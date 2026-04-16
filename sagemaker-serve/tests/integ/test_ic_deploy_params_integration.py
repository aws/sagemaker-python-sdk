# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Integration test for IC-level deploy parameters (data_cache_config, variant_name).

Uses ModelBuilder with a simple PyTorch model and ResourceRequirements to deploy
via the IC-based endpoint path, then verifies DataCacheConfig and custom VariantName
on the created InferenceComponent.
"""
from __future__ import absolute_import

import json
import os
import tempfile
import uuid
import logging

import boto3
import pytest
import torch
import torch.nn as nn

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.spec.inference_spec import InferenceSpec
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.resources import EndpointConfig

logger = logging.getLogger(__name__)


class SimpleModel(nn.Module):
    """Tiny PyTorch model for testing."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return torch.softmax(self.linear(x), dim=1)


class SimpleInferenceSpec(InferenceSpec):
    """InferenceSpec for the simple model."""

    def load(self, model_dir: str):
        model = SimpleModel()
        model_path = os.path.join(model_dir, "model.pth")
        if os.path.exists(model_path):
            model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
        return model

    def invoke(self, input_object, model):
        input_tensor = torch.tensor(input_object, dtype=torch.float32)
        with torch.no_grad():
            return model(input_tensor).tolist()


def _save_model(path):
    """Save a traced PyTorch model to disk."""
    os.makedirs(path, exist_ok=True)
    m = SimpleModel()
    traced = torch.jit.trace(m, torch.randn(1, 4))
    torch.jit.save(traced, os.path.join(path, "model.pth"))


def _cleanup(endpoint_name, sagemaker_client):
    """Best-effort cleanup."""
    try:
        paginator = sagemaker_client.get_paginator("list_inference_components")
        for page in paginator.paginate(EndpointNameEquals=endpoint_name):
            for ic in page.get("InferenceComponents", []):
                try:
                    sagemaker_client.delete_inference_component(
                        InferenceComponentName=ic["InferenceComponentName"]
                    )
                except Exception:
                    pass
    except Exception:
        pass
    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
    except Exception:
        pass
    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
    except Exception:
        pass


@pytest.mark.slow_test
def test_deploy_ic_with_data_cache_config_and_variant_name():
    """Deploy a simple model via ModelBuilder IC path with data_cache_config and variant_name.

    Uses a tiny PyTorch model on ml.m5.xlarge (CPU) to keep costs low and avoid
    GPU capacity issues. Verifies the IC was created with the correct DataCacheConfig
    and VariantName via boto3 describe.
    """
    uid = uuid.uuid4().hex[:8]
    endpoint_name = f"ic-params-ep-{uid}"
    ic_name = f"ic-params-component-{uid}"
    custom_variant = f"Variant-{uid}"
    model_name = f"ic-params-model-{uid}"

    sagemaker_client = boto3.client("sagemaker", region_name="us-west-2")
    model_path = tempfile.mkdtemp()
    _save_model(model_path)

    try:
        schema = SchemaBuilder(
            sample_input=[[0.1, 0.2, 0.3, 0.4]],
            sample_output=[[0.6, 0.4]],
        )

        # Use a PyTorch inference image that works across CI (py310) and local (py312).
        # The container has its own Python, so py310 works regardless of host Python.
        from sagemaker.core import image_uris
        inference_image = image_uris.retrieve(
            framework="pytorch",
            region="us-west-2",
            version="2.2.0",
            py_version="py310",
            instance_type="ml.m5.xlarge",
            image_scope="inference",
        )

        model_builder = ModelBuilder(
            inference_spec=SimpleInferenceSpec(),
            model_path=model_path,
            model_server=ModelServer.TORCHSERVE,
            schema_builder=schema,
            instance_type="ml.m5.xlarge",
            image_uri=inference_image,
            dependencies={"auto": False},
        )

        model_builder.build(model_name=model_name)
        logger.info("Model built: %s", model_name)

        resources = ResourceRequirements(
            requests={"memory": 1024, "num_cpus": 1, "copies": 1}
        )

        endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=1,
            inference_config=resources,
            inference_component_name=ic_name,
            data_cache_config={"enable_caching": True},
            variant_name=custom_variant,
        )
        logger.info("Endpoint deployed: %s", endpoint.endpoint_name)

        # Wait for the IC to be fully ready before describing it.
        # deploy() creates the IC with wait=False, so it may still be Creating.
        import time
        for _ in range(40):
            ic_status = sagemaker_client.describe_inference_component(
                InferenceComponentName=ic_name
            ).get("InferenceComponentStatus")
            if ic_status == "InService":
                break
            logger.info("IC status: %s, waiting...", ic_status)
            time.sleep(15)
        logger.info("IC InService: %s", ic_name)

        # Verify the IC was created with correct params
        ic_desc = sagemaker_client.describe_inference_component(
            InferenceComponentName=ic_name
        )

        # Check DataCacheConfig
        spec = ic_desc.get("Specification", {})
        data_cache = spec.get("DataCacheConfig", {})
        assert data_cache.get("EnableCaching") is True, (
            f"Expected DataCacheConfig.EnableCaching=True, got {data_cache}"
        )

        # Check VariantName
        actual_variant = ic_desc.get("VariantName")
        assert actual_variant == custom_variant, (
            f"Expected VariantName='{custom_variant}', got '{actual_variant}'"
        )

        logger.info("Test passed: IC has correct DataCacheConfig and VariantName")

    finally:
        _cleanup(endpoint_name, sagemaker_client)
