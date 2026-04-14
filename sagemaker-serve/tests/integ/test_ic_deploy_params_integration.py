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
"""Integration tests for IC-level deploy parameters (data_cache_config, variant_name)."""
from __future__ import absolute_import

import json
import uuid
import time
import random
import logging

import boto3
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.inference_config import ResourceRequirements
from sagemaker.core.resources import (
    Endpoint,
    EndpointConfig,
    InferenceComponent,
)
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# Use the same JumpStart model as test_jumpstart_integration.py
MODEL_ID = "huggingface-llm-falcon-7b-bf16"

# Training job for model customization path (same as test_model_customization_deployment.py)
TRAINING_JOB_NAME = "meta-textgeneration-llama-3-2-1b-instruct-sft-20251201172445"


def _cleanup_endpoint(endpoint_name, sagemaker_client):
    """Delete endpoint, endpoint config, and all inference components."""
    try:
        # Delete inference components first
        paginator = sagemaker_client.get_paginator("list_inference_components")
        for page in paginator.paginate(EndpointNameEquals=endpoint_name):
            for ic in page.get("InferenceComponents", []):
                ic_name = ic["InferenceComponentName"]
                try:
                    sagemaker_client.delete_inference_component(
                        InferenceComponentName=ic_name
                    )
                    logger.info("Deleted inference component: %s", ic_name)
                except Exception as e:
                    logger.warning("Failed to delete IC %s: %s", ic_name, e)
    except Exception as e:
        logger.warning("Failed to list/delete ICs for %s: %s", endpoint_name, e)

    try:
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        logger.info("Deleted endpoint: %s", endpoint_name)
    except Exception as e:
        logger.warning("Failed to delete endpoint %s: %s", endpoint_name, e)

    try:
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        logger.info("Deleted endpoint config: %s", endpoint_name)
    except Exception as e:
        logger.warning("Failed to delete endpoint config %s: %s", endpoint_name, e)


def _cleanup_model(model_name, sagemaker_client):
    """Delete a SageMaker model."""
    try:
        sagemaker_client.delete_model(ModelName=model_name)
        logger.info("Deleted model: %s", model_name)
    except Exception as e:
        logger.warning("Failed to delete model %s: %s", model_name, e)


@pytest.mark.slow_test
def test_deploy_with_data_cache_config_and_variant_name_via_ic_path():
    """Deploy a JumpStart model via the IC-based path with data_cache_config and custom variant_name.

    Verifies:
    - The IC was created with DataCacheConfig.EnableCaching == True
    - The variant name matches the custom value (not 'AllTraffic')
    """
    unique_id = uuid.uuid4().hex[:8]
    model_name = f"ic-params-test-model-{unique_id}"
    endpoint_name = f"ic-params-test-ep-{unique_id}"
    custom_variant = f"Variant-{unique_id}"

    sagemaker_client = boto3.client("sagemaker")
    ic_name = None

    try:
        # Build
        compute = Compute(instance_type="ml.g5.2xlarge")
        jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
        model_builder = ModelBuilder.from_jumpstart_config(
            jumpstart_config=jumpstart_config, compute=compute
        )
        core_model = model_builder.build(model_name=model_name)
        logger.info("Model created: %s", core_model.model_name)

        # Deploy with IC path (ResourceRequirements triggers IC-based endpoint)
        resources = ResourceRequirements(
            requests={
                "memory": 8192,
                "num_accelerators": 1,
                "num_cpus": 2,
                "copies": 1,
            }
        )
        core_endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=1,
            inference_config=resources,
            data_cache_config={"enable_caching": True},
            variant_name=custom_variant,
        )
        logger.info("Endpoint created: %s", core_endpoint.endpoint_name)

        # Find the inference component that was created
        ic_name = model_builder.inference_component_name
        assert ic_name is not None, "inference_component_name should be set after deploy"

        # Describe the inference component via boto3
        ic_desc = sagemaker_client.describe_inference_component(
            InferenceComponentName=ic_name
        )

        # Verify DataCacheConfig.EnableCaching == True
        spec = ic_desc.get("Specification", {})
        data_cache = spec.get("DataCacheConfig", {})
        assert data_cache.get("EnableCaching") is True, (
            f"Expected DataCacheConfig.EnableCaching=True, got {data_cache}"
        )

        # Verify variant name matches custom value
        actual_variant = ic_desc.get("VariantName")
        assert actual_variant == custom_variant, (
            f"Expected VariantName='{custom_variant}', got '{actual_variant}'"
        )

        logger.info(
            "Test passed: IC '%s' has DataCacheConfig.EnableCaching=True and VariantName='%s'",
            ic_name,
            custom_variant,
        )

    finally:
        _cleanup_endpoint(endpoint_name, sagemaker_client)
        _cleanup_model(model_name, sagemaker_client)


@pytest.mark.slow_test
def test_deploy_with_data_cache_config_via_model_customization_path():
    """Deploy a fine-tuned model via _deploy_model_customization with data_cache_config.

    Verifies:
    - The IC was created with DataCacheConfig.EnableCaching == True
    - The variant_name defaults to endpoint_name (backward compat) when not explicitly provided
    """
    from sagemaker.core.resources import TrainingJob

    unique_id = uuid.uuid4().hex[:8]
    model_name = f"ic-mc-test-model-{unique_id}"
    endpoint_name = f"ic-mc-test-ep-{unique_id}"

    sagemaker_client = boto3.client("sagemaker")

    try:
        training_job = TrainingJob.get(training_job_name=TRAINING_JOB_NAME)
        model_builder = ModelBuilder(
            model=training_job, instance_type="ml.g5.4xlarge"
        )
        model_builder.accept_eula = True
        core_model = model_builder.build(model_name=model_name)
        logger.info("Model created: %s", core_model.model_name)

        # Deploy with data_cache_config but WITHOUT explicit variant_name
        # so it should default to endpoint_name for model customization path
        endpoint = model_builder.deploy(
            endpoint_name=endpoint_name,
            initial_instance_count=1,
            data_cache_config={"enable_caching": True},
        )
        logger.info("Endpoint created: %s", endpoint.endpoint_name)

        # Find inference components on this endpoint
        paginator = sagemaker_client.get_paginator("list_inference_components")
        ic_names = []
        for page in paginator.paginate(EndpointNameEquals=endpoint_name):
            for ic in page.get("InferenceComponents", []):
                ic_names.append(ic["InferenceComponentName"])

        assert len(ic_names) > 0, (
            f"Expected at least one inference component on endpoint '{endpoint_name}'"
        )

        # Check the first (or base) IC for DataCacheConfig
        # For LORA, the base IC should have data_cache_config; for non-LORA, the single IC.
        peft_type = model_builder._fetch_peft()
        if peft_type == "LORA":
            # Base IC is named <endpoint_name>-inference-component
            base_ic_name = f"{endpoint_name}-inference-component"
        else:
            base_ic_name = f"{endpoint_name}-inference-component"

        ic_desc = sagemaker_client.describe_inference_component(
            InferenceComponentName=base_ic_name
        )

        # Verify DataCacheConfig.EnableCaching == True
        spec = ic_desc.get("Specification", {})
        data_cache = spec.get("DataCacheConfig", {})
        assert data_cache.get("EnableCaching") is True, (
            f"Expected DataCacheConfig.EnableCaching=True, got {data_cache}"
        )

        # Verify variant_name defaults to endpoint_name (backward compat)
        actual_variant = ic_desc.get("VariantName")
        assert actual_variant == endpoint_name, (
            f"Expected VariantName='{endpoint_name}' (backward compat default), "
            f"got '{actual_variant}'"
        )

        logger.info(
            "Test passed: IC '%s' has DataCacheConfig.EnableCaching=True "
            "and VariantName='%s' (backward compat default)",
            base_ic_name,
            endpoint_name,
        )

    finally:
        _cleanup_endpoint(endpoint_name, sagemaker_client)
        _cleanup_model(model_name, sagemaker_client)
