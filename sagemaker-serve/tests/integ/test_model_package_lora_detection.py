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
"""Integration tests for ModelPackage LoRA detection in ModelBuilder.

Build-only tests that verify _fetch_peft() correctly detects LoRA from
ModelPackage recipe names, that the build path sets the right attributes,
and that the created SageMaker Model has the correct container configuration.
No deployment or GPU instances required.
"""
from __future__ import absolute_import

import os
import time
import random
import pytest
import boto3


# LoRA model package (recipe name contains "lora")
LORA_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1"
)

# Non-LoRA model package (DPO recipe, no "lora" in name)
NON_LORA_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/264"
)

REGION = "us-west-2"


@pytest.fixture(scope="session", autouse=True)
def set_region():
    """Ensure us-west-2 region for all tests."""
    original = os.environ.get("SAGEMAKER_REGION")
    os.environ["SAGEMAKER_REGION"] = REGION
    yield
    if original:
        os.environ["SAGEMAKER_REGION"] = original
    elif "SAGEMAKER_REGION" in os.environ:
        del os.environ["SAGEMAKER_REGION"]


@pytest.fixture(scope="session")
def sm_client():
    """Boto3 SageMaker client for validating created models."""
    return boto3.client("sagemaker", region_name=REGION)


class TestModelPackageLoraBuild:
    """Test build() from LoRA and non-LoRA ModelPackages."""

    def test_build_lora_model_package(self, sm_client):
        """Build from a LoRA ModelPackage: verify peft detection, adapter URI, and model config."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)
        model_builder.accept_eula = True

        # Verify _fetch_peft detects LoRA
        peft_type = model_builder._fetch_peft()
        assert peft_type == "LORA", (
            f"Expected 'LORA' but got '{peft_type}' for {LORA_MODEL_PACKAGE_ARN}"
        )

        model_name = f"integ-lora-test-{int(time.time())}-{random.randint(100, 10000)}"
        model = model_builder.build(model_name=model_name)

        try:
            assert model is not None
            assert model.model_arn is not None

            # Verify _adapter_s3_uri is set to a valid S3 path
            assert hasattr(model_builder, "_adapter_s3_uri"), (
                "_adapter_s3_uri should be set after build() for LoRA ModelPackage"
            )
            assert model_builder._adapter_s3_uri is not None
            assert model_builder._adapter_s3_uri.startswith("s3://"), (
                f"_adapter_s3_uri should be an S3 URI, got: {model_builder._adapter_s3_uri}"
            )

            # Use boto3 to validate the actual model configuration
            describe_resp = sm_client.describe_model(ModelName=model_name)
            containers = describe_resp.get("Containers", [])
            assert len(containers) == 1, f"Expected 1 container, got {len(containers)}"

            container = containers[0]

            # LoRA path: container should point at JumpStart base model, NOT the adapter
            model_data = container.get("ModelDataSource", {})
            s3_source = model_data.get("S3DataSource", {})
            s3_uri = s3_source.get("S3Uri", "")
            assert s3_uri, "Container should have an S3 model data source"
            # The S3 URI should NOT be the adapter URI — it should be the base model
            assert s3_uri != model_builder._adapter_s3_uri, (
                f"LoRA container S3 URI should point to base model, not adapter. "
                f"Got: {s3_uri}, adapter: {model_builder._adapter_s3_uri}"
            )

            # LoRA path: container should have accept_eula in model access config
            access_config = s3_source.get("ModelAccessConfig", {})
            assert access_config.get("AcceptEula") is True, (
                "LoRA container should have AcceptEula=True in ModelAccessConfig"
            )

            # LoRA path: container should have environment variables set
            env_vars = container.get("Environment", {})
            assert len(env_vars) > 0, (
                "LoRA container should have environment variables set"
            )
        finally:
            try:
                model.delete()
            except Exception:
                pass

    def test_build_non_lora_model_package(self, sm_client):
        """Build from a non-LoRA ModelPackage: verify no adapter URI and env vars in container."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=NON_LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)
        model_builder.accept_eula = True

        # Verify _fetch_peft does NOT detect LoRA
        peft_type = model_builder._fetch_peft()
        assert peft_type is None, (
            f"Expected None but got '{peft_type}' for {NON_LORA_MODEL_PACKAGE_ARN}"
        )

        model_name = f"integ-nonlora-test-{int(time.time())}-{random.randint(100, 10000)}"
        model = model_builder.build(model_name=model_name)

        try:
            assert model is not None
            assert model.model_arn is not None

            # Verify no adapter URI is set
            adapter_uri = getattr(model_builder, "_adapter_s3_uri", None)
            assert adapter_uri is None, (
                f"_adapter_s3_uri should not be set for non-LoRA, got: {adapter_uri}"
            )

            # Use boto3 to validate the actual model configuration
            describe_resp = sm_client.describe_model(ModelName=model_name)
            containers = describe_resp.get("Containers", [])
            assert len(containers) == 1, f"Expected 1 container, got {len(containers)}"

            container = containers[0]

            # Non-LoRA path: container should have environment variables (bug fix validation)
            env_vars = container.get("Environment", {})
            assert len(env_vars) > 0, (
                "Non-LoRA container should have environment variables set (env_vars bug fix)"
            )

            # Non-LoRA path: container should point at training output S3 URI
            model_data = container.get("ModelDataSource", {})
            s3_source = model_data.get("S3DataSource", {})
            s3_uri = s3_source.get("S3Uri", "")
            assert s3_uri, "Non-LoRA container should have an S3 model data source"
        finally:
            try:
                model.delete()
            except Exception:
                pass
