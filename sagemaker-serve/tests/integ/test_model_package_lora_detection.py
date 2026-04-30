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
ModelPackage recipe names and that the build path sets the right attributes.
No deployment or GPU instances required.
"""
from __future__ import absolute_import

import os
import time
import random
import pytest


# LoRA model package (recipe name contains "lora")
LORA_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1"
)

# Non-LoRA model package (DPO recipe, no "lora" in name)
NON_LORA_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/264"
)


@pytest.fixture(scope="session", autouse=True)
def set_region():
    """Ensure us-west-2 region for all tests."""
    original = os.environ.get("SAGEMAKER_REGION")
    os.environ["SAGEMAKER_REGION"] = "us-west-2"
    yield
    if original:
        os.environ["SAGEMAKER_REGION"] = original
    elif "SAGEMAKER_REGION" in os.environ:
        del os.environ["SAGEMAKER_REGION"]


class TestModelPackageLoraDetection:
    """Test _fetch_peft() LoRA detection from real ModelPackage resources."""

    def test_fetch_peft_returns_lora_for_lora_model_package(self):
        """_fetch_peft() returns 'LORA' for a ModelPackage with a LoRA recipe name."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)

        peft_type = model_builder._fetch_peft()
        assert peft_type == "LORA", (
            f"Expected 'LORA' but got '{peft_type}' for {LORA_MODEL_PACKAGE_ARN}"
        )

    def test_fetch_peft_returns_none_for_non_lora_model_package(self):
        """_fetch_peft() returns None for a ModelPackage without LoRA recipe name."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=NON_LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)

        peft_type = model_builder._fetch_peft()
        assert peft_type is None, (
            f"Expected None but got '{peft_type}' for {NON_LORA_MODEL_PACKAGE_ARN}"
        )


class TestModelPackageLoraBuild:
    """Test that build() from a LoRA ModelPackage sets the right attributes."""

    def test_build_lora_model_package_sets_adapter_s3_uri(self):
        """build() from a LoRA ModelPackage sets _adapter_s3_uri."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)
        model_builder.accept_eula = True

        model_name = f"integ-lora-test-{int(time.time())}-{random.randint(100, 10000)}"
        model = model_builder.build(model_name=model_name)

        try:
            # Verify model was created
            assert model is not None
            assert model.model_arn is not None

            # Verify LoRA-specific attributes
            assert hasattr(model_builder, "_adapter_s3_uri"), (
                "_adapter_s3_uri should be set after build() for LoRA ModelPackage"
            )
            assert model_builder._adapter_s3_uri is not None
            assert model_builder._adapter_s3_uri.startswith("s3://"), (
                f"_adapter_s3_uri should be an S3 URI, got: {model_builder._adapter_s3_uri}"
            )
        finally:
            # Cleanup: delete the created model
            try:
                model.delete()
            except Exception:
                pass

    def test_build_non_lora_model_package_no_adapter_uri(self):
        """build() from a non-LoRA ModelPackage does NOT set _adapter_s3_uri."""
        from sagemaker.core.resources import ModelPackage
        from sagemaker.serve import ModelBuilder

        model_package = ModelPackage.get(model_package_name=NON_LORA_MODEL_PACKAGE_ARN)
        model_builder = ModelBuilder(model=model_package)
        model_builder.accept_eula = True

        model_name = f"integ-nonlora-test-{int(time.time())}-{random.randint(100, 10000)}"
        model = model_builder.build(model_name=model_name)

        try:
            # Verify model was created
            assert model is not None
            assert model.model_arn is not None

            # Verify no adapter URI is set (non-LoRA path)
            adapter_uri = getattr(model_builder, "_adapter_s3_uri", None)
            assert adapter_uri is None, (
                f"_adapter_s3_uri should not be set for non-LoRA ModelPackage, got: {adapter_uri}"
            )
        finally:
            # Cleanup: delete the created model
            try:
                model.delete()
            except Exception:
                pass
