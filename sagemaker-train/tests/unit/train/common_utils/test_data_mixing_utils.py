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
"""Unit tests for data mixing utility functions."""
from __future__ import absolute_import

import os

import pytest

from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.train.common_utils.data_mixing_utils import (
    validate_data_mixing_platform,
    validate_data_mixing_model,
    validate_data_mixing_categories,
    HyperPodTemplateContext,
)
from sagemaker.train.common_utils.model_aliases import MODEL_NAME_ALIASES
from sagemaker.core.training.constants import TrainingPlatform


class TestValidateDataMixingPlatform:
    """Tests for validate_data_mixing_platform()."""

    def test_sagemaker_hyperpod_passes(self):
        """SAGEMAKER_HYPERPOD platform should pass validation without raising."""
        validate_data_mixing_platform(TrainingPlatform.SAGEMAKER_HYPERPOD)

    def test_sagemaker_training_job_raises_value_error(self):
        """SAGEMAKER_TRAINING_JOB platform should raise ValueError — does NOT support data mixing."""
        with pytest.raises(ValueError, match="Data mixing is only supported on HyperPod"):
            validate_data_mixing_platform(TrainingPlatform.SAGEMAKER_TRAINING_JOB_SERVERFUL)

    def test_serverless_platform_passes(self):
        """SAGEMAKER_TRAINING_JOB_SERVERLESS platform should pass validation without raising."""
        validate_data_mixing_platform(TrainingPlatform.SAGEMAKER_TRAINING_JOB_SERVERLESS)

    @pytest.mark.parametrize(
        "platform",
        [
            "local",
            "sagemaker",
            "ec2",
            "",
        ],
        ids=[
            "local",
            "sagemaker",
            "ec2",
            "empty_string",
        ],
    )
    def test_unsupported_platform_raises_value_error(self, platform):
        """Unsupported platforms should raise ValueError with descriptive message."""
        with pytest.raises(ValueError, match="Data mixing is only supported on HyperPod"):
            validate_data_mixing_platform(platform)


class TestValidateDataMixingModel:
    """Tests for validate_data_mixing_model()."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "amazon.nova-2-lite-v1",
            "amazon.nova-lite-v1",
            "amazon.nova-pro-v1",
            "amazon.nova-micro-v1",
            "nova-textgeneration-lite-v2",
            "nova-textgeneration-lite",
            "nova-textgeneration-pro",
            "nova-textgeneration-micro",
        ],
        ids=[
            "bedrock_nova_2_lite",
            "bedrock_nova_lite_v1",
            "bedrock_nova_pro_v1",
            "bedrock_nova_micro_v1",
            "hub_nova_lite_v2",
            "hub_nova_lite",
            "hub_nova_pro",
            "hub_nova_micro",
        ],
    )
    def test_known_nova_models_pass(self, model_name):
        """Known Nova model identifiers (from MODEL_NAME_ALIASES) should pass validation."""
        validate_data_mixing_model(model_name)

    def test_nova_model_in_longer_name_passes(self):
        """A model name containing a Nova identifier substring should pass."""
        validate_data_mixing_model("hub-content-amazon-nova-micro-v1")

    def test_nova_model_case_insensitive(self):
        """Model name matching should be case-insensitive."""
        validate_data_mixing_model("Amazon.Nova-Pro-V1")

    @pytest.mark.parametrize(
        "model_name",
        [
            "meta-textgeneration-llama-3-2-1b-instruct",
            "anthropic-claude-v2",
            "mistral-7b",
            "falcon-40b",
            "amazon-titan-express",
        ],
        ids=[
            "llama",
            "claude",
            "mistral",
            "falcon",
            "titan",
        ],
    )
    def test_non_nova_models_raise_value_error(self, model_name):
        """Non-Nova models should raise ValueError listing supported models."""
        with pytest.raises(ValueError, match="Data mixing is only supported for Nova models"):
            validate_data_mixing_model(model_name)

    def test_model_name_aliases_used_for_validation(self):
        """validate_data_mixing_model should accept all keys and values from MODEL_NAME_ALIASES."""
        for alias_key in MODEL_NAME_ALIASES.keys():
            validate_data_mixing_model(alias_key)
        for hub_name in MODEL_NAME_ALIASES.values():
            validate_data_mixing_model(hub_name)


class TestValidateDataMixingCategories:
    """Tests for validate_data_mixing_categories()."""

    RECIPE_CATEGORIES = {
        "en-entertainment": 25.0,
        "code": 25.0,
        "math": 25.0,
        "en-scientific": 25.0,
    }

    def test_valid_subset_of_recipe_keys_passes(self):
        """User-provided keys that are a valid subset of recipe keys should pass."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"code": 50.0, "math": 50.0},
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        assert result.nova_data_percentages is not None
        assert result.nova_data_percentages["code"] == 50.0
        assert result.nova_data_percentages["math"] == 50.0

    def test_all_recipe_keys_provided_passes(self):
        """Providing all recipe keys should pass validation."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={
                "en-entertainment": 25.0,
                "code": 25.0,
                "math": 25.0,
                "en-scientific": 25.0,
            },
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        assert result.nova_data_percentages == {
            "en-entertainment": 25.0,
            "code": 25.0,
            "math": 25.0,
            "en-scientific": 25.0,
        }

    def test_unrecognized_keys_raise_value_error(self):
        """Category keys not in recipe should raise ValueError."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"invalid-key": 50.0, "also-invalid": 50.0},
        )
        with pytest.raises(ValueError, match="Unrecognized data mixing categories"):
            validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

    def test_unrecognized_keys_error_lists_valid_categories(self):
        """Error message should list valid categories from recipe."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"bad-key": 100.0},
        )
        with pytest.raises(ValueError, match="Valid categories from recipe"):
            validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

    def test_mix_of_valid_and_invalid_keys_raises(self):
        """A mix of valid and invalid keys should raise for the invalid ones."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"code": 50.0, "nonexistent": 50.0},
        )
        with pytest.raises(ValueError, match="Unrecognized data mixing categories"):
            validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

    def test_unspecified_categories_zeroed_out(self):
        """Categories in recipe but not specified by user should be set to 0."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"code": 100.0},
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

        # All recipe categories should be present
        assert set(result.nova_data_percentages.keys()) == set(self.RECIPE_CATEGORIES.keys())

        # User-specified value preserved
        assert result.nova_data_percentages["code"] == 100.0

        # Unspecified categories zeroed out
        assert result.nova_data_percentages["en-entertainment"] == 0.0
        assert result.nova_data_percentages["math"] == 0.0
        assert result.nova_data_percentages["en-scientific"] == 0.0

    def test_none_nova_percentages_uses_recipe_defaults(self):
        """When nova_data_percentages is None, recipe defaults are used."""
        config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)

        # All recipe categories should be present with their default values
        assert result.nova_data_percentages == self.RECIPE_CATEGORIES
        # Customer data percent preserved
        assert result.customer_data_percent == 50.0

    def test_none_nova_percentages_preserves_customer_percent(self):
        """When using recipe defaults, customer_data_percent is preserved."""
        config = DataMixingConfig(
            customer_data_percent=75.0,
            nova_data_percentages=None,
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        assert result.customer_data_percent == 75.0

    def test_result_contains_all_recipe_category_keys(self):
        """Returned config always contains all recipe category keys."""
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={"math": 100.0},
        )
        result = validate_data_mixing_categories(config, self.RECIPE_CATEGORIES)
        assert set(result.nova_data_percentages.keys()) == set(self.RECIPE_CATEGORIES.keys())


class TestHyperPodTemplateContext:
    def _make_context(self, **overrides):
        """Helper to construct a HyperPodTemplateContext with sensible defaults."""
        defaults = {
            "raw_template": "apiVersion: v1\nkind: ConfigMap\n",
            "overrides": {"nova_code_percent": {"default": 25.0}},
            "recipe_name": "nova_pro_sft_lora_datamix",
            "datamix_keyword": "text_with_datamix",
            "categories": {"code": 25.0, "math": 25.0},
            "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/train:latest",
        }
        defaults.update(overrides)
        return HyperPodTemplateContext(**defaults)

    def test_valid_construction_with_all_fields(self):
        """HyperPodTemplateContext should construct successfully with all fields provided."""
        ctx = self._make_context()
        assert ctx.raw_template == "apiVersion: v1\nkind: ConfigMap\n"
        assert ctx.overrides == {"nova_code_percent": {"default": 25.0}}
        assert ctx.recipe_name == "nova_pro_sft_lora_datamix"
        assert ctx.datamix_keyword == "text_with_datamix"
        assert ctx.categories == {"code": 25.0, "math": 25.0}
        assert ctx.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/train:latest"

    def test_construction_with_image_uri_none(self):
        """HyperPodTemplateContext should allow image_uri=None (the default)."""
        ctx = self._make_context(image_uri=None)
        assert ctx.image_uri is None
        # All other fields should still be set correctly
        assert ctx.raw_template == "apiVersion: v1\nkind: ConfigMap\n"
        assert ctx.recipe_name == "nova_pro_sft_lora_datamix"

    def test_empty_raw_template_raises_value_error(self):
        """Empty raw_template should raise ValueError mentioning the field."""
        with pytest.raises(ValueError, match="raw_template must not be empty"):
            self._make_context(raw_template="")

    def test_empty_recipe_name_raises_value_error(self):
        """Empty recipe_name should raise ValueError mentioning the field."""
        with pytest.raises(ValueError, match="recipe_name must not be empty"):
            self._make_context(recipe_name="")

    def test_immutability_raises_frozen_instance_error(self):
        """Assigning to a field on a frozen dataclass should raise FrozenInstanceError."""
        from dataclasses import FrozenInstanceError

        ctx = self._make_context()
        with pytest.raises(FrozenInstanceError):
            ctx.raw_template = "modified"
        with pytest.raises(FrozenInstanceError):
            ctx.recipe_name = "modified"
        with pytest.raises(FrozenInstanceError):
            ctx.image_uri = "modified"


class TestResolveHyperPodDatamixContext:
    MOCK_TEMPLATE_CONTENT = (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: training-config.yaml\n"
        "data:\n  config.yaml: |-\n    model:\n      name: nova-pro\n"
    )

    MOCK_TEMPLATE_WITH_IMAGE = (
        "apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: training-config.yaml\n"
        "data:\n  config.yaml: |-\n    model:\n      name: nova-pro\n"
        "containers:\n  - name: pytorch\n    image: 123456789012.dkr.ecr.us-west-2.amazonaws.com/train:latest\n"
    )

    MOCK_OVERRIDES = {
        "nova_code_percent": {"default": 25.0, "type": "float"},
        "nova_math_percent": {"default": 25.0, "type": "float"},
        "nova_en-entertainment_percent": {"default": 25.0, "type": "float"},
        "nova_en-scientific_percent": {"default": 25.0, "type": "float"},
        "other_param": {"default": "value"},
    }

    MOCK_RECIPE = {
        "Name": "nova_pro_sft_lora_text_with_datamix",
        "CustomizationTechnique": "SFT",
        "HpEksPayloadTemplateS3Uri": "s3://my-bucket/templates/recipe.yaml",
        "HpEksOverrideParamsS3Uri": "s3://my-bucket/overrides/params.json",
    }

    MOCK_HUB_METADATA = {
        "hub_content_document": {
            "RecipeCollection": [
                {
                    "Name": "nova_pro_sft_lora_text_with_datamix",
                    "CustomizationTechnique": "SFT",
                    "HpEksPayloadTemplateS3Uri": "s3://my-bucket/templates/recipe.yaml",
                    "HpEksOverrideParamsS3Uri": "s3://my-bucket/overrides/params.json",
                }
            ]
        }
    }

    def _make_mock_session(self):
        """Create a mock sagemaker_session with boto_session, s3 client, and sts client."""
        from unittest.mock import MagicMock, patch
        import io
        import json as json_mod

        session = MagicMock()
        session.boto_session.region_name = "us-west-2"

        # STS mock
        sts_client = MagicMock()
        sts_client.get_caller_identity.return_value = {"Account": "123456789012"}

        # S3 mock
        s3_client = MagicMock()

        def client_factory(service_name, **kwargs):
            if service_name == "sts":
                return sts_client
            elif service_name == "s3":
                return s3_client
            return MagicMock()

        session.boto_session.client.side_effect = client_factory
        return session, s3_client, sts_client

    def _setup_s3_responses(self, s3_client, template_content=None, overrides=None):
        """Configure the s3_client mock to return template and overrides content."""
        import io
        import json as json_mod
        from unittest.mock import MagicMock

        if template_content is None:
            template_content = self.MOCK_TEMPLATE_CONTENT
        if overrides is None:
            overrides = self.MOCK_OVERRIDES

        template_body = MagicMock()
        template_body.read.return_value = template_content.encode("utf-8")

        overrides_body = MagicMock()
        overrides_body.read.return_value = json_mod.dumps(overrides).encode("utf-8")

        # First call returns template, second returns overrides
        s3_client.get_object.side_effect = [
            {"Body": template_body},
            {"Body": overrides_body},
        ]

    @pytest.fixture
    def mock_hub_metadata(self):
        """Patch _get_hub_content_metadata and get_sagemaker_hub_name."""
        from unittest.mock import patch

        with patch(
            "sagemaker.train.common_utils.data_mixing_utils._get_hub_content_metadata"
        ) as mock_get_hub, patch(
            "sagemaker.train.common_utils.data_mixing_utils.get_sagemaker_hub_name"
        ) as mock_hub_name:
            mock_hub_name.return_value = "SageMakerPublicHub"
            mock_get_hub.return_value = self.MOCK_HUB_METADATA
            yield mock_get_hub, mock_hub_name

    # --- Test cases ---

    def test_successful_resolve_returns_populated_context(self, mock_hub_metadata):
        """Successful resolve should return a correctly populated HyperPodTemplateContext."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()
        self._setup_s3_responses(s3_client)

        ctx = resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

        assert isinstance(ctx, HyperPodTemplateContext)
        assert ctx.raw_template == self.MOCK_TEMPLATE_CONTENT
        assert ctx.overrides == self.MOCK_OVERRIDES
        assert ctx.recipe_name == "nova_pro_sft_lora_text_with_datamix"
        assert ctx.datamix_keyword == "text_with_datamix"
        assert ctx.categories == {
            "code": 25.0,
            "math": 25.0,
            "en-entertainment": 25.0,
            "en-scientific": 25.0,
        }

    def test_exactly_two_s3_get_object_calls(self, mock_hub_metadata):
        """Resolve should make exactly 2 S3 GetObject calls (template + overrides)."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()
        self._setup_s3_responses(s3_client)

        resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

        assert s3_client.get_object.call_count == 2

    def test_missing_hp_eks_payload_template_s3_uri_raises_value_error(self, mock_hub_metadata):
        """Missing HpEksPayloadTemplateS3Uri should raise ValueError with field name."""
        from unittest.mock import patch
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        # Override hub metadata to have a recipe without HpEksPayloadTemplateS3Uri
        mock_get_hub, _ = mock_hub_metadata
        mock_get_hub.return_value = {
            "hub_content_document": {
                "RecipeCollection": [
                    {
                        "Name": "nova_pro_sft_lora_text_with_datamix",
                        "CustomizationTechnique": "SFT",
                        # Missing HpEksPayloadTemplateS3Uri
                        "HpEksOverrideParamsS3Uri": "s3://my-bucket/overrides/params.json",
                    }
                ]
            }
        }

        session, s3_client, _ = self._make_mock_session()

        with pytest.raises(ValueError, match="HpEksPayloadTemplateS3Uri"):
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

    def test_missing_hp_eks_override_params_s3_uri_raises_value_error(self, mock_hub_metadata):
        """Missing HpEksOverrideParamsS3Uri should raise ValueError with field name."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        mock_get_hub, _ = mock_hub_metadata
        mock_get_hub.return_value = {
            "hub_content_document": {
                "RecipeCollection": [
                    {
                        "Name": "nova_pro_sft_lora_text_with_datamix",
                        "CustomizationTechnique": "SFT",
                        "HpEksPayloadTemplateS3Uri": "s3://my-bucket/templates/recipe.yaml",
                        # Missing HpEksOverrideParamsS3Uri
                    }
                ]
            }
        }

        session, s3_client, _ = self._make_mock_session()

        with pytest.raises(ValueError, match="HpEksOverrideParamsS3Uri"):
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

    def test_s3_client_error_raises_value_error_with_forge_iam_message(self, mock_hub_metadata):
        """S3 ClientError should raise ValueError referencing Forge subscription and s3:GetObject."""
        from unittest.mock import MagicMock
        from botocore.exceptions import ClientError
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()

        # S3 get_object raises ClientError
        s3_client.get_object.side_effect = ClientError(
            error_response={"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            operation_name="GetObject",
        )

        with pytest.raises(ValueError) as exc_info:
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

        error_msg = str(exc_info.value)
        assert "Forge" in error_msg or "forge" in error_msg.lower()
        assert "s3:GetObject" in error_msg

    def test_no_matching_recipe_raises_value_error_with_all_filter_identifiers(
        self, mock_hub_metadata
    ):
        """No matching recipe should raise ValueError with model_name, datamix_keyword,
        customization_technique, and training_type."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        mock_get_hub, _ = mock_hub_metadata
        # Return a recipe collection that doesn't match the requested filters
        mock_get_hub.return_value = {
            "hub_content_document": {
                "RecipeCollection": [
                    {
                        "Name": "nova_pro_cpt_full_mm_with_datamix",
                        "CustomizationTechnique": "CPT",
                        "HpEksPayloadTemplateS3Uri": "s3://bucket/template.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://bucket/overrides.json",
                    }
                ]
            }
        }

        session, _, _ = self._make_mock_session()

        with pytest.raises(ValueError) as exc_info:
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

        error_msg = str(exc_info.value)
        assert "nova-pro" in error_msg
        assert "text_with_datamix" in error_msg
        assert "SFT" in error_msg
        assert "LORA" in error_msg

    def test_no_nova_percent_fields_raises_value_error(self, mock_hub_metadata):
        """Overrides with no nova_*_percent fields should raise ValueError."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()
        # Overrides with no nova_*_percent fields
        self._setup_s3_responses(
            s3_client,
            overrides={"some_param": {"default": "value"}, "another_param": 42},
        )

        with pytest.raises(ValueError, match="No Nova data category fields found"):
            resolve_hyperpod_datamix_context(
                model_name="nova-pro",
                is_multimodal=False,
                sagemaker_session=session,
                training_type="LORA",
                customization_technique="SFT",
            )

    def test_customer_id_placeholder_resolution(self, mock_hub_metadata):
        """S3 URIs containing {customer_id} should be resolved with the actual account ID."""
        from unittest.mock import MagicMock, call
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        mock_get_hub, _ = mock_hub_metadata
        mock_get_hub.return_value = {
            "hub_content_document": {
                "RecipeCollection": [
                    {
                        "Name": "nova_pro_sft_lora_text_with_datamix",
                        "CustomizationTechnique": "SFT",
                        "HpEksPayloadTemplateS3Uri": "s3://bucket-{customer_id}/templates/recipe.yaml",
                        "HpEksOverrideParamsS3Uri": "s3://bucket-{customer_id}/overrides/params.json",
                    }
                ]
            }
        }

        session, s3_client, _ = self._make_mock_session()
        self._setup_s3_responses(s3_client)

        resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

        # Verify the S3 calls used the resolved account ID (123456789012)
        calls = s3_client.get_object.call_args_list
        assert len(calls) == 2
        # The bucket name should contain the resolved account ID, not the placeholder
        first_call_bucket = calls[0][1]["Bucket"] if "Bucket" in calls[0][1] else calls[0][0][0]
        assert "123456789012" in str(calls[0])
        assert "{customer_id}" not in str(calls[0])
        assert "{customer_id}" not in str(calls[1])

    def test_image_uri_extracted_when_present(self, mock_hub_metadata):
        """Image URI should be extracted from Helm chart template when present."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()
        self._setup_s3_responses(s3_client, template_content=self.MOCK_TEMPLATE_WITH_IMAGE)

        ctx = resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

        assert ctx.image_uri == "123456789012.dkr.ecr.us-west-2.amazonaws.com/train:latest"

    def test_image_uri_none_when_absent(self, mock_hub_metadata):
        """Image URI should be None when not found in template."""
        from sagemaker.train.common_utils.data_mixing_utils import (
            resolve_hyperpod_datamix_context,
        )

        session, s3_client, _ = self._make_mock_session()
        # Template without any image URI pattern
        plain_template = "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\n"
        self._setup_s3_responses(s3_client, template_content=plain_template)

        ctx = resolve_hyperpod_datamix_context(
            model_name="nova-pro",
            is_multimodal=False,
            sagemaker_session=session,
            training_type="LORA",
            customization_technique="SFT",
        )

        assert ctx.image_uri is None


class TestBuildHyperPodDatamixRecipeFromContext:
    MOCK_HELM_TEMPLATE = (
        "---\n"
        "# Source: nova-training/templates/training-config.yaml\n"
        "apiVersion: v1\n"
        "kind: ConfigMap\n"
        "metadata:\n"
        "  name: training-config\n"
        "data:\n"
        "  config.yaml: |-\n"
        "    model:\n"
        "      name: nova-pro\n"
        "    data_mixing:\n"
        "      sources:\n"
        "        customer_data:\n"
        "          percent: '{{percent}}'\n"
        "        nova_data:\n"
        "          code: '{{code}}'\n"
        "          math: '{{math}}'\n"
        "          en-entertainment: '{{en-entertainment}}'\n"
        "          en-scientific: '{{en-scientific}}'\n"
        "    optimizer:\n"
        "      name: distributed_fused_adam\n"
        "      lr: '{{lr}}'\n"
        "---\n"
    )

    MOCK_OVERRIDES = {
        "nova_code_percent": {"default": 25.0, "type": "float"},
        "nova_math_percent": {"default": 25.0, "type": "float"},
        "nova_en-entertainment_percent": {"default": 25.0, "type": "float"},
        "nova_en-scientific_percent": {"default": 25.0, "type": "float"},
        "percent": {"default": 50.0, "type": "float"},
        "lr": {"default": 0.001, "type": "float"},
    }

    MOCK_CATEGORIES = {
        "code": 25.0,
        "math": 25.0,
        "en-entertainment": 25.0,
        "en-scientific": 25.0,
    }

    IMAGE_URI = "123456789012.dkr.ecr.us-west-2.amazonaws.com/train:latest"

    def _make_context(self, **overrides):
        """Helper to construct a HyperPodTemplateContext with realistic defaults."""
        defaults = {
            "raw_template": self.MOCK_HELM_TEMPLATE,
            "overrides": dict(self.MOCK_OVERRIDES),
            "recipe_name": "nova_pro_sft_lora_text_with_datamix",
            "datamix_keyword": "text_with_datamix",
            "categories": dict(self.MOCK_CATEGORIES),
            "image_uri": self.IMAGE_URI,
        }
        defaults.update(overrides)
        return HyperPodTemplateContext(**defaults)

    def _make_validated_config(self, customer_data_percent=70.0, nova_data_percentages=None):
        """Helper to build a validated DataMixingConfig."""
        if nova_data_percentages is None:
            nova_data_percentages = {
                "code": 40.0,
                "math": 30.0,
                "en-entertainment": 20.0,
                "en-scientific": 10.0,
            }
        return DataMixingConfig(
            customer_data_percent=customer_data_percent,
            nova_data_percentages=nova_data_percentages,
        )

    def test_successful_build_writes_correct_data_mixing_sources(self):
        """Build should write YAML with correct data_mixing.sources.customer_data.percent
        and data_mixing.sources.nova_data values."""
        import yaml
        from io import StringIO
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        config = self._make_validated_config(
            customer_data_percent=70.0,
            nova_data_percentages={
                "code": 40.0,
                "math": 30.0,
                "en-entertainment": 20.0,
                "en-scientific": 10.0,
            },
        )

        written_content = {}
        real_open = open

        def capture_write(path, mode="r", **kwargs):
            """Capture content written to files."""
            if mode == "w":
                sio = StringIO()
                sio.name = path
                written_content["path"] = path
                written_content["buffer"] = sio
                sio.real_close = sio.close
                sio.close = lambda: None
                return sio
            return real_open(path, mode, **kwargs)

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", side_effect=capture_write):
                    build_hyperpod_datamix_recipe_from_context(context, config)

        # Parse the written YAML and verify data_mixing.sources values
        assert "buffer" in written_content
        written_yaml = written_content["buffer"].getvalue()
        parsed = yaml.safe_load(written_yaml)

        assert parsed["data_mixing"]["sources"]["customer_data"]["percent"] == 70
        assert parsed["data_mixing"]["sources"]["nova_data"]["code"] == 40
        assert parsed["data_mixing"]["sources"]["nova_data"]["math"] == 30
        assert parsed["data_mixing"]["sources"]["nova_data"]["en-entertainment"] == 20
        assert parsed["data_mixing"]["sources"]["nova_data"]["en-scientific"] == 10

    def test_return_value_is_absolute_recipe_path_and_image_uri(self):
        """Return value should be (absolute_recipe_path_with_extension, image_uri).

        The path must be a loadable filesystem path (absolute, .yaml extension
        intact) because the recipe resolver opens it as a user recipe file.
        """
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        config = self._make_validated_config()

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        m_open = mock_open()
        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", m_open):
                    result = build_hyperpod_datamix_recipe_from_context(context, config)

        recipe_path, image_uri = result

        # recipe_path must be an absolute filesystem path
        assert os.path.isabs(recipe_path)
        # recipe_path must retain the .yaml extension so it can be opened
        assert recipe_path.endswith(".yaml")
        # recipe_path should live under the HyperPod CLI recipes fine-tuning/nova dir
        assert "fine-tuning/nova" in recipe_path
        # image_uri should match the context
        assert image_uri == self.IMAGE_URI

    def test_no_s3_calls_made_during_build(self):
        """Build phase must not make any S3 API calls."""
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        config = self._make_validated_config()

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        mock_boto3_client = MagicMock()

        m_open = mock_open()
        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", m_open):
                    with patch("boto3.client", mock_boto3_client):
                        build_hyperpod_datamix_recipe_from_context(context, config)

        # Assert no boto3 client was called
        mock_boto3_client.assert_not_called()

    def test_missing_hyperpod_cli_raises_runtime_error(self):
        """Missing hyperpod_cli package should raise RuntimeError with install guidance."""
        import sys
        import importlib
        from unittest.mock import patch

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        config = self._make_validated_config()

        # Remove hyperpod_cli from sys.modules if present, and make import fail
        original_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

        def mock_import(name, *args, **kwargs):
            if name == "hyperpod_cli":
                raise ModuleNotFoundError("No module named 'hyperpod_cli'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(RuntimeError, match="HyperPod CLI is a required dependency"):
                build_hyperpod_datamix_recipe_from_context(context, config)

    def test_malformed_yaml_raises_value_error(self):
        """Template content that cannot be parsed as YAML should raise ValueError."""
        from unittest.mock import patch, MagicMock

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        # Create a template with a valid Helm chart structure but malformed YAML in the config section
        malformed_template = (
            "---\n"
            "# Source: nova-training/templates/training-config.yaml\n"
            "apiVersion: v1\n"
            "kind: ConfigMap\n"
            "data:\n"
            "  config.yaml: |-\n"
            "    model:\n"
            "      name: nova-pro\n"
            "      invalid_yaml: [unclosed bracket\n"
            "      bad: : colon\n"
            "---\n"
        )

        context = self._make_context(raw_template=malformed_template)
        config = self._make_validated_config()

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with pytest.raises(ValueError, match="malformed"):
                build_hyperpod_datamix_recipe_from_context(context, config)

    def test_nonzero_category_not_in_template_raises_value_error_with_sorted_names(self):
        """Non-zero category not present in template nova_data should raise ValueError
        with sorted unsupported and sorted valid category names."""
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        # Create config with a non-zero category that doesn't exist in the template
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={
                "code": 30.0,
                "math": 30.0,
                "en-entertainment": 20.0,
                "en-scientific": 10.0,
                "nonexistent-cat": 10.0,  # This doesn't exist in template
            },
        )

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        m_open = mock_open()
        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", m_open):
                    with pytest.raises(ValueError) as exc_info:
                        build_hyperpod_datamix_recipe_from_context(context, config)

        error_msg = str(exc_info.value)
        # Should mention the unsupported category (sorted)
        assert "nonexistent-cat" in error_msg
        # Should list valid categories (sorted)
        assert "code" in error_msg
        assert "en-entertainment" in error_msg
        assert "en-scientific" in error_msg
        assert "math" in error_msg

    def test_customer_data_100_with_non_summing_nova_does_not_raise(self):
        """When customer_data_percent=100, nova percentages that don't sum to 100
        should NOT raise in the build phase (sum validation is Pydantic's job, and it
        skips when customer_data_percent=100)."""
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
        )

        context = self._make_context()
        # customer_data_percent=100 skips sum validation in Pydantic,
        # so we can pass non-summing nova values
        config = DataMixingConfig(
            customer_data_percent=100.0,
            nova_data_percentages={
                "code": 10.0,
                "math": 10.0,
                "en-entertainment": 5.0,
                "en-scientific": 5.0,
            },
        )

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        m_open = mock_open()
        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", m_open):
                    # Should not raise any error
                    result = build_hyperpod_datamix_recipe_from_context(context, config)

        # Should still return a valid tuple
        assert result is not None
        assert len(result) == 2

    def test_nova_data_percentages_none_fills_template_defaults(self):
        """When nova_data_percentages is None (after validate_data_mixing_categories
        populates defaults), the build should use those defaults in the output."""
        import yaml
        from io import StringIO
        from unittest.mock import patch, MagicMock, mock_open

        from sagemaker.train.common_utils.data_mixing_utils import (
            build_hyperpod_datamix_recipe_from_context,
            validate_data_mixing_categories,
        )

        context = self._make_context()

        # Simulate what validate_data_mixing_categories does when nova_data_percentages=None:
        # It returns a config with all recipe defaults populated
        input_config = DataMixingConfig(
            customer_data_percent=50.0,
            nova_data_percentages=None,
        )
        validated_config = validate_data_mixing_categories(input_config, self.MOCK_CATEGORIES)

        # validated_config now has all defaults filled in
        assert validated_config.nova_data_percentages == self.MOCK_CATEGORIES

        written_content = {}
        real_open = open

        def capture_write(path, mode="r", **kwargs):
            from io import StringIO as SIO
            if mode == "w":
                sio = SIO()
                sio.name = path
                written_content["buffer"] = sio
                sio.close = lambda: None
                return sio
            return real_open(path, mode, **kwargs)

        mock_hyperpod_cli = MagicMock()
        mock_hyperpod_cli.__file__ = "/fake/hyperpod_cli/__init__.py"

        with patch.dict("sys.modules", {"hyperpod_cli": mock_hyperpod_cli}):
            with patch("os.makedirs"):
                with patch("builtins.open", side_effect=capture_write):
                    build_hyperpod_datamix_recipe_from_context(context, validated_config)

        # Parse the written YAML and verify defaults are present
        assert "buffer" in written_content
        written_yaml = written_content["buffer"].getvalue()
        parsed = yaml.safe_load(written_yaml)

        nova_data = parsed["data_mixing"]["sources"]["nova_data"]
        assert nova_data["code"] == 25
        assert nova_data["math"] == 25
        assert nova_data["en-entertainment"] == 25
        assert nova_data["en-scientific"] == 25
