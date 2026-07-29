"""Unit tests for Nova SMI config validation in ModelBuilder."""

import unittest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder


class TestNovaSmiValidation(unittest.TestCase):
    """Test _validate_nova_smi_config method using Tiers in _NOVA_HOSTING_CONFIGS."""

    def _create_builder_with_nova_model(self, hub_content_name, instance_type, env_vars=None):
        """Create a ModelBuilder with mocked Nova model internals."""
        with patch.object(ModelBuilder, "__init__", lambda self, **kwargs: None):
            builder = ModelBuilder()

        builder.env_vars = env_vars or {}
        builder.instance_type = instance_type

        mock_base_model = Mock()
        mock_base_model.hub_content_name = hub_content_name

        mock_container = Mock()
        mock_container.base_model = mock_base_model

        mock_inference_spec = Mock()
        mock_inference_spec.containers = [mock_container]

        mock_model_package = Mock()
        mock_model_package.inference_specification = mock_inference_spec

        builder._fetch_model_package = Mock(return_value=mock_model_package)

        return builder

    # --- Valid configs at various tiers ---

    def test_valid_config_micro_p5_low_context_high_concurrency(self):
        """Valid: context<=16000 allows up to 128 concurrency on micro/p5."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "128"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_micro_p5_mid_context(self):
        """Valid: context<=64000 allows up to 32 concurrency on micro/p5."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "64000", "MAX_CONCURRENCY": "32"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_micro_p5_high_context(self):
        """Valid: context<=128000 allows up to 8 concurrency on micro/p5."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "128000", "MAX_CONCURRENCY": "8"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_micro_g5_first_tier(self):
        """Valid: context<=4000 allows up to 12 concurrency on micro/g5.12xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.g5.12xlarge",
            env_vars={"CONTEXT_LENGTH": "4000", "MAX_CONCURRENCY": "12"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_micro_g5_second_tier(self):
        """Valid: context<=8000 allows up to 6 concurrency on micro/g5.12xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.g5.12xlarge",
            env_vars={"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "6"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_lite_g6_48xlarge_first_tier(self):
        """Valid: NOVA_LITE on ml.g6.48xlarge, context<=4000 allows 16 concurrency."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-lite",
            instance_type="ml.g6.48xlarge",
            env_vars={"CONTEXT_LENGTH": "4000", "MAX_CONCURRENCY": "16"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_pro_p5(self):
        """Valid: NOVA_PRO on ml.p5.48xlarge, context<=8000 allows 8 concurrency."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-pro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "8000", "MAX_CONCURRENCY": "8"},
        )
        builder._validate_nova_smi_config()

    def test_valid_config_lite_v2_p5(self):
        """Valid: NOVA_LITE_2 on ml.p5.48xlarge, context<=16000 allows 128."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-lite-v2",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "128"},
        )
        builder._validate_nova_smi_config()

    # --- Context length violations ---

    def test_exceeds_context_length_micro_p5(self):
        """CONTEXT_LENGTH exceeds max (128000) for NOVA_MICRO on ml.p5.48xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "200000", "MAX_CONCURRENCY": "1"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("CONTEXT_LENGTH=200000", str(ctx.exception))
        self.assertIn("128000", str(ctx.exception))

    def test_exceeds_context_length_micro_g5(self):
        """CONTEXT_LENGTH exceeds max (8000) for NOVA_MICRO on ml.g5.12xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.g5.12xlarge",
            env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "1"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("CONTEXT_LENGTH=16000", str(ctx.exception))
        self.assertIn("8000", str(ctx.exception))

    def test_exceeds_context_length_pro_p5(self):
        """CONTEXT_LENGTH exceeds max (24000) for NOVA_PRO on ml.p5.48xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-pro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "32000", "MAX_CONCURRENCY": "1"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("CONTEXT_LENGTH=32000", str(ctx.exception))
        self.assertIn("24000", str(ctx.exception))

    # --- Per-tier concurrency violations ---

    def test_exceeds_concurrency_micro_p5_low_tier(self):
        """context<=16000 allows max 128; 200 should fail."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "200"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=200", str(ctx.exception))
        self.assertIn("128", str(ctx.exception))

    def test_exceeds_concurrency_micro_p5_mid_tier(self):
        """context<=64000 allows max 32; 50 should fail."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "64000", "MAX_CONCURRENCY": "50"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=50", str(ctx.exception))
        self.assertIn("32", str(ctx.exception))

    def test_exceeds_concurrency_micro_p5_high_tier(self):
        """context<=128000 allows max 8; 10 should fail."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "100000", "MAX_CONCURRENCY": "10"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=10", str(ctx.exception))
        self.assertIn("8", str(ctx.exception))

    def test_exceeds_concurrency_micro_g5_first_tier(self):
        """context<=4000 allows max 12; 15 should fail on micro/g5.12xlarge."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.g5.12xlarge",
            env_vars={"CONTEXT_LENGTH": "3000", "MAX_CONCURRENCY": "15"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=15", str(ctx.exception))
        self.assertIn("12", str(ctx.exception))

    def test_exceeds_concurrency_lite_g6_first_tier(self):
        """NOVA_LITE on ml.g6.48xlarge: context<=4000 allows 16; 20 should fail."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-lite",
            instance_type="ml.g6.48xlarge",
            env_vars={"CONTEXT_LENGTH": "4000", "MAX_CONCURRENCY": "20"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=20", str(ctx.exception))
        self.assertIn("16", str(ctx.exception))

    def test_exceeds_concurrency_pro_p5_mid_tier(self):
        """NOVA_PRO on ml.p5.48xlarge: context<=16000 allows 2; 5 should fail."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-pro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "5"},
        )
        with self.assertRaises(ValueError) as ctx:
            builder._validate_nova_smi_config()
        self.assertIn("MAX_CONCURRENCY=5", str(ctx.exception))
        self.assertIn("2", str(ctx.exception))

    # --- Skip cases (no validation) ---

    def test_skips_when_context_length_missing(self):
        """No validation when CONTEXT_LENGTH is not in env_vars."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"MAX_CONCURRENCY": "999"},
        )
        builder._validate_nova_smi_config()

    def test_skips_when_max_concurrency_missing(self):
        """No validation when MAX_CONCURRENCY is not in env_vars."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "999999"},
        )
        builder._validate_nova_smi_config()

    def test_skips_when_env_vars_empty(self):
        """No validation when env_vars is empty."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.p5.48xlarge",
            env_vars={},
        )
        builder._validate_nova_smi_config()

    def test_skips_when_no_model_package(self):
        """No validation when model package cannot be fetched."""
        with patch.object(ModelBuilder, "__init__", lambda self, **kwargs: None):
            builder = ModelBuilder()
        builder.env_vars = {"CONTEXT_LENGTH": "999999", "MAX_CONCURRENCY": "999"}
        builder.instance_type = "ml.p5.48xlarge"
        builder._fetch_model_package = Mock(return_value=None)
        builder._validate_nova_smi_config()

    def test_skips_for_unknown_instance_type(self):
        """No validation when instance type is not in _NOVA_HOSTING_CONFIGS."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="nova-textgeneration-micro",
            instance_type="ml.g5.xlarge",
            env_vars={"CONTEXT_LENGTH": "999999", "MAX_CONCURRENCY": "999"},
        )
        builder._validate_nova_smi_config()

    def test_skips_for_unknown_model(self):
        """No validation when hub_content_name is not in _NOVA_HOSTING_CONFIGS."""
        builder = self._create_builder_with_nova_model(
            hub_content_name="some-other-model",
            instance_type="ml.p5.48xlarge",
            env_vars={"CONTEXT_LENGTH": "999999", "MAX_CONCURRENCY": "999"},
        )
        builder._validate_nova_smi_config()


class TestNovaSmiValidationIntegrationWithFetchConfig(unittest.TestCase):
    """Test that validation is called during _fetch_and_cache_recipe_config for Nova models."""

    def _create_builder_for_fetch_config(self, user_env_vars=None):
        """Create a builder that exercises the _fetch_and_cache_recipe_config Nova path."""
        with patch.object(ModelBuilder, "__init__", lambda self, **kwargs: None):
            builder = ModelBuilder()

        builder.env_vars = user_env_vars if user_env_vars is not None else {}
        builder.instance_type = "ml.p5.48xlarge"
        builder.image_uri = None
        builder.s3_upload_path = None
        builder.sagemaker_session = Mock()
        builder.sagemaker_session.boto_region_name = "us-east-1"
        builder._user_provided_instance_type = True

        mock_base_model = Mock()
        mock_base_model.hub_content_name = "nova-textgeneration-micro"
        mock_base_model.recipe_name = "nova-micro-recipe"

        mock_container = Mock()
        mock_container.base_model = mock_base_model

        mock_inference_spec = Mock()
        mock_inference_spec.containers = [mock_container]

        mock_model_package = Mock()
        mock_model_package.inference_specification = mock_inference_spec

        builder._fetch_model_package = Mock(return_value=mock_model_package)
        builder._fetch_model_package_arn = Mock(
            return_value="arn:aws:sagemaker:us-east-1:123456789012:model-package/test"
        )
        builder._fetch_hub_document_for_custom_model = Mock(
            return_value={"RecipeCollection": []}
        )

        return builder

    def test_user_override_exceeds_context_raises_at_config_time(self):
        """User CONTEXT_LENGTH override that exceeds bounds should raise during config fetch."""
        builder = self._create_builder_for_fetch_config(
            user_env_vars={"CONTEXT_LENGTH": "200000", "MAX_CONCURRENCY": "1"}
        )

        with self.assertRaises(ValueError) as ctx:
            builder._fetch_and_cache_recipe_config()
        self.assertIn("CONTEXT_LENGTH=200000", str(ctx.exception))

    def test_user_override_exceeds_concurrency_raises_at_config_time(self):
        """User MAX_CONCURRENCY override that exceeds tier bounds should raise."""
        builder = self._create_builder_for_fetch_config(
            user_env_vars={"CONTEXT_LENGTH": "64000", "MAX_CONCURRENCY": "50"}
        )

        with self.assertRaises(ValueError) as ctx:
            builder._fetch_and_cache_recipe_config()
        self.assertIn("MAX_CONCURRENCY=50", str(ctx.exception))
        self.assertIn("32", str(ctx.exception))

    def test_user_override_valid_passes_config(self):
        """Valid user env var overrides should pass config fetch without error."""
        builder = self._create_builder_for_fetch_config(
            user_env_vars={"CONTEXT_LENGTH": "16000", "MAX_CONCURRENCY": "64"}
        )

        builder._fetch_and_cache_recipe_config()
        self.assertEqual(builder.env_vars["CONTEXT_LENGTH"], "16000")
        self.assertEqual(builder.env_vars["MAX_CONCURRENCY"], "64")

    def test_user_overrides_take_priority_over_defaults(self):
        """User-provided env vars should override Nova defaults."""
        builder = self._create_builder_for_fetch_config(
            user_env_vars={"CONTEXT_LENGTH": "64000", "MAX_CONCURRENCY": "32"}
        )

        builder._fetch_and_cache_recipe_config()
        self.assertEqual(builder.env_vars["CONTEXT_LENGTH"], "64000")
        self.assertEqual(builder.env_vars["MAX_CONCURRENCY"], "32")

    def test_no_user_overrides_uses_defaults(self):
        """Without user overrides, Nova defaults are used."""
        builder = self._create_builder_for_fetch_config(user_env_vars=None)

        builder._fetch_and_cache_recipe_config()
        self.assertEqual(builder.env_vars["CONTEXT_LENGTH"], "128000")
        self.assertEqual(builder.env_vars["MAX_CONCURRENCY"], "8")


if __name__ == "__main__":
    unittest.main()
