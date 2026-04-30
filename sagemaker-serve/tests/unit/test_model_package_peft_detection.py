"""Unit tests for ModelPackage LoRA detection in _fetch_peft() and related paths.

Tests verify that:
1. _fetch_peft() returns "LORA" for ModelPackage with lora recipe name
2. _fetch_peft() returns None for ModelPackage with non-lora recipe name
3. _fetch_peft() returns None for ModelPackage with no recipe name
4. _adapter_s3_uri is correctly set from ModelPackage container S3 URI
5. env vars are applied in the non-LoRA ContainerDefinition path
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.resources import ModelPackage, TrainingJob


class TestModelPackagePeftDetection:
    """Test _fetch_peft() behavior with ModelPackage input."""

    def _create_model_package_mock(self, recipe_name=None):
        """Helper to create a mock ModelPackage with a given recipe name."""
        mock_package = Mock(spec=ModelPackage)
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = recipe_name
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        return mock_package

    def test_fetch_peft_returns_lora_for_lora_recipe(self):
        """_fetch_peft() returns 'LORA' when recipe name contains 'lora'."""
        mock_package = self._create_model_package_mock(
            recipe_name="verl-grpo-rlvr-qwen-3-32b-lora"
        )
        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() == "LORA"

    def test_fetch_peft_returns_lora_case_insensitive(self):
        """_fetch_peft() matches 'lora' case-insensitively."""
        mock_package = self._create_model_package_mock(
            recipe_name="some-model-LoRA-adapter"
        )
        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() == "LORA"

    def test_fetch_peft_returns_none_for_fft_recipe(self):
        """_fetch_peft() returns None when recipe name does not contain 'lora'."""
        mock_package = self._create_model_package_mock(
            recipe_name="verl-grpo-rlvr-qwen-3-32b-fft"
        )
        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() is None

    def test_fetch_peft_returns_none_for_no_recipe_name(self):
        """_fetch_peft() returns None when recipe name is None."""
        mock_package = self._create_model_package_mock(recipe_name=None)
        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() is None

    def test_fetch_peft_returns_none_when_base_model_missing(self):
        """_fetch_peft() returns None when base_model attribute is missing."""
        mock_package = Mock(spec=ModelPackage)
        mock_container = Mock()
        mock_container.base_model = None
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]

        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() is None

    def test_fetch_peft_returns_none_when_containers_empty(self):
        """_fetch_peft() returns None when containers list is empty."""
        mock_package = Mock(spec=ModelPackage)
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = []

        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        assert builder._fetch_peft() is None


class TestModelPackageAdapterS3Uri:
    """Test _adapter_s3_uri is correctly set from ModelPackage."""

    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_fetch_hub_document_for_custom_model")
    @patch.object(ModelBuilder, "_fetch_and_cache_recipe_config")
    @patch.object(ModelBuilder, "_is_nova_model", return_value=False)
    @patch.object(ModelBuilder, "_is_model_customization")
    @patch("sagemaker.core.resources.Model.create")
    def test_adapter_s3_uri_set_from_model_package(
        self,
        mock_model_create,
        mock_is_customization,
        mock_is_nova_model,
        mock_fetch_and_cache_recipe,
        mock_fetch_hub,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
    ):
        """_adapter_s3_uri is set from ModelPackage container S3 URI for LORA."""
        mock_is_customization.return_value = True
        mock_fetch_peft.return_value = "LORA"

        expected_adapter_uri = "s3://bucket/adapter-weights/"

        mock_package = Mock(spec=ModelPackage)
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "verl-grpo-rlvr-qwen-3-32b-lora"
        mock_container.model_data_source = Mock()
        mock_container.model_data_source.s3_data_source = Mock()
        mock_container.model_data_source.s3_data_source.s3_uri = expected_adapter_uri
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        mock_fetch_hub.return_value = {
            "HostingArtifactUri": "s3://jumpstart-bucket/base-model-artifacts/"
        }

        mock_model_create.return_value = Mock()

        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        builder.accept_eula = True
        builder._build_single_modelbuilder()

        assert builder._adapter_s3_uri == expected_adapter_uri


class TestNonLoraEnvVars:
    """Test env vars are applied in the non-LoRA ContainerDefinition path."""

    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_fetch_and_cache_recipe_config")
    @patch.object(ModelBuilder, "_is_nova_model", return_value=False)
    @patch.object(ModelBuilder, "_is_model_customization")
    @patch("sagemaker.core.resources.Model.create")
    def test_env_vars_passed_to_non_lora_container_def(
        self,
        mock_model_create,
        mock_is_customization,
        mock_is_nova_model,
        mock_fetch_and_cache_recipe,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
    ):
        """Non-LoRA ContainerDefinition includes environment vars."""
        mock_is_customization.return_value = True
        mock_fetch_peft.return_value = None  # Not LORA

        mock_package = Mock(spec=ModelPackage)
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "verl-grpo-rlvr-qwen-3-32b-fft"
        mock_container.model_data_source = Mock()
        mock_container.model_data_source.s3_data_source = Mock()
        mock_container.model_data_source.s3_data_source.s3_uri = "s3://bucket/model/"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        mock_model_create.return_value = Mock()

        expected_env = {"SM_MODEL_ID": "test-model", "CUSTOM_VAR": "value"}

        builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
            env_vars=expected_env,
        )
        builder._build_single_modelbuilder()

        # Verify Model.create was called and the container has environment set
        assert mock_model_create.called
        create_call = mock_model_create.call_args
        containers = create_call[1].get("containers", [])
        assert len(containers) == 1
        container_def = containers[0]
        assert container_def.environment == expected_env
