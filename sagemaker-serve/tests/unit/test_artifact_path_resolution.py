"""
Unit tests for ModelBuilder artifact path resolution.
Tests the _resolve_model_artifact_uri method with various scenarios.

Requirements: 7.3
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode


class TestArtifactPathResolution(unittest.TestCase):
    """Test artifact path resolution - Requirements 7.3"""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.settings = Mock()
        self.mock_session.settings.include_jumpstart_tags = False
        
        mock_credentials = Mock()
        mock_credentials.access_key = "test-key"
        mock_credentials.secret_key = "test-secret"
        mock_credentials.token = None
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.get_credentials.return_value = mock_credentials
        self.mock_session.boto_session.region_name = "us-west-2"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_base_model_artifact_uri_retrieval(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_hub
    ):
        """Test base model artifact URI retrieval from JumpStart metadata."""
        # Setup: Base model (not LORA, not full fine-tuned)
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"  # Not LORA
        
        # Setup: Model package with base_model but no model_data_source
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        # No model_data_source attribute (base model)
        mock_container.model_data_source = None
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document with HostingArtifactUri
        mock_fetch_hub.return_value = {
            'HostingArtifactUri': 's3://jumpstart-bucket/base-model/artifacts.tar.gz'
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return HostingArtifactUri from JumpStart
        assert artifact_uri == 's3://jumpstart-bucket/base-model/artifacts.tar.gz'
        assert mock_fetch_hub.called

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_fine_tuned_adapter_artifact_location(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package
    ):
        """Test fine-tuned adapter artifact location from ModelPackage."""
        # Setup: Full fine-tuned model (not LORA)
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Model package with model_data_source (fine-tuned model)
        mock_package = Mock()
        mock_container = Mock()
        mock_s3_data_source = Mock()
        mock_s3_data_source.s3_uri = 's3://my-bucket/fine-tuned-model/model.tar.gz'
        mock_model_data_source = Mock()
        mock_model_data_source.s3_data_source = mock_s3_data_source
        mock_container.model_data_source = mock_model_data_source
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return model_data_source S3 URI
        assert artifact_uri == 's3://my-bucket/fine-tuned-model/model.tar.gz'

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_lora_adapter_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft
    ):
        """Test that LORA adapters return None (no artifact URI needed)."""
        # Setup: LORA adapter
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "LORA"
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None for LORA adapters
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_missing_hosting_artifact_uri_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_hub
    ):
        """Test error handling when HostingArtifactUri is missing from metadata."""
        # Setup: Base model without HostingArtifactUri
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Model package with base_model but no model_data_source
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_container.model_data_source = None
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document WITHOUT HostingArtifactUri
        mock_fetch_hub.return_value = {}
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None and log warning
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_hub_document_fetch_exception_handling(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_hub
    ):
        """Test error handling when hub document fetch fails."""
        # Setup: Base model
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Model package with base_model
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_container.model_data_source = None
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document fetch raises exception
        mock_fetch_hub.side_effect = Exception("Hub service unavailable")
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute - should not raise exception
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None and log warning
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_non_model_customization_returns_none(
        self,
        mock_is_model_customization
    ):
        """Test that non-model-customization deployments return None."""
        # Setup: Not a model customization deployment
        mock_is_model_customization.return_value = False
        
        builder = ModelBuilder(
            model="my-model",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None for non-model-customization
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_missing_model_package_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package
    ):
        """Test error handling when model package is not available."""
        # Setup: Model customization but no model package
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        mock_fetch_package.return_value = None
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None when model package is unavailable
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_missing_inference_specification_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package
    ):
        """Test error handling when model package has no inference specification."""
        # Setup: Model package without inference_specification
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        mock_package = Mock()
        mock_package.inference_specification = None
        mock_fetch_package.return_value = mock_package
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None when inference_specification is missing
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_empty_containers_list_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package
    ):
        """Test error handling when containers list is empty."""
        # Setup: Model package with empty containers list
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        mock_package = Mock()
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = []
        mock_fetch_package.return_value = mock_package
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None when containers list is empty
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_base_model_without_base_model_attribute_returns_none(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_hub
    ):
        """Test error handling when container has no base_model attribute."""
        # Setup: Container without base_model attribute
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        mock_package = Mock()
        mock_container = Mock()
        mock_container.model_data_source = None
        mock_container.base_model = None  # No base_model
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return None when base_model is not present
        assert artifact_uri is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_fine_tuned_model_with_nested_s3_data_source(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package
    ):
        """Test fine-tuned model with properly nested s3_data_source structure."""
        # Setup: Full fine-tuned model with nested structure
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Properly nested model_data_source structure
        mock_package = Mock()
        mock_container = Mock()
        
        # Create nested structure: container -> model_data_source -> s3_data_source -> s3_uri
        mock_s3_data_source = Mock()
        mock_s3_data_source.s3_uri = 's3://custom-bucket/my-fine-tuned-model/artifacts.tar.gz'
        
        mock_model_data_source = Mock()
        mock_model_data_source.s3_data_source = mock_s3_data_source
        
        mock_container.model_data_source = mock_model_data_source
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return the nested S3 URI
        assert artifact_uri == 's3://custom-bucket/my-fine-tuned-model/artifacts.tar.gz'

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_base_model_with_multiple_hosting_artifact_uris(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_hub
    ):
        """Test base model retrieval when hub document has HostingArtifactUri."""
        # Setup: Base model
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Model package with base_model
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_container.model_data_source = None
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document with HostingArtifactUri
        mock_fetch_hub.return_value = {
            'HostingArtifactUri': 's3://jumpstart-cache/base-model-v2/model.tar.gz',
            'HostingEcrUri': '123456789012.dkr.ecr.us-west-2.amazonaws.com/jumpstart:latest'
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        artifact_uri = builder._resolve_model_artifact_uri()
        
        # Verify: Should return HostingArtifactUri
        assert artifact_uri == 's3://jumpstart-cache/base-model-v2/model.tar.gz'


if __name__ == "__main__":
    unittest.main()
