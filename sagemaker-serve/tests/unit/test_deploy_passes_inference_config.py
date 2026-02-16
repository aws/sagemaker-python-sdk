"""
Test to verify that deploy() method passes inference_config to _deploy_model_customization.
This test validates task 4.4 requirements.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.inference_config import ResourceRequirements  # Correct import!


class TestDeployPassesInferenceConfig(unittest.TestCase):
    """Test that deploy() correctly passes inference_config to _deploy_model_customization."""

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

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_default_instance_type_for_custom_model')
    def test_deploy_passes_inference_config_to_deploy_model_customization(
        self,
        mock_fetch_default_instance,
        mock_is_model_customization,
        mock_deploy_model_customization
    ):
        """Test that deploy() passes inference_config parameter to _deploy_model_customization."""
        # Setup: Mock model customization check
        mock_is_model_customization.return_value = True
        mock_fetch_default_instance.return_value = "ml.g5.12xlarge"
        
        # Setup: Mock _deploy_model_customization to return a mock endpoint
        mock_endpoint = Mock()
        mock_deploy_model_customization.return_value = mock_endpoint
        
        # Create ModelBuilder
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        )
        
        # Mark as built
        builder.built_model = Mock()
        
        # Create inference_config
        inference_config = ResourceRequirements(
            requests={
                "num_cpus": 8,
                "memory": 16384,
                "num_accelerators": 4
            }
        )
        
        # Execute: Call deploy() with inference_config
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=inference_config,
            initial_instance_count=1,
            wait=True
        )
        
        # Verify: _deploy_model_customization was called with inference_config
        assert mock_deploy_model_customization.called
        call_kwargs = mock_deploy_model_customization.call_args[1]
        
        # Verify inference_config was passed through
        assert 'inference_config' in call_kwargs
        assert call_kwargs['inference_config'] == inference_config
        
        # Verify other parameters were also passed
        assert call_kwargs['endpoint_name'] == "test-endpoint"
        assert call_kwargs['initial_instance_count'] == 1
        assert call_kwargs['wait'] == True
        
        # Verify the result is the mock endpoint
        assert result == mock_endpoint

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_default_instance_type_for_custom_model')
    def test_deploy_passes_none_when_inference_config_not_provided(
        self,
        mock_fetch_default_instance,
        mock_is_model_customization,
        mock_deploy_model_customization
    ):
        """Test backward compatibility: deploy() passes None when inference_config not provided."""
        # Setup
        mock_is_model_customization.return_value = True
        mock_fetch_default_instance.return_value = "ml.g5.12xlarge"
        mock_endpoint = Mock()
        mock_deploy_model_customization.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        )
        
        builder.built_model = Mock()
        
        # Execute: Call deploy() WITHOUT inference_config
        result = builder.deploy(
            endpoint_name="test-endpoint",
            initial_instance_count=1
        )
        
        # Verify: _deploy_model_customization was called with inference_config=None
        assert mock_deploy_model_customization.called
        call_kwargs = mock_deploy_model_customization.call_args[1]
        
        # Verify inference_config is None (backward compatibility)
        assert 'inference_config' in call_kwargs
        assert call_kwargs['inference_config'] is None

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_default_instance_type_for_custom_model')
    def test_deploy_only_passes_resource_requirements_type(
        self,
        mock_fetch_default_instance,
        mock_is_model_customization,
        mock_deploy_model_customization
    ):
        """Test that deploy() only passes inference_config if it's ResourceRequirements type."""
        # Setup
        mock_is_model_customization.return_value = True
        mock_fetch_default_instance.return_value = "ml.g5.12xlarge"
        mock_endpoint = Mock()
        mock_deploy_model_customization.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"
        )
        
        builder.built_model = Mock()
        
        # Create a non-ResourceRequirements inference_config (e.g., ServerlessInferenceConfig)
        from sagemaker.core.inference_config import ServerlessInferenceConfig
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=4096,
            max_concurrency=10
        )
        
        # Execute: Call deploy() with ServerlessInferenceConfig
        # This should NOT pass it to _deploy_model_customization
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=serverless_config
        )
        
        # Verify: _deploy_model_customization was called with inference_config=None
        # because ServerlessInferenceConfig is not ResourceRequirements
        assert mock_deploy_model_customization.called
        call_kwargs = mock_deploy_model_customization.call_args[1]
        
        # Verify inference_config is None (not ServerlessInferenceConfig)
        assert 'inference_config' in call_kwargs
        assert call_kwargs['inference_config'] is None


if __name__ == "__main__":
    unittest.main()
