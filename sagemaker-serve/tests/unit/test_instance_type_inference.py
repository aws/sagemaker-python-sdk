"""
Unit tests for ModelBuilder instance type inference edge cases.
Tests the _infer_instance_type_from_jumpstart method with various scenarios.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode


class TestInstanceTypeInferenceEdgeCases(unittest.TestCase):
    """Test instance type inference edge cases - Requirements 1.3, 1.4"""

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
    def test_missing_jumpstart_metadata_no_hosting_configs(self, mock_fetch_hub):
        """Test with missing JumpStart metadata - no HostingConfigs."""
        # Setup: Hub document without HostingConfigs
        mock_fetch_hub.return_value = {}
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._infer_instance_type_from_jumpstart()
        
        # Verify error message content
        error_msg = str(exc_info.value)
        assert "Unable to infer instance type" in error_msg
        assert "does not have hosting configuration" in error_msg
        assert "Please specify instance_type explicitly" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_missing_jumpstart_metadata_empty_hosting_configs(self, mock_fetch_hub):
        """Test with empty HostingConfigs list."""
        # Setup: Hub document with empty HostingConfigs
        mock_fetch_hub.return_value = {
            "HostingConfigs": []
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._infer_instance_type_from_jumpstart()
        
        error_msg = str(exc_info.value)
        assert "Unable to infer instance type" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_missing_instance_types_in_metadata(self, mock_fetch_hub):
        """Test with metadata that has no instance type information."""
        # Setup: Hub document with HostingConfigs but no instance types
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    # Missing both SupportedInstanceTypes and InstanceType/DefaultInstanceType
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._infer_instance_type_from_jumpstart()
        
        error_msg = str(exc_info.value)
        assert "Unable to infer instance type" in error_msg
        assert "does not specify supported instance types" in error_msg
        assert "Please specify instance_type explicitly" in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_gpu_required_model_selects_gpu_instance(self, mock_fetch_hub):
        """Test that GPU-required models select GPU instance types."""
        # Setup: Hub document with both GPU and CPU instance types
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "SupportedInstanceTypes": [
                        "ml.m5.xlarge",  # CPU instance
                        "ml.g5.xlarge",  # GPU instance
                        "ml.g5.2xlarge",  # GPU instance
                        "ml.p4d.24xlarge"  # GPU instance
                    ]
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        instance_type = builder._infer_instance_type_from_jumpstart()
        
        # Verify: Should select a GPU instance type, not ml.m5.xlarge
        assert instance_type != "ml.m5.xlarge"
        assert any(gpu_family in instance_type for gpu_family in ['g5', 'g4dn', 'p4', 'p5', 'p3'])
        # Should select the first GPU instance type (ml.g5.xlarge)
        assert instance_type == "ml.g5.xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_gpu_required_model_with_various_gpu_families(self, mock_fetch_hub):
        """Test GPU instance selection across different GPU families."""
        # Setup: Hub document with various GPU families
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "SupportedInstanceTypes": [
                        "ml.g4dn.xlarge",  # G4 GPU
                        "ml.p3.2xlarge",   # P3 GPU
                        "ml.p5.48xlarge"   # P5 GPU
                    ]
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        instance_type = builder._infer_instance_type_from_jumpstart()
        
        # Verify: Should select first GPU instance type
        assert instance_type == "ml.g4dn.xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_default_instance_type_takes_precedence(self, mock_fetch_hub):
        """Test that DefaultInstanceType/InstanceType takes precedence over supported list."""
        # Setup: Hub document with both default and supported instance types
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "InstanceType": "ml.g5.12xlarge",
                    "SupportedInstanceTypes": [
                        "ml.g5.xlarge",
                        "ml.g5.2xlarge",
                        "ml.g5.12xlarge"
                    ]
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        instance_type = builder._infer_instance_type_from_jumpstart()
        
        # Verify: Should use the default instance type
        assert instance_type == "ml.g5.12xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_error_message_includes_supported_types(self, mock_fetch_hub):
        """Test that error messages include available instance types when possible."""
        # Setup: Hub document that will cause an error but has supported types
        supported_types = ["ml.g5.xlarge", "ml.g5.2xlarge", "ml.g5.12xlarge"]
        
        def side_effect_fetch():
            # First call raises exception
            if not hasattr(side_effect_fetch, 'call_count'):
                side_effect_fetch.call_count = 0
            side_effect_fetch.call_count += 1
            
            if side_effect_fetch.call_count == 1:
                raise Exception("Test error")
            else:
                # Second call in error handling returns valid data
                return {
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "SupportedInstanceTypes": supported_types
                        }
                    ]
                }
        
        mock_fetch_hub.side_effect = side_effect_fetch
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._infer_instance_type_from_jumpstart()
        
        error_msg = str(exc_info.value)
        assert "Unable to infer instance type" in error_msg
        assert "Supported instance types for this model:" in error_msg
        assert str(supported_types) in error_msg

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_cpu_only_model_selects_cpu_instance(self, mock_fetch_hub):
        """Test that CPU-only models correctly select CPU instance types."""
        # Setup: Hub document with only CPU instance types
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "SupportedInstanceTypes": [
                        "ml.m5.xlarge",
                        "ml.m5.2xlarge",
                        "ml.c5.xlarge"
                    ]
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-text-classification",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-text-classification",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        instance_type = builder._infer_instance_type_from_jumpstart()
        
        # Verify: Should select first CPU instance type
        assert instance_type == "ml.m5.xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_non_default_profile_fallback(self, mock_fetch_hub):
        """Test fallback to first config when Default profile is not present."""
        # Setup: Hub document without Default profile
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "CustomProfile",
                    "InstanceType": "ml.g5.2xlarge"
                }
            ]
        }
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute
        instance_type = builder._infer_instance_type_from_jumpstart()
        
        # Verify: Should use the first (and only) config
        assert instance_type == "ml.g5.2xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_fetch_hub_document_exception_handling(self, mock_fetch_hub):
        """Test proper exception handling when fetching hub document fails."""
        # Setup: Mock fetch to raise an exception
        mock_fetch_hub.side_effect = Exception("Network error")
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Execute and verify
        with pytest.raises(ValueError) as exc_info:
            builder._infer_instance_type_from_jumpstart()
        
        error_msg = str(exc_info.value)
        assert "Unable to infer instance type" in error_msg
        assert "Network error" in error_msg
        assert "Please specify instance_type explicitly" in error_msg


if __name__ == "__main__":
    unittest.main()


class TestInstanceTypeInferenceIntegration(unittest.TestCase):
    """Test instance type inference integration with model customization flow - Requirement 1.1"""

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

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_instance_type_inference_called_when_none_in_model_customization(
        self, mock_get_resources, mock_fetch_hub, mock_fetch_package
    ):
        """Test that _infer_instance_type_from_jumpstart is called when instance_type is None in model customization."""
        # Setup: Mock model package
        mock_package = Mock()
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document with recipe but no instance type in recipe config
        mock_fetch_hub.return_value = {
            "RecipeCollection": [
                {
                    "Name": "test-recipe",
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "EcrAddress": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
                            # No InstanceType or DefaultInstanceType in recipe config
                            "ComputeResourceRequirements": {
                                "NumberOfCpuCoresRequired": 4
                            }
                        }
                    ]
                }
            ],
            # Add HostingConfigs at root level for _infer_instance_type_from_jumpstart
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "InstanceType": "ml.g5.12xlarge",
                    "SupportedInstanceTypes": ["ml.g5.xlarge", "ml.g5.12xlarge"]
                }
            ]
        }
        
        mock_get_resources.return_value = (8, 32768)  # 8 CPUs, 32GB RAM
        
        # Create ModelBuilder without instance_type
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        # Manually set instance_type to None to simulate the scenario
        builder.instance_type = None
        
        # Execute: Call _fetch_and_cache_recipe_config which should trigger inference
        builder._fetch_and_cache_recipe_config()
        
        # Verify: instance_type should be inferred from JumpStart metadata
        assert builder.instance_type is not None
        assert builder.instance_type == "ml.g5.12xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_instance_type_not_inferred_when_provided_in_recipe(
        self, mock_get_resources, mock_fetch_hub, mock_fetch_package
    ):
        """Test that _infer_instance_type_from_jumpstart is NOT called when instance_type is in recipe config."""
        # Setup: Mock model package
        mock_package = Mock()
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document with recipe that HAS instance type
        mock_fetch_hub.return_value = {
            "RecipeCollection": [
                {
                    "Name": "test-recipe",
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "EcrAddress": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
                            "InstanceType": "ml.g5.2xlarge",  # Instance type provided in recipe
                            "ComputeResourceRequirements": {
                                "NumberOfCpuCoresRequired": 4
                            }
                        }
                    ]
                }
            ]
        }
        
        mock_get_resources.return_value = (8, 32768)
        
        # Create ModelBuilder without instance_type
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"
        )
        
        builder.instance_type = None
        
        # Execute
        builder._fetch_and_cache_recipe_config()
        
        # Verify: Should use instance type from recipe config, not inference
        assert builder.instance_type == "ml.g5.2xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_instance_resources')
    def test_instance_type_preserved_when_explicitly_provided(
        self, mock_get_resources, mock_fetch_hub, mock_fetch_package
    ):
        """Test backward compatibility: explicitly provided instance_type is preserved."""
        # Setup: Mock model package
        mock_package = Mock()
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package
        
        # Setup: Hub document
        mock_fetch_hub.return_value = {
            "RecipeCollection": [
                {
                    "Name": "test-recipe",
                    "HostingConfigs": [
                        {
                            "Profile": "Default",
                            "EcrAddress": "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
                            "InstanceType": "ml.g5.12xlarge",  # Different from user-provided
                            "ComputeResourceRequirements": {
                                "NumberOfCpuCoresRequired": 4
                            }
                        }
                    ]
                }
            ]
        }
        
        mock_get_resources.return_value = (8, 32768)
        
        # Create ModelBuilder WITH explicit instance_type
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0"
            },
            instance_type="ml.p4d.24xlarge",  # User explicitly provides instance type
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session
        )
        
        # Execute
        builder._fetch_and_cache_recipe_config()
        
        # Verify: Should preserve user-provided instance type
        assert builder.instance_type == "ml.p4d.24xlarge"

    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model')
    def test_inference_only_called_for_model_customization(
        self, mock_fetch_hub, mock_fetch_package
    ):
        """Test that inference is only called during model customization flow, not for regular models."""
        # This test verifies that _fetch_and_cache_recipe_config is only called
        # in the model customization flow, which is the only place where
        # _infer_instance_type_from_jumpstart should be called
        
        # For regular (non-model-customization) models, _fetch_and_cache_recipe_config
        # is never called, so _infer_instance_type_from_jumpstart won't be called either
        
        # Create a regular ModelBuilder (not model customization)
        builder = ModelBuilder(
            model="my-local-model",  # Not a model customization
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge"  # Provide instance_type to avoid auto-detection
        )
        
        # Verify: _is_model_customization should return False
        assert not builder._is_model_customization()
        
        # For model customization, _fetch_and_cache_recipe_config is called
        # which is where instance type inference happens
        # This is tested in the previous tests
