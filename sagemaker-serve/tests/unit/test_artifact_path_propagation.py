"""
Unit tests to verify artifact path propagation to CreateInferenceComponent API.
Tests that _resolve_model_artifact_uri is called and its result is used in deployment.

Requirements: 4.3, 4.4
Task: 5.4
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode


class TestArtifactPathPropagation(unittest.TestCase):
    """Test artifact path propagation to CreateInferenceComponent - Requirements 4.3, 4.4"""

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

    @patch('sagemaker.core.resources.InferenceComponent.create')
    @patch('sagemaker.core.resources.Endpoint.get')
    @patch('sagemaker.core.resources.Endpoint.create')
    @patch('sagemaker.core.resources.EndpointConfig.create')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._resolve_model_artifact_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_base_model_artifact_uri_propagated_to_inference_component(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_resolve_artifact,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create
    ):
        """Test that base model artifact URI is propagated to InferenceComponent.create."""
        # Setup: Model customization deployment
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Artifact URI resolution returns JumpStart HostingArtifactUri
        expected_artifact_uri = 's3://jumpstart-bucket/base-model/artifacts.tar.gz'
        mock_resolve_artifact.return_value = expected_artifact_uri
        
        # Setup: Model package
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn
        
        # Setup: Endpoint mocks
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint
        
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
        
        # Mark as built with cached compute requirements
        builder.built_model = Mock()
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=16384,
            number_of_cpu_cores_required=8.0,
            number_of_accelerator_devices_required=4.0
        )
        
        # Execute: Deploy
        builder._deploy_model_customization(
            endpoint_name="test-endpoint",
            initial_instance_count=1
        )
        
        # Verify: _resolve_model_artifact_uri was called
        assert mock_resolve_artifact.called
        
        # Verify: InferenceComponent.create was called with correct artifact_url
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        
        # Extract the specification
        ic_spec = call_kwargs['specification']
        
        # Verify artifact_url matches the resolved URI
        assert ic_spec.container.artifact_url == expected_artifact_uri

    @patch('sagemaker.core.resources.InferenceComponent.create')
    @patch('sagemaker.core.resources.Endpoint.get')
    @patch('sagemaker.core.resources.Endpoint.create')
    @patch('sagemaker.core.resources.EndpointConfig.create')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._resolve_model_artifact_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_fine_tuned_model_artifact_uri_propagated_to_inference_component(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_resolve_artifact,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create
    ):
        """Test that fine-tuned model artifact URI is propagated to InferenceComponent.create."""
        # Setup: Model customization deployment
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Artifact URI resolution returns None for fine-tuned models
        mock_resolve_artifact.return_value = None
        
        # Setup: Model package
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn
        
        # Setup: Endpoint mocks
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint
        
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
        
        # Mark as built with cached compute requirements
        builder.built_model = Mock()
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=16384,
            number_of_cpu_cores_required=8.0,
            number_of_accelerator_devices_required=4.0
        )
        
        # Execute: Deploy
        builder._deploy_model_customization(
            endpoint_name="test-endpoint",
            initial_instance_count=1
        )
        
        # Verify: _resolve_model_artifact_uri was called
        assert mock_resolve_artifact.called
        
        # Verify: InferenceComponent.create was called with correct artifact_url
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        
        # Extract the specification
        ic_spec = call_kwargs['specification']
        
        # Verify artifact_url is None for fine-tuned models (model data handled by recipe)
        assert ic_spec.container.artifact_url is None

    @patch('sagemaker.core.resources.InferenceComponent.create')
    @patch('sagemaker.core.resources.InferenceComponent.get_all')
    @patch('sagemaker.core.resources.Tag.get_all')
    @patch('sagemaker.core.resources.Endpoint.get')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._resolve_model_artifact_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_lora_adapter_no_artifact_uri_propagated(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_resolve_artifact,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_get,
        mock_tag_get_all,
        mock_ic_get_all,
        mock_ic_create
    ):
        """Test that LORA adapters have None artifact_url (no artifact needed)."""
        # Setup: Model customization deployment with LORA adapter
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "LORA"
        
        # Setup: Artifact URI resolution returns None for LORA
        mock_resolve_artifact.return_value = None
        
        # Setup: Model package
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn
        
        # Setup: Existing endpoint with base component
        mock_endpoint = Mock()
        mock_endpoint_get.return_value = mock_endpoint
        
        # Setup: Base inference component
        mock_base_component = Mock()
        mock_base_component.inference_component_name = "base-component"
        mock_base_component.inference_component_arn = "arn:aws:sagemaker:us-west-2:123456789012:inference-component/base"
        mock_ic_get_all.return_value = [mock_base_component]
        
        # Setup: Tags for base component
        mock_tag = Mock()
        mock_tag.key = "Base"
        mock_tag.value = "test-recipe"
        mock_tag_get_all.return_value = [mock_tag]
        
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
        
        # Mark as built with cached compute requirements
        builder.built_model = Mock()
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=16384,
            number_of_cpu_cores_required=8.0,
            number_of_accelerator_devices_required=4.0
        )
        
        # Execute: Deploy to existing endpoint (LORA adapter)
        builder._deploy_model_customization(
            endpoint_name="test-endpoint",
            initial_instance_count=1
        )
        
        # Verify: _resolve_model_artifact_uri was called
        assert mock_resolve_artifact.called
        
        # Verify: InferenceComponent.create was called with artifact_url=None
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        
        # Extract the specification
        ic_spec = call_kwargs['specification']
        
        # Verify artifact_url is None for LORA adapters
        assert ic_spec.container.artifact_url is None
        
        # Verify base_inference_component_name is set
        assert ic_spec.base_inference_component_name == "base-component"

    @patch('sagemaker.core.resources.InferenceComponent.create')
    @patch('sagemaker.core.resources.Endpoint.get')
    @patch('sagemaker.core.resources.Endpoint.create')
    @patch('sagemaker.core.resources.EndpointConfig.create')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_model_package')
    @patch('sagemaker.serve.model_builder.ModelBuilder._resolve_model_artifact_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._fetch_peft')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_model_customization')
    def test_environment_variables_propagated_with_artifact_path(
        self,
        mock_is_model_customization,
        mock_fetch_peft,
        mock_resolve_artifact,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create
    ):
        """Test that environment variables are propagated along with artifact path."""
        # Setup: Model customization deployment
        mock_is_model_customization.return_value = True
        mock_fetch_peft.return_value = "FULL"
        
        # Setup: Artifact URI resolution
        expected_artifact_uri = 's3://jumpstart-bucket/base-model/artifacts.tar.gz'
        mock_resolve_artifact.return_value = expected_artifact_uri
        
        # Setup: Model package
        mock_package = Mock()
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-recipe"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_package.model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn
        
        # Setup: Endpoint mocks
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint
        
        # Create ModelBuilder with custom environment variables
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
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
            env_vars={
                "CUSTOM_VAR": "custom_value",
                "MODEL_TIMEOUT": "300"
            }
        )
        
        # Mark as built with cached compute requirements
        builder.built_model = Mock()
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=16384,
            number_of_cpu_cores_required=8.0,
            number_of_accelerator_devices_required=4.0
        )
        
        # Execute: Deploy
        builder._deploy_model_customization(
            endpoint_name="test-endpoint",
            initial_instance_count=1
        )
        
        # Verify: InferenceComponent.create was called
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        
        # Extract the specification
        ic_spec = call_kwargs['specification']
        
        # Verify both artifact_url and environment variables are set
        assert ic_spec.container.artifact_url == expected_artifact_uri
        assert ic_spec.container.environment == builder.env_vars
        assert ic_spec.container.environment["CUSTOM_VAR"] == "custom_value"
        assert ic_spec.container.environment["MODEL_TIMEOUT"] == "300"


if __name__ == "__main__":
    unittest.main()
