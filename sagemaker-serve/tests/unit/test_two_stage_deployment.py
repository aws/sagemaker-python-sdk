"""Unit tests for two-stage deployment support (base model + adapter).

Tests verify that:
1. Base models are correctly tagged as "Base"
2. Full fine-tuned models are NOT tagged as "Base"
3. LORA adapters correctly reference base components
4. Separate inference components are created for base and adapter
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.resources import ModelPackage, TrainingJob


class TestTwoStageDeployment:
    """Test two-stage deployment pattern for base models and adapters."""

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_does_endpoint_exist")
    @patch.object(ModelBuilder, "_fetch_hub_document_for_custom_model")
    @patch.object(ModelBuilder, "_is_model_customization")
    def test_base_model_deployment_tagged_correctly(
        self,
        mock_is_customization,
        mock_fetch_hub,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create,
        mock_ic_get,
    ):
        """Test that base model deployments are correctly tagged as 'Base'."""
        # Setup: Base model (no model_data_source, has base_model)
        mock_is_customization.return_value = True
        mock_endpoint_exists.return_value = False
        mock_fetch_peft.return_value = None  # Not a LORA adapter

        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-base-model"
        mock_container.model_data_source = None  # No model_data_source = base model
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        mock_fetch_hub.return_value = {
            "HostingArtifactUri": "s3://jumpstart-bucket/base-model-artifacts/"
        }

        # Mock endpoint creation
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Mock inference component for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
        )
        mock_ic_get.return_value = mock_ic

        # Create ModelBuilder and deploy
        model_builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )

        # Mock the cached compute requirements
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        model_builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
        )

        # Deploy with mocked lineage tracking
        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            model_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify: InferenceComponent.create was called with Base tag
        assert mock_ic_create.called
        create_call = mock_ic_create.call_args
        tags = create_call[1].get("tags", [])

        # Should have exactly one tag with key="Base"
        assert len(tags) == 1
        assert tags[0]["key"] == "Base"
        assert tags[0]["value"] == "test-base-model"

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_does_endpoint_exist")
    @patch.object(ModelBuilder, "_is_model_customization")
    def test_full_fine_tuned_model_not_tagged_as_base(
        self,
        mock_is_customization,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create,
        mock_ic_get,
    ):
        """Test that full fine-tuned models are NOT tagged as 'Base'."""
        # Setup: Full fine-tuned model (has model_data_source)
        mock_is_customization.return_value = True
        mock_endpoint_exists.return_value = False
        mock_fetch_peft.return_value = None  # Not a LORA adapter

        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-base-model"

        # Has model_data_source = full fine-tuned model
        mock_container.model_data_source = Mock()
        mock_container.model_data_source.s3_data_source = Mock()
        mock_container.model_data_source.s3_data_source.s3_uri = "s3://bucket/fine-tuned-model/"

        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        # Mock endpoint creation
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Mock inference component for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
        )
        mock_ic_get.return_value = mock_ic

        # Create ModelBuilder and deploy
        model_builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )

        # Mock the cached compute requirements
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        model_builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
        )

        # Deploy with mocked lineage tracking
        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            model_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify: InferenceComponent.create was called WITHOUT Base tag
        assert mock_ic_create.called
        create_call = mock_ic_create.call_args
        tags = create_call[1].get("tags", [])

        # Should have NO tags (empty list)
        assert len(tags) == 0

    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Tag.get_all")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_does_endpoint_exist")
    @patch.object(ModelBuilder, "_is_model_customization")
    def test_lora_adapter_references_base_component(
        self,
        mock_is_customization,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_endpoint_get,
        mock_tag_get_all,
        mock_ic_create,
        mock_ic_get_all,
    ):
        """Test that LORA adapters correctly reference the base component."""
        # Setup: LORA adapter deployment on existing endpoint
        mock_is_customization.return_value = True
        mock_endpoint_exists.return_value = True
        mock_fetch_peft.return_value = "LORA"  # This is a LORA adapter

        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-base-model"
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package

        # Mock existing endpoint
        mock_endpoint = Mock()
        mock_endpoint_get.return_value = mock_endpoint

        # Mock existing base inference component with Base tag
        mock_base_ic = Mock()
        mock_base_ic.inference_component_name = "test-endpoint-base-component"
        mock_base_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/base"
        )

        mock_ic_get_all.return_value = [mock_base_ic]

        # Mock Tag.get_all to return Base tag
        mock_tag = Mock()
        mock_tag.key = "Base"
        mock_tag.value = "test-base-model"
        mock_tag_get_all.return_value = [mock_tag]

        # Create ModelBuilder and deploy
        model_builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )

        # Mock the cached compute requirements
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        model_builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
        )

        # Deploy
        model_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify: InferenceComponent.create was called with base_inference_component_name
        assert mock_ic_create.called
        create_call = mock_ic_create.call_args
        spec = create_call[1].get("specification")

        # Should reference the base component
        assert spec.base_inference_component_name == "test-endpoint-base-component"

        # Should have artifact_url = None for LORA
        assert spec.container.artifact_url is None

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_does_endpoint_exist")
    @patch.object(ModelBuilder, "_fetch_hub_document_for_custom_model")
    @patch.object(ModelBuilder, "_is_model_customization")
    def test_base_model_uses_hosting_artifact_uri(
        self,
        mock_is_customization,
        mock_fetch_hub,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_ic_create,
        mock_ic_get,
    ):
        """Test that base model deployment uses HostingArtifactUri from JumpStart."""
        # Setup: Base model
        mock_is_customization.return_value = True
        mock_endpoint_exists.return_value = False
        mock_fetch_peft.return_value = None

        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-base-model"
        mock_container.model_data_source = None  # Base model
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        expected_artifact_uri = "s3://jumpstart-bucket/base-model-artifacts/"
        mock_fetch_hub.return_value = {"HostingArtifactUri": expected_artifact_uri}

        # Mock endpoint creation
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Mock inference component for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
        )
        mock_ic_get.return_value = mock_ic

        # Create ModelBuilder and deploy
        model_builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )

        # Mock the cached compute requirements
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        model_builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
        )

        # Deploy with mocked lineage tracking
        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            model_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify: InferenceComponent.create was called with HostingArtifactUri
        assert mock_ic_create.called
        create_call = mock_ic_create.call_args
        spec = create_call[1].get("specification")

        # Should use HostingArtifactUri
        assert spec.container.artifact_url == expected_artifact_uri

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.Tag.get_all")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch.object(ModelBuilder, "_fetch_model_package_arn")
    @patch.object(ModelBuilder, "_fetch_model_package")
    @patch.object(ModelBuilder, "_fetch_peft")
    @patch.object(ModelBuilder, "_does_endpoint_exist")
    @patch.object(ModelBuilder, "_fetch_hub_document_for_custom_model")
    @patch.object(ModelBuilder, "_is_model_customization")
    def test_sequential_base_then_adapter_deployment(
        self,
        mock_is_customization,
        mock_fetch_hub,
        mock_endpoint_exists,
        mock_fetch_peft,
        mock_fetch_package,
        mock_fetch_package_arn,
        mock_endpoint_config_create,
        mock_endpoint_create,
        mock_endpoint_get,
        mock_tag_get_all,
        mock_ic_create,
        mock_ic_get_all,
        mock_ic_get,
    ):
        """Test deploying base model first, then adapter as separate operation.

        Validates Requirements 5.3: Sequential Base-Then-Adapter Deployment
        """
        # Setup: Base model deployment first
        mock_is_customization.return_value = True
        mock_fetch_peft.return_value = None  # Not a LORA adapter initially

        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_container = Mock()
        mock_container.base_model = Mock()
        mock_container.base_model.recipe_name = "test-base-model"
        mock_container.model_data_source = None  # Base model
        mock_package.inference_specification = Mock()
        mock_package.inference_specification.containers = [mock_container]
        mock_fetch_package.return_value = mock_package
        mock_fetch_package_arn.return_value = mock_package.model_package_arn

        mock_fetch_hub.return_value = {
            "HostingArtifactUri": "s3://jumpstart-bucket/base-model-artifacts/"
        }

        # Mock endpoint creation for base model
        mock_endpoint = Mock()
        mock_endpoint.endpoint_name = "test-endpoint"
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint
        mock_endpoint_get.return_value = mock_endpoint

        # Mock inference component for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test-ic"
        )
        mock_ic_get.return_value = mock_ic

        # First deployment: Base model
        mock_endpoint_exists.return_value = False

        model_builder = ModelBuilder(
            model=mock_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )

        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements

        model_builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
        )

        # Deploy base model with mocked lineage tracking
        with patch("sagemaker.core.resources.Action"), patch(
            "sagemaker.core.resources.Association"
        ), patch("sagemaker.core.resources.Artifact"):
            model_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify base model was deployed
        assert mock_endpoint_create.called
        assert mock_ic_create.call_count == 1
        base_create_call = mock_ic_create.call_args
        base_tags = base_create_call[1].get("tags", [])
        assert len(base_tags) == 1
        assert base_tags[0]["key"] == "Base"

        # Reset mocks for adapter deployment
        mock_ic_create.reset_mock()

        # Second deployment: Adapter on existing endpoint
        mock_endpoint_exists.return_value = True
        mock_fetch_peft.return_value = "LORA"  # Now deploying LORA adapter

        # Mock adapter package
        mock_adapter_package = Mock()
        mock_adapter_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/adapter-package"
        )
        mock_adapter_container = Mock()
        mock_adapter_container.base_model = Mock()
        mock_adapter_container.base_model.recipe_name = "test-base-model"
        mock_adapter_package.inference_specification = Mock()
        mock_adapter_package.inference_specification.containers = [mock_adapter_container]
        mock_fetch_package.return_value = mock_adapter_package

        # Mock base inference component
        mock_base_ic = Mock()
        mock_base_ic.inference_component_name = "test-endpoint-inference-component"
        mock_base_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/base"
        )

        # Mock Tag.get_all to return Base tag from base component
        mock_tag = Mock()
        mock_tag.key = "Base"
        mock_tag.value = "test-base-model"
        mock_tag_get_all.return_value = [mock_tag]

        # Mock get_all to return base component
        mock_ic_get_all.return_value = [mock_base_ic]

        # Create new ModelBuilder for adapter
        adapter_builder = ModelBuilder(
            model=mock_adapter_package,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            instance_type="ml.g5.12xlarge",
        )
        adapter_builder._cached_compute_requirements = (
            InferenceComponentComputeResourceRequirements(
                min_memory_required_in_mb=24576, number_of_accelerator_devices_required=4
            )
        )

        # Deploy adapter (no lineage tracking for existing endpoint)
        adapter_builder._deploy_model_customization(endpoint_name="test-endpoint")

        # Verify adapter was deployed
        assert mock_ic_create.call_count == 1
        adapter_create_call = mock_ic_create.call_args

        # Verify adapter references base component
        spec = adapter_create_call[1].get("specification")
        assert spec.base_inference_component_name == "test-endpoint-inference-component"

        # Verify adapter has no tags (only base model is tagged)
        adapter_tags = adapter_create_call[1].get("tags", [])
        assert len(adapter_tags) == 0

        # Verify endpoint was not recreated
        assert mock_endpoint_create.call_count == 1  # Only called once for base model

    def test_routing_with_both_base_and_adapter_components(self):
        """Test that inference requests can be routed to specific components.

        Validates Requirements 5.4: Multi-Component Routing
        """
        # Setup: Mock endpoint with both base and adapter components
        mock_endpoint = Mock()
        mock_endpoint.endpoint_name = "test-endpoint"

        # Mock base inference component
        mock_base_ic = Mock()
        mock_base_ic.inference_component_name = "test-endpoint-base-component"

        # Mock adapter inference component
        mock_adapter_ic = Mock()
        mock_adapter_ic.inference_component_name = "test-endpoint-adapter-component"

        # Mock invoke responses
        mock_base_response = Mock()
        mock_base_response.body = b'{"result": "base model response"}'
        mock_base_response.content_type = "application/json"

        mock_adapter_response = Mock()
        mock_adapter_response.body = b'{"result": "adapter response"}'
        mock_adapter_response.content_type = "application/json"

        # Test 1: Invoke base component
        mock_endpoint.invoke = Mock(return_value=mock_base_response)

        response = mock_endpoint.invoke(
            body={"input": "test"}, inference_component_name="test-endpoint-base-component"
        )

        # Verify base component was invoked
        assert mock_endpoint.invoke.called
        call_args = mock_endpoint.invoke.call_args
        assert call_args.kwargs.get("inference_component_name") == "test-endpoint-base-component"
        assert response.body == b'{"result": "base model response"}'

        # Reset mock
        mock_endpoint.invoke.reset_mock()

        # Test 2: Invoke adapter component
        mock_endpoint.invoke = Mock(return_value=mock_adapter_response)

        response = mock_endpoint.invoke(
            body={"input": "test"}, inference_component_name="test-endpoint-adapter-component"
        )

        # Verify adapter component was invoked
        assert mock_endpoint.invoke.called
        call_args = mock_endpoint.invoke.call_args
        assert call_args.kwargs.get("inference_component_name") == "test-endpoint-adapter-component"
        assert response.body == b'{"result": "adapter response"}'

        # Test 3: Invoke without specifying component (default routing)
        mock_endpoint.invoke.reset_mock()
        mock_endpoint.invoke = Mock(return_value=mock_base_response)

        response = mock_endpoint.invoke(body={"input": "test"})

        # Verify invoke was called without inference_component_name
        assert mock_endpoint.invoke.called
        call_args = mock_endpoint.invoke.call_args
        # When no component is specified, the parameter should be absent or None
        inference_component = call_args.kwargs.get("inference_component_name")
        assert inference_component is None or "inference_component_name" not in call_args.kwargs
