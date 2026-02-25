"""
Unit tests for ModelBuilder inference_config parameter handling.
Tests the _deploy_model_customization method with inference_config parameter.

Requirements: 2.3, 2.4, 2.5
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements


class TestInferenceConfigParameterHandling(unittest.TestCase):
    """Test inference_config parameter handling in deployment - Requirements 2.3, 2.4, 2.5"""

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

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    def test_inference_config_provided_all_fields(
        self,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
    ):
        """Test deployment with inference_config containing all ResourceRequirements fields."""
        # Setup: Mock model package
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        # Setup: Mock endpoint doesn't exist (new deployment)
        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        # Setup: Mock endpoint creation
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        # Create ModelBuilder
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # User provides inference_config with all fields
        inference_config = ResourceRequirements(
            requests={"num_cpus": 8, "memory": 16384, "num_accelerators": 4},
            limits={"memory": 32768},
        )

        # Execute
        builder._deploy_model_customization(
            endpoint_name="test-endpoint", inference_config=inference_config
        )

        # Verify: InferenceComponent.create was called with correct compute requirements
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]

        # Extract compute requirements from the specification
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify all fields are present
        assert compute_reqs.number_of_cpu_cores_required == 8
        assert compute_reqs.min_memory_required_in_mb == 16384
        assert compute_reqs.max_memory_required_in_mb == 32768
        assert compute_reqs.number_of_accelerator_devices_required == 4

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    def test_inference_config_provided_partial_fields(
        self,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
    ):
        """Test deployment with inference_config containing only some fields."""
        # Setup
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # User provides inference_config with only accelerator count and memory
        inference_config = ResourceRequirements(
            requests={"num_accelerators": 2, "memory": 8192}  # Required field
        )

        # Execute
        builder._deploy_model_customization(
            endpoint_name="test-endpoint", inference_config=inference_config
        )

        # Verify: InferenceComponent.create was called with accelerator count
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify accelerator count and memory are set
        assert compute_reqs.number_of_accelerator_devices_required == 2
        assert compute_reqs.min_memory_required_in_mb == 8192
        # CPU cores should be None (not set)
        assert compute_reqs.number_of_cpu_cores_required is None

    @patch("sagemaker.serve.model_builder.ModelBuilder._infer_accelerator_count_from_instance_type")
    @patch("sagemaker.serve.model_builder.ModelBuilder._is_gpu_instance")
    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_hub_document_for_custom_model")
    @patch("sagemaker.serve.model_builder.ModelBuilder._get_instance_resources")
    def test_inference_config_not_provided_uses_cached_requirements(
        self,
        mock_get_resources,
        mock_fetch_hub,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
        mock_is_gpu,
        mock_infer_accel,
    ):
        """Test deployment without inference_config uses cached compute requirements from build()."""
        # Setup: Mock GPU detection for g5.12xlarge
        mock_is_gpu.return_value = True
        mock_infer_accel.return_value = 4

        # Setup: Mock hub document with default compute requirements
        mock_fetch_hub.return_value = {
            "HostingConfigs": [
                {
                    "Profile": "Default",
                    "ComputeResourceRequirements": {
                        "NumberOfCpuCoresRequired": 4,
                        "MinMemoryRequiredInMb": 8192,
                        "NumberOfAcceleratorDevicesRequired": 4,
                    },
                }
            ]
        }
        mock_get_resources.return_value = (48, 196608)  # g5.12xlarge specs

        # Setup: Mock model package
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # Simulate build() being called which resolves compute requirements
        cached_requirements = builder._resolve_compute_requirements(
            instance_type="ml.g5.12xlarge", user_resource_requirements=None
        )
        builder._cached_compute_requirements = cached_requirements

        # Execute deployment WITHOUT inference_config
        builder._deploy_model_customization(endpoint_name="test-endpoint", inference_config=None)

        # Verify: InferenceComponent.create was called with cached requirements
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify cached requirements were used
        assert compute_reqs.number_of_cpu_cores_required == 4
        assert compute_reqs.min_memory_required_in_mb == 1024
        assert compute_reqs.number_of_accelerator_devices_required == 4

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    def test_inference_config_overrides_cached_requirements(
        self,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
    ):
        """Test that inference_config takes precedence over cached requirements."""
        # Setup
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # Set cached requirements (from build())
        from sagemaker.core.utils.utils import Unassigned

        cached_requirements = InferenceComponentComputeResourceRequirements(
            number_of_cpu_cores_required=4,
            min_memory_required_in_mb=8192,
            number_of_accelerator_devices_required=2,
        )
        builder._cached_compute_requirements = cached_requirements

        # User provides different inference_config
        inference_config = ResourceRequirements(
            requests={"num_cpus": 16, "memory": 32768, "num_accelerators": 8}
        )

        # Execute
        builder._deploy_model_customization(
            endpoint_name="test-endpoint", inference_config=inference_config
        )

        # Verify: InferenceComponent.create was called with inference_config values, not cached
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify inference_config values were used (not cached)
        assert compute_reqs.number_of_cpu_cores_required == 16
        assert compute_reqs.min_memory_required_in_mb == 32768
        assert compute_reqs.number_of_accelerator_devices_required == 8

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    def test_all_resource_requirements_fields_reach_api_call(
        self,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
    ):
        """Test that all ResourceRequirements fields reach the CreateInferenceComponent API call."""
        # Setup
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # Provide inference_config with all possible fields
        inference_config = ResourceRequirements(
            requests={"num_cpus": 12, "memory": 24576, "num_accelerators": 4},
            limits={"memory": 49152},
        )

        # Execute
        builder._deploy_model_customization(
            endpoint_name="test-endpoint", inference_config=inference_config
        )

        # Verify: All fields are present in the API call
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify each field individually
        assert (
            compute_reqs.number_of_cpu_cores_required == 12
        ), "number_of_cpu_cores_required should be 12"
        assert (
            compute_reqs.min_memory_required_in_mb == 24576
        ), "min_memory_required_in_mb should be 24576"
        assert (
            compute_reqs.max_memory_required_in_mb == 49152
        ), "max_memory_required_in_mb should be 49152"
        assert (
            compute_reqs.number_of_accelerator_devices_required == 4
        ), "number_of_accelerator_devices_required should be 4"

    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    @patch("sagemaker.core.resources.Tag.get_all")
    def test_inference_config_with_existing_endpoint_lora_adapter(
        self,
        mock_tag_get_all,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
    ):
        """Test inference_config with existing endpoint (LORA adapter deployment)."""
        # Setup: Mock model package
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        # Setup: Endpoint exists
        mock_does_endpoint_exist.return_value = True
        mock_fetch_peft.return_value = "LORA"

        mock_endpoint = Mock()
        mock_endpoint_get.return_value = mock_endpoint

        # Setup: Mock existing base inference component
        mock_base_component = Mock()
        mock_base_component.inference_component_name = "base-component"
        mock_base_component.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/base"
        )
        mock_ic_get_all.return_value = [mock_base_component]

        # Setup: Mock tags for base component
        mock_tag = Mock()
        mock_tag.key = "Base"
        mock_tag.value = "test-recipe"
        mock_tag_get_all.return_value = [mock_tag]

        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-llm-mistral-7b",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.g5.12xlarge",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # User provides inference_config for adapter
        inference_config = ResourceRequirements(requests={"num_accelerators": 1, "memory": 4096})

        # Execute
        builder._deploy_model_customization(
            endpoint_name="existing-endpoint", inference_config=inference_config
        )

        # Verify: InferenceComponent.create was called with inference_config
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify inference_config values were used
        assert compute_reqs.number_of_accelerator_devices_required == 1
        assert compute_reqs.min_memory_required_in_mb == 4096

        # Verify base_inference_component_name is set for LORA
        assert ic_spec.base_inference_component_name == "base-component"

    @patch("sagemaker.core.resources.InferenceComponent.get")
    @patch("sagemaker.core.resources.Action.create")
    @patch("sagemaker.core.resources.Artifact.get_all")
    @patch("sagemaker.core.resources.Association.add")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package_arn")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_peft")
    @patch("sagemaker.serve.model_builder.ModelBuilder._fetch_model_package")
    @patch("sagemaker.serve.model_builder.ModelBuilder._does_endpoint_exist")
    @patch("sagemaker.core.resources.EndpointConfig.create")
    @patch("sagemaker.core.resources.Endpoint.create")
    @patch("sagemaker.core.resources.Endpoint.get")
    @patch("sagemaker.core.resources.InferenceComponent.create")
    @patch("sagemaker.core.resources.InferenceComponent.get_all")
    def test_inference_config_with_zero_accelerators(
        self,
        mock_ic_get_all,
        mock_ic_create,
        mock_endpoint_get,
        mock_endpoint_create,
        mock_endpoint_config_create,
        mock_does_endpoint_exist,
        mock_fetch_package,
        mock_fetch_peft,
        mock_fetch_package_arn,
        mock_association_add,
        mock_artifact_get_all,
        mock_action_create,
        mock_ic_get,
    ):
        """Test inference_config with zero accelerators (CPU-only deployment)."""
        # Setup
        mock_package = Mock()
        mock_package.model_package_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        )
        mock_package.inference_specification.containers = [Mock()]
        mock_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_package.inference_specification.containers[
            0
        ].model_data_source.s3_data_source.s3_uri = "s3://test-bucket/model"
        mock_fetch_package.return_value = mock_package

        mock_does_endpoint_exist.return_value = False
        mock_fetch_peft.return_value = "FULL"

        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_endpoint_create.return_value = mock_endpoint

        # Setup: Mock InferenceComponent.get for lineage tracking
        mock_ic = Mock()
        mock_ic.inference_component_arn = (
            "arn:aws:sagemaker:us-west-2:123456789012:inference-component/test"
        )
        mock_ic_get.return_value = mock_ic

        # Setup: Mock lineage tracking
        mock_fetch_package_arn.return_value = (
            "arn:aws:sagemaker:us-west-2:123456789012:model-package/test"
        )
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test"
        mock_artifact_get_all.return_value = [mock_artifact]

        builder = ModelBuilder(
            model="huggingface-text-classification",
            model_metadata={
                "CUSTOM_MODEL_ID": "huggingface-text-classification",
                "CUSTOM_MODEL_VERSION": "1.0.0",
            },
            instance_type="ml.m5.2xlarge",  # CPU instance
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self.mock_session,
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest",
        )

        # User explicitly sets 0 accelerators for CPU deployment
        inference_config = ResourceRequirements(
            requests={"num_cpus": 4, "memory": 8192, "num_accelerators": 0}
        )

        # Execute
        builder._deploy_model_customization(
            endpoint_name="test-endpoint", inference_config=inference_config
        )

        # Verify: InferenceComponent.create was called with 0 accelerators
        assert mock_ic_create.called
        call_kwargs = mock_ic_create.call_args[1]
        ic_spec = call_kwargs["specification"]
        compute_reqs = ic_spec.compute_resource_requirements

        # Verify 0 accelerators is accepted
        assert compute_reqs.number_of_accelerator_devices_required == 0
        assert compute_reqs.number_of_cpu_cores_required == 4
        assert compute_reqs.min_memory_required_in_mb == 8192


if __name__ == "__main__":
    unittest.main()
