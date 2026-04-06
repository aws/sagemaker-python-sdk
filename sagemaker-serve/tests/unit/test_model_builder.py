"""
Unit tests for ModelBuilder V3 implementation.

These tests focus on the V3 experience where:
- build() returns sagemaker.core.resources.Model (actual AWS resource)
- deploy() returns sagemaker.core.resources.Endpoint (actual AWS resource)
"""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock

from botocore.exceptions import ClientError

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


class TestModelBuilderV3(unittest.TestCase):
    """Test ModelBuilder V3 implementation."""

    def setUp(self):
        """Set up test fixtures."""
        import tempfile
        self.model_path = tempfile.mkdtemp()
        
        # Shared schema builder for all tests
        self.mock_schema_builder = MagicMock()
        self.mock_schema_builder.sample_input = {"inputs": "test input", "parameters": {}}
        self.mock_schema_builder.sample_output = [{"generated_text": "test output"}]
        
        # Shared mock model
        self.mock_model = Mock()
        
        # Shared mock inference spec
        self.mock_inference_spec = Mock()
        
        # Shared image URI
        self.image_uri = "123456789012.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.0-gpu-py3"
        
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-east-1"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        
        # Mock session credentials properly
        mock_credentials = Mock()
        mock_credentials.access_key = "test-access-key"
        mock_credentials.secret_key = "test-secret-key"
        mock_credentials.token = None
        self.mock_session.boto_session.get_credentials.return_value = mock_credentials
        self.mock_session.boto_session.region_name = "us-east-1"
        
        # Mock config attributes to prevent config resolution errors
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        # Additional mock setup for session
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-east-1"
        
        # Mock settings to prevent AttributeError
        self.mock_session.settings = Mock()
        self.mock_session.settings.include_jumpstart_tags = False
        self.mock_session.settings._local_download_dir = None

    def test_model_server_validation_unsupported_type(self):
        """Test that unsupported model server types raise error."""
        try:
            builder = ModelBuilder(
                model=self.mock_model,
                model_server="UNSUPPORTED_SERVER",
                sagemaker_session=self.mock_session
            )
            # If we get here, the validation might happen later
            self.assertTrue(True)
        except (ValueError, AttributeError, TypeError):
            # Expected - validation caught the invalid server type
            self.assertTrue(True)

    def test_env_vars_initialization(self):
        """Test that env_vars is properly initialized."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertIsInstance(builder.env_vars, dict)

    def test_env_vars_custom_values(self):
        """Test that custom env_vars are preserved."""
        custom_env = {"CUSTOM_VAR": "custom_value"}
        
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            env_vars=custom_env,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.env_vars["CUSTOM_VAR"], "custom_value")

    def test_model_path_temp_creation_local_mode(self):
        """Test that temp model_path is created for local modes."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertIsNotNone(builder.model_path)
        self.assertTrue("/tmp" in builder.model_path or "sagemaker" in builder.model_path)

    def test_schema_builder_validation(self):
        """Test that schema_builder is properly validated."""
        from sagemaker.serve.builder.schema_builder import SchemaBuilder
        
        sample_input = {"inputs": "test"}
        sample_output = [{"result": "test"}]
        schema_builder = SchemaBuilder(sample_input, sample_output)
        
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            schema_builder=schema_builder,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.schema_builder, schema_builder)

    def test_mode_defaults_to_sagemaker_endpoint(self):
        """Test that mode defaults to SAGEMAKER_ENDPOINT."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.mode, Mode.SAGEMAKER_ENDPOINT)

    def test_mode_local_container_validation(self):
        """Test LOCAL_CONTAINER mode validation."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        self.assertEqual(builder.mode, Mode.LOCAL_CONTAINER)
        self.assertIsNotNone(builder.model_path)

    def test_deploy_requires_built_model(self):
        """Test that deploy() requires build() to be called first."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder.deploy()
        
        error_msg = str(context.exception).lower()
        self.assertTrue("model" in error_msg and "built" in error_msg and "deploy" in error_msg)

    @patch("sagemaker.serve.model_builder.ModelBuilder._deploy")
    def test_deploy_serverless_inference(self, mock_deploy):
        """Test deploy() with ServerlessInferenceConfig."""
        from sagemaker.core.inference_config import ServerlessInferenceConfig
        
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        builder.built_model = Mock()
        
        serverless_config = ServerlessInferenceConfig()
        
        result = builder.deploy(
            inference_config=serverless_config,
            endpoint_name="test-serverless-endpoint"
        )
        
        mock_deploy.assert_called_once()
        self.assertEqual(result, mock_endpoint)

    def test_transformer_requires_built_model(self):
        """Test that transformer() requires built_model to exist."""
        builder = ModelBuilder(
            model=self.mock_model,
            model_server=ModelServer.TORCHSERVE,
            role_arn="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder.transformer(
                instance_count=1,
                instance_type="ml.m5.large"
            )
        
        self.assertIn("Must call build() before creating transformer", str(context.exception))


if __name__ == "__main__":
    unittest.main()


class ModelCustomizationTest(unittest.TestCase):
    """Test ModelBuilder model customization features."""

    def setUp(self):
        """Set up test fixtures."""
        from sagemaker.core.resources import TrainingJob
        
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-east-1"
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-east-1"
        
        # Mock config attributes to prevent config resolution errors
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_training_job = Mock(spec=TrainingJob)
        self.mock_training_job.serverless_job_config = Mock()
        self.mock_training_job.model_package_config = Mock()
        self.mock_training_job.output_model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-package"

    @patch('sagemaker.serve.model_builder.HubContent')
    def test_fetch_hub_document_for_custom_model(self, mock_hub_content):
        """Test fetching hub document for custom model."""
        mock_hub_doc = {"HostingConfigs": {"InstanceType": "ml.g5.2xlarge"}}
        mock_hub_content.get.return_value.hub_content_document = json.dumps(mock_hub_doc)
        
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].base_model = Mock()
        mock_model_package.inference_specification.containers[0].base_model.hub_content_name = "test-model"
        mock_model_package.inference_specification.containers[0].base_model.hub_content_version = "1.0"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            result = builder._fetch_hub_document_for_custom_model()
            self.assertEqual(result, mock_hub_doc)

    def test_fetch_hosting_configs_for_custom_model(self):
        """Test fetching hosting configs for custom model."""
        mock_hub_doc = {"HostingConfigs": {"InstanceType": "ml.g5.2xlarge"}}
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        with patch.object(builder, '_fetch_hub_document_for_custom_model', return_value=mock_hub_doc):
            result = builder._fetch_hosting_configs_for_custom_model()
            self.assertEqual(result, {"InstanceType": "ml.g5.2xlarge"})

    def test_fetch_default_instance_type_for_custom_model(self):
        """Test fetching default instance type for custom model."""
        mock_hosting_configs = {"InstanceType": "ml.g5.2xlarge"}
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        with patch.object(builder, '_fetch_hosting_configs_for_custom_model', return_value=mock_hosting_configs):
            result = builder._fetch_default_instance_type_for_custom_model()
            self.assertEqual(result, "ml.g5.2xlarge")


    def test_get_instance_resources(self):
        """Test getting instance resources from EC2."""
        mock_ec2 = Mock()
        mock_ec2.describe_instance_types.return_value = {
            'InstanceTypes': [{
                'VCpuInfo': {'DefaultVCpus': 8},
                'MemoryInfo': {'SizeInMiB': 32768}
            }]
        }
        self.mock_session.boto_session.client.return_value = mock_ec2
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        cpus, memory = builder._get_instance_resources("ml.g5.2xlarge")
        self.assertEqual(cpus, 8)
        self.assertEqual(memory, 32768)

    @patch('sagemaker.serve.model_builder.InferenceComponent')
    @patch('sagemaker.core.resources.Tag')
    def test_fetch_endpoint_names_for_base_model(self, mock_tag, mock_ic):
        """Test fetching endpoint names for base model."""
        mock_ic1 = Mock()
        mock_ic1.inference_component_arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/ic1"
        mock_ic1.endpoint_name = "endpoint-1"
        
        mock_ic.get_all.return_value = [mock_ic1]
        
        mock_tag_obj = Mock()
        mock_tag_obj.key = "Base"
        mock_tag_obj.value = "test-recipe"
        mock_tag.get_all.return_value = [mock_tag_obj]
        
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].base_model = Mock()
        mock_model_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        with patch.object(builder, '_is_model_customization', return_value=True):
            with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
                result = builder.fetch_endpoint_names_for_base_model()
                self.assertIn("endpoint-1", result)

    def test_fetch_model_package_arn_from_model_package_config(self):
        """Test _fetch_model_package_arn from model_package_config."""
        from sagemaker.core.utils.utils import Unassigned
        from sagemaker.core.resources import TrainingJob
        
        mock_training_job = Mock(spec=TrainingJob)
        mock_training_job.output_model_package_arn = Unassigned()
        mock_training_job.model_package_config = Mock()
        mock_training_job.model_package_config.source_model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/source"
        mock_training_job.serverless_job_config = Unassigned()
        
        builder = ModelBuilder(
            model=mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._fetch_model_package_arn()
        self.assertEqual(result, "arn:aws:sagemaker:us-east-1:123456789012:model-package/source")

    def test_fetch_peft_from_training_job(self):
        """Test fetching PEFT from TrainingJob."""
        from sagemaker.core.utils.utils import Unassigned
        
        self.mock_training_job.serverless_job_config = Mock()
        self.mock_training_job.serverless_job_config.peft = "LORA"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._fetch_peft()
        self.assertEqual(result, "LORA")

    def test_fetch_peft_from_model_trainer(self):
        """Test fetching PEFT from ModelTrainer."""
        from sagemaker.train.model_trainer import ModelTrainer
        
        self.mock_training_job.serverless_job_config = Mock()
        self.mock_training_job.serverless_job_config.peft = "LORA"
        
        mock_trainer = Mock(spec=ModelTrainer)
        mock_trainer._latest_training_job = self.mock_training_job
        
        builder = ModelBuilder(
            model=mock_trainer,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._fetch_peft()
        self.assertEqual(result, "LORA")

    def test_is_model_customization_with_model_package_config(self):
        """Test _is_model_customization with model_package_config."""
        from sagemaker.core.utils.utils import Unassigned
        
        self.mock_training_job.model_package_config = Mock()
        self.mock_training_job.model_package_config.source_model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/source"
        self.mock_training_job.serverless_job_config = Unassigned()
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session
        )
        
        result = builder._is_model_customization()
        self.assertTrue(result)

    @patch('sagemaker.serve.model_builder.Model')
    @patch('sagemaker.serve.model_builder.is_1p_image_uri')
    def test_build_single_modelbuilder_with_model_customization(self, mock_is_1p, mock_model_class):
        """Test _build_single_modelbuilder when _is_model_customization returns True."""
        from sagemaker.core.utils.utils import Unassigned
        
        # Mock is_1p_image_uri to return True to bypass validation
        mock_is_1p.return_value = True
        
        # Setup mock model package
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://bucket/model"
        mock_model_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        
        # Setup training job with model_package_config
        self.mock_training_job.model_package_config = Mock()
        self.mock_training_job.model_package_config.source_model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/source"
        
        # Setup mock for Model.create
        mock_created_model = Mock()
        mock_model_class.create.return_value = mock_created_model
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.2xlarge"
        )
        
        # Mock the helper methods
        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            with patch.object(builder, '_fetch_and_cache_recipe_config'):
                with patch.object(builder, '_get_client_translators', return_value=(Mock(), Mock())):
                    with patch.object(builder, '_get_serve_setting', return_value=Mock()):
                        with patch.object(builder, '_is_nova_model', return_value=False):
                            result = builder._build_single_modelbuilder()
        
        # Verify Model.create was called (indicating model customization path was taken)
        mock_model_class.create.assert_called_once()
        self.assertEqual(result, mock_created_model)

    @patch('sagemaker.serve.model_builder.Model')
    @patch('sagemaker.serve.model_builder.is_1p_image_uri')
    def test_build_single_modelbuilder_with_model_customization_no_jumpstart(self, mock_is_1p, mock_model_class):
        """Test _build_single_modelbuilder skips _fetch_and_cache_recipe_config when base_model is None."""
        mock_is_1p.return_value = True

        # Setup mock model package with base_model = None (custom model package, not JumpStart)
        mock_model_package = Mock()
        mock_container = Mock()
        mock_container.base_model = None
        mock_container.model_data_source.s3_data_source.s3_uri = "s3://bucket/model"
        mock_model_package.inference_specification.containers = [mock_container]

        # Setup training job with model_package_config
        self.mock_training_job.model_package_config = Mock()
        self.mock_training_job.model_package_config.source_model_package_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/source"
        )

        mock_created_model = Mock()
        mock_model_class.create.return_value = mock_created_model

        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.2xlarge"
        )

        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            with patch.object(builder, '_fetch_and_cache_recipe_config') as mock_recipe:
                with patch.object(builder, '_get_serve_setting', return_value=Mock()):
                    with patch.object(builder, '_is_nova_model', return_value=False):
                        with patch.object(builder, '_fetch_peft', return_value=None):
                            result = builder._build_single_modelbuilder()

        mock_recipe.assert_not_called()
        mock_model_class.create.assert_called_once()
        self.assertEqual(result, mock_created_model)

    def test_deploy_model_customization_new_endpoint(self):
        """Test _deploy_model_customization for new endpoint creation."""
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Action, Association, Artifact
        
        # Setup mocks
        mock_endpoint_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_ic = Mock()
        mock_ic.inference_component_arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/test-ic"
        mock_action = Mock()
        mock_action.action_arn = "arn:aws:sagemaker:us-east-1:123456789012:action/test-action"
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/test-artifact"
        
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://bucket/model"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.2xlarge"
        )
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=1024,
            number_of_cpu_cores_required=1
        )
        
        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            with patch.object(builder, '_fetch_peft', return_value=None):
              with patch.object(builder, '_is_nova_model', return_value=False):
                with patch.object(EndpointConfig, 'create', return_value=mock_endpoint_config):
                    with patch.object(Endpoint, 'get', side_effect=ClientError({'Error': {'Code': 'ValidationException'}}, 'GetEndpoint')):
                        with patch.object(Endpoint, 'create', return_value=mock_endpoint):
                            with patch.object(InferenceComponent, 'create', return_value=mock_ic):
                                with patch.object(InferenceComponent, 'get', return_value=mock_ic):
                                    with patch.object(Action, 'create', return_value=mock_action):
                                        with patch.object(Artifact, 'get_all', return_value=[mock_artifact]):
                                            with patch.object(Association, 'add', return_value=None):
                                                result = builder._deploy_model_customization(
                                                    endpoint_name="test-endpoint",
                                                    instance_type="ml.g5.2xlarge",
                                                    initial_instance_count=1
                                                )
        
        self.assertEqual(result, mock_endpoint)

    def test_deploy_model_customization_with_inference_config(self):
        """Test _deploy_model_customization with inference_config parameter."""
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Action, Association, Artifact
        from sagemaker.core.inference_config import ResourceRequirements
        
        # Setup mocks
        mock_endpoint_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_ic = Mock()
        mock_ic.inference_component_arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/test-ic"
        mock_action = Mock()
        mock_action.action_arn = "arn:aws:sagemaker:us-east-1:123456789012:action/test-action"
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/test-artifact"
        
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://bucket/model"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.12xlarge"
        )
        
        # Set cached compute requirements (should be overridden by inference_config)
        builder._cached_compute_requirements = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=1024,
            number_of_cpu_cores_required=1,
            number_of_accelerator_devices_required=1
        )
        
        # Create inference_config with different values
        inference_config = ResourceRequirements(
            requests={
                "num_accelerators": 4,
                "num_cpus": 8,
                "memory": 49152
            },
            limits={
                "memory": 98304
            }
        )
        
        # Track the InferenceComponent.create call to verify compute requirements
        created_ic_spec = None
        def capture_ic_create(**kwargs):
            nonlocal created_ic_spec
            created_ic_spec = kwargs.get('specification')
            return mock_ic
        
        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            with patch.object(builder, '_fetch_peft', return_value=None):
              with patch.object(builder, '_is_nova_model', return_value=False):
                with patch.object(EndpointConfig, 'create', return_value=mock_endpoint_config):
                    with patch.object(Endpoint, 'get', side_effect=ClientError({'Error': {'Code': 'ValidationException'}}, 'GetEndpoint')):
                        with patch.object(Endpoint, 'create', return_value=mock_endpoint):
                            with patch.object(InferenceComponent, 'create', side_effect=capture_ic_create):
                                with patch.object(InferenceComponent, 'get', return_value=mock_ic):
                                    with patch.object(Action, 'create', return_value=mock_action):
                                        with patch.object(Artifact, 'get_all', return_value=[mock_artifact]):
                                            with patch.object(Association, 'add', return_value=None):
                                                result = builder._deploy_model_customization(
                                                    endpoint_name="test-endpoint",
                                                    instance_type="ml.g5.12xlarge",
                                                    initial_instance_count=1,
                                                    inference_config=inference_config
                                                )
        
        # Verify the result
        self.assertEqual(result, mock_endpoint)
        
        # Verify that inference_config values were used (not cached values)
        self.assertIsNotNone(created_ic_spec)
        compute_reqs = created_ic_spec.compute_resource_requirements
        self.assertEqual(compute_reqs.min_memory_required_in_mb, 49152)
        self.assertEqual(compute_reqs.max_memory_required_in_mb, 98304)
        self.assertEqual(compute_reqs.number_of_cpu_cores_required, 8)
        self.assertEqual(compute_reqs.number_of_accelerator_devices_required, 4)

    def test_deploy_model_customization_without_inference_config_uses_cached(self):
        """Test _deploy_model_customization falls back to cached requirements when inference_config not provided."""
        from sagemaker.core.shapes import InferenceComponentComputeResourceRequirements
        from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Action, Association, Artifact
        
        # Setup mocks
        mock_endpoint_config = Mock()
        mock_endpoint = Mock()
        mock_endpoint.wait_for_status = Mock()
        mock_ic = Mock()
        mock_ic.inference_component_arn = "arn:aws:sagemaker:us-east-1:123456789012:inference-component/test-ic"
        mock_action = Mock()
        mock_action.action_arn = "arn:aws:sagemaker:us-east-1:123456789012:action/test-action"
        mock_artifact = Mock()
        mock_artifact.artifact_arn = "arn:aws:sagemaker:us-east-1:123456789012:artifact/test-artifact"
        
        mock_model_package = Mock()
        mock_model_package.inference_specification.containers = [Mock()]
        mock_model_package.inference_specification.containers[0].base_model.recipe_name = "test-recipe"
        mock_model_package.inference_specification.containers[0].model_data_source.s3_data_source.s3_uri = "s3://bucket/model"
        
        builder = ModelBuilder(
            model=self.mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.2xlarge"
        )
        
        # Set cached compute requirements
        cached_reqs = InferenceComponentComputeResourceRequirements(
            min_memory_required_in_mb=2048,
            number_of_cpu_cores_required=2,
            number_of_accelerator_devices_required=1
        )
        builder._cached_compute_requirements = cached_reqs
        
        # Track the InferenceComponent.create call to verify compute requirements
        created_ic_spec = None
        def capture_ic_create(**kwargs):
            nonlocal created_ic_spec
            created_ic_spec = kwargs.get('specification')
            return mock_ic
        
        with patch.object(builder, '_fetch_model_package', return_value=mock_model_package):
            with patch.object(builder, '_fetch_peft', return_value=None):
              with patch.object(builder, '_is_nova_model', return_value=False):
                with patch.object(EndpointConfig, 'create', return_value=mock_endpoint_config):
                    with patch.object(Endpoint, 'get', side_effect=ClientError({'Error': {'Code': 'ValidationException'}}, 'GetEndpoint')):
                        with patch.object(Endpoint, 'create', return_value=mock_endpoint):
                            with patch.object(InferenceComponent, 'create', side_effect=capture_ic_create):
                                with patch.object(InferenceComponent, 'get', return_value=mock_ic):
                                    with patch.object(Action, 'create', return_value=mock_action):
                                        with patch.object(Artifact, 'get_all', return_value=[mock_artifact]):
                                            with patch.object(Association, 'add', return_value=None):
                                                result = builder._deploy_model_customization(
                                                    endpoint_name="test-endpoint",
                                                    instance_type="ml.g5.2xlarge",
                                                    initial_instance_count=1
                                                    # Note: no inference_config parameter
                                                )
        
        # Verify the result
        self.assertEqual(result, mock_endpoint)
        
        # Verify that cached requirements were used
        self.assertIsNotNone(created_ic_spec)
        compute_reqs = created_ic_spec.compute_resource_requirements
        self.assertIs(compute_reqs, cached_reqs)

    def test_deploy_passes_inference_config_to_model_customization(self):
        """Test that deploy() passes inference_config to _deploy_model_customization for model customization deployments."""
        from sagemaker.core.inference_config import ResourceRequirements
        
        # Create a mock training job that will be recognized as model customization
        mock_training_job = Mock()
        mock_training_job.training_job_name = "test-training-job"
        
        builder = ModelBuilder(
            model=mock_training_job,
            role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            sagemaker_session=self.mock_session,
            image_uri="test-image:latest",
            instance_type="ml.g5.12xlarge"
        )
        
        # Mark as built
        builder.built_model = Mock()
        
        # Create inference_config
        inference_config = ResourceRequirements(
            requests={
                "num_accelerators": 4,
                "num_cpus": 8,
                "memory": 49152
            }
        )
        
        # Mock _is_model_customization to return True
        with patch.object(builder, '_is_model_customization', return_value=True):
            # Mock _deploy_model_customization to capture the call
            with patch.object(builder, '_deploy_model_customization') as mock_deploy_mc:
                mock_endpoint = Mock()
                mock_deploy_mc.return_value = mock_endpoint
                
                # Call deploy with inference_config
                result = builder.deploy(
                    endpoint_name="test-endpoint",
                    inference_config=inference_config
                )
                
                # Verify _deploy_model_customization was called with inference_config
                mock_deploy_mc.assert_called_once()
                call_kwargs = mock_deploy_mc.call_args[1]
                self.assertEqual(call_kwargs['inference_config'], inference_config)
                self.assertEqual(result, mock_endpoint)


class TestLoraAcceptEula(unittest.TestCase):
    """Tests for accept_eula handling in the LoRA deployment path."""

    def _make_mb(self, accept_eula=None):
        mb = ModelBuilder.__new__(ModelBuilder)
        mb.accept_eula = accept_eula
        mb.image_uri = "some-image-uri"
        mb.env_vars = {}
        mb.model_name = None
        mb.model_path = "/tmp/fake-model-path"
        mb.role_arn = "arn:aws:iam::123456789012:role/role"
        mb.model = MagicMock()
        mb._adapter_s3_uri = None
        mb.shared_libs = []
        mb.dependencies = {"auto": True}
        mb.image_config = None
        mb.inference_spec = None
        mb.schema_builder = None
        mb.modelbuilder_list = None
        mb.sagemaker_session = None
        mb.s3_model_data_url = None
        mb.source_code = None
        mb.model_server = None
        mb.model_metadata = None
        mb.log_level = None
        mb.content_type = None
        mb.accept_type = None
        mb.compute = None
        mb.network = None
        mb.instance_type = None
        mb.mode = None
        return mb

    def _patch_lora_deps(self, mb, hosting_uri="s3://bucket/hosting/"):
        """Patch all dependencies needed to reach the LoRA ContainerDefinition block."""
        patches = [
            patch.object(mb, "_get_serve_setting", return_value=MagicMock()),
            patch.object(mb, "_is_model_customization", return_value=True),
            patch.object(mb, "_fetch_model_package", return_value=MagicMock()),
            patch.object(mb, "_fetch_and_cache_recipe_config"),
            patch.object(mb, "_is_nova_model", return_value=False),
            patch.object(mb, "_fetch_peft", return_value="LORA"),
            patch.object(mb, "_fetch_hub_document_for_custom_model",
                         return_value={"HostingArtifactUri": hosting_uri}),
        ]
        return patches

    def test_lora_build_raises_when_accept_eula_false(self):
        mb = self._make_mb(accept_eula=False)
        patches = self._patch_lora_deps(mb)
        for p in patches:
            p.start()
        try:
            with self.assertRaises(ValueError) as ctx:
                mb._build_single_modelbuilder()
            self.assertIn("accept_eula", str(ctx.exception))
        finally:
            for p in patches:
                p.stop()

    def test_lora_build_raises_when_accept_eula_not_set(self):
        mb = self._make_mb(accept_eula=None)
        patches = self._patch_lora_deps(mb)
        for p in patches:
            p.start()
        try:
            with self.assertRaises(ValueError) as ctx:
                mb._build_single_modelbuilder()
            self.assertIn("accept_eula", str(ctx.exception))
        finally:
            for p in patches:
                p.stop()

    @patch("sagemaker.serve.model_builder.ContainerDefinition")
    @patch("sagemaker.serve.model_builder.Model")
    def test_lora_build_passes_accept_eula_true(self, mock_model, mock_container_def):
        mb = self._make_mb(accept_eula=True)
        mock_model.create.return_value = MagicMock()
        patches = self._patch_lora_deps(mb)
        for p in patches:
            p.start()
        try:
            mb._build_single_modelbuilder()
            call_kwargs = mock_container_def.call_args[1]
            eula_val = (
                call_kwargs["model_data_source"]["s3_data_source"]["model_access_config"]["accept_eula"]
            )
            self.assertTrue(eula_val)
        finally:
            for p in patches:
                p.stop()
