"""
Unit tests for V3 ModelBuilder - Testing build() and deploy() methods.

V3 Changes:
- build() returns sagemaker.core.resources.Model (not PySDK Model)
- deploy() returns sagemaker.core.resources.Endpoint (not Predictor)
- Use endpoint.invoke() instead of predictor.predict()
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import uuid

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model, Endpoint, TrainingJob
from sagemaker.core.enums import EndpointType
from sagemaker.core.inference_config import (
    AsyncInferenceConfig,
    ServerlessInferenceConfig,
    ResourceRequirements
)


class TestModelBuilderV3Build(unittest.TestCase):
    """Test V3 ModelBuilder.build() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        
        # Mock sagemaker client
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"
        self.mock_image_uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/test:latest"

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_returns_model_resource(self, mock_get_serve_setting, mock_build_single):
        """Test that build() returns a sagemaker.core.resources.Model (V3 behavior)."""
        # Setup
        mock_model = Mock(spec=Model)
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test-model"
        mock_build_single.return_value = mock_model
        mock_get_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        # Initialize built_model attribute before patching
        builder.built_model = None
        
        # Mock the built_model attribute that gets set during build
        with patch.object(builder, 'built_model', mock_model):
            # Execute
            result = builder.build()
            
            # Assert - V3 returns Model resource, not PySDK Model
            self.assertIsInstance(result, Mock)
            self.assertEqual(result, mock_model)
            mock_build_single.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_with_model_name_parameter(self, mock_get_serve_setting, mock_build_single):
        """Test build() with model_name parameter."""
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_get_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = None  # Initialize to avoid AttributeError
        
        result = builder.build(model_name="custom-model-name")
        
        self.assertEqual(builder.model_name, "custom-model-name")
        self.assertIsNotNone(result)

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_with_mode_override(self, mock_get_serve_setting, mock_build_single):
        """Test build() with mode parameter override."""
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_get_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = None  # Initialize to avoid AttributeError
        
        result = builder.build(mode=Mode.SAGEMAKER_ENDPOINT)
        
        self.assertEqual(builder.mode, Mode.SAGEMAKER_ENDPOINT)
        self.assertIsNotNone(result)

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_with_region_change(self, mock_get_serve_setting, mock_build_single):
        """Test build() with region parameter that differs from initialization."""
        mock_model = Mock(spec=Model)
        mock_build_single.return_value = mock_model
        mock_get_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        builder.region = "us-east-1"
        builder.built_model = None  # Initialize to avoid AttributeError
        
        with patch.object(builder, '_create_session_with_region') as mock_create_session:
            mock_create_session.return_value = self.mock_session
            result = builder.build(region="us-west-2")
        
        self.assertEqual(builder.region, "us-west-2")
        mock_create_session.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_warns_on_rebuild(self, mock_get_serve_setting, mock_build_single):
        """Test that build() warns when called multiple times."""
        mock_model = Mock(spec=Model)
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test-model"
        mock_build_single.return_value = mock_model
        mock_get_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        # Initialize built_model attribute
        builder.built_model = None
        
        # First build - set built_model
        with patch.object(builder, 'built_model', None):
            builder.build()
        
        # Now set built_model to simulate first build completed
        builder.built_model = mock_model
        
        # Second build should warn
        with patch('sagemaker.core.utils.utils.logger.warning') as mock_warning:
            with patch.object(builder, 'built_model', mock_model):
                builder.build()
                # Check that warning was called with message about rebuild
                self.assertTrue(mock_warning.called)
                call_args = str(mock_warning.call_args)
                self.assertIn("already been called", call_args)


class TestModelBuilderV3Deploy(unittest.TestCase):
    """Test V3 ModelBuilder.deploy() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_deploy_raises_error_without_build(self):
        """Test that deploy() raises error if build() was not called first."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        with self.assertRaises(ValueError) as context:
            builder.deploy()
        
        self.assertIn("Model needs to be built before deploying", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_returns_endpoint_resource(self, mock_deploy):
        """Test that deploy() returns sagemaker.core.resources.Endpoint (V3 behavior)."""
        # Setup
        mock_endpoint = Mock(spec=Endpoint)
        mock_endpoint.endpoint_name = "test-endpoint"
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        # Execute
        result = builder.deploy(endpoint_name="test-endpoint", wait=False)
        
        # Assert - V3 returns Endpoint resource, not Predictor
        self.assertIsInstance(result, Mock)
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_instance_based_config(self, mock_deploy):
        """Test deploy() with instance-based configuration."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            instance_type="ml.m5.xlarge",
            initial_instance_count=2,
            wait=False
        )
        
        self.assertIsNotNone(result)
        # Verify _deploy was called with correct parameters
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['instance_type'], "ml.m5.xlarge")
        self.assertEqual(call_kwargs['initial_instance_count'], 2)
        self.assertEqual(call_kwargs['endpoint_type'], EndpointType.MODEL_BASED)

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_serverless_config(self, mock_deploy):
        """Test deploy() with ServerlessInferenceConfig."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        serverless_config = ServerlessInferenceConfig(
            memory_size_in_mb=2048,
            max_concurrency=10
        )
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        builder.instance_type = None
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=serverless_config,
            wait=False
        )
        
        self.assertIsNotNone(result)
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['serverless_inference_config'], serverless_config)

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_async_config(self, mock_deploy):
        """Test deploy() with AsyncInferenceConfig."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        async_config = AsyncInferenceConfig(
            output_path="s3://bucket/output",
            max_concurrent_invocations_per_instance=5
        )
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=async_config,
            wait=False
        )
        
        self.assertIsNotNone(result)
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['async_inference_config'], async_config)

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_update_endpoint(self, mock_deploy):
        """Test deploy() with update_endpoint=True."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(
            endpoint_name="existing-endpoint",
            update_endpoint=True,
            wait=False
        )
        
        self.assertIsNotNone(result)
        call_kwargs = mock_deploy.call_args[1]
        self.assertTrue(call_kwargs['update_endpoint'])

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_generates_unique_endpoint_name(self, mock_deploy):
        """Test that deploy() generates unique endpoint name when not provided."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(wait=False)
        
        # Verify endpoint name was generated
        self.assertIsNotNone(builder.endpoint_name)
        self.assertTrue(builder.endpoint_name.startswith("endpoint-"))

    def test_deploy_warns_on_multiple_calls(self):
        """Test that deploy() warns when called multiple times."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = Mock(spec=Model)
        
        with patch.object(builder, '_deploy') as mock_deploy:
            mock_deploy.return_value = Mock(spec=Endpoint)
            
            # First deploy
            builder.deploy(wait=False)
            
            # Second deploy should warn
            with patch('sagemaker.core.utils.utils.logger.warning') as mock_warning:
                builder.deploy(wait=False)
                mock_warning.assert_called()


class TestModelBuilderV3BuildSingleModelBuilder(unittest.TestCase):
    """Test V3 ModelBuilder._build_single_modelbuilder() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.settings = Mock()
        self.mock_session.settings._local_download_dir = "/tmp/test"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    def test_build_single_with_pipeline_models(self, mock_create_model):
        """Test _build_single_modelbuilder with pipeline models (list of Models)."""
        mock_model1 = Mock(spec=Model)
        mock_model2 = Mock(spec=Model)
        mock_created_model = Mock(spec=Model)
        mock_create_model.return_value = mock_created_model
        
        builder = ModelBuilder(
            model=[mock_model1, mock_model2],
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_created_model)
        mock_create_model.assert_called_once()

    def test_build_single_with_invalid_pipeline_models(self):
        """Test _build_single_modelbuilder raises error for invalid pipeline models."""
        builder = ModelBuilder(
            model=[Mock(spec=Model), "not-a-model"],
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_single_modelbuilder()
        
        self.assertIn("must be sagemaker.core.resources.Model instances", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_torchserve')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_single_with_torchserve(
        self, mock_mlflow, mock_validations, mock_serve_setting, 
        mock_translators, mock_build_torchserve
    ):
        """Test _build_single_modelbuilder with TorchServe model server."""
        mock_model = Mock(spec=Model)
        mock_build_torchserve.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_torchserve.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_passthrough')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_single_with_passthrough(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_passthrough
    ):
        """Test _build_single_modelbuilder with passthrough mode."""
        mock_model = Mock(spec=Model)
        mock_build_passthrough.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            image_uri="123456789012.dkr.ecr.us-west-2.amazonaws.com/custom:latest",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        builder._passthrough = True
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_passthrough.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_single_with_jumpstart_model_id(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_is_js, mock_build_js
    ):
        """Test _build_single_modelbuilder with JumpStart model ID."""
        mock_model = Mock(spec=Model)
        mock_build_js.return_value = mock_model
        mock_is_js.return_value = True
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b-bf16",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_js.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_model_server')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_single_with_explicit_model_server(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_server
    ):
        """Test _build_single_modelbuilder with explicit model_server."""
        mock_model = Mock(spec=Model)
        mock_build_server.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TRITON
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_server.assert_called_once()



class TestModelBuilderV3TrainingJobIntegration(unittest.TestCase):
    """Test V3 ModelBuilder with TrainingJob integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.settings = Mock()
        self.mock_session.settings._local_download_dir = "/tmp/test"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_torchserve')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_with_training_job(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_torchserve
    ):
        """Test build() with TrainingJob as model input."""
        mock_training_job = Mock(spec=TrainingJob)
        mock_training_job.model_artifacts = Mock()
        mock_training_job.model_artifacts.s3_model_artifacts = "s3://bucket/model.tar.gz"
        
        mock_model = Mock(spec=Model)
        mock_build_torchserve.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=mock_training_job,
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE,
            inference_spec=Mock()  # Add inference_spec to avoid validation error
        )
        
        result = builder._build_single_modelbuilder()
        
        # Verify model_path was set from TrainingJob
        self.assertEqual(builder.model_path, "s3://bucket/model.tar.gz")
        self.assertIsNone(builder.model)
        self.assertEqual(result, mock_model)


class TestModelBuilderV3Validations(unittest.TestCase):
    """Test V3 ModelBuilder validation methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    def test_validation_model_and_inference_spec_mutually_exclusive(self):
        """Test that model and inference_spec cannot both be set."""
        from sagemaker.serve.spec.inference_spec import InferenceSpec
        
        mock_inference_spec = Mock(spec=InferenceSpec)
        
        builder = ModelBuilder(
            model=Mock(),
            inference_spec=mock_inference_spec,
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        
        self.assertIn("Can only set one of the following: model, inference_spec", str(context.exception))

    def test_validation_custom_image_requires_model_server(self):
        """Test that custom image_uri requires model_server to be set."""
        builder = ModelBuilder(
            image_uri="custom-image:latest",
            model=Mock(),  # Add model to avoid passthrough mode
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_validations()
        
        self.assertIn("Model_server must be set when non-first-party image_uri is set", str(context.exception))

    def test_validation_passthrough_with_first_party_image(self):
        """Test passthrough mode with first-party image."""
        builder = ModelBuilder(
            image_uri="763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:2.0-gpu-py310",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        # Should not raise - passthrough is allowed with 1P images
        builder._build_validations()
        self.assertTrue(builder._passthrough)


class TestModelBuilderV3WaitForEndpoint(unittest.TestCase):
    """Test V3 ModelBuilder._wait_for_endpoint() method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        
        self.mock_client = Mock()
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder._wait_until')
    def test_wait_for_endpoint_with_wait_true(self, mock_wait_until):
        """Test _wait_for_endpoint with wait=True."""
        # Setup mock to return successful endpoint status
        mock_wait_until.return_value = {'EndpointStatus': 'InService'}
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        builder.model_server = ModelServer.TORCHSERVE
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        
        # Call _wait_for_endpoint
        builder._wait_for_endpoint("test-endpoint", wait=True, show_progress=False)
        
        # Verify _wait_until was called
        mock_wait_until.assert_called_once()

    def test_wait_for_endpoint_with_wait_false(self):
        """Test _wait_for_endpoint with wait=False (no waiting)."""
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        with patch('sagemaker.core.utils.utils.logger.info') as mock_logger:
            builder._wait_for_endpoint("test-endpoint", wait=False, show_progress=False)
            # Should log deployment started message
            mock_logger.assert_called()


class TestModelBuilderV3EndToEnd(unittest.TestCase):
    """Test V3 ModelBuilder end-to-end workflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.boto_session.region_name = "us-west-2"
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.default_bucket_prefix = "test-prefix"
        self.mock_session.settings = Mock()
        self.mock_session.settings._local_download_dir = "/tmp/test"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_then_deploy_workflow(self, mock_serve_setting, mock_build_single, mock_deploy):
        """Test complete V3 workflow: build() -> deploy() -> invoke()."""
        # Setup mocks
        mock_model = Mock(spec=Model)
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test-model"
        mock_build_single.return_value = mock_model
        mock_serve_setting.return_value = Mock()
        
        mock_endpoint = Mock(spec=Endpoint)
        mock_endpoint.endpoint_name = "test-endpoint"
        mock_endpoint.invoke = Mock(return_value={"predictions": [1, 2, 3]})
        mock_deploy.return_value = mock_endpoint
        
        # Create builder
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            instance_type="ml.m5.xlarge",
            model_server=ModelServer.TORCHSERVE
        )
        
        # Initialize built_model attribute
        builder.built_model = None
        
        # Build model (V3 returns Model resource)
        with patch.object(builder, 'built_model', mock_model):
            model = builder.build()
            self.assertIsInstance(model, Mock)
            self.assertEqual(model, mock_model)
        
        # Set built_model for deploy
        builder.built_model = mock_model
        
        # Deploy model (V3 returns Endpoint resource)
        endpoint = builder.deploy(endpoint_name="test-endpoint", wait=False)
        self.assertIsInstance(endpoint, Mock)
        self.assertEqual(endpoint, mock_endpoint)
        
        # Invoke endpoint (V3 uses endpoint.invoke(), not predictor.predict())
        result = endpoint.invoke(data={"input": "test"})
        self.assertEqual(result, {"predictions": [1, 2, 3]})

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_with_different_modes(self, mock_serve_setting, mock_build_single, mock_deploy):
        """Test building with different deployment modes."""
        mock_model = Mock(spec=Model)
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123456789012:model/test-model"
        mock_build_single.return_value = mock_model
        mock_serve_setting.return_value = Mock()
        
        # Test SAGEMAKER_ENDPOINT mode
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_server=ModelServer.TORCHSERVE
        )
        builder.built_model = None  # Initialize attribute
        with patch.object(builder, 'built_model', mock_model):
            result = builder.build()
            self.assertEqual(builder.mode, Mode.SAGEMAKER_ENDPOINT)
        
        # Test LOCAL_CONTAINER mode
        builder2 = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            mode=Mode.LOCAL_CONTAINER,
            model_server=ModelServer.TORCHSERVE
        )
        builder2.built_model = None  # Initialize attribute
        with patch.object(builder2, 'built_model', mock_model):
            result2 = builder2.build()
            self.assertEqual(builder2.mode, Mode.LOCAL_CONTAINER)


class TestModelBuilderV3ModelServers(unittest.TestCase):
    """Test V3 ModelBuilder with different model servers."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = Mock()
        self.mock_session.boto_region_name = "us-west-2"
        self.mock_session.boto_session = Mock()
        self.mock_session.config = {}
        self.mock_session.sagemaker_config = {}
        self.mock_session.default_bucket.return_value = "test-bucket"
        self.mock_session.settings = Mock()
        self.mock_session.settings._local_download_dir = "/tmp/test"
        
        self.mock_client = Mock()
        self.mock_client._user_agent_creator = Mock()
        self.mock_client._user_agent_creator.to_string = Mock(return_value="test-agent")
        self.mock_session.sagemaker_client = self.mock_client
        
        self.mock_role_arn = "arn:aws:iam::123456789012:role/TestRole"

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_torchserve')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_with_torchserve(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_torchserve
    ):
        """Test build with TorchServe model server."""
        mock_model = Mock(spec=Model)
        mock_build_torchserve.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TORCHSERVE
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_torchserve.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_triton')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_with_triton(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_triton
    ):
        """Test build with Triton model server."""
        mock_model = Mock(spec=Model)
        mock_build_triton.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.TRITON
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_triton.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_djl')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    def test_build_with_djl(
        self, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_build_djl
    ):
        """Test build with DJL Serving model server."""
        mock_model = Mock(spec=Model)
        mock_build_djl.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        
        builder = ModelBuilder(
            model=Mock(),
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session,
            model_server=ModelServer.DJL_SERVING
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_djl.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tgi')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_huggingface_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder.get_huggingface_model_metadata')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_validations')
    @patch('sagemaker.serve.model_builder.ModelBuilder._handle_mlflow_input')
    @patch('sagemaker.serve.model_builder.ModelBuilder._hf_schema_builder_init')
    def test_build_with_tgi_for_text_generation(
        self, mock_hf_schema_init, mock_mlflow, mock_validations, mock_serve_setting,
        mock_translators, mock_hf_metadata, mock_is_hf, mock_build_tgi
    ):
        """Test build with TGI for text-generation models."""
        mock_model = Mock(spec=Model)
        mock_build_tgi.return_value = mock_model
        mock_translators.return_value = (Mock(), Mock())
        mock_serve_setting.return_value = Mock()
        mock_is_hf.return_value = True
        mock_hf_metadata.return_value = {"pipeline_tag": "text-generation"}
        mock_hf_schema_init.return_value = None  # Skip schema initialization
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=self.mock_role_arn,
            sagemaker_session=self.mock_session
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_tgi.assert_called_once()


if __name__ == '__main__':
    unittest.main()
