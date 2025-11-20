"""
Comprehensive workflow tests for ModelBuilder to improve coverage.
Focuses on build() and deploy() methods with various configurations.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, call
import tempfile
import os

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer, ModelHub
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.constants import Framework
from sagemaker.core.resources import Model, Endpoint
from sagemaker.core.inference_config import ServerlessInferenceConfig, AsyncInferenceConfig, ResourceRequirements
from sagemaker.serve.batch_inference.batch_transform_inference_config import BatchTransformInferenceConfig

from .test_fixtures import (
    mock_sagemaker_session,
    mock_model_object,
    MOCK_ROLE_ARN,
    MOCK_IMAGE_URI,
    MOCK_S3_URI
)


class TestModelBuilderBuildMethod(unittest.TestCase):
    """Test build() method with various configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    def test_build_simple_model_returns_model(self, mock_get_serve, mock_build_single):
        """Test that build() returns a Model for simple case."""
        mock_model = Mock(spec=Model)
        mock_model.model_name = "test-model"
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123:model/test"
        
        # Mock _build_single_modelbuilder to set built_model as a side effect
        def set_built_model(*args, **kwargs):
            builder.built_model = mock_model
            return mock_model
        
        mock_build_single.side_effect = set_built_model
        mock_get_serve.return_value = Mock()
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder.model_name = "test-model"
        builder.model_server = ModelServer.TORCHSERVE
        builder.modelbuilder_list = None
        builder.inference_spec = None
        
        result = builder.build()
        
        self.assertIsNotNone(result)
        # build() sets built_model as a side effect
        self.assertEqual(builder.built_model, mock_model)
        self.assertEqual(result, mock_model)
        mock_build_single.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    def test_build_with_modelbuilder_list_raises_for_local_mode(self, mock_build_single):
        """Test that bulk building raises error for LOCAL_CONTAINER mode."""
        from sagemaker.serve.spec.inference_spec import InferenceSpec
        
        # Create a ModelBuilder with LOCAL_CONTAINER mode
        mb1 = ModelBuilder(
            inference_spec=Mock(spec=InferenceSpec),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.LOCAL_CONTAINER
        )
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder.modelbuilder_list = [mb1]
        
        with self.assertRaises(ValueError) as context:
            builder.build()
        
        self.assertIn("only supported for SageMaker Endpoint Mode", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_single_modelbuilder')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_serve_setting')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_inference_component_resource_requirements')
    def test_build_with_modelbuilder_list_builds_inference_components(
        self, mock_get_ic_reqs, mock_get_serve, mock_build_single
    ):
        """Test bulk building with inference components."""
        from sagemaker.serve.spec.inference_spec import InferenceSpec
        
        # Setup mocks
        mock_model = Mock(spec=Model)
        mock_model.model_name = "test-model"
        mock_model.model_arn = "arn:aws:sagemaker:us-west-2:123:model/test"
        mock_build_single.return_value = mock_model
        mock_get_serve.return_value = Mock()
        
        # Create ModelBuilder with inference component
        mb1 = ModelBuilder(
            inference_spec=Mock(spec=InferenceSpec),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        mb1.inference_component_name = "ic-1"
        mb1.resource_requirements = ResourceRequirements(
            requests={"memory": 1024, "copies": 1},
            limits={}
        )
        mb1.model_name = "test-model-1"
        mb1.model_server = ModelServer.TORCHSERVE
        mb1.built_model = mock_model
        mock_get_ic_reqs.return_value = mb1
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder.modelbuilder_list = [mb1]
        builder.model_name = "test-model"
        builder.model_server = ModelServer.TORCHSERVE
        
        result = builder.build()
        
        self.assertIsNotNone(result)
        self.assertIn("InferenceComponents", builder._deployables)
        self.assertEqual(len(builder._deployables["InferenceComponents"]), 1)


class TestModelBuilderDeployMethod(unittest.TestCase):
    """Test deploy() method with various configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_deploy_without_build_raises_error(self):
        """Test that deploy() raises error if build() wasn't called."""
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        
        with self.assertRaises(ValueError) as context:
            builder.deploy()
        
        self.assertIn("Model needs to be built before deploying", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_generates_unique_endpoint_name(self, mock_deploy):
        """Test that deploy() generates unique endpoint name when not provided."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            instance_type="ml.m5.large"
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(instance_type="ml.m5.large")
        
        self.assertIsNotNone(result)
        # Verify endpoint name was generated (contains uuid)
        self.assertIn("endpoint-", builder.endpoint_name)
        mock_deploy.assert_called_once()

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
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.built_model = Mock(spec=Model)
        builder.instance_type = "ml.m5.large"
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            inference_config=serverless_config
        )
        
        self.assertIsNotNone(result)
        mock_deploy.assert_called_once()
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['serverless_inference_config'], serverless_config)

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy')
    def test_deploy_with_async_config(self, mock_deploy):
        """Test deploy() with AsyncInferenceConfig."""
        mock_endpoint = Mock(spec=Endpoint)
        mock_deploy.return_value = mock_endpoint
        
        async_config = AsyncInferenceConfig(
            output_path=MOCK_S3_URI,
            max_concurrent_invocations_per_instance=5
        )
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            instance_type="ml.m5.large"
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder.deploy(
            endpoint_name="test-endpoint",
            instance_type="ml.m5.large",
            inference_config=async_config
        )
        
        self.assertIsNotNone(result)
        mock_deploy.assert_called_once()
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['async_inference_config'], async_config)

    @patch('sagemaker.serve.model_builder.Transformer')
    def test_deploy_with_batch_transform_config(self, mock_transformer_class):
        """Test deploy() with BatchTransformInferenceConfig."""
        mock_transformer = Mock()
        mock_transformer_class.return_value = mock_transformer
        
        batch_config = BatchTransformInferenceConfig(
            instance_count=1,
            instance_type="ml.m5.large",
            output_path=MOCK_S3_URI,
            max_payload_in_mb=6,
            max_concurrent_transforms=4
        )
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            instance_type="ml.m5.large"
        )
        builder.built_model = Mock(spec=Model)
        builder.built_model.model_name = "test-model"
        
        result = builder.deploy(
            endpoint_name="test-job",
            instance_type="ml.m5.large",
            inference_config=batch_config
        )
        
        self.assertIsNotNone(result)
        mock_transformer_class.assert_called_once()

    def test_deploy_warns_on_multiple_calls(self):
        """Test that deploy() warns when called multiple times."""
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI,
            instance_type="ml.m5.large"
        )
        builder.built_model = Mock(spec=Model)
        builder._deployed = True
        
        with patch('sagemaker.serve.model_builder.ModelBuilder._deploy') as mock_deploy:
            mock_deploy.return_value = Mock(spec=Endpoint)
            
            with patch('sagemaker.serve.model_builder.logger') as mock_logger:
                builder.deploy(instance_type="ml.m5.large")
                
                # Verify warning was logged
                mock_logger.warning.assert_called()
                warning_msg = mock_logger.warning.call_args[0][0]
                self.assertIn("already been called", warning_msg)


class TestModelBuilderJumpStartWorkflow(unittest.TestCase):
    """Test JumpStart model workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    def test_build_single_with_jumpstart_model_id(self, mock_get_trans, mock_build_js, mock_is_js):
        """Test _build_single_modelbuilder with JumpStart model ID."""
        mock_is_js.return_value = True
        mock_model = Mock(spec=Model)
        mock_build_js.return_value = mock_model
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.model_hub, ModelHub.JUMPSTART)
        mock_build_js.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    def test_build_single_jumpstart_raises_for_in_process_mode(self, mock_get_trans, mock_is_js):
        """Test that JumpStart models raise error for IN_PROCESS mode."""
        mock_is_js.return_value = True
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.IN_PROCESS,
            image_uri=MOCK_IMAGE_URI
        )
        
        with self.assertRaises(ValueError) as context:
            builder._build_single_modelbuilder()
        
        self.assertIn("not supported for JumpStart models", str(context.exception))


class TestModelBuilderHuggingFaceWorkflow(unittest.TestCase):
    """Test HuggingFace model workflows."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_huggingface_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder.get_huggingface_model_metadata')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tgi')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._use_jumpstart_equivalent')
    @patch('sagemaker.serve.model_builder.ModelBuilder._hf_schema_builder_init')
    def test_build_single_with_hf_text_generation(self, mock_schema_init, mock_use_js, mock_is_js, mock_get_trans, mock_build_tgi, mock_get_md, mock_is_hf):
        """Test _build_single_modelbuilder with HF text-generation model."""
        mock_is_hf.return_value = True
        mock_is_js.return_value = False
        mock_use_js.return_value = False
        mock_get_md.return_value = {"pipeline_tag": "text-generation"}
        mock_model = Mock(spec=Model)
        mock_build_tgi.return_value = mock_model
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.model_hub, ModelHub.HUGGINGFACE)
        mock_build_tgi.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_huggingface_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder.get_huggingface_model_metadata')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tei')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._use_jumpstart_equivalent')
    @patch('sagemaker.serve.model_builder.ModelBuilder._hf_schema_builder_init')
    def test_build_single_with_hf_sentence_similarity(self, mock_schema_init, mock_use_js, mock_is_js, mock_get_trans, mock_build_tei, mock_get_md, mock_is_hf):
        """Test _build_single_modelbuilder with HF sentence-similarity model."""
        mock_is_hf.return_value = True
        mock_is_js.return_value = False
        mock_use_js.return_value = False
        mock_get_md.return_value = {"pipeline_tag": "sentence-similarity"}
        mock_model = Mock(spec=Model)
        mock_build_tei.return_value = mock_model
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_tei.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_huggingface_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder.get_huggingface_model_metadata')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_transformers')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._use_jumpstart_equivalent')
    @patch('sagemaker.serve.model_builder.ModelBuilder._hf_schema_builder_init')
    def test_build_single_with_hf_other_task(self, mock_schema_init, mock_use_js, mock_is_js, mock_get_trans, mock_build_transformers, mock_get_md, mock_is_hf):
        """Test _build_single_modelbuilder with HF other task types."""
        mock_is_hf.return_value = True
        mock_is_js.return_value = False
        mock_use_js.return_value = False
        mock_get_md.return_value = {"pipeline_tag": "image-classification"}
        mock_model = Mock(spec=Model)
        mock_build_transformers.return_value = mock_model
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="google/vit-base-patch16-224",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_transformers.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._is_huggingface_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_djl')
    @patch('sagemaker.serve.model_builder.ModelBuilder._get_client_translators')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder.ModelBuilder._use_jumpstart_equivalent')
    def test_build_single_with_hf_djl_server(self, mock_use_js, mock_is_js, mock_get_trans, mock_build_djl, mock_is_hf):
        """Test _build_single_modelbuilder with HF model using DJL server."""
        mock_is_hf.return_value = True
        mock_is_js.return_value = False
        mock_use_js.return_value = False
        mock_model = Mock(spec=Model)
        mock_build_djl.return_value = mock_model
        mock_get_trans.return_value = (None, None)
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT,
            model_server=ModelServer.DJL_SERVING,
            image_uri=MOCK_IMAGE_URI
        )
        
        result = builder._build_single_modelbuilder()
        
        self.assertEqual(result, mock_model)
        mock_build_djl.assert_called_once()


if __name__ == "__main__":
    unittest.main()
