"""
Comprehensive tests for model_builder_servers.py to boost coverage.
Tests the _ModelBuilderServers mixin methods.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model

from .test_fixtures import (
    mock_sagemaker_session,
    mock_model_object,
    MOCK_ROLE_ARN,
    MOCK_IMAGE_URI,
    MOCK_S3_URI
)


class TestBuildForModelServer(unittest.TestCase):
    """Test _build_for_model_server routing logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    def test_build_for_model_server_unsupported_raises(self):
        """Test that unsupported model server raises error."""
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = "UNSUPPORTED_SERVER"
        
        with self.assertRaises(ValueError) as context:
            builder._build_for_model_server()
        
        self.assertIn("is not supported yet", str(context.exception))

    def test_build_for_model_server_without_model_raises(self):
        """Test that missing model/inference_spec raises error."""
        builder = ModelBuilder(
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model = None
        builder.inference_spec = None
        builder.model_metadata = None
        builder.model_server = ModelServer.TORCHSERVE
        
        with self.assertRaises(ValueError) as context:
            builder._build_for_model_server()
        
        self.assertIn("Missing required parameter", str(context.exception))

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_torchserve')
    def test_build_for_model_server_routes_to_torchserve(self, mock_build):
        """Test routing to TorchServe builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TORCHSERVE
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_djl')
    def test_build_for_model_server_routes_to_djl(self, mock_build):
        """Test routing to DJL builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.DJL_SERVING
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tgi')
    def test_build_for_model_server_routes_to_tgi(self, mock_build):
        """Test routing to TGI builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TGI
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tei')
    def test_build_for_model_server_routes_to_tei(self, mock_build):
        """Test routing to TEI builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TEI
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_triton')
    def test_build_for_model_server_routes_to_triton(self, mock_build):
        """Test routing to Triton builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TRITON
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tensorflow_serving')
    def test_build_for_model_server_routes_to_tensorflow(self, mock_build):
        """Test routing to TensorFlow Serving builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TENSORFLOW_SERVING
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_transformers')
    def test_build_for_model_server_routes_to_mms(self, mock_build):
        """Test routing to MMS/Transformers builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.MMS
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_smd')
    def test_build_for_model_server_routes_to_smd(self, mock_build):
        """Test routing to SMD builder."""
        mock_model = Mock(spec=Model)
        mock_build.return_value = mock_model
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.SMD
        
        result = builder._build_for_model_server()
        
        self.assertEqual(result, mock_model)
        mock_build.assert_called_once()


class TestBuildForTorchServe(unittest.TestCase):
    """Test _build_for_torchserve method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder.save_pkl')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder._detect_framework_and_version')
    @patch('sagemaker.serve.model_builder._get_model_base')
    def test_build_for_torchserve_with_model_object(self, mock_get_base, mock_detect_fw, mock_prepare, mock_create, mock_save_pkl):
        """Test TorchServe build with model object."""
        mock_model = Mock(spec=Model)
        mock_create.return_value = mock_model
        mock_get_base.return_value = mock_model_object()
        mock_detect_fw.return_value = ("pytorch", "1.13.0")
        
        builder = ModelBuilder(
            model=mock_model_object(),
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.IN_PROCESS,  # Use IN_PROCESS to skip prepare_for_torchserve
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TORCHSERVE
        
        result = builder._build_for_torchserve()
        
        self.assertEqual(result, mock_model)
        mock_save_pkl.assert_called()
        mock_create.assert_called_once()

    @patch('sagemaker.serve.model_builder.save_pkl')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    def test_build_for_torchserve_with_hf_model_id(self, mock_is_js, mock_prepare, mock_create, mock_save_pkl):
        """Test TorchServe build with HuggingFace model ID."""
        mock_is_js.return_value = False
        mock_model = Mock(spec=Model)
        mock_create.return_value = mock_model
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.IN_PROCESS  # Use IN_PROCESS to skip prepare_for_torchserve
        )
        builder.model_server = ModelServer.TORCHSERVE
        builder.env_vars = {}
        builder.image_uri = MOCK_IMAGE_URI
        
        result = builder._build_for_torchserve()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertIsNone(builder.s3_upload_path)

    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._save_model_inference_spec')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    def test_build_for_torchserve_with_hf_token(self, mock_is_js, mock_save, mock_prepare, mock_create):
        """Test TorchServe build with HuggingFace token."""
        mock_is_js.return_value = False
        mock_model = Mock(spec=Model)
        mock_create.return_value = mock_model
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.IN_PROCESS
        )
        builder.model_server = ModelServer.TORCHSERVE
        builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "hf_token_123"}
        
        result = builder._build_for_torchserve()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.env_vars["HF_TOKEN"], "hf_token_123")


class TestBuildForJumpStart(unittest.TestCase):
    """Test _build_for_jumpstart method."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    @patch('sagemaker.core.jumpstart.factory.utils.get_init_kwargs')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    def test_build_for_jumpstart_unsupported_image_raises(self, mock_prepare, mock_get_kwargs):
        """Test that unsupported JumpStart image raises error."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "unsupported-image:latest"
        mock_init_kwargs.env = {}
        mock_get_kwargs.return_value = mock_init_kwargs
        
        builder = ModelBuilder(
            model="some-model-id",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder._optimizing = False
        
        with self.assertRaises(ValueError) as context:
            builder._build_for_jumpstart()
        
        self.assertIn("Unsupported JumpStart image URI", str(context.exception))

    @patch('sagemaker.core.jumpstart.factory.utils.get_init_kwargs')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_djl_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    def test_build_for_jumpstart_routes_to_djl(self, mock_prepare, mock_build_djl, mock_get_kwargs):
        """Test JumpStart routing to DJL builder."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.0-cu117"
        mock_init_kwargs.env = {}
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock(spec=Model)
        mock_build_djl.return_value = mock_model

        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder._optimizing = False

        result = builder._build_for_jumpstart()

        self.assertEqual(result, mock_model)
        self.assertEqual(builder.model_server, ModelServer.DJL_SERVING)
        mock_build_djl.assert_called_once()

    @patch('sagemaker.core.jumpstart.factory.utils.get_init_kwargs')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_djl_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    def test_build_for_jumpstart_passes_config_name(self, mock_prepare, mock_build_djl, mock_get_kwargs):
        """Test that config_name is forwarded to get_init_kwargs."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.21.0-deepspeed0.8.0-cu117"
        mock_init_kwargs.env = {}
        mock_get_kwargs.return_value = mock_init_kwargs

        mock_model = Mock(spec=Model)
        mock_build_djl.return_value = mock_model

        builder = ModelBuilder(
            model="meta-textgeneration-llama-3-3-70b-instruct",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder._optimizing = False
        builder.config_name = "lmi-optimized"

        builder._build_for_jumpstart()

        mock_get_kwargs.assert_called_once()
        call_kwargs = mock_get_kwargs.call_args
        self.assertEqual(call_kwargs.kwargs.get("config_name") or call_kwargs[1].get("config_name"), "lmi-optimized")

    @patch('sagemaker.core.jumpstart.factory.utils.get_init_kwargs')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_tgi_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    def test_build_for_jumpstart_routes_to_tgi(self, mock_prepare, mock_build_tgi, mock_get_kwargs):
        """Test JumpStart routing to TGI builder."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-tgi-inference:2.0.1-tgi0.9.3-gpu-py39-cu118-ubuntu20.04"
        mock_init_kwargs.env = {}
        mock_get_kwargs.return_value = mock_init_kwargs
        
        mock_model = Mock(spec=Model)
        mock_build_tgi.return_value = mock_model
        
        builder = ModelBuilder(
            model="huggingface-llm-mistral-7b",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder._optimizing = False
        
        result = builder._build_for_jumpstart()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.model_server, ModelServer.TGI)
        mock_build_tgi.assert_called_once()

    @patch('sagemaker.core.jumpstart.factory.utils.get_init_kwargs')
    @patch('sagemaker.serve.model_builder.ModelBuilder._build_for_mms_jumpstart')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    def test_build_for_jumpstart_routes_to_mms(self, mock_prepare, mock_build_mms, mock_get_kwargs):
        """Test JumpStart routing to MMS builder."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/huggingface-pytorch-inference:1.13.1-transformers4.26.0-gpu-py39-cu117-ubuntu20.04"
        mock_init_kwargs.env = {}
        mock_get_kwargs.return_value = mock_init_kwargs
        
        mock_model = Mock(spec=Model)
        mock_build_mms.return_value = mock_model
        
        builder = ModelBuilder(
            model="pytorch-ic-mobilenet-v2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            mode=Mode.SAGEMAKER_ENDPOINT
        )
        builder._optimizing = False
        
        result = builder._build_for_jumpstart()
        
        self.assertEqual(result, mock_model)
        self.assertEqual(builder.model_server, ModelServer.MMS)
        mock_build_mms.assert_called_once()


class TestDeployWrappers(unittest.TestCase):
    """Test deploy wrapper methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = mock_sagemaker_session()

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint')
    def test_djl_deploy_wrapper_sets_timeout(self, mock_deploy):
        """Test DJL deploy wrapper sets model data download timeout."""
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.DJL_SERVING
        builder.built_model = Mock(spec=Model)
        
        result = builder._djl_model_builder_deploy_wrapper(
            model_data_download_timeout=1800
        )
        
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()
        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs['model_data_download_timeout'], 1800)

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint')
    def test_tgi_deploy_wrapper_calls_core_deploy(self, mock_deploy):
        """Test TGI deploy wrapper calls core deploy."""
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="gpt2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TGI
        builder.built_model = Mock(spec=Model)
        
        result = builder._tgi_model_builder_deploy_wrapper(
            endpoint_name="test-endpoint"
        )
        
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint')
    def test_tei_deploy_wrapper_calls_core_deploy(self, mock_deploy):
        """Test TEI deploy wrapper calls core deploy."""
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="sentence-transformers/all-MiniLM-L6-v2",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.TEI
        builder.built_model = Mock(spec=Model)
        
        result = builder._tei_model_builder_deploy_wrapper(
            endpoint_name="test-endpoint"
        )
        
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint')
    def test_js_deploy_wrapper_calls_core_deploy(self, mock_deploy):
        """Test JumpStart deploy wrapper calls core deploy."""
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="huggingface-llm-falcon-7b",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.built_model = Mock(spec=Model)
        
        result = builder._js_builder_deploy_wrapper(
            endpoint_name="test-endpoint"
        )
        
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()

    @patch('sagemaker.serve.model_builder.ModelBuilder._deploy_core_endpoint')
    def test_transformers_deploy_wrapper_calls_core_deploy(self, mock_deploy):
        """Test Transformers deploy wrapper calls core deploy."""
        mock_endpoint = Mock()
        mock_deploy.return_value = mock_endpoint
        
        builder = ModelBuilder(
            model="bert-base-uncased",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            image_uri=MOCK_IMAGE_URI
        )
        builder.model_server = ModelServer.MMS
        builder.built_model = Mock(spec=Model)
        
        result = builder._transformers_model_builder_deploy_wrapper(
            endpoint_name="test-endpoint"
        )
        
        self.assertEqual(result, mock_endpoint)
        mock_deploy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
