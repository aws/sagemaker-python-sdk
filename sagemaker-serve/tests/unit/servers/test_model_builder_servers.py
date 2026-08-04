"""Unit tests for ModelBuilderServers class."""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import unittest

# Prevent JumpStart from loading region config during import
os.environ["SAGEMAKER_INTERNAL_SKIP_REGION_CONFIG"] = "1"

from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_builder_servers import _ModelBuilderServers


class MockModelBuilderServers(_ModelBuilderServers):
    """Mock class that inherits _ModelBuilderServers behavior."""

    def __init__(self):
        self.model_server = ModelServer.TORCHSERVE
        self.model = None
        self.model_metadata = {}
        self.inference_spec = None
        self.mode = Mode.SAGEMAKER_ENDPOINT
        self.model_path = tempfile.mkdtemp()
        self.shared_libs = []
        self.dependencies = {}
        self.sagemaker_session = Mock()
        self.image_uri = "test-image-uri"
        self.secret_key = ""
        self.env_vars = {}
        self.schema_builder = Mock()
        self.schema_builder.sample_input = {"inputs": "test"}
        self.hf_model_config = {}
        self.s3_upload_path = None
        self.s3_model_data_url = None
        self.instance_type = "ml.m5.large"
        self._user_provided_instance_type = False
        self._optimizing = False
        self.model_data_download_timeout = None
        self.role_arn = "arn:aws:iam::123456789012:role/test"
        self.region = "us-east-1"
        self.model_version = None
        self.framework = None
        self.framework_version = None
        self._is_mlflow_model = False
        self.config_name = None
        self._enable_network_isolation = False

    def _deploy_local_endpoint(self, **kwargs):
        return Mock()

    def _deploy_core_endpoint(self, *args, **kwargs):
        return Mock()

    def _save_model_inference_spec(self):
        pass

    def _is_jumpstart_model_id(self):
        return False

    def _auto_detect_image_uri(self):
        pass

    def _prepare_for_mode(self, should_upload_artifacts=False):
        return ("s3://bucket/model.tar.gz", None)

    def _create_model(self):
        return Mock()

    def _validate_tgi_serving_sample_data(self):
        pass

    def _validate_djl_serving_sample_data(self):
        pass

    def _validate_for_triton(self):
        pass

    def _auto_detect_image_for_triton(self):
        pass

    def _save_inference_spec(self):
        pass

    def _prepare_for_triton(self):
        pass

    def get_huggingface_model_metadata(self, model_id, token=None):
        return {}

    def _normalize_framework_to_enum(self, framework):
        return framework

    def _get_processing_unit(self):
        return "cpu"

    def _get_smd_image_uri(self, processing_unit):
        return "smd-image-uri"

    def _create_conda_env(self):
        pass


class TestBuildForModelServer(unittest.TestCase):
    """Test _build_for_model_server method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    def test_unsupported_model_server(self):
        """Test error for unsupported model server."""
        self.builder.model_server = "INVALID_SERVER"
        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_model_server()
        self.assertIn("not supported", str(ctx.exception))

    def test_missing_required_parameters(self):
        """Test error when model, MLflow path, and inference_spec are all missing."""
        self.builder.model = None
        self.builder.model_metadata = {}
        self.builder.inference_spec = None
        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_model_server()
        self.assertIn("Missing required parameter", str(ctx.exception))

    @patch.object(MockModelBuilderServers, "_build_for_torchserve")
    def test_route_to_torchserve(self, mock_build):
        """Test routing to TorchServe builder."""
        self.builder.model_server = ModelServer.TORCHSERVE
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_triton")
    def test_route_to_triton(self, mock_build):
        """Test routing to Triton builder."""
        self.builder.model_server = ModelServer.TRITON
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tensorflow_serving")
    def test_route_to_tensorflow_serving(self, mock_build):
        """Test routing to TensorFlow Serving builder."""
        self.builder.model_server = ModelServer.TENSORFLOW_SERVING
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_djl")
    def test_route_to_djl(self, mock_build):
        """Test routing to DJL builder."""
        self.builder.model_server = ModelServer.DJL_SERVING
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tei")
    def test_route_to_tei(self, mock_build):
        """Test routing to TEI builder."""
        self.builder.model_server = ModelServer.TEI
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_tgi")
    def test_route_to_tgi(self, mock_build):
        """Test routing to TGI builder."""
        self.builder.model_server = ModelServer.TGI
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_transformers")
    def test_route_to_mms(self, mock_build):
        """Test routing to MMS builder."""
        self.builder.model_server = ModelServer.MMS
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()

    @patch.object(MockModelBuilderServers, "_build_for_smd")
    def test_route_to_smd(self, mock_build):
        """Test routing to SMD builder."""
        self.builder.model_server = ModelServer.SMD
        self.builder.model = Mock()
        mock_build.return_value = Mock()
        self.builder._build_for_model_server()
        mock_build.assert_called_once()


class TestBuildForTorchServe(unittest.TestCase):
    """Test _build_for_torchserve method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TORCHSERVE

    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_id(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_save
    ):
        """Test building with HuggingFace model ID."""
        mock_js.return_value = False
        mock_create.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS
        self.builder.model = "bert-base-uncased"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "test-token"}

        result = self.builder._build_for_torchserve()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "bert-base-uncased")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "test-token")
        self.assertIsNone(self.builder.s3_upload_path)
        mock_save.assert_called_once()
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_for_torchserve")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_local_container_mode(
        self, mock_create, mock_prepare, mock_detect, mock_save, mock_ts_prepare
    ):
        """Test building for LOCAL_CONTAINER mode."""
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = Mock()
        mock_ts_prepare.return_value = ""
        mock_create.return_value = Mock()

        result = self.builder._build_for_torchserve()

        mock_ts_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_for_torchserve")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_mode(
        self, mock_create, mock_prepare, mock_detect, mock_save, mock_ts_prepare
    ):
        """Test building for SAGEMAKER_ENDPOINT mode."""
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        mock_ts_prepare.return_value = ""
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        result = self.builder._build_for_torchserve()

        mock_ts_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_prepare.assert_called_with(should_upload_artifacts=True)


class TestBuildForTGI(unittest.TestCase):
    """Test _build_for_tgi method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TGI

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_notebook_instance(
        self, mock_create, mock_prepare, mock_detect, mock_validate, mock_dir, mock_nb
    ):
        """Test building with notebook instance detection."""
        mock_nb.return_value = "ml.g4dn.xlarge"
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.instance_type, "ml.g4dn.xlarge")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tgi_config,
        mock_hf_config,
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_tgi_config.return_value = ({"MAX_INPUT_LENGTH": "1024"}, 512)
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(self.builder.env_vars["SHARDED"], "false")
        self.assertEqual(self.builder.env_vars["NUM_SHARD"], "1")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_with_gpu(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
    ):
        """Test building for SAGEMAKER_ENDPOINT with GPU sharding."""
        mock_nb.return_value = None
        mock_gpu.return_value = 4
        mock_tp.return_value = 2
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        self.builder.hf_model_config = {"model_type": "gpt2"}

        result = self.builder._build_for_tgi()

        self.assertEqual(self.builder.env_vars["NUM_SHARD"], "2")
        self.assertEqual(self.builder.env_vars["SHARDED"], "true")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info_fallback")
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_tgi_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_gpu_fallback(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
        mock_fallback,
    ):
        """Test GPU info fallback when primary method fails."""
        mock_nb.return_value = None
        mock_gpu.side_effect = Exception("GPU info failed")
        mock_fallback.return_value = 2
        mock_tp.return_value = 1
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()

        result = self.builder._build_for_tgi()

        mock_fallback.assert_called_once()
        mock_create.assert_called_once()


class TestBuildForDJL(unittest.TestCase):
    """Test _build_for_djl method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.DJL_SERVING

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_timeout(
        self, mock_create, mock_prepare, mock_detect, mock_validate, mock_dir, mock_nb
    ):
        """Test building with model_data_download_timeout."""
        mock_nb.return_value = None
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model_data_download_timeout = 600

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["MODEL_LOADING_TIMEOUT"], "600")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_djl_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_js,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_djl_config,
        mock_hf_config,
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_djl_config.return_value = ({"OPTION_ENGINE": "Python"}, 512)
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        self.assertEqual(self.builder.env_vars["OPTION_ENGINE"], "Python")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_gpu_info")
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_validate_djl_serving_sample_data")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_tensor_parallel(
        self,
        mock_create,
        mock_prepare,
        mock_detect,
        mock_validate,
        mock_dir,
        mock_nb,
        mock_tp,
        mock_gpu,
    ):
        """Test building for SAGEMAKER_ENDPOINT with tensor parallelism."""
        mock_nb.return_value = None
        mock_gpu.return_value = 4
        mock_tp.return_value = 4
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = Mock()
        self.builder.hf_model_config = {"model_type": "gpt2"}

        result = self.builder._build_for_djl()

        self.assertEqual(self.builder.env_vars["TENSOR_PARALLEL_DEGREE"], "4")
        mock_create.assert_called_once()


class TestBuildForTriton(unittest.TestCase):
    """Test _build_for_triton method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TRITON

    @patch.object(MockModelBuilderServers, "get_huggingface_model_metadata")
    @patch.object(MockModelBuilderServers, "_validate_for_triton")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_save_inference_spec")
    @patch.object(MockModelBuilderServers, "_prepare_for_triton")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_string(
        self,
        mock_create,
        mock_prepare_mode,
        mock_prepare_triton,
        mock_save,
        mock_js,
        mock_validate,
        mock_hf_meta,
    ):
        """Test building with HuggingFace model string."""
        mock_js.return_value = False
        mock_hf_meta.return_value = {"pipeline_tag": "text-generation"}
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_triton()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        self.assertEqual(self.builder.env_vars["HF_TASK"], "text-generation")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._detect_framework_and_version")
    @patch("sagemaker.serve.model_builder_servers._get_model_base")
    @patch.object(MockModelBuilderServers, "_normalize_framework_to_enum")
    @patch.object(MockModelBuilderServers, "_validate_for_triton")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_for_triton")
    @patch.object(MockModelBuilderServers, "_save_inference_spec")
    @patch.object(MockModelBuilderServers, "_prepare_for_triton")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_model_object(
        self,
        mock_create,
        mock_prepare_mode,
        mock_prepare_triton,
        mock_save,
        mock_detect_img,
        mock_validate,
        mock_normalize,
        mock_base,
        mock_detect_fw,
    ):
        """Test building with model object."""
        mock_base.return_value = "pytorch_model"
        mock_detect_fw.return_value = ("pytorch", "1.8.0")
        mock_normalize.return_value = "PYTORCH"
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.image_uri = None

        result = self.builder._build_for_triton()

        self.assertEqual(self.builder.framework_version, "1.8.0")
        mock_detect_img.assert_called_once()
        mock_create.assert_called_once()


class TestBuildForTensorFlowServing(unittest.TestCase):
    """Test _build_for_tensorflow_serving method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TENSORFLOW_SERVING
        self.builder._is_mlflow_model = True

    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    @patch("sagemaker.serve.model_builder_servers.prepare_for_tf_serving")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_mlflow_model(self, mock_create, mock_prepare_mode, mock_tf_prepare, mock_save):
        """Test building MLflow model for TensorFlow Serving."""
        mock_tf_prepare.return_value = ""
        mock_create.return_value = Mock()
        mock_prepare_mode.return_value = ("s3://bucket/model.tar.gz", None)

        result = self.builder._build_for_tensorflow_serving()

        self.assertEqual(self.builder.secret_key, "")
        mock_save.assert_called_once()
        mock_create.assert_called_once()

    def test_build_non_mlflow_model_error(self):
        """Test error when building non-MLflow model."""
        self.builder._is_mlflow_model = False

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tensorflow_serving()
        self.assertIn("mlflow", str(ctx.exception).lower())

    def test_build_missing_image_uri_error(self):
        """Test error when image_uri is missing."""
        self.builder.image_uri = None

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tensorflow_serving()
        self.assertIn("image_uri", str(ctx.exception))


class TestBuildForTEI(unittest.TestCase):
    """Test _build_for_tei method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.TEI

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_dir, mock_nb, mock_hf_config
    ):
        """Test building with HuggingFace model."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "bert"}
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "bert-base-uncased"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_tei()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "bert-base-uncased")
        self.assertEqual(self.builder.env_vars["HF_TOKEN"], "token")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_missing_instance_type(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test error when instance_type is missing for SAGEMAKER_ENDPOINT."""
        mock_nb.return_value = None
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = None
        self.builder.model = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_tei()
        self.assertIn("Instance type", str(ctx.exception))


class TestBuildForSMD(unittest.TestCase):
    """Test _build_for_smd method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.SMD

    @patch("sagemaker.serve.model_builder_servers.prepare_for_smd")
    @patch.object(MockModelBuilderServers, "_save_model_inference_spec")
    @patch.object(MockModelBuilderServers, "_get_processing_unit")
    @patch.object(MockModelBuilderServers, "_get_smd_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_auto_image(
        self,
        mock_create,
        mock_prepare_mode,
        mock_get_img,
        mock_get_unit,
        mock_save,
        mock_smd_prepare,
    ):
        """Test building with auto-detected image."""
        mock_get_unit.return_value = "gpu"
        mock_get_img.return_value = "smd-image-uri"
        mock_smd_prepare.return_value = ""
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.model = Mock()

        result = self.builder._build_for_smd()

        self.assertEqual(self.builder.image_uri, "smd-image-uri")
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()


class TestBuildForTransformers(unittest.TestCase):
    """Test _build_for_transformers method."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model_server = ModelServer.MMS

    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    @patch("sagemaker.serve.model_builder_servers.prepare_for_mms")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_create_conda_env")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_inference_spec_local_container(
        self,
        mock_create,
        mock_prepare_mode,
        mock_conda,
        mock_detect,
        mock_dir,
        mock_nb,
        mock_mms_prepare,
        mock_save,
    ):
        """Test building with inference_spec for LOCAL_CONTAINER."""
        mock_nb.return_value = None
        mock_mms_prepare.return_value = ""
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.inference_spec = Mock()

        result = self.builder._build_for_transformers()

        mock_save.assert_called_once()
        mock_mms_prepare.assert_called_once()
        self.assertEqual(self.builder.secret_key, "")
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_is_jumpstart_model_id")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_with_hf_model_string(
        self, mock_create, mock_prepare, mock_detect, mock_js, mock_dir, mock_nb, mock_hf_config
    ):
        """Test building with HuggingFace model string."""
        mock_js.return_value = False
        mock_nb.return_value = None
        mock_hf_config.return_value = {"model_type": "gpt2"}
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = "gpt2"
        self.builder.env_vars = {"HUGGING_FACE_HUB_TOKEN": "token"}

        result = self.builder._build_for_transformers()

        self.assertEqual(self.builder.env_vars["HF_MODEL_ID"], "gpt2")
        mock_hf_config.assert_called_once_with(
            "gpt2",
            "token",
        )
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_missing_instance_type(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test error when instance_type is missing for SAGEMAKER_ENDPOINT."""
        mock_nb.return_value = None
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = None
        self.builder.model = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_transformers()
        self.assertIn("Instance type", str(ctx.exception))

    @patch("sagemaker.serve.model_builder_servers._get_nb_instance")
    @patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure")
    @patch.object(MockModelBuilderServers, "_auto_detect_image_uri")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_clean_empty_secret_key(
        self, mock_create, mock_prepare, mock_detect, mock_dir, mock_nb
    ):
        """Test cleaning empty secret key from env_vars."""
        mock_nb.return_value = None
        mock_create.return_value = Mock()
        mock_prepare.return_value = ("s3://bucket/model.tar.gz", None)
        self.builder.model = Mock()
        self.builder.env_vars["SAGEMAKER_SERVE_SECRET_KEY"] = ""

        result = self.builder._build_for_transformers()

        self.assertNotIn("SAGEMAKER_SERVE_SECRET_KEY", self.builder.env_vars)
        mock_create.assert_called_once()


class TestBuildForJumpStart(unittest.TestCase):
    """Test _build_for_jumpstart and related methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()
        self.builder.model = "huggingface-llm-falcon-7b"

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_djl_local_container(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test building DJL JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {"TEST": "value"}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.DJL_SERVING)
        self.assertTrue(self.builder.prepared_for_djl)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_tgi_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_tgi_local_container(
        self, mock_create, mock_prepare_mode, mock_tgi_res, mock_init
    ):
        """Test building TGI JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "tgi-inference:1.0.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_tgi_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.TGI)
        self.assertTrue(self.builder.prepared_for_tgi)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_mms_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_mms_local_container(
        self, mock_create, mock_prepare_mode, mock_mms_res, mock_init
    ):
        """Test building MMS JumpStart model for LOCAL_CONTAINER."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "huggingface-pytorch-inference:1.10.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_mms_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        self.assertEqual(self.builder.model_server, ModelServer.MMS)
        self.assertTrue(self.builder.prepared_for_mms)
        mock_create.assert_called_once()

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    def test_build_unsupported_image_uri(self, mock_init):
        """Test error for unsupported JumpStart image URI."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "unsupported-image:1.0.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = None
        mock_init.return_value = mock_init_kwargs
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None

        with self.assertRaises(ValueError) as ctx:
            self.builder._build_for_jumpstart()
        self.assertIn("Local container mode is not yet supported", str(ctx.exception))

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_passes_config_name_to_get_init_kwargs(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test that config_name is forwarded to get_init_kwargs."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {"TEST": "value"}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.config_name = "lmi-optimized"

        self.builder._build_for_jumpstart()

        mock_init.assert_called_once_with(
            model_id=self.builder.model,
            model_version="*",
            region=self.builder.region,
            instance_type=self.builder.instance_type,
            sagemaker_session=self.builder.sagemaker_session,
            tolerate_vulnerable_model=None,
            tolerate_deprecated_model=None,
            config_name="lmi-optimized",
        )

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_passes_none_config_name_when_not_set(
        self, mock_create, mock_prepare_mode, mock_djl_res, mock_init
    ):
        """Test that config_name defaults to None when not set."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.image_uri = None
        self.builder.config_name = None

        self.builder._build_for_jumpstart()

        mock_init.assert_called_once_with(
            model_id=self.builder.model,
            model_version="*",
            region=self.builder.region,
            instance_type=self.builder.instance_type,
            sagemaker_session=self.builder.sagemaker_session,
            tolerate_vulnerable_model=None,
            tolerate_deprecated_model=None,
            config_name=None,
        )

    @patch("sagemaker.core.jumpstart.factory.utils.get_init_kwargs")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_sagemaker_endpoint_djl(self, mock_create, mock_prepare, mock_init):
        """Test building DJL JumpStart for SAGEMAKER_ENDPOINT."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.image_uri = "djl-inference:0.21.0"
        mock_init_kwargs.env = {}
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_init.return_value = mock_init_kwargs
        mock_create.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.image_uri = None

        result = self.builder._build_for_jumpstart()

        mock_create.assert_called_once()


class TestJumpStartAdditionalModelDataSourcesUserFlow(unittest.TestCase):
    """End-to-end tests for additional model data sources, asserting the
    CreateModel request shape.

    Each test uses the documented single-call customer API:

        builder = ModelBuilder.from_jumpstart_config(
            jumpstart_config=JumpStartConfig(
                model_id="...", accept_eula=True, inference_config_name="..."),
            compute=Compute(instance_type="..."), ...)
        builder.build()

    and asserts on what the SDK sends to the CreateModel API -- the
    ``container_defs`` captured from ``sagemaker_session.create_model`` -- not
    on builder internals. This is the request boto receives, so the assertions
    cover the full downstream impact: primary container image, model data
    source, and AdditionalModelDataSources.

    The spec fixtures under tests/unit/servers/data/jumpstart_specs/ are real,
    unmodified specs captured from the production JumpStart content bucket
    (jumpstart-cache-prod-us-west-2):

    - pytorch-ic-mobilenet-v2: public model, no additional data sources.
    - openai-reasoning-gpt-oss-20b: public model whose default (lmi-optimized)
      config carries an ungated EAGLE speculative-decoding source -- the model
      that surfaced the propagation bug this suite guards.
    - meta-textgeneration-llama-3-1-70b: gated model whose lmi-optimized
      config carries a GATED draft_model source
      (hosting_eula_key=fmhMetadata/eula/llama3_2Eula.txt).

    Patching is centralized in setUp on the same seams the legacy master-v2
    JumpStartModel tests patched:

    1. The spec-fetch boundary: JumpStartModelsAccessor.get_model_specs routes
       by model id into the captured spec files (master-v2's
       PROTOTYPICAL_MODEL_SPECS_DICT pattern); _get_manifest serves the
       captured manifest headers.
    2. The AWS boundary: a mock session (whose create_model call is the
       assertion target), IAM role validation, artifact staging
       (_prepare_for_mode), and the post-create Model.get describe call.

    Everything in between runs for real: JumpStart model-id detection,
    JumpStartModelSpecs parsing, inference-config resolution, the
    get_init_kwargs factory (image URI resolution, content-bucket injection,
    PascalCase shaping), the propagation in _build_for_jumpstart, the
    accept_eula application in container_def (shared with the primary model),
    and _prepare_container_def_base assembling the CreateModel request.
    """

    SPEC_DIR = Path(__file__).parent / "data" / "jumpstart_specs"

    ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"

    # The content buckets the sources must resolve into (the raw specs publish
    # bucket-less key prefixes): public sources land in the public content
    # bucket, gated sources in the private one.
    PUBLIC_CONTENT_BUCKET = "jumpstart-cache-prod-us-west-2"
    PRIVATE_CONTENT_BUCKET = "jumpstart-private-cache-prod-us-west-2"

    @classmethod
    def _spec_additional_source(cls, model_id, config_name="lmi-optimized"):
        """The raw additional data source exactly as published in the captured
        spec fixture: snake_case keys, bucket-less s3_uri key prefix, and
        hosting_eula_key present when the source is gated."""
        with open(cls.SPEC_DIR / f"{model_id}.json") as f:
            spec = json.load(f)
        sources = spec["inference_config_components"][config_name][
            "hosting_additional_data_sources"
        ]["speculative_decoding"]
        assert len(sources) == 1, f"expected one source in {model_id}/{config_name}"
        return sources[0]

    @classmethod
    def _expected_create_model_source(cls, model_id, bucket, accept_eula=None):
        """The shape the CreateModel request must carry for the fixture's
        source. Data values (channel name, compression, data type, key prefix)
        come from the spec fixture so it stays the single source of truth; the
        transformation under test is spelled out declaratively here instead of
        reusing production code (which would make the assertion a tautology):

        - snake_case spec keys become the PascalCase API keys
        - the bucket-less key prefix resolves into the given content bucket
        - HostingEulaKey never appears (the API rejects it)
        - accept_eula, when set, is folded into S3DataSource.ModelAccessConfig
          exactly like the primary model's ModelDataSource
        """
        raw = cls._spec_additional_source(model_id)
        expected = {
            "ChannelName": raw["channel_name"],
            "S3DataSource": {
                "CompressionType": raw["s3_data_source"]["compression_type"],
                "S3DataType": raw["s3_data_source"]["s3_data_type"],
                "S3Uri": f"s3://{bucket}/{raw['s3_data_source']['s3_uri']}",
            },
        }
        if accept_eula is not None:
            expected["S3DataSource"]["ModelAccessConfig"] = {"AcceptEula": accept_eula}
        return expected

    def setUp(self):
        import sagemaker.serve.model_builder as model_builder_module
        from sagemaker.core.jumpstart.accessors import JumpStartModelsAccessor
        from sagemaker.core.jumpstart.types import JumpStartModelHeader, JumpStartModelSpecs
        from sagemaker.serve.model_builder import ModelBuilder

        with open(self.SPEC_DIR / "manifest.json") as f:
            manifest = [JumpStartModelHeader(header) for header in json.load(f)]

        spec_dir = self.SPEC_DIR

        def get_captured_model_specs(*args, **kwargs):
            """Routes spec lookups by model id into the captured real spec
            files, like master-v2's PROTOTYPICAL_MODEL_SPECS_DICT loaders."""
            model_id = kwargs.get("model_id") or args[1]
            with open(spec_dir / f"{model_id}.json") as f:
                return JumpStartModelSpecs(json.load(f))

        patchers = [
            patch.object(
                JumpStartModelsAccessor, "get_model_specs", side_effect=get_captured_model_specs
            ),
            patch.object(JumpStartModelsAccessor, "_get_manifest", return_value=manifest),
            patch.object(ModelBuilder, "_prepare_for_mode", return_value=None),
            patch.object(
                model_builder_module, "resolve_and_validate_role", return_value=self.ROLE_ARN
            ),
            patch.object(model_builder_module.Model, "get", return_value=Mock()),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        session = Mock()
        session.boto_region_name = "us-west-2"
        session.get_caller_identity_arn = Mock(return_value=self.ROLE_ARN)
        session.sagemaker_config = {}
        session.config = None
        session.settings.include_jumpstart_tags = False
        session.default_bucket = Mock(return_value="sagemaker-us-west-2-123456789012")
        session.default_bucket_prefix = None
        session.local_mode = False
        self.session = session

    def _build(self, model_id, accept_eula=None, inference_config_name=None, instance_type=None):
        """The customer flow, verbatim: one from_jumpstart_config call carrying
        the model id, EULA decision, and config selection, then build()."""
        from sagemaker.core.jumpstart.configs import JumpStartConfig
        from sagemaker.core.training.configs import Compute
        from sagemaker.serve.model_builder import ModelBuilder

        builder = ModelBuilder.from_jumpstart_config(
            jumpstart_config=JumpStartConfig(
                model_id=model_id,
                accept_eula=accept_eula,
                inference_config_name=inference_config_name,
            ),
            role_arn=self.ROLE_ARN,
            compute=Compute(instance_type=instance_type) if instance_type else None,
            sagemaker_session=self.session,
        )
        builder.build()

    def _create_model_container_def(self):
        """The container definition sent to the CreateModel API."""
        self.session.create_model.assert_called_once()
        return self.session.create_model.call_args.kwargs["container_defs"]

    def test_model_without_additional_sources_sends_none_to_create_model(self):
        """Base case: the CreateModel request carries no
        AdditionalModelDataSources field at all."""
        self._build("pytorch-ic-mobilenet-v2")

        container_def = self._create_model_container_def()
        self.assertNotIn("AdditionalModelDataSources", container_def)

    def test_ungated_additional_source_reaches_create_model_without_eula(self):
        """The bug this suite guards: an ungated speculative-decoding source
        in the model's default config must reach CreateModel even though the
        customer never touches accept_eula."""
        self._build("openai-reasoning-gpt-oss-20b", instance_type="ml.g7e.2xlarge")

        container_def = self._create_model_container_def()
        self.assertEqual(
            container_def["AdditionalModelDataSources"],
            [
                self._expected_create_model_source(
                    "openai-reasoning-gpt-oss-20b", bucket=self.PUBLIC_CONTENT_BUCKET
                )
            ],
        )

    def test_accept_eula_applies_to_every_model_data_source_uniformly(self):
        """The single accept_eula knob folds the same ModelAccessConfig into
        the primary model AND each additional source, exactly like the primary
        model's ModelDataSource handling. Gatedness is not the SDK's call: the
        service determines it from the artifact's bucket and ignores the config
        on ungated sources."""
        self._build(
            "openai-reasoning-gpt-oss-20b", accept_eula=True, instance_type="ml.g7e.2xlarge"
        )

        container_def = self._create_model_container_def()
        self.assertEqual(
            container_def["AdditionalModelDataSources"],
            [
                self._expected_create_model_source(
                    "openai-reasoning-gpt-oss-20b",
                    bucket=self.PUBLIC_CONTENT_BUCKET,
                    accept_eula=True,
                )
            ],
        )
        self.assertEqual(
            container_def["ModelDataSource"]["S3DataSource"]["ModelAccessConfig"],
            {"AcceptEula": True},
        )

    def test_gated_additional_source_with_accepted_eula_sends_model_access_config(self):
        """Accepting the EULA folds ModelAccessConfig into the gated
        additional source AND the primary model in the CreateModel request."""
        self._build(
            "meta-textgeneration-llama-3-1-70b",
            accept_eula=True,
            inference_config_name="lmi-optimized",
            instance_type="ml.p4d.24xlarge",
        )

        container_def = self._create_model_container_def()
        self.assertEqual(
            container_def["AdditionalModelDataSources"],
            [
                self._expected_create_model_source(
                    "meta-textgeneration-llama-3-1-70b",
                    bucket=self.PRIVATE_CONTENT_BUCKET,
                    accept_eula=True,
                )
            ],
        )
        self.assertEqual(
            container_def["ModelDataSource"]["S3DataSource"]["ModelAccessConfig"],
            {"AcceptEula": True},
        )

    def test_gated_additional_source_without_eula_sends_no_model_access_config(self):
        """No EULA decision: the request carries the gated source without any
        ModelAccessConfig. Enforcement is the service's: the SageMaker control
        plane resolves the private-bucket URI as gated and rejects CreateModel
        with an EULA validation error, the same as for a gated primary model."""
        self._build(
            "meta-textgeneration-llama-3-1-70b",
            inference_config_name="lmi-optimized",
            instance_type="ml.p4d.24xlarge",
        )

        container_def = self._create_model_container_def()
        self.assertEqual(
            container_def["AdditionalModelDataSources"],
            [
                self._expected_create_model_source(
                    "meta-textgeneration-llama-3-1-70b", bucket=self.PRIVATE_CONTENT_BUCKET
                )
            ],
        )

    def test_gated_additional_source_with_rejected_eula_sends_acceptance_false(self):
        """Explicit accept_eula=False is transmitted faithfully as
        ModelAccessConfig={"AcceptEula": False} on every source, exactly like
        the primary model. The service rejects the gated sources."""
        self._build(
            "meta-textgeneration-llama-3-1-70b",
            accept_eula=False,
            inference_config_name="lmi-optimized",
            instance_type="ml.p4d.24xlarge",
        )

        container_def = self._create_model_container_def()
        self.assertEqual(
            container_def["AdditionalModelDataSources"],
            [
                self._expected_create_model_source(
                    "meta-textgeneration-llama-3-1-70b",
                    bucket=self.PRIVATE_CONTENT_BUCKET,
                    accept_eula=False,
                )
            ],
        )
        self.assertEqual(
            container_def["ModelDataSource"]["S3DataSource"]["ModelAccessConfig"],
            {"AcceptEula": False},
        )

    def test_unselected_config_sources_do_not_leak_into_create_model(self):
        """Config resolution gates which sources apply: the same gated model on
        its default (lmi) config has no additional sources, so the CreateModel
        request must not carry the lmi-optimized config's gated draft model."""
        self._build(
            "meta-textgeneration-llama-3-1-70b",
            accept_eula=True,
            instance_type="ml.p4d.24xlarge",
        )

        container_def = self._create_model_container_def()
        self.assertNotIn("AdditionalModelDataSources", container_def)


class TestDeployWrappers(unittest.TestCase):
    """Test deploy wrapper methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_djl_deploy_in_process(self, mock_deploy):
        """Test DJL deploy wrapper for IN_PROCESS mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS

        result = self.builder._djl_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_djl_deploy_local_container(self, mock_deploy):
        """Test DJL deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._djl_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_djl_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test DJL deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper(model_data_download_timeout=600)

        self.assertEqual(self.builder.env_vars["MODEL_LOADING_TIMEOUT"], "600")
        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_djl_deploy_with_defaults(self, mock_deploy):
        """Test DJL deploy wrapper sets default values."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper()

        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs["endpoint_logging"], True)
        self.assertEqual(call_kwargs["initial_instance_count"], 1)

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_tgi_deploy_local_container(self, mock_deploy):
        """Test TGI deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._tgi_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_tgi_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test TGI deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._tgi_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_tei_deploy_in_process(self, mock_deploy):
        """Test TEI deploy wrapper for IN_PROCESS mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.IN_PROCESS

        result = self.builder._tei_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_tei_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test TEI deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._tei_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_js_deploy_local_container(self, mock_deploy):
        """Test JumpStart deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._js_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_js_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test JumpStart deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.instance_type = "ml.g5.xlarge"

        result = self.builder._js_builder_deploy_wrapper()

        call_kwargs = mock_deploy.call_args[1]
        self.assertEqual(call_kwargs["instance_type"], "ml.g5.xlarge")
        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_local_endpoint")
    def test_transformers_deploy_local_container(self, mock_deploy):
        """Test Transformers deploy wrapper for LOCAL_CONTAINER mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER

        result = self.builder._transformers_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_transformers_deploy_sagemaker_endpoint(self, mock_deploy):
        """Test Transformers deploy wrapper for SAGEMAKER_ENDPOINT mode."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._transformers_model_builder_deploy_wrapper()

        mock_deploy.assert_called_once()

    @patch.object(MockModelBuilderServers, "_deploy_core_endpoint")
    def test_deploy_wrapper_removes_mode_and_role(self, mock_deploy):
        """Test deploy wrapper removes mode and role from kwargs."""
        mock_deploy.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT

        result = self.builder._djl_model_builder_deploy_wrapper(
            mode=Mode.LOCAL_CONTAINER, role="arn:aws:iam::123456789012:role/test"
        )

        call_kwargs = mock_deploy.call_args[1]
        self.assertNotIn("mode", call_kwargs)
        self.assertNotIn("role", call_kwargs)
        self.assertEqual(self.builder.role_arn, "arn:aws:iam::123456789012:role/test")


class TestJumpStartBuilders(unittest.TestCase):
    """Test JumpStart-specific builder methods."""

    def setUp(self):
        self.builder = MockModelBuilderServers()

    @patch("sagemaker.serve.model_builder_servers.prepare_djl_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_djl_jumpstart_local(self, mock_create, mock_prepare, mock_djl_res):
        """Test _build_for_djl_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_djl_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_djl_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.DJL_SERVING)
        self.assertTrue(self.builder.prepared_for_djl)
        mock_djl_res.assert_called_once()
        mock_create.assert_called_once()

    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_djl_jumpstart_sagemaker(self, mock_create):
        """Test _build_for_djl_jumpstart for SAGEMAKER_ENDPOINT mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_create.return_value = Mock()
        self.builder.mode = Mode.SAGEMAKER_ENDPOINT
        self.builder.model = "jumpstart-model-id"

        result = self.builder._build_for_djl_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.s3_upload_path, "s3://bucket/model.tar.gz")
        self.assertTrue(self.builder.prepared_for_djl)
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_tgi_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_tgi_jumpstart_local(self, mock_create, mock_prepare, mock_tgi_res):
        """Test _build_for_tgi_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_tgi_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_tgi_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.TGI)
        self.assertTrue(self.builder.prepared_for_tgi)
        mock_tgi_res.assert_called_once()
        mock_create.assert_called_once()

    @patch("sagemaker.serve.model_builder_servers.prepare_mms_js_resources")
    @patch.object(MockModelBuilderServers, "_prepare_for_mode")
    @patch.object(MockModelBuilderServers, "_create_model")
    def test_build_for_mms_jumpstart_local(self, mock_create, mock_prepare, mock_mms_res):
        """Test _build_for_mms_jumpstart for local mode."""
        mock_init_kwargs = Mock()
        mock_init_kwargs.model_data = "s3://bucket/model.tar.gz"
        mock_mms_res.return_value = ({"config": "value"}, True)
        mock_create.return_value = Mock()
        self.builder.mode = Mode.LOCAL_CONTAINER
        self.builder.model = "jumpstart-model-id"
        self.builder.s3_model_data_url = "s3://bucket/model.tar.gz"

        result = self.builder._build_for_mms_jumpstart(mock_init_kwargs)

        self.assertEqual(self.builder.model_server, ModelServer.MMS)
        self.assertTrue(self.builder.prepared_for_mms)
        mock_mms_res.assert_called_once()
        mock_create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
