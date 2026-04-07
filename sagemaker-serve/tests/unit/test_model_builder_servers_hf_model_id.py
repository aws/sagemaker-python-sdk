"""Unit tests to verify HF_MODEL_ID is not overwritten when user provides it."""
import unittest
from unittest.mock import Mock, patch, MagicMock, PropertyMock

from sagemaker.serve.model_builder_servers import _ModelBuilderServers
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


def _create_mock_builder(env_vars=None, model="Qwen/Qwen3-VL-4B-Instruct"):
    """Create a mock builder with common attributes set."""
    builder = MagicMock(spec=_ModelBuilderServers)
    builder.model = model
    builder.env_vars = env_vars if env_vars is not None else {}
    builder.model_path = "/tmp/test_model_path"
    builder.mode = Mode.SAGEMAKER_ENDPOINT
    builder.model_server = ModelServer.DJL_SERVING
    builder.secret_key = ""
    builder.s3_upload_path = None
    builder.s3_model_data_url = None
    builder.shared_libs = []
    builder.dependencies = {}
    builder.image_uri = "test-image-uri"
    builder.instance_type = "ml.g5.2xlarge"
    builder.sagemaker_session = Mock()
    builder.schema_builder = MagicMock()
    builder.schema_builder.sample_input = {"inputs": "Hello", "parameters": {}}
    builder.inference_spec = None
    builder.hf_model_config = {}
    builder.model_data_download_timeout = None
    builder._user_provided_instance_type = True
    builder._is_jumpstart_model_id = Mock(return_value=False)
    builder._auto_detect_image_uri = Mock()
    builder._prepare_for_mode = Mock(return_value=("s3://model-data", None))
    builder._create_model = Mock(return_value=Mock())
    builder._optimizing = False
    builder._validate_djl_serving_sample_data = Mock()
    builder._validate_tgi_serving_sample_data = Mock()
    builder._validate_for_triton = Mock()
    builder.get_huggingface_model_metadata = Mock(return_value={"pipeline_tag": "text-generation"})
    builder.role_arn = "arn:aws:iam::123456789012:role/SageMakerRole"
    return builder


class TestDjlPreservesHfModelId(unittest.TestCase):
    """Test that _build_for_djl preserves user-provided HF_MODEL_ID."""

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_djl_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info", return_value=1)
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree", return_value=1)
    def test_preserves_user_provided_s3_uri(self, mock_tp, mock_gpu, mock_nb, mock_djl_config, mock_hf_config):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten."""
        mock_hf_config.return_value = {}
        mock_djl_config.return_value = ({}, 256)

        s3_path = "s3://my-bucket/models/Qwen/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})

        with patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_djl(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_djl_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info", return_value=1)
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree", return_value=1)
    def test_sets_hf_model_id_when_not_provided(self, mock_tp, mock_gpu, mock_nb, mock_djl_config, mock_hf_config):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        mock_hf_config.return_value = {}
        mock_djl_config.return_value = ({}, 256)

        builder = _create_mock_builder(env_vars={})

        with patch("sagemaker.serve.model_server.djl_serving.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_djl(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")


class TestTgiPreservesHfModelId(unittest.TestCase):
    """Test that _build_for_tgi preserves user-provided HF_MODEL_ID."""

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info", return_value=1)
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree", return_value=1)
    def test_preserves_user_provided_s3_uri(self, mock_tp, mock_gpu, mock_nb, mock_tgi_config, mock_hf_config):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten."""
        mock_hf_config.return_value = {}
        mock_tgi_config.return_value = ({}, 256)

        s3_path = "s3://my-bucket/models/Qwen/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.TGI

        with patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_tgi(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_default_tgi_configurations")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    @patch("sagemaker.serve.model_builder_servers._get_gpu_info", return_value=1)
    @patch("sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree", return_value=1)
    def test_sets_hf_model_id_when_not_provided(self, mock_tp, mock_gpu, mock_nb, mock_tgi_config, mock_hf_config):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        mock_hf_config.return_value = {}
        mock_tgi_config.return_value = ({}, 256)

        builder = _create_mock_builder(env_vars={})
        builder.model_server = ModelServer.TGI

        with patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_tgi(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")


class TestTeiPreservesHfModelId(unittest.TestCase):
    """Test that _build_for_tei preserves user-provided HF_MODEL_ID."""

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    def test_preserves_user_provided_s3_uri(self, mock_nb, mock_hf_config):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten."""
        mock_hf_config.return_value = {}

        s3_path = "s3://my-bucket/models/embedding-model/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.TEI

        with patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_tei(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    def test_sets_hf_model_id_when_not_provided(self, mock_nb, mock_hf_config):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        mock_hf_config.return_value = {}

        builder = _create_mock_builder(env_vars={})
        builder.model_server = ModelServer.TEI

        with patch("sagemaker.serve.model_server.tgi.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_tei(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")


class TestTorchservePreservesHfModelId(unittest.TestCase):
    """Test that _build_for_torchserve preserves user-provided HF_MODEL_ID."""

    def test_preserves_user_provided_s3_uri(self):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten."""
        s3_path = "s3://my-bucket/models/my-model/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.TORCHSERVE
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder._save_model_inference_spec = Mock()

        _ModelBuilderServers._build_for_torchserve(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    def test_sets_hf_model_id_when_not_provided(self):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        builder = _create_mock_builder(env_vars={})
        builder.model_server = ModelServer.TORCHSERVE
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder._save_model_inference_spec = Mock()

        _ModelBuilderServers._build_for_torchserve(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")


class TestTritonPreservesHfModelId(unittest.TestCase):
    """Test that _build_for_triton preserves user-provided HF_MODEL_ID."""

    def test_preserves_user_provided_s3_uri(self):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten."""
        s3_path = "s3://my-bucket/models/my-model/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.TRITON
        builder._save_inference_spec = Mock()
        builder._prepare_for_triton = Mock()
        builder._auto_detect_image_for_triton = Mock()

        _ModelBuilderServers._build_for_triton(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    def test_sets_hf_model_id_when_not_provided(self):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        builder = _create_mock_builder(env_vars={})
        builder.model_server = ModelServer.TRITON
        builder._save_inference_spec = Mock()
        builder._prepare_for_triton = Mock()
        builder._auto_detect_image_for_triton = Mock()

        _ModelBuilderServers._build_for_triton(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")


class TestTransformersPreservesHfModelId(unittest.TestCase):
    """Test that _build_for_transformers preserves user-provided HF_MODEL_ID."""

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    def test_preserves_user_provided_s3_uri_with_model_string(self, mock_nb, mock_hf_config):
        """User-provided S3 URI for HF_MODEL_ID should not be overwritten when model is a string."""
        mock_hf_config.return_value = {}

        s3_path = "s3://my-bucket/models/my-model/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.MMS
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder.model_data_download_timeout = None

        with patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_transformers(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    def test_sets_hf_model_id_when_not_provided_with_model_string(self, mock_nb, mock_hf_config):
        """HF_MODEL_ID should be set from self.model when user doesn't provide it."""
        mock_hf_config.return_value = {}

        builder = _create_mock_builder(env_vars={})
        builder.model_server = ModelServer.MMS
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder.model_data_download_timeout = None

        with patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure"):
            _ModelBuilderServers._build_for_transformers(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "Qwen/Qwen3-VL-4B-Instruct")

    @patch("sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf")
    @patch("sagemaker.serve.model_builder_servers._get_nb_instance", return_value=None)
    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    def test_preserves_user_provided_hf_model_id_with_inference_spec(self, mock_pkl, mock_nb, mock_hf_config):
        """User-provided HF_MODEL_ID should not be overwritten when inference_spec provides a model ID."""
        mock_hf_config.return_value = {}

        s3_path = "s3://my-bucket/models/my-model/"
        builder = _create_mock_builder(env_vars={"HF_MODEL_ID": s3_path})
        builder.model_server = ModelServer.MMS
        builder.mode = Mode.SAGEMAKER_ENDPOINT
        builder.model_data_download_timeout = None
        builder.model = None  # No model string, using inference_spec
        builder.inference_spec = Mock()
        builder.inference_spec.get_model.return_value = "some-hf-model-id"
        builder._is_jumpstart_model_id = Mock(return_value=False)

        with patch("sagemaker.serve.model_server.multi_model_server.prepare._create_dir_structure"):
            with patch("os.makedirs"):
                _ModelBuilderServers._build_for_transformers(builder)

        self.assertEqual(builder.env_vars["HF_MODEL_ID"], s3_path)


if __name__ == "__main__":
    unittest.main()
