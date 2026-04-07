"""Tests for DJL builder HF cache environment variables and HF_MODEL_ID handling.

Verifies that _build_for_djl() correctly:
- Sets HF_HOME and HUGGINGFACE_HUB_CACHE to /tmp for writable cache
- Preserves user-provided HF_MODEL_ID values (uses setdefault)
- Sets HF_MODEL_ID when not provided by user
- Sets HF_HUB_OFFLINE in local modes
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import shutil

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model


def _mock_sagemaker_session():
    """Create a mock SageMaker session."""
    session = Mock()
    session.boto_region_name = "us-east-1"
    session.sagemaker_config = {}
    session.default_bucket.return_value = "mock-bucket"
    session.upload_data.return_value = "s3://mock-bucket/model.tar.gz"
    return session


MOCK_ROLE_ARN = "arn:aws:iam::123456789012:role/SageMakerRole"
MOCK_IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.36.0-lmi22.0.0-cu129"
MOCK_HF_MODEL_CONFIG = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}


class TestDjlHfCacheEnv(unittest.TestCase):
    """Test DJL builder HF cache environment variable handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_session = _mock_sagemaker_session()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up temp directory."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    @patch('sagemaker.serve.model_builder_servers._get_gpu_info')
    @patch('sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_sets_hf_home_to_tmp(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create,
        mock_tp_degree, mock_gpu_info
    ):
        """Verify HF_HOME=/tmp is set in SAGEMAKER_ENDPOINT mode."""
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)
        mock_prepare.return_value = ("s3://bucket/model", None)
        mock_gpu_info.return_value = 4
        mock_tp_degree.return_value = 4

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
            instance_type="ml.g6e.12xlarge",
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        self.assertEqual(builder.env_vars.get("HF_HOME"), "/tmp")

    @patch('sagemaker.serve.model_builder_servers._get_gpu_info')
    @patch('sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_sets_huggingface_hub_cache_to_tmp(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create,
        mock_tp_degree, mock_gpu_info
    ):
        """Verify HUGGINGFACE_HUB_CACHE=/tmp is set in SAGEMAKER_ENDPOINT mode."""
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)
        mock_prepare.return_value = ("s3://bucket/model", None)
        mock_gpu_info.return_value = 4
        mock_tp_degree.return_value = 4

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
            instance_type="ml.g6e.12xlarge",
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        self.assertEqual(builder.env_vars.get("HUGGINGFACE_HUB_CACHE"), "/tmp")

    @patch('sagemaker.serve.model_builder_servers._get_gpu_info')
    @patch('sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_preserves_user_provided_hf_model_id(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create,
        mock_tp_degree, mock_gpu_info
    ):
        """Verify user-provided HF_MODEL_ID is NOT overridden."""
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)
        mock_prepare.return_value = ("s3://bucket/model", None)
        mock_gpu_info.return_value = 4
        mock_tp_degree.return_value = 4

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
            instance_type="ml.g6e.12xlarge",
            env_vars={"HF_MODEL_ID": "/opt/ml/model"},
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        # User-provided value should be preserved, NOT overridden by model param
        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "/opt/ml/model")

    @patch('sagemaker.serve.model_builder_servers._get_gpu_info')
    @patch('sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_sets_hf_model_id_when_not_provided(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create,
        mock_tp_degree, mock_gpu_info
    ):
        """Verify HF_MODEL_ID is set from model param when not user-provided."""
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)
        mock_prepare.return_value = ("s3://bucket/model", None)
        mock_gpu_info.return_value = 4
        mock_tp_degree.return_value = 4

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
            instance_type="ml.g6e.12xlarge",
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        # When no user-provided HF_MODEL_ID, it should be set from model param
        self.assertEqual(builder.env_vars["HF_MODEL_ID"], "chromadb/context-1")

    @patch('sagemaker.serve.model_builder_servers._get_gpu_info')
    @patch('sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree')
    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_with_source_code_and_hf_model_id(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create,
        mock_tp_degree, mock_gpu_info
    ):
        """Verify HF cache env vars are set to /tmp when source_code is provided.
        
        This is the key scenario from the bug: source_code makes /opt/ml/model
        read-only, so HF cache must be redirected to /tmp.
        """
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)
        mock_prepare.return_value = ("s3://bucket/model", None)
        mock_gpu_info.return_value = 4
        mock_tp_degree.return_value = 4

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.SAGEMAKER_ENDPOINT,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
            instance_type="ml.g6e.12xlarge",
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        # HF cache should be redirected to /tmp to avoid read-only /opt/ml/model
        self.assertEqual(builder.env_vars.get("HF_HOME"), "/tmp")
        self.assertEqual(builder.env_vars.get("HUGGINGFACE_HUB_CACHE"), "/tmp")

    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri')
    @patch('sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data')
    @patch('sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id')
    @patch('sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf')
    @patch('sagemaker.serve.model_builder_servers._get_default_djl_configurations')
    @patch('sagemaker.serve.model_builder_servers._get_nb_instance')
    def test_build_for_djl_local_mode_sets_hf_hub_offline(
        self, mock_nb, mock_djl_config, mock_hf_config, mock_is_js,
        mock_validate, mock_auto_detect, mock_prepare, mock_create
    ):
        """Verify HF_HUB_OFFLINE=1 is set in LOCAL_CONTAINER mode."""
        mock_nb.return_value = None
        mock_is_js.return_value = False
        mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
        mock_djl_config.return_value = ({}, 256)
        mock_create.return_value = Mock(spec=Model)

        builder = ModelBuilder(
            model="chromadb/context-1",
            role_arn=MOCK_ROLE_ARN,
            sagemaker_session=self.mock_session,
            model_path=self.temp_dir,
            mode=Mode.LOCAL_CONTAINER,
            image_uri=MOCK_IMAGE_URI,
            model_server=ModelServer.DJL_SERVING,
        )
        builder.schema_builder = Mock()
        builder.schema_builder.sample_input = {"inputs": "Hello"}
        builder._optimizing = False
        builder.hf_model_config = MOCK_HF_MODEL_CONFIG

        builder._build_for_djl()

        self.assertEqual(builder.env_vars.get("HF_HUB_OFFLINE"), "1")


if __name__ == "__main__":
    unittest.main()
