"""Tests for DJL builder HF cache environment variables and HF_MODEL_ID handling.

Verifies that _build_for_djl() correctly:
- Sets HF_HOME and HUGGINGFACE_HUB_CACHE to /tmp for writable cache
- Preserves user-provided HF_MODEL_ID values (uses setdefault)
- Sets HF_MODEL_ID from model param when not provided by user
- Preserves user-provided HF_HOME and HUGGINGFACE_HUB_CACHE values
"""

import pytest
from unittest.mock import Mock, patch

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.core.resources import Model


MOCK_ROLE_ARN = "arn:aws:iam::000000000000:role/SageMakerRole"
MOCK_IMAGE_URI = "000000000000.dkr.ecr.us-east-1.amazonaws.com/djl-inference:latest"
MOCK_HF_MODEL_CONFIG = {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]}


# Common patches needed for _build_for_djl
_DJL_PATCHES = [
    "sagemaker.serve.model_builder_servers._get_nb_instance",
    "sagemaker.serve.model_builder_servers._get_default_djl_configurations",
    "sagemaker.serve.model_builder_servers._get_model_config_properties_from_hf",
    "sagemaker.serve.model_builder.ModelBuilder._is_jumpstart_model_id",
    "sagemaker.serve.model_builder.ModelBuilder._validate_djl_serving_sample_data",
    "sagemaker.serve.model_builder.ModelBuilder._auto_detect_image_uri",
    "sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode",
    "sagemaker.serve.model_builder.ModelBuilder._create_model",
    "sagemaker.serve.model_builder_servers._get_default_tensor_parallel_degree",
    "sagemaker.serve.model_builder_servers._get_gpu_info",
]


def _mock_sagemaker_session():
    """Create a mock SageMaker session."""
    session = Mock()
    session.boto_region_name = "us-east-1"
    session.sagemaker_config = {}
    session.default_bucket.return_value = "mock-bucket"
    session.upload_data.return_value = "s3://mock-bucket/model.tar.gz"
    return session


def _create_djl_builder(tmp_path, env_vars=None, mode=Mode.SAGEMAKER_ENDPOINT):
    """Create a ModelBuilder configured for DJL serving tests."""
    builder = ModelBuilder(
        model="test-org/test-model",
        role_arn=MOCK_ROLE_ARN,
        sagemaker_session=_mock_sagemaker_session(),
        model_path=str(tmp_path),
        mode=mode,
        image_uri=MOCK_IMAGE_URI,
        model_server=ModelServer.DJL_SERVING,
        instance_type="ml.g6e.12xlarge",
        env_vars=env_vars or {},
    )
    builder.schema_builder = Mock()
    builder.schema_builder.sample_input = {"inputs": "Hello"}
    builder._optimizing = False
    builder.hf_model_config = MOCK_HF_MODEL_CONFIG
    return builder


def _setup_mocks(mocks):
    """Configure common mock return values for DJL build."""
    # mocks are in reverse order of _DJL_PATCHES
    mock_gpu_info = mocks[-1]
    mock_tp_degree = mocks[-2]
    mock_create = mocks[-3]
    mock_prepare = mocks[-4]
    # mock_auto_detect = mocks[-5]  # no setup needed
    # mock_validate = mocks[-6]  # no setup needed
    mock_is_js = mocks[-7]
    mock_hf_config = mocks[-8]
    mock_djl_config = mocks[-9]
    mock_nb = mocks[-10]

    mock_nb.return_value = None
    mock_djl_config.return_value = ({}, 256)
    mock_hf_config.return_value = MOCK_HF_MODEL_CONFIG
    mock_is_js.return_value = False
    mock_prepare.return_value = ("s3://bucket/model", None)
    mock_create.return_value = Mock(spec=Model)
    mock_tp_degree.return_value = 4
    mock_gpu_info.return_value = 4


class TestDjlHfCacheAndModelId:
    """Tests for DJL builder HF cache env vars and HF_MODEL_ID handling."""

    @pytest.fixture(autouse=True)
    def _patch_djl(self):
        """Apply all DJL-related patches for each test."""
        patchers = [patch(p) for p in _DJL_PATCHES]
        self._mocks = [p.start() for p in patchers]
        _setup_mocks(self._mocks)
        yield
        for p in patchers:
            p.stop()

    def test_sets_hf_cache_env_vars_to_tmp(self, tmp_path):
        """HF_HOME and HUGGINGFACE_HUB_CACHE should be /tmp in endpoint mode."""
        builder = _create_djl_builder(tmp_path)
        builder._build_for_djl()

        assert builder.env_vars["HF_HOME"] == "/tmp"
        assert builder.env_vars["HUGGINGFACE_HUB_CACHE"] == "/tmp"

    def test_preserves_user_provided_hf_model_id(self, tmp_path):
        """User-provided HF_MODEL_ID must NOT be overridden by model param."""
        builder = _create_djl_builder(
            tmp_path, env_vars={"HF_MODEL_ID": "/opt/ml/model"}
        )
        builder._build_for_djl()

        assert builder.env_vars["HF_MODEL_ID"] == "/opt/ml/model"

    def test_sets_hf_model_id_from_model_param_when_not_provided(self, tmp_path):
        """When no user-provided HF_MODEL_ID, it should come from model param."""
        builder = _create_djl_builder(tmp_path)
        builder._build_for_djl()

        assert builder.env_vars["HF_MODEL_ID"] == "test-org/test-model"

    def test_preserves_user_provided_hf_cache_dirs(self, tmp_path):
        """User-provided HF_HOME and HUGGINGFACE_HUB_CACHE should be preserved."""
        builder = _create_djl_builder(
            tmp_path,
            env_vars={
                "HF_HOME": "/my/custom/cache",
                "HUGGINGFACE_HUB_CACHE": "/my/custom/hub",
            },
        )
        builder._build_for_djl()

        assert builder.env_vars["HF_HOME"] == "/my/custom/cache"
        assert builder.env_vars["HUGGINGFACE_HUB_CACHE"] == "/my/custom/hub"

    def test_local_mode_sets_hf_hub_offline(self, tmp_path):
        """HF_HUB_OFFLINE=1 should be set in LOCAL_CONTAINER mode."""
        builder = _create_djl_builder(tmp_path, mode=Mode.LOCAL_CONTAINER)
        # Local mode doesn't need GPU info mocks for instance_type validation
        builder.instance_type = None
        builder._build_for_djl()

        assert builder.env_vars["HF_HUB_OFFLINE"] == "1"
