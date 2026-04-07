"""Unit tests to verify HF_MODEL_ID is not overwritten when user provides it."""
from __future__ import annotations

from typing import Optional
from unittest.mock import Mock, patch, MagicMock

import pytest

from sagemaker.serve.model_builder_servers import _ModelBuilderServers
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


def _create_mock_builder(
    env_vars: Optional[dict[str, str]] = None,
    model: str = "Qwen/Qwen3-VL-4B-Instruct",
) -> MagicMock:
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
    builder.schema_builder.sample_input = {
        "inputs": "Hello",
        "parameters": {},
    }
    builder.inference_spec = None
    builder.hf_model_config = {}
    builder.model_data_download_timeout = None
    builder._user_provided_instance_type = True
    builder._is_jumpstart_model_id = Mock(return_value=False)
    builder._auto_detect_image_uri = Mock()
    builder._prepare_for_mode = Mock(
        return_value=("s3://model-data", None)
    )
    builder._create_model = Mock(return_value=Mock())
    builder._optimizing = False
    builder._validate_djl_serving_sample_data = Mock()
    builder._validate_tgi_serving_sample_data = Mock()
    builder._validate_for_triton = Mock()
    builder.get_huggingface_model_metadata = Mock(
        return_value={"pipeline_tag": "text-generation"}
    )
    builder.role_arn = (
        "arn:aws:iam::123456789012:role/SageMakerRole"
    )
    return builder


@pytest.fixture
def mock_builder() -> MagicMock:
    """Create a mock builder with default (empty) env_vars."""
    return _create_mock_builder(env_vars={})


@pytest.fixture
def mock_builder_with_s3() -> MagicMock:
    """Create a mock builder with user-provided S3 HF_MODEL_ID."""
    return _create_mock_builder(
        env_vars={"HF_MODEL_ID": "s3://my-bucket/models/Qwen/"}
    )


S3_PATH = "s3://my-bucket/models/Qwen/"
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


# ---------------------------------------------------------------------------
# DJL Serving
# ---------------------------------------------------------------------------
class TestBuildForDjlHfModelId:
    """Test _build_for_djl preserves user-provided HF_MODEL_ID."""

    _patches = [
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_default_tensor_parallel_degree",
            return_value=1,
        ),
        patch(
            "sagemaker.serve.model_builder_servers._get_gpu_info",
            return_value=1,
        ),
        patch(
            "sagemaker.serve.model_builder_servers._get_nb_instance",
            return_value=None,
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_default_djl_configurations",
            return_value=({}, 256),
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_model_config_properties_from_hf",
            return_value={},
        ),
        patch(
            "sagemaker.serve.model_server.djl_serving"
            ".prepare._create_dir_structure",
        ),
    ]

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        """User-provided S3 URI should not be overwritten."""
        builder = mock_builder_with_s3
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_djl(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        """HF_MODEL_ID should default to self.model."""
        builder = mock_builder
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_djl(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# TGI
# ---------------------------------------------------------------------------
class TestBuildForTgiHfModelId:
    """Test _build_for_tgi preserves user-provided HF_MODEL_ID."""

    _patches = [
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_default_tensor_parallel_degree",
            return_value=1,
        ),
        patch(
            "sagemaker.serve.model_builder_servers._get_gpu_info",
            return_value=1,
        ),
        patch(
            "sagemaker.serve.model_builder_servers._get_nb_instance",
            return_value=None,
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_default_tgi_configurations",
            return_value=({}, 256),
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_model_config_properties_from_hf",
            return_value={},
        ),
        patch(
            "sagemaker.serve.model_server.tgi"
            ".prepare._create_dir_structure",
        ),
    ]

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.TGI
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_tgi(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        builder = mock_builder
        builder.model_server = ModelServer.TGI
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_tgi(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# TEI
# ---------------------------------------------------------------------------
class TestBuildForTeiHfModelId:
    """Test _build_for_tei preserves user-provided HF_MODEL_ID."""

    _patches = [
        patch(
            "sagemaker.serve.model_builder_servers._get_nb_instance",
            return_value=None,
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_model_config_properties_from_hf",
            return_value={},
        ),
        patch(
            "sagemaker.serve.model_server.tgi"
            ".prepare._create_dir_structure",
        ),
    ]

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.TEI
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_tei(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        builder = mock_builder
        builder.model_server = ModelServer.TEI
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_tei(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# TorchServe
# ---------------------------------------------------------------------------
class TestBuildForTorchserveHfModelId:
    """Test _build_for_torchserve preserves user-provided HF_MODEL_ID."""

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.TORCHSERVE
        builder._save_model_inference_spec = Mock()
        _ModelBuilderServers._build_for_torchserve(builder)
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        builder = mock_builder
        builder.model_server = ModelServer.TORCHSERVE
        builder._save_model_inference_spec = Mock()
        _ModelBuilderServers._build_for_torchserve(builder)
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Triton
# ---------------------------------------------------------------------------
class TestBuildForTritonHfModelId:
    """Test _build_for_triton preserves user-provided HF_MODEL_ID."""

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.TRITON
        builder._save_inference_spec = Mock()
        builder._prepare_for_triton = Mock()
        builder._auto_detect_image_for_triton = Mock()
        _ModelBuilderServers._build_for_triton(builder)
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        builder = mock_builder
        builder.model_server = ModelServer.TRITON
        builder._save_inference_spec = Mock()
        builder._prepare_for_triton = Mock()
        builder._auto_detect_image_for_triton = Mock()
        _ModelBuilderServers._build_for_triton(builder)
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------------------
# Transformers (MMS)
# ---------------------------------------------------------------------------
class TestBuildForTransformersHfModelId:
    """Test _build_for_transformers preserves user-provided HF_MODEL_ID."""

    _patches = [
        patch(
            "sagemaker.serve.model_builder_servers._get_nb_instance",
            return_value=None,
        ),
        patch(
            "sagemaker.serve.model_builder_servers"
            "._get_model_config_properties_from_hf",
            return_value={},
        ),
        patch(
            "sagemaker.serve.model_server.multi_model_server"
            ".prepare._create_dir_structure",
        ),
    ]

    def test_preserves_user_provided_s3_uri(
        self, mock_builder_with_s3
    ):
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.MMS
        builder.model_data_download_timeout = None
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_transformers(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self, mock_builder
    ):
        builder = mock_builder
        builder.model_server = ModelServer.MMS
        builder.model_data_download_timeout = None
        for p in self._patches:
            p.start()
        try:
            _ModelBuilderServers._build_for_transformers(builder)
        finally:
            for p in self._patches:
                p.stop()
        assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL

    @patch("sagemaker.serve.model_builder_servers.save_pkl")
    @patch(
        "sagemaker.serve.model_builder_servers"
        "._get_model_config_properties_from_hf",
        return_value={},
    )
    @patch(
        "sagemaker.serve.model_builder_servers._get_nb_instance",
        return_value=None,
    )
    @patch(
        "sagemaker.serve.model_server.multi_model_server"
        ".prepare._create_dir_structure",
    )
    @patch("os.makedirs")
    def test_preserves_with_inference_spec(
        self,
        _mock_makedirs,
        _mock_dir,
        _mock_nb,
        _mock_hf,
        _mock_pkl,
    ):
        """User-provided HF_MODEL_ID preserved with inference_spec."""
        builder = _create_mock_builder(
            env_vars={"HF_MODEL_ID": S3_PATH}
        )
        builder.model_server = ModelServer.MMS
        builder.model_data_download_timeout = None
        builder.model = None
        builder.inference_spec = Mock()
        builder.inference_spec.get_model.return_value = (
            "some-hf-model-id"
        )
        builder._is_jumpstart_model_id = Mock(
            return_value=False
        )
        _ModelBuilderServers._build_for_transformers(builder)
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH
