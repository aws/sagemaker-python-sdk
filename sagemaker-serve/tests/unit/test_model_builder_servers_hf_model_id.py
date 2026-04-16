"""Unit tests: HF_MODEL_ID is not overwritten when user provides it."""
from __future__ import annotations

from typing import Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest

from sagemaker.serve.model_builder_servers import (
    _ModelBuilderServers,
)
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.mode.function_pointers import Mode


S3_PATH = "s3://my-bucket/models/Qwen/"
DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def _create_mock_builder(
    env_vars: Optional[Dict[str, str]] = None,
    model: str = DEFAULT_MODEL,
) -> MagicMock:
    """Create a mock builder with common attributes set."""
    builder = MagicMock(spec=_ModelBuilderServers)
    builder.model = model
    builder.env_vars = (
        env_vars if env_vars is not None else {}
    )
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
    builder._is_jumpstart_model_id = Mock(
        return_value=False
    )
    builder._auto_detect_image_uri = Mock()
    builder._prepare_for_mode = Mock(
        return_value=("s3://model-data", None)
    )
    builder._create_model = Mock(return_value=Mock())
    builder._optimizing = False
    builder._validate_djl_serving_sample_data = Mock()
    builder._validate_tgi_serving_sample_data = Mock()
    builder._validate_for_triton = Mock()
    builder._save_model_inference_spec = Mock()
    builder._save_inference_spec = Mock()
    builder._prepare_for_triton = Mock()
    builder._auto_detect_image_for_triton = Mock()
    builder.get_huggingface_model_metadata = Mock(
        return_value={"pipeline_tag": "text-generation"}
    )
    builder.role_arn = (
        "arn:aws:iam::123456789012:role/SageMakerRole"
    )
    return builder


@pytest.fixture
def mock_builder() -> MagicMock:
    """Create a mock builder with default env_vars."""
    return _create_mock_builder(env_vars={})


@pytest.fixture
def mock_builder_with_s3() -> MagicMock:
    """Mock builder with user-provided S3 HF_MODEL_ID."""
    return _create_mock_builder(
        env_vars={"HF_MODEL_ID": S3_PATH}
    )


# -- Patch targets for each server type --------------------------

_DJL_PATCHES: List[str] = [
    "sagemaker.serve.model_builder_servers"
    "._get_default_tensor_parallel_degree",
    "sagemaker.serve.model_builder_servers"
    "._get_gpu_info",
    "sagemaker.serve.model_builder_servers"
    "._get_nb_instance",
    "sagemaker.serve.model_builder_servers"
    "._get_default_djl_configurations",
    "sagemaker.serve.model_builder_servers"
    "._get_model_config_properties_from_hf",
    "sagemaker.serve.model_server.djl_serving"
    ".prepare._create_dir_structure",
]

_DJL_RETURN_VALUES = [
    1,          # tensor_parallel_degree
    1,          # gpu_info
    None,       # nb_instance
    ({}, 256),  # djl_configurations
    {},         # hf_model_config
    None,       # _create_dir_structure
]

_TGI_PATCHES: List[str] = [
    "sagemaker.serve.model_builder_servers"
    "._get_default_tensor_parallel_degree",
    "sagemaker.serve.model_builder_servers"
    "._get_gpu_info",
    "sagemaker.serve.model_builder_servers"
    "._get_nb_instance",
    "sagemaker.serve.model_builder_servers"
    "._get_default_tgi_configurations",
    "sagemaker.serve.model_builder_servers"
    "._get_model_config_properties_from_hf",
    "sagemaker.serve.model_server.tgi"
    ".prepare._create_dir_structure",
]

_TGI_RETURN_VALUES = [
    1,          # tensor_parallel_degree
    1,          # gpu_info
    None,       # nb_instance
    ({}, 256),  # tgi_configurations
    {},         # hf_model_config
    None,       # _create_dir_structure
]

_TEI_PATCHES: List[str] = [
    "sagemaker.serve.model_builder_servers"
    "._get_nb_instance",
    "sagemaker.serve.model_builder_servers"
    "._get_model_config_properties_from_hf",
    "sagemaker.serve.model_server.tgi"
    ".prepare._create_dir_structure",
]

_TEI_RETURN_VALUES = [
    None,  # nb_instance
    {},    # hf_model_config
    None,  # _create_dir_structure
]

_TORCHSERVE_PATCHES: List[str] = [
    "sagemaker.serve.model_builder_servers"
    ".prepare_for_torchserve",
]

_TORCHSERVE_RETURN_VALUES = [
    "mock-secret-key",  # prepare_for_torchserve
]

_TRITON_PATCHES: List[str] = []
_TRITON_RETURN_VALUES: list = []

_MMS_PATCHES: List[str] = [
    "sagemaker.serve.model_builder_servers"
    "._get_nb_instance",
    "sagemaker.serve.model_builder_servers"
    "._get_model_config_properties_from_hf",
    "sagemaker.serve.model_server.multi_model_server"
    ".prepare._create_dir_structure",
]

_MMS_RETURN_VALUES = [
    None,  # nb_instance
    {},    # hf_model_config
    None,  # _create_dir_structure
]


def _apply_patches(
    targets: List[str],
    return_values: list,
) -> List:
    """Start patches and return the list of patchers."""
    patchers = []
    for target, rv in zip(targets, return_values):
        p = patch(target, return_value=rv)
        p.start()
        patchers.append(p)
    return patchers


def _stop_patches(patchers: List) -> None:
    """Stop all patchers."""
    for p in patchers:
        p.stop()


# ---------------------------------------------------------------
# Parametrised tests: preserve user-provided HF_MODEL_ID
# ---------------------------------------------------------------
@pytest.mark.parametrize(
    "build_method, server_type, patch_targets, patch_rvs",
    [
        (
            "_build_for_djl",
            ModelServer.DJL_SERVING,
            _DJL_PATCHES,
            _DJL_RETURN_VALUES,
        ),
        (
            "_build_for_tgi",
            ModelServer.TGI,
            _TGI_PATCHES,
            _TGI_RETURN_VALUES,
        ),
        (
            "_build_for_tei",
            ModelServer.TEI,
            _TEI_PATCHES,
            _TEI_RETURN_VALUES,
        ),
        (
            "_build_for_torchserve",
            ModelServer.TORCHSERVE,
            _TORCHSERVE_PATCHES,
            _TORCHSERVE_RETURN_VALUES,
        ),
        (
            "_build_for_triton",
            ModelServer.TRITON,
            _TRITON_PATCHES,
            _TRITON_RETURN_VALUES,
        ),
    ],
    ids=[
        "djl",
        "tgi",
        "tei",
        "torchserve",
        "triton",
    ],
)
def test_preserves_user_provided_hf_model_id(
    build_method: str,
    server_type: ModelServer,
    patch_targets: List[str],
    patch_rvs: list,
    mock_builder_with_s3: MagicMock,
) -> None:
    """User-provided HF_MODEL_ID must not be overwritten."""
    builder = mock_builder_with_s3
    builder.model_server = server_type
    patchers = _apply_patches(patch_targets, patch_rvs)
    try:
        getattr(
            _ModelBuilderServers, build_method
        )(builder)
    finally:
        _stop_patches(patchers)
    assert builder.env_vars["HF_MODEL_ID"] == S3_PATH


@pytest.mark.parametrize(
    "build_method, server_type, patch_targets, patch_rvs",
    [
        (
            "_build_for_djl",
            ModelServer.DJL_SERVING,
            _DJL_PATCHES,
            _DJL_RETURN_VALUES,
        ),
        (
            "_build_for_tgi",
            ModelServer.TGI,
            _TGI_PATCHES,
            _TGI_RETURN_VALUES,
        ),
        (
            "_build_for_tei",
            ModelServer.TEI,
            _TEI_PATCHES,
            _TEI_RETURN_VALUES,
        ),
        (
            "_build_for_torchserve",
            ModelServer.TORCHSERVE,
            _TORCHSERVE_PATCHES,
            _TORCHSERVE_RETURN_VALUES,
        ),
        (
            "_build_for_triton",
            ModelServer.TRITON,
            _TRITON_PATCHES,
            _TRITON_RETURN_VALUES,
        ),
    ],
    ids=[
        "djl",
        "tgi",
        "tei",
        "torchserve",
        "triton",
    ],
)
def test_sets_default_hf_model_id_when_not_provided(
    build_method: str,
    server_type: ModelServer,
    patch_targets: List[str],
    patch_rvs: list,
    mock_builder: MagicMock,
) -> None:
    """HF_MODEL_ID should default to self.model."""
    builder = mock_builder
    builder.model_server = server_type
    patchers = _apply_patches(patch_targets, patch_rvs)
    try:
        getattr(
            _ModelBuilderServers, build_method
        )(builder)
    finally:
        _stop_patches(patchers)
    assert builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL


# ---------------------------------------------------------------
# Transformers (MMS) — needs extra patches
# ---------------------------------------------------------------
class TestBuildForTransformersHfModelId:
    """_build_for_transformers preserves HF_MODEL_ID."""

    def test_preserves_user_provided_s3_uri(
        self,
        mock_builder_with_s3: MagicMock,
    ) -> None:
        """User S3 URI is preserved."""
        builder = mock_builder_with_s3
        builder.model_server = ModelServer.MMS
        patchers = _apply_patches(
            _MMS_PATCHES, _MMS_RETURN_VALUES
        )
        try:
            _ModelBuilderServers._build_for_transformers(
                builder
            )
        finally:
            _stop_patches(patchers)
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH

    def test_sets_default_when_not_provided(
        self,
        mock_builder: MagicMock,
    ) -> None:
        """HF_MODEL_ID defaults to self.model."""
        builder = mock_builder
        builder.model_server = ModelServer.MMS
        patchers = _apply_patches(
            _MMS_PATCHES, _MMS_RETURN_VALUES
        )
        try:
            _ModelBuilderServers._build_for_transformers(
                builder
            )
        finally:
            _stop_patches(patchers)
        assert (
            builder.env_vars["HF_MODEL_ID"] == DEFAULT_MODEL
        )

    @patch(
        "sagemaker.serve.model_builder_servers.save_pkl"
    )
    @patch(
        "sagemaker.serve.model_builder_servers"
        "._get_model_config_properties_from_hf",
        return_value={},
    )
    @patch(
        "sagemaker.serve.model_builder_servers"
        "._get_nb_instance",
        return_value=None,
    )
    @patch(
        "sagemaker.serve.model_server"
        ".multi_model_server"
        ".prepare._create_dir_structure",
    )
    @patch("os.makedirs")
    def test_preserves_with_inference_spec(
        self,
        _mock_makedirs: Mock,
        _mock_dir: Mock,
        _mock_nb: Mock,
        _mock_hf: Mock,
        _mock_pkl: Mock,
    ) -> None:
        """User HF_MODEL_ID preserved with inference_spec."""
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
        _ModelBuilderServers._build_for_transformers(
            builder
        )
        assert builder.env_vars["HF_MODEL_ID"] == S3_PATH
