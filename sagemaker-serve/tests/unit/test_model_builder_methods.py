# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Tests for ModelBuilder simple methods"""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.serve.builder.schema_builder import SchemaBuilder
from sagemaker.core.serializers import NumpySerializer, TorchTensorSerializer
from sagemaker.core.deserializers import JSONDeserializer, TorchTensorDeserializer
from sagemaker.serve.constants import Framework
from sagemaker.serve.mode.function_pointers import Mode


class TestModelBuilderSimpleMethods:
    """Test simple utility methods in ModelBuilder"""
    
    def test_is_mms_version_true(self):
        """Test _is_mms_version returns True for version >= 1.2"""
        builder = ModelBuilder(model=Mock())
        builder.framework_version = "1.2.0"
        assert builder._is_mms_version() is True
        
        builder.framework_version = "1.3.0"
        assert builder._is_mms_version() is True
        
        builder.framework_version = "2.0.0"
        assert builder._is_mms_version() is True
    
    def test_is_mms_version_false(self):
        """Test _is_mms_version returns False for version < 1.2"""
        builder = ModelBuilder(model=Mock())
        builder.framework_version = "1.1.0"
        assert builder._is_mms_version() is False
        
        builder.framework_version = "1.0.0"
        assert builder._is_mms_version() is False
    
    def test_is_mms_version_none(self):
        """Test _is_mms_version returns False when framework_version is None"""
        builder = ModelBuilder(model=Mock())
        builder.framework_version = None
        assert builder._is_mms_version() is False
    
    def test_get_container_env_no_log_level(self):
        """Test _get_container_env returns env when no container_log_level"""
        builder = ModelBuilder(model=Mock())
        builder.env = {"KEY": "value"}
        builder._container_log_level = None
        
        result = builder._get_container_env()
        assert result == {"KEY": "value"}
    
    def test_get_container_env_with_valid_log_level(self):
        """Test _get_container_env adds log level to env"""
        builder = ModelBuilder(model=Mock())
        builder.env = {"KEY": "value"}
        builder._container_log_level = "INFO"
        builder.LOG_LEVEL_MAP = {"INFO": "20", "DEBUG": "10"}
        builder.LOG_LEVEL_PARAM_NAME = "SAGEMAKER_CONTAINER_LOG_LEVEL"
        
        result = builder._get_container_env()
        assert result["KEY"] == "value"
        assert result["SAGEMAKER_CONTAINER_LOG_LEVEL"] == "20"
    
    def test_get_container_env_with_invalid_log_level(self):
        """Test _get_container_env ignores invalid log level"""
        builder = ModelBuilder(model=Mock())
        builder.env = {"KEY": "value"}
        builder._container_log_level = "INVALID"
        builder.LOG_LEVEL_MAP = {"INFO": "20", "DEBUG": "10"}
        
        result = builder._get_container_env()
        assert result == {"KEY": "value"}
    
    def test_get_source_code_env_vars_none(self):
        """Test _get_source_code_env_vars returns empty dict when no source_code"""
        builder = ModelBuilder(model=Mock())
        builder.source_code = None
        
        result = builder._get_source_code_env_vars()
        assert result == {}
    
    def test_get_source_code_env_vars_with_local_dir(self):
        """Test _get_source_code_env_vars with local source directory"""
        builder = ModelBuilder(model=Mock())
        builder.source_code = Mock()
        builder.source_code.entry_script = "inference.py"
        builder.source_code.source_dir = "/local/path"
        builder.region = "us-west-2"
        
        result = builder._get_source_code_env_vars()
        
        assert result["SAGEMAKER_PROGRAM"] == "inference.py"
        assert result["SAGEMAKER_SUBMIT_DIRECTORY"] == "file:///local/path"
        assert result["SAGEMAKER_CONTAINER_LOG_LEVEL"] == "20"
        assert result["SAGEMAKER_REGION"] == "us-west-2"
    
    def test_get_source_code_env_vars_with_s3_dir(self):
        """Test _get_source_code_env_vars with S3 source directory"""
        builder = ModelBuilder(model=Mock())
        builder.source_code = Mock()
        builder.source_code.entry_script = "train.py"
        builder.source_code.source_dir = "s3://bucket/path"
        builder.region = "us-east-1"
        
        result = builder._get_source_code_env_vars()
        
        assert result["SAGEMAKER_PROGRAM"] == "train.py"
        assert result["SAGEMAKER_SUBMIT_DIRECTORY"] == "s3://bucket/path"
        assert result["SAGEMAKER_REGION"] == "us-east-1"
    
    def test_to_string_regular_object(self):
        """Test to_string with regular object"""
        builder = ModelBuilder(model=Mock())
        
        result = builder.to_string("test_string")
        assert result == "test_string"
        
        result = builder.to_string(123)
        assert result == "123"
    
    def test_to_string_pipeline_variable(self):
        """Test to_string with PipelineVariable"""
        builder = ModelBuilder(model=Mock())
        
        mock_pipeline_var = Mock()
        mock_pipeline_var.to_string.return_value = "pipeline_value"
        
        with patch("sagemaker.serve.model_builder.is_pipeline_variable", return_value=True):
            result = builder.to_string(mock_pipeline_var)
            assert result == "pipeline_value"
            mock_pipeline_var.to_string.assert_called_once()
    
    def test_is_repack_false_no_source_dir(self):
        """Test is_repack returns False when source_dir is None"""
        builder = ModelBuilder(model=Mock())
        builder.source_dir = None
        builder.entry_point = "inference.py"
        
        assert builder.is_repack() is False
    
    def test_is_repack_false_no_entry_point(self):
        """Test is_repack returns False when entry_point is None"""
        builder = ModelBuilder(model=Mock())
        builder.source_dir = "/path"
        builder.entry_point = None
        
        assert builder.is_repack() is False
    
    def test_is_repack_false_with_key_prefix(self):
        """Test is_repack returns False when key_prefix is set"""
        builder = ModelBuilder(model=Mock())
        builder.source_dir = "/path"
        builder.entry_point = "inference.py"
        builder.key_prefix = "prefix"
        builder.git_config = None
        
        assert builder.is_repack() is False
    
    def test_is_repack_false_with_git_config(self):
        """Test is_repack returns False when git_config is set"""
        builder = ModelBuilder(model=Mock())
        builder.source_dir = "/path"
        builder.entry_point = "inference.py"
        builder.key_prefix = None
        builder.git_config = {"repo": "url"}
        
        assert builder.is_repack() is False
    
    def test_is_repack_true(self):
        """Test is_repack returns True when conditions are met"""
        builder = ModelBuilder(model=Mock())
        builder.source_dir = "/path"
        builder.entry_point = "inference.py"
        builder.key_prefix = None
        builder.git_config = None
        
        assert builder.is_repack() is True
    
    def test_get_client_translators_with_npy_content_type(self):
        """Test _get_client_translators with numpy content type"""
        builder = ModelBuilder(model=Mock())
        builder.content_type = "application/x-npy"
        builder.accept_type = "application/json"
        builder.schema_builder = None
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            serializer, deserializer = builder._get_client_translators()
            
            assert isinstance(serializer, NumpySerializer)
            assert isinstance(deserializer, JSONDeserializer)
    
    def test_get_client_translators_with_torch_tensor(self):
        """Test _get_client_translators with torch tensor types"""
        builder = ModelBuilder(model=Mock())
        builder.content_type = "tensor/pt"
        builder.accept_type = "tensor/pt"
        builder.schema_builder = None
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            serializer, deserializer = builder._get_client_translators()
            
            assert isinstance(serializer, TorchTensorSerializer)
            assert isinstance(deserializer, TorchTensorDeserializer)
    
    def test_get_client_translators_with_schema_builder(self):
        """Test _get_client_translators uses schema_builder serializers"""
        mock_schema = Mock(spec=SchemaBuilder)
        mock_serializer = Mock()
        mock_deserializer = Mock()
        mock_schema.input_serializer = mock_serializer
        mock_schema.output_deserializer = mock_deserializer
        
        builder = ModelBuilder(model=Mock())
        builder.content_type = None
        builder.accept_type = None
        builder.schema_builder = mock_schema
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            serializer, deserializer = builder._get_client_translators()
            
            assert serializer == mock_serializer
            assert deserializer == mock_deserializer
    
    def test_get_client_translators_with_custom_translators(self):
        """Test _get_client_translators uses custom translators from schema_builder"""
        mock_schema = Mock(spec=SchemaBuilder)
        mock_input_translator = Mock()
        mock_output_translator = Mock()
        mock_schema.custom_input_translator = mock_input_translator
        mock_schema.custom_output_translator = mock_output_translator
        
        builder = ModelBuilder(model=Mock())
        builder.content_type = None
        builder.accept_type = None
        builder.schema_builder = mock_schema
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            serializer, deserializer = builder._get_client_translators()
            
            assert serializer == mock_input_translator
            assert deserializer == mock_output_translator
    
    def test_get_client_translators_fallback_to_framework(self):
        """Test _get_client_translators falls back to framework defaults"""
        mock_auto_serializer = Mock()
        mock_auto_deserializer = Mock()
        
        builder = ModelBuilder(model=Mock())
        builder.content_type = None
        builder.accept_type = None
        builder.schema_builder = None
        builder.framework = "pytorch"
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', 
                         return_value=(mock_auto_serializer, mock_auto_deserializer)):
            serializer, deserializer = builder._get_client_translators()
            
            assert serializer == mock_auto_serializer
            assert deserializer == mock_auto_deserializer
    
    def test_get_client_translators_raises_on_no_serializer(self):
        """Test _get_client_translators raises ValueError when serializer cannot be determined"""
        builder = ModelBuilder(model=Mock())
        builder.content_type = None
        builder.accept_type = "application/json"
        builder.schema_builder = None
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            with pytest.raises(ValueError, match="Cannot determine serializer"):
                builder._get_client_translators()
    
    def test_get_client_translators_raises_on_no_deserializer(self):
        """Test _get_client_translators raises ValueError when deserializer cannot be determined"""
        builder = ModelBuilder(model=Mock())
        builder.content_type = "application/x-npy"
        builder.accept_type = None
        builder.schema_builder = None
        builder.framework = None
        
        with patch.object(builder, '_fetch_serializer_and_deserializer_for_framework', return_value=(None, None)):
            with pytest.raises(ValueError, match="Cannot determine deserializer"):
                builder._get_client_translators()


class TestBuildForPassthroughLocalContainer:
    """Tests for _build_for_passthrough() with LOCAL_CONTAINER mode.

    Validates: Requirements 2.1, 2.2, 3.1
    """

    def _make_mock_session(self):
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        mock_session.config = {}
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        return mock_session

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_initializes_secret_key(
        self, mock_prepare, mock_create
    ):
        """Test that _build_for_passthrough initializes secret_key for LOCAL_CONTAINER mode.

        Bug 1 fix: secret_key must be set to empty string so _deploy_local_endpoint()
        can pass it to LocalEndpoint.create() without raising AttributeError.
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )

        builder._build_for_passthrough()

        assert builder.secret_key == ""

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_calls_prepare_for_mode_local_container(
        self, mock_prepare, mock_create
    ):
        """Test that _build_for_passthrough calls _prepare_for_mode for LOCAL_CONTAINER.

        Bug 2 fix: _prepare_for_mode() must be called so that
        self.modes[str(Mode.LOCAL_CONTAINER)] contains a LocalContainerMode object,
        preventing KeyError when _deploy_local_endpoint() accesses it.
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )

        builder._build_for_passthrough()

        mock_prepare.assert_called_once()

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_does_not_call_prepare_for_mode_sagemaker_endpoint(
        self, mock_prepare, mock_create
    ):
        """Test that _build_for_passthrough does NOT call _prepare_for_mode for SAGEMAKER_ENDPOINT.

        Preservation: SAGEMAKER_ENDPOINT mode passthrough must continue to work without
        calling _prepare_for_mode() (endpoint mode has its own preparation in _create_model).
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )

        builder._build_for_passthrough()

        mock_prepare.assert_not_called()


class TestBuildForPassthroughS3PathPreservation:
    """Tests for _build_for_passthrough() S3 path preservation with ModelTrainer.

    Validates: Requirements 2.4, 3.5
    """

    def _make_mock_session(self):
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        mock_session.config = {}
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        return mock_session

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_preserves_model_path_as_s3_upload_path(
        self, mock_prepare, mock_create
    ):
        """Test that _build_for_passthrough preserves model_path as s3_upload_path.

        Bug 4 fix: When ModelTrainer has set model_path to an S3 URI,
        _build_for_passthrough() must preserve it as s3_upload_path instead of
        unconditionally setting it to None.

        **Validates: Requirements 2.4**
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )
        builder.model_path = "s3://bucket/model.tar.gz"

        builder._build_for_passthrough()

        assert builder.s3_upload_path == "s3://bucket/model.tar.gz"

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_sets_s3_upload_path_none_when_no_model_path(
        self, mock_prepare, mock_create
    ):
        """Test that _build_for_passthrough sets s3_upload_path to None when no model_path.

        Preservation: When no model artifacts are involved (model_path is None),
        s3_upload_path should still be set to None as before.

        **Validates: Requirements 3.5**
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )
        builder.model_path = None

        builder._build_for_passthrough()

        assert builder.s3_upload_path is None


class TestGetDockerClient:
    """Tests for _get_docker_client() Studio Docker client initialization.

    Validates: Requirements 2.3, 3.3
    """

    @patch("sagemaker.serve.mode.local_container_mode.os.path.exists")
    @patch("sagemaker.serve.mode.local_container_mode.check_for_studio", return_value=True)
    @patch("sagemaker.serve.mode.local_container_mode.docker")
    def test_pull_image_studio_uses_socket_path(
        self, mock_docker, mock_check_studio, mock_path_exists
    ):
        """Test that in Studio, _get_docker_client uses the proxy socket path.

        Bug 3 fix: When check_for_studio() returns True and the proxy socket exists,
        DockerClient should be initialized with base_url pointing to that socket.
        """
        from sagemaker.serve.mode.local_container_mode import _get_docker_client

        mock_path_exists.side_effect = lambda p: p == "/docker/proxy/docker.sock"
        mock_client = Mock()
        mock_docker.DockerClient.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            result = _get_docker_client()

        mock_docker.DockerClient.assert_called_once_with(
            base_url="unix:///docker/proxy/docker.sock"
        )
        assert result == mock_client

    @patch("sagemaker.serve.mode.local_container_mode.check_for_studio", return_value=False)
    @patch("sagemaker.serve.mode.local_container_mode.docker")
    def test_pull_image_non_studio_uses_from_env(self, mock_docker, mock_check_studio):
        """Test that in non-Studio environments, _get_docker_client uses docker.from_env().

        Preservation: Standard Docker environments must continue to use docker.from_env().
        """
        from sagemaker.serve.mode.local_container_mode import _get_docker_client

        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("os.environ", {}, clear=True):
            result = _get_docker_client()

        mock_docker.from_env.assert_called_once()
        assert result == mock_client

    @patch("sagemaker.serve.mode.local_container_mode.check_for_studio")
    @patch("sagemaker.serve.mode.local_container_mode.docker")
    def test_pull_image_with_docker_host_env_var(self, mock_docker, mock_check_studio):
        """Test that when DOCKER_HOST is set, docker.from_env() is used regardless of Studio.

        When the user explicitly sets DOCKER_HOST, we should respect that and skip
        Studio detection entirely.
        """
        from sagemaker.serve.mode.local_container_mode import _get_docker_client

        mock_client = Mock()
        mock_docker.from_env.return_value = mock_client

        with patch.dict("os.environ", {"DOCKER_HOST": "tcp://localhost:2375"}):
            result = _get_docker_client()

        mock_docker.from_env.assert_called_once()
        mock_check_studio.assert_not_called()
        assert result == mock_client


class TestPreservationNonPassthroughBehavior:
    """Tests for Property 4: Preservation - Non-Passthrough and Non-LOCAL_CONTAINER Behavior.

    Verifies that the bugfix does not alter behavior for:
    - Model-server-specific build methods (e.g., _build_for_torchserve)
    - SAGEMAKER_ENDPOINT passthrough mode

    **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5**
    """

    def _make_mock_session(self):
        mock_session = Mock()
        mock_session.boto_region_name = "us-west-2"
        mock_session.config = {}
        mock_session.boto_session = Mock()
        mock_session.boto_session.region_name = "us-west-2"
        return mock_session

    @patch('sagemaker.serve.model_builder.ModelBuilder._create_model')
    @patch('sagemaker.serve.model_builder.ModelBuilder._prepare_for_mode')
    @patch('sagemaker.serve.model_builder.ModelBuilder._save_model_inference_spec')
    def test_build_for_torchserve_still_calls_prepare_for_mode(
        self, mock_save_spec, mock_prepare, mock_create
    ):
        """Test that _build_for_torchserve still calls _prepare_for_mode for LOCAL_CONTAINER.

        Preservation: The torchserve build path already calls _prepare_for_mode()
        for local modes. This must remain unchanged after the passthrough bugfix.

        **Validates: Requirements 3.2**
        """
        mock_create.return_value = Mock()

        builder = ModelBuilder(
            model=Mock(),
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.LOCAL_CONTAINER,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )
        builder.model_server = Mock()
        builder.env_vars = {}
        builder.shared_libs = []
        builder.dependencies = {}
        builder.inference_spec = None
        builder.model_path = None

        builder._build_for_torchserve()

        mock_prepare.assert_called_once()

    @patch.object(ModelBuilder, '_create_model')
    @patch.object(ModelBuilder, '_prepare_for_mode')
    def test_build_for_passthrough_sagemaker_endpoint_unchanged(
        self, mock_prepare, mock_create
    ):
        """Test that SAGEMAKER_ENDPOINT passthrough is fully unchanged by the bugfix.

        Preservation: SAGEMAKER_ENDPOINT passthrough must continue to:
        - Initialize secret_key to empty string
        - NOT call _prepare_for_mode()
        - Set s3_upload_path to None when no model_path
        - Return the model from _create_model()

        **Validates: Requirements 3.1, 3.5**
        """
        mock_model = Mock()
        mock_create.return_value = mock_model

        builder = ModelBuilder(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            mode=Mode.SAGEMAKER_ENDPOINT,
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            sagemaker_session=self._make_mock_session(),
        )
        builder.model_path = None

        result = builder._build_for_passthrough()

        assert builder.secret_key == ""
        assert builder.s3_upload_path is None
        mock_prepare.assert_not_called()
        mock_create.assert_called_once()
        assert result == mock_model

