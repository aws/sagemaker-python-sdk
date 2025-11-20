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
