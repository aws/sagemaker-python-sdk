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
"""Tests for triton_schema_builder module"""
from __future__ import absolute_import

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from sagemaker.serve.builder.triton_schema_builder import (
    TritonSchemaBuilder,
    TORCH_TENSOR,
    TF_TENSOR,
    NUMPY_ARRAY,
    PYTHON_LIST,
    SUPPORTED_TYPES,
    CLASS_TO_TRANSLATOR_MAP,
    PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP,
    TENSORFLOW_TO_TRITON_DTYPE_MAP,
    NUMPY_ARRAY_TRITON_DTYPE_MAP,
    DEFAULT_DTYPE,
)


class TestTritonSchemaBuilder:
    def test_init(self):
        builder = TritonSchemaBuilder()
        assert builder._input_class_name is None
        assert builder._output_class_name is None
        assert builder._input_triton_dtype is None
        assert builder._output_triton_dtype is None
        assert builder._sample_input_ndarray is None
        assert builder._sample_output_ndarray is None
    
    def test_detect_class_numpy_arrays(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = np.array([1, 2, 3])
        builder.sample_output = np.array([4, 5, 6])
        
        builder._detect_class_of_sample_input_and_output()
        
        assert builder._input_class_name == NUMPY_ARRAY
        assert builder._output_class_name == NUMPY_ARRAY
    
    def test_detect_class_unsupported_input(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = "unsupported_type"
        builder.sample_output = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Unable to update input serializer"):
            builder._detect_class_of_sample_input_and_output()
    
    def test_detect_class_unsupported_output(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = np.array([1, 2, 3])
        builder.sample_output = "unsupported_type"
        
        with pytest.raises(ValueError, match="Unable to update output serializer"):
            builder._detect_class_of_sample_input_and_output()
    
    def test_detect_dtype_for_numpy(self):
        builder = TritonSchemaBuilder()
        
        # Test float32
        data = np.array([1.0, 2.0], dtype=np.float32)
        result = builder._detect_dtype_for_numpy(data)
        assert result == "TYPE_FP32"
        
        # Test int64
        data = np.array([1, 2], dtype=np.int64)
        result = builder._detect_dtype_for_numpy(data)
        assert result == "TYPE_INT64"
        
        # Test bool
        data = np.array([True, False], dtype=bool)
        result = builder._detect_dtype_for_numpy(data)
        assert result == "TYPE_BOOL"
        
        # Test unsupported dtype (should return default)
        data = np.array([1, 2], dtype=np.complex64)
        result = builder._detect_dtype_for_numpy(data)
        assert result == DEFAULT_DTYPE
    
    def test_detect_dtype_for_pytorch_tensor(self):
        builder = TritonSchemaBuilder()
        
        # Mock torch tensor
        mock_tensor = Mock()
        mock_tensor.dtype = "torch.float32"
        result = builder._detect_dtype_for_pytorch_tensor(mock_tensor)
        assert result == "TYPE_FP32"
        
        # Test unsupported dtype
        mock_tensor.dtype = "torch.unknown"
        result = builder._detect_dtype_for_pytorch_tensor(mock_tensor)
        assert result == DEFAULT_DTYPE
    
    def test_detect_dtype_for_tensorflow(self):
        builder = TritonSchemaBuilder()
        
        # Mock tensorflow tensor
        mock_tensor = Mock()
        mock_tensor.dtype.name = "float32"
        result = builder._detect_dtype_for_tensorflow(mock_tensor)
        assert result == "TYPE_FP32"
        
        # Test unsupported dtype
        mock_tensor.dtype.name = "unknown"
        result = builder._detect_dtype_for_tensorflow(mock_tensor)
        assert result == DEFAULT_DTYPE
    
    def test_detect_dtype_for_triton_numpy(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = np.array([1.0, 2.0], dtype=np.float32)
        builder.sample_output = np.array([3, 4], dtype=np.int32)
        builder._input_class_name = NUMPY_ARRAY
        builder._output_class_name = NUMPY_ARRAY
        
        builder._detect_dtype_for_triton()
        
        assert builder._input_triton_dtype == "TYPE_FP32"
        assert builder._output_triton_dtype == "TYPE_INT32"
    
    def test_detect_dtype_for_triton_torch(self):
        builder = TritonSchemaBuilder()
        
        # Mock torch tensors
        mock_input = Mock()
        mock_input.dtype = "torch.float16"
        mock_output = Mock()
        mock_output.dtype = "torch.int64"
        
        builder.sample_input = mock_input
        builder.sample_output = mock_output
        builder._input_class_name = TORCH_TENSOR
        builder._output_class_name = TORCH_TENSOR
        
        builder._detect_dtype_for_triton()
        
        assert builder._input_triton_dtype == "TYPE_FP16"
        assert builder._output_triton_dtype == "TYPE_INT64"
    
    def test_detect_dtype_for_triton_tensorflow(self):
        builder = TritonSchemaBuilder()
        
        # Mock tensorflow tensors
        mock_input = Mock()
        mock_input.dtype.name = "float64"
        mock_output = Mock()
        mock_output.dtype.name = "int16"
        
        builder.sample_input = mock_input
        builder.sample_output = mock_output
        builder._input_class_name = TF_TENSOR
        builder._output_class_name = TF_TENSOR
        
        builder._detect_dtype_for_triton()
        
        assert builder._input_triton_dtype == "TYPE_FP64"
        assert builder._output_triton_dtype == "TYPE_INT16"
    
    def test_detect_dtype_for_triton_default(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = Mock()
        builder.sample_output = Mock()
        builder._input_class_name = PYTHON_LIST
        builder._output_class_name = PYTHON_LIST
        
        builder._detect_dtype_for_triton()
        
        assert builder._input_triton_dtype == DEFAULT_DTYPE
        assert builder._output_triton_dtype == DEFAULT_DTYPE
    
    def test_update_serializer_deserializer_for_triton_numpy(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = np.array([1, 2, 3])
        builder.sample_output = np.array([4, 5, 6])
        
        builder._update_serializer_deserializer_for_triton()
        
        assert builder._input_class_name == NUMPY_ARRAY
        assert builder._output_class_name == NUMPY_ARRAY
        assert builder.input_serializer is not None
        assert builder.input_deserializer is not None
        assert builder.output_serializer is not None
        assert builder.output_deserializer is not None
        assert builder._sample_input_ndarray is not None
        assert builder._sample_output_ndarray is not None
    
    def test_update_serializer_deserializer_for_triton_validation_error(self):
        builder = TritonSchemaBuilder()
        builder.sample_input = np.array([1, 2, 3])
        builder.sample_output = np.array([4, 5, 6])
        
        # Mock serializer to raise exception
        from unittest.mock import patch
        with patch.object(
            CLASS_TO_TRANSLATOR_MAP[NUMPY_ARRAY],
            'serialize',
            side_effect=Exception("Serialization failed")
        ):
            with pytest.raises(ValueError, match="Validation of serialization and deserialization failed"):
                builder._update_serializer_deserializer_for_triton()


class TestConstants:
    def test_supported_types(self):
        assert TORCH_TENSOR in SUPPORTED_TYPES
        assert TF_TENSOR in SUPPORTED_TYPES
        assert NUMPY_ARRAY in SUPPORTED_TYPES
        assert PYTHON_LIST not in SUPPORTED_TYPES
    
    def test_class_to_translator_map(self):
        assert TORCH_TENSOR in CLASS_TO_TRANSLATOR_MAP
        assert TF_TENSOR in CLASS_TO_TRANSLATOR_MAP
        assert NUMPY_ARRAY in CLASS_TO_TRANSLATOR_MAP
        assert PYTHON_LIST in CLASS_TO_TRANSLATOR_MAP
    
    def test_pytorch_dtype_mappings(self):
        assert PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP["torch.float32"] == "TYPE_FP32"
        assert PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP["torch.int64"] == "TYPE_INT64"
        assert PYTORCH_TENSOR_TO_TRITON_DTYPE_MAP["torch.bool"] == "TYPE_BOOL"
    
    def test_tensorflow_dtype_mappings(self):
        assert TENSORFLOW_TO_TRITON_DTYPE_MAP["float32"] == "TYPE_FP32"
        assert TENSORFLOW_TO_TRITON_DTYPE_MAP["int64"] == "TYPE_INT64"
        assert TENSORFLOW_TO_TRITON_DTYPE_MAP["bool"] == "TYPE_BOOL"
    
    def test_numpy_dtype_mappings(self):
        assert NUMPY_ARRAY_TRITON_DTYPE_MAP["float32"] == "TYPE_FP32"
        assert NUMPY_ARRAY_TRITON_DTYPE_MAP["int64"] == "TYPE_INT64"
        assert NUMPY_ARRAY_TRITON_DTYPE_MAP["bool"] == "TYPE_BOOL"
