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
"""Unit tests for sagemaker.core.serializers.utils module."""
from __future__ import absolute_import

import pytest
import struct
import numpy as np
from io import BytesIO
from unittest.mock import Mock, patch

from sagemaker.core.serializers.utils import (
    _write_recordio,
    read_recordio,
    _resolve_type,
)


# Note: Tests for functions that depend on sagemaker.core.amazon.record_pb2.Record
# have been removed as that module has been deprecated:
# - TestWriteFeatureTensor
# - TestWriteLabelTensor
# - TestWriteKeysTensor
# - TestWriteShape
# - TestWriteNumpyToDenseTensor
# - TestWriteSpmatrixToSparseTensor
# - TestReadRecords


class TestWriteRecordio:
    """Test _write_recordio function."""

    def test_write_recordio_basic(self):
        """Test writing recordio format."""
        file = BytesIO()
        data = b"test data"
        
        _write_recordio(file, data)
        
        file.seek(0)
        content = file.read()
        assert len(content) > len(data)

    def test_write_recordio_with_padding(self):
        """Test writing recordio with padding."""
        file = BytesIO()
        data = b"x"  # Single byte requires padding
        
        _write_recordio(file, data)
        
        file.seek(0)
        content = file.read()
        # Should have magic number (4 bytes) + length (4 bytes) + data (1 byte) + padding (3 bytes)
        assert len(content) == 12


class TestReadRecordio:
    """Test read_recordio function."""

    def test_read_recordio_basic(self):
        """Test reading recordio format."""
        file = BytesIO()
        data1 = b"first record"
        data2 = b"second record"
        
        _write_recordio(file, data1)
        _write_recordio(file, data2)
        
        file.seek(0)
        records = list(read_recordio(file))
        
        assert len(records) == 2
        assert records[0] == data1
        assert records[1] == data2

    def test_read_recordio_empty_file(self):
        """Test reading from empty file."""
        file = BytesIO()
        records = list(read_recordio(file))
        
        assert len(records) == 0

    def test_read_recordio_single_record(self):
        """Test reading single record."""
        file = BytesIO()
        data = b"single record"
        
        _write_recordio(file, data)
        
        file.seek(0)
        records = list(read_recordio(file))
        
        assert len(records) == 1
        assert records[0] == data


class TestResolveType:
    """Test _resolve_type function."""

    def test_resolve_type_int(self):
        """Test resolving int type."""
        result = _resolve_type(np.dtype(int))
        assert result == "Int32"

    def test_resolve_type_float(self):
        """Test resolving float type."""
        result = _resolve_type(np.dtype(float))
        assert result == "Float64"

    def test_resolve_type_float32(self):
        """Test resolving float32 type."""
        result = _resolve_type(np.dtype("float32"))
        assert result == "Float32"

    def test_resolve_type_unsupported(self):
        """Test resolving unsupported type."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            _resolve_type(np.dtype("complex64"))


class TestRecordioRoundTrip:
    """Test round-trip conversion."""

    def test_recordio_round_trip(self):
        """Test writing and reading back data."""
        file = BytesIO()
        original_data = [b"record1", b"record2", b"record3"]
        
        for data in original_data:
            _write_recordio(file, data)
        
        file.seek(0)
        read_data = list(read_recordio(file))
        
        assert read_data == original_data
