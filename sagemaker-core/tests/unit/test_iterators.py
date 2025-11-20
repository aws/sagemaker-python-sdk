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
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.core.iterators import (
    handle_stream_errors,
    ByteIterator,
    LineIterator,
)
from sagemaker.core.exceptions import ModelStreamError, InternalStreamFailure


def test_handle_stream_errors_model_stream_error():
    """Test handle_stream_errors raises ModelStreamError."""
    chunk = {
        "ModelStreamError": {
            "Message": "Model error occurred",
            "ErrorCode": "ModelError"
        }
    }
    
    with pytest.raises(ModelStreamError) as exc_info:
        handle_stream_errors(chunk)
    
    assert "Model error occurred" in str(exc_info.value)


def test_handle_stream_errors_internal_stream_failure():
    """Test handle_stream_errors raises InternalStreamFailure."""
    chunk = {
        "InternalStreamFailure": {
            "Message": "Internal failure occurred"
        }
    }
    
    with pytest.raises(InternalStreamFailure) as exc_info:
        handle_stream_errors(chunk)
    
    assert "Internal failure occurred" in str(exc_info.value)


def test_handle_stream_errors_no_error():
    """Test handle_stream_errors does nothing when no error in chunk."""
    chunk = {"PayloadPart": {"Bytes": b"test data"}}
    
    # Should not raise any exception
    handle_stream_errors(chunk)


def test_byte_iterator_initialization():
    """Test ByteIterator initialization."""
    mock_stream = []
    iterator = ByteIterator(mock_stream)
    
    assert iterator.event_stream == mock_stream
    assert hasattr(iterator, 'byte_iterator')


def test_byte_iterator_iter():
    """Test ByteIterator __iter__ returns self."""
    mock_stream = []
    iterator = ByteIterator(mock_stream)
    
    assert iterator.__iter__() == iterator


def test_byte_iterator_next_with_payload():
    """Test ByteIterator __next__ returns bytes from PayloadPart."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b"chunk1"}},
        {"PayloadPart": {"Bytes": b"chunk2"}},
    ]
    iterator = ByteIterator(mock_stream)
    
    assert next(iterator) == b"chunk1"
    assert next(iterator) == b"chunk2"


def test_byte_iterator_next_with_model_error():
    """Test ByteIterator __next__ raises ModelStreamError."""
    mock_stream = [
        {"ModelStreamError": {"Message": "Error", "ErrorCode": "500"}}
    ]
    iterator = ByteIterator(mock_stream)
    
    with pytest.raises(ModelStreamError):
        next(iterator)


def test_byte_iterator_next_with_internal_failure():
    """Test ByteIterator __next__ raises InternalStreamFailure."""
    mock_stream = [
        {"InternalStreamFailure": {"Message": "Failure"}}
    ]
    iterator = ByteIterator(mock_stream)
    
    with pytest.raises(InternalStreamFailure):
        next(iterator)


def test_byte_iterator_next_stop_iteration():
    """Test ByteIterator __next__ raises StopIteration when stream ends."""
    mock_stream = []
    iterator = ByteIterator(mock_stream)
    
    with pytest.raises(StopIteration):
        next(iterator)


def test_byte_iterator_multiple_chunks():
    """Test ByteIterator can iterate through multiple chunks."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b"chunk1"}},
        {"PayloadPart": {"Bytes": b"chunk2"}},
        {"PayloadPart": {"Bytes": b"chunk3"}},
    ]
    iterator = ByteIterator(mock_stream)
    
    chunks = list(iterator)
    assert len(chunks) == 3
    assert chunks[0] == b"chunk1"
    assert chunks[1] == b"chunk2"
    assert chunks[2] == b"chunk3"


def test_line_iterator_initialization():
    """Test LineIterator initialization."""
    mock_stream = []
    iterator = LineIterator(mock_stream)
    
    assert iterator.event_stream == mock_stream
    assert hasattr(iterator, 'byte_iterator')
    assert hasattr(iterator, 'buffer')
    assert iterator.read_pos == 0


def test_line_iterator_iter():
    """Test LineIterator __iter__ returns self."""
    mock_stream = []
    iterator = LineIterator(mock_stream)
    
    assert iterator.__iter__() == iterator


def test_line_iterator_next_single_line():
    """Test LineIterator __next__ returns single complete line."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b'{"outputs": [" test"]}\n'}},
    ]
    iterator = LineIterator(mock_stream)
    
    line = next(iterator)
    assert line == b'{"outputs": [" test"]}'


def test_line_iterator_next_multiple_lines():
    """Test LineIterator __next__ returns multiple lines."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b'{"outputs": [" line1"]}\n'}},
        {"PayloadPart": {"Bytes": b'{"outputs": [" line2"]}\n'}},
    ]
    iterator = LineIterator(mock_stream)
    
    line1 = next(iterator)
    line2 = next(iterator)
    assert line1 == b'{"outputs": [" line1"]}'
    assert line2 == b'{"outputs": [" line2"]}'


def test_line_iterator_split_json():
    """Test LineIterator handles JSON split across multiple chunks."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b'{"outputs": '}},
        {"PayloadPart": {"Bytes": b'[" test"]}\n'}},
    ]
    iterator = LineIterator(mock_stream)
    
    line = next(iterator)
    assert line == b'{"outputs": [" test"]}'


def test_line_iterator_with_model_error():
    """Test LineIterator __next__ raises ModelStreamError."""
    mock_stream = [
        {"ModelStreamError": {"Message": "Error", "ErrorCode": "500"}}
    ]
    iterator = LineIterator(mock_stream)
    
    with pytest.raises(ModelStreamError):
        next(iterator)


def test_line_iterator_with_internal_failure():
    """Test LineIterator __next__ raises InternalStreamFailure."""
    mock_stream = [
        {"InternalStreamFailure": {"Message": "Failure"}}
    ]
    iterator = LineIterator(mock_stream)
    
    with pytest.raises(InternalStreamFailure):
        next(iterator)


def test_line_iterator_stop_iteration():
    """Test LineIterator __next__ raises StopIteration when stream ends."""
    mock_stream = []
    iterator = LineIterator(mock_stream)
    
    with pytest.raises(StopIteration):
        next(iterator)


def test_line_iterator_multiple_lines_in_single_chunk():
    """Test LineIterator handles multiple lines in a single chunk."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b'{"outputs": [" line1"]}\n{"outputs": [" line2"]}\n'}},
    ]
    iterator = LineIterator(mock_stream)
    
    line1 = next(iterator)
    line2 = next(iterator)
    assert line1 == b'{"outputs": [" line1"]}'
    assert line2 == b'{"outputs": [" line2"]}'


def test_line_iterator_incomplete_line_at_end():
    """Test LineIterator handles incomplete line at end of stream."""
    mock_stream = [
        {"PayloadPart": {"Bytes": b'{"outputs": [" complete"]}\n'}},
    ]
    iterator = LineIterator(mock_stream)
    
    line1 = next(iterator)
    assert line1 == b'{"outputs": [" complete"]}'
    
    # After consuming all complete lines, should raise StopIteration
    with pytest.raises(StopIteration):
        next(iterator)
