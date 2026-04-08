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
"""Tests for serializer and deserializer implementations."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from sagemaker.core.serializers.base import (
    CSVSerializer,
    NumpySerializer,
    JSONSerializer,
    IdentitySerializer,
    JSONLinesSerializer,
    StringSerializer,
    DataSerializer,
    LibSVMSerializer,
)
from sagemaker.core.deserializers.base import (
    StringDeserializer,
    BytesDeserializer,
    CSVDeserializer,
    NumpyDeserializer,
    JSONDeserializer,
    JSONLinesDeserializer,
    StreamDeserializer,
)


class TestCSVSerializer:
    def test_serialize_list(self):
        serializer = CSVSerializer()
        result = serializer.serialize([1, 2, 3])
        assert result == "1,2,3"

    def test_serialize_numpy_array(self):
        serializer = CSVSerializer()
        result = serializer.serialize(np.array([1, 2, 3]))
        assert result == "1,2,3"

    def test_serialize_2d_list(self):
        serializer = CSVSerializer()
        result = serializer.serialize([[1, 2], [3, 4]])
        assert result == "1,2\n3,4"

    def test_serialize_string(self):
        serializer = CSVSerializer()
        result = serializer.serialize("hello")
        assert result == "hello"

    def test_content_type(self):
        serializer = CSVSerializer()
        assert serializer.CONTENT_TYPE == "text/csv"


class TestNumpySerializer:
    def test_serialize_numpy_array(self):
        serializer = NumpySerializer()
        data = np.array([1.0, 2.0, 3.0])
        result = serializer.serialize(data)
        assert result is not None
        loaded = np.load(io.BytesIO(result))
        assert np.array_equal(loaded, data)

    def test_serialize_list(self):
        serializer = NumpySerializer()
        result = serializer.serialize([1, 2, 3])
        assert result is not None

    def test_serialize_empty_array_raises(self):
        serializer = NumpySerializer()
        with pytest.raises(ValueError, match="Cannot serialize empty array"):
            serializer.serialize(np.array([]))

    def test_content_type(self):
        serializer = NumpySerializer()
        assert serializer.CONTENT_TYPE == "application/x-npy"


class TestJSONSerializer:
    def test_serialize_dict(self):
        serializer = JSONSerializer()
        result = serializer.serialize({"key": "value"})
        assert json.loads(result) == {"key": "value"}

    def test_serialize_list(self):
        serializer = JSONSerializer()
        result = serializer.serialize([1, 2, 3])
        assert json.loads(result) == [1, 2, 3]

    def test_serialize_numpy_array(self):
        serializer = JSONSerializer()
        result = serializer.serialize(np.array([1, 2, 3]))
        assert json.loads(result) == [1, 2, 3]

    def test_content_type(self):
        serializer = JSONSerializer()
        assert serializer.CONTENT_TYPE == "application/json"


class TestIdentitySerializer:
    def test_serialize(self):
        serializer = IdentitySerializer()
        data = b"raw bytes"
        assert serializer.serialize(data) == data

    def test_content_type(self):
        serializer = IdentitySerializer()
        assert serializer.CONTENT_TYPE == "application/octet-stream"


class TestJSONLinesSerializer:
    def test_serialize_iterable(self):
        serializer = JSONLinesSerializer()
        result = serializer.serialize([{"a": 1}, {"b": 2}])
        lines = result.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0]) == {"a": 1}
        assert json.loads(lines[1]) == {"b": 2}

    def test_serialize_string(self):
        serializer = JSONLinesSerializer()
        result = serializer.serialize("already formatted")
        assert result == "already formatted"

    def test_content_type(self):
        serializer = JSONLinesSerializer()
        assert serializer.CONTENT_TYPE == "application/jsonlines"


class TestStringSerializer:
    def test_serialize_string(self):
        serializer = StringSerializer()
        result = serializer.serialize("hello")
        assert result == b"hello"

    def test_serialize_non_string_raises(self):
        serializer = StringSerializer()
        with pytest.raises(ValueError, match="is not String serializable"):
            serializer.serialize(123)

    def test_content_type(self):
        serializer = StringSerializer()
        assert serializer.CONTENT_TYPE == "text/plain"


class TestLibSVMSerializer:
    def test_serialize_string(self):
        serializer = LibSVMSerializer()
        data = "1 1:1 2:2\n0 1:3 2:4"
        assert serializer.serialize(data) == data

    def test_serialize_invalid_raises(self):
        serializer = LibSVMSerializer()
        with pytest.raises(ValueError, match="Unable to handle input format"):
            serializer.serialize(123)

    def test_content_type(self):
        serializer = LibSVMSerializer()
        assert serializer.CONTENT_TYPE == "text/libsvm"


class TestDataSerializer:
    def test_serialize_bytes(self):
        serializer = DataSerializer()
        data = b"raw bytes"
        assert serializer.serialize(data) == data

    def test_serialize_invalid_raises(self):
        serializer = DataSerializer()
        with pytest.raises(ValueError, match="is not Data serializable"):
            serializer.serialize(123)

    def test_content_type(self):
        serializer = DataSerializer()
        assert serializer.CONTENT_TYPE == "file-path/raw-bytes"


class MockStream:
    """Mock stream for testing deserializers."""

    def __init__(self, data):
        self._stream = io.BytesIO(data)

    def read(self):
        return self._stream.read()

    def close(self):
        self._stream.close()


class TestStringDeserializer:
    def test_deserialize(self):
        deserializer = StringDeserializer()
        stream = MockStream(b"hello world")
        result = deserializer.deserialize(stream, "application/json")
        assert result == "hello world"


class TestBytesDeserializer:
    def test_deserialize(self):
        deserializer = BytesDeserializer()
        stream = MockStream(b"raw bytes")
        result = deserializer.deserialize(stream, "application/octet-stream")
        assert result == b"raw bytes"


class TestCSVDeserializer:
    def test_deserialize(self):
        deserializer = CSVDeserializer()
        stream = MockStream(b"1,2,3\n4,5,6")
        result = deserializer.deserialize(stream, "text/csv")
        assert result == [["1", "2", "3"], ["4", "5", "6"]]


class TestNumpyDeserializer:
    def test_deserialize_npy(self):
        deserializer = NumpyDeserializer()
        array = np.array([1.0, 2.0, 3.0])
        buffer = io.BytesIO()
        np.save(buffer, array)
        stream = MockStream(buffer.getvalue())
        result = deserializer.deserialize(stream, "application/x-npy")
        assert np.array_equal(result, array)

    def test_deserialize_csv(self):
        deserializer = NumpyDeserializer()
        stream = MockStream(b"1,2,3")
        result = deserializer.deserialize(stream, "text/csv")
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_deserialize_json(self):
        deserializer = NumpyDeserializer()
        stream = MockStream(b"[1, 2, 3]")
        result = deserializer.deserialize(stream, "application/json")
        assert np.array_equal(result, np.array([1, 2, 3]))


class TestJSONDeserializer:
    def test_deserialize(self):
        deserializer = JSONDeserializer()
        stream = MockStream(json.dumps({"key": "value"}).encode("utf-8"))
        result = deserializer.deserialize(stream, "application/json")
        assert result == {"key": "value"}


class TestJSONLinesDeserializer:
    def test_deserialize(self):
        deserializer = JSONLinesDeserializer()
        data = '{"a": 1}\n{"b": 2}'.encode("utf-8")
        stream = MockStream(data)
        result = deserializer.deserialize(stream, "application/jsonlines")
        assert result == [{"a": 1}, {"b": 2}]


class TestStreamDeserializer:
    def test_deserialize(self):
        deserializer = StreamDeserializer()
        stream = MockStream(b"data")
        result_stream, result_type = deserializer.deserialize(stream, "application/octet-stream")
        assert result_type == "application/octet-stream"


class TestTorchTensorSerializer:
    """Tests for TorchTensorSerializer that require torch."""

    def test_serialize(self):
        torch = pytest.importorskip("torch")
        from sagemaker.core.serializers.base import TorchTensorSerializer

        serializer = TorchTensorSerializer()
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = serializer.serialize(tensor)
        assert result is not None
        array = np.load(io.BytesIO(result))
        assert np.array_equal(array, np.array([1.0, 2.0, 3.0]))

    def test_serialize_non_tensor_raises(self):
        pytest.importorskip("torch")
        from sagemaker.core.serializers.base import TorchTensorSerializer

        serializer = TorchTensorSerializer()
        with pytest.raises(ValueError, match="is not a torch.Tensor"):
            serializer.serialize("not a tensor")

    def test_content_type(self):
        pytest.importorskip("torch")
        from sagemaker.core.serializers.base import TorchTensorSerializer

        serializer = TorchTensorSerializer()
        assert serializer.CONTENT_TYPE == "tensor/pt"


class TestTorchTensorDeserializer:
    """Tests for TorchTensorDeserializer that require torch."""

    def test_deserialize(self):
        torch = pytest.importorskip("torch")
        from sagemaker.core.deserializers.base import TorchTensorDeserializer

        deserializer = TorchTensorDeserializer()
        array = np.array([1.0, 2.0, 3.0])
        buffer = io.BytesIO()
        np.save(buffer, array)
        buffer.seek(0)
        result = deserializer.deserialize(buffer, "tensor/pt")
        assert isinstance(result, torch.Tensor)
        assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))

    def test_content_type(self):
        pytest.importorskip("torch")
        from sagemaker.core.deserializers.base import TorchTensorDeserializer

        deserializer = TorchTensorDeserializer()
        assert deserializer.ACCEPT == ("tensor/pt",)
