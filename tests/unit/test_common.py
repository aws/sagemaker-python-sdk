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

import numpy as np
import tempfile
import pytest
import itertools
from scipy.sparse import coo_matrix
from sagemaker.amazon.common import (
    RecordDeserializer,
    write_numpy_to_dense_tensor,
    read_recordio,
    RecordSerializer,
    write_spmatrix_to_sparse_tensor,
)
from sagemaker.amazon.record_pb2 import Record


def test_serializer():
    s = RecordSerializer()
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    buf = s.serialize(np.array(array_data))
    for record_data, expected in zip(read_recordio(buf), array_data):
        record = Record()
        record.ParseFromString(record_data)
        assert record.features["values"].float64_tensor.values == expected


def test_serializer_accepts_one_dimensional_array():
    s = RecordSerializer()
    array_data = [1.0, 2.0, 3.0]
    buf = s.serialize(np.array(array_data))
    record_data = next(read_recordio(buf))
    record = Record()
    record.ParseFromString(record_data)
    assert record.features["values"].float64_tensor.values == array_data


def test_deserializer():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    s = RecordSerializer()
    buf = s.serialize(np.array(array_data))
    d = RecordDeserializer()
    for record, expected in zip(d.deserialize(buf, "who cares"), array_data):
        assert record.features["values"].float64_tensor.values == expected


def test_float_write_numpy_to_dense_tensor():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    array = np.array(array_data)
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array)
        f.seek(0)
        for record_data, expected in zip(read_recordio(f), array_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float64_tensor.values == expected


def test_float32_write_numpy_to_dense_tensor():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    array = np.array(array_data).astype(np.dtype("float32"))
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array)
        f.seek(0)
        for record_data, expected in zip(read_recordio(f), array_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float32_tensor.values == expected


def test_int_write_numpy_to_dense_tensor():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array)
        f.seek(0)
        for record_data, expected in zip(read_recordio(f), array_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected


def test_int_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97])
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected, label in zip(read_recordio(f), array_data, label_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected
            assert record.label["values"].int32_tensor.values == [label]


def test_float32_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97]).astype(np.dtype("float32"))
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected, label in zip(read_recordio(f), array_data, label_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected
            assert record.label["values"].float32_tensor.values == [label]


def test_float_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97]).astype(np.dtype("float64"))
    with tempfile.TemporaryFile() as f:
        write_numpy_to_dense_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected, label in zip(read_recordio(f), array_data, label_data):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected
            assert record.label["values"].float64_tensor.values == [label]


def test_invalid_array():
    array_data = [[[1, 2, 3], [10, 20, 3]], [[1, 2, 3], [10, 20, 3]]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97]).astype(np.dtype("float64"))
    with tempfile.TemporaryFile() as f:
        with pytest.raises(ValueError):
            write_numpy_to_dense_tensor(f, array, label_data)


def test_invalid_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97, 1000]).astype(np.dtype("float64"))
    with tempfile.TemporaryFile() as f:
        with pytest.raises(ValueError):
            write_numpy_to_dense_tensor(f, array, label_data)


def test_dense_float_write_spmatrix_to_sparse_tensor():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data))
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float64_tensor.values == expected_data
            assert record.features["values"].float64_tensor.keys == expected_keys
            assert record.features["values"].float64_tensor.shape == [len(expected_data)]


def test_dense_float32_write_spmatrix_to_sparse_tensor():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data).astype(np.dtype("float32")))
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float32_tensor.values == expected_data
            assert record.features["values"].float32_tensor.keys == expected_keys
            assert record.features["values"].float32_tensor.shape == [len(expected_data)]


def test_dense_int_write_spmatrix_to_sparse_tensor():
    array_data = [[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data).astype(np.dtype("int")))
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected_data
            assert record.features["values"].int32_tensor.keys == expected_keys
            assert record.features["values"].int32_tensor.shape == [len(expected_data)]


def test_dense_int_spmatrix_to_sparse_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data))
    label_data = np.array([99, 98, 97])
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected_data, expected_keys, label in zip(
            read_recordio(f), array_data, keys_data, label_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected_data
            assert record.features["values"].int32_tensor.keys == expected_keys
            assert record.label["values"].int32_tensor.values == [label]
            assert record.features["values"].int32_tensor.shape == [len(expected_data)]


def test_dense_float32_spmatrix_to_sparse_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data).astype("float32"))
    label_data = np.array([99, 98, 97])
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected_data, expected_keys, label in zip(
            read_recordio(f), array_data, keys_data, label_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float32_tensor.values == expected_data
            assert record.features["values"].float32_tensor.keys == expected_keys
            assert record.label["values"].int32_tensor.values == [label]
            assert record.features["values"].float32_tensor.shape == [len(expected_data)]


def test_dense_float64_spmatrix_to_sparse_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    keys_data = [[0, 1, 2], [0, 1, 2]]
    array = coo_matrix(np.array(array_data).astype("float64"))
    label_data = np.array([99, 98, 97])
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array, label_data)
        f.seek(0)
        for record_data, expected_data, expected_keys, label in zip(
            read_recordio(f), array_data, keys_data, label_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float64_tensor.values == expected_data
            assert record.features["values"].float64_tensor.keys == expected_keys
            assert record.label["values"].int32_tensor.values == [label]
            assert record.features["values"].float64_tensor.shape == [len(expected_data)]


def test_invalid_sparse_label():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = coo_matrix(np.array(array_data))
    label_data = np.array([99, 98, 97, 1000]).astype(np.dtype("float64"))
    with tempfile.TemporaryFile() as f:
        with pytest.raises(ValueError):
            write_spmatrix_to_sparse_tensor(f, array, label_data)


def test_sparse_float_write_spmatrix_to_sparse_tensor():
    n = 4
    array_data = [[1.0, 2.0], [10.0, 30.0], [100.0, 200.0, 300.0, 400.0], [1000.0, 2000.0, 3000.0]]
    keys_data = [[0, 1], [1, 2], [0, 1, 2, 3], [0, 2, 3]]

    flatten_data = list(itertools.chain.from_iterable(array_data))
    y_indices = list(itertools.chain.from_iterable(keys_data))
    x_indices = [[i] * len(keys_data[i]) for i in range(len(keys_data))]
    x_indices = list(itertools.chain.from_iterable(x_indices))

    array = coo_matrix((flatten_data, (x_indices, y_indices)), dtype="float64")
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float64_tensor.values == expected_data
            assert record.features["values"].float64_tensor.keys == expected_keys
            assert record.features["values"].float64_tensor.shape == [n]


def test_sparse_float32_write_spmatrix_to_sparse_tensor():
    n = 4
    array_data = [[1.0, 2.0], [10.0, 30.0], [100.0, 200.0, 300.0, 400.0], [1000.0, 2000.0, 3000.0]]
    keys_data = [[0, 1], [1, 2], [0, 1, 2, 3], [0, 2, 3]]

    flatten_data = list(itertools.chain.from_iterable(array_data))
    y_indices = list(itertools.chain.from_iterable(keys_data))
    x_indices = [[i] * len(keys_data[i]) for i in range(len(keys_data))]
    x_indices = list(itertools.chain.from_iterable(x_indices))

    array = coo_matrix((flatten_data, (x_indices, y_indices)), dtype="float32")
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].float32_tensor.values == expected_data
            assert record.features["values"].float32_tensor.keys == expected_keys
            assert record.features["values"].float32_tensor.shape == [n]


def test_sparse_int_write_spmatrix_to_sparse_tensor():
    n = 4
    array_data = [[1.0, 2.0], [10.0, 30.0], [100.0, 200.0, 300.0, 400.0], [1000.0, 2000.0, 3000.0]]
    keys_data = [[0, 1], [1, 2], [0, 1, 2, 3], [0, 2, 3]]

    flatten_data = list(itertools.chain.from_iterable(array_data))
    y_indices = list(itertools.chain.from_iterable(keys_data))
    x_indices = [[i] * len(keys_data[i]) for i in range(len(keys_data))]
    x_indices = list(itertools.chain.from_iterable(x_indices))

    array = coo_matrix((flatten_data, (x_indices, y_indices)), dtype="int")
    with tempfile.TemporaryFile() as f:
        write_spmatrix_to_sparse_tensor(f, array)
        f.seek(0)
        for record_data, expected_data, expected_keys in zip(
            read_recordio(f), array_data, keys_data
        ):
            record = Record()
            record.ParseFromString(record_data)
            assert record.features["values"].int32_tensor.values == expected_data
            assert record.features["values"].int32_tensor.keys == expected_keys
            assert record.features["values"].int32_tensor.shape == [n]


def test_dense_to_sparse():
    array_data = [[1, 2, 3], [10, 20, 3]]
    array = np.array(array_data)
    label_data = np.array([99, 98, 97]).astype(np.dtype("float64"))
    with tempfile.TemporaryFile() as f:
        with pytest.raises(TypeError):
            write_spmatrix_to_sparse_tensor(f, array, label_data)
