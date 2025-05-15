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

import io
import json

import numpy as np
import pandas as pd
import pytest

from sagemaker.utils.base_deserializers import (
    StringDeserializer,
    BytesDeserializer,
    CSVDeserializer,
    StreamDeserializer,
    NumpyDeserializer,
    JSONDeserializer,
    PandasDeserializer,
    JSONLinesDeserializer,
)


def test_string_deserializer():
    deserializer = StringDeserializer()

    result = deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == "[1, 2, 3]"


def test_bytes_deserializer():
    deserializer = BytesDeserializer()

    result = deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == b"[1, 2, 3]"


@pytest.fixture
def csv_deserializer():
    return CSVDeserializer()


def test_csv_deserializer_single_element(csv_deserializer):
    result = csv_deserializer.deserialize(io.BytesIO(b"1"), "text/csv")
    assert result == [["1"]]


def test_csv_deserializer_array(csv_deserializer):
    result = csv_deserializer.deserialize(io.BytesIO(b"1,2,3"), "text/csv")
    assert result == [["1", "2", "3"]]


def test_csv_deserializer_2dimensional(csv_deserializer):
    result = csv_deserializer.deserialize(io.BytesIO(b"1,2,3\n3,4,5"), "text/csv")
    assert result == [["1", "2", "3"], ["3", "4", "5"]]


def test_csv_deserializer_posix_compliant(csv_deserializer):
    result = csv_deserializer.deserialize(io.BytesIO(b"1,2,3\n3,4,5\n"), "text/csv")
    assert result == [["1", "2", "3"], ["3", "4", "5"]]


def test_stream_deserializer():
    deserializer = StreamDeserializer()

    stream, content_type = deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")
    try:
        result = stream.read()
    finally:
        stream.close()

    assert result == b"[1, 2, 3]"
    assert content_type == "application/json"


@pytest.fixture
def numpy_deserializer():
    return NumpyDeserializer()


def test_numpy_deserializer_from_csv(numpy_deserializer):
    stream = io.BytesIO(b"1,2,3\n4,5,6")
    array = numpy_deserializer.deserialize(stream, "text/csv")
    assert np.array_equal(array, np.array([[1, 2, 3], [4, 5, 6]]))


def test_numpy_deserializer_from_csv_ragged(numpy_deserializer):
    stream = io.BytesIO(b"1,2,3\n4,5,6,7")
    with pytest.raises(ValueError) as error:
        numpy_deserializer.deserialize(stream, "text/csv")
    assert "errors were detected" in str(error)


def test_numpy_deserializer_from_csv_alpha():
    numpy_deserializer = NumpyDeserializer(dtype="U5")
    stream = io.BytesIO(b"hello,2,3\n4,5,6")
    array = numpy_deserializer.deserialize(stream, "text/csv")
    assert np.array_equal(array, np.array([["hello", 2, 3], [4, 5, 6]]))


def test_numpy_deserializer_from_json(numpy_deserializer):
    stream = io.BytesIO(b"[[1,2,3],\n[4,5,6]]")
    array = numpy_deserializer.deserialize(stream, "application/json")
    assert np.array_equal(array, np.array([[1, 2, 3], [4, 5, 6]]))


# Sadly, ragged arrays work fine in JSON (giving us a 1D array of Python lists)
def test_numpy_deserializer_from_json_ragged(numpy_deserializer):
    stream = io.BytesIO(b"[[1,2,3],\n[4,5,6,7]]")
    with pytest.raises(ValueError) as error:
        numpy_deserializer.deserialize(stream, "application/json")
    assert "requested array has an inhomogeneous shape" in str(error)


def test_numpy_deserializer_from_json_alpha():
    numpy_deserializer = NumpyDeserializer(dtype="U5")
    stream = io.BytesIO(b'[["hello",2,3],\n[4,5,6]]')
    array = numpy_deserializer.deserialize(stream, "application/json")
    assert np.array_equal(array, np.array([["hello", 2, 3], [4, 5, 6]]))


def test_numpy_deserializer_from_npy(numpy_deserializer):
    array = np.ones((2, 3))
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = numpy_deserializer.deserialize(stream, "application/x-npy")

    assert np.array_equal(array, result)


def test_numpy_deserializer_from_npy_object_array():
    numpy_deserializer = NumpyDeserializer(allow_pickle=True)
    array = np.array([{"a": "", "b": ""}, {"c": "", "d": ""}])
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = numpy_deserializer.deserialize(stream, "application/x-npy")

    assert np.array_equal(array, result)


def test_numpy_deserializer_from_npy_object_array_with_allow_pickle_false():
    numpy_deserializer = NumpyDeserializer(allow_pickle=False)

    array = np.array([{"a": "", "b": ""}, {"c": "", "d": ""}])
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    with pytest.raises(ValueError):
        numpy_deserializer.deserialize(stream, "application/x-npy")


def test_numpy_deserializer_from_npz(numpy_deserializer):
    arrays = {
        "alpha": np.ones((2, 3)),
        "beta": np.zeros((3, 2)),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    stream.seek(0)

    result = numpy_deserializer.deserialize(stream, "application/x-npz")

    assert isinstance(result, np.lib.npyio.NpzFile)
    assert set(arrays.keys()) == set(result.keys())
    for key, arr in arrays.items():
        assert np.array_equal(arr, result[key])


@pytest.fixture
def json_deserializer():
    return JSONDeserializer()


def test_json_deserializer_array(json_deserializer):
    result = json_deserializer.deserialize(io.BytesIO(b"[1, 2, 3]"), "application/json")

    assert result == [1, 2, 3]


def test_json_deserializer_2dimensional(json_deserializer):
    result = json_deserializer.deserialize(
        io.BytesIO(b"[[1, 2, 3], [3, 4, 5]]"), "application/json"
    )

    assert result == [[1, 2, 3], [3, 4, 5]]


def test_json_deserializer_invalid_data(json_deserializer):
    with pytest.raises(ValueError) as error:
        json_deserializer.deserialize(io.BytesIO(b"[[1]"), "application/json")
    assert "column" in str(error)


@pytest.fixture
def pandas_deserializer():
    return PandasDeserializer()


def test_pandas_deserializer_json(pandas_deserializer):
    data = {"col 1": {"row 1": "a", "row 2": "c"}, "col 2": {"row 1": "b", "row 2": "d"}}
    stream = io.BytesIO(json.dumps(data).encode("utf-8"))
    result = pandas_deserializer.deserialize(stream, "application/json")
    expected = pd.DataFrame(
        [["a", "b"], ["c", "d"]], index=["row 1", "row 2"], columns=["col 1", "col 2"]
    )
    assert result.equals(expected)


def test_pandas_deserializer_csv(pandas_deserializer):
    stream = io.BytesIO(b"col 1,col 2\na,b\nc,d")
    result = pandas_deserializer.deserialize(stream, "text/csv")
    expected = pd.DataFrame([["a", "b"], ["c", "d"]], columns=["col 1", "col 2"])
    assert result.equals(expected)


@pytest.fixture
def json_lines_deserializer():
    return JSONLinesDeserializer()


@pytest.mark.parametrize(
    "source, expected",
    [
        (b'["Name", "Score"]\n["Gilbert", 24]', [["Name", "Score"], ["Gilbert", 24]]),
        (b'["Name", "Score"]\n["Gilbert", 24]\n', [["Name", "Score"], ["Gilbert", 24]]),
        (
            b'{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
            [{"Name": "Gilbert", "Score": 24}, {"Name": "Alexa", "Score": 29}],
        ),
    ],
)
def test_json_lines_deserializer(json_lines_deserializer, source, expected):
    stream = io.BytesIO(source)
    content_type = "application/jsonlines"
    actual = json_lines_deserializer.deserialize(stream, content_type)
    assert actual == expected
