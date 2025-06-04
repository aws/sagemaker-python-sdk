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
import os

import numpy as np
import pytest
import scipy.sparse

from sagemaker.utils.base_serializers import (
    CSVSerializer,
    NumpySerializer,
    JSONSerializer,
    IdentitySerializer,
    SparseMatrixSerializer,
    JSONLinesSerializer,
    LibSVMSerializer,
    DataSerializer,
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@pytest.fixture
def csv_serializer():
    return CSVSerializer()


def test_csv_serializer_str(csv_serializer):
    original = "1,2,3"
    result = csv_serializer.serialize("1,2,3")

    assert result == original


def test_csv_serializer_python_array(csv_serializer):
    result = csv_serializer.serialize([1, 2, 3])

    assert result == "1,2,3"


def test_csv_serializer_numpy_valid(csv_serializer):
    result = csv_serializer.serialize(np.array([1, 2, 3]))

    assert result == "1,2,3"


def test_csv_serializer_numpy_valid_2dimensional(csv_serializer):
    result = csv_serializer.serialize(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == "1,2,3\n3,4,5"


def test_csv_serializer_list_of_str(csv_serializer):
    result = csv_serializer.serialize(["1,2,3", "4,5,6"])

    assert result == "1,2,3\n4,5,6"


def test_csv_serializer_list_of_list(csv_serializer):
    result = csv_serializer.serialize([[1, 2, 3], [3, 4, 5]])

    assert result == "1,2,3\n3,4,5"


def test_csv_serializer_list_of_empty(csv_serializer):
    with pytest.raises(ValueError) as invalid_input:
        csv_serializer.serialize(np.array([[], []]))

    assert "empty array" in str(invalid_input)


def test_csv_serializer_numpy_invalid_empty(csv_serializer):
    with pytest.raises(ValueError) as invalid_input:
        csv_serializer.serialize(np.array([]))

    assert "empty array" in str(invalid_input)


def test_csv_serializer_python_invalid_empty(csv_serializer):
    with pytest.raises(ValueError) as error:
        csv_serializer.serialize([])
    assert "empty array" in str(error)


def test_csv_serializer_csv_reader(csv_serializer):
    csv_file_path = os.path.join(DATA_DIR, "with_integers.csv")
    with open(csv_file_path) as csv_file:
        validation_data = csv_file.read()
        csv_file.seek(0)
        result = csv_serializer.serialize(csv_file)
        assert result == validation_data


@pytest.fixture
def numpy_serializer():
    return NumpySerializer()


def test_numpy_serializer_python_array(numpy_serializer):
    array = [1, 2, 3]
    result = numpy_serializer.serialize(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_numpy_serializer_python_array_with_dtype():
    numpy_serializer = NumpySerializer(dtype="float16")
    array = [1, 2, 3]

    result = numpy_serializer.serialize(array)

    deserialized = np.load(io.BytesIO(result))
    assert np.array_equal(array, deserialized)
    assert deserialized.dtype == "float16"


def test_numpy_serializer_numpy_valid_2_dimensional(numpy_serializer):
    array = np.array([[1, 2, 3], [3, 4, 5]])
    result = numpy_serializer.serialize(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_numpy_serializer_numpy_valid_multidimensional(numpy_serializer):
    array = np.ones((10, 10, 10, 10))
    result = numpy_serializer.serialize(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_numpy_serializer_numpy_valid_list_of_strings(numpy_serializer):
    array = np.array(["one", "two", "three"])
    result = numpy_serializer.serialize(array)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_numpy_serializer_from_buffer_or_file(numpy_serializer):
    array = np.ones((2, 3))
    stream = io.BytesIO()
    np.save(stream, array)
    stream.seek(0)

    result = numpy_serializer.serialize(stream)

    assert np.array_equal(array, np.load(io.BytesIO(result)))


def test_numpy_serializer_object(numpy_serializer):
    object = {1, 2, 3}

    result = numpy_serializer.serialize(object)

    assert np.array_equal(np.array(object), np.load(io.BytesIO(result), allow_pickle=True))


def test_numpy_serializer_list_of_empty(numpy_serializer):
    with pytest.raises(ValueError) as invalid_input:
        numpy_serializer.serialize(np.array([[], []]))

    assert "empty array" in str(invalid_input)


def test_numpy_serializer_numpy_invalid_empty(numpy_serializer):
    with pytest.raises(ValueError) as invalid_input:
        numpy_serializer.serialize(np.array([]))

    assert "empty array" in str(invalid_input)


def test_numpy_serializer_python_invalid_empty(numpy_serializer):
    with pytest.raises(ValueError) as error:
        numpy_serializer.serialize([])
    assert "empty array" in str(error)


@pytest.fixture
def json_serializer():
    return JSONSerializer()


def test_json_serializer_numpy_valid(json_serializer):
    result = json_serializer.serialize(np.array([1, 2, 3]))

    assert result == "[1, 2, 3]"


def test_json_serializer_numpy_valid_2dimensional(json_serializer):
    result = json_serializer.serialize(np.array([[1, 2, 3], [3, 4, 5]]))

    assert result == "[[1, 2, 3], [3, 4, 5]]"


def test_json_serializer_empty(json_serializer):
    assert json_serializer.serialize(np.array([])) == "[]"


def test_json_serializer_python_array(json_serializer):
    result = json_serializer.serialize([1, 2, 3])

    assert result == "[1, 2, 3]"


def test_json_serializer_python_dictionary(json_serializer):
    d = {"gender": "m", "age": 22, "city": "Paris"}

    result = json_serializer.serialize(d)

    assert json.loads(result) == d


def test_json_serializer_python_invalid_empty(json_serializer):
    assert json_serializer.serialize([]) == "[]"


def test_json_serializer_python_dictionary_invalid_empty(json_serializer):
    assert json_serializer.serialize({}) == "{}"


def test_json_serializer_csv_buffer(json_serializer):
    csv_file_path = os.path.join(DATA_DIR, "with_integers.csv")
    with open(csv_file_path) as csv_file:
        validation_value = csv_file.read()
        csv_file.seek(0)
        result = json_serializer.serialize(csv_file)
        assert result == validation_value


def test_identity_serializer():
    identity_serializer = IdentitySerializer()
    assert identity_serializer.serialize(b"{}") == b"{}"


def test_identity_serializer_with_custom_content_type():
    identity_serializer = IdentitySerializer(content_type="text/csv")
    assert identity_serializer.serialize(b"a,b\n1,2") == b"a,b\n1,2"
    assert identity_serializer.CONTENT_TYPE == "text/csv"


@pytest.fixture
def json_lines_serializer():
    return JSONLinesSerializer()


@pytest.mark.parametrize(
    "input, expected",
    [
        ('["Name", "Score"]\n["Gilbert", 24]', '["Name", "Score"]\n["Gilbert", 24]'),
        (
            '{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
            '{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
        ),
    ],
)
def test_json_lines_serializer_string(json_lines_serializer, input, expected):
    actual = json_lines_serializer.serialize(input)
    assert actual == expected


@pytest.mark.parametrize(
    "input, expected",
    [
        ([["Name", "Score"], ["Gilbert", 24]], '["Name", "Score"]\n["Gilbert", 24]'),
        (
            [{"Name": "Gilbert", "Score": 24}, {"Name": "Alexa", "Score": 29}],
            '{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
        ),
    ],
)
def test_json_lines_serializer_list(json_lines_serializer, input, expected):
    actual = json_lines_serializer.serialize(input)
    assert actual == expected


@pytest.mark.parametrize(
    "source, expected",
    [
        ('["Name", "Score"]\n["Gilbert", 24]', '["Name", "Score"]\n["Gilbert", 24]'),
        (
            '{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
            '{"Name": "Gilbert", "Score": 24}\n{"Name": "Alexa", "Score": 29}',
        ),
    ],
)
def test_json_lines_serializer_file_like(json_lines_serializer, source, expected):
    input = io.StringIO(source)
    actual = json_lines_serializer.serialize(input)
    assert actual == expected


@pytest.fixture
def sparse_matrix_serializer():
    return SparseMatrixSerializer()


def test_sparse_matrix_serializer(sparse_matrix_serializer):
    data = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))
    stream = io.BytesIO(sparse_matrix_serializer.serialize(data))
    result = scipy.sparse.load_npz(stream).toarray()
    expected = data.toarray()
    assert np.array_equal(result, expected)


@pytest.fixture
def libsvm_serializer():
    return LibSVMSerializer()


def test_libsvm_serializer_str(libsvm_serializer):
    original = "0 0:1 5:1"
    result = libsvm_serializer.serialize("0 0:1 5:1")
    assert result == original


def test_libsvm_serializer_file_like(libsvm_serializer):
    libsvm_file_path = os.path.join(DATA_DIR, "xgboost_abalone", "abalone")
    with open(libsvm_file_path) as libsvm_file:
        validation_data = libsvm_file.read()
        libsvm_file.seek(0)
        result = libsvm_serializer.serialize(libsvm_file)
        assert result == validation_data


@pytest.fixture
def data_serializer():
    return DataSerializer()


def test_data_serializer_raw(data_serializer):
    input_image_file_path = os.path.join(DATA_DIR, "", "cuteCat.jpg")
    with open(input_image_file_path, "rb") as image:
        input_image = image.read()
    input_image_data = data_serializer.serialize(input_image)
    validation_image_file_path = os.path.join(DATA_DIR, "", "cuteCat.raw")
    with open(validation_image_file_path, "rb") as f:
        validation_image_data = f.read()
    assert input_image_data == validation_image_data


def test_data_serializer_file_like(data_serializer):
    input_image_file_path = os.path.join(DATA_DIR, "", "cuteCat.jpg")
    validation_image_file_path = os.path.join(DATA_DIR, "", "cuteCat.raw")
    input_image_data = data_serializer.serialize(input_image_file_path)
    with open(validation_image_file_path, "rb") as f:
        validation_image_data = f.read()
    assert input_image_data == validation_image_data
