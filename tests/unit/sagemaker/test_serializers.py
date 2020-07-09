# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import numpy as np
import pytest

from sagemaker.serializers import NumpySerializer


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
