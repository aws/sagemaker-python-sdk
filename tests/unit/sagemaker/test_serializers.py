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

import os

import numpy as np
import pytest

from sagemaker.serializers import CSVSerializer
from tests.unit import DATA_DIR


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
