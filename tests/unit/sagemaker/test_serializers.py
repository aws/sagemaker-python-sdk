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

import json
import os

import numpy as np
import pytest

from sagemaker.serializers import JSONSerializer
from tests.unit import DATA_DIR


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
