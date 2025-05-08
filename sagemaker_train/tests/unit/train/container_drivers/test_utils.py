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
"""Container Utils Unit Tests."""
from __future__ import absolute_import
import os

from sagemaker.train.container_drivers.common.utils import (
    safe_deserialize,
    safe_serialize,
    hyperparameters_to_cli_args,
    get_process_count,
)

SM_HPS = {
    "boolean": "true",
    "dict": '{"string":"value","integer":3,"list":[1,2,3],"dict":{"key":"value"},"boolean":true}',
    "float": "3.14",
    "integer": "1",
    "list": "[1,2,3]",
    "string": "Hello World",
}


def test_hyperparameters_to_cli_args():
    args = hyperparameters_to_cli_args(SM_HPS)

    assert args == [
        "--boolean",
        "true",
        "--dict",
        '{"string": "value", "integer": 3, "list": [1, 2, 3], "dict": {"key": "value"}, "boolean": true}',
        "--float",
        "3.14",
        "--integer",
        "1",
        "--list",
        "[1, 2, 3]",
        "--string",
        "Hello World",
    ]


def test_safe_deserialize_not_a_string():
    assert safe_deserialize(123) == 123
    assert safe_deserialize([1, 2, 3]) == [1, 2, 3]
    assert safe_deserialize({"key": "value"}) == {"key": "value"}


def test_safe_deserialize_boolean_strings():
    assert safe_deserialize("true") is True
    assert safe_deserialize("false") is False
    assert safe_deserialize("True") is True
    assert safe_deserialize("False") is False


def test_safe_deserialize_valid_json_string():
    json_data = '{"key": "value", "number": 123, "boolean": true}'
    expected_output = {"key": "value", "number": 123, "boolean": True}
    assert safe_deserialize(json_data) == expected_output

    assert safe_deserialize("Hello World") == "Hello World"
    assert safe_deserialize("12345") == 12345

    assert safe_deserialize("3.14") == 3.14
    assert safe_deserialize("[1,2,3]") == [1, 2, 3]


def test_safe_deserialize_invalid_json_string():
    invalid_json = '{"key": value}'  # Missing quotes around value so not valid json
    assert safe_deserialize(invalid_json) == invalid_json


def test_safe_deserialize_null_string():
    assert safe_deserialize("null") == None  # noqa: E711
    assert safe_deserialize("None") == "None"


def test_safe_serialize_string():
    assert safe_serialize("Hello World") == "Hello World"
    assert safe_serialize("12345") == "12345"
    assert safe_serialize("true") == "true"


def test_safe_serialize_serializable_data():
    assert safe_serialize({"key": "value", "number": 123, "boolean": True}) == (
        '{"key": "value", "number": 123, "boolean": true}'
    )
    assert safe_serialize([1, 2, 3]) == "[1, 2, 3]"
    assert safe_serialize(123) == "123"
    assert safe_serialize(3.14) == "3.14"
    assert safe_serialize(True) == "true"
    assert safe_serialize(False) == "false"
    assert safe_serialize(None) == "null"


def test_safe_serialize_custom_object():
    class CustomObject:
        def __str__(self):
            return "CustomObject"

    obj = CustomObject()
    assert safe_serialize(obj) == "CustomObject"


def test_safe_serialize_invalid_data():
    invalid_data = {"key": set([1, 2, 3])}  # Sets are not JSON serializable
    assert safe_serialize(invalid_data) == str(invalid_data)


def test_safe_serialize_empty_data():
    assert safe_serialize("") == ""
    assert safe_serialize([]) == "[]"
    assert safe_serialize({}) == "{}"


def test_get_process_count():
    assert get_process_count() == 1
    assert get_process_count(2) == 2
    os.environ["SM_NUM_GPUS"] = "4"
    assert get_process_count() == 4
    os.environ["SM_NUM_GPUS"] = "0"
    os.environ["SM_NUM_NEURONS"] = "8"
    assert get_process_count() == 8
    os.environ["SM_NUM_NEURONS"] = "0"
    assert get_process_count() == 1
    del os.environ["SM_NUM_GPUS"]
    del os.environ["SM_NUM_NEURONS"]
    assert get_process_count() == 1
