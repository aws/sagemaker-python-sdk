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
"""Unit tests for sagemaker.train.utils – specifically safe_serialize."""
from __future__ import absolute_import

import json

from sagemaker.train.utils import safe_serialize
from sagemaker.core.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)


# ---------------------------------------------------------------------------
# PipelineVariable inputs – should be returned as-is (identity)
# ---------------------------------------------------------------------------

def test_safe_serialize_with_pipeline_variable_integer_returns_object_directly():
    param = ParameterInteger(name="MaxDepth", default_value=5)
    result = safe_serialize(param)
    assert result is param


def test_safe_serialize_with_pipeline_variable_string_returns_object_directly():
    param = ParameterString(name="Optimizer", default_value="sgd")
    result = safe_serialize(param)
    assert result is param


def test_safe_serialize_with_pipeline_variable_float_returns_object_directly():
    param = ParameterFloat(name="LearningRate", default_value=0.01)
    result = safe_serialize(param)
    assert result is param


# ---------------------------------------------------------------------------
# Regular / primitive inputs
# ---------------------------------------------------------------------------

def test_safe_serialize_with_string_returns_string_as_is():
    assert safe_serialize("hello") == "hello"
    assert safe_serialize("12345") == "12345"


def test_safe_serialize_with_int_returns_json_string():
    assert safe_serialize(5) == "5"
    assert safe_serialize(0) == "0"


def test_safe_serialize_with_dict_returns_json_string():
    data = {"key": "value", "num": 1}
    assert safe_serialize(data) == json.dumps(data)


def test_safe_serialize_with_bool_returns_json_string():
    assert safe_serialize(True) == "true"
    assert safe_serialize(False) == "false"


def test_safe_serialize_with_custom_object_returns_str():
    class CustomObject:
        def __str__(self):
            return "CustomObject"

    obj = CustomObject()
    assert safe_serialize(obj) == "CustomObject"


def test_safe_serialize_with_none_returns_json_null():
    assert safe_serialize(None) == "null"


def test_safe_serialize_with_list_returns_json_string():
    assert safe_serialize([1, 2, 3]) == "[1, 2, 3]"


def test_safe_serialize_with_empty_string():
    assert safe_serialize("") == ""
