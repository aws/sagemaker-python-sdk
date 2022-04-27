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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest

from sagemaker.workflow.parameters import (
    ParameterBoolean,
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)


def test_parameter():
    param = ParameterBoolean("MyBool")
    assert param.to_request() == {"Name": "MyBool", "Type": "Boolean"}
    assert param.expr == {"Get": "Parameters.MyBool"}
    assert param.parameter_type.python_type == bool


def test_parameter_with_default():
    param = ParameterFloat(name="MyFloat", default_value=1.2)
    assert param.to_request() == {"Name": "MyFloat", "Type": "Float", "DefaultValue": 1.2}
    assert param.expr == {"Get": "Parameters.MyFloat"}
    assert param.parameter_type.python_type == float


def test_parameter_with_default_value_zero():
    param = ParameterInteger(name="MyInteger", default_value=0)
    assert param.to_request() == {"Name": "MyInteger", "Type": "Integer", "DefaultValue": 0}
    assert param.expr == {"Get": "Parameters.MyInteger"}
    assert param.parameter_type.python_type == int


def test_parameter_string_with_enum_values():
    param = ParameterString("MyString", enum_values=["a", "b"])
    assert param.to_request() == {"Name": "MyString", "Type": "String", "EnumValues": ["a", "b"]}
    assert param.expr == {"Get": "Parameters.MyString"}
    assert param.parameter_type.python_type == str

    param = ParameterString("MyString", default_value="a", enum_values=["a", "b"])
    assert param.to_request() == {
        "Name": "MyString",
        "Type": "String",
        "DefaultValue": "a",
        "EnumValues": ["a", "b"],
    }
    assert param.expr == {"Get": "Parameters.MyString"}
    assert param.parameter_type.python_type == str


def test_parameter_with_invalid_default():
    with pytest.raises(TypeError):
        ParameterFloat(name="MyFloat", default_value="abc")


def test_parameter_to_string_and_string_implicit_value():
    param = ParameterString("MyString", "1")

    assert param.to_string() == param

    with pytest.raises(TypeError) as error:
        str(param)

    assert "Pipeline variables do not support __str__ operation." in str(error.value)


def test_parameter_integer_implicit_value():
    param = ParameterInteger("MyInteger", 1)

    with pytest.raises(TypeError) as error:
        int(param)

    assert str(error.value) == "Pipeline variables do not support __int__ operation."


def test_parameter_float_implicit_value():
    param = ParameterFloat("MyFloat", 1.1)

    with pytest.raises(TypeError) as error:
        float(param)

    assert str(error.value) == "Pipeline variables do not support __float__ operation."


def test_add_func():
    param_str = ParameterString(name="MyString", default_value="s3://foo/bar/baz.csv")
    param_int = ParameterInteger(name="MyInteger", default_value=3)
    param_float = ParameterFloat(name="MyFloat", default_value=1.5)
    param_bool = ParameterBoolean(name="MyBool")

    with pytest.raises(TypeError) as error:
        param_str + param_int
    assert str(error.value) == "Pipeline variables do not support concatenation."

    with pytest.raises(TypeError) as error:
        param_int + param_float
    assert str(error.value) == "Pipeline variables do not support concatenation."

    with pytest.raises(TypeError) as error:
        param_float + param_bool
    assert str(error.value) == "Pipeline variables do not support concatenation."

    with pytest.raises(TypeError) as error:
        param_bool + param_str
    assert str(error.value) == "Pipeline variables do not support concatenation."
