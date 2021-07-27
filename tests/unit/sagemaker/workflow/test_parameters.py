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

from urllib.parse import urlparse

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


def test_parameter_with_default_value_zero():
    param = ParameterInteger(name="MyInteger", default_value=0)
    assert param.to_request() == {"Name": "MyInteger", "Type": "Integer", "DefaultValue": 0}


def test_parameter_string_with_enum_values():
    param = ParameterString("MyString", enum_values=["a", "b"])
    assert param.to_request() == {"Name": "MyString", "Type": "String", "EnumValues": ["a", "b"]}
    param = ParameterString("MyString", default_value="a", enum_values=["a", "b"])
    assert param.to_request() == {
        "Name": "MyString",
        "Type": "String",
        "DefaultValue": "a",
        "EnumValues": ["a", "b"],
    }


def test_parameter_with_invalid_default():
    with pytest.raises(TypeError):
        ParameterFloat(name="MyFloat", default_value="abc")


def test_parameter_string_implicit_value():
    param = ParameterString("MyString")
    assert param.__str__() == ""
    param1 = ParameterString("MyString", "1")
    assert param1.__str__() == "1"
    param2 = ParameterString("MyString", default_value="2")
    assert param2.__str__() == "2"
    param3 = ParameterString(name="MyString", default_value="3")
    assert param3.__str__() == "3"
    param3 = ParameterString(name="MyString", default_value="3", enum_values=["3"])
    assert param3.__str__() == "3"


def test_parameter_integer_implicit_value():
    param = ParameterInteger("MyInteger")
    assert param.__int__() == 0
    param1 = ParameterInteger("MyInteger", 1)
    assert param1.__int__() == 1
    param2 = ParameterInteger("MyInteger", default_value=2)
    assert param2.__int__() == 2
    param3 = ParameterInteger(name="MyInteger", default_value=3)
    assert param3.__int__() == 3


def test_parameter_float_implicit_value():
    param = ParameterFloat("MyFloat")
    assert param.__float__() == 0.0
    param1 = ParameterFloat("MyFloat", 1.1)
    assert param1.__float__() == 1.1
    param2 = ParameterFloat("MyFloat", default_value=2.1)
    assert param2.__float__() == 2.1
    param3 = ParameterFloat(name="MyFloat", default_value=3.1)
    assert param3.__float__() == 3.1


def test_parsable_parameter_string():
    param = ParameterString("MyString", default_value="s3://foo/bar/baz.csv")
    assert urlparse(param).scheme == "s3"
