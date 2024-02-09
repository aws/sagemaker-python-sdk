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

from sagemaker.workflow.conditions import (
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
    ConditionIn,
    ConditionNot,
    ConditionOr,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
)
from sagemaker.workflow.properties import Properties


def test_condition_equals():
    param = ParameterInteger(name="MyInt")
    cond = ConditionEquals(left=param, right=1)
    assert cond.to_request() == {
        "Type": "Equals",
        "LeftValue": param,
        "RightValue": 1,
    }


def test_condition_equals_parameter():
    param1 = ParameterInteger(name="MyInt1")
    param2 = ParameterInteger(name="MyInt2")
    cond = ConditionEquals(left=param1, right=param2)
    assert cond.to_request() == {
        "Type": "Equals",
        "LeftValue": param1,
        "RightValue": param2,
    }


def test_condition_greater_than():
    var = ExecutionVariables.START_DATETIME
    cond = ConditionGreaterThan(left=var, right="2020-12-01")
    assert cond.to_request() == {
        "Type": "GreaterThan",
        "LeftValue": var,
        "RightValue": "2020-12-01",
    }


def test_condition_greater_than_or_equal_to():
    var = ExecutionVariables.START_DATETIME
    param = ParameterString(name="StartDateTime")
    cond = ConditionGreaterThanOrEqualTo(left=var, right=param)
    assert cond.to_request() == {
        "Type": "GreaterThanOrEqualTo",
        "LeftValue": var,
        "RightValue": param,
    }


def test_condition_less_than():
    var = ExecutionVariables.START_DATETIME
    cond = ConditionLessThan(left=var, right="2020-12-01")
    assert cond.to_request() == {
        "Type": "LessThan",
        "LeftValue": var,
        "RightValue": "2020-12-01",
    }


def test_condition_less_than_or_equal_to():
    var = ExecutionVariables.START_DATETIME
    param = ParameterString(name="StartDateTime")
    cond = ConditionLessThanOrEqualTo(left=var, right=param)
    assert cond.to_request() == {
        "Type": "LessThanOrEqualTo",
        "LeftValue": var,
        "RightValue": param,
    }


def test_condition_in():
    param = ParameterString(name="MyStr")
    cond_in = ConditionIn(value=param, in_values=["abc", "def"])
    assert cond_in.to_request() == {
        "Type": "In",
        "QueryValue": param,
        "Values": ["abc", "def"],
    }


def test_condition_in_mixed():
    param = ParameterString(name="MyStr")
    prop = Properties("foo")
    var = ExecutionVariables.START_DATETIME
    cond_in = ConditionIn(value=param, in_values=["abc", prop, var])
    assert cond_in.to_request() == {
        "Type": "In",
        "QueryValue": param,
        "Values": ["abc", prop, var],
    }


def test_condition_not():
    param = ParameterString(name="MyStr")
    cond_eq = ConditionEquals(left=param, right="foo")
    cond_not = ConditionNot(expression=cond_eq)
    assert cond_not.to_request() == {
        "Type": "Not",
        "Condition": {
            "Type": "Equals",
            "LeftValue": param,
            "RightValue": "foo",
        },
    }


def test_condition_not_in():
    param = ParameterString(name="MyStr")
    cond_in = ConditionIn(value=param, in_values=["abc", "def"])
    cond_not = ConditionNot(expression=cond_in)
    assert cond_not.to_request() == {
        "Type": "Not",
        "Condition": {
            "Type": "In",
            "QueryValue": param,
            "Values": ["abc", "def"],
        },
    }


def test_condition_or():
    var = ExecutionVariables.START_DATETIME
    cond = ConditionGreaterThan(left=var, right="2020-12-01")
    param = ParameterString(name="MyStr")
    cond_in = ConditionIn(value=param, in_values=["abc", "def"])
    cond_or = ConditionOr(conditions=[cond, cond_in])
    assert cond_or.to_request() == {
        "Type": "Or",
        "Conditions": [
            {
                "Type": "GreaterThan",
                "LeftValue": var,
                "RightValue": "2020-12-01",
            },
            {
                "Type": "In",
                "QueryValue": param,
                "Values": ["abc", "def"],
            },
        ],
    }


def test_left_and_right_primitives():
    cond = ConditionEquals(left=2, right=1)
    assert cond.to_request() == {
        "Type": "Equals",
        "LeftValue": 2,
        "RightValue": 1,
    }
