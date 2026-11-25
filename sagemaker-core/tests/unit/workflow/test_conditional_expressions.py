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
"""Unit tests for argument-level conditional expressions."""

import pytest

from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.conditional_expressions import (
    ElseIf,
    If,
    Omit,
)
from sagemaker.core.workflow.conditions import ConditionEquals
from sagemaker.core.workflow.parameters import ParameterBoolean, ParameterString


class _TestVariable(PipelineVariable):
    def __init__(self, expression, referenced_steps):
        self._expression = expression
        self._steps = referenced_steps

    @property
    def expr(self):
        return self._expression

    @property
    def _referenced_steps(self):
        return self._steps


def test_omit_as_branch_serializes_to_std_omit():
    expression = If(condition=ConditionEquals(left=True, right=True), then="a", else_=Omit())

    assert expression.expr["Std:If"]["Else"] == {"Std:Omit": True}
    assert Omit()._referenced_steps == []


@pytest.mark.parametrize(
    "value",
    [Omit(), {"Key": Omit()}, [Omit()]],
    ids=["bare", "inside-map-in-branch", "inside-list-in-branch"],
)
def test_omit_outside_a_complete_branch_is_rejected(value):
    if not isinstance(value, Omit):
        value = If(condition=ConditionEquals(left=True, right=True), then=value, else_="b")

    with pytest.raises(ValueError, match="complete then or else_ branch"):
        value.expr


def test_if_depth_two_is_accepted():
    cond = ConditionEquals(left=True, right=True)

    If(cond, then=If(cond, then="a", else_="b"), else_=Omit())


@pytest.mark.parametrize(
    "build",
    [
        lambda c, inner: If(c, then=If(c, then=inner, else_="d"), else_="e"),
        lambda c, inner: If(
            c, then="a", else_if=[ElseIf(c, then=If(c, then=inner, else_="d"))], else_="e"
        ),
        lambda c, inner: If(c, then={"Nested": [If(c, then=inner, else_="d")]}, else_="e"),
    ],
    ids=["then", "else-if", "inside-structure"],
)
def test_if_depth_three_is_rejected(build):
    cond = ConditionEquals(left=True, right=True)
    inner = If(cond, then="a", else_="b")

    with pytest.raises(ValueError, match="depth cannot exceed 2"):
        build(cond, inner)


def test_if_expression_with_literals():
    expression = If(
        condition=ConditionEquals(left=True, right=True),
        then="selected",
        else_="fallback",
    )

    assert expression.expr == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": True,
                "RightValue": True,
            },
            "Then": "selected",
            "Else": "fallback",
        }
    }


def test_if_expression_serializes_parameters_in_condition_and_branches():
    enabled = ParameterBoolean("Enabled")
    primary = ParameterString("Primary")
    fallback = ParameterString("Fallback")
    expression = If(
        condition=ConditionEquals(left=enabled, right=True),
        then=primary,
        else_=fallback,
    )

    assert expression.expr == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": {"Get": "Parameters.Enabled"},
                "RightValue": True,
            },
            "Then": {"Get": "Parameters.Primary"},
            "Else": {"Get": "Parameters.Fallback"},
        }
    }


def test_nested_if_expression_is_canonical_else_if():
    mode = ParameterString("Mode")
    expression = If(
        condition=ConditionEquals(left=mode, right="primary"),
        then={"Subnets": ["subnet-primary"]},
        else_=If(
            condition=ConditionEquals(left=mode, right="fallback"),
            then={"Subnets": ["subnet-fallback"]},
            else_=Omit(),
        ),
    )

    result = expression.expr
    assert "ElseIf" not in str(result)
    assert result["Std:If"]["Else"] == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": {"Get": "Parameters.Mode"},
                "RightValue": "fallback",
            },
            "Then": {"Subnets": ["subnet-fallback"]},
            "Else": {"Std:Omit": True},
        }
    }


def test_if_expression_recursively_serializes_structures_lists_and_maps():
    value = ParameterString("Value")
    expression = If(
        condition=ConditionEquals(left=True, right=True),
        then={"Items": [value, {"Nested": value}]},
        else_=Omit(),
    )

    assert expression.expr["Std:If"]["Then"] == {
        "Items": [
            {"Get": "Parameters.Value"},
            {"Nested": {"Get": "Parameters.Value"}},
        ]
    }


def test_if_expression_collects_references_from_condition_and_both_branches():
    condition_value = _TestVariable({"Get": "Steps.Condition.Value"}, ["Condition"])
    then = _TestVariable({"Get": "Steps.Then.Value"}, ["Then"])
    else_ = _TestVariable({"Get": "Steps.Else.Value"}, ["Else"])
    expression = If(
        condition=ConditionEquals(left=condition_value, right=True),
        then={"Nested": [then]},
        else_=else_,
    )

    assert expression._referenced_steps == ["Condition", "Then", "Else"]


def test_if_expression_rejects_non_condition():
    with pytest.raises(TypeError):
        If(condition=True, then="yes", else_="no")


def test_omit_to_string_stays_omit():
    assert isinstance(Omit().to_string(), Omit)


def test_if_expression_to_string_stringifies_branches_and_keeps_if_outermost():
    number = _TestVariable({"Get": "Steps.Train.Number"}, ["Train"])
    expression = If(
        condition=ConditionEquals(left=True, right=True),
        then=number,
        else_if=[ElseIf(condition=ConditionEquals(left=1, right=1), then=5)],
        else_=Omit(),
    )

    result = expression.to_string().expr["Std:If"]

    assert result["Then"] == {"Std:Join": {"On": "", "Values": [{"Get": "Steps.Train.Number"}]}}
    assert result["ElseIf"][0]["Then"] == "5"
    assert result["Else"] == {"Std:Omit": True}
    assert result["Condition"] == expression.expr["Std:If"]["Condition"]
