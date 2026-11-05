# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for argument-level conditional expressions."""

import pytest

from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.conditional_expressions import (
    ElseIfBranch,
    IfExpression,
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


def test_omit_expression():
    assert Omit().expr == {"Std:Omit": True}
    assert Omit()._referenced_steps == []


def test_if_expression_with_literals():
    expression = IfExpression(
        condition=ConditionEquals(left=True, right=True),
        then_value="selected",
        else_value="fallback",
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
    expression = IfExpression(
        condition=ConditionEquals(left=enabled, right=True),
        then_value=primary,
        else_value=fallback,
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
    expression = IfExpression(
        condition=ConditionEquals(left=mode, right="primary"),
        then_value={"Subnets": ["subnet-primary"]},
        else_value=IfExpression(
            condition=ConditionEquals(left=mode, right="fallback"),
            then_value={"Subnets": ["subnet-fallback"]},
            else_value=Omit(),
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
    expression = IfExpression(
        condition=ConditionEquals(left=True, right=True),
        then_value={"Items": [value, {"Nested": value}]},
        else_value=Omit(),
    )

    assert expression.expr["Std:If"]["Then"] == {
        "Items": [
            {"Get": "Parameters.Value"},
            {"Nested": {"Get": "Parameters.Value"}},
        ]
    }


def test_if_expression_collects_references_from_condition_and_both_branches():
    condition_value = _TestVariable({"Get": "Steps.Condition.Value"}, ["Condition"])
    then_value = _TestVariable({"Get": "Steps.Then.Value"}, ["Then"])
    else_value = _TestVariable({"Get": "Steps.Else.Value"}, ["Else"])
    expression = IfExpression(
        condition=ConditionEquals(left=condition_value, right=True),
        then_value={"Nested": [then_value]},
        else_value=else_value,
    )

    assert expression._referenced_steps == ["Condition", "Then", "Else"]


def test_if_expression_rejects_non_condition():
    with pytest.raises(TypeError):
        IfExpression(condition=True, then_value="yes", else_value="no")


def test_omit_to_string_stays_omit():
    assert Omit().to_string().expr == {"Std:Omit": True}


def test_if_expression_to_string_stringifies_branches_and_keeps_if_outermost():
    number = _TestVariable({"Get": "Steps.Train.Number"}, ["Train"])
    expression = IfExpression(
        condition=ConditionEquals(left=True, right=True),
        then_value=number,
        else_if_branches=[ElseIfBranch(condition=ConditionEquals(left=1, right=1), then_value=5)],
        else_value=Omit(),
    )

    result = expression.to_string().expr["Std:If"]

    assert result["Then"] == {"Std:Join": {"On": "", "Values": [{"Get": "Steps.Train.Number"}]}}
    assert result["ElseIf"][0]["Then"] == "5"
    assert result["Else"] == {"Std:Omit": True}
    assert result["Condition"] == expression.expr["Std:If"]["Condition"]
