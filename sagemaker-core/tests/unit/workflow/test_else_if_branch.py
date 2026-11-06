# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Unit tests for ordered ElseIf support in argument-level conditional expressions."""

import pytest

from sagemaker.core.workflow import ElseIfBranch, IfExpression, Omit
from sagemaker.core.workflow.conditions import (
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionIn,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
    ConditionNot,
    ConditionOr,
)
from sagemaker.core.workflow.parameters import (
    ParameterBoolean,
    ParameterInteger,
    ParameterString,
)


def test_empty_else_if_list_matches_legacy_wire_shape():
    mode = ParameterString("Mode")
    expression = IfExpression(
        condition=ConditionEquals(left=mode, right="primary"),
        then_value="primary",
        else_value=Omit(),
    )
    assert "ElseIf" not in expression.expr["Std:If"]
    assert expression.expr == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": {"Get": "Parameters.Mode"},
                "RightValue": "primary",
            },
            "Then": "primary",
            "Else": {"Std:Omit": True},
        }
    }


def test_single_else_if_serializes_between_then_and_else():
    mode = ParameterString("Mode")
    expression = IfExpression(
        condition=ConditionEquals(left=mode, right="primary"),
        then_value="primary",
        else_if_branches=[
            ElseIfBranch(
                condition=ConditionEquals(left=mode, right="fallback"),
                then_value="fallback",
            )
        ],
        else_value=Omit(),
    )
    assert expression.expr == {
        "Std:If": {
            "Condition": {
                "Type": "Equals",
                "LeftValue": {"Get": "Parameters.Mode"},
                "RightValue": "primary",
            },
            "Then": "primary",
            "ElseIf": [
                {
                    "Condition": {
                        "Type": "Equals",
                        "LeftValue": {"Get": "Parameters.Mode"},
                        "RightValue": "fallback",
                    },
                    "Then": "fallback",
                }
            ],
            "Else": {"Std:Omit": True},
        }
    }


def test_multiple_else_if_branches_preserve_declared_order():
    mode = ParameterString("Mode")
    expression = IfExpression(
        condition=ConditionEquals(left=mode, right="a"),
        then_value="a",
        else_if_branches=[
            ElseIfBranch(condition=ConditionEquals(left=mode, right="b"), then_value="b"),
            ElseIfBranch(condition=ConditionEquals(left=mode, right="c"), then_value="c"),
            ElseIfBranch(condition=ConditionEquals(left=mode, right="d"), then_value="d"),
        ],
        else_value="z",
    )
    else_if = expression.expr["Std:If"]["ElseIf"]
    assert [branch["Then"] for branch in else_if] == ["b", "c", "d"]


def test_else_if_supports_every_condition_step_operator():
    score = ParameterInteger("Score")
    mode = ParameterString("Mode")
    flag = ParameterBoolean("Flag")

    expression = IfExpression(
        condition=ConditionGreaterThan(left=score, right=100),
        then_value="high",
        else_if_branches=[
            ElseIfBranch(
                condition=ConditionGreaterThanOrEqualTo(left=score, right=50),
                then_value="medium",
            ),
            ElseIfBranch(
                condition=ConditionLessThan(left=score, right=10),
                then_value="tiny",
            ),
            ElseIfBranch(
                condition=ConditionLessThanOrEqualTo(left=score, right=25),
                then_value="small",
            ),
            ElseIfBranch(
                condition=ConditionIn(value=mode, in_values=["allow", "permit"]),
                then_value="listed",
            ),
            ElseIfBranch(
                condition=ConditionNot(expression=ConditionEquals(left=flag, right=True)),
                then_value="not-flag",
            ),
            ElseIfBranch(
                condition=ConditionOr(
                    conditions=[
                        ConditionEquals(left=mode, right="x"),
                        ConditionEquals(left=mode, right="y"),
                    ]
                ),
                then_value="x-or-y",
            ),
        ],
        else_value=Omit(),
    )
    body = expression.expr["Std:If"]
    branches = body["ElseIf"]
    assert [branch["Then"] for branch in branches] == [
        "medium",
        "tiny",
        "small",
        "listed",
        "not-flag",
        "x-or-y",
    ]

    types = [branch["Condition"]["Type"] for branch in branches]
    assert types == [
        "GreaterThanOrEqualTo",
        "LessThan",
        "LessThanOrEqualTo",
        "In",
        "Not",
        "Or",
    ]


def test_else_if_recursively_serializes_pipeline_variables_and_structures():
    mode = ParameterString("Mode")
    subnet = ParameterString("SecondarySubnet")
    expression = IfExpression(
        condition=ConditionEquals(left=mode, right="primary"),
        then_value={"Subnets": ["subnet-primary"]},
        else_if_branches=[
            ElseIfBranch(
                condition=ConditionEquals(left=mode, right="secondary"),
                then_value={"Subnets": [subnet]},
            )
        ],
        else_value=Omit(),
    )
    assert expression.expr["Std:If"]["ElseIf"][0]["Then"] == {
        "Subnets": [{"Get": "Parameters.SecondarySubnet"}]
    }


def test_else_if_collects_references_from_every_branch_in_order():
    from sagemaker.core.helper.pipeline_variable import PipelineVariable

    class _TestVariable(PipelineVariable):
        def __init__(self, name, steps):
            self._expression = {"Get": f"Steps.{name}"}
            self._steps = steps

        @property
        def expr(self):
            return self._expression

        @property
        def _referenced_steps(self):
            return self._steps

    condition_value = _TestVariable("ConditionValue", ["ConditionStep"])
    then_value = _TestVariable("Then", ["ThenStep"])
    branch_condition = _TestVariable("BranchCondition", ["BranchConditionStep"])
    branch_then = _TestVariable("BranchThen", ["BranchThenStep"])
    else_value = _TestVariable("Else", ["ElseStep"])

    expression = IfExpression(
        condition=ConditionEquals(left=condition_value, right="yes"),
        then_value=then_value,
        else_if_branches=[
            ElseIfBranch(
                condition=ConditionEquals(left=branch_condition, right="yes"),
                then_value=branch_then,
            )
        ],
        else_value=else_value,
    )
    assert expression._referenced_steps == [
        "ConditionStep",
        "ThenStep",
        "BranchConditionStep",
        "BranchThenStep",
        "ElseStep",
    ]


def test_else_if_branch_rejects_non_condition():
    with pytest.raises(TypeError):
        ElseIfBranch(condition=True, then_value="yes")


def test_if_expression_rejects_non_branch_in_else_if_list():
    mode = ParameterString("Mode")
    with pytest.raises(TypeError):
        IfExpression(
            condition=ConditionEquals(left=mode, right="primary"),
            then_value="primary",
            else_if_branches=[("not-a-branch",)],
            else_value=Omit(),
        )
