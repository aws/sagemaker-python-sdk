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
"""Unit tests for ordered ElseIf support in argument-level conditional expressions."""

import pytest

from sagemaker.core.workflow import ElseIf, If, Omit
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
    expression = If(
        condition=ConditionEquals(left=mode, right="primary"),
        then="primary",
        else_=Omit(),
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
    expression = If(
        condition=ConditionEquals(left=mode, right="primary"),
        then="primary",
        else_if=[
            ElseIf(
                condition=ConditionEquals(left=mode, right="fallback"),
                then="fallback",
            )
        ],
        else_=Omit(),
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
    expression = If(
        condition=ConditionEquals(left=mode, right="a"),
        then="a",
        else_if=[
            ElseIf(condition=ConditionEquals(left=mode, right="b"), then="b"),
            ElseIf(condition=ConditionEquals(left=mode, right="c"), then="c"),
            ElseIf(condition=ConditionEquals(left=mode, right="d"), then="d"),
        ],
        else_="z",
    )
    else_if = expression.expr["Std:If"]["ElseIf"]
    assert [branch["Then"] for branch in else_if] == ["b", "c", "d"]


def test_else_if_supports_every_condition_step_operator():
    score = ParameterInteger("Score")
    mode = ParameterString("Mode")
    flag = ParameterBoolean("Flag")

    expression = If(
        condition=ConditionGreaterThan(left=score, right=100),
        then="high",
        else_if=[
            ElseIf(
                condition=ConditionGreaterThanOrEqualTo(left=score, right=50),
                then="medium",
            ),
            ElseIf(
                condition=ConditionLessThan(left=score, right=10),
                then="tiny",
            ),
            ElseIf(
                condition=ConditionLessThanOrEqualTo(left=score, right=25),
                then="small",
            ),
            ElseIf(
                condition=ConditionIn(value=mode, in_values=["allow", "permit"]),
                then="listed",
            ),
            ElseIf(
                condition=ConditionNot(expression=ConditionEquals(left=flag, right=True)),
                then="not-flag",
            ),
            ElseIf(
                condition=ConditionOr(
                    conditions=[
                        ConditionEquals(left=mode, right="x"),
                        ConditionEquals(left=mode, right="y"),
                    ]
                ),
                then="x-or-y",
            ),
        ],
        else_=Omit(),
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
    expression = If(
        condition=ConditionEquals(left=mode, right="primary"),
        then={"Subnets": ["subnet-primary"]},
        else_if=[
            ElseIf(
                condition=ConditionEquals(left=mode, right="secondary"),
                then={"Subnets": [subnet]},
            )
        ],
        else_=Omit(),
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
    then = _TestVariable("Then", ["ThenStep"])
    branch_condition = _TestVariable("BranchCondition", ["BranchConditionStep"])
    branch_then = _TestVariable("BranchThen", ["BranchThenStep"])
    else_ = _TestVariable("Else", ["ElseStep"])

    expression = If(
        condition=ConditionEquals(left=condition_value, right="yes"),
        then=then,
        else_if=[
            ElseIf(
                condition=ConditionEquals(left=branch_condition, right="yes"),
                then=branch_then,
            )
        ],
        else_=else_,
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
        ElseIf(condition=True, then="yes")


def test_if_expression_rejects_non_branch_in_else_if_list():
    mode = ParameterString("Mode")
    with pytest.raises(TypeError):
        If(
            condition=ConditionEquals(left=mode, right="primary"),
            then="primary",
            else_if=[("not-a-branch",)],
            else_=Omit(),
        )
