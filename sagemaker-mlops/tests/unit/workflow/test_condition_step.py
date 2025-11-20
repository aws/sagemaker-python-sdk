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
"""Unit tests for workflow condition_step."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock

from sagemaker.mlops.workflow.condition_step import ConditionStep
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum
from sagemaker.core.workflow.conditions import ConditionEquals


@pytest.fixture
def mock_condition():
    condition = Mock(spec=ConditionEquals)
    condition.to_request.return_value = {"Type": "Equals", "LeftValue": 1, "RightValue": 1}
    return condition


@pytest.fixture
def mock_step():
    step = Mock(spec=Step)
    step.name = "test-step"
    step.to_request.return_value = {"Name": "test-step"}
    return step


def test_condition_step_init(mock_condition, mock_step):
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[mock_condition],
        if_steps=[mock_step],
        else_steps=[]
    )
    assert condition_step.name == "condition-step"
    assert condition_step.step_type == StepTypeEnum.CONDITION
    assert len(condition_step.conditions) == 1
    assert len(condition_step.if_steps) == 1


def test_condition_step_arguments(mock_condition, mock_step):
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[mock_condition],
        if_steps=[mock_step],
        else_steps=[]
    )
    args = condition_step.arguments
    assert "Conditions" in args
    assert "IfSteps" in args
    assert "ElseSteps" in args


def test_condition_step_properties(mock_condition):
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[mock_condition]
    )
    assert hasattr(condition_step.properties, "Outcome")
