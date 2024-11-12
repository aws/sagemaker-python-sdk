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
from __future__ import absolute_import

import json

import pytest

from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from tests.unit.sagemaker.workflow.helpers import ordered


def test_fail_step():
    fail_step = FailStep(
        name="MyFailStep",
        depends_on=["TestStep"],
        error_message="Test error message",
    )
    fail_step.add_depends_on(["SecondTestStep"])
    assert fail_step.to_request() == {
        "Name": "MyFailStep",
        "Type": "Fail",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": {"ErrorMessage": "Test error message"},
    }


def test_fail_step_with_no_error_message():
    fail_step = FailStep(
        name="MyFailStep",
        depends_on=["TestStep"],
    )
    fail_step.add_depends_on(["SecondTestStep"])
    assert fail_step.to_request() == {
        "Name": "MyFailStep",
        "Type": "Fail",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": {"ErrorMessage": ""},
    }


def test_fail_step_with_join_fn_in_error_message():
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[],
        else_steps=[],
    )
    step_fail = FailStep(
        name="FailStep",
        error_message=Join(
            on=": ", values=["Failed due to xxx == yyy returns", step_cond.properties.Outcome]
        ),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_cond, step_fail],
        parameters=[param],
    )

    _expected_dsl = [
        {
            "Name": "CondStep",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [
                    {"Type": "Equals", "LeftValue": {"Get": "Parameters.MyInt"}, "RightValue": 1}
                ],
                "IfSteps": [],
                "ElseSteps": [],
            },
        },
        {
            "Name": "FailStep",
            "Type": "Fail",
            "Arguments": {
                "ErrorMessage": {
                    "Std:Join": {
                        "On": ": ",
                        "Values": [
                            "Failed due to xxx == yyy returns",
                            {"Get": "Steps.CondStep.Outcome"},
                        ],
                    }
                }
            },
        },
    ]

    assert json.loads(pipeline.definition())["Steps"] == _expected_dsl
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered({"CondStep": ["FailStep"], "FailStep": []})


def test_fail_step_with_properties_ref():
    fail_step = FailStep(
        name="MyFailStep",
        error_message="Test error message",
    )

    with pytest.raises(Exception) as error:
        fail_step.properties()

    assert (
        str(error.value)
        == "FailStep is a terminal step and the Properties object is not available for it."
    )
