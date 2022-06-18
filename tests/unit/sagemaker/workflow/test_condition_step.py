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

import pytest
from mock import Mock, MagicMock
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
        account_id=Mock(),
    )
    return session_mock


def test_condition_step():
    param = ParameterInteger(name="MyInt")
    cond = ConditionEquals(left=param, right=1)
    step1 = CustomStep(name="MyStep1")
    step2 = CustomStep(name="MyStep2")
    cond_step = ConditionStep(
        name="MyConditionStep",
        depends_on=["TestStep"],
        conditions=[cond],
        if_steps=[step1],
        else_steps=[step2],
    )
    cond_step.add_depends_on(["SecondTestStep"])
    assert cond_step.to_request() == {
        "Name": "MyConditionStep",
        "Type": "Condition",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "Arguments": {
            "Conditions": [
                {
                    "Type": "Equals",
                    "LeftValue": {"Get": "Parameters.MyInt"},
                    "RightValue": 1,
                },
            ],
            "IfSteps": [
                {
                    "Name": "MyStep1",
                    "Type": "Training",
                    "Arguments": {},
                },
            ],
            "ElseSteps": [
                {
                    "Name": "MyStep2",
                    "Type": "Training",
                    "Arguments": {},
                }
            ],
        },
    }
    assert cond_step.properties.Outcome.expr == {"Get": "Steps.MyConditionStep.Outcome"}


def test_pipeline(sagemaker_session):
    param = ParameterInteger(name="MyInt", default_value=2)
    cond = ConditionEquals(left=param, right=1)
    custom_step1 = CustomStep("IfStep")
    custom_step2 = CustomStep("ElseStep")
    step_cond = ConditionStep(
        name="CondStep",
        conditions=[cond],
        if_steps=[custom_step1],
        else_steps=[custom_step2],
    )
    pipeline = Pipeline(
        name="MyPipeline",
        steps=[step_cond],
        sagemaker_session=sagemaker_session,
        parameters=[param],
    )
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"CondStep": ["IfStep", "ElseStep"], "IfStep": [], "ElseStep": []}
    )
