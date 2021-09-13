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

from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.parameters import ParameterInteger
from sagemaker.workflow.steps import (
    Step,
    StepTypeEnum,
)
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.condition_step import ConditionStep


class CustomStep(Step):
    def __init__(self, name, display_name=None, description=None):
        super(CustomStep, self).__init__(name, display_name, description, StepTypeEnum.TRAINING)
        self._properties = Properties(path=f"Steps.{name}")

    @property
    def arguments(self):
        return dict()

    @property
    def properties(self):
        return self._properties


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
