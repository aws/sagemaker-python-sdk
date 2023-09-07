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

from sagemaker.workflow.conditions import (
    ConditionEquals,
    ConditionGreaterThan,
    ConditionGreaterThanOrEqualTo,
    ConditionIn,
    ConditionLessThan,
    ConditionLessThanOrEqualTo,
    ConditionNot,
    ConditionOr,
)
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.properties import Properties
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


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
                    "LeftValue": param,
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


def test_pipeline_condition_step_interpolated(sagemaker_session):
    param1 = ParameterInteger(name="MyInt1")
    param2 = ParameterInteger(name="MyInt2")
    param3 = ParameterString(name="MyStr")
    var = ExecutionVariables.START_DATETIME
    prop = Properties("foo")

    cond_eq = ConditionEquals(left=param1, right=param2)
    cond_gt = ConditionGreaterThan(left=var, right="2020-12-01")
    cond_gte = ConditionGreaterThanOrEqualTo(left=var, right=param3)
    cond_lt = ConditionLessThan(left=var, right="2020-12-01")
    cond_lte = ConditionLessThanOrEqualTo(left=var, right=param3)
    cond_in = ConditionIn(value=param3, in_values=["abc", "def"])
    cond_in_mixed = ConditionIn(value=param3, in_values=["abc", prop, var])
    cond_not_eq = ConditionNot(expression=cond_eq)
    cond_not_in = ConditionNot(expression=cond_in)
    cond_or = ConditionOr(conditions=[cond_gt, cond_in])

    step1 = CustomStep(name="MyStep1")
    step2 = CustomStep(name="MyStep2")
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[
            cond_eq,
            cond_gt,
            cond_gte,
            cond_lt,
            cond_lte,
            cond_in,
            cond_in_mixed,
            cond_not_eq,
            cond_not_in,
            cond_or,
        ],
        if_steps=[step1],
        else_steps=[step2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[param1, param2, param3],
        steps=[cond_step],
        sagemaker_session=sagemaker_session,
    )
    assert json.loads(pipeline.definition()) == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [
            {"Name": "MyInt1", "Type": "Integer"},
            {"Name": "MyInt2", "Type": "Integer"},
            {"Name": "MyStr", "Type": "String"},
        ],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "MyConditionStep",
                "Type": "Condition",
                "Arguments": {
                    "Conditions": [
                        {
                            "Type": "Equals",
                            "LeftValue": {"Get": "Parameters.MyInt1"},
                            "RightValue": {"Get": "Parameters.MyInt2"},
                        },
                        {
                            "Type": "GreaterThan",
                            "LeftValue": {"Get": "Execution.StartDateTime"},
                            "RightValue": "2020-12-01",
                        },
                        {
                            "Type": "GreaterThanOrEqualTo",
                            "LeftValue": {"Get": "Execution.StartDateTime"},
                            "RightValue": {"Get": "Parameters.MyStr"},
                        },
                        {
                            "Type": "LessThan",
                            "LeftValue": {"Get": "Execution.StartDateTime"},
                            "RightValue": "2020-12-01",
                        },
                        {
                            "Type": "LessThanOrEqualTo",
                            "LeftValue": {"Get": "Execution.StartDateTime"},
                            "RightValue": {"Get": "Parameters.MyStr"},
                        },
                        {
                            "Type": "In",
                            "QueryValue": {"Get": "Parameters.MyStr"},
                            "Values": ["abc", "def"],
                        },
                        {
                            "Type": "In",
                            "QueryValue": {"Get": "Parameters.MyStr"},
                            "Values": [
                                "abc",
                                {"Get": "Steps.foo"},
                                {"Get": "Execution.StartDateTime"},
                            ],
                        },
                        {
                            "Type": "Not",
                            "Expression": {
                                "Type": "Equals",
                                "LeftValue": {"Get": "Parameters.MyInt1"},
                                "RightValue": {"Get": "Parameters.MyInt2"},
                            },
                        },
                        {
                            "Type": "Not",
                            "Expression": {
                                "Type": "In",
                                "QueryValue": {"Get": "Parameters.MyStr"},
                                "Values": ["abc", "def"],
                            },
                        },
                        {
                            "Type": "Or",
                            "Conditions": [
                                {
                                    "Type": "GreaterThan",
                                    "LeftValue": {"Get": "Execution.StartDateTime"},
                                    "RightValue": "2020-12-01",
                                },
                                {
                                    "Type": "In",
                                    "QueryValue": {"Get": "Parameters.MyStr"},
                                    "Values": ["abc", "def"],
                                },
                            ],
                        },
                    ],
                    "IfSteps": [{"Name": "MyStep1", "Type": "Training", "Arguments": {}}],
                    "ElseSteps": [{"Name": "MyStep2", "Type": "Training", "Arguments": {}}],
                },
            }
        ],
    }


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
