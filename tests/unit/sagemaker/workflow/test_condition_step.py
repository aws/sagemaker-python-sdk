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

from sagemaker.remote_function.job import _JobSettings
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
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


IF_STEP = CustomStep(name="MyStep1")
ELSE_STEP = CustomStep(name="MyStep2")
STEP_SETTINGS = dict(
    instance_type="ml.m5.large",
    image_uri="test_image_uri",
)
SERIALIZE_OUTPUT_TO_JSON_FLAG = "'--serialize_output_to_json', 'true'"


def get_mock_job_settings():
    mock_job_settings = _JobSettings(
        s3_root_uri="s3://bucket",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
        s3_kms_key="kms-key",
    )

    return mock_job_settings


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

    step_0 = CustomStep(name="MyStep0")

    cond_eq = ConditionEquals(left=param1, right=param2)
    cond_gt = ConditionGreaterThan(left=var, right="2020-12-01")
    cond_gte = ConditionGreaterThanOrEqualTo(left=var, right=param3)
    cond_lt = ConditionLessThan(left=var, right="2020-12-01")
    cond_lte = ConditionLessThanOrEqualTo(left=var, right=param3)
    cond_in = ConditionIn(value=param3, in_values=["abc", "def"])
    cond_in_mixed = ConditionIn(
        value=param3, in_values=["abc", step_0.properties.TrainingJobName, var]
    )
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
                                {"Get": "Steps.MyStep0.TrainingJobName"},
                                {"Get": "Execution.StartDateTime"},
                            ],
                        },
                        {
                            "Type": "Not",
                            "Condition": {
                                "Type": "Equals",
                                "LeftValue": {"Get": "Parameters.MyInt1"},
                                "RightValue": {"Get": "Parameters.MyInt2"},
                            },
                        },
                        {
                            "Type": "Not",
                            "Condition": {
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
            },
            {
                "Name": "MyStep0",
                "Type": "Training",
                "Arguments": {},
            },
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


def test_condition_step_with_delayed_return(sagemaker_session_mock):
    @step(
        name="step_name",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def dummy():
        return 1

    step_output = dummy()
    cond_gt = ConditionGreaterThan(left=step_output, right=1)
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_gt],
        if_steps=[IF_STEP],
        else_steps=[ELSE_STEP],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {step_output._step.name}
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 1
            assert step_dsl["Arguments"]["Conditions"][0]["Type"] == "GreaterThan"
            cond_gt_dsl = step_dsl["Arguments"]["Conditions"][0]
            assert cond_gt_dsl["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output._step.name, path="Result"
            )
        else:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)


def test_condition_step_with_delayed_return_sequence_return(sagemaker_session_mock):
    @step(
        name="step_name",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def dummy() -> tuple:
        return 1, 2, 3

    step_output = dummy()
    cond_gt = ConditionGreaterThan(left=step_output[1], right=1)
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_gt],
        if_steps=[IF_STEP],
        else_steps=[ELSE_STEP],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 2
    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {step_output._step.name}
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 1
            assert step_dsl["Arguments"]["Conditions"][0]["Type"] == "GreaterThan"
            cond_gt_dsl = step_dsl["Arguments"]["Conditions"][0]
            assert cond_gt_dsl["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output._step.name, path="Result[1]"
            )
        else:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)


def test_depends_on_multiple_upstream_delayed_returns(sagemaker_session_mock):
    @step(name="step1", **STEP_SETTINGS)
    def func1():
        return 1

    @step(name="step2", **STEP_SETTINGS)
    def func2():
        return 2

    step_output1 = func1()
    step_output2 = func2()
    cond_gt = ConditionGreaterThan(left=step_output1, right=0)
    cond_lt = ConditionLessThan(left=step_output2, right=3)

    depend_step = CustomStep(name="MyStep3")
    cond_step = ConditionStep(
        name="MyConditionStep",
        depends_on=[depend_step],
        conditions=[cond_gt, cond_lt],
        if_steps=[IF_STEP],
        else_steps=[ELSE_STEP],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 4
    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {
                depend_step.name,
                step_output1._step.name,
                step_output2._step.name,
            }
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 2
            assert step_dsl["Arguments"]["Conditions"][0]["Type"] == "GreaterThan"
            assert step_dsl["Arguments"]["Conditions"][1]["Type"] == "LessThan"
            cond_gt_dsl = step_dsl["Arguments"]["Conditions"][0]
            cond_lt_dsl = step_dsl["Arguments"]["Conditions"][1]
            assert cond_gt_dsl["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output1._step.name, path="Result"
            )
            assert cond_lt_dsl["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output2._step.name, path="Result"
            )
        elif step_dsl["Name"] != depend_step.name:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)


def test_depends_on_condition_or_upstream_delayed_returns(sagemaker_session_mock):
    @step(name="step1", **STEP_SETTINGS)
    def func1():
        return 1

    @step(name="step2", **STEP_SETTINGS)
    def func2():
        return 2

    @step(name="step3", **STEP_SETTINGS)
    def func3():
        return 3

    step_output1 = func1()
    step_output2 = func2()
    step_output3 = func2()

    cond_gt = ConditionGreaterThan(left=step_output1, right=0)
    cond_lt = ConditionLessThan(left=step_output2, right=3)
    cond_lt2 = ConditionLessThan(left=step_output3, right=3)

    # test recursive unpacking
    cond_or_nested = ConditionOr(conditions=[cond_gt, cond_lt])
    cond_or = ConditionOr(conditions=[cond_or_nested, cond_lt2])

    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_or],
        if_steps=[IF_STEP],
        else_steps=[ELSE_STEP],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 4
    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {
                step_output3._step.name,
                step_output1._step.name,
                step_output2._step.name,
            }
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 1
            condition_dsl = step_dsl["Arguments"]["Conditions"][0]
            assert condition_dsl["Type"] == "Or"
            assert len(condition_dsl["Conditions"]) == 2
            assert condition_dsl["Conditions"][0]["Type"] == "Or"
            assert condition_dsl["Conditions"][1]["Type"] == "LessThan"
            cond_or_dsl = condition_dsl["Conditions"][0]
            cond_less_than_dsl = condition_dsl["Conditions"][1]
            assert len(cond_or_dsl["Conditions"]) == 2
            assert cond_or_dsl["Conditions"][0]["Type"] == "GreaterThan"
            assert cond_or_dsl["Conditions"][0]["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output1._step.name,
                path="Result",
            )
            assert cond_or_dsl["Conditions"][1]["Type"] == "LessThan"
            assert cond_or_dsl["Conditions"][1]["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output2._step.name,
                path="Result",
            )
            assert cond_less_than_dsl["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output3._step.name, path="Result"
            )
        else:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)


def test_depends_on_condition_not_upstream_delayed_returns(sagemaker_session_mock):
    @step(name="step1", **STEP_SETTINGS)
    def func1():
        return 1

    @step(name="step2", **STEP_SETTINGS)
    def func2():
        return 2

    step_output1 = func1()
    step_output2 = func2()

    cond_gt = ConditionGreaterThan(left=step_output1, right=0)
    cond_lt = ConditionLessThan(left=step_output2, right=3)

    # test recursive unpacking
    cond_or_nested = ConditionOr(conditions=[cond_gt, cond_lt])
    cond_not_nested = ConditionNot(expression=cond_or_nested)
    cond_not = ConditionNot(expression=cond_not_nested)

    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_not],
        if_steps=[IF_STEP],
        else_steps=[ELSE_STEP],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 3

    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {step_output1._step.name, step_output2._step.name}
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 1
            condition_dsl = step_dsl["Arguments"]["Conditions"][0]
            assert condition_dsl["Type"] == "Not"
            cond_expr_dsl = condition_dsl["Condition"]
            assert cond_expr_dsl["Type"] == "Not"
            cond_inner_expr_dsl = cond_expr_dsl["Condition"]
            assert cond_inner_expr_dsl["Type"] == "Or"
            assert len(cond_inner_expr_dsl["Conditions"]) == 2
            assert cond_inner_expr_dsl["Conditions"][0]["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output1._step.name, path="Result"
            )
            assert cond_inner_expr_dsl["Conditions"][1]["LeftValue"] == _get_expected_jsonget_expr(
                step_name=step_output2._step.name, path="Result"
            )
        else:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)


def test_depends_on_condition_in_upstream_delayed_returns(sagemaker_session_mock):
    @step(name="step1", **STEP_SETTINGS)
    def func1():
        return 1

    @step(name="step2", **STEP_SETTINGS)
    def func2():
        return 2

    @step(name="step3", **STEP_SETTINGS)
    def func3():
        return 3

    @step(name="step4", **STEP_SETTINGS)
    def func4():
        return 4

    step_output1 = func1()
    step_output2 = func2()
    step_output3 = func3()
    step_output4 = func4()

    cond_in = ConditionIn(step_output3, [step_output1, step_output2])
    cond_not = ConditionNot(expression=cond_in)

    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[cond_not],
        depends_on=[step_output4],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[cond_step],
        sagemaker_session=sagemaker_session_mock,
    )

    step_dsl_list = json.loads(pipeline.definition())["Steps"]
    assert len(step_dsl_list) == 5
    for step_dsl in step_dsl_list:
        if step_dsl["Type"] == "Condition":
            # Both the user input depends_on and the upstream steps fetched from condition
            # are put into the DependsOn list
            assert set(step_dsl["DependsOn"]) == {
                step_output4._step.name,
                step_output3._step.name,
                step_output1._step.name,
                step_output2._step.name,
            }
            # Check JsonGet expr in Conditions
            assert len(step_dsl["Arguments"]["Conditions"]) == 1
            condition_dsl = step_dsl["Arguments"]["Conditions"][0]
            assert condition_dsl["Type"] == "Not"
            cond_expr_dsl = condition_dsl["Condition"]
            assert cond_expr_dsl["Type"] == "In"
            assert cond_expr_dsl["QueryValue"] == _get_expected_jsonget_expr(
                step_name=step_output3._step.name, path="Result"
            )
            assert cond_expr_dsl["Values"] == [
                _get_expected_jsonget_expr(step_name=step_output1._step.name, path="Result"),
                _get_expected_jsonget_expr(step_name=step_output2._step.name, path="Result"),
            ]
        elif step_dsl["Name"] != step_output4._step.name:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert SERIALIZE_OUTPUT_TO_JSON_FLAG in str(container_args)
        else:
            container_args = step_dsl["Arguments"]["AlgorithmSpecification"]["ContainerArguments"]
            assert "--serialize_output_to_json" not in str(container_args)


def _get_expected_jsonget_expr(step_name: str, path: str):
    return {
        "Std:JsonGet": {
            "S3Uri": {
                "Std:Join": {
                    "On": "/",
                    "Values": [
                        {"Get": f"Steps.{step_name}.OutputDataConfig.S3OutputPath"},
                        "results.json",
                    ],
                }
            },
            "Path": path,
        }
    }
