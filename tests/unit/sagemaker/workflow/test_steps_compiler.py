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

import pytest

from sagemaker.workflow.steps import Step, StepTypeEnum
from sagemaker.workflow.function_step import step
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow._steps_compiler import StepsCompiler
from sagemaker.workflow.utilities import load_step_compilation_context, list_to_request


class CustomStep(Step):
    def __init__(
        self,
        name,
        display_name=None,
        description=None,
        depends_on=None,
        input_data=None,
    ):
        super(CustomStep, self).__init__(
            name, display_name, description, StepTypeEnum.TRAINING, depends_on
        )
        self._input_data = input_data
        # for testing property reference, we just use DescribeTrainingJobResponse shape here.
        self._properties = Properties(name, step=self, shape_name="DescribeTrainingJobResponse")

    @property
    def arguments(self):
        context = load_step_compilation_context()
        args = {"Output": f"s3://bucket/{context.pipeline_name}/{context.step_name}"}
        if self._input_data:
            args["Input"] = self._input_data
        return args

    @property
    def properties(self):
        return self._properties


class CustomStepCollection(StepCollection):
    def __init__(self, name, depends_on=None):
        step_1 = CustomStep(name=f"{name}-0", depends_on=depends_on)
        step_2 = CustomStep(name=f"{name}-1", depends_on=[step_1])
        super(CustomStepCollection, self).__init__(
            name=name, steps=[step_1, step_2], depends_on=depends_on
        )


def test_compile_steps_with_explicit_dependencies():
    # a diamond graph
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1")
    step_1.add_depends_on([step_0])
    step_2 = CustomStep(name="custom-step-2")
    step_2.add_depends_on([step_0])
    step_3 = CustomStep(name="custom-step-3")
    step_3.add_depends_on([step_1, step_2])

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_3],
    )
    compiled_steps = compiler.build()

    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-3",
            "Type": "Training",
            "DependsOn": ["custom-step-1", "custom-step-2"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-3",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-2",
            "Type": "Training",
            "DependsOn": ["custom-step-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-2",
            },
        },
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
    ]


def test_compile_steps_with_data_dependencies():
    # a diamond graph
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(
        name="custom-step-1",
        input_data=[step_0.properties.OutputDataConfig.S3OutputPath],
    )
    step_2 = CustomStep(
        name="custom-step-2",
        input_data=[step_0.properties.OutputDataConfig.S3OutputPath],
    )
    step_3 = CustomStep(
        name="custom-step-3",
        input_data=[
            step_1.properties.OutputDataConfig.S3OutputPath,
            step_2.properties.OutputDataConfig.S3OutputPath,
        ],
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_3],
    )
    compiled_steps = compiler.build()

    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-3",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_1.properties.OutputDataConfig.S3OutputPath,
                    step_2.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-3",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_0.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-2",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_0.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-2",
            },
        },
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
    ]


def test_compile_steps_with_missing_input_steps():
    # a diamond graph
    CustomStep(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1")
    step_1.add_depends_on(["custom-step-0"])
    step_2 = CustomStep(name="custom-step-2")
    step_2.add_depends_on(["custom-step-0"])
    step_3 = CustomStep(name="custom-step-3")
    step_3.add_depends_on([step_1, step_2])

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_3],
    )

    with pytest.raises(
        ValueError, match="The input steps do not contain the step of name: custom-step-0"
    ):
        compiler.build()


def test_compile_steps_with_cycle():
    # a diamond graph
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1")
    step_1.add_depends_on([step_0])
    step_2 = CustomStep(name="custom-step-2")
    step_2.add_depends_on([step_1])
    step_0.add_depends_on([step_2])

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_2],
    )
    compiled_steps = compiler.build()

    # the compiler doesn't detect cycles.
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-2",
            "Type": "Training",
            "DependsOn": ["custom-step-1"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-2",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "DependsOn": ["custom-step-2"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
    ]


def test_compile_condition_step_with_nested_step_collection():
    step_0 = CustomStep(name="custom-step-0")
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[
            ConditionEquals(
                left=step_0.properties.TrainingJobStatus,
                right="Completed",
            )
        ],
        if_steps=[CustomStep(name="custom-step-1")],
        else_steps=[CustomStepCollection(name="custom-step-2")],
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_0, condition_step],
    )
    compiled_steps = compiler.build()
    assert compiled_steps[0].step_type == StepTypeEnum.TRAINING
    assert compiled_steps[1].step_type == StepTypeEnum.CONDITION
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
        {
            "Name": "condition-step",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "Equals",
                        "LeftValue": step_0.properties.TrainingJobStatus,
                        "RightValue": "Completed",
                    }
                ],
                "IfSteps": [
                    {
                        "Name": "custom-step-1",
                        "Type": "Training",
                        "Arguments": {
                            "Output": "s3://bucket/test-pipeline/custom-step-1",
                        },
                    }
                ],
                "ElseSteps": [
                    {
                        "Name": "custom-step-2-0",
                        "Type": "Training",
                        "Arguments": {
                            "Output": "s3://bucket/test-pipeline/custom-step-2-0",
                        },
                    },
                    {
                        "Name": "custom-step-2-1",
                        "Type": "Training",
                        "Arguments": {
                            "Output": "s3://bucket/test-pipeline/custom-step-2-1",
                        },
                        "DependsOn": ["custom-step-2-0"],
                    },
                ],
            },
        },
    ]


def test_compile_condition_step_explicit_depending_on_another_step():
    step_0 = CustomStep(name="custom-step-0")
    condition_step = ConditionStep(
        name="condition-step",
        depends_on=[step_0],
        conditions=[
            ConditionEquals(
                left="Failed",
                right="Completed",
            )
        ],
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_0, condition_step],
    )
    compiled_steps = compiler.build()

    assert compiled_steps[0].step_type == StepTypeEnum.TRAINING
    assert compiled_steps[1].step_type == StepTypeEnum.CONDITION
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
        {
            "Name": "condition-step",
            "Type": "Condition",
            "DependsOn": ["custom-step-0"],
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "Equals",
                        "LeftValue": "Failed",
                        "RightValue": "Completed",
                    }
                ],
                "ElseSteps": [],
                "IfSteps": [],
            },
        },
    ]


def test_compile_condition_step_with_sub_steps_and_conditions_sharing_common_dependency():
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(
        name="custom-step-1",
        input_data=[step_0.properties.OutputDataConfig.S3OutputPath],
    )
    step_2 = CustomStep(
        name="custom-step-2",
        input_data=[step_0.properties.OutputDataConfig.S3OutputPath],
    )
    step_3 = CustomStep(
        name="custom-step-3",
        input_data=[
            step_1.properties.OutputDataConfig.S3OutputPath,
            step_2.properties.OutputDataConfig.S3OutputPath,
        ],
    )

    condition_step = ConditionStep(
        name="condition-step",
        conditions=[
            ConditionEquals(
                left=step_0.properties.TrainingJobStatus,
                right="Completed",
            )
        ],
        if_steps=[step_3],
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[condition_step],
    )
    compiled_steps = compiler.build()
    assert list_to_request(compiled_steps) == [
        {
            "Name": "condition-step",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "Equals",
                        "LeftValue": step_0.properties.TrainingJobStatus,
                        "RightValue": "Completed",
                    }
                ],
                "IfSteps": [
                    {
                        "Name": "custom-step-3",
                        "Type": "Training",
                        "Arguments": {
                            "Input": [
                                step_1.properties.OutputDataConfig.S3OutputPath,
                                step_2.properties.OutputDataConfig.S3OutputPath,
                            ],
                            "Output": "s3://bucket/test-pipeline/custom-step-3",
                        },
                    },
                ],
                "ElseSteps": [],
            },
        },
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_0.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-2",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_0.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-2",
            },
        },
    ]


def test_condition_step_substep_and_step_outside_sharing_common_dependency():
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(
        name="custom-step-1",
    )
    step_out = CustomStep(
        name="custom-step-out",
        input_data=[step_1.properties.OutputDataConfig.S3OutputPath],
    )
    step_in = CustomStep(
        name="custom-step-in",
        input_data=[
            step_1.properties.OutputDataConfig.S3OutputPath,
        ],
    )

    condition_step = ConditionStep(
        name="condition-step",
        conditions=[
            ConditionEquals(
                left=step_0.properties.TrainingJobStatus,
                right="Completed",
            )
        ],
        else_steps=[step_in],
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[condition_step, step_out],
    )
    compiled_steps = compiler.build()
    assert list_to_request(compiled_steps) == [
        {
            "Name": "condition-step",
            "Type": "Condition",
            "Arguments": {
                "Conditions": [
                    {
                        "Type": "Equals",
                        "LeftValue": step_0.properties.TrainingJobStatus,
                        "RightValue": "Completed",
                    }
                ],
                "IfSteps": [],
                "ElseSteps": [
                    {
                        "Name": "custom-step-in",
                        "Type": "Training",
                        "Arguments": {
                            "Input": [
                                step_1.properties.OutputDataConfig.S3OutputPath,
                            ],
                            "Output": "s3://bucket/test-pipeline/custom-step-in",
                        },
                    },
                ],
            },
        },
        {
            "Name": "custom-step-out",
            "Type": "Training",
            "Arguments": {
                "Input": [
                    step_1.properties.OutputDataConfig.S3OutputPath,
                ],
                "Output": "s3://bucket/test-pipeline/custom-step-out",
            },
        },
        {
            "Name": "custom-step-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
    ]


def test_step_depends_on_substep_in_condition_step():
    condition_step = ConditionStep(
        name="condition-step",
        conditions=[ConditionEquals(left=1, right=1)],
        if_steps=[
            CustomStep(name="custom-step-0"),
        ],
    )

    step_1 = CustomStep(name="custom-step-1", depends_on=[condition_step.if_steps[0]])

    compiled_steps = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_1, condition_step],
    ).build()

    assert list_to_request(compiled_steps) == [
        {
            "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-1"},
            "DependsOn": ["custom-step-0"],
            "Name": "custom-step-1",
            "Type": "Training",
        },
        {
            "Arguments": {
                "Conditions": [
                    {
                        "LeftValue": 1,
                        "RightValue": 1,
                        "Type": "Equals",
                    }
                ],
                "ElseSteps": [],
                "IfSteps": [
                    {
                        "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-0"},
                        "Name": "custom-step-0",
                        "Type": "Training",
                    }
                ],
            },
            "Name": "condition-step",
            "Type": "Condition",
        },
    ]


def test_nested_condition_step_in_condition_step():
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1", depends_on=[step_0])
    step_2 = CustomStep(name="custom-step-2", depends_on=[step_1])

    condition_step = ConditionStep(
        name="condition-step",
        conditions=[ConditionEquals(left=1, right=1)],
        if_steps=[
            step_2,
            ConditionStep(
                name="nested-condition-step",
                conditions=[ConditionEquals(left=1, right=1)],
                if_steps=[step_1],
            ),
        ],
    )

    compiled_steps = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[condition_step],
    ).build()

    assert list_to_request(compiled_steps) == [
        {
            "Arguments": {
                "Conditions": [
                    {
                        "LeftValue": 1,
                        "RightValue": 1,
                        "Type": "Equals",
                    },
                ],
                "ElseSteps": [],
                "IfSteps": [
                    {
                        "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-2"},
                        "Name": "custom-step-2",
                        "Type": "Training",
                        "DependsOn": ["custom-step-1"],
                    },
                    {
                        "Arguments": {
                            "Conditions": [
                                {
                                    "LeftValue": 1,
                                    "RightValue": 1,
                                    "Type": "Equals",
                                },
                            ],
                            "ElseSteps": [],
                            "IfSteps": [
                                {
                                    "Arguments": {
                                        "Output": "s3://bucket/test-pipeline/custom-step-1"
                                    },
                                    "Name": "custom-step-1",
                                    "Type": "Training",
                                    "DependsOn": ["custom-step-0"],
                                },
                            ],
                        },
                        "Name": "nested-condition-step",
                        "Type": "Condition",
                    },
                ],
            },
            "Name": "condition-step",
            "Type": "Condition",
        },
        {
            "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-0"},
            "Name": "custom-step-0",
            "Type": "Training",
        },
    ]


def test_condition_step_if_else_branch_sharing_common_dependency():
    step_0 = CustomStep(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1", depends_on=[step_0])
    step_2 = CustomStep(name="custom-step-2", depends_on=[step_0])

    condition_step = ConditionStep(
        name="condition-step",
        conditions=[ConditionEquals(left=1, right=1)],
        if_steps=[
            step_1,
        ],
        else_steps=[
            step_2,
        ],
    )

    compiled_steps = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[condition_step],
    ).build()

    assert list_to_request(compiled_steps) == [
        {
            "Arguments": {
                "Conditions": [
                    {
                        "LeftValue": 1,
                        "RightValue": 1,
                        "Type": "Equals",
                    },
                ],
                "ElseSteps": [
                    {
                        "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-2"},
                        "Name": "custom-step-2",
                        "Type": "Training",
                        "DependsOn": ["custom-step-0"],
                    },
                ],
                "IfSteps": [
                    {
                        "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-1"},
                        "Name": "custom-step-1",
                        "Type": "Training",
                        "DependsOn": ["custom-step-0"],
                    },
                ],
            },
            "Name": "condition-step",
            "Type": "Condition",
        },
        {
            "Arguments": {"Output": "s3://bucket/test-pipeline/custom-step-0"},
            "Name": "custom-step-0",
            "Type": "Training",
        },
    ]


def test_depends_on_step_collection_via_name():
    step_0 = CustomStepCollection(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1", depends_on=["custom-step-0"])

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_0, step_1],
    )
    compiled_steps = compiler.build()
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-0-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-0",
            },
        },
        {
            "Name": "custom-step-0-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-1",
            },
        },
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0-0", "custom-step-0-1"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
    ]


def test_depends_on_step_collection_via_step_instance():
    step_0 = CustomStepCollection(name="custom-step-0")
    step_1 = CustomStep(name="custom-step-1", depends_on=[step_0])

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_1],
    )
    compiled_steps = compiler.build()
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0-0", "custom-step-0-1"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-0-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-0",
            },
        },
        {
            "Name": "custom-step-0-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-1",
            },
        },
    ]


def test_depends_on_step_collection_via_data_dependency():
    step_0 = CustomStepCollection(name="custom-step-0")
    step_1 = CustomStep(
        name="custom-step-1", input_data=[step_0.properties.OutputDataConfig.S3OutputPath]
    )

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step_1],
    )
    compiled_steps = compiler.build()
    assert list_to_request(compiled_steps) == [
        {
            "Name": "custom-step-1",
            "Type": "Training",
            "Arguments": {
                "Input": [step_0.steps[-1].properties.OutputDataConfig.S3OutputPath],
                "Output": "s3://bucket/test-pipeline/custom-step-1",
            },
        },
        {
            "Name": "custom-step-0-1",
            "Type": "Training",
            "DependsOn": ["custom-step-0-0"],
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-1",
            },
        },
        {
            "Name": "custom-step-0-0",
            "Type": "Training",
            "Arguments": {
                "Output": "s3://bucket/test-pipeline/custom-step-0-0",
            },
        },
    ]


def test_invoke_build_multiple_times():
    step = CustomStepCollection(name="custom-step")

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=None,
        pipeline_definition_config=None,
        steps=[step],
    )
    compiler.build()
    with pytest.raises(RuntimeError):
        compiler.build()


def test_consistent_setting_across_function_steps(sagemaker_session_mock):
    @step(
        name="func_1",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def func_1():
        return 1

    @step(
        name="func_2",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def func_2():
        return 1

    step_output_1 = func_1()
    step_output_2 = func_2()

    compiler = StepsCompiler(
        pipeline_name="test-pipeline",
        sagemaker_session=sagemaker_session_mock,
        pipeline_definition_config=PipelineDefinitionConfig(False),
        steps=[step_output_1, step_output_2],
    )

    compiled_steps = compiler.build()

    assert len(compiled_steps) == 2
    assert compiled_steps[0].arguments["Environment"] == compiled_steps[1].arguments["Environment"]


def test_same_function_step_built_for_different_pipelines(sagemaker_session_mock):
    @step(
        name="step_name",
        instance_type="ml.m5.large",
        image_uri="test_image_uri",
    )
    def dummy():
        return 1

    step_output = dummy()
    cond_step = ConditionStep(
        name="MyConditionStep",
        conditions=[ConditionEquals(left=step_output, right=1)],
        if_steps=[CustomStep(name="custom-step-0")],
    )

    compiled_steps_1 = StepsCompiler(
        pipeline_name="test-pipeline-1",
        sagemaker_session=sagemaker_session_mock,
        pipeline_definition_config=PipelineDefinitionConfig(False),
        steps=[cond_step],
    ).build()

    assert "'--serialize_output_to_json', 'true'" in str(
        compiled_steps_1[1].arguments["AlgorithmSpecification"]["ContainerArguments"]
    )

    compiled_steps_2 = StepsCompiler(
        pipeline_name="test-pipeline-2",
        sagemaker_session=sagemaker_session_mock,
        pipeline_definition_config=PipelineDefinitionConfig(False),
        steps=[step_output],
    ).build()

    assert "'--serialize_output_to_json', 'true'" not in str(
        compiled_steps_2[0].arguments["AlgorithmSpecification"]["ContainerArguments"]
    )
