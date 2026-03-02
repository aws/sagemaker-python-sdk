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

from mock import Mock

from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.callback_step import CallbackStep, CallbackOutput, CallbackOutputTypeEnum
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock()
    session_mock.boto_session.client = Mock()
    session_mock.sagemaker_config = None
    return session_mock


def test_callback_step():
    param = ParameterInteger(name="MyInt")
    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    outputParam2 = CallbackOutput(output_name="output2", output_type=CallbackOutputTypeEnum.Boolean)
    cb_step = CallbackStep(
        name="MyCallbackStep",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1, outputParam2],
    )
    cb_step.add_depends_on(["SecondTestStep"])
    assert cb_step.to_request() == {
        "Name": "MyCallbackStep",
        "Type": "Callback",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "SqsQueueUrl": "https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        "OutputParameters": [
            {"OutputName": "output1", "OutputType": "String"},
            {"OutputName": "output2", "OutputType": "Boolean"},
        ],
        "Arguments": {"arg1": "foo", "arg2": 5, "arg3": param},
    }


def test_callback_step_default_values():
    param = ParameterInteger(name="MyInt")
    outputParam1 = CallbackOutput(output_name="output1")
    cb_step = CallbackStep(
        name="MyCallbackStep",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1],
    )
    cb_step.add_depends_on(["SecondTestStep"])
    assert cb_step.to_request() == {
        "Name": "MyCallbackStep",
        "Type": "Callback",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "SqsQueueUrl": "https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        "OutputParameters": [
            {"OutputName": "output1", "OutputType": "String"},
        ],
        "Arguments": {"arg1": "foo", "arg2": 5, "arg3": param},
    }


def test_callback_step_output_expr():
    param = ParameterInteger(name="MyInt")
    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    outputParam2 = CallbackOutput(output_name="output2", output_type=CallbackOutputTypeEnum.Boolean)
    cb_step = CallbackStep(
        name="MyCallbackStep",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1, outputParam2],
    )

    assert cb_step.properties.Outputs["output1"].expr == {
        "Get": "Steps.MyCallbackStep.OutputParameters['output1']"
    }
    assert cb_step.properties.Outputs["output2"].expr == {
        "Get": "Steps.MyCallbackStep.OutputParameters['output2']"
    }


def test_pipeline_interpolates_callback_outputs(sagemaker_session_mock):
    parameter = ParameterString("MyStr")
    custom_step = CustomStep("TestStep")
    outputParam1 = CallbackOutput(output_name="output1", output_type=CallbackOutputTypeEnum.String)
    outputParam2 = CallbackOutput(output_name="output2", output_type=CallbackOutputTypeEnum.String)
    cb_step1 = CallbackStep(
        name="MyCallbackStep1",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )
    cb_step2 = CallbackStep(
        name="MyCallbackStep2",
        depends_on=["TestStep"],
        sqs_queue_url="https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
        inputs={"arg1": outputParam1},
        outputs=[outputParam2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[cb_step1, cb_step2, custom_step],
        sagemaker_session=sagemaker_session_mock,
    )

    assert json.loads(pipeline.definition()) == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": {"Get": "Execution.PipelineName"},
            "TrialName": {"Get": "Execution.PipelineExecutionId"},
        },
        "Steps": [
            {
                "Name": "MyCallbackStep1",
                "Type": "Callback",
                "Arguments": {"arg1": "foo"},
                "DependsOn": ["TestStep"],
                "SqsQueueUrl": "https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
                "OutputParameters": [{"OutputName": "output1", "OutputType": "String"}],
            },
            {
                "Name": "MyCallbackStep2",
                "Type": "Callback",
                "Arguments": {"arg1": {"Get": "Steps.MyCallbackStep1.OutputParameters['output1']"}},
                "DependsOn": ["TestStep"],
                "SqsQueueUrl": "https://sqs.us-east-2.amazonaws.com/123456789012/MyQueue",
                "OutputParameters": [{"OutputName": "output2", "OutputType": "String"}],
            },
            {
                "Name": "TestStep",
                "Type": "Training",
                "Arguments": {},
            },
        ],
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {
            "MyCallbackStep1": [],
            "MyCallbackStep2": [],
            "TestStep": ["MyCallbackStep1", "MyCallbackStep2"],
        }
    )
