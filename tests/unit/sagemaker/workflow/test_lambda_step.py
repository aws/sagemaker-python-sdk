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
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name="us-west-2")
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
    )
    return session_mock


def test_lambda_step(sagemaker_session):
    param = ParameterInteger(name="MyInt")
    outputParam1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    outputParam2 = LambdaOutput(output_name="output2", output_type=LambdaOutputTypeEnum.Boolean)
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        display_name="MyLambdaStep",
        description="MyLambdaStepDescription",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1, outputParam2],
    )
    lambda_step.add_depends_on(["SecondTestStep"])
    assert lambda_step.to_request() == {
        "Name": "MyLambdaStep",
        "Type": "Lambda",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "DisplayName": "MyLambdaStep",
        "Description": "MyLambdaStepDescription",
        "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
        "OutputParameters": [
            {"OutputName": "output1", "OutputType": "String"},
            {"OutputName": "output2", "OutputType": "Boolean"},
        ],
        "Arguments": {"arg1": "foo", "arg2": 5, "arg3": param},
    }


def test_lambda_step_output_expr(sagemaker_session):
    param = ParameterInteger(name="MyInt")
    outputParam1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    outputParam2 = LambdaOutput(output_name="output2", output_type=LambdaOutputTypeEnum.Boolean)
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[outputParam1, outputParam2],
    )

    assert lambda_step.properties.Outputs["output1"].expr == {
        "Get": "Steps.MyLambdaStep.OutputParameters['output1']"
    }
    assert lambda_step.properties.Outputs["output2"].expr == {
        "Get": "Steps.MyLambdaStep.OutputParameters['output2']"
    }


def test_pipeline_interpolates_lambda_outputs(sagemaker_session):
    parameter = ParameterString("MyStr")
    outputParam1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    outputParam2 = LambdaOutput(output_name="output2", output_type=LambdaOutputTypeEnum.String)
    lambda_step1 = LambdaStep(
        name="MyLambdaStep1",
        depends_on=["TestStep"],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": "foo"},
        outputs=[outputParam1],
    )
    lambda_step2 = LambdaStep(
        name="MyLambdaStep2",
        depends_on=["TestStep"],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": outputParam1},
        outputs=[outputParam2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[lambda_step1, lambda_step2],
        sagemaker_session=sagemaker_session,
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
                "Name": "MyLambdaStep1",
                "Type": "Lambda",
                "Arguments": {"arg1": "foo"},
                "DependsOn": ["TestStep"],
                "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
                "OutputParameters": [{"OutputName": "output1", "OutputType": "String"}],
            },
            {
                "Name": "MyLambdaStep2",
                "Type": "Lambda",
                "Arguments": {"arg1": {"Get": "Steps.MyLambdaStep1.OutputParameters['output1']"}},
                "DependsOn": ["TestStep"],
                "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
                "OutputParameters": [{"OutputName": "output2", "OutputType": "String"}],
            },
        ],
    }


def test_lambda_step_no_inputs_outputs(sagemaker_session):
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={},
        outputs=[],
    )
    lambda_step.add_depends_on(["SecondTestStep"])
    assert lambda_step.to_request() == {
        "Name": "MyLambdaStep",
        "Type": "Lambda",
        "DependsOn": ["TestStep", "SecondTestStep"],
        "FunctionArn": "arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
        "OutputParameters": [],
        "Arguments": {},
    }
