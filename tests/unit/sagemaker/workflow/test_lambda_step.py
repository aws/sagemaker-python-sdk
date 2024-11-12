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

from mock import Mock, MagicMock

from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.lambda_step import LambdaStep, LambdaOutput, LambdaOutputTypeEnum
from sagemaker.lambda_helper import Lambda
from sagemaker.workflow.steps import CacheConfig
from tests.unit.sagemaker.workflow.helpers import CustomStep, ordered


@pytest.fixture()
def sagemaker_session_cn():
    boto_mock = Mock(name="boto_session", region_name="cn-north-1")
    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="cn-north-1",
        config=None,
        local_mode=False,
    )
    session_mock.account_id.return_value = "234567890123"
    return session_mock


def test_lambda_step(sagemaker_session):
    custom_step1 = CustomStep("TestStep")
    custom_step2 = CustomStep("SecondTestStep")
    param = ParameterInteger(name="MyInt")
    output_param1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    output_param2 = LambdaOutput(output_name="output2", output_type=LambdaOutputTypeEnum.Boolean)
    cache_config = CacheConfig(enable_caching=True, expire_after="PT1H")
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=[custom_step1],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        display_name="MyLambdaStep",
        description="MyLambdaStepDescription",
        inputs={"arg1": "foo", "arg2": 5, "arg3": param},
        outputs=[output_param1, output_param2],
        cache_config=cache_config,
    )
    lambda_step.add_depends_on([custom_step2])
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[param],
        steps=[lambda_step, custom_step1, custom_step2],
        sagemaker_session=sagemaker_session,
    )
    assert json.loads(pipeline.definition())["Steps"][0] == {
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
        "Arguments": {"arg1": "foo", "arg2": 5, "arg3": {"Get": "Parameters.MyInt"}},
        "CacheConfig": {"Enabled": True, "ExpireAfter": "PT1H"},
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyLambdaStep": [], "TestStep": ["MyLambdaStep"], "SecondTestStep": ["MyLambdaStep"]}
    )


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
    custom_step = CustomStep("TestStep")
    parameter = ParameterString("MyStr")
    output_param1 = LambdaOutput(output_name="output1", output_type=LambdaOutputTypeEnum.String)
    output_param2 = LambdaOutput(output_name="output2", output_type=LambdaOutputTypeEnum.String)
    lambda_step1 = LambdaStep(
        name="MyLambdaStep1",
        depends_on=[custom_step],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": "foo"},
        outputs=[output_param1],
    )
    lambda_step2 = LambdaStep(
        name="MyLambdaStep2",
        depends_on=[custom_step],
        lambda_func=Lambda(
            function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
            session=sagemaker_session,
        ),
        inputs={"arg1": output_param1},
        outputs=[output_param2],
    )

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[lambda_step1, lambda_step2, custom_step],
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
            {
                "Name": "TestStep",
                "Type": "Training",
                "Arguments": {},
            },
        ],
    }
    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered(
        {"MyLambdaStep1": [], "MyLambdaStep2": [], "TestStep": ["MyLambdaStep1", "MyLambdaStep2"]}
    )


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


def test_lambda_step_with_function_arn_no_lambda_update(sagemaker_session):
    lambda_func = MagicMock(
        function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
        session=sagemaker_session,
        zipped_code_dir=None,
        script=None,
    )
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=lambda_func,
        inputs={},
        outputs=[],
    )
    function_arn = lambda_step._get_function_arn()
    assert function_arn == "arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda"
    lambda_func.upsert.assert_not_called()


def test_lambda_step_with_function_arn_lambda_updated(sagemaker_session):
    lambda_func = MagicMock(
        function_arn="arn:aws:lambda:us-west-2:123456789012:function:sagemaker_test_lambda",
        zipped_code_dir=None,
        script="code",
        session=sagemaker_session,
    )
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=lambda_func,
        inputs={},
        outputs=[],
    )
    lambda_step._get_function_arn()
    lambda_func.update.assert_called_once()


def test_lambda_step_without_function_arn(sagemaker_session):
    lambda_func = MagicMock(
        function_arn=None,
        function_name="name",
        execution_role_arn="arn:aws:lambda:us-west-2:123456789012:execution_role",
        zipped_code_dir="",
        handler="",
        session=sagemaker_session,
    )
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=lambda_func,
        inputs={},
        outputs=[],
    )
    lambda_step._get_function_arn()
    lambda_func.upsert.assert_called_once()


def test_lambda_step_without_function_arn_and_with_error(sagemaker_session_cn):
    lambda_func = MagicMock(
        function_arn=None,
        function_name="name",
        execution_role_arn="arn:aws:lambda:us-west-2:123456789012:execution_role",
        zipped_code_dir="",
        handler="",
        session=sagemaker_session_cn,
    )

    lambda_func.upsert.side_effect = ValueError()
    lambda_step = LambdaStep(
        name="MyLambdaStep",
        depends_on=["TestStep"],
        lambda_func=lambda_func,
        inputs={},
        outputs=[],
    )
    with pytest.raises(ValueError):
        lambda_step._get_function_arn()
