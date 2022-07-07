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

from botocore.exceptions import ClientError

from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.model import Model
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.local.local_session import LocalSession
from sagemaker.local.pipeline import LocalPipelineExecutor, StepExecutionException
from sagemaker.local.entities import _LocalPipelineExecution
from tests.unit.sagemaker.workflow.helpers import CustomStep

STRING_PARAMETER = ParameterString("MyStr", "DefaultParameter")
INPUT_STEP = CustomStep(name="InputStep")


@pytest.fixture()
def local_sagemaker_session():
    return LocalSession()


@pytest.fixture
def role_arn():
    return "arn:role"


def test_evaluate_parameter(local_sagemaker_session):
    step = CustomStep(name="MyStep", input_data=STRING_PARAMETER)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline, {"MyStr": "test_string"})
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == "test_string"


def test_evaluate_parameter_undefined(local_sagemaker_session, role_arn):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )
    with pytest.raises(ClientError) as error:
        pipeline.create(role_arn, "test pipeline")
        pipeline.start()
    assert f"Parameter '{parameter.name}' is undefined." in str(error.value)


def test_evaluate_parameter_unknown(local_sagemaker_session, role_arn):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )
    with pytest.raises(ClientError) as error:
        pipeline.create(role_arn, "test pipeline")
        pipeline.start({"MyStr": "test-test", "UnknownParameterFoo": "foo"})
    assert "Unknown parameter 'UnknownParameterFoo'" in str(error.value)


def test_evaluate_parameter_wrong_type(local_sagemaker_session, role_arn):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )
    with pytest.raises(ClientError) as error:
        pipeline.create(role_arn, "test pipeline")
        pipeline.start({"MyStr": True})
    assert (
        f"Unexpected type for parameter '{parameter.name}'. Expected "
        f"{parameter.parameter_type.python_type} but found {type(True)}." in str(error.value)
    )


@pytest.mark.parametrize(
    "property_reference, expected",
    [
        (INPUT_STEP.properties.TrainingJobArn, "my-training-arn"),
        (INPUT_STEP.properties.ExperimentConfig.TrialName, "trial-bar"),
        (INPUT_STEP.properties.FinalMetricDataList[0].Value, 24),
        (INPUT_STEP.properties.FailureReason, "Error: bad input!"),
        (INPUT_STEP.properties.AlgorithmSpecification.AlgorithmName, "fooAlgorithm"),
        (INPUT_STEP.properties.AlgorithmSpecification.MetricDefinitions[0].Name, "mse"),
        (INPUT_STEP.properties.Environment["max-depth"], "10"),
    ],
)
def test_evaluate_property_reference(local_sagemaker_session, property_reference, expected):
    step = CustomStep(name="MyStep", input_data=property_reference)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution[INPUT_STEP.name].properties = {
        "AlgorithmSpecification": {
            "AlgorithmName": "fooAlgorithm",
            "MetricDefinitions": [{"Name": "mse", "Regex": ".*MeanSquaredError.*"}],
        },
        "TrainingJobArn": "my-training-arn",
        "FinalMetricDataList": [{"MetricName": "mse", "Timestamp": 1656281030, "Value": 24}],
        "ExperimentConfig": {
            "ExperimentName": "my-exp",
            "TrialComponentDisplayName": "trial-component-foo",
            "TrialName": "trial-bar",
        },
        "Environment": {"max-depth": "10"},
        "FailureReason": "Error: bad input!",
    }
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == expected


def test_evaluate_property_reference_undefined(local_sagemaker_session):
    step = CustomStep(name="MyStep", input_data=INPUT_STEP.properties.FailureReason)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution[INPUT_STEP.name].properties = {"TrainingJobArn": "my-training-arn"}
    with pytest.raises(StepExecutionException) as e:
        LocalPipelineExecutor(execution, local_sagemaker_session).evaluate_step_arguments(step)
    assert f"{INPUT_STEP.properties.FailureReason.expr} is undefined." in str(e.value)


@pytest.mark.parametrize(
    "join_value, expected",
    [
        (ExecutionVariables.PIPELINE_NAME, "blah-MyPipeline-blah"),
        (STRING_PARAMETER, "blah-DefaultParameter-blah"),
        (INPUT_STEP.properties.TrainingJobArn, "blah-my-training-arn-blah"),
        (Join(on=".", values=["test1", "test2", "test3"]), "blah-test1.test2.test3-blah"),
        (
            Join(on=".", values=["test", ExecutionVariables.PIPELINE_NAME, "test"]),
            "blah-test.MyPipeline.test-blah",
        ),
    ],
)
def test_evaluate_join_function(local_sagemaker_session, join_value, expected):
    step = CustomStep(name="TestStep", input_data=Join(on="-", values=["blah", join_value, "blah"]))
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[INPUT_STEP, step],
        sagemaker_session=local_sagemaker_session,
    )

    execution = _LocalPipelineExecution("my-execution", pipeline)
    execution.step_execution["InputStep"].properties = {"TrainingJobArn": "my-training-arn"}
    evaluated_args = LocalPipelineExecutor(
        execution, local_sagemaker_session
    ).evaluate_step_arguments(step)
    assert evaluated_args["input_data"] == expected


def test_execute_unsupported_step_type(role_arn, local_sagemaker_session):
    step = CreateModelStep(
        name="MyRegisterModelStep",
        model=Model(image_uri="mock_image_uri"),
    )
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[STRING_PARAMETER],
        steps=[step],
        sagemaker_session=local_sagemaker_session,
    )
    create_pipeline_response = pipeline.create(role_arn, "test pipeline")
    assert create_pipeline_response["PipelineArn"] == "MyPipeline"
    with pytest.raises(ClientError) as e:
        pipeline.start()
    assert f"Step type {step.step_type.value} is not supported in local mode." in str(e.value)
