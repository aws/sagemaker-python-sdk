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

import json

import pytest

from mock import Mock, patch

from sagemaker import s3
from sagemaker.session_settings import SessionSettings
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionEquals
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline, PipelineGraph
from sagemaker.workflow.parallelism_config import ParallelismConfiguration
from sagemaker.workflow.pipeline_experiment_config import (
    PipelineExperimentConfig,
    PipelineExperimentConfigProperties,
)
from sagemaker.workflow.step_collections import StepCollection
from tests.unit.sagemaker.workflow.helpers import ordered, CustomStep
from sagemaker.local.local_session import LocalSession
from botocore.exceptions import ClientError
from sagemaker.workflow.selective_execution_config import SelectiveExecutionConfig


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def sagemaker_session_mock():
    session_mock = Mock()
    session_mock.default_bucket = Mock(name="default_bucket", return_value="s3_bucket")
    session_mock.local_mode = False
    # For tests which doesn't verify config file injection, operate with empty config
    session_mock.sagemaker_config = {}
    session_mock._append_sagemaker_config_tags = Mock(
        name="_append_sagemaker_config_tags", side_effect=lambda tags, config_path_to_tags: tags
    )
    return session_mock


def test_pipeline_create_and_update_without_role_arn(sagemaker_session_mock):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    with pytest.raises(ValueError):
        pipeline.create()
    with pytest.raises(ValueError):
        pipeline.update()
    with pytest.raises(ValueError):
        pipeline.upsert()


def test_pipeline_create_and_update_with_config_injection(sagemaker_session_mock):
    # For tests which doesn't verify config file injection, operate with empty config
    pipeline_role_arn = "arn:aws:iam::111111111111:role/ConfigRole"
    sagemaker_session_mock.sagemaker_config = {
        "SchemaVersion": "1.0",
        "SageMaker": {"Pipeline": {"RoleArn": pipeline_role_arn}},
    }
    sagemaker_session_mock.sagemaker_client.describe_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.settings = SessionSettings()
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create()
    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=pipeline_role_arn,
    )
    pipeline.update()
    sagemaker_session_mock.sagemaker_client.update_pipeline.assert_called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=pipeline_role_arn,
    )
    pipeline.upsert()
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=pipeline_role_arn,
    )


def test_pipeline_create(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )


def test_pipeline_create_with_parallelism_config(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        pipeline_experiment_config=ParallelismConfiguration(max_parallel_execution_steps=10),
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.create_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=role_arn,
        ParallelismConfiguration={"MaxParallelExecutionSteps": 10},
    )


@patch("sagemaker.s3.S3Uploader.upload_string_as_file_body")
def test_large_pipeline_create(sagemaker_session_mock, role_arn):
    sagemaker_session_mock.sagemaker_config = {}
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=_generate_large_pipeline_steps(parameter),
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline.create(role_arn=role_arn)

    assert s3.S3Uploader.upload_string_as_file_body.called_with(
        body=pipeline.definition(), s3_uri="s3://s3_bucket/MyPipeline"
    )

    assert sagemaker_session_mock.sagemaker_client.create_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinitionS3Location={"Bucket": "s3_bucket", "ObjectKey": "MyPipeline"},
        RoleArn=role_arn,
    )


def test_pipeline_update(sagemaker_session_mock, role_arn):
    sagemaker_session_mock.sagemaker_config = {}
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.update(role_arn=role_arn)
    assert len(json.loads(pipeline.definition())["Steps"]) == 0
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )

    step1 = CustomStep(name="MyStep1")
    step2 = CustomStep(name="MyStep2", input_data=step1.properties)
    step_collection = StepCollection(name="MyStepCollection", steps=[step1, step2])
    cond_step = ConditionStep(
        name="MyConditionStep",
        depends_on=[],
        conditions=[ConditionEquals(left=2, right=1)],
        if_steps=[step_collection],
        else_steps=[],
    )
    step3 = CustomStep(name="MyStep3", depends_on=[step_collection])
    pipeline.steps = [cond_step, step3]
    pipeline.update(role_arn=role_arn)
    assert len(json.loads(pipeline.definition())["Steps"]) > 0
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )


def test_pipeline_update_with_parallelism_config(sagemaker_session_mock, role_arn):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        pipeline_experiment_config=ParallelismConfiguration(max_parallel_execution_steps=10),
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.create(role_arn=role_arn)
    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinition=pipeline.definition(),
        RoleArn=role_arn,
        ParallelismConfiguration={"MaxParallelExecutionSteps": 10},
    )


@patch("sagemaker.s3.S3Uploader.upload_string_as_file_body")
def test_large_pipeline_update(sagemaker_session_mock, role_arn):
    sagemaker_session_mock.sagemaker_config = {}
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=_generate_large_pipeline_steps(parameter),
        sagemaker_session=sagemaker_session_mock,
    )

    pipeline.create(role_arn=role_arn)

    assert s3.S3Uploader.upload_string_as_file_body.called_with(
        body=pipeline.definition(), s3_uri="s3://s3_bucket/MyPipeline"
    )

    assert sagemaker_session_mock.sagemaker_client.update_pipeline.called_with(
        PipelineName="MyPipeline",
        PipelineDefinitionS3Location={"Bucket": "s3_bucket", "ObjectKey": "MyPipeline"},
        RoleArn=role_arn,
    )


def test_pipeline_upsert_resource_already_exists(sagemaker_session_mock, role_arn):

    # case 1: resource already exists
    def _raise_does_already_exists_client_error(**kwargs):
        response = {"Error": {"Code": "ValidationException", "Message": "Resource already exists."}}
        raise ClientError(error_response=response, operation_name="create_pipeline")

    sagemaker_session_mock.sagemaker_client.create_pipeline = Mock(
        name="create_pipeline", side_effect=_raise_does_already_exists_client_error
    )

    sagemaker_session_mock.sagemaker_client.update_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.list_tags.return_value = {
        "Tags": [{"Key": "dummy", "Value": "dummy_tag"}]
    }

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    tags = [
        {"Key": "foo", "Value": "abc"},
        {"Key": "bar", "Value": "xyz"},
    ]
    pipeline.upsert(role_arn=role_arn, tags=tags)

    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_once_with(
        PipelineName="MyPipeline",
        RoleArn=role_arn,
        PipelineDefinition=pipeline.definition(),
        Tags=tags,
    )

    sagemaker_session_mock.sagemaker_client.update_pipeline.assert_called_once_with(
        PipelineName="MyPipeline", PipelineDefinition=pipeline.definition(), RoleArn=role_arn
    )
    assert sagemaker_session_mock.sagemaker_client.list_tags.called_with(
        ResourceArn="mock_pipeline_arn"
    )

    tags.append({"Key": "dummy", "Value": "dummy_tag"})
    assert sagemaker_session_mock.sagemaker_client.add_tags.called_with(
        ResourceArn="mock_pipeline_arn", Tags=tags
    )


def test_pipeline_upsert_create_unexpected_failure(sagemaker_session_mock, role_arn):

    # case 2: unexpected failure on create
    def _raise_unexpected_client_error(**kwargs):
        response = {
            "Error": {"Code": "ValidationException", "Message": "Name does not satisfy expression."}
        }
        raise ClientError(error_response=response, operation_name="foo")

    sagemaker_session_mock.sagemaker_client.create_pipeline = Mock(
        name="create_pipeline", side_effect=_raise_unexpected_client_error
    )

    sagemaker_session_mock.sagemaker_client.update_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.list_tags.return_value = {
        "Tags": [{"Key": "dummy", "Value": "dummy_tag"}]
    }

    tags = [
        {"Key": "foo", "Value": "abc"},
        {"Key": "bar", "Value": "xyz"},
    ]

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    with pytest.raises(ClientError):
        pipeline.upsert(role_arn=role_arn, tags=tags)

    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_once_with(
        PipelineName="MyPipeline",
        RoleArn=role_arn,
        PipelineDefinition=pipeline.definition(),
        Tags=tags,
    )
    sagemaker_session_mock.sagemaker_client.update_pipeline.assert_not_called()
    sagemaker_session_mock.sagemaker_client.list_tags.assert_not_called()
    sagemaker_session_mock.sagemaker_client.add_tags.assert_not_called()


def test_pipeline_upsert_resourse_doesnt_exist(sagemaker_session_mock, role_arn):

    # case 3: resource does not exist
    sagemaker_session_mock.sagemaker_client.create_pipeline = Mock(name="create_pipeline")

    sagemaker_session_mock.sagemaker_client.update_pipeline.return_value = {
        "PipelineArn": "pipeline-arn"
    }
    sagemaker_session_mock.sagemaker_client.list_tags.return_value = {
        "Tags": [{"Key": "dummy", "Value": "dummy_tag"}]
    }

    tags = [
        {"Key": "foo", "Value": "abc"},
        {"Key": "bar", "Value": "xyz"},
    ]

    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    try:
        pipeline.upsert(role_arn=role_arn, tags=tags)
    except ClientError:
        assert False, "Unexpected ClientError raised"

    sagemaker_session_mock.sagemaker_client.create_pipeline.assert_called_once_with(
        PipelineName="MyPipeline",
        RoleArn=role_arn,
        PipelineDefinition=pipeline.definition(),
        Tags=tags,
    )

    sagemaker_session_mock.sagemaker_client.update_pipeline.assert_not_called()
    sagemaker_session_mock.sagemaker_client.list_tags.assert_not_called()
    sagemaker_session_mock.sagemaker_client.add_tags.assert_not_called()


def test_pipeline_delete(sagemaker_session_mock):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.delete()
    assert sagemaker_session_mock.sagemaker_client.delete_pipeline.called_with(
        PipelineName="MyPipeline",
    )


def test_pipeline_describe(sagemaker_session_mock):
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.describe()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline.called_with(
        PipelineName="MyPipeline",
    )


def test_pipeline_start(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[ParameterString("alpha", "beta"), ParameterString("gamma", "delta")],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    pipeline.start()
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline",
    )

    pipeline.start(execution_display_name="pipeline-execution")
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline", PipelineExecutionDisplayName="pipeline-execution"
    )

    pipeline.start(parameters=dict(alpha="epsilon"))
    assert sagemaker_session_mock.start_pipeline_execution.called_with(
        PipelineName="MyPipeline", PipelineParameters=[{"Name": "alpha", "Value": "epsilon"}]
    )


def test_pipeline_start_selective_execution(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )

    # Case 1: Happy path
    selective_execution_config = SelectiveExecutionConfig(
        source_pipeline_execution_arn="foo-arn", selected_steps=["step-1", "step-2", "step-3"]
    )
    pipeline.start(selective_execution_config=selective_execution_config)
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.assert_called_with(
        PipelineName="MyPipeline",
        SelectiveExecutionConfig={
            "SelectedSteps": [
                {"StepName": "step-1"},
                {"StepName": "step-2"},
                {"StepName": "step-3"},
            ],
            "SourcePipelineExecutionArn": "foo-arn",
        },
    )

    # Case 2: Start selective execution without SourcePipelineExecutionArn
    sagemaker_session_mock.sagemaker_client.list_pipeline_executions.return_value = {
        "PipelineExecutionSummaries": [
            {
                "PipelineExecutionArn": "my:latest:execution:arn",
                "PipelineExecutionDisplayName": "Latest",
            }
        ]
    }
    selective_execution_config = SelectiveExecutionConfig(
        selected_steps=["step-1", "step-2", "step-3"]
    )
    pipeline.start(selective_execution_config=selective_execution_config)
    sagemaker_session_mock.sagemaker_client.list_pipeline_executions.assert_called_with(
        PipelineName="MyPipeline",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.assert_called_with(
        PipelineName="MyPipeline",
        SelectiveExecutionConfig={
            "SelectedSteps": [
                {"StepName": "step-1"},
                {"StepName": "step-2"},
                {"StepName": "step-3"},
            ],
            "SourcePipelineExecutionArn": "my:latest:execution:arn",
        },
    )


def test_pipeline_basic():
    parameter = ParameterString("MyStr")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)],
        sagemaker_session=sagemaker_session_mock,
    )
    assert pipeline.to_request() == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": ExecutionVariables.PIPELINE_NAME,
            "TrialName": ExecutionVariables.PIPELINE_EXECUTION_ID,
        },
        "Steps": [{"Name": "MyStep", "Type": "Training", "Arguments": {"input_data": parameter}}],
    }
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [{"Name": "MyStr", "Type": "String"}],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": {"Get": "Parameters.MyStr"}},
                }
            ],
        }
    )


def test_pipeline_two_step(sagemaker_session_mock):
    parameter = ParameterString("MyStr")
    step1 = CustomStep(
        name="MyStep1",
        input_data=[
            parameter,  # parameter reference
            ExecutionVariables.PIPELINE_EXECUTION_ID,  # execution variable
            PipelineExperimentConfigProperties.EXPERIMENT_NAME,  # experiment config property
        ],
    )
    step2 = CustomStep(
        name="MyStep2", input_data=[step1.properties.ModelArtifacts.S3ModelArtifacts]
    )  # step property
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step1, step2],
        sagemaker_session=sagemaker_session_mock,
    )
    assert pipeline.to_request() == {
        "Version": "2020-12-01",
        "Metadata": {},
        "Parameters": [{"Name": "MyStr", "Type": "String"}],
        "PipelineExperimentConfig": {
            "ExperimentName": ExecutionVariables.PIPELINE_NAME,
            "TrialName": ExecutionVariables.PIPELINE_EXECUTION_ID,
        },
        "Steps": [
            {
                "Name": "MyStep1",
                "Type": "Training",
                "Arguments": {
                    "input_data": [
                        parameter,
                        ExecutionVariables.PIPELINE_EXECUTION_ID,
                        PipelineExperimentConfigProperties.EXPERIMENT_NAME,
                    ]
                },
            },
            {
                "Name": "MyStep2",
                "Type": "Training",
                "Arguments": {"input_data": [step1.properties.ModelArtifacts.S3ModelArtifacts]},
            },
        ],
    }
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [{"Name": "MyStr", "Type": "String"}],
            "PipelineExperimentConfig": {
                "ExperimentName": {"Get": "Execution.PipelineName"},
                "TrialName": {"Get": "Execution.PipelineExecutionId"},
            },
            "Steps": [
                {
                    "Name": "MyStep1",
                    "Type": "Training",
                    "Arguments": {
                        "input_data": [
                            {"Get": "Parameters.MyStr"},
                            {"Get": "Execution.PipelineExecutionId"},
                            {"Get": "PipelineExperimentConfig.ExperimentName"},
                        ]
                    },
                },
                {
                    "Name": "MyStep2",
                    "Type": "Training",
                    "Arguments": {
                        "input_data": [{"Get": "Steps.MyStep1.ModelArtifacts.S3ModelArtifacts"}]
                    },
                },
            ],
        }
    )

    adjacency_list = PipelineGraph.from_pipeline(pipeline).adjacency_list
    assert ordered(adjacency_list) == ordered({"MyStep1": ["MyStep2"], "MyStep2": []})


def test_pipeline_override_experiment_config():
    pipeline = Pipeline(
        name="MyPipeline",
        pipeline_experiment_config=PipelineExperimentConfig("MyExperiment", "MyTrial"),
        steps=[CustomStep(name="MyStep", input_data="input")],
        sagemaker_session=sagemaker_session_mock,
    )
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": {"ExperimentName": "MyExperiment", "TrialName": "MyTrial"},
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": "input"},
                }
            ],
        }
    )


def test_pipeline_disable_experiment_config():
    pipeline = Pipeline(
        name="MyPipeline",
        pipeline_experiment_config=None,
        steps=[CustomStep(name="MyStep", input_data="input")],
        sagemaker_session=sagemaker_session_mock,
    )
    assert ordered(json.loads(pipeline.definition())) == ordered(
        {
            "Version": "2020-12-01",
            "Metadata": {},
            "Parameters": [],
            "PipelineExperimentConfig": None,
            "Steps": [
                {
                    "Name": "MyStep",
                    "Type": "Training",
                    "Arguments": {"input_data": "input"},
                }
            ],
        }
    )


def test_pipeline_list_executions(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.list_pipeline_executions.return_value = {
        "PipelineExecutionSummaries": [Mock()],
        "ResponseMetadata": "metadata",
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[ParameterString("alpha", "beta"), ParameterString("gamma", "delta")],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    executions = pipeline.list_executions()
    assert len(executions) == 1
    assert len(executions["PipelineExecutionSummaries"]) == 1
    sagemaker_session_mock.sagemaker_client.list_pipeline_executions.return_value = {
        "PipelineExecutionSummaries": [Mock(), Mock()],
        "NextToken": "token",
        "ResponseMetadata": "metadata",
    }
    executions = pipeline.list_executions()
    assert len(executions) == 2
    assert len(executions["PipelineExecutionSummaries"]) == 2
    assert executions["NextToken"] == "token"


def test_pipeline_execution_basics(sagemaker_session_mock):
    sagemaker_session_mock.sagemaker_client.start_pipeline_execution.return_value = {
        "PipelineExecutionArn": "my:arn"
    }
    sagemaker_session_mock.sagemaker_client.list_pipeline_execution_steps.return_value = {
        "PipelineExecutionSteps": [Mock()]
    }
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[ParameterString("alpha", "beta"), ParameterString("gamma", "delta")],
        steps=[],
        sagemaker_session=sagemaker_session_mock,
    )
    execution = pipeline.start()
    execution.stop()
    assert sagemaker_session_mock.sagemaker_client.stop_pipeline_execution.called_with(
        PipelineExecutionArn="my:arn"
    )
    execution.describe()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline_execution.called_with(
        PipelineExecutionArn="my:arn"
    )
    steps = execution.list_steps()
    assert sagemaker_session_mock.sagemaker_client.describe_pipeline_execution_steps.called_with(
        PipelineExecutionArn="my:arn"
    )
    assert len(steps) == 1


def _generate_large_pipeline_steps(input_data: object):
    steps = []
    for i in range(2000):
        steps.append(CustomStep(name=f"MyStep{i}", input_data=input_data))
    return steps


def test_local_pipeline():
    parameter = ParameterString("MyStr", default_value="test")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)],
        sagemaker_session=LocalSession(),
    )
    pipeline.create("dummy-role", "pipeline-description")

    pipeline_describe_response1 = pipeline.describe()
    assert pipeline_describe_response1["PipelineArn"] == "MyPipeline"
    assert pipeline_describe_response1["PipelineDefinition"] == pipeline.definition()
    assert pipeline_describe_response1["PipelineDescription"] == "pipeline-description"

    pipeline.update("dummy-role", "pipeline-description-2")
    pipeline_describe_response2 = pipeline.describe()
    assert pipeline_describe_response2["PipelineDescription"] == "pipeline-description-2"
    assert (
        pipeline_describe_response2["CreationTime"]
        != pipeline_describe_response2["LastModifiedTime"]
    )

    pipeline_execution_describe_response = pipeline.start().describe()
    assert pipeline_execution_describe_response["PipelineArn"] == "MyPipeline"
    assert pipeline_execution_describe_response["PipelineExecutionArn"] is not None
