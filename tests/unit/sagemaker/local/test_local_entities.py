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

import os
import pytest

from mock import patch, Mock

from botocore.exceptions import ClientError

import sagemaker.local
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.lambda_step import LambdaStep
from tests.unit.sagemaker.workflow.helpers import CustomStep


@pytest.fixture(scope="session")
def local_transform_job(sagemaker_local_session):
    with patch(
        "sagemaker.local.local_session.LocalSagemakerClient.describe_model"
    ) as describe_model:
        describe_model.return_value = {
            "PrimaryContainer": {"Environment": {}, "Image": "some-image:1.0"}
        }
        job = sagemaker.local.entities._LocalTransformJob(
            "my-transform-job", "some-model", sagemaker_local_session
        )
        return job


@patch(
    "sagemaker.local.local_session.LocalSagemakerClient.describe_model",
    Mock(return_value={"PrimaryContainer": {}}),
)
def test_local_transform_job_init(sagemaker_local_session):
    job = sagemaker.local.entities._LocalTransformJob(
        "my-transform-job", "some-model", sagemaker_local_session
    )
    assert job.name == "my-transform-job"
    assert job.state == sagemaker.local.entities._LocalTransformJob._CREATING


def test_local_transform_job_container_environment(local_transform_job):
    transform_kwargs = {"MaxPayloadInMB": 3, "BatchStrategy": "MultiRecord"}
    container_env = local_transform_job._get_container_environment(**transform_kwargs)

    assert "SAGEMAKER_BATCH" in container_env
    assert "SAGEMAKER_MAX_PAYLOAD_IN_MB" in container_env
    assert "SAGEMAKER_BATCH_STRATEGY" in container_env
    assert "SAGEMAKER_MAX_CONCURRENT_TRANSFORMS" in container_env

    transform_kwargs = {"BatchStrategy": "SingleRecord"}

    container_env = local_transform_job._get_container_environment(**transform_kwargs)

    assert "SAGEMAKER_BATCH" in container_env
    assert "SAGEMAKER_BATCH_STRATEGY" in container_env
    assert "SAGEMAKER_MAX_CONCURRENT_TRANSFORMS" in container_env

    transform_kwargs = {"Environment": {"MY_ENV": 3}}

    container_env = local_transform_job._get_container_environment(**transform_kwargs)

    assert "SAGEMAKER_BATCH" in container_env
    assert "SAGEMAKER_MAX_PAYLOAD_IN_MB" not in container_env
    assert "SAGEMAKER_BATCH_STRATEGY" not in container_env
    assert "SAGEMAKER_MAX_CONCURRENT_TRANSFORMS" in container_env
    assert "MY_ENV" in container_env


def test_local_transform_job_defaults_with_empty_args(local_transform_job):
    transform_kwargs = {}
    defaults = local_transform_job._get_required_defaults(**transform_kwargs)
    assert "BatchStrategy" in defaults
    assert "MaxPayloadInMB" in defaults


def test_local_transform_job_defaults_with_batch_strategy(local_transform_job):
    transform_kwargs = {"BatchStrategy": "my-own"}
    defaults = local_transform_job._get_required_defaults(**transform_kwargs)
    assert "BatchStrategy" not in defaults
    assert "MaxPayloadInMB" in defaults


def test_local_transform_job_defaults_with_max_payload(local_transform_job):
    transform_kwargs = {"MaxPayloadInMB": 322}
    defaults = local_transform_job._get_required_defaults(**transform_kwargs)
    assert "BatchStrategy" in defaults
    assert "MaxPayloadInMB" not in defaults


@patch("sagemaker.local.entities._SageMakerContainer", Mock())
@patch("sagemaker.local.entities._wait_for_serving_container", Mock())
@patch("sagemaker.local.entities._perform_request")
@patch("sagemaker.local.entities._LocalTransformJob._perform_batch_inference")
def test_start_local_transform_job(_perform_batch_inference, _perform_request, local_transform_job):
    input_data = {}
    output_data = {}
    transform_resources = {"InstanceType": "local"}

    response = Mock()
    _perform_request.return_value = (response, 200)
    response.data = '{"BatchStrategy": "SingleRecord"}'.encode("UTF-8")
    local_transform_job.primary_container["ModelDataUrl"] = "file:///some/model"
    local_transform_job.start(input_data, output_data, transform_resources, Environment={})

    _perform_batch_inference.assert_called()
    response = local_transform_job.describe()
    assert response["TransformJobStatus"] == "Completed"


@patch("sagemaker.local.data.get_batch_strategy_instance")
@patch("sagemaker.local.data.get_data_source_instance")
@patch("sagemaker.local.entities.move_to_destination")
@patch("sagemaker.local.entities.get_config_value")
def test_local_transform_job_perform_batch_inference(
    get_config_value,
    move_to_destination,
    get_data_source_instance,
    get_batch_strategy_instance,
    local_transform_job,
    tmpdir,
):
    input_data = {
        "DataSource": {"S3DataSource": {"S3Uri": "s3://some_bucket/nice/data"}},
        "ContentType": "text/csv",
    }

    output_data = {"S3OutputPath": "s3://bucket/output", "AssembleWith": "Line"}

    transform_kwargs = {"MaxPayloadInMB": 3, "BatchStrategy": "MultiRecord"}

    data_source = Mock()
    data_source.get_file_list.return_value = ["/tmp/file1", "/tmp/file2"]
    data_source.get_root_dir.return_value = "/tmp"
    get_data_source_instance.return_value = data_source

    batch_strategy = Mock()
    batch_strategy.pad.return_value = "some data"
    get_batch_strategy_instance.return_value = batch_strategy

    get_config_value.return_value = str(tmpdir)

    runtime_client = Mock()
    response_object = Mock()
    response_object.read.return_value = b"data"
    runtime_client.invoke_endpoint.return_value = {"Body": response_object}
    local_transform_job.local_session.sagemaker_runtime_client = runtime_client

    local_transform_job.container = Mock()

    local_transform_job._perform_batch_inference(input_data, output_data, **transform_kwargs)

    dir, output, job_name, session = move_to_destination.call_args[0]
    assert output == "s3://bucket/output"
    output_files = os.listdir(dir)
    assert len(output_files) == 2
    assert "file1.out" in output_files
    assert "file2.out" in output_files


@patch("sagemaker.local.entities._SageMakerContainer", Mock())
@patch("sagemaker.local.entities.get_docker_host")
@patch("sagemaker.local.entities._perform_request")
@patch("sagemaker.local.entities._LocalTransformJob._perform_batch_inference")
def test_start_local_transform_job_from_remote_docker_host(
    m_perform_batch_inference, m_perform_request, m_get_docker_host, local_transform_job
):
    input_data = {}
    output_data = {}
    transform_resources = {"InstanceType": "local"}
    m_get_docker_host.return_value = "some_host"
    response = Mock()
    m_perform_request.return_value = (response, 200)
    response.data = '{"BatchStrategy": "SingleRecord"}'.encode("UTF-8")
    local_transform_job.primary_container["ModelDataUrl"] = "file:///some/model"
    local_transform_job.start(input_data, output_data, transform_resources, Environment={})
    endpoints = [
        "http://%s:%d/ping" % ("some_host", 8080),
        "http://%s:%d/execution-parameters" % ("some_host", 8080),
    ]
    calls = m_perform_request.call_args_list
    for call, endpoint in zip(calls, endpoints):
        assert call[0][0] == endpoint


@patch("sagemaker.local.pipeline.LocalPipelineExecutor.execute")
def test_start_local_pipeline(mock_local_pipeline_executor, sagemaker_local_session):
    parameter = ParameterString("MyStr", default_value="test")
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[CustomStep(name="MyStep", input_data=parameter)],
        sagemaker_session=sagemaker_local_session,
    )
    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)

    describe_pipeline_response = local_pipeline.describe()
    assert describe_pipeline_response["PipelineArn"] == "MyPipeline"
    assert describe_pipeline_response["CreationTime"] is not None

    mock_executor_return_value = sagemaker.local.entities._LocalPipelineExecution(
        "execution-id", pipeline
    )
    mock_executor_return_value.step_execution["MyStep"].status = "Executing"
    mock_local_pipeline_executor.return_value = mock_executor_return_value
    pipeline_execution = local_pipeline.start()

    describe_pipeline_execution_response = pipeline_execution.describe()
    assert describe_pipeline_execution_response["PipelineArn"] == "MyPipeline"
    assert describe_pipeline_execution_response["PipelineExecutionArn"] == "execution-id"
    assert describe_pipeline_execution_response["CreationTime"] is not None

    list_steps_response = pipeline_execution.list_steps()
    assert list_steps_response["PipelineExecutionSteps"][0]["StepName"] == "MyStep"
    assert list_steps_response["PipelineExecutionSteps"][0]["StepStatus"] == "Executing"


def test_start_local_pipeline_with_unsupported_step_type(sagemaker_local_session):
    mock_lambda_func = Mock()
    mock_lambda_func.update.return_value = {"FunctionArn": "function_arn"}
    step = LambdaStep(name="MyLambdaStep", lambda_func=mock_lambda_func)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[],
        steps=[step],
        sagemaker_session=sagemaker_local_session,
    )
    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)

    with pytest.raises(ClientError) as e:
        local_pipeline.start()
    assert f"Step type {step.step_type.value} is not supported in local mode." in str(e.value)


def test_start_local_pipeline_with_undefined_parameter(sagemaker_local_session):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=sagemaker_local_session,
    )
    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)
    with pytest.raises(ClientError) as error:
        local_pipeline.start()
    assert f"Parameter '{parameter.name}' is undefined." in str(error.value)


def test_start_local_pipeline_with_unknown_parameter(sagemaker_local_session):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=sagemaker_local_session,
    )
    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)
    with pytest.raises(ClientError) as error:
        local_pipeline.start(
            PipelineParameters={"MyStr": "test-test", "UnknownParameterFoo": "foo"}
        )
    assert "Unknown parameter 'UnknownParameterFoo'" in str(error.value)


def test_start_local_pipeline_with_wrong_parameter_type(sagemaker_local_session):
    parameter = ParameterString("MyStr")
    step = CustomStep(name="MyStep", input_data=parameter)
    pipeline = Pipeline(
        name="MyPipeline",
        parameters=[parameter],
        steps=[step],
        sagemaker_session=sagemaker_local_session,
    )
    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)
    with pytest.raises(ClientError) as error:
        local_pipeline.start(PipelineParameters={"MyStr": True})
    assert (
        f"Unexpected type for parameter '{parameter.name}'. Expected "
        f"{parameter.parameter_type.python_type} but found {type(True)}." in str(error.value)
    )


def test_start_local_pipeline_with_empty_parameter_string_value(
    local_pipeline_session,
):
    param_str_name = "MyParameterString"
    param_str = ParameterString(name=param_str_name, default_value="default")
    fail_step = FailStep(
        name="MyFailStep",
        error_message=param_str,
    )

    pipeline = Pipeline(
        name="MyPipeline",
        steps=[fail_step],
        sagemaker_session=local_pipeline_session,
        parameters=[param_str],
    )

    local_pipeline = sagemaker.local.entities._LocalPipeline(pipeline)
    with pytest.raises(ClientError) as error:
        local_pipeline.start(PipelineParameters={param_str_name: ""})
    assert (
        f'Parameter {param_str_name} value "" is too short (length: 0, required minimum: 1).'
    ) in str(error)
