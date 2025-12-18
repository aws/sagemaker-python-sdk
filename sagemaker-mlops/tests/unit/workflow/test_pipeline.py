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
"""Unit tests for workflow pipeline."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch

from sagemaker.mlops.workflow.pipeline import Pipeline, PipelineGraph, format_start_parameters
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


@pytest.fixture
def mock_session():
    session = Mock()
    session.boto_session.client.return_value = Mock()
    session.sagemaker_client = Mock()
    session.local_mode = False
    return session


@pytest.fixture
def mock_step():
    step = Mock(spec=Step)
    step.name = "test-step"
    step.step_type = StepTypeEnum.TRAINING
    step.to_request.return_value = {"Name": "test-step", "Type": "Training"}
    return step


def test_pipeline_init(mock_session, mock_step):
    pipeline = Pipeline(
        name="test-pipeline",
        steps=[mock_step],
        sagemaker_session=mock_session
    )
    assert pipeline.name == "test-pipeline"
    assert len(pipeline.steps) == 1
    assert pipeline.sagemaker_session == mock_session


def test_pipeline_create_without_role_raises_error(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value=None):
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            pipeline.create()


def test_format_start_parameters():
    params = {"param1": "value1", "param2": 123}
    result = format_start_parameters(params)
    assert result == [{"Name": "param1", "Value": "value1"}, {"Name": "param2", "Value": "123"}]


def test_format_start_parameters_none():
    assert format_start_parameters(None) is None


def test_pipeline_graph_detects_cycle():
    step1 = Mock(spec=Step)
    step1.name = "step1"
    step1._find_step_dependencies.return_value = ["step2"]
    
    step2 = Mock(spec=Step)
    step2.name = "step2"
    step2._find_step_dependencies.return_value = ["step1"]
    
    with pytest.raises(ValueError, match="Cycle detected"):
        PipelineGraph([step1, step2])


def test_pipeline_create_local_mode(mock_session, mock_step):
    mock_session.local_mode = True
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        pipeline.create(role_arn="role-arn", description="test", parallelism_config={"MaxParallelExecutionSteps": 2})
        mock_session.sagemaker_client.create_pipeline.assert_called_once()


def test_pipeline_create_large_definition(mock_session, mock_step):
    mock_session.local_mode = False
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    large_definition = "x" * (1024 * 101)
    with patch.object(pipeline, "definition", return_value=large_definition):
        with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
            with patch("sagemaker.mlops.workflow.pipeline.s3.determine_bucket_and_prefix", return_value=("bucket", "key")):
                with patch("sagemaker.mlops.workflow.pipeline.s3.S3Uploader.upload_string_as_file_body"):
                    pipeline.create(role_arn="role-arn")
                    mock_session.sagemaker_client.create_pipeline.assert_called_once()


def test_pipeline_update_without_role_raises_error(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value=None):
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            pipeline.update()


def test_pipeline_update_local_mode(mock_session, mock_step):
    mock_session.local_mode = True
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        pipeline.update(role_arn="role-arn", description="test", parallelism_config={"MaxParallelExecutionSteps": 2})
        mock_session.sagemaker_client.update_pipeline.assert_called_once()


def test_pipeline_upsert_without_role_raises_error(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value=None):
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            pipeline.upsert()


def test_pipeline_upsert_existing_pipeline(mock_session, mock_step):
    from botocore.exceptions import ClientError
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    error = ClientError({"Error": {"Code": "ValidationException", "Message": "already exists"}}, "create_pipeline")
    update_response = {"PipelineArn": "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test"}
    mock_session.sagemaker_client.list_tags.return_value = {"Tags": [{"Key": "old", "Value": "tag"}]}
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        with patch("sagemaker.mlops.workflow.pipeline.format_tags", return_value=[{"Key": "new", "Value": "tag"}]):
            with patch.object(pipeline, "create", side_effect=error):
                with patch.object(pipeline, "update", return_value=update_response):
                    pipeline.upsert(role_arn="role-arn", tags=[{"Key": "new", "Value": "tag"}])
                    mock_session.sagemaker_client.add_tags.assert_called_once()


def test_pipeline_start_with_selective_execution(mock_session, mock_step):
    from sagemaker.mlops.workflow.selective_execution_config import SelectiveExecutionConfig
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    mock_session.sagemaker_client.start_pipeline_execution.return_value = {"PipelineExecutionArn": "arn"}
    mock_session.sagemaker_client.list_pipeline_executions.return_value = {
        "PipelineExecutionSummaries": [{"PipelineExecutionArn": "latest-arn"}]
    }
    
    config = SelectiveExecutionConfig(selected_steps=["step1"], reference_latest_execution=True)
    pipeline.start(selective_execution_config=config)
    mock_session.sagemaker_client.start_pipeline_execution.assert_called_once()


def test_pipeline_start_local_mode(mock_session, mock_step):
    mock_session.local_mode = True
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    pipeline.start(parameters={"param": "value"})
    mock_session.sagemaker_client.start_pipeline_execution.assert_called_once()


def test_pipeline_get_latest_execution_arn_none(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    mock_session.sagemaker_client.list_pipeline_executions.return_value = {"PipelineExecutionSummaries": []}
    
    result = pipeline._get_latest_execution_arn()
    assert result is None


def test_pipeline_build_parameters_from_execution(mock_session, mock_step):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    mock_session.sagemaker_client.list_pipeline_parameters_for_execution.return_value = {
        "PipelineParameters": [{"Name": "param1", "Value": "value1"}]
    }
    
    result = pipeline.build_parameters_from_execution("arn", {"param1": "new_value"})
    assert result == {"param1": "new_value"}


def test_pipeline_validate_parameter_overrides_invalid():
    from sagemaker.mlops.workflow.pipeline import Pipeline
    
    with pytest.raises(ValueError, match="not present in the pipeline execution"):
        Pipeline._validate_parameter_overrides("arn", {"param1": "value1"}, {"param2": "value2"})


def test_pipeline_put_triggers_without_role_raises_error(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value=None):
        with pytest.raises(ValueError, match="AWS IAM role is required"):
            pipeline.put_triggers([])


def test_pipeline_put_triggers_empty_list_raises_error(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        with pytest.raises(TypeError, match="No Triggers provided"):
            pipeline.put_triggers([])


def test_pipeline_put_triggers_pipeline_not_exists(mock_session, mock_step):
    from botocore.exceptions import ClientError
    from sagemaker.mlops.workflow.triggers import PipelineSchedule
    
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    error = ClientError({"Error": {"Code": "ResourceNotFound"}}, "describe_pipeline")
    mock_session.sagemaker_client.describe_pipeline.side_effect = error
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        with pytest.raises(RuntimeError, match="does not exist"):
            pipeline.put_triggers([PipelineSchedule(rate=(1, "hour"))], role_arn="role-arn")


def test_pipeline_put_triggers_unsupported_type(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    mock_session.sagemaker_client.describe_pipeline.return_value = {"PipelineArn": "arn"}
    
    with patch("sagemaker.mlops.workflow.pipeline.resolve_value_from_config", return_value="role-arn"):
        with patch("sagemaker.mlops.workflow.pipeline.validate_default_parameters_for_schedules"):
            with pytest.raises(TypeError, match="Unsupported TriggerType"):
                pipeline.put_triggers([Mock()], role_arn="role-arn")


def test_pipeline_describe_trigger_empty_name(mock_session, mock_step):
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with pytest.raises(TypeError, match="No trigger name provided"):
        pipeline.describe_trigger("")


def test_pipeline_describe_trigger_success(mock_session, mock_step):
    from datetime import datetime
    import pytz
    
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    mock_schedule = {
        "Arn": "arn", 
        "ScheduleExpression": "rate(1 hour)", 
        "State": "ENABLED",
        "StartDate": datetime.now(tz=pytz.utc),
        "Target": {"RoleArn": "role-arn"}
    }
    pipeline._event_bridge_scheduler_helper.describe_schedule = Mock(return_value=mock_schedule)
    
    result = pipeline.describe_trigger("trigger-name")
    assert "Schedule_Arn" in result


def test_pipeline_delete_triggers_not_found(mock_session, mock_step):
    from botocore.exceptions import ClientError
    
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    error = ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "delete_schedule")
    pipeline._event_bridge_scheduler_helper.delete_schedule = Mock(side_effect=error)
    
    pipeline.delete_triggers(["trigger-name"])


def test_pipeline_execution_stop(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    
    execution = _PipelineExecution(arn="arn", sagemaker_session=mock_session)
    execution.stop()
    mock_session.sagemaker_client.stop_pipeline_execution.assert_called_once()


def test_pipeline_execution_describe(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    
    execution = _PipelineExecution(arn="arn", sagemaker_session=mock_session)
    execution.describe()
    mock_session.sagemaker_client.describe_pipeline_execution.assert_called_once()


def test_pipeline_execution_list_steps(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    
    mock_session.sagemaker_client.list_pipeline_execution_steps.return_value = {"PipelineExecutionSteps": []}
    execution = _PipelineExecution(arn="arn", sagemaker_session=mock_session)
    result = execution.list_steps()
    assert result == []


def test_pipeline_execution_list_parameters(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    
    execution = _PipelineExecution(arn="arn", sagemaker_session=mock_session)
    execution.list_parameters(max_results=10, next_token="token")
    mock_session.sagemaker_client.list_pipeline_parameters_for_execution.assert_called_once()


def test_pipeline_execution_wait(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    import botocore.waiter
    
    execution = _PipelineExecution(arn="arn", sagemaker_session=mock_session)
    with patch("botocore.waiter.create_waiter_with_client") as mock_waiter:
        mock_waiter.return_value.wait = Mock()
        execution.wait(delay=10, max_attempts=5)
        mock_waiter.return_value.wait.assert_called_once()


def test_get_function_step_result_invalid_step(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    
    with pytest.raises(ValueError, match="Invalid step name"):
        get_function_step_result("invalid", [], "exec-id", mock_session)


def test_get_function_step_result_local_mode_no_metadata():
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    from sagemaker.core.local.local_session import LocalSession
    
    local_session = Mock(spec=LocalSession)
    step_list = [{"StepName": "step1"}]
    
    with pytest.raises(RuntimeError, match="not in Completed status"):
        get_function_step_result("step1", step_list, "exec-id", local_session)


def test_get_function_step_result_wrong_step_type(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    
    step_list = [{"StepName": "step1", "Metadata": {"Processing": {"Arn": "arn"}}}]
    
    with pytest.raises(ValueError, match="@step decorator"):
        get_function_step_result("step1", step_list, "exec-id", mock_session)


def test_get_function_step_result_wrong_container(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    
    step_list = [{"StepName": "step1", "Metadata": {"TrainingJob": {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/job"}}}]
    mock_session.describe_training_job.return_value = {
        "AlgorithmSpecification": {"ContainerEntrypoint": ["python"]},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/path"}
    }
    
    with pytest.raises(ValueError, match="@step decorator"):
        get_function_step_result("step1", step_list, "exec-id", mock_session)


def test_get_function_step_result_incomplete_job(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    from sagemaker.train.remote_function.job import JOBS_CONTAINER_ENTRYPOINT
    from sagemaker.train.remote_function.errors import RemoteFunctionError
    
    step_list = [{"StepName": "step1", "Metadata": {"TrainingJob": {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/job"}}}]
    mock_session.describe_training_job.return_value = {
        "AlgorithmSpecification": {"ContainerEntrypoint": JOBS_CONTAINER_ENTRYPOINT},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/path"},
        "TrainingJobStatus": "Failed",
    }
    
    with pytest.raises(RemoteFunctionError, match="not in Completed status"):
        get_function_step_result("step1", step_list, "exec-id", mock_session)


def test_get_function_step_result_success(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    from sagemaker.train.remote_function.job import JOBS_CONTAINER_ENTRYPOINT
    
    step_list = [{"StepName": "step1", "Metadata": {"TrainingJob": {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/job"}}}]
    mock_session.describe_training_job.return_value = {
        "AlgorithmSpecification": {"ContainerEntrypoint": JOBS_CONTAINER_ENTRYPOINT},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/path/exec-id/step1/results"},
        "TrainingJobStatus": "Completed",
    }
    
    with patch("sagemaker.mlops.workflow.pipeline.deserialize_obj_from_s3", return_value="result"):
        result = get_function_step_result("step1", step_list, "exec-id", mock_session)
        assert result == "result"


def test_pipeline_graph_from_pipeline(mock_session, mock_step):
    from sagemaker.mlops.workflow.pipeline import PipelineGraph
    
    mock_step._find_step_dependencies.return_value = []
    pipeline = Pipeline(name="test-pipeline", steps=[mock_step], sagemaker_session=mock_session)
    
    with patch("sagemaker.mlops.workflow.pipeline.StepsCompiler") as mock_compiler:
        mock_compiler.return_value.build.return_value = [mock_step]
        graph = PipelineGraph.from_pipeline(pipeline)
        assert graph is not None


def test_pipeline_graph_get_steps_in_sub_dag_invalid_step(mock_step):
    from sagemaker.mlops.workflow.pipeline import PipelineGraph
    
    mock_step._find_step_dependencies.return_value = []
    graph = PipelineGraph([mock_step])
    
    invalid_step = Mock(spec=Step)
    invalid_step.name = "invalid"
    
    with pytest.raises(ValueError, match="does not exist"):
        graph.get_steps_in_sub_dag(invalid_step)


def test_pipeline_graph_iteration(mock_step):
    from sagemaker.mlops.workflow.pipeline import PipelineGraph
    
    mock_step._find_step_dependencies.return_value = []
    graph = PipelineGraph([mock_step])
    
    steps = list(graph)
    assert len(steps) == 1


def test_pipeline_execution_result_waiter_error(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    from botocore.exceptions import WaiterError
    
    execution = _PipelineExecution(arn="arn:aws:sagemaker:us-west-2:123456789012:pipeline/test/execution/exec-id", sagemaker_session=mock_session)
    
    with patch.object(execution, "wait", side_effect=WaiterError("name", "reason", {})):
        with pytest.raises(WaiterError):
            execution.result("step1")


def test_pipeline_execution_result_terminal_failure(mock_session):
    from sagemaker.mlops.workflow.pipeline import _PipelineExecution
    from botocore.exceptions import WaiterError
    from sagemaker.train.remote_function.job import JOBS_CONTAINER_ENTRYPOINT
    
    execution = _PipelineExecution(arn="arn:aws:sagemaker:us-west-2:123456789012:pipeline/test/execution/exec-id", sagemaker_session=mock_session)
    mock_session.sagemaker_client.list_pipeline_execution_steps.return_value = {
        "PipelineExecutionSteps": [{"StepName": "step1", "Metadata": {"TrainingJob": {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/job"}}}]
    }
    mock_session.describe_training_job.return_value = {
        "AlgorithmSpecification": {"ContainerEntrypoint": JOBS_CONTAINER_ENTRYPOINT},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/path/exec-id/step1/results"},
        "TrainingJobStatus": "Completed",
    }
    
    with patch.object(execution, "wait", side_effect=WaiterError("name", "Waiter encountered a terminal failure state", {})):
        with patch("sagemaker.mlops.workflow.pipeline.deserialize_obj_from_s3", return_value="result"):
            result = execution.result("step1")
            assert result == "result"


def test_get_function_step_result_obsolete_s3_path(mock_session):
    from sagemaker.mlops.workflow.pipeline import get_function_step_result
    from sagemaker.train.remote_function.job import JOBS_CONTAINER_ENTRYPOINT
    
    step_list = [{"StepName": "step1", "Metadata": {"TrainingJob": {"Arn": "arn:aws:sagemaker:us-west-2:123456789012:training-job/job"}}}]
    mock_session.describe_training_job.return_value = {
        "AlgorithmSpecification": {"ContainerEntrypoint": JOBS_CONTAINER_ENTRYPOINT},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/different/path"},
        "TrainingJobStatus": "Completed",
    }
    
    with patch("sagemaker.mlops.workflow.pipeline.deserialize_obj_from_s3", return_value="result"):
        result = get_function_step_result("step1", step_list, "exec-id", mock_session)
        assert result == "result"
