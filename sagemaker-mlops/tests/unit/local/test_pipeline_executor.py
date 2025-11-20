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
"""Unit tests for local pipeline executor."""
from __future__ import absolute_import

import pytest
import json
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError

from sagemaker.mlops.local.pipeline import (
    LocalPipelineExecutor,
    _TrainingStepExecutor,
    _ProcessingStepExecutor,
    _ConditionStepExecutor,
    _TransformStepExecutor,
    _CreateModelStepExecutor,
    _FailStepExecutor,
    _StepExecutorFactory,
)
from sagemaker.mlops.local.exceptions import StepExecutionException
from sagemaker.mlops.workflow.steps import StepTypeEnum
from sagemaker.core.workflow.parameters import ParameterString, ParameterInteger
from sagemaker.core.workflow.execution_variables import ExecutionVariables
from sagemaker.core.workflow.functions import Join, JsonGet
from sagemaker.core.workflow.properties import Properties


@pytest.fixture
def mock_session():
    session = Mock()
    session.sagemaker_client = Mock()
    session.read_s3_file = Mock()
    return session


@pytest.fixture
def mock_execution():
    execution = Mock()
    execution.pipeline = Mock()
    execution.pipeline.name = "test-pipeline"
    execution.pipeline_parameters = {}
    execution.step_execution = {}
    execution.pipeline_execution_name = "exec-123"
    execution.creation_time = "2024-01-01T00:00:00"
    execution.mark_step_executing = Mock()
    execution.update_step_properties = Mock()
    execution.update_step_failure = Mock()
    execution.update_execution_success = Mock()
    execution.update_execution_failure = Mock()
    return execution


@pytest.fixture
def mock_pipeline_dag():
    dag = Mock()
    dag.step_map = {}
    dag.__iter__ = Mock(return_value=iter([]))
    return dag


class TestLocalPipelineExecutor:
    """Tests for LocalPipelineExecutor class."""

    def test_init(self, mock_execution, mock_session):
        """Test LocalPipelineExecutor initialization."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)

            assert executor.sagemaker_session == mock_session
            assert executor.execution == mock_execution
            assert executor.pipeline_dag is not None
            assert len(executor._blocked_steps) == 0

    def test_execute_empty_pipeline(self, mock_execution, mock_session):
        """Test execute with empty pipeline."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            result = executor.execute()

            assert result == mock_execution
            mock_execution.update_execution_success.assert_called_once()

    def test_execute_with_step_failure(self, mock_execution, mock_session):
        """Test execute handles step execution exception."""
        mock_step = Mock()
        mock_step.name = "failing-step"
        mock_step.step_type = StepTypeEnum.TRAINING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {"failing-step": mock_step}
            mock_dag.__iter__ = Mock(return_value=iter([mock_step]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            with patch.object(LocalPipelineExecutor, '_execute_step') as mock_execute:
                mock_execute.side_effect = StepExecutionException(
                    "failing-step", "Test error"
                )

                executor = LocalPipelineExecutor(mock_execution, mock_session)
                result = executor.execute()

                assert result == mock_execution
                mock_execution.update_execution_failure.assert_called_once_with(
                    "failing-step", "Test error"
                )

    def test_evaluate_pipeline_variable_parameter(self, mock_execution, mock_session):
        """Test evaluate_pipeline_variable with Parameter."""
        param = ParameterString(name="test-param", default_value="default")
        mock_execution.pipeline_parameters = {"test-param": "test-value"}

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            result = executor.evaluate_pipeline_variable(param, "test-step")

            assert result == "test-value"

    def test_evaluate_pipeline_variable_primitive(self, mock_execution, mock_session):
        """Test evaluate_pipeline_variable with primitive value."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            assert executor.evaluate_pipeline_variable("string", "test-step") == "string"
            assert executor.evaluate_pipeline_variable(123, "test-step") == 123
            assert executor.evaluate_pipeline_variable(True, "test-step") is True
            assert executor.evaluate_pipeline_variable(3.14, "test-step") == 3.14

    def test_evaluate_join_function(self, mock_execution, mock_session):
        """Test _evaluate_join_function."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            join_func = Join(on="/", values=["s3://bucket", "prefix", "file.txt"])
            result = executor._evaluate_join_function(join_func, "test-step")

            assert result == "s3://bucket/prefix/file.txt"

    def test_evaluate_execution_variable_pipeline_name(self, mock_execution, mock_session):
        """Test _evaluate_execution_variable for pipeline name."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            result = executor._evaluate_execution_variable(ExecutionVariables.PIPELINE_NAME)
            assert result == "test-pipeline"

    def test_evaluate_execution_variable_pipeline_arn(self, mock_execution, mock_session):
        """Test _evaluate_execution_variable for pipeline ARN."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            result = executor._evaluate_execution_variable(ExecutionVariables.PIPELINE_ARN)
            assert result == "test-pipeline"

    def test_evaluate_execution_variable_execution_id(self, mock_execution, mock_session):
        """Test _evaluate_execution_variable for execution ID."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            result = executor._evaluate_execution_variable(ExecutionVariables.PIPELINE_EXECUTION_ID)
            assert result == "exec-123"

    def test_evaluate_execution_variable_start_datetime(self, mock_execution, mock_session):
        """Test _evaluate_execution_variable for start datetime."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            result = executor._evaluate_execution_variable(ExecutionVariables.START_DATETIME)
            assert result == "2024-01-01T00:00:00"

    def test_evaluate_execution_variable_current_datetime(self, mock_execution, mock_session):
        """Test _evaluate_execution_variable for current datetime."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            result = executor._evaluate_execution_variable(ExecutionVariables.CURRENT_DATETIME)
            # Should return a datetime object
            assert result is not None

    def test_parse_arguments_dict(self, mock_execution, mock_session):
        """Test _parse_arguments with dictionary."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            args = {"key1": "value1", "key2": 123}
            result = executor._parse_arguments(args, "test-step")

            assert result == {"key1": "value1", "key2": 123}

    def test_parse_arguments_list(self, mock_execution, mock_session):
        """Test _parse_arguments with list."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            args = ["value1", 123, True]
            result = executor._parse_arguments(args, "test-step")

            assert result == ["value1", 123, True]

    def test_parse_arguments_nested(self, mock_execution, mock_session):
        """Test _parse_arguments with nested structures."""
        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            
            args = {
                "outer": {
                    "inner": ["value1", 123]
                },
                "list": [{"key": "value"}]
            }
            result = executor._parse_arguments(args, "test-step")

            assert result == args

    def test_evaluate_step_arguments(self, mock_execution, mock_session):
        """Test evaluate_step_arguments."""
        mock_step = Mock()
        mock_step.arguments = {"TrainingJobName": "job-123"}

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            executor = LocalPipelineExecutor(mock_execution, mock_session)
            result = executor.evaluate_step_arguments(mock_step)

            assert result == {"TrainingJobName": "job-123"}


class TestTrainingStepExecutor:
    """Tests for _TrainingStepExecutor class."""

    def test_execute_success(self, mock_execution, mock_session):
        """Test execute creates training job successfully."""
        mock_step = Mock()
        mock_step.name = "training-step"
        mock_step.step_type = StepTypeEnum.TRAINING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={
                "TrainingJobName": "job-123",
                "RoleArn": "arn:aws:iam::123:role/SageMakerRole"
            })

            mock_session.sagemaker_client.create_training_job = Mock()
            mock_session.sagemaker_client.describe_training_job = Mock(return_value={
                "TrainingJobName": "job-123",
                "TrainingJobStatus": "Completed"
            })

            executor = _TrainingStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["TrainingJobName"] == "job-123"
            mock_session.sagemaker_client.create_training_job.assert_called_once()

    def test_execute_failure(self, mock_execution, mock_session):
        """Test execute handles training job failure."""
        mock_step = Mock()
        mock_step.name = "training-step"
        mock_step.step_type = StepTypeEnum.TRAINING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={})

            mock_session.sagemaker_client.create_training_job = Mock(
                side_effect=Exception("Training failed")
            )

            executor = _TrainingStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            # Should return None and update execution with failure
            assert result is None
            mock_execution.update_step_failure.assert_called_once()


class TestProcessingStepExecutor:
    """Tests for _ProcessingStepExecutor class."""

    def test_execute_success(self, mock_execution, mock_session):
        """Test execute creates processing job successfully."""
        mock_step = Mock()
        mock_step.name = "processing-step"
        mock_step.step_type = StepTypeEnum.PROCESSING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={
                "ProcessingJobName": "proc-123"
            })

            mock_session.sagemaker_client.create_processing_job = Mock()
            mock_session.sagemaker_client.describe_processing_job = Mock(return_value={
                "ProcessingJobName": "proc-123",
                "ProcessingJobStatus": "Completed",
                "ProcessingOutputConfig": {
                    "Outputs": [
                        {"OutputName": "output1", "S3Output": {"S3Uri": "s3://bucket/output"}}
                    ]
                },
                "ProcessingInputs": [
                    {"InputName": "input1", "S3Input": {"S3Uri": "s3://bucket/input"}}
                ]
            })

            executor = _ProcessingStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["ProcessingJobName"] == "proc-123"
            # Verify outputs were converted to dict
            assert isinstance(result["ProcessingOutputConfig"]["Outputs"], dict)
            assert "output1" in result["ProcessingOutputConfig"]["Outputs"]


class TestConditionStepExecutor:
    """Tests for _ConditionStepExecutor class."""

    def test_execute_condition_true(self, mock_execution, mock_session):
        """Test execute with condition evaluating to true."""
        mock_step = Mock()
        mock_step.name = "condition-step"
        mock_step.step_type = StepTypeEnum.CONDITION
        mock_step.if_steps = []
        mock_step.else_steps = []
        mock_step.step_only_arguments = {
            "Conditions": [
                {"Type": "Equals", "LeftValue": 1, "RightValue": 1}
            ]
        }

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)

            executor = _ConditionStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["Outcome"] is True

    def test_execute_condition_false(self, mock_execution, mock_session):
        """Test execute with condition evaluating to false."""
        mock_step = Mock()
        mock_step.name = "condition-step"
        mock_step.step_type = StepTypeEnum.CONDITION
        mock_step.if_steps = []
        mock_step.else_steps = []
        mock_step.step_only_arguments = {
            "Conditions": [
                {"Type": "Equals", "LeftValue": 1, "RightValue": 2}
            ]
        }

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)

            executor = _ConditionStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["Outcome"] is False


class TestTransformStepExecutor:
    """Tests for _TransformStepExecutor class."""

    def test_execute_success(self, mock_execution, mock_session):
        """Test execute creates transform job successfully."""
        mock_step = Mock()
        mock_step.name = "transform-step"
        mock_step.step_type = StepTypeEnum.TRANSFORM

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={
                "TransformJobName": "transform-123"
            })

            mock_session.sagemaker_client.create_transform_job = Mock()
            mock_session.sagemaker_client.describe_transform_job = Mock(return_value={
                "TransformJobName": "transform-123",
                "TransformJobStatus": "Completed"
            })

            executor = _TransformStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["TransformJobName"] == "transform-123"


class TestCreateModelStepExecutor:
    """Tests for _CreateModelStepExecutor class."""

    def test_execute_success(self, mock_execution, mock_session):
        """Test execute creates model successfully."""
        mock_step = Mock()
        mock_step.name = "create-model-step"
        mock_step.step_type = StepTypeEnum.CREATE_MODEL

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={})

            mock_session.sagemaker_client.create_model = Mock()
            mock_session.sagemaker_client.describe_model = Mock(return_value={
                "ModelName": "model-123"
            })

            executor = _CreateModelStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            assert result["ModelName"] == "model-123"


class TestFailStepExecutor:
    """Tests for _FailStepExecutor class."""

    def test_execute(self, mock_execution, mock_session):
        """Test execute marks step as failed."""
        mock_step = Mock()
        mock_step.name = "fail-step"
        mock_step.step_type = StepTypeEnum.FAIL

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            pipeline_executor.evaluate_step_arguments = Mock(return_value={
                "ErrorMessage": "Test failure message"
            })

            executor = _FailStepExecutor(pipeline_executor, mock_step)
            result = executor.execute()

            # Should update step properties and then fail
            mock_execution.update_step_properties.assert_called_once()
            mock_execution.update_step_failure.assert_called_once_with(
                "fail-step", "Test failure message"
            )


class TestStepExecutorFactory:
    """Tests for _StepExecutorFactory class."""

    def test_get_training_executor(self, mock_execution, mock_session):
        """Test get returns TrainingStepExecutor for training step."""
        mock_step = Mock()
        mock_step.name = "training-step"
        mock_step.step_type = StepTypeEnum.TRAINING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _TrainingStepExecutor)

    def test_get_processing_executor(self, mock_execution, mock_session):
        """Test get returns ProcessingStepExecutor for processing step."""
        mock_step = Mock()
        mock_step.name = "processing-step"
        mock_step.step_type = StepTypeEnum.PROCESSING

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _ProcessingStepExecutor)

    def test_get_condition_executor(self, mock_execution, mock_session):
        """Test get returns ConditionStepExecutor for condition step."""
        mock_step = Mock()
        mock_step.name = "condition-step"
        mock_step.step_type = StepTypeEnum.CONDITION

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _ConditionStepExecutor)

    def test_get_transform_executor(self, mock_execution, mock_session):
        """Test get returns TransformStepExecutor for transform step."""
        mock_step = Mock()
        mock_step.name = "transform-step"
        mock_step.step_type = StepTypeEnum.TRANSFORM

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _TransformStepExecutor)

    def test_get_create_model_executor(self, mock_execution, mock_session):
        """Test get returns CreateModelStepExecutor for create model step."""
        mock_step = Mock()
        mock_step.name = "create-model-step"
        mock_step.step_type = StepTypeEnum.CREATE_MODEL

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _CreateModelStepExecutor)

    def test_get_fail_executor(self, mock_execution, mock_session):
        """Test get returns FailStepExecutor for fail step."""
        mock_step = Mock()
        mock_step.name = "fail-step"
        mock_step.step_type = StepTypeEnum.FAIL

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            executor = factory.get(mock_step)

            assert isinstance(executor, _FailStepExecutor)

    def test_get_unsupported_step_type(self, mock_execution, mock_session):
        """Test get handles unsupported step type."""
        mock_step = Mock()
        mock_step.name = "unsupported-step"
        mock_step.step_type = StepTypeEnum.LAMBDA  # Unsupported in local mode

        with patch('sagemaker.mlops.local.pipeline.PipelineGraph') as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_dag.__iter__ = Mock(return_value=iter([]))
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            pipeline_executor = LocalPipelineExecutor(mock_execution, mock_session)
            factory = _StepExecutorFactory(pipeline_executor)

            result = factory.get(mock_step)

            # Should return None and update execution with failure
            assert result is None
            mock_execution.update_step_failure.assert_called_once()
