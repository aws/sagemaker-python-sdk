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
"""Unit tests for local pipeline entities."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, MagicMock, patch
from botocore.exceptions import ClientError
from datetime import datetime

from sagemaker.mlops.local.pipeline_entities import (
    _LocalPipeline,
    _LocalPipelineExecution,
    _LocalPipelineExecutionStep,
    _LocalExecutionStatus,
)
from sagemaker.mlops.local.exceptions import StepExecutionException
from sagemaker.mlops.workflow.steps import StepTypeEnum


@pytest.fixture
def mock_pipeline():
    pipeline = Mock()
    pipeline.name = "test-pipeline"
    pipeline.parameters = []
    pipeline.definition = Mock(return_value='{"Steps": []}')
    return pipeline


@pytest.fixture
def mock_local_session():
    session = Mock()
    session.sagemaker_client = Mock()
    return session


class TestLocalPipeline:
    """Tests for _LocalPipeline class."""

    def test_init_with_session(self, mock_pipeline, mock_local_session):
        """Test _LocalPipeline initialization with session."""
        local_pipeline = _LocalPipeline(
            pipeline=mock_pipeline,
            pipeline_description="Test description",
            local_session=mock_local_session,
        )

        assert local_pipeline.pipeline == mock_pipeline
        assert local_pipeline.pipeline_description == "Test description"
        assert local_pipeline.local_session == mock_local_session
        assert local_pipeline.creation_time > 0
        assert local_pipeline.last_modified_time == local_pipeline.creation_time

    def test_init_without_session(self, mock_pipeline):
        """Test _LocalPipeline initialization without session creates default."""
        with patch("sagemaker.core.local.LocalSession") as mock_session_class:
            mock_session_instance = Mock()
            mock_session_class.return_value = mock_session_instance

            local_pipeline = _LocalPipeline(pipeline=mock_pipeline)

            assert local_pipeline.local_session == mock_session_instance
            mock_session_class.assert_called_once()

    def test_describe(self, mock_pipeline, mock_local_session):
        """Test describe returns pipeline metadata."""
        local_pipeline = _LocalPipeline(
            pipeline=mock_pipeline,
            pipeline_description="Test description",
            local_session=mock_local_session,
        )

        result = local_pipeline.describe()

        assert result["PipelineArn"] == "test-pipeline"
        assert result["PipelineName"] == "test-pipeline"
        assert result["PipelineDescription"] == "Test description"
        assert result["PipelineStatus"] == "Active"
        assert result["RoleArn"] == "<no_role>"
        assert "CreationTime" in result
        assert "LastModifiedTime" in result
        assert "PipelineDefinition" in result

    def test_start_creates_execution(self, mock_pipeline, mock_local_session):
        """Test start creates and executes a pipeline execution."""
        # Make pipeline.steps iterable and parameters empty
        mock_pipeline.steps = []
        mock_pipeline.parameters = []
        
        with patch("sagemaker.mlops.local.pipeline.LocalPipelineExecutor") as mock_executor:
            mock_execution_result = Mock()
            mock_executor_instance = Mock()
            mock_executor_instance.execute = Mock(return_value=mock_execution_result)
            mock_executor.return_value = mock_executor_instance

            local_pipeline = _LocalPipeline(
                pipeline=mock_pipeline, local_session=mock_local_session
            )
            initial_modified_time = local_pipeline.last_modified_time

            result = local_pipeline.start()

            assert result == mock_execution_result
            assert len(local_pipeline._executions) == 1
            assert local_pipeline.last_modified_time > initial_modified_time
            mock_executor.assert_called_once()
            mock_executor_instance.execute.assert_called_once()


class TestLocalPipelineExecution:
    """Tests for _LocalPipelineExecution class."""

    @pytest.fixture
    def mock_pipeline_with_params(self):
        pipeline = Mock()
        pipeline.name = "test-pipeline"
        
        param1 = Mock()
        param1.name = "param1"
        param1.default_value = "default1"
        param1.parameter_type = Mock()
        param1.parameter_type.python_type = str
        
        pipeline.parameters = [param1]
        return pipeline

    def test_init(self, mock_pipeline, mock_local_session):
        """Test _LocalPipelineExecution initialization."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                PipelineExecutionDescription="Test execution",
                PipelineExecutionDisplayName="Test Display",
                local_session=mock_local_session,
            )

            assert execution.pipeline == mock_pipeline
            assert execution.pipeline_execution_name == "exec-123"
            assert execution.pipeline_execution_description == "Test execution"
            assert execution.pipeline_execution_display_name == "Test Display"
            assert execution.status == _LocalExecutionStatus.EXECUTING.value
            assert execution.failure_reason is None
            assert execution.creation_time > 0

    def test_describe(self, mock_pipeline, mock_local_session):
        """Test describe returns execution metadata."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            result = execution.describe()

            assert result["PipelineArn"] == "test-pipeline"
            assert result["PipelineExecutionArn"] == "exec-123"
            assert result["PipelineExecutionStatus"] == "Executing"
            assert "CreationTime" in result
            assert "LastModifiedTime" in result
            # FailureReason should not be in result when None
            assert "FailureReason" not in result

    def test_list_steps_empty(self, mock_pipeline, mock_local_session):
        """Test list_steps with no steps."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            result = execution.list_steps()

            assert result == {"PipelineExecutionSteps": []}

    def test_update_execution_success(self, mock_pipeline, mock_local_session):
        """Test update_execution_success marks execution as succeeded."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )
            initial_time = execution.last_modified_time

            execution.update_execution_success()

            assert execution.status == _LocalExecutionStatus.SUCCEEDED.value
            assert execution.last_modified_time > initial_time

    def test_update_execution_failure(self, mock_pipeline, mock_local_session):
        """Test update_execution_failure marks execution as failed."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            execution.update_execution_failure("test-step", "Test error message")

            assert execution.status == _LocalExecutionStatus.FAILED.value
            assert "test-step" in execution.failure_reason
            assert "Test error message" in execution.failure_reason

    def test_update_step_properties(self, mock_pipeline, mock_local_session):
        """Test update_step_properties updates step properties."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            
            # Create a proper Step mock
            from sagemaker.mlops.workflow.steps import Step
            mock_step = Mock(spec=Step)
            mock_step.name = "test-step"
            mock_step.step_type = StepTypeEnum.TRAINING
            mock_step.description = "Test step"
            mock_step.display_name = "Test Display"
            
            mock_dag.step_map = {"test-step": mock_step}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            properties = {"TrainingJobName": "job-123"}
            execution.update_step_properties("test-step", properties)

            assert execution.step_execution["test-step"].properties == properties
            assert execution.step_execution["test-step"].status == _LocalExecutionStatus.SUCCEEDED.value

    def test_update_step_failure(self, mock_pipeline, mock_local_session):
        """Test update_step_failure marks step as failed and raises exception."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            
            # Create a proper Step mock
            from sagemaker.mlops.workflow.steps import Step
            mock_step = Mock(spec=Step)
            mock_step.name = "test-step"
            mock_step.step_type = StepTypeEnum.TRAINING
            mock_step.description = "Test step"
            mock_step.display_name = "Test Display"
            
            mock_dag.step_map = {"test-step": mock_step}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            # update_step_failure raises StepExecutionException
            with pytest.raises(StepExecutionException):
                execution.update_step_failure("test-step", "Test failure")

            assert execution.step_execution["test-step"].status == _LocalExecutionStatus.FAILED.value
            assert execution.step_execution["test-step"].failure_reason == "Test failure"

    def test_mark_step_executing(self, mock_pipeline, mock_local_session):
        """Test mark_step_executing updates step status."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            
            # Create a proper Step mock
            from sagemaker.mlops.workflow.steps import Step
            mock_step = Mock(spec=Step)
            mock_step.name = "test-step"
            mock_step.step_type = StepTypeEnum.TRAINING
            mock_step.description = "Test step"
            mock_step.display_name = "Test Display"
            
            mock_dag.step_map = {"test-step": mock_step}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline,
                local_session=mock_local_session,
            )

            execution.mark_step_executing("test-step")

            assert execution.step_execution["test-step"].status == _LocalExecutionStatus.EXECUTING.value
            assert execution.step_execution["test-step"].start_time is not None

    def test_initialize_parameters_with_defaults(self, mock_pipeline_with_params, mock_local_session):
        """Test parameter initialization uses defaults when no overrides."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline_with_params,
                local_session=mock_local_session,
            )

            assert execution.pipeline_parameters == {"param1": "default1"}

    def test_initialize_parameters_with_overrides(self, mock_pipeline_with_params, mock_local_session):
        """Test parameter initialization with overrides."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            execution = _LocalPipelineExecution(
                execution_id="exec-123",
                pipeline=mock_pipeline_with_params,
                PipelineParameters={"param1": "override1"},
                local_session=mock_local_session,
            )

            assert execution.pipeline_parameters == {"param1": "override1"}

    def test_initialize_parameters_unknown_parameter(self, mock_pipeline_with_params, mock_local_session):
        """Test parameter initialization raises error for unknown parameter."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            with pytest.raises(ClientError) as exc_info:
                _LocalPipelineExecution(
                    execution_id="exec-123",
                    pipeline=mock_pipeline_with_params,
                    PipelineParameters={"unknown_param": "value"},
                    local_session=mock_local_session,
                )

            assert "Unknown parameter" in str(exc_info.value)

    def test_initialize_parameters_wrong_type(self, mock_pipeline_with_params, mock_local_session):
        """Test parameter initialization raises error for wrong type."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            with pytest.raises(ClientError) as exc_info:
                _LocalPipelineExecution(
                    execution_id="exec-123",
                    pipeline=mock_pipeline_with_params,
                    PipelineParameters={"param1": 123},  # Should be string
                    local_session=mock_local_session,
                )

            assert "Unexpected type" in str(exc_info.value)

    def test_initialize_parameters_empty_string(self, mock_pipeline_with_params, mock_local_session):
        """Test parameter initialization raises error for empty string."""
        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            with pytest.raises(ClientError) as exc_info:
                _LocalPipelineExecution(
                    execution_id="exec-123",
                    pipeline=mock_pipeline_with_params,
                    PipelineParameters={"param1": ""},
                    local_session=mock_local_session,
                )

            assert "too short" in str(exc_info.value)

    def test_initialize_parameters_missing_required(self, mock_local_session):
        """Test parameter initialization raises error for missing required parameter."""
        pipeline = Mock()
        pipeline.name = "test-pipeline"
        
        param1 = Mock()
        param1.name = "param1"
        param1.default_value = None  # No default
        param1.parameter_type = Mock()
        param1.parameter_type.python_type = str
        
        pipeline.parameters = [param1]

        with patch("sagemaker.mlops.workflow.pipeline.PipelineGraph") as mock_graph:
            mock_dag = Mock()
            mock_dag.step_map = {}
            mock_graph.from_pipeline = Mock(return_value=mock_dag)

            with pytest.raises(ClientError) as exc_info:
                _LocalPipelineExecution(
                    execution_id="exec-123",
                    pipeline=pipeline,
                    local_session=mock_local_session,
                )

            assert "is undefined" in str(exc_info.value)


class TestLocalPipelineExecutionStep:
    """Tests for _LocalPipelineExecutionStep class."""

    def test_init(self):
        """Test _LocalPipelineExecutionStep initialization."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
            display_name="Test Display",
        )

        assert step.name == "test-step"
        assert step.type == StepTypeEnum.TRAINING
        assert step.description == "Test description"
        assert step.display_name == "Test Display"
        assert step.status is None
        assert step.failure_reason is None
        assert step.properties == {}

    def test_update_step_properties(self):
        """Test update_step_properties updates properties and status."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
        )

        properties = {"TrainingJobName": "job-123", "TrainingJobStatus": "Completed"}
        step.update_step_properties(properties)

        assert step.properties == properties
        assert step.status == _LocalExecutionStatus.SUCCEEDED.value
        assert step.end_time is not None

    def test_update_step_failure(self):
        """Test update_step_failure raises StepExecutionException."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
        )

        with pytest.raises(StepExecutionException) as exc_info:
            step.update_step_failure("Test failure message")

        assert step.status == _LocalExecutionStatus.FAILED.value
        assert step.failure_reason == "Test failure message"
        assert step.end_time is not None
        assert exc_info.value.step_name == "test-step"
        assert exc_info.value.message == "Test failure message"

    def test_mark_step_executing(self):
        """Test mark_step_executing updates status and start time."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
        )

        step.mark_step_executing()

        assert step.status == _LocalExecutionStatus.EXECUTING.value
        assert step.start_time is not None

    def test_to_list_steps_response_training(self):
        """Test to_list_steps_response for training step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
            display_name="Test Display",
        )
        step.mark_step_executing()
        step.update_step_properties({"TrainingJobName": "job-123"})

        result = step.to_list_steps_response()

        assert result["StepName"] == "test-step"
        assert result["StepDescription"] == "Test description"
        assert result["StepDisplayName"] == "Test Display"
        assert result["StepStatus"] == "Succeeded"
        assert "Metadata" in result
        assert result["Metadata"]["TrainingJob"]["Arn"] == "job-123"

    def test_to_list_steps_response_processing(self):
        """Test to_list_steps_response for processing step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.PROCESSING,
            description="Test description",
        )
        step.update_step_properties({"ProcessingJobName": "proc-123"})

        result = step.to_list_steps_response()

        assert result["Metadata"]["ProcessingJob"]["Arn"] == "proc-123"

    def test_to_list_steps_response_transform(self):
        """Test to_list_steps_response for transform step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRANSFORM,
            description="Test description",
        )
        step.update_step_properties({"TransformJobName": "transform-123"})

        result = step.to_list_steps_response()

        assert result["Metadata"]["TransformJob"]["Arn"] == "transform-123"

    def test_to_list_steps_response_create_model(self):
        """Test to_list_steps_response for create model step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.CREATE_MODEL,
            description="Test description",
        )
        step.update_step_properties({"ModelName": "model-123"})

        result = step.to_list_steps_response()

        assert result["Metadata"]["Model"]["Arn"] == "model-123"

    def test_to_list_steps_response_condition(self):
        """Test to_list_steps_response for condition step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.CONDITION,
            description="Test description",
        )
        step.update_step_properties({"Outcome": True})

        result = step.to_list_steps_response()

        assert result["Metadata"]["Condition"]["Outcome"] is True

    def test_to_list_steps_response_fail(self):
        """Test to_list_steps_response for fail step."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.FAIL,
            description="Test description",
        )
        step.update_step_properties({"ErrorMessage": "Test error"})

        result = step.to_list_steps_response()

        assert result["Metadata"]["Fail"]["ErrorMessage"] == "Test error"

    def test_to_list_steps_response_no_properties(self):
        """Test to_list_steps_response with no properties."""
        step = _LocalPipelineExecutionStep(
            name="test-step",
            step_type=StepTypeEnum.TRAINING,
            description="Test description",
        )
        step.mark_step_executing()

        result = step.to_list_steps_response()

        assert "Metadata" not in result
        assert result["StepStatus"] == "Executing"


class TestLocalExecutionStatus:
    """Tests for _LocalExecutionStatus enum."""

    def test_enum_values(self):
        """Test enum has correct values."""
        assert _LocalExecutionStatus.EXECUTING.value == "Executing"
        assert _LocalExecutionStatus.SUCCEEDED.value == "Succeeded"
        assert _LocalExecutionStatus.FAILED.value == "Failed"
