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
"""Tests for SageMaker Evaluation Execution Module."""
from __future__ import absolute_import

import json
import pytest
import time
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch, ANY, PropertyMock
from botocore.exceptions import ClientError

from sagemaker.core.resources import Pipeline, PipelineExecution
from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

from sagemaker.train.evaluate.execution import (
    EvaluationPipelineExecution,
    BenchmarkEvaluationExecution,
    LLMAJEvaluationExecution,
    StepDetail,
    PipelineExecutionStatus,
    _create_evaluation_pipeline,
    _clean_unassigned_value,
    _clean_unassigned_from_dict,
    _extract_eval_type_from_arn,
    _get_or_create_pipeline,
    _start_pipeline_execution,
    _create_execution_from_pipeline_execution,
    _extract_output_s3_location_from_steps,
)
from sagemaker.train.evaluate.constants import EvalType, _get_pipeline_name, _get_pipeline_name_prefix

# Test constants
DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_PIPELINE_NAME = "SagemakerEvaluation-BenchmarkEvaluation-test-uuid-123"
DEFAULT_PIPELINE_ARN = f"arn:aws:sagemaker:{DEFAULT_REGION}:123456789012:pipeline/{DEFAULT_PIPELINE_NAME}"
DEFAULT_EXECUTION_ARN = f"{DEFAULT_PIPELINE_ARN}/execution/test-execution-123"
DEFAULT_S3_OUTPUT_PATH = "s3://test-bucket/evaluation/output"
DEFAULT_PIPELINE_DEFINITION = json.dumps({"Version": "2020-12-01", "Steps": []})


class MockUnassigned:
    """Mock class to simulate sagemaker_core Unassigned objects."""
    pass


@pytest.fixture
def mock_session():
    """Mock SageMaker session."""
    session = MagicMock()
    session.boto_region_name = DEFAULT_REGION
    session.client.return_value = MagicMock()
    return session


@pytest.fixture
def mock_pipeline():
    """Mock Pipeline object."""
    pipeline = MagicMock(spec=Pipeline)
    pipeline.pipeline_name = DEFAULT_PIPELINE_NAME
    pipeline.pipeline_arn = DEFAULT_PIPELINE_ARN
    return pipeline


@pytest.fixture
def mock_pipeline_execution():
    """Mock PipelineExecution object."""
    pe = MagicMock(spec=PipelineExecution)
    pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
    pe.pipeline_execution_status = "Executing"
    pe.pipeline_arn = DEFAULT_PIPELINE_ARN
    pe.creation_time = datetime.now()
    pe.last_modified_time = datetime.now()
    pe.failure_reason = None
    return pe


# ============================================================================
# Tests for Helper Functions
# ============================================================================

class TestCreateEvaluationPipeline:
    """Tests for _create_evaluation_pipeline function."""

    @patch("sagemaker.train.evaluate.execution._get_pipeline_name")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_create_pipeline_success(self, mock_pipeline_class, mock_get_name, mock_session):
        """Test successful pipeline creation."""
        mock_get_name.return_value = DEFAULT_PIPELINE_NAME
        mock_pipeline = MagicMock()
        mock_pipeline_class.create.return_value = mock_pipeline
        
        result = _create_evaluation_pipeline(
            eval_type=EvalType.BENCHMARK,
            role_arn=DEFAULT_ROLE,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        mock_pipeline_class.create.assert_called_once()
        assert mock_pipeline_class.create.call_args.kwargs["pipeline_name"] == DEFAULT_PIPELINE_NAME
        assert mock_pipeline_class.create.call_args.kwargs["role_arn"] == DEFAULT_ROLE
        assert mock_pipeline_class.create.call_args.kwargs["pipeline_definition"] == DEFAULT_PIPELINE_DEFINITION
        assert result == mock_pipeline

    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_create_pipeline_waits_for_status(self, mock_pipeline_class, mock_session):
        """Test that pipeline waits for active status."""
        mock_pipeline = MagicMock()
        mock_pipeline_class.create.return_value = mock_pipeline
        
        _create_evaluation_pipeline(
            eval_type=EvalType.BENCHMARK,
            role_arn=DEFAULT_ROLE,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        mock_pipeline.wait_for_status.assert_called_once_with(
            target_status="Active",
            poll=5,
            timeout=300
        )

    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_create_pipeline_wait_timeout_continues(self, mock_pipeline_class, mock_session):
        """Test that pipeline creation continues even if wait times out."""
        mock_pipeline = MagicMock()
        mock_pipeline.wait_for_status.side_effect = Exception("Timeout")
        mock_pipeline_class.create.return_value = mock_pipeline
        
        result = _create_evaluation_pipeline(
            eval_type=EvalType.BENCHMARK,
            role_arn=DEFAULT_ROLE,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        # Should still return the pipeline even if wait fails
        assert result == mock_pipeline


class TestCleanUnassignedValue:
    """Tests for _clean_unassigned_value function."""

    def test_clean_unassigned_object(self):
        """Test cleaning an Unassigned object returns None."""
        unassigned = MockUnassigned()
        result = _clean_unassigned_value(unassigned)
        assert result is None

    def test_clean_normal_value(self):
        """Test cleaning a normal value returns it unchanged."""
        assert _clean_unassigned_value("test") == "test"
        assert _clean_unassigned_value(123) == 123
        assert _clean_unassigned_value(None) is None


class TestCleanUnassignedFromDict:
    """Tests for _clean_unassigned_from_dict function."""

    def test_clean_dict_with_failure_reason(self):
        """Test cleaning dict with Unassigned failure_reason."""
        unassigned = MockUnassigned()
        data = {
            "status": {
                "failure_reason": unassigned
            }
        }
        
        result = _clean_unassigned_from_dict(data)
        assert result["status"]["failure_reason"] is None

    def test_clean_dict_without_failure_reason(self):
        """Test cleaning dict without failure_reason."""
        data = {
            "status": {
                "other_field": "value"
            }
        }
        
        result = _clean_unassigned_from_dict(data)
        assert result == data


class TestExtractEvalTypeFromArn:
    """Tests for _extract_eval_type_from_arn function."""

    @pytest.mark.parametrize("eval_type", [
        EvalType.BENCHMARK,
        EvalType.CUSTOM_SCORER,
        EvalType.LLM_AS_JUDGE
    ])
    def test_extract_from_pipeline_arn(self, eval_type):
        """Test extracting eval type from pipeline ARN."""
        pipeline_name = _get_pipeline_name(eval_type, unique_id="test-uuid")
        arn = f"arn:aws:sagemaker:{DEFAULT_REGION}:123456789012:pipeline/{pipeline_name}"
        
        result = _extract_eval_type_from_arn(arn)
        assert result == eval_type

    @pytest.mark.parametrize("eval_type", [
        EvalType.BENCHMARK,
        EvalType.CUSTOM_SCORER,
        EvalType.LLM_AS_JUDGE
    ])
    def test_extract_from_execution_arn(self, eval_type):
        """Test extracting eval type from execution ARN."""
        pipeline_name = _get_pipeline_name(eval_type, unique_id="test-uuid")
        arn = f"arn:aws:sagemaker:{DEFAULT_REGION}:123456789012:pipeline/{pipeline_name}/execution/exec-123"
        
        result = _extract_eval_type_from_arn(arn)
        assert result == eval_type

    def test_extract_invalid_arn(self):
        """Test extracting from invalid ARN returns None."""
        result = _extract_eval_type_from_arn("invalid-arn")
        assert result is None

    def test_extract_unknown_pipeline_name(self):
        """Test extracting from unknown pipeline name returns None."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/unknown-pipeline"
        result = _extract_eval_type_from_arn(arn)
        assert result is None

    def test_extract_with_exception(self):
        """Test extracting handles exceptions gracefully."""
        # Test with malformed ARN that causes exception during parsing
        result = _extract_eval_type_from_arn("arn")
        assert result is None


class TestGetOrCreatePipeline:
    """Tests for _get_or_create_pipeline function."""

    @patch("sagemaker.train.evaluate.execution.Tag")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_get_existing_pipeline_and_update(self, mock_pipeline_class, mock_tag_class, mock_session):
        """Test getting and updating existing pipeline via Pipeline.get_all with prefix."""
        # Mock pipeline with matching tag
        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = DEFAULT_PIPELINE_NAME
        mock_pipeline.pipeline_arn = DEFAULT_PIPELINE_ARN
        mock_pipeline_class.get_all.return_value = iter([mock_pipeline])
        
        # Mock tags
        mock_tag = MagicMock()
        mock_tag.key = "SagemakerModelEvaluation"
        mock_tag.value = "true"
        mock_tag_class.get_all.return_value = iter([mock_tag])
        
        result = _get_or_create_pipeline(
            eval_type=EvalType.BENCHMARK,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        # Should call get_all with prefix
        mock_pipeline_class.get_all.assert_called_once()
        # Should update the found pipeline
        mock_pipeline.update.assert_called_once()
        assert result == mock_pipeline

    @patch("sagemaker.train.evaluate.execution._create_evaluation_pipeline")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_create_pipeline_when_not_found(self, mock_pipeline_class, mock_create, mock_session):
        """Test creating pipeline when it doesn't exist."""
        error_response = {"Error": {"Code": "ResourceNotFound"}}
        mock_pipeline_class.get.side_effect = ClientError(error_response, "DescribePipeline")
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        result = _get_or_create_pipeline(
            eval_type=EvalType.BENCHMARK,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        mock_create.assert_called_once_with(
            EvalType.BENCHMARK,
            DEFAULT_ROLE,
            DEFAULT_PIPELINE_DEFINITION,
            mock_session,
            DEFAULT_REGION,
            []
        )
        assert result == mock_pipeline

    @patch("sagemaker.train.evaluate.execution._create_evaluation_pipeline")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_create_pipeline_when_not_found_with_jumpstart_tags(self, mock_pipeline_class, mock_create, mock_session):
        """Test creating pipeline when it doesn't exist."""
        error_response = {"Error": {"Code": "ResourceNotFound"}}
        mock_pipeline_class.get.side_effect = ClientError(error_response, "DescribePipeline")
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        create_tags = [{"key": "sagemaker-sdk:jumpstart-model-id", "value": "dummy-js-model-id"}]
        
        result = _get_or_create_pipeline(
            eval_type=EvalType.BENCHMARK,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session,
            region=DEFAULT_REGION,
            create_tags=create_tags
        )
        
        mock_create.assert_called_once_with(
            EvalType.BENCHMARK,
            DEFAULT_ROLE,
            DEFAULT_PIPELINE_DEFINITION,
            mock_session,
            DEFAULT_REGION,
            create_tags
        )
        assert result == mock_pipeline

    @patch("sagemaker.train.evaluate.execution._create_evaluation_pipeline")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    def test_fallback_to_create_on_exception(self, mock_pipeline_class, mock_create, mock_session):
        """Test fallback to create when Pipeline.get raises generic exception."""
        mock_pipeline_class.get.side_effect = Exception("Generic error")
        mock_pipeline = MagicMock()
        mock_create.return_value = mock_pipeline
        
        result = _get_or_create_pipeline(
            eval_type=EvalType.BENCHMARK,
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        mock_create.assert_called_once()
        assert result == mock_pipeline


class TestStartPipelineExecution:
    """Tests for _start_pipeline_execution function."""

    @patch("boto3.client")
    def test_start_execution_without_session(self, mock_boto3_client):
        """Test starting pipeline execution without session."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": DEFAULT_EXECUTION_ARN
        }
        
        result = _start_pipeline_execution(
            pipeline_name=DEFAULT_PIPELINE_NAME,
            name="test-execution",
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result == DEFAULT_EXECUTION_ARN
        mock_boto3_client.assert_called_once_with(
            'sagemaker',
            region_name=DEFAULT_REGION,
            endpoint_url=None
        )
        mock_client.start_pipeline_execution.assert_called_once()

    def test_start_execution_with_session(self, mock_session):
        """Test starting pipeline execution with session."""
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": DEFAULT_EXECUTION_ARN
        }
        
        result = _start_pipeline_execution(
            pipeline_name=DEFAULT_PIPELINE_NAME,
            name="test-execution",
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        assert result == DEFAULT_EXECUTION_ARN
        mock_client.start_pipeline_execution.assert_called_once()


class TestCreateExecutionFromPipelineExecution:
    """Tests for _create_execution_from_pipeline_execution function."""

    def test_create_execution_basic(self, mock_pipeline_execution):
        """Test creating execution from pipeline execution."""
        result = _create_execution_from_pipeline_execution(
            pe=mock_pipeline_execution,
            eval_type=EvalType.BENCHMARK
        )
        
        assert isinstance(result, EvaluationPipelineExecution)
        assert result.arn == DEFAULT_EXECUTION_ARN
        assert result.status.overall_status == "Executing"
        assert result.eval_type == EvalType.BENCHMARK

    def test_create_execution_with_unassigned_failure_reason(self):
        """Test creating execution with Unassigned failure reason."""
        pe = MagicMock()
        pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        pe.pipeline_execution_status = "Failed"
        pe.failure_reason = MockUnassigned()
        pe.last_modified_time = datetime.now()
        
        result = _create_execution_from_pipeline_execution(pe, EvalType.BENCHMARK)
        
        assert result.status.failure_reason is None


class TestExtractOutputS3LocationFromSteps:
    """Tests for _extract_output_s3_location_from_steps function."""

    @patch("boto3.client")
    def test_extract_from_custom_model_step(self, mock_boto3_client):
        """Test extracting S3 path from EvaluateCustomModel step."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.describe_training_job.return_value = {
            "OutputDataConfig": {
                "S3OutputPath": DEFAULT_S3_OUTPUT_PATH
            }
        }
        
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateCustomModel"
        mock_step.metadata.training_job.arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result == DEFAULT_S3_OUTPUT_PATH

    @patch("boto3.client")
    def test_extract_from_base_model_step(self, mock_boto3_client):
        """Test extracting S3 path from EvaluateBaseModel step."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.describe_training_job.return_value = {
            "OutputDataConfig": {
                "S3OutputPath": DEFAULT_S3_OUTPUT_PATH
            }
        }
        
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateBaseModel"
        mock_step.metadata.training_job.arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result == DEFAULT_S3_OUTPUT_PATH

    def test_extract_no_evaluation_steps(self):
        """Test extracting when no evaluation steps exist."""
        mock_step = MagicMock()
        mock_step.step_name = "OtherStep"
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result is None

    @patch("boto3.client")
    def test_extract_describe_training_job_fails(self, mock_boto3_client):
        """Test extracting when DescribeTrainingJob fails."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        error_response = {"Error": {"Code": "ResourceNotFound"}}
        mock_client.describe_training_job.side_effect = ClientError(error_response, "DescribeTrainingJob")
        
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateCustomModel"
        mock_step.metadata.training_job.arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result is None

    @patch("boto3.client")
    def test_extract_with_generic_exception(self, mock_boto3_client):
        """Test extracting handles generic exceptions gracefully."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.describe_training_job.side_effect = Exception("Unexpected error")
        
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateCustomModel"
        mock_step.metadata.training_job.arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/test-job"
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result is None

    @patch("boto3.client")
    def test_extract_with_malformed_step(self, mock_boto3_client):
        """Test extracting with step that raises exception on attribute access."""
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateCustomModel"
        # Make metadata.training_job.arn raise an exception
        type(mock_step).metadata = PropertyMock(side_effect=Exception("Attribute error"))
        
        result = _extract_output_s3_location_from_steps(
            raw_steps=[mock_step],
            session=None,
            region=DEFAULT_REGION
        )
        
        assert result is None


# ============================================================================
# Tests for Pydantic Models
# ============================================================================

class TestStepDetail:
    """Tests for StepDetail model."""

    def test_create_step_detail(self):
        """Test creating StepDetail with all fields."""
        step = StepDetail(
            name="TestStep",
            status="Completed",
            start_time="2024-01-01T00:00:00Z",
            end_time="2024-01-01T00:01:00Z",
            display_name="Test Step",
            failure_reason=None
        )
        
        assert step.name == "TestStep"
        assert step.status == "Completed"
        assert step.start_time == "2024-01-01T00:00:00Z"

    def test_create_step_detail_minimal(self):
        """Test creating StepDetail with minimal fields."""
        step = StepDetail(
            name="TestStep",
            status="Waiting"
        )
        
        assert step.name == "TestStep"
        assert step.status == "Waiting"
        assert step.start_time is None
        assert step.end_time is None


class TestPipelineExecutionStatus:
    """Tests for PipelineExecutionStatus model."""

    def test_create_status(self):
        """Test creating PipelineExecutionStatus."""
        status = PipelineExecutionStatus(
            overall_status="Executing",
            step_details=[],
            failure_reason=None
        )
        
        assert status.overall_status == "Executing"
        assert len(status.step_details) == 0

    def test_create_status_with_steps(self):
        """Test creating status with step details."""
        step = StepDetail(name="Step1", status="Completed")
        status = PipelineExecutionStatus(
            overall_status="Executing",
            step_details=[step]
        )
        
        assert len(status.step_details) == 1
        assert status.step_details[0].name == "Step1"


# ============================================================================
# Tests for EvaluationPipelineExecution Class
# ============================================================================

class TestEvaluationPipelineExecutionStart:
    """Tests for EvaluationPipelineExecution.start() method."""

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    @patch("sagemaker.train.evaluate.execution._start_pipeline_execution")
    @patch("sagemaker.train.evaluate.execution._get_or_create_pipeline")
    def test_start_execution_success(
        self, mock_get_pipeline, mock_start, mock_pe_class, mock_session
    ):
        """Test successfully starting a pipeline execution."""
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_start.return_value = DEFAULT_EXECUTION_ARN
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Executing"
        mock_pe.creation_time = datetime.now()
        mock_pe_class.get.return_value = mock_pe
        
        execution = EvaluationPipelineExecution.start(
            eval_type=EvalType.BENCHMARK,
            name="test-evaluation",
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            s3_output_path=DEFAULT_S3_OUTPUT_PATH,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        assert execution.arn == DEFAULT_EXECUTION_ARN
        assert execution.eval_type == EvalType.BENCHMARK
        assert execution.s3_output_path == DEFAULT_S3_OUTPUT_PATH
        assert isinstance(execution, BenchmarkEvaluationExecution)

    @patch("sagemaker.train.evaluate.execution._get_or_create_pipeline")
    def test_start_execution_invalid_json(self, mock_get_pipeline, mock_session):
        """Test starting with invalid JSON pipeline definition."""
        with pytest.raises(ValueError, match="Invalid pipeline definition JSON"):
            EvaluationPipelineExecution.start(
                eval_type=EvalType.BENCHMARK,
                name="test-evaluation",
                pipeline_definition="invalid json",
                role_arn=DEFAULT_ROLE,
                session=mock_session
            )

    @patch("sagemaker.train.evaluate.execution._start_pipeline_execution")
    @patch("sagemaker.train.evaluate.execution._get_or_create_pipeline")
    def test_start_execution_client_error(
        self, mock_get_pipeline, mock_start, mock_session
    ):
        """Test handling ClientError during start."""
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        mock_start.side_effect = ClientError(error_response, "StartPipelineExecution")
        
        execution = EvaluationPipelineExecution.start(
            eval_type=EvalType.BENCHMARK,
            name="test-evaluation",
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session
        )
        
        assert execution.status.overall_status == "Failed"
        assert "Access denied" in execution.status.failure_reason

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    @patch("sagemaker.train.evaluate.execution._start_pipeline_execution")
    @patch("sagemaker.train.evaluate.execution._get_or_create_pipeline")
    def test_start_execution_unexpected_error(
        self, mock_get_pipeline, mock_start, mock_pe_class, mock_session
    ):
        """Test handling unexpected error during start."""
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_start.side_effect = Exception("Unexpected error")
        
        execution = EvaluationPipelineExecution.start(
            eval_type=EvalType.BENCHMARK,
            name="test-evaluation",
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session
        )
        
        assert execution.status.overall_status == "Failed"
        assert "Unexpected error" in execution.status.failure_reason


class TestEvaluationPipelineExecutionGet:
    """Tests for EvaluationPipelineExecution.get() method."""

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_execution_success(self, mock_pe_class, mock_session):
        """Test successfully getting a pipeline execution."""
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.pipeline_arn = DEFAULT_PIPELINE_ARN
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        mock_pe_class.get.return_value = mock_pe
        
        execution = EvaluationPipelineExecution.get(
            arn=DEFAULT_EXECUTION_ARN,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        assert execution.arn == DEFAULT_EXECUTION_ARN
        assert execution.status.overall_status == "Succeeded"
        assert execution.eval_type == EvalType.BENCHMARK

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_execution_client_error(self, mock_pe_class, mock_session):
        """Test handling ClientError when getting execution."""
        error_response = {"Error": {"Code": "ResourceNotFound", "Message": "Not found"}}
        mock_pe_class.get.side_effect = ClientError(error_response, "DescribePipelineExecution")
        
        execution = EvaluationPipelineExecution.get(
            arn=DEFAULT_EXECUTION_ARN,
            session=mock_session
        )
        
        assert execution.status.overall_status == "Error"
        assert "Not found" in execution.status.failure_reason

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_execution_generic_exception(self, mock_pe_class, mock_session):
        """Test handling generic exception when getting execution."""
        mock_pe_class.get.side_effect = Exception("Unexpected error")
        
        execution = EvaluationPipelineExecution.get(
            arn=DEFAULT_EXECUTION_ARN,
            session=mock_session
        )
        
        assert execution.status.overall_status == "Error"
        assert "Unexpected error" in execution.status.failure_reason


class TestEvaluationPipelineExecutionGetAll:
    """Tests for EvaluationPipelineExecution.get_all() method."""

    @patch("sagemaker.train.evaluate.execution.Tag")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_executions(self, mock_pe_class, mock_pipeline_class, mock_tag_class, mock_session):
        """Test getting all executions."""
        # Mock pipeline with correct name and ARN
        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = DEFAULT_PIPELINE_NAME
        mock_pipeline.pipeline_arn = DEFAULT_PIPELINE_ARN
        mock_pipeline_class.get_all.return_value = iter([mock_pipeline])
        
        # Mock tags with required tag
        mock_tag = MagicMock()
        mock_tag.key = "SagemakerModelEvaluation"
        mock_tag.value = "true"
        mock_tag_class.get_all.return_value = iter([mock_tag])
        
        # Mock pipeline execution
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        mock_pe_class.get_all.return_value = iter([mock_pe])
        
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=mock_session,
            region=DEFAULT_REGION
        ))
        
        assert len(executions) == 1
        assert executions[0].arn == DEFAULT_EXECUTION_ARN
        
        # Verify Pipeline.get_all was called with prefix
        mock_pipeline_class.get_all.assert_called_once()
        # Verify Tag.get_all was called to validate pipeline
        mock_tag_class.get_all.assert_called_once()
        # Verify PipelineExecution.get_all was called with the pipeline name
        mock_pe_class.get_all.assert_called_once()

    @patch("sagemaker.train.evaluate.execution.Tag")
    @patch("sagemaker.train.evaluate.execution.Pipeline")
    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_multiple_eval_types(self, mock_pe_class, mock_pipeline_class, mock_tag_class, mock_session):
        """Test getting all executions for multiple eval types."""
        # Mock pipeline for BENCHMARK type
        mock_pipeline = MagicMock()
        mock_pipeline.pipeline_name = DEFAULT_PIPELINE_NAME
        mock_pipeline.pipeline_arn = DEFAULT_PIPELINE_ARN
        
        # Pipeline.get_all called 3 times (once per eval type)
        # Return pipeline for BENCHMARK, empty for others
        mock_pipeline_class.get_all.side_effect = [
            iter([mock_pipeline]),  # BENCHMARK - found
            iter([]),              # CUSTOM_SCORER - not found
            iter([])               # LLM_AS_JUDGE - not found
        ]
        
        # Mock tags with required tag (only called for BENCHMARK since others have no pipelines)
        mock_tag = MagicMock()
        mock_tag.key = "SagemakerModelEvaluation"
        mock_tag.value = "true"
        mock_tag_class.get_all.return_value = iter([mock_tag])
        
        # Mock pipeline execution for BENCHMARK
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        mock_pe_class.get_all.return_value = iter([mock_pe])
        
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=None,
            session=mock_session
        ))
        
        assert len(executions) == 1
        assert mock_pipeline_class.get_all.call_count == 3  # Called for each eval type

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_pipeline_not_found(self, mock_pe_class, mock_session):
        """Test get_all when pipeline doesn't exist."""
        error_response = {"Error": {"Code": "ResourceNotFound"}}
        mock_pe_class.get_all.side_effect = ClientError(error_response, "ListPipelineExecutions")
        
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=mock_session
        ))
        
        assert len(executions) == 0

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_with_validation_exception(self, mock_pe_class, mock_session):
        """Test get_all with ValidationException."""
        error_response = {"Error": {"Code": "ValidationException", "Message": "Invalid input"}}
        mock_pe_class.get_all.side_effect = ClientError(error_response, "ListPipelineExecutions")
        
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=mock_session
        ))
        
        # Should continue and return empty list
        assert len(executions) == 0

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_with_generic_exception(self, mock_pe_class, mock_session):
        """Test get_all handles generic exceptions."""
        mock_pe_class.get_all.side_effect = Exception("Unexpected error")
        
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=mock_session
        ))
        
        # Should handle exception and return empty list
        assert len(executions) == 0

    @patch("sagemaker.train.evaluate.execution._create_execution_from_pipeline_execution")
    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_get_all_with_exception_during_enrichment(self, mock_pe_class, mock_create_exec, mock_session):
        """Test get_all handles exceptions during execution enrichment."""
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe_class.get_all.return_value = iter([mock_pe])
        
        # Make _create_execution_from_pipeline_execution raise an exception
        mock_create_exec.side_effect = Exception("Enrichment error")
        
        # Should catch the exception and continue
        executions = list(EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=mock_session
        ))
        
        # Should return empty list due to exception
        assert len(executions) == 0


class TestEvaluationPipelineExecutionRefresh:
    """Tests for EvaluationPipelineExecution.refresh() method."""

    def test_refresh_execution(self):
        """Test refreshing execution status."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        execution.refresh()
        
        assert execution.status.overall_status == "Succeeded"
        mock_pe.refresh.assert_called_once()

    def test_refresh_without_pipeline_execution(self):
        """Test refresh when no pipeline execution is set."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Unknown")
        )
        
        # Should not raise an error
        execution.refresh()
        assert execution.status.overall_status == "Unknown"

    def test_refresh_with_client_error(self):
        """Test refresh handling ClientError."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        error_response = {"Error": {"Code": "ResourceNotFound", "Message": "Not found"}}
        mock_pe.refresh.side_effect = ClientError(error_response, "DescribePipelineExecution")
        
        execution._pipeline_execution = mock_pe
        
        # Should not raise, just log error
        execution.refresh()

    def test_refresh_with_generic_exception(self):
        """Test refresh handling generic exception."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.refresh.side_effect = Exception("Unexpected error")
        
        execution._pipeline_execution = mock_pe
        
        # Should not raise, just log error
        execution.refresh()


class TestEvaluationPipelineExecutionStop:
    """Tests for EvaluationPipelineExecution.stop() method."""

    @patch("boto3.client")
    def test_stop_execution(self, mock_boto3_client):
        """Test stopping a pipeline execution."""
        execution = EvaluationPipelineExecution(
            arn=DEFAULT_EXECUTION_ARN,
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Stopping"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        mock_pe._session = MagicMock()
        mock_pe._session.boto_session.client.return_value = mock_client
        
        execution._pipeline_execution = mock_pe
        execution.stop()
        
        assert execution.status.overall_status == "Stopping"

    def test_stop_without_arn(self):
        """Test stop when no ARN is set."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        # Should not raise an error
        execution.stop()

    @patch("boto3.client")
    def test_stop_with_client_error(self, mock_boto3_client):
        """Test stop handling ClientError."""
        execution = EvaluationPipelineExecution(
            arn=DEFAULT_EXECUTION_ARN,
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_client = MagicMock()
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Access denied"}}
        mock_client.stop_pipeline_execution.side_effect = ClientError(error_response, "StopPipelineExecution")
        mock_boto3_client.return_value = mock_client
        
        # Should not raise, just log error
        execution.stop()

    @patch("boto3.client")
    def test_stop_with_generic_exception(self, mock_boto3_client):
        """Test stop handling generic exception."""
        execution = EvaluationPipelineExecution(
            arn=DEFAULT_EXECUTION_ARN,
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_client = MagicMock()
        mock_client.stop_pipeline_execution.side_effect = Exception("Unexpected error")
        mock_boto3_client.return_value = mock_client
        
        # Should not raise, just log error
        execution.stop()

    @patch("boto3.client")
    def test_stop_with_session_client(self, mock_boto3_client):
        """Test stop with session that doesn't have boto_session attribute."""
        execution = EvaluationPipelineExecution(
            arn=DEFAULT_EXECUTION_ARN,
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Stopping"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        mock_pe._session = MagicMock()
        # Session without boto_session attribute
        delattr(mock_pe._session, 'boto_session')
        mock_pe._session.client.return_value = mock_client
        
        execution._pipeline_execution = mock_pe
        execution.stop()
        
        assert execution.status.overall_status == "Stopping"


class TestEvaluationPipelineExecutionWait:
    """Tests for EvaluationPipelineExecution.wait() method."""

    def test_wait_reaches_target_status(self):
        """Test wait method when target status is reached."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Succeeded")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        
        # Should return immediately since already at target status
        execution.wait(target_status="Succeeded", poll=1)

    def test_wait_fails_on_failed_status(self):
        """Test wait raises exception when execution fails."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Failed"
        mock_pe.failure_reason = "Test failure"
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        
        with pytest.raises(FailedStatusError):
            execution.wait(target_status="Succeeded", poll=1)

    @patch("time.time")
    def test_wait_timeout_exceeded(self, mock_time):
        """Test wait raises exception on timeout."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Executing"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        
        # Mock time to simulate timeout
        mock_time.side_effect = [0, 10, 20, 30, 40, 50, 60]  # Exceeds timeout
        
        with pytest.raises(TimeoutExceededError):
            execution.wait(target_status="Succeeded", poll=1, timeout=5)

    def test_wait_without_pipeline_execution(self):
        """Test wait when no pipeline execution is set."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        # Should return immediately without error
        execution.wait()


class TestEvaluationPipelineExecutionConvertToSubclass:
    """Tests for EvaluationPipelineExecution._convert_to_subclass() method."""

    def test_convert_to_benchmark(self):
        """Test converting to BenchmarkEvaluationExecution."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Succeeded"),
            eval_type=EvalType.BENCHMARK
        )
        
        result = execution._convert_to_subclass(EvalType.BENCHMARK)
        
        assert isinstance(result, BenchmarkEvaluationExecution)
        assert result.name == "test"

    def test_convert_to_custom_scorer(self):
        """Test converting to BenchmarkEvaluationExecution (custom scorer uses same class)."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Succeeded"),
            eval_type=EvalType.CUSTOM_SCORER
        )
        
        result = execution._convert_to_subclass(EvalType.CUSTOM_SCORER)
        
        assert isinstance(result, BenchmarkEvaluationExecution)

    def test_convert_to_llmaj(self):
        """Test converting to LLMAJEvaluationExecution."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Succeeded"),
            eval_type=EvalType.LLM_AS_JUDGE
        )
        
        result = execution._convert_to_subclass(EvalType.LLM_AS_JUDGE)
        
        assert isinstance(result, LLMAJEvaluationExecution)

    def test_convert_preserves_pipeline_execution(self):
        """Test that conversion preserves internal pipeline execution reference."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Succeeded"),
            eval_type=EvalType.BENCHMARK
        )
        
        mock_pe = MagicMock()
        execution._pipeline_execution = mock_pe
        
        result = execution._convert_to_subclass(EvalType.BENCHMARK)
        
        assert result._pipeline_execution == mock_pe


class TestEvaluationPipelineExecutionUpdateStepDetails:
    """Tests for EvaluationPipelineExecution._update_step_details_from_raw_steps() method."""

    def test_update_step_details_from_steps(self):
        """Test updating step details from raw steps."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_step = MagicMock()
        mock_step.step_name = "TestStep"
        mock_step.step_status = "Completed"
        mock_step.start_time = datetime.now()
        mock_step.end_time = datetime.now()
        mock_step.step_display_name = "Test Step Display"
        mock_step.failure_reason = None
        
        execution._update_step_details_from_raw_steps([mock_step])
        
        assert len(execution.status.step_details) == 1
        assert execution.status.step_details[0].name == "TestStep"
        assert execution.status.step_details[0].status == "Completed"

    def test_update_step_details_handles_unassigned(self):
        """Test updating step details with Unassigned objects."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_step = MagicMock()
        mock_step.step_name = "TestStep"
        mock_step.step_status = "Completed"
        mock_step.start_time = None
        mock_step.end_time = None
        mock_step.step_display_name = MockUnassigned()
        mock_step.failure_reason = MockUnassigned()
        
        execution._update_step_details_from_raw_steps([mock_step])
        
        assert len(execution.status.step_details) == 1
        assert execution.status.step_details[0].display_name is None
        assert execution.status.step_details[0].failure_reason is None

    def test_update_step_details_handles_errors(self):
        """Test updating step details handles errors gracefully."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_step = MagicMock()
        mock_step.step_name.side_effect = Exception("Error")
        
        # Should not raise, just skip the problematic step
        execution._update_step_details_from_raw_steps([mock_step])
        
        assert len(execution.status.step_details) == 0


class TestEvaluationPipelineExecutionEnrichWithStepDetails:
    """Tests for EvaluationPipelineExecution._enrich_with_step_details() method."""

    @patch("sagemaker.train.evaluate.execution._extract_output_s3_location_from_steps")
    def test_enrich_with_step_details(self, mock_extract):
        """Test enriching execution with step details."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_step = MagicMock()
        mock_step.step_name = "TestStep"
        mock_step.step_status = "Completed"
        mock_step.start_time = None
        mock_step.end_time = None
        mock_step.step_display_name = "Test Step"
        mock_step.failure_reason = None
        
        mock_pe.get_all_steps.return_value = iter([mock_step])
        mock_extract.return_value = DEFAULT_S3_OUTPUT_PATH
        
        execution._pipeline_execution = mock_pe
        execution._enrich_with_step_details()
        
        assert len(execution.status.step_details) == 1
        assert execution.s3_output_path == DEFAULT_S3_OUTPUT_PATH

    def test_enrich_without_pipeline_execution(self):
        """Test enrich when no pipeline execution is set."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        # Should not raise an error
        execution._enrich_with_step_details()

    @patch("sagemaker.train.evaluate.execution._extract_output_s3_location_from_steps")
    def test_enrich_with_exception(self, mock_extract):
        """Test enrich handles exceptions gracefully."""
        execution = EvaluationPipelineExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.get_all_steps.side_effect = Exception("Unexpected error")
        
        execution._pipeline_execution = mock_pe
        
        # Should not raise, just log warning
        execution._enrich_with_step_details()


# ============================================================================
# Tests for Subclasses
# ============================================================================

class TestBenchmarkEvaluationExecution:
    """Tests for BenchmarkEvaluationExecution subclass."""

    def test_show_results_success(self):
        """Test show_results when execution succeeded - skipped (tests external dependency)."""
        pytest.skip("Skipping test for external show_results_utils module")

    def test_show_results_not_succeeded(self):
        """Test show_results raises error when not succeeded."""
        execution = BenchmarkEvaluationExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Executing")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Executing"
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        
        with pytest.raises(ValueError, match="Cannot show results"):
            execution.show_results()


class TestLLMAJEvaluationExecution:
    """Tests for LLMAJEvaluationExecution subclass."""

    def test_show_results_success(self):
        """Test show_results when execution succeeded - skipped (tests external dependency)."""
        pytest.skip("Skipping test for external show_results_utils module")

    def test_show_results_not_succeeded(self):
        """Test show_results raises error when not succeeded."""
        execution = LLMAJEvaluationExecution(
            name="test",
            status=PipelineExecutionStatus(overall_status="Failed")
        )
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Failed"
        mock_pe.failure_reason = "Test failure"
        mock_pe.last_modified_time = datetime.now()
        mock_pe.get_all_steps.return_value = iter([])
        
        execution._pipeline_execution = mock_pe
        
        with pytest.raises(ValueError, match="Cannot show results"):
            execution.show_results()

    def test_show_results_default_params(self):
        """Test show_results with default parameters - skipped (tests external dependency)."""
        pytest.skip("Skipping test for external show_results_utils module")


# ============================================================================
# Integration-style Tests
# ============================================================================

class TestEvaluationPipelineExecutionIntegration:
    """Integration-style tests for complete workflows."""

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    @patch("sagemaker.train.evaluate.execution._start_pipeline_execution")
    @patch("sagemaker.train.evaluate.execution._get_or_create_pipeline")
    def test_complete_start_workflow(
        self, mock_get_pipeline, mock_start, mock_pe_class, mock_session
    ):
        """Test complete workflow from start to execution."""
        # Setup mocks
        mock_pipeline = MagicMock()
        mock_get_pipeline.return_value = mock_pipeline
        mock_start.return_value = DEFAULT_EXECUTION_ARN
        
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_status = "Executing"
        mock_pe.creation_time = datetime.now()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe_class.get.return_value = mock_pe
        
        # Start execution
        execution = EvaluationPipelineExecution.start(
            eval_type=EvalType.LLM_AS_JUDGE,
            name="test-evaluation",
            pipeline_definition=DEFAULT_PIPELINE_DEFINITION,
            role_arn=DEFAULT_ROLE,
            session=mock_session,
            region=DEFAULT_REGION
        )
        
        # Verify correct subclass was created
        assert isinstance(execution, LLMAJEvaluationExecution)
        assert execution.arn == DEFAULT_EXECUTION_ARN
        assert execution.eval_type == EvalType.LLM_AS_JUDGE
        
        # Verify pipeline was created/updated
        mock_get_pipeline.assert_called_once()
        
        # Verify execution was started
        mock_start.assert_called_once()

    @patch("sagemaker.train.evaluate.execution.PipelineExecution")
    def test_complete_get_workflow(self, mock_pe_class, mock_session):
        """Test complete workflow for getting an execution."""
        # Setup mock
        mock_pe = MagicMock()
        mock_pe.pipeline_execution_arn = DEFAULT_EXECUTION_ARN
        mock_pe.pipeline_execution_status = "Succeeded"
        mock_pe.pipeline_arn = DEFAULT_PIPELINE_ARN
        mock_pe.failure_reason = None
        mock_pe.last_modified_time = datetime.now()
        
        mock_step = MagicMock()
        mock_step.step_name = "EvaluateCustomModel"
        mock_step.step_status = "Succeeded"
        mock_step.start_time = datetime.now()
        mock_step.end_time = datetime.now()
        mock_step.step_display_name = "Evaluate Custom Model"
        mock_step.failure_reason = None
        
        mock_pe.get_all_steps.return_value = iter([mock_step])
        mock_pe_class.get.return_value = mock_pe
        
        # Get execution
        execution = EvaluationPipelineExecution.get(
            arn=DEFAULT_EXECUTION_ARN,
            session=mock_session
        )
        
        # Verify execution was retrieved
        assert execution.arn == DEFAULT_EXECUTION_ARN
        assert execution.status.overall_status == "Succeeded"
        assert execution.eval_type == EvalType.BENCHMARK
        
        # Verify steps were populated
        assert len(execution.status.step_details) == 1
        assert execution.status.step_details[0].name == "EvaluateCustomModel"



# Additional tests for improved coverage - removed as they don't add significant value
