"""Unit tests for stream_logs() on AgentRFTJob, MultiTurnRLTrainer, and evaluators."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.agent_rft_job import AgentRFTJob
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.evaluate.base_evaluator import BaseEvaluator


def _make_client_error(code, message="error"):
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "operation_name",
    )


def _make_mock_job(**overrides):
    job = MagicMock()
    job.job_name = "test-mtrl-job"
    job.job_arn = "arn:aws:sagemaker:us-west-2:123456789012:job/test-mtrl-job"
    job.job_status = "Training"
    job.job_category = "AgentRFT"
    job.job_config_document = "{}"
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


class TestAgentRFTJobStreamLogs:
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_stream_logs_exits_on_completed(self, mock_get_session):
        """stream_logs exits when job status reaches Completed."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_get_session.return_value = mock_session

        logs_client = mock_session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "test-mtrl-job/algo-1"}]}
        ]
        logs_client.get_log_events.return_value = {
            "events": [{"timestamp": 1700000000000, "message": "done\n"}],
            "nextForwardToken": "token-1",
        }

        mock_job = _make_mock_job()
        # Job becomes Completed after refresh
        mock_job.refresh.side_effect = lambda: setattr(mock_job, "job_status", "Completed")

        rft_job = AgentRFTJob(mock_job)

        with patch("time.sleep"):
            rft_job.stream_logs(poll=1)

        # Verify it exited (didn't hang)
        assert mock_job.refresh.called

    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_stream_logs_raises_on_access_denied(self, mock_get_session):
        """stream_logs propagates AccessDeniedException."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_get_session.return_value = mock_session

        logs_client = mock_session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "AccessDeniedException"
        )

        mock_job = _make_mock_job()
        rft_job = AgentRFTJob(mock_job)

        with pytest.raises(ClientError) as exc_info:
            rft_job.stream_logs(poll=1)
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    def test_stream_logs_handles_resource_not_found_terminal(self, mock_get_session):
        """stream_logs returns cleanly when log group not found and job is terminal."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_get_session.return_value = mock_session

        logs_client = mock_session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.side_effect = _make_client_error(
            "ResourceNotFoundException"
        )

        mock_job = _make_mock_job(job_status="Failed")
        rft_job = AgentRFTJob(mock_job)

        with patch("time.sleep"):
            rft_job.stream_logs(poll=1)

    def test_stream_logs_validates_poll(self):
        """stream_logs raises ValueError for invalid poll."""
        mock_job = _make_mock_job()
        rft_job = AgentRFTJob(mock_job)

        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            rft_job.stream_logs(poll=0)

        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            rft_job.stream_logs(poll=500)

    def test_stream_logs_validates_start_time_type(self):
        """stream_logs raises TypeError for invalid start_time."""
        mock_job = _make_mock_job()
        rft_job = AgentRFTJob(mock_job)

        with pytest.raises(TypeError, match="start_time must be datetime or int"):
            rft_job.stream_logs(start_time="2023-01-01")


class TestMultiTurnRLTrainerStreamLogs:
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.core.resources.Job.get")
    def test_stream_logs_uses_correct_log_group(self, mock_job_get, mock_get_session):
        """MultiTurnRLTrainer.stream_logs uses /aws/sagemaker/Job/AgentRFT."""
        mock_session = MagicMock()
        mock_session.boto_session.region_name = "us-west-2"
        mock_get_session.return_value = mock_session

        logs_client = mock_session.boto_session.client.return_value
        logs_client.get_paginator.return_value.paginate.return_value = [
            {"logStreams": [{"logStreamName": "my-job/algo-1"}]}
        ]
        logs_client.get_log_events.return_value = {
            "events": [],
            "nextForwardToken": "token-1",
        }

        # Job.get returns terminal status
        mock_job_obj = MagicMock()
        mock_job_obj.job_status = "Completed"
        mock_job_get.return_value = mock_job_obj

        # Create trainer with _latest_job set
        trainer = MagicMock(spec=MultiTurnRLTrainer)
        trainer._latest_job = MagicMock()
        trainer._latest_job.job_name = "my-job"
        trainer.sagemaker_session = None

        # Call the actual method
        with patch("time.sleep"):
            MultiTurnRLTrainer.stream_logs(trainer, poll=1)

        # Verify Job.get was called with correct category
        mock_job_get.assert_called_with(job_name="my-job", job_category="AgentRFT")

    def test_stream_logs_raises_when_no_job(self):
        """MultiTurnRLTrainer.stream_logs raises ValueError if no job exists."""
        trainer = MagicMock(spec=MultiTurnRLTrainer)
        trainer._latest_job = None

        with pytest.raises(ValueError, match="No training job found"):
            MultiTurnRLTrainer.stream_logs(trainer)


class TestBaseEvaluatorStreamLogs:
    def test_stream_logs_raises_when_no_execution(self):
        """BaseEvaluator.stream_logs raises ValueError if no evaluation executed."""
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator._latest_execution = None

        with pytest.raises(ValueError, match="No evaluation executed yet"):
            BaseEvaluator.stream_logs(evaluator)

    def test_stream_logs_validates_poll(self):
        """BaseEvaluator.stream_logs validates poll parameter."""
        evaluator = MagicMock(spec=BaseEvaluator)
        evaluator._latest_execution = MagicMock()

        with pytest.raises(ValueError, match="poll must be between 1 and 300"):
            BaseEvaluator.stream_logs(evaluator, poll=0)

    def test_log_group_for_training_job_arn(self):
        """_log_group_for_step_arn resolves training-job ARNs correctly."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/my-eval-job"
        assert BaseEvaluator._log_group_for_step_arn(arn) == "/aws/sagemaker/TrainingJobs"

    def test_log_group_for_job_arn(self):
        """_log_group_for_step_arn resolves Job API ARNs correctly."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:job/my-eval-job"
        assert BaseEvaluator._log_group_for_step_arn(arn) == "/aws/sagemaker/Job/AgentRFTEvaluation"

    def test_job_name_from_training_job_arn(self):
        """_job_name_from_arn extracts job name from training-job ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:training-job/my-eval-job-abc123"
        assert BaseEvaluator._job_name_from_arn(arn) == "my-eval-job-abc123"

    def test_job_name_from_job_arn(self):
        """_job_name_from_arn extracts job name from Job API ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:job/my-mtrl-eval-job"
        assert BaseEvaluator._job_name_from_arn(arn) == "my-mtrl-eval-job"

    def test_job_name_from_unknown_arn_returns_none(self):
        """_job_name_from_arn returns None for unrecognized ARN format."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/my-pipeline"
        assert BaseEvaluator._job_name_from_arn(arn) is None
