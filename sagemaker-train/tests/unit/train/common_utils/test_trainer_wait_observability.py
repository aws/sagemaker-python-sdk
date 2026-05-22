"""Tests for training job observability prints in script/terminal mode."""
import time
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.common_utils.trainer_wait import wait, _is_unassigned_attribute


class MockUnassigned:
    pass


class MockTrainingJob:
    def __init__(self, status="Completed", failure_reason=None):
        self.training_job_name = "test-sft-job-2026"
        self.training_job_arn = "arn:aws:sagemaker:us-west-2:123456789:training-job/test-sft-job-2026"
        self.training_job_status = status
        self.secondary_status = "Training"
        self.secondary_status_transitions = []
        self.progress_info = MockUnassigned()
        self.failure_reason = failure_reason
        self.mlflow_config = MockUnassigned()
        self._call_count = 0

    def refresh(self):
        self._call_count += 1


class TestTrainingObservabilityAtStart:
    """Test that job info is printed at start in terminal mode."""

    @patch("sagemaker.train.common_utils.trainer_wait._is_jupyter_environment", return_value=False)
    @patch("sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration", return_value=(None, None, None))
    def test_prints_job_info_at_start(self, mock_mlflow, mock_jupyter, capsys):
        job = MockTrainingJob(status="Completed")
        wait(job, poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "Training job started: test-sft-job-2026" in captured.out
        assert "Log group: /aws/sagemaker/TrainingJobs" in captured.out
        assert "Log stream prefix: test-sft-job-2026" in captured.out


class TestTrainingObservabilityOnFailure:
    """Test that debug info is printed on failure."""

    @patch("sagemaker.train.common_utils.trainer_wait._is_jupyter_environment", return_value=False)
    @patch("sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration", return_value=(None, None, None))
    def test_prints_debug_info_on_failure(self, mock_mlflow, mock_jupyter, capsys):
        job = MockTrainingJob(status="Failed", failure_reason="OOM error")
        with pytest.raises(Exception):
            wait(job, poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "Failure reason: OOM error" in captured.out
        assert "Log group: /aws/sagemaker/TrainingJobs" in captured.out
        assert "Log stream prefix: test-sft-job-2026" in captured.out
        assert "CloudWatch Logs:" in captured.out

    @patch("sagemaker.train.common_utils.trainer_wait._is_jupyter_environment", return_value=False)
    @patch("sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration", return_value=(None, None, None))
    def test_prints_cloudwatch_url_on_failure(self, mock_mlflow, mock_jupyter, capsys):
        job = MockTrainingJob(status="Failed", failure_reason="ClientError")
        with pytest.raises(Exception):
            wait(job, poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "us-west-2.console.aws.amazon.com/cloudwatch" in captured.out


class TestTrainingObservabilityOnSuccess:
    """Test that MLflow link is printed on success (existing behavior preserved)."""

    @patch("sagemaker.train.common_utils.trainer_wait._is_jupyter_environment", return_value=False)
    @patch("sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration", return_value=("https://mlflow.example.com", None, None))
    def test_prints_mlflow_on_success(self, mock_mlflow, mock_jupyter, capsys):
        job = MockTrainingJob(status="Completed")
        wait(job, poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "mlflow.example.com" in captured.out
