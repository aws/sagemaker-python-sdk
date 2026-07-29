"""Tests for eval pipeline observability prints in terminal mode."""
from unittest.mock import patch, MagicMock

import pytest

from sagemaker.train.evaluate.execution import (
    EvaluationPipelineExecution,
    PipelineExecutionStatus,
    StepDetail,
)


def _make_execution(status="Succeeded", step_details=None, failure_reason=None, s3_output_path=None):
    exec_obj = EvaluationPipelineExecution(
        name="benchmark-eval-mmlu",
        arn="arn:aws:sagemaker:us-west-2:123456789:pipeline/sm-eval-benchmark-abc/execution/exec-123",
        status=PipelineExecutionStatus(
            overall_status=status,
            step_details=step_details or [],
            failure_reason=failure_reason,
        ),
        s3_output_path=s3_output_path,
    )
    exec_obj._pipeline_execution = MagicMock()
    return exec_obj


class TestEvalObservabilityAtStart:
    @patch("sagemaker.train.evaluate.execution.time.sleep")
    @patch.object(EvaluationPipelineExecution, "refresh")
    def test_prints_pipeline_info_at_start(self, mock_refresh, mock_sleep, capsys):
        exec_obj = _make_execution(status="Succeeded")
        exec_obj.wait(poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "Evaluation started: benchmark-eval-mmlu" in captured.out
        assert "Pipeline: sm-eval-benchmark-abc" in captured.out
        assert "Execution ARN:" in captured.out


class TestEvalObservabilityStepTransitions:
    @patch("sagemaker.train.evaluate.execution.time.sleep")
    @patch.object(EvaluationPipelineExecution, "refresh")
    def test_prints_step_transitions(self, mock_refresh, mock_sleep, capsys):
        steps = [
            StepDetail(name="EvaluateBaseModel", status="Succeeded", display_name="EvaluateBaseModel",
                      start_time="2026-01-01T00:00:00Z", end_time="2026-01-01T00:01:00Z",
                      job_arn="arn:aws:sagemaker:us-west-2:123456789:training-job/eval-base-xyz"),
        ]
        exec_obj = _make_execution(status="Succeeded", step_details=steps)
        exec_obj.wait(poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "✓ EvaluateBaseModel: Succeeded" in captured.out
        assert "(60.0s)" in captured.out

    @patch("sagemaker.train.evaluate.execution.time.sleep")
    @patch.object(EvaluationPipelineExecution, "refresh")
    def test_prints_job_arn_for_executing_step(self, mock_refresh, mock_sleep, capsys):
        steps = [
            StepDetail(name="EvaluateCustomModel", status="Executing", display_name="EvaluateCustomModel",
                      start_time="2026-01-01T00:00:00Z",
                      job_arn="arn:aws:sagemaker:us-west-2:123456789:training-job/eval-custom-xyz"),
        ]
        # First poll shows Executing, then Succeeded
        call_count = [0]
        def side_effect():
            call_count[0] += 1
            if call_count[0] > 1:
                exec_obj.status.overall_status = "Succeeded"
                exec_obj.status.step_details[0].status = "Succeeded"
                exec_obj.status.step_details[0].end_time = "2026-01-01T00:01:00Z"
        exec_obj = _make_execution(status="Executing", step_details=steps)
        mock_refresh.side_effect = side_effect
        exec_obj.wait(poll=0, timeout=5)
        captured = capsys.readouterr()
        assert "Job ARN: arn:aws:sagemaker:us-west-2:123456789:training-job/eval-custom-xyz" in captured.out


class TestEvalObservabilityOnSuccess:
    @patch("sagemaker.train.evaluate.execution.time.sleep")
    @patch.object(EvaluationPipelineExecution, "refresh")
    def test_prints_s3_output_on_success(self, mock_refresh, mock_sleep, capsys):
        exec_obj = _make_execution(status="Succeeded", s3_output_path="s3://bucket/eval-results/")
        exec_obj.wait(poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "Results S3: s3://bucket/eval-results/" in captured.out


class TestEvalObservabilityOnFailure:
    @patch("sagemaker.train.evaluate.execution.time.sleep")
    @patch.object(EvaluationPipelineExecution, "refresh")
    def test_prints_failed_step_info(self, mock_refresh, mock_sleep, capsys):
        steps = [
            StepDetail(name="EvaluateCustomModel", status="Failed",
                      display_name="EvaluateCustomModel",
                      failure_reason="ResourceLimitExceeded",
                      job_arn="arn:aws:sagemaker:us-west-2:123456789:training-job/eval-custom-xyz"),
        ]
        exec_obj = _make_execution(status="Failed", step_details=steps, failure_reason="Step failed")
        with pytest.raises(Exception):
            exec_obj.wait(poll=0, timeout=1)
        captured = capsys.readouterr()
        assert "Failed step: EvaluateCustomModel" in captured.out
        assert "ResourceLimitExceeded" in captured.out
        assert "Job ARN: arn:aws:sagemaker:us-west-2:123456789:training-job/eval-custom-xyz" in captured.out
        assert "Log group: /aws/sagemaker/TrainingJobs" in captured.out
        assert "Log stream prefix: eval-custom-xyz" in captured.out
        assert "CloudWatch Logs:" in captured.out
