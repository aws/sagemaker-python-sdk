"""Unit tests for job_wait utilities."""
import collections
import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.common_utils.job_wait import (
    DEFAULT_LOG_GROUP_PREFIX,
    MAX_LOG_LINES,
    _calculate_job_progress,
    _create_log_stream_handler,
    _drain_log_events,
    _flush_log_events,
    _get_cloudwatch_logs_url,
    _get_mlflow_run_name,
    _get_progress_info,
    _get_rollout_info,
    _parse_region_from_arn,
)


class TestParseRegionFromArn:
    def test_standard_arn(self):
        assert _parse_region_from_arn("arn:aws:sagemaker:us-west-2:123456789012:job/my-job") == "us-west-2"

    def test_other_region(self):
        assert _parse_region_from_arn("arn:aws:sagemaker:eu-west-1:123456789012:job/j") == "eu-west-1"

    def test_invalid_arn(self):
        assert _parse_region_from_arn("not-an-arn") is None

    def test_empty_string(self):
        assert _parse_region_from_arn("") is None


class TestGetCloudwatchLogsUrl:
    def test_builds_url(self):
        url = _get_cloudwatch_logs_url(
            "arn:aws:sagemaker:us-west-2:123:job/my-job",
            "my-job",
            "/aws/sagemaker/FineTuningJob",
        )
        assert "us-west-2" in url
        assert "my-job" in url
        assert "FineTuningJob" in url
        assert url.startswith("https://us-west-2.console.aws.amazon.com/cloudwatch/")

    def test_bad_arn_returns_empty(self):
        assert _get_cloudwatch_logs_url("bad", "job", "/aws/sagemaker/FineTuningJob") == ""

    def test_custom_log_group(self):
        url = _get_cloudwatch_logs_url(
            "arn:aws:sagemaker:eu-west-1:123:job/j",
            "j",
            "/custom/log/group",
        )
        assert "eu-west-1" in url
        assert "custom" in url


class TestDefaultLogGroupPrefix:
    def test_value(self):
        assert DEFAULT_LOG_GROUP_PREFIX == "/aws/sagemaker/Job"


class TestGetProgressInfo:
    def test_valid_progress(self):
        config = {
            "ServiceOutput": {
                "ProgressInfo": {
                    "MaxEpoch": 3,
                    "StepsPerEpoch": 100,
                    "CurrentEpoch": 2,
                    "CurrentStep": 50,
                }
            }
        }
        info = _get_progress_info(config)
        assert info["MaxEpoch"] == 3
        assert info["CurrentStep"] == 50

    def test_step_only_progress(self):
        config = {
            "ServiceOutput": {
                "ProgressInfo": {
                    "MaxSteps": 200,
                    "CurrentStep": 75,
                }
            }
        }
        info = _get_progress_info(config)
        assert info["MaxSteps"] == 200
        assert info["CurrentStep"] == 75

    def test_missing_progress(self):
        assert _get_progress_info({"ServiceOutput": {}}) is None

    def test_incomplete_progress(self):
        config = {"ServiceOutput": {"ProgressInfo": {"CurrentEpoch": 1}}}
        assert _get_progress_info(config) is None

    def test_empty_config(self):
        assert _get_progress_info({}) is None


class TestGetRolloutProgress:
    def test_valid_rollout(self):
        config = {"ServiceOutput": {"RolloutInfo": {"Total": 1000, "Completed": 900}}}
        assert _get_rollout_info(config) == (900, 1000)

    def test_missing_rollout(self):
        assert _get_rollout_info({"ServiceOutput": {}}) is None

    def test_missing_total(self):
        config = {"ServiceOutput": {"RolloutInfo": {"Completed": 50}}}
        assert _get_rollout_info(config) is None

    def test_missing_completed(self):
        config = {"ServiceOutput": {"RolloutInfo": {"Total": 100}}}
        assert _get_rollout_info(config) is None

    def test_zero_completed(self):
        config = {"ServiceOutput": {"RolloutInfo": {"Total": 100, "Completed": 0}}}
        assert _get_rollout_info(config) == (0, 100)

    def test_empty_config(self):
        assert _get_rollout_info({}) is None


class TestGetMlflowRunName:
    def test_present(self):
        config = {"TrainingConfig": {"MlflowConfig": {"MlflowRunName": "run-1"}}}
        assert _get_mlflow_run_name(config) == "run-1"

    def test_absent(self):
        assert _get_mlflow_run_name({}) is None


class TestCalculateJobProgress:
    def test_basic_progress(self):
        info = {"MaxEpoch": 2, "StepsPerEpoch": 100, "CurrentEpoch": 1, "CurrentStep": 50}
        pct, text = _calculate_job_progress(info, None, None, None)
        assert pct == pytest.approx(24.5)
        assert "Epoch 1/2" in text
        assert "Step 50/100" in text

    def test_zero_max_epoch(self):
        info = {"MaxEpoch": 0, "StepsPerEpoch": 100, "CurrentEpoch": 0, "CurrentStep": 0}
        pct, text = _calculate_job_progress(info, None, None, None)
        assert pct is None

    def test_zero_total_steps(self):
        info = {"MaxEpoch": 2, "StepsPerEpoch": 0, "CurrentEpoch": 1, "CurrentStep": 0}
        pct, text = _calculate_job_progress(info, None, None, None)
        assert pct is None

    def test_step_only_progress(self):
        info = {"MaxSteps": 200, "CurrentStep": 100}
        pct, text = _calculate_job_progress(info, None, None, None)
        assert pct == pytest.approx(49.5)
        assert "Step 100/200" in text
        assert "Epoch" not in text

    def test_step_only_zero_max(self):
        info = {"MaxSteps": 0, "CurrentStep": 0}
        pct, text = _calculate_job_progress(info, None, None, None)
        assert pct is None


class TestCreateLogStreamHandler:
    @patch("sagemaker.train.common_utils.job_wait.MultiLogStreamHandler", create=True)
    def test_creates_handler(self, mock_cls):
        with patch(
            "sagemaker.core.utils.logs.MultiLogStreamHandler", mock_cls
        ):
            handler = _create_log_stream_handler("/aws/sagemaker/FineTuningJob", "my-job")
            mock_cls.assert_called_once_with(
                log_group_name="/aws/sagemaker/FineTuningJob",
                log_stream_name_prefix="my-job",
                expected_stream_count=1,
            )
            assert handler is not None

    @patch(
        "sagemaker.train.common_utils.job_wait.MultiLogStreamHandler",
        create=True,
        side_effect=ImportError("no module"),
    )
    def test_returns_none_on_import_error(self, _):
        with patch(
            "sagemaker.core.utils.logs.MultiLogStreamHandler",
            side_effect=ImportError("no module"),
        ):
            assert _create_log_stream_handler("/group", "job") is None

    @patch("sagemaker.train.common_utils.job_wait.MultiLogStreamHandler", create=True)
    def test_custom_instance_count(self, mock_cls):
        with patch(
            "sagemaker.core.utils.logs.MultiLogStreamHandler", mock_cls
        ):
            _create_log_stream_handler("/group", "job", instance_count=4)
            mock_cls.assert_called_once_with(
                log_group_name="/group",
                log_stream_name_prefix="job",
                expected_stream_count=4,
            )


class TestFlushLogEvents:
    def test_prints_messages(self, capsys):
        handler = MagicMock()
        handler.get_latest_log_events.return_value = [
            ("stream/0", {"message": "line 1\n", "timestamp": 1}),
            ("stream/0", {"message": "line 2\n", "timestamp": 2}),
        ]
        _flush_log_events(handler)
        captured = capsys.readouterr()
        assert "line 1" in captured.out
        assert "line 2" in captured.out

    def test_no_events(self, capsys):
        handler = MagicMock()
        handler.get_latest_log_events.return_value = []
        _flush_log_events(handler)
        assert capsys.readouterr().out == ""

    def test_handles_exception_gracefully(self, capsys):
        handler = MagicMock()
        handler.get_latest_log_events.side_effect = Exception("cw error")
        _flush_log_events(handler)
        # Should not raise
        assert capsys.readouterr().out == ""


class TestDrainLogEvents:
    def test_appends_to_deque(self):
        handler = MagicMock()
        handler.get_latest_log_events.return_value = [
            ("stream/0", {"message": "line 1\n", "timestamp": 1}),
            ("stream/0", {"message": "line 2\n", "timestamp": 2}),
        ]
        buf = collections.deque(maxlen=MAX_LOG_LINES)
        _drain_log_events(handler, buf)
        assert list(buf) == ["line 1", "line 2"]

    def test_respects_maxlen(self):
        handler = MagicMock()
        handler.get_latest_log_events.return_value = [
            ("s", {"message": f"line {i}\n", "timestamp": i})
            for i in range(30)
        ]
        buf = collections.deque(maxlen=MAX_LOG_LINES)
        _drain_log_events(handler, buf)
        assert len(buf) == MAX_LOG_LINES
        assert buf[-1] == "line 29"
        assert buf[0] == "line 10"

    def test_skips_empty_messages(self):
        handler = MagicMock()
        handler.get_latest_log_events.return_value = [
            ("s", {"message": "", "timestamp": 1}),
            ("s", {"message": "real line\n", "timestamp": 2}),
        ]
        buf = collections.deque(maxlen=MAX_LOG_LINES)
        _drain_log_events(handler, buf)
        assert list(buf) == ["real line"]

    def test_handles_exception_gracefully(self):
        handler = MagicMock()
        handler.get_latest_log_events.side_effect = Exception("cw error")
        buf = collections.deque(maxlen=MAX_LOG_LINES)
        _drain_log_events(handler, buf)
        assert len(buf) == 0
