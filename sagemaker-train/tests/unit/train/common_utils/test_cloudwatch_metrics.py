"""Unit tests for cloudwatch_metrics module."""

from __future__ import absolute_import

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas
import pytest

from sagemaker.core.training.configs import Compute, HyperPodCompute
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common_utils.cloudwatch_metrics import (
    _fetch_smhp_logs,
    _fetch_smtj_logs,
    fetch_and_plot_metrics,
    parse_metrics_from_logs,
)


FAKE_SFT_LOGS = [
    {"message": "Training epoch 0, iteration 0/9 | lr: 6.25e-07 | global_batch_size: 32 | global_step: 1 | reduced_train_loss: 9.240 | ..."},
    {"message": "Training epoch 0, iteration 1/9 | lr: 1.25e-06 | global_batch_size: 32 | global_step: 2 | reduced_train_loss: 7.750 | ..."},
    {"message": "Training epoch 0, iteration 2/9 | lr: 1.87e-06 | global_batch_size: 32 | global_step: 3 | reduced_train_loss: 6.615 | ..."},
    {"message": "Some other log line without any metrics"},
]

FAKE_RLVR_SMTJ_LOGS = [
    {"message": "global_step=1 critic/rewards/mean=0.123"},
    {"message": "global_step=2 critic/rewards/mean=0.456"},
    {"message": "PPO iteration complete, buffers flushed"},
]

FAKE_RLVR_SMHP_LOGS = [
    {"message": "global_step: 1 train_rm_score: 0.55"},
    {"message": "global_step: 2 train_rm_score: 0.72"},
]


class TestParseMetrics:

    def test_sft_extracts_loss_and_lr(self):
        df = parse_metrics_from_logs(FAKE_SFT_LOGS, "smtj", "SFT")

        assert len(df) == 3
        assert list(df.columns) == ["global_step", "training_loss", "lr"]
        assert df["training_loss"].iloc[0] == pytest.approx(9.240)
        assert df["lr"].iloc[0] == pytest.approx(6.25e-07)

    def test_rlvr_smtj_extracts_reward_score(self):
        df = parse_metrics_from_logs(FAKE_RLVR_SMTJ_LOGS, "smtj", "RLVR")

        assert len(df) == 2
        assert list(df.columns) == ["global_step", "reward_score"]
        assert df["reward_score"].iloc[0] == pytest.approx(0.123)

    def test_rlvr_smhp_extracts_reward_score(self):
        df = parse_metrics_from_logs(FAKE_RLVR_SMHP_LOGS, "smhp", "RLVR")

        assert len(df) == 2
        assert df["reward_score"].iloc[0] == pytest.approx(0.55)

    def test_subset_metrics_lr_only(self):
        df = parse_metrics_from_logs(FAKE_SFT_LOGS, "smtj", "SFT", metrics=["lr"])

        assert list(df.columns) == ["global_step", "lr"]
        assert "training_loss" not in df.columns

    def test_partial_metrics_fills_nan(self):
        """Lines missing a metric get NaN for that column."""
        logs = [
            {"message": "global_step=1 reduced_train_loss=5.0"},
            {"message": "global_step=2 reduced_train_loss=4.0 lr=1e-5"},
        ]
        df = parse_metrics_from_logs(logs, "smtj", "SFT")

        assert len(df) == 2
        assert pandas.isna(df["lr"].iloc[0])
        assert df["lr"].iloc[1] == pytest.approx(1e-5)

    def test_dpo_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not supported for DPO"):
            parse_metrics_from_logs([], "smtj", "DPO")

    def test_empty_logs_returns_empty_dataframe(self):
        df = parse_metrics_from_logs([], "smtj", "SFT")
        assert df.empty


class TestFetchLogs:

    def test_smtj_fetches_from_dedicated_stream(self):
        mock_client = MagicMock()
        mock_client.describe_log_streams.return_value = {
            "logStreams": [{"logStreamName": "my-job/algo-1"}]
        }
        mock_client.get_log_events.side_effect = [
            {"events": [{"message": "global_step=1 reduced_train_loss=5.0"}], "nextBackwardToken": "t1"},
            {"events": [], "nextBackwardToken": "t1"},
        ]

        events = _fetch_smtj_logs("my-job", mock_client, "/aws/sagemaker/TrainingJobs")
        assert len(events) == 1

    def test_smtj_no_stream_returns_empty(self):
        mock_client = MagicMock()
        mock_client.describe_log_streams.return_value = {"logStreams": []}

        events = _fetch_smtj_logs("nonexistent-job", mock_client, "/aws/sagemaker/TrainingJobs")
        assert events == []

    def test_smhp_uses_filter_with_job_id(self):
        mock_client = MagicMock()
        mock_client.filter_log_events.return_value = {
            "events": [{"message": "global_step: 1 train_rm_score: 0.5"}],
        }

        events = _fetch_smhp_logs("hp-job-123", mock_client, "/aws/sagemaker/Clusters/c/id")
        assert len(events) == 1
        call_kwargs = mock_client.filter_log_events.call_args[1]
        assert '"hp-job-123"' in call_kwargs["filterPattern"]


class TestFetchAndPlotMetrics:

    def _session(self):
        s = MagicMock()
        s.boto_session.region_name = "us-east-1"
        return s

    @patch("sagemaker.train.common_utils.cloudwatch_metrics.plot_metrics")
    @patch("sagemaker.train.common_utils.cloudwatch_metrics._fetch_smtj_logs")
    def test_smtj_sft_end_to_end(self, mock_fetch, mock_plot):
        mock_fetch.return_value = FAKE_SFT_LOGS

        df = fetch_and_plot_metrics(
            "my-job", Compute(instance_type="ml.p5.48xlarge", instance_count=1),
            "SFT", self._session(),
        )

        assert len(df) == 3
        assert "training_loss" in df.columns
        mock_plot.assert_called_once()

    @patch("sagemaker.train.common_utils.cloudwatch_metrics.plot_metrics")
    @patch("sagemaker.train.common_utils.cloudwatch_metrics._fetch_smhp_logs")
    @patch("sagemaker.train.common_utils.cloudwatch_metrics._get_smhp_log_group")
    def test_smhp_rlvr_end_to_end(self, mock_lg, mock_fetch, mock_plot):
        mock_lg.return_value = "/aws/sagemaker/Clusters/c/id"
        mock_fetch.return_value = FAKE_RLVR_SMHP_LOGS

        df = fetch_and_plot_metrics(
            "hp-job", HyperPodCompute(cluster_name="c", instance_type="ml.p5.48xlarge", node_count=1),
            "RLVR", self._session(),
        )

        assert len(df) == 2
        assert "reward_score" in df.columns

    def test_invalid_technique_raises_before_fetching(self):
        with pytest.raises(ValueError, match="not a supported training technique"):
            fetch_and_plot_metrics(
                "job", Compute(instance_type="ml.p5.48xlarge", instance_count=1),
                "RFT", self._session(),
            )

    @patch("sagemaker.train.common_utils.cloudwatch_metrics._fetch_smtj_logs")
    def test_no_logs_found_raises(self, mock_fetch):
        mock_fetch.return_value = []

        with pytest.raises(ValueError, match="No CloudWatch logs found"):
            fetch_and_plot_metrics(
                "missing-job", Compute(instance_type="ml.p5.48xlarge", instance_count=1),
                "SFT", self._session(),
            )

    @patch("sagemaker.train.common_utils.cloudwatch_metrics.plot_metrics")
    @patch("sagemaker.train.common_utils.cloudwatch_metrics._fetch_smtj_logs")
    def test_result_sorted_by_step(self, mock_fetch, mock_plot):
        """Out-of-order events (startFromHead=False) are sorted in returned DataFrame."""
        mock_fetch.return_value = [
            {"message": "global_step=5 reduced_train_loss=1.0"},
            {"message": "global_step=1 reduced_train_loss=5.0"},
        ]

        df = fetch_and_plot_metrics(
            "job", Compute(instance_type="ml.p5.48xlarge", instance_count=1),
            "SFT", self._session(), metrics=["training_loss"],
        )

        assert df["global_step"].tolist() == [1, 5]

class TestStreamLogs:
    """Tests for BaseTrainer.stream_logs() dispatch and behavior."""

    def _make_trainer(self, compute=None, latest_job=None):
        """Create a minimal trainer stub for stream_logs testing."""
        class _StubTrainer(BaseTrainer):
            _customization_technique = "SFT"

            def train(self, *args, **kwargs):
                pass

        trainer = _StubTrainer.__new__(_StubTrainer)
        trainer.compute = compute
        trainer.sagemaker_session = None
        trainer._latest_training_job = latest_job
        return trainer

    def test_no_job_raises_valueerror(self):
        """stream_logs() raises if no training job exists."""
        trainer = self._make_trainer(latest_job=None)

        with pytest.raises(ValueError, match="No training job found"):
            trainer.stream_logs()

    @patch("sagemaker.train.base_trainer.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.common_utils.cloudwatch_metrics._get_smhp_log_group")
    def test_smhp_dispatches_with_start_time(self, mock_log_group, mock_session):
        """SMHP stream_logs passes start_time to the polling loop."""
        mock_log_group.return_value = "/aws/sagemaker/Clusters/c/id"
        mock_sess = MagicMock()
        mock_sess.boto_session.region_name = "us-east-1"
        mock_logs_client = MagicMock()
        mock_sess.boto_session.client.return_value = mock_logs_client
        mock_session.return_value = mock_sess

        mock_logs_client.filter_log_events.return_value = {
            "events": [{"eventId": "e1", "message": "hello", "timestamp": 1000}]
        }

        trainer = self._make_trainer(
            compute=HyperPodCompute(cluster_name="c", instance_type="ml.p5.48xlarge", node_count=1),
            latest_job="my-hp-job",
        )

        # Simulate KeyboardInterrupt on first sleep to stop the loop
        with patch("time.sleep", side_effect=KeyboardInterrupt):
            trainer.stream_logs(
                start_time=datetime(2026, 7, 8, 14, 0, 0, tzinfo=timezone.utc)
            )

        # Verify filter_log_events was called with the user-provided startTime
        call_kwargs = mock_logs_client.filter_log_events.call_args[1]
        expected_ts = int(datetime(2026, 7, 8, 14, 0, 0, tzinfo=timezone.utc).timestamp() * 1000)
        assert call_kwargs["startTime"] == expected_ts

    @patch("sagemaker.core.resources.TrainingJob.get")
    @patch("sagemaker.train.base_trainer.MultiLogStreamHandler")
    def test_smtj_stops_on_completed(self, mock_handler_cls, mock_get_job):
        """SMTJ stream_logs exits when job status is Completed."""
        # Mock the handler to return one event then empty
        mock_handler = MagicMock()
        mock_handler.get_latest_log_events.side_effect = [
            iter([("stream", {"message": "Training complete"})]),
            iter([]),  # final flush
        ]
        mock_handler_cls.return_value = mock_handler

        # Mock job status as Completed
        mock_job = MagicMock()
        mock_job.training_job_status = "Completed"
        mock_get_job.return_value = mock_job

        trainer = self._make_trainer(
            compute=Compute(instance_type="ml.p5.48xlarge", instance_count=1),
            latest_job=MagicMock(training_job_name="my-smtj-job"),
        )

        # Should return without hanging (job is already Completed)
        trainer.stream_logs()

        mock_get_job.assert_called()


    def test_show_metrics_oss_without_mlflow_raises(self):
        """show_metrics() raises ValueError for non-Nova models without MLflow configured."""
        trainer = self._make_trainer(latest_job=MagicMock(
            training_job_name="some-job",
            mlflow_config=None,
            mlflow_details=None,
        ))
        trainer._model_name = "test-oss-model"

        with pytest.raises(ValueError, match="requires MLflow to be configured"):
            trainer.show_metrics()

    @patch("sagemaker.train.base_trainer.plot_training_metrics")
    def test_show_metrics_oss_with_mlflow_delegates(self, mock_plot):
        """show_metrics() for OSS models with MLflow configured calls plot_training_metrics."""
        mock_job = MagicMock()
        mock_job.training_job_name = "oss-sft-job"
        mock_job.mlflow_config.mlflow_resource_arn = "arn:aws:sagemaker:us-east-1:012345678910:mlflow-app/app-123"
        mock_job.mlflow_details.mlflow_run_id = "run-abc123"

        trainer = self._make_trainer(latest_job=mock_job)
        trainer._model_name = "test-oss-model"

        trainer.show_metrics(metrics=["loss"])

        mock_plot.assert_called_once_with(mock_job, metrics=["loss"])
