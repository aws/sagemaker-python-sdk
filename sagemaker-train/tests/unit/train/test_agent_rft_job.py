"""Unit tests for AgentRFTJob."""
import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.agent_rft_job import AgentRFTJob


SAMPLE_CONFIG_DOC = json.dumps(
    {
        "AgentConfig": {"EndpointConfig": {"BedrockAgentCoreConfig": {"AgentArn": "arn:agent"}}},
        "OutputDataConfig": {"S3OutputPath": "s3://bucket/output"},
        "TrainingConfig": {"BaseModelArn": "arn:model"},
        "ServiceOutput": {
            "OutputModelPackageArn": "arn:aws:sagemaker:us-west-2:123:model-package/pkg",
            "MlflowDetails": {
                "ExperimentName": "exp",
                "RunName": "run",
                "ExperimentId": "eid",
                "RunId": "rid",
            },
            "BillableTokenUsage": {
                "TrainTokenCount": 42000,
                "PrefillTokenCount": 10000,
                "SampleTokenCount": 5000,
            },
        },
    }
)


def _make_mock_job(**overrides):
    job = MagicMock()
    job.job_name = "test-job"
    job.job_arn = "arn:aws:sagemaker:us-west-2:123:job/test-job"
    job.job_status = "Completed"
    job.secondary_status = "Training"
    job.secondary_status_transitions = []
    job.failure_reason = None
    job.creation_time = "2026-01-01T00:00:00Z"
    job.last_modified_time = "2026-01-01T01:00:00Z"
    job.end_time = "2026-01-01T02:00:00Z"
    job.job_config_document = SAMPLE_CONFIG_DOC
    for k, v in overrides.items():
        setattr(job, k, v)
    return job


class TestAgentRFTJobProperties:
    def test_delegated_properties(self):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        assert rft.job_name == "test-job"
        assert rft.job_arn == "arn:aws:sagemaker:us-west-2:123:job/test-job"
        assert rft.job_status == "Completed"
        assert rft.secondary_status == "Training"
        assert rft.failure_reason is None

    def test_output_model_package_arn(self):
        rft = AgentRFTJob(_make_mock_job())
        assert rft.output_model_package_arn == "arn:aws:sagemaker:us-west-2:123:model-package/pkg"

    def test_mlflow_details(self):
        rft = AgentRFTJob(_make_mock_job())
        assert rft.mlflow_details["ExperimentName"] == "exp"
        assert rft.mlflow_details["RunId"] == "rid"

    def test_s3_output_path(self):
        rft = AgentRFTJob(_make_mock_job())
        assert rft.s3_output_path == "s3://bucket/output"

    def test_billable_token_usage(self):
        rft = AgentRFTJob(_make_mock_job())
        assert rft.billable_token_usage["TrainTokenCount"] == 42000
        assert rft.billable_token_usage["PrefillTokenCount"] == 10000
        assert rft.billable_token_usage["SampleTokenCount"] == 5000

    def test_output_model_package_arn_none_before_completion(self):
        config = json.dumps({"OutputDataConfig": {"S3OutputPath": "s3://b/o"}})
        rft = AgentRFTJob(_make_mock_job(job_config_document=config, job_status="InProgress"))
        assert rft.output_model_package_arn is None

    def test_empty_config_document(self):
        rft = AgentRFTJob(_make_mock_job(job_config_document=None))
        assert rft.output_model_package_arn is None
        assert rft.mlflow_details is None
        assert rft.s3_output_path is None
        assert rft.billable_token_usage is None
        assert rft.progress_info is None

    def test_progress_info(self):
        config = json.dumps({
            "ServiceOutput": {
                "ProgressInfo": {
                    "MaxEpoch": 3,
                    "StepsPerEpoch": 100,
                    "CurrentEpoch": 2,
                    "CurrentStep": 50,
                }
            }
        })
        rft = AgentRFTJob(_make_mock_job(job_config_document=config))
        info = rft.progress_info
        assert info["MaxEpoch"] == 3
        assert info["CurrentEpoch"] == 2
        assert info["CurrentStep"] == 50

    def test_progress_info_none_when_missing(self):
        config = json.dumps({"ServiceOutput": {}})
        rft = AgentRFTJob(_make_mock_job(job_config_document=config))
        assert rft.progress_info is None

    def test_progress_info_none_when_incomplete(self):
        config = json.dumps({
            "ServiceOutput": {"ProgressInfo": {"CurrentEpoch": 1}}
        })
        rft = AgentRFTJob(_make_mock_job(job_config_document=config))
        assert rft.progress_info is None


class TestAgentRFTJobLifecycle:
    def test_refresh_invalidates_cache(self):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        _ = rft.output_model_package_arn  # populate cache
        assert rft._cached_config is not None
        rft.refresh()
        assert rft._cached_config is None
        job.refresh.assert_called_once()

    def test_stop_delegates(self):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        rft.stop()
        job.stop.assert_called_once()

    def test_delete_delegates(self):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        rft.delete()
        job.delete.assert_called_once()

    @patch("sagemaker.train.common_utils.job_wait.wait")
    def test_wait_defaults(self, mock_wait):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        rft.description = "test desc"
        rft.wait()
        mock_wait.assert_called_once_with(
            job, poll=5, timeout=3000, description="test desc", max_log_lines=20
        )

    @patch("sagemaker.train.common_utils.job_wait.wait")
    def test_wait_custom_max_log_lines(self, mock_wait):
        job = _make_mock_job()
        rft = AgentRFTJob(job)
        rft.description = "test desc"
        rft.wait(max_log_lines=50)
        mock_wait.assert_called_once_with(
            job, poll=5, timeout=3000, description="test desc", max_log_lines=50
        )

    def test_from_job(self):
        job = _make_mock_job()
        rft = AgentRFTJob.from_job(job)
        assert rft._job is job

    @patch("sagemaker.train.agent_rft_job.Job")
    def test_get(self, mock_job_cls):
        mock_job_cls.get.return_value = _make_mock_job()
        rft = AgentRFTJob.get("my-job")
        mock_job_cls.get.assert_called_once_with(
            job_name="my-job", job_category="AgentRFT", session=None
        )
        assert rft.job_name == "test-job"

    @patch("sagemaker.train.agent_rft_job.Job")
    def test_get_all(self, mock_job_cls):
        mock_jobs = [_make_mock_job(job_name="job-1"), _make_mock_job(job_name="job-2")]
        mock_job_cls.get_all.return_value = iter(mock_jobs)
        results = list(AgentRFTJob.get_all(session="fake-session", name_contains="job"))
        mock_job_cls.get_all.assert_called_once_with(
            job_category="AgentRFT", session="fake-session", name_contains="job"
        )
        assert len(results) == 2
        assert all(isinstance(r, AgentRFTJob) for r in results)
        assert results[0].job_name == "job-1"
        assert results[1].job_name == "job-2"


class TestAgentRFTJobMlflowUrl:
    @patch("sagemaker.train.common_utils.job_wait._get_mlflow_presigned_url")
    def test_get_mlflow_url(self, mock_presigned):
        mock_presigned.return_value = "https://mlflow.example.com/presigned"
        config = json.dumps({
            "TrainingConfig": {
                "MlflowConfig": {"MlflowResourceArn": "arn:mlflow", "MlflowExperimentName": "exp"}
            },
            "ServiceOutput": {
                "MlflowDetails": {"ExperimentId": "123", "RunId": "456"}
            },
        })
        rft = AgentRFTJob(_make_mock_job(job_config_document=config))
        url = rft.get_mlflow_url()
        assert url == "https://mlflow.example.com/presigned"
        mock_presigned.assert_called_once_with(
            "arn:mlflow", "exp", experiment_id="123", run_id="456"
        )

    @patch("sagemaker.train.common_utils.job_wait._is_jupyter_environment", return_value=True)
    @patch("sagemaker.train.common_utils.job_wait._get_mlflow_presigned_url")
    def test_get_mlflow_url_displays_in_jupyter(self, mock_presigned, _mock_jupyter):
        import sys
        mock_ipython_display = MagicMock()
        sys.modules["IPython"] = MagicMock()
        sys.modules["IPython.display"] = mock_ipython_display
        try:
            mock_presigned.return_value = "https://mlflow.example.com/presigned"
            config = json.dumps({
                "TrainingConfig": {
                    "MlflowConfig": {"MlflowResourceArn": "arn:mlflow", "MlflowExperimentName": "exp"}
                },
                "ServiceOutput": {
                    "MlflowDetails": {"ExperimentId": "123", "RunId": "456"}
                },
            })
            rft = AgentRFTJob(_make_mock_job(job_config_document=config))
            url = rft.get_mlflow_url()
            assert url == "https://mlflow.example.com/presigned"
        finally:
            sys.modules.pop("IPython.display", None)
            sys.modules.pop("IPython", None)

    def test_get_mlflow_url_no_config(self):
        rft = AgentRFTJob(_make_mock_job(job_config_document="{}"))
        assert rft.get_mlflow_url() is None


class TestAgentRFTJobTrainingMetrics:
    MLFLOW_CONFIG_DOC = json.dumps({
        "TrainingConfig": {
            "MlflowConfig": {
                "MlflowResourceArn": "arn:mlflow",
                "MlflowExperimentName": "exp",
                "MlflowRunName": "run1",
            }
        },
        "ServiceOutput": {
            "MlflowDetails": {"ExperimentId": "eid", "RunId": "rid"},
        },
    })

    @patch("sagemaker.train.common_utils.job_wait._setup_mlflow_metrics_util")
    def test_returns_per_step_metrics(self, mock_setup):
        mock_util = MagicMock()
        mock_setup.return_value = mock_util
        mock_util._get_run_ids.return_value = ["rid"]
        mock_util.get_metric_history.side_effect = lambda rid, name: {
            "training/num_trajectories": [
                {"step": 1, "value": 80, "timestamp": 0},
                {"step": 2, "value": 80, "timestamp": 1},
            ],
            "rollout/turns/mean": [
                {"step": 1, "value": 2.55, "timestamp": 0},
                {"step": 2, "value": 3.10, "timestamp": 1},
            ],
            "rollout/reward/mean": [
                {"step": 1, "value": 0.75, "timestamp": 0},
                {"step": 2, "value": 0.82, "timestamp": 1},
            ],
            "training/total_tokens": [
                {"step": 1, "value": 5000, "timestamp": 0},
                {"step": 2, "value": 12000, "timestamp": 1},
            ],
        }[name]

        rft = AgentRFTJob(_make_mock_job(job_config_document=self.MLFLOW_CONFIG_DOC))
        rows = rft.get_training_metrics()

        assert len(rows) == 2
        assert rows[0] == {
            "step": 1,
            "training/num_trajectories": 80,
            "rollout/turns/mean": 2.55,
            "rollout/reward/mean": 0.75,
            "training/total_tokens": 5000,
        }
        assert rows[1] == {
            "step": 2,
            "training/num_trajectories": 80,
            "rollout/turns/mean": 3.10,
            "rollout/reward/mean": 0.82,
            "training/total_tokens": 12000,
        }

    @patch("sagemaker.train.common_utils.job_wait._setup_mlflow_metrics_util")
    def test_returns_empty_when_no_mlflow(self, mock_setup):
        rft = AgentRFTJob(_make_mock_job(job_config_document="{}"))
        assert rft.get_training_metrics() == []

    @patch("sagemaker.train.common_utils.job_wait._setup_mlflow_metrics_util")
    def test_handles_partial_metrics(self, mock_setup):
        mock_util = MagicMock()
        mock_setup.return_value = mock_util
        mock_util._get_run_ids.return_value = ["rid"]

        def history_side_effect(rid, name):
            if name == "rollout/reward/mean":
                return [{"step": 1, "value": 0.5, "timestamp": 0}]
            raise Exception("metric not found")

        mock_util.get_metric_history.side_effect = history_side_effect

        rft = AgentRFTJob(_make_mock_job(job_config_document=self.MLFLOW_CONFIG_DOC))
        rows = rft.get_training_metrics()

        assert len(rows) == 1
        assert rows[0]["rollout/reward/mean"] == 0.5
        assert rows[0]["training/num_trajectories"] is None
        assert rows[0]["rollout/turns/mean"] is None
