"""Unit tests for metrics_visualizer module."""
import pytest
from unittest.mock import Mock, patch, MagicMock


class TestParseJobArn:
    def test_training_job_arn(self):
        from sagemaker.train.common_utils.metrics_visualizer import _parse_job_arn
        result = _parse_job_arn("arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job")
        assert result == ("us-west-2", "training-job/my-job")

    def test_processing_job_arn(self):
        from sagemaker.train.common_utils.metrics_visualizer import _parse_job_arn
        result = _parse_job_arn("arn:aws:sagemaker:us-east-1:123456789012:processing-job/my-job")
        assert result == ("us-east-1", "processing-job/my-job")

    def test_invalid_arn_returns_none(self):
        from sagemaker.train.common_utils.metrics_visualizer import _parse_job_arn
        assert _parse_job_arn("not-an-arn") is None


class TestGetConsoleJobUrl:
    def test_training_job(self):
        from sagemaker.train.common_utils.metrics_visualizer import get_console_job_url
        url = get_console_job_url("arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job")
        assert url == "https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs/my-job"

    def test_invalid_arn_returns_empty(self):
        from sagemaker.train.common_utils.metrics_visualizer import get_console_job_url
        assert get_console_job_url("not-an-arn") == ""

    def test_unknown_job_type_returns_empty(self):
        from sagemaker.train.common_utils.metrics_visualizer import get_console_job_url
        assert get_console_job_url("arn:aws:sagemaker:us-west-2:123456789012:unknown-job/my-job") == ""


class TestGetCloudwatchLogsUrl:
    def test_training_job(self):
        from sagemaker.train.common_utils.metrics_visualizer import get_cloudwatch_logs_url
        url = get_cloudwatch_logs_url("arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job")
        assert "us-west-2" in url
        assert "TrainingJobs" in url
        assert "my-job" in url

    def test_invalid_arn_returns_empty(self):
        from sagemaker.train.common_utils.metrics_visualizer import get_cloudwatch_logs_url
        assert get_cloudwatch_logs_url("not-an-arn") == ""


class TestGetStudioUrl:
    @patch("sagemaker.train.common_utils.metrics_visualizer._get_studio_base_url")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_with_training_job_object(self, mock_client_cls, mock_base_url):
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        mock_client_cls.return_value.region_name = "us-west-2"
        mock_base_url.return_value = "https://studio-d-abc.studio.us-west-2.sagemaker.aws"

        mock_job = Mock()
        mock_job.training_job_name = "my-job"

        url = get_studio_url(mock_job)
        assert url == "https://studio-d-abc.studio.us-west-2.sagemaker.aws/jobs/train/my-job"
        mock_base_url.assert_called_once_with("us-west-2")

    @patch("sagemaker.train.common_utils.metrics_visualizer._get_studio_base_url")
    def test_with_arn_string(self, mock_base_url):
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        mock_base_url.return_value = "https://studio-d-abc.studio.us-west-2.sagemaker.aws"

        url = get_studio_url("arn:aws:sagemaker:us-west-2:123456789012:training-job/my-job")
        assert url == "https://studio-d-abc.studio.us-west-2.sagemaker.aws/jobs/train/my-job"
        mock_base_url.assert_called_once_with("us-west-2")

    @patch("sagemaker.train.common_utils.metrics_visualizer._get_studio_base_url")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    @patch("sagemaker.train.common_utils.metrics_visualizer.TrainingJob")
    def test_with_job_name_string(self, mock_tj_cls, mock_client_cls, mock_base_url):
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        mock_client_cls.return_value.region_name = "us-west-2"
        mock_base_url.return_value = "https://studio-d-abc.studio.us-west-2.sagemaker.aws"
        mock_tj_cls.get.return_value.training_job_name = "my-job"

        url = get_studio_url("my-job")
        assert url == "https://studio-d-abc.studio.us-west-2.sagemaker.aws/jobs/train/my-job"

    @patch("sagemaker.train.common_utils.metrics_visualizer._get_studio_base_url")
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_returns_empty_when_no_domain(self, mock_client_cls, mock_base_url):
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        mock_client_cls.return_value.region_name = "us-west-2"
        mock_base_url.return_value = ""

        url = get_studio_url(Mock(training_job_name="my-job"))
        assert url == ""


class TestGetAvailableMetrics:
    @patch("sagemaker.train.common_utils.metrics_visualizer.TrainingJob")
    def test_returns_empty_when_no_mlflow_config(self, _):
        from sagemaker.train.common_utils.metrics_visualizer import get_available_metrics
        mock_job = Mock(spec=[])  # no mlflow_config attribute
        assert get_available_metrics(mock_job) == []

    @patch("sagemaker.train.common_utils.metrics_visualizer.TrainingJob")
    def test_returns_empty_when_mlflow_config_falsy(self, _):
        from sagemaker.train.common_utils.metrics_visualizer import get_available_metrics
        mock_job = Mock()
        mock_job.mlflow_config = None
        assert get_available_metrics(mock_job) == []

    @patch("mlflow.get_run")
    @patch("mlflow.set_tracking_uri")
    def test_returns_metric_names(self, mock_set_uri, mock_get_run):
        from sagemaker.train.common_utils.metrics_visualizer import get_available_metrics
        mock_job = Mock()
        mock_job.mlflow_config.mlflow_resource_arn = "arn:aws:sagemaker:us-west-2:123:mlflow-tracking/abc"
        mock_job.mlflow_details.mlflow_run_id = "run-123"
        mock_get_run.return_value.data.metrics = {"loss": 0.5, "accuracy": 0.9}

        result = get_available_metrics(mock_job)
        assert set(result) == {"loss", "accuracy"}
