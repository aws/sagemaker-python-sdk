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
from __future__ import absolute_import

import pytest
from datetime import datetime
from mock import Mock, patch

from sagemaker.local.entities import _LocalTrainingJob


class TestLocalTrainingJobFinalMetrics:
    """Test cases for FinalMetricDataList functionality in _LocalTrainingJob."""

    def test_describe_includes_final_metric_data_list(self):
        """Test that describe() includes FinalMetricDataList field."""
        container = Mock()
        job = _LocalTrainingJob(container)
        job.training_job_name = "test-job"
        job.state = "Completed"
        job.start_time = datetime.now()
        job.end_time = datetime.now()
        job.model_artifacts = "/path/to/model"
        job.output_data_config = {}
        job.environment = {}

        response = job.describe()

        assert "FinalMetricDataList" in response
        assert isinstance(response["FinalMetricDataList"], list)

    def test_extract_final_metrics_no_logs(self):
        """Test _extract_final_metrics returns empty list when no logs."""
        container = Mock()
        container.logs = None
        job = _LocalTrainingJob(container)

        result = job._extract_final_metrics()

        assert result == []

    def test_extract_final_metrics_no_metric_definitions(self):
        """Test _extract_final_metrics returns empty list when no metric definitions."""
        container = Mock()
        container.logs = "some logs"
        container.metric_definitions = []
        job = _LocalTrainingJob(container)

        result = job._extract_final_metrics()

        assert result == []

    def test_extract_final_metrics_with_valid_metrics(self):
        """Test _extract_final_metrics extracts metrics correctly."""
        container = Mock()
        container.logs = "Training started\nGAN_loss=0.138318;\nTraining complete"
        container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"}
        ]
        job = _LocalTrainingJob(container)
        job.end_time = datetime(2023, 1, 1, 12, 0, 0)

        result = job._extract_final_metrics()

        assert len(result) == 1
        assert result[0]["MetricName"] == "ganloss"
        assert result[0]["Value"] == 0.138318
        assert result[0]["Timestamp"] == job.end_time

    def test_extract_final_metrics_multiple_matches_uses_last(self):
        """Test _extract_final_metrics uses the last match for each metric."""
        container = Mock()
        container.logs = "GAN_loss=0.5;\nGAN_loss=0.3;\nGAN_loss=0.138318;"
        container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"}
        ]
        job = _LocalTrainingJob(container)
        job.end_time = datetime(2023, 1, 1, 12, 0, 0)

        result = job._extract_final_metrics()

        assert len(result) == 1
        assert result[0]["Value"] == 0.138318

    def test_extract_final_metrics_multiple_metrics(self):
        """Test _extract_final_metrics handles multiple different metrics."""
        container = Mock()
        container.logs = "GAN_loss=0.138318;\nAccuracy=0.95;\nLoss=1.234;"
        container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"},
            {"Name": "accuracy", "Regex": r"Accuracy=([\d\.]+);"},
            {"Name": "loss", "Regex": r"Loss=([\d\.]+);"}
        ]
        job = _LocalTrainingJob(container)
        job.end_time = datetime(2023, 1, 1, 12, 0, 0)

        result = job._extract_final_metrics()

        assert len(result) == 3
        metric_names = [m["MetricName"] for m in result]
        assert "ganloss" in metric_names
        assert "accuracy" in metric_names
        assert "loss" in metric_names

    def test_extract_final_metrics_no_matches(self):
        """Test _extract_final_metrics returns empty list when regex doesn't match."""
        container = Mock()
        container.logs = "Training started\nTraining complete"
        container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"}
        ]
        job = _LocalTrainingJob(container)

        result = job._extract_final_metrics()

        assert result == []

    def test_extract_final_metrics_invalid_metric_definition(self):
        """Test _extract_final_metrics skips invalid metric definitions."""
        container = Mock()
        container.logs = "GAN_loss=0.138318;"
        container.metric_definitions = [
            {"Name": "ganloss"},  # Missing Regex
            {"Regex": r"GAN_loss=([\d\.]+);"},  # Missing Name
            {"Name": "valid", "Regex": r"GAN_loss=([\d\.]+);"}  # Valid
        ]
        job = _LocalTrainingJob(container)
        job.end_time = datetime(2023, 1, 1, 12, 0, 0)

        result = job._extract_final_metrics()

        assert len(result) == 1
        assert result[0]["MetricName"] == "valid"

    @patch("sagemaker.local.entities.datetime")
    def test_extract_final_metrics_uses_current_time_when_no_end_time(self, mock_datetime):
        """Test _extract_final_metrics uses current time when end_time is None."""
        container = Mock()
        container.logs = "GAN_loss=0.138318;"
        container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"}
        ]
        job = _LocalTrainingJob(container)
        job.end_time = None

        mock_now = datetime(2023, 1, 1, 12, 0, 0)
        mock_datetime.now.return_value = mock_now
        
        result = job._extract_final_metrics()

        assert len(result) == 1
        assert result[0]["Timestamp"] == mock_now

    @patch("sagemaker.local.image._SageMakerContainer.train", return_value="/some/path/to/model")
    def test_integration_describe_training_job_with_metrics(self, mock_train):
        """Integration test: describe_training_job includes FinalMetricDataList."""
        from sagemaker.local.local_session import LocalSagemakerClient
        
        local_sagemaker_client = LocalSagemakerClient()
        
        algo_spec = {"TrainingImage": "my-image:1.0"}
        input_data_config = [{
            "ChannelName": "training",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3Uri": "s3://bucket/data"
                }
            }
        }]
        output_data_config = {}
        resource_config = {"InstanceType": "local", "InstanceCount": 1}
        
        # Create training job
        local_sagemaker_client.create_training_job(
            "test-job",
            algo_spec,
            output_data_config,
            resource_config,
            InputDataConfig=input_data_config,
            HyperParameters={}
        )
        
        # Mock the container logs and metric definitions
        training_job = local_sagemaker_client._training_jobs["test-job"]
        training_job.container.logs = "GAN_loss=0.138318;"
        training_job.container.metric_definitions = [
            {"Name": "ganloss", "Regex": r"GAN_loss=([\d\.]+);"}
        ]
        
        response = local_sagemaker_client.describe_training_job("test-job")
        
        assert "FinalMetricDataList" in response
        assert len(response["FinalMetricDataList"]) == 1
        assert response["FinalMetricDataList"][0]["MetricName"] == "ganloss"
        assert response["FinalMetricDataList"][0]["Value"] == 0.138318
