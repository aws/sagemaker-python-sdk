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
"""Tests for local entities module."""
from __future__ import absolute_import

import datetime
import pytest
from unittest.mock import MagicMock

from sagemaker.train.local.entities import _LocalTrainingJob


class TestLocalTrainingJob:
    """Test _LocalTrainingJob class."""

    def test_init(self):
        """Test initialization."""
        mock_container = MagicMock()
        job = _LocalTrainingJob(mock_container)
        
        assert job.container is mock_container
        assert job.model_artifacts is None
        assert job.state == "created"
        assert job.start_time is None
        assert job.end_time is None
        assert job.environment is None
        assert job.training_job_name == ""
        assert job.output_data_config is None

    def test_start_with_s3_data_source(self):
        """Test start with S3 data source."""
        mock_container = MagicMock()
        mock_container.train.return_value = "s3://bucket/model.tar.gz"
        
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/data",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ]
        output_data_config = {"S3OutputPath": "s3://bucket/output"}
        hyperparameters = {"epochs": "10"}
        environment = {"ENV_VAR": "value"}
        job_name = "test-job"
        
        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)
        
        assert job.state == job._COMPLETED
        assert job.model_artifacts == "s3://bucket/model.tar.gz"
        assert job.training_job_name == job_name
        assert job.output_data_config == output_data_config
        assert job.environment == environment
        assert job.start_time is not None
        assert job.end_time is not None
        assert isinstance(job.start_time, datetime.datetime)
        assert isinstance(job.end_time, datetime.datetime)
        
        mock_container.train.assert_called_once_with(
            input_data_config, output_data_config, hyperparameters, environment, job_name
        )

    def test_start_with_file_data_source(self):
        """Test start with file data source."""
        mock_container = MagicMock()
        mock_container.train.return_value = "file:///tmp/model.tar.gz"
        
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "FileDataSource": {
                        "FileUri": "file:///data",
                        "FileDataDistributionType": "FullyReplicated",
                    }
                },
            }
        ]
        output_data_config = {"LocalPath": "/tmp/output"}
        hyperparameters = {}
        environment = {}
        job_name = "test-job"
        
        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)
        
        assert job.state == job._COMPLETED
        assert input_data_config[0]["DataUri"] == "file:///data"

    def test_start_with_default_distribution(self):
        """Test start with default data distribution."""
        mock_container = MagicMock()
        mock_container.train.return_value = "s3://bucket/model.tar.gz"
        
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/data",
                        # No S3DataDistributionType specified
                    }
                },
            }
        ]
        output_data_config = {}
        hyperparameters = {}
        environment = {}
        job_name = "test-job"
        
        # Should not raise error
        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)
        
        assert job.state == job._COMPLETED

    def test_start_raises_error_for_invalid_data_source(self):
        """Test start raises error for invalid data source."""
        mock_container = MagicMock()
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {},  # Missing S3DataSource or FileDataSource
            }
        ]
        output_data_config = {}
        hyperparameters = {}
        environment = {}
        job_name = "test-job"
        
        with pytest.raises(ValueError, match="Need channel\\['DataSource'\\]"):
            job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)

    def test_start_raises_error_for_unsupported_distribution(self):
        """Test start raises error for unsupported distribution type."""
        mock_container = MagicMock()
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/data",
                        "S3DataDistributionType": "ShardedByS3Key",  # Not supported
                    }
                },
            }
        ]
        output_data_config = {}
        hyperparameters = {}
        environment = {}
        job_name = "test-job"
        
        with pytest.raises(RuntimeError, match="Invalid DataDistribution"):
            job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)

    def test_describe(self):
        """Test describe method."""
        mock_container = MagicMock()
        mock_container.instance_count = 1
        mock_container.container_entrypoint = ["python", "train.py"]
        mock_container.train.return_value = "s3://bucket/model.tar.gz"
        
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/data",
                    }
                },
            }
        ]
        output_data_config = {"S3OutputPath": "s3://bucket/output"}
        hyperparameters = {"epochs": "10"}
        environment = {"ENV_VAR": "value"}
        job_name = "test-job"
        
        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)
        
        response = job.describe()
        
        assert response["TrainingJobName"] == job_name
        assert response["TrainingJobArn"] == "unused-arn"
        assert response["ResourceConfig"]["InstanceCount"] == 1
        assert response["TrainingJobStatus"] == job._COMPLETED
        assert response["TrainingStartTime"] == job.start_time
        assert response["TrainingEndTime"] == job.end_time
        assert response["ModelArtifacts"]["S3ModelArtifacts"] == "s3://bucket/model.tar.gz"
        assert response["OutputDataConfig"] == output_data_config
        assert response["Environment"] == environment
        assert response["AlgorithmSpecification"]["ContainerEntrypoint"] == ["python", "train.py"]

    def test_describe_before_start(self):
        """Test describe before job starts."""
        mock_container = MagicMock()
        mock_container.instance_count = 1
        mock_container.container_entrypoint = None
        
        job = _LocalTrainingJob(mock_container)
        
        response = job.describe()
        
        assert response["TrainingJobName"] == ""
        assert response["TrainingJobStatus"] == "created"
        assert response["TrainingStartTime"] is None
        assert response["TrainingEndTime"] is None
        assert response["ModelArtifacts"]["S3ModelArtifacts"] is None

    def test_state_constants(self):
        """Test state constants are defined."""
        assert _LocalTrainingJob._STARTING == "Starting"
        assert _LocalTrainingJob._TRAINING == "Training"
        assert _LocalTrainingJob._COMPLETED == "Completed"
        assert _LocalTrainingJob._states == ["Starting", "Training", "Completed"]

    def test_multiple_channels(self):
        """Test start with multiple input channels."""
        mock_container = MagicMock()
        mock_container.train.return_value = "s3://bucket/model.tar.gz"
        
        job = _LocalTrainingJob(mock_container)
        
        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/train",
                    }
                },
            },
            {
                "ChannelName": "validation",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/val",
                    }
                },
            },
        ]
        output_data_config = {}
        hyperparameters = {}
        environment = {}
        job_name = "test-job"
        
        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)
        
        assert job.state == job._COMPLETED
        assert input_data_config[0]["DataUri"] == "s3://bucket/train"
        assert input_data_config[1]["DataUri"] == "s3://bucket/val"
