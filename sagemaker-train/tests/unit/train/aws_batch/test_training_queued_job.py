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
"""Unit tests for training_queued_job module"""

import pytest
import time
import asyncio
from unittest.mock import Mock, patch, MagicMock

from sagemaker.train.aws_batch.training_queued_job import TrainingQueuedJob
from sagemaker.train.aws_batch.exception import NoTrainingJob, MissingRequiredArgument
from .conftest import (
    JOB_NAME,
    JOB_ARN,
    REASON,
    TRAINING_JOB_NAME,
    TRAINING_JOB_ARN,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    JOB_STATUS_SUCCEEDED,
    JOB_STATUS_FAILED,
    DESCRIBE_SERVICE_JOB_RESP_RUNNING,
    DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED,
    DESCRIBE_SERVICE_JOB_RESP_FAILED,
    DESCRIBE_SERVICE_JOB_RESP_PENDING,
)


class TestTrainingQueuedJobInit:
    """Tests for TrainingQueuedJob initialization"""

    def test_training_queued_job_init(self):
        """Test TrainingQueuedJob initialization"""
        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        assert queued_job.job_arn == JOB_ARN
        assert queued_job.job_name == JOB_NAME


class TestTrainingQueuedJobDescribe:
    """Tests for TrainingQueuedJob.describe method"""

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_describe(self, mock_describe_service_job):
        """Test describe returns job details"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_RUNNING

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.describe()

        assert result["status"] == JOB_STATUS_RUNNING
        mock_describe_service_job.assert_called_once_with(JOB_ARN)


class TestTrainingQueuedJobTerminate:
    """Tests for TrainingQueuedJob.terminate method"""

    @patch("sagemaker.train.aws_batch.training_queued_job.terminate_service_job")
    def test_terminate(self, mock_terminate_service_job):
        """Test terminate calls terminate API"""
        mock_terminate_service_job.return_value = {}

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        queued_job.terminate(REASON)

        mock_terminate_service_job.assert_called_once_with(JOB_ARN, REASON)

    @patch("sagemaker.train.aws_batch.training_queued_job.terminate_service_job")
    def test_terminate_default_reason(self, mock_terminate_service_job):
        """Test terminate with default reason"""
        mock_terminate_service_job.return_value = {}

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        queued_job.terminate()

        call_kwargs = mock_terminate_service_job.call_args[0]
        assert call_kwargs[0] == JOB_ARN


class TestTrainingQueuedJobWait:
    """Tests for TrainingQueuedJob.wait method"""

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_wait_immediate_completion(self, mock_describe_service_job):
        """Test wait returns immediately when job is completed"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.wait()

        assert result["status"] == JOB_STATUS_SUCCEEDED

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_wait_with_polling(self, mock_describe_service_job):
        """Test wait polls until job completes"""
        mock_describe_service_job.side_effect = [
            DESCRIBE_SERVICE_JOB_RESP_RUNNING,
            DESCRIBE_SERVICE_JOB_RESP_RUNNING,
            DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED,
        ]

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.wait()

        assert result["status"] == JOB_STATUS_SUCCEEDED
        assert mock_describe_service_job.call_count == 3

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_wait_with_timeout(self, mock_describe_service_job):
        """Test wait respects timeout"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_RUNNING

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        start_time = time.time()
        result = queued_job.wait(timeout=2)
        end_time = time.time()

        # Should timeout after approximately 2 seconds
        assert end_time - start_time >= 2
        assert result["status"] == JOB_STATUS_RUNNING

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_wait_job_failed(self, mock_describe_service_job):
        """Test wait returns failed status"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_FAILED

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.wait()

        assert result["status"] == JOB_STATUS_FAILED


class TestTrainingQueuedJobGetModelTrainer:
    """Tests for TrainingQueuedJob.get_model_trainer method"""

    @patch("sagemaker.train.aws_batch.training_queued_job._remove_system_tags_in_place_in_model_trainer_object")
    @patch("sagemaker.train.aws_batch.training_queued_job._construct_model_trainer_from_training_job_name")
    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_get_model_trainer_success(self, mock_describe_service_job, mock_construct_trainer, mock_remove_tags):
        """Test get_model_trainer returns ModelTrainer when training job created"""
        # Return a real dict (not a mock) so nested dict access works
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED
        
        mock_trainer = Mock()
        mock_construct_trainer.return_value = mock_trainer

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.get_model_trainer()

        assert result == mock_trainer
        mock_construct_trainer.assert_called_once()
        mock_remove_tags.assert_called_once_with(mock_trainer)

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_get_model_trainer_no_training_job_pending(self, mock_describe_service_job):
        """Test get_model_trainer raises error when job still pending"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_PENDING

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

        with pytest.raises(NoTrainingJob):
            queued_job.get_model_trainer()

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_get_model_trainer_no_latest_attempt(self, mock_describe_service_job):
        """Test get_model_trainer raises error when latestAttempt missing"""
        resp = DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED.copy()
        del resp["latestAttempt"]
        mock_describe_service_job.return_value = resp

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

        with pytest.raises(MissingRequiredArgument):
            queued_job.get_model_trainer()


class TestTrainingQueuedJobResult:
    """Tests for TrainingQueuedJob.result method"""

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_result_success(self, mock_describe_service_job):
        """Test result returns job result when completed"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = queued_job.result(timeout=100)

        assert result["status"] == JOB_STATUS_SUCCEEDED

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_result_timeout(self, mock_describe_service_job):
        """Test result raises TimeoutError when timeout exceeded"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_RUNNING

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

        with pytest.raises(TimeoutError):
            queued_job.result(timeout=1)


class TestTrainingQueuedJobAsync:
    """Tests for TrainingQueuedJob async methods"""

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_fetch_job_results_success(self, mock_describe_service_job):
        """Test fetch_job_results returns result when job succeeds"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        result = asyncio.run(queued_job.fetch_job_results())

        assert result["status"] == JOB_STATUS_SUCCEEDED

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_fetch_job_results_failed(self, mock_describe_service_job):
        """Test fetch_job_results raises error when job fails"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_FAILED

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

        with pytest.raises(RuntimeError):
            asyncio.run(queued_job.fetch_job_results())

    @patch("sagemaker.train.aws_batch.training_queued_job.describe_service_job")
    def test_fetch_job_results_timeout(self, mock_describe_service_job):
        """Test fetch_job_results raises TimeoutError when timeout exceeded"""
        mock_describe_service_job.return_value = DESCRIBE_SERVICE_JOB_RESP_RUNNING

        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

        with pytest.raises(TimeoutError):
            asyncio.run(queued_job.fetch_job_results(timeout=1))


class TestTrainingQueuedJobTrainingJobCreated:
    """Tests for TrainingQueuedJob._training_job_created method"""

    def test_training_job_created_running(self):
        """Test _training_job_created returns True for RUNNING status"""
        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        assert queued_job._training_job_created(JOB_STATUS_RUNNING) is True

    def test_training_job_created_succeeded(self):
        """Test _training_job_created returns True for SUCCEEDED status"""
        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        assert queued_job._training_job_created(JOB_STATUS_SUCCEEDED) is True

    def test_training_job_created_failed(self):
        """Test _training_job_created returns True for FAILED status"""
        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        assert queued_job._training_job_created(JOB_STATUS_FAILED) is True

    def test_training_job_created_pending(self):
        """Test _training_job_created returns False for PENDING status"""
        queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
        assert queued_job._training_job_created(JOB_STATUS_PENDING) is False
