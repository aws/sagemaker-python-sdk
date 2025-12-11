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
"""Unit tests for training_queue module"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.train.aws_batch.training_queue import TrainingQueue
from sagemaker.train.model_trainer import ModelTrainer, Mode
from .conftest import (
    JOB_NAME,
    JOB_QUEUE,
    JOB_ARN,
    JOB_ID,
    SCHEDULING_PRIORITY,
    SHARE_IDENTIFIER,
    TIMEOUT_CONFIG,
    BATCH_TAGS,
    DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
    SUBMIT_SERVICE_JOB_RESP,
    LIST_SERVICE_JOB_RESP_WITH_JOBS,
    LIST_SERVICE_JOB_RESP_EMPTY,
    TRAINING_JOB_PAYLOAD,
)


class TestTrainingQueueInit:
    """Tests for TrainingQueue initialization"""

    def test_training_queue_init(self):
        """Test TrainingQueue initialization"""
        queue = TrainingQueue(JOB_QUEUE)
        assert queue.queue_name == JOB_QUEUE


class TestTrainingQueueSubmit:
    """Tests for TrainingQueue.submit method"""

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_submit_model_trainer(self, mock_submit_service_job):
        """Test submit with ModelTrainer"""
        mock_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)
        queued_job = queue.submit(
            trainer,
            [],
            JOB_NAME,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
        )

        assert queued_job.job_name == JOB_NAME
        assert queued_job.job_arn == JOB_ARN
        mock_submit_service_job.assert_called_once()

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_submit_with_default_timeout(self, mock_submit_service_job):
        """Test submit uses default timeout when not provided"""
        mock_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)
        queue.submit(
            trainer,
            [],
            JOB_NAME,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            None,  # No timeout
            BATCH_TAGS,
        )

        call_kwargs = mock_submit_service_job.call_args[0]
        # Timeout should be set to default
        assert call_kwargs[5] is not None

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_submit_with_generated_job_name(self, mock_submit_service_job):
        """Test submit generates job name from payload if not provided"""
        mock_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)
        queue.submit(
            trainer,
            [],
            None,  # No job name provided
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
        )

        call_kwargs = mock_submit_service_job.call_args[0]
        # Job name should come from payload
        assert call_kwargs[1] == TRAINING_JOB_PAYLOAD["TrainingJobName"]

    def test_submit_invalid_training_job_type(self):
        """Test submit raises error for invalid training job type"""
        queue = TrainingQueue(JOB_QUEUE)

        with pytest.raises(TypeError, match="training_job must be an instance of ModelTrainer"):
            queue.submit(
                "not-a-trainer",
                [],
                JOB_NAME,
                DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
                SCHEDULING_PRIORITY,
                SHARE_IDENTIFIER,
                TIMEOUT_CONFIG,
                BATCH_TAGS,
            )

    def test_submit_invalid_training_mode(self):
        """Test submit raises error for invalid training mode"""
        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.LOCAL_CONTAINER

        queue = TrainingQueue(JOB_QUEUE)

        with pytest.raises(ValueError, match="Mode.SAGEMAKER_TRAINING_JOB"):
            queue.submit(
                trainer,
                [],
                JOB_NAME,
                DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
                SCHEDULING_PRIORITY,
                SHARE_IDENTIFIER,
                TIMEOUT_CONFIG,
                BATCH_TAGS,
            )

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_submit_missing_job_arn_in_response(self, mock_submit_service_job):
        """Test submit raises error when jobArn missing from response"""
        mock_submit_service_job.return_value = {"jobName": JOB_NAME}  # Missing jobArn

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)

        with pytest.raises(Exception):  # MissingRequiredArgument
            queue.submit(
                trainer,
                [],
                JOB_NAME,
                DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
                SCHEDULING_PRIORITY,
                SHARE_IDENTIFIER,
                TIMEOUT_CONFIG,
                BATCH_TAGS,
            )


class TestTrainingQueueMap:
    """Tests for TrainingQueue.map method"""

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_map_multiple_inputs(self, mock_submit_service_job):
        """Test map submits multiple jobs"""
        mock_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)
        inputs = ["input1", "input2", "input3"]
        queued_jobs = queue.map(
            trainer,
            inputs,
            None,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
        )

        assert len(queued_jobs) == 3
        assert mock_submit_service_job.call_count == 3

    @patch("sagemaker.train.aws_batch.training_queue.submit_service_job")
    def test_map_with_job_names(self, mock_submit_service_job):
        """Test map with explicit job names"""
        mock_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
        trainer._create_training_job_args.return_value = TRAINING_JOB_PAYLOAD

        queue = TrainingQueue(JOB_QUEUE)
        inputs = ["input1", "input2"]
        job_names = ["job1", "job2"]
        queued_jobs = queue.map(
            trainer,
            inputs,
            job_names,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
        )

        assert len(queued_jobs) == 2

    def test_map_mismatched_job_names_length(self):
        """Test map raises error when job names length doesn't match inputs"""
        trainer = Mock(spec=ModelTrainer)
        trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB

        queue = TrainingQueue(JOB_QUEUE)
        inputs = ["input1", "input2"]
        job_names = ["job1"]  # Mismatch

        with pytest.raises(ValueError, match="number of job names must match"):
            queue.map(
                trainer,
                inputs,
                job_names,
                DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
                SCHEDULING_PRIORITY,
                SHARE_IDENTIFIER,
                TIMEOUT_CONFIG,
                BATCH_TAGS,
            )


class TestTrainingQueueList:
    """Tests for TrainingQueue.list_jobs method"""

    @patch("sagemaker.train.aws_batch.training_queue.list_service_job")
    def test_list_jobs_default(self, mock_list_service_job):
        """Test list_jobs with default parameters"""
        mock_list_service_job.return_value = iter([LIST_SERVICE_JOB_RESP_WITH_JOBS])

        queue = TrainingQueue(JOB_QUEUE)
        jobs = queue.list_jobs()

        assert len(jobs) == 2
        assert jobs[0].job_name == JOB_NAME

    @patch("sagemaker.train.aws_batch.training_queue.list_service_job")
    def test_list_jobs_with_name_filter(self, mock_list_service_job):
        """Test list_jobs with job name filter"""
        mock_list_service_job.return_value = iter([LIST_SERVICE_JOB_RESP_WITH_JOBS])

        queue = TrainingQueue(JOB_QUEUE)
        jobs = queue.list_jobs(job_name=JOB_NAME)

        # Verify list_service_job was called
        mock_list_service_job.assert_called_once()
        
        # Get the call arguments - list_service_job is called with positional args:
        # list_service_job(queue_name, status, filters, next_token)
        call_args = mock_list_service_job.call_args[0]
        
        # The 3rd positional argument (index 2) is filters
        filters = call_args[2] if len(call_args) > 2 else None
        
        # Verify filters contain the job name
        assert filters is not None, "Filters should be passed to list_service_job"
        assert filters[0]["name"] == "JOB_NAME", "JOB_NAME filter should be present"
        assert filters[0]["values"] == [JOB_NAME], "Filter values should contain the job name"

    @patch("sagemaker.train.aws_batch.training_queue.list_service_job")
    def test_list_jobs_empty(self, mock_list_service_job):
        """Test list_jobs returns empty list"""
        mock_list_service_job.return_value = iter([LIST_SERVICE_JOB_RESP_EMPTY])

        queue = TrainingQueue(JOB_QUEUE)
        jobs = queue.list_jobs()

        assert len(jobs) == 0


class TestTrainingQueueGet:
    """Tests for TrainingQueue.get_job method"""

    @patch("sagemaker.train.aws_batch.training_queue.list_service_job")
    def test_get_job_found(self, mock_list_service_job):
        """Test get_job returns job when found"""
        mock_list_service_job.return_value = iter([LIST_SERVICE_JOB_RESP_WITH_JOBS])

        queue = TrainingQueue(JOB_QUEUE)
        job = queue.get_job(JOB_NAME)

        assert job.job_name == JOB_NAME
        assert job.job_arn == JOB_ARN

    @patch("sagemaker.train.aws_batch.training_queue.list_service_job")
    def test_get_job_not_found(self, mock_list_service_job):
        """Test get_job raises error when job not found"""
        mock_list_service_job.return_value = iter([LIST_SERVICE_JOB_RESP_EMPTY])

        queue = TrainingQueue(JOB_QUEUE)

        with pytest.raises(ValueError, match="Cannot find job"):
            queue.get_job(JOB_NAME)
