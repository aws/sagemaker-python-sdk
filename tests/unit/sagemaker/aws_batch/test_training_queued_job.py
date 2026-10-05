from __future__ import absolute_import

import pytest
import time
from mock.mock import patch
from unittest.mock import Mock

from sagemaker.aws_batch.exception import NoTrainingJob, MissingRequiredArgument
from sagemaker.aws_batch.training_queued_job import TrainingQueuedJob
from sagemaker.config import SAGEMAKER, TRAINING_JOB
from .constants import (
    JOB_ARN,
    JOB_NAME,
    REASON,
    TRAINING_IMAGE,
    JOB_STATUS_RUNNING,
    JOB_STATUS_RUNNABLE,
    JOB_STATUS_FAILED,
    JOB_STATUS_COMPLETED,
    EXECUTION_ROLE,
    TRAINING_JOB_ARN,
)
from tests.unit import SAGEMAKER_CONFIG_TRAINING_JOB


@patch("sagemaker.aws_batch.training_queued_job.terminate_service_job")
def test_queued_job_terminate(patched_terminate_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    queued_job.terminate(REASON)
    patched_terminate_service_job.assert_called_once_with(queued_job.job_arn, REASON)


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_describe(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    queued_job.describe()
    patched_describe_service_job.assert_called_once_with(queued_job.job_arn)


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_estimator_no_training_job_created(patched_describe_service_job):
    patched_describe_service_job.return_value = {"status": JOB_STATUS_RUNNABLE}
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    with pytest.raises(NoTrainingJob):
        queued_job.get_estimator()


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_estimator_missing_required_argument(patched_describe_service_job):
    patched_describe_service_job.return_value = {"status": JOB_STATUS_RUNNING}
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    with pytest.raises(MissingRequiredArgument):
        queued_job.get_estimator()


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
@patch("sagemaker.aws_batch.training_queued_job._construct_estimator_from_training_job_name")
def test_queued_job_estimator_happy_case(
    patched_construct_estimator_from_training_job_name, patched_describe_service_job
):
    training_job_config = SAGEMAKER_CONFIG_TRAINING_JOB[SAGEMAKER][TRAINING_JOB]
    training_job_config["image_uri"] = TRAINING_IMAGE
    training_job_config["job_name"] = JOB_NAME
    training_job_config["role"] = EXECUTION_ROLE
    describe_resp = {
        "status": JOB_STATUS_RUNNING,
        "latestAttempt": {
            "serviceResourceId": {"name": "trainingJobArn", "value": TRAINING_JOB_ARN}
        },
    }
    patched_describe_service_job.return_value = describe_resp

    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    queued_job.get_estimator()
    patched_construct_estimator_from_training_job_name.assert_called_once_with(JOB_NAME)


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_wait_no_timeout(patched_describe_service_job):
    patched_describe_service_job.return_value = {"status": JOB_STATUS_COMPLETED}
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    result = queued_job.wait()
    assert result.get("status", "") == JOB_STATUS_COMPLETED


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_wait_with_timeout_succeeds(patched_describe_service_job):
    patched_describe_service_job.side_effect = [
        {"status": JOB_STATUS_RUNNING},
        {"status": JOB_STATUS_RUNNING},
        {"status": JOB_STATUS_COMPLETED},
    ]
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    start_time = time.time()
    result = queued_job.wait(timeout=15)
    end_time = time.time()

    assert end_time - start_time < 15
    assert result.get("status", "") == JOB_STATUS_COMPLETED
    assert patched_describe_service_job.call_count == 3


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queued_job_wait_with_timeout_times_out(patched_describe_service_job):
    patched_describe_service_job.return_value = {"status": JOB_STATUS_RUNNING}
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    start_time = time.time()
    result = queued_job.wait(timeout=5)
    end_time = time.time()

    assert end_time - start_time > 5
    assert result.get("status", "") == JOB_STATUS_RUNNING


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
@pytest.mark.asyncio
async def test_queued_job_async_fetch_job_results_happy_case(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

    queued_job.wait = Mock()
    # queued_job.describe.return_value = {"status": JOB_STATUS_COMPLETED}
    patched_describe_service_job.return_value = {"status": JOB_STATUS_COMPLETED}

    result = await queued_job.fetch_job_results()
    assert result == {"status": JOB_STATUS_COMPLETED}


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
@pytest.mark.asyncio
async def test_queued_job_async_fetch_job_results_job_failed(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

    queued_job.wait = Mock()
    patched_describe_service_job.return_value = {
        "status": JOB_STATUS_FAILED,
        "statusReason": "Job failed",
    }

    with pytest.raises(RuntimeError):
        await queued_job.fetch_job_results()


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
@pytest.mark.asyncio
async def test_queued_job_async_fetch_job_results_timeout(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)

    queued_job.wait = Mock()
    patched_describe_service_job.return_value = {"status": JOB_STATUS_RUNNING}

    with pytest.raises(TimeoutError):
        await queued_job.fetch_job_results(timeout=1)


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queue_result_happy_case(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    patched_describe_service_job.return_value = {"status": JOB_STATUS_COMPLETED}

    result = queued_job.result(100)
    assert result == {"status": JOB_STATUS_COMPLETED}


@patch("sagemaker.aws_batch.training_queued_job.describe_service_job")
def test_queue_result_job_times_out(patched_describe_service_job):
    queued_job = TrainingQueuedJob(JOB_ARN, JOB_NAME)
    patched_describe_service_job.return_value = {"status": JOB_STATUS_RUNNING}

    with pytest.raises(TimeoutError):
        queued_job.result(1)
