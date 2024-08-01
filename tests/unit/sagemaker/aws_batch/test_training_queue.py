from __future__ import absolute_import
from sagemaker.aws_batch.constants import DEFAULT_TIMEOUT
from sagemaker.aws_batch.exception import MissingRequiredArgument
from sagemaker.aws_batch.training_queue import TrainingQueue

from unittest.mock import Mock, call
from mock.mock import patch
import pytest

from sagemaker.modules.train.model_trainer import ModelTrainer, Mode
from sagemaker.estimator import _TrainingJob
from .constants import (
    JOB_QUEUE,
    JOB_NAME,
    DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
    SCHEDULING_PRIORITY,
    SHARE_IDENTIFIER,
    TIMEOUT_CONFIG,
    BATCH_TAGS,
    JOB_ARN,
    SUBMIT_SERVICE_JOB_RESP,
    JOB_NAME_IN_PAYLOAD,
    JOB_STATUS_RUNNING,
    EMPTY_LIST_SERVICE_JOB_RESP,
    FIRST_LIST_SERVICE_JOB_RESP,
    INCORRECT_FIRST_LIST_SERVICE_JOB_RESP,
    EXPERIMENT_CONFIG_EMPTY,
    SECOND_LIST_SERVICE_JOB_RESP,
    TRAINING_JOB_PAYLOAD_IN_PASCALCASE,
)
from .mock_estimator import Estimator, PyTorch


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_submit_with_timeout(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

    queue = TrainingQueue(JOB_QUEUE)
    queue_job = queue.submit(
        Estimator(),
        {},
        JOB_NAME,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        TIMEOUT_CONFIG,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    patched_submit_service_job.assert_called_once_with(
        TRAINING_JOB_PAYLOAD_IN_PASCALCASE,
        JOB_NAME,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        TIMEOUT_CONFIG,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )
    assert queue_job.job_name == JOB_NAME
    assert queue_job.job_arn == JOB_ARN


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_submit_use_default_timeout(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

    queue = TrainingQueue(JOB_QUEUE)
    queue.submit(
        Estimator(),
        {},
        JOB_NAME,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        None,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    patched_submit_service_job.assert_called_once_with(
        TRAINING_JOB_PAYLOAD_IN_PASCALCASE,
        JOB_NAME,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        DEFAULT_TIMEOUT,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_submit_with_job_name(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

    queue = TrainingQueue(JOB_QUEUE)
    queue.submit(
        Estimator(),
        {},
        None,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        TIMEOUT_CONFIG,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    patched_submit_service_job.assert_called_once_with(
        TRAINING_JOB_PAYLOAD_IN_PASCALCASE,
        JOB_NAME_IN_PAYLOAD,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        TIMEOUT_CONFIG,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_submit_encounter_error(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = {}

    queue = TrainingQueue(JOB_QUEUE)
    with pytest.raises(MissingRequiredArgument):
        queue.submit(
            Estimator(),
            {},
            JOB_NAME,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
            EXPERIMENT_CONFIG_EMPTY,
        )


def test_queue_map_with_job_names_mismatch_input_length_encounter_error():
    queue = TrainingQueue(JOB_QUEUE)
    with pytest.raises(ValueError):
        queue.map(Estimator(), {}, [JOB_NAME])


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_map_happy_case(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
    input_list = {"test-input", "test-input-2"}

    queue = TrainingQueue(JOB_QUEUE)
    queue.map(
        Estimator(),
        input_list,
        None,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        TIMEOUT_CONFIG,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    assert patched_submit_service_job.call_count == len(input_list)


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_queue_map_with_job_names(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP
    input_list = {"test-input", "test-input-2"}
    job_names = [JOB_NAME, "job-name-2"]

    queue = TrainingQueue(JOB_QUEUE)
    queue.map(
        Estimator(),
        input_list,
        job_names,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        TIMEOUT_CONFIG,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    assert patched_submit_service_job.call_count == len(input_list)


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_list_default_argument(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    patched_list_service_job.return_value = [{"jobSummaryList": [], "nextToken": None}]
    queue.list_jobs()
    patched_list_service_job.assert_has_calls([call(JOB_QUEUE, JOB_STATUS_RUNNING, None, None)])


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_list_happy_case_with_job_name(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]

    patched_list_service_job.return_value = [{"jobSummaryList": [], "nextToken": None}]

    queue.list_jobs(JOB_NAME, None)
    patched_list_service_job.assert_has_calls([call(JOB_QUEUE, None, filters, None)])


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_list_happy_case_with_job_status(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = None

    patched_list_service_job.return_value = [EMPTY_LIST_SERVICE_JOB_RESP]

    queue.list_jobs(None, JOB_STATUS_RUNNING)
    patched_list_service_job.assert_has_calls([call(JOB_QUEUE, JOB_STATUS_RUNNING, filters, None)])


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_list_happy_case_has_next_token(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]

    first_output = FIRST_LIST_SERVICE_JOB_RESP
    second_output = SECOND_LIST_SERVICE_JOB_RESP
    third_output = EMPTY_LIST_SERVICE_JOB_RESP
    patched_list_service_job.return_value = iter([first_output, second_output, third_output])

    jobs = queue.list_jobs(JOB_NAME, JOB_STATUS_RUNNING)
    patched_list_service_job.assert_has_calls(
        [call(JOB_QUEUE, None, filters, None)],
        any_order=False,
    )
    assert len(jobs) == 3
    assert jobs[0].job_arn == JOB_ARN
    assert jobs[0].job_name == JOB_NAME


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_list_without_job_arn_in_list_resp(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]

    first_output = INCORRECT_FIRST_LIST_SERVICE_JOB_RESP
    second_output = EMPTY_LIST_SERVICE_JOB_RESP
    patched_list_service_job.return_value = iter([first_output, second_output])

    jobs = queue.list_jobs(JOB_NAME, JOB_STATUS_RUNNING)
    patched_list_service_job.assert_has_calls(
        [call(JOB_QUEUE, None, filters, None)],
        any_order=False,
    )
    assert len(jobs) == 0


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_get_happy_case_job_exists(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]

    patched_list_service_job.return_value = [FIRST_LIST_SERVICE_JOB_RESP]

    job = queue.get_job(JOB_NAME)
    patched_list_service_job.assert_has_calls(
        [call(JOB_QUEUE, None, filters, None)],
        any_order=False,
    )
    assert job.job_name == JOB_NAME


@patch("sagemaker.aws_batch.training_queue.list_service_job")
def test_queue_get_job_not_found_encounter_error(patched_list_service_job):
    queue = TrainingQueue(JOB_QUEUE)
    filters = [{"name": "JOB_NAME", "values": [JOB_NAME]}]

    patched_list_service_job.return_value = [EMPTY_LIST_SERVICE_JOB_RESP]

    with pytest.raises(ValueError):
        queue.get_job(JOB_NAME)
    patched_list_service_job.assert_has_calls([call(JOB_QUEUE, None, filters, None)])


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_submit_model_trainer(patch_submit_service_job):
    trainer = Mock(spec=ModelTrainer)
    trainer.training_mode = Mode.SAGEMAKER_TRAINING_JOB
    payload = {
        "TrainingJobName": JOB_NAME,
        "ResourceConfig": {
            "InstanceType": "ml.m5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
    }
    trainer._create_training_job_args.return_value = payload

    patch_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

    queue = TrainingQueue(JOB_QUEUE)
    queue_job = queue.submit(
        trainer,
        [],
        JOB_NAME,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        TIMEOUT_CONFIG,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    patch_submit_service_job.assert_called_once_with(
        payload,
        JOB_NAME,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        TIMEOUT_CONFIG,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )
    assert queue_job.job_name == JOB_NAME
    assert queue_job.job_arn == JOB_ARN


def test_submit_model_trainer_fail():
    trainer = Mock(spec=ModelTrainer)
    trainer.training_mode = Mode.LOCAL_CONTAINER

    with pytest.raises(
        ValueError,
        match="TrainingQueue requires using a ModelTrainer with Mode.SAGEMAKER_TRAINING_JOB",
    ):
        queue = TrainingQueue(JOB_QUEUE)
        queue.submit(
            trainer,
            [],
            JOB_NAME,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
            EXPERIMENT_CONFIG_EMPTY,
        )


@patch("sagemaker.aws_batch.training_queue.submit_service_job")
def test_submit_pytorch_estimator(patched_submit_service_job):
    training_job_cls = _TrainingJob
    training_job_cls.get_train_args = Mock(return_value=TRAINING_JOB_PAYLOAD_IN_PASCALCASE)

    patched_submit_service_job.return_value = SUBMIT_SERVICE_JOB_RESP

    queue = TrainingQueue(JOB_QUEUE)
    queue_job = queue.submit(
        PyTorch(),
        {},
        JOB_NAME,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        SHARE_IDENTIFIER,
        DEFAULT_TIMEOUT,
        BATCH_TAGS,
        EXPERIMENT_CONFIG_EMPTY,
    )
    patched_submit_service_job.assert_called_once_with(
        TRAINING_JOB_PAYLOAD_IN_PASCALCASE,
        JOB_NAME,
        JOB_QUEUE,
        DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
        SCHEDULING_PRIORITY,
        DEFAULT_TIMEOUT,
        SHARE_IDENTIFIER,
        BATCH_TAGS,
    )
    assert queue_job.job_name == JOB_NAME
    assert queue_job.job_arn == JOB_ARN


def test_submit_with_invalid_training_job():
    with pytest.raises(
        TypeError,
        match="training_job must be an instance of EstimatorBase or ModelTrainer",
    ):
        queue = TrainingQueue(JOB_QUEUE)
        queue.submit(
            TrainingQueue("NotAnEstimatorOrModelTrainer"),
            [],
            JOB_NAME,
            DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG,
            SCHEDULING_PRIORITY,
            SHARE_IDENTIFIER,
            TIMEOUT_CONFIG,
            BATCH_TAGS,
            EXPERIMENT_CONFIG_EMPTY,
        )
