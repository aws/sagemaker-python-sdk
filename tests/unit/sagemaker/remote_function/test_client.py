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

import threading
import time

import pytest
from mock import MagicMock, patch, Mock, ANY, call
from sagemaker.exceptions import UnexpectedStatusException

from botocore.exceptions import ClientError
from sagemaker import Session
from sagemaker.experiments.experiment import Experiment
from sagemaker.experiments.run import Run
from sagemaker.remote_function.client import (
    remote,
    RemoteExecutor,
    Future,
    get_future,
    list_futures,
)
from sagemaker.remote_function.errors import DeserializationError, RemoteFunctionError, ServiceError
from sagemaker.remote_function.runtime_environment.runtime_environment_manager import (
    RuntimeEnvironmentError,
)
from sagemaker.remote_function.job import _RunInfo

from tests.unit.sagemaker.experiments.helpers import (
    mock_tc_load_or_create_func,
    mock_trial_load_or_create_func,
)

TRAINING_JOB_ARN = "training-job-arn"
TRAINING_JOB_NAME = "job-name"
IMAGE = "image_uri"
BUCKET = "my-s3-bucket"
S3_URI = f"s3://{BUCKET}/keyprefix"
EXPECTED_JOB_RESULT = [1, 2, 3]
PATH_TO_SRC_DIR = "path/to/src/dir"
HMAC_KEY = "some-hmac-key"


def describe_training_job_response(job_status):
    return {
        "TrainingJobName": TRAINING_JOB_NAME,
        "TrainingJobArn": TRAINING_JOB_ARN,
        "TrainingJobStatus": job_status,
        "ResourceConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.c4.xlarge",
            "VolumeSizeInGB": 30,
        },
        "OutputDataConfig": {"S3OutputPath": "s3://sagemaker-123/image_uri/output"},
        "Environment": {"REMOTE_FUNCTION_SECRET_KEY": HMAC_KEY},
    }


COMPLETED_TRAINING_JOB = describe_training_job_response("Completed")
INPROGRESS_TRAINING_JOB = describe_training_job_response("InProgress")
CANCELLED_TRAINING_JOB = describe_training_job_response("Stopped")
FAILED_TRAINING_JOB = describe_training_job_response("Failed")

API_CALL_LIMIT = {
    "SubmittingIntervalInSecs": 0.005,
    "MinBatchPollingIntervalInSecs": 0.01,
    "PollingIntervalInSecs": 0.01,
}


@pytest.fixture
def client():
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture
def sagemaker_session(client):
    return Session(
        sagemaker_client=client,
    )


@pytest.fixture
def run_obj(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.update_trial_component.return_value = {}
    client.associate_trial_component.return_value = {}
    with patch(
        "sagemaker.experiments.run.Experiment._load_or_create",
        MagicMock(
            return_value=Experiment(experiment_name="test-exp", sagemaker_session=sagemaker_session)
        ),
    ):
        with patch(
            "sagemaker.experiments.run._TrialComponent._load_or_create",
            MagicMock(side_effect=mock_tc_load_or_create_func),
        ):
            with patch(
                "sagemaker.experiments.run._Trial._load_or_create",
                MagicMock(side_effect=mock_trial_load_or_create_func),
            ):
                run = Run(
                    experiment_name="test-exp",
                    sagemaker_session=sagemaker_session,
                )
                run._artifact_uploader = Mock()
                run._lineage_artifact_tracker = Mock()
                run._metrics_manager = Mock()

                return run


def create_mock_job(job_name, describe_return):
    mock_job = Mock(job_name=job_name, s3_uri=S3_URI)
    mock_job.describe.return_value = describe_return

    return mock_job


def job_function(a, b=1, *, c, d=3):
    return a * b * c * d


def job_function2(a, b):
    # uses positional-only args
    return a**b


def inner_func_0():
    return 1 / 0


def inner_func_1():
    return inner_func_0()


def inner_func_2():
    raise ValueError("some value error")


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator(mock_start, mock_job_settings, mock_deserialize_obj_from_s3):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = COMPLETED_TRAINING_JOB

    mock_start.return_value = mock_job

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    result = square(5)
    assert result == EXPECTED_JOB_RESULT
    assert mock_job_settings.call_args.kwargs["image_uri"] == IMAGE


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_exception_from_s3",
    return_value=ZeroDivisionError("division by zero"),
)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_failed_remote_error_client_function(
    mock_start, mock_job_settings, mock_deserialize_exception_from_s3
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(ZeroDivisionError, match=r"division by zero"):
        square(5)


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_failed_no_exception_in_s3(
    mock_start, mock_job_settings, read_bytes
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = FAILED_TRAINING_JOB
    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        RemoteFunctionError,
        match=r"Failed to execute remote function. Check corresponding job for details.",
    ):
        square(5)


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_failed_runtime_environment_error(
    mock_start, mock_job_settings, read_bytes
):
    failed_training_job = FAILED_TRAINING_JOB.copy()
    failed_training_job.update(
        {"FailureReason": "RuntimeEnvironmentError: failure while installing dependencies."}
    )
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = failed_training_job
    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        RuntimeEnvironmentError,
        match=r"failure while installing dependencies.",
    ):
        square(5)


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_job_failed_failure_reason_without_runtime_environment_error(
    mock_start, mock_job_settings, read_bytes
):
    failed_training_job = FAILED_TRAINING_JOB.copy()
    failed_training_job.update({"FailureReason": "failure while installing dependencies."})
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = failed_training_job
    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        RemoteFunctionError,
        match=r"Failed to execute remote function. Check corresponding job for details.",
    ):
        square(5)


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_failed_local_error_service_error(
    mock_start, mock_job_settings, read_bytes
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = FAILED_TRAINING_JOB
    re = RuntimeError("some error when reading from s3")
    read_bytes.side_effect = re

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        ServiceError,
        match=r"Failed to read serialized bytes from .+: RuntimeError\('some error when reading from s3'\)",
    ):
        square(5)


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_exception_from_s3",
    side_effect=DeserializationError("Failed to deserialize the exception."),
)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_failed_local_error_remote_function_error(
    mock_start, mock_job_settings, mock_deserialize_exception_from_s3
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Failed",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        DeserializationError,
        match=r"Failed to deserialize the exception.",
    ):
        square(5)
    assert mock_job_settings.call_args.kwargs["image_uri"] == IMAGE


@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_stopped_somehow(mock_start, mock_job_settings):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = CANCELLED_TRAINING_JOB

    mock_start.return_value = mock_job

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(RemoteFunctionError, match=r"Job for remote function has been aborted."):
        square(5)


def test_decorator_instance_count_greater_than_one():
    @remote(image_uri=IMAGE, s3_root_uri=S3_URI, instance_count=2)
    def square(x):
        return x * x

    with pytest.raises(
        ValueError, match=r"Remote function do not support training on multi instances."
    ):
        square(5)


@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_underlying_job_timed_out(mock_start, mock_job_settings):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = INPROGRESS_TRAINING_JOB

    mock_start.return_value = mock_job
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="InProgress",
    )

    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def square(x):
        return x * x

    with pytest.raises(
        TimeoutError,
        match=r"Job for remote function timed out before reaching a termination status.",
    ):
        square(5)


@patch(
    "sagemaker.remote_function.core.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_no_arguments(mock_start, mock_job_settings, mock_deserialize):
    mock_job = Mock(job_name=TRAINING_JOB_NAME)
    mock_job.describe.return_value = COMPLETED_TRAINING_JOB

    mock_start.return_value = mock_job

    @remote
    def square(x):
        return x * x

    result = square(5)
    assert result == EXPECTED_JOB_RESULT
    assert mock_job_settings.call_args.kwargs["image_uri"] is None


@pytest.mark.parametrize(
    "args, kwargs, error_message",
    [
        (
            [1, 2, 3],
            {},
            "decorated_function() missing 2 required keyword-only arguments: 'd', and 'e'",
        ),
        ([1, 2, 3], {"d": 4}, "decorated_function() missing 1 required keyword-only argument: 'e'"),
        (
            [1, 2, 3],
            {"d": 3, "e": 4, "g": "extra_arg"},
            "decorated_function() got an unexpected keyword argument 'g'",
        ),
        (
            [],
            {"c": 3, "d": 4},
            "decorated_function() missing 2 required positional arguments: 'a', and 'b'",
        ),
        ([1], {"c": 3, "d": 4}, "decorated_function() missing 1 required positional argument: 'b'"),
        (
            [1, 2, 3, "extra_arg"],
            {"d": 3, "e": 4},
            "decorated_function() takes 3 positional arguments but 4 were given.",
        ),
        ([], {"a": 1, "b": 2, "d": 3, "e": 2}, None),
        (
            (1, 2),
            {"a": 1, "c": 3, "d": 2},
            "decorated_function() got multiple values for argument 'a'",
        ),
        (
            (1, 2),
            {"b": 1, "c": 3, "d": 2},
            "decorated_function() got multiple values for argument 'b'",
        ),
    ],
)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_decorator_invalid_function_args(
    mock_job_start, mock_job_settings, args, kwargs, error_message
):
    @remote(image_uri=IMAGE, s3_root_uri=S3_URI)
    def decorated_function(a, b, c=1, *, d, e, f=3):
        return a * b * c * d * e * f

    if error_message:
        with pytest.raises(TypeError) as e:
            decorated_function(*args, **kwargs)
        assert error_message in str(e.value)
    else:
        try:
            decorated_function(*args, **kwargs)
        except Exception as ex:
            pytest.fail("Unexpected Exception: " + str(ex))


def test_executor_invalid_arguments():
    with pytest.raises(ValueError):
        with RemoteExecutor(max_parallel_jobs=0, s3_root_uri="s3://bucket/") as e:
            e.submit(job_function, 1, 2, c=3, d=4)


@patch("sagemaker.remote_function.client._JobSettings")
def test_executor_submit_after_shutdown(*args):
    with pytest.raises(RuntimeError):
        with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as e:
            pass
        e.submit(job_function, 1, 2, c=3, d=4)


@pytest.mark.parametrize("parallelism", [1, 2, 3, 4])
@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_submit_happy_case(mock_start, mock_job_settings, parallelism):
    mock_job_1 = create_mock_job("job_1", COMPLETED_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", COMPLETED_TRAINING_JOB)
    mock_job_3 = create_mock_job("job_3", COMPLETED_TRAINING_JOB)
    mock_job_4 = create_mock_job("job_4", COMPLETED_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2, mock_job_3, mock_job_4]

    with RemoteExecutor(max_parallel_jobs=parallelism, s3_root_uri="s3://bucket/") as e:
        future_1 = e.submit(job_function, 1, 2, c=3, d=4)
        future_2 = e.submit(job_function, 5, 6, c=7, d=8)
        future_3 = e.submit(job_function, 9, 10, c=11, d=12)
        future_4 = e.submit(job_function, 13, 14, c=15, d=16)

    mock_start.assert_has_calls(
        [
            call(ANY, job_function, (1, 2), {"c": 3, "d": 4}, None),
            call(ANY, job_function, (5, 6), {"c": 7, "d": 8}, None),
            call(ANY, job_function, (9, 10), {"c": 11, "d": 12}, None),
            call(ANY, job_function, (13, 14), {"c": 15, "d": 16}, None),
        ]
    )
    mock_job_1.describe.assert_called()
    mock_job_2.describe.assert_called()
    mock_job_3.describe.assert_called()
    mock_job_4.describe.assert_called()

    assert future_1.done()
    assert future_2.done()
    assert future_3.done()
    assert future_4.done()


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_submit_with_run(mock_start, mock_job_settings, run_obj):
    mock_job_1 = create_mock_job("job_1", COMPLETED_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", COMPLETED_TRAINING_JOB)
    mock_job_3 = create_mock_job("job_3", COMPLETED_TRAINING_JOB)
    mock_job_4 = create_mock_job("job_4", COMPLETED_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2, mock_job_3, mock_job_4]

    run_info = _RunInfo(run_obj.experiment_name, run_obj.run_name)

    with run_obj:
        with RemoteExecutor(max_parallel_jobs=2, s3_root_uri="s3://bucket/") as e:
            future_1 = e.submit(job_function, 1, 2, c=3, d=4)
            future_2 = e.submit(job_function, 5, 6, c=7, d=8)

    mock_start.assert_has_calls(
        [
            call(ANY, job_function, (1, 2), {"c": 3, "d": 4}, run_info),
            call(ANY, job_function, (5, 6), {"c": 7, "d": 8}, run_info),
        ]
    )
    mock_job_1.describe.assert_called()
    mock_job_2.describe.assert_called()

    assert future_1.done()
    assert future_2.done()

    with RemoteExecutor(max_parallel_jobs=2, s3_root_uri="s3://bucket/") as e:
        with run_obj:
            future_3 = e.submit(job_function, 9, 10, c=11, d=12)
            future_4 = e.submit(job_function, 13, 14, c=15, d=16)

    mock_start.assert_has_calls(
        [
            call(ANY, job_function, (9, 10), {"c": 11, "d": 12}, run_info),
            call(ANY, job_function, (13, 14), {"c": 15, "d": 16}, run_info),
        ]
    )
    mock_job_3.describe.assert_called()
    mock_job_4.describe.assert_called()

    assert future_3.done()
    assert future_4.done()


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_submit_enforcing_max_parallel_jobs(mock_start, *args):
    mock_job_1 = create_mock_job("job_1", INPROGRESS_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", INPROGRESS_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2]

    e = RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/")
    future_1 = e.submit(job_function, 1, 2, c=3, d=4)
    future_2 = e.submit(job_function, 5, 6, c=7, d=8)

    time.sleep(0.02)

    assert future_1.running()
    assert not future_2.running()
    mock_start.assert_called_with(ANY, job_function, (1, 2), {"c": 3, "d": 4}, None)

    mock_job_1.describe.return_value = COMPLETED_TRAINING_JOB
    mock_job_2.describe.return_value = COMPLETED_TRAINING_JOB

    e.shutdown()

    mock_start.assert_called_with(ANY, job_function, (5, 6), {"c": 7, "d": 8}, None)
    mock_job_1.describe.assert_called()
    mock_job_2.describe.assert_called()

    assert future_1.done()
    assert future_2.done()


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_fails_to_start_job(mock_start, *args):
    mock_job = Mock()
    mock_job.describe.return_value = COMPLETED_TRAINING_JOB

    mock_start.side_effect = [TypeError(), mock_job]

    with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as e:
        future_1 = e.submit(job_function, 1, 2, c=3, d=4)
        future_2 = e.submit(job_function, 5, 6, c=7, d=8)

    with pytest.raises(TypeError):
        future_1.result()
    print(future_2._state)
    assert future_2.done()


def test_executor_instance_count_greater_than_one():
    with pytest.raises(
        ValueError, match=r"Remote function do not support training on multi instances."
    ):
        with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/", instance_count=2) as e:
            e.submit(job_function, 1, 2, c=3, d=4)


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_submit_and_cancel(mock_start, *args):
    mock_job_1 = create_mock_job("job_1", INPROGRESS_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", INPROGRESS_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2]

    e = RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/")

    # submit first job and stay in progress
    future_1 = e.submit(job_function, 1, 2, c=3, d=4)

    # submit second job and cancel
    future_2 = e.submit(job_function, 5, 6, c=7, d=8)
    future_2.cancel()

    # let the first job complete
    mock_job_1.describe.return_value = COMPLETED_TRAINING_JOB
    e.shutdown()

    mock_start.assert_called_once_with(ANY, job_function, (1, 2), {"c": 3, "d": 4}, None)
    mock_job_1.describe.assert_called()

    assert future_1.done()
    assert future_2.cancelled()


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_describe_job_throttled_temporarily(mock_start, *args):
    throttling_error = ClientError(
        error_response={"Error": {"Code": "LimitExceededException"}},
        operation_name="SomeOperation",
    )
    mock_job = Mock()
    mock_job.describe.side_effect = [
        throttling_error,
        throttling_error,
        COMPLETED_TRAINING_JOB,
        COMPLETED_TRAINING_JOB,
        COMPLETED_TRAINING_JOB,
        COMPLETED_TRAINING_JOB,
    ]
    mock_start.return_value = mock_job

    with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as e:
        # submit first job
        future_1 = e.submit(job_function, 1, 2, c=3, d=4)
        # submit second job
        future_2 = e.submit(job_function, 5, 6, c=7, d=8)

    assert future_1.done()
    assert future_2.done()


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
def test_executor_describe_job_failed_permanently(mock_start, *args):
    mock_job = Mock()
    mock_job.describe.side_effect = RuntimeError()
    mock_start.return_value = mock_job

    with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as e:
        # submit first job
        future_1 = e.submit(job_function, 1, 2, c=3, d=4)
        # submit second job
        future_2 = e.submit(job_function, 5, 6, c=7, d=8)

    with pytest.raises(RuntimeError):
        future_1.done()
    with pytest.raises(RuntimeError):
        future_2.done()


@pytest.mark.parametrize(
    "args, kwargs, error_message",
    [
        ((1, 2), {}, "job_function() missing 1 required keyword-only argument: 'c'"),
        (
            (1, 2),
            {"c": 3, "d": 4, "e": "extra_arg"},
            "job_function() got an unexpected keyword argument 'e'",
        ),
        ((), {"c": 3, "d": 4}, "job_function() missing 1 required positional argument: 'a'"),
        (
            (1, 2, "extra_Arg"),
            {"c": 3, "d": 4},
            "job_function() takes 2 positional arguments but 3 were given.",
        ),
    ],
)
@patch("sagemaker.remote_function.client._JobSettings")
def test_executor_submit_invalid_function_args(mock_job_settings, args, kwargs, error_message):
    with pytest.raises(TypeError) as e:
        with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as executor:
            executor.submit(job_function, *args, **kwargs)
    assert error_message in str(e.value)


@patch("sagemaker.remote_function.client._Job.start")
def test_future_cancel_before_job_starts(mock_start):
    mock_job = Mock()
    mock_start.return_value = mock_job

    future = Future()

    # cancel
    assert future.cancel()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)

    assert future.cancelled()
    assert not future.done()
    assert future.result() is None
    mock_job.stop.assert_not_called()


@patch("sagemaker.remote_function.client._Job.start")
def test_future_cancel_after_job_starts(mock_start):
    mock_job = Mock()
    mock_start.return_value = mock_job

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    # cancel
    assert future.cancel()

    assert future.cancelled()
    assert not future.done()
    assert future.result() is None
    mock_job.stop.assert_called_once()


@patch("sagemaker.remote_function.client._Job.start")
def test_future_cancel_when_job_starting(mock_start):
    mock_job = Mock()
    mock_start.return_value = mock_job

    future = Future()

    t = threading.Thread(
        target=lambda f: f._start_and_notify(Mock(), job_function, None, None),
        args=[future],
    )
    t.start()

    future.cancel()

    t.join()

    assert future.cancelled()


@patch("sagemaker.remote_function.client._Job.start")
def test_future_cancel_after_job_fails_to_start(mock_start):
    mock_start.side_effect = TypeError()

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.done()

    # cancel
    assert not future.cancel()

    assert not future.cancelled()
    assert future.done()


@patch("sagemaker.remote_function.client._Job.start")
def test_future_wait_after_job_start(mock_start):
    mock_job = Mock()
    mock_start.return_value = mock_job

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    future.wait()

    mock_job.wait.assert_called_once()


@patch("sagemaker.remote_function.client._Job.start")
def test_future_wait_before_job_start(mock_start):
    mock_job = Mock()
    mock_start.return_value = mock_job

    future = Future()

    # wait for the future to resolve until timeout
    future.wait(timeout=0.01)
    mock_job.wait.assert_not_called()

    # start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    future.wait()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_completed_job(mock_start, mock_deserialize):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI)
    mock_job.describe.return_value = COMPLETED_TRAINING_JOB

    mock_start.return_value = mock_job

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    result = future.result()

    assert result is EXPECTED_JOB_RESULT
    assert future.done()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_exception_from_s3",
    return_value=ZeroDivisionError("division by zero"),
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_failed_job_remote_error_client_function(
    mock_start, mock_deserialize
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI, hmac_key=HMAC_KEY)
    mock_start.return_value = mock_job
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(ZeroDivisionError, match=r"division by zero"):
        future.result()

    assert future.done()
    mock_job.wait.assert_called_once()
    mock_deserialize.assert_called_with(
        sagemaker_session=ANY, s3_uri=f"{S3_URI}/exception", hmac_key=HMAC_KEY
    )


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_failed_job_no_exception_in_s3(mock_start, read_bytes):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI)
    mock_start.return_value = mock_job
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(
        RemoteFunctionError,
        match=r"Failed to execute remote function. Check corresponding job for details.",
    ):
        future.result()

    assert future.done()
    mock_job.wait.assert_called_once()


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_failed_job_runtime_environment_error(mock_start, read_bytes):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI)
    mock_start.return_value = mock_job
    failed_training_job = FAILED_TRAINING_JOB.copy()
    failed_training_job.update(
        {"FailureReason": "RuntimeEnvironmentError: failure while installing dependencies."}
    )
    mock_job.describe.return_value = failed_training_job

    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(
        RuntimeEnvironmentError,
        match=r"failure while installing dependencies.",
    ):
        future.result()

    assert future.done()
    mock_job.wait.assert_called_once()


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_failed_job_local_error_service_error(mock_start, read_bytes):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI)
    mock_start.return_value = mock_job
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    read_bytes.side_effect = RuntimeError("some error when reading from s3")

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(
        ServiceError,
        match=r"Failed to read serialized bytes from .+: RuntimeError\('some error when reading from s3'\)",
    ):
        future.result()

    assert future.done()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_exception_from_s3",
    side_effect=DeserializationError("Failed to deserialize the exception."),
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_failed_job_local_error_remote_function_error(
    mock_start, mock_deserialize
):
    mock_job = Mock(job_name=TRAINING_JOB_NAME, s3_uri=S3_URI)
    mock_start.return_value = mock_job
    mock_job.describe.return_value = FAILED_TRAINING_JOB

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(
        DeserializationError,
        match=r"Failed to deserialize the exception.",
    ):
        future.result()

    assert future.done()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_in_progress_job(mock_start, mock_deserialize):
    mock_job = Mock()
    mock_start.return_value = mock_job
    mock_job.describe.return_value = INPROGRESS_TRAINING_JOB
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="InProgress",
    )

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(
        TimeoutError,
        match=r"Job for remote function timed out before reaching a termination status.",
    ):
        future.result()

    assert future.running()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_stopped_job(mock_start, mock_deserialize):
    mock_job = Mock()
    mock_start.return_value = mock_job
    mock_job.describe.return_value = CANCELLED_TRAINING_JOB
    mock_job.wait.side_effect = UnexpectedStatusException(
        message="some message",
        allowed_statuses=["Completed", "Stopped"],
        actual_status="Stopped",
    )

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.running()

    with pytest.raises(RemoteFunctionError, match=r"Job for remote function has been aborted."):
        future.result()

    assert future.cancelled()
    mock_job.wait.assert_called_once()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    return_value=EXPECTED_JOB_RESULT,
)
@patch("sagemaker.remote_function.client._Job.start")
def test_future_get_result_from_job_failed_to_start(mock_start, mock_deserialize):
    mock_start.side_effect = TypeError()

    future = Future()

    # try to start the job
    future._start_and_notify(Mock(), job_function, None, None)
    assert future.done()

    with pytest.raises(TypeError):
        future.result()


def test_future_get_result_from_not_yet_started_job():
    future = Future()

    # wait for the future to resolve until timeout
    with pytest.raises(RuntimeError):
        future.result(timeout=0.01)


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
@patch("sagemaker.remote_function.client.serialization.deserialize_obj_from_s3")
def test_executor_map_happy_case(mock_deserialized, mock_start, mock_job_settings):
    mock_job_1 = create_mock_job("job_1", COMPLETED_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", COMPLETED_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2]

    mock_deserialized.side_effect = [1, 16]

    with RemoteExecutor(max_parallel_jobs=1, s3_root_uri="s3://bucket/") as executor:
        results = executor.map(job_function2, [1, 2], [3, 4])

    mock_start.assert_has_calls(
        [
            call(ANY, job_function2, (1, 3), {}, None),
            call(ANY, job_function2, (2, 4), {}, None),
        ]
    )
    mock_job_1.describe.assert_called()
    mock_job_2.describe.assert_called()

    assert results[0] == 1
    assert results[1] == 16


@patch("sagemaker.remote_function.client._API_CALL_LIMIT", new=API_CALL_LIMIT)
@patch("sagemaker.remote_function.client._JobSettings")
@patch("sagemaker.remote_function.client._Job.start")
@patch("sagemaker.remote_function.client.serialization.deserialize_obj_from_s3")
def test_executor_map_with_run(mock_deserialized, mock_start, mock_job_settings, run_obj):
    mock_job_1 = create_mock_job("job_1", COMPLETED_TRAINING_JOB)
    mock_job_2 = create_mock_job("job_2", COMPLETED_TRAINING_JOB)
    mock_job_3 = create_mock_job("job_3", COMPLETED_TRAINING_JOB)
    mock_job_4 = create_mock_job("job_4", COMPLETED_TRAINING_JOB)
    mock_start.side_effect = [mock_job_1, mock_job_2, mock_job_3, mock_job_4]

    mock_deserialized.side_effect = [1, 16]

    run_info = _RunInfo(run_obj.experiment_name, run_obj.run_name)

    with run_obj:
        with RemoteExecutor(max_parallel_jobs=2, s3_root_uri="s3://bucket/") as executor:
            results_12 = executor.map(job_function2, [1, 2], [3, 4])

    mock_start.assert_has_calls(
        [
            call(ANY, job_function2, (1, 3), {}, run_info),
            call(ANY, job_function2, (2, 4), {}, run_info),
        ]
    )
    mock_job_1.describe.assert_called()
    mock_job_2.describe.assert_called()

    assert results_12[0] == 1
    assert results_12[1] == 16

    mock_deserialized.side_effect = [1, 16]

    with RemoteExecutor(max_parallel_jobs=2, s3_root_uri="s3://bucket/") as executor:
        with run_obj:
            results_34 = executor.map(job_function2, [1, 2], [3, 4])

    mock_start.assert_has_calls(
        [
            call(ANY, job_function2, (1, 3), {}, run_info),
            call(ANY, job_function2, (2, 4), {}, run_info),
        ]
    )
    mock_job_3.describe.assert_called()
    mock_job_4.describe.assert_called()

    assert results_34[0] == 1
    assert results_34[1] == 16


@patch("sagemaker.remote_function.client.Session")
@patch("sagemaker.remote_function.client.serialization.deserialize_obj_from_s3")
def test_get_future_completed_job(mock_deserialized, mock_session):
    job_return_val = "4.666"

    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        COMPLETED_TRAINING_JOB
    )
    mock_deserialized.return_value = job_return_val

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    assert future.result() == job_return_val


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_exception_from_s3",
    return_value=ZeroDivisionError("division by zero"),
)
@patch("sagemaker.remote_function.client.Session")
def test_get_future_failed_job(mock_session, *args):
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        FAILED_TRAINING_JOB
    )

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    with pytest.raises(ZeroDivisionError, match=r"division by zero"):
        future.result()


@patch(
    "sagemaker.remote_function.client.serialization.deserialize_obj_from_s3",
    side_effect=DeserializationError("Failed to deserialize the results."),
)
@patch("sagemaker.remote_function.client.Session")
def test_get_future_completed_job_deserialization_error(mock_session, mock_deserialize):
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        COMPLETED_TRAINING_JOB
    )

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    with pytest.raises(DeserializationError, match=r"Failed to deserialize the results."):
        future.result()

    mock_deserialize.assert_called_with(
        sagemaker_session=ANY,
        s3_uri="s3://sagemaker-123/image_uri/output/results",
        hmac_key=HMAC_KEY,
    )


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client.Session")
def test_get_future_completed_job_s3_read_error(mock_session, read_bytes):
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        COMPLETED_TRAINING_JOB
    )

    read_bytes.side_effect = RuntimeError("some error when reading from s3")

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    with pytest.raises(
        ServiceError,
        match=r"Failed to read serialized bytes from .+: RuntimeError\('some error when reading from s3'\)",
    ):
        future.result()


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client.Session")
def test_get_future_failed_job_S3_404_service_error(mock_session, read_bytes):
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        FAILED_TRAINING_JOB
    )

    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    with pytest.raises(
        RemoteFunctionError,
        match=r"Failed to execute remote function. Check corresponding job for details.",
    ):
        future.result()


@patch("sagemaker.s3.S3Downloader.read_bytes")
@patch("sagemaker.remote_function.client.Session")
def test_get_future_failed_job_S3_404_runtime_environment_error(mock_session, read_bytes):
    failed_training_job = FAILED_TRAINING_JOB.copy()
    failed_training_job.update(
        {"FailureReason": "RuntimeEnvironmentError: failure while installing dependencies."}
    )
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        failed_training_job
    )

    read_bytes.side_effect = ClientError(
        error_response={"Error": {"Code": "404", "Message": "Not Found"}},
        operation_name="HeadObject",
    )

    future = get_future(TRAINING_JOB_NAME)

    assert future.done()
    with pytest.raises(
        RuntimeEnvironmentError,
        match=r"failure while installing dependencies.",
    ):
        future.result()


@patch("sagemaker.remote_function.client.Session")
def test_get_future_incomplete_job(mock_session):
    mock_session.return_value.sagemaker_client.describe_training_job.return_value = (
        INPROGRESS_TRAINING_JOB
    )

    future = get_future(TRAINING_JOB_NAME)

    assert future.running()


@patch("sagemaker.remote_function.client.Session")
def test_list_future(mock_session):
    job_name_prefix = "foobarbaz"
    next_token = "next-token-1"
    mock_session.return_value.sagemaker_client.list_training_jobs.side_effect = [
        {
            "TrainingJobSummaries": [{"TrainingJobName": "job-1"}, {"TrainingJobName": "job-2"}],
            "NextToken": next_token,
        },
        {"TrainingJobSummaries": [{"TrainingJobName": "job-3"}]},
    ]
    mock_session.return_value.sagemaker_client.describe_training_job.side_effect = [
        INPROGRESS_TRAINING_JOB,
        COMPLETED_TRAINING_JOB,
        FAILED_TRAINING_JOB,
    ]

    futures = list(list_futures(job_name_prefix))

    assert futures[0].running()
    assert futures[1].done()
    assert futures[2].done()

    mock_session.return_value.sagemaker_client.list_training_jobs.assert_has_calls(
        [
            call(NameContains=job_name_prefix),
            call(NameContains=job_name_prefix, NextToken=next_token),
        ]
    )

    mock_session.return_value.sagemaker_client.describe_training_job.assert_has_calls(
        [
            call(TrainingJobName="job-1"),
            call(TrainingJobName="job-2"),
            call(TrainingJobName="job-3"),
        ]
    )
