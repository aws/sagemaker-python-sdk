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
"""Generated ``wait()`` methods take ``raise_on_stopped`` and raise on a stopped resource.

Before the change ``wait()`` raised ``FailedStatusError`` only for a status that contains
``failed``. ``Stopped`` is a terminal state too, so ``TrainingJob.wait()`` returned as if
the job completed, and three training templates in the SageMaker agent skills read
``training_job_status`` after the wait and raised on ``Stopped`` themselves.
"""

from __future__ import absolute_import

import inspect
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core import resources
from sagemaker.core.resources import Base, ProcessingJob, TrainingJob
from sagemaker.core.shapes import ProcessingClusterConfig, ProcessingResources
from sagemaker.core.utils.exceptions import FailedStatusError

JOB_NAME = "my-job"


def _describe_training_job(status, failure_reason=None):
    response = {
        "TrainingJobName": JOB_NAME,
        "TrainingJobArn": f"arn:aws:sagemaker:us-west-2:123456789012:training-job/{JOB_NAME}",
        "TrainingJobStatus": status,
        "SecondaryStatus": status,
    }
    if failure_reason:
        response["FailureReason"] = failure_reason
    return response


@pytest.fixture
def training_client():
    client = MagicMock()
    with patch.object(Base, "get_sagemaker_client", return_value=client):
        yield client


def test_every_generated_wait_takes_raise_on_stopped():
    wait_methods = [
        cls.wait
        for _, cls in inspect.getmembers(resources, inspect.isclass)
        if issubclass(cls, Base) and "wait" in vars(cls)
    ]

    assert len(wait_methods) >= 10
    for wait in wait_methods:
        parameter = inspect.signature(wait).parameters["raise_on_stopped"]
        assert parameter.default is False, wait.__qualname__


def test_wait_returns_on_stopped_by_default(training_client):
    training_client.describe_training_job.return_value = _describe_training_job("Stopped")

    assert TrainingJob(training_job_name=JOB_NAME).wait(poll=0) is None


def test_wait_raises_on_stopped_when_asked(training_client):
    training_client.describe_training_job.return_value = _describe_training_job(
        "Stopped", failure_reason="Stopped by user"
    )

    with pytest.raises(FailedStatusError) as raised:
        TrainingJob(training_job_name=JOB_NAME).wait(poll=0, raise_on_stopped=True)

    message = str(raised.value)
    assert "TrainingJob" in message
    assert "Stopped" in message
    assert "Stopped by user" in message


def test_wait_with_raise_on_stopped_returns_on_completed(training_client):
    training_client.describe_training_job.return_value = _describe_training_job("Completed")

    assert TrainingJob(training_job_name=JOB_NAME).wait(poll=0, raise_on_stopped=True) is None


def test_wait_with_raise_on_stopped_still_raises_on_failed(training_client):
    training_client.describe_training_job.return_value = _describe_training_job(
        "Failed", failure_reason="AlgorithmError"
    )

    with pytest.raises(FailedStatusError, match="Failed"):
        TrainingJob(training_job_name=JOB_NAME).wait(poll=0, raise_on_stopped=True)


def test_wait_polls_through_stopping_before_it_raises(training_client):
    training_client.describe_training_job.side_effect = [
        _describe_training_job("Stopping"),
        _describe_training_job("Stopped"),
    ]

    with pytest.raises(FailedStatusError):
        TrainingJob(training_job_name=JOB_NAME).wait(poll=0, raise_on_stopped=True)

    assert training_client.describe_training_job.call_count == 2


def test_processing_job_wait_raises_on_stopped_when_asked(training_client):
    training_client.describe_processing_job.return_value = {
        "ProcessingJobName": JOB_NAME,
        "ProcessingJobStatus": "Stopped",
    }
    # The generated ProcessingJob.wait reads the instance count before the first refresh.
    job = ProcessingJob(
        processing_job_name=JOB_NAME,
        processing_resources=ProcessingResources(
            cluster_config=ProcessingClusterConfig(
                instance_count=1, instance_type="ml.m5.large", volume_size_in_gb=30
            )
        ),
    )

    with pytest.raises(FailedStatusError, match="ProcessingJob"):
        job.wait(poll=0, raise_on_stopped=True)
