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
"""``Processor._start_new`` returns a session-bound ``ProcessingJob`` with its ARN.

Before the change ``_start_new`` built ``ProcessingJob(**transformed)`` from the create
request only. The job had no ``processing_job_arn``, no ``processing_job_status``, and no
client context, so ``processor.latest_job.refresh()`` polled the process default region
(issue #5796) and four processing templates in the SageMaker agent skills called
``describe_processing_job`` on a boto3 client to read the ARN.
"""

from __future__ import absolute_import

from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.core.local.local_session import LocalSession
from sagemaker.core.processing import Processor
from sagemaker.core.resources import Base
from sagemaker.core.utils.utils import Unassigned
from sagemaker.core.workflow.pipeline_context import PipelineSession

JOB_NAME = "test-job"
JOB_ARN = f"arn:aws:sagemaker:eu-west-1:123456789012:processing-job/{JOB_NAME}"
ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"
PROCESS_ARGS = {
    "job_name": JOB_NAME,
    "inputs": [],
    "output_config": {"Outputs": []},
    "resources": {
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "VolumeSizeInGB": 30,
        }
    },
    "stopping_condition": None,
    "app_specification": {"ImageUri": "test-image:latest"},
    "environment": None,
    "network_config": None,
    "role_arn": ROLE,
    "tags": [],
}
DESCRIBE_RESPONSE = {
    "ProcessingJobName": JOB_NAME,
    "ProcessingJobArn": JOB_ARN,
    "ProcessingJobStatus": "InProgress",
    "AppSpecification": {"ImageUri": "test-image:latest"},
    "ProcessingResources": {
        "ClusterConfig": {
            "InstanceCount": 1,
            "InstanceType": "ml.m5.xlarge",
            "VolumeSizeInGB": 30,
        }
    },
    "RoleArn": ROLE,
}


def _session(spec=None):
    session = Mock(spec=spec) if spec else Mock()
    session.boto_session = Mock()
    session.boto_session.region_name = "eu-west-1"
    session.boto_region_name = "eu-west-1"
    session.sagemaker_client = Mock()
    session.sagemaker_config = {}
    session._intercept_create_request = Mock(
        side_effect=lambda request, create, name: create(request)
    )
    return session


@pytest.fixture
def mock_session():
    return _session()


@pytest.fixture
def recorded_clients():
    """Patch Base.get_sagemaker_client; record (session, region) per call."""
    calls = []
    client = MagicMock()
    client.describe_processing_job.return_value = DESCRIBE_RESPONSE

    def fake(session=None, region_name=None, service_name="sagemaker"):
        calls.append((session, region_name))
        return client

    with patch.object(Base, "get_sagemaker_client", side_effect=fake):
        yield calls, client


def _processor(session):
    processor = Processor(
        role=ROLE,
        image_uri="test-image:latest",
        instance_count=1,
        instance_type="ml.m5.xlarge",
        sagemaker_session=session,
    )
    processor._current_job_name = JOB_NAME
    return processor


def _start_new(processor):
    with patch.object(processor, "_get_process_args", return_value=dict(PROCESS_ARGS)):
        return processor._start_new([], [], None)


def test_start_new_returns_the_job_with_its_arn_and_status(mock_session, recorded_clients):
    _, client = recorded_clients

    job = _start_new(_processor(mock_session))

    mock_session.sagemaker_client.create_processing_job.assert_called_once()
    client.describe_processing_job.assert_called_once_with(ProcessingJobName=JOB_NAME)
    assert job.processing_job_name == JOB_NAME
    assert job.processing_job_arn == JOB_ARN
    assert job.processing_job_status == "InProgress"


def test_start_new_describes_through_the_session_and_its_region(mock_session, recorded_clients):
    calls, _ = recorded_clients

    job = _start_new(_processor(mock_session))

    assert calls == [(mock_session.boto_session, "eu-west-1")]
    assert job._session is mock_session.boto_session
    assert job._region == "eu-west-1"


def test_refresh_on_the_returned_job_keeps_the_session_region(mock_session, recorded_clients):
    calls, client = recorded_clients
    job = _start_new(_processor(mock_session))
    calls.clear()

    job.refresh()

    assert calls == [(mock_session.boto_session, "eu-west-1")]
    assert client.describe_processing_job.call_count == 2


def test_pipeline_session_captures_the_step_arguments_without_a_describe(recorded_clients):
    calls, client = recorded_clients
    pipeline_session = _session(spec=PipelineSession)
    pipeline_session._intercept_create_request = Mock(return_value=None)

    result = _start_new(_processor(pipeline_session))

    assert result is None
    pipeline_session._intercept_create_request.assert_called_once()
    pipeline_session.sagemaker_client.create_processing_job.assert_not_called()
    client.describe_processing_job.assert_not_called()
    assert calls == []


def test_local_session_returns_the_request_job_without_a_describe(recorded_clients):
    calls, client = recorded_clients
    local_session = _session(spec=LocalSession)

    job = _start_new(_processor(local_session))

    local_session.sagemaker_client.create_processing_job.assert_called_once()
    client.describe_processing_job.assert_not_called()
    assert calls == []
    assert job.processing_job_name == JOB_NAME
    assert isinstance(job.processing_job_arn, Unassigned)
