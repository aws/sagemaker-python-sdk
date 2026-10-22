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
"""Transformer keeps its configured session after submission (issue #6503).

``transform()`` submitted through the configured ``sagemaker_session`` and then
built ``TransformJob(**transformed)`` with no session, so ``wait()``, ``stop()``
and the log stream fell back to the process default boto session: the job was
created in the target account and the wait failed against the default one.
Local mode was worse -- the wait could poll AWS for a job that only ever
existed in the local session.
"""

from unittest.mock import MagicMock, Mock, patch

import boto3
import pytest

from sagemaker.core.resources import Base, TransformJob
from sagemaker.core.shapes import TransformOutput, TransformResources
from sagemaker.core.transformer import Transformer
from sagemaker.core.utils.utils import SageMakerClient

REGION = "us-east-2"
DEFAULT_REGION = "us-east-1"
MODEL_NAME = "transformer-session-propagation-model"
JOB_NAME = "transformer-session-propagation-job"
INPUT_URI = "s3://target-bucket/input/"
OUTPUT_URI = "s3://target-bucket/output/"

# The session the caller configures (target account) and the process default one.
TARGET_BOTO_SESSION = boto3.Session(
    aws_access_key_id="AKIDTARGET", aws_secret_access_key="secret", region_name=REGION
)
DEFAULT_BOTO_SESSION = boto3.Session(
    aws_access_key_id="AKIDDEFAULT", aws_secret_access_key="secret", region_name=DEFAULT_REGION
)


@pytest.fixture(autouse=True)
def _reset_client_cache():
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()


def _configured_session(local_mode=False):
    session = MagicMock()
    session.boto_session = TARGET_BOTO_SESSION
    session.boto_region_name = REGION
    session.local_mode = local_mode
    session.default_bucket.return_value = "target-bucket"
    session.default_bucket_prefix = None
    session.sagemaker_config = {}
    # Run the submit callback inline; the sagemaker_client is a mock, so no AWS call.
    session._intercept_create_request.side_effect = lambda request, submit, name: (
        submit(request) if submit else None
    )
    return session


def _described_job():
    """A TransformJob resource shaped like a DescribeTransformJob response."""
    return TransformJob(
        transform_job_name=JOB_NAME,
        model_name=MODEL_NAME,
        transform_resources=TransformResources(instance_type="ml.m5.large", instance_count=1),
        transform_output=TransformOutput(s3_output_path=OUTPUT_URI),
    )


def _transformer(session):
    return Transformer(
        model_name=MODEL_NAME,
        instance_count=1,
        instance_type="ml.m5.large",
        output_path=OUTPUT_URI,
        sagemaker_session=session,
    )


@pytest.fixture
def recorded_clients():
    """Record the (session, region) every SageMaker client request is built with."""
    calls = []
    client = MagicMock()
    client.describe_transform_job.return_value = {
        "TransformJobName": JOB_NAME,
        "TransformJobStatus": "Completed",
    }

    def fake(session=None, region_name=None, service_name="sagemaker"):
        calls.append((session, region_name))
        return client

    with patch.object(Base, "get_sagemaker_client", side_effect=fake):
        yield calls, client


class TestRemoteTransformBindsConfiguredSession:
    def test_transform_binds_session_and_region_to_the_resource(self):
        transformer = _transformer(_configured_session())

        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)

        job = transformer.latest_transform_job
        assert job._session is TARGET_BOTO_SESSION
        assert job._region == REGION

    def test_waiting_inside_transform_uses_the_configured_session(self, recorded_clients):
        calls, _ = recorded_clients
        transformer = _transformer(_configured_session())

        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=True, logs=False)

        assert calls, "the wait made no client request"
        assert all(call == (TARGET_BOTO_SESSION, REGION) for call in calls), calls

    def test_wait_after_transform_uses_the_configured_session(self, recorded_clients):
        calls, _ = recorded_clients
        transformer = _transformer(_configured_session())
        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)
        calls.clear()

        transformer.wait(logs=False)

        assert all(call == (TARGET_BOTO_SESSION, REGION) for call in calls), calls

    def test_stop_transform_job_uses_the_configured_session(self, recorded_clients):
        calls, _ = recorded_clients
        transformer = _transformer(_configured_session())
        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)
        calls.clear()

        transformer.stop_transform_job(wait=False)

        assert calls, "stop made no client request"
        assert all(call == (TARGET_BOTO_SESSION, REGION) for call in calls), calls

    def test_bound_client_is_signed_with_the_configured_credentials(self):
        """End to end through SageMakerClient, with a default client built first."""
        default = SageMakerClient(session=DEFAULT_BOTO_SESSION)
        transformer = _transformer(_configured_session())

        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)
        client = transformer.latest_transform_job._get_client()

        assert client is not default.sagemaker_client
        assert client._request_signer._credentials.access_key == "AKIDTARGET"


class TestLocalTransformNeverCallsAws:
    def test_local_wait_does_not_request_an_aws_client(self, recorded_clients):
        calls, _ = recorded_clients
        transformer = _transformer(_configured_session(local_mode=True))

        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=True, logs=False)

        assert calls == [], f"a local transform must not poll AWS, got {calls}"

    def test_local_stop_does_not_request_an_aws_client(self, recorded_clients):
        calls, _ = recorded_clients
        transformer = _transformer(_configured_session(local_mode=True))
        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)
        calls.clear()

        transformer.stop_transform_job(wait=False)

        assert calls == [], f"a local transform has nothing to stop in AWS, got {calls}"

    def test_remote_transform_is_not_marked_local(self):
        transformer = _transformer(_configured_session())

        transformer.transform(data=INPUT_URI, job_name=JOB_NAME, wait=False)

        assert transformer._latest_transform_job_is_local is False


class TestAttachUsesTheConfiguredSession:
    def test_attach_passes_the_boto_session_and_reuses_the_resource(self):
        session = _configured_session()
        described = _described_job()

        with patch.object(TransformJob, "get", return_value=described) as mock_get:
            transformer = Transformer.attach(JOB_NAME, sagemaker_session=session)

        mock_get.assert_called_once_with(
            transform_job_name=JOB_NAME,
            session=TARGET_BOTO_SESSION,
            region=REGION,
        )
        assert transformer.latest_transform_job is described

    def test_attach_raises_when_the_job_is_missing(self):
        with patch.object(TransformJob, "get", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                Transformer.attach(JOB_NAME, sagemaker_session=_configured_session())

    def test_attached_transformer_is_not_marked_local(self):
        session = _configured_session()
        described = _described_job()

        with patch.object(TransformJob, "get", return_value=described):
            transformer = Transformer.attach(JOB_NAME, sagemaker_session=session)

        assert transformer._latest_transform_job_is_local is False


class TestEnsureLastTransformJob:
    def test_wait_without_a_job_raises(self):
        transformer = _transformer(_configured_session())

        with pytest.raises(ValueError, match="No transform job available"):
            transformer.wait()

    def test_stop_without_a_job_raises(self):
        transformer = _transformer(Mock(local_mode=False, sagemaker_config={}))

        with pytest.raises(ValueError, match="No transform job available"):
            transformer.stop_transform_job()
