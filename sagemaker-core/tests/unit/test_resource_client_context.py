"""Generated resources keep using the session/region they were loaded with.

Regression coverage for #6069: ``TrainingJob.create(session=...)`` used the
caller's session, but ``refresh()``/``wait()``/``stop()`` called
``Base.get_sagemaker_client()`` with no session and fell back to the process
default client, so a job created in one account was polled in another.
"""

from unittest.mock import MagicMock, patch

import boto3
import pytest

from sagemaker.core.resources import Base, TrainingJob
from sagemaker.core.utils.utils import SageMakerClient

JOB = "my-job"


def _session(key):
    return boto3.Session(
        aws_access_key_id=key, aws_secret_access_key="secret", region_name="us-west-2"
    )


@pytest.fixture(autouse=True)
def _reset_client_cache():
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()


@pytest.fixture
def recorded_clients():
    """Patch Base.get_sagemaker_client; record (session, region) per call."""
    calls = []
    client = MagicMock()
    client.describe_training_job.return_value = {
        "TrainingJobName": JOB,
        "TrainingJobStatus": "Completed",
        "SecondaryStatus": "Completed",
    }
    client.list_training_jobs.return_value = {
        "TrainingJobSummaries": [{"TrainingJobName": JOB, "TrainingJobStatus": "Completed"}]
    }

    def fake(session=None, region_name=None, service_name="sagemaker"):
        calls.append((session, region_name))
        return client

    with patch.object(Base, "get_sagemaker_client", side_effect=fake):
        yield calls, client


def test_get_binds_session_for_refresh_wait_and_stop(recorded_clients):
    calls, _ = recorded_clients
    session_b = _session("AKIDB")

    job = TrainingJob.get(JOB, session=session_b, region="eu-west-1")
    job.refresh()
    job.wait()
    job.stop()

    assert calls, "no client was requested"
    assert all(call == (session_b, "eu-west-1") for call in calls), calls


def test_explicit_session_on_object_method_overrides_bound_one(recorded_clients):
    calls, _ = recorded_clients
    session_b, session_c = _session("AKIDB"), _session("AKIDC")

    job = TrainingJob.get(JOB, session=session_b)
    calls.clear()
    job._get_client(session=session_c)

    assert calls == [(session_c, None)]


def test_get_all_binds_session_on_each_resource(recorded_clients):
    calls, _ = recorded_clients
    session_b = _session("AKIDB")

    jobs = list(TrainingJob.get_all(session=session_b, region="eu-west-1"))

    assert [j.training_job_name for j in jobs] == [JOB]
    # list call plus the iterator's per-item refresh, all on session_b
    assert len(calls) >= 2
    assert all(call == (session_b, "eu-west-1") for call in calls), calls


def test_unbound_resource_still_uses_process_default(recorded_clients):
    calls, _ = recorded_clients

    TrainingJob(training_job_name=JOB).refresh()

    assert calls == [(None, None)]


def test_bound_refresh_signs_with_bound_session_after_default_exists():
    """End to end through SageMakerClient: the default-chain client built first
    must not be used by refresh() on a resource loaded with another session."""
    default = SageMakerClient(session=_session("AKIDDEFAULT"))
    session_b = _session("AKIDB")

    job = TrainingJob(training_job_name=JOB)._set_client_context(session=session_b)
    client = job._get_client()

    assert client is not default.sagemaker_client
    assert client._request_signer._credentials.access_key == "AKIDB"


def test_client_context_is_not_a_model_field():
    job = TrainingJob(training_job_name=JOB)._set_client_context(session=_session("AKIDB"))

    assert "_session" not in vars(job)
    assert "session" not in TrainingJob.model_fields
