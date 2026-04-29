"""Tests for session propagation fix (GitHub issue #5765).

Verifies that when a user passes a custom session to create() or get(),
the session is stored on the resource instance and used by all instance
methods (refresh, update, delete, stop, wait).

Before this fix, instance methods called Base.get_sagemaker_client() with
no session argument, falling back to ambient/default credentials. This
caused NoCredentialsError when the user's session used different credentials
(e.g., assumed-role via STS).
"""

import unittest
from unittest.mock import patch, MagicMock, call

from boto3.session import Session as BotoSession

from sagemaker.core.resources import (
    Base,
    TrainingJob,
    ProcessingJob,
    Endpoint,
    TransformJob,
    Model,
)


class TestSessionStoredOnGet(unittest.TestCase):
    """Test that get() stores the session on the resource instance."""

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_get_stores_session_on_training_job(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed",
        }
        mock_transform.return_value = {
            "training_job_name": "test-job",
            "training_job_status": "Completed",
        }

        job = TrainingJob.get(training_job_name="test-job", session=mock_session)

        assert hasattr(job, "_session")
        assert job._session is mock_session

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_get_stores_session_on_processing_job(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_processing_job.return_value = {
            "ProcessingJobName": "test-proc",
            "ProcessingJobStatus": "Completed",
        }
        mock_transform.return_value = {
            "processing_job_name": "test-proc",
            "processing_job_status": "Completed",
        }

        job = ProcessingJob.get(processing_job_name="test-proc", session=mock_session)

        assert hasattr(job, "_session")
        assert job._session is mock_session

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_get_stores_session_on_endpoint(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_endpoint.return_value = {
            "EndpointName": "test-ep",
            "EndpointStatus": "InService",
        }
        mock_transform.return_value = {
            "endpoint_name": "test-ep",
            "endpoint_status": "InService",
        }

        ep = Endpoint.get(endpoint_name="test-ep", session=mock_session)

        assert hasattr(ep, "_session")
        assert ep._session is mock_session

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_get_stores_none_when_no_session_passed(self, mock_get_client, mock_transform):
        """Backward compatibility: _session is None when no session is passed."""
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed",
        }
        mock_transform.return_value = {
            "training_job_name": "test-job",
            "training_job_status": "Completed",
        }

        job = TrainingJob.get(training_job_name="test-job")

        assert hasattr(job, "_session")
        assert job._session is None


class TestSessionUsedByRefresh(unittest.TestCase):
    """Test that refresh() uses the stored session."""

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_refresh_uses_stored_session(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed",
        }

        job = TrainingJob(training_job_name="test-job")
        job._session = mock_session

        job.refresh()

        # Verify get_sagemaker_client was called with the stored session
        mock_get_client.assert_called_with(session=mock_session)

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_refresh_uses_none_session_when_not_set(self, mock_get_client, mock_transform):
        """Backward compatibility: refresh works when _session is not set."""
        client = MagicMock()
        mock_get_client.return_value = client
        client.describe_processing_job.return_value = {
            "ProcessingJobName": "test-proc",
            "ProcessingJobStatus": "Completed",
        }

        job = ProcessingJob(processing_job_name="test-proc")
        # Don't set _session — getattr should return None

        job.refresh()

        mock_get_client.assert_called_with(session=None)


class TestSessionUsedByDelete(unittest.TestCase):
    """Test that delete() uses the stored session."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_delete_uses_stored_session(self, mock_get_client):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.delete_model.return_value = {}

        model = Model(model_name="test-model")
        model._session = mock_session

        model.delete()

        mock_get_client.assert_called_with(session=mock_session)


class TestSessionUsedByStop(unittest.TestCase):
    """Test that stop() uses the stored session instead of SageMakerClient()."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_stop_uses_stored_session(self, mock_get_client):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.stop_training_job.return_value = {}

        job = TrainingJob(training_job_name="test-job")
        job._session = mock_session

        job.stop()

        mock_get_client.assert_called_with(session=mock_session)

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_stop_uses_none_session_when_not_set(self, mock_get_client):
        """Backward compatibility: stop works when _session is not set."""
        client = MagicMock()
        mock_get_client.return_value = client
        client.stop_training_job.return_value = {}

        job = TrainingJob(training_job_name="test-job")

        job.stop()

        mock_get_client.assert_called_with(session=None)


class TestSessionUsedByUpdate(unittest.TestCase):
    """Test that update() uses the stored session."""

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_update_uses_stored_session(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client
        client.update_training_job.return_value = {}
        client.describe_training_job.return_value = {
            "TrainingJobName": "test-job",
            "TrainingJobStatus": "Completed",
        }

        job = TrainingJob(training_job_name="test-job")
        job._session = mock_session

        job.update()

        # update() calls get_sagemaker_client, then refresh() also calls it
        # Both should use the stored session
        for c in mock_get_client.call_args_list:
            assert c == call(session=mock_session)


class TestSessionFlowsThroughCreate(unittest.TestCase):
    """Test that create() -> get() stores session on the returned instance."""

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_create_stores_session_via_get(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client

        # create() calls the API then calls get()
        client.create_model.return_value = {"ModelArn": "arn:aws:sagemaker:us-west-2:123:model/m"}
        client.describe_model.return_value = {
            "ModelName": "test-model",
            "ModelArn": "arn:aws:sagemaker:us-west-2:123:model/m",
        }
        mock_transform.return_value = {
            "model_name": "test-model",
            "model_arn": "arn:aws:sagemaker:us-west-2:123:model/m",
        }

        model = Model.create(
            model_name="test-model",
            session=mock_session,
        )

        # The session should be stored on the instance returned by get()
        assert hasattr(model, "_session")
        assert model._session is mock_session


class TestSessionPropagationEndToEnd(unittest.TestCase):
    """End-to-end test: create with session, then refresh uses that session."""

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_create_then_refresh_uses_same_session(self, mock_get_client, mock_transform):
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client

        client.create_model.return_value = {"ModelArn": "arn:aws:sagemaker:us-west-2:123:model/m"}
        client.describe_model.return_value = {
            "ModelName": "test-model",
            "ModelArn": "arn:aws:sagemaker:us-west-2:123:model/m",
        }
        mock_transform.return_value = {
            "model_name": "test-model",
            "model_arn": "arn:aws:sagemaker:us-west-2:123:model/m",
        }

        # Step 1: Create with session
        model = Model.create(model_name="test-model", session=mock_session)

        # Reset mock to track refresh calls
        mock_get_client.reset_mock()

        # Step 2: Refresh should use the same session
        model.refresh()

        mock_get_client.assert_called_with(session=mock_session)


class TestAllResourceTypesHaveSession(unittest.TestCase):
    """Verify that all resource types with get() store _session."""

    @patch("sagemaker.core.resources.validate_call", lambda **kwargs: lambda func: func)
    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_multiple_resource_types_store_session(self, mock_get_client, mock_transform):
        """Spot-check several resource types to verify _session is stored."""
        mock_session = MagicMock(spec=BotoSession)
        client = MagicMock()
        mock_get_client.return_value = client

        test_cases = [
            (TrainingJob, "describe_training_job", "TrainingJobName", "training_job_name"),
            (ProcessingJob, "describe_processing_job", "ProcessingJobName", "processing_job_name"),
            (TransformJob, "describe_transform_job", "TransformJobName", "transform_job_name"),
        ]

        for resource_cls, describe_method, api_key, attr_key in test_cases:
            with self.subTest(resource=resource_cls.__name__):
                getattr(client, describe_method).return_value = {
                    api_key: "test-name",
                }
                mock_transform.return_value = {
                    attr_key: "test-name",
                }

                instance = resource_cls.get(**{attr_key: "test-name"}, session=mock_session)

                assert hasattr(instance, "_session"), (
                    f"{resource_cls.__name__} missing _session attribute"
                )
                assert instance._session is mock_session, (
                    f"{resource_cls.__name__}._session is not the passed session"
                )


if __name__ == "__main__":
    unittest.main()
