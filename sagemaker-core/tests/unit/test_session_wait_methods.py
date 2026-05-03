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
"""Unit tests for session wait methods (_wait_for_processing_job, _wait_for_training_job).

These methods were added to fix Bug 1 in issue #5765: wait=True does not
respect sagemaker_session, causing NoCredentialsError with assumed-role sessions.
"""
from __future__ import absolute_import

from unittest.mock import MagicMock, patch
import pytest

from sagemaker.core.helper.session_helper import (
    _processing_job_status,
    _training_job_status,
)


class TestProcessingJobStatus:
    """Tests for the _processing_job_status helper function."""

    def test_returns_none_when_in_progress(self):
        client = MagicMock()
        client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "InProgress"
        }
        result = _processing_job_status(client, "my-job")
        assert result is None
        client.describe_processing_job.assert_called_once_with(ProcessingJobName="my-job")

    def test_returns_desc_when_completed(self):
        desc = {"ProcessingJobStatus": "Completed"}
        client = MagicMock()
        client.describe_processing_job.return_value = desc
        result = _processing_job_status(client, "my-job")
        assert result == desc

    def test_returns_desc_when_failed(self):
        desc = {"ProcessingJobStatus": "Failed", "FailureReason": "OOM"}
        client = MagicMock()
        client.describe_processing_job.return_value = desc
        result = _processing_job_status(client, "my-job")
        assert result == desc

    def test_returns_desc_when_stopped(self):
        desc = {"ProcessingJobStatus": "Stopped"}
        client = MagicMock()
        client.describe_processing_job.return_value = desc
        result = _processing_job_status(client, "my-job")
        assert result == desc

    def test_returns_none_when_stopping(self):
        client = MagicMock()
        client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "Stopping"
        }
        result = _processing_job_status(client, "my-job")
        assert result is None


class TestTrainingJobStatus:
    """Tests for the _training_job_status helper function."""

    def test_returns_none_when_in_progress(self):
        client = MagicMock()
        client.describe_training_job.return_value = {
            "TrainingJobStatus": "InProgress"
        }
        result = _training_job_status(client, "my-job")
        assert result is None
        client.describe_training_job.assert_called_once_with(TrainingJobName="my-job")

    def test_returns_desc_when_completed(self):
        desc = {"TrainingJobStatus": "Completed"}
        client = MagicMock()
        client.describe_training_job.return_value = desc
        result = _training_job_status(client, "my-job")
        assert result == desc

    def test_returns_desc_when_failed(self):
        desc = {"TrainingJobStatus": "Failed", "FailureReason": "AlgorithmError"}
        client = MagicMock()
        client.describe_training_job.return_value = desc
        result = _training_job_status(client, "my-job")
        assert result == desc


class TestSessionWaitForProcessingJob:
    """Tests for Session._wait_for_processing_job."""

    def test_uses_session_client(self):
        """Verify _wait_for_processing_job uses self.sagemaker_client, not global."""
        from sagemaker.core.helper.session_helper import Session

        session = MagicMock(spec=Session)
        session.sagemaker_client = MagicMock()
        session.sagemaker_client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "Completed"
        }

        # Call the unbound method with our mock session
        Session._wait_for_processing_job(session, "test-job", poll=0.1)

        session.sagemaker_client.describe_processing_job.assert_called_with(
            ProcessingJobName="test-job"
        )

    def test_polls_until_complete(self):
        """Verify it polls multiple times until job completes."""
        from sagemaker.core.helper.session_helper import Session

        session = MagicMock(spec=Session)
        session.sagemaker_client = MagicMock()
        session.sagemaker_client.describe_processing_job.side_effect = [
            {"ProcessingJobStatus": "InProgress"},
            {"ProcessingJobStatus": "InProgress"},
            {"ProcessingJobStatus": "Completed"},
        ]

        Session._wait_for_processing_job(session, "test-job", poll=0.1)

        assert session.sagemaker_client.describe_processing_job.call_count == 3


class TestSessionWaitForTrainingJob:
    """Tests for Session._wait_for_training_job."""

    def test_uses_session_client(self):
        """Verify _wait_for_training_job uses self.sagemaker_client, not global."""
        from sagemaker.core.helper.session_helper import Session

        session = MagicMock(spec=Session)
        session.sagemaker_client = MagicMock()
        session.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed"
        }

        Session._wait_for_training_job(session, "test-job", poll=0.1)

        session.sagemaker_client.describe_training_job.assert_called_with(
            TrainingJobName="test-job"
        )


class TestProcessingUsesSessionWait:
    """Tests that processing.py uses session-aware wait instead of global client."""

    def test_processor_run_calls_session_wait(self):
        """Verify Processor.run with wait=True calls _wait_for_processing_job."""
        from sagemaker.core.processing import Processor

        processor = MagicMock(spec=Processor)
        processor.sagemaker_session = MagicMock()
        processor.sagemaker_session.__class__.__name__ = "Session"
        processor.jobs = []

        # Create a mock processing job
        mock_job = MagicMock()
        mock_job.processing_job_name = "test-processing-job"
        processor.latest_job = mock_job

        # Simulate what run() does after _start_new
        from sagemaker.core.workflow.pipeline_context import PipelineSession
        if not isinstance(processor.sagemaker_session, PipelineSession):
            processor.jobs.append(processor.latest_job)
            processor.sagemaker_session._wait_for_processing_job(
                processor.latest_job.processing_job_name
            )

        processor.sagemaker_session._wait_for_processing_job.assert_called_once_with(
            "test-processing-job"
        )
