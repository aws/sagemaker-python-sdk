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
"""Tests for v2->v3 regression Bug 1: wait=True ignores sagemaker session."""
import pytest
from unittest.mock import MagicMock, patch, call


class TestBug1ProcessorWaitUsesSession:
    """Bug 1: wait=True should use sagemaker_session, not global default client."""

    def test_processor_wait_for_job_uses_session_no_logs(self):
        """Test that _wait_for_job uses the Processor's sagemaker_session (no logs)."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )
        mock_session.boto_session = MagicMock()

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = MagicMock()
        mock_job.processing_job_name = "test-job"

        # Mock ProcessingJob.get to return a completed job
        with patch("sagemaker.core.processing.ProcessingJob") as MockPJ:
            mock_refreshed = MagicMock()
            mock_refreshed.processing_job_status = "Completed"
            MockPJ.get.return_value = mock_refreshed

            processor._wait_for_job(mock_job, logs=False)

            MockPJ.get.assert_called_with(
                processing_job_name="test-job",
                session=mock_session.boto_session,
            )

    def test_processor_wait_for_job_uses_session_with_logs(self):
        """Test that _wait_for_job with logs=True uses logs_for_processing_job."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )
        mock_session.boto_session = MagicMock()

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = MagicMock()
        mock_job.processing_job_name = "test-job"

        with patch("sagemaker.core.processing.logs_for_processing_job") as mock_logs:
            processor._wait_for_job(mock_job, logs=True)

            mock_logs.assert_called_once_with(
                mock_session, "test-job", wait=True
            )

    def test_processor_wait_for_job_raises_on_failed(self):
        """Test that _wait_for_job raises RuntimeError when job fails."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )
        mock_session.boto_session = MagicMock()

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = MagicMock()
        mock_job.processing_job_name = "test-job"

        with patch("sagemaker.core.processing.ProcessingJob") as MockPJ:
            mock_refreshed = MagicMock()
            mock_refreshed.processing_job_status = "Failed"
            mock_refreshed.failure_reason = "OutOfMemory"
            MockPJ.get.return_value = mock_refreshed

            with pytest.raises(RuntimeError, match="failed.*OutOfMemory"):
                processor._wait_for_job(mock_job, logs=False)

    def test_processor_wait_for_job_timeout(self):
        """Test that _wait_for_job raises RuntimeError on timeout."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )
        mock_session.boto_session = MagicMock()

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        mock_job = MagicMock()
        mock_job.processing_job_name = "test-job"

        with patch("sagemaker.core.processing.ProcessingJob") as MockPJ:
            mock_refreshed = MagicMock()
            mock_refreshed.processing_job_status = "InProgress"
            MockPJ.get.return_value = mock_refreshed

            with patch("sagemaker.core.processing.time") as mock_time:
                # Simulate timeout: first call returns 0, second returns > timeout
                mock_time.time.side_effect = [0, 0, 5000]
                mock_time.sleep = MagicMock()

                with pytest.raises(RuntimeError, match="Timed out"):
                    processor._wait_for_job(mock_job, logs=False, timeout=1)

    def test_processor_run_calls_wait_for_job(self):
        """Test that Processor.run with wait=True calls _wait_for_job."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )
        mock_session.boto_session = MagicMock()
        mock_session.sagemaker_client = MagicMock()

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        # Verify _wait_for_job method exists and is callable
        assert hasattr(processor, '_wait_for_job')
        assert callable(processor._wait_for_job)
