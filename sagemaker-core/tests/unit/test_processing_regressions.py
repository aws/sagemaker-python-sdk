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
"""Tests for v2->v3 regression bugs in sagemaker.core.processing."""
import os
import pytest
from unittest.mock import MagicMock, patch


class TestBug1ProcessorWaitUsesSession:
    """Bug 1: wait=True should use sagemaker_session, not global default client."""

    def test_processor_wait_for_job_uses_session(self):
        """Test that _wait_for_job uses the Processor's sagemaker_session."""
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


class TestBug2CodeLocation:
    """Bug 2: code_location should be used for S3 uploads."""

    def test_framework_processor_code_location_used_in_upload(self):
        """Test that code_location is used when uploading code."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            code_location="s3://my-custom-bucket/my-prefix",
            sagemaker_session=mock_session,
        )

        bucket, prefix = processor._get_code_upload_bucket_and_prefix()
        assert bucket == "my-custom-bucket"
        assert prefix == "my-prefix"

    def test_framework_processor_code_location_none_uses_default(self):
        """Test that default bucket is used when code_location is None."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = "default-prefix"
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        bucket, prefix = processor._get_code_upload_bucket_and_prefix()
        assert bucket == "default-bucket"
        assert prefix == "default-prefix"


class TestBug3CodeArtifactFrameworkProcessor:
    """Bug 3: CodeArtifact support in FrameworkProcessor."""

    def test_run_accepts_codeartifact_repo_arn(self):
        """Test that FrameworkProcessor.run() accepts codeartifact_repo_arn."""
        import inspect
        from sagemaker.core.processing import FrameworkProcessor

        sig = inspect.signature(FrameworkProcessor.run)
        assert "codeartifact_repo_arn" in sig.parameters

    def test_get_codeartifact_command_parses_arn_correctly(self):
        """Test that _get_codeartifact_command correctly parses the ARN."""
        from sagemaker.core.processing import FrameworkProcessor

        arn = (
            "arn:aws:codeartifact:us-west-2:123456789012"
            ":repository/my-domain/my-repo"
        )
        command = FrameworkProcessor._get_codeartifact_command(arn)

        assert "--domain my-domain" in command
        assert "--domain-owner 123456789012" in command
        assert "--repository my-repo" in command
        assert "--region us-west-2" in command
        assert "aws codeartifact login --tool pip" in command

    def test_get_codeartifact_command_rejects_invalid_arn(self):
        """Test that _get_codeartifact_command raises ValueError for bad ARN."""
        from sagemaker.core.processing import FrameworkProcessor

        with pytest.raises(ValueError, match="Invalid CodeArtifact repository ARN"):
            FrameworkProcessor._get_codeartifact_command("not-a-valid-arn")

    def test_generate_framework_script_with_codeartifact(self):
        """Test that _generate_framework_script injects CodeArtifact login."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._codeartifact_repo_arn = (
            "arn:aws:codeartifact:us-west-2:123456789012"
            ":repository/my-domain/my-repo"
        )

        script = processor._generate_framework_script("my_script.py")
        assert "aws codeartifact login --tool pip" in script
        assert "--domain my-domain" in script
        assert "--repository my-repo" in script

    def test_generate_framework_script_without_codeartifact(self):
        """Test script does NOT inject CodeArtifact login when not set."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = (
            "arn:aws:iam::123456789:role/MyRole"
        )

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        script = processor._generate_framework_script("my_script.py")
        assert "aws codeartifact login" not in script
        assert "pip install -r requirements.txt" in script
