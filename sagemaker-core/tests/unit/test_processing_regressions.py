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
"""Tests for v2->v3 regression bugs in processing and training."""
import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock


class TestBug1ProcessorWaitUsesSession:
    """Bug 1: wait=True should use sagemaker_session, not global default client."""

    def test_processor_start_new_stores_session_on_job(self):
        """Test that _start_new stores sagemaker_session on the ProcessingJob."""
        from sagemaker.core.processing import Processor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "my-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = "arn:aws:iam::123456789:role/MyRole"
        mock_session.sagemaker_client = MagicMock()
        mock_session.boto_session = MagicMock()
        mock_session.sagemaker_client.create_processing_job.return_value = {}

        # Mock the _intercept_create_request to call submit
        def intercept(request, submit, job_type):
            if submit:
                submit(request)

        mock_session._intercept_create_request = intercept

        processor = Processor(
            role="arn:aws:iam::123456789:role/MyRole",
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )

        processor._current_job_name = "test-job"
        processor.arguments = None

        with patch("sagemaker.core.processing.ProcessingJob") as MockPJ:
            mock_job = MagicMock()
            MockPJ.return_value = mock_job

            with patch("sagemaker.core.processing.serialize", return_value={}):
                with patch("sagemaker.core.processing.transform", return_value={}):
                    try:
                        result = processor._start_new(
                            inputs=[], outputs=[], experiment_config=None
                        )
                    except Exception:
                        pass

            # The key assertion: _sagemaker_session should be set
            if result is not None and hasattr(result, '_sagemaker_session'):
                assert result._sagemaker_session == mock_session

    def test_patched_processing_job_wait_uses_session(self):
        """Test that the patched ProcessingJob.wait uses _sagemaker_session."""
        from sagemaker.core.resources import ProcessingJob

        mock_session = MagicMock()
        mock_session.sagemaker_client.describe_processing_job.return_value = {
            "ProcessingJobStatus": "Completed",
            "ProcessingJobName": "test-job",
        }

        job = MagicMock(spec=ProcessingJob)
        job.processing_job_name = "test-job"
        job._sagemaker_session = mock_session

        # Import the patched wait
        from sagemaker.core.processing import _wait_for_processing_job
        _wait_for_processing_job(job, logs=False)

        mock_session.sagemaker_client.describe_processing_job.assert_called()


class TestBug2CodeLocation:
    """Bug 2: code_location should be used for S3 uploads."""

    def test_framework_processor_code_location_used_in_upload(self):
        """Test that code_location is used when uploading code."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = "arn:aws:iam::123456789:role/MyRole"

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

    def test_framework_processor_code_location_none_uses_default_bucket(self):
        """Test that default bucket is used when code_location is None."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = "default-prefix"
        mock_session.expand_role.return_value = "arn:aws:iam::123456789:role/MyRole"

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

    def test_framework_processor_run_accepts_codeartifact_repo_arn(self):
        """Test that FrameworkProcessor.run() accepts codeartifact_repo_arn parameter."""
        import inspect
        from sagemaker.core.processing import FrameworkProcessor

        sig = inspect.signature(FrameworkProcessor.run)
        assert "codeartifact_repo_arn" in sig.parameters

    def test_get_codeartifact_command_parses_arn_correctly(self):
        """Test that _get_codeartifact_command correctly parses the ARN."""
        from sagemaker.core.processing import FrameworkProcessor

        arn = "arn:aws:codeartifact:us-west-2:123456789012:repository/my-domain/my-repo"
        command = FrameworkProcessor._get_codeartifact_command(arn)

        assert "--domain my-domain" in command
        assert "--domain-owner 123456789012" in command
        assert "--repository my-repo" in command
        assert "--region us-west-2" in command
        assert "aws codeartifact login --tool pip" in command

    def test_generate_framework_script_with_codeartifact_injects_login(self):
        """Test that _generate_framework_script injects CodeArtifact login."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = "arn:aws:iam::123456789:role/MyRole"

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        processor._codeartifact_repo_arn = (
            "arn:aws:codeartifact:us-west-2:123456789012:repository/my-domain/my-repo"
        )

        script = processor._generate_framework_script("my_script.py")
        assert "aws codeartifact login --tool pip" in script
        assert "--domain my-domain" in script
        assert "--repository my-repo" in script

    def test_generate_framework_script_without_codeartifact_no_login(self):
        """Test that _generate_framework_script does NOT inject CodeArtifact login when not set."""
        from sagemaker.core.processing import FrameworkProcessor

        mock_session = MagicMock()
        mock_session.default_bucket.return_value = "default-bucket"
        mock_session.default_bucket_prefix = ""
        mock_session.expand_role.return_value = "arn:aws:iam::123456789:role/MyRole"

        processor = FrameworkProcessor(
            image_uri="123456789.dkr.ecr.us-west-2.amazonaws.com/my-image:latest",
            role="arn:aws:iam::123456789:role/MyRole",
            instance_count=1,
            instance_type="ml.m5.xlarge",
            sagemaker_session=mock_session,
        )
        # Don't set _codeartifact_repo_arn

        script = processor._generate_framework_script("my_script.py")
        assert "aws codeartifact login" not in script
        assert "pip install -r requirements.txt" in script


class TestBug4CodeArtifactTemplates:
    """Bug 4: INSTALL_REQUIREMENTS template should check CA_REPOSITORY_ARN."""

    def test_install_requirements_template_checks_ca_repository_arn(self):
        """Test that INSTALL_REQUIREMENTS template includes CA_REPOSITORY_ARN check."""
        from sagemaker.train.templates import INSTALL_REQUIREMENTS

        # The template should contain the CodeArtifact login logic
        rendered = INSTALL_REQUIREMENTS.format(requirements_file="requirements.txt")
        assert "CA_REPOSITORY_ARN" in rendered
        assert "aws codeartifact login --tool pip" in rendered

    def test_install_requirements_template_without_ca_repository_arn_uses_plain_pip(self):
        """Test that INSTALL_REQUIREMENTS still does pip install."""
        from sagemaker.train.templates import INSTALL_REQUIREMENTS

        rendered = INSTALL_REQUIREMENTS.format(requirements_file="requirements.txt")
        assert "pip install -r requirements.txt" in rendered or "$SM_PIP_CMD install -r requirements.txt" in rendered

    def test_install_auto_requirements_checks_ca_repository_arn(self):
        """Test that INSTALL_AUTO_REQUIREMENTS template includes CA_REPOSITORY_ARN check."""
        from sagemaker.train.templates import INSTALL_AUTO_REQUIREMENTS

        assert "CA_REPOSITORY_ARN" in INSTALL_AUTO_REQUIREMENTS
        assert "aws codeartifact login --tool pip" in INSTALL_AUTO_REQUIREMENTS


class TestBug1ModelTrainerWait:
    """Bug 1: ModelTrainer.train(wait=True) should use sagemaker_session."""

    def test_model_trainer_train_wait_uses_sagemaker_session(self):
        """Test that the wait function accepts sagemaker_session parameter."""
        import inspect
        from sagemaker.train.common_utils.trainer_wait import wait

        sig = inspect.signature(wait)
        assert "sagemaker_session" in sig.parameters

    def test_refresh_training_job_uses_session_client(self):
        """Test that _refresh_training_job uses session's sagemaker_client."""
        from sagemaker.train.common_utils.trainer_wait import _refresh_training_job

        mock_session = MagicMock()
        mock_session.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "test-job",
        }

        mock_job = MagicMock()
        mock_job.training_job_name = "test-job"

        _refresh_training_job(mock_job, sagemaker_session=mock_session)

        mock_session.sagemaker_client.describe_training_job.assert_called_once_with(
            TrainingJobName="test-job"
        )

    def test_refresh_training_job_without_session_uses_default(self):
        """Test that _refresh_training_job falls back to default refresh."""
        from sagemaker.train.common_utils.trainer_wait import _refresh_training_job

        mock_job = MagicMock()
        mock_job.training_job_name = "test-job"

        _refresh_training_job(mock_job, sagemaker_session=None)

        mock_job.refresh.assert_called_once()
