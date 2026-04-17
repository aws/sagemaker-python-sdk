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
"""Tests for v2->v3 regression bugs in sagemaker.train."""
import inspect
import pytest
from unittest.mock import MagicMock


class TestBug1ModelTrainerWait:
    """Bug 1: ModelTrainer.train(wait=True) should use sagemaker_session."""

    def test_wait_function_accepts_sagemaker_session(self):
        """Test that the wait function accepts sagemaker_session parameter."""
        from sagemaker.train.common_utils.trainer_wait import wait

        sig = inspect.signature(wait)
        assert "sagemaker_session" in sig.parameters

    def test_refresh_training_job_uses_session_client(self):
        """Test that _refresh_training_job uses session's sagemaker_client."""
        from sagemaker.train.common_utils.trainer_wait import (
            _refresh_training_job,
        )

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
        from sagemaker.train.common_utils.trainer_wait import (
            _refresh_training_job,
        )

        mock_job = MagicMock()
        mock_job.training_job_name = "test-job"

        _refresh_training_job(mock_job, sagemaker_session=None)

        mock_job.refresh.assert_called_once()


class TestBug4CodeArtifactTemplates:
    """Bug 4: INSTALL_REQUIREMENTS template should check CA_REPOSITORY_ARN."""

    def test_install_requirements_template_has_ca_support(self):
        """Test that INSTALL_REQUIREMENTS includes CA_REPOSITORY_ARN check."""
        from sagemaker.train.templates import INSTALL_REQUIREMENTS

        rendered = INSTALL_REQUIREMENTS.format(
            requirements_file="requirements.txt"
        )
        assert "CA_REPOSITORY_ARN" in rendered
        assert "aws codeartifact login --tool pip" in rendered

    def test_install_requirements_template_does_pip_install(self):
        """Test that INSTALL_REQUIREMENTS still does pip install."""
        from sagemaker.train.templates import INSTALL_REQUIREMENTS

        rendered = INSTALL_REQUIREMENTS.format(
            requirements_file="requirements.txt"
        )
        has_pip = (
            "pip install -r requirements.txt" in rendered
            or "$SM_PIP_CMD install -r requirements.txt" in rendered
        )
        assert has_pip

    def test_install_requirements_format_does_not_raise(self):
        """Test that .format() does not raise KeyError."""
        from sagemaker.train.templates import INSTALL_REQUIREMENTS

        # This was the CI failure - KeyError: 'region'
        try:
            rendered = INSTALL_REQUIREMENTS.format(
                requirements_file="requirements.txt"
            )
        except KeyError as e:
            pytest.fail(
                f"INSTALL_REQUIREMENTS.format() raised KeyError: {e}"
            )

    def test_install_auto_requirements_has_ca_support(self):
        """Test that INSTALL_AUTO_REQUIREMENTS includes CA_REPOSITORY_ARN."""
        from sagemaker.train.templates import INSTALL_AUTO_REQUIREMENTS

        assert "CA_REPOSITORY_ARN" in INSTALL_AUTO_REQUIREMENTS
        assert (
            "aws codeartifact login --tool pip"
            in INSTALL_AUTO_REQUIREMENTS
        )
