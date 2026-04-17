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

    def test_to_snake_case(self):
        """Test the _to_snake_case helper function."""
        from sagemaker.train.common_utils.trainer_wait import _to_snake_case

        assert _to_snake_case("TrainingJobStatus") == "training_job_status"
        assert _to_snake_case("TrainingJobName") == "training_job_name"
        assert _to_snake_case("SecondaryStatus") == "secondary_status"
        assert _to_snake_case("already_snake") == "already_snake"

    def test_refresh_training_job_updates_attributes(self):
        """Test that _refresh_training_job updates job attributes from describe response."""
        from sagemaker.train.common_utils.trainer_wait import (
            _refresh_training_job,
        )

        mock_session = MagicMock()
        mock_session.sagemaker_client.describe_training_job.return_value = {
            "TrainingJobStatus": "Completed",
            "TrainingJobName": "test-job",
            "SecondaryStatus": "Completed",
        }

        mock_job = MagicMock()
        mock_job.training_job_name = "test-job"
        mock_job.training_job_status = "InProgress"
        mock_job.secondary_status = "Training"

        _refresh_training_job(mock_job, sagemaker_session=mock_session)

        # Verify attributes were updated via setattr
        mock_session.sagemaker_client.describe_training_job.assert_called_once()
