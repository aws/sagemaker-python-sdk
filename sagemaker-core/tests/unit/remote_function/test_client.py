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

import pytest
from unittest.mock import Mock, patch
from collections import deque

from sagemaker.core.remote_function.client import (
    remote,
    RemoteExecutor,
    _submit_worker,
    _polling_worker,
    _API_CALL_LIMIT,
    _PENDING,
    _RUNNING,
    _CANCELLED,
    _FINISHED,
)

TRAINING_PLAN_ARN = "arn:aws:sagemaker:us-west-2:123456789012:training-plan/test-plan"


class TestTrainingPlanArnForwarding:
    """Ensure training_plan_arn is forwarded into _JobSettings by both entry points."""

    @patch("sagemaker.core.remote_function.client._JobSettings")
    def test_remote_decorator_forwards_training_plan_arn(self, mock_job_settings):
        @remote(instance_type="ml.m5.xlarge", training_plan_arn=TRAINING_PLAN_ARN)
        def my_func():
            pass

        _, kwargs = mock_job_settings.call_args
        assert kwargs["training_plan_arn"] == TRAINING_PLAN_ARN

    @patch("sagemaker.core.remote_function.client._JobSettings")
    def test_remote_executor_forwards_training_plan_arn(self, mock_job_settings):
        RemoteExecutor(instance_type="ml.m5.xlarge", training_plan_arn=TRAINING_PLAN_ARN)

        _, kwargs = mock_job_settings.call_args
        assert kwargs["training_plan_arn"] == TRAINING_PLAN_ARN


class TestConstants:
    """Test module constants"""

    def test_api_call_limit_constants(self):
        assert _API_CALL_LIMIT["SubmittingIntervalInSecs"] == 1
        assert _API_CALL_LIMIT["MinBatchPollingIntervalInSecs"] == 10
        assert _API_CALL_LIMIT["PollingIntervalInSecs"] == 0.5

    def test_future_state_constants(self):
        assert _PENDING == "PENDING"
        assert _RUNNING == "RUNNING"
        assert _CANCELLED == "CANCELLED"
        assert _FINISHED == "FINISHED"


class TestRemoteExecutorValidation:
    """Test RemoteExecutor argument validation"""

    def test_validate_submit_args_with_valid_args(self):
        def my_function(x, y, z=10):
            return x + y + z

        RemoteExecutor._validate_submit_args(my_function, 1, 2, z=3)

    def test_validate_submit_args_with_missing_args(self):
        def my_function(x, y):
            return x + y

        with pytest.raises(TypeError):
            RemoteExecutor._validate_submit_args(my_function, 1)

    def test_validate_submit_args_with_extra_args(self):
        def my_function(x):
            return x

        with pytest.raises(TypeError):
            RemoteExecutor._validate_submit_args(my_function, 1, 2)

    def test_validate_env_names_valid(self):
        """Test valid conda environment names"""
        valid_names = [
            "myenv",
            "base",
            "py39",
            "env123",
        ]
        for name in valid_names:
            RemoteExecutor._validate_env_name(name)

    def test_validate_env_names_invalid(self):
        """Test invalid conda environment names"""
        invalid_names = [
            "env && echo PWNED",
            "env > /tmp/output.txt",
            "sagemaker-rce-env; echo PWNED_FROM_CONDA_ENV > /tmp/conda_rce.txt #",
        ]
        for name in invalid_names:
            with pytest.raises(ValueError):
                RemoteExecutor._validate_env_name(name)


class TestWorkerFunctions:
    """Test worker thread functions"""

    def test_submit_worker_exits_on_none(self):
        """Test that submit worker exits when None is in queue"""
        executor = Mock()
        executor._pending_request_queue = deque([None])
        executor._running_jobs = {}
        executor.max_parallel_jobs = 1

        mock_condition = Mock()
        mock_condition.__enter__ = Mock(return_value=mock_condition)
        mock_condition.__exit__ = Mock(return_value=False)
        mock_condition.wait_for = Mock(return_value=True)
        executor._state_condition = mock_condition

        _submit_worker(executor)

        assert len(executor._pending_request_queue) == 0

    def test_polling_worker_exits_on_shutdown(self):
        """Test that polling worker exits when shutdown flag is set"""
        executor = Mock()
        executor._running_jobs = {}
        executor._pending_request_queue = deque()
        executor._shutdown = True
        executor._state_condition = Mock()

        _polling_worker(executor)
