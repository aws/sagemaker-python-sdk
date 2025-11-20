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
from unittest.mock import Mock
from collections import deque

from sagemaker.core.remote_function.client import (
    RemoteExecutor,
    _submit_worker,
    _polling_worker,
    _API_CALL_LIMIT,
    _PENDING,
    _RUNNING,
    _CANCELLED,
    _FINISHED,
)


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
