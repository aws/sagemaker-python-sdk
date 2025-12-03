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
"""Unit tests for trainer_wait module."""

import pytest
import time
from unittest.mock import MagicMock, patch, Mock, call
from datetime import datetime, timedelta

from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

from sagemaker.train.common_utils.trainer_wait import (
    _setup_mlflow_integration,
    _is_jupyter_environment,
    _is_unassigned_attribute,
    _calculate_training_progress,
    _calculate_transition_duration,
    wait
)


class MockUnassignedAttribute:
    """Mock class to simulate unassigned attributes."""
    def __init__(self):
        self.__class__.__name__ = 'UnassignedValue'


class TestSetupMLflowIntegration:
    """Test cases for _setup_mlflow_integration function."""

    @patch('boto3.client')
    def test_successful_mlflow_setup(self, mock_boto3_client):
        """Test successful MLflow integration setup."""
        # Mock training job with MLflow config
        training_job = MagicMock()
        training_job.mlflow_config.mlflow_resource_arn = 'arn:aws:sagemaker:us-west-2:123456789:mlflow-tracking-server/test-server'
        training_job.mlflow_config.mlflow_run_name = 'test-run'
        training_job.mlflow_config.mlflow_experiment_name = 'test-experiment'

        # Mock SageMaker client
        mock_sm_client = MagicMock()
        mock_sm_client.create_presigned_mlflow_app_url.return_value = {
            'AuthorizedUrl': 'https://test-mlflow-url.com'
        }
        mock_boto3_client.return_value = mock_sm_client

        with patch('sagemaker.train.common_utils.trainer_wait._MLflowMetricsUtil') as mock_metrics_util:
            mock_util_instance = MagicMock()
            mock_metrics_util.return_value = mock_util_instance

            mlflow_url, metrics_util, mlflow_run_name = _setup_mlflow_integration(training_job)

            assert mlflow_url == 'https://test-mlflow-url.com'
            assert metrics_util == mock_util_instance
            assert mlflow_run_name == 'test-run'
            
            mock_boto3_client.assert_called_once_with('sagemaker')
            mock_sm_client.create_presigned_mlflow_app_url.assert_called_once_with(
                Arn='arn:aws:sagemaker:us-west-2:123456789:mlflow-tracking-server/test-server'
            )
            mock_metrics_util.assert_called_once_with(
                tracking_uri='arn:aws:sagemaker:us-west-2:123456789:mlflow-tracking-server/test-server',
                experiment_name='test-experiment'
            )

    def test_mlflow_setup_exception(self):
        """Test MLflow setup when exception occurs."""
        training_job = MagicMock()
        training_job.mlflow_config.mlflow_resource_arn = 'arn:aws:sagemaker:us-west-2:123456789:mlflow-tracking-server/test-server'

        with patch('boto3.client', side_effect=Exception("boto3 error")):
            mlflow_url, metrics_util, mlflow_run_name = _setup_mlflow_integration(training_job)

            assert mlflow_url is None
            assert metrics_util is None
            assert mlflow_run_name is None

    def test_mlflow_setup_no_config(self):
        """Test MLflow setup when training job has no MLflow config."""
        training_job = MagicMock()
        training_job.mlflow_config = None

        with patch('boto3.client') as mock_boto3_client:
            mock_boto3_client.side_effect = AttributeError("'NoneType' object has no attribute")
            
            mlflow_url, metrics_util, mlflow_run_name = _setup_mlflow_integration(training_job)

            assert mlflow_url is None
            assert metrics_util is None
            assert mlflow_run_name is None


class TestIsJupyterEnvironment:
    """Test cases for _is_jupyter_environment function."""

    @patch('IPython.get_ipython')
    def test_jupyter_environment_true(self, mock_get_ipython):
        """Test detection of Jupyter environment."""
        mock_ipython = MagicMock()
        mock_ipython.config = {'IPKernelApp': {}}
        mock_get_ipython.return_value = mock_ipython

        result = _is_jupyter_environment()
        assert result is True

    @patch('IPython.get_ipython')
    def test_jupyter_environment_false_no_ipkernel(self, mock_get_ipython):
        """Test non-Jupyter environment without IPKernelApp."""
        mock_ipython = MagicMock()
        mock_ipython.config = {}
        mock_get_ipython.return_value = mock_ipython

        result = _is_jupyter_environment()
        assert result is False

    @patch('IPython.get_ipython')
    def test_jupyter_environment_false_no_ipython(self, mock_get_ipython):
        """Test non-Jupyter environment without IPython."""
        mock_get_ipython.return_value = None

        result = _is_jupyter_environment()
        assert result is False

    def test_jupyter_environment_import_error(self):
        """Test ImportError when IPython is not available."""
        with patch('builtins.__import__', side_effect=ImportError):
            result = _is_jupyter_environment()
            assert result is False


class TestIsUnassignedAttribute:
    """Test cases for _is_unassigned_attribute function."""

    def test_unassigned_attribute_true(self):
        """Test detection of unassigned attribute."""
        attr = MockUnassignedAttribute()
        result = _is_unassigned_attribute(attr)
        assert result is True

    def test_unassigned_attribute_false(self):
        """Test normal attribute."""
        attr = "normal_value"
        result = _is_unassigned_attribute(attr)
        assert result is False

    def test_unassigned_attribute_none(self):
        """Test None attribute."""
        attr = None
        result = _is_unassigned_attribute(attr)
        assert result is False

    def test_unassigned_attribute_no_class(self):
        """Test attribute without __class__."""
        attr = 42
        result = _is_unassigned_attribute(attr)
        assert result is False


class TestCalculateTrainingProgress:
    """Test cases for _calculate_training_progress function."""

    def test_calculate_progress_success(self):
        """Test successful progress calculation."""
        progress_info = MagicMock()
        progress_info.max_epoch = 10
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = 5
        progress_info.current_step = 50

        metrics_util = MagicMock()
        metrics_util._get_most_recent_total_loss.return_value = 0.123456789
        
        training_job = MagicMock()
        training_job.mlflow_details.mlflow_run_id = 'test-run-id'

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, metrics_util, 'test-run', training_job
        )

        expected_pct = ((5 - 1) * 100 + 50) / (10 * 100) * 100  # 45%
        assert progress_pct == expected_pct
        assert "Epoch 5/10, Step 50/100" in progress_text
        assert "loss: 0.1234568" in progress_text

    def test_calculate_progress_no_progress_info(self):
        """Test progress calculation with no progress info."""
        progress_pct, progress_text = _calculate_training_progress(
            None, None, None, None
        )

        assert progress_pct is None
        assert progress_text == ""

    def test_calculate_progress_unassigned_progress_info(self):
        """Test progress calculation with unassigned progress info."""
        progress_info = MockUnassignedAttribute()
        
        progress_pct, progress_text = _calculate_training_progress(
            progress_info, None, None, None
        )

        assert progress_pct is None
        assert progress_text == ""

    def test_calculate_progress_missing_required_fields(self):
        """Test progress calculation with missing required fields."""
        progress_info = MagicMock()
        progress_info.max_epoch = MockUnassignedAttribute()
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = 5
        progress_info.current_step = 50

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, None, None, None
        )

        assert progress_pct is None
        assert progress_text == ""

    def test_calculate_progress_zero_values(self):
        """Test progress calculation with zero max_epoch or total_steps."""
        progress_info = MagicMock()
        progress_info.max_epoch = 0
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = 5
        progress_info.current_step = 50

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, None, None, None
        )

        assert progress_pct is None
        assert progress_text == ""

    def test_calculate_progress_none_current_values(self):
        """Test progress calculation with None current values."""
        progress_info = MagicMock()
        progress_info.max_epoch = 10
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = None
        progress_info.current_step = None

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, None, None, None
        )

        expected_pct = ((0 - 1) * 100 + 0) / (10 * 100) * 100  # -1%
        assert progress_pct == expected_pct
        assert "Epoch 0/10, Step 0/100" in progress_text

    def test_calculate_progress_metrics_exception(self):
        """Test progress calculation when metrics retrieval fails."""
        progress_info = MagicMock()
        progress_info.max_epoch = 10
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = 5
        progress_info.current_step = 50

        metrics_util = MagicMock()
        metrics_util._get_most_recent_total_loss.side_effect = Exception("metrics error")
        
        training_job = MagicMock()

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, metrics_util, 'test-run', training_job
        )

        expected_pct = ((5 - 1) * 100 + 50) / (10 * 100) * 100  # 45%
        assert progress_pct == expected_pct
        assert "Epoch 5/10, Step 50/100" in progress_text
        assert "loss:" not in progress_text

    def test_calculate_progress_no_metrics_util(self):
        """Test progress calculation without metrics util."""
        progress_info = MagicMock()
        progress_info.max_epoch = 10
        progress_info.total_step_count_per_epoch = 100
        progress_info.current_epoch = 5
        progress_info.current_step = 50

        progress_pct, progress_text = _calculate_training_progress(
            progress_info, None, 'test-run', None
        )

        expected_pct = ((5 - 1) * 100 + 50) / (10 * 100) * 100  # 45%
        assert progress_pct == expected_pct
        assert "Epoch 5/10, Step 50/100" in progress_text
        assert "loss:" not in progress_text


class TestCalculateTransitionDuration:
    """Test cases for _calculate_transition_duration function."""

    def test_calculate_duration_completed(self):
        """Test duration calculation for completed transition."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10.5)
        
        trans = MagicMock()
        trans.start_time = start_time
        trans.end_time = end_time

        duration, check = _calculate_transition_duration(trans)

        assert duration == "10.5s"
        assert check == "✓"

    def test_calculate_duration_running(self):
        """Test duration calculation for running transition."""
        trans = MagicMock()
        trans.start_time = datetime.now()
        trans.end_time = None

        duration, check = _calculate_transition_duration(trans)

        assert duration == "Running..."
        assert check == "⋯"

    def test_calculate_duration_no_start_time(self):
        """Test duration calculation with no start time."""
        trans = MagicMock()
        trans.start_time = None
        trans.end_time = None

        duration, check = _calculate_transition_duration(trans)

        assert duration == ""
        assert check == ""

class TestWaitFunction:
    """Test cases for wait function."""

    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_completed_non_jupyter(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with completed job in non-Jupyter environment."""
        mock_is_jupyter.return_value = False
        mock_setup_mlflow.return_value = (None, None, None)
        mock_time.side_effect = [0, 5, 10]  # start_time, elapsed times

        # Mock training job
        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'Completed'
        training_job.secondary_status = 'Completed'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = None

        wait(training_job, poll=1)

        training_job.refresh.assert_called()
        mock_sleep.assert_called_with(1)

    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    @patch('sagemaker.train.common_utils.trainer_wait._is_unassigned_attribute')
    def test_wait_failed_job(self, mock_is_unassigned, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with failed job."""
        mock_is_jupyter.return_value = False
        mock_setup_mlflow.return_value = (None, None, None)
        mock_time.side_effect = [0, 5]
        mock_is_unassigned.return_value = False  # failure_reason is not unassigned

        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'InProgress'  # Not in terminal states yet
        training_job.secondary_status = 'Failed'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = 'Job failed due to error'

        with pytest.raises(FailedStatusError):
            wait(training_job, poll=1)


    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_with_transitions_non_jupyter(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with status transitions in non-Jupyter environment."""
        mock_is_jupyter.return_value = False
        mock_setup_mlflow.return_value = (None, None, None)
        mock_time.side_effect = [0, 5, 10]

        # Mock transition
        trans = MagicMock()
        trans.status = 'Training'
        trans.status_message = 'Training in progress'
        trans.start_time = datetime.now()
        trans.end_time = None

        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'Completed'
        training_job.secondary_status = 'Completed'
        training_job.secondary_status_transitions = [trans]
        training_job.failure_reason = None
        training_job.progress_info = None

        wait(training_job, poll=1)

        training_job.refresh.assert_called()

    def test_wait_exception_handling(self):
        """Test wait function exception handling."""
        training_job = MagicMock()
        training_job.refresh.side_effect = Exception("Unexpected error")

        with pytest.raises(RuntimeError, match="Training job monitoring failed"):
            wait(training_job)

    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_with_mlflow_metrics_completed(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with MLflow metrics for completed job."""
        mock_is_jupyter.return_value = False
        
        # Mock MLflow setup
        metrics_util = MagicMock()
        metrics_util._get_loss_metrics_by_epoch.return_value = {
            0: {'loss': 0.5, 'accuracy': 0.8},
            1: {'loss': 0.3, 'accuracy': 0.9}
        }
        mock_setup_mlflow.return_value = ('https://mlflow.com', metrics_util, 'test-run')
        mock_time.side_effect = [0, 5]

        # Mock progress info
        progress_info = MagicMock()
        progress_info.total_step_count_per_epoch = 100

        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'Completed'
        training_job.secondary_status = 'Completed'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = None
        training_job.progress_info = progress_info

        wait(training_job, poll=1)

        metrics_util._get_loss_metrics_by_epoch.assert_called_once()

    @patch('time.sleep')
    @patch('time.time')  
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_unassigned_failure_reason(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with unassigned failure reason."""
        mock_is_jupyter.return_value = False
        mock_setup_mlflow.return_value = (None, None, None)
        mock_time.side_effect = [0, 5]

        training_job = MagicMock()
        training_job.training_job_name = 'test-job' 
        training_job.training_job_status = 'Completed'
        training_job.secondary_status = 'Completed'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = MockUnassignedAttribute()

        wait(training_job, poll=1)

        training_job.refresh.assert_called()

    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_stopped_job(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with stopped job."""
        mock_is_jupyter.return_value = False
        mock_setup_mlflow.return_value = (None, None, None)
        mock_time.side_effect = [0, 5]

        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'Stopped'
        training_job.secondary_status = 'Stopped'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = None

        wait(training_job, poll=1)

        training_job.refresh.assert_called()

    @patch('time.sleep')
    @patch('time.time')
    @patch('sagemaker.train.common_utils.trainer_wait._setup_mlflow_integration')
    @patch('sagemaker.train.common_utils.trainer_wait._is_jupyter_environment')
    def test_wait_metrics_exception_non_jupyter(self, mock_is_jupyter, mock_setup_mlflow, mock_time, mock_sleep):
        """Test wait function with metrics exception in non-Jupyter environment."""
        mock_is_jupyter.return_value = False
        
        # Mock MLflow setup with exception
        metrics_util = MagicMock()
        metrics_util._get_loss_metrics_by_epoch.side_effect = Exception("metrics error")
        mock_setup_mlflow.return_value = ('https://mlflow.com', metrics_util, 'test-run')
        mock_time.side_effect = [0, 5]

        # Mock progress info
        progress_info = MagicMock()
        progress_info.total_step_count_per_epoch = 100

        training_job = MagicMock()
        training_job.training_job_name = 'test-job'
        training_job.training_job_status = 'Completed'
        training_job.secondary_status = 'Completed'
        training_job.secondary_status_transitions = []
        training_job.failure_reason = None
        training_job.progress_info = progress_info

        wait(training_job, poll=1)

        # Should complete successfully despite metrics exception
        training_job.refresh.assert_called()
