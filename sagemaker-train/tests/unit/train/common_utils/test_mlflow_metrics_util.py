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
"""Unit tests for mlflow_metrics_util module."""

import pytest
from unittest.mock import MagicMock, patch, Mock
import pandas as pd

from sagemaker.train.common_utils.mlflow_metrics_util import (
    _MLflowMetricsUtil,
    _MLflowMetricsError,
)
from sagemaker.train.common_utils.constants import (
    _ErrorConstants,
    _MLflowConstants,
    _ValidationConstants,
)


class TestMLflowMetricsUtil:
    """Test cases for MLflowMetricsUtil class."""

    def test_init_success_with_standard_uri(self):
        """Test successful initialization with standard tracking URI."""
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.get_experiment_by_name') as mock_get_exp:
            
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = 'exp123'
            mock_get_exp.return_value = mock_experiment
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            assert util.experiment_name == 'test_experiment'
            assert util.tracking_server_arn is None
            mock_set_uri.assert_called_once_with('http://localhost:5000')
            mock_get_exp.assert_called_once_with('test_experiment')

    def test_init_success_with_sagemaker_arn(self):
        """Test successful initialization with SageMaker ARN."""
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('sagemaker.train.common_utils.mlflow_metrics_util.sagemaker_mlflow', create=True):
            
            mock_experiment = MagicMock()
            mock_get_exp.return_value = mock_experiment
            
            arn = 'arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test'
            util = _MLflowMetricsUtil(arn, 'test_experiment')
            
            assert util.tracking_server_arn == arn
            mock_set_uri.assert_called_once_with(arn)

    def test_init_sagemaker_arn_without_sagemaker_mlflow(self):
        """Test initialization with SageMaker ARN but no sagemaker-mlflow package."""
        with patch('mlflow.get_experiment_by_name'), \
             patch('sagemaker.train.common_utils.mlflow_metrics_util.sagemaker_mlflow', None):
            
            arn = 'arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test'
            
            with pytest.raises(ImportError, match=_MLflowConstants.SAGEMAKER_MLFLOW_REQUIRED_MSG):
                _MLflowMetricsUtil(arn, 'test_experiment')

    def test_init_empty_tracking_uri(self):
        """Test initialization with empty tracking URI."""
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_TRACKING_URI_MSG):
            _MLflowMetricsUtil('', 'test_experiment')
        
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_TRACKING_URI_MSG):
            _MLflowMetricsUtil(None, 'test_experiment')

    def test_init_empty_experiment_name(self):
        """Test initialization with empty experiment name."""
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_EXPERIMENT_NAME_MSG):
            _MLflowMetricsUtil('http://localhost:5000', '')
        
        with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_EXPERIMENT_NAME_MSG):
            _MLflowMetricsUtil('http://localhost:5000', None)

    def test_init_experiment_not_found(self):
        """Test initialization when experiment is not found."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name', return_value=None):
            
            with pytest.raises(_MLflowMetricsError) as exc_info:
                _MLflowMetricsUtil('http://localhost:5000', 'nonexistent_experiment')
            
            assert _ErrorConstants.EXPERIMENT_NOT_FOUND.format('nonexistent_experiment') in str(exc_info.value)

    def test_list_runs_success(self):
        """Test successful run listing."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.search_runs') as mock_search:
            
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = 'exp123'
            mock_get_exp.return_value = mock_experiment
            
            mock_runs_df = pd.DataFrame([{'run_id': 'run1', 'status': 'FINISHED'}])
            mock_search.return_value = mock_runs_df
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            runs = util._list_runs()
            
            assert len(runs) == 1
            assert runs[0]['run_id'] == 'run1'
            mock_search.assert_called_once_with(
                experiment_ids=['exp123'],
                filter_string=None
            )

    def test_list_runs_with_filter(self):
        """Test run listing with run name filter."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.search_runs') as mock_search:
            
            mock_experiment = MagicMock()
            mock_experiment.experiment_id = 'exp123'
            mock_get_exp.return_value = mock_experiment
            
            mock_runs_df = pd.DataFrame([])
            mock_search.return_value = mock_runs_df
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            runs = util._list_runs('specific_run')
            
            assert runs == []
            expected_filter = f"tags.{_MLflowConstants.MLFLOW_RUN_NAME_TAG} = 'specific_run'"
            mock_search.assert_called_once_with(
                experiment_ids=['exp123'],
                filter_string=expected_filter
            )

    def test_get_loss_metrics_success(self):
        """Test successful loss metrics retrieval."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.tracking.MlflowClient') as mock_client_class:
            
            mock_experiment = MagicMock()
            mock_get_exp.return_value = mock_experiment
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Mock run data
            mock_run = MagicMock()
            mock_run.data.metrics = {'total_loss': 0.5, 'accuracy': 0.9}
            mock_client.get_run.return_value = mock_run
            
            # Mock metric history
            mock_metric_point = MagicMock()
            mock_metric_point.step = 1
            mock_metric_point.value = 0.5
            mock_metric_point.timestamp = 1234567890
            mock_client.get_metric_history.return_value = [mock_metric_point]
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            with patch.object(util, '_get_run_ids', return_value=['run123']):
                metrics = util._get_loss_metrics(run_id='run123')
            
            assert 'run123' in metrics
            assert len(metrics['run123']) == 1
            assert metrics['run123'][0]['metric_name'] == 'total_loss'
            assert metrics['run123'][0]['value'] == 0.5

    def test_get_all_metrics_success(self):
        """Test successful retrieval of all metrics."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.tracking.MlflowClient') as mock_client_class:
            
            mock_experiment = MagicMock()
            mock_get_exp.return_value = mock_experiment
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_run = MagicMock()
            mock_run.data.metrics = {'loss': 0.1, 'accuracy': 0.95}
            mock_client.get_run.return_value = mock_run
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            metrics = util._get_all_metrics('run123')
            
            assert metrics == {'loss': 0.1, 'accuracy': 0.95}
            mock_client.get_run.assert_called_once_with('run123')

    def test_get_all_metrics_empty_run_id(self):
        """Test get_all_metrics with empty run_id."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_RUN_ID_MSG):
                util._get_all_metrics('')

    def test_get_metric_history_success(self):
        """Test successful metric history retrieval."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name') as mock_get_exp, \
             patch('mlflow.tracking.MlflowClient') as mock_client_class:
            
            mock_experiment = MagicMock()
            mock_get_exp.return_value = mock_experiment
            
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            mock_point1 = MagicMock()
            mock_point1.step = 1
            mock_point1.value = 0.5
            mock_point1.timestamp = 1234567890
            
            mock_point2 = MagicMock()
            mock_point2.step = 2
            mock_point2.value = 0.3
            mock_point2.timestamp = 1234567891
            
            mock_client.get_metric_history.return_value = [mock_point1, mock_point2]
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            history = util.get_metric_history('run123', 'loss')
            
            assert len(history) == 2
            assert history[0]['step'] == 1
            assert history[0]['value'] == 0.5
            assert history[1]['step'] == 2
            assert history[1]['value'] == 0.3

    def test_get_metric_history_empty_inputs(self):
        """Test get_metric_history with empty inputs."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_RUN_ID_MSG):
                util.get_metric_history('', 'loss')
            
            with pytest.raises(ValueError, match=_ValidationConstants.EMPTY_METRIC_NAME_MSG):
                util.get_metric_history('run123', '')

    def test_get_most_recent_total_loss_success(self):
        """Test successful retrieval of most recent total loss."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            mock_loss_metrics = {
                'run123': [
                    {
                        'metric_name': 'total_loss',
                        'value': 0.1,
                        'history': [
                            {'step': 1, 'value': 0.5, 'timestamp': 1234567890},
                            {'step': 2, 'value': 0.3, 'timestamp': 1234567891},
                            {'step': 3, 'value': 0.1, 'timestamp': 1234567892}
                        ]
                    }
                ]
            }
            
            with patch.object(util, '_get_loss_metrics', return_value=mock_loss_metrics):
                recent_loss = util._get_most_recent_total_loss('run123')
            
            assert recent_loss == 0.1

    def test_get_most_recent_total_loss_not_found(self):
        """Test get_most_recent_total_loss when no total_loss found."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            with patch.object(util, '_get_loss_metrics', return_value={}):
                recent_loss = util._get_most_recent_total_loss('run123')
            
            assert recent_loss is None

    def test_get_loss_metrics_by_step_success(self):
        """Test successful loss metrics by step retrieval."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            mock_loss_metrics = {
                'run123': [
                    {
                        'metric_name': 'total_loss',
                        'history': [
                            {'step': 1, 'value': 0.5},
                            {'step': 2, 'value': 0.3}
                        ]
                    }
                ]
            }
            
            with patch.object(util, '_get_loss_metrics', return_value=mock_loss_metrics):
                step_metrics = util._get_loss_metrics_by_step('run123')
            
            assert 1 in step_metrics
            assert 2 in step_metrics
            assert step_metrics[1]['total_loss'] == 0.5
            assert step_metrics[2]['total_loss'] == 0.3

    def test_get_loss_metrics_by_epoch_with_steps_per_epoch(self):
        """Test loss metrics by epoch with steps_per_epoch parameter."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            mock_loss_metrics = {
                'run123': [
                    {
                        'metric_name': 'total_loss',
                        'history': [
                            {'step': 0, 'value': 0.8},
                            {'step': 1, 'value': 0.5},
                            {'step': 2, 'value': 0.3},
                            {'step': 3, 'value': 0.2}
                        ]
                    }
                ]
            }
            
            with patch.object(util, '_get_loss_metrics', return_value=mock_loss_metrics):
                epoch_metrics = util._get_loss_metrics_by_epoch('run123', steps_per_epoch=2)
            
            assert 0 in epoch_metrics  # steps 0,1 -> epoch 0
            assert 1 in epoch_metrics  # steps 2,3 -> epoch 1
            assert epoch_metrics[1]['total_loss'] == 0.2  # Last value in epoch

    def test_get_run_ids_with_run_id(self):
        """Test _get_run_ids with explicit run_id."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            run_ids = util._get_run_ids('run123', None)
            
            assert run_ids == ['run123']

    def test_get_run_ids_with_run_name(self):
        """Test _get_run_ids with run_name."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            mock_runs = [{'run_id': 'run456', 'status': 'FINISHED'}]
            with patch.object(util, '_list_runs', return_value=mock_runs):
                run_ids = util._get_run_ids(None, 'test_run')
            
            assert run_ids == ['run456']

    def test_get_run_ids_no_runs_found(self):
        """Test _get_run_ids when no runs are found."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            with patch.object(util, '_list_runs', return_value=[]):
                with pytest.raises(_MLflowMetricsError) as exc_info:
                    util._get_run_ids(None, 'nonexistent_run')
                
                expected_error = _ErrorConstants.NO_RUNS_FOUND.format(
                    'test_experiment', " with run_name 'nonexistent_run'"
                )
                assert expected_error in str(exc_info.value)

    def test_error_handling_in_methods(self):
        """Test error handling in various methods."""
        with patch('mlflow.set_tracking_uri'), \
             patch('mlflow.get_experiment_by_name'):
            
            util = _MLflowMetricsUtil('http://localhost:5000', 'test_experiment')
            
            # Test error in get_loss_metrics
            with patch.object(util, '_get_run_ids', side_effect=Exception('Test error')):
                with pytest.raises(_MLflowMetricsError) as exc_info:
                    util._get_loss_metrics('run123')
                assert _ErrorConstants.LOSS_METRICS_ERROR.format('Test error') in str(exc_info.value)

    def test_mlflow_metrics_error_creation(self):
        """Test MLflowMetricsError exception class."""
        error_msg = "Test MLflow metrics error"
        error = _MLflowMetricsError(error_msg)
        assert str(error) == error_msg
        assert isinstance(error, Exception)

    def test_whitespace_handling(self):
        """Test that whitespace is handled correctly in inputs."""
        with patch('mlflow.set_tracking_uri') as mock_set_uri, \
             patch('mlflow.get_experiment_by_name') as mock_get_exp:
            
            mock_experiment = MagicMock()
            mock_get_exp.return_value = mock_experiment
            
            util = _MLflowMetricsUtil('  http://localhost:5000  ', '  test_experiment  ')
            
            assert util.experiment_name == 'test_experiment'
            mock_set_uri.assert_called_once_with('http://localhost:5000')
            mock_get_exp.assert_called_once_with('test_experiment')
