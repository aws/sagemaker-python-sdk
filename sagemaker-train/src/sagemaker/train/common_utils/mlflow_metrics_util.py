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
"""Utility module for MLflow metrics management and retrieval."""

from typing import Any, Optional

import mlflow

from sagemaker.train.common_utils.constants import (
    _ErrorConstants,
    _MLflowConstants,
    _ValidationConstants,
)

# Import sagemaker-mlflow for proper SageMaker integration
try:
    import sagemaker_mlflow
except ImportError:
    sagemaker_mlflow = None


class _MLflowMetricsError(Exception):
    """Raised when MLflow metrics operations fail."""
    pass


class _MLflowMetricsUtil:
    """Utility class for retrieving and managing MLflow metrics data.
    
    This class provides methods to retrieve loss metrics and other training
    metrics from MLflow experiments, with support for both standard MLflow
    tracking URIs and SageMaker ARNs.
    
    Example:
        .. code:: python

            util = MLflowMetricsUtil(
                tracking_uri="http://localhost:5000",
                experiment_name="my_experiment"
            )
            
            # Get recent total loss
            loss = util.get_most_recent_total_loss(run_name="training_run_1")
            
            # Get loss metrics by epoch
            epoch_metrics = util.get_loss_metrics_by_epoch(
                run_name="training_run_1",
                steps_per_epoch=100
            )
    
    Args:
        tracking_uri (str): MLflow tracking server URI or SageMaker ARN.
        experiment_name (str): Name of the MLflow experiment.
    """
    
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize MLflow metrics utility.
        
        Args:
            tracking_uri (str): MLflow tracking server URI or SageMaker ARN.
            experiment_name (str): Name of the MLflow experiment.
            
        Raises:
            ValueError: If tracking_uri or experiment_name is empty or invalid.
            ImportError: If sagemaker-mlflow is required but not installed.
            MLflowMetricsError: If experiment cannot be found or initialized.
        """
        if not tracking_uri or not tracking_uri.strip():
            raise ValueError(_ValidationConstants.EMPTY_TRACKING_URI_MSG)
        
        if not experiment_name or not experiment_name.strip():
            raise ValueError(_ValidationConstants.EMPTY_EXPERIMENT_NAME_MSG)
        
        self.tracking_server_arn: Optional[str] = None
        self.experiment_name = experiment_name.strip()
        
        try:
            # Handle SageMaker ARN
            if tracking_uri.startswith(_MLflowConstants.SAGEMAKER_ARN_PREFIX):
                if sagemaker_mlflow is None:
                    raise ImportError(_MLflowConstants.SAGEMAKER_MLFLOW_REQUIRED_MSG)
                
                self.tracking_server_arn = tracking_uri
                mlflow.set_tracking_uri(tracking_uri)
            else:
                mlflow.set_tracking_uri(tracking_uri.strip())
            
            self.experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not self.experiment:
                raise _MLflowMetricsError(_ErrorConstants.EXPERIMENT_NOT_FOUND.format(self.experiment_name))
                
        except Exception as e:
            if isinstance(e, (ValueError, ImportError, _MLflowMetricsError)):
                raise
            raise _MLflowMetricsError(_ErrorConstants.MLFLOW_INIT_ERROR.format(e)) from e
    
    def _list_runs(self, run_name: Optional[str] = None) -> list[dict[str, Any]]:
        """List all runs in the experiment.
        
        Args:
            run_name (Optional[str]): Optional filter by run name.
            
        Returns:
            list[dict[str, Any]]: List of run information dictionaries.
            
        Raises:
            MLflowMetricsError: If unable to retrieve runs.
        """
        try:
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                filter_string=f"tags.{_MLflowConstants.MLFLOW_RUN_NAME_TAG} = '{run_name}'" if run_name else None
            )
            
            return runs.to_dict('records') if not runs.empty else []
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.RUNS_LIST_ERROR.format(e)) from e
    
    def _get_loss_metrics(
        self, 
        run_id: Optional[str] = None, 
        run_name: Optional[str] = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Fetch loss-related metrics from runs.
        
        Args:
            run_id (Optional[str]): Specific run ID to fetch metrics from.
            run_name (Optional[str]): Specific run name to fetch metrics from.
            
        Returns:
            dict[str, list[dict[str, Any]]]: Dictionary with run_id as key and 
                list of loss metrics as value.
                
        Raises:
            MLflowMetricsError: If unable to retrieve loss metrics.
        """
        try:
            loss_metrics = {}
            run_ids = self._get_run_ids(run_id, run_name)
            
            for rid in run_ids:
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(rid)
                
                loss_data = []
                for metric_key in run.data.metrics:
                    if _MLflowConstants.TOTAL_LOSS_METRIC == metric_key.lower():
                        metric_history = client.get_metric_history(rid, metric_key)
                        loss_data.append({
                            'metric_name': metric_key,
                            'value': run.data.metrics[metric_key],
                            'history': [
                                {
                                    'step': m.step, 
                                    'value': m.value, 
                                    'timestamp': m.timestamp
                                } 
                                for m in metric_history
                            ]
                        })
                
                loss_metrics[rid] = loss_data
            
            return loss_metrics
        
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.LOSS_METRICS_ERROR.format(e)) from e
    
    def _get_all_metrics(self, run_id: str) -> dict[str, Any]:
        """Get all metrics for a specific run.
        
        Args:
            run_id (str): Run ID to fetch metrics from.
            
        Returns:
            dict[str, Any]: Dictionary of all metrics.
            
        Raises:
            ValueError: If run_id is empty.
            MLflowMetricsError: If unable to retrieve metrics.
        """
        if not run_id or not run_id.strip():
            raise ValueError(_ValidationConstants.EMPTY_RUN_ID_MSG)
        
        try:
            client = mlflow.tracking.MlflowClient()
            run = client.get_run(run_id.strip())
            return run.data.metrics
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.ALL_METRICS_ERROR.format(run_id, e)) from e
    
    def get_metric_history(self, run_id: str, metric_name: str) -> list[dict[str, Any]]:
        """Get history of a specific metric.
        
        Args:
            run_id (str): Run ID.
            metric_name (str): Name of the metric.
            
        Returns:
            list[dict[str, Any]]: List of metric history points.
            
        Raises:
            ValueError: If run_id or metric_name is empty.
            MLflowMetricsError: If unable to retrieve metric history.
        """
        if not run_id or not run_id.strip():
            raise ValueError(_ValidationConstants.EMPTY_RUN_ID_MSG)
        if not metric_name or not metric_name.strip():
            raise ValueError(_ValidationConstants.EMPTY_METRIC_NAME_MSG)
        
        try:
            client = mlflow.tracking.MlflowClient()
            metric_history = client.get_metric_history(run_id.strip(), metric_name.strip())
            
            return [
                {'step': m.step, 'value': m.value, 'timestamp': m.timestamp} 
                for m in metric_history
            ]
        except Exception as e:
            raise _MLflowMetricsError(
                _ErrorConstants.METRIC_HISTORY_ERROR.format(metric_name, run_id, e)
            ) from e
    
    def _get_loss_metrics_by_step(
        self, 
        run_id: Optional[str] = None, 
        run_name: Optional[str] = None
    ) -> dict[int, dict[str, float]]:
        """Get loss metrics organized by step.
        
        Args:
            run_id (Optional[str]): Specific run ID to fetch metrics from.
            run_name (Optional[str]): Specific run name to fetch metrics from.
            
        Returns:
            dict[int, dict[str, float]]: Dictionary with step as key and loss 
                metrics as value.
                
        Raises:
            MLflowMetricsError: If unable to retrieve loss metrics by step.
        """
        try:
            loss_metrics = self._get_loss_metrics(run_id, run_name)
            step_data = {}
            
            for rid, metrics in loss_metrics.items():
                for metric in metrics:
                    for point in metric['history']:
                        step = point['step']
                        if step not in step_data:
                            step_data[step] = {}
                        step_data[step][metric['metric_name']] = point['value']
            
            return dict(sorted(step_data.items()))
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.LOSS_METRICS_STEP_ERROR.format(e)) from e
    
    def _get_loss_metrics_by_epoch(
        self, 
        run_id: Optional[str] = None, 
        run_name: Optional[str] = None, 
        steps_per_epoch: Optional[int] = None
    ) -> dict[int, dict[str, float]]:
        """Get loss metrics organized by epoch.
        
        Args:
            run_id (Optional[str]): Specific run ID to fetch metrics from.
            run_name (Optional[str]): Specific run name to fetch metrics from.
            steps_per_epoch (Optional[int]): Number of steps per epoch. If None, 
                tries to infer from metric names.
            
        Returns:
            dict[int, dict[str, float]]: Dictionary with epoch as key and loss 
                metrics as value.
                
        Raises:
            MLflowMetricsError: If unable to retrieve loss metrics by epoch.
        """
        try:
            loss_metrics = self._get_loss_metrics(run_id, run_name)
            epoch_data = {}
            
            for rid, metrics in loss_metrics.items():
                for metric in metrics:
                    # Check if metric name contains epoch info
                    if _MLflowConstants.EPOCH_KEYWORD in metric['metric_name'].lower():
                        # For epoch-based metrics, use the metric value directly
                        for point in metric['history']:
                            epoch = point['step']  # Assuming step represents epoch for epoch metrics
                            if epoch not in epoch_data:
                                epoch_data[epoch] = {}
                            epoch_data[epoch][metric['metric_name']] = point['value']
                    elif steps_per_epoch and steps_per_epoch > 0:
                        # Convert step-based metrics to epoch-based using steps_per_epoch
                        for point in metric['history']:
                            epoch = point['step'] // steps_per_epoch
                            if epoch not in epoch_data:
                                epoch_data[epoch] = {}
                            # Use the last value in each epoch
                            epoch_data[epoch][metric['metric_name']] = point['value']
            
            return dict(sorted(epoch_data.items()))
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.LOSS_METRICS_EPOCH_ERROR.format(e)) from e
    
    def _get_most_recent_total_loss(
        self, 
        run_id: Optional[str] = None, 
        run_name: Optional[str] = None
    ) -> Optional[float]:
        """Get the most recent total_loss metric value.
        
        Args:
            run_id (Optional[str]): Specific run ID to fetch metrics from.
            run_name (Optional[str]): Specific run name to fetch metrics from.
            
        Returns:
            Optional[float]: Most recent total_loss value or None if not found.
            
        Raises:
            MLflowMetricsError: If unable to retrieve total loss metric.
        """
        try:
            loss_metrics = self._get_loss_metrics(run_id, run_name)
            
            for rid, metrics in loss_metrics.items():
                for metric in metrics:
                    if metric['metric_name'].lower() == _MLflowConstants.TOTAL_LOSS_METRIC:
                        if metric['history']:
                            # Get the most recent entry (last in history)
                            return metric['history'][-1]['value']
            
            return None
        except Exception as e:
            raise _MLflowMetricsError(_ErrorConstants.TOTAL_LOSS_ERROR.format(e)) from e
    
    def _get_run_ids(self, run_id: Optional[str], run_name: Optional[str]) -> list[str]:
        """Get run IDs based on provided run_id or run_name.
        
        Args:
            run_id (Optional[str]): Specific run ID.
            run_name (Optional[str]): Specific run name.
            
        Returns:
            List[str]: List of run IDs.
            
        Raises:
            MLflowMetricsError: If no runs are found.
        """
        if run_id:
            return [run_id.strip()]
        
        runs = self._list_runs(run_name)
        if runs:
            # Use only the latest run (first in the list as they're sorted by start_time desc)
            return [runs[0]['run_id']]
        
        raise _MLflowMetricsError(
            _ErrorConstants.NO_RUNS_FOUND.format(
                self.experiment_name, 
                f" with run_name '{run_name}'" if run_name else ""
            )
        )
