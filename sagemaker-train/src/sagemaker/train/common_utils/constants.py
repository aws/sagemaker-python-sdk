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
"""Constants used across training utilities modules."""

class _MLflowConstants:
    """Constants related to MLflow functionality."""
    
    # ARN patterns and prefixes
    SAGEMAKER_ARN_PREFIX = 'arn:aws:sagemaker:'
    
    # Metric names
    TOTAL_LOSS_METRIC = 'total_loss'
    EPOCH_KEYWORD = 'epoch'
    
    # MLflow run tags
    MLFLOW_RUN_NAME_TAG = 'mlflow.runName'
    
    # Error messages
    SAGEMAKER_MLFLOW_REQUIRED_MSG = (
        "sagemaker-mlflow package is required for SageMaker ARN support. "
        "Install with: pip install sagemaker-mlflow"
    )


class _TrainingJobConstants:
    """Constants related to training job monitoring."""
    
    # Status values
    TERMINAL_STATUSES = ["Completed", "Failed", "Stopped"]
    TRAINING_STATUS = "Training"
    COMPLETED_STATUS = "Completed"
    FAILED_STATUS = "Failed"
    
    # Default values
    DEFAULT_POLL_INTERVAL = 3
    DEFAULT_AWS_REGION = 'us-west-2'
    DEFAULT_PROGRESS_WAIT_TIME = 20
    
    # UI constants
    JUPYTER_KERNEL_APP = 'IPKernelApp'
    PANEL_WIDTH_RATIO = 0.8
    DEFAULT_PANEL_WIDTH = 80
    PROGRESS_BAR_SEGMENTS = 20
    PROGRESS_BAR_DIVISOR = 5
    
    # Display messages and formatting
    TRAINING_COMPLETED_MSG = "✓ Training completed! View metrics in MLflow: {}"
    MLFLOW_URL_ERROR_MSG = "Could not get MLflow URL: {}"
    LOSS_METRICS_HEADER = "\n------------ Loss Metrics by Epoch ------------"
    LOSS_METRICS_FOOTER = "----------------------------------------------"
    STATUS_SEPARATOR = "\n--------------------------------------\n"
    
    # Progress indicators
    COMPLETED_CHECK = "✓"
    RUNNING_CHECK = "⋯"
    RUNNING_DURATION = "Running..."
    
    # Hardcoded server name (should be made configurable in production)
    DEFAULT_MLFLOW_SERVER = 'mmlu-eval-experiment'


class _ValidationConstants:
    """Constants for input validation."""
    
    # Error messages
    EMPTY_TRACKING_URI_MSG = "tracking_uri cannot be empty"
    EMPTY_EXPERIMENT_NAME_MSG = "experiment_name cannot be empty"
    EMPTY_RUN_ID_MSG = "run_id cannot be empty"
    EMPTY_METRIC_NAME_MSG = "metric_name cannot be empty"
    EMPTY_TRACKING_SERVER_NAME_MSG = "tracking_server_name cannot be empty"
    EMPTY_REGION_MSG = "region cannot be empty"
    POSITIVE_POLL_MSG = "Poll interval must be positive"
    POSITIVE_TIMEOUT_MSG = "Timeout must be positive or None"
    
    # Validation patterns
    MIN_POLL_INTERVAL = 1
    MIN_TIMEOUT = 1


class _ErrorConstants:
    """Constants for error handling and messages."""
    
    # MLflow errors
    MLFLOW_INIT_ERROR = "Failed to initialize MLflow metrics utility: {}"
    EXPERIMENT_NOT_FOUND = "Experiment '{}' not found"
    RUNS_LIST_ERROR = "Failed to list runs: {}"
    LOSS_METRICS_ERROR = "Failed to retrieve loss metrics: {}"
    ALL_METRICS_ERROR = "Failed to retrieve metrics for run {}: {}"
    METRIC_HISTORY_ERROR = "Failed to retrieve metric history for {} in run {}: {}"
    LOSS_METRICS_STEP_ERROR = "Failed to get loss metrics by step: {}"
    LOSS_METRICS_EPOCH_ERROR = "Failed to get loss metrics by epoch: {}"
    TOTAL_LOSS_ERROR = "Failed to get most recent total loss: {}"
    NO_RUNS_FOUND = "No runs found for experiment '{}'{}"
    
    # Endpoint errors
    NO_TRACKING_URL = "No tracking server URL found for server '{}'"
    ENDPOINT_RETRIEVAL_ERROR = "Failed to retrieve tracking server endpoint: {}"
    RESOURCE_NOT_FOUND_ERROR = "MLflow tracking server '{}' not found in region '{}'"
    
    # General error prefixes
    ERROR_PREFIX = "[ERROR] Exception: {}: {}"
