"""MLflow metrics visualization utilities for SageMaker training jobs."""

import logging
from typing import Optional, List, Dict, Any
import boto3
from sagemaker.core.resources import TrainingJob

logger = logging.getLogger(__name__)


def get_studio_url(training_job: TrainingJob, domain_id: str = None) -> str:
    """Get SageMaker Studio URL for training job logs.
    
    Args:
        training_job: SageMaker TrainingJob object or job name string
        domain_id: Studio domain ID (e.g., 'd-xxxxxxxxxxxx'). If not provided, attempts to auto-detect
        
    Returns:
        Studio URL pointing to the training job details
        
    Example:
        >>> from sagemaker.train import get_studio_url
        >>> url = get_studio_url('my-training-job')
    """
    if isinstance(training_job, str):
        training_job = TrainingJob.get(training_job_name=training_job)
    
    region = training_job.region if hasattr(training_job, 'region') and training_job.region else 'us-east-1'
    job_name = training_job.training_job_name
    
    sm_client = boto3.client('sagemaker', region_name=region)
    
    # Auto-detect domain if not provided
    if not domain_id:
        try:
            domains = sm_client.list_domains()['Domains']
            if domains:
                domain_id = domains[0]['DomainId']
        except Exception:
            pass
    
    if not domain_id:
        # Fallback to console URL
        return f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/jobs/{job_name}"
    
    # Studio URL format: https://studio-{domain_id}.studio.{region}.sagemaker.aws/jobs/train/{job_name}
    return f"https://studio-{domain_id}.studio.{region}.sagemaker.aws/jobs/train/{job_name}"


def plot_training_metrics(
    training_job: TrainingJob,
    metrics: Optional[List[str]] = None,
    figsize: tuple = (12, 6)
) -> None:
    """Plot training metrics from MLflow for a completed training job.
    
    Args:
        training_job: SageMaker TrainingJob object or job name string
        metrics: List of metric names to plot. If None, plots all available metrics.
        figsize: Figure size as (width, height)
    """
    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient
    from IPython.display import display
    import logging
    
    logging.getLogger('botocore.credentials').setLevel(logging.WARNING)
    
    if isinstance(training_job, str):
        training_job = TrainingJob.get(training_job_name=training_job)
    
    run_id = training_job.mlflow_details.mlflow_run_id
    
    mlflow.set_tracking_uri(training_job.mlflow_config.mlflow_resource_arn)
    client = MlflowClient()
    
    run = mlflow.get_run(run_id)
    available_metrics = list(run.data.metrics.keys())
    metrics_to_plot = metrics if metrics else available_metrics
    
    # Fetch metric histories
    metric_data = {}
    for metric_name in metrics_to_plot:
        history = client.get_metric_history(run_id, metric_name)
        if history:
            metric_data[metric_name] = history
    
    # Plot
    num_metrics = len(metric_data)
    rows = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(figsize[0], figsize[1] * rows))
    axes = axes.flatten() if num_metrics > 1 else [axes]
    
    for idx, (metric_name, history) in enumerate(metric_data.items()):
        steps = [h.step for h in history]
        values = [h.value for h in history]
        axes[idx].plot(steps, values, linewidth=2, marker='o', markersize=4)
        axes[idx].set_xlabel('Step')
        axes[idx].set_ylabel('Value')
        axes[idx].set_title(metric_name, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
    
    for idx in range(len(metric_data), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Training Metrics: {training_job.training_job_name}', fontweight='bold', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave small space for suptitle
    display(fig)
    plt.close()


def get_available_metrics(training_job: TrainingJob) -> List[str]:
    """Get list of available metrics for a training job.
    
    Args:
        training_job: SageMaker TrainingJob object or job name string
        
    Returns:
        List of metric names
    """
    try:
        import mlflow
    except ImportError:
        logger.error("mlflow package not installed")
        return []
    
    # Handle string input
    if isinstance(training_job, str):
        training_job = TrainingJob.get(training_job_name=training_job)
    
    if not hasattr(training_job, 'mlflow_config') or not training_job.mlflow_config:
        return []
    
    mlflow_details = training_job.mlflow_details
    if not mlflow_details or not mlflow_details.mlflow_run_id:
        return []
    
    mlflow.set_tracking_uri(training_job.mlflow_config.mlflow_resource_arn)
    run = mlflow.get_run(mlflow_details.mlflow_run_id)
    
    return list(run.data.metrics.keys())
