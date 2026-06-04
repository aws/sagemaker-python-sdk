"""MLflow metrics visualization utilities for SageMaker training jobs."""

import logging
from typing import Optional, List, Dict, Any
from sagemaker.core.resources import TrainingJob

logger = logging.getLogger(__name__)


def _is_in_studio() -> bool:
    """Check if running inside SageMaker Studio."""
    from sagemaker.train.common_utils.finetune_utils import _read_domain_id_from_metadata
    return _read_domain_id_from_metadata() is not None


def _get_studio_base_url(region: str) -> str:
    """Get Studio base URL, or empty string if domain not resolvable."""
    from sagemaker.train.common_utils.finetune_utils import _read_domain_id_from_metadata
    domain_id = _read_domain_id_from_metadata()
    if not domain_id or not region:
        return ""
    return f"https://studio-{domain_id}.studio.{region}.sagemaker.aws"


def _parse_job_arn(job_arn: str):
    """Parse a SageMaker job ARN into (region, resource) or None."""
    import re
    m = re.match(r'arn:aws(?:-[a-z]+)?:sagemaker:([a-z0-9-]+):\d+:(\S+)', job_arn)
    return (m.group(1), m.group(2)) if m else None


def get_console_job_url(job_arn: str) -> str:
    """Get AWS Console URL for a SageMaker job ARN.
    
    Args:
        job_arn: Full ARN like arn:aws:sagemaker:us-east-1:123:training-job/my-job
        
    Returns:
        Console URL or empty string.
    """
    parsed = _parse_job_arn(job_arn)
    if not parsed:
        return ""
    region, resource = parsed
    job_type_map = {
        "training-job/": "#/jobs/",
        "processing-job/": "#/processing-jobs/",
        "transform-job/": "#/transform-jobs/",
    }
    for prefix, fragment in job_type_map.items():
        if resource.startswith(prefix):
            job_name = resource.split("/", 1)[1]
            return f"https://{region}.console.aws.amazon.com/sagemaker/home?region={region}{fragment}{job_name}"
    return ""


def get_cloudwatch_logs_url(job_arn: str) -> str:
    """Get CloudWatch Logs console URL for a SageMaker job ARN.
    
    Returns:
        CloudWatch console URL or empty string.
    """
    parsed = _parse_job_arn(job_arn)
    if not parsed:
        return ""
    region, resource = parsed
    log_group_map = {
        "training-job/": "/aws/sagemaker/TrainingJobs",
        "processing-job/": "/aws/sagemaker/ProcessingJobs",
        "transform-job/": "/aws/sagemaker/TransformJobs",
    }
    for prefix, log_group in log_group_map.items():
        if resource.startswith(prefix):
            job_name = resource.split("/", 1)[1]
            encoded_group = log_group.replace("/", "$252F")
            return (
                f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
                f"#logsV2:log-groups/log-group/{encoded_group}"
                f"$3FlogStreamNameFilter$3D{job_name}"
            )
    return ""


def get_studio_url(training_job, domain_id: str = None) -> str:
    """Get SageMaker Studio URL for training job logs.
    
    Args:
        training_job: SageMaker TrainingJob object, job name string, or job ARN string
        domain_id: Studio domain ID (e.g., 'd-xxxxxxxxxxxx'). If not provided, attempts to auto-detect
        
    Returns:
        Studio URL pointing to the training job details, or empty string if not resolvable
        
    Example:
        >>> from sagemaker.train import get_studio_url
        >>> url = get_studio_url('my-training-job')
        >>> url = get_studio_url('arn:aws:sagemaker:us-west-2:123456789:training-job/my-job')
    """
    import re

    if isinstance(training_job, str):
        arn_match = re.match(
            r'arn:aws(?:-[a-z]+)?:sagemaker:([a-z0-9-]+):\d+:training-job/(.+)',
            training_job,
        )
        if arn_match:
            region = arn_match.group(1)
            job_name = arn_match.group(2)
        else:
            # Plain job name — use session region
            training_job = TrainingJob.get(training_job_name=training_job)
            from sagemaker.core.utils.utils import SageMakerClient
            region = SageMakerClient().region_name
            job_name = training_job.training_job_name
    else:
        from sagemaker.core.utils.utils import SageMakerClient
        region = SageMakerClient().region_name
        job_name = training_job.training_job_name
    
    base = _get_studio_base_url(region)
    if not base:
        return ""
    return f"{base}/jobs/train/{job_name}"


def display_job_links_html(rows: list, as_html: bool = False):
    """Render job/resource links with copy-to-clipboard buttons as a Jupyter HTML table.

    Args:
        rows: List of dicts, each with keys:
            - label (str): Row label (e.g. step name, "Training Job", "MLflow Experiment")
            - arn (str): The ARN or URI to display and copy
            - url (Optional[str]): Clickable link URL. If None, resolved via get_studio_url for job ARNs.
            - url_text (Optional[str]): Link display text. Defaults to "🔗 link"
            - url_hint (Optional[str]): Hint text after link. Defaults to "(please sign in to Studio first)"
        as_html: If True, return HTML object instead of displaying it.

    Returns:
        HTML object if as_html=True, otherwise None.
    """
    from IPython.display import display, HTML
    import html as html_mod

    html_rows = ""
    for row in rows:
        escaped_arn = html_mod.escape(row['arn'])
        escaped_label = html_mod.escape(row['label'])

        url = row.get('url')
        if url is None:
            url = get_studio_url(row['arn'])
        url_text = row.get('url_text', '🔗 link')
        url_hint = row.get('url_hint', '(please sign in to Studio first)')

        link_html = ""
        if url:
            link_html = (
                f'<a href="{html_mod.escape(url)}" target="_blank" '
                f'style="color:var(--jp-brand-color1,#4a90d9);text-decoration:none;">{html_mod.escape(url_text)}</a>'
                f' <span style="color:var(--jp-ui-font-color2,#888);font-size:11px;">{html_mod.escape(url_hint)}</span>'
            )

        copy_btn = (
            f'<button onclick="navigator.clipboard.writeText(\'{escaped_arn}\')'
            f'.then(()=>{{this.textContent=\'✓\';setTimeout(()=>this.textContent=\'📋\',1500)}})"'
            f' style="border:1px solid var(--jp-border-color1,#555);'
            f'background:var(--jp-layout-color2,#333);color:var(--jp-ui-font-color0,white);'
            f'border-radius:3px;cursor:pointer;font-size:11px;padding:1px 5px;"'
            f' title="Copy">📋</button>'
        )

        html_rows += (
            f'<tr>'
            f'<td style="padding:4px 8px;text-align:left;font-weight:bold;color:var(--jp-brand-color1,#4fc3f7);">{escaped_label}</td>'
            f'<td style="padding:4px 8px;text-align:left;">{link_html}</td>'
            f'<td style="padding:4px 8px;text-align:left;">'
            f'<code style="font-size:12px;word-break:break-all;">{escaped_arn}</code>'
            f' {copy_btn}</td>'
            f'</tr>'
        )

    result = HTML(
        f'<table style="border-collapse:collapse;margin:4px 0;color:var(--jp-ui-font-color0,inherit);">'
        f'<tr style="border-bottom:1px solid var(--jp-border-color1,#555);">'
        f'<th style="padding:4px 8px;text-align:left;color:var(--jp-brand-color2,#ce93d8);">Step</th>'
        f'<th style="padding:4px 8px;text-align:left;color:var(--jp-brand-color2,#ce93d8);">Job Link</th>'
        f'<th style="padding:4px 8px;text-align:left;color:var(--jp-brand-color2,#ce93d8);">Job ARN</th>'
        f'</tr>{html_rows}</table>'
    )

    if as_html:
        return result
    display(result)


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
