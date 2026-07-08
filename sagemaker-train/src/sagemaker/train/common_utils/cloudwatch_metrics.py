"""CloudWatch log-based training metrics extraction and plotting.

Parses CloudWatch log events from SageMaker Training Jobs and HyperPod clusters
to extract step-level training metrics (loss, reward scores) and plot them.

Supports:
- SFT/CPT on SMTJ and SMHP: reduced_train_loss
- RLVR on SMTJ: critic/rewards/mean
- RLVR on SMHP: train_rm_score
"""

from __future__ import absolute_import

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GLOBAL_STEP_REGEX = r"global_step[=:]\s*([\d.]+)"
TRAINING_LOSS_REGEX = r"reduced_train_loss[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
LEARNING_RATE_REGEX = r"(?<![a-z_])lr[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMHP_RLVR_REWARD_SCORE_REGEX = r"train_rm_score:\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"
SMTJ_RLVR_REWARD_SCORE_REGEX = r"critic/rewards/mean[=:]\s*(-?[\d.]+(?:[eE][+-]?\d+)?)"

# Metric patterns keyed by (platform, customization_technique)
AVAILABLE_METRICS: Dict[str, Dict[str, Dict[str, str]]] = {
    "smtj": {
        "SFT": {"training_loss": TRAINING_LOSS_REGEX, "lr": LEARNING_RATE_REGEX},
        "CPT": {"training_loss": TRAINING_LOSS_REGEX, "lr": LEARNING_RATE_REGEX},
        "RLVR": {"reward_score": SMTJ_RLVR_REWARD_SCORE_REGEX},
    },
    "smhp": {
        "SFT": {"training_loss": TRAINING_LOSS_REGEX, "lr": LEARNING_RATE_REGEX},
        "CPT": {"training_loss": TRAINING_LOSS_REGEX, "lr": LEARNING_RATE_REGEX},
        "RLVR": {"reward_score": SMHP_RLVR_REWARD_SCORE_REGEX},
    },
}

_UNSUPPORTED_TECHNIQUES = {"DPO", "RLAIF"}

def _get_smtj_log_group() -> str:
    """Return the CW log group for SageMaker Training Jobs."""
    return "/aws/sagemaker/TrainingJobs"


def _get_smhp_log_group(cluster_name: str, sagemaker_session) -> str:
    """Return the CW log group for a HyperPod cluster.

    The log group follows the pattern:
        /aws/sagemaker/Clusters/{cluster_name}/{cluster_id}
    """
    region_name = sagemaker_session.boto_session.region_name
    sagemaker_client = sagemaker_session.boto_session.client("sagemaker", region_name=region_name)
    response = sagemaker_client.describe_cluster(ClusterName=cluster_name)
    cluster_arn = response["ClusterArn"]
    cluster_id = cluster_arn.split("/")[-1]
    return f"/aws/sagemaker/Clusters/{cluster_name}/{cluster_id}"


def _fetch_smtj_logs(
    job_name: str,
    logs_client,
    log_group: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch CloudWatch log events for an SMTJ training job."""
    # Find the job's dedicated log stream
    try:
        response = logs_client.describe_log_streams(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
        )
    except Exception as e:
        logger.warning(f"Could not describe log streams for job '{job_name}': {e}")
        return []

    streams = response.get("logStreams", [])
    if not streams:
        logger.warning(f"No log stream found for job '{job_name}' in {log_group}")
        return []

    log_stream_name = streams[0]["logStreamName"]

    # Read events with get_log_events
    all_events: List[Dict[str, Any]] = []
    next_token = None
    end_time_ms = end_time or int(datetime.now().timestamp() * 1000)

    while True:
        params: Dict[str, Any] = {
            "logGroupName": log_group,
            "logStreamName": log_stream_name,
            "startFromHead": False,
            "endTime": end_time_ms,
        }
        if start_time:
            params["startTime"] = start_time
        if next_token:
            params["nextToken"] = next_token

        response = logs_client.get_log_events(**params)
        events = response.get("events", [])
        all_events.extend(events)

        current_token = next_token
        next_token = response.get("nextBackwardToken")
        if next_token == current_token:
            break

    return all_events


def _fetch_smhp_logs(
    job_id: str,
    logs_client,
    log_group: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Fetch CloudWatch log events for a HyperPod training job.

    HyperPod doesn't separate log streams by job — uses filter_log_events
    with the job ID as the filter pattern.
    """
    all_events: List[Dict[str, Any]] = []
    next_token = None
    end_time_ms = end_time or int(datetime.now().timestamp() * 1000)

    while True:
        params: Dict[str, Any] = {
            "logGroupName": log_group,
            "logStreamNamePrefix": "SagemakerHyperPodTrainingJob",
            "filterPattern": f'"{job_id}"',
            "endTime": end_time_ms,
        }
        if start_time:
            params["startTime"] = start_time
        if next_token:
            params["nextToken"] = next_token

        try:
            response = logs_client.filter_log_events(**params)
        except Exception as e:
            logger.warning(f"Could not filter log events for HP job '{job_id}': {e}")
            return all_events

        events = response.get("events", [])
        all_events.extend(events)

        next_token = response.get("nextToken")
        if not next_token:
            break

    return all_events


def parse_metrics_from_logs(
    logs: List[Dict[str, Any]],
    platform: str,
    customization_technique: str,
    metrics: Optional[List[str]] = None,
) -> "pandas.DataFrame":
    """Parse training metrics from CloudWatch log events.

    Scans each log line for global_step, then extracts the relevant metric
    value from the same line based on the platform and training technique.

    Args:
        logs: List of CW log event dicts (each with a "message" key).
        platform: "smtj" or "smhp".
        customization_technique: "SFT", "CPT", or "RLVR".
        metrics: Optional list of metric names to extract. If None, extracts all
            available metrics for the given platform/technique combination.

    Returns:
        pandas DataFrame with columns ["global_step", <metric_name>, ...].

    Raises:
        ImportError: If pandas is not installed.
        NotImplementedError: If the technique is not supported.
        ValueError: If a requested metric is not available.
    """
    try:
        import pandas
    except ImportError:
        raise ImportError(
            "pandas is required for metric extraction. "
            "Install it with: pip install pandas\n"
        )

    technique = customization_technique.upper()

    if technique in _UNSUPPORTED_TECHNIQUES:
        raise NotImplementedError(
            f"Training metrics extraction is not supported for {technique} jobs. "
            f"Supported techniques: SFT, CPT, RLVR."
        )

    available = AVAILABLE_METRICS.get(platform, {}).get(technique)
    if not available:
        raise NotImplementedError(
            f"No metric patterns defined for technique '{technique}' on platform '{platform}'. "
            f"Supported: {list(AVAILABLE_METRICS.get(platform, {}).keys())}"
        )

    if not metrics:
        metrics = list(available.keys())

    patterns = []
    for metric_name in metrics:
        if metric_name not in available:
            raise ValueError(
                f"Metric '{metric_name}' is not available for {technique} on {platform}. "
                f"Available metrics: {list(available.keys())}"
            )
        patterns.append(available[metric_name])

    # Parse log lines — extract each metric independently per line.
    # A line must have global_step plus at least one requested metric to be included.
    all_rows: List[List] = []
    log_lines = [line for log in logs for line in log.get("message", "").splitlines()]

    for line in log_lines:
        step_match = re.search(GLOBAL_STEP_REGEX, line)
        if not step_match:
            continue

        step_value = int(float(step_match.group(1)))
        row_values: List = [step_value]
        found_any = False

        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                row_values.append(float(match.group(1)))
                found_any = True
            else:
                row_values.append(None)

        if found_any:
            all_rows.append(row_values)

    return pandas.DataFrame(all_rows, columns=["global_step"] + metrics)


def plot_metrics(
    metrics_df: "pandas.DataFrame",
    title: str = "Training Metrics",
    starting_step: Optional[int] = None,
    ending_step: Optional[int] = None,
) -> None:
    """Plot training metrics using matplotlib.

    Args:
        metrics_df: DataFrame with "global_step" column and one or more metric columns.
        title: Plot title.
        starting_step: Filter to steps >= this value.
        ending_step: Filter to steps <= this value.

    Raises:
        ImportError: If matplotlib is not installed.
        ValueError: If no metrics found in the specified range.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting metrics. "
            "Install it with: pip install matplotlib\n"
        )

    if metrics_df.empty:
        raise ValueError("No metrics data available to plot.")

    # Deduplicate and filter by step range
    df = metrics_df.drop_duplicates(subset=["global_step"], keep="last").copy()

    if starting_step is not None:
        df = df[df["global_step"] >= starting_step]
    if ending_step is not None:
        df = df[df["global_step"] <= ending_step]

    if df.empty:
        range_desc = f"[{starting_step or 'start'} - {ending_step or 'end'}]"
        raise ValueError(f"No metrics found in the specified step range {range_desc}")

    df = df.sort_values("global_step").reset_index(drop=True)

    # Plot each metric in its own subplot (stacked vertically)
    metric_columns = [col for col in df.columns if col != "global_step"]
    # Only plot columns that have at least one non-null value
    metric_columns = [col for col in metric_columns if df[col].notna().any()]

    num_metrics = len(metric_columns)
    if num_metrics == 0:
        raise ValueError("No plottable metric data found.")

    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics), squeeze=False)

    for idx, col in enumerate(metric_columns):
        ax = axes[idx, 0]
        col_data = df[["global_step", col]].dropna(subset=[col])
        ax.plot(col_data["global_step"], col_data[col], linewidth=1.5)
        ax.set_xlabel("Global Step")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    fig.tight_layout()
    plt.show()


def fetch_and_plot_metrics(
    job_id: str,
    platform: str,
    customization_technique: str,
    sagemaker_session,
    cluster_name: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    starting_step: Optional[int] = None,
    ending_step: Optional[int] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> "pandas.DataFrame":
    """Fetch CW logs, parse metrics, and plot them.

    This is the main entry point used by BaseTrainer.show_metrics().

    Args:
        job_id: Training job name (SMTJ) or HyperPod job name.
        platform: "smtj" or "smhp".
        customization_technique: "SFT", "CPT", or "RLVR".
        sagemaker_session: SageMaker session (provides boto_session).
        cluster_name: Required for SMHP platform — the HyperPod cluster name.
        metrics: Optional list of metric names to extract.
        starting_step: Filter to steps >= this value.
        ending_step: Filter to steps <= this value.
        start_time: Optional epoch ms to filter logs from (speeds up retrieval).
        end_time: Optional epoch ms to filter logs until.

    Returns:
        pandas DataFrame with the extracted metrics.

    Raises:
        NotImplementedError: If the technique is not supported (e.g., DPO).
        ValueError: If no logs or metrics are found.
    """
    technique = customization_technique.upper()

    if technique in _UNSUPPORTED_TECHNIQUES:
        raise NotImplementedError(
            f"show_metrics() is not supported for {technique} jobs. "
            f"Supported training techniques: SFT, CPT, RLVR."
        )

    # Validate technique is recognized before doing any expensive log fetching
    available = AVAILABLE_METRICS.get(platform, {}).get(technique)
    if not available:
        supported = list(AVAILABLE_METRICS.get(platform, {}).keys())
        raise ValueError(
            f"'{customization_technique}' is not a supported training technique. "
            f"Supported techniques for platform '{platform}': {supported}"
        )

    region_name = sagemaker_session.boto_session.region_name
    logs_client = sagemaker_session.boto_session.client("logs", region_name=region_name)

    # Fetch logs based on platform
    if platform == "smhp":
        if not cluster_name:
            raise ValueError(
                "cluster_name is required for HyperPod metrics. "
                "This should be available from the compute configuration."
            )
        log_group = _get_smhp_log_group(cluster_name, sagemaker_session)
        log_events = _fetch_smhp_logs(
            job_id=job_id,
            logs_client=logs_client,
            log_group=log_group,
            start_time=start_time,
            end_time=end_time,
        )
    else:
        # SMTJ (serverless or serverful)
        log_group = _get_smtj_log_group()
        log_events = _fetch_smtj_logs(
            job_name=job_id,
            logs_client=logs_client,
            log_group=log_group,
            start_time=start_time,
            end_time=end_time,
        )

    if not log_events:
        raise ValueError(
            f"No CloudWatch logs found for job '{job_id}' in log group '{log_group}'. "
            f"The job may still be starting, or logs may not be available yet."
        )

    # Parse metrics from logs
    metrics_df = parse_metrics_from_logs(
        logs=log_events,
        platform=platform,
        customization_technique=technique,
        metrics=metrics,
    )

    if metrics_df.empty:
        raise ValueError(
            f"No training metrics could be extracted from logs for job '{job_id}'. "
            f"The job may not have started training steps yet."
        )

    # Sort by global_step before plotting and returning
    metrics_df = metrics_df.sort_values("global_step").reset_index(drop=True)

    # Plot
    title = f"Training Metrics: {job_id}"
    plot_metrics(
        metrics_df=metrics_df,
        title=title,
        starting_step=starting_step,
        ending_step=ending_step,
    )

    return metrics_df
