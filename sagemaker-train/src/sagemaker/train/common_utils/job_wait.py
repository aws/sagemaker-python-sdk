"""Job wait utilities for SageMaker CreateJob API.

Adapted from trainer_wait.py for the Job resource (used by MultiTurnRLTrainer).
MLflow is optional — all mlflow imports are guarded.
"""
from __future__ import annotations

import collections
import json
import logging
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from sagemaker.core.resources import Job
from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("Completed", "Failed", "Stopped")

DEFAULT_LOG_GROUP_PREFIX = "/aws/sagemaker/Job"

MAX_LOG_LINES = 20


def _parse_region_from_arn(job_arn: str) -> Optional[str]:
    """Extract region from a SageMaker ARN."""
    import re

    m = re.match(r"arn:aws(?:-[a-z]+)?:sagemaker:([a-z0-9-]+):\d+:", job_arn)
    return m.group(1) if m else None


def _get_cloudwatch_logs_url(job_arn: str, job_name: str, log_group: str) -> str:
    """Build a CloudWatch Logs console URL for a job.

    Args:
        job_arn: Job ARN (used to extract region).
        job_name: Job name (used as log stream filter).
        log_group: CloudWatch log group name.

    Returns:
        CloudWatch console URL or empty string.
    """
    region = _parse_region_from_arn(job_arn)
    if not region:
        return ""
    encoded_group = log_group.replace("/", "$252F")
    return (
        f"https://{region}.console.aws.amazon.com/cloudwatch/home?region={region}"
        f"#logsV2:log-groups/log-group/{encoded_group}"
        f"$3FlogStreamNameFilter$3D{job_name}"
    )


def _is_unassigned_attribute(attr) -> bool:
    return hasattr(attr, "__class__") and "Unassigned" in attr.__class__.__name__


def _is_jupyter_environment() -> bool:
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        return ipython is not None and "IPKernelApp" in ipython.config
    except ImportError:
        return False


@contextmanager
def _suppress_info_logging():
    root = logging.getLogger()
    original_level = root.level
    root.setLevel(logging.WARNING)
    try:
        yield
    finally:
        root.setLevel(original_level)


def _parse_config_document(job: Job) -> dict:
    doc = job.job_config_document
    if doc and not _is_unassigned_attribute(doc):
        try:
            return json.loads(doc)
        except (json.JSONDecodeError, TypeError):
            pass
    return {}


def _get_mlflow_arn_from_config(config: dict) -> Optional[str]:
    return config.get("TrainingConfig", {}).get("MlflowConfig", {}).get("MlflowResourceArn")


def _get_mlflow_experiment_name(config: dict) -> Optional[str]:
    return config.get("TrainingConfig", {}).get("MlflowConfig", {}).get("MlflowExperimentName")


def _get_mlflow_output_details(config: dict) -> tuple:
    """Extract experiment_id and run_id from ServiceOutput.MlflowDetails."""
    details = config.get("ServiceOutput", {}).get("MlflowDetails", {})
    return details.get("ExperimentId"), details.get("RunId")


def _get_progress_info(config: dict) -> Optional[dict]:
    """Extract ProgressInfo from ServiceOutput of JobConfigDocument.

    Supports two formats:
    - Epoch-based: ``{MaxEpoch, StepsPerEpoch, CurrentEpoch, CurrentStep}``
    - Step-only: ``{MaxSteps, CurrentStep}``

    Returns:
        ProgressInfo dict, or None if not available.
    """
    info = config.get("ServiceOutput", {}).get("ProgressInfo")
    if not info:
        return None
    has_epoch = info.get("MaxEpoch") and info.get("StepsPerEpoch")
    has_steps = info.get("MaxSteps")
    if not has_epoch and not has_steps:
        return None
    return info


def _get_rollout_info(config: dict) -> Optional[Tuple[int, int]]:
    """Extract RolloutInfo from ServiceOutput of JobConfigDocument.

    Returns:
        Tuple of (completed, total), or None if not available.
    """
    info = config.get("ServiceOutput", {}).get("RolloutInfo")
    if not info:
        return None
    total = info.get("Total")
    completed = info.get("Completed")
    if not total or completed is None:
        return None
    return completed, total


def _build_description_with_progress(description: Optional[str], progress_info: Optional[dict]) -> Optional[str]:
    """Enrich description with training details from ProgressInfo.

    Appends max steps, batch size, and dataset size when available.
    """
    if not progress_info:
        return description
    parts = []
    max_epoch = progress_info.get("MaxEpoch", 0)
    steps_per_epoch = progress_info.get("StepsPerEpoch", 0)
    max_steps = progress_info.get("MaxSteps", 0)
    batch_size = progress_info.get("BatchSize", 0)

    if max_epoch and steps_per_epoch:
        parts.append(f"{max_epoch} epochs, {steps_per_epoch} steps/epoch")
    elif max_steps:
        parts.append(f"{max_steps} max steps")
    if batch_size:
        parts.append(f"batch size {batch_size}")
    if batch_size and steps_per_epoch:
        dataset_size = steps_per_epoch * batch_size
        parts.append(f"dataset size {dataset_size}")

    if not parts:
        return description
    detail = ", ".join(parts)
    return f"{description}, {detail}" if description else detail


def _get_mlflow_run_name(config: dict) -> Optional[str]:
    """Extract MLflow run name from MlflowConfig."""
    return config.get("TrainingConfig", {}).get("MlflowConfig", {}).get("MlflowRunName")


def _setup_mlflow_metrics_util(mlflow_arn: str, experiment_name: str):
    """Create an _MLflowMetricsUtil instance, or None on failure."""
    try:
        from sagemaker.train.common_utils.mlflow_metrics_util import _MLflowMetricsUtil

        return _MLflowMetricsUtil(
            tracking_uri=mlflow_arn,
            experiment_name=experiment_name,
        )
    except Exception as e:
        logger.debug("Failed to initialize MLflow metrics util: %s", e)
        return None


MTRL_METRIC_KEYS = (
    "rollout/reward/mean",
    "rollout/turns/mean",
    "training/total_tokens",
    "training/num_trajectories",
)


def _get_step_metrics(
    metrics_util,
    run_name: Optional[str],
    run_id: Optional[str],
    metric_keys: tuple[str, ...] = MTRL_METRIC_KEYS,
    mlflow_arn: Optional[str] = None,
    _rest_cache: Optional[dict] = None,
) -> list[dict]:
    """Fetch per-step metrics from MLflow for the given metric keys.

    Tries ``metrics_util`` first; falls back to a direct mlflow client
    when ``run_id`` and ``mlflow_arn`` are available.

    Args:
        metrics_util: An ``_MLflowMetricsUtil`` instance (may be None).
        run_name: MLflow run name filter.
        run_id: MLflow run ID.
        metric_keys: Tuple of MLflow metric names to retrieve.
        mlflow_arn: MLflow tracking server ARN for direct client fallback.

    Returns:
        List of dicts, one per step.  Each dict has a ``"step"`` key plus
        one key per metric name with its value (or ``None`` if missing).
    """
    # Resolve which run ID to use
    rid = run_id
    if metrics_util is not None and not rid:
        try:
            run_ids = metrics_util._get_run_ids(run_id, run_name)
            rid = run_ids[0] if run_ids else None
        except Exception:
            pass
    if not rid:
        logger.debug("No MLflow run_id available, cannot fetch metrics.")
        return []

    # Try metrics_util first
    if metrics_util is not None:
        try:
            step_data: dict[int, dict[str, float]] = {}
            for metric_name in metric_keys:
                try:
                    for pt in metrics_util.get_metric_history(rid, metric_name):
                        step_data.setdefault(pt["step"], {})[metric_name] = pt["value"]
                except Exception:
                    pass
            if step_data:
                return [
                    {"step": step, **{k: step_data[step].get(k) for k in metric_keys}}
                    for step in sorted(step_data)
                ]
        except Exception:
            pass

    # Fall back to MLflow REST API via presigned URL
    if mlflow_arn:
        try:
            import requests

            # _rest_cache is a mutable dict passed by the caller to persist
            # the session across calls.  Keys: session, base_url, ts.
            if _rest_cache is None:
                _rest_cache = {}

            def _init_rest_session():
                url = _get_mlflow_presigned_url(mlflow_arn, None)
                if not url:
                    return False
                sess = requests.Session()
                resp = sess.get(url.split("#")[0], timeout=10)
                _rest_cache["session"] = sess
                _rest_cache["base_url"] = resp.url.split("#")[0].split("?")[0].rstrip("/")
                _rest_cache["ts"] = time.time()
                return True

            # Init or refresh session (every 240s)
            if "session" not in _rest_cache or (time.time() - _rest_cache.get("ts", 0)) > 240:
                if not _init_rest_session():
                    return []

            sess = _rest_cache["session"]
            base_url = _rest_cache["base_url"]
            step_data: dict[int, dict[str, float]] = {}
            for metric_name in metric_keys:
                try:
                    resp = sess.get(
                        f"{base_url}/api/2.0/mlflow/metrics/get-history",
                        params={"run_id": rid, "metric_key": metric_name},
                        timeout=10,
                    )
                    if resp.status_code == 403:
                        # Session expired, refresh once and retry
                        if _init_rest_session():
                            sess = _rest_cache["session"]
                            base_url = _rest_cache["base_url"]
                            resp = sess.get(
                                f"{base_url}/api/2.0/mlflow/metrics/get-history",
                                params={"run_id": rid, "metric_key": metric_name},
                                timeout=10,
                            )
                    if resp.status_code == 200:
                        for m in resp.json().get("metrics", []):
                            step_data.setdefault(int(m["step"]), {})[metric_name] = float(m["value"])
                except Exception:
                    pass
            if step_data:
                return [
                    {"step": step, **{k: step_data[step].get(k) for k in metric_keys}}
                    for step in sorted(step_data)
                ]
        except Exception as e:
            logger.warning("MLflow REST API fallback failed: %s", e)

    return []


def _calculate_job_progress(
    progress_info: dict,
    metrics_util,
    mlflow_run_name: Optional[str],
    mlflow_run_id: Optional[str],
) -> Tuple[Optional[float], str]:
    """Calculate training progress percentage and text from ProgressInfo dict.

    Supports two formats:
    - Epoch-based: ``{MaxEpoch, StepsPerEpoch, CurrentEpoch, CurrentStep}``
    - Step-only: ``{MaxSteps, CurrentStep}``

    Args:
        progress_info: ProgressInfo dict.
        metrics_util: Optional _MLflowMetricsUtil instance.
        mlflow_run_name: MLflow run name.
        mlflow_run_id: MLflow run id.

    Returns:
        Tuple of (progress_pct, progress_text).
    """
    current_step = progress_info.get("CurrentStep", 0)
    max_epoch = progress_info.get("MaxEpoch", 0)
    total_steps = progress_info.get("StepsPerEpoch", 0)
    max_steps = progress_info.get("MaxSteps", 0)

    if max_epoch and total_steps:
        # Epoch-based progress
        current_epoch = progress_info.get("CurrentEpoch", 0)
        progress_pct = max(0, ((current_epoch - 1) * total_steps + current_step - 1)) / (max_epoch * total_steps) * 100
        progress_text = f"\n- Epoch {current_epoch}/{max_epoch}, Step {current_step}/{total_steps}"
    elif max_steps:
        # Step-only progress
        progress_pct = max(0, current_step - 1) / max_steps * 100
        progress_text = f"\n- Step {current_step}/{max_steps}"
    else:
        return None, ""

    if metrics_util and mlflow_run_name:
        try:
            loss = metrics_util._get_most_recent_total_loss(
                run_name=mlflow_run_name, run_id=mlflow_run_id
            )
            if loss is not None:
                progress_text += f"\n- loss: {loss:.7f}"
        except Exception:
            pass

    return progress_pct, progress_text


def _get_mlflow_presigned_url(mlflow_arn: str, experiment_name: Optional[str] = None,
                              experiment_id: Optional[str] = None,
                              run_id: Optional[str] = None) -> Optional[str]:
    """Get presigned MLflow URL. Handles both mlflow-app and mlflow-tracking-server ARNs."""
    try:
        import boto3

        # MLflow tracking servers are in prod, use default endpoint
        sm_client = boto3.Session().client("sagemaker")

        if ":mlflow-tracking-server/" in mlflow_arn:
            server_name = mlflow_arn.rsplit("/", 1)[-1]
            response = sm_client.create_presigned_mlflow_tracking_server_url(
                TrackingServerName=server_name
            )
        else:
            response = sm_client.create_presigned_mlflow_app_url(Arn=mlflow_arn)

        base_url = response.get("AuthorizedUrl")

        from sagemaker.train.common_utils.mlflow_url_utils import _build_mlflow_deep_link

        if experiment_id:
            return _build_mlflow_deep_link(base_url, experiment_id, run_id)

        if experiment_name:
            try:
                import mlflow
                from mlflow.tracking import MlflowClient

                tracking_url = base_url.split("?")[0].replace("/auth", "")
                mlflow.set_tracking_uri(tracking_url)
                client = MlflowClient(tracking_uri=tracking_url)
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment:
                    return _build_mlflow_deep_link(base_url, experiment.experiment_id)
            except ImportError:
                pass
            except Exception:
                pass

        return base_url
    except Exception as e:
        logger.warning("Failed to get MLflow presigned URL: %s", e)
        return None


def _calculate_transition_duration(trans) -> Tuple[str, str]:
    duration = ""
    check = ""
    if trans.start_time and trans.end_time:
        if not _is_unassigned_attribute(trans.end_time):
            duration = f"{(trans.end_time - trans.start_time).total_seconds():.1f}s"
            check = "✓"
    elif trans.start_time:
        duration = "Running..."
        check = "⋯"
    return duration, check


def wait(
    job: Job,
    poll: int = 5,
    timeout: Optional[int] = 3000,
    log_group: Optional[str] = None,
    description: Optional[str] = None,
    max_log_lines: int = MAX_LOG_LINES,
) -> None:
    """Wait for a Job resource to reach terminal status with rich progress display.

    Shows job name/ARN, status, secondary status, transitions, and MLflow link.

    Args:
        job: The sagemaker-core Job instance.
        poll: Polling interval in seconds.
        timeout: Maximum wait time in seconds.
        log_group: CloudWatch log group name for log links. Defaults to
            ``"/aws/sagemaker/Job/<JobCategory>"``. Set to ``False`` to disable.
        description: Optional job description shown in the status display.
        max_log_lines: Maximum number of recent log lines to display.
            Defaults to ``MAX_LOG_LINES`` (20).

    Raises:
        FailedStatusError: If the job fails.
        TimeoutExceededError: If the timeout is exceeded.
    """
    try:
        start_time = time.time()

        if log_group is None:
            job_category = job.job_category
            if job_category and not _is_unassigned_attribute(job_category):
                log_group = f"{DEFAULT_LOG_GROUP_PREFIX}/{job_category}"

        is_jupyter = _is_jupyter_environment()

        if is_jupyter:
            _wait_jupyter(job, poll, timeout, start_time, log_group, description, max_log_lines)
        else:
            _wait_terminal(job, poll, timeout, start_time, log_group, description)

    except (FailedStatusError, TimeoutExceededError):
        raise
    except Exception as e:
        raise RuntimeError(f"Job monitoring failed: {e}") from e


def _wait_jupyter(
    job: Job,
    poll: int,
    timeout: Optional[int],
    start_time: float,
    log_group: Optional[str] = None,
    description: Optional[str] = None,
    max_log_lines: int = MAX_LOG_LINES,
) -> None:
    from IPython.display import clear_output
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text

    with _suppress_info_logging():
        console = Console(force_jupyter=True)

        mlflow_arn = None
        mlflow_experiment_name = None
        mlflow_experiment_id = None
        mlflow_run_id = None
        mlflow_run_name = None
        mlflow_link_cache = {"url": None, "timestamp": 0, "error": None}
        config_parsed = False
        metrics_util = None
        progress_started = False
        cached_mtrl_rows = None
        metrics_fetch_needed = True
        rest_cache = {}

        log_handler = _create_log_stream_handler(log_group, job.job_name) if log_group else None
        log_buf: collections.deque[str] = collections.deque(maxlen=max_log_lines)

        def _ensure_config_parsed():
            nonlocal mlflow_arn, mlflow_experiment_name, mlflow_experiment_id
            nonlocal mlflow_run_id, mlflow_run_name, config_parsed, metrics_util
            if config_parsed and mlflow_experiment_id and mlflow_run_id:
                return
            config = _parse_config_document(job)
            mlflow_arn = _get_mlflow_arn_from_config(config)
            mlflow_experiment_name = _get_mlflow_experiment_name(config)
            mlflow_experiment_id, mlflow_run_id = _get_mlflow_output_details(config)
            mlflow_run_name = _get_mlflow_run_name(config)
            # Fall back to ServiceOutput for experiment/run name
            svc_details = config.get("ServiceOutput", {}).get("MlflowDetails", {})
            if not mlflow_experiment_name:
                mlflow_experiment_name = svc_details.get("ExperimentName")
            if not mlflow_run_name:
                mlflow_run_name = svc_details.get("RunName")
            config_parsed = bool(mlflow_arn) or bool(mlflow_experiment_name)
            if mlflow_arn and mlflow_experiment_name and metrics_util is None:
                metrics_util = _setup_mlflow_metrics_util(mlflow_arn, mlflow_experiment_name)

        def _refresh_mlflow_session():
            """Create a presigned URL and init REST session."""
            try:
                import requests
                presigned = _get_mlflow_presigned_url(mlflow_arn, None)
                if not presigned:
                    return
                sess = requests.Session()
                resp = sess.get(presigned.split("#")[0], timeout=10)
                rest_cache["session"] = sess
                rest_cache["base_url"] = resp.url.split("#")[0].split("?")[0].rstrip("/")
                rest_cache["ts"] = time.time()
            except Exception:
                pass

        def get_cached_mlflow_url():
            now = time.time()
            # Refresh REST session every 240s
            if mlflow_arn and ("session" not in rest_cache or (now - rest_cache.get("ts", 0)) > 240):
                _refresh_mlflow_session()
            # Refresh link URL every 30s so it stays fresh for clicking
            if mlflow_arn and (mlflow_link_cache["url"] is None or (now - mlflow_link_cache["timestamp"]) > 30):
                mlflow_link_cache["url"] = _get_mlflow_presigned_url(
                    mlflow_arn, mlflow_experiment_name,
                    experiment_id=mlflow_experiment_id, run_id=mlflow_run_id,
                )
                mlflow_link_cache["timestamp"] = now
            return mlflow_link_cache["url"]

        last_status = None
        last_secondary = None
        iteration = 0

        _ensure_config_parsed()

        while True:
            iteration += 1
            time.sleep(0.5)
            if iteration >= poll * 2:
                job.refresh()
                _ensure_config_parsed()
                metrics_fetch_needed = True
                if log_handler:
                    _drain_log_events(log_handler, log_buf)
                iteration = 0

            status = job.job_status
            secondary_status = job.secondary_status
            if _is_unassigned_attribute(secondary_status):
                secondary_status = ""
            elapsed = time.time() - start_time

            should_render = (
                status != last_status
                or secondary_status != last_secondary
                or iteration % 4 == 0
            )
            if not should_render:
                continue

            last_status = status
            last_secondary = secondary_status

            clear_output(wait=True)

            # Parse config early for header and progress
            config = _parse_config_document(job)
            progress_info = _get_progress_info(config)

            # Header
            header_table = Table(show_header=False, box=None, padding=(0, 1))
            header_table.add_column("Property", style="cyan bold", width=20)
            header_table.add_column("Value", style="dim", overflow="fold")

            header_table.add_row("Job Name", f"[bold green]{job.job_name}[/bold green]")
            job_arn = job.job_arn if not _is_unassigned_attribute(job.job_arn) else ""
            header_table.add_row("Job ARN", f"[dim]{job_arn}[/dim]")
            if description or progress_info:
                enriched = _build_description_with_progress(description, progress_info)
                if enriched:
                    header_table.add_row("Description", enriched)

            # Links
            links_row1 = []
            links_row2 = []
            try:
                from sagemaker.train.common_utils.metrics_visualizer import (
                    _is_in_studio, _get_studio_base_url,
                )
                if _is_in_studio() and job_arn:
                    region = _parse_region_from_arn(job_arn)
                    if region:
                        base = _get_studio_base_url(region)
                        if base:
                            studio_url = f"{base}/jobs/{job.job_name}"
                            links_row1.append(
                                f"[bright_blue underline][link={studio_url}]🔗 Job (Studio)[/link][/bright_blue underline]"
                            )
            except Exception:
                pass
            if log_group and job_arn:
                cw_url = _get_cloudwatch_logs_url(job_arn, job.job_name, log_group)
                if cw_url:
                    links_row2.append(
                        f"[bright_blue underline][link={cw_url}]🔗 CloudWatch Logs[/link][/bright_blue underline]"
                    )
            if mlflow_experiment_id:
                cached_url = get_cached_mlflow_url()
                if cached_url:
                    links_row2.append(
                        f"[bright_blue underline][link={cached_url}]🔗 MLflow Experiment[/link][/bright_blue underline]"
                    )
            if links_row1:
                header_table.add_row("Links", " | ".join(links_row1))
            if links_row2:
                header_table.add_row("" if links_row1 else "Links", " | ".join(links_row2))

            # Status
            status_table = Table(show_header=False, box=None, padding=(0, 1))
            status_table.add_column("Property", style="cyan bold", width=20)
            status_table.add_column("Value", style="dim")

            status_table.add_row("Job Status", f"[bold][orange3]{status}[/][/]")
            if secondary_status:
                status_table.add_row("Secondary Status", f"[bold yellow]{secondary_status}[/bold yellow]")
            status_table.add_row("Elapsed Time", f"[bold bright_red]{elapsed:.1f}s[/bold bright_red]")

            failure_reason = job.failure_reason
            if failure_reason and not _is_unassigned_attribute(failure_reason):
                status_table.add_row("Failure Reason", f"[bright_red]{failure_reason}[/bright_red]")

            # Training progress from ServiceOutput.ProgressInfo
            training_progress_pct = None
            training_progress_text = ""
            if secondary_status == "Training" and progress_info:
                if not progress_started:
                    progress_started = True
                    time.sleep(poll)
                    job.refresh()
                    _ensure_config_parsed()
                    config = _parse_config_document(job)
                    progress_info = _get_progress_info(config)

                if progress_info:
                    training_progress_pct, training_progress_text = _calculate_job_progress(
                        progress_info, metrics_util, mlflow_run_name, mlflow_run_id,
                    )

            # Transitions
            transitions_table = None
            transitions = job.secondary_status_transitions
            if transitions and not _is_unassigned_attribute(transitions):
                from rich.box import SIMPLE

                transitions_table = Table(
                    show_header=True, header_style="bold magenta", box=SIMPLE, padding=(0, 1)
                )
                transitions_table.add_column("", style="green", width=2)
                transitions_table.add_column("Step", style="cyan", width=15)
                transitions_table.add_column("Details", style="orange3", width=35)
                transitions_table.add_column("Duration", style="green", width=12)

                # Find the last Training transition (the one actually in progress)
                last_training_idx = None
                for i, trans in enumerate(transitions):
                    if trans.status == "Training":
                        last_training_idx = i

                for i, trans in enumerate(transitions):
                    duration, check = _calculate_transition_duration(trans)
                    msg = trans.status_message if not _is_unassigned_attribute(trans.status_message) else ""
                    if trans.status == "Training" and i == last_training_idx and training_progress_pct is not None:
                        bar = (
                            f"[green][{'█' * int(training_progress_pct / 5)}"
                            f"{'░' * (20 - int(training_progress_pct / 5))}][/green] "
                            f"{training_progress_pct:.1f}% {training_progress_text}"
                        )
                        if msg:
                            bar += f"\n{msg}"
                        transitions_table.add_row(check, trans.status, bar, duration)
                    else:
                        transitions_table.add_row(check, trans.status, msg or "", duration)

            # Training metrics table (live during training + on completion)
            metrics_table = None
            if secondary_status == "Training" or status in TERMINAL_STATUSES:
                # Ensure metrics_util is available
                if metrics_util is None and mlflow_arn and mlflow_experiment_name:
                    metrics_util = _setup_mlflow_metrics_util(mlflow_arn, mlflow_experiment_name)
                # Only fetch metrics on refresh cycles, not every render
                if metrics_fetch_needed:
                    with _suppress_info_logging():
                        cached_mtrl_rows = _get_step_metrics(
                            metrics_util, mlflow_run_name, mlflow_run_id,
                            mlflow_arn=mlflow_arn,
                            _rest_cache=rest_cache,
                        )
                    metrics_fetch_needed = False
                try:
                    from rich.box import SIMPLE

                    if cached_mtrl_rows:
                        metrics_table = Table(
                            show_header=True, header_style="bold magenta",
                            box=SIMPLE, padding=(0, 1),
                        )
                        metrics_table.add_column("Step", style="cyan", width=6, justify="right")
                        metric_keys = [k for k in cached_mtrl_rows[0] if k != "step"]
                        for k in metric_keys:
                            parts = k.split("/")
                            col_name = "/".join(parts[-2:]) if len(parts) > 1 else parts[0]
                            col_name = col_name.replace("_", " ").title()
                            metrics_table.add_column(col_name, style="white", width=14, justify="right")
                        for r in cached_mtrl_rows:
                            vals = []
                            for k in metric_keys:
                                v = r.get(k)
                                if v is None:
                                    vals.append("—")
                                elif isinstance(v, float) and v != int(v):
                                    vals.append(f"{v:.4f}")
                                else:
                                    vals.append(str(int(v)))
                            metrics_table.add_row(str(r["step"]), *vals, style="yellow")
                except Exception as e:
                    logger.warning("Metrics table render error: %s", e)

            # Assemble panel
            if status in TERMINAL_STATUSES and log_handler:
                _drain_log_events(log_handler, log_buf)
            parts = [header_table, Text(""), status_table]
            if transitions_table:
                parts += [Text(""), Text("Status Transitions", style="bold magenta"), transitions_table]
            rollout = _get_rollout_info(config)
            if rollout:
                completed, total = rollout
                rollout_pct = completed / total * 100
                filled = int(rollout_pct / 5)
                rollout_bar = (
                    f"[green][{'█' * filled}{'░' * (20 - filled)}][/green] "
                    f"{rollout_pct:.1f}% ({completed}/{total})"
                )
                parts += [Text(""), Text("Rollouts", style="bold magenta"), Text.from_markup(rollout_bar)]
            if metrics_table:
                parts += [Text(""), Text("Training Metrics", style="bold magenta"), metrics_table]
            if log_buf:
                parts += [
                    Text(""),
                    Text(f"Recent Logs (last {len(log_buf)})", style="bold magenta"),
                    Text("\n".join(log_buf), style="dim"),
                ]
            combined = Group(*parts)

            panel_width = 80
            if console.width and not _is_unassigned_attribute(console.width):
                panel_width = int(console.width * 0.8)
            console.print(
                Panel(
                    combined,
                    title="[bold bright_blue]Agentic RFT Job Status[/bold bright_blue]",
                    border_style="orange3",
                    width=panel_width,
                )
            )

            if status in TERMINAL_STATUSES:
                return

            if status == "Failed" or (failure_reason and not _is_unassigned_attribute(failure_reason)):
                raise FailedStatusError(resource_type="Job", status=status, reason=failure_reason)

            if timeout and elapsed >= timeout:
                raise TimeoutExceededError(resource_type="Job", status=status)


def _create_log_stream_handler(
    log_group: str,
    job_name: str,
    instance_count: int = 1,
) -> Optional["MultiLogStreamHandler"]:
    """Create a MultiLogStreamHandler for streaming CloudWatch logs.

    Args:
        log_group: CloudWatch log group name.
        job_name: Job name used as the log stream name prefix.
        instance_count: Expected number of log streams (one per instance).

    Returns:
        A MultiLogStreamHandler, or None if the import fails.
    """
    try:
        from sagemaker.core.utils.logs import MultiLogStreamHandler

        return MultiLogStreamHandler(
            log_group_name=log_group,
            log_stream_name_prefix=job_name,
            expected_stream_count=instance_count,
        )
    except Exception as e:
        logger.debug("Failed to create log stream handler: %s", e)
        return None


def _flush_log_events(handler: "MultiLogStreamHandler") -> None:
    """Print all new log events from the handler."""
    try:
        for stream_name, event in handler.get_latest_log_events():
            message = event.get("message", "").rstrip()
            print(message)
    except Exception as e:
        logger.debug("Error reading log events: %s", e)


def _drain_log_events(
    handler: "MultiLogStreamHandler",
    buf: collections.deque,
) -> None:
    """Drain new log events from the handler into a rolling deque buffer."""
    try:
        for stream_name, event in handler.get_latest_log_events():
            message = event.get("message", "").rstrip()
            if message:
                buf.append(message)
    except Exception as e:
        logger.debug("Error reading log events: %s", e)


def _wait_terminal(
    job: Job,
    poll: int,
    timeout: Optional[int],
    start_time: float,
    log_group: Optional[str] = None,
    description: Optional[str] = None,
) -> None:
    print(f"\nJob Name: {job.job_name}")
    if description:
        print(f"Description: {description}")
    if log_group:
        job_arn = job.job_arn if not _is_unassigned_attribute(job.job_arn) else ""
        if job_arn:
            cw_url = _get_cloudwatch_logs_url(job_arn, job.job_name, log_group)
            if cw_url:
                print(f"CloudWatch Logs: {cw_url}")

    log_handler = None
    if log_group:
        log_handler = _create_log_stream_handler(log_group, job.job_name)

    mlflow_arn = None
    metrics_util = None
    mlflow_run_name = None
    mlflow_run_id = None
    progress_started = False
    description_enriched = False

    while True:
        time.sleep(poll)
        job.refresh()

        # Stream new CloudWatch log events
        if log_handler:
            _flush_log_events(log_handler)

        config = _parse_config_document(job)

        # Lazily resolve MLflow ARN and metrics util
        if metrics_util is None:
            if mlflow_arn is None:
                mlflow_arn = _get_mlflow_arn_from_config(config) or False
            if mlflow_arn:
                svc_details = config.get("ServiceOutput", {}).get("MlflowDetails", {})
                exp_name = _get_mlflow_experiment_name(config) or svc_details.get("ExperimentName")
                mlflow_run_name = _get_mlflow_run_name(config) or svc_details.get("RunName")
                if exp_name:
                    metrics_util = _setup_mlflow_metrics_util(mlflow_arn, exp_name)

        _, mlflow_run_id = _get_mlflow_output_details(config)

        status = job.job_status
        secondary_status = job.secondary_status
        if _is_unassigned_attribute(secondary_status):
            secondary_status = ""
        elapsed = time.time() - start_time

        # Progress
        progress_info = _get_progress_info(config)
        if not description_enriched and progress_info:
            enriched = _build_description_with_progress(description, progress_info)
            if enriched:
                print(f"Description: {enriched}")
            description_enriched = True
        progress_pct = None
        progress_text = ""
        if secondary_status == "Training" and progress_info:
            if not progress_started:
                progress_started = True
                time.sleep(poll)
                job.refresh()
                config = _parse_config_document(job)
                progress_info = _get_progress_info(config)
            if progress_info:
                progress_pct, progress_text = _calculate_job_progress(
                    progress_info, metrics_util, mlflow_run_name, mlflow_run_id,
                )

        transitions = job.secondary_status_transitions
        if transitions and not _is_unassigned_attribute(transitions):
            print("\n--------------------------------------\n")
            print("Status Transitions:")
            # Find the last Training transition (the one actually in progress)
            last_training_idx = None
            for i, trans in enumerate(transitions):
                if trans.status == "Training":
                    last_training_idx = i
            for i, trans in enumerate(transitions):
                duration, check = _calculate_transition_duration(trans)
                msg = trans.status_message if not _is_unassigned_attribute(trans.status_message) else ""
                step_msg = f"  {check} {trans.status}: {msg} ({duration})"
                if trans.status == "Training" and i == last_training_idx and progress_pct is not None:
                    step_msg += f" - {progress_pct:.1f}%{progress_text.replace(chr(10), ', ')}"
                print(step_msg)

        rollout = _get_rollout_info(config)
        if rollout:
            completed, total = rollout
            rollout_pct = completed / total * 100
            filled = int(rollout_pct / 5)
            print(f"  Rollouts: [{'█' * filled}{'░' * (20 - filled)}] {rollout_pct:.1f}% ({completed}/{total})")

        print(f"\nStatus: {status} - {secondary_status} (Elapsed: {elapsed:.1f}s)")

        if status in TERMINAL_STATUSES:
            # Final flush of any remaining log events
            if log_handler:
                _flush_log_events(log_handler)
            if status == "Completed" and mlflow_arn:
                exp_id, run_id = _get_mlflow_output_details(config)
                mlflow_url = _get_mlflow_presigned_url(
                    mlflow_arn, _get_mlflow_experiment_name(config),
                    experiment_id=exp_id, run_id=run_id,
                )
                if mlflow_url:
                    print(f"\n✓ Job completed! View metrics in MLflow: {mlflow_url}")
                if metrics_util or (mlflow_arn and mlflow_run_id):
                    try:
                        from sagemaker.train.agent_rft_job import AgentRFTJob
                        mtrl_rows = _get_step_metrics(
                            metrics_util, mlflow_run_name, mlflow_run_id,
                            mlflow_arn=mlflow_arn,
                        )
                        if mtrl_rows:
                            AgentRFTJob._print_metrics_table(mtrl_rows)
                    except Exception:
                        pass
            return

        failure_reason = job.failure_reason
        if status == "Failed" or (failure_reason and not _is_unassigned_attribute(failure_reason)):
            raise FailedStatusError(resource_type="Job", status=status, reason=failure_reason)

        if timeout and elapsed >= timeout:
            raise TimeoutExceededError(resource_type="Job", status=status)
