"""Training job wait utilities for SageMaker.

This module provides functionality to wait for training jobs to complete
with progress tracking and MLflow integration.
"""

import logging
import time
from contextlib import contextmanager
from typing import Optional, Tuple

from sagemaker.core.resources import TrainingJob
from sagemaker.core.utils.exceptions import FailedStatusError, TimeoutExceededError

from sagemaker.train.common_utils.mlflow_metrics_util import _MLflowMetricsUtil


@contextmanager
def _suppress_info_logging():
    """Context manager to temporarily suppress INFO level logging."""
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def _setup_mlflow_integration(training_job: TrainingJob) -> Tuple[
    Optional[str], Optional[_MLflowMetricsUtil], Optional[str]]:
    """Setup MLflow integration for training job monitoring.

    Args:
        training_job (TrainingJob): The training job to setup MLflow for.

    Returns:
        Tuple containing mlflow_url, mlflow_server, metrics_util, and mlflow_run_name.
    """
    try:
        import boto3

        sm_client = boto3.client('sagemaker')
        mlflow_arn = training_job.mlflow_config.mlflow_resource_arn

        response = sm_client.create_presigned_mlflow_app_url(
            Arn=mlflow_arn
        )
        mlflow_url = response.get('AuthorizedUrl')
        mlflow_run_name = training_job.mlflow_config.mlflow_run_name

        metrics_util = _MLflowMetricsUtil(
            tracking_uri=training_job.mlflow_config.mlflow_resource_arn,
            experiment_name=training_job.mlflow_config.mlflow_experiment_name
        )

        return mlflow_url, metrics_util, mlflow_run_name

    except Exception:
        return None, None, None


def _is_jupyter_environment() -> bool:
    """Check if running in Jupyter environment.

    Returns:
        bool: True if running in Jupyter, False otherwise.
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        return ipython is not None and 'IPKernelApp' in ipython.config
    except ImportError:
        return False


def _is_unassigned_attribute(attr) -> bool:
    """Check if an attribute is unassigned.

    Args:
        attr: The attribute to check.

    Returns:
        bool: True if the attribute is unassigned, False otherwise.
    """
    return hasattr(attr, '__class__') and 'Unassigned' in attr.__class__.__name__


def _calculate_training_progress(progress_info, metrics_util: Optional[_MLflowMetricsUtil],
                                 mlflow_run_name: Optional[str], training_job: TrainingJob) -> Tuple[
    Optional[float], str]:
    """Calculate training progress percentage and text.

    Args:
        progress_info: Training job progress information.
        metrics_util: MLflow metrics utility instance.
        mlflow_run_name: MLflow run name.
        training_job: Training job instance.

    Returns:
        Tuple of progress percentage and progress text.
    """
    if not progress_info or _is_unassigned_attribute(progress_info):
        return None, ""

    if (_is_unassigned_attribute(progress_info.max_epoch) or
            _is_unassigned_attribute(progress_info.total_step_count_per_epoch) or
            _is_unassigned_attribute(progress_info.current_epoch) or
            not progress_info.max_epoch or not progress_info.total_step_count_per_epoch):
        return None, ""

    current_epoch = progress_info.current_epoch if progress_info.current_epoch is not None else 0
    current_step = progress_info.current_step if progress_info.current_step is not None else 0
    max_epoch = progress_info.max_epoch
    total_steps = progress_info.total_step_count_per_epoch

    progress_pct = ((current_epoch - 1) * total_steps + current_step) / (max_epoch * total_steps) * 100

    progress_text = f"\n- Epoch {current_epoch}/{max_epoch}, Step {current_step}/{total_steps}"

    if metrics_util and mlflow_run_name:
        try:
            loss_metrics = metrics_util._get_most_recent_total_loss(
                run_name=mlflow_run_name,
                run_id=training_job.mlflow_details.mlflow_run_id
            )
            progress_text += f"\n- loss: {loss_metrics:.7f}"
        except Exception:
            pass

    return progress_pct, progress_text


def _calculate_transition_duration(trans) -> Tuple[str, str]:
    """Calculate duration and check mark for a transition.

    Args:
        trans: Transition object.

    Returns:
        Tuple of duration string and check mark.
    """
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
        training_job: TrainingJob,
        poll: int = 5,
        timeout: Optional[int] = 3000
) -> None:
    """Wait for training job to complete with progress tracking.

    Args:
        training_job (TrainingJob): The SageMaker training job to monitor.
        poll (int): Polling interval in seconds. Defaults to 3.
        timeout (Optional[int]): Maximum wait time in seconds. Defaults to None.

    Raises:
        FailedStatusError: If the training job fails.
        TimeoutExceededError: If the timeout is exceeded.
    """
    try:
        start_time = time.time()
        progress_started = False

        # Setup MLflow integration
        mlflow_url, metrics_util, mlflow_run_name = _setup_mlflow_integration(training_job)

        is_jupyter = _is_jupyter_environment()

        if is_jupyter:
            from IPython.display import clear_output
            from rich.console import Console
            from rich.table import Table
            from rich.panel import Panel
            from rich.text import Text
            from rich.console import Group
            with _suppress_info_logging():
                console = Console(force_jupyter=True)

                iteration = 0
                while True:
                    iteration += 1
                    time.sleep(1)
                    if iteration == poll:
                        training_job.refresh()
                        iteration = 0
                    clear_output(wait=True)

                    status = training_job.training_job_status
                    secondary_status = training_job.secondary_status
                    elapsed = time.time() - start_time

                    # Header section with training job name and MLFlow URL
                    header_table = Table(show_header=False, box=None, padding=(0, 1))
                    header_table.add_column("Property", style="cyan bold", width=20)
                    header_table.add_column("Value", style="white")
                    header_table.add_row("TrainingJob Name", f"[bold green]{training_job.training_job_name}[/bold green]")
                    if mlflow_url:
                        header_table.add_row("MLFlow URL",
                                             f"[link={mlflow_url}][bold bright_blue underline]{mlflow_run_name}(link valid for 5 mins)[/bright_blue bold underline][/link]")

                    status_table = Table(show_header=False, box=None, padding=(0, 1))
                    status_table.add_column("Property", style="cyan bold", width=20)
                    status_table.add_column("Value", style="white")

                    status_table.add_row("Job Status", f"[bold][orange3]{status}[/][/]")
                    status_table.add_row("Secondary Status", f"[bold yellow]{secondary_status}[/bold yellow]")
                    status_table.add_row("Elapsed Time", f"[bold bright_red]{elapsed:.1f}s[/bold bright_red]")

                    failure_reason = training_job.failure_reason
                    if failure_reason and not _is_unassigned_attribute(failure_reason):
                        status_table.add_row("Failure Reason", f"[bright_red]{failure_reason}[/bright_red]")

                    # Calculate training progress
                    training_progress_pct = None
                    training_progress_text = ""
                    if secondary_status == "Training" and training_job.progress_info:
                        if not progress_started:
                            progress_started = True
                            time.sleep(poll)
                            training_job.refresh()

                        training_progress_pct, training_progress_text = _calculate_training_progress(
                            training_job.progress_info, metrics_util, mlflow_run_name, training_job
                        )

                    # Build transitions table if available
                    transitions_table = None
                    if training_job.secondary_status_transitions:
                        from rich.box import SIMPLE
                        transitions_table = Table(show_header=True, header_style="bold magenta", box=SIMPLE, padding=(0, 1))
                        transitions_table.add_column("", style="green", width=2)
                        transitions_table.add_column("Step", style="cyan", width=15)
                        transitions_table.add_column("Details", style="orange3", width=35)
                        transitions_table.add_column("Duration", style="green", width=12)

                        for trans in training_job.secondary_status_transitions:
                            duration, check = _calculate_transition_duration(trans)

                            # Add progress bar for Training step
                            if trans.status == "Training" and training_progress_pct is not None:
                                bar = f"[green][{'█' * int(training_progress_pct / 5)}{'░' * (20 - int(training_progress_pct / 5))}][/green] {training_progress_pct:.1f}% {training_progress_text}"
                                transitions_table.add_row(check, trans.status, bar, duration)
                            else:
                                transitions_table.add_row(check, trans.status, trans.status_message or "", duration)

                    # Prepare metrics table for terminal states
                    metrics_table = None
                    if status in ["Completed", "Failed", "Stopped"]:
                        try:
                            steps_per_epoch = training_job.progress_info.total_step_count_per_epoch
                            loss_metrics_by_epoch = metrics_util._get_loss_metrics_by_epoch(run_name=mlflow_run_name,
                                                                                           steps_per_epoch=steps_per_epoch)
                            if loss_metrics_by_epoch:
                                metrics_table = Table(show_header=True, header_style="bold magenta", box=SIMPLE,
                                                      padding=(0, 1))
                                metrics_table.add_column("Epochs", style="cyan", width=8)
                                metrics_table.add_column("Loss Metrics", style="white")

                                for epoch, metrics in list(loss_metrics_by_epoch.items())[:-1]:
                                    metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
                                    metrics_table.add_row(str(epoch + 1), metrics_str, style="yellow")
                        except Exception:
                            pass

                    # Build combined group with metrics if available
                    if training_job.secondary_status_transitions:
                        if metrics_table:
                            combined = Group(header_table, Text(""), status_table, Text(""),
                                             Text("Status Transitions", style="bold magenta"), transitions_table, Text(""),
                                             Text("Loss Metrics by Epoch", style="bold magenta"), metrics_table)
                        else:
                            combined = Group(header_table, Text(""), status_table, Text(""),
                                             Text("Status Transitions", style="bold magenta"), transitions_table)
                    else:
                        if metrics_table:
                            combined = Group(header_table, Text(""), status_table, Text(""),
                                             Text("Loss Metrics by Epoch", style="bold magenta"), metrics_table)
                        else:
                            combined = Group(header_table, Text(""), status_table)

                    panel_width = 80
                    if console.width and not _is_unassigned_attribute(console.width):
                        panel_width = int(console.width * 0.8)
                    console.print(Panel(combined, title="[bold bright_blue]Training Job Status[/bold bright_blue]",
                                        border_style="orange3", width=panel_width))

                    if status in ["Completed", "Failed", "Stopped"]:
                        return

                    if status == "Failed" or (failure_reason and not _is_unassigned_attribute(failure_reason)):
                        raise FailedStatusError(resource_type="TrainingJob", status=status, reason=failure_reason)

                    if timeout and elapsed >= timeout:
                        raise TimeoutExceededError(resource_type="TrainingJob", status=status)

        else:
            print(f"\nTrainingJob Name: {training_job.training_job_name}")
            iteration = 0
            while True:
                iteration += 1
                time.sleep(poll)
                training_job.refresh()

                status = training_job.training_job_status
                secondary_status = training_job.secondary_status
                elapsed = time.time() - start_time

                # Show transitions with checkmarks
                if training_job.secondary_status_transitions:
                    print("\n--------------------------------------\n")
                    print("Status Transitions:")
                    for trans in training_job.secondary_status_transitions:
                        duration, check = _calculate_transition_duration(trans)

                        step_msg = f"  {check} {trans.status}: {trans.status_message or ''} ({duration})"

                        # Add progress for Training step
                        if trans.status == "Training" and secondary_status == "Training" and training_job.progress_info:
                            if not progress_started:
                                progress_started = True
                                time.sleep(20)
                                training_job.refresh()

                            progress_pct, progress_text = _calculate_training_progress(
                                training_job.progress_info, metrics_util, mlflow_run_name, training_job
                            )
                            if progress_pct is not None:
                                step_msg += f" - {progress_pct:.1f}%{progress_text.replace(chr(10), ', ')}"

                        print(step_msg)
                print(f"\nStatus: {status} - {secondary_status} (Elapsed: {elapsed:.1f}s)")

                if status in ["Completed", "Failed", "Stopped"]:
                    if status == "Completed":
                        if mlflow_url:
                            print(f"\n✓ Training completed! View metrics in MLflow: {mlflow_url}")
                        try:
                            steps_per_epoch = training_job.progress_info.total_step_count_per_epoch
                            loss_metrics_by_epoch = metrics_util._get_loss_metrics_by_epoch(run_name=mlflow_run_name,
                                                                                           steps_per_epoch=steps_per_epoch)
                            if loss_metrics_by_epoch:
                                print("\n------------ Loss Metrics by Epoch ------------")
                                for epoch, metrics in list(loss_metrics_by_epoch.items())[:-1]:
                                    print(f"Epoch {epoch}: {metrics}")
                                print("----------------------------------------------")
                        except Exception:
                            pass
                    return

                failure_reason = training_job.failure_reason
                if status == "Failed" or (failure_reason and not _is_unassigned_attribute(failure_reason)):
                    raise FailedStatusError(resource_type="TrainingJob", status=status, reason=failure_reason)

                if timeout and elapsed >= timeout:
                    raise TimeoutExceededError(resource_type="TrainingJob", status=status)


    except (FailedStatusError, TimeoutExceededError):
        raise
    except Exception as e:
        raise RuntimeError(f"Training job monitoring failed: {e}") from e
