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

"""AgentRFTJob — wrapper around sagemaker-core Job for AgentRFT job category."""
from __future__ import annotations

import json
import logging
from typing import Optional

from sagemaker.core.resources import Job
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

logger = logging.getLogger(__name__)

JOB_CATEGORY = "AgentRFT"
TERMINAL_STATUSES = ("Completed", "Failed", "Stopped")


class AgentRFTJob:
    """Wrapper around sagemaker-core Job for AgentRFT job category.

    Delegates lifecycle methods to the underlying Job and adds typed
    convenience properties by parsing the JobConfigDocument JSON string.

    Args:
        job: The sagemaker-core Job instance to wrap.
    """

    JOB_CATEGORY = JOB_CATEGORY

    def __init__(self, job: Job):
        self._job = job
        self._cached_config: dict | None = None
        self.description: str | None = None
        self.sagemaker_session = None

    @classmethod
    def from_job(cls, job: Job) -> AgentRFTJob:
        """Create an AgentRFTJob from a sagemaker-core Job instance."""
        return cls(job)

    # --- Delegated properties ---

    @property
    def job_name(self) -> str:
        return self._job.job_name

    @property
    def job_arn(self) -> str:
        return self._job.job_arn

    @property
    def job_status(self) -> str:
        return self._job.job_status

    @property
    def secondary_status(self) -> str:
        return self._job.secondary_status

    @property
    def secondary_status_transitions(self) -> list:
        return self._job.secondary_status_transitions

    @property
    def failure_reason(self) -> str | None:
        return self._job.failure_reason

    @property
    def creation_time(self):
        return self._job.creation_time

    @property
    def last_modified_time(self):
        return self._job.last_modified_time

    @property
    def end_time(self):
        return self._job.end_time

    # --- Delegated lifecycle methods ---

    def refresh(self):
        """Refresh job state from DescribeJob API."""
        self._job.refresh()
        self._cached_config = None

    def wait(self, poll: int = 5, timeout: Optional[int] = 3000, max_log_lines: int = 20):
        """Wait for job to reach terminal status.

        Args:
            poll: Seconds between polls.
            timeout: Maximum seconds to wait.
            max_log_lines: Maximum number of log lines to display. Defaults to 20.
        """
        from sagemaker.train.common_utils.job_wait import wait as _job_wait

        _job_wait(self._job, poll=poll, timeout=timeout, description=self.description, max_log_lines=max_log_lines)

    def stop(self):
        """Stop the job via StopJob API."""
        self._job.stop()

    def delete(self):
        """Delete the job via DeleteJob API."""
        self._job.delete()

    def wait_for_delete(self):
        """Wait for job deletion to complete."""
        self._job.wait_for_delete()

    # --- Parsed properties from JobConfigDocument ---

    def _parse_config_document(self) -> dict:
        """Parse JobConfigDocument JSON string into a dict. Cached after refresh."""
        if self._cached_config is None:
            doc = self._job.job_config_document
            self._cached_config = json.loads(doc) if doc else {}
        return self._cached_config

    @property
    def output_model_package_arn(self) -> str | None:
        """ARN of the output model package from ServiceOutput, or None."""
        config = self._parse_config_document()
        return config.get("ServiceOutput", {}).get("OutputModelPackageArn")

    @property
    def mlflow_details(self) -> dict | None:
        """MLflow experiment/run details from ServiceOutput.

        Returns dict with keys: ExperimentName, RunName, ExperimentId, RunId.
        """
        config = self._parse_config_document()
        return config.get("ServiceOutput", {}).get("MlflowDetails")

    def get_mlflow_url(self) -> str | None:
        """Generate a fresh presigned MLflow URL for this job's experiment/run.

        In Jupyter notebooks, also renders a clickable link.

        Returns:
            Presigned URL string, or None if MLflow is not configured.
        """
        from sagemaker.train.common_utils.job_wait import (
            _get_mlflow_arn_from_config,
            _get_mlflow_output_details,
            _get_mlflow_experiment_name,
            _get_mlflow_presigned_url,
            _is_jupyter_environment,
        )

        config = self._parse_config_document()
        mlflow_arn = _get_mlflow_arn_from_config(config)
        if not mlflow_arn:
            return None
        exp_id, run_id = _get_mlflow_output_details(config)
        exp_name = _get_mlflow_experiment_name(config)
        url = _get_mlflow_presigned_url(mlflow_arn, exp_name, experiment_id=exp_id, run_id=run_id)
        if url and _is_jupyter_environment():
            from IPython.display import display as ipy_display, HTML

            ipy_display(HTML(
                f'🔗 <a href="{url}" target="_blank">Open MLflow Experiment</a>'
            ))
        return url

    @property
    def s3_output_path(self) -> str | None:
        """S3 output path from OutputDataConfig."""
        config = self._parse_config_document()
        return config.get("OutputDataConfig", {}).get("S3OutputPath")

    @property
    def billable_token_usage(self) -> dict | None:
        """Billable token usage from ServiceOutput.

        Returns dict with keys: TrainTokenCount, PrefillTokenCount, SampleTokenCount.
        """
        config = self._parse_config_document()
        return config.get("ServiceOutput", {}).get("BillableTokenUsage")

    @property
    def progress_info(self) -> dict | None:
        """Training progress from ServiceOutput.

        Supports two formats:
        - Epoch-based: dict with MaxEpoch, StepsPerEpoch, CurrentEpoch, CurrentStep.
        - Step-only: dict with MaxSteps, CurrentStep.

        Returns None if not available.
        """
        config = self._parse_config_document()
        info = config.get("ServiceOutput", {}).get("ProgressInfo")
        if not info:
            return None
        has_epoch = info.get("MaxEpoch") and info.get("StepsPerEpoch")
        has_steps = info.get("MaxSteps")
        if not has_epoch and not has_steps:
            return None
        return info

    @property
    def training_config(self) -> dict | None:
        """Full TrainingConfig section from JobConfigDocument."""
        return self._parse_config_document().get("TrainingConfig")

    @property
    def agent_config(self) -> dict | None:
        """Full AgentConfig section from JobConfigDocument."""
        return self._parse_config_document().get("AgentConfig")

    # --- Training metrics ---

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION, func_name="AgentRFTJob.get_training_metrics"
    )
    def get_training_metrics(self) -> list[dict]:
        """Fetch per-step MTRL training metrics from MLflow.

        Retrieves ``rollout/reward/mean``, ``rollout/turns/mean``,
        ``training/total_tokens``, and ``training/num_trajectories``
        for each training step and prints a summary table.

        Returns:
            List of dicts, one per step, with keys ``step``,
            ``rollout/reward/mean``, ``rollout/turns/mean``,
            ``training/total_tokens``, and ``training/num_trajectories``.
        """
        from sagemaker.train.common_utils.job_wait import (
            _get_mlflow_arn_from_config,
            _get_mlflow_experiment_name,
            _get_mlflow_output_details,
            _get_mlflow_run_name,
            _get_step_metrics,
            _setup_mlflow_metrics_util,
            MTRL_METRIC_KEYS,
        )

        config = self._parse_config_document()
        mlflow_arn = _get_mlflow_arn_from_config(config)
        exp_name = _get_mlflow_experiment_name(config)
        svc_details = config.get("ServiceOutput", {}).get("MlflowDetails", {})
        if not exp_name:
            exp_name = svc_details.get("ExperimentName")
        if not mlflow_arn or not exp_name:
            logger.warning("MLflow not configured for this job.")
            return []

        util = _setup_mlflow_metrics_util(mlflow_arn, exp_name)

        run_name = _get_mlflow_run_name(config) or svc_details.get("RunName")
        _, run_id = _get_mlflow_output_details(config)
        rows = _get_step_metrics(util, run_name, run_id, MTRL_METRIC_KEYS, mlflow_arn=mlflow_arn)
        if rows:
            self._print_metrics_table(rows)
        return rows

    @staticmethod
    def _print_metrics_table(rows: list[dict]) -> None:
        """Print a formatted metrics table.

        Columns are derived from the dict keys (excluding ``step``).
        """
        if not rows:
            return
        metric_keys = [k for k in rows[0] if k != "step"]
        # Build column headers from metric names
        def _col_name(k: str) -> str:
            parts = k.split("/")
            label = "/".join(parts[-2:]) if len(parts) > 1 else parts[0]
            return label.replace("_", " ").title()

        col_names = [_col_name(k) for k in metric_keys]
        col_width = max(14, *(len(c) for c in col_names))
        header = f"{'Step':>6}" + "".join(f"  {c:>{col_width}}" for c in col_names)
        sep = "-" * len(header)
        print(f"\n{sep}\n  Training Metrics\n{sep}")
        print(header)
        print(sep)
        for r in rows:
            line = f"{r['step']:>6}"
            for k in metric_keys:
                v = r.get(k)
                if v is None:
                    line += f"  {'—':>{col_width}}"
                elif isinstance(v, float) and v != int(v):
                    line += f"  {v:>{col_width}.4f}"
                else:
                    line += f"  {int(v):>{col_width}}"
            print(line)
        print(sep)

    # --- Class methods ---

    @classmethod
    def get(cls, job_name: str, session=None) -> AgentRFTJob:
        """Attach to an existing AgentRFT job by name.

        Args:
            job_name: The name of the job.
            session: Optional boto3 session.

        Returns:
            AgentRFTJob wrapping the existing job.
        """
        job = Job.get(job_name=job_name, job_category=cls.JOB_CATEGORY, session=session)
        return cls.from_job(job)

    @classmethod
    def get_all(cls, session=None, **kwargs):
        """List all AgentRFT jobs.

        Delegates to Job.get_all with job_category pre-filled. Additional
        keyword arguments (e.g. creation_time_after, name_contains,
        sort_by, sort_order, status_equals) are forwarded.

        Args:
            session: Optional boto3 session.
            **kwargs: Additional filter arguments forwarded to Job.get_all.

        Yields:
            AgentRFTJob instances.
        """
        for job in Job.get_all(job_category=cls.JOB_CATEGORY, session=session, **kwargs):
            yield cls.from_job(job)
