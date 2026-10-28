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
"""Job resource class for DataPreparer.

Wraps CreateJob/DescribeJob/StopJob and implements wait() with a live
status panel for monitoring job progress in notebooks.
"""
from __future__ import absolute_import

import json
import logging
import re
import time
from datetime import datetime
from typing import TYPE_CHECKING, Dict, List, Optional

from botocore.exceptions import ClientError
from rich.panel import Panel

from sagemaker.core.helper.session_helper import Session

if TYPE_CHECKING:  # import cycle at runtime: the summary module reaches back here
    from sagemaker.ai_registry.data_preparer.auto_labeling.summary import LabelingSummary

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {"Completed", "Failed", "Stopped"}
_LOG_GROUP_PREFIX = "/aws/sagemaker/Job"

# Transient error codes returned by ``DescribeJob`` that ``wait()`` should
# silently skip on the polling cadence. Anything outside this set is treated
# as a hard failure and propagated immediately.
_TRANSIENT_CLIENT_ERROR_CODES = frozenset({
    "ThrottlingException",
    "Throttling",
    "ThrottledException",
    "TooManyRequestsException",
    "RequestLimitExceeded",
    "ServiceUnavailable",
    "ServiceUnavailableException",
    "InternalServerError",
    "InternalFailure",
})

# Quiet_Period default. Five minutes from the last panel print (or from
# ``wait()`` entry if no panel has been printed since) before the backoff
# message is emitted and sentinel scanning stops.
_QUIET_PERIOD_SECONDS = 5 * 60


def _is_transient_client_error(error: BaseException) -> bool:
    """Return True iff ``error`` is a ``botocore.exceptions.ClientError`` whose
    error code or HTTP status indicates a transient server-side condition
    (throttling, 5xx). Used by ``wait()`` to decide whether to retry the
    next describe tick or propagate immediately.
    """
    if not isinstance(error, ClientError):
        return False
    response = getattr(error, "response", None) or {}
    code = response.get("Error", {}).get("Code", "")
    if code in _TRANSIENT_CLIENT_ERROR_CODES:
        return True
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
    try:
        return int(status) >= 500
    except (TypeError, ValueError):
        return False

# Pattern for the Python ``logging`` default timestamp prefix:
#   "YYYY-MM-DD HH:MM:SS,mmm" or "YYYY-MM-DD HH:MM:SS.mmm"
_LOG_TIMESTAMP_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}[,.]\d{3})"
)
# Pattern for extracting the phase description after a [MILESTONE] sentinel.
_MILESTONE_RE = re.compile(r"\[MILESTONE\]\s+(.*)")
# Python logging level token as formatted by the job container
# ("<timestamp> - DEBUG - <logger> - <message>").
_DEBUG_LEVEL_RE = re.compile(r"\s-\sDEBUG\s-\s")


def _is_jupyter_environment() -> bool:
    """Checks whether the process is running inside a Jupyter kernel.

    Relies on the IPKernelApp config key being present on
    ``IPython.get_ipython().config``. Returns False when IPython is not
    importable or no kernel is attached (plain Python REPL / script).
    """
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        ipython = get_ipython()
        return ipython is not None and "IPKernelApp" in ipython.config
    except ImportError:
        return False
    except Exception:
        # Defensive: any IPython-internal failure => not a Jupyter env.
        return False


def _extract_log_timestamp(line: str) -> Optional[datetime]:
    """Extract the timestamp prefix from a CloudWatch log line, if present.

    Timestamps follow the Python logging default format:
    ``YYYY-MM-DD HH:MM:SS,mmm``. We accept either ``,`` or ``.`` as the
    milliseconds separator.

    Returns ``None`` on any parse failure (no prefix, malformed prefix,
    or any other error). The caller treats ``None`` as "timestamp
    unknown" and stores it in the milestone transitions list as-is.
    """
    try:
        match = _LOG_TIMESTAMP_RE.match(line)
        if not match:
            return None
        ts_str = match.group(1).replace(",", ".")
        return datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return None


class Job:
    """Job resource wrapping CreateJob/DescribeJob/StopJob via boto3."""

    def __init__(self, *, job_name, job_category, session, **kwargs):
        self.job_name = job_name
        self.job_category = job_category
        self._session = session
        self._client = session.sagemaker_client
        self.job_arn = kwargs.get("job_arn")
        self.job_status = kwargs.get("job_status")
        self.secondary_status = kwargs.get("secondary_status")
        self.failure_reason = kwargs.get("failure_reason")
        self.job_config_document = kwargs.get("job_config_document")
        # Schema version of the JobConfigDocument. DataPreparer._rehydrate_datasets
        # dispatches DqShape.for_version() on this — without it every completed
        # DQ v2 (0.2.0) job fell back to the V1 shape and failed rehydration
        # with "Completed job missing DatasetArn in JobConfigDocument".
        self.job_config_schema_version = kwargs.get("job_config_schema_version")
        self._status_message = None
        # Cached CloudWatch Logs stream name. The service surfaces this in
        # SecondaryStatusTransitions[].statusMessage during the
        # Executing phase as `stream=<name>`, but later transitions
        # (Publishing, Finalizing, Completed) overwrite _status_message
        # with new text that does not contain `stream=`. Cache the first
        # observation so the final panel still renders the link.
        self._log_stream_name: Optional[str] = None
        self.created_at: Optional[str] = kwargs.get("created_at")
        # SDG-only progress tracking (DataPreparer sets these when applicable).
        # Counted live from the service's verified_samples.jsonl via S3 Select.
        self._expected_samples = kwargs.get("expected_samples")
        self._progress_output_s3_uri = kwargs.get("progress_output_s3_uri")
        # Lock-in: once we discover the right S3 key that *provably* belongs
        # to this job, reuse it on subsequent ticks. Guards against drifting
        # to a concurrent/past job whose JobName shares a string prefix with
        # ours (e.g. "foo-sdg" vs "foo-sdg-extra").
        self._progress_target_key: Optional[str] = None

        # CloudWatch log polling cursor — instance state so successive wait() calls
        # on the same Job resume from where the previous call left off.
        # Without this, switching modes (logs=False -> logs=True) or
        # re-entering wait() would replay every event from the start of
        # the stream.
        self._log_next_token: Optional[str] = None
        # Phase transitions sourced from [MILESTONE] sentinel lines emitted
        # by the container at phase boundaries. Append-only — entries are
        # never removed or overwritten.
        self._milestone_transitions: List[Dict] = []
        # Quiet-period backoff message has fired exactly once for the
        # current wait() invocation. Reset by wait() on entry so
        # subsequent calls start fresh.
        self._quiet_period_notified: bool = False
        # Most recently observed sample count, populated by ``wait()``
        # via ``_count_samples()``. Exposed as instance state so the
        # panel renderer can show it independently of the wait loop.
        self._current_samples: Optional[int] = None
        # ``time.time()`` snapshot taken at the top of ``wait()`` so the
        # panel's ``Elapsed:`` line reflects time-since-wait-began rather
        # than time-since-job-started. ``None`` until ``wait()`` runs;
        # the panel falls back to ``0m 00s`` in that case (e.g. when
        # ``_render_panel`` is called directly from a unit test).
        self._wait_start_time: Optional[float] = None
        # Full list of ``SecondaryStatusTransitions`` from the latest
        # ``DescribeJob`` response. Used by ``_render_panel`` to populate
        # the API-sourced rows of the Status_Transitions_Table. Stored
        # alongside ``_status_message`` (which only carries the latest
        # message) so the panel does not need to re-issue DescribeJob.
        self._secondary_status_transitions: List[Dict] = []

    @classmethod
    def create(
        cls,
        job_name: str,
        role_arn: str,
        job_category: str,
        job_config_schema_version: str,
        job_config_document: str,
        session: Optional[Session] = None,
        job_arn: Optional[str] = None,
        customer_details: Optional[dict] = None,
        expected_samples: Optional[int] = None,
        progress_output_s3_uri: Optional[str] = None,
        tags: Optional[list] = None,
        **kwargs,
    ) -> "Job":
        """Create a Job via the CreateJob API.

        The envelope carries the required fields for job creation. This
        resource silently drops any field the currently-loaded service model
        does not know about, so new fields only start flowing once the service
        model catches up. Internal-only fields are never forwarded.
        """
        client = session.sagemaker_client
        request: dict = {
            "JobName": job_name,
            "RoleArn": role_arn,
            "JobCategory": job_category,
            "JobConfigSchemaVersion": job_config_schema_version,
            "JobConfigDocument": job_config_document,
        }
        if job_arn is not None:
            request["JobArn"] = job_arn
        if customer_details is not None:
            request["CustomerDetails"] = customer_details
        if tags is not None:
            request["Tags"] = [
                {"Key": t.key, "Value": t.value} if hasattr(t, "key") else t
                for t in tags
            ]

        # Intersect the request with fields actually accepted by the loaded
        # service model. This prevents ParamValidationError when the SDK
        # ships a richer envelope than the service model has published yet.
        try:
            op_model = client.meta.service_model.operation_model("CreateJob")
            accepted_members = set(op_model.input_shape.members.keys())
            request = {k: v for k, v in request.items() if k in accepted_members and v is not None}
        except Exception:
            # If introspection fails for any reason, fall back to non-None filter.
            request = {k: v for k, v in request.items() if v is not None}

        logger.info("Creating Job: %s (%s)", job_name, job_category)
        response = client.create_job(**request)
        return cls(
            job_name=job_name,
            job_category=job_category,
            session=session,
            job_arn=response.get("JobArn", job_arn),
            job_config_schema_version=job_config_schema_version,
            expected_samples=expected_samples,
            progress_output_s3_uri=progress_output_s3_uri,
        )

    @classmethod
    def get(
        cls,
        job_name: str,
        job_category: str,
        session: Optional[Session] = None,
    ) -> "Job":
        """Re-attach to an existing job by name and load its state via DescribeJob.

        Companion to an asynchronous (wait=False) submission: persist
        ``job.job_name`` from the returned handle, then rebuild the handle in a
        later step or process with ``Job.get(job_name, job_category)`` and call
        ``refresh()``/``wait()`` or read the job state off it.
        """
        session = session or Session()
        return cls(
            job_name=job_name, job_category=job_category, session=session
        ).refresh()

    def refresh(self) -> "Job":
        """Refresh job state via DescribeJob."""
        resp = self._client.describe_job(
            JobName=self.job_name, JobCategory=self.job_category
        )
        self.job_arn = resp.get("JobArn")
        self.job_status = resp.get("JobStatus")
        self.secondary_status = resp.get("SecondaryStatus")
        self.failure_reason = resp.get("FailureReason")
        self.job_config_document = resp.get("JobConfigDocument")
        self.job_config_schema_version = resp.get("JobConfigSchemaVersion")
        self.created_at = resp.get("CreationTime")  # datetime or ISO 8601 string from DescribeJob
        self._status_message = None
        transitions = resp.get("SecondaryStatusTransitions", []) or []
        self._secondary_status_transitions = transitions
        if transitions:
            self._status_message = transitions[-1].get("StatusMessage")
        return self

    def _scan_for_milestones(self) -> List[str]:
        """Drain the CW log stream and return only ``[MILESTONE]`` lines.

        Calls ``_tail_logs(self._log_next_token)`` and updates
        ``self._log_next_token`` as a side effect so successive calls
        never replay the same events. Returns only lines containing
        ``[MILESTONE]`` — non-milestone lines are discarded since this
        method is used by ``logs=False`` mode where general log content
        is intentionally not surfaced.
        """
        new_token, all_lines = self._tail_logs(self._log_next_token)
        self._log_next_token = new_token
        return [line for line in all_lines if "[MILESTONE]" in line]

    def _process_milestone(self, line: str) -> None:
        """Parse a ``[MILESTONE]`` line and append a transition entry.

        Extracts the phase description (everything after ``[MILESTONE] ``)
        and the timestamp from the log line prefix. The entry is always
        appended — if the timestamp prefix is missing or unparseable,
        ``timestamp`` is stored as ``None`` and no exception is raised
        (the caller treats that as "timestamp unknown"). Lines that do
        not contain a ``[MILESTONE]`` token are silently ignored.
        """
        match = _MILESTONE_RE.search(line)
        if not match:
            return
        phase_description = match.group(1).strip()
        timestamp = _extract_log_timestamp(line)
        self._milestone_transitions.append(
            {
                "phase": phase_description,
                "timestamp": timestamp,
            }
        )

    def _cloudwatch_logs_url(self) -> Optional[str]:
        """Build the CloudWatch Logs console URL for this job's log stream.

        Returns ``None`` when the stream name has not yet been surfaced via
        ``SecondaryStatusTransitions[].statusMessage`` — callers render a
        pending placeholder in that case. The URL points at the customer's
        active region (best-effort; falls back to an empty region segment if
        the session does not expose one).
        """
        stream_name = self._extract_log_stream_name()
        if not stream_name:
            return None
        try:
            region = self._session.boto_region_name or ""
        except Exception:
            region = ""
        log_group = f"{_LOG_GROUP_PREFIX}/{self.job_category}/"
        from urllib.parse import quote

        return (
            f"https://console.aws.amazon.com/cloudwatch/home?region={region}"
            f"#logsV2:log-groups/log-group/{quote(log_group, safe='')}"
            f"/log-events/{quote(stream_name, safe='')}"
        )

    def _output_s3_url(self) -> Optional[str]:
        """Build the S3 console URL for this job's output directory.

        Prefers the discovered job-specific subdir
        (``{output_s3_uri}/{JobName}-{ts}/``) once the progress tracker
        has locked onto the verified-samples file. Falls back to the
        customer's ``output_s3_uri`` prefix while discovery is still
        pending. Returns ``None`` when ``output_s3_uri`` is unset (e.g.
        DQ jobs that don't have an output URI configured at all).
        """
        from urllib.parse import quote

        # Resolve (bucket, prefix) from the most specific source available.
        bucket: Optional[str] = None
        prefix: Optional[str] = None

        # Tier 1: discovered subdir — use the parent directory of the
        # locked verified-samples key. The shape is always
        # ``{customer_prefix}/<JobName>-<ts>/<job_metadata|output|record_level>/<file>``,
        # so trim from the trailing 2 segments to get the
        # ``<JobName>-<ts>/`` directory.
        if self._progress_target_key:
            try:
                # _progress_output_s3_uri is the source of truth for bucket
                # — the locked key is bucket-relative.
                if self._progress_output_s3_uri and self._progress_output_s3_uri.startswith("s3://"):
                    rest = self._progress_output_s3_uri[len("s3://"):].rstrip("/")
                    bucket = rest.split("/", 1)[0]
                # Trim two trailing segments (subdir + filename) to land on
                # ``{customer_prefix}/<JobName>-<ts>/``.
                parts = self._progress_target_key.rstrip("/").split("/")
                if len(parts) >= 2:
                    prefix = "/".join(parts[:-2]) + "/"
            except Exception:
                bucket, prefix = None, None

        # Tier 2: bare customer output_s3_uri.
        if not (bucket and prefix):
            uri = self._progress_output_s3_uri
            if not uri or not uri.startswith("s3://"):
                return None
            rest = uri[len("s3://"):].rstrip("/")
            split = rest.split("/", 1)
            bucket = split[0]
            prefix = (split[1] + "/") if len(split) > 1 else ""

        try:
            region = self._session.boto_region_name or ""
        except Exception:
            region = ""

        return (
            f"https://s3.console.aws.amazon.com/s3/buckets/{bucket}"
            f"?prefix={quote(prefix, safe='')}&region={region}"
        )

    def _extract_dataset_arn(self) -> Optional[str]:
        """Extract ``DatasetArn`` from ``self.job_config_document``.

        The service surfaces the customer-visible Dataset hub-content ARN
        in the JobConfigDocument JSON string once the job completes (or
        even mid-flight if the dataset was created early — depends on the
        service implementation). Returns ``None`` when the document is
        missing, unparseable, or does not contain a ``DatasetArn`` key.
        Failure modes are silent — the panel renders without the row in
        all of them.
        """
        doc = self.job_config_document
        if not doc:
            return None
        try:
            parsed = json.loads(doc) if isinstance(doc, str) else doc
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        arn = parsed.get("DatasetArn")
        return arn if isinstance(arn, str) and arn else None

    @property
    def labeling_summary(self) -> "LabelingSummary":
        """This auto-labeling job's summary report.

        Reads the DataSet the job registered and follows its labeling pointer.
        See
        :class:`~sagemaker.ai_registry.data_preparer.auto_labeling.summary.LabelingSummary`.

        Raises:
            DataPreparerError: If the job has not registered a DataSet yet, or
                if it was not produced by an auto-labeling job.
        """
        # Local imports: the summary module reaches back into preparer, which
        # imports this module.
        from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerError
        from sagemaker.ai_registry.data_preparer.auto_labeling.summary import LabelingSummary

        dataset_arn = self._extract_dataset_arn()
        if not dataset_arn:
            raise DataPreparerError(
                f"Job '{self.job_name}' has no DataSet yet; wait for it to complete first. "
                "The labeling summary is read from the DataSet the job registers."
            )
        return LabelingSummary.from_dataset(dataset_arn, sagemaker_session=self._session)

    def _job_category_label(self) -> str:
        """Return the raw ``job_category`` to use as a humanized prefix on
        header rows (``"SyntheticDataGenerationJob Name"``,
        ``"DataQualityEvaluationJob ARN"``, …). The category value is
        already PascalCase, no spacing transform needed. Falls back to
        ``"Job"`` when the category is unset/empty so the rendered label
        remains coherent.
        """
        return self.job_category or "Job"

    @staticmethod
    def _render_progress_bar(done: int, total: int) -> str:
        """Build a 20-block green progress bar fragment.

        Returns a string like ``"[green][██████░░░░░░░░░░░░░░][/green] 40.0%"``
        where the outer brackets are literal characters (not rich markup) so
        the rendered output reads as ``[██████░░░░░░░░░░░░░░] 40.0%``. The
        block count is fixed at 20 and each block represents 5%.
        """
        pct = (done / total * 100.0) if total > 0 else 0.0
        filled = int(pct / 5)
        empty = 20 - filled
        return f"[green][{'█' * filled}{'░' * empty}][/green] {pct:.1f}%"

    def _render_panel(self, is_final: bool = False) -> Panel:
        """Build the live Status_Panel for this job's current state.

        Renders a status panel with a header sub-table of property labels
        with dim values (JobName, JobArn, JobCategory, Links), a status
        sub-table (Job Status, Secondary Status, Elapsed Time, optional
        Sample Count, optional Failure Reason), and a SIMPLE-boxed
        Status_Transitions_Table with a bold magenta header rendered
        separately above it.

        Conditional rows are populated as their underlying state becomes
        available:

          * the JobArn row only when ``self.job_arn`` is set;
          * the Secondary Status row only when ``self.secondary_status`` is
            truthy;
          * the Sample Count row only when ``self._expected_samples`` is set
            (containing a literal ``N / M`` substring so callers can grep
            for progress);
          * the Failure Reason row only when the job is ``Failed`` and a
            failure_reason is present.

        ``is_final`` switches the border to green and the title color to
        bold green — used by ``wait()`` once a terminal state is observed.
        The unit tests call this directly to inspect body fields without
        driving the full wait loop, which is why the rendering is split
        out from ``wait()``.
        """
        from rich.console import Console, Group
        from rich.table import Table
        from rich.text import Text

        # --- Header sub-table ------------------------------------------------------
        # The Property column is widened to 34 to accommodate the longest
        # category-prefixed label (``SyntheticDataGenerationJob Name`` at
        # 31 chars). The status_table below keeps the original width=20
        # because its labels stay short ("Job Status", "Elapsed Time", …).
        header_table = Table(show_header=False, box=None, padding=(0, 1))
        header_table.add_column("Property", style="cyan bold", width=34)
        header_table.add_column("Value", style="dim", overflow="fold")

        category_label = self._job_category_label()
        header_table.add_row(
            f"{category_label}Job Name",
            f"[bold green]{self.job_name}[/bold green]",
        )
        if self.job_arn:
            header_table.add_row(
                f"{category_label}Job ARN", f"[dim]{self.job_arn}[/dim]"
            )
        # DatasetArn is surfaced inside JobConfigDocument once the service
        # has registered the dataset with Hub. Conditional row — only
        # appears when present, mirroring the JobArn row's gating.
        dataset_arn = self._extract_dataset_arn()
        if dataset_arn:
            header_table.add_row(
                "Dataset ARN", f"[dim]{dataset_arn}[/dim]"
            )
        header_table.add_row("JobCategory", f"{self.job_category}")

        cw_url = self._cloudwatch_logs_url()
        if cw_url:
            cw_link = (
                f"[bright_blue underline][link={cw_url}]"
                f"🔗 CloudWatch Logs[/link][/bright_blue underline]"
            )
        else:
            cw_link = "[dim]🔗 CloudWatch Logs (stream name pending)[/dim]"

        # Output S3 URI link — points at the customer's output directory
        # and upgrades to the discovered job-specific subdir once the
        # progress tracker has locked onto the verified-samples file.
        out_url = self._output_s3_url()
        if out_url:
            out_link = (
                f"[bright_blue underline][link={out_url}]"
                f"🔗 OutputS3Uri[/link][/bright_blue underline]"
            )
            header_table.add_row("Links", f"{cw_link} | {out_link}")
        else:
            header_table.add_row("Links", cw_link)

        # --- Status sub-table ------------------------------------------------------
        status_table = Table(show_header=False, box=None, padding=(0, 1))
        status_table.add_column("Property", style="cyan bold", width=20)
        status_table.add_column("Value", style="dim", overflow="fold")

        status_table.add_row(
            "Job Status",
            f"[bold][orange3]{self.job_status or 'Pending'}[/][/]",
        )
        if self.secondary_status:
            status_table.add_row(
                "Secondary Status",
                f"[bold yellow]{self.secondary_status}[/bold yellow]",
            )

        if self._wait_start_time is not None:
            elapsed = max(0.0, time.time() - self._wait_start_time)
        else:
            elapsed = 0.0
        status_table.add_row(
            "Elapsed Time", f"[bold bright_red]{elapsed:.1f}s[/bold bright_red]"
        )

        if self._expected_samples is not None:
            done = self._current_samples if self._current_samples is not None else 0
            total = self._expected_samples
            bar = self._render_progress_bar(done, total)
            sample_label = (
                "Records Evaluated"
                if self.job_category == "DataQualityEvaluation"
                else "Sample Count"
            )
            status_table.add_row(sample_label, f"{done} / {total} samples {bar}")

        if self.job_status == "Failed" and self.failure_reason:
            status_table.add_row(
                "Failure Reason", f"[bright_red]{self.failure_reason}[/bright_red]"
            )

        # --- Status_Transitions_Table ----------------------------------------------
        transitions_table = self._build_transitions_table()

        # --- Compose Group ---------------------------------------------------------
        if transitions_table is not None:
            combined = Group(
                header_table,
                Text(""),
                status_table,
                Text(""),
                Text("Status Transitions", style="bold magenta"),
                transitions_table,
            )
        else:
            combined = Group(header_table, Text(""), status_table)

        # --- Wrap in Panel ---------------------------------------------------------
        panel_width = 80
        # Use a fresh Console solely to read the terminal width — not to print.
        # Only shrink to 80% of the console width when we're attached to a real
        # terminal; in non-TTY contexts (pytest, redirected stdout) Rich falls
        # back to width=80, and shrinking that further wraps long values like
        # failure_reason mid-string.
        console = Console()
        if console.is_terminal and console.width:
            panel_width = int(console.width * 0.8)

        if is_final:
            title = "[bold green]DataPreparer Job Status[/bold green]"
            border_style = "green"
        else:
            title = "[bold bright_blue]DataPreparer Job Status[/bold bright_blue]"
            border_style = "orange3"

        return Panel(
            combined,
            title=title,
            border_style=border_style,
            width=panel_width,
        )

    def _extract_log_stream_name(self) -> Optional[str]:
        """Return the CloudWatch Log stream name surfaced by the service.

        The stream is published during the ``Executing`` phase via
        ``SecondaryStatusTransitions[].statusMessage`` as ``stream=<name>``.
        Later transitions (Publishing, Finalizing, Completed) overwrite
        ``_status_message`` with text that does not contain ``stream=``,
        so we cache the first match on the instance and prefer the cached
        value on subsequent reads. Falls back to scanning the full
        ``_secondary_status_transitions`` list before giving up.
        """
        if self._log_stream_name:
            return self._log_stream_name

        # Try the most recent statusMessage first (cheapest path).
        if self._status_message:
            match = re.search(r"stream=([^\s,]+)", self._status_message)
            if match:
                self._log_stream_name = match.group(1)
                return self._log_stream_name

        # Fallback: scan every transition's StatusMessage. The service
        # may have already moved past Executing by the time we check.
        for entry in self._secondary_status_transitions or []:
            msg = entry.get("StatusMessage") or ""
            match = re.search(r"stream=([^\s,]+)", msg)
            if match:
                self._log_stream_name = match.group(1)
                return self._log_stream_name

        return None

    def _build_transitions_table(self) -> Optional["Table"]:
        """Build the Status_Transitions_Table for the panel body.

        Top-level rows come from ``self._secondary_status_transitions``
        (API-sourced). Sentinel sub-rows from ``self._milestone_transitions``
        are nested under the ``Executing`` row by indenting their phase
        column with ``↳ ``.

        For each row:
          * Running rows show ``Running...`` in the detail column;
          * Completed rows show the elapsed duration and a ``✓`` glyph;
          * The active Sample Generation sub-row replaces ``Running...``
            with the sample count progress bar when ``_expected_samples``
            is set.

        Once any sentinel sub-row is present beneath the ``Executing`` API
        row, the ``Executing`` row's detail column is left blank to avoid
        redundancy with the sub-rows below it.

        The table itself uses ``rich.box.SIMPLE`` with a bold-magenta header
        for consistent styling. The ``Status Transitions`` heading is
        rendered separately as a magenta ``Text`` block above the table by
        ``_render_panel`` — this method does not set a title on the table.

        Returns ``None`` when no transitions have been recorded yet.
        """
        from rich.box import SIMPLE
        from rich.table import Table

        api_rows = self._secondary_status_transitions or []
        sentinel_rows = self._milestone_transitions or []
        if not api_rows and not sentinel_rows:
            return None

        table = Table(
            show_header=True,
            header_style="bold magenta",
            box=SIMPLE,
            padding=(0, 1),
        )
        table.add_column("", style="green", width=2)  # status glyph column
        table.add_column("Step", style="cyan", width=15)
        table.add_column("Details", style="orange3", width=35, no_wrap=False)
        table.add_column("Duration", style="green", width=12)

        # Pair sentinel sub-rows into start/complete groups so we can render
        # each phase as a single row with the appropriate state.
        sentinel_phases = self._pair_sentinels(sentinel_rows)

        executing_index: Optional[int] = None
        is_terminal = self.job_status in _TERMINAL_STATES
        for idx, api_row in enumerate(api_rows):
            api_status = (api_row.get("Status") or "").lower()
            is_executing = api_status == "executing" or api_status == "inprogress"
            # A row is completed if it has an EndTime, or a subsequent row
            # exists (implying this step finished before the next began), or
            # the overall job reached a terminal state.
            is_completed = (
                bool(api_row.get("EndTime"))
                or idx < len(api_rows) - 1
                or is_terminal
            )
            glyph = "✓" if is_completed else "⋯"
            phase_label = api_row.get("Status") or "Unknown"
            # If this is the Executing row and we have sentinel sub-rows to
            # render under it, suppress the API detail column to avoid
            # duplicating the sentinel-sourced phase information below.
            if is_executing and sentinel_phases:
                detail = ""
            else:
                detail = api_row.get("StatusMessage") or (
                    "Running..." if not is_completed else ""
                )
            next_row = api_rows[idx + 1] if idx < len(api_rows) - 1 else None
            duration = self._format_transition_duration(
                api_row, next_row=next_row, is_terminal=is_terminal
            )
            table.add_row(glyph, phase_label, detail, duration)
            if is_executing:
                executing_index = idx

        # If no Executing row appeared but sentinels exist, render them as
        # top-level rows so they remain visible. This degrades gracefully
        # when the API hasn't surfaced an Executing row yet.
        if executing_index is None and sentinel_phases:
            for phase in sentinel_phases:
                glyph, label, detail, duration = self._format_sentinel_row(phase)
                table.add_row(glyph, label, detail, duration)
        else:
            for phase in sentinel_phases:
                glyph, label, detail, duration = self._format_sentinel_row(phase)
                # Indent the phase label so it visually nests under Executing.
                table.add_row(glyph, f"↳ {label}", detail, duration)

        return table

    @staticmethod
    def _pair_sentinels(sentinel_rows: List[Dict]) -> List[Dict]:
        """Collapse ``Starting <phase>`` / ``Completed <phase>`` pairs into
        a single phase entry with start / end timestamps and a derived
        label. Unpaired ``Starting`` lines stay as in-progress phases.
        """
        phases: Dict[str, Dict] = {}
        order: List[str] = []
        for entry in sentinel_rows:
            phase = (entry.get("phase") or "").strip()
            ts = entry.get("timestamp")
            if phase.startswith("Starting "):
                key = phase[len("Starting "):].strip()
                if key not in phases:
                    phases[key] = {"label": key, "start": ts, "end": None}
                    order.append(key)
                else:
                    phases[key]["start"] = ts
            elif phase.startswith("Completed "):
                key = phase[len("Completed "):].strip()
                if key not in phases:
                    phases[key] = {"label": key, "start": None, "end": ts}
                    order.append(key)
                else:
                    phases[key]["end"] = ts
            else:
                # Free-form phase string — render as a standalone, unpaired row.
                phases[phase] = {"label": phase, "start": ts, "end": None}
                order.append(phase)
        return [phases[k] for k in order]

    def _format_sentinel_row(self, phase: Dict) -> tuple:
        """Convert a paired sentinel phase entry into a table row tuple
        ``(glyph, label, detail, duration)``.

        For the Sample Generation phase, the detail column is replaced
        with the live sample-count progress bar when ``_expected_samples``
        is set.
        """
        label = self._humanize_phase_label(phase["label"])
        completed = phase.get("end") is not None
        glyph = "✓" if completed else "⋯"
        # Detail column: progress bar for active Sample Generation, plain
        # status text otherwise.
        is_sample_generation = "sample generation" in phase["label"].lower()
        if (
            not completed
            and is_sample_generation
            and self._expected_samples is not None
        ):
            done = self._current_samples if self._current_samples is not None else 0
            total = self._expected_samples
            detail = f"{done} / {total} {self._render_progress_bar(done, total)}"
        elif completed and is_sample_generation and self._current_samples is not None:
            detail = f"{self._current_samples} samples"
        elif completed:
            detail = "Completed"
        else:
            detail = "Running..."
        # Duration: elapsed between start and end if both are present.
        duration = "—"
        start = phase.get("start")
        end = phase.get("end")
        if start is not None and end is not None:
            try:
                seconds = (end - start).total_seconds()
                duration = f"{seconds:.1f}s"
            except Exception:
                duration = "—"
        elif start is not None and not completed:
            try:
                duration = (
                    f"{(datetime.now(start.tzinfo if hasattr(start, 'tzinfo') else None) - start).total_seconds():.1f}s"
                )
            except Exception:
                duration = "—"
        return glyph, label, detail, duration

    @staticmethod
    def _humanize_phase_label(label: str) -> str:
        """Map a raw sentinel phase string (e.g. ``code generation``) to a
        Title Case label (``Code Generation``) for display.
        """
        return " ".join(word.capitalize() for word in label.split())

    @staticmethod
    def _format_transition_duration(
        api_row: Dict,
        next_row: Optional[Dict] = None,
        is_terminal: bool = False,
    ) -> str:
        """Return a step's Duration for the Status_Transitions_Table.

        The service does not populate ``EndTime`` on
        ``SecondaryStatusTransitions`` — not even on a Completed job — so a
        step's end must be recovered from the *next* transition's
        ``StartTime`` (the instant the following step began is the instant
        this one ended). The resolution order is:

          1. This row's own ``EndTime`` if the service ever populates it
             (future-proofing — preferred when present).
          2. ``next_row``'s ``StartTime`` — the frozen duration of a
             completed step. This is the fix for F17: without it, a
             completed step fell through to ``now − StartTime`` and its
             Duration kept incrementing (an age, not a duration).
          3. ``now − StartTime`` — the live-ticking elapsed time, used
             ONLY for the genuinely in-progress step (no successor row and
             the job has not reached a terminal state).
          4. ``—`` — no successor and the job is terminal (e.g. the
             trailing ``Completed`` marker row, which is not a
             duration-bearing phase). Rendering a live age here would be
             misleading, so we omit it.
        """
        start = api_row.get("StartTime")
        if start is None:
            return "—"

        start_dt = Job._coerce_datetime(start)
        if start_dt is None:
            return "—"

        # 1. Prefer an explicit EndTime if the service ever surfaces one.
        end = api_row.get("EndTime")
        if end is not None:
            end_dt = Job._coerce_datetime(end)
            if end_dt is None:
                return "—"
            return f"{(end_dt - start_dt).total_seconds():.1f}s"

        # 2. Frozen duration: derive the end from the next step's start.
        if next_row is not None:
            next_start_dt = Job._coerce_datetime(next_row.get("StartTime"))
            if next_start_dt is not None:
                return f"{(next_start_dt - start_dt).total_seconds():.1f}s"

        # 3. In-progress step: live elapsed since start, ticks each redraw.
        if not is_terminal:
            elapsed = (datetime.now(start_dt.tzinfo) - start_dt).total_seconds()
            return f"{elapsed:.1f}s"

        # 4. Terminal marker row with no successor — nothing to measure.
        return "—"

    @staticmethod
    def _coerce_datetime(value) -> Optional[datetime]:
        """Best-effort conversion of an API timestamp to a ``datetime``.

        Accepts ``datetime``, ISO 8601 strings (with optional Z suffix), and
        ``int``/``float`` epoch seconds. Returns ``None`` on any unrecognized
        input or parse failure — callers render "—" in that case.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except Exception:
                return None
        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(value)
            except Exception:
                return None
        return None

    def wait(
        self,
        poll: int = 10,
        describe_interval: int = 30,
        timeout: Optional[int] = None,
        logs: bool = False,
    ) -> None:
        """Poll until terminal state, rendering live status panels.

        ``logs=False`` (default) is Panel_Mode: a live status panel is
        rendered in place using ``clear_output(wait=True)`` +
        ``Console(force_jupyter=True)`` while polling, with a final
        ANSI-rendered panel printed via ``Console(force_jupyter=False,
        force_terminal=True)`` so it serializes to the saved ``.ipynb``
        file. The CW log tailing path is fully disabled in Panel_Mode —
        only the ``[MILESTONE]`` sentinel scan runs against the CW stream,
        populating the Status_Transitions_Table.

        ``logs=True`` is Log_Tail_Mode: an append-only stream of
        INFO+/MILESTONE log lines is printed alongside periodic panels,
        using ``force_terminal_console`` throughout (no ``clear_output``).

        Cadence model:
          * ``poll`` (default 10s) — loop sleep. Sentinel/log polls run
            on this cadence so bursty jobs surface promptly.
          * ``describe_interval`` (default 30s) — gates ``DescribeJob``
            and S3-progress queries. Terminal status is observed on
            describe ticks; a 2s grace sleep + final sentinel scan after
            terminal catches trailing events that ingest after status
            flip.

        Other rules:
          * Re-entrancy: ``self._log_next_token`` and
            ``self._milestone_transitions`` persist across calls so a
            second ``wait()`` on the same Job continues from where the
            first left off.
          * Quiet_Period (default 5m): after no panel for 5m, emit a
            backoff message exactly once. In ``logs=False``, sentinel
            scanning stops after the backoff; in ``logs=True``, CW
            polling continues unchanged.
          * Transient ``DescribeJob`` failures: skip the tick silently;
            propagate after 3 consecutive failures.
          * Already-terminal entry: print final panel immediately and
            return without entering the loop.
          * Timeout: print the current state panel before raising
            ``DataPreparerJobTimeoutError``.
        """
        # Lazy import — avoids a circular dependency at module load time
        # (preparer.py imports _job_resource, exceptions imports constants).
        from sagemaker.ai_registry.data_preparer.exceptions import (
            DataPreparerJobTimeoutError,
        )

        # Reset per-invocation flags so a re-entered wait() starts fresh.
        # Persistent state (_log_next_token, _milestone_transitions) is
        # intentionally NOT reset.
        self._quiet_period_notified = False
        self._wait_start_time = time.time()

        if logs:
            self._wait_log_tail_mode(
                poll=poll,
                describe_interval=describe_interval,
                timeout=timeout,
                timeout_exc_cls=DataPreparerJobTimeoutError,
            )
        else:
            self._wait_panel_mode(
                poll=poll,
                describe_interval=describe_interval,
                timeout=timeout,
                timeout_exc_cls=DataPreparerJobTimeoutError,
            )

        logger.info("Final Resource Status: %s", self.job_status)
        if self.job_status == "Failed":
            logger.error("Job %s failed: %s", self.job_name, self.failure_reason)

    # ------------------------------------------------------------------
    # logs=False — Panel_Mode
    # ------------------------------------------------------------------

    def _wait_panel_mode(
        self,
        poll: int,
        describe_interval: int,
        timeout: Optional[int],
        timeout_exc_cls: type,
    ) -> None:
        """Implementation of ``wait(logs=False)`` — the live status panel
        path. See ``wait()`` docstring for the cadence and rendering
        rules.
        """
        from rich.console import Console

        is_jupyter = _is_jupyter_environment()

        # Single live console — same one used for in-loop updates and the
        # terminal render. Uses clear_output + console.print pattern where
        # clear_output(wait=True) + console.print(Panel(...)) on every
        # tick produces a single cell-persisted panel that ends on the
        # terminal-state frame. Using two consoles previously caused a
        # duplicate panel: the in-loop one rendered via force_jupyter and
        # the "final" one rendered via a separate force_terminal console
        # both ended up in the saved notebook output.
        if is_jupyter:
            live_console = Console(force_jupyter=True)
        else:
            live_console = Console(
                force_jupyter=False, force_terminal=True, width=120
            )

        start_time = self._wait_start_time or time.time()
        last_describe = 0.0
        last_panel_time = start_time
        consecutive_describe_failures = 0
        progress_changed = False
        # Track values whose mutation should trigger a redraw. Initial
        # values reflect what the constructor / pre-loop refresh produced;
        # any subsequent change in the loop sets progress_changed.
        last_status = self.job_status
        last_status_message = self._status_message
        last_transitions_count = len(self._secondary_status_transitions)
        last_milestone_count = len(self._milestone_transitions)

        # Print the initial panel BEFORE any DescribeJob call.
        # The customer sees something immediately even if the very first
        # describe is slow or throttled.
        if is_jupyter:
            self._clear_jupyter_output(wait=True)
        live_console.print(self._render_panel(is_final=False))
        last_panel_time = time.time()

        # Initial describe — retry transient errors silently like the
        # main loop. If this returns terminal status, print
        # the final panel via the live console and bail out
        # without entering the polling loop.
        try:
            self.refresh()
            consecutive_describe_failures = 0
            last_describe = time.time()
            if self._progress_output_s3_uri:
                initial_count = self._count_samples()
                if initial_count is not None:
                    self._current_samples = initial_count
            if self.job_status in _TERMINAL_STATES:
                # Drain pending sentinels so the final panel reflects
                # the complete transitions table. No grace sleep on the
                # already-terminal path — the job is done and the
                # customer should not wait.
                try:
                    for milestone_line in self._scan_for_milestones():
                        self._process_milestone(milestone_line)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.debug("Terminal sentinel scan failed: %s", exc)
                # Snap sample count to expected on successful completion (see helper docstring).
                self._apply_completion_fallback()
                self._render_terminal_panel(live_console)
                return
        except ClientError as e:
            if _is_transient_client_error(e):
                consecutive_describe_failures += 1
                # Don't update last_describe — the loop's first describe
                # tick should retry on the next iteration.
            else:
                raise

        while True:
            time.sleep(poll)
            now = time.time()

            # --- Describe tick ---------------------------------------------------
            if now - last_describe >= describe_interval:
                try:
                    self.refresh()
                    consecutive_describe_failures = 0
                    last_describe = now
                    if self._progress_output_s3_uri:
                        new_count = self._count_samples()
                        if (
                            new_count is not None
                            and new_count != self._current_samples
                        ):
                            self._current_samples = new_count
                            progress_changed = True
                    # Job status flipped (e.g. Pending → InProgress) is a
                    # redraw trigger — without this the panel stays frozen
                    # on the initial Pending state until the first sample
                    # arrives, which can be many minutes later.
                    if self.job_status != last_status:
                        progress_changed = True
                        last_status = self.job_status
                    # Status message change covers the stream-name resolution
                    # transition ("(stream name pending)" → real CW URL) plus
                    # any later message updates the API surfaces.
                    if self._status_message != last_status_message:
                        progress_changed = True
                        last_status_message = self._status_message
                    # New SecondaryStatusTransitions rows were appended.
                    new_transitions_count = len(
                        self._secondary_status_transitions
                    )
                    if new_transitions_count != last_transitions_count:
                        progress_changed = True
                        last_transitions_count = new_transitions_count
                except ClientError as e:
                    if _is_transient_client_error(e):
                        consecutive_describe_failures += 1
                        if consecutive_describe_failures >= 3:
                            raise
                        # Skip this tick silently — do not advance
                        # ``last_describe`` so we retry on the next poll
                        # rather than waiting another describe interval.
                    else:
                        raise

            # --- Sentinel scan (every poll tick, gated by quiet period) ---------
            # New milestones are also a panel-render trigger — without this,
            # the panel can freeze for an entire phase that produces no
            # samples (e.g., Phase 1 code generation), since sample-count
            # increase would otherwise be the only mid-loop trigger.
            if not self._quiet_period_notified:
                try:
                    for milestone_line in self._scan_for_milestones():
                        self._process_milestone(milestone_line)
                    new_milestone_count = len(self._milestone_transitions)
                    if new_milestone_count != last_milestone_count:
                        progress_changed = True
                        last_milestone_count = new_milestone_count
                except Exception as exc:  # pylint: disable=broad-except
                    # Transient CW Logs failures (stream not yet created,
                    # throttling, network blip) must not kill the panel
                    # loop. Log at DEBUG and continue — the next tick
                    # will retry.
                    logger.debug(
                        "Sentinel scan failed (will retry): %s", exc
                    )

            # --- Terminal check --------------------------------------------------
            if self.job_status in _TERMINAL_STATES:
                # Grace sleep to let CW ingest any trailing events, then
                # one final sentinel scan + sample-count probe so the
                # terminal panel has the complete picture.
                time.sleep(2)
                try:
                    for milestone_line in self._scan_for_milestones():
                        self._process_milestone(milestone_line)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.debug("Terminal sentinel scan failed: %s", exc)
                if self._progress_output_s3_uri:
                    final_count = self._count_samples()
                    if final_count is not None:
                        self._current_samples = final_count
                # Snap sample count to expected on successful completion (see helper docstring).
                self._apply_completion_fallback()
                self._render_terminal_panel(live_console)
                return

            # --- Panel render trigger -------------------------------------------
            # Print on sample count progression, on a freshly observed
            # milestone, or on a periodic clock tick so the Elapsed Time /
            # active-row Duration fields advance during long quiet phases
            # (e.g., multi-minute Code Generation).
            elapsed_since_panel = now - last_panel_time
            # Periodic redraw so Elapsed Time and active-row Durations
            # advance even when no event has fired (e.g. during a
            # multi-minute Code Generation phase). The quiet-period flag
            # only suppresses sentinel scanning — the panel itself must
            # keep ticking so the customer sees live elapsed time even
            # after the backoff message has fired.
            should_tick_redraw = elapsed_since_panel >= describe_interval

            if progress_changed or should_tick_redraw:
                if is_jupyter:
                    self._clear_jupyter_output(wait=True)
                live_console.print(self._render_panel(is_final=False))
                last_panel_time = now
                progress_changed = False

            # --- Quiet_Period -----------------------------------------------------
            if (
                not self._quiet_period_notified
                and (now - last_panel_time) >= _QUIET_PERIOD_SECONDS
            ):
                self._quiet_period_notified = True
                self._emit_backoff_message(
                    console=live_console,
                    elapsed_quiet_seconds=int(now - last_panel_time),
                )
                # Note: do NOT reset last_panel_time here — we want the
                # backoff to fire exactly once. Subsequent ticks see
                # _quiet_period_notified=True and skip the block above.

            # --- Timeout ---------------------------------------------------------
            if timeout is not None and now - start_time >= timeout:
                # Print a panel reflecting the current state BEFORE raising
                # so the customer has a visible record.
                self._render_terminal_panel(
                    live_console, is_timeout=True
                )
                raise timeout_exc_cls(
                    job_arn=self.job_arn or "unknown",
                    elapsed_seconds=now - start_time,
                    job_name=self.job_name,
                )

    def _render_terminal_panel(
        self, console, is_timeout: bool = False
    ) -> None:
        """Render the final / timeout panel.

        In Jupyter, clears the previous live frame so the cell ends with a
        single persisted panel showing the terminal state — so a saved
        notebook reopens to one panel showing the final job state, not a
        stack of progress + final panels.
        """
        if _is_jupyter_environment():
            self._clear_jupyter_output(wait=True)
        console.print(self._render_panel(is_final=True))

    @staticmethod
    def _clear_jupyter_output(wait: bool = True) -> None:
        """Best-effort ``IPython.display.clear_output`` — silently no-ops
        when IPython is not importable (e.g. during unit tests or in a
        plain Python REPL).
        """
        try:
            from IPython.display import clear_output  # type: ignore

            clear_output(wait=wait)
        except Exception:
            pass

    def _emit_backoff_message(
        self, console, elapsed_quiet_seconds: int
    ) -> None:
        """Print the Quiet_Period backoff message exactly once.

        Includes the elapsed quiet duration and a CloudWatch link. The
        substring ``"still running"`` is part of the customer-visible text
        so unit tests can assert the message was emitted via grep.
        """
        minutes = elapsed_quiet_seconds // 60
        seconds = elapsed_quiet_seconds % 60
        log_group = f"{_LOG_GROUP_PREFIX}/{self.job_category}/"
        stream = self._extract_log_stream_name()
        cw_path = (
            f"{log_group}{stream}" if stream else f"{log_group}<stream pending>"
        )
        msg = (
            f"No new activity for {minutes}m {seconds:02d}s. "
            f"Job is still running.\n"
            f"Monitor logs at: {cw_path}\n"
            f"(DescribeJob polling continues every {30}s)"
        )
        try:
            console.print(msg, highlight=False, markup=False)
        except Exception:
            # Best-effort — never let a console rendering glitch crash wait().
            logger.info(msg)

    # ------------------------------------------------------------------
    # logs=True — Log_Tail_Mode
    # ------------------------------------------------------------------

    @staticmethod
    def _is_printable_log_line(line: str) -> bool:
        """Return whether a raw CloudWatch log line should be shown in log-tail mode.

        Customer-relevant lines (INFO, WARNING, ERROR, CRITICAL and
        ``[MILESTONE]`` sentinels) are printed. Container DEBUG lines and
        CloudWatch embedded-metric (EMF) JSON blobs are suppressed. Lines with
        no recognizable level token (for example traceback continuation lines)
        are printed, so error output is never hidden.
        """
        if not line or not line.strip():
            return False
        if "[MILESTONE]" in line:
            return True
        stripped = line.lstrip()
        if stripped.startswith("{") and '"_aws"' in stripped:
            return False
        return not _DEBUG_LEVEL_RE.search(line)

    def _wait_log_tail_mode(
        self,
        poll: int,
        describe_interval: int,
        timeout: Optional[int],
        timeout_exc_cls: type,
    ) -> None:
        """Implementation of ``wait(logs=True)`` — pure log-tailing mode.

        Append-only stream of customer-relevant CW log lines (INFO+ and
        ``[MILESTONE]`` sentinels) printed to the console. No panels are
        rendered during the loop — only a simple summary line is printed
        at the end when a terminal state is reached or timeout fires.

        Cadence:
          * ``poll`` (default 10s) — loop sleep. ``_tail_logs`` runs on
            every poll tick, even after the quiet-period backoff.
          * ``describe_interval`` (default 30s) — gates ``DescribeJob``
            and S3 progress queries so terminal status is detected.
          * Terminal: 2s grace sleep, one final ``_tail_logs`` drain,
            then a simple status summary line.

        Other rules:
          * Each line returned by ``_tail_logs`` is printed if non-empty;
            empty lines are silently dropped.
          * ``[MILESTONE]`` lines are both printed AND fed into
            ``_process_milestone`` so milestone transitions stay in sync.
          * Quiet_Period: emit the backoff message exactly once, but
            continue calling ``_tail_logs`` on subsequent ticks.
          * Transient ``DescribeJob`` failures: skip the tick silently;
            propagate after 3 consecutive failures.
          * Timeout: print status summary before raising
            ``DataPreparerJobTimeoutError``.
        """
        from rich.console import Console

        # Append-only console — used throughout logs=True mode.
        # In Jupyter, use force_jupyter=True so Rich renders hyperlinks as
        # clickable HTML rather than raw OSC 8 escape sequences (which
        # Jupyter's output cell doesn't interpret, causing ]8;...\ noise).
        if _is_jupyter_environment():
            force_terminal_console = Console(force_jupyter=True)
        else:
            force_terminal_console = Console(
                force_jupyter=False, force_terminal=True, width=120
            )

        start_time = self._wait_start_time or time.time()
        last_describe = 0.0
        consecutive_describe_failures = 0

        while True:
            now = time.time()

            # --- Describe tick ---------------------------------------------------
            if now - last_describe >= describe_interval:
                try:
                    self.refresh()
                    consecutive_describe_failures = 0
                    last_describe = now
                    if self._progress_output_s3_uri:
                        new_count = self._count_samples()
                        if (
                            new_count is not None
                            and new_count != self._current_samples
                        ):
                            self._current_samples = new_count
                except ClientError as e:
                    if _is_transient_client_error(e):
                        consecutive_describe_failures += 1
                        if consecutive_describe_failures >= 3:
                            raise
                        # Skip this tick silently — do not advance
                        # ``last_describe`` so we retry on the next poll.
                    else:
                        raise

            # --- Tail and print log lines (every tick, even after quiet) --------
            try:
                self._log_next_token, raw_lines = self._tail_logs(
                    self._log_next_token
                )
                for line in raw_lines:
                    if self._is_printable_log_line(line):
                        force_terminal_console.print(
                            line, highlight=False, markup=False
                        )
                    if line and "[MILESTONE]" in line:
                        self._process_milestone(line)
            except Exception as exc:  # pylint: disable=broad-except
                # Transient CW Logs failures must not kill the loop.
                logger.debug("Log tail failed (will retry): %s", exc)

            # --- Terminal check --------------------------------------------------
            if self.job_status in _TERMINAL_STATES:
                # Grace sleep so CW catches up, then drain any trailing
                # events through the same printability filter.
                time.sleep(2)
                try:
                    self._log_next_token, final_lines = self._tail_logs(
                        self._log_next_token
                    )
                    for line in final_lines:
                        if self._is_printable_log_line(line):
                            force_terminal_console.print(
                                line, highlight=False, markup=False
                            )
                        if line and "[MILESTONE]" in line:
                            self._process_milestone(line)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.debug("Terminal log tail failed: %s", exc)
                if self._progress_output_s3_uri:
                    final_count = self._count_samples()
                    if final_count is not None:
                        self._current_samples = final_count
                # Snap sample count to expected on successful completion (see helper docstring).
                self._apply_completion_fallback()
                force_terminal_console.print(
                    f"\nJob {self.job_name} finished with status: {self.job_status}",
                    highlight=False,
                )
                return

            # --- Quiet_Period (logs=True: emit once, keep tailing) --------------
            if (
                not self._quiet_period_notified
                and (now - start_time) >= _QUIET_PERIOD_SECONDS
            ):
                self._quiet_period_notified = True
                self._emit_backoff_message(
                    console=force_terminal_console,
                    elapsed_quiet_seconds=int(now - start_time),
                )
                # Note: do NOT reset start_time — backoff fires
                # exactly once per wait() invocation. CW polling
                # continues unchanged on subsequent ticks.

            # --- Timeout ---------------------------------------------------------
            if timeout is not None and now - start_time >= timeout:
                # Print status summary before raising so the customer has
                # a visible record.
                force_terminal_console.print(
                    f"\nJob {self.job_name} timed out (status: {self.job_status})",
                    highlight=False,
                )
                raise timeout_exc_cls(
                    job_arn=self.job_arn or "unknown",
                    elapsed_seconds=now - start_time,
                    job_name=self.job_name,
                )

            time.sleep(poll)

    def _tail_logs(self, next_token):
        """Fetch new CloudWatch log events up to steady-state. Returns (next_token, list_of_lines).

        ``get_log_events`` returns at most 1 MB / 10,000 events per call and
        always yields a forward token, so we loop until the token stops
        advancing — this drains everything new in the stream in one poll
        instead of leaving batches for later ticks.

        Log location is derived from the job category and name using the
        service's canonical namespace /aws/sagemaker/Job/{JobCategory}/.
        The stream name is {jobName}/generation-{createdAt} or
        {jobName}/evaluation-{createdAt}, matching the Fluent Bit config.
        If the service surfaces the log location in a future DescribeJob
        field (e.g. CloudWatchLogGroupName / CloudWatchLogStreamName), this
        method should prefer those over the derived values.
        """
        import boto3

        # Derive log group from job category — matches Fluent Bit tail-output.conf
        # NOTE: trailing slash is required — the service creates the log group with it.
        log_group = f"{_LOG_GROUP_PREFIX}/{self.job_category}/"

        # Derive stream prefix from job category — matches LOG_STREAM_SUFFIX in ASL
        category_stream_suffix_map = {
            "SyntheticDataGeneration": "generation",
            "DataQualityEvaluation": "evaluation",
        }
        stream_suffix = category_stream_suffix_map.get(self.job_category, "log")

        # Try to extract the exact stream name from the StatusMessage if the
        # service has already surfaced it (e.g. "stream=<jobName>/generation-<ts>").
        # Falls back to prefix-based discovery if not available.
        log_stream = None
        if self._status_message:
            m = re.search(r"stream=([^\s,]+)", self._status_message)
            if m:
                log_stream = m.group(1)

        if not log_stream:
            return next_token, []

        collected: list = []
        current_token = next_token
        try:
            log_client = boto3.client("logs", region_name=self._session.boto_region_name)
            for _ in range(20):  # safety bound: drain up to 20 pages per poll
                kwargs = {
                    "logGroupName": log_group,
                    "logStreamName": log_stream,
                    "startFromHead": True,
                }
                if current_token:
                    kwargs["nextToken"] = current_token
                resp = log_client.get_log_events(**kwargs)
                events = resp.get("events", [])
                collected.extend(e["message"].rstrip() for e in events)
                new_token = resp.get("nextForwardToken")
                if not events or new_token == current_token:
                    current_token = new_token or current_token
                    break
                current_token = new_token
            return current_token, collected
        except Exception:
            return next_token, collected

    def _count_samples(self) -> Optional[int]:
        """Count rows in the SDG output file via S3 GetObject + newline count.

        The service writes samples under a non-deterministic subdirectory; in
        some flows it inserts an intermediate tier such as ``refined/`` or
        ``samples/`` between ``OutputS3Uri`` and the ``{JobName}-{ts}/`` dir:

            {OutputS3Uri}/[{tier}/]{JobName}-{service-UTC-timestamp}/job_metadata/verified_samples.jsonl
            {OutputS3Uri}/[{tier}/]{JobName}-{service-UTC-timestamp}/output/output_samples.jsonl

        For tabular SDG jobs, the output may be a ``.csv`` file. The
        newline-count approach works identically for CSV (one row per line).

        The subdirectory's exact name (service-assigned timestamp) is not
        known client-side until the container first writes. Matching only by
        ``{JobName}-`` prefix is unsafe because sibling jobs can share a
        string prefix (``my-sdg`` vs ``my-sdg-v2``). We enforce the
        structural requirement that the path component immediately after
        ``OutputS3Uri`` is exactly ``{JobName}-{iso-ish-suffix}`` where the
        suffix starts with a digit (service timestamps always begin with a
        year) and contains a ``T`` (the date/time separator in ISO format).
        Any match that fails this shape is rejected.

        Once we find a matching object, we cache its S3 key on the instance
        and reuse it for subsequent polls — this both speeds up polling and
        ensures we never drift to a different job's output directory.

        Discovery first tries a narrow listing under
        ``{output_s3_uri}/{JobName}-`` so the result set is bounded to
        this job's directory. This avoids exhausting the listing cap
        when older test runs share the same customer prefix. If the
        narrow listing finds nothing, the function falls back to a
        broad listing under ``{output_s3_uri}/`` to cover the case
        where the service inserts an intermediate tier (e.g.
        ``refined/``, ``samples/``) between the customer prefix and
        the ``{JobName}-{ts}/`` subdir.

        **Implementation note — why not S3 Select:** S3 Select
        (``SelectObjectContent``) was deprecated by AWS in 2024 and returns
        ``MethodNotAllowed`` in all regions as of mid-2025. We instead do a
        cheap ``GetObject`` + newline count on the JSONL body (each JSONL
        record is one line by definition), which is accurate and works on
        any S3 bucket. For very large files, swap to a range-read chunked
        counter — not needed at current SDG output volumes.

        Returns ``None`` if progress tracking is not configured, the output
        directory has not yet been created, or the GetObject call fails for
        any transient reason.
        """
        import boto3

        uri = self._progress_output_s3_uri
        if not uri or not uri.startswith("s3://"):
            return None

        rest = uri[len("s3://"):].rstrip("/")
        parts = rest.split("/", 1)
        bucket = parts[0]
        prefix = (parts[1] + "/") if len(parts) > 1 else ""

        s3 = boto3.client("s3", region_name=self._session.boto_region_name)

        def get_line_count(key: str) -> Optional[int]:
            r"""GET s3://{bucket}/{key} and count non-empty lines.

            Uses ``splitlines()`` to handle ``\r\n`` line endings and trailing
            blank lines. Blank lines (whitespace-only) are not counted — they
            are not valid JSONL records. Returns ``None`` on ``NoSuchKey`` so
            callers can distinguish "file not found" from "file found, 0 records".
            """
            try:
                resp = s3.get_object(Bucket=bucket, Key=key)
                body = resp["Body"].read()
                if not body:
                    return 0
                return sum(1 for line in body.splitlines() if line.strip())
            except s3.exceptions.NoSuchKey:
                return None
            except Exception as e:  # pragma: no cover — best-effort telemetry
                logger.debug("GetObject on %s failed: %s", key, e)
                return None

        # Fast path: we already discovered the file on a prior tick. Re-query it.
        if self._progress_target_key:
            count = get_line_count(self._progress_target_key)
            if count is not None:
                return count
            # NoSuchKey: the service may have finalized and moved to the output
            # dir. Re-scan to pick up the new file.
            self._progress_target_key = None

        # Discovery uses two strategies in order:
        #   1. Narrow listing under "{prefix}{JobName}-": always bounds to
        #      this job's output dir (~3–5 files), avoids the lexicographic
        #      cap problem when many older test runs share the same
        #      customer prefix.
        #   2. Broad listing under "{prefix}": fallback that handles the
        #      service-inserted intermediate tier case (e.g. ``refined/``,
        #      ``samples/``) where the {JobName}-{ts}/ subdir does not
        #      live directly under the customer's prefix.
        # The is_ours() segment match is identical for both strategies — it
        # rejects sibling jobs whose names are string prefixes of ours.
        def _list_under(search_prefix: str) -> Optional[List[Dict]]:
            try:
                paginator = s3.get_paginator("list_objects_v2")
                collected: List[Dict] = []
                for page in paginator.paginate(
                    Bucket=bucket,
                    Prefix=search_prefix,
                    PaginationConfig={"MaxItems": 1000},
                ):
                    collected.extend(page.get("Contents", []) or [])
                return collected
            except Exception as e:  # pragma: no cover
                logger.debug(
                    "S3 list_objects_v2 failed (prefix=%s): %s",
                    search_prefix, e,
                )
                return None

        narrow_prefix = f"{prefix}{self.job_name}-"
        objects = _list_under(narrow_prefix)
        if not objects:
            # Fallback for the intermediate-tier case.
            objects = _list_under(prefix)
        if not objects:
            return None

        # Match {JobName}-{ts}/ as a path segment anywhere in the key.
        # The service timestamp suffix always starts with a year digit and
        # contains a `T` (ISO 8601 separator), so the regex
        #     /{JobName}-\d[^/]*T[^/]*/
        # uniquely identifies this job's output dir without colliding with
        # sibling jobs whose JobName is a prefix of ours (e.g. `my-sdg` vs
        # `my-sdg-v2` produce different segments because the `-` after the
        # JobName terminates the first segment).
        segment_re = re.compile(
            r"(?:^|/)" + re.escape(self.job_name) + r"-\d[^/]*T[^/]*/"
        )

        def is_ours(key: str) -> bool:
            return bool(segment_re.search(key))

        # Discover the streaming data file belonging to this job.
        # Priority: verified_samples > evaluated_data > output_samples.
        # Only match line-countable extensions (.jsonl, .csv) — binary
        # formats like .parquet are not trackable via newline counting.
        _COUNTABLE_EXTENSIONS = (".jsonl", ".csv")
        keys = [o["Key"] for o in objects if is_ours(o["Key"])]

        def _is_countable(k: str) -> bool:
            return any(k.endswith(ext) for ext in _COUNTABLE_EXTENSIONS)

        target = next(
            (k for k in keys if "verified_samples" in k.rsplit("/", 1)[-1] and _is_countable(k)),
            None,
        )
        if target is None:
            target = next(
                (k for k in keys if "evaluated_data" in k.rsplit("/", 1)[-1] and _is_countable(k)),
                None,
            )
        if target is None:
            target = next(
                (k for k in keys if "output_samples" in k.rsplit("/", 1)[-1] and _is_countable(k)),
                None,
            )
        if target is None:
            return None

        # Log on first discovery so operators can verify the match.
        if self._progress_target_key != target:
            logger.info(
                "Locked onto sample-progress file for job %s: s3://%s/%s",
                self.job_name, bucket, target,
            )
        self._progress_target_key = target

        return get_line_count(target)

    def _apply_completion_fallback(self) -> None:
        """Snap sample count to expected on successful completion.

        When the service skips writing verified_samples.jsonl (e.g.,
        code-gen-only path where Sample Generation is a no-op), the
        count probe returns None even though all expected samples were
        produced. This surfaces "expected/expected" on the terminal panel
        instead of a misleading "0/expected". Only fires on Completed —
        Failed/Stopped jobs keep whatever count was last observed.

        Importantly, this snap is ONLY applied when the count is None
        (the output file was never written). A count of 0 means the file
        was found but contains zero records — a genuinely empty SDG output
        that should NOT be masked (bug-bash SDG #9: job Completed, empty
        output, panel incorrectly showed N/N instead of 0/N).
        """
        if (
            self.job_status == "Completed"
            and self._expected_samples is not None
            and self._current_samples is None
        ):
            self._current_samples = self._expected_samples

    def stop(self) -> None:
        """Stop the job."""
        self._client.stop_job(JobName=self.job_name, JobCategory=self.job_category)
        logger.info("Stop requested for Job %s", self.job_name)
