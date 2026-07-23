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
"""Parsing of benchmark output artifacts from S3."""
from __future__ import absolute_import

import io
import json
import tarfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import boto3

PROFILE_EXPORT_FILENAME = "profile_export_aiperf.json"
OUTPUT_ARCHIVE_FILENAME = "output.tar.gz"
# A concurrency search / magic-list sweep writes this at the artifact root
# instead of a single top-level profile_export_aiperf.json. Its presence in the
# archive is how we tell a sweep run apart from a single run.
SEARCH_HISTORY_FILENAME = "search_history.json"


@dataclass
class BenchmarkMetric:
    """A single benchmark metric with its statistical aggregates."""

    name: str
    unit: Optional[str] = None
    avg: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    p50: Optional[float] = None
    p90: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None
    stddev: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "BenchmarkMetric":
        return cls(
            name=name,
            unit=data.get("unit"),
            avg=_as_float(data.get("avg")),
            min=_as_float(data.get("min")),
            max=_as_float(data.get("max")),
            p50=_as_float(data.get("p50")),
            p90=_as_float(data.get("p90")),
            p95=_as_float(data.get("p95")),
            p99=_as_float(data.get("p99")),
            stddev=_as_float(data.get("stddev") or data.get("std")),
            raw=dict(data),
        )


@dataclass
class BenchmarkMetrics:
    """Typed access to the well-known AIPerf metrics.

    Use ``.get(name)`` to look up any metric by its raw key. ``print()``-ing
    this object renders every metric in a table; ``print(result)`` (the
    parent ``BenchmarkResult``) shows just the well-known metrics.
    """

    request_throughput: Optional[BenchmarkMetric] = None
    request_latency: Optional[BenchmarkMetric] = None
    time_to_first_token: Optional[BenchmarkMetric] = None
    inter_token_latency: Optional[BenchmarkMetric] = None
    output_token_throughput: Optional[BenchmarkMetric] = None
    all_metrics: Dict[str, BenchmarkMetric] = field(default_factory=dict)

    def get(self, name: str) -> Optional[BenchmarkMetric]:
        return self.all_metrics.get(name)

    def __str__(self) -> str:
        rest, http = [], []
        for name in sorted(self.all_metrics):
            bucket = http if name.startswith("http_") else rest
            bucket.append((name, self.all_metrics[name]))
        return _format_metrics_table(rest + http)

    def __repr__(self) -> str:
        return f"BenchmarkMetrics({len(self.all_metrics)} metrics; print() for the table)"

    def _repr_pretty_(self, p, cycle):
        # Render the full table in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))

    @classmethod
    def from_profile_json(cls, profile: Dict[str, Any]) -> "BenchmarkMetrics":
        all_metrics: Dict[str, BenchmarkMetric] = {}
        for key, value in profile.items():
            if isinstance(value, dict) and any(
                f in value for f in ("avg", "min", "max", "p50", "p90", "p99")
            ):
                all_metrics[key] = BenchmarkMetric.from_dict(key, value)

        return cls(
            request_throughput=all_metrics.get("request_throughput"),
            request_latency=all_metrics.get("request_latency"),
            time_to_first_token=all_metrics.get("time_to_first_token"),
            inter_token_latency=all_metrics.get("inter_token_latency"),
            output_token_throughput=all_metrics.get("output_token_throughput"),
            all_metrics=all_metrics,
        )


_KEY_METRIC_FIELDS = (
    "request_throughput",
    "request_latency",
    "time_to_first_token",
    "inter_token_latency",
    "output_token_throughput",
    "e2e_output_token_throughput",
    "input_sequence_length",
    "output_sequence_length",
    "benchmark_duration",
)


@dataclass
class BenchmarkSearchResult:
    """Outcome of a concurrency search / magic-list sweep benchmark.

    A search run does not produce a single ``profile_export_aiperf.json``; it
    sweeps a dimension (typically concurrency) and records the outcome in
    ``search_history.json``. This captures the parts callers care about: which
    swept value won, and the raw history for anything deeper.

    Attributes:
        swept_dim: dotted path of the swept dimension, e.g.
            ``"phases.profiling.concurrency"``.
        winner: the largest feasible swept value (``boundary_summary.feasible_max.value``).
            ``None`` if no feasible point was found (e.g. every level breached the SLA).
        winner_objective: the objective value at the winning point, when reported.
        infeasible_min: the smallest swept value that breached a constraint, if any.
        first_breach: details of the constraint the ``infeasible_min`` level breached
            (metric tag / stat / threshold / observed), if reported.
        raw: the full parsed ``search_history.json`` for callers who need more.
    """

    swept_dim: Optional[str] = None
    winner: Optional[float] = None
    winner_objective: Optional[float] = None
    infeasible_min: Optional[float] = None
    first_breach: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_history_json(cls, history: Dict[str, Any]) -> "BenchmarkSearchResult":
        # boundary_summary is present for a single-dimension search with a
        # resolved boundary; it is null/absent for a multi-dim search or one
        # that never ran, in which case we still return a result carrying the
        # raw history rather than fabricating a winner.
        boundary = history.get("boundary_summary")
        if not isinstance(boundary, dict):
            return cls(raw=history)
        feasible_max = boundary.get("feasible_max")
        if not isinstance(feasible_max, dict):
            feasible_max = {}
        infeasible = boundary.get("infeasible_min")
        if not isinstance(infeasible, dict):
            infeasible = {}
        first_breach = infeasible.get("first_breach")
        return cls(
            swept_dim=boundary.get("swept_dim_path"),
            winner=_as_float(feasible_max.get("value")),
            winner_objective=_as_float(feasible_max.get("objective_value")),
            infeasible_min=_as_float(infeasible.get("value")),
            first_breach=first_breach if isinstance(first_breach, dict) else None,
            raw=history,
        )

    def __str__(self) -> str:
        breach = ""
        if self.infeasible_min is not None:
            metric = (self.first_breach or {}).get("metric_tag", "?")
            breach = f"\n  first_breach:       {metric} at {self.infeasible_min}"
        return (
            f"BenchmarkSearchResult\n"
            f"  swept_dim:          {self.swept_dim or '-'}\n"
            f"  winner:             {_fmt_number(self.winner)}\n"
            f"  winner_objective:   {_fmt_number(self.winner_objective)}"
            f"{breach}\n"
            f"  raw history available via .raw"
        )

    def __repr__(self) -> str:
        return f"BenchmarkSearchResult(swept_dim={self.swept_dim!r}, winner={self.winner!r})"

    def _repr_pretty_(self, p, cycle):
        # Render the full summary in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))


@dataclass
class BenchmarkResult:
    """Parsed result of a completed benchmark job.

    For a single run, ``metrics``/``profile`` carry the AIPerf profile export
    and ``search`` is ``None``. For a concurrency search / magic-list sweep,
    ``search`` carries the sweep outcome (the winning level); ``metrics`` is
    empty and ``profile`` holds the raw ``search_history.json`` — a sweep has
    no single headline profile to report.
    """

    metrics: BenchmarkMetrics
    s3_output_location: str
    endpoint: Optional[str] = None
    workload_config: Optional[str] = None
    tool_version: Optional[str] = None
    profile: Dict[str, Any] = field(default_factory=dict)
    search: Optional[BenchmarkSearchResult] = None

    @property
    def is_search(self) -> bool:
        """True if this result came from a concurrency search / sweep run."""
        return self.search is not None

    def __str__(self) -> str:
        # A search/sweep run has no single headline profile; render the sweep
        # outcome (winning level) instead of an (empty) metrics table.
        if self.search is not None:
            return (
                f"BenchmarkResult (search)\n"
                f"  endpoint:           {self.endpoint or '-'}\n"
                f"  workload_config:    {self.workload_config or '-'}\n"
                f"  tool_version:       {self.tool_version or '-'}\n"
                f"  s3_output_location: {self.s3_output_location}\n"
                f"  search:\n{_indent(str(self.search), '    ')}"
            )
        # Order: well-known headline metrics first, then everything else
        # alphabetized, then HTTP-level transport metrics last (they're
        # noise for most readers, useful only for debugging).
        seen = set()
        headline = []
        for name in _KEY_METRIC_FIELDS:
            metric = self.metrics.all_metrics.get(name)
            if metric is not None:
                headline.append((name, metric))
                seen.add(name)

        rest, http = [], []
        for name in sorted(self.metrics.all_metrics):
            if name in seen:
                continue
            bucket = http if name.startswith("http_") else rest
            bucket.append((name, self.metrics.all_metrics[name]))

        ordered = headline + rest + http
        table = _format_metrics_table(ordered)
        return (
            f"BenchmarkResult\n"
            f"  endpoint:           {self.endpoint or '-'}\n"
            f"  workload_config:    {self.workload_config or '-'}\n"
            f"  tool_version:       {self.tool_version or '-'}\n"
            f"  s3_output_location: {self.s3_output_location}\n"
            f"  metrics:\n{_indent(table, '    ')}\n"
            f"  raw profile available via .profile"
        )

    def __repr__(self) -> str:
        return (
            f"BenchmarkResult(endpoint={self.endpoint!r}, "
            f"metrics={len(self.metrics.all_metrics)}; print() for the table)"
        )

    def _repr_pretty_(self, p, cycle):
        # Render the full table in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))

    @classmethod
    def from_job(
        cls,
        job,
        *,
        session: Optional[boto3.session.Session] = None,
    ) -> "BenchmarkResult":
        """Download and parse the benchmark output for a completed ``AIBenchmarkJob``.

        Populates ``endpoint``, ``workload_config``, and ``tool_version`` from
        the job's ``BenchmarkTarget`` and ``WorkloadConfigIdentifier`` plus the
        AIPerf profile metadata so the parsed result is self-describing.

        Args:
            job: An ``AIBenchmarkJob`` (or ``BenchmarkJob`` re-export) that has
                reached a terminal state.
            session: Optional boto3 session. Defaults to the ambient session.

        Returns:
            A parsed ``BenchmarkResult``.

        Raises:
            RuntimeError: if the job has no S3 output location set.
        """
        # Refresh unless the job is already known-terminal, so a stale
        # create-time status/output is not read.
        terminal_states = ("Completed", "Failed", "Stopped")
        if (
            getattr(job, "ai_benchmark_job_status", None) not in terminal_states
            or job.output_config is None
            or not getattr(job.output_config, "s3_output_location", None)
        ):
            job.refresh()
        status = job.ai_benchmark_job_status
        if status in ("InProgress", "Pending"):
            raise RuntimeError(
                f"AIBenchmarkJob {job.get_name()} has not finished "
                f"(status={status}). Call job.wait() (or pass wait=True to "
                f"start_benchmark) before BenchmarkResult.from_job()."
            )
        if job.output_config is None or not getattr(
            job.output_config, "s3_output_location", None
        ):
            failure_reason = getattr(job, "failure_reason", None)
            hint = (
                f"Job failed: {failure_reason or 'no reason provided'}."
                if status == "Failed"
                else "Job produced no S3 output."
            )
            raise RuntimeError(
                f"AIBenchmarkJob {job.get_name()} has no S3OutputLocation "
                f"(status={status}). {hint}"
            )
        workload_config = getattr(job, "ai_workload_config_identifier", None)
        return cls.from_s3(
            job.output_config.s3_output_location,
            session=session,
            endpoint=_extract_endpoint(job),
            # Normalize falsy sentinels (e.g. unset optional fields) to None
            # so the result renders cleanly when fields are missing.
            workload_config=workload_config or None,
        )

    @classmethod
    def from_s3(
        cls,
        s3_output_location: str,
        *,
        session: Optional[boto3.session.Session] = None,
        endpoint: Optional[str] = None,
        workload_config: Optional[str] = None,
    ) -> "BenchmarkResult":
        """Download and parse the benchmark output artifact from S3.

        Args:
            s3_output_location: ``s3://bucket/prefix/`` location written by
                the benchmark job.
            session: Optional boto3 session. Defaults to the ambient session.
            endpoint: Optional endpoint identifier to attach to the result.
                Threaded through by :meth:`from_job`.
            workload_config: Optional workload-config identifier to attach.
                Threaded through by :meth:`from_job`.

        Returns:
            A parsed ``BenchmarkResult``.
        """
        bucket, prefix = _parse_s3_uri(s3_output_location)
        s3 = (session or boto3).client("s3")
        archive_key = _find_object(s3, bucket, prefix, OUTPUT_ARCHIVE_FILENAME)
        body = s3.get_object(Bucket=bucket, Key=archive_key)["Body"].read()

        # A concurrency search / magic-list sweep writes search_history.json at
        # the artifact root and NO top-level profile_export_aiperf.json (each
        # swept level has its own per-trial profile export nested in a subdir).
        # Check for the search history FIRST: those per-trial exports share the
        # profile_export_aiperf.json name, so a plain suffix match would
        # otherwise silently return one arbitrary level's metrics as if they
        # were the whole benchmark.
        history_bytes = _read_member_from_tar_gz(body, SEARCH_HISTORY_FILENAME)
        if history_bytes is not None:
            history = json.loads(history_bytes.decode("utf-8"))
            return cls(
                metrics=BenchmarkMetrics.from_profile_json({}),
                s3_output_location=s3_output_location,
                endpoint=endpoint,
                workload_config=workload_config,
                tool_version=_extract_tool_version(history),
                profile=history,
                search=BenchmarkSearchResult.from_history_json(history),
            )

        profile_bytes = _read_member_from_tar_gz(body, PROFILE_EXPORT_FILENAME)
        if profile_bytes is None:
            raise FileNotFoundError(
                f"Neither {PROFILE_EXPORT_FILENAME} nor {SEARCH_HISTORY_FILENAME} "
                f"found in s3://{bucket}/{archive_key}"
            )
        profile = json.loads(profile_bytes.decode("utf-8"))
        return cls(
            metrics=BenchmarkMetrics.from_profile_json(profile),
            s3_output_location=s3_output_location,
            endpoint=endpoint,
            workload_config=workload_config,
            tool_version=_extract_tool_version(profile),
            profile=profile,
        )


def _extract_endpoint(job) -> Optional[str]:
    target = getattr(job, "benchmark_target", None) or None
    endpoint = (getattr(target, "endpoint", None) or None) if target else None
    identifier = getattr(endpoint, "identifier", None) if endpoint else None
    return identifier or None


def _extract_tool_version(profile: Dict[str, Any]) -> Optional[str]:
    """Best-effort lookup of the AIPerf tool version from the profile JSON.

    AIPerf has no single canonical key; we check a few plausible top-level
    locations and return the first string we find.
    """
    for key in ("aiperf_version", "tool_version", "version"):
        value = profile.get(key)
        if isinstance(value, str):
            return value
    meta = profile.get("metadata") or profile.get("meta") or {}
    if isinstance(meta, dict):
        for key in ("aiperf_version", "tool_version", "version"):
            value = meta.get(key)
            if isinstance(value, str):
                return value
    return None


def _parse_s3_uri(uri: str) -> tuple:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _find_object(s3_client, bucket: str, prefix: str, suffix: str) -> str:
    # Paginate: a shared/reused output prefix can hold more than one page
    # (1000 keys), and the target may sit beyond the first page.
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj.get("Key", "")
            if key.endswith(suffix):
                return key
    raise FileNotFoundError(
        f"No object ending in {suffix!r} under s3://{bucket}/{prefix}"
    )


def _read_member_from_tar_gz(archive_bytes: bytes, suffix: str) -> Optional[bytes]:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(suffix):
                fh = tar.extractfile(member)
                if fh is not None:
                    return fh.read()
    return None


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Optional[float]) -> str:
    """Render a number compact for the metrics table; '-' for None."""
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _format_metrics_table(name_metric_pairs) -> str:
    """Render an iterable of (name, BenchmarkMetric) pairs as a table."""
    rows = []
    for _name, metric in name_metric_pairs:
        rows.append([
            metric.name,
            metric.unit or "-",
            _fmt_number(metric.avg),
            _fmt_number(metric.p50),
            _fmt_number(metric.p90),
            _fmt_number(metric.p99),
        ])
    return _format_table(
        headers=["metric", "unit", "avg", "p50", "p90", "p99"],
        rows=rows,
    )


def _format_table(headers, rows) -> str:
    """Tiny stdlib-only table formatter. No external deps.

    Returns a str like:

        metric              unit  avg     p50    p90   p99
        ──────────────────  ────  ──────  ─────  ────  ────
        request_throughput  -     0.169   -      -     -
        request_latency     ms    5896    408    5989  50247
    """
    if not rows:
        return "(no metrics)"
    widths = [len(str(h)) for h in headers]
    str_rows = [[str(c) for c in row] for row in rows]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    header_line = "  ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers))
    sep_line = "  ".join("─" * widths[i] for i in range(len(headers)))
    body = "\n".join(
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        for row in str_rows
    )
    return f"{header_line}\n{sep_line}\n{body}"
