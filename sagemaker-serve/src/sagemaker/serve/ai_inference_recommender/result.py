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
from typing import Any, Dict, List, Optional, Sequence
from urllib.parse import urlparse

import boto3

PROFILE_EXPORT_FILENAME = "profile_export_aiperf.json"
OUTPUT_ARCHIVE_FILENAME = "output.tar.gz"
# A concurrency search / magic-list sweep writes this at the artifact root
# instead of a single top-level profile_export_aiperf.json. Its presence in the
# archive is how we tell a sweep run apart from a single run.
SEARCH_HISTORY_FILENAME = "search_history.json"

# Statistic columns exposed by ``to_dataframe()``. Superset of the printed
# table's avg/p50/p90/p99 — the DataFrame also carries min/max/p95/stddev,
# which the text table omits for width.
_METRIC_STAT_COLUMNS = ("avg", "min", "max", "p50", "p90", "p95", "p99", "stddev")


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

    def _ordered_metric_pairs(self):
        """(name, metric) pairs in display order: non-HTTP metrics alphabetically,
        then ``http_*`` transport metrics last. Shared by ``__str__`` and
        ``to_dataframe()`` so the printed table and the frame stay in sync."""
        rest, http = [], []
        for name in sorted(self.all_metrics):
            bucket = http if name.startswith("http_") else rest
            bucket.append((name, self.all_metrics[name]))
        return rest + http

    def __str__(self) -> str:
        return _format_metrics_table(self._ordered_metric_pairs())

    def __repr__(self) -> str:
        return f"BenchmarkMetrics({len(self.all_metrics)} metrics; print() for the table)"

    def _repr_pretty_(self, p, cycle):
        # Render the full table in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))

    def to_dataframe(self):
        """Return the metrics as a pandas ``DataFrame`` indexed by metric name.

        One row per metric, one column per statistic (``unit`` plus
        ``avg``/``min``/``max``/``p50``/``p90``/``p95``/``p99``/``stddev``).
        Rows are ordered exactly as the printed table: non-HTTP metrics
        alphabetically, then ``http_*`` transport metrics last. Unlike the text
        table, the frame keeps ``min``/``max``/``p95``/``stddev``.

        Requires pandas.
        """
        pd = _require_pandas()
        return _metrics_dataframe(pd, self._ordered_metric_pairs())

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
        table = _format_metrics_table(self._ordered_metric_pairs())
        return (
            f"BenchmarkResult\n"
            f"  endpoint:           {self.endpoint or '-'}\n"
            f"  workload_config:    {self.workload_config or '-'}\n"
            f"  tool_version:       {self.tool_version or '-'}\n"
            f"  s3_output_location: {self.s3_output_location}\n"
            f"  metrics:\n{_indent(table, '    ')}\n"
            f"  raw profile available via .profile"
        )

    def _ordered_metric_pairs(self):
        """(name, metric) pairs in display order: well-known headline metrics
        first (canonical order), then the rest alphabetized, then ``http_*``
        transport metrics last. Shared by ``__str__`` and ``to_dataframe()`` so
        the printed table and the frame stay in the same order."""
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
        return headline + rest + http

    def to_dataframe(self):
        """Return this result's metrics as a pandas ``DataFrame``.

        One row per metric (indexed by metric name), one column per statistic —
        the same shape as :meth:`BenchmarkMetrics.to_dataframe`, but ordered as
        this result prints: headline metrics first, then the rest alphabetized,
        then ``http_*`` transport metrics last.

        A search/sweep result has no single metric profile; call
        ``result.search`` for its outcome instead.

        Requires pandas.
        """
        pd = _require_pandas()
        if self.search is not None:
            raise ValueError(
                "This is a search/sweep result with no single metric profile to "
                "tabulate. Inspect result.search for the sweep outcome instead."
            )
        return _metrics_dataframe(pd, self._ordered_metric_pairs())

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
        if job.output_config is None or not getattr(job.output_config, "s3_output_location", None):
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
    raise FileNotFoundError(f"No object ending in {suffix!r} under s3://{bucket}/{prefix}")


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


def _require_pandas():
    """Import pandas lazily, only when ``to_dataframe()`` is called, so this
    module's printed tables stay stdlib-only."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - trivial re-raise
        raise ImportError(
            "to_dataframe() requires pandas, which is not installed. "
            "Install it with `pip install pandas`."
        ) from exc
    return pd


def _fmt_number(value: Optional[float]) -> str:
    """Render a number compact for the metrics table; '-' for None."""
    if value is None:
        return "-"
    if abs(value) >= 100:
        return f"{value:.1f}"
    return f"{value:.3g}"


def _indent(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line else line for line in text.splitlines())


def _coerce_numeric(pd, frame, numeric_cols):
    """Cast the named columns to float64 so an all-missing column is ``NaN``,
    not ``object`` holding ``None`` (on which sort/nlargest/mean would raise)."""
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _metrics_dataframe(pd, name_metric_pairs):
    """Build a metric-indexed DataFrame from (name, BenchmarkMetric) pairs.

    Columns are ``unit`` plus every stat in ``_METRIC_STAT_COLUMNS``; row order
    follows the input. Empty input yields an empty frame with the same columns.
    """
    columns = ["unit", *_METRIC_STAT_COLUMNS]
    names = [name for name, _ in name_metric_pairs]
    data = [
        {
            "unit": metric.unit,
            **{stat: getattr(metric, stat, None) for stat in _METRIC_STAT_COLUMNS},
        }
        for _name, metric in name_metric_pairs
    ]
    frame = pd.DataFrame(data, columns=columns, index=pd.Index(names, name="metric"))
    return _coerce_numeric(pd, frame, _METRIC_STAT_COLUMNS)


def _format_metrics_table(name_metric_pairs) -> str:
    """Render an iterable of (name, BenchmarkMetric) pairs as a table."""
    rows = []
    for _name, metric in name_metric_pairs:
        rows.append(
            [
                metric.name,
                metric.unit or "-",
                _fmt_number(metric.avg),
                _fmt_number(metric.p50),
                _fmt_number(metric.p90),
                _fmt_number(metric.p99),
            ]
        )
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
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in str_rows
    )
    return f"{header_line}\n{sep_line}\n{body}"


# Per-metric direction, used to orient a delta so "+" reads as an improvement.
# Metrics absent from both sets (sequence lengths, token counts, HTTP counters)
# have no better/worse direction, so they get no signed delta.
_HIGHER_IS_BETTER = frozenset(
    {
        "request_throughput",
        "output_token_throughput",
        "e2e_output_token_throughput",
    }
)
_LOWER_IS_BETTER = frozenset(
    {
        "request_latency",
        "time_to_first_token",
        "inter_token_latency",
        "benchmark_duration",
    }
)


@dataclass
class BenchmarkComparison:
    """Side-by-side comparison of two or more ``BenchmarkResult`` runs.

    The first run is the baseline; each other run's ``__str__`` shows its value
    for every key metric alongside the percentage change from the baseline,
    signed so a ``+`` is always an improvement (higher throughput / lower
    latency) regardless of the metric's direction.

    Attributes:
        results: the compared results, baseline first.
        names: display label per result (defaults to ``run1``, ``run2``, ...).
        stat: which per-metric statistic is compared (``avg`` by default; any of
            ``avg``/``p50``/``p90``/``p95``/``p99``/``min``/``max``).
    """

    results: List["BenchmarkResult"]
    names: List[str]
    stat: str = "avg"

    def _metric_names(self) -> List[str]:
        """Key metrics first (in canonical order), then any other metric present
        in at least one run — so the table covers everything, headline first."""
        ordered = [
            name
            for name in _KEY_METRIC_FIELDS
            if any(name in r.metrics.all_metrics for r in self.results)
        ]
        seen = set(ordered)
        for r in self.results:
            for name in sorted(r.metrics.all_metrics):
                if name not in seen:
                    ordered.append(name)
                    seen.add(name)
        return ordered

    def _value(self, result: "BenchmarkResult", metric_name: str) -> Optional[float]:
        metric = result.metrics.all_metrics.get(metric_name)
        return getattr(metric, self.stat, None) if metric is not None else None

    def _delta_value(self, metric_name: str, baseline, value) -> Optional[float]:
        """Signed percentage change vs. baseline, oriented so ``+`` is better.

        Only computed for metrics with a known direction (throughput = higher
        better, latency/duration = lower better). A directionless metric
        (sequence length, token count, HTTP transport counter) has no
        better/worse orientation, so a signed "improvement" delta would be
        misleading under the ``+Δ = better`` header — it returns ``None``.
        ``None`` is also returned when the delta is undefined (missing value or
        zero baseline). ``__str__`` formats it; ``to_dataframe()`` keeps it
        numeric (``NaN`` for ``None``).
        """
        if metric_name not in _HIGHER_IS_BETTER and metric_name not in _LOWER_IS_BETTER:
            return None
        if baseline is None or value is None or baseline == 0:
            return None
        pct = (value - baseline) / abs(baseline) * 100.0
        if metric_name in _LOWER_IS_BETTER:
            # Lower-is-better: flip the sign so a drop reads as +.
            pct = -pct
        return pct

    def _delta_cell(self, metric_name: str, baseline, value) -> str:
        """Signed percentage change vs. baseline as a display string; ``-`` when
        no oriented delta applies (directionless or undefined)."""
        pct = self._delta_value(metric_name, baseline, value)
        return "-" if pct is None else f"{pct:+.1f}%"

    def _unit_for(self, metric_name: str) -> Optional[str]:
        for r in self.results:
            m = r.metrics.all_metrics.get(metric_name)
            if m is not None and m.unit:
                return m.unit
        return None

    def __str__(self) -> str:
        metric_names = self._metric_names()
        if not metric_names:
            return "BenchmarkComparison (no metrics to compare)"

        # Columns: metric, unit, one value column per run, and a Δ% column per
        # non-baseline run (vs. the baseline, the first run).
        headers = ["metric", "unit"]
        headers += list(self.names)
        headers += [f"Δ% {name}" for name in self.names[1:]]

        rows = []
        for metric_name in metric_names:
            unit = self._unit_for(metric_name) or "-"
            values = [self._value(r, metric_name) for r in self.results]
            row = [metric_name, unit] + [_fmt_number(v) for v in values]
            baseline = values[0]
            row += [self._delta_cell(metric_name, baseline, v) for v in values[1:]]
            rows.append(row)

        table = _format_table(headers=headers, rows=rows)
        baseline_note = f"baseline: {self.names[0]}  |  stat: {self.stat}  (+Δ = better)"
        return f"BenchmarkComparison\n  {baseline_note}\n{_indent(table, '  ')}"

    def to_dataframe(self):
        """Return the comparison as a pandas ``DataFrame``.

        Mirrors the printed table: one row per metric (indexed by metric name),
        a ``unit`` column, one column per run (named by :attr:`names`, holding
        the compared ``stat``), and a ``Δ% <run>`` column per non-baseline run —
        signed so ``+`` is always an improvement. Delta values are numeric
        percentages (``NaN`` where undefined), not preformatted strings.

        Requires pandas.
        """
        pd = _require_pandas()
        columns = ["unit", *self.names, *(f"Δ% {name}" for name in self.names[1:])]
        metric_names = self._metric_names()
        data = []
        for metric_name in metric_names:
            values = [self._value(r, metric_name) for r in self.results]
            baseline = values[0]
            record = {"unit": self._unit_for(metric_name)}
            for name, value in zip(self.names, values):
                record[name] = value
            for name, value in zip(self.names[1:], values[1:]):
                record[f"Δ% {name}"] = self._delta_value(metric_name, baseline, value)
            data.append(record)
        frame = pd.DataFrame(data, columns=columns, index=pd.Index(metric_names, name="metric"))
        return _coerce_numeric(pd, frame, [c for c in columns if c != "unit"])

    def __repr__(self) -> str:
        return (
            f"BenchmarkComparison({len(self.results)} runs: "
            f"{', '.join(self.names)}; print() for the table)"
        )

    def _repr_pretty_(self, p, cycle):
        p.text("..." if cycle else str(self))


def compare_benchmarks(
    *results: "BenchmarkResult",
    names: Optional[Sequence[str]] = None,
    stat: str = "avg",
) -> BenchmarkComparison:
    """Compare two or more benchmark runs and return a tabular comparison.

    The first result is the baseline; each subsequent run is reported with a
    signed percentage change from it (oriented so ``+`` is always better —
    higher throughput or lower latency). ``print()`` the returned object for the
    table.

    Args:
        *results: two or more ``BenchmarkResult`` objects (from
            ``job.show_result()``). The first is the baseline.
        names: optional display label per result; defaults to ``run1``,
            ``run2``, .... Must match the number of results when given.
        stat: which per-metric statistic to compare — one of ``avg`` (default),
            ``p50``, ``p90``, ``p95``, ``p99``, ``min``, ``max``.

    Returns:
        BenchmarkComparison: renders a metric-by-run table with per-run deltas.

    Raises:
        ValueError: if fewer than two results are given, if ``names`` length
            does not match, if ``stat`` is not a known statistic, or if any
            result is a concurrency-search/sweep run (which has no single metric
            profile to compare).
    """
    if len(results) < 2:
        raise ValueError("compare_benchmarks() needs at least two results to compare.")
    if any(r.is_search for r in results):
        raise ValueError(
            "compare_benchmarks() compares single-run results; a search/sweep "
            "result has no single metric profile. Compare the winning runs instead."
        )
    valid_stats = {"avg", "p50", "p90", "p95", "p99", "min", "max"}
    if stat not in valid_stats:
        raise ValueError(f"stat must be one of {sorted(valid_stats)}, got {stat!r}.")
    if names is not None:
        if len(names) != len(results):
            raise ValueError(
                f"names has {len(names)} entries but {len(results)} results were given."
            )
        labels = list(names)
        # Each name is a distinct DataFrame column; a duplicate or "unit" would
        # collide and make the frame disagree with the printed table.
        if len(set(labels)) != len(labels):
            duplicates = sorted({n for n in labels if labels.count(n) > 1})
            raise ValueError(f"names must be unique; duplicated: {duplicates}.")
        if "unit" in labels:
            raise ValueError("names cannot include the reserved column name 'unit'.")
    else:
        labels = [f"run{i + 1}" for i in range(len(results))]
    return BenchmarkComparison(results=list(results), names=labels, stat=stat)
