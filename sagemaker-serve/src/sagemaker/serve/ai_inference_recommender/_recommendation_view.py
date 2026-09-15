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
"""Pretty-printing wrapper over an AIRecommendation row.

Wraps each row to replace the default repr without owning the data;
attribute access forwards to the underlying shape transparently.
"""
from __future__ import absolute_import

from collections import defaultdict
from typing import Any, Dict, List, Optional

from sagemaker.serve.ai_inference_recommender.result import (
    _coerce_numeric,
    _fmt_number,
    _format_table,
    _indent,
    _require_pandas,
)


class _ExpectedPerformanceMetric:
    """Aggregated stats for a single metric in ``expected_performance``.

    Each metric on the recommendation row is reported as one or more rows
    keyed by ``stat`` (avg, p50, p90, p99, ...). This view groups the rows
    so customers can do ``rec.expected_performance.request_throughput.avg``
    or ``.p99`` directly.
    """

    __slots__ = ("_stats", "unit")

    def __init__(self, stats: Dict[str, float], unit: Optional[str]):
        object.__setattr__(self, "_stats", stats)
        object.__setattr__(self, "unit", unit)

    @property
    def avg(self) -> Optional[float]:
        return self._stats.get("avg")

    @property
    def p50(self) -> Optional[float]:
        return self._stats.get("p50")

    @property
    def p90(self) -> Optional[float]:
        return self._stats.get("p90")

    @property
    def p99(self) -> Optional[float]:
        return self._stats.get("p99")

    @property
    def stats(self) -> Dict[str, float]:
        return dict(self._stats)

    def __repr__(self) -> str:
        parts = ", ".join(f"{stat}={_fmt_number(v)}" for stat, v in self._stats.items())
        unit = f" {self.unit}" if self.unit else ""
        return f"<{parts}{unit}>"


class _ExpectedPerformanceView:
    """Typed + dict-style accessor over a recommendation's expected_performance.

    Service shape is ``List[AIRecommendationPerformanceMetric]`` with one row
    per (metric, stat). This view groups rows by metric name so customers
    can do ``view.request_throughput.avg`` (snake_case attribute), or
    ``view.get("RequestThroughput").p99`` (raw service name).
    """

    __slots__ = ("_by_metric",)

    def __init__(self, raw_rows: Optional[List[Any]]):
        by_metric: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"unit": None, "stats": {}})
        for row in raw_rows or []:
            metric = getattr(row, "metric", None)
            if not metric:
                continue
            stat = getattr(row, "stat", None) or "value"
            value = _to_float(getattr(row, "value", None))
            if value is None:
                continue
            entry = by_metric[metric]
            entry["stats"][stat] = value
            unit = getattr(row, "unit", None)
            if unit and not entry["unit"]:
                entry["unit"] = unit

        compiled: Dict[str, _ExpectedPerformanceMetric] = {
            name: _ExpectedPerformanceMetric(entry["stats"], entry["unit"])
            for name, entry in by_metric.items()
        }
        object.__setattr__(self, "_by_metric", compiled)

    def get(self, name: str) -> Optional[_ExpectedPerformanceMetric]:
        """Look up a metric by raw service name (e.g. ``"RequestThroughput"``)."""
        return self._by_metric.get(name)

    def __getattr__(self, name: str) -> _ExpectedPerformanceMetric:
        # snake_case attribute access. Translate to CamelCase service name.
        service_name = _snake_to_camel(name)
        metric = self._by_metric.get(service_name) or self._by_metric.get(name)
        if metric is None:
            raise AttributeError(
                f"No expected_performance metric named {name!r}. "
                f"Available: {sorted(self._by_metric)}"
            )
        return metric

    def __contains__(self, name: str) -> bool:
        return name in self._by_metric or _snake_to_camel(name) in self._by_metric

    def __iter__(self):
        return iter(self._by_metric)

    def keys(self):
        return self._by_metric.keys()

    def items(self):
        return self._by_metric.items()

    def values(self):
        return self._by_metric.values()

    def __len__(self) -> int:
        return len(self._by_metric)

    def __repr__(self) -> str:
        return (
            "{" + ", ".join(f"{name}: {metric!r}" for name, metric in self._by_metric.items()) + "}"
        )


def _to_float(value):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _snake_to_camel(name: str) -> str:
    return "".join(word.capitalize() for word in name.split("_"))


class _RecommendationView:
    """Read-only view over a single recommendation row."""

    __slots__ = ("_raw", "_index", "_expected_performance")

    def __init__(self, raw, index: int = 0):
        # Use object.__setattr__ to avoid triggering __getattr__ during init.
        object.__setattr__(self, "_raw", raw)
        object.__setattr__(self, "_index", index)
        object.__setattr__(
            self,
            "_expected_performance",
            _ExpectedPerformanceView(getattr(raw, "expected_performance", None)),
        )

    @property
    def raw(self):
        """The underlying ``AIRecommendation`` shape."""
        return self._raw

    @property
    def expected_performance(self) -> _ExpectedPerformanceView:
        """Typed + dict-style accessor for the recommendation's expected metrics."""
        return self._expected_performance

    @property
    def recommendation_spec_name(self) -> Optional[str]:
        """The recommendation's spec name.

        Same value passed to ``mb.deploy(recommendation_spec_name=...)``.
        """
        md = getattr(self._raw, "model_details", None)
        return getattr(md, "inference_specification_name", None) if md else None

    def __getattr__(self, name):
        return getattr(self._raw, name)

    def __str__(self) -> str:
        md = getattr(self._raw, "model_details", None)
        dc = getattr(self._raw, "deployment_configuration", None)

        config_lines = [
            f"instance_type:        {_safe_str(dc, 'instance_type')}",
            f"instance_count:       {_safe_str(dc, 'instance_count')}",
            f"copy_count_per_instance: {_safe_str(dc, 'copy_count_per_instance')}",
            f"image_uri:            {_safe_str(dc, 'image_uri')}",
            f"model_package:        {_safe_str(md, 'model_package_arn')}",
            f"recommendation_spec_name: {self.recommendation_spec_name or '-'}",
        ]

        env_vars = getattr(dc, "environment_variables", None) if dc else None
        env_block = ""
        if env_vars:
            try:
                items = list(dict(env_vars).items())
            except (TypeError, ValueError):
                items = []
            if items:
                env_lines = [f"  {k} = {v}" for k, v in items]
                env_block = "\nenv vars ({0}):\n{1}".format(len(items), "\n".join(env_lines))

        # Same records to_dataframe() uses, formatted for display here.
        perf_rows = [
            [
                rec["metric"] if rec["metric"] not in (None, "") else "-",
                rec["stat"] if rec["stat"] not in (None, "") else "-",
                _fmt_number(rec["value"]),
                rec["unit"] if rec["unit"] not in (None, "") else "-",
            ]
            for rec in self._perf_records()
        ]
        perf_table = _format_table(
            headers=["metric", "stat", "value", "unit"],
            rows=perf_rows,
        )

        return (
            f"Recommendation[{self._index}]\n"
            f"{_indent(chr(10).join(config_lines), '  ')}"
            f"{_indent(env_block, '  ')}\n"
            f"  expected performance:\n{_indent(perf_table, '    ')}"
        )

    def __repr__(self) -> str:
        dc = getattr(self._raw, "deployment_configuration", None)
        return (
            f"Recommendation[{self._index}]("
            f"spec_name={self.recommendation_spec_name or '-'}, "
            f"instance_type={_safe_str(dc, 'instance_type')}; print() for detail)"
        )

    def _repr_pretty_(self, p, cycle):
        # Render the full table in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))

    def _perf_records(self) -> List[Dict[str, Any]]:
        """(metric, stat, value, unit) records for this row's expected
        performance — the rows of the printed table, one per (metric, stat)."""
        ep = getattr(self._raw, "expected_performance", None) or []
        return [
            {
                "metric": getattr(m, "metric", None),
                "stat": getattr(m, "stat", None),
                "value": _safe_float(m, "value"),
                "unit": getattr(m, "unit", None),
            }
            for m in ep
        ]

    def to_dataframe(self):
        """Return this recommendation's expected performance as a pandas
        ``DataFrame`` — the same ``metric``/``stat``/``value``/``unit`` rows the
        printed ``expected performance`` table shows, one row per (metric, stat).

        Requires pandas.
        """
        pd = _require_pandas()
        return pd.DataFrame(self._perf_records(), columns=["metric", "stat", "value", "unit"])


def _safe_str(obj, attr) -> str:
    if obj is None:
        return "-"
    value = getattr(obj, attr, None)
    if value is None:
        return "-"
    text = str(value)
    return text if text else "-"


def _safe_float(obj, attr):
    if obj is None:
        return None
    value = getattr(obj, attr, None)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


class _RecommendationsView(list):
    """List of recommendation rows with a comparative table and a ``.best`` shortcut."""

    @property
    def best(self):
        """The top-ranked recommendation row."""
        if not self:
            raise IndexError(
                "No recommendations available. The AIRecommendationJob may "
                "still be running, or it may have failed before producing "
                "recommendations."
            )
        return self[0]

    def __repr__(self) -> str:
        return f"Recommendations[{len(self)}]  (.best = top row; print() for the table)"

    def _repr_pretty_(self, p, cycle):
        # Render the full table in notebooks (Jupyter uses this hook).
        p.text("..." if cycle else str(self))

    # Column labels for the comparative table / DataFrame, in display order.
    _TABLE_COLUMNS = (
        "idx",
        "spec_name",
        "instance_type",
        "instances",
        "copies/inst",
        "container",
        "req/s",
        "tok/s",
        "lat_p50",
        "lat_p90",
        "lat_p99",
        "ttft_p50",
        "itl_p50",
    )
    # Metric columns holding numbers; the rest are strings or the idx.
    _NUMERIC_COLUMNS = (
        "req/s",
        "tok/s",
        "lat_p50",
        "lat_p90",
        "lat_p99",
        "ttft_p50",
        "itl_p50",
    )

    def _row_records(self) -> List[Dict[str, Any]]:
        """One record per row, keyed by ``_TABLE_COLUMNS``, with native values.

        Shared by ``__str__`` and ``to_dataframe()`` so they cannot drift.
        """
        records = []
        for view in self:
            dc = getattr(view.raw, "deployment_configuration", None)
            ep = view.expected_performance
            records.append(
                {
                    "idx": view._index,
                    "spec_name": view.recommendation_spec_name,
                    "instance_type": getattr(dc, "instance_type", None) if dc else None,
                    "instances": getattr(dc, "instance_count", None) if dc else None,
                    "copies/inst": (getattr(dc, "copy_count_per_instance", None) if dc else None),
                    # None when absent (not "-"), so the DataFrame keeps it as
                    # missing; __str__ renders the dash.
                    "container": (
                        _short_container_tag(dc.image_uri)
                        if dc is not None and getattr(dc, "image_uri", None)
                        else None
                    ),
                    "req/s": _get_metric_stat(ep, "request_throughput", "avg"),
                    "tok/s": _get_metric_stat(ep, "output_token_throughput", "avg"),
                    "lat_p50": _get_metric_stat(ep, "request_latency", "p50"),
                    "lat_p90": _get_metric_stat(ep, "request_latency", "p90"),
                    "lat_p99": _get_metric_stat(ep, "request_latency", "p99"),
                    "ttft_p50": _get_metric_stat(ep, "time_to_first_token", "p50"),
                    "itl_p50": _get_metric_stat(ep, "inter_token_latency", "p50"),
                }
            )
        return records

    def __str__(self) -> str:
        if not self:
            return "Recommendations[0]  (no rows)"

        rows = []
        for rec in self._row_records():
            row = []
            for col in self._TABLE_COLUMNS:
                value = rec[col]
                if col == "idx":
                    row.append(f"[{value}]")
                elif col in self._NUMERIC_COLUMNS:
                    row.append(_fmt_number(value))
                else:
                    row.append("-" if value in (None, "") else str(value))
            rows.append(row)

        table = _format_table(headers=list(self._TABLE_COLUMNS), rows=rows)

        return (
            f"Recommendations[{len(self)}]  (.best = top row; index by [N] for full detail)\n"
            f"{table}\n"
            f"lat/ttft/itl in ms; req/s = requests/sec; tok/s = output tokens/sec"
        )

    def to_dataframe(self):
        """Return the recommendations as a pandas ``DataFrame`` — one row per
        recommendation, columns matching the printed comparative table
        (``instance_type``, ``instances``, ``req/s``, ``lat_p50``, ...), indexed
        by the recommendation index ``idx``.

        Latency columns are milliseconds; ``req/s`` is requests/sec and ``tok/s``
        is output tokens/sec, same as the printed table's footnote. Numeric
        columns hold real numbers (``NaN`` where a metric is absent), not
        preformatted strings.

        Requires pandas.
        """
        pd = _require_pandas()
        columns = [c for c in self._TABLE_COLUMNS if c != "idx"]
        records = self._row_records()
        frame = pd.DataFrame(
            [{c: rec[c] for c in columns} for rec in records],
            columns=columns,
            index=pd.Index([rec["idx"] for rec in records], name="idx"),
        )
        return _coerce_numeric(pd, frame, self._NUMERIC_COLUMNS)


def _get_metric_stat(ep_view, metric_name: str, stat: str):
    try:
        metric = getattr(ep_view, metric_name)
    except AttributeError:
        return None
    return getattr(metric, stat, None)


def _short_container_tag(image_uri: str) -> str:
    """Extract a comparison-friendly tag from a full container image URI.

    ".../djl-inference:0.36.0-lmi25.0.0-cu130" -> "lmi25.0.0"
    ".../image:vllm0.6.0-cu121"                -> "vllm0.6.0"
    ".../image:1.2.3"                          -> "1.2.3"
    Empty / "-"                                -> "-"
    """
    if not image_uri or image_uri == "-":
        return "-"
    tag = image_uri.rsplit(":", 1)[-1]
    for token in tag.split("-"):
        lower = token.lower()
        if lower.startswith("lmi") or lower.startswith("vllm"):
            return token
    return tag
