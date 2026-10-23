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
"""Hand-written behavior of the generated ``sagemaker.core.resources.Endpoint`` class.

``EndpointMixin`` gives ``Endpoint`` one call that collects what an operator reads first when
an endpoint misbehaves: ``diagnostics()`` returns the endpoint status and failure reason, one
``VariantDiagnostics`` per production variant with the instance type from the endpoint
config, the 11 CloudWatch metric series of the SageMaker endpoint dashboard, and the latest
lines of the ``/aws/sagemaker/Endpoints/<name>`` log group. ``recent_logs()`` returns the log
lines alone.

``DescribeEndpoint`` does not return the instance type. ``ProductionVariantSummary`` carries
counts, weights, and statuses, and the instance type lives on the endpoint config, so the
mixin reads ``EndpointConfig.get(endpoint_config_name)`` and matches the variants by name.
The mixin keeps no state and adds no model field. Every client call uses the session and
region the resource was bound with, so ``Endpoint.get(name, session=..., region=...)``
reads the metrics and logs of that account and region.
"""

from __future__ import absolute_import

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from botocore.exceptions import BotoCoreError, ClientError

from sagemaker.core.utils.utils import SageMakerClient, Unassigned

SAGEMAKER_NAMESPACE = "AWS/SageMaker"
ENDPOINT_INSTANCE_NAMESPACE = "/aws/sagemaker/Endpoints"
LOG_GROUP_PREFIX = "/aws/sagemaker/Endpoints/"
METRIC_PERIOD_SECONDS = 60
DEFAULT_VARIANT_NAME = "AllTraffic"
DEFAULT_METRIC_WINDOW_MINUTES = 5
DEFAULT_LOG_WINDOW_MINUTES = 15
DEFAULT_DIAGNOSTICS_LOG_LIMIT = 50
DEFAULT_LOG_LIMIT = 100
MAX_LOG_EVENTS_PER_PAGE = 10000


class MetricQuery(NamedTuple):
    """One ``GetMetricData`` query of the endpoint dashboard."""

    query_id: str
    namespace: str
    metric_name: str
    stat: str

    @property
    def key(self) -> str:
        """The ``EndpointDiagnostics.metrics`` key, ``"<MetricName>/<Stat>"``."""
        return f"{self.metric_name}/{self.stat}"


METRIC_QUERIES = (
    MetricQuery("invocations", SAGEMAKER_NAMESPACE, "Invocations", "Sum"),
    MetricQuery("errors_4xx", SAGEMAKER_NAMESPACE, "Invocation4XXErrors", "Sum"),
    MetricQuery("errors_5xx", SAGEMAKER_NAMESPACE, "Invocation5XXErrors", "Sum"),
    MetricQuery("invocation_model_errors", SAGEMAKER_NAMESPACE, "InvocationModelErrors", "Sum"),
    MetricQuery("model_latency_avg", SAGEMAKER_NAMESPACE, "ModelLatency", "Average"),
    MetricQuery("model_latency_p99", SAGEMAKER_NAMESPACE, "ModelLatency", "p99"),
    MetricQuery("overhead_latency", SAGEMAKER_NAMESPACE, "OverheadLatency", "Average"),
    MetricQuery("cpu_util", ENDPOINT_INSTANCE_NAMESPACE, "CPUUtilization", "Average"),
    MetricQuery("memory_util", ENDPOINT_INSTANCE_NAMESPACE, "MemoryUtilization", "Average"),
    MetricQuery("gpu_util", ENDPOINT_INSTANCE_NAMESPACE, "GPUUtilization", "Average"),
    MetricQuery("gpu_memory_util", ENDPOINT_INSTANCE_NAMESPACE, "GPUMemoryUtilization", "Average"),
)


class TimeWindow(NamedTuple):
    """A UTC window that ends at the time it was built."""

    start: datetime
    end: datetime

    @classmethod
    def last(cls, minutes: int) -> "TimeWindow":
        """The window of the last ``minutes`` minutes."""
        end = datetime.now(timezone.utc)
        return cls(start=end - timedelta(minutes=minutes), end=end)

    @property
    def start_millis(self) -> int:
        """The start as epoch milliseconds, the unit of ``FilterLogEvents``."""
        return int(self.start.timestamp() * 1000)

    @property
    def end_millis(self) -> int:
        """The end as epoch milliseconds, the unit of ``FilterLogEvents``."""
        return int(self.end.timestamp() * 1000)


@dataclass(frozen=True)
class VariantDiagnostics:
    """The state of one production variant, with the instance type from the endpoint config."""

    variant_name: str
    instance_type: Optional[str]
    current_instance_count: Optional[int]
    desired_instance_count: Optional[int]
    current_weight: Optional[float]
    variant_status: List[str]


@dataclass(frozen=True)
class MetricSeries:
    """One CloudWatch metric statistic over the diagnostics window, oldest value first."""

    namespace: str
    metric_name: str
    stat: str
    values: List[float]
    timestamps: List[datetime]


@dataclass(frozen=True)
class EndpointDiagnostics:
    """What ``Endpoint.diagnostics()`` collected.

    ``metrics`` is keyed ``"<MetricName>/<Stat>"``, for example ``"ModelLatency/p99"``.
    ``errors`` holds one line per data source that failed. The endpoint config, CloudWatch,
    and CloudWatch Logs never raise through ``diagnostics()``.
    """

    endpoint_name: str
    endpoint_status: str
    failure_reason: Optional[str]
    endpoint_config_name: Optional[str]
    variants: List[VariantDiagnostics]
    metrics: Dict[str, MetricSeries]
    recent_logs: List[str]
    errors: List[str]


class EndpointMixin:
    """``diagnostics()`` and ``recent_logs()`` for the generated ``Endpoint`` resource.

    Mixed into ``sagemaker.core.resources.Endpoint``. The methods read the generated
    attributes at call time and build every boto3 client from the session and region the
    resource is bound to, so the mixin adds no model field and no state.
    """

    endpoint_name: Any
    endpoint_config_name: Any
    endpoint_status: Any
    failure_reason: Any
    production_variants: Any

    def diagnostics(
        self,
        window_minutes: int = DEFAULT_METRIC_WINDOW_MINUTES,
        variant_name: Optional[str] = None,
        log_limit: int = DEFAULT_DIAGNOSTICS_LOG_LIMIT,
    ) -> EndpointDiagnostics:
        """Collect the endpoint state, the dashboard metrics, and the latest log lines.

        Args:
            window_minutes: The metric window, counted back from now. The log window is
                this value or 15 minutes, whichever is longer.
            variant_name: The variant whose metrics to read. Defaults to the first
                production variant, or ``AllTraffic`` when the endpoint has none.
            log_limit: The maximum number of log lines.

        Returns:
            The collected ``EndpointDiagnostics``. A CloudWatch, CloudWatch Logs, or
            endpoint config failure lands in ``errors``.

        Raises:
            botocore.exceptions.ClientError: ``DescribeEndpoint`` failed.
        """
        self.refresh()
        variants = _assigned(self.production_variants) or []
        instance_types, config_error = self._instance_types()
        metric_variant = variant_name or _first_variant_name(variants)
        metrics, metric_error = self._metrics(metric_variant, TimeWindow.last(window_minutes))
        log_window_minutes = max(window_minutes, DEFAULT_LOG_WINDOW_MINUTES)
        recent_logs, log_error = self._recent_logs_or_error(log_window_minutes, log_limit)
        return EndpointDiagnostics(
            endpoint_name=self.endpoint_name,
            endpoint_status=_assigned(self.endpoint_status) or "",
            failure_reason=_assigned(self.failure_reason),
            endpoint_config_name=_assigned(self.endpoint_config_name),
            variants=[_variant_diagnostics(summary, instance_types) for summary in variants],
            metrics=metrics,
            recent_logs=recent_logs,
            errors=[error for error in (config_error, metric_error, log_error) if error],
        )

    def recent_logs(
        self,
        window_minutes: int = DEFAULT_LOG_WINDOW_MINUTES,
        limit: int = DEFAULT_LOG_LIMIT,
    ) -> List[str]:
        """The latest lines of the ``/aws/sagemaker/Endpoints/<endpoint name>`` log group.

        Args:
            window_minutes: The log window, counted back from now.
            limit: The maximum number of lines. Pages until the log group has no next
                token, returns an empty page, or the limit is reached.

        Returns:
            One ``"<ISO timestamp> <log stream name> <message>"`` line per event.

        Raises:
            botocore.exceptions.ClientError: ``FilterLogEvents`` failed, for example
                because the log group does not exist yet.
        """
        window = TimeWindow.last(window_minutes)
        client = self._boto_client("logs")
        request = {
            "logGroupName": LOG_GROUP_PREFIX + self.endpoint_name,
            "startTime": window.start_millis,
            "endTime": window.end_millis,
        }
        lines: List[str] = []
        next_token: Optional[str] = None
        while len(lines) < limit:
            page_request = dict(request, limit=min(limit - len(lines), MAX_LOG_EVENTS_PER_PAGE))
            if next_token:
                page_request["nextToken"] = next_token
            page = client.filter_log_events(**page_request)
            events = page.get("events", [])
            lines.extend(_format_log_event(event) for event in events)
            next_token = page.get("nextToken")
            if not events or not next_token:
                break
        return lines[:limit]

    def _instance_types(self) -> Tuple[Dict[str, Optional[str]], Optional[str]]:
        """The instance type per variant name from the endpoint config, or the failure line."""
        # resources.py imports this module, so the resource class is imported at call time.
        from sagemaker.core.resources import EndpointConfig

        config_name = _assigned(self.endpoint_config_name)
        if not config_name:
            return {}, None
        try:
            config = EndpointConfig.get(
                endpoint_config_name=config_name, session=self._session, region=self._region
            )
        except (BotoCoreError, ClientError) as error:
            return {}, f"DescribeEndpointConfig failed: {error}"
        config_variants = _assigned(config.production_variants) or []
        return {
            variant.variant_name: _assigned(variant.instance_type) for variant in config_variants
        }, None

    def _metrics(
        self, variant_name: str, window: TimeWindow
    ) -> Tuple[Dict[str, MetricSeries], Optional[str]]:
        """The 11 dashboard metric series of ``variant_name``, or the failure line."""
        try:
            results = self._metric_data_results(variant_name, window)
        except (BotoCoreError, ClientError) as error:
            return {}, f"GetMetricData failed: {error}"
        return _metric_series_by_key(results), None

    def _metric_data_results(self, variant_name: str, window: TimeWindow) -> List[Dict[str, Any]]:
        """Every ``MetricDataResults`` entry of the window, across all ``GetMetricData`` pages."""
        client = self._boto_client("cloudwatch")
        request = {
            "MetricDataQueries": metric_data_queries(self.endpoint_name, variant_name),
            "StartTime": window.start,
            "EndTime": window.end,
            "ScanBy": "TimestampAscending",
        }
        results: List[Dict[str, Any]] = []
        next_token: Optional[str] = None
        while True:
            page_request = dict(request, NextToken=next_token) if next_token else request
            page = client.get_metric_data(**page_request)
            results.extend(page.get("MetricDataResults", []))
            next_token = page.get("NextToken")
            if not next_token:
                return results

    def _recent_logs_or_error(
        self, window_minutes: int, limit: int
    ) -> Tuple[List[str], Optional[str]]:
        try:
            return self.recent_logs(window_minutes=window_minutes, limit=limit), None
        except (BotoCoreError, ClientError) as error:
            return [], f"FilterLogEvents failed: {error}"

    def _boto_client(self, service_name: str) -> Any:
        """A boto3 client on the session, region, and config this resource is bound to.

        ``SageMakerClient`` caches only the SageMaker family of clients, so this method builds
        the CloudWatch and CloudWatch Logs clients from its resolved session, region, and config.
        """
        sagemaker_client = SageMakerClient(session=self._session, region_name=self._region)
        return sagemaker_client.session.client(
            service_name, sagemaker_client.region_name, config=sagemaker_client.config
        )


def metric_data_queries(endpoint_name: str, variant_name: str) -> List[Dict[str, Any]]:
    """The ``MetricDataQueries`` of the endpoint dashboard for one variant.

    SageMaker publishes every endpoint metric with the ``EndpointName`` and ``VariantName``
    dimensions, and CloudWatch matches a dimension set exactly, so each query names the two.
    """
    dimensions = [
        {"Name": "EndpointName", "Value": endpoint_name},
        {"Name": "VariantName", "Value": variant_name},
    ]
    return [
        {
            "Id": query.query_id,
            "MetricStat": {
                "Metric": {
                    "Namespace": query.namespace,
                    "MetricName": query.metric_name,
                    "Dimensions": dimensions,
                },
                "Period": METRIC_PERIOD_SECONDS,
                "Stat": query.stat,
            },
        }
        for query in METRIC_QUERIES
    ]


def _assigned(value: Any) -> Any:
    """Return ``value`` unless it is the generated ``Unassigned()`` sentinel or ``None``."""
    if value is None or isinstance(value, Unassigned):
        return None
    return value


def _first_variant_name(variants: List[Any]) -> str:
    return variants[0].variant_name if variants else DEFAULT_VARIANT_NAME


def _variant_diagnostics(
    summary: Any, instance_types: Mapping[str, Optional[str]]
) -> VariantDiagnostics:
    statuses = _assigned(summary.variant_status) or []
    return VariantDiagnostics(
        variant_name=summary.variant_name,
        instance_type=instance_types.get(summary.variant_name),
        current_instance_count=_assigned(summary.current_instance_count),
        desired_instance_count=_assigned(summary.desired_instance_count),
        current_weight=_assigned(summary.current_weight),
        variant_status=[status.status for status in statuses],
    )


def _metric_series_by_key(results: List[Mapping[str, Any]]) -> Dict[str, MetricSeries]:
    """Group the results by ``"<MetricName>/<Stat>"``. A paged query id appends to its series."""
    queries_by_id = {query.query_id: query for query in METRIC_QUERIES}
    series: Dict[str, MetricSeries] = {}
    for result in results:
        query = queries_by_id.get(result.get("Id"))
        if not query:
            continue
        previous = series.get(query.key)
        series[query.key] = MetricSeries(
            namespace=query.namespace,
            metric_name=query.metric_name,
            stat=query.stat,
            values=(previous.values if previous else []) + list(result.get("Values", [])),
            timestamps=(previous.timestamps if previous else [])
            + list(result.get("Timestamps", [])),
        )
    return series


def _format_log_event(event: Mapping[str, Any]) -> str:
    timestamp = datetime.fromtimestamp(event.get("timestamp", 0) / 1000, tz=timezone.utc)
    return f"{timestamp.isoformat()} {event.get('logStreamName', '')} {event.get('message', '')}"
