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
"""Describe the traffic one model on a SageMaker endpoint served, to size a workload from it.

Concurrency comes from the endpoint's invocation metrics in CloudWatch: each hour's maximum
``ConcurrentRequestsPerCopy`` (an inference component) or ``ConcurrentRequestsPerModel`` (a model
on a production variant), times the current copies or instances.

Token sizes come from detailed observability. With it on (the endpoint configuration's
``MetricsConfig.EnableDetailedObservability``), the serving container's histograms of each
completed request's prompt and generated tokens are published to CloudWatch, which serves them
through its Prometheus-compatible query API. Those calls are signed with the session's
credentials and need ``cloudwatch:GetMetricData`` and ``cloudwatch:ListMetrics``.
``_TOKEN_METRICS`` lists the histograms read for each serving framework.
"""

from __future__ import absolute_import

import itertools
import json
import logging
import math
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.exceptions import BotoCoreError, ClientError

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.serve.ai_inference_recommender.listing import _boto_session

logger = logging.getLogger(__name__)

# The longest window that can be read: a CloudWatch PromQL query spans at most seven days.
MAX_WINDOW = timedelta(days=7)
_DAY_SECONDS = 86_400
_HOUR_SECONDS = 3_600
_ENDPOINT_LABEL = "aws.sagemaker.endpoint.name"
_COMPONENT_LABEL = "aws.sagemaker.inference_component.name"
_VARIANT_LABEL = "aws.sagemaker.variant.name"
_STAT_LABEL = "stat"
# Attempts per query, with exponential backoff, on throttling, server errors and dropped connections.
_QUERY_ATTEMPTS = 3
_RETRY_BASE_SECONDS = 0.5
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class _TokenMetrics:
    """The histograms a serving framework publishes of each request's prompt and generated tokens."""

    framework: str
    prompt_tokens: str
    generation_tokens: str


# Read in this order; the first framework with requests in the window is used. To support
# another framework, add its histograms here.
_TOKEN_METRICS = (
    _TokenMetrics("vllm", "vllm:request_prompt_tokens", "vllm:request_generation_tokens"),
)


@dataclass(frozen=True)
class TrafficStats:
    """The distribution of one quantity across a window: percentiles, mean and standard deviation.

    ``max`` is the 100th percentile.
    """

    min: float
    p25: float
    p50: float
    p70: float
    p80: float
    p90: float
    p95: float
    p99: float
    max: float
    mean: float
    stddev: float


_PERCENTILES = ("p25", "p50", "p70", "p80", "p90", "p95", "p99")


@dataclass(frozen=True)
class EndpointTraffic:
    """The traffic one model on an endpoint served from ``start_time`` to ``end_time``.

    Attributes:
        endpoint_name: The endpoint described.
        inference_component_name: The inference component described (an adapter's base
            component when an adapter is passed), or None for a model on a production variant.
        variant_name: The production variant the model runs on.
        start_time: The window's start, in UTC.
        end_time: The window's end, in UTC.
        concurrency: The concurrent requests on the whole deployment across the window's hours:
            each hour's per-copy or per-instance maximum times the current copies or instances (at
            least one). ``max`` is the busiest hour's. None when the window had no traffic. If the
            copies or instances autoscale, the count in an hour may differ from the current one.
        detailed_observability: The endpoint configuration's
            ``MetricsConfig.EnableDetailedObservability``, or None when the configuration doesn't
            say or can't be read. Token sizes aren't read when it's False.
        framework: The serving framework whose token histograms were read, or None without them.
        requests: Requests the model completed in the window, from its token histograms. 0
            without them.
        input_tokens: Prompt tokens per request, or None without token histograms.
        output_tokens: Generated tokens per request, or None without token histograms.
        token_metrics_error: Why the token histograms couldn't be read, for example a missing
            permission or an unreachable query API, or None when the read didn't fail.
        data_capture_s3_uri: Where SageMaker Data Capture writes the variant's requests, or None
            when the endpoint doesn't capture them. The variant's captures include the requests of
            every inference component on it.
    """

    endpoint_name: str
    inference_component_name: Optional[str]
    variant_name: Optional[str]
    start_time: datetime
    end_time: datetime
    concurrency: Optional[TrafficStats]
    detailed_observability: Optional[bool]
    framework: Optional[str]
    requests: int
    input_tokens: Optional[TrafficStats]
    output_tokens: Optional[TrafficStats]
    token_metrics_error: Optional[str]
    data_capture_s3_uri: Optional[str]


@dataclass(frozen=True)
class _Target:
    """The model on an endpoint whose traffic is read, and how many copies or instances it has."""

    endpoint_name: str
    inference_component_name: Optional[str]
    variant_name: Optional[str]
    units: int


@dataclass
class _Histogram:
    """One side's requests in the window: their count, token sum and weighted buckets.

    ``buckets`` maps a bucket's ``(lower, upper)`` bounds to its requests in the window.
    """

    count: float = 0.0
    total: float = 0.0
    buckets: Dict[Tuple[float, float], float] = field(default_factory=dict)


@_telemetry_emitter(
    feature=Feature.INFERENCE_RECOMMENDER,
    func_name="ai_inference_recommender.describe_endpoint_traffic",
)
def describe_endpoint_traffic(
    endpoint_name: str,
    *,
    inference_component_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    lookback: Optional[timedelta] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    sagemaker_session: Optional[Union[boto3.session.Session, Session]] = None,
) -> EndpointTraffic:
    """Describe the concurrency and token sizes of the traffic one model on an endpoint served.

    Pass the inference component or the production variant to describe. When the endpoint has a
    single base inference component, or no inference components and a single variant, it is
    found for you. An adapter's requests run in its base component's container, so passing an
    adapter describes its base component, whose traffic includes every adapter on it.

    The window is relative or absolute: ``lookback`` back from ``end_time``, or ``start_time`` to
    ``end_time``. ``end_time`` defaults to now and ``lookback`` to seven days, the longest window.
    Times must be timezone-aware; they are compared and reported in UTC.

    The request count and the token means are exact, unless a serving container restarted in the
    window: the requests it completed between the previous daily snapshot and the restart aren't
    counted. The token percentiles and standard deviations are estimated from the histograms'
    buckets. Metrics can take a few minutes to arrive, so a window that ends in the last few
    minutes can miss its newest requests. Token sizes need detailed observability on and a serving
    container whose token histograms are read (vLLM); without them, ``input_tokens`` and
    ``output_tokens`` are None. Reading them is best effort: when the read fails,
    ``token_metrics_error`` says why and a warning is logged.

    Args:
        endpoint_name: The SageMaker endpoint name.
        inference_component_name: The inference component to describe, for an endpoint that has
            them.
        variant_name: The production variant to describe, for a model deployed on a variant.
        lookback: How far back from ``end_time`` to read, up to seven days. Can't be combined
            with ``start_time``.
        start_time: The window's start, a timezone-aware ``datetime``.
        end_time: The window's end, a timezone-aware ``datetime`` no later than now.
        sagemaker_session: Optional session — a ``boto3.session.Session`` or a sagemaker
            ``Session`` wrapping one. A default is created if omitted. Use one in the endpoint's
            region.

    Returns:
        An ``EndpointTraffic``.

    Raises:
        ValueError: If the arguments are invalid, or the model to describe can't be determined.
        botocore.exceptions.ClientError: If a SageMaker or CloudWatch call fails, for example
            when the endpoint doesn't exist.
    """
    if not endpoint_name:
        raise ValueError("endpoint_name is required.")
    if inference_component_name and variant_name:
        raise ValueError("Pass at most one of inference_component_name and variant_name.")
    start, end = _window(
        lookback, start_time, end_time, datetime.fromtimestamp(time.time(), tz=timezone.utc)
    )

    boto_session = _boto_session(sagemaker_session) or boto3.session.Session()
    sagemaker = boto_session.client("sagemaker")
    endpoint = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    target = _resolve_target(sagemaker, endpoint, inference_component_name, variant_name)
    observability = _detailed_observability(sagemaker, endpoint)
    cloudwatch = boto_session.client("cloudwatch")
    concurrency = _concurrency(cloudwatch, target, start, end)

    framework, requests, input_tokens, output_tokens, token_error = None, 0, None, None, None
    if observability is not False:
        try:
            snapshots = _snapshots(_PromQLClient(boto_session, cloudwatch), target, start, end)
        except RuntimeError as error:
            token_error = str(error)
            logger.warning(
                "Couldn't read the token metrics of endpoint %s: %s", endpoint_name, error
            )
        else:
            for metrics in _TOKEN_METRICS:
                prompt = _window_histogram(snapshots, f"{metrics.framework}/input")
                generation = _window_histogram(snapshots, f"{metrics.framework}/output")
                if prompt is not None and generation is not None:
                    framework, requests = metrics.framework, round(prompt.count)
                    input_tokens = _histogram_stats(prompt)
                    output_tokens = _histogram_stats(generation)
                    break
    return EndpointTraffic(
        endpoint_name=endpoint_name,
        inference_component_name=target.inference_component_name,
        variant_name=target.variant_name,
        start_time=start,
        end_time=end,
        concurrency=concurrency,
        detailed_observability=observability,
        framework=framework,
        requests=requests,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_metrics_error=token_error,
        data_capture_s3_uri=_data_capture_uri(endpoint, target.variant_name),
    )


def _window(
    lookback: Optional[timedelta],
    start_time: Optional[datetime],
    end_time: Optional[datetime],
    now: datetime,
) -> Tuple[datetime, datetime]:
    """The window's start and end in UTC, to the second."""
    for name, value in (("start_time", start_time), ("end_time", end_time)):
        if value is not None and (not isinstance(value, datetime) or value.utcoffset() is None):
            raise ValueError(
                f"{name} must be a timezone-aware datetime, for example "
                f"datetime(2026, 1, 31, tzinfo=timezone.utc); got {value!r}."
            )
    if lookback is not None:
        if not isinstance(lookback, timedelta) or lookback <= timedelta(0):
            raise ValueError(f"lookback must be a positive timedelta, got {lookback!r}.")
        if start_time is not None:
            raise ValueError("Pass lookback or start_time, not both.")
    end = (end_time or now).astimezone(timezone.utc).replace(microsecond=0)
    if end > now:
        raise ValueError(f"end_time can't be later than now; got {end_time!r}.")
    if start_time is not None:
        start = start_time.astimezone(timezone.utc).replace(microsecond=0)
    else:
        start = end - (lookback or MAX_WINDOW)
    if start >= end:
        raise ValueError(f"start_time must be before end_time; got {start} and {end}.")
    if end - start > MAX_WINDOW:
        raise ValueError(f"The window can be at most {MAX_WINDOW.days} days; got {end - start}.")
    return start, end


def _resolve_target(
    client, endpoint: Dict[str, Any], component: Optional[str], variant: Optional[str]
) -> _Target:
    """Find the inference component or production variant to read, and its copies or instances."""
    endpoint_name = endpoint["EndpointName"]
    variants = {v["VariantName"]: v for v in endpoint.get("ProductionVariants", [])}
    if component is None and variant is None:
        components = _base_components(client, endpoint_name)
        if len(components) > 1:
            raise ValueError(
                f"Endpoint {endpoint_name!r} has inference components {components}; pass "
                "inference_component_name to choose one."
            )
        if components:
            component = components[0]
        elif len(variants) > 1:
            raise ValueError(
                f"Endpoint {endpoint_name!r} has production variants {sorted(variants)}; pass "
                "variant_name to choose one."
            )
        elif variants:
            variant = next(iter(variants))
        else:
            raise ValueError(f"Endpoint {endpoint_name!r} has no production variant.")
    if component is not None:
        described = client.describe_inference_component(InferenceComponentName=component)
        if described.get("EndpointName") != endpoint_name:
            raise ValueError(
                f"Inference component {component!r} is not on endpoint {endpoint_name!r}."
            )
        base = (described.get("Specification") or {}).get("BaseInferenceComponentName")
        if base:
            # An adapter's requests run in its base component's container, which publishes the
            # concurrency and the token histograms for them.
            component = base
            described = client.describe_inference_component(InferenceComponentName=base)
        copies = (described.get("RuntimeConfig") or {}).get("CurrentCopyCount") or 0
        return _Target(endpoint_name, component, described.get("VariantName"), copies)
    if variant not in variants:
        raise ValueError(
            f"Endpoint {endpoint_name!r} has no production variant {variant!r}; it has "
            f"{sorted(variants)}."
        )
    instances = variants[variant].get("CurrentInstanceCount") or 0
    return _Target(endpoint_name, None, variant, instances)


def _base_components(client, endpoint_name: str) -> List[str]:
    """The endpoint's inference components that aren't adapters.

    An adapter's traffic is read from its base component, so only base components are choices.
    """
    components = _inference_components(client, endpoint_name)
    if len(components) <= 1:
        return components
    return [
        name
        for name in components
        if not (
            client.describe_inference_component(InferenceComponentName=name).get("Specification")
            or {}
        ).get("BaseInferenceComponentName")
    ]


def _inference_components(client, endpoint_name: str) -> List[str]:
    """The names of the endpoint's inference components."""
    names: List[str] = []
    kwargs: Dict[str, Any] = {"EndpointNameEquals": endpoint_name}
    while True:
        page = client.list_inference_components(**kwargs)
        names += [ic["InferenceComponentName"] for ic in page.get("InferenceComponents", [])]
        if not page.get("NextToken"):
            return names
        kwargs["NextToken"] = page["NextToken"]


def _detailed_observability(client, endpoint: Dict[str, Any]) -> Optional[bool]:
    """The endpoint configuration's ``EnableDetailedObservability``, or None if it doesn't say.

    An endpoint configuration can be deleted while an endpoint runs on it; then it can't say.
    """
    try:
        config = client.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    except ClientError:
        return None
    enabled = (config.get("MetricsConfig") or {}).get("EnableDetailedObservability")
    return enabled if isinstance(enabled, bool) else None


def _data_capture_uri(endpoint: Dict[str, Any], variant: Optional[str]) -> Optional[str]:
    """Where Data Capture writes the variant's requests, or None when capture is off."""
    capture = endpoint.get("DataCaptureConfig") or {}
    destination = capture.get("DestinationS3Uri")
    if not capture.get("EnableCapture") or not destination or not variant:
        return None
    return f"{destination.rstrip('/')}/{endpoint['EndpointName']}/{variant}/"


def _concurrency(client, target: _Target, start: datetime, end: datetime) -> Optional[TrafficStats]:
    """The deployment's concurrent requests across the window's hours, or None without traffic.

    SageMaker publishes concurrency per copy or per instance, so each hour's maximum is
    multiplied by the current copies or instances.
    """
    if target.inference_component_name is not None:
        metric = "ConcurrentRequestsPerCopy"
        dimensions = [{"Name": "InferenceComponentName", "Value": target.inference_component_name}]
    else:
        metric = "ConcurrentRequestsPerModel"
        dimensions = [
            {"Name": "EndpointName", "Value": target.endpoint_name},
            {"Name": "VariantName", "Value": target.variant_name},
        ]
    response = client.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=metric,
        Dimensions=dimensions,
        StartTime=start,
        EndTime=end,
        Period=_HOUR_SECONDS,
        Statistics=["Maximum"],
    )
    units = max(target.units, 1)
    hourly = sorted(point["Maximum"] * units for point in response.get("Datapoints", []))
    if not hourly or hourly[-1] <= 0:
        return None
    return _sample_stats(hourly)


def _sample_stats(values: List[float]) -> TrafficStats:
    """The distribution of sorted samples; percentiles interpolate linearly between ranks."""

    def percentile(fraction: float) -> float:
        rank = fraction * (len(values) - 1)
        low = math.floor(rank)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (rank - low)

    mean = sum(values) / len(values)
    stddev = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
    return _stats(percentile, values[0], values[-1], mean, stddev)


def _stats(
    percentile: Callable[[float], float], low: float, high: float, mean: float, stddev: float
) -> TrafficStats:
    """A ``TrafficStats`` from a percentile function, rounded to one decimal place."""
    values = {name: percentile(int(name[1:]) / 100) for name in _PERCENTILES}
    return TrafficStats(
        min=round(low, 1),
        **{name: round(value, 1) for name, value in values.items()},
        max=round(high, 1),
        mean=round(mean, 1),
        stddev=round(stddev, 1),
    )


def _snapshots(
    client: "_PromQLClient", target: _Target, start: datetime, end: datetime
) -> List[Dict[Tuple, Dict[str, Any]]]:
    """Each token histogram series' last snapshot a day at a time, newest first, then at the start.

    All but the last entry are read at the window's end and a day at a time back while after its
    start; together they hold each series' newest snapshot in the window. The last is read at the
    start. The histograms are cumulative, so a series' newest snapshot in the window less its
    snapshot at the start is the window's requests.
    """
    labels = {_ENDPOINT_LABEL: target.endpoint_name}
    if target.inference_component_name is not None:
        labels[_COMPONENT_LABEL] = target.inference_component_name
    else:
        labels[_VARIANT_LABEL] = target.variant_name
    parts = [
        f'label_replace(last_over_time({_selector(metric, labels)}[1d]), "{_STAT_LABEL}", '
        f'"{metrics.framework}/{side}", "", "")'
        for metrics in _TOKEN_METRICS
        for side, metric in (
            ("input", metrics.prompt_tokens),
            ("output", metrics.generation_tokens),
        )
    ]
    query = " or ".join(parts)
    first, last = int(end.timestamp()), int(start.timestamp())
    times = list(range(first, last, -_DAY_SECONDS)) + [last]
    return [client.histograms(query, at) for at in times]


def _window_histogram(
    snapshots: List[Dict[Tuple, Dict[str, Any]]], stat: str
) -> Optional[_Histogram]:
    """One side's requests in the window, or None without requests.

    The histograms are cumulative, so a series' requests between two of its snapshots are the later
    one less the earlier, summed over its snapshots from the start's through the window's days. A
    snapshot whose count, sum or any bucket is below the one before it follows a restart of its
    container, and counts from zero, as does a series that started in the window.
    """
    *window, before = snapshots
    previous = {key: sample for key, sample in before.items() if dict(key).get(_STAT_LABEL) == stat}
    histogram = _Histogram()
    for snapshot in reversed(window):
        for key, sample in snapshot.items():
            if dict(key).get(_STAT_LABEL) != stat:
                continue
            earlier = previous.get(key)
            _add(histogram, sample, 1.0)
            if earlier is not None and not _restarted(earlier, sample):
                _add(histogram, earlier, -1.0)
            previous[key] = sample
    if histogram.count <= 0 or histogram.total < 0 or not any(histogram.buckets.values()):
        return None
    return histogram


def _restarted(earlier: Dict[str, Any], later: Dict[str, Any]) -> bool:
    """Whether a cumulative snapshot went down from an earlier one: its container restarted."""
    if float(later.get("count", 0)) < float(earlier.get("count", 0)):
        return True
    if float(later.get("sum", 0)) < float(earlier.get("sum", 0)):
        return True
    counts = {(float(low), float(high)): float(n) for _, low, high, n in earlier.get("buckets", [])}
    return any(
        float(n) < counts.get((float(low), float(high)), 0.0)
        for _, low, high, n in later.get("buckets", [])
    )


def _add(histogram: _Histogram, sample: Dict[str, Any], sign: float) -> None:
    """Add a snapshot's count, sum and buckets to a histogram, or subtract them."""
    histogram.count += sign * float(sample.get("count", 0))
    histogram.total += sign * float(sample.get("sum", 0))
    for _, lower, upper, observations in sample.get("buckets", []):
        bounds = (float(lower), float(upper))
        histogram.buckets[bounds] = histogram.buckets.get(bounds, 0.0) + sign * float(observations)


def _histogram_stats(histogram: _Histogram) -> TrafficStats:
    """A histogram's distribution: the exact mean, and the rest estimated from its buckets.

    Requests are spread evenly within a bucket; an open-ended bucket counts at its finite bound.
    Percentiles interpolate linearly within buckets, and the standard deviation uses the buckets'
    midpoints. ``min`` and ``max`` are the bounds of the lowest and highest buckets with requests.
    """
    count = histogram.count
    tolerance = count * 1e-9
    buckets = [
        (*_finite(lower, upper), weight)
        for (lower, upper), weight in histogram.buckets.items()
        if abs(weight) > tolerance
    ]

    def below(point: float, inclusive: bool) -> float:
        total = 0.0
        for low, high, weight in buckets:
            if low == high:
                total += weight if point > high or (inclusive and point == high) else 0.0
            elif point >= high:
                total += weight
            elif point > low:
                total += weight * (point - low) / (high - low)
        return total

    # The cumulative count just below and at each bound, so a bucket collapsed to a point is a
    # step. A running maximum keeps it from dipping where buckets were resized between snapshots.
    bounds = sorted({bound for low, high, _ in buckets for bound in (low, high)})
    knots = [(bound, below(bound, inclusive)) for bound in bounds for inclusive in (False, True)]
    seen = list(itertools.accumulate((cumulative for _, cumulative in knots), max))

    def percentile(fraction: float) -> float:
        target = fraction * count - tolerance
        index = next((i for i, value in enumerate(seen) if value >= target), len(knots) - 1)
        if index == 0 or seen[index] <= seen[index - 1]:
            return knots[index][0]
        low, high = knots[index - 1][0], knots[index][0]
        before, after = seen[index - 1], seen[index]
        return low + (high - low) * (target - before) / (after - before)

    first = next((i for i, value in enumerate(seen) if value > tolerance), 0)
    lowest = knots[first - 1][0] if first > 0 else knots[0][0]
    midpoint_sum = sum(weight * (low + high) / 2 for low, high, weight in buckets)
    square_sum = sum(weight * ((low + high) / 2) ** 2 for low, high, weight in buckets)
    spread = math.sqrt(max(square_sum / count - (midpoint_sum / count) ** 2, 0.0))
    return _stats(percentile, lowest, percentile(1.0), histogram.total / count, spread)


def _finite(lower: float, upper: float) -> Tuple[float, float]:
    """A bucket's bounds, with an open-ended bucket collapsed to its finite bound."""
    if math.isinf(upper):
        return lower, lower
    if math.isinf(lower):
        return upper, upper
    return lower, upper


def _selector(metric: str, labels: Dict[str, str]) -> str:
    """A PromQL series selector: the quoted metric name and one exact-match matcher per label."""
    matchers = [_quote(metric)]
    matchers += [f"{_quote(name)}={_quote(value)}" for name, value in labels.items()]
    return "{" + ", ".join(matchers) + "}"


def _quote(text: str) -> str:
    """A PromQL string literal: backslashes and double quotes escaped."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


class _PromQLClient:
    """Signed GET requests to CloudWatch's Prometheus-compatible query API in one region.

    Requests go through the CloudWatch client's HTTP session, so they use its proxy, CA bundle and
    timeout settings, and a query is retried with backoff on throttling, server errors and dropped
    connections.
    """

    def __init__(self, boto_session: boto3.session.Session, cloudwatch) -> None:
        credentials = boto_session.get_credentials()
        if credentials is None:
            raise RuntimeError("No AWS credentials found. Configure credentials and try again.")
        if not boto_session.region_name:
            raise RuntimeError(
                "The session has no region. Set the endpoint's region and try again."
            )
        self._credentials = credentials
        self._region = boto_session.region_name
        # The CloudWatch client resolves the region's endpoint in its partition.
        self._base_url = cloudwatch.meta.endpoint_url
        self._http = cloudwatch._endpoint.http_session  # pylint: disable=protected-access

    def histograms(self, expression: str, at: int) -> Dict[Tuple, Dict[str, Any]]:
        """Evaluate an expression at one time; return {series labels: histogram sample}."""
        data = self._get("/api/v1/query", {"query": expression, "time": str(at)})
        return {
            tuple(sorted(series.get("metric", {}).items())): series["histogram"][1]
            for series in data.get("result", [])
            if "histogram" in series
        }

    def _get(self, path: str, params: Dict[str, str]) -> Dict[str, Any]:
        query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
        url = f"{self._base_url}{path}?{query}"
        attempt = 1
        while True:
            request = AWSRequest(method="GET", url=url)
            SigV4Auth(
                self._credentials.get_frozen_credentials(), "monitoring", self._region
            ).add_auth(request)
            try:
                response = self._http.send(request.prepare())
            except BotoCoreError as error:
                if attempt == _QUERY_ATTEMPTS:
                    raise RuntimeError(
                        f"Couldn't reach CloudWatch's query API in {self._region}: {error}"
                    ) from None
            else:
                if response.status_code not in _RETRYABLE_STATUS or attempt == _QUERY_ATTEMPTS:
                    return _query_data(response)
            time.sleep(_RETRY_BASE_SECONDS * 2 ** (attempt - 1))
            attempt += 1


def _query_data(response) -> Dict[str, Any]:
    """The ``data`` of a query API response, or a RuntimeError saying why there's none."""
    detail = response.content.decode("utf-8", "replace")[:300]
    if response.status_code == 403:
        raise RuntimeError(
            "CloudWatch denied the query (HTTP 403). Reading token metrics needs "
            f"cloudwatch:GetMetricData and cloudwatch:ListMetrics. {detail}"
        )
    if response.status_code != 200:
        raise RuntimeError(f"CloudWatch's query API returned HTTP {response.status_code}: {detail}")
    try:
        body = json.loads(response.content)
    except ValueError:
        raise RuntimeError(
            f"CloudWatch's query API returned a body that isn't JSON: {detail}"
        ) from None
    if body.get("status") != "success":
        raise RuntimeError(f"The CloudWatch query failed: {body.get('error', detail)}")
    return body.get("data", {})
