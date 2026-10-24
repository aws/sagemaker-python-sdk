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
"""``Endpoint`` collects its own diagnostics through ``EndpointMixin``.

Before the mixin, ``Endpoint`` had no ``diagnostics()`` and no ``recent_logs()``. The
SageMaker agent skill ``collect_diagnostics.py`` built raw boto3 clients from a region string,
read ``InstanceType`` from ``ProductionVariantSummary`` where it never exists, so the instance
type was always ``None``, fetched one unpaged page of 100 log events, and ignored the session
a caller had bound the resource to.
"""

from __future__ import absolute_import

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from sagemaker.core.endpoint import (
    METRIC_QUERIES,
    EndpointDiagnostics,
    EndpointMixin,
    MetricSeries,
    VariantDiagnostics,
)
from sagemaker.core.resources import Endpoint, EndpointConfig
from sagemaker.core.shapes import (
    ProductionVariant,
    ProductionVariantStatus,
    ProductionVariantSummary,
)

ENDPOINT_NAME = "ep"
CONFIG_NAME = "cfg"
VARIANT_NAME = "AllTraffic"
SHADOW_VARIANT_NAME = "Canary"
INSTANCE_TYPE = "ml.g5.xlarge"
LOG_GROUP = "/aws/sagemaker/Endpoints/ep"
TIMESTAMPS = [
    datetime(2026, 10, 23, 12, 0, tzinfo=timezone.utc),
    datetime(2026, 10, 23, 12, 1, tzinfo=timezone.utc),
]
EXPECTED_QUERY_IDS = [
    "invocations",
    "errors_4xx",
    "errors_5xx",
    "invocation_model_errors",
    "model_latency_avg",
    "model_latency_p99",
    "overhead_latency",
    "cpu_util",
    "memory_util",
    "gpu_util",
    "gpu_memory_util",
]
EXPECTED_SAGEMAKER_METRICS = {
    ("Invocations", "Sum"),
    ("Invocation4XXErrors", "Sum"),
    ("Invocation5XXErrors", "Sum"),
    ("InvocationModelErrors", "Sum"),
    ("ModelLatency", "Average"),
    ("ModelLatency", "p99"),
    ("OverheadLatency", "Average"),
}
EXPECTED_INSTANCE_METRICS = {
    ("CPUUtilization", "Average"),
    ("MemoryUtilization", "Average"),
    ("GPUUtilization", "Average"),
    ("GPUMemoryUtilization", "Average"),
}


def _endpoint(**overrides):
    kwargs = {
        "endpoint_name": ENDPOINT_NAME,
        "endpoint_config_name": CONFIG_NAME,
        "endpoint_status": "InService",
        "production_variants": [
            ProductionVariantSummary(
                variant_name=VARIANT_NAME,
                current_instance_count=2,
                desired_instance_count=3,
                current_weight=1.0,
                variant_status=[
                    ProductionVariantStatus(status="Updating"),
                    ProductionVariantStatus(status="Baking"),
                ],
            ),
            ProductionVariantSummary(variant_name=SHADOW_VARIANT_NAME, current_instance_count=1),
        ],
    }
    kwargs.update(overrides)
    return Endpoint(**kwargs)


def _endpoint_config():
    return EndpointConfig(
        endpoint_config_name=CONFIG_NAME,
        production_variants=[
            ProductionVariant(variant_name=SHADOW_VARIANT_NAME, instance_type="ml.m5.large"),
            ProductionVariant(variant_name=VARIANT_NAME, instance_type=INSTANCE_TYPE),
        ],
    )


def _metric_data_response():
    return {
        "MetricDataResults": [
            {"Id": "model_latency_p99", "Values": [120.0, 480.0], "Timestamps": TIMESTAMPS},
            {"Id": "invocations", "Values": [10.0, 12.0], "Timestamps": TIMESTAMPS},
            {"Id": "not_a_query", "Values": [1.0], "Timestamps": TIMESTAMPS[:1]},
        ]
    }


def _log_event(seconds, message):
    return {
        "timestamp": 1_700_000_000_000 + seconds * 1000,
        "logStreamName": "AllTraffic/i-0abc",
        "message": message,
    }


def _client_error(operation):
    return ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "no " + operation}}, operation
    )


@pytest.fixture
def clients():
    """Canned CloudWatch and Logs clients, keyed the way ``_boto_client`` is called."""
    cloudwatch = Mock(name="cloudwatch")
    cloudwatch.get_metric_data.return_value = _metric_data_response()
    logs = Mock(name="logs")
    logs.filter_log_events.return_value = {"events": [_log_event(0, "ready")]}
    return {"cloudwatch": cloudwatch, "logs": logs}


@pytest.fixture
def mocked_sources(clients):
    with (
        patch.object(Endpoint, "refresh") as refresh,
        patch.object(EndpointConfig, "get", return_value=_endpoint_config()) as get_config,
        patch.object(
            EndpointMixin, "_boto_client", side_effect=lambda service_name: clients[service_name]
        ),
    ):
        yield {"refresh": refresh, "get_config": get_config, "clients": clients}


def test_endpoint_is_an_endpoint_mixin_without_new_fields():
    assert issubclass(Endpoint, EndpointMixin)
    assert not {"diagnostics", "recent_logs"} & set(Endpoint.model_fields)
    assert "diagnostics" not in _endpoint().model_dump()


def test_diagnostics_refreshes_and_reads_the_endpoint_state(mocked_sources):
    endpoint = _endpoint(failure_reason="CannotStartContainerError")

    result = endpoint.diagnostics()

    mocked_sources["refresh"].assert_called_once_with()
    assert isinstance(result, EndpointDiagnostics)
    assert result.endpoint_name == ENDPOINT_NAME
    assert result.endpoint_status == "InService"
    assert result.failure_reason == "CannotStartContainerError"
    assert result.endpoint_config_name == CONFIG_NAME
    assert result.errors == []


def test_diagnostics_reads_the_instance_type_from_the_endpoint_config(mocked_sources):
    """``DescribeEndpoint`` never returns ``InstanceType``. The skill read ``None`` forever."""
    session = boto3.Session(
        aws_access_key_id="AKIAEXAMPLE", aws_secret_access_key="secret", region_name="eu-west-1"
    )
    endpoint = _endpoint()._set_client_context(session=session, region="eu-west-1")

    result = endpoint.diagnostics()

    mocked_sources["get_config"].assert_called_once_with(
        endpoint_config_name=CONFIG_NAME, session=session, region="eu-west-1"
    )
    assert result.variants == [
        VariantDiagnostics(
            variant_name=VARIANT_NAME,
            instance_type=INSTANCE_TYPE,
            current_instance_count=2,
            desired_instance_count=3,
            current_weight=1.0,
            variant_status=["Updating", "Baking"],
        ),
        VariantDiagnostics(
            variant_name=SHADOW_VARIANT_NAME,
            instance_type="ml.m5.large",
            current_instance_count=1,
            desired_instance_count=None,
            current_weight=None,
            variant_status=[],
        ),
    ]


def test_diagnostics_sends_the_eleven_dashboard_queries_for_the_first_variant(mocked_sources):
    _endpoint().diagnostics(window_minutes=30)

    call = mocked_sources["clients"]["cloudwatch"].get_metric_data.call_args
    queries = call.kwargs["MetricDataQueries"]
    assert len(queries) == len(METRIC_QUERIES) == 11
    assert [query["Id"] for query in queries] == EXPECTED_QUERY_IDS
    expected_dimensions = [
        {"Name": "EndpointName", "Value": ENDPOINT_NAME},
        {"Name": "VariantName", "Value": VARIANT_NAME},
    ]
    by_namespace = {}
    for query in queries:
        metric_stat = query["MetricStat"]
        assert metric_stat["Metric"]["Dimensions"] == expected_dimensions
        assert metric_stat["Period"] == 60
        by_namespace.setdefault(metric_stat["Metric"]["Namespace"], set()).add(
            (metric_stat["Metric"]["MetricName"], metric_stat["Stat"])
        )
    assert by_namespace == {
        "AWS/SageMaker": EXPECTED_SAGEMAKER_METRICS,
        "/aws/sagemaker/Endpoints": EXPECTED_INSTANCE_METRICS,
    }
    assert call.kwargs["ScanBy"] == "TimestampAscending"
    window = call.kwargs["EndTime"] - call.kwargs["StartTime"]
    assert window.total_seconds() == 30 * 60


def test_diagnostics_targets_the_requested_variant(mocked_sources):
    _endpoint().diagnostics(variant_name=SHADOW_VARIANT_NAME)

    queries = mocked_sources["clients"]["cloudwatch"].get_metric_data.call_args.kwargs[
        "MetricDataQueries"
    ]
    assert {query["MetricStat"]["Metric"]["Dimensions"][1]["Value"] for query in queries} == {
        SHADOW_VARIANT_NAME
    }


def test_diagnostics_defaults_to_all_traffic_without_variants(mocked_sources):
    _endpoint(production_variants=[]).diagnostics()

    queries = mocked_sources["clients"]["cloudwatch"].get_metric_data.call_args.kwargs[
        "MetricDataQueries"
    ]
    assert queries[0]["MetricStat"]["Metric"]["Dimensions"][1]["Value"] == "AllTraffic"


def test_diagnostics_keys_metrics_by_name_and_stat(mocked_sources):
    result = _endpoint().diagnostics()

    assert sorted(result.metrics) == ["Invocations/Sum", "ModelLatency/p99"]
    assert result.metrics["ModelLatency/p99"] == MetricSeries(
        namespace="AWS/SageMaker",
        metric_name="ModelLatency",
        stat="p99",
        values=[120.0, 480.0],
        timestamps=TIMESTAMPS,
    )


def test_diagnostics_joins_paged_metric_results(mocked_sources):
    cloudwatch = mocked_sources["clients"]["cloudwatch"]
    cloudwatch.get_metric_data.side_effect = [
        {
            "MetricDataResults": [
                {"Id": "invocations", "Values": [1.0], "Timestamps": TIMESTAMPS[:1]}
            ],
            "NextToken": "page-2",
        },
        {
            "MetricDataResults": [
                {"Id": "invocations", "Values": [2.0], "Timestamps": TIMESTAMPS[1:]}
            ]
        },
    ]

    result = _endpoint().diagnostics()

    assert result.metrics["Invocations/Sum"].values == [1.0, 2.0]
    assert result.metrics["Invocations/Sum"].timestamps == TIMESTAMPS
    second_call = cloudwatch.get_metric_data.call_args_list[1]
    assert second_call.kwargs["NextToken"] == "page-2"


def test_diagnostics_formats_and_limits_the_log_lines(mocked_sources):
    logs = mocked_sources["clients"]["logs"]
    logs.filter_log_events.return_value = {
        "events": [_log_event(0, "model loaded"), _log_event(1, "worker started")],
        "nextToken": "more",
    }

    result = _endpoint().diagnostics(log_limit=2)

    assert result.recent_logs == [
        "2023-11-14T22:13:20+00:00 AllTraffic/i-0abc model loaded",
        "2023-11-14T22:13:21+00:00 AllTraffic/i-0abc worker started",
    ]
    logs.filter_log_events.assert_called_once()
    request = logs.filter_log_events.call_args.kwargs
    assert request["logGroupName"] == LOG_GROUP
    assert request["limit"] == 2
    assert request["endTime"] - request["startTime"] == 15 * 60 * 1000


def test_diagnostics_widens_the_log_window_to_the_metric_window(mocked_sources):
    _endpoint().diagnostics(window_minutes=60)

    request = mocked_sources["clients"]["logs"].filter_log_events.call_args.kwargs
    assert request["endTime"] - request["startTime"] == 60 * 60 * 1000


def test_diagnostics_collects_a_cloudwatch_error_instead_of_raising(mocked_sources):
    mocked_sources["clients"]["cloudwatch"].get_metric_data.side_effect = _client_error(
        "GetMetricData"
    )

    result = _endpoint().diagnostics()

    assert result.metrics == {}
    assert len(result.errors) == 1
    assert result.errors[0].startswith("GetMetricData failed: ")
    assert "no GetMetricData" in result.errors[0]
    assert result.recent_logs == ["2023-11-14T22:13:20+00:00 AllTraffic/i-0abc ready"]


def test_diagnostics_collects_a_logs_error_and_a_config_error(mocked_sources):
    mocked_sources["clients"]["logs"].filter_log_events.side_effect = _client_error(
        "FilterLogEvents"
    )
    mocked_sources["get_config"].side_effect = EndpointConnectionError(
        endpoint_url="https://api.sagemaker.eu-west-1.amazonaws.com"
    )

    result = _endpoint().diagnostics()

    assert result.recent_logs == []
    assert [variant.instance_type for variant in result.variants] == [None, None]
    assert [error.split(":")[0] for error in result.errors] == [
        "DescribeEndpointConfig failed",
        "FilterLogEvents failed",
    ]
    assert len(result.metrics) == 2


def test_diagnostics_skips_the_config_lookup_without_a_config_name(mocked_sources):
    result = _endpoint(endpoint_config_name=None).diagnostics()

    mocked_sources["get_config"].assert_not_called()
    assert result.endpoint_config_name is None
    assert result.variants[0].instance_type is None
    assert result.errors == []


def test_diagnostics_raises_when_describe_endpoint_fails(mocked_sources):
    mocked_sources["refresh"].side_effect = _client_error("DescribeEndpoint")

    with pytest.raises(ClientError, match="no DescribeEndpoint"):
        _endpoint().diagnostics()


def test_recent_logs_pages_until_the_next_token_is_absent(mocked_sources):
    logs = mocked_sources["clients"]["logs"]
    logs.filter_log_events.side_effect = [
        {"events": [_log_event(0, "one")], "nextToken": "page-2"},
        {"events": [_log_event(1, "two")], "nextToken": "page-3"},
        {"events": [_log_event(2, "three")]},
    ]

    lines = _endpoint().recent_logs(window_minutes=10, limit=100)

    assert [line.split(" ", 2)[2] for line in lines] == ["one", "two", "three"]
    requests = [call.kwargs for call in logs.filter_log_events.call_args_list]
    assert [request.get("nextToken") for request in requests] == [None, "page-2", "page-3"]
    assert [request["limit"] for request in requests] == [100, 99, 98]
    assert requests[0]["endTime"] - requests[0]["startTime"] == 10 * 60 * 1000


def test_recent_logs_stops_on_an_empty_page_with_a_token(mocked_sources):
    """CloudWatch Logs can return a token with no events forever. The pager must not loop."""
    logs = mocked_sources["clients"]["logs"]
    logs.filter_log_events.side_effect = [
        {"events": [_log_event(0, "one")], "nextToken": "page-2"},
        {"events": [], "nextToken": "page-3"},
    ]

    lines = _endpoint().recent_logs()

    assert len(lines) == 1
    assert logs.filter_log_events.call_count == 2


def test_recent_logs_stops_at_the_limit(mocked_sources):
    logs = mocked_sources["clients"]["logs"]
    logs.filter_log_events.return_value = {
        "events": [_log_event(0, "one"), _log_event(1, "two"), _log_event(2, "three")],
        "nextToken": "page-2",
    }

    lines = _endpoint().recent_logs(limit=3)

    assert len(lines) == 3
    logs.filter_log_events.assert_called_once()


def test_recent_logs_raises_a_client_error(mocked_sources):
    mocked_sources["clients"]["logs"].filter_log_events.side_effect = _client_error(
        "FilterLogEvents"
    )

    with pytest.raises(ClientError, match="no FilterLogEvents"):
        _endpoint().recent_logs()


def test_boto_clients_use_the_bound_session_and_region():
    """The skill built clients from a region string and lost the caller's credentials."""
    session = boto3.Session(
        aws_access_key_id="AKIABOUND0000000000", aws_secret_access_key="s", region_name="us-east-1"
    )
    endpoint = _endpoint()._set_client_context(session=session, region="eu-central-1")

    cloudwatch = endpoint._boto_client("cloudwatch")
    logs = endpoint._boto_client("logs")

    assert cloudwatch.meta.service_model.service_name == "cloudwatch"
    assert logs.meta.service_model.service_name == "logs"
    assert cloudwatch.meta.region_name == logs.meta.region_name == "eu-central-1"
    assert cloudwatch._request_signer._credentials.access_key == "AKIABOUND0000000000"
    assert logs._request_signer._credentials.access_key == "AKIABOUND0000000000"
