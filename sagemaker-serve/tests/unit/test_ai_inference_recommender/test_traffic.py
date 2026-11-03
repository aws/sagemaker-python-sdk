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
"""Unit tests for describe_endpoint_traffic and Workload.from_endpoint."""

from __future__ import absolute_import

import json
import math
import urllib.parse
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError

from sagemaker.serve.ai_inference_recommender import (
    EndpointTraffic,
    TrafficStats,
    Workload,
    describe_endpoint_traffic,
)
from sagemaker.serve.ai_inference_recommender import traffic as traffic_module

DAY = 86_400
NOW = 1_800_000_000
NOW_UTC = datetime.fromtimestamp(NOW, tz=timezone.utc)
ENDPOINT = "my-endpoint"
COMPONENT = "my-ic"


@pytest.fixture(autouse=True)
def _offline_telemetry():
    """Keep the telemetry decorator from sending anything."""
    with patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request"), patch(
        "sagemaker.core.telemetry.telemetry_logging._get_default_sagemaker_session",
        return_value=None,
    ):
        yield


def _histogram(values):
    """A cumulative snapshot of these observations, one bucket of width 10 per distinct value."""
    buckets = {}
    for value in values:
        buckets[value] = buckets.get(value, 0) + 1
    return {
        "count": str(len(values)),
        "sum": str(sum(values)),
        "buckets": [[0, str(v - 5), str(v + 5), str(n)] for v, n in sorted(buckets.items())],
    }


def _both(container, values, framework="vllm"):
    """The same snapshot for both sides, outputs a tenth of the inputs."""
    return {
        (f"{framework}/input", container): _histogram(values),
        (f"{framework}/output", container): _histogram([v / 10 for v in values]),
    }


def _response(status_code=200, body=None):
    response = MagicMock()
    response.status_code = status_code
    response.content = json.dumps(body if body is not None else {}).encode()
    return response


def _params(prepared):
    return dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(prepared.url).query))


class _FakeQueryAPI:
    """Answers instant queries from {days_ago: {(stat, container): snapshot}}; records requests."""

    def __init__(self, days=None):
        self.days = days or {}
        self.requests = []

    def send(self, prepared):
        self.requests.append(prepared)
        at = int(_params(prepared)["time"])
        result = [
            {
                "metric": {"stat": stat, "aws.sagemaker.container.id": container},
                "histogram": [at, snapshot],
            }
            for (stat, container), snapshot in self.days.get((NOW - at) // DAY, {}).items()
        ]
        return _response(
            body={"status": "success", "data": {"resultType": "vector", "result": result}}
        )


def _sagemaker_client(
    components=(COMPONENT,),
    copies=2,
    variants=(("AllTraffic", 3),),
    observability=True,
    capture=None,
):
    client = MagicMock()
    client.describe_endpoint.return_value = {
        "EndpointName": ENDPOINT,
        "EndpointConfigName": "my-config",
        "ProductionVariants": [
            {"VariantName": name, "CurrentInstanceCount": count} for name, count in variants
        ],
        **({"DataCaptureConfig": capture} if capture is not None else {}),
    }
    client.describe_endpoint_config.return_value = (
        {"MetricsConfig": {"EnableDetailedObservability": observability}}
        if observability is not None
        else {}
    )
    client.list_inference_components.return_value = {
        "InferenceComponents": [{"InferenceComponentName": name} for name in components]
    }
    client.describe_inference_component.return_value = {
        "EndpointName": ENDPOINT,
        "VariantName": "AllTraffic",
        "RuntimeConfig": {"CurrentCopyCount": copies},
    }
    return client


def _cloudwatch_client(peaks=(1.0, 2.5), api=None):
    client = MagicMock()
    client.meta.endpoint_url = "https://monitoring.us-east-1.amazonaws.com"
    client._endpoint.http_session = api if api is not None else _FakeQueryAPI()
    client.get_metric_statistics.return_value = {
        "Datapoints": [{"Maximum": peak} for peak in peaks]
    }
    return client


def _session(sagemaker_client, cloudwatch_client):
    session = boto3.session.Session(
        aws_access_key_id="AKIDEXAMPLE",
        aws_secret_access_key="SECRET",
        aws_session_token="TOKEN",
        region_name="us-east-1",
    )
    clients = {"sagemaker": sagemaker_client, "cloudwatch": cloudwatch_client}
    session.client = MagicMock(side_effect=lambda name, **_: clients[name])
    return session


def _describe(api, sagemaker_client=None, cloudwatch_client=None, **kwargs):
    cloudwatch = cloudwatch_client or _cloudwatch_client()
    cloudwatch._endpoint.http_session = api
    session = _session(sagemaker_client or _sagemaker_client(), cloudwatch)
    with patch.object(traffic_module.time, "time", return_value=NOW), patch.object(
        traffic_module.time, "sleep"
    ):
        return describe_endpoint_traffic(ENDPOINT, sagemaker_session=session, **kwargs)


def _times(api):
    return [int(_params(prepared)["time"]) for prepared in api.requests]


class TestTokenSizes:
    def test_the_window_is_each_series_newest_snapshot_less_its_snapshot_at_the_start(self):
        # Container a served 100 and 300 before the window and 200, 200 and 800 in it; container
        # b started in the window and served 500.
        api = _FakeQueryAPI(
            {
                0: {**_both("a", [100, 300, 200, 200, 800]), **_both("b", [500])},
                3: _both("a", [100, 300, 200]),
                7: _both("a", [100, 300]),
            }
        )

        traffic = _describe(api)

        window = [200, 200, 800, 500]
        mean = sum(window) / len(window)
        stddev = round(math.sqrt(sum((x - mean) ** 2 for x in window) / len(window)), 1)
        assert traffic.framework == "vllm"
        assert traffic.requests == 4
        tokens = traffic.input_tokens
        assert (tokens.min, tokens.p50, tokens.max) == (195.0, 205.0, 805.0)
        assert (tokens.mean, tokens.stddev) == (mean, stddev)
        assert traffic.output_tokens.mean == mean / 10
        assert traffic.output_tokens.stddev == round(stddev / 10, 1)

    def test_percentiles_interpolate_within_buckets(self):
        # 100 requests, one in each bucket from [0, 10] to [990, 1000].
        api = _FakeQueryAPI({0: _both("a", list(range(5, 1000, 10)))})

        tokens = _describe(api).input_tokens

        assert tokens == TrafficStats(
            min=0.0,
            p25=250.0,
            p50=500.0,
            p70=700.0,
            p80=800.0,
            p90=900.0,
            p95=950.0,
            p99=990.0,
            max=1000.0,
            mean=500.0,
            stddev=tokens.stddev,
        )
        assert tokens.stddev == pytest.approx(288.7, abs=0.1)

    def test_an_open_ended_bucket_counts_at_its_finite_bound(self):
        snapshot = {
            "count": "4",
            "sum": "10",
            "buckets": [[0, "-Inf", "0", "1"], [0, "0", "4", "2"], [0, "4", "+Inf", "1"]],
        }
        api = _FakeQueryAPI({0: {("vllm/input", "a"): snapshot, ("vllm/output", "a"): snapshot}})

        tokens = _describe(api).input_tokens

        assert (tokens.min, tokens.p25, tokens.p50, tokens.p70, tokens.p80, tokens.max) == (
            0.0,
            0.0,
            2.0,
            3.6,
            4.0,
            4.0,
        )
        assert (tokens.mean, tokens.stddev) == (2.5, round(math.sqrt(2), 1))

    def test_buckets_resized_between_snapshots_keep_the_percentiles_in_order(self):
        # The start snapshot's buckets are finer than the end's: [0, 5] and [5, 10] against [0, 10].
        start = {"count": "2", "sum": "8", "buckets": [[0, "0", "5", "1"], [0, "5", "10", "1"]]}
        end = {"count": "6", "sum": "48", "buckets": [[0, "0", "10", "4"], [0, "10", "20", "2"]]}
        api = _FakeQueryAPI(
            {
                0: {("vllm/input", "a"): end, ("vllm/output", "a"): end},
                7: {("vllm/input", "a"): start, ("vllm/output", "a"): start},
            }
        )

        traffic = _describe(api)

        tokens = traffic.input_tokens
        assert traffic.requests == 4
        assert tokens.mean == 10.0
        percentiles = [tokens.min, tokens.p25, tokens.p50, tokens.p90, tokens.p99, tokens.max]
        assert percentiles == sorted(percentiles)
        assert tokens.max == 20.0

    def test_buckets_short_of_the_count_put_the_top_percentiles_at_the_highest_bound(self):
        # The snapshot counts 4 requests but its buckets hold 2.
        snapshot = {
            "count": "4",
            "sum": "40",
            "buckets": [[0, "0", "10", "1"], [0, "10", "20", "1"]],
        }
        api = _FakeQueryAPI({0: {("vllm/input", "a"): snapshot, ("vllm/output", "a"): snapshot}})

        tokens = _describe(api).input_tokens

        assert (tokens.p25, tokens.p50, tokens.p99, tokens.max) == (10.0, 20.0, 20.0, 20.0)

    def test_reads_each_day_back_from_the_end_and_the_start_with_signed_instant_queries(self):
        api = _FakeQueryAPI({0: _both("a", [10])})

        _describe(api, lookback=timedelta(days=3))

        assert _times(api) == [NOW, NOW - DAY, NOW - 2 * DAY, NOW - 3 * DAY]
        for prepared in api.requests:
            url = urllib.parse.urlsplit(prepared.url)
            query = _params(prepared)["query"]
            assert url.netloc == "monitoring.us-east-1.amazonaws.com"
            assert url.path == "/api/v1/query"
            assert '"aws.sagemaker.endpoint.name"="my-endpoint"' in query
            assert '"aws.sagemaker.inference_component.name"="my-ic"' in query
            assert 'last_over_time({"vllm:request_prompt_tokens"' in query
            assert 'last_over_time({"vllm:request_generation_tokens"' in query
            assert '"stat", "vllm/input"' in query and '"stat", "vllm/output"' in query
            assert "increase(" not in query
            assert "/us-east-1/monitoring/aws4_request" in prepared.headers["Authorization"]

    def test_a_model_on_a_variant_is_selected_by_its_variant_label(self):
        api = _FakeQueryAPI({0: _both("a", [10])})

        traffic = _describe(
            api, sagemaker_client=_sagemaker_client(components=()), variant_name="AllTraffic"
        )

        query = _params(api.requests[0])["query"]
        assert traffic.inference_component_name is None
        assert traffic.variant_name == "AllTraffic"
        assert '"aws.sagemaker.variant.name"="AllTraffic"' in query
        assert "inference_component" not in query

    def test_a_restarted_series_counts_from_zero(self):
        # The container restarted after the window began: its count went from 5 to 2.
        api = _FakeQueryAPI({0: _both("a", [40, 60]), 7: _both("a", [1, 1, 1, 1, 1])})

        traffic = _describe(api)

        assert traffic.requests == 2
        assert traffic.input_tokens.mean == 50.0

    def test_a_restart_between_days_adds_the_requests_from_before_it(self):
        # The series had served 2 requests before the window. In the window it served 3 more,
        # restarted, and served 2 after the restart.
        api = _FakeQueryAPI(
            {
                7: _both("a", [10, 10]),
                3: _both("a", [10, 10, 30, 30, 30]),
                0: _both("a", [50, 50]),
            }
        )

        traffic = _describe(api)

        assert traffic.requests == 5
        assert traffic.input_tokens.mean == (3 * 30 + 2 * 50) / 5
        assert traffic.input_tokens.min == 25.0

    def test_a_bucket_that_goes_down_is_a_restart(self):
        # The count and sum grew, but the bucket around 10 went from 3 to 1: the container
        # restarted, so the later snapshot counts from zero.
        api = _FakeQueryAPI({7: _both("a", [10, 10, 10]), 0: _both("a", [10, 90, 90, 90])})

        traffic = _describe(api)

        assert traffic.requests == 4
        assert traffic.input_tokens.mean == 70.0

    def test_a_sum_that_goes_down_is_a_restart(self):
        api = _FakeQueryAPI({7: _both("a", [50, 50]), 0: _both("a", [10, 10, 10])})

        traffic = _describe(api)

        assert traffic.requests == 3
        assert traffic.input_tokens.mean == 10.0

    def test_a_series_seen_only_before_the_window_is_left_out(self):
        api = _FakeQueryAPI({0: _both("a", [10]), 7: {**_both("a", []), **_both("old", [999])}})

        traffic = _describe(api)

        assert traffic.requests == 1
        assert traffic.input_tokens.mean == 10.0

    def test_the_newest_snapshot_in_the_window_is_the_one_read(self):
        # The series stopped two days ago; the snapshot from then is its newest.
        api = _FakeQueryAPI({2: _both("a", [10, 30]), 5: _both("a", [10])})

        traffic = _describe(api)

        assert traffic.requests == 2
        assert traffic.input_tokens.mean == 20.0

    def test_no_histogram_in_the_window_leaves_the_token_sizes_out(self):
        traffic = _describe(_FakeQueryAPI())

        assert (traffic.framework, traffic.requests) == (None, 0)
        assert traffic.input_tokens is None and traffic.output_tokens is None

    def test_detailed_observability_off_skips_the_token_metrics(self):
        api = _FakeQueryAPI({0: _both("a", [10])})

        traffic = _describe(api, sagemaker_client=_sagemaker_client(observability=False))

        assert api.requests == []
        assert traffic.detailed_observability is False
        assert traffic.input_tokens is None and traffic.framework is None
        assert traffic.concurrency is not None

    def test_an_unset_detailed_observability_still_reads_the_token_metrics(self):
        api = _FakeQueryAPI({0: _both("a", [10])})

        traffic = _describe(api, sagemaker_client=_sagemaker_client(observability=None))

        assert traffic.detailed_observability is None
        assert traffic.input_tokens.mean == 10.0

    def test_a_deleted_endpoint_configuration_still_reads_the_token_metrics(self):
        sagemaker = _sagemaker_client()
        sagemaker.describe_endpoint_config.side_effect = ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find"}},
            "DescribeEndpointConfig",
        )
        api = _FakeQueryAPI({0: _both("a", [10])})

        traffic = _describe(api, sagemaker_client=sagemaker)

        assert traffic.detailed_observability is None
        assert traffic.input_tokens.mean == 10.0

    def test_another_frameworks_histograms_are_read_when_listed(self):
        frameworks = (
            traffic_module._TokenMetrics("vllm", "vllm:prompt", "vllm:generation"),
            traffic_module._TokenMetrics("other", "other:prompt", "other:generation"),
        )
        api = _FakeQueryAPI({0: _both("a", [10], framework="other")})

        with patch.object(traffic_module, "_TOKEN_METRICS", frameworks):
            traffic = _describe(api)

        query = _params(api.requests[0])["query"]
        assert '{"other:prompt"' in query and '"stat", "other/output"' in query
        assert traffic.framework == "other"
        assert traffic.input_tokens.mean == 10.0

    def test_label_values_are_escaped(self):
        assert traffic_module._selector("m", {"l": 'a"b\\c'}) == '{"m", "l"="a\\"b\\\\c"}'


class TestWindow:
    def test_the_default_is_the_seven_days_up_to_now_in_utc(self):
        cloudwatch = _cloudwatch_client()

        traffic = _describe(_FakeQueryAPI(), cloudwatch_client=cloudwatch)

        assert (traffic.start_time, traffic.end_time) == (NOW_UTC - timedelta(days=7), NOW_UTC)
        assert traffic.end_time.tzinfo == timezone.utc
        kwargs = cloudwatch.get_metric_statistics.call_args.kwargs
        assert (kwargs["StartTime"], kwargs["EndTime"]) == (traffic.start_time, traffic.end_time)

    def test_a_lookback_ends_at_the_end_time(self):
        end = NOW_UTC - timedelta(days=10)
        api = _FakeQueryAPI()

        traffic = _describe(api, lookback=timedelta(days=2), end_time=end)

        assert (traffic.start_time, traffic.end_time) == (end - timedelta(days=2), end)
        assert _times(api) == [NOW - 10 * DAY, NOW - 11 * DAY, NOW - 12 * DAY]

    def test_an_absolute_window_in_another_time_zone_is_read_in_utc(self):
        pacific = timezone(timedelta(hours=-8))
        start = datetime(2027, 1, 10, 0, 0, tzinfo=pacific)
        end = datetime(2027, 1, 12, 12, 0, tzinfo=pacific)
        api = _FakeQueryAPI()

        traffic = _describe(api, start_time=start, end_time=end)

        assert traffic.start_time == datetime(2027, 1, 10, 8, 0, tzinfo=timezone.utc)
        assert traffic.end_time == datetime(2027, 1, 12, 20, 0, tzinfo=timezone.utc)
        first, last = int(end.timestamp()), int(start.timestamp())
        assert _times(api) == [first, first - DAY, first - 2 * DAY, last]

    def test_a_window_of_part_of_a_day_reads_the_end_and_the_start(self):
        api = _FakeQueryAPI()

        _describe(api, lookback=timedelta(hours=6))

        assert _times(api) == [NOW, NOW - 6 * 3_600]

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"start_time": datetime(2027, 1, 10)}, "timezone-aware"),
            ({"end_time": datetime(2027, 1, 10)}, "timezone-aware"),
            ({"end_time": "2027-01-10"}, "timezone-aware"),
            ({"lookback": timedelta(0)}, "positive timedelta"),
            ({"lookback": 3}, "positive timedelta"),
            (
                {"lookback": timedelta(days=1), "start_time": NOW_UTC - timedelta(days=1)},
                "not both",
            ),
            ({"end_time": NOW_UTC + timedelta(minutes=1)}, "later than now"),
            ({"start_time": NOW_UTC, "end_time": NOW_UTC}, "before end_time"),
            ({"lookback": timedelta(days=7, seconds=1)}, "at most 7 days"),
            ({"start_time": NOW_UTC - timedelta(days=8)}, "at most 7 days"),
        ],
    )
    def test_bad_windows_are_refused(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _describe(_FakeQueryAPI(), **kwargs)


class TestConcurrency:
    def test_a_components_hourly_peaks_per_copy_are_multiplied_by_its_copies(self):
        cloudwatch = _cloudwatch_client(peaks=(2.5, 1.0, 0.5, 2.0))

        traffic = _describe(_FakeQueryAPI(), cloudwatch_client=cloudwatch)

        # Each hour's maximum per copy, times 2 copies: 1, 2, 4 and 5.
        concurrency = traffic.concurrency
        assert (concurrency.min, concurrency.p50, concurrency.max) == (1.0, 3.0, 5.0)
        assert (concurrency.p25, concurrency.p90, concurrency.mean) == (1.8, 4.7, 3.0)
        assert concurrency.stddev == round(math.sqrt(2.5), 1)
        call = cloudwatch.get_metric_statistics.call_args.kwargs
        assert call["MetricName"] == "ConcurrentRequestsPerCopy"
        assert call["Dimensions"] == [{"Name": "InferenceComponentName", "Value": COMPONENT}]
        assert (call["Namespace"], call["Period"], call["Statistics"]) == (
            "AWS/SageMaker",
            3600,
            ["Maximum"],
        )

    def test_a_variants_hourly_peaks_per_model_are_multiplied_by_its_instances(self):
        cloudwatch = _cloudwatch_client(peaks=(1.5,))

        traffic = _describe(
            _FakeQueryAPI(),
            sagemaker_client=_sagemaker_client(components=()),
            cloudwatch_client=cloudwatch,
        )

        assert traffic.concurrency.max == 4.5
        assert traffic.concurrency.p25 == 4.5
        call = cloudwatch.get_metric_statistics.call_args.kwargs
        assert call["MetricName"] == "ConcurrentRequestsPerModel"
        assert call["Dimensions"] == [
            {"Name": "EndpointName", "Value": ENDPOINT},
            {"Name": "VariantName", "Value": "AllTraffic"},
        ]

    def test_no_traffic_is_no_concurrency(self):
        for peaks in ((), (0.0, 0.0)):
            traffic = _describe(_FakeQueryAPI(), cloudwatch_client=_cloudwatch_client(peaks=peaks))

            assert traffic.concurrency is None

    def test_a_component_with_no_copies_now_counts_one(self):
        traffic = _describe(
            _FakeQueryAPI(),
            sagemaker_client=_sagemaker_client(copies=0),
            cloudwatch_client=_cloudwatch_client(peaks=(3.0,)),
        )

        assert traffic.concurrency.max == 3.0


class TestTarget:
    def test_the_only_component_is_found(self):
        sagemaker = _sagemaker_client()

        traffic = _describe(_FakeQueryAPI(), sagemaker_client=sagemaker)

        assert (traffic.inference_component_name, traffic.variant_name) == (
            COMPONENT,
            "AllTraffic",
        )
        sagemaker.list_inference_components.assert_any_call(EndpointNameEquals=ENDPOINT)
        sagemaker.describe_endpoint_config.assert_called_once_with(EndpointConfigName="my-config")

    @staticmethod
    def _with_adapter(sagemaker):
        """The endpoint's components: base my-ic with 2 copies, and adapter my-adapter on it."""
        described = {
            COMPONENT: {
                "EndpointName": ENDPOINT,
                "VariantName": "AllTraffic",
                "RuntimeConfig": {"CurrentCopyCount": 2},
            },
            "my-adapter": {
                "EndpointName": ENDPOINT,
                "VariantName": "AllTraffic",
                "Specification": {"BaseInferenceComponentName": COMPONENT},
            },
        }
        sagemaker.list_inference_components.return_value = {
            "InferenceComponents": [
                {"InferenceComponentName": COMPONENT},
                {"InferenceComponentName": "my-adapter"},
            ]
        }
        sagemaker.describe_inference_component.side_effect = lambda InferenceComponentName: (
            described[InferenceComponentName]
        )
        return sagemaker

    def test_an_adapter_reads_its_base_component(self):
        api = _FakeQueryAPI({0: _both("a", [10])})
        cloudwatch = _cloudwatch_client(peaks=(3.0,))

        traffic = _describe(
            api,
            sagemaker_client=self._with_adapter(_sagemaker_client()),
            cloudwatch_client=cloudwatch,
            inference_component_name="my-adapter",
        )

        assert traffic.inference_component_name == COMPONENT
        assert traffic.concurrency.max == 6.0
        assert cloudwatch.get_metric_statistics.call_args.kwargs["Dimensions"] == [
            {"Name": "InferenceComponentName", "Value": COMPONENT}
        ]
        assert (
            '"aws.sagemaker.inference_component.name"="my-ic"' in _params(api.requests[0])["query"]
        )

    def test_finding_the_component_skips_adapters(self):
        traffic = _describe(
            _FakeQueryAPI(), sagemaker_client=self._with_adapter(_sagemaker_client())
        )

        assert traffic.inference_component_name == COMPONENT

    def test_several_components_need_a_choice(self):
        with pytest.raises(ValueError, match="inference_component_name"):
            _describe(_FakeQueryAPI(), sagemaker_client=_sagemaker_client(components=("a", "b")))

    def test_several_variants_need_a_choice(self):
        sagemaker = _sagemaker_client(components=(), variants=(("a", 1), ("b", 1)))

        with pytest.raises(ValueError, match="variant_name"):
            _describe(_FakeQueryAPI(), sagemaker_client=sagemaker)

    def test_a_component_on_another_endpoint_is_refused(self):
        sagemaker = _sagemaker_client()
        sagemaker.describe_inference_component.return_value["EndpointName"] = "other"

        with pytest.raises(ValueError, match="is not on endpoint"):
            _describe(
                _FakeQueryAPI(), sagemaker_client=sagemaker, inference_component_name=COMPONENT
            )

    def test_an_unknown_variant_is_refused(self):
        with pytest.raises(ValueError, match="no production variant 'nope'"):
            _describe(_FakeQueryAPI(), variant_name="nope")

    def test_an_endpoint_without_variants_is_refused(self):
        sagemaker = _sagemaker_client(components=(), variants=())

        with pytest.raises(ValueError, match="no production variant"):
            _describe(_FakeQueryAPI(), sagemaker_client=sagemaker)

    def test_components_are_listed_across_pages(self):
        sagemaker = _sagemaker_client()
        sagemaker.list_inference_components.side_effect = [
            {"InferenceComponents": [{"InferenceComponentName": "a"}], "NextToken": "t"},
            {"InferenceComponents": [{"InferenceComponentName": "b"}]},
        ]

        with pytest.raises(ValueError, match=r"\['a', 'b'\]"):
            _describe(_FakeQueryAPI(), sagemaker_client=sagemaker)
        assert sagemaker.list_inference_components.call_args.kwargs["NextToken"] == "t"

    def test_a_component_and_a_variant_together_are_refused(self):
        with pytest.raises(ValueError, match="at most one"):
            _describe(_FakeQueryAPI(), inference_component_name="a", variant_name="b")

    def test_an_endpoint_name_is_required(self):
        with pytest.raises(ValueError, match="endpoint_name"):
            describe_endpoint_traffic("")


class TestDataCapture:
    def test_the_variants_capture_prefix_is_reported(self):
        capture = {"EnableCapture": True, "DestinationS3Uri": "s3://bucket/capture/"}

        traffic = _describe(_FakeQueryAPI(), sagemaker_client=_sagemaker_client(capture=capture))

        assert traffic.data_capture_s3_uri == "s3://bucket/capture/my-endpoint/AllTraffic/"

    @pytest.mark.parametrize(
        "capture",
        [
            None,
            {"EnableCapture": False, "DestinationS3Uri": "s3://bucket/capture"},
            {"EnableCapture": True},
        ],
    )
    def test_no_capture_is_no_prefix(self, capture):
        traffic = _describe(_FakeQueryAPI(), sagemaker_client=_sagemaker_client(capture=capture))

        assert traffic.data_capture_s3_uri is None


class TestQueryAPIErrors:
    """Token metrics are read best effort: a failed read is recorded, and the rest is returned."""

    def test_access_denied_names_the_permissions(self, caplog):
        api = MagicMock()
        api.send.return_value = _response(403, {"message": "not authorized"})

        traffic = _describe(api)

        assert "cloudwatch:GetMetricData and cloudwatch:ListMetrics" in traffic.token_metrics_error
        assert traffic.input_tokens is None and traffic.concurrency is not None
        assert "Couldn't read the token metrics" in caplog.text
        assert api.send.call_count == 1

    def test_other_http_errors_and_failed_queries_are_recorded(self):
        api = MagicMock()
        api.send.return_value = _response(422, {"error": "range too long"})
        assert "HTTP 422" in _describe(api).token_metrics_error

        api.send.return_value = _response(200, {"status": "error", "error": "parse error"})
        assert "parse error" in _describe(api).token_metrics_error

    def test_a_body_that_isnt_json_is_recorded(self):
        api = MagicMock()
        response = _response(200)
        response.content = b"<html>proxy error</html>"
        api.send.return_value = response

        assert "isn't JSON" in _describe(api).token_metrics_error

    def test_throttling_and_server_errors_are_retried(self):
        fake = _FakeQueryAPI({0: _both("a", [10])})
        failures = iter([_response(429), _response(503)])
        api = MagicMock()
        api.send.side_effect = lambda prepared: next(failures, None) or fake.send(prepared)

        traffic = _describe(api)

        assert traffic.token_metrics_error is None
        assert traffic.input_tokens.mean == 10.0

    def test_a_query_gives_up_after_its_attempts(self):
        api = MagicMock()
        api.send.return_value = _response(503)

        traffic = _describe(api)

        assert "HTTP 503" in traffic.token_metrics_error
        assert api.send.call_count == traffic_module._QUERY_ATTEMPTS

    def test_an_unreachable_api_is_retried_then_recorded(self):
        api = MagicMock()
        api.send.side_effect = EndpointConnectionError(endpoint_url="https://monitoring.example")

        traffic = _describe(api)

        assert "Couldn't reach" in traffic.token_metrics_error
        assert api.send.call_count == traffic_module._QUERY_ATTEMPTS

    def test_queries_go_through_the_cloudwatch_clients_http_session(self):
        api = _FakeQueryAPI({0: _both("a", [10])})

        _describe(api)

        assert api.requests and all(
            "/us-east-1/monitoring/aws4_request" in p.headers["Authorization"] for p in api.requests
        )

    def test_no_credentials_raise(self):
        session = MagicMock(spec=boto3.session.Session)
        session.get_credentials.return_value = None
        session.region_name = "us-east-1"

        with pytest.raises(RuntimeError, match="No AWS credentials"):
            traffic_module._PromQLClient(session, _cloudwatch_client())

    def test_no_region_raises(self):
        session = MagicMock(spec=boto3.session.Session)
        session.region_name = None

        with pytest.raises(RuntimeError, match="no region"):
            traffic_module._PromQLClient(session, _cloudwatch_client())


class TestSessionHandling:
    def test_a_sagemaker_session_is_unwrapped(self):
        boto_session = _session(_sagemaker_client(), _cloudwatch_client())
        sm_session = MagicMock()
        sm_session.boto_session = boto_session
        sm_session.sagemaker_config = {}
        with patch.object(traffic_module.time, "time", return_value=NOW):
            traffic = describe_endpoint_traffic(ENDPOINT, sagemaker_session=sm_session)

        assert traffic.inference_component_name == COMPONENT
        boto_session.client.assert_any_call("sagemaker")


def _stats(mean, stddev, peak=None):
    """A TrafficStats with this mean, standard deviation and maximum."""
    peak = mean if peak is None else peak
    return TrafficStats(
        min=0.0,
        p25=mean,
        p50=mean,
        p70=mean,
        p80=mean,
        p90=mean,
        p95=mean,
        p99=peak,
        max=peak,
        mean=mean,
        stddev=stddev,
    )


class TestWorkloadFromEndpoint:
    _DESCRIBE = "sagemaker.serve.ai_inference_recommender.workload.describe_endpoint_traffic"

    def _traffic(self, **overrides):
        values = {
            "endpoint_name": ENDPOINT,
            "inference_component_name": COMPONENT,
            "variant_name": "AllTraffic",
            "start_time": NOW_UTC - timedelta(days=7),
            "end_time": NOW_UTC,
            "concurrency": _stats(3.0, 1.2, peak=7.5),
            "detailed_observability": True,
            "framework": "vllm",
            "requests": 1234,
            "input_tokens": _stats(812.4, 230.1),
            "output_tokens": _stats(196.6, 75.3),
            "token_metrics_error": None,
            "data_capture_s3_uri": None,
        }
        values.update(overrides)
        return EndpointTraffic(**values)

    def test_sizes_a_synthetic_workload_from_the_token_metrics(self):
        session = MagicMock()
        start, end = NOW_UTC - timedelta(days=3), NOW_UTC - timedelta(days=1)
        with patch(self._DESCRIBE, return_value=self._traffic()) as describe:
            workload = Workload.from_endpoint(
                ENDPOINT,
                inference_component_name=COMPONENT,
                start_time=start,
                end_time=end,
                sagemaker_session=session,
                tokenizer="org/model",
                hf_token="arn:aws:secretsmanager:us-east-1:123456789012:secret:hf-AbCdEf",
                request_count=200,
            )

        describe.assert_called_once_with(
            ENDPOINT,
            inference_component_name=COMPONENT,
            variant_name=None,
            lookback=None,
            start_time=start,
            end_time=end,
            sagemaker_session=session,
        )
        assert workload.parameters == {
            "tokenizer": "org/model",
            "concurrency": 8,
            "request_count": 200,
            "prompt_input_tokens_mean": 812,
            "prompt_input_tokens_stddev": 230.1,
            "output_tokens_mean": 197,
            "output_tokens_stddev": 75.3,
            "streaming": True,
        }
        assert workload.secrets == {
            "hf_token": "arn:aws:secretsmanager:us-east-1:123456789012:secret:hf-AbCdEf"
        }
        assert workload.dataset_channels == []

    _CAPTURES = "s3://bucket/capture/my-endpoint/AllTraffic/"

    def test_captured_requests_are_replayed_when_asked(self, caplog):
        with patch(self._DESCRIBE, return_value=self._traffic(data_capture_s3_uri=self._CAPTURES)):
            workload = Workload.from_endpoint(ENDPOINT, use_data_capture=True, request_count=50)

        assert [channel.s3_uri for channel in workload.dataset_channels] == [self._CAPTURES]
        assert workload.parameters["custom_dataset_type"] == "sagemaker-datacapture"
        assert (workload.parameters["concurrency"], workload.parameters["request_count"]) == (8, 50)
        assert self._CAPTURES in caplog.text

    def test_captured_requests_arent_replayed_by_default(self):
        with patch(self._DESCRIBE, return_value=self._traffic(data_capture_s3_uri=self._CAPTURES)):
            workload = Workload.from_endpoint(ENDPOINT, tokenizer="org/model")

        assert workload.dataset_channels == []
        assert workload.parameters["prompt_input_tokens_mean"] == 812

    def test_asking_for_a_replay_without_captures_uses_synthetic_prompts(self, caplog):
        with patch(self._DESCRIBE, return_value=self._traffic()):
            workload = Workload.from_endpoint(
                ENDPOINT, use_data_capture=True, tokenizer="org/model"
            )

        assert workload.dataset_channels == []
        assert "doesn't capture its requests" in caplog.text

    def test_a_given_concurrency_wins(self):
        with patch(self._DESCRIBE, return_value=self._traffic(concurrency=None)):
            workload = Workload.from_endpoint(ENDPOINT, tokenizer="org/model", concurrency=4)

        assert workload.parameters["concurrency"] == 4

    @pytest.mark.parametrize(
        "observability, reason",
        [(False, "turns detailed observability off"), (None, "need detailed observability on")],
    )
    def test_no_token_metrics_is_refused(self, observability, reason):
        traffic = self._traffic(
            input_tokens=None,
            output_tokens=None,
            requests=0,
            framework=None,
            detailed_observability=observability,
        )
        with patch(self._DESCRIBE, return_value=traffic):
            with pytest.raises(ValueError, match=f"published no token metrics .*{reason}"):
                Workload.from_endpoint(ENDPOINT, tokenizer="org/model")

    def test_a_failed_token_read_says_why(self):
        traffic = self._traffic(
            input_tokens=None,
            output_tokens=None,
            requests=0,
            framework=None,
            token_metrics_error="CloudWatch denied the query (HTTP 403).",
        )
        with patch(self._DESCRIBE, return_value=traffic):
            with pytest.raises(ValueError, match=r"couldn't be read \(CloudWatch denied"):
                Workload.from_endpoint(ENDPOINT, tokenizer="org/model")

    def test_no_traffic_without_a_concurrency_is_refused(self):
        with patch(self._DESCRIBE, return_value=self._traffic(concurrency=None)):
            with pytest.raises(ValueError, match="pass concurrency"):
                Workload.from_endpoint(ENDPOINT, tokenizer="org/model")
