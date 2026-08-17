# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for BenchmarkResult / BenchmarkMetrics."""
from __future__ import absolute_import

import io
import json
import tarfile

import boto3
import pytest
from botocore.stub import Stubber

from sagemaker.serve.ai_inference_recommender import (
    BenchmarkMetric,
    BenchmarkMetrics,
    BenchmarkResult,
    BenchmarkSearchResult,
)


SAMPLE_PROFILE = {
    "request_throughput": {"avg": 12.5, "min": 10.0, "max": 15.0, "unit": "req/s"},
    "request_latency": {
        "avg": 800.0,
        "min": 200.0,
        "max": 1500.0,
        "p50": 700.0,
        "p90": 1200.0,
        "p99": 1450.0,
        "unit": "ms",
    },
    "time_to_first_token": {"avg": 150.5, "min": 50.0, "max": 300.0, "unit": "ms"},
    "inter_token_latency": {"avg": 25.0, "min": 5.0, "max": 100.0, "unit": "ms"},
    "output_token_throughput": {"avg": 320.0, "unit": "tokens/s"},
    "config": {"some": "scalar-not-a-metric"},
}


def _make_output_archive() -> bytes:
    """Build an in-memory output.tar.gz that contains profile_export_aiperf.json."""
    profile_bytes = json.dumps(SAMPLE_PROFILE).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="profile_export_aiperf.json")
        info.size = len(profile_bytes)
        tar.addfile(info, io.BytesIO(profile_bytes))
    return buf.getvalue()


class TestBenchmarkMetricsFromProfileJson:
    def test_typed_metrics_populated(self):
        metrics = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE)
        assert metrics.request_throughput.avg == 12.5
        assert metrics.request_latency.p50 == 700.0
        assert metrics.request_latency.p90 == 1200.0
        assert metrics.time_to_first_token.unit == "ms"
        assert metrics.inter_token_latency.avg == 25.0
        assert metrics.output_token_throughput.avg == 320.0

    def test_non_metric_keys_ignored(self):
        metrics = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE)
        assert "config" not in metrics.all_metrics

    def test_get_returns_known_metric(self):
        metrics = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE)
        assert metrics.get("request_throughput").avg == 12.5
        assert metrics.get("does_not_exist") is None

    def test_metric_raw_preserves_full_dict(self):
        metric = BenchmarkMetric.from_dict("x", {"avg": 1, "unit": "s", "extra": "kept"})
        assert metric.raw == {"avg": 1, "unit": "s", "extra": "kept"}


class TestBenchmarkResultFromS3:
    def test_downloads_archive_and_parses(self):
        archive_bytes = _make_output_archive()
        s3_client = boto3.session.Session(region_name="us-east-1").client("s3")
        with Stubber(s3_client) as stub:
            stub.add_response(
                "list_objects_v2",
                {
                    "Contents": [
                        {"Key": "results/job-abc/output.tar.gz", "Size": len(archive_bytes)},
                    ],
                    "KeyCount": 1,
                },
                expected_params={"Bucket": "my-bucket", "Prefix": "results/job-abc/"},
            )
            stub.add_response(
                "get_object",
                {"Body": _StreamingBody(archive_bytes)},
                expected_params={"Bucket": "my-bucket", "Key": "results/job-abc/output.tar.gz"},
            )

            class _SessionStub:
                def client(self, name):
                    return s3_client

            result = BenchmarkResult.from_s3(
                "s3://my-bucket/results/job-abc/", session=_SessionStub()
            )

        assert isinstance(result, BenchmarkResult)
        assert result.s3_output_location == "s3://my-bucket/results/job-abc/"
        assert result.metrics.request_throughput.avg == 12.5
        assert result.profile == SAMPLE_PROFILE


class _StreamingBody:
    """Minimal Body shim returning bytes for botocore stubbed responses."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, *_, **__):
        return self._data


class TestBenchmarkResultRepr:
    """Lock down the table-formatted __repr__ on BenchmarkResult and BenchmarkMetrics."""

    def _result(self) -> BenchmarkResult:
        metrics = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE)
        return BenchmarkResult(
            metrics=metrics,
            s3_output_location="s3://bucket/results/",
            profile=dict(SAMPLE_PROFILE),
        )

    def test_str_contains_header_and_table_columns(self):
        text = str(self._result())
        assert "BenchmarkResult" in text
        assert "s3://bucket/results/" in text
        # table headers
        for column in ("metric", "unit", "avg", "p50", "p90", "p99"):
            assert column in text

    def test_str_lists_well_known_metrics(self):
        text = str(self._result())
        for metric_name in (
            "request_throughput",
            "request_latency",
            "time_to_first_token",
            "inter_token_latency",
            "output_token_throughput",
        ):
            assert metric_name in text

    def test_metrics_str_independently_renders_table(self):
        text = str(self._result().metrics)
        assert "request_throughput" in text
        assert "request_latency" in text
        # BenchmarkMetrics.__str__ is a bare table — no wrapping header.
        assert "BenchmarkResult" not in text

    def test_repr_is_concise_single_line(self):
        # The multi-line table lives in __str__; __repr__ stays short so it
        # doesn't spam logs/tracebacks/nested containers.
        text = repr(self._result())
        assert "\n" not in text
        assert text.startswith("BenchmarkResult(")


pd = pytest.importorskip("pandas")


class TestToDataFrame:
    """to_dataframe() on BenchmarkMetrics and BenchmarkResult."""

    def _result(self) -> BenchmarkResult:
        # SAMPLE_PROFILE plus an http_* metric, so ordering can be asserted.
        profile = dict(SAMPLE_PROFILE)
        profile["http_req_waiting"] = {"avg": 90.0, "p50": 80.0, "unit": "ms"}
        return BenchmarkResult(
            metrics=BenchmarkMetrics.from_profile_json(profile),
            s3_output_location="s3://bucket/results/",
        )

    def test_metrics_dataframe_shape_and_columns(self):
        df = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE).to_dataframe()
        assert df.index.name == "metric"
        assert list(df.columns) == [
            "unit",
            "avg",
            "min",
            "max",
            "p50",
            "p90",
            "p95",
            "p99",
            "stddev",
        ]
        # one row per parsed metric
        assert set(df.index) == set(BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE).all_metrics)

    def test_metrics_dataframe_carries_stats_the_text_table_drops(self):
        # min/max are absent from the printed table but present in the frame.
        df = BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE).to_dataframe()
        assert df.loc["request_throughput", "avg"] == 12.5
        assert df.loc["request_throughput", "min"] == 10.0
        assert df.loc["request_throughput", "max"] == 15.0
        # request_latency carries min/max/p99 that the printed table omits.
        assert df.loc["request_latency", "min"] == 200.0
        assert df.loc["request_latency", "p99"] == 1450.0

    def test_result_dataframe_orders_headline_first_http_last(self):
        df = self._result().to_dataframe()
        names = list(df.index)
        # headline metric leads; http_* transport metric trails.
        assert names[0] == "request_throughput"
        assert names[-1] == "http_req_waiting"

    def test_result_and_metrics_frames_share_columns(self):
        result = self._result()
        assert list(result.to_dataframe().columns) == list(result.metrics.to_dataframe().columns)

    def test_search_result_dataframe_raises(self):
        search = BenchmarkResult(
            metrics=BenchmarkMetrics.from_profile_json({}),
            s3_output_location="s3://b/s/",
            search=BenchmarkSearchResult(swept_dim="concurrency", winner=8),
        )
        with pytest.raises(ValueError, match="search/sweep"):
            search.to_dataframe()

    def test_empty_metrics_yield_empty_frame_with_schema(self):
        df = BenchmarkMetrics.from_profile_json({}).to_dataframe()
        assert len(df) == 0
        assert "avg" in df.columns

    def _table_metric_order(self, text):
        """Metric names in the order the printed table lists them — the first
        column of each body row, after the header + separator lines."""
        names = []
        known = self._all_names()
        for line in text.splitlines():
            # Keep the first token of each line only when it is a known metric
            # name, skipping the header, separator, and prose lines.
            first = line.strip().split(" ")[0] if line.strip() else ""
            if first in known:
                names.append(first)
        return names

    def _all_names(self):
        profile = dict(SAMPLE_PROFILE)
        profile["http_req_waiting"] = {"avg": 90.0, "unit": "ms"}
        return set(BenchmarkMetrics.from_profile_json(profile).all_metrics)

    def test_metrics_str_and_dataframe_index_agree_on_order(self):
        """BenchmarkMetrics.__str__ and .to_dataframe() list metrics in the
        same order, via the shared _ordered_metric_pairs()."""
        metrics = self._result().metrics
        table_order = self._table_metric_order(str(metrics))
        frame_order = list(metrics.to_dataframe().index)
        assert table_order == frame_order

    def test_result_str_and_dataframe_index_agree_on_order(self):
        """Same order invariant for BenchmarkResult (headline-first)."""
        result = self._result()
        table_order = self._table_metric_order(str(result))
        frame_order = list(result.to_dataframe().index)
        assert table_order == frame_order


class TestBenchmarkResultMetadataFields:
    """Lock down endpoint / workload_config / tool_version on BenchmarkResult."""

    def _result(self, **overrides) -> BenchmarkResult:
        kwargs = dict(
            metrics=BenchmarkMetrics.from_profile_json(SAMPLE_PROFILE),
            s3_output_location="s3://bucket/results/",
            endpoint="my-endpoint",
            workload_config="my-wl-cfg",
            tool_version="0.6.0",
            profile=dict(SAMPLE_PROFILE),
        )
        kwargs.update(overrides)
        return BenchmarkResult(**kwargs)

    def test_str_shows_endpoint_workload_and_tool_version(self):
        text = str(self._result())
        assert "my-endpoint" in text
        assert "my-wl-cfg" in text
        assert "0.6.0" in text

    def test_str_renders_dashes_when_metadata_missing(self):
        text = str(self._result(endpoint=None, workload_config=None, tool_version=None))
        assert "endpoint:           -" in text
        assert "workload_config:    -" in text
        assert "tool_version:       -" in text

    def test_from_s3_threads_endpoint_and_workload_config_through(self):
        archive_bytes = _make_output_archive()
        s3_client = boto3.session.Session(region_name="us-east-1").client("s3")
        with Stubber(s3_client) as stub:
            stub.add_response(
                "list_objects_v2",
                {
                    "Contents": [
                        {"Key": "results/job-x/output.tar.gz", "Size": len(archive_bytes)},
                    ],
                    "KeyCount": 1,
                },
                expected_params={"Bucket": "b", "Prefix": "results/job-x/"},
            )
            stub.add_response(
                "get_object",
                {"Body": _StreamingBody(archive_bytes)},
                expected_params={"Bucket": "b", "Key": "results/job-x/output.tar.gz"},
            )

            class _SessionStub:
                def client(self, name):
                    return s3_client

            result = BenchmarkResult.from_s3(
                "s3://b/results/job-x/",
                session=_SessionStub(),
                endpoint="ep-1",
                workload_config="wl-1",
            )
        assert result.endpoint == "ep-1"
        assert result.workload_config == "wl-1"

    def test_tool_version_pulled_from_profile_top_level(self):
        archive_bytes = self._archive_with({**SAMPLE_PROFILE, "aiperf_version": "0.6.1"})
        result = self._parse_archive(archive_bytes)
        assert result.tool_version == "0.6.1"

    def test_tool_version_pulled_from_profile_metadata(self):
        archive_bytes = self._archive_with(
            {
                **SAMPLE_PROFILE,
                "metadata": {"version": "0.6.2"},
            }
        )
        result = self._parse_archive(archive_bytes)
        assert result.tool_version == "0.6.2"

    def test_tool_version_none_when_not_present(self):
        result = self._parse_archive(_make_output_archive())
        assert result.tool_version is None

    @staticmethod
    def _archive_with(profile) -> bytes:
        profile_bytes = json.dumps(profile).encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="profile_export_aiperf.json")
            info.size = len(profile_bytes)
            tar.addfile(info, io.BytesIO(profile_bytes))
        return buf.getvalue()

    @staticmethod
    def _parse_archive(archive_bytes: bytes) -> BenchmarkResult:
        s3_client = boto3.session.Session(region_name="us-east-1").client("s3")
        with Stubber(s3_client) as stub:
            stub.add_response(
                "list_objects_v2",
                {
                    "Contents": [
                        {"Key": "p/output.tar.gz", "Size": len(archive_bytes)},
                    ],
                    "KeyCount": 1,
                },
                expected_params={"Bucket": "b", "Prefix": "p/"},
            )
            stub.add_response(
                "get_object",
                {"Body": _StreamingBody(archive_bytes)},
                expected_params={"Bucket": "b", "Key": "p/output.tar.gz"},
            )

            class _SessionStub:
                def client(self, name):
                    return s3_client

            return BenchmarkResult.from_s3("s3://b/p/", session=_SessionStub())


# A concurrency search writes search_history.json at the artifact root. Schema
# per aiperf v0.11.0 (docs/sweeping/search-recipes.md): boundary_summary carries
# the winning (feasible_max) and first-breaching (infeasible_min) swept values.
SAMPLE_SEARCH_HISTORY = {
    "aiperf_version": "0.11.0",
    "boundary_summary": {
        "swept_dim_path": "phases.profiling.concurrency",
        "feasible_max": {"value": 256, "iteration_idx": 0, "objective_value": 4172.3},
        "infeasible_min": {
            "value": 512,
            "first_breach": {
                "metric_tag": "time_to_first_token",
                "stat": "p95",
                "op": "lt",
                "threshold": 200.0,
                "observed": 213.4,
            },
        },
    },
}


def _make_search_archive(history=None, include_per_trial_profile=True) -> bytes:
    """Build an output.tar.gz for a sweep run.

    Mirrors what the benchmark container ships: search_history.json at the root
    and (optionally) a per-trial profile_export_aiperf.json nested in a swept
    level's subdir — the exact shape that would trip a naive suffix match.
    """
    history_bytes = json.dumps(history or SAMPLE_SEARCH_HISTORY).encode("utf-8")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="search_history.json")
        info.size = len(history_bytes)
        tar.addfile(info, io.BytesIO(history_bytes))
        if include_per_trial_profile:
            # A per-level export shares the single-run profile filename; it must
            # NOT be mistaken for the headline profile.
            trial = json.dumps({"request_throughput": {"avg": 999.0}}).encode("utf-8")
            tinfo = tarfile.TarInfo(
                name="search_iter_0000/profile_runs/run_0000/profile_export_aiperf.json"
            )
            tinfo.size = len(trial)
            tar.addfile(tinfo, io.BytesIO(trial))
    return buf.getvalue()


def _parse_archive_from_s3(archive_bytes: bytes) -> BenchmarkResult:
    s3_client = boto3.session.Session(region_name="us-east-1").client("s3")
    with Stubber(s3_client) as stub:
        stub.add_response(
            "list_objects_v2",
            {
                "Contents": [{"Key": "p/output.tar.gz", "Size": len(archive_bytes)}],
                "KeyCount": 1,
            },
            expected_params={"Bucket": "b", "Prefix": "p/"},
        )
        stub.add_response(
            "get_object",
            {"Body": _StreamingBody(archive_bytes)},
            expected_params={"Bucket": "b", "Key": "p/output.tar.gz"},
        )

        class _SessionStub:
            def client(self, name):
                return s3_client

        return BenchmarkResult.from_s3("s3://b/p/", session=_SessionStub())


class TestBenchmarkSearchResultFromHistory:
    def test_winner_and_boundary_parsed(self):
        result = BenchmarkSearchResult.from_history_json(SAMPLE_SEARCH_HISTORY)
        assert result.swept_dim == "phases.profiling.concurrency"
        assert result.winner == 256.0
        assert result.winner_objective == 4172.3
        assert result.infeasible_min == 512.0
        assert result.first_breach["metric_tag"] == "time_to_first_token"
        assert result.raw == SAMPLE_SEARCH_HISTORY

    def test_no_infeasible_min_when_no_sla_breach(self):
        # No SLA filter → nothing breaches → infeasible_min is null, winner is
        # the highest swept value.
        history = {
            "boundary_summary": {
                "swept_dim_path": "phases.profiling.concurrency",
                "feasible_max": {"value": 1024},
                "infeasible_min": None,
            }
        }
        result = BenchmarkSearchResult.from_history_json(history)
        assert result.winner == 1024.0
        assert result.infeasible_min is None
        assert result.first_breach is None

    def test_null_boundary_summary_yields_no_winner(self):
        # Multi-dim search (or one that never resolved a boundary) → null block.
        # We keep the raw history but report no winner rather than fabricating one.
        history = {"boundary_summary": None, "iterations": []}
        result = BenchmarkSearchResult.from_history_json(history)
        assert result.winner is None
        assert result.swept_dim is None
        assert result.raw == history

    def test_missing_boundary_summary_key(self):
        result = BenchmarkSearchResult.from_history_json({"iterations": []})
        assert result.winner is None
        assert result.raw == {"iterations": []}


class TestBenchmarkResultFromS3Search:
    def test_search_run_parsed_from_history_not_per_trial_profile(self):
        result = _parse_archive_from_s3(_make_search_archive())
        # The result must reflect the search, NOT the stray per-trial profile
        # (request_throughput avg 999.0) that shares the single-run filename.
        assert result.is_search is True
        assert result.search is not None
        assert result.search.winner == 256.0
        assert result.metrics.all_metrics == {}
        assert result.metrics.request_throughput is None
        assert result.profile == SAMPLE_SEARCH_HISTORY
        assert result.tool_version == "0.11.0"

    def test_search_history_preferred_even_without_per_trial_profile(self):
        result = _parse_archive_from_s3(_make_search_archive(include_per_trial_profile=False))
        assert result.is_search is True
        assert result.search.swept_dim == "phases.profiling.concurrency"

    def test_single_run_still_not_flagged_as_search(self):
        # Regression guard: the single-run path must be untouched.
        result = _parse_archive_from_s3(_make_output_archive())
        assert result.is_search is False
        assert result.search is None
        assert result.metrics.request_throughput.avg == 12.5

    def test_str_renders_search_summary(self):
        result = _parse_archive_from_s3(_make_search_archive())
        text = str(result)
        assert "BenchmarkResult (search)" in text
        assert "phases.profiling.concurrency" in text
        assert "256" in text


class TestFindObjectPagination:
    """_find_object must paginate so keys beyond the first page are found."""

    def _client_with_pages(self, pages):
        class _Paginator:
            def paginate(self, **kwargs):
                for page in pages:
                    yield page

        class _Client:
            def get_paginator(self, name):
                assert name == "list_objects_v2"
                return _Paginator()

        return _Client()

    def test_finds_object_on_second_page(self):
        from sagemaker.serve.ai_inference_recommender.result import _find_object

        pages = [
            {"Contents": [{"Key": "p/other-000{}.json".format(i)} for i in range(3)]},
            {"Contents": [{"Key": "p/nested/output.tar.gz"}]},
        ]
        key = _find_object(self._client_with_pages(pages), "b", "p/", "output.tar.gz")
        assert key == "p/nested/output.tar.gz"

    def test_raises_when_absent_across_all_pages(self):
        from sagemaker.serve.ai_inference_recommender.result import _find_object

        pages = [{"Contents": [{"Key": "p/a.json"}]}, {"Contents": [{"Key": "p/b.json"}]}]
        with pytest.raises(FileNotFoundError, match="output.tar.gz"):
            _find_object(self._client_with_pages(pages), "b", "p/", "output.tar.gz")
