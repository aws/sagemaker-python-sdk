# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for BenchmarkResult.from_job and the _RecommendationView wrapper."""
from __future__ import absolute_import

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sagemaker.core.shapes.shapes import (
    AIBenchmarkEndpoint,
    AIBenchmarkOutputResult,
    AIBenchmarkTarget,
)
from sagemaker.serve.ai_inference_recommender import (
    BenchmarkJob,
    BenchmarkResult,
)
from sagemaker.serve.ai_inference_recommender._recommendation_view import (
    _RecommendationView,
    _RecommendationsView,
)


def _bench_job(uri):
    return BenchmarkJob(
        ai_benchmark_job_name="bench-1",
        output_config=AIBenchmarkOutputResult(s3_output_location=uri),
    )


class TestBenchmarkResultFromJob:
    def test_calls_from_s3_with_known_location(self):
        job = _bench_job("s3://bucket/results/bench-1/")
        with patch.object(BenchmarkResult, "from_s3", return_value="PARSED") as from_s3:
            assert BenchmarkResult.from_job(job) == "PARSED"
            from_s3.assert_called_once_with(
                "s3://bucket/results/bench-1/",
                session=None,
                endpoint=None,
                workload_config=None,
            )

    def test_refreshes_when_output_missing_then_succeeds(self):
        job = BenchmarkJob(ai_benchmark_job_name="bench-1")

        def _refresh():
            job.output_config = AIBenchmarkOutputResult(
                s3_output_location="s3://bucket/results/bench-1/"
            )
            return job

        with patch.object(BenchmarkJob, "refresh", side_effect=_refresh) as mock_refresh:
            with patch.object(BenchmarkResult, "from_s3", return_value="PARSED") as from_s3:
                assert BenchmarkResult.from_job(job) == "PARSED"
                mock_refresh.assert_called_once()
                from_s3.assert_called_once()

    def test_raises_with_wait_hint_when_job_in_progress(self):
        job = BenchmarkJob(
            ai_benchmark_job_name="bench-1", ai_benchmark_job_status="InProgress"
        )
        with patch.object(BenchmarkJob, "refresh", return_value=job):
            with pytest.raises(RuntimeError, match="has not finished.*job.wait"):
                BenchmarkResult.from_job(job)

    def test_raises_with_failure_reason_when_job_failed(self):
        job = BenchmarkJob(
            ai_benchmark_job_name="bench-1",
            ai_benchmark_job_status="Failed",
            failure_reason="capacity error",
        )
        with patch.object(BenchmarkJob, "refresh", return_value=job):
            with pytest.raises(RuntimeError, match="capacity error"):
                BenchmarkResult.from_job(job)

    def test_threads_endpoint_and_workload_config_into_from_s3(self):
        job = BenchmarkJob(
            ai_benchmark_job_name="bench-1",
            output_config=AIBenchmarkOutputResult(
                s3_output_location="s3://bucket/results/bench-1/"
            ),
            benchmark_target=AIBenchmarkTarget(
                endpoint=AIBenchmarkEndpoint(identifier="my-ep")
            ),
            ai_workload_config_identifier="my-wl-cfg",
        )
        with patch.object(BenchmarkResult, "from_s3", return_value="PARSED") as from_s3:
            assert BenchmarkResult.from_job(job) == "PARSED"
            from_s3.assert_called_once_with(
                "s3://bucket/results/bench-1/",
                session=None,
                endpoint="my-ep",
                workload_config="my-wl-cfg",
            )


def _fake_recommendation_row():
    return SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package/p/1",
            inference_specification_name="my-spec",
        ),
        deployment_configuration=SimpleNamespace(
            instance_type="ml.g6.12xlarge",
            instance_count=1,
            copy_count_per_instance=4,
            image_uri="111122223333.dkr.ecr.us-west-2.amazonaws.com/example:latest",
            environment_variables={"MY_VAR_A": "128", "MY_VAR_B": "1"},
        ),
        expected_performance=[
            SimpleNamespace(metric="RequestThroughput", stat="avg", value=28.42, unit="Requests/Second"),
            SimpleNamespace(metric="RequestLatency", stat="p99", value=4639.0, unit="Milliseconds"),
        ],
    )


class TestRecommendationView:
    def test_repr_renders_config_and_perf_table(self):
        text = repr(_RecommendationView(_fake_recommendation_row(), index=0))
        assert "Recommendation[0]" in text
        # Deployment config
        assert "ml.g6.12xlarge" in text
        assert "model-package/p/1" in text
        assert "my-spec" in text
        # The label uses the new "recommendation_spec_name" naming, not the
        # legacy "inference_spec_name" (which collides with the model-package
        # InferenceSpecification concept).
        assert "recommendation_spec_name:" in text
        assert "inference_spec_name:" not in text
        # Env vars
        assert "MY_VAR_A = 128" in text
        # Expected performance table
        for column in ("metric", "stat", "value", "unit"):
            assert column in text
        assert "RequestThroughput" in text

    def test_attribute_access_forwards_to_raw(self):
        raw = _fake_recommendation_row()
        view = _RecommendationView(raw)
        assert view.deployment_configuration is raw.deployment_configuration
        assert view.deployment_configuration.instance_type == "ml.g6.12xlarge"
        assert view.raw is raw

    def test_recommendation_spec_name_property(self):
        view = _RecommendationView(_fake_recommendation_row())
        assert view.recommendation_spec_name == "my-spec"
        # And on a row whose model_details lacks the field:
        sparse = SimpleNamespace(model_details=None, deployment_configuration=None, expected_performance=[])
        assert _RecommendationView(sparse).recommendation_spec_name is None

    def test_handles_missing_optional_fields(self):
        sparse = SimpleNamespace(
            model_details=None,
            deployment_configuration=None,
            expected_performance=[],
        )
        text = repr(_RecommendationView(sparse))
        assert "Recommendation[0]" in text
        assert "instance_type:        -" in text
        assert "model_package:        -" in text


def _fake_two_recommendations():
    """Two rows that differ enough that the comparative table can render them distinctly."""
    row_a = SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn="arn:.../p/A",
            inference_specification_name="high-otps-on-g5-2xlarge",
        ),
        deployment_configuration=SimpleNamespace(
            instance_type="ml.g5.2xlarge",
            instance_count=1,
            copy_count_per_instance=1,
            image_uri="111122223333.dkr.ecr.us-west-2.amazonaws.com/example:0.36.0-lmi25.0.0-cu130",
            environment_variables={},
        ),
        expected_performance=[
            SimpleNamespace(metric="RequestThroughput", stat="avg", value=152.94, unit="Requests/Second"),
            SimpleNamespace(metric="OutputTokenThroughput", stat="avg", value=4893.7, unit="Tokens/Second"),
            SimpleNamespace(metric="RequestLatency", stat="p50", value=402.6, unit="Milliseconds"),
            SimpleNamespace(metric="RequestLatency", stat="p99", value=481.6, unit="Milliseconds"),
        ],
    )
    row_b = SimpleNamespace(
        model_details=SimpleNamespace(
            model_package_arn="arn:.../p/B",
            inference_specification_name="high-otps-on-g5-2xlarge-1",
        ),
        deployment_configuration=SimpleNamespace(
            instance_type="ml.g5.2xlarge",
            instance_count=1,
            copy_count_per_instance=1,
            image_uri="111122223333.dkr.ecr.us-west-2.amazonaws.com/example:0.36.0-lmi24.0.0-cu129",
            environment_variables={},
        ),
        expected_performance=[
            SimpleNamespace(metric="RequestThroughput", stat="avg", value=151.6, unit="Requests/Second"),
            SimpleNamespace(metric="OutputTokenThroughput", stat="avg", value=4851.1, unit="Tokens/Second"),
            SimpleNamespace(metric="RequestLatency", stat="p50", value=425.2, unit="Milliseconds"),
            SimpleNamespace(metric="RequestLatency", stat="p99", value=474.6, unit="Milliseconds"),
        ],
    )
    return [row_a, row_b]


class TestRecommendationsView:
    def _make(self, rows=None):
        rows = rows if rows is not None else _fake_two_recommendations()
        return _RecommendationsView(
            _RecommendationView(row, index=i) for i, row in enumerate(rows)
        )

    def test_behaves_like_list(self):
        view = self._make()
        assert len(view) == 2
        assert isinstance(view[0], _RecommendationView)
        # iteration
        names = [getattr(getattr(r.raw, "model_details", None), "inference_specification_name", None) for r in view]
        assert names == ["high-otps-on-g5-2xlarge", "high-otps-on-g5-2xlarge-1"]

    def test_best_returns_first_row(self):
        view = self._make()
        assert view.best is view[0]

    def test_best_on_empty_raises_indexerror_with_helpful_message(self):
        view = _RecommendationsView()
        with pytest.raises(IndexError, match="No recommendations available"):
            _ = view.best

    def test_repr_renders_comparative_table_with_both_rows(self):
        text = repr(self._make())
        # Header / framing
        assert "Recommendations[2]" in text
        # Both spec names land in the table
        assert "high-otps-on-g5-2xlarge" in text
        assert "high-otps-on-g5-2xlarge-1" in text
        for column in (
            "idx", "spec_name", "instance_type",
            "instances", "copies/inst", "container",
            "req/s", "lat_p99",
        ):
            assert column in text
        # The two rows have different throughput values that should both render
        assert "152.9" in text or "152.94" in text
        assert "151.6" in text or "151.60" in text
        # Container tags should render as the short LMI version, not the full URI
        assert "lmi25.0.0" in text
        assert "lmi24.0.0" in text
        assert "djl-inference" not in text  # full URI must not leak in

    def test_repr_when_empty(self):
        text = repr(_RecommendationsView())
        # Should not crash and should clearly indicate emptiness
        assert "Recommendations[0]" in text or "no rows" in text


class TestShortContainerTag:
    """_short_container_tag should pick the friendly version token from a full image URI."""

    def _short(self, uri):
        from sagemaker.serve.ai_inference_recommender._recommendation_view import _short_container_tag
        return _short_container_tag(uri)

    def test_picks_lmi_token(self):
        assert self._short("111122223333.dkr.ecr.us-west-2.amazonaws.com/example:0.36.0-lmi25.0.0-cu130") == "lmi25.0.0"

    def test_picks_vllm_token(self):
        assert self._short("example/img:vllm0.6.0-cu121") == "vllm0.6.0"

    def test_falls_back_to_full_tag_when_no_known_token(self):
        assert self._short("example/img:1.2.3") == "1.2.3"

    def test_handles_dash_and_empty(self):
        assert self._short("-") == "-"
        assert self._short("") == "-"
