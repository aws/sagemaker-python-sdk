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
"""Unit tests for list_benchmarks / list_recommendations client-side filtering."""
from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.core.resources import AIBenchmarkJob, AIRecommendationJob
from sagemaker.core.shapes.shapes import (
    AIBenchmarkEndpoint,
    AIBenchmarkTarget,
    AIModelSource,
    AIModelSourceS3,
    AIRecommendation,
    AIRecommendationModelDetails,
    AIRecommendationOutputResult,
)
from sagemaker.serve.ai_inference_recommender import (
    list_benchmarks,
    list_recommendations,
)
from sagemaker.serve.ai_inference_recommender.jobs import BenchmarkJob, RecommendationJob


# Stand-ins are base AIBenchmarkJob / AIRecommendationJob instances, matching
# what get_all yields, so the retype to the show_result subclass can be asserted
# rather than being a silent no-op. Nested fields are set directly.


@pytest.fixture(autouse=True)
def _no_op_refresh():
    """Stub refresh() as a no-op MagicMock so hydration does not hit AWS and its
    call count stays assertable; the stand-ins already carry their fields."""
    with patch.object(AIBenchmarkJob, "refresh", MagicMock()), patch.object(
        AIRecommendationJob, "refresh", MagicMock()
    ):
        yield


def _bench(name, endpoint_identifier=None):
    job = AIBenchmarkJob(ai_benchmark_job_name=name)
    job.benchmark_target = (
        AIBenchmarkTarget(endpoint=AIBenchmarkEndpoint(identifier=endpoint_identifier))
        if endpoint_identifier is not None
        else None
    )
    return job


def _rec(name, s3_uri=None, group=None, row_arns=()):
    job = AIRecommendationJob(ai_recommendation_job_name=name)
    job.model_source = AIModelSource(s3=AIModelSourceS3(s3_uri=s3_uri)) if s3_uri else None
    job.output_config = (
        AIRecommendationOutputResult(
            s3_output_location="s3://bucket/out/",
            model_package_group_identifier=group,
        )
        if group is not None
        else None
    )
    job.recommendations = [
        AIRecommendation(model_details=AIRecommendationModelDetails(model_package_arn=arn))
        for arn in row_arns
    ]
    return job


class TestListBenchmarks:
    def test_no_filter_returns_all_native(self):
        jobs = [_bench("b1"), _bench("b2")]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks()
        assert [j.ai_benchmark_job_name for j in out] == ["b1", "b2"]

    def test_endpoint_filter_matches_by_name(self):
        jobs = [
            _bench("b1", endpoint_identifier="ep-A"),
            _bench("b2", endpoint_identifier="ep-B"),
            _bench("b3", endpoint_identifier=None),
        ]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks(endpoint="ep-A")
        assert [j.ai_benchmark_job_name for j in out] == ["b1"]

    def test_endpoint_filter_matches_arn_suffix(self):
        arn = "arn:aws:sagemaker:us-west-2:1:endpoint/ep-A"
        jobs = [_bench("b1", endpoint_identifier=arn)]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks(endpoint="ep-A")
        assert [j.ai_benchmark_job_name for j in out] == ["b1"]

    def test_max_results_caps_output(self):
        jobs = [_bench(f"b{i}") for i in range(10)]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks(max_results=3)
        assert len(out) == 3

    def test_native_filters_forwarded(self):
        with patch.object(BenchmarkJob, "get_all", return_value=iter([])) as get_all:
            list_benchmarks(status="Completed", name_contains="qwen")
        kwargs = get_all.call_args.kwargs
        assert kwargs["status_equals"] == "Completed"
        assert kwargs["name_contains"] == "qwen"


class TestListRecommendations:
    def test_model_filter_matches_source_uri(self):
        jobs = [
            _rec("r1", s3_uri="s3://bucket/model-a/"),
            _rec("r2", s3_uri="s3://bucket/model-b/"),
        ]
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations(model="s3://bucket/model-a/")
        assert [j.ai_recommendation_job_name for j in out] == ["r1"]

    def test_model_filter_ignores_trailing_slash(self):
        jobs = [_rec("r1", s3_uri="s3://bucket/model-a/")]
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations(model="s3://bucket/model-a")
        assert [j.ai_recommendation_job_name for j in out] == ["r1"]

    def test_model_package_matches_output_group(self):
        jobs = [
            _rec("r1", group="my-group"),
            _rec("r2", group="other-group"),
        ]
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations(model_package="my-group")
        assert [j.ai_recommendation_job_name for j in out] == ["r1"]

    def test_model_package_matches_recommendation_row_arn(self):
        arn = "arn:aws:sagemaker:us-west-2:1:model-package/g/1"
        jobs = [_rec("r1", row_arns=(arn,)), _rec("r2", row_arns=())]
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations(model_package=arn)
        assert [j.ai_recommendation_job_name for j in out] == ["r1"]

    def test_model_and_model_package_mutually_exclusive(self):
        with pytest.raises(ValueError, match="only one of"):
            list_recommendations(model="s3://x/", model_package="arn:y")

    def test_no_filter_returns_all(self):
        jobs = [_rec("r1"), _rec("r2")]
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations()
        assert [j.ai_recommendation_job_name for j in out] == ["r1", "r2"]


class TestRetypesToSubclass:
    """list_* re-types base get_all output to the show_result subclass."""

    def test_benchmarks_retyped_to_benchmark_job(self):
        jobs = [_bench("b1")]
        assert type(jobs[0]) is AIBenchmarkJob  # base going in
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks()
        assert isinstance(out[0], BenchmarkJob)  # subclass coming out
        assert hasattr(out[0], "show_result")

    def test_recommendations_retyped_to_recommendation_job(self):
        jobs = [_rec("r1")]
        assert type(jobs[0]) is AIRecommendationJob
        with patch.object(RecommendationJob, "get_all", return_value=iter(jobs)):
            out = list_recommendations()
        assert isinstance(out[0], RecommendationJob)
        assert hasattr(out[0], "show_result")


class TestScanBound:
    """max_scan bounds the Describe fan-out, independent of max_results."""

    def test_max_scan_caps_candidates_examined_when_filter_rarely_matches(self):
        # 500 candidates, only the last matches; max_scan stops the scan early
        # so the rare-match filter cannot fan out across the whole stream.
        jobs = [_bench(f"b{i}") for i in range(500)]
        jobs[-1].benchmark_target = AIBenchmarkTarget(
            endpoint=AIBenchmarkEndpoint(identifier="ep-match")
        )
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks(endpoint="ep-match", max_scan=10)
        # Scan stopped at 10, well before the lone match at index 499.
        assert out == []

    def test_scan_reaching_the_match_within_budget_returns_it(self):
        jobs = [_bench(f"b{i}") for i in range(20)]
        jobs[5].benchmark_target = AIBenchmarkTarget(
            endpoint=AIBenchmarkEndpoint(identifier="ep-match")
        )
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            out = list_benchmarks(endpoint="ep-match", max_scan=10)
        assert [j.ai_benchmark_job_name for j in out] == ["b5"]

    def test_truncated_scan_logs_warning(self, caplog):
        jobs = [_bench(f"b{i}") for i in range(50)]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            with caplog.at_level("WARNING"):
                list_benchmarks(endpoint="no-match", max_scan=5)
        assert any("max_scan" in r.message for r in caplog.records)


class TestBadCandidateResilience:
    """A candidate whose Describe fails is skipped, not fatal to the listing."""

    def _iterator_that_raises_on(self, jobs, bad_index, exc):
        """Stand-in for sagemaker-core's ResourceIterator: raises from __next__
        at the bad candidate but advances its index first, so a later next()
        resumes at the following item (a plain generator dies once it raises)."""

        class _ResumableIterator:
            def __init__(self):
                self._i = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._i >= len(jobs):
                    raise StopIteration
                i = self._i
                self._i += 1  # advance BEFORE the (simulated) refresh, like the real one
                if i == bad_index:
                    raise exc
                return jobs[i]

        return _ResumableIterator()

    def test_one_undescribable_job_does_not_abort_the_listing(self):
        jobs = [
            _bench("b1", endpoint_identifier="ep-A"),
            _bench("b2", endpoint_identifier="ep-A"),  # never reached: bad one is #1
            _bench("b3", endpoint_identifier="ep-A"),
        ]
        it = self._iterator_that_raises_on(jobs, 1, RuntimeError("AccessDenied"))
        with patch.object(BenchmarkJob, "get_all", return_value=it):
            out = list_benchmarks(endpoint="ep-A")
        # b1 returned; the raise at #1 is skipped; b3 still examined and returned.
        assert [j.ai_benchmark_job_name for j in out] == ["b1", "b3"]

    def test_skipped_jobs_logged_at_warning(self, caplog):
        jobs = [_bench("b1", endpoint_identifier="ep-A"), _bench("b2")]
        it = self._iterator_that_raises_on(jobs, 0, RuntimeError("Throttled"))
        with patch.object(BenchmarkJob, "get_all", return_value=it):
            with caplog.at_level("WARNING"):
                list_benchmarks(endpoint="ep-A")
        msgs = " ".join(r.message for r in caplog.records)
        assert "could not be described" in msgs
        assert "skipped" in msgs

    def _iterator_all_raise(self, n, exc):
        """Stand-in iterator whose every __next__ raises (a fully-denied role)."""

        class _AllRaise:
            def __init__(self):
                self._i = 0

            def __iter__(self):
                return self

            def __next__(self):
                if self._i >= n:
                    raise StopIteration
                self._i += 1
                raise exc

        return _AllRaise()

    def test_per_candidate_skip_warnings_are_capped(self, caplog):
        # A role that cannot Describe anything must not emit one warning per
        # candidate (up to max_scan); only the first few, plus the summary.
        it = self._iterator_all_raise(50, RuntimeError("AccessDenied"))
        with patch.object(BenchmarkJob, "get_all", return_value=it):
            with caplog.at_level("WARNING"):
                out = list_benchmarks(endpoint="ep-A", max_scan=50)
        assert out == []
        # Per-candidate lines start with "Skipping a"; the summary is separate.
        per_candidate = [r for r in caplog.records if r.message.startswith("Skipping a")]
        assert len(per_candidate) <= 5
        assert any("50 job(s) were skipped" in r.message for r in caplog.records)

    def test_hitting_max_results_is_logged(self, caplog):
        jobs = [_bench(f"b{i}", endpoint_identifier="ep-A") for i in range(5)]
        with patch.object(BenchmarkJob, "get_all", return_value=iter(jobs)):
            with caplog.at_level("INFO"):
                out = list_benchmarks(endpoint="ep-A", max_results=2)
        assert len(out) == 2
        assert any("max_results" in r.message for r in caplog.records)


class TestSessionHandling:
    """Accept a boto3 Session or a sagemaker Session (unwrapping the latter)."""

    def test_boto3_session_passed_through(self):
        import boto3

        session = boto3.session.Session(region_name="us-west-2")
        with patch.object(BenchmarkJob, "get_all", return_value=iter([])) as get_all:
            list_benchmarks(sagemaker_session=session)
        assert get_all.call_args.kwargs["session"] is session

    def test_sagemaker_session_unwrapped_to_boto_session(self):
        import boto3

        boto_session = boto3.session.Session(region_name="us-west-2")
        # A sagemaker-Session-like object: exposes .boto_session, and a real
        # sagemaker_config dict so the telemetry decorator's config validation
        # (which runs before the call) does not choke on a bare MagicMock.
        sm_session = MagicMock()
        sm_session.boto_session = boto_session
        sm_session.sagemaker_config = {}
        with patch.object(BenchmarkJob, "get_all", return_value=iter([])) as get_all:
            list_benchmarks(sagemaker_session=sm_session)
        # get_all receives the unwrapped boto3 session (what pydantic validate_call wants).
        assert get_all.call_args.kwargs["session"] is boto_session

    def test_none_session_stays_none(self):
        with patch.object(BenchmarkJob, "get_all", return_value=iter([])) as get_all:
            list_benchmarks()
        assert get_all.call_args.kwargs["session"] is None
