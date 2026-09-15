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
"""End-to-end integration tests for the inference-recommender enhancements:
``list_benchmarks`` / ``list_recommendations`` filtering, ``deploy`` from a
recommendation row (``mb.recommendations.best``), and ``compare_benchmarks``.
"""
from __future__ import absolute_import

import logging
import time
import uuid

import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.resources import (
    AIBenchmarkJob,
    AIRecommendationJob,
    AIWorkloadConfig,
    EndpointConfig,
    Model,
    ModelPackage,
)
from sagemaker.serve.ai_inference_recommender import (
    Workload,
    compare_benchmarks,
    list_benchmarks,
    list_recommendations,
    start_benchmark,
)
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

MODEL_ID = "huggingface-reasoning-qwen3-06b"
INSTANCE_TYPE = "ml.g6.2xlarge"
WORKLOAD_TOKENIZER = "gpt2"


def _synthetic_workload():
    return Workload.synthetic(
        tokenizer=WORKLOAD_TOKENIZER,
        concurrency=1,
        request_count=10,
        prompt_input_tokens_mean=32,
        output_tokens_mean=32,
        streaming=True,
    )


def test_list_benchmarks_and_recommendations_plumbing():
    """The listing helpers execute against the live API (list + describe +
    client-side filter) and a filter with no match returns an empty list.

    No GPU: this exercises the ``get_all`` / describe / filter plumbing without
    creating any resource, so it runs fast and independently of the e2e below.

    This test collects on ordinary PR checks (it has no ``gpu_intensive`` mark),
    so every call caps ``max_scan`` tightly: a client-side filter that matches
    rarely would otherwise Describe up to ``max_scan`` jobs, and on a busy shared
    account that is enough to throttle the suite. A small cap keeps the plumbing
    check cheap regardless of how many jobs the account holds.
    """
    logger.info("Listing plumbing: list_benchmarks / list_recommendations ...")

    benches = list_benchmarks(max_results=5, max_scan=5)
    assert isinstance(benches, list)
    logger.info("list_benchmarks() returned %d job(s)", len(benches))

    recs = list_recommendations(max_results=5, max_scan=5)
    assert isinstance(recs, list)
    logger.info("list_recommendations() returned %d job(s)", len(recs))

    # A filter that cannot match returns an empty list (not an error). Cap the
    # scan hard: the whole point is that no candidate matches, so without a cap
    # this is exactly the full-account Describe sweep we must not trigger here.
    no_match_ep = f"no-such-endpoint-{uuid.uuid4().hex}"
    assert list_benchmarks(endpoint=no_match_ep, max_results=5, max_scan=10) == []

    no_match_model = f"s3://no-such-bucket-{uuid.uuid4().hex}/model/"
    assert list_recommendations(model=no_match_model, max_results=5, max_scan=10) == []
    logger.info("Non-matching filters correctly returned empty lists.")


@pytest.mark.slow_test
@pytest.mark.gpu_intensive
def test_recommendation_deploy_best_and_compare_e2e():
    """Full flow across all three enhancements, sharing one rec job + endpoint:

    1. run a recommendation job,
    2. ``list_recommendations(model=...)`` finds it (client-side filter),
    3. ``deploy(recommendation=mb.recommendations.best)`` reaches InService,
    4. run two benchmarks against that endpoint,
    5. ``list_benchmarks(endpoint=...)`` finds them,
    6. ``compare_benchmarks`` renders a two-run comparison.
    """
    unique_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    role = get_execution_role(sagemaker_session=Session())
    rec_job_name = f"air-enh-rec-{unique_id}"
    rec_wl_name = f"air-enh-rec-wl-{unique_id}"
    src_model_name = f"air-enh-src-{unique_id}"
    dep_model_name = f"air-enh-model-{unique_id}"
    dep_config_name = f"air-enh-cfg-{unique_id}"
    endpoint_name = f"air-enh-ep-{unique_id}"
    bench_a_name = f"air-enh-bench-a-{unique_id}"
    bench_b_name = f"air-enh-bench-b-{unique_id}"
    bench_a_wl = f"air-enh-bench-a-wl-{unique_id}"
    bench_b_wl = f"air-enh-bench-b-wl-{unique_id}"

    source_model = None
    endpoint = None
    rec_model_package_arn = None
    model_uri = None

    try:
        mb = ModelBuilder.from_jumpstart_config(
            jumpstart_config=JumpStartConfig(model_id=MODEL_ID),
            compute=Compute(instance_type=INSTANCE_TYPE),
            role_arn=role,
        )
        source_model = mb.build(model_name=src_model_name)
        model_uri = _model_source_uri(src_model_name)

        # (1) recommendation job
        rec_job = mb.generate_deployment_recommendations(
            workload=_synthetic_workload(),
            performance_target="throughput",
            instance_types=[INSTANCE_TYPE],
            advanced_optimization=False,
            framework="LMI",
            role_arn=role,
            job_name=rec_job_name,
            workload_config_name=rec_wl_name,
            wait=True,
        )
        assert rec_job.ai_recommendation_job_status == "Completed", (
            f"Recommendation job did not complete: "
            f"{rec_job.ai_recommendation_job_status} / "
            f"{getattr(rec_job, 'failure_reason', None)}"
        )
        rows = mb.recommendations
        assert rows, "mb.recommendations is empty after a completed job"
        rec_model_package_arn = getattr(
            getattr(rows.best, "model_details", None), "model_package_arn", None
        )
        logger.info("Recommendation complete; best spec=%s", rows.best.recommendation_spec_name)

        # (2) list_recommendations model filter returns rows that all match the
        # requested model. We do NOT assert this specific just-created job is in
        # the result: the model URI is a shared JumpStart cache path many jobs
        # reuse, and freshly-created jobs are subject to List eventual
        # consistency — so pinning on "find my exact job" here is racy. Instead
        # assert the filter's contract: every returned job's model_source matches
        # the requested URI. (Deterministic filter coverage lives in the unit
        # tests and the no-GPU plumbing test.)
        if model_uri:
            found = list_recommendations(model=model_uri, max_results=25)
            # We just created a job on this model_uri, so the filter must return
            # at least one row. Assert non-empty first, otherwise the per-job
            # contract check below passes vacuously and a filter regressing to
            # always-False would go unnoticed.
            assert found, (
                "list_recommendations(model=...) returned no jobs, but this run "
                f"created one on {model_uri}"
            )
            for job in found:
                src = getattr(getattr(job, "model_source", None), "s3", None)
                uri = getattr(src, "s3_uri", None) if src else None
                assert uri is not None and uri.rstrip("/") == model_uri.rstrip("/"), (
                    f"list_recommendations(model=...) returned a non-matching job: "
                    f"{job.ai_recommendation_job_name} has model_source {uri}"
                )
            logger.info("list_recommendations(model=...) returned %d matching job(s).", len(found))

        # (3) deploy the best recommendation row directly (no magic index)
        endpoint = mb.deploy(
            endpoint_name=endpoint_name,
            recommendation=rows.best,
            model_name=dep_model_name,
            endpoint_config_name=dep_config_name,
            role=role,
            auto_approve=True,
            wait=True,
        )
        assert (
            endpoint.endpoint_status == "InService"
        ), f"Endpoint did not reach InService: {endpoint.endpoint_status}"
        logger.info("Deployed mb.recommendations.best -> %s InService", endpoint_name)

        # (4) two benchmarks against the endpoint
        bench_a = start_benchmark(
            endpoint=endpoint_name,
            workload=_synthetic_workload(),
            role=role,
            name=bench_a_name,
            workload_config_name=bench_a_wl,
            wait=True,
        )
        bench_b = start_benchmark(
            endpoint=endpoint_name,
            workload=_synthetic_workload(),
            role=role,
            name=bench_b_name,
            workload_config_name=bench_b_wl,
            wait=True,
        )
        for job in (bench_a, bench_b):
            assert job.ai_benchmark_job_status == "Completed", (
                f"Benchmark {job.get_name()} did not complete: " f"{job.ai_benchmark_job_status}"
            )

        # (5) list_benchmarks(endpoint=...) filters by this run's (unique)
        # endpoint. Hard-assert the filter contract: every returned job targets
        # this endpoint. Finding both specific jobs is best-effort (List is
        # eventually consistent for freshly-created jobs), logged not gated —
        # which also means `listed` may legitimately be empty here, so the
        # per-job loop below is allowed to be vacuous. `both_present` is the
        # signal that the filter actually returned our jobs when List has caught
        # up; it is logged rather than asserted precisely because of that race.
        listed = list_benchmarks(endpoint=endpoint_name, max_results=25)
        for job in listed:
            target = getattr(job, "benchmark_target", None)
            ep = getattr(target, "endpoint", None) if target else None
            identifier = getattr(ep, "identifier", None) if ep else None
            assert identifier and (
                identifier == endpoint_name or identifier.endswith(f"/{endpoint_name}")
            ), (
                f"list_benchmarks(endpoint=...) returned a non-matching job: "
                f"{job.ai_benchmark_job_name} targets {identifier}"
            )
        listed_names = [j.ai_benchmark_job_name for j in listed]
        both_present = bench_a_name in listed_names and bench_b_name in listed_names
        logger.info(
            "list_benchmarks(endpoint=...) returned %d job(s); both this run's "
            "benchmarks present: %s",
            len(listed),
            both_present,
        )

        # (6) compare the two runs
        result_a = bench_a.show_result()
        result_b = bench_b.show_result()
        comparison = compare_benchmarks(result_a, result_b, names=["run_a", "run_b"])
        rendered = str(comparison)
        assert "BenchmarkComparison" in rendered
        assert "run_a" in rendered and "run_b" in rendered
        logger.info("compare_benchmarks rendered:\n%s", rendered)

    finally:
        _delete_quietly(lambda: Model.get(model_name=dep_model_name), f"Model {dep_model_name}")
        if endpoint is not None:
            _delete_quietly(lambda: endpoint, f"Endpoint {endpoint_name}")
        _delete_quietly(
            lambda: EndpointConfig.get(endpoint_config_name=dep_config_name),
            f"EndpointConfig {dep_config_name}",
        )
        if source_model is not None:
            _delete_quietly(lambda: source_model, f"Model {src_model_name}")
        for job_name in (bench_a_name, bench_b_name):
            _delete_quietly(
                lambda n=job_name: AIBenchmarkJob.get(ai_benchmark_job_name=n),
                f"AIBenchmarkJob {job_name}",
            )
        _delete_quietly(
            lambda: AIRecommendationJob.get(ai_recommendation_job_name=rec_job_name),
            f"AIRecommendationJob {rec_job_name}",
        )
        for wl in (rec_wl_name, bench_a_wl, bench_b_wl):
            _delete_quietly(
                lambda n=wl: AIWorkloadConfig.get(ai_workload_config_name=n),
                f"AIWorkloadConfig {wl}",
            )
        if rec_model_package_arn:
            _delete_quietly(
                lambda: ModelPackage.get(model_package_name=rec_model_package_arn),
                f"ModelPackage {rec_model_package_arn}",
            )


def _model_source_uri(model_name):
    """Resolve the S3 artifact URI of a built model, to filter recs by model."""
    container = getattr(Model.get(model_name=model_name), "primary_container", None)
    if container is None:
        return None
    mds = getattr(container, "model_data_source", None)
    if mds is not None:
        s3 = getattr(mds, "s3_data_source", None)
        if s3 is not None and getattr(s3, "s3_uri", None):
            return s3.s3_uri
    return getattr(container, "model_data_url", None)


def _delete_quietly(resource_factory, label):
    """Best-effort delete; log and continue on any failure."""
    try:
        resource_factory().delete()
        logger.info("Deleted %s", label)
    except Exception as exc:
        logger.warning("Failed to delete %s: %s", label, exc)
