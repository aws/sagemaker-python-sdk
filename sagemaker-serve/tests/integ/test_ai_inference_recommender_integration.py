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
"""End-to-end integration tests for the AI inference recommender feature."""
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
)
from sagemaker.serve.ai_inference_recommender import (
    Workload,
    start_benchmark,
)
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# Qwen2-1.5B-Instruct on the JumpStart DJL-LMI-22 0.36 container natively
# serves /v1/chat/completions, matching the OpenAI api_standard the
# synthetic Workload sends.
MODEL_ID = "huggingface-llm-qwen2-1-5b-instruct"
INSTANCE_TYPE = "ml.g5.2xlarge"
WORKLOAD_TOKENIZER = "gpt2"
WORKLOAD_REQUEST_COUNT = 10
WORKLOAD_CONCURRENCY = 1
WORKLOAD_INPUT_TOKENS = 32
WORKLOAD_OUTPUT_TOKENS = 32


def _build_synthetic_workload():
    return Workload.synthetic(
        tokenizer=WORKLOAD_TOKENIZER,
        concurrency=WORKLOAD_CONCURRENCY,
        request_count=WORKLOAD_REQUEST_COUNT,
        prompt_input_tokens_mean=WORKLOAD_INPUT_TOKENS,
        output_tokens_mean=WORKLOAD_OUTPUT_TOKENS,
        streaming=True,
    )


def _build_jumpstart_model_builder(role_arn):
    compute = Compute(instance_type=INSTANCE_TYPE)
    jumpstart_config = JumpStartConfig(model_id=MODEL_ID)
    return ModelBuilder.from_jumpstart_config(
        jumpstart_config=jumpstart_config, compute=compute, role_arn=role_arn
    )


@pytest.mark.slow_test
def test_benchmark_workflow_end_to_end():
    """Deploy a JumpStart endpoint, run a benchmark against it, parse the result."""
    logger.info("Starting AI inference recommender benchmark integration test...")

    unique_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    role = get_execution_role(sagemaker_session=Session())
    bench_job_name = f"air-bench-job-{unique_id}"
    workload_config_name = f"air-bench-wl-{unique_id}"
    core_model = None
    core_endpoint = None

    try:
        model_builder = _build_jumpstart_model_builder(role_arn=role)
        core_model = model_builder.build(model_name=f"air-bench-model-{unique_id}")
        logger.info(f"Model created: {core_model.model_name}")

        core_endpoint = model_builder.deploy(
            endpoint_name=f"air-bench-ep-{unique_id}"
        )
        logger.info(f"Endpoint InService: {core_endpoint.endpoint_name}")

        benchmark_job = start_benchmark(
            endpoint=core_endpoint,
            workload=_build_synthetic_workload(),
            role=role,
            name=bench_job_name,
            workload_config_name=workload_config_name,
            wait=True,
        )
        logger.info(
            f"Benchmark job terminal state: {benchmark_job.ai_benchmark_job_status}"
        )
        assert benchmark_job.ai_benchmark_job_status == "Completed", (
            f"Benchmark did not complete successfully: "
            f"{benchmark_job.ai_benchmark_job_status} / "
            f"{getattr(benchmark_job, 'failure_reason', None)}"
        )

        result = benchmark_job.show_result()
        assert result.metrics.all_metrics, "Expected at least one parsed metric"
        assert result.metrics.request_throughput is not None, (
            f"request_throughput missing from parsed BenchmarkResult; "
            f"available metrics: {list(result.metrics.all_metrics)}"
        )
        logger.info(f"Parsed {len(result.metrics.all_metrics)} benchmark metrics")

    finally:
        if core_endpoint and core_model:
            cleanup_resources(core_model, core_endpoint)
        _delete_quietly(
            lambda: AIBenchmarkJob.get(ai_benchmark_job_name=bench_job_name),
            f"AIBenchmarkJob {bench_job_name}",
        )
        _delete_quietly(
            lambda: AIWorkloadConfig.get(ai_workload_config_name=workload_config_name),
            f"AIWorkloadConfig {workload_config_name}",
        )


@pytest.mark.slow_test
def test_recommendation_workflow_end_to_end():
    """Run an AI recommendation via generate_deployment_recommendations and deploy the top recommendation."""
    logger.info("Starting AI inference recommender recommendation integration test...")

    unique_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    role = get_execution_role(sagemaker_session=Session())
    rec_job_name = f"air-rec-job-{unique_id}"
    workload_config_name = f"air-rec-wl-{unique_id}"
    rec_endpoint_config_name = f"air-rec-cfg-{unique_id}"
    rec_model_name = f"air-rec-model-{unique_id}"

    source_model = None
    rec_model = None
    rec_endpoint = None

    try:
        model_builder = _build_jumpstart_model_builder(role_arn=role)
        source_model = model_builder.build(model_name=f"air-rec-source-{unique_id}")

        recommendation_job = model_builder.generate_deployment_recommendations(
            workload=_build_synthetic_workload(),
            performance_target="throughput",
            instance_types=[INSTANCE_TYPE],
            advanced_optimization=False,
            framework="LMI",
            role_arn=role,
            job_name=rec_job_name,
            workload_config_name=workload_config_name,
            wait=True,
        )
        logger.info(
            f"Recommendation job terminal state: "
            f"{recommendation_job.ai_recommendation_job_status}"
        )
        assert recommendation_job.ai_recommendation_job_status == "Completed", (
            f"Recommendation did not complete successfully: "
            f"{recommendation_job.ai_recommendation_job_status} / "
            f"{getattr(recommendation_job, 'failure_reason', None)}"
        )

        rows = model_builder.recommendations
        assert rows, (
            "ModelBuilder.recommendations is empty after generate_deployment_recommendations"
        )
        top = rows.best
        assert top is rows[0], "rows.best should equal rows[0]"
        assert getattr(getattr(top, "model_details", None), "model_package_arn", None), (
            f"Top recommendation has no ModelPackageArn. Raw: {top}"
        )
        # Comparative table __repr__ should include both row indices and column headers.
        rec_table = repr(rows)
        assert f"Recommendations[{len(rows)}]" in rec_table
        assert "spec_name" in rec_table and "req/s" in rec_table
        logger.info("Got %d recommendations; deploying top one", len(rows))

        rec_endpoint = model_builder.deploy(
            endpoint_name=f"air-rec-ep-{unique_id}",
            model_name=rec_model_name,
            endpoint_config_name=rec_endpoint_config_name,
            role=role,
            wait=True,
        )
        logger.info(f"Recommendation endpoint deployed: {rec_endpoint.endpoint_name}")
        assert rec_endpoint.endpoint_status == "InService", (
            f"Endpoint did not reach InService: {rec_endpoint.endpoint_status}"
        )
        rec_model = Model.get(model_name=rec_model_name)

    finally:
        if rec_endpoint and rec_model:
            cleanup_resources(rec_model, rec_endpoint, rec_endpoint_config_name)
        if source_model:
            source_model.delete()
        _delete_quietly(
            lambda: AIRecommendationJob.get(ai_recommendation_job_name=rec_job_name),
            f"AIRecommendationJob {rec_job_name}",
        )
        _delete_quietly(
            lambda: AIWorkloadConfig.get(ai_workload_config_name=workload_config_name),
            f"AIWorkloadConfig {workload_config_name}",
        )


def cleanup_resources(core_model, core_endpoint, endpoint_config_name=None):
    """Delete the model, endpoint, and endpoint config in reverse order of creation.

    ``endpoint_config_name`` defaults to ``core_endpoint.endpoint_name`` for the
    benchmark-test path where ``ModelBuilder.deploy()`` uses a single name for
    both. Must be passed explicitly for the recommendation-test path, where
    ``ModelBuilder.deploy(recommendation_index=N)`` generates a distinct
    endpoint-config name.
    """
    config_name = endpoint_config_name or core_endpoint.endpoint_name
    _delete_quietly(lambda: core_model, f"Model {core_model.model_name}")
    _delete_quietly(lambda: core_endpoint, f"Endpoint {core_endpoint.endpoint_name}")
    _delete_quietly(
        lambda: EndpointConfig.get(endpoint_config_name=config_name),
        f"EndpointConfig {config_name}",
    )


def _delete_quietly(resource_factory, label):
    """Best-effort delete; log and continue on any failure."""
    try:
        resource_factory().delete()
    except Exception as exc:
        logger.warning("Failed to delete %s: %s", label, exc)
