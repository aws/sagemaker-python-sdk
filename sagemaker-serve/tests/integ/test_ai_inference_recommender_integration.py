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
    Model,
    ModelPackage,
)
from sagemaker.serve.ai_inference_recommender import (
    Workload,
    start_benchmark,
)
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train.configs import Compute

from cleanup_helpers import cleanup_by_name, delete_quietly

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
    # Names are generated up front so cleanup can run by name even if build()
    # or deploy() raises or is killed before returning a resource handle.
    model_name = f"air-bench-model-{unique_id}"
    endpoint_name = f"air-bench-ep-{unique_id}"

    try:
        model_builder = _build_jumpstart_model_builder(role_arn=role)
        core_model = model_builder.build(model_name=model_name)
        logger.info(f"Model created: {core_model.model_name}")

        core_endpoint = model_builder.deploy(
            endpoint_name=endpoint_name
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
        # Best-effort cleanup by name; runs even if deploy() failed/hung so a
        # Failed or half-created endpoint is still torn down.
        cleanup_by_name(endpoint_name=endpoint_name, model_name=model_name)
        delete_quietly(
            lambda: AIBenchmarkJob.get(ai_benchmark_job_name=bench_job_name).delete(),
            f"AIBenchmarkJob {bench_job_name}",
        )
        delete_quietly(
            lambda: AIWorkloadConfig.get(
                ai_workload_config_name=workload_config_name
            ).delete(),
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
    rec_endpoint_name = f"air-rec-ep-{unique_id}"
    rec_model_name = f"air-rec-model-{unique_id}"
    source_model_name = f"air-rec-source-{unique_id}"

    rec_model_package_arn = None

    try:
        model_builder = _build_jumpstart_model_builder(role_arn=role)
        model_builder.build(model_name=source_model_name)

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
        rec_model_package_arn = getattr(
            getattr(top, "model_details", None), "model_package_arn", None
        )
        assert rec_model_package_arn, (
            f"Top recommendation has no ModelPackageArn. Raw: {top}"
        )
        # The comparative table lives in str(); it includes row count + headers.
        rec_table = str(rows)
        assert f"Recommendations[{len(rows)}]" in rec_table
        assert "spec_name" in rec_table and "req/s" in rec_table
        logger.info("Got %d recommendations; deploying top one", len(rows))

        # The recommendation's ModelPackage is unapproved; opt in to approving
        # it as part of deploy for this end-to-end test.
        rec_endpoint = model_builder.deploy(
            endpoint_name=rec_endpoint_name,
            model_name=rec_model_name,
            endpoint_config_name=rec_endpoint_config_name,
            role=role,
            auto_approve=True,
            wait=True,
        )
        logger.info(f"Recommendation endpoint deployed: {rec_endpoint.endpoint_name}")
        assert rec_endpoint.endpoint_status == "InService", (
            f"Endpoint did not reach InService: {rec_endpoint.endpoint_status}"
        )

    finally:
        # Best-effort cleanup by name; runs even if deploy() failed/hung so a
        # Failed or half-created endpoint is still torn down. The
        # recommendation path uses a distinct endpoint-config name, so it is
        # passed explicitly rather than defaulting to the endpoint name.
        cleanup_by_name(
            endpoint_name=rec_endpoint_name,
            endpoint_config_name=rec_endpoint_config_name,
            model_name=rec_model_name,
        )
        delete_quietly(
            lambda: Model.get(model_name=source_model_name).delete(),
            f"Model {source_model_name}",
        )
        delete_quietly(
            lambda: AIRecommendationJob.get(
                ai_recommendation_job_name=rec_job_name
            ).delete(),
            f"AIRecommendationJob {rec_job_name}",
        )
        delete_quietly(
            lambda: AIWorkloadConfig.get(
                ai_workload_config_name=workload_config_name
            ).delete(),
            f"AIWorkloadConfig {workload_config_name}",
        )
        # The recommendation job publishes (and this test approves) a
        # ModelPackage; delete it so repeated runs don't accumulate packages.
        if rec_model_package_arn:
            delete_quietly(
                lambda: ModelPackage.get(
                    model_package_name=rec_model_package_arn
                ).delete(),
                f"ModelPackage {rec_model_package_arn}",
            )
