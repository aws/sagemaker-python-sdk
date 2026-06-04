# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for the recommender helpers in _model_builder_methods."""
from __future__ import absolute_import

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.serve.ai_inference_recommender import Workload
from sagemaker.serve.ai_inference_recommender._model_builder_methods import (
    start_benchmark,
    run_recommendation_job,
)


@pytest.fixture
def patch_session():
    with patch(
        "sagemaker.serve.ai_inference_recommender._model_builder_methods.Session"
    ) as Session, patch(
        "sagemaker.serve.ai_inference_recommender._model_builder_methods.get_execution_role",
        return_value="arn:aws:iam::123456789012:role/role",
    ):
        sess = MagicMock()
        sess.default_bucket.return_value = "default-bucket"
        Session.return_value = sess
        yield sess


@pytest.fixture
def patch_resources():
    with patch(
        "sagemaker.serve.ai_inference_recommender._model_builder_methods.AIWorkloadConfig"
    ) as AIWorkloadConfig, patch(
        "sagemaker.serve.ai_inference_recommender._model_builder_methods.AIBenchmarkJob"
    ) as AIBenchmarkJob, patch(
        "sagemaker.serve.ai_inference_recommender._model_builder_methods.AIRecommendationJob"
    ) as AIRecommendationJob:
        AIBenchmarkJob.create.return_value = MagicMock()
        AIRecommendationJob.create.return_value = MagicMock()
        yield AIWorkloadConfig, AIBenchmarkJob, AIRecommendationJob


def _builder(s3_uri: str = "s3://my-models/llama/") -> SimpleNamespace:
    return SimpleNamespace(
        model_path=s3_uri, s3_upload_path=None, s3_model_data_url=None
    )


class TestStartBenchmark:
    def test_creates_workload_config_and_benchmark_job(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, AIBenchmarkJob, _ = patch_resources

        start_benchmark(
            endpoint="my-ep",
            workload=Workload.synthetic(tokenizer="t"),
            name="bench-1",
        )

        AIWorkloadConfig.create.assert_called_once()
        wc_kwargs = AIWorkloadConfig.create.call_args.kwargs
        assert wc_kwargs["ai_workload_config_name"].startswith("sm-wl-")
        assert wc_kwargs["ai_workload_configs"].workload_spec.inline.startswith("{")

        AIBenchmarkJob.create.assert_called_once()
        job_kwargs = AIBenchmarkJob.create.call_args.kwargs
        assert job_kwargs["ai_benchmark_job_name"] == "bench-1"
        assert job_kwargs["benchmark_target"].endpoint.identifier == "my-ep"
        assert job_kwargs["output_config"].s3_output_location.startswith(
            "s3://default-bucket/benchmarks/"
        )
        assert job_kwargs["role_arn"] == "arn:aws:iam::123456789012:role/role"
        assert job_kwargs["ai_workload_config_identifier"] == wc_kwargs["ai_workload_config_name"]

    def test_endpoint_resource_object_accepted(self, patch_session, patch_resources):
        _, AIBenchmarkJob, _ = patch_resources
        from sagemaker.core.resources import Endpoint

        endpoint = Endpoint(endpoint_name="ep-from-resource")
        start_benchmark(
            endpoint=endpoint,
            workload=Workload.synthetic(tokenizer="t"),
        )
        target = AIBenchmarkJob.create.call_args.kwargs["benchmark_target"]
        assert target.endpoint.identifier == "ep-from-resource"

    def test_existing_workload_config_string_passes_through(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, AIBenchmarkJob, _ = patch_resources
        start_benchmark(endpoint="ep", workload="existing-config")
        AIWorkloadConfig.create.assert_not_called()
        assert (
            AIBenchmarkJob.create.call_args.kwargs["ai_workload_config_identifier"]
            == "existing-config"
        )

    def test_dataset_workload_plumbs_dataset_config(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, _, _ = patch_resources
        wl = Workload.from_dataset(
            s3_uri="s3://my-bucket/datasets/traffic/",
            custom_dataset_type="openai-chat",
            tokenizer="t",
        )
        start_benchmark(endpoint="ep", workload=wl)
        wc_kwargs = AIWorkloadConfig.create.call_args.kwargs
        ds_config = wc_kwargs["dataset_config"]
        assert ds_config is not None
        channels = ds_config.input_data_config
        assert len(channels) == 1
        assert channels[0].channel_name == "dataset"
        assert channels[0].data_source.s3_data_source.s3_uri == (
            "s3://my-bucket/datasets/traffic/"
        )

    def test_synthetic_workload_omits_dataset_config(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, _, _ = patch_resources
        start_benchmark(
            endpoint="ep",
            workload=Workload.synthetic(tokenizer="t"),
        )
        wc_kwargs = AIWorkloadConfig.create.call_args.kwargs
        assert wc_kwargs["dataset_config"] is None

    def test_inline_workload_kwargs_construct_synthetic(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, _, _ = patch_resources
        start_benchmark(
            endpoint="ep",
            tokenizer="meta-llama/Llama-3.2-1B",
            concurrency=4,
            request_count=200,
        )
        wc_kwargs = AIWorkloadConfig.create.call_args.kwargs
        spec = wc_kwargs["ai_workload_configs"].workload_spec.inline
        assert "meta-llama/Llama-3.2-1B" in spec
        assert "\"concurrency\": 4" in spec

    def test_rejects_workload_and_inline_kwargs_together(
        self, patch_session, patch_resources
    ):
        with pytest.raises(ValueError, match="either workload= or inline"):
            start_benchmark(
                endpoint="ep",
                workload=Workload.synthetic(tokenizer="t"),
                tokenizer="other",
            )

    def test_rejects_no_workload_provided(self, patch_session, patch_resources):
        with pytest.raises(ValueError, match="requires either"):
            start_benchmark(endpoint="ep")

    def test_inference_components_routed_into_target(
        self, patch_session, patch_resources
    ):
        _, AIBenchmarkJob, _ = patch_resources
        start_benchmark(
            endpoint="ep",
            workload=Workload.synthetic(tokenizer="t"),
            inference_components=["ic-llama", "ic-qwen"],
        )
        target = AIBenchmarkJob.create.call_args.kwargs["benchmark_target"]
        ids = [c.identifier for c in target.endpoint.inference_components]
        assert ids == ["ic-llama", "ic-qwen"]

    def test_explicit_role_overrides_execution_role(
        self, patch_session, patch_resources
    ):
        _, AIBenchmarkJob, _ = patch_resources
        start_benchmark(
            endpoint="ep",
            workload=Workload.synthetic(tokenizer="t"),
            role="arn:aws:iam::1:role/explicit",
        )
        assert (
            AIBenchmarkJob.create.call_args.kwargs["role_arn"]
            == "arn:aws:iam::1:role/explicit"
        )

    def test_explicit_output_path_used_verbatim(self, patch_session, patch_resources):
        _, AIBenchmarkJob, _ = patch_resources
        start_benchmark(
            endpoint="ep",
            workload=Workload.synthetic(tokenizer="t"),
            output_path="s3://my-out/here/",
        )
        assert (
            AIBenchmarkJob.create.call_args.kwargs["output_config"].s3_output_location
            == "s3://my-out/here/"
        )

    def test_wait_blocks(self, patch_session, patch_resources):
        _, AIBenchmarkJob, _ = patch_resources
        job = AIBenchmarkJob.create.return_value
        start_benchmark(
            endpoint="ep",
            workload=Workload.synthetic(tokenizer="t"),
            wait=True,
        )
        job.wait.assert_called_once()


class TestRunRecommendationJob:
    def test_creates_workload_config_and_recommendation_job(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, _, AIRecommendationJob = patch_resources

        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            name="rec-1",
        )

        AIWorkloadConfig.create.assert_called_once()
        job_kwargs = AIRecommendationJob.create.call_args.kwargs
        assert job_kwargs["ai_recommendation_job_name"] == "rec-1"
        assert job_kwargs["model_source"].s3.s3_uri == "s3://my-models/llama/"
        assert job_kwargs["output_config"].s3_output_location.startswith(
            "s3://default-bucket/recommendations/"
        )
        constraints = job_kwargs["performance_target"].constraints
        assert [c.metric for c in constraints] == ["throughput"]
        assert job_kwargs["role_arn"] == "arn:aws:iam::123456789012:role/role"

    def test_raises_when_no_s3_model_path(self, patch_session, patch_resources):
        builder = _builder()
        builder.model_path = "/local/path"
        with pytest.raises(ValueError, match="S3 model_path"):
            run_recommendation_job(
                builder=builder,
                workload=Workload.synthetic(tokenizer="t"),
                performance_target="throughput",
            )

    def test_resolves_s3_model_data_url_for_jumpstart_builds(
        self, patch_session, patch_resources
    ):
        _, _, AIRecommendationJob = patch_resources
        builder = SimpleNamespace(
            model_path="/local/jumpstart-cache",
            s3_upload_path=None,
            s3_model_data_url="s3://jumpstart-cache-prod/model/",
        )
        run_recommendation_job(
            builder=builder,
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
        )
        model_source = AIRecommendationJob.create.call_args.kwargs["model_source"]
        assert model_source.s3.s3_uri == "s3://jumpstart-cache-prod/model/"

    def test_compute_spec_built_from_instance_types(
        self, patch_session, patch_resources
    ):
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            instance_types=["ml.g6.12xlarge", "ml.p4d.24xlarge"],
        )
        cs = AIRecommendationJob.create.call_args.kwargs["compute_spec"]
        assert cs.instance_types == ["ml.g6.12xlarge", "ml.p4d.24xlarge"]

    def test_too_many_instance_types_rejected(self, patch_session, patch_resources):
        with pytest.raises(ValueError, match="At most"):
            run_recommendation_job(
                builder=_builder(),
                workload=Workload.synthetic(tokenizer="t"),
                performance_target="throughput",
                instance_types=["a", "b", "c", "d"],
            )

    def test_capacity_reservation_arns_built(self, patch_session, patch_resources):
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            instance_types=["ml.g6.12xlarge"],
            capacity_reservation_arns=["arn:aws:ec2:..:cr/cr-1"],
        )
        cs = AIRecommendationJob.create.call_args.kwargs["compute_spec"]
        assert (
            cs.capacity_reservation_config.capacity_reservation_preference
            == "capacity-reservations-only"
        )
        assert cs.capacity_reservation_config.ml_reservation_arns == [
            "arn:aws:ec2:..:cr/cr-1"
        ]

    def test_framework_routed_to_inference_specification(
        self, patch_session, patch_resources
    ):
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            framework="VLLM",
        )
        spec = AIRecommendationJob.create.call_args.kwargs["inference_specification"]
        assert spec.framework == "VLLM"

    def test_model_package_group_passed_through(self, patch_session, patch_resources):
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            model_package_group="mpg",
        )
        out = AIRecommendationJob.create.call_args.kwargs["output_config"]
        assert out.model_package_group_identifier == "mpg"

    def test_advanced_optimization_default_true(self, patch_session, patch_resources):
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
        )
        assert AIRecommendationJob.create.call_args.kwargs["optimize_model"] is True

    def test_advanced_optimization_false_preserved(self, patch_session, patch_resources):
        """Verify advanced_optimization=False reaches the underlying create() call."""
        _, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(),
            workload=Workload.synthetic(tokenizer="t"),
            performance_target="throughput",
            advanced_optimization=False,
        )
        assert AIRecommendationJob.create.call_args.kwargs["optimize_model"] is False

    def test_existing_workload_config_string_passes_through(
        self, patch_session, patch_resources
    ):
        AIWorkloadConfig, _, AIRecommendationJob = patch_resources
        run_recommendation_job(
            builder=_builder(), workload="existing-config", performance_target="throughput"
        )
        AIWorkloadConfig.create.assert_not_called()
        assert (
            AIRecommendationJob.create.call_args.kwargs["ai_workload_config_identifier"]
            == "existing-config"
        )


class TestModelBuilderMethodsRegistration:
    def test_methods_present_on_class(self):
        from sagemaker.serve import ModelBuilder

        assert hasattr(ModelBuilder, "optimize")
        assert hasattr(ModelBuilder, "deploy")
        assert hasattr(ModelBuilder, "recommendations")
        assert hasattr(ModelBuilder, "generate_deployment_recommendations")
        assert hasattr(ModelBuilder, "from_recommendation_job")
        # start_benchmark is now a free function in the recommender module,
        # not a method on ModelBuilder.
        assert not hasattr(ModelBuilder, "start_benchmark")
