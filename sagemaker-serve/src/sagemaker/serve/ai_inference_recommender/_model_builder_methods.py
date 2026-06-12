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
"""Internal helpers backing the public start_benchmark function and ModelBuilder.generate_deployment_recommendations."""
from __future__ import absolute_import

import time
import uuid
from typing import Any, List, Literal, Optional, Union

PerformanceTarget = Literal["throughput", "ttft-ms", "cost"]
Framework = Literal["LMI", "VLLM"]

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import (
    AIBenchmarkJob,
    AIRecommendationJob,
    AIWorkloadConfig,
    Endpoint,
)
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.shapes.shapes import (
    AIBenchmarkEndpoint,
    AIBenchmarkInferenceComponent,
    AIBenchmarkNetworkConfig,
    AIBenchmarkOutputConfig,
    AIBenchmarkTarget,
    AICapacityReservationConfig,
    AIDatasetConfig,
    AIModelSource,
    AIModelSourceS3,
    AIRecommendationComputeSpec,
    AIRecommendationConstraint,
    AIRecommendationInferenceSpecification,
    AIRecommendationOutputConfig,
    AIRecommendationPerformanceTarget,
    AIWorkloadConfigs,
    AIWorkloadDataSource,
    AIWorkloadInputDataConfig,
    AIWorkloadS3DataSource,
    Tag,
    VpcConfig,
    WorkloadSpec,
)
from sagemaker.serve.ai_inference_recommender._constants import MAX_INSTANCE_TYPES
from sagemaker.serve.ai_inference_recommender.workload import Workload


@_telemetry_emitter(
    feature=Feature.MODEL_CUSTOMIZATION, func_name="ai_inference_recommender.start_benchmark"
)
def start_benchmark(
    endpoint: Union[Endpoint, str],
    workload: Optional[Union[Workload, str]] = None,
    *,
    output_path: Optional[str] = None,
    role: Optional[str] = None,
    inference_components: Optional[List[str]] = None,
    vpc_config: Optional[VpcConfig] = None,
    tags: Optional[List[Tag]] = None,
    name: Optional[str] = None,
    workload_config_name: Optional[str] = None,
    wait: bool = True,
    **workload_kwargs: Any,
) -> AIBenchmarkJob:
    """Start an AI benchmark job against a SageMaker endpoint.

    Args:
        endpoint: An ``Endpoint`` resource, or the name/ARN of an existing
            endpoint to benchmark.
        workload: Optional. A ``Workload`` instance, or the name/ARN of an
            existing ``AIWorkloadConfig``. Omit this and pass workload
            keyword arguments inline (``tokenizer=``, ``concurrency=``,
            etc.) to construct a synthetic workload on the fly.
        output_path: ``s3://`` URI for benchmark output. Defaults to the
            session's default bucket.
        role: IAM execution role ARN. Defaults to the SageMaker execution
            role from the ambient session.
        inference_components: Optional list of inference component names to
            target on the endpoint.
        vpc_config: Optional ``VpcConfig`` for VPC-only endpoints.
        tags: Optional resource tags.
        name: Optional benchmark job name. Auto-generated if omitted.
        workload_config_name: Optional name for the auto-created workload
            config. Auto-generated if omitted.
        wait: If True (default), block until the job reaches a terminal
            state.
        **workload_kwargs: Inline workload parameters. Only used when
            ``workload`` is omitted; forwarded to ``Workload.synthetic``.

    Returns:
        The created :class:`BenchmarkJob`. Once terminal, call
        ``job.show_result()`` to download and parse the metrics.
    """
    if workload is None:
        if not workload_kwargs:
            raise ValueError(
                "start_benchmark requires either a workload= argument or "
                "inline workload keyword arguments (e.g. tokenizer=...)."
            )
        workload = Workload.synthetic(**workload_kwargs)
    elif workload_kwargs:
        raise ValueError(
            "start_benchmark accepts either workload= or inline workload "
            "keyword arguments, not both."
        )

    sagemaker_session = Session()
    role_arn = role or get_execution_role(sagemaker_session=sagemaker_session)
    output_location = output_path or _default_output_path(sagemaker_session, "benchmarks")

    workload_config_id = _ensure_workload_config(workload, workload_config_name, tags=tags)

    endpoint_name = endpoint.endpoint_name if isinstance(endpoint, Endpoint) else endpoint
    components = (
        [AIBenchmarkInferenceComponent(identifier=ic) for ic in inference_components]
        if inference_components
        else None
    )
    target = AIBenchmarkTarget(
        endpoint=AIBenchmarkEndpoint(
            identifier=endpoint_name,
            inference_components=components,
        )
    )
    network_config = (
        AIBenchmarkNetworkConfig(vpc_config=vpc_config) if vpc_config else None
    )

    suffix = uuid.uuid4().hex[:8]
    job_name = name or f"sm-bench-{int(time.time())}-{suffix}"

    job = AIBenchmarkJob.create(
        ai_benchmark_job_name=job_name,
        benchmark_target=target,
        output_config=AIBenchmarkOutputConfig(s3_output_location=output_location),
        ai_workload_config_identifier=workload_config_id,
        role_arn=role_arn,
        network_config=network_config,
        tags=tags,
    )
    # Surface the BenchmarkJob subclass (which adds show_result) on the
    # returned instance.
    from sagemaker.serve.ai_inference_recommender.jobs import BenchmarkJob

    job.__class__ = BenchmarkJob
    if wait:
        job.wait()
    return job


def run_recommendation_job(
    builder,  # ModelBuilder; not annotated to avoid a circular import.
    workload: Union[Workload, str],
    performance_target: PerformanceTarget,
    *,
    output_path: Optional[str] = None,
    role_arn: Optional[str] = None,
    instance_types: Optional[List[str]] = None,
    capacity_reservation_arns: Optional[List[str]] = None,
    advanced_optimization: bool = True,
    framework: Optional[Framework] = None,
    model_package_group: Optional[str] = None,
    tags: Optional[List[Tag]] = None,
    name: Optional[str] = None,
    workload_config_name: Optional[str] = None,
    wait: bool = True,
) -> AIRecommendationJob:
    """Submit an ``AIRecommendationJob`` for the model configured on this builder.

    Backs :meth:`ModelBuilder.generate_deployment_recommendations`. Not intended
    to be called directly.

    Args:
        workload: Either a ``Workload`` (auto-creates a workload config) or
            the name/ARN of an existing ``AIWorkloadConfig``.
        performance_target: One of ``"throughput"``, ``"ttft-ms"``, or
            ``"cost"``.
        output_path: ``s3://`` URI for recommendation output. Defaults to
            the session's default bucket.
        role_arn: IAM execution role ARN. Defaults to the SageMaker execution
            role from the ambient session.
        instance_types: Up to 3 instance types to evaluate.
        capacity_reservation_arns: Optional list of ML reservation ARNs.
        advanced_optimization: If True (default), allow the service to apply
            model optimizations such as speculative decoding and kernel
            tuning.
        framework: Inference framework. ``"LMI"`` or ``"VLLM"``.
        model_package_group: Optional model package group identifier in
            which to register the optimized model.
        tags: Optional resource tags.
        name: Optional recommendation job name. Auto-generated if omitted.
        workload_config_name: Optional name for the auto-created workload
            config. Auto-generated if omitted.
        wait: If True (default), block until the job reaches a terminal state.

    Returns:
        The created ``AIRecommendationJob`` resource.
    """
    sagemaker_session = Session()
    resolved_role_arn = role_arn or get_execution_role(sagemaker_session=sagemaker_session)
    output_location = output_path or _default_output_path(
        sagemaker_session, "recommendations"
    )

    s3_uri = _resolve_model_s3_uri(builder)
    if not s3_uri:
        raise ValueError(
            "ModelBuilder must be configured with an S3 model_path before "
            "calling generate_deployment_recommendations. Call build() first."
        )

    if instance_types and len(instance_types) > MAX_INSTANCE_TYPES:
        raise ValueError(
            f"At most {MAX_INSTANCE_TYPES} instance_types are accepted; "
            f"got {len(instance_types)}."
        )

    workload_config_id = _ensure_workload_config(workload, workload_config_name, tags=tags)

    suffix = uuid.uuid4().hex[:8]
    job_name = name or f"sm-rec-{int(time.time())}-{suffix}"

    compute_spec = None
    if instance_types or capacity_reservation_arns:
        capacity = (
            AICapacityReservationConfig(
                capacity_reservation_preference="capacity-reservations-only",
                ml_reservation_arns=capacity_reservation_arns,
            )
            if capacity_reservation_arns
            else None
        )
        compute_spec = AIRecommendationComputeSpec(
            instance_types=instance_types,
            capacity_reservation_config=capacity,
        )

    inference_spec = (
        AIRecommendationInferenceSpecification(framework=framework) if framework else None
    )

    job = AIRecommendationJob.create(
        ai_recommendation_job_name=job_name,
        model_source=AIModelSource(s3=AIModelSourceS3(s3_uri=s3_uri)),
        output_config=AIRecommendationOutputConfig(
            s3_output_location=output_location,
            model_package_group_identifier=model_package_group,
        ),
        ai_workload_config_identifier=workload_config_id,
        performance_target=AIRecommendationPerformanceTarget(
            constraints=[AIRecommendationConstraint(metric=performance_target)],
        ),
        role_arn=resolved_role_arn,
        inference_specification=inference_spec,
        optimize_model=advanced_optimization,
        compute_spec=compute_spec,
        tags=tags,
    )
    # Surface the RecommendationJob subclass (which adds show_result) on the
    # returned instance.
    from sagemaker.serve.ai_inference_recommender.jobs import RecommendationJob

    job.__class__ = RecommendationJob
    if wait:
        job.wait()
    return job


def _resolve_model_s3_uri(builder) -> Optional[str]:
    for attr in ("model_path", "s3_upload_path", "s3_model_data_url"):
        candidate = getattr(builder, attr, None)
        if isinstance(candidate, str) and candidate.startswith("s3://"):
            return candidate
    return None


def _ensure_workload_config(
    workload: Union[Workload, str],
    name: Optional[str],
    *,
    tags: Optional[List[Tag]] = None,
) -> str:
    if isinstance(workload, str):
        return workload

    config_name = name or f"sm-wl-{int(time.time())}-{uuid.uuid4().hex[:8]}"
    dataset_config = None
    if workload.dataset_channels:
        dataset_config = AIDatasetConfig(
            input_data_config=[
                AIWorkloadInputDataConfig(
                    channel_name=channel.channel_name,
                    data_source=AIWorkloadDataSource(
                        s3_data_source=AIWorkloadS3DataSource(s3_uri=channel.s3_uri),
                    ),
                )
                for channel in workload.dataset_channels
            ],
        )
    AIWorkloadConfig.create(
        ai_workload_config_name=config_name,
        ai_workload_configs=AIWorkloadConfigs(
            workload_spec=WorkloadSpec(inline=workload.to_inline()),
        ),
        dataset_config=dataset_config,
        tags=tags,
    )
    return config_name


def _default_output_path(session: Session, prefix: str) -> str:
    bucket = session.default_bucket()
    return f"s3://{bucket}/{prefix}/"
