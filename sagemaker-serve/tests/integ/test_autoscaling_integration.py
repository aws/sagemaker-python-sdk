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
"""Integration tests for sagemaker.serve.autoscaling.

Each test creates its own CPU endpoint from a small public JumpStart model (MobileNet v2),
exercises autoscaling on it, then deletes everything it created. The execution role is
``SAGEMAKER_INTEG_ROLE_ARN`` when set, else the caller's role.
"""

from __future__ import absolute_import

import json
import logging
import os
import tarfile
import tempfile
import uuid

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Model
from sagemaker.core.shapes import (
    AsyncInferenceConfig,
    AsyncInferenceOutputConfig,
    ContainerDefinition,
    InferenceComponentComputeResourceRequirements,
    InferenceComponentRuntimeConfig,
    InferenceComponentSpecification,
    ModelDataSource,
    ProductionVariant,
    ProductionVariantManagedInstanceScaling,
    S3ModelDataSource,
)
from sagemaker.serve.autoscaling import (
    Alarm,
    CustomMetric,
    ScheduledAction,
    StepAdjustment,
    StepScalingPolicy,
    TargetTrackingPolicy,
    describe_autoscaling,
    disable_autoscaling,
    enable_autoscaling,
    resume_autoscaling,
    suspend_autoscaling,
)

logger = logging.getLogger(__name__)

HUB_CONTENT_NAME = "pytorch-ic-mobilenet-v2"
INSTANCE_TYPE = "ml.m5.large"
TRACKING = "sagemaker-sdk-target-tracking"
FROM_ZERO = "sagemaker-sdk-step-scaling-scale-out-from-zero"


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return os.environ.get("SAGEMAKER_INTEG_ROLE_ARN") or get_execution_role(
        sagemaker_session=sagemaker_session
    )


@pytest.fixture(scope="module")
def container(sagemaker_session):
    """The public hub's hosting image, artifacts and environment for the model."""
    document = json.loads(
        sagemaker_session.sagemaker_client.describe_hub_content(
            HubName="SageMakerPublicHub", HubContentType="Model", HubContentName=HUB_CONTENT_NAME
        )["HubContentDocument"]
    )
    environment = {
        variable["Name"]: str(variable["Default"])
        for variable in document.get("InferenceEnvironmentVariables", [])
        if variable.get("Default") is not None
    }
    return ContainerDefinition(
        image=document["HostingEcrUri"],
        model_data_source=ModelDataSource(
            s3_data_source=S3ModelDataSource(
                s3_uri=document["HostingArtifactUri"],
                s3_data_type="S3Prefix",
                compression_type="None",
            )
        ),
        environment=environment,
    )


class _Resources:
    """Creates a test endpoint's resources and deletes them in reverse order."""

    def __init__(self, sagemaker_session, role, container, prefix):
        self.session = sagemaker_session
        self.boto = sagemaker_session.boto_session
        self.role = role
        self.container = container
        self.name = f"{prefix}-{uuid.uuid4().hex[:8]}"
        self.created = []

    def endpoint(self, variant, **config_kwargs):
        self.created.append(
            Model.create(
                model_name=self.name,
                primary_container=self.container,
                execution_role_arn=self.role,
                session=self.boto,
            )
        )
        self.created.append(
            EndpointConfig.create(
                endpoint_config_name=self.name,
                production_variants=[variant],
                session=self.boto,
                **config_kwargs,
            )
        )
        endpoint = Endpoint.create(
            endpoint_name=self.name, endpoint_config_name=self.name, session=self.boto
        )
        self.created.append(endpoint)
        endpoint.wait_for_status("InService")
        return endpoint

    def component(self):
        component = InferenceComponent.create(
            inference_component_name=f"{self.name}-ic",
            endpoint_name=self.name,
            variant_name="AllTraffic",
            specification=InferenceComponentSpecification(
                model_name=self.name,
                compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                    number_of_cpu_cores_required=1, min_memory_required_in_mb=1024
                ),
            ),
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
            session=self.boto,
        )
        self.created.append(component)
        component.wait_for_status("InService")
        return component

    def cleanup(self, **scaled):
        try:
            disable_autoscaling(sagemaker_session=self.session, wait=False, **scaled)
        except Exception as error:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Could not turn off autoscaling on %s: %s", scaled, error)
        for resource in reversed(self.created):
            try:
                resource.delete()
                if isinstance(resource, (InferenceComponent, Endpoint)):
                    resource.wait_for_delete()
            except Exception as error:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Could not delete %s: %s", resource, error)


def _packed_artifact(sagemaker_session, container, key):
    """Pack the hub's uncompressed artifact into model.tar.gz, which a serverless endpoint needs."""
    s3 = sagemaker_session.boto_session.client("s3")
    source = container.model_data_source.s3_data_source.s3_uri
    bucket, prefix = source[len("s3://") :].split("/", 1)
    with tempfile.TemporaryDirectory() as tmp:
        archive = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(archive, "w:gz") as tar:
            for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
                for item in page.get("Contents", []):
                    relative = item["Key"][len(prefix) :]
                    if not relative:
                        continue
                    local = os.path.join(tmp, "files", relative)
                    os.makedirs(os.path.dirname(local), exist_ok=True)
                    s3.download_file(bucket, item["Key"], local)
                    tar.add(local, arcname=relative)
        target_bucket = sagemaker_session.default_bucket()
        s3.upload_file(archive, target_bucket, key)
    return f"s3://{target_bucket}/{key}"


def _alarm_count(sagemaker_session, name):
    cloudwatch = boto3.client("cloudwatch", region_name=sagemaker_session.boto_region_name)
    return len(cloudwatch.describe_alarms(AlarmNames=[name])["MetricAlarms"])


def _policy_names(state):
    return {policy["PolicyName"] for policy in state["policies"]}


def _realtime_variant(name, **kwargs):
    return ProductionVariant(
        variant_name="AllTraffic",
        model_name=name,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=1,
        **kwargs,
    )


@pytest.mark.slow_test
def test_variant_policies_schedules_and_suspension(sagemaker_session, role, container):
    resources = _Resources(sagemaker_session, role, container, "autoscale-variant")
    name = resources.name
    try:
        resources.endpoint(_realtime_variant(name))
        cpu = CustomMetric("CPUUtilization", namespace="/aws/sagemaker/Endpoints")
        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=1,
            max_capacity=3,
            policies=[
                TargetTrackingPolicy(target_value=5),
                TargetTrackingPolicy(target_value=70, metric=cpu, name="cpu"),
                StepScalingPolicy(
                    steps=[StepAdjustment(scaling_adjustment=1, lower_bound=0)],
                    alarm=Alarm(metric=cpu, threshold=90, evaluation_periods=2),
                    name="cpu-burst",
                ),
            ],
            scheduled_actions=[
                ScheduledAction(
                    name="night",
                    schedule="cron(0 2 * * ? *)",
                    min_capacity=1,
                    max_capacity=2,
                    timezone="America/Los_Angeles",
                )
            ],
            sagemaker_session=sagemaker_session,
        )
        assert state["resource_id"] == f"endpoint/{name}/variant/AllTraffic"
        assert (
            state["scalable_target"]["MinCapacity"],
            state["scalable_target"]["MaxCapacity"],
        ) == (
            1,
            3,
        )
        assert _policy_names(state) == {
            TRACKING,
            f"{TRACKING}-cpu",
            "sagemaker-sdk-step-scaling-cpu-burst",
        }
        by_name = {policy["PolicyName"]: policy for policy in state["policies"]}
        custom = by_name[f"{TRACKING}-cpu"]["TargetTrackingScalingPolicyConfiguration"]
        assert custom["CustomizedMetricSpecification"]["MetricName"] == "CPUUtilization"
        step_alarm = f"sagemaker-sdk-step-scaling-cpu-burst-{name}-AllTraffic"
        assert _alarm_count(sagemaker_session, step_alarm) == 1
        (schedule,) = state["scheduled_actions"]
        assert (schedule["ScheduledActionName"], schedule["Timezone"]) == (
            "sagemaker-sdk-night",
            "America/Los_Angeles",
        )

        state = suspend_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)
        assert set(state["scalable_target"]["SuspendedState"].values()) == {True}
        state = resume_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)
        assert set(state["scalable_target"]["SuspendedState"].values()) == {False}

        # Declarative: a policy and schedule no longer given are removed, with the step alarm.
        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=1,
            max_capacity=2,
            target_value=100,
            metric="invocations",
            sagemaker_session=sagemaker_session,
        )
        assert _policy_names(state) == {TRACKING}
        assert state["scheduled_actions"] == []
        assert _alarm_count(sagemaker_session, step_alarm) == 0
        (policy,) = state["policies"]
        assert policy["TargetTrackingScalingPolicyConfiguration"]["PredefinedMetricSpecification"][
            "PredefinedMetricType"
        ] == ("SageMakerVariantInvocationsPerInstance")

        result = disable_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)
        assert result["deregistered"] is True
        assert (
            describe_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)[
                "scalable_target"
            ]
            is None
        )
    finally:
        resources.cleanup(endpoint_name=name)


@pytest.mark.slow_test
def test_inference_component_scale_to_zero_on_change_off(sagemaker_session, role, container):
    resources = _Resources(sagemaker_session, role, container, "autoscale-ic")
    name = resources.name
    component_name = f"{name}-ic"
    try:
        # MinInstanceCount 0 goes through sagemaker-core shape serialization.
        resources.endpoint(
            ProductionVariant(
                variant_name="AllTraffic",
                instance_type=INSTANCE_TYPE,
                initial_instance_count=1,
                managed_instance_scaling=ProductionVariantManagedInstanceScaling(
                    status="ENABLED", min_instance_count=0, max_instance_count=2
                ),
            ),
            execution_role_arn=role,
        )
        scaling = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=name
        )["ProductionVariants"][0]["ManagedInstanceScaling"]
        assert scaling == {"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2}
        resources.component()

        state = enable_autoscaling(
            inference_component_name=component_name,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
            sagemaker_session=sagemaker_session,
        )
        assert state["scalable_dimension"] == "sagemaker:inference-component:DesiredCopyCount"
        assert state["scalable_target"]["MinCapacity"] == 0
        assert _policy_names(state) == {TRACKING, FROM_ZERO}
        alarm = f"{FROM_ZERO}-{component_name}"
        assert _alarm_count(sagemaker_session, alarm) == 1

        # Moving the minimum back to 1 removes the scale-out-from-zero policy and its alarm.
        state = enable_autoscaling(
            inference_component_name=component_name,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
            sagemaker_session=sagemaker_session,
        )
        assert state["scalable_target"]["MinCapacity"] == 1
        assert _policy_names(state) == {TRACKING}
        assert _alarm_count(sagemaker_session, alarm) == 0

        result = disable_autoscaling(
            inference_component_name=component_name, sagemaker_session=sagemaker_session
        )
        assert result["deregistered"] is True
        assert result["remaining"]["scalable_target"] is None
    finally:
        resources.cleanup(inference_component_name=component_name)


@pytest.mark.slow_test
def test_asynchronous_endpoint_scales_to_zero_on_backlog(sagemaker_session, role, container):
    resources = _Resources(sagemaker_session, role, container, "autoscale-async")
    name = resources.name
    output = f"s3://{sagemaker_session.default_bucket()}/autoscaling-integ/{name}/"
    try:
        resources.endpoint(
            _realtime_variant(name),
            async_inference_config=AsyncInferenceConfig(
                output_config=AsyncInferenceOutputConfig(s3_output_path=output)
            ),
        )
        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=0,
            max_capacity=2,
            target_value=5,
            scale_to_zero=True,
            sagemaker_session=sagemaker_session,
        )
        assert state["scalable_target"]["MinCapacity"] == 0
        assert _policy_names(state) == {TRACKING, FROM_ZERO}
        by_name = {policy["PolicyName"]: policy for policy in state["policies"]}
        tracking = by_name[TRACKING]["TargetTrackingScalingPolicyConfiguration"]
        assert tracking["CustomizedMetricSpecification"]["MetricName"] == (
            "ApproximateBacklogSizePerInstance"
        )
        alarm = f"{FROM_ZERO}-{name}-AllTraffic"
        assert _alarm_count(sagemaker_session, alarm) == 1

        result = disable_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)
        assert result["deregistered"] is True
        assert _alarm_count(sagemaker_session, alarm) == 0
    finally:
        resources.cleanup(endpoint_name=name)


@pytest.mark.slow_test
def test_serverless_provisioned_concurrency(sagemaker_session, role, container):
    resources = _Resources(sagemaker_session, role, container, "autoscale-serverless")
    name = resources.name
    client = sagemaker_session.sagemaker_client
    key = f"autoscaling-integ/{name}/model.tar.gz"
    try:
        artifact = _packed_artifact(sagemaker_session, container, key)
        # Created with the client: a serverless endpoint needs a model.tar.gz (ModelDataUrl), and
        # sagemaker-core serializes memory_size_in_mb as MemorySizeInMb.
        client.create_model(
            ModelName=name,
            ExecutionRoleArn=role,
            PrimaryContainer={
                "Image": container.image,
                "ModelDataUrl": artifact,
                "Environment": container.environment,
            },
        )
        resources.created.append(Model.get(model_name=name, session=resources.boto))
        client.create_endpoint_config(
            EndpointConfigName=name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": name,
                    "ServerlessConfig": {
                        "MemorySizeInMB": 2048,
                        "MaxConcurrency": 4,
                        "ProvisionedConcurrency": 1,
                    },
                }
            ],
        )
        resources.created.append(
            EndpointConfig.get(endpoint_config_name=name, session=resources.boto)
        )
        endpoint = Endpoint.create(
            endpoint_name=name, endpoint_config_name=name, session=resources.boto
        )
        resources.created.append(endpoint)
        endpoint.wait_for_status("InService")
        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=1,
            max_capacity=2,
            target_value=0.5,
            sagemaker_session=sagemaker_session,
        )
        assert state["scalable_dimension"] == "sagemaker:variant:DesiredProvisionedConcurrency"
        (policy,) = state["policies"]
        assert policy["TargetTrackingScalingPolicyConfiguration"]["PredefinedMetricSpecification"][
            "PredefinedMetricType"
        ] == ("SageMakerVariantProvisionedConcurrencyUtilization")

        result = disable_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session)
        assert result["deregistered"] is True
    finally:
        resources.cleanup(endpoint_name=name)
        sagemaker_session.boto_session.client("s3").delete_object(
            Bucket=sagemaker_session.default_bucket(), Key=key
        )
