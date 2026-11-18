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

Each test creates its own CPU endpoint from a small public JumpStart model (MobileNet v2 on
ml.m5.large), turns autoscaling on, changes it and turns it off, then deletes everything it
created. The execution role is ``SAGEMAKER_INTEG_ROLE_ARN`` when set, else the caller's role.
"""

from __future__ import absolute_import

import json
import logging
import os
import uuid

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.resources import Endpoint, EndpointConfig, InferenceComponent, Model
from sagemaker.core.shapes import (
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
    describe_autoscaling,
    disable_autoscaling,
    enable_autoscaling,
)

logger = logging.getLogger(__name__)

HUB_CONTENT_NAME = "pytorch-ic-mobilenet-v2"
INSTANCE_TYPE = "ml.m5.large"


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


def _delete(*resources):
    for resource in resources:
        if resource is None:
            continue
        try:
            resource.delete()
        except Exception as error:  # noqa: BLE001 - best-effort cleanup
            logger.warning("Could not delete %s: %s", resource, error)


def _alarm_count(sagemaker_session, name):
    cloudwatch = boto3.client("cloudwatch", region_name=sagemaker_session.boto_region_name)
    return len(cloudwatch.describe_alarms(AlarmNames=[name])["MetricAlarms"])


@pytest.mark.slow_test
def test_variant_autoscaling_on_change_off(sagemaker_session, role, container):
    name = f"autoscale-variant-{uuid.uuid4().hex[:8]}"
    model = config = endpoint = None
    try:
        model = Model.create(
            model_name=name,
            primary_container=container,
            execution_role_arn=role,
            session=sagemaker_session.boto_session,
        )
        config = EndpointConfig.create(
            endpoint_config_name=name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    model_name=name,
                    instance_type=INSTANCE_TYPE,
                    initial_instance_count=1,
                )
            ],
            session=sagemaker_session.boto_session,
        )
        endpoint = Endpoint.create(
            endpoint_name=name, endpoint_config_name=name, session=sagemaker_session.boto_session
        )
        endpoint.wait_for_status("InService")

        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=1,
            max_capacity=2,
            target_value=5,
            sagemaker_session=sagemaker_session,
        )
        assert state["resource_id"] == f"endpoint/{name}/variant/AllTraffic"
        assert (
            state["scalable_target"]["MinCapacity"],
            state["scalable_target"]["MaxCapacity"],
        ) == (
            1,
            2,
        )
        (policy,) = state["policies"]
        tracking = policy["TargetTrackingScalingPolicyConfiguration"]
        assert tracking["TargetValue"] == 5.0
        assert tracking["PredefinedMetricSpecification"]["PredefinedMetricType"] == (
            "SageMakerVariantConcurrentRequestsPerModelHighResolution"
        )

        # A change updates the bounds and the policy in place.
        state = enable_autoscaling(
            endpoint_name=name,
            min_capacity=1,
            max_capacity=3,
            target_value=100,
            metric="invocations",
            sagemaker_session=sagemaker_session,
        )
        assert state["scalable_target"]["MaxCapacity"] == 3
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
        if endpoint is not None:
            disable_autoscaling(endpoint_name=name, sagemaker_session=sagemaker_session, wait=False)
        _delete(endpoint, config, model)


@pytest.mark.slow_test
def test_inference_component_scale_to_zero_on_change_off(sagemaker_session, role, container):
    name = f"autoscale-ic-{uuid.uuid4().hex[:8]}"
    component_name = f"{name}-ic"
    model = config = endpoint = component = None
    try:
        model = Model.create(
            model_name=name,
            primary_container=container,
            execution_role_arn=role,
            session=sagemaker_session.boto_session,
        )
        # MinInstanceCount 0 goes through sagemaker-core shape serialization.
        config = EndpointConfig.create(
            endpoint_config_name=name,
            production_variants=[
                ProductionVariant(
                    variant_name="AllTraffic",
                    instance_type=INSTANCE_TYPE,
                    initial_instance_count=1,
                    managed_instance_scaling=ProductionVariantManagedInstanceScaling(
                        status="ENABLED", min_instance_count=0, max_instance_count=2
                    ),
                )
            ],
            execution_role_arn=role,
            session=sagemaker_session.boto_session,
        )
        scaling = sagemaker_session.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=name
        )["ProductionVariants"][0]["ManagedInstanceScaling"]
        assert scaling == {"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2}

        endpoint = Endpoint.create(
            endpoint_name=name, endpoint_config_name=name, session=sagemaker_session.boto_session
        )
        endpoint.wait_for_status("InService")
        component = InferenceComponent.create(
            inference_component_name=component_name,
            endpoint_name=name,
            variant_name="AllTraffic",
            specification=InferenceComponentSpecification(
                model_name=name,
                compute_resource_requirements=InferenceComponentComputeResourceRequirements(
                    number_of_cpu_cores_required=1, min_memory_required_in_mb=1024
                ),
            ),
            runtime_config=InferenceComponentRuntimeConfig(copy_count=1),
            session=sagemaker_session.boto_session,
        )
        component.wait_for_status("InService")

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
        assert {p["PolicyName"] for p in state["policies"]} == {
            f"{component_name}-target-tracking",
            f"{component_name}-scale-out-from-zero",
        }
        alarm = f"{component_name}-no-capacity-invocations"
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
        assert {p["PolicyName"] for p in state["policies"]} == {f"{component_name}-target-tracking"}
        assert _alarm_count(sagemaker_session, alarm) == 0

        result = disable_autoscaling(
            inference_component_name=component_name, sagemaker_session=sagemaker_session
        )
        assert result["deregistered"] is True
        assert result["remaining"]["scalable_target"] is None
    finally:
        if component is not None:
            disable_autoscaling(
                inference_component_name=component_name,
                sagemaker_session=sagemaker_session,
                wait=False,
            )
            _delete(component)
            try:
                component.wait_for_delete()
            except Exception as error:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Could not wait for %s to delete: %s", component_name, error)
        _delete(endpoint, config, model)
