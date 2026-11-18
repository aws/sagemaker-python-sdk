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
"""Target-tracking autoscaling for SageMaker real-time endpoints and inference components.

A single-model endpoint scales its production variant's instance count
(``sagemaker:variant:DesiredInstanceCount``). An endpoint that hosts inference components
scales one component's copy count (``sagemaker:inference-component:DesiredCopyCount``), and its
managed instance scaling adds or removes the instances those copies need. An inference
component can also scale to zero copies, with a step-scaling policy and a CloudWatch alarm on
``NoCapacityInvocationFailures`` that add a copy when a request finds none.

Example::

    from sagemaker.serve.autoscaling import enable_autoscaling, disable_autoscaling

    enable_autoscaling(endpoint_name="my-endpoint", min_capacity=1, max_capacity=4, target_value=4)
    disable_autoscaling(endpoint_name="my-endpoint")

Everything this module creates is tagged, and ``disable_autoscaling`` deletes only what it
created: other scaling policies, scheduled actions and a scalable target registered by someone
else stay.
"""

from __future__ import absolute_import

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

SERVICE_NAMESPACE = "sagemaker"
OWNER_TAG_KEY = "sagemaker-sdk:autoscaling"
OWNER_TAG_VALUE = "managed"

#: Predefined target-tracking metrics, by kind and metric name.
METRICS = {
    "variant": {
        "concurrent-requests": "SageMakerVariantConcurrentRequestsPerModelHighResolution",
        "invocations": "SageMakerVariantInvocationsPerInstance",
    },
    "inference-component": {
        "concurrent-requests": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution",
        "invocations": "SageMakerInferenceComponentInvocationsPerCopy",
    },
}
DIMENSIONS = {
    "variant": "sagemaker:variant:DesiredInstanceCount",
    "inference-component": "sagemaker:inference-component:DesiredCopyCount",
}
_READ_BACK_ATTEMPTS = 6
_READ_BACK_DELAY_SECONDS = 5


@dataclass
class ScalingTarget:
    """What autoscaling applies to: a production variant or an inference component."""

    kind: str
    endpoint_name: str
    resource_id: str
    name: str
    variant_name: Optional[str] = None
    inference_component_name: Optional[str] = None

    @property
    def scalable_dimension(self) -> str:
        """The Application Auto Scaling dimension this target scales."""
        return DIMENSIONS[self.kind]

    @property
    def policy_name(self) -> str:
        """The name of the target-tracking policy this module puts."""
        return f"{self.name}-target-tracking"

    @property
    def step_policy_name(self) -> str:
        """The name of the scale-out-from-zero step policy (scale to zero only)."""
        return f"{self.name}-scale-out-from-zero"

    @property
    def alarm_name(self) -> str:
        """The name of the CloudWatch alarm that drives the step policy (scale to zero only)."""
        return f"{self.name}-no-capacity-invocations"

    def target_kwargs(self) -> Dict[str, str]:
        """The keyword arguments that name this target in Application Auto Scaling calls."""
        return {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": self.resource_id,
            "ScalableDimension": self.scalable_dimension,
        }


class _Clients:
    """The SageMaker, Application Auto Scaling and CloudWatch clients of a session."""

    def __init__(self, sagemaker_session: Optional[Session]):
        session = sagemaker_session or Session()
        region = session.boto_region_name
        self.sagemaker = session.sagemaker_client
        self.autoscaling = session.boto_session.client(
            "application-autoscaling", region_name=region
        )
        self.cloudwatch = session.boto_session.client("cloudwatch", region_name=region)


def enable_autoscaling(
    *,
    min_capacity: int,
    max_capacity: int,
    target_value: float,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    metric: str = "concurrent-requests",
    scale_in_cooldown: int = 300,
    scale_out_cooldown: int = 300,
    disable_scale_in: bool = False,
    scale_to_zero: bool = False,
    wait: bool = True,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Turn on, or change, target-tracking autoscaling for an endpoint or inference component.

    Args:
        min_capacity (int): Minimum instances (variant) or copies (inference component); at
            least 1, or 0 with ``scale_to_zero``.
        max_capacity (int): Maximum instances or copies.
        target_value (float): The metric's value per instance or copy that the policy keeps the
            endpoint near.
        endpoint_name (str): The endpoint, to scale its production variant's instance count.
        variant_name (str): The variant, when the endpoint has more than one.
        inference_component_name (str): The inference component, to scale its copy count. On an
            endpoint with LoRA adapters, the base component (the adapters run on its copies).
        metric (str): ``"concurrent-requests"`` (default; sampled every 10 seconds, so it scales
            out sooner) or ``"invocations"`` (requests per minute).
        scale_in_cooldown (int): Seconds after a scale-in before the next one (default 300).
        scale_out_cooldown (int): Seconds after a scale-out before the next one (default 300).
        disable_scale_in (bool): Keep the policy from removing capacity (default False).
        scale_to_zero (bool): Inference components only: allow 0 copies, and add the step-scaling
            policy and CloudWatch alarm that add a copy when a request finds none. Needs
            ``min_capacity=0`` and managed instance scaling with ``MinInstanceCount`` 0. The first
            request at zero fails while a copy starts, which takes several minutes.
        wait (bool): Wait until the scalable target and policy read back (default True).
        sagemaker_session (Session): The session to use; a default one if omitted.

    Returns:
        dict: The resulting configuration, as :func:`describe_autoscaling` returns it.

    Raises:
        ValueError: If the arguments or the endpoint cannot be autoscaled as asked. Nothing is
            changed.
    """
    _validate_capacity(min_capacity, max_capacity, scale_to_zero)
    if not isinstance(target_value, (int, float)) or isinstance(target_value, bool):
        raise ValueError(f"target_value must be a number, got {target_value!r}.")
    if target_value <= 0:
        raise ValueError(f"target_value must be greater than 0, got {target_value}.")
    for name, value in (
        ("scale_in_cooldown", scale_in_cooldown),
        ("scale_out_cooldown", scale_out_cooldown),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be an integer >= 0, got {value!r}.")

    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=True
    )
    metric_type = METRICS[target.kind].get(metric)
    if metric_type is None:
        raise ValueError(f"metric must be one of {sorted(METRICS[target.kind])}, got {metric!r}.")
    adapters = _adapters(clients, target) if target.kind == "inference-component" else []
    if adapters and metric == "invocations":
        raise ValueError(
            f"{target.name} has adapters {adapters}: use metric 'concurrent-requests', which "
            "counts the requests its adapters serve."
        )
    if scale_to_zero:
        _check_scale_to_zero(clients, target, adapters)

    current = _read(clients, target)
    registered_before = current["scalable_target"] is not None
    # With scale to zero, keep a minimum of 1 until the alarm that scales out from 0 exists, so
    # a failed alarm write cannot leave a component that scales in to 0 and never back out.
    lower_last = scale_to_zero and not (
        registered_before and current["scalable_target"]["MinCapacity"] == 0
    )
    register = dict(
        target.target_kwargs(),
        MinCapacity=1 if lower_last else min_capacity,
        MaxCapacity=max_capacity,
    )
    if not registered_before:
        register["Tags"] = {OWNER_TAG_KEY: OWNER_TAG_VALUE}
    clients.autoscaling.register_scalable_target(**register)
    clients.autoscaling.put_scaling_policy(
        **target.target_kwargs(),
        PolicyName=target.policy_name,
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": float(target_value),
            "PredefinedMetricSpecification": {"PredefinedMetricType": metric_type},
            "ScaleInCooldown": scale_in_cooldown,
            "ScaleOutCooldown": scale_out_cooldown,
            "DisableScaleIn": disable_scale_in,
        },
    )
    have = {p["PolicyName"] for p in current["policies"]}
    if scale_to_zero:
        _put_scale_out_from_zero(clients, target)
        if lower_last:
            clients.autoscaling.register_scalable_target(
                **target.target_kwargs(), MinCapacity=0, MaxCapacity=max_capacity
            )
    elif target.step_policy_name in have:
        # A change away from scale to zero.
        clients.autoscaling.delete_scaling_policy(
            **target.target_kwargs(), PolicyName=target.step_policy_name
        )
        _delete_alarm(clients, target)

    if wait:
        return _wait_for(
            clients,
            target,
            lambda state: state["scalable_target"] is not None
            and target.policy_name in _policy_names(state)
            and (target.step_policy_name in _policy_names(state)) == scale_to_zero,
        )
    return _read(clients, target)


def disable_autoscaling(
    *,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    wait: bool = True,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Turn off the autoscaling that :func:`enable_autoscaling` set up.

    Deletes this module's policies and alarm, then deregisters the scalable target when this
    module registered it and no other scaling policy or scheduled action is on it. Otherwise the
    target and everything else on it stay. The instance or copy count stays where scaling left
    it.

    Args:
        endpoint_name (str): The endpoint whose production variant was scaled.
        variant_name (str): The variant, when the endpoint has more than one.
        inference_component_name (str): The inference component whose copies were scaled.
        wait (bool): Wait until the change reads back (default True).
        sagemaker_session (Session): The session to use; a default one if omitted.

    Returns:
        dict: ``{"deregistered": bool, "deleted_policies": [...], "remaining": {...}}``, where
        ``remaining`` is what :func:`describe_autoscaling` returns afterwards.
    """
    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=False
    )
    current = _read(clients, target)
    ours = [
        name
        for name in (target.policy_name, target.step_policy_name)
        if name in _policy_names(current)
    ]
    others = sorted(_policy_names(current) - set(ours))
    deregister = (
        current["scalable_target"] is not None
        and _is_ours(clients, current["scalable_target"])
        and not others
        and not current["scheduled_actions"]
    )
    if deregister:
        # Deregistering deletes the target's policies too.
        clients.autoscaling.deregister_scalable_target(**target.target_kwargs())
    else:
        for name in ours:
            clients.autoscaling.delete_scaling_policy(**target.target_kwargs(), PolicyName=name)
        if current["scalable_target"] is not None and not ours:
            logger.warning(
                "%s has no scaling policy created by enable_autoscaling; nothing was deleted.",
                target.resource_id,
            )
    _delete_alarm(clients, target)

    if wait:
        remaining = _wait_for(
            clients,
            target,
            lambda state: (
                (state["scalable_target"] is None)
                if deregister
                else not (_policy_names(state) & set(ours))
            ),
        )
    else:
        remaining = _read(clients, target)
    return {"deregistered": deregister, "deleted_policies": ours, "remaining": remaining}


def describe_autoscaling(
    *,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """The scalable target, scaling policies and scheduled actions of an endpoint or component.

    Returns:
        dict: ``{"resource_id": str, "scalable_dimension": str, "scalable_target": dict or None,
        "policies": [...], "scheduled_actions": [...]}``, with the Application Auto Scaling
        responses' fields.
    """
    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=False
    )
    return _read(clients, target)


def _validate_capacity(min_capacity, max_capacity, scale_to_zero):
    for name, value in (("min_capacity", min_capacity), ("max_capacity", max_capacity)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
    if scale_to_zero and min_capacity != 0:
        raise ValueError("scale_to_zero needs min_capacity=0.")
    if not scale_to_zero and min_capacity < 1:
        raise ValueError(
            "min_capacity must be at least 1; only an inference component with "
            "scale_to_zero=True scales to 0."
        )
    if max_capacity < max(min_capacity, 1):
        raise ValueError(
            f"max_capacity ({max_capacity}) must be at least min_capacity ({min_capacity}) and 1."
        )


def _resolve_target(clients, endpoint_name, variant_name, inference_component_name, for_apply):
    if not endpoint_name and not inference_component_name:
        raise ValueError("Pass endpoint_name, or inference_component_name, to scale.")
    sm = clients.sagemaker
    if inference_component_name:
        component = sm.describe_inference_component(InferenceComponentName=inference_component_name)
        if endpoint_name and component["EndpointName"] != endpoint_name:
            raise ValueError(
                f"{inference_component_name} is on {component['EndpointName']}, "
                f"not {endpoint_name}."
            )
        base = component.get("Specification", {}).get("BaseInferenceComponentName")
        if base:
            raise ValueError(
                f"{inference_component_name} is an adapter on {base}, whose copies it runs on: "
                f"scale inference_component_name={base!r}."
            )
        endpoint_name = component["EndpointName"]
        variant_name = component["VariantName"]
    endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
    if for_apply and endpoint["EndpointStatus"] != "InService":
        raise ValueError(f"{endpoint_name} is {endpoint['EndpointStatus']}, not InService.")
    config = sm.describe_endpoint_config(EndpointConfigName=endpoint["EndpointConfigName"])
    if config.get("AsyncInferenceConfig"):
        raise ValueError(
            f"{endpoint_name} is an asynchronous endpoint, which this module does not scale."
        )
    variants = {v["VariantName"]: v for v in config["ProductionVariants"]}
    if variant_name is None:
        if len(variants) != 1:
            raise ValueError(f"{endpoint_name} has variants {sorted(variants)}: pass variant_name.")
        variant_name = next(iter(variants))
    if variant_name not in variants:
        raise ValueError(f"{endpoint_name} has no variant {variant_name!r}.")
    variant = variants[variant_name]
    if variant.get("ServerlessConfig"):
        raise ValueError(f"Variant {variant_name} of {endpoint_name} is serverless.")

    if inference_component_name:
        if for_apply and (variant.get("ManagedInstanceScaling") or {}).get("Status") != "ENABLED":
            raise ValueError(
                f"{endpoint_name} has no managed instance scaling, so it cannot add instances "
                "for new copies. Managed instance scaling is set in the endpoint configuration "
                "(deploy(managed_instance_scaling=...))."
            )
        return ScalingTarget(
            kind="inference-component",
            endpoint_name=endpoint_name,
            resource_id=f"inference-component/{inference_component_name}",
            name=inference_component_name,
            variant_name=variant_name,
            inference_component_name=inference_component_name,
        )
    if not variant.get("ModelName"):
        raise ValueError(
            f"Variant {variant_name} of {endpoint_name} hosts inference components: pass "
            "inference_component_name to scale a component's copies."
        )
    return ScalingTarget(
        kind="variant",
        endpoint_name=endpoint_name,
        resource_id=f"endpoint/{endpoint_name}/variant/{variant_name}",
        name=f"{endpoint_name}-{variant_name}",
        variant_name=variant_name,
    )


def _adapters(clients, target) -> List[str]:
    names = []
    paginator = clients.sagemaker.get_paginator("list_inference_components")
    for page in paginator.paginate(EndpointNameEquals=target.endpoint_name):
        for summary in page["InferenceComponents"]:
            name = summary["InferenceComponentName"]
            if name == target.inference_component_name:
                continue
            spec = clients.sagemaker.describe_inference_component(InferenceComponentName=name).get(
                "Specification", {}
            )
            if spec.get("BaseInferenceComponentName") == target.inference_component_name:
                names.append(name)
    return names


def _check_scale_to_zero(clients, target, adapters):
    if target.kind != "inference-component":
        raise ValueError(
            "scale_to_zero applies to an inference component; a production variant keeps at "
            "least 1 instance."
        )
    endpoint = clients.sagemaker.describe_endpoint(EndpointName=target.endpoint_name)
    config = clients.sagemaker.describe_endpoint_config(
        EndpointConfigName=endpoint["EndpointConfigName"]
    )
    variant = next(
        v for v in config["ProductionVariants"] if v["VariantName"] == target.variant_name
    )
    minimum = (variant.get("ManagedInstanceScaling") or {}).get("MinInstanceCount")
    if minimum != 0:
        raise ValueError(
            f"Managed instance scaling on {target.endpoint_name} keeps at least {minimum} "
            "instance(s), so the endpoint cannot reach zero: it needs MinInstanceCount 0."
        )
    if adapters:
        raise ValueError(
            f"scale_to_zero is not supported for a base component with adapters {adapters}: "
            "a request names its adapter, so the scale-out alarm would not see it."
        )


def _put_scale_out_from_zero(clients, target):
    # Target tracking cannot scale out from 0 copies (there is no metric to track), so a step
    # policy adds one copy when a request finds no capacity.
    policy_arn = clients.autoscaling.put_scaling_policy(
        **target.target_kwargs(),
        PolicyName=target.step_policy_name,
        PolicyType="StepScaling",
        StepScalingPolicyConfiguration={
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": 60,
            "StepAdjustments": [{"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1}],
        },
    )["PolicyARN"]
    clients.cloudwatch.put_metric_alarm(
        AlarmName=target.alarm_name,
        AlarmActions=[policy_arn],
        AlarmDescription=f"Adds a copy of {target.inference_component_name} when a request "
        "finds none.",
        Namespace="AWS/SageMaker",
        MetricName="NoCapacityInvocationFailures",
        Dimensions=[{"Name": "InferenceComponentName", "Value": target.inference_component_name}],
        Statistic="Sum",
        Period=60,
        EvaluationPeriods=1,
        DatapointsToAlarm=1,
        Threshold=1,
        ComparisonOperator="GreaterThanOrEqualToThreshold",
        TreatMissingData="notBreaching",
        Tags=[{"Key": OWNER_TAG_KEY, "Value": OWNER_TAG_VALUE}],
    )


def _delete_alarm(clients, target):
    # A step policy's alarm outlives the policy; delete it only when it watches this component.
    if target.kind != "inference-component":
        return
    wanted = {"Name": "InferenceComponentName", "Value": target.inference_component_name}
    for alarm in clients.cloudwatch.describe_alarms(AlarmNames=[target.alarm_name])["MetricAlarms"]:
        if alarm.get("MetricName") == "NoCapacityInvocationFailures" and wanted in alarm.get(
            "Dimensions", []
        ):
            clients.cloudwatch.delete_alarms(AlarmNames=[target.alarm_name])


def _is_ours(clients, scalable_target) -> bool:
    arn = scalable_target.get("ScalableTargetARN")
    if not arn:
        return False
    tags = clients.autoscaling.list_tags_for_resource(ResourceARN=arn).get("Tags", {})
    return tags.get(OWNER_TAG_KEY) == OWNER_TAG_VALUE


def _read(clients, target) -> Dict[str, Any]:
    aas = clients.autoscaling
    targets = aas.describe_scalable_targets(
        ServiceNamespace=SERVICE_NAMESPACE,
        ResourceIds=[target.resource_id],
        ScalableDimension=target.scalable_dimension,
    )["ScalableTargets"]
    return {
        "resource_id": target.resource_id,
        "scalable_dimension": target.scalable_dimension,
        "scalable_target": targets[0] if targets else None,
        "policies": aas.describe_scaling_policies(**target.target_kwargs())["ScalingPolicies"],
        "scheduled_actions": aas.describe_scheduled_actions(**target.target_kwargs())[
            "ScheduledActions"
        ],
    }


def _policy_names(state) -> set:
    return {p["PolicyName"] for p in state["policies"]}


def _wait_for(clients, target, settled) -> Dict[str, Any]:
    # Application Auto Scaling is eventually consistent: a write can take seconds to read back.
    for attempt in range(_READ_BACK_ATTEMPTS):
        state = _read(clients, target)
        if settled(state):
            return state
        if attempt < _READ_BACK_ATTEMPTS - 1:
            time.sleep(_READ_BACK_DELAY_SECONDS)
    logger.warning(
        "The change to %s was accepted but does not read back yet; check describe_autoscaling.",
        target.resource_id,
    )
    return state
