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
"""Application Auto Scaling for SageMaker endpoints and inference components.

What scales depends on the endpoint:

* A real-time or asynchronous endpoint that hosts a model on its production variant scales the
  variant's instance count (``sagemaker:variant:DesiredInstanceCount``).
* An endpoint that hosts inference components scales one component's copy count
  (``sagemaker:inference-component:DesiredCopyCount``); its managed instance scaling adds or
  removes the instances those copies need.
* A serverless endpoint with provisioned concurrency scales that concurrency
  (``sagemaker:variant:DesiredProvisionedConcurrency``).

A scalable target can carry any mix of target-tracking policies (on a predefined metric, a
custom CloudWatch metric, or metric math), step-scaling policies driven by a CloudWatch alarm,
and scheduled actions. Inference components and asynchronous
endpoints can also scale to zero.

Example::

    from sagemaker.serve.autoscaling import (
        CustomMetric, ScheduledAction, TargetTrackingPolicy, enable_autoscaling,
        disable_autoscaling,
    )

    # One target-tracking policy on the default metric.
    enable_autoscaling(endpoint_name="my-endpoint", min_capacity=1, max_capacity=4, target_value=4)

    # Several policies and a schedule.
    enable_autoscaling(
        endpoint_name="my-endpoint",
        min_capacity=1,
        max_capacity=8,
        policies=[
            TargetTrackingPolicy(target_value=4),
            TargetTrackingPolicy(
                target_value=70,
                metric=CustomMetric("GPUUtilization", namespace="/aws/sagemaker/Endpoints"),
                name="gpu",
            ),
        ],
        scheduled_actions=[
            ScheduledAction(name="business-hours", schedule="cron(0 8 ? * MON-FRI *)",
                            min_capacity=2, timezone="America/New_York"),
        ],
    )

    disable_autoscaling(endpoint_name="my-endpoint")

``enable_autoscaling`` is declarative: the policies and scheduled actions it is given are the
set this module manages on the target, so calling it again changes them and removes the ones
no longer given. It names what it creates with the ``sagemaker-sdk-`` prefix and tags the
scalable target and alarms it creates. Policies, scheduled actions, alarms and scalable targets
created any other way are left alone, and ``disable_autoscaling`` deregisters a target only when
this module registered it and nothing else is on it.
"""

from __future__ import absolute_import

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from sagemaker.core.helper.session_helper import Session

logger = logging.getLogger(__name__)

SERVICE_NAMESPACE = "sagemaker"
NAME_PREFIX = "sagemaker-sdk-"
OWNER_TAG_KEY = "sagemaker-sdk:autoscaling"
OWNER_TAG_VALUE = "managed"

VARIANT = "variant"
INFERENCE_COMPONENT = "inference-component"
PROVISIONED_CONCURRENCY = "provisioned-concurrency"

DIMENSIONS = {
    VARIANT: "sagemaker:variant:DesiredInstanceCount",
    INFERENCE_COMPONENT: "sagemaker:inference-component:DesiredCopyCount",
    PROVISIONED_CONCURRENCY: "sagemaker:variant:DesiredProvisionedConcurrency",
}

#: Short names for the predefined target-tracking metrics, by kind. A full predefined metric
#: type (such as ``"SageMakerVariantInvocationsPerInstance"``) is accepted too.
METRICS = {
    VARIANT: {
        "concurrent-requests": "SageMakerVariantConcurrentRequestsPerModelHighResolution",
        "invocations": "SageMakerVariantInvocationsPerInstance",
    },
    INFERENCE_COMPONENT: {
        "concurrent-requests": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution",
        "invocations": "SageMakerInferenceComponentInvocationsPerCopy",
    },
    PROVISIONED_CONCURRENCY: {
        "provisioned-concurrency-utilization": "SageMakerVariantProvisionedConcurrencyUtilization",
    },
}
#: The asynchronous-endpoint metric SageMaker recommends for target tracking.
BACKLOG_PER_INSTANCE = "backlog-per-instance"

_READ_BACK_ATTEMPTS = 6
_READ_BACK_DELAY_SECONDS = 5
_SCALE_OUT_FROM_ZERO = "scale-out-from-zero"


@dataclass
class CustomMetric:
    """A CloudWatch metric to scale on, for a target-tracking policy or a step-scaling alarm.

    Args:
        metric_name (str): The metric's name, such as ``"CPUUtilization"``.
        namespace (str): Its namespace (default ``"AWS/SageMaker"``; instance metrics such as
            ``CPUUtilization`` and ``GPUUtilization`` are in ``"/aws/sagemaker/Endpoints"``).
        dimensions (dict): Dimension names to values. ``None`` (the default) uses the scaled
            resource's own dimensions: ``EndpointName`` and ``VariantName`` for a variant,
            ``InferenceComponentName`` for an inference component.
        statistic (str): ``"Average"`` (default), ``"Sum"``, ``"Minimum"``, ``"Maximum"`` or
            ``"SampleCount"``.
        unit (str): The metric's unit, if it has one.
    """

    metric_name: str
    namespace: str = "AWS/SageMaker"
    dimensions: Optional[Dict[str, str]] = None
    statistic: str = "Average"
    unit: Optional[str] = None

    def _dimension_list(self, target) -> List[Dict[str, str]]:
        dimensions = self.dimensions if self.dimensions is not None else target.metric_dimensions
        return [{"Name": name, "Value": value} for name, value in dimensions.items()]


@dataclass
class TargetTrackingPolicy:
    """Keep a metric near a target value by adding and removing capacity.

    Args:
        target_value (float): The metric's value per instance, copy or unit of provisioned
            concurrency that the policy keeps the target near.
        metric: The metric to track. A short name from :data:`METRICS` for the target's kind
            (``"concurrent-requests"``, the default for a real-time endpoint, or
            ``"invocations"``; ``"provisioned-concurrency-utilization"`` for a serverless
            endpoint; ``"backlog-per-instance"``, the default for an asynchronous endpoint), a full
            predefined metric type, a :class:`CustomMetric`, or a
            ``CustomizedMetricSpecification`` dict as Application Auto Scaling takes it (for
            metric math).
        scale_in_cooldown (int): Seconds after a scale-in before the next one (default 300).
        scale_out_cooldown (int): Seconds after a scale-out before the next one (default 300).
        disable_scale_in (bool): Keep the policy from removing capacity (default False).
        name (str): Distinguishes this policy from the target's other target-tracking policies.
    """

    target_value: float
    metric: Union[str, CustomMetric, Dict[str, Any], None] = None
    scale_in_cooldown: int = 300
    scale_out_cooldown: int = 300
    disable_scale_in: bool = False
    name: Optional[str] = None


@dataclass
class StepAdjustment:
    """One step of a step-scaling policy.

    The bounds are relative to the alarm threshold: a step applies when the metric minus the
    threshold falls in ``[lower_bound, upper_bound)``; ``None`` is unbounded.
    """

    scaling_adjustment: int
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


@dataclass
class Alarm:
    """The CloudWatch alarm that drives a step-scaling policy.

    Args:
        metric (CustomMetric): The metric the alarm watches.
        threshold (float): The value it compares the metric with.
        comparison_operator (str): Default ``"GreaterThanOrEqualToThreshold"``.
        period (int): Seconds per evaluated data point (default 60).
        evaluation_periods (int): Data points to evaluate (default 1).
        datapoints_to_alarm (int): Breaching data points that trigger it (default all).
        treat_missing_data (str): ``"missing"`` (default), ``"notBreaching"``, ``"breaching"`` or
            ``"ignore"``.
    """

    metric: CustomMetric
    threshold: float
    comparison_operator: str = "GreaterThanOrEqualToThreshold"
    period: int = 60
    evaluation_periods: int = 1
    datapoints_to_alarm: Optional[int] = None
    treat_missing_data: str = "missing"


@dataclass
class StepScalingPolicy:
    """Change capacity by fixed steps when a CloudWatch alarm fires.

    Args:
        steps (list[StepAdjustment]): The adjustments, by how far the metric is past the
            threshold.
        alarm (Alarm): The alarm this module creates for the policy.
        adjustment_type (str): ``"ChangeInCapacity"`` (default), ``"ExactCapacity"`` or
            ``"PercentChangeInCapacity"``.
        metric_aggregation_type (str): ``"Average"`` (default), ``"Minimum"`` or ``"Maximum"``.
        cooldown (int): Seconds after a scaling activity before the next one (default 300).
        min_adjustment_magnitude (int): For ``"PercentChangeInCapacity"``, the smallest change.
        name (str): Distinguishes this policy from the target's other step-scaling policies.
    """

    steps: List[StepAdjustment]
    alarm: Alarm
    adjustment_type: str = "ChangeInCapacity"
    metric_aggregation_type: str = "Average"
    cooldown: int = 300
    min_adjustment_magnitude: Optional[int] = None
    name: Optional[str] = None


@dataclass
class ScheduledAction:
    """Change the scalable target's bounds on a schedule.

    Args:
        name (str): The action's name, unique on the target.
        schedule (str): ``"at(yyyy-mm-ddThh:mm:ss)"``, ``"rate(value unit)"`` or
            ``"cron(fields)"``.
        min_capacity (int): The minimum from then on.
        max_capacity (int): The maximum from then on.
        timezone (str): The schedule's time zone, such as ``"America/New_York"`` (default UTC).
        start_time (datetime): When a recurring schedule starts.
        end_time (datetime): When a recurring schedule ends.
    """

    name: str
    schedule: str
    min_capacity: Optional[int] = None
    max_capacity: Optional[int] = None
    timezone: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ScalingTarget:
    """What autoscaling applies to: a production variant or an inference component."""

    kind: str
    endpoint_name: str
    resource_id: str
    name: str
    variant_name: Optional[str] = None
    inference_component_name: Optional[str] = None
    asynchronous: bool = False
    adapters: List[str] = field(default_factory=list)
    managed_instance_scaling: Optional[Dict[str, Any]] = None

    @property
    def scalable_dimension(self) -> str:
        """The Application Auto Scaling dimension this target scales."""
        return DIMENSIONS[self.kind]

    @property
    def metric_dimensions(self) -> Dict[str, str]:
        """The CloudWatch dimensions of this target's own metrics."""
        if self.kind == INFERENCE_COMPONENT:
            return {"InferenceComponentName": self.inference_component_name}
        return {"EndpointName": self.endpoint_name, "VariantName": self.variant_name}

    def target_kwargs(self) -> Dict[str, str]:
        """The keyword arguments that name this target in Application Auto Scaling calls."""
        return {
            "ServiceNamespace": SERVICE_NAMESPACE,
            "ResourceId": self.resource_id,
            "ScalableDimension": self.scalable_dimension,
        }

    def alarm_name(self, policy_name: str) -> str:
        """The name of the alarm this module creates for a step-scaling policy."""
        return f"{policy_name}-{self.name}"


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
    target_value: Optional[float] = None,
    metric: Union[str, CustomMetric, Dict[str, Any], None] = None,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    policies: Optional[List[Union[TargetTrackingPolicy, StepScalingPolicy]]] = None,
    scheduled_actions: Optional[List[ScheduledAction]] = None,
    scale_in_cooldown: int = 300,
    scale_out_cooldown: int = 300,
    disable_scale_in: bool = False,
    scale_to_zero: bool = False,
    wait: bool = True,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Turn on, or change, autoscaling for an endpoint or inference component.

    Pass ``target_value`` (with ``metric`` and the cooldown arguments) for one target-tracking
    policy, or ``policies`` for any mix of policies; ``scheduled_actions`` can go with either, or
    alone. The policies and scheduled actions given replace the ones this module set before.

    Args:
        min_capacity (int): Minimum instances, copies or provisioned concurrency; at least 1,
            or 0 for an asynchronous endpoint or, with ``scale_to_zero``, an inference component.
        max_capacity (int): Maximum instances, copies or provisioned concurrency.
        target_value (float): For one target-tracking policy: the metric's target value.
        metric: For one target-tracking policy: the metric (see :class:`TargetTrackingPolicy`).
        endpoint_name (str): The endpoint, to scale its production variant.
        variant_name (str): The variant, when the endpoint has more than one.
        inference_component_name (str): The inference component, to scale its copies. On an
            endpoint with LoRA adapters, the base component (the adapters run on its copies).
        policies (list): :class:`TargetTrackingPolicy` and :class:`StepScalingPolicy` objects.
        scheduled_actions (list[ScheduledAction]): Scheduled changes to the bounds.
        scale_in_cooldown (int): For one target-tracking policy (default 300).
        scale_out_cooldown (int): For one target-tracking policy (default 300).
        disable_scale_in (bool): For one target-tracking policy (default False).
        scale_to_zero (bool): Add the step-scaling policy and CloudWatch alarm that add capacity
            when a request arrives at zero: on ``NoCapacityInvocationFailures`` for an inference
            component (which needs ``min_capacity=0`` and managed instance scaling with
            ``MinInstanceCount`` 0), on ``HasBacklogWithoutCapacity`` for an asynchronous
            endpoint. A request to an inference component at zero fails while a copy starts,
            which takes several minutes; an asynchronous request waits in the queue.
        wait (bool): Wait until the change reads back (default True).
        sagemaker_session (Session): The session to use; a default one if omitted.

    Returns:
        dict: The resulting configuration, as :func:`describe_autoscaling` returns it.

    Raises:
        ValueError: If the arguments or the endpoint cannot be autoscaled as asked. Nothing is
            changed.
    """
    if policies is None and target_value is not None:
        policies = [
            TargetTrackingPolicy(
                target_value=target_value,
                metric=metric,
                scale_in_cooldown=scale_in_cooldown,
                scale_out_cooldown=scale_out_cooldown,
                disable_scale_in=disable_scale_in,
            )
        ]
    elif policies is not None and (target_value is not None or metric is not None):
        raise ValueError("Pass target_value and metric, or policies, not both.")
    policies = list(policies or [])
    scheduled_actions = list(scheduled_actions or [])
    if not policies and not scheduled_actions and not scale_to_zero:
        raise ValueError("Pass target_value, policies or scheduled_actions.")

    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=True
    )
    _validate_capacity(target, min_capacity, max_capacity, scale_to_zero)
    requests = [_policy_request(target, policy) for policy in policies]
    if scale_to_zero:
        requests.append(_policy_request(target, _scale_out_from_zero_policy(target)))
    _check_unique([request["PolicyName"] for request in requests], "policy")
    schedule_requests = [_schedule_request(action) for action in scheduled_actions]
    _check_unique(
        [request["ScheduledActionName"] for request in schedule_requests], "scheduled action"
    )

    current = _read(clients, target)
    registered_before = current["scalable_target"] is not None
    # With scale to zero on an inference component, keep a minimum of 1 until the alarm that
    # scales out from 0 exists, so a failed alarm write cannot leave a component that scales in
    # to 0 and never back out.
    lower_last = (
        scale_to_zero
        and target.kind == INFERENCE_COMPONENT
        and not (registered_before and current["scalable_target"]["MinCapacity"] == 0)
    )
    register = dict(
        target.target_kwargs(),
        MinCapacity=1 if lower_last else min_capacity,
        MaxCapacity=max_capacity,
    )
    if not registered_before:
        register["Tags"] = {OWNER_TAG_KEY: OWNER_TAG_VALUE}
    clients.autoscaling.register_scalable_target(**register)

    for request in requests:
        alarm = request.pop("_alarm", None)
        arn = clients.autoscaling.put_scaling_policy(**target.target_kwargs(), **request)[
            "PolicyARN"
        ]
        if alarm is not None:
            clients.cloudwatch.put_metric_alarm(**alarm, AlarmActions=[arn])
    for request in schedule_requests:
        clients.autoscaling.put_scheduled_action(**target.target_kwargs(), **request)

    wanted = {request["PolicyName"] for request in requests}
    for name in sorted(_managed(_policy_names(current)) - wanted):
        clients.autoscaling.delete_scaling_policy(**target.target_kwargs(), PolicyName=name)
        _delete_alarm(clients, target.alarm_name(name))
    wanted_schedules = {request["ScheduledActionName"] for request in schedule_requests}
    for name in sorted(_managed(_schedule_names(current)) - wanted_schedules):
        clients.autoscaling.delete_scheduled_action(
            **target.target_kwargs(), ScheduledActionName=name
        )

    if lower_last:
        clients.autoscaling.register_scalable_target(
            **target.target_kwargs(), MinCapacity=0, MaxCapacity=max_capacity
        )

    if wait:
        return _wait_for(
            clients,
            target,
            lambda state: state["scalable_target"] is not None
            and _managed(_policy_names(state)) == wanted
            and _managed(_schedule_names(state)) == wanted_schedules
            and state["scalable_target"]["MinCapacity"] == min_capacity
            and state["scalable_target"]["MaxCapacity"] == max_capacity,
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

    Deletes this module's policies, their alarms and its scheduled actions, then deregisters
    the scalable target when this module registered it and no other policy or scheduled action
    is on it. Otherwise the target and everything else on it stay. The capacity stays where
    scaling left it.

    Args:
        endpoint_name (str): The endpoint whose production variant was scaled.
        variant_name (str): The variant, when the endpoint has more than one.
        inference_component_name (str): The inference component whose copies were scaled.
        wait (bool): Wait until the change reads back (default True).
        sagemaker_session (Session): The session to use; a default one if omitted.

    Returns:
        dict: ``{"deregistered": bool, "deleted_policies": [...],
        "deleted_scheduled_actions": [...], "remaining": {...}}``, where ``remaining`` is what
        :func:`describe_autoscaling` returns afterwards.
    """
    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=False
    )
    current = _read(clients, target)
    policies = sorted(_managed(_policy_names(current)))
    schedules = sorted(_managed(_schedule_names(current)))
    others = (_policy_names(current) - set(policies)) | (_schedule_names(current) - set(schedules))
    deregister = (
        current["scalable_target"] is not None
        and _is_ours(clients, current["scalable_target"])
        and not others
    )
    if deregister:
        # Deregistering deletes the target's policies and scheduled actions too.
        clients.autoscaling.deregister_scalable_target(**target.target_kwargs())
    else:
        for name in policies:
            clients.autoscaling.delete_scaling_policy(**target.target_kwargs(), PolicyName=name)
        for name in schedules:
            clients.autoscaling.delete_scheduled_action(
                **target.target_kwargs(), ScheduledActionName=name
            )
        if current["scalable_target"] is not None and not policies and not schedules:
            logger.warning(
                "%s has no policy or scheduled action created by enable_autoscaling; nothing was "
                "deleted.",
                target.resource_id,
            )
    for name in policies:
        _delete_alarm(clients, target.alarm_name(name))

    if wait:
        remaining = _wait_for(
            clients,
            target,
            lambda state: (
                (state["scalable_target"] is None)
                if deregister
                else not (_managed(_policy_names(state)) or _managed(_schedule_names(state)))
            ),
        )
    else:
        remaining = _read(clients, target)
    return {
        "deregistered": deregister,
        "deleted_policies": policies,
        "deleted_scheduled_actions": schedules,
        "remaining": remaining,
    }


def suspend_autoscaling(
    *,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    scale_in: bool = True,
    scale_out: bool = True,
    scheduled: bool = True,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Pause scaling on a registered target, keeping its policies and scheduled actions.

    Args:
        scale_in (bool): Pause scale-in by the scaling policies (default True).
        scale_out (bool): Pause scale-out by the scaling policies (default True).
        scheduled (bool): Pause the scheduled actions (default True).

    Returns:
        dict: The resulting configuration, as :func:`describe_autoscaling` returns it.
    """
    return _set_suspended(
        endpoint_name,
        variant_name,
        inference_component_name,
        {
            "DynamicScalingInSuspended": scale_in,
            "DynamicScalingOutSuspended": scale_out,
            "ScheduledScalingSuspended": scheduled,
        },
        sagemaker_session,
    )


def resume_autoscaling(
    *,
    endpoint_name: Optional[str] = None,
    variant_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    sagemaker_session: Optional[Session] = None,
) -> Dict[str, Any]:
    """Resume scaling that :func:`suspend_autoscaling` paused.

    Returns:
        dict: The resulting configuration, as :func:`describe_autoscaling` returns it.
    """
    return _set_suspended(
        endpoint_name,
        variant_name,
        inference_component_name,
        {
            "DynamicScalingInSuspended": False,
            "DynamicScalingOutSuspended": False,
            "ScheduledScalingSuspended": False,
        },
        sagemaker_session,
    )


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
        responses' fields. The scalable target's ``SuspendedState`` says whether scaling is
        paused.
    """
    clients = _Clients(sagemaker_session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=False
    )
    return _read(clients, target)


def _set_suspended(endpoint_name, variant_name, inference_component_name, state, session):
    clients = _Clients(session)
    target = _resolve_target(
        clients, endpoint_name, variant_name, inference_component_name, for_apply=False
    )
    if _read(clients, target)["scalable_target"] is None:
        raise ValueError(f"{target.resource_id} is not registered for autoscaling.")
    clients.autoscaling.register_scalable_target(**target.target_kwargs(), SuspendedState=state)
    return _wait_for(
        clients,
        target,
        lambda current: current["scalable_target"] is not None
        and all(
            current["scalable_target"].get("SuspendedState", {}).get(key, False) == value
            for key, value in state.items()
        ),
    )


def _int(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")


def _validate_capacity(target, min_capacity, max_capacity, scale_to_zero):
    _int("max_capacity", max_capacity, 1)
    if scale_to_zero:
        if target.kind == PROVISIONED_CONCURRENCY:
            raise ValueError(
                "A serverless endpoint's provisioned concurrency does not scale to zero."
            )
        if target.kind == VARIANT and not target.asynchronous:
            raise ValueError(
                "scale_to_zero applies to an inference component or an asynchronous endpoint; a "
                "real-time production variant keeps at least 1 instance."
            )
        if min_capacity != 0:
            raise ValueError("scale_to_zero needs min_capacity=0.")
        if target.kind == INFERENCE_COMPONENT:
            minimum = (target.managed_instance_scaling or {}).get("MinInstanceCount")
            if minimum != 0:
                raise ValueError(
                    f"Managed instance scaling on {target.endpoint_name} keeps at least "
                    f"{minimum} instance(s), so the endpoint cannot reach zero: it needs "
                    "MinInstanceCount 0."
                )
            if target.adapters:
                raise ValueError(
                    f"scale_to_zero is not supported for a base component with adapters "
                    f"{target.adapters}: a request names its adapter, so the scale-out alarm "
                    "would not see it."
                )
    _int("min_capacity", min_capacity, 0 if (scale_to_zero or target.asynchronous) else 1)
    if min_capacity > max_capacity:
        raise ValueError(
            f"min_capacity ({min_capacity}) must not exceed max_capacity ({max_capacity})."
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
    variants = {v["VariantName"]: v for v in config["ProductionVariants"]}
    if variant_name is None:
        if len(variants) != 1:
            raise ValueError(f"{endpoint_name} has variants {sorted(variants)}: pass variant_name.")
        variant_name = next(iter(variants))
    if variant_name not in variants:
        raise ValueError(f"{endpoint_name} has no variant {variant_name!r}.")
    variant = variants[variant_name]
    managed = variant.get("ManagedInstanceScaling") or None

    if inference_component_name:
        if for_apply and (managed or {}).get("Status") != "ENABLED":
            raise ValueError(
                f"{endpoint_name} has no managed instance scaling, so it cannot add instances "
                "for new copies. Managed instance scaling is set in the endpoint configuration "
                "(deploy(managed_instance_scaling=...))."
            )
        target = ScalingTarget(
            kind=INFERENCE_COMPONENT,
            endpoint_name=endpoint_name,
            resource_id=f"inference-component/{inference_component_name}",
            name=inference_component_name,
            variant_name=variant_name,
            inference_component_name=inference_component_name,
            managed_instance_scaling=managed,
        )
        target.adapters = _adapters(clients, target) if for_apply else []
        return target

    serverless = variant.get("ServerlessConfig")
    if serverless:
        if not serverless.get("ProvisionedConcurrency"):
            raise ValueError(
                f"Variant {variant_name} of {endpoint_name} is serverless without provisioned "
                "concurrency; SageMaker scales on-demand serverless capacity itself."
            )
        kind = PROVISIONED_CONCURRENCY
    else:
        if not variant.get("ModelName"):
            raise ValueError(
                f"Variant {variant_name} of {endpoint_name} hosts inference components: pass "
                "inference_component_name to scale a component's copies."
            )
        kind = VARIANT
    return ScalingTarget(
        kind=kind,
        endpoint_name=endpoint_name,
        resource_id=f"endpoint/{endpoint_name}/variant/{variant_name}",
        name=f"{endpoint_name}-{variant_name}",
        variant_name=variant_name,
        asynchronous=bool(config.get("AsyncInferenceConfig")),
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


def _metric_specification(target, metric) -> Dict[str, Any]:
    """The metric part of a target-tracking configuration."""
    if metric is None:
        metric = BACKLOG_PER_INSTANCE if target.asynchronous else next(iter(METRICS[target.kind]))
    if isinstance(metric, dict):
        return {"CustomizedMetricSpecification": metric}
    if isinstance(metric, CustomMetric):
        specification = {
            "MetricName": metric.metric_name,
            "Namespace": metric.namespace,
            "Dimensions": metric._dimension_list(target),  # pylint: disable=protected-access
            "Statistic": metric.statistic,
        }
        if metric.unit:
            specification["Unit"] = metric.unit
        return {"CustomizedMetricSpecification": specification}
    if metric == BACKLOG_PER_INSTANCE:
        if not target.asynchronous:
            raise ValueError("metric 'backlog-per-instance' applies to an asynchronous endpoint.")
        return {
            "CustomizedMetricSpecification": {
                "MetricName": "ApproximateBacklogSizePerInstance",
                "Namespace": "AWS/SageMaker",
                "Dimensions": [{"Name": "EndpointName", "Value": target.endpoint_name}],
                "Statistic": "Average",
            }
        }
    predefined = METRICS[target.kind].get(metric)
    if predefined is None and metric in METRICS[target.kind].values():
        predefined = metric
    if predefined is None:
        names = sorted(METRICS[target.kind]) + (
            [BACKLOG_PER_INSTANCE] if target.asynchronous else []
        )
        raise ValueError(
            f"metric {metric!r} does not apply to a {target.kind}: use one of {names}, a full "
            "predefined metric type, or a CustomMetric."
        )
    if target.adapters and predefined == METRICS[INFERENCE_COMPONENT]["invocations"]:
        raise ValueError(
            f"{target.name} has adapters {target.adapters}: use metric 'concurrent-requests', "
            "which counts the requests its adapters serve."
        )
    return {"PredefinedMetricSpecification": {"PredefinedMetricType": predefined}}


def _policy_request(target, policy) -> Dict[str, Any]:
    """A PutScalingPolicy request (without the target) and, for step scaling, its alarm."""
    name = getattr(policy, "name", None)
    suffix = f"-{name}" if name else ""
    if isinstance(policy, TargetTrackingPolicy):
        if (
            isinstance(policy.target_value, bool)
            or not isinstance(policy.target_value, (int, float))
            or policy.target_value <= 0
        ):
            raise ValueError(f"target_value must be a number > 0, got {policy.target_value!r}.")
        _int("scale_in_cooldown", policy.scale_in_cooldown, 0)
        _int("scale_out_cooldown", policy.scale_out_cooldown, 0)
        configuration = {
            "TargetValue": float(policy.target_value),
            "ScaleInCooldown": policy.scale_in_cooldown,
            "ScaleOutCooldown": policy.scale_out_cooldown,
            "DisableScaleIn": policy.disable_scale_in,
        }
        configuration.update(_metric_specification(target, policy.metric))
        return {
            "PolicyName": f"{NAME_PREFIX}target-tracking{suffix}",
            "PolicyType": "TargetTrackingScaling",
            "TargetTrackingScalingPolicyConfiguration": configuration,
        }
    if isinstance(policy, StepScalingPolicy):
        if not policy.steps:
            raise ValueError("A StepScalingPolicy needs at least one step.")
        _int("cooldown", policy.cooldown, 0)
        steps = []
        for step in policy.steps:
            adjustment = {"ScalingAdjustment": step.scaling_adjustment}
            if step.lower_bound is not None:
                adjustment["MetricIntervalLowerBound"] = step.lower_bound
            if step.upper_bound is not None:
                adjustment["MetricIntervalUpperBound"] = step.upper_bound
            steps.append(adjustment)
        configuration = {
            "AdjustmentType": policy.adjustment_type,
            "MetricAggregationType": policy.metric_aggregation_type,
            "Cooldown": policy.cooldown,
            "StepAdjustments": steps,
        }
        if policy.min_adjustment_magnitude is not None:
            configuration["MinAdjustmentMagnitude"] = policy.min_adjustment_magnitude
        name = f"{NAME_PREFIX}step-scaling{suffix}"
        return {
            "PolicyName": name,
            "PolicyType": "StepScaling",
            "StepScalingPolicyConfiguration": configuration,
            "_alarm": _alarm_request(target, target.alarm_name(name), policy.alarm),
        }
    raise ValueError(
        f"policies must be TargetTrackingPolicy or StepScalingPolicy objects, got "
        f"{type(policy).__name__}."
    )


def _alarm_request(target, name, alarm: Alarm) -> Dict[str, Any]:
    request = {
        "AlarmName": name,
        "AlarmDescription": f"Drives a scaling policy of {target.resource_id}.",
        "Namespace": alarm.metric.namespace,
        "MetricName": alarm.metric.metric_name,
        "Dimensions": alarm.metric._dimension_list(target),  # pylint: disable=protected-access
        "Statistic": alarm.metric.statistic,
        "Period": alarm.period,
        "EvaluationPeriods": alarm.evaluation_periods,
        "DatapointsToAlarm": alarm.datapoints_to_alarm or alarm.evaluation_periods,
        "Threshold": alarm.threshold,
        "ComparisonOperator": alarm.comparison_operator,
        "TreatMissingData": alarm.treat_missing_data,
        "Tags": [{"Key": OWNER_TAG_KEY, "Value": OWNER_TAG_VALUE}],
    }
    if alarm.metric.unit:
        request["Unit"] = alarm.metric.unit
    return request


def _scale_out_from_zero_policy(target) -> StepScalingPolicy:
    """The step policy that adds capacity when a request arrives at zero."""
    if target.kind == INFERENCE_COMPONENT:
        # Target tracking cannot scale out from 0 copies (there is no metric to track).
        alarm = Alarm(
            metric=CustomMetric("NoCapacityInvocationFailures", statistic="Sum"),
            threshold=1,
            treat_missing_data="notBreaching",
        )
        return StepScalingPolicy(
            steps=[StepAdjustment(scaling_adjustment=1, lower_bound=0)],
            alarm=alarm,
            metric_aggregation_type="Maximum",
            cooldown=60,
            name=_SCALE_OUT_FROM_ZERO,
        )
    # An asynchronous endpoint otherwise waits until the backlog exceeds the tracking target.
    alarm = Alarm(
        metric=CustomMetric(
            "HasBacklogWithoutCapacity", dimensions={"EndpointName": target.endpoint_name}
        ),
        threshold=1,
        evaluation_periods=2,
    )
    return StepScalingPolicy(
        steps=[StepAdjustment(scaling_adjustment=1, lower_bound=0)],
        alarm=alarm,
        name=_SCALE_OUT_FROM_ZERO,
    )


def _schedule_request(action: ScheduledAction) -> Dict[str, Any]:
    if not isinstance(action, ScheduledAction):
        raise ValueError(f"scheduled_actions must be ScheduledAction objects, got {action!r}.")
    if not action.name:
        raise ValueError("A ScheduledAction needs a name.")
    if not str(action.schedule).startswith(("at(", "rate(", "cron(")):
        raise ValueError(
            f"schedule must be at(...), rate(...) or cron(...), got {action.schedule!r}."
        )
    bounds = {}
    for key, value in (("MinCapacity", action.min_capacity), ("MaxCapacity", action.max_capacity)):
        if value is not None:
            _int(key, value, 0)
            bounds[key] = value
    if not bounds:
        raise ValueError(f"ScheduledAction {action.name!r} needs min_capacity or max_capacity.")
    request = {
        "ScheduledActionName": f"{NAME_PREFIX}{action.name}",
        "Schedule": action.schedule,
        "ScalableTargetAction": bounds,
    }
    if action.timezone:
        request["Timezone"] = action.timezone
    if action.start_time:
        request["StartTime"] = action.start_time
    if action.end_time:
        request["EndTime"] = action.end_time
    return request


def _check_unique(names, what):
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Each {what} needs a distinct name; {duplicates} repeat.")


def _delete_alarm(clients, name):
    # A step policy's alarm outlives the policy; delete it only when this module created it.
    for alarm in clients.cloudwatch.describe_alarms(AlarmNames=[name])["MetricAlarms"]:
        tags = clients.cloudwatch.list_tags_for_resource(ResourceARN=alarm["AlarmArn"])["Tags"]
        if {"Key": OWNER_TAG_KEY, "Value": OWNER_TAG_VALUE} in tags:
            clients.cloudwatch.delete_alarms(AlarmNames=[name])


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


def _schedule_names(state) -> set:
    return {a["ScheduledActionName"] for a in state["scheduled_actions"]}


def _managed(names) -> set:
    return {name for name in names if name.startswith(NAME_PREFIX)}


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
