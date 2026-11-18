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
"""Unit tests for sagemaker.serve.autoscaling."""

from __future__ import absolute_import

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from sagemaker.serve import autoscaling
from sagemaker.serve.autoscaling import (
    OWNER_TAG_KEY,
    OWNER_TAG_VALUE,
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

ENDPOINT = "my-endpoint"
VARIANT_ID = f"endpoint/{ENDPOINT}/variant/AllTraffic"
IC = "my-ic"
IC_ID = f"inference-component/{IC}"
TRACKING = "sagemaker-sdk-target-tracking"
FROM_ZERO = "sagemaker-sdk-step-scaling-scale-out-from-zero"
IC_FROM_ZERO_ALARM = f"{FROM_ZERO}-{IC}"
OWNER_TAGS = [{"Key": OWNER_TAG_KEY, "Value": OWNER_TAG_VALUE}]
WRITES = {
    "register_scalable_target",
    "deregister_scalable_target",
    "put_scaling_policy",
    "delete_scaling_policy",
    "put_scheduled_action",
    "delete_scheduled_action",
    "put_metric_alarm",
    "delete_alarms",
}


class FakeAWS:
    """SageMaker, Application Auto Scaling and CloudWatch over one endpoint's state."""

    def __init__(
        self,
        *,
        status="InService",
        variants=("AllTraffic",),
        model_name="my-model",
        managed=None,
        serverless=None,
        async_config=None,
        components=None,
        registered=None,
        ours=True,
        policies=(),
        scheduled_actions=(),
        alarms=None,
    ):
        self.calls = []
        self.status = status
        self.variants = variants
        self.model_name = model_name
        self.managed = managed
        self.serverless = serverless
        self.async_config = async_config
        self.components = components or {}  # {name: {"base": ..., "endpoint": ...}}
        self.target = None
        self.tags = {}
        if registered is not None:
            self.target = {"MinCapacity": registered[0], "MaxCapacity": registered[1]}
            self.tags = {OWNER_TAG_KEY: OWNER_TAG_VALUE} if ours else {}
        self.policies = {
            name: {"PolicyName": name, "PolicyType": "TargetTrackingScaling"} for name in policies
        }
        self.scheduled = {n: {"ScheduledActionName": n} for n in scheduled_actions}
        # {name: (alarm, tags)}
        self.alarms = alarms or {}
        self.sagemaker = Mock(name="sagemaker")
        self.autoscaling = Mock(name="application-autoscaling")
        self.cloudwatch = Mock(name="cloudwatch")
        self._wire()

    def _record(self, client, name, func):
        def call(*args, **kwargs):
            self.calls.append((name, kwargs))
            return func(**kwargs)

        setattr(client, name, Mock(side_effect=call))

    def _wire(self):
        sm, aas, cw = self.sagemaker, self.autoscaling, self.cloudwatch
        self._record(sm, "describe_endpoint", self._describe_endpoint)
        self._record(sm, "describe_endpoint_config", self._describe_endpoint_config)
        self._record(sm, "describe_inference_component", self._describe_component)
        paginator = Mock()
        paginator.paginate.side_effect = lambda EndpointNameEquals: [
            {
                "InferenceComponents": [
                    {"InferenceComponentName": n}
                    for n, c in self.components.items()
                    if c.get("endpoint", ENDPOINT) == EndpointNameEquals
                ]
            }
        ]
        sm.get_paginator.return_value = paginator
        for name in (
            "describe_scalable_targets",
            "describe_scaling_policies",
            "describe_scheduled_actions",
            "list_tags_for_resource",
            "register_scalable_target",
            "deregister_scalable_target",
            "put_scaling_policy",
            "delete_scaling_policy",
            "put_scheduled_action",
            "delete_scheduled_action",
        ):
            self._record(aas, name, getattr(self, "_aas_" + name))
        for name in (
            "describe_alarms",
            "put_metric_alarm",
            "delete_alarms",
            "list_tags_for_resource",
        ):
            self._record(cw, name, getattr(self, "_cw_" + name))

    # SageMaker
    def _describe_endpoint(self, EndpointName):
        return {
            "EndpointName": EndpointName,
            "EndpointStatus": self.status,
            "EndpointConfigName": f"{EndpointName}-config",
        }

    def _describe_endpoint_config(self, EndpointConfigName):
        variants = []
        for name in self.variants:
            variant = {"VariantName": name}
            if self.model_name:
                variant["ModelName"] = self.model_name
            if self.managed:
                variant["ManagedInstanceScaling"] = self.managed
            if self.serverless:
                variant["ServerlessConfig"] = self.serverless
            variants.append(variant)
        config = {"ProductionVariants": variants}
        if self.async_config:
            config["AsyncInferenceConfig"] = self.async_config
        return config

    def _describe_component(self, InferenceComponentName):
        component = self.components[InferenceComponentName]
        spec = {"BaseInferenceComponentName": component["base"]} if component.get("base") else {}
        return {
            "InferenceComponentName": InferenceComponentName,
            "EndpointName": component.get("endpoint", ENDPOINT),
            "VariantName": "AllTraffic",
            "Specification": spec,
        }

    # Application Auto Scaling
    def _aas_describe_scalable_targets(self, **kwargs):
        if not self.target:
            return {"ScalableTargets": []}
        return {"ScalableTargets": [dict(self.target, ScalableTargetARN="arn:target")]}

    def _aas_describe_scaling_policies(self, **kwargs):
        return {"ScalingPolicies": [dict(p) for p in self.policies.values()]}

    def _aas_describe_scheduled_actions(self, **kwargs):
        return {"ScheduledActions": [dict(a) for a in self.scheduled.values()]}

    def _aas_list_tags_for_resource(self, ResourceARN):
        return {"Tags": dict(self.tags)}

    def _aas_register_scalable_target(self, **kwargs):
        if not self.target:
            self.tags = dict(kwargs.get("Tags", {}))
            self.target = {}
        for key in ("MinCapacity", "MaxCapacity", "SuspendedState"):
            if key in kwargs:
                self.target[key] = kwargs[key]

    def _aas_deregister_scalable_target(self, **kwargs):
        self.target, self.tags, self.policies, self.scheduled = None, {}, {}, {}

    def _aas_put_scaling_policy(self, **kwargs):
        arn = f"arn:policy/{kwargs['PolicyName']}"
        self.policies[kwargs["PolicyName"]] = {
            "PolicyName": kwargs["PolicyName"],
            "PolicyType": kwargs["PolicyType"],
            "PolicyARN": arn,
        }
        return {"PolicyARN": arn}

    def _aas_delete_scaling_policy(self, **kwargs):
        self.policies.pop(kwargs["PolicyName"], None)

    def _aas_put_scheduled_action(self, **kwargs):
        self.scheduled[kwargs["ScheduledActionName"]] = {
            "ScheduledActionName": kwargs["ScheduledActionName"],
            "Schedule": kwargs["Schedule"],
        }

    def _aas_delete_scheduled_action(self, **kwargs):
        self.scheduled.pop(kwargs["ScheduledActionName"], None)

    # CloudWatch
    def _cw_describe_alarms(self, AlarmNames):
        return {
            "MetricAlarms": [
                {"AlarmName": n, "AlarmArn": f"arn:alarm/{n}"}
                for n in AlarmNames
                if n in self.alarms
            ]
        }

    def _cw_list_tags_for_resource(self, ResourceARN):
        return {"Tags": list(self.alarms[ResourceARN.split("/", 1)[1]][1])}

    def _cw_put_metric_alarm(self, **kwargs):
        self.alarms[kwargs["AlarmName"]] = (kwargs, kwargs.get("Tags", []))

    def _cw_delete_alarms(self, AlarmNames):
        for name in AlarmNames:
            self.alarms.pop(name, None)

    def called(self, name):
        return [kwargs for call, kwargs in self.calls if call == name]

    def writes(self):
        return [call for call, _ in self.calls if call in WRITES]


def _ic_aws(**kwargs):
    kwargs.setdefault("model_name", None)
    kwargs.setdefault(
        "managed", {"Status": "ENABLED", "MinInstanceCount": 1, "MaxInstanceCount": 4}
    )
    kwargs.setdefault("components", {IC: {}})
    return FakeAWS(**kwargs)


def _zero_ic_aws(**kwargs):
    kwargs.setdefault(
        "managed", {"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2}
    )
    return _ic_aws(**kwargs)


@pytest.fixture
def run():
    def _run(aws, func, **kwargs):
        with patch.object(autoscaling, "_Clients", return_value=aws), patch.object(
            autoscaling.time, "sleep"
        ) as sleep:
            result = func(**kwargs)
        return result, sleep

    return _run


def _policy(aws, name):
    return next(p for p in aws.called("put_scaling_policy") if p["PolicyName"] == name)


class TestTargetTracking:
    def test_registers_the_variant_and_puts_a_target_tracking_policy(self, run):
        aws = FakeAWS()
        result, _ = run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            target_value=8,
        )
        (register,) = aws.called("register_scalable_target")
        assert register == {
            "ServiceNamespace": "sagemaker",
            "ResourceId": VARIANT_ID,
            "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
            "MinCapacity": 1,
            "MaxCapacity": 4,
            "Tags": {OWNER_TAG_KEY: OWNER_TAG_VALUE},
        }
        (policy,) = aws.called("put_scaling_policy")
        assert policy["PolicyName"] == TRACKING
        assert policy["PolicyType"] == "TargetTrackingScaling"
        assert policy["TargetTrackingScalingPolicyConfiguration"] == {
            "TargetValue": 8.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantConcurrentRequestsPerModelHighResolution"
            },
            "ScaleInCooldown": 300,
            "ScaleOutCooldown": 300,
            "DisableScaleIn": False,
        }
        assert result["scalable_target"]["MaxCapacity"] == 4
        assert aws.called("put_metric_alarm") == []

    def test_a_change_does_not_retag_an_existing_target(self, run):
        aws = FakeAWS(registered=(1, 2), ours=False, policies=(TRACKING,))
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=2,
            max_capacity=6,
            target_value=5,
            metric="invocations",
            scale_in_cooldown=60,
            scale_out_cooldown=30,
            disable_scale_in=True,
        )
        (register,) = aws.called("register_scalable_target")
        assert "Tags" not in register
        config = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert config["PredefinedMetricSpecification"] == {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        }
        assert (
            config["ScaleInCooldown"],
            config["ScaleOutCooldown"],
            config["DisableScaleIn"],
        ) == (
            60,
            30,
            True,
        )

    def test_a_full_predefined_metric_type_is_accepted(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=70,
            metric="SageMakerVariantInvocationsPerInstance",
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["PredefinedMetricSpecification"]["PredefinedMetricType"] == (
            "SageMakerVariantInvocationsPerInstance"
        )

    def test_a_custom_metric_defaults_to_the_targets_dimensions(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=50,
            metric=CustomMetric(
                "CPUUtilization", namespace="/aws/sagemaker/Endpoints", unit="Percent"
            ),
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["CustomizedMetricSpecification"] == {
            "MetricName": "CPUUtilization",
            "Namespace": "/aws/sagemaker/Endpoints",
            "Dimensions": [
                {"Name": "EndpointName", "Value": ENDPOINT},
                {"Name": "VariantName", "Value": "AllTraffic"},
            ],
            "Statistic": "Average",
            "Unit": "Percent",
        }
        assert "PredefinedMetricSpecification" not in spec

    def test_a_metric_math_dict_is_passed_through(self, run):
        math = {
            "Metrics": [
                {
                    "Id": "m1",
                    "MetricStat": {
                        "Metric": {"MetricName": "Invocations", "Namespace": "AWS/SageMaker"},
                        "Stat": "Sum",
                    },
                    "ReturnData": False,
                },
                {"Id": "e1", "Expression": "m1 / 60", "ReturnData": True},
            ]
        }
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=10,
            metric=math,
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["CustomizedMetricSpecification"] is math

    def test_several_target_tracking_policies_need_distinct_names(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            policies=[
                TargetTrackingPolicy(target_value=4),
                TargetTrackingPolicy(
                    target_value=70,
                    metric=CustomMetric("GPUUtilization", namespace="/aws/sagemaker/Endpoints"),
                    name="gpu",
                ),
            ],
        )
        assert [p["PolicyName"] for p in aws.called("put_scaling_policy")] == [
            TRACKING,
            f"{TRACKING}-gpu",
        ]
        aws = FakeAWS()
        with pytest.raises(ValueError, match="distinct name"):
            run(
                aws,
                enable_autoscaling,
                endpoint_name=ENDPOINT,
                min_capacity=1,
                max_capacity=4,
                policies=[
                    TargetTrackingPolicy(target_value=4),
                    TargetTrackingPolicy(target_value=5),
                ],
            )
        assert aws.writes() == []

    def test_names_the_variant_when_the_endpoint_has_several(self, run):
        aws = FakeAWS(variants=("A", "B"))
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            variant_name="B",
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )
        assert aws.called("register_scalable_target")[0]["ResourceId"] == (
            f"endpoint/{ENDPOINT}/variant/B"
        )

    def test_wait_false_returns_without_retrying(self, run):
        aws = FakeAWS()
        _, sleep = run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
            wait=False,
        )
        assert sleep.call_count == 0


class TestStepScaling:
    def test_a_step_policy_gets_its_own_tagged_alarm(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=6,
            policies=[
                StepScalingPolicy(
                    steps=[
                        StepAdjustment(scaling_adjustment=1, lower_bound=0, upper_bound=20),
                        StepAdjustment(scaling_adjustment=3, lower_bound=20),
                    ],
                    alarm=Alarm(
                        metric=CustomMetric("CPUUtilization", namespace="/aws/sagemaker/Endpoints"),
                        threshold=70,
                        evaluation_periods=3,
                        datapoints_to_alarm=2,
                    ),
                    adjustment_type="ChangeInCapacity",
                    cooldown=120,
                    name="cpu",
                )
            ],
        )
        (policy,) = aws.called("put_scaling_policy")
        assert policy["PolicyName"] == "sagemaker-sdk-step-scaling-cpu"
        assert policy["StepScalingPolicyConfiguration"] == {
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Average",
            "Cooldown": 120,
            "StepAdjustments": [
                {
                    "ScalingAdjustment": 1,
                    "MetricIntervalLowerBound": 0,
                    "MetricIntervalUpperBound": 20,
                },
                {"ScalingAdjustment": 3, "MetricIntervalLowerBound": 20},
            ],
        }
        (alarm,) = aws.called("put_metric_alarm")
        assert alarm["AlarmName"] == f"sagemaker-sdk-step-scaling-cpu-{ENDPOINT}-AllTraffic"
        assert alarm["AlarmActions"] == ["arn:policy/sagemaker-sdk-step-scaling-cpu"]
        assert (
            alarm["MetricName"],
            alarm["Threshold"],
            alarm["EvaluationPeriods"],
            alarm["DatapointsToAlarm"],
        ) == ("CPUUtilization", 70, 3, 2)
        assert alarm["Tags"] == OWNER_TAGS

    def test_percent_change_with_a_minimum_magnitude(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=6,
            policies=[
                StepScalingPolicy(
                    steps=[StepAdjustment(scaling_adjustment=50, lower_bound=0)],
                    alarm=Alarm(
                        metric=CustomMetric("Invocations", statistic="Sum", unit="Count"),
                        threshold=1000,
                    ),
                    adjustment_type="PercentChangeInCapacity",
                    min_adjustment_magnitude=2,
                )
            ],
        )
        config = aws.called("put_scaling_policy")[0]["StepScalingPolicyConfiguration"]
        assert aws.called("put_metric_alarm")[0]["Unit"] == "Count"
        assert (config["AdjustmentType"], config["MinAdjustmentMagnitude"]) == (
            "PercentChangeInCapacity",
            2,
        )

    @pytest.mark.parametrize(
        "policy, message",
        [
            (StepScalingPolicy(steps=[], alarm=Alarm(CustomMetric("X"), 1)), "at least one step"),
            (
                StepScalingPolicy(
                    steps=[StepAdjustment(1, 0)], alarm=Alarm(CustomMetric("X"), 1), cooldown=-1
                ),
                "cooldown",
            ),
            (TargetTrackingPolicy(target_value=0), "target_value"),
            (TargetTrackingPolicy(target_value=True), "target_value"),
            (TargetTrackingPolicy(target_value=4, scale_in_cooldown=-1), "scale_in_cooldown"),
            ("not a policy", "policies must be"),
        ],
    )
    def test_bad_policies(self, run, policy, message):
        aws = FakeAWS()
        with pytest.raises(ValueError, match=message):
            run(
                aws,
                enable_autoscaling,
                endpoint_name=ENDPOINT,
                min_capacity=1,
                max_capacity=2,
                policies=[policy],
            )
        assert aws.writes() == []


class TestScheduledActions:
    def test_puts_scheduled_actions_with_their_bounds_and_time_zone(self, run):
        start = datetime(2026, 12, 1)
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            target_value=4,
            scheduled_actions=[
                ScheduledAction(
                    name="morning",
                    schedule="cron(0 8 ? * MON-FRI *)",
                    min_capacity=2,
                    timezone="America/New_York",
                    start_time=start,
                ),
                ScheduledAction(
                    name="night",
                    schedule="cron(0 20 * * ? *)",
                    min_capacity=1,
                    max_capacity=2,
                    end_time=start,
                ),
            ],
        )
        morning, night = aws.called("put_scheduled_action")
        assert morning == {
            "ServiceNamespace": "sagemaker",
            "ResourceId": VARIANT_ID,
            "ScalableDimension": "sagemaker:variant:DesiredInstanceCount",
            "ScheduledActionName": "sagemaker-sdk-morning",
            "Schedule": "cron(0 8 ? * MON-FRI *)",
            "ScalableTargetAction": {"MinCapacity": 2},
            "Timezone": "America/New_York",
            "StartTime": start,
        }
        assert night["ScalableTargetAction"] == {"MinCapacity": 1, "MaxCapacity": 2}
        assert night["EndTime"] == start

    def test_scheduled_actions_alone(self, run):
        aws = FakeAWS()
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            scheduled_actions=[ScheduledAction(name="a", schedule="rate(1 day)", max_capacity=3)],
        )
        assert aws.called("put_scaling_policy") == []
        assert len(aws.called("put_scheduled_action")) == 1

    @pytest.mark.parametrize(
        "action, message",
        [
            (ScheduledAction(name="a", schedule="every day", min_capacity=1), "at\\(...\\)"),
            (ScheduledAction(name="a", schedule="rate(1 day)"), "min_capacity or max_capacity"),
            (ScheduledAction(name="", schedule="rate(1 day)", min_capacity=1), "needs a name"),
            (ScheduledAction(name="a", schedule="rate(1 day)", min_capacity=-1), "MinCapacity"),
            ("not an action", "ScheduledAction objects"),
        ],
    )
    def test_bad_scheduled_actions(self, run, action, message):
        aws = FakeAWS()
        with pytest.raises(ValueError, match=message):
            run(
                aws,
                enable_autoscaling,
                endpoint_name=ENDPOINT,
                min_capacity=1,
                max_capacity=2,
                scheduled_actions=[action],
            )
        assert aws.writes() == []


class TestDeclarativeChanges:
    def test_policies_and_schedules_no_longer_given_are_removed_and_others_stay(self, run):
        aws = FakeAWS(
            registered=(1, 4),
            policies=(TRACKING, "sagemaker-sdk-step-scaling-cpu", "theirs"),
            scheduled_actions=("sagemaker-sdk-old", "their-schedule"),
            alarms={f"sagemaker-sdk-step-scaling-cpu-{ENDPOINT}-AllTraffic": ({}, OWNER_TAGS)},
        )
        result, _ = run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            target_value=4,
        )
        assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [
            "sagemaker-sdk-step-scaling-cpu"
        ]
        assert aws.called("delete_alarms") == [
            {"AlarmNames": [f"sagemaker-sdk-step-scaling-cpu-{ENDPOINT}-AllTraffic"]}
        ]
        assert [d["ScheduledActionName"] for d in aws.called("delete_scheduled_action")] == [
            "sagemaker-sdk-old"
        ]
        assert {p["PolicyName"] for p in result["policies"]} == {TRACKING, "theirs"}
        assert {a["ScheduledActionName"] for a in result["scheduled_actions"]} == {"their-schedule"}

    def test_an_alarm_it_did_not_create_is_left_alone(self, run):
        name = f"sagemaker-sdk-step-scaling-cpu-{ENDPOINT}-AllTraffic"
        aws = FakeAWS(
            registered=(1, 4),
            policies=("sagemaker-sdk-step-scaling-cpu",),
            alarms={name: ({}, [{"Key": "team", "Value": "other"}])},
        )
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=4,
            target_value=4,
        )
        assert aws.called("delete_alarms") == []


class TestInferenceComponents:
    def test_scales_the_copy_count_on_the_component_metric(self, run):
        aws = _ic_aws(components={IC: {}, "adapter-1": {"base": IC}})
        run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=3,
            target_value=4,
        )
        (register,) = aws.called("register_scalable_target")
        assert (register["ResourceId"], register["ScalableDimension"]) == (
            IC_ID,
            "sagemaker:inference-component:DesiredCopyCount",
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["PredefinedMetricSpecification"] == {
            "PredefinedMetricType": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution"
        }

    def test_a_custom_metric_uses_the_component_dimension(self, run):
        aws = _ic_aws()
        run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=3,
            target_value=70,
            metric=CustomMetric("GPUUtilizationNormalized"),
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["CustomizedMetricSpecification"]["Dimensions"] == [
            {"Name": "InferenceComponentName", "Value": IC}
        ]

    def test_scale_to_zero_lowers_the_minimum_only_after_the_alarm_exists(self, run):
        aws = _zero_ic_aws()
        result, _ = run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
        )
        assert [r.get("MinCapacity") for r in aws.called("register_scalable_target")] == [1, 0]
        writes = aws.writes()
        assert writes.index("put_metric_alarm") < len(writes) - 1 - writes[::-1].index(
            "register_scalable_target"
        )
        step = _policy(aws, FROM_ZERO)
        assert step["StepScalingPolicyConfiguration"] == {
            "AdjustmentType": "ChangeInCapacity",
            "MetricAggregationType": "Maximum",
            "Cooldown": 60,
            "StepAdjustments": [{"ScalingAdjustment": 1, "MetricIntervalLowerBound": 0}],
        }
        (alarm,) = aws.called("put_metric_alarm")
        assert alarm["AlarmName"] == IC_FROM_ZERO_ALARM
        assert alarm["AlarmActions"] == [f"arn:policy/{FROM_ZERO}"]
        assert (alarm["Namespace"], alarm["MetricName"], alarm["Statistic"]) == (
            "AWS/SageMaker",
            "NoCapacityInvocationFailures",
            "Sum",
        )
        assert alarm["Dimensions"] == [{"Name": "InferenceComponentName", "Value": IC}]
        assert alarm["TreatMissingData"] == "notBreaching"
        assert result["scalable_target"]["MinCapacity"] == 0

    def test_a_failed_alarm_write_leaves_a_minimum_of_one(self, run):
        aws = _zero_ic_aws()
        aws.cloudwatch.put_metric_alarm.side_effect = RuntimeError("AccessDenied")
        with pytest.raises(RuntimeError):
            run(
                aws,
                enable_autoscaling,
                inference_component_name=IC,
                min_capacity=0,
                max_capacity=2,
                target_value=4,
                scale_to_zero=True,
            )
        assert aws.target["MinCapacity"] == 1

    def test_scale_to_zero_already_at_zero_registers_once(self, run):
        aws = _zero_ic_aws(
            registered=(0, 2),
            policies=(TRACKING, FROM_ZERO),
            alarms={IC_FROM_ZERO_ALARM: ({}, OWNER_TAGS)},
        )
        run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=3,
            target_value=4,
            scale_to_zero=True,
        )
        assert [r["MinCapacity"] for r in aws.called("register_scalable_target")] == [0]

    def test_a_change_away_from_zero_deletes_the_step_policy_and_alarm(self, run):
        aws = _ic_aws(
            registered=(0, 2),
            policies=(TRACKING, FROM_ZERO),
            alarms={IC_FROM_ZERO_ALARM: ({}, OWNER_TAGS)},
        )
        result, _ = run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )
        assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [FROM_ZERO]
        assert aws.called("delete_alarms") == [{"AlarmNames": [IC_FROM_ZERO_ALARM]}]
        assert [p["PolicyName"] for p in result["policies"]] == [TRACKING]


class TestAsynchronousAndServerless:
    def test_an_asynchronous_endpoint_tracks_its_backlog_and_can_reach_zero(self, run):
        aws = FakeAWS(async_config={"OutputConfig": {"S3OutputPath": "s3://b/o"}})
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=0,
            max_capacity=3,
            target_value=5,
        )
        assert aws.called("register_scalable_target")[0]["MinCapacity"] == 0
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["CustomizedMetricSpecification"] == {
            "MetricName": "ApproximateBacklogSizePerInstance",
            "Namespace": "AWS/SageMaker",
            "Dimensions": [{"Name": "EndpointName", "Value": ENDPOINT}],
            "Statistic": "Average",
        }

    def test_asynchronous_scale_to_zero_scales_out_on_backlog_without_capacity(self, run):
        aws = FakeAWS(async_config={"OutputConfig": {"S3OutputPath": "s3://b/o"}})
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=0,
            max_capacity=3,
            target_value=5,
            scale_to_zero=True,
        )
        assert [r["MinCapacity"] for r in aws.called("register_scalable_target")] == [0]
        (alarm,) = aws.called("put_metric_alarm")
        assert (alarm["MetricName"], alarm["EvaluationPeriods"], alarm["DatapointsToAlarm"]) == (
            "HasBacklogWithoutCapacity",
            2,
            2,
        )
        assert alarm["Dimensions"] == [{"Name": "EndpointName", "Value": ENDPOINT}]
        assert _policy(aws, FROM_ZERO)["PolicyType"] == "StepScaling"

    def test_serverless_provisioned_concurrency(self, run):
        aws = FakeAWS(
            serverless={"MemorySizeInMB": 2048, "MaxConcurrency": 20, "ProvisionedConcurrency": 2}
        )
        run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=10,
            target_value=0.5,
        )
        assert aws.called("register_scalable_target")[0]["ScalableDimension"] == (
            "sagemaker:variant:DesiredProvisionedConcurrency"
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["PredefinedMetricSpecification"]["PredefinedMetricType"] == (
            "SageMakerVariantProvisionedConcurrencyUtilization"
        )


def _refuses(run, aws, message, **kwargs):
    with pytest.raises(ValueError, match=message):
        run(aws, enable_autoscaling, **kwargs)
    assert aws.writes() == []


class TestRefusesBeforeWriting:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(min_capacity=0, max_capacity=2, target_value=4), "min_capacity must be"),
            (dict(min_capacity=3, max_capacity=2, target_value=4), "must not exceed"),
            (dict(min_capacity=True, max_capacity=2, target_value=4), "min_capacity"),
            (dict(min_capacity=1, max_capacity=0, target_value=4), "max_capacity"),
            (dict(min_capacity=1, max_capacity=2, target_value=4, metric="cpu"), "does not apply"),
            (
                dict(min_capacity=1, max_capacity=2, target_value=4, metric="backlog-per-instance"),
                "asynchronous endpoint",
            ),
            (
                dict(min_capacity=0, max_capacity=2, target_value=4, scale_to_zero=True),
                "real-time production variant",
            ),
            (dict(min_capacity=1, max_capacity=2), "Pass target_value"),
            (
                dict(
                    min_capacity=1,
                    max_capacity=2,
                    target_value=4,
                    policies=[TargetTrackingPolicy(target_value=4)],
                ),
                "not both",
            ),
        ],
    )
    def test_bad_arguments(self, run, kwargs, message):
        _refuses(run, FakeAWS(), message, endpoint_name=ENDPOINT, **kwargs)

    def test_no_target_named(self, run):
        _refuses(
            run, FakeAWS(), "Pass endpoint_name", min_capacity=1, max_capacity=2, target_value=4
        )

    @pytest.mark.parametrize(
        "aws_kwargs, message",
        [
            (dict(status="Updating"), "not InService"),
            (dict(serverless={"MaxConcurrency": 5}), "without provisioned concurrency"),
            (dict(variants=("A", "B")), "pass variant_name"),
            (dict(model_name=None), "pass inference_component_name"),
        ],
    )
    def test_endpoints_that_cannot_scale_this_way(self, run, aws_kwargs, message):
        _refuses(
            run,
            FakeAWS(**aws_kwargs),
            message,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )

    def test_provisioned_concurrency_does_not_scale_to_zero(self, run):
        aws = FakeAWS(serverless={"MaxConcurrency": 5, "ProvisionedConcurrency": 1})
        _refuses(
            run,
            aws,
            "does not scale to zero",
            endpoint_name=ENDPOINT,
            min_capacity=0,
            max_capacity=2,
            target_value=0.5,
            scale_to_zero=True,
        )

    def test_an_unknown_variant(self, run):
        _refuses(
            run,
            FakeAWS(),
            "no variant 'B'",
            endpoint_name=ENDPOINT,
            variant_name="B",
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )

    def test_a_component_without_managed_instance_scaling(self, run):
        _refuses(
            run,
            _ic_aws(managed={"Status": "DISABLED"}),
            "no managed instance scaling",
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )

    def test_an_adapter_names_its_base(self, run):
        aws = _ic_aws(components={IC: {}, "adapter-1": {"base": IC}})
        _refuses(
            run,
            aws,
            f"inference_component_name='{IC}'",
            inference_component_name="adapter-1",
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )

    def test_a_component_on_another_endpoint(self, run):
        _refuses(
            run,
            _ic_aws(components={IC: {"endpoint": "other"}}),
            "not my-endpoint",
            endpoint_name=ENDPOINT,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )

    def test_invocations_on_a_base_with_adapters(self, run):
        aws = _ic_aws(components={IC: {}, "adapter-1": {"base": IC}})
        _refuses(
            run,
            aws,
            "concurrent-requests",
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
            metric="invocations",
        )

    def test_scale_to_zero_on_a_component(self, run):
        _refuses(
            run,
            _ic_aws(),
            "MinInstanceCount 0",
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
        )
        _refuses(
            run,
            _zero_ic_aws(),
            "min_capacity=0",
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
        )
        aws = _zero_ic_aws(components={IC: {}, "adapter-1": {"base": IC}})
        _refuses(
            run,
            aws,
            "adapters",
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
        )

    def test_a_component_without_scale_to_zero_keeps_a_minimum_of_one(self, run):
        _refuses(
            run,
            _zero_ic_aws(),
            "min_capacity must be",
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
        )


class TestDisable:
    def test_deregisters_a_target_it_registered_and_deletes_its_alarms(self, run):
        aws = _ic_aws(
            registered=(0, 2),
            policies=(TRACKING, FROM_ZERO),
            scheduled_actions=("sagemaker-sdk-night",),
            alarms={IC_FROM_ZERO_ALARM: ({}, OWNER_TAGS)},
        )
        result, _ = run(aws, disable_autoscaling, inference_component_name=IC)
        assert len(aws.called("deregister_scalable_target")) == 1
        assert aws.called("delete_alarms") == [{"AlarmNames": [IC_FROM_ZERO_ALARM]}]
        assert result["deregistered"] is True
        assert result["deleted_policies"] == sorted([TRACKING, FROM_ZERO])
        assert result["deleted_scheduled_actions"] == ["sagemaker-sdk-night"]
        assert result["remaining"]["scalable_target"] is None

    def test_a_target_it_did_not_register_stays(self, run):
        aws = FakeAWS(
            registered=(1, 2),
            ours=False,
            policies=(TRACKING,),
            scheduled_actions=("sagemaker-sdk-night",),
        )
        result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
        assert aws.called("deregister_scalable_target") == []
        assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [TRACKING]
        assert [d["ScheduledActionName"] for d in aws.called("delete_scheduled_action")] == [
            "sagemaker-sdk-night"
        ]
        assert result["remaining"]["scalable_target"] is not None

    def test_other_policies_and_scheduled_actions_keep_the_target(self, run):
        for extra in (
            dict(policies=(TRACKING, "theirs")),
            dict(policies=(TRACKING,), scheduled_actions=("nightly",)),
        ):
            aws = FakeAWS(registered=(1, 2), **extra)
            result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
            assert aws.called("deregister_scalable_target") == []
            assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [TRACKING]
            assert result["deregistered"] is False

    def test_runs_while_the_endpoint_updates_or_after_managed_scaling_is_off(self, run):
        aws = _ic_aws(status="Updating", managed=None, registered=(1, 2), policies=(TRACKING,))
        result, _ = run(aws, disable_autoscaling, inference_component_name=IC)
        assert result["deregistered"] is True

    def test_nothing_registered_changes_nothing(self, run):
        aws = FakeAWS()
        result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
        assert aws.writes() == []
        assert (result["deregistered"], result["deleted_policies"]) == (False, [])

    def test_warns_when_a_target_has_nothing_it_created(self, run):
        aws = FakeAWS(registered=(1, 2), ours=False, policies=("theirs",))
        with patch.object(autoscaling.logger, "warning") as warning:
            result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT, wait=False)
        assert aws.writes() == []
        assert "nothing was deleted" in warning.call_args[0][0]
        assert [p["PolicyName"] for p in result["remaining"]["policies"]] == ["theirs"]


class TestSuspendResume:
    def test_suspend_and_resume_set_the_suspended_state(self, run):
        aws = FakeAWS(registered=(1, 2), policies=(TRACKING,))
        result, _ = run(aws, suspend_autoscaling, endpoint_name=ENDPOINT, scheduled=False)
        assert aws.called("register_scalable_target")[-1]["SuspendedState"] == {
            "DynamicScalingInSuspended": True,
            "DynamicScalingOutSuspended": True,
            "ScheduledScalingSuspended": False,
        }
        assert "MinCapacity" not in aws.called("register_scalable_target")[-1]
        assert result["scalable_target"]["SuspendedState"]["DynamicScalingInSuspended"] is True
        result, _ = run(aws, resume_autoscaling, endpoint_name=ENDPOINT)
        assert set(result["scalable_target"]["SuspendedState"].values()) == {False}

    def test_an_unregistered_target_cannot_be_suspended(self, run):
        aws = FakeAWS()
        with pytest.raises(ValueError, match="not registered"):
            run(aws, suspend_autoscaling, endpoint_name=ENDPOINT)
        assert aws.writes() == []


class TestReadBack:
    def test_retries_until_the_write_reads_back(self, run):
        aws = FakeAWS()
        reads = {"n": 0}
        real = aws.autoscaling.describe_scalable_targets.side_effect

        def lagging(*args, **kwargs):
            reads["n"] += 1
            # The first read is before the write; the next two lag behind it.
            return {"ScalableTargets": []} if 1 < reads["n"] <= 3 else real(*args, **kwargs)

        aws.autoscaling.describe_scalable_targets.side_effect = lagging
        result, sleep = run(
            aws,
            enable_autoscaling,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )
        assert sleep.call_count == 2
        assert result["scalable_target"] is not None

    def test_warns_when_the_write_never_reads_back(self, run):
        aws = FakeAWS()
        aws.autoscaling.describe_scalable_targets.side_effect = lambda **kw: {"ScalableTargets": []}
        with patch.object(autoscaling.logger, "warning") as warning:
            result, sleep = run(
                aws,
                enable_autoscaling,
                endpoint_name=ENDPOINT,
                min_capacity=1,
                max_capacity=2,
                target_value=4,
            )
        assert sleep.call_count == autoscaling._READ_BACK_ATTEMPTS - 1
        assert "does not read back yet" in warning.call_args[0][0]
        assert result["scalable_target"] is None


def test_describe_returns_target_policies_and_scheduled_actions(run):
    aws = FakeAWS(registered=(1, 3), policies=(TRACKING,), scheduled_actions=("nightly",))
    result, _ = run(aws, describe_autoscaling, endpoint_name=ENDPOINT)
    assert result["resource_id"] == VARIANT_ID
    assert result["scalable_target"]["MaxCapacity"] == 3
    assert [p["PolicyName"] for p in result["policies"]] == [TRACKING]
    assert result["scheduled_actions"] == [{"ScheduledActionName": "nightly"}]
    assert aws.writes() == []


def test_a_target_without_an_arn_is_not_treated_as_its_own():
    assert autoscaling._is_ours(Mock(), {"MinCapacity": 1}) is False


def test_clients_come_from_the_session_region():
    session = Mock(boto_region_name="eu-west-1")
    clients = autoscaling._Clients(session)
    assert clients.sagemaker is session.sagemaker_client
    session.boto_session.client.assert_any_call("application-autoscaling", region_name="eu-west-1")
    session.boto_session.client.assert_any_call("cloudwatch", region_name="eu-west-1")
