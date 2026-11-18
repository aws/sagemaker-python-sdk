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

from unittest.mock import Mock, patch

import pytest

from sagemaker.serve import autoscaling
from sagemaker.serve.autoscaling import (
    OWNER_TAG_KEY,
    OWNER_TAG_VALUE,
    describe_autoscaling,
    disable_autoscaling,
    enable_autoscaling,
)

ENDPOINT = "my-endpoint"
VARIANT_ID = f"endpoint/{ENDPOINT}/variant/AllTraffic"
VARIANT_POLICY = f"{ENDPOINT}-AllTraffic-target-tracking"
IC = "my-ic"
IC_ID = f"inference-component/{IC}"
IC_POLICY = f"{IC}-target-tracking"
IC_STEP_POLICY = f"{IC}-scale-out-from-zero"
IC_ALARM = f"{IC}-no-capacity-invocations"
WRITES = {
    "register_scalable_target",
    "deregister_scalable_target",
    "put_scaling_policy",
    "delete_scaling_policy",
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
        # {name: {"base": ..., "endpoint": ...}}
        self.components = components or {}
        self.target = None
        self.tags = {}
        if registered is not None:
            self.target = {"MinCapacity": registered[0], "MaxCapacity": registered[1]}
            self.tags = {OWNER_TAG_KEY: OWNER_TAG_VALUE} if ours else {}
        self.policies = {
            name: {"PolicyName": name, "PolicyType": "TargetTrackingScaling"} for name in policies
        }
        self.scheduled = [{"ScheduledActionName": n} for n in scheduled_actions]
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
        ):
            self._record(aas, name, getattr(self, "_" + name))
        for name in ("describe_alarms", "put_metric_alarm", "delete_alarms"):
            self._record(cw, name, getattr(self, "_" + name))

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
    def _describe_scalable_targets(self, **kwargs):
        if not self.target:
            return {"ScalableTargets": []}
        return {"ScalableTargets": [dict(self.target, ScalableTargetARN="arn:target")]}

    def _describe_scaling_policies(self, **kwargs):
        return {"ScalingPolicies": [dict(p) for p in self.policies.values()]}

    def _describe_scheduled_actions(self, **kwargs):
        return {"ScheduledActions": list(self.scheduled)}

    def _list_tags_for_resource(self, ResourceARN):
        return {"Tags": dict(self.tags)}

    def _register_scalable_target(self, **kwargs):
        if not self.target:
            self.tags = dict(kwargs.get("Tags", {}))
        self.target = {"MinCapacity": kwargs["MinCapacity"], "MaxCapacity": kwargs["MaxCapacity"]}

    def _deregister_scalable_target(self, **kwargs):
        self.target, self.tags, self.policies = None, {}, {}

    def _put_scaling_policy(self, **kwargs):
        arn = f"arn:policy/{kwargs['PolicyName']}"
        self.policies[kwargs["PolicyName"]] = {
            "PolicyName": kwargs["PolicyName"],
            "PolicyType": kwargs["PolicyType"],
            "PolicyARN": arn,
        }
        return {"PolicyARN": arn}

    def _delete_scaling_policy(self, **kwargs):
        self.policies.pop(kwargs["PolicyName"], None)

    # CloudWatch
    def _describe_alarms(self, AlarmNames):
        return {"MetricAlarms": [dict(self.alarms[n]) for n in AlarmNames if n in self.alarms]}

    def _put_metric_alarm(self, **kwargs):
        self.alarms[kwargs["AlarmName"]] = {
            k: kwargs[k] for k in ("AlarmName", "MetricName", "Dimensions")
        }

    def _delete_alarms(self, AlarmNames):
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


def _our_alarm(component=IC):
    return {
        IC_ALARM: {
            "AlarmName": IC_ALARM,
            "MetricName": "NoCapacityInvocationFailures",
            "Dimensions": [{"Name": "InferenceComponentName", "Value": component}],
        }
    }


@pytest.fixture
def run():
    def _run(aws, func, **kwargs):
        with patch.object(autoscaling, "_Clients", return_value=aws), patch.object(
            autoscaling.time, "sleep"
        ) as sleep:
            result = func(**kwargs)
        return result, sleep

    return _run


class TestEnableVariant:
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
        assert policy["PolicyName"] == VARIANT_POLICY
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
        assert [p["PolicyName"] for p in result["policies"]] == [VARIANT_POLICY]
        assert aws.called("put_metric_alarm") == []

    def test_a_change_does_not_retag_an_existing_target(self, run):
        aws = FakeAWS(registered=(1, 2), ours=False, policies=(VARIANT_POLICY,))
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
        ) == (60, 30, True)

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


class TestEnableInferenceComponent:
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
        (policy,) = aws.called("put_scaling_policy")
        assert policy["PolicyName"] == IC_POLICY
        assert policy["TargetTrackingScalingPolicyConfiguration"][
            "PredefinedMetricSpecification"
        ] == {
            "PredefinedMetricType": "SageMakerInferenceComponentConcurrentRequestsPerCopyHighResolution"
        }

    def test_invocations_per_copy(self, run):
        aws = _ic_aws()
        run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=3,
            target_value=10,
            metric="invocations",
        )
        spec = aws.called("put_scaling_policy")[0]["TargetTrackingScalingPolicyConfiguration"]
        assert spec["PredefinedMetricSpecification"] == {
            "PredefinedMetricType": "SageMakerInferenceComponentInvocationsPerCopy"
        }

    def test_scale_to_zero_lowers_the_minimum_only_after_the_alarm_exists(self, run):
        aws = _ic_aws(managed={"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2})
        result, _ = run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=0,
            max_capacity=2,
            target_value=4,
            scale_to_zero=True,
        )
        assert [r["MinCapacity"] for r in aws.called("register_scalable_target")] == [1, 0]
        writes = aws.writes()
        assert writes.index("put_metric_alarm") < len(writes) - 1 - writes[::-1].index(
            "register_scalable_target"
        )
        target_tracking, step = aws.called("put_scaling_policy")
        assert (target_tracking["PolicyName"], step["PolicyName"]) == (IC_POLICY, IC_STEP_POLICY)
        assert step["StepScalingPolicyConfiguration"]["StepAdjustments"] == [
            {"MetricIntervalLowerBound": 0, "ScalingAdjustment": 1}
        ]
        (alarm,) = aws.called("put_metric_alarm")
        assert alarm["AlarmName"] == IC_ALARM
        assert alarm["AlarmActions"] == [f"arn:policy/{IC_STEP_POLICY}"]
        assert (alarm["Namespace"], alarm["MetricName"]) == (
            "AWS/SageMaker",
            "NoCapacityInvocationFailures",
        )
        assert alarm["Dimensions"] == [{"Name": "InferenceComponentName", "Value": IC}]
        assert alarm["Tags"] == [{"Key": OWNER_TAG_KEY, "Value": OWNER_TAG_VALUE}]
        assert result["scalable_target"]["MinCapacity"] == 0

    def test_a_failed_alarm_write_leaves_a_minimum_of_one(self, run):
        aws = _ic_aws(managed={"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2})
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
        aws = _ic_aws(
            managed={"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2},
            registered=(0, 2),
            policies=(IC_POLICY, IC_STEP_POLICY),
            alarms=_our_alarm(),
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
        aws = _ic_aws(registered=(0, 2), policies=(IC_POLICY, IC_STEP_POLICY), alarms=_our_alarm())
        result, _ = run(
            aws,
            enable_autoscaling,
            inference_component_name=IC,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
        )
        assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [IC_STEP_POLICY]
        assert aws.called("delete_alarms") == [{"AlarmNames": [IC_ALARM]}]
        assert [p["PolicyName"] for p in result["policies"]] == [IC_POLICY]


def _refuses(run, aws, message, **kwargs):
    with pytest.raises(ValueError, match=message):
        run(aws, enable_autoscaling, **kwargs)
    assert aws.writes() == []


class TestEnableRefusesBeforeWriting:
    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (dict(min_capacity=0, max_capacity=2, target_value=4), "at least 1"),
            (dict(min_capacity=3, max_capacity=2, target_value=4), "at least min_capacity"),
            (dict(min_capacity=True, max_capacity=2, target_value=4), "integer"),
            (dict(min_capacity=1, max_capacity=2, target_value=0), "greater than 0"),
            (dict(min_capacity=1, max_capacity=2, target_value=True), "number"),
            (
                dict(min_capacity=1, max_capacity=2, target_value=4, scale_in_cooldown=-1),
                "scale_in_cooldown",
            ),
            (dict(min_capacity=1, max_capacity=2, target_value=4, metric="cpu"), "metric must be"),
            (
                dict(min_capacity=1, max_capacity=2, target_value=4, scale_to_zero=True),
                "min_capacity=0",
            ),
            (
                dict(min_capacity=0, max_capacity=2, target_value=4, scale_to_zero=True),
                "inference component",
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
            (dict(async_config={"OutputConfig": {}}), "asynchronous"),
            (dict(serverless={"MaxConcurrency": 5}), "serverless"),
            (dict(variants=("A", "B")), "pass variant_name"),
            (dict(model_name=None), "pass inference_component_name"),
        ],
    )
    def test_endpoints_with_no_variant_scaling(self, run, aws_kwargs, message):
        _refuses(
            run,
            FakeAWS(**aws_kwargs),
            message,
            endpoint_name=ENDPOINT,
            min_capacity=1,
            max_capacity=2,
            target_value=4,
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

    def test_scale_to_zero_without_a_managed_minimum_of_zero(self, run):
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

    def test_scale_to_zero_on_a_base_with_adapters(self, run):
        aws = _ic_aws(
            managed={"Status": "ENABLED", "MinInstanceCount": 0, "MaxInstanceCount": 2},
            components={IC: {}, "adapter-1": {"base": IC}},
        )
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


class TestDisable:
    def test_deregisters_a_target_it_registered_and_deletes_the_alarm(self, run):
        aws = _ic_aws(registered=(0, 2), policies=(IC_POLICY, IC_STEP_POLICY), alarms=_our_alarm())
        result, _ = run(aws, disable_autoscaling, inference_component_name=IC)
        assert len(aws.called("deregister_scalable_target")) == 1
        assert aws.called("delete_alarms") == [{"AlarmNames": [IC_ALARM]}]
        assert result["deregistered"] is True
        assert sorted(result["deleted_policies"]) == sorted([IC_POLICY, IC_STEP_POLICY])
        assert result["remaining"]["scalable_target"] is None

    def test_a_target_it_did_not_register_stays(self, run):
        aws = FakeAWS(registered=(1, 2), ours=False, policies=(VARIANT_POLICY,))
        result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
        assert aws.called("deregister_scalable_target") == []
        assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [VARIANT_POLICY]
        assert result["remaining"]["scalable_target"] is not None

    def test_other_policies_and_scheduled_actions_keep_the_target(self, run):
        for extra in (
            dict(policies=(VARIANT_POLICY, "theirs")),
            dict(policies=(VARIANT_POLICY,), scheduled_actions=("nightly",)),
        ):
            aws = FakeAWS(registered=(1, 2), **extra)
            result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
            assert aws.called("deregister_scalable_target") == []
            assert [d["PolicyName"] for d in aws.called("delete_scaling_policy")] == [
                VARIANT_POLICY
            ]
            assert result["deregistered"] is False

    def test_a_variant_makes_no_cloudwatch_call(self, run):
        aws = FakeAWS(registered=(1, 2), policies=(VARIANT_POLICY,))
        run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
        assert aws.called("describe_alarms") == []

    def test_an_alarm_of_the_same_name_on_another_component_stays(self, run):
        aws = _ic_aws(registered=(0, 2), policies=(IC_POLICY,), alarms=_our_alarm("someone-else"))
        run(aws, disable_autoscaling, inference_component_name=IC)
        assert aws.called("delete_alarms") == []

    def test_runs_while_the_endpoint_updates_or_after_managed_scaling_is_off(self, run):
        aws = _ic_aws(status="Updating", managed=None, registered=(1, 2), policies=(IC_POLICY,))
        result, _ = run(aws, disable_autoscaling, inference_component_name=IC)
        assert result["deregistered"] is True

    def test_nothing_registered_changes_nothing(self, run):
        aws = FakeAWS()
        result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT)
        assert aws.writes() == []
        assert result == {
            "deregistered": False,
            "deleted_policies": [],
            "remaining": result["remaining"],
        }


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
    aws = FakeAWS(registered=(1, 3), policies=(VARIANT_POLICY,), scheduled_actions=("nightly",))
    result, _ = run(aws, describe_autoscaling, endpoint_name=ENDPOINT)
    assert result["resource_id"] == VARIANT_ID
    assert result["scalable_target"]["MaxCapacity"] == 3
    assert [p["PolicyName"] for p in result["policies"]] == [VARIANT_POLICY]
    assert result["scheduled_actions"] == [{"ScheduledActionName": "nightly"}]
    assert aws.writes() == []


def test_clients_come_from_the_session_region():
    session = Mock(boto_region_name="eu-west-1")
    clients = autoscaling._Clients(session)
    assert clients.sagemaker is session.sagemaker_client
    session.boto_session.client.assert_any_call("application-autoscaling", region_name="eu-west-1")
    session.boto_session.client.assert_any_call("cloudwatch", region_name="eu-west-1")


def test_disable_warns_when_a_target_has_none_of_its_policies(run):
    aws = FakeAWS(registered=(1, 2), ours=False, policies=("theirs",))
    with patch.object(autoscaling.logger, "warning") as warning:
        result, _ = run(aws, disable_autoscaling, endpoint_name=ENDPOINT, wait=False)
    assert aws.writes() == []
    assert "nothing was deleted" in warning.call_args[0][0]
    assert [p["PolicyName"] for p in result["remaining"]["policies"]] == ["theirs"]


def test_a_target_without_an_arn_is_not_treated_as_its_own():
    assert autoscaling._is_ours(Mock(), {"MinCapacity": 1}) is False
