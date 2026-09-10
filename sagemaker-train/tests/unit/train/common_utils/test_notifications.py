"""Unit tests for job notifications module."""

from __future__ import absolute_import

from unittest.mock import MagicMock, patch

import pytest

from sagemaker.train.common_utils.notifications import (
    _build_event_pattern,
    _get_rule_name,
    _normalize_events,
    delete_notification_rule,
    enable_notifications,
    list_notification_rules,
)


class TestRuleNaming:

    def test_deterministic_from_same_config(self):
        """Same config always produces the same rule name."""
        arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
        events = ["Completed", "Failed"]
        assert _get_rule_name(arn, events) == _get_rule_name(arn, events)

    def test_different_arns_different_names(self):
        """Different ARNs produce different rule names."""
        events = ["Completed", "Failed"]
        arn1 = "arn:aws:sns:us-east-1:123456789012:topic-a"
        arn2 = "arn:aws:sns:us-east-1:123456789012:topic-b"
        assert _get_rule_name(arn1, events) != _get_rule_name(arn2, events)

    def test_same_topic_different_events_different_names(self):
        """Same topic with different events produces different rule names."""
        arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
        assert _get_rule_name(arn, ["Completed"]) != _get_rule_name(arn, ["Completed", "Failed"])

    def test_same_topic_different_prefix_different_names(self):
        """Same topic with different job_name_prefix produces different rule names."""
        arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
        events = ["Completed", "Failed"]
        assert _get_rule_name(arn, events, "team-a-") != _get_rule_name(arn, events, "team-b-")

    def test_same_topic_same_prefix_same_name(self):
        """Same topic + same events + same prefix = same rule (idempotent)."""
        arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
        events = ["Completed", "Failed"]
        assert _get_rule_name(arn, events, "my-prefix-") == _get_rule_name(arn, events, "my-prefix-")

    def test_event_order_does_not_matter(self):
        """Events are sorted internally, so order doesn't affect the hash."""
        arn = "arn:aws:sns:us-east-1:123456789012:my-topic"
        assert _get_rule_name(arn, ["Failed", "Completed"]) == _get_rule_name(arn, ["Completed", "Failed"])

    def test_prefix_present(self):
        """Rule name starts with the SDK prefix."""
        name = _get_rule_name("arn:aws:sns:us-east-1:123456789012:topic", ["Completed"])
        assert name.startswith("sm-pysdk-job-notif-")

    def test_no_prefix_vs_none_prefix_same(self):
        """No prefix and None prefix produce the same rule."""
        arn = "arn:aws:sns:us-east-1:123456789012:topic"
        events = ["Completed"]
        assert _get_rule_name(arn, events) == _get_rule_name(arn, events, None)


class TestNormalizeEvents:

    def test_defaults_to_all_terminal(self):
        assert _normalize_events(None) == ["Completed", "Failed", "Stopped"]

    def test_capitalizes_input(self):
        assert _normalize_events(["completed", "failed"]) == ["Completed", "Failed"]

    def test_handles_inprogress(self):
        assert _normalize_events(["inprogress"]) == ["InProgress"]

    def test_invalid_event_raises(self):
        with pytest.raises(ValueError, match="Invalid notification event"):
            _normalize_events(["invalid_status"])


class TestBuildEventPattern:

    def test_basic_pattern(self):
        import json
        pattern = json.loads(_build_event_pattern(["Completed", "Failed"]))

        assert pattern["source"] == ["aws.sagemaker"]
        assert pattern["detail-type"] == ["SageMaker Training Job State Change"]
        assert pattern["detail"]["TrainingJobStatus"] == ["Completed", "Failed"]
        assert "TrainingJobName" not in pattern["detail"]

    def test_with_job_name_prefix(self):
        import json
        pattern = json.loads(_build_event_pattern(["Completed"], job_name_prefix="my-team-"))

        assert pattern["detail"]["TrainingJobName"] == [{"prefix": "my-team-"}]


class TestEnableNotifications:

    def _session(self):
        s = MagicMock()
        s.boto_session.region_name = "us-east-1"
        return s

    def test_invalid_arn_raises(self):
        with pytest.raises(ValueError, match="Invalid SNS topic ARN"):
            enable_notifications("not-an-arn", self._session())

    def test_creates_rule_and_target(self):
        session = self._session()
        events_client = MagicMock()
        sns_client = MagicMock()
        session.boto_session.client.side_effect = lambda svc, **kw: (
            events_client if svc == "events" else sns_client
        )

        events_client.put_rule.return_value = {"RuleArn": "arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-notif-abc"}
        events_client.put_targets.return_value = {"FailedEntryCount": 0}
        events_client.list_rules.return_value = {"Rules": []}

        arn = enable_notifications(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sagemaker_session=session,
        )

        assert arn == "arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-notif-abc"
        events_client.put_rule.assert_called_once()
        events_client.put_targets.assert_called_once()

    def test_with_custom_events_and_prefix(self):
        session = self._session()
        events_client = MagicMock()
        sns_client = MagicMock()
        session.boto_session.client.side_effect = lambda svc, **kw: (
            events_client if svc == "events" else sns_client
        )

        events_client.put_rule.return_value = {"RuleArn": "arn:rule"}
        events_client.put_targets.return_value = {"FailedEntryCount": 0}
        events_client.list_rules.return_value = {"Rules": []}

        enable_notifications(
            sns_topic_arn="arn:aws:sns:us-east-1:123456789012:my-topic",
            sagemaker_session=session,
            events=["completed"],
            job_name_prefix="ealynnh-",
        )

        call_kwargs = events_client.put_rule.call_args[1]
        import json
        pattern = json.loads(call_kwargs["EventPattern"])
        assert pattern["detail"]["TrainingJobStatus"] == ["Completed"]
        assert pattern["detail"]["TrainingJobName"] == [{"prefix": "ealynnh-"}]


class TestDeleteNotificationRules:

    def test_deletes_specific_rule(self):
        session = MagicMock()
        session.boto_session.region_name = "us-east-1"
        events_client = MagicMock()
        session.boto_session.client.return_value = events_client

        events_client.list_targets_by_rule.return_value = {
            "Targets": [{"Id": "target-1"}]
        }

        deleted = delete_notification_rule(
            sagemaker_session=session,
            rule_arn="arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-job-notif-abc123",
        )

        assert deleted == "sm-pysdk-job-notif-abc123"
        events_client.remove_targets.assert_called_once()
        events_client.delete_rule.assert_called_once()


class TestBaseTrainerNotifications:

    def _make_trainer(self, compute=None):
        from sagemaker.train.base_trainer import BaseTrainer

        class _StubTrainer(BaseTrainer):
            _customization_technique = "SFT"
            def train(self, *args, **kwargs):
                pass

        trainer = _StubTrainer.__new__(_StubTrainer)
        trainer.compute = compute
        trainer.sagemaker_session = None
        trainer._latest_training_job = None
        return trainer

    def test_hyperpod_raises_not_implemented(self):
        from sagemaker.core.training.configs import HyperPodCompute
        trainer = self._make_trainer(
            compute=HyperPodCompute(cluster_name="c", instance_type="ml.p5.48xlarge")
        )

        with pytest.raises(NotImplementedError, match="not supported for HyperPod"):
            trainer._setup_notifications({"sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:topic"})

    def test_missing_sns_arn_raises(self):
        trainer = self._make_trainer()

        with pytest.raises(ValueError, match="requires 'sns_topic_arn'"):
            trainer._setup_notifications({"events": ["completed"]})

    def test_invalid_config_type_raises(self):
        trainer = self._make_trainer()

        with pytest.raises(ValueError, match="must be a dict"):
            trainer._setup_notifications("not-a-dict")


class TestListNotificationRules:

    def test_lists_sdk_rules(self):
        session = MagicMock()
        session.boto_session.region_name = "us-east-1"
        events_client = MagicMock()
        session.boto_session.client.return_value = events_client

        paginator = MagicMock()
        paginator.paginate.return_value = [
            {"Rules": [
                {"Name": "sm-pysdk-job-notif-aaa", "Arn": "arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-job-notif-aaa", "State": "ENABLED"},
                {"Name": "sm-pysdk-job-notif-bbb", "Arn": "arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-job-notif-bbb", "State": "ENABLED"},
            ]}
        ]
        events_client.get_paginator.return_value = paginator

        rules = list_notification_rules(sagemaker_session=session)

        assert len(rules) == 2
        assert rules[0]["name"] == "sm-pysdk-job-notif-aaa"
        assert rules[0]["arn"] == "arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-job-notif-aaa"
