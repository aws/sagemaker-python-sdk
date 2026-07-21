"""Job notification utilities for SageMaker training jobs.

Manages EventBridge rules that route SageMaker Training Job status change
events to user-provided SNS topics. Supports SMTJ (serverless and serverful)
training jobs only.
"""

from __future__ import absolute_import

import hashlib
import json
import logging
import re
from typing import Dict, List, Optional
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# Rule name prefix used to identify SDK-created rules
_RULE_NAME_PREFIX = "sm-pysdk-job-notif"
_DEFAULT_EVENTS = ["Completed", "Failed", "Stopped"]
_VALID_EVENTS = {"Completed", "Failed", "Stopped", "InProgress"}


def _get_rule_name(sns_topic_arn: str, events: List[str], job_name_prefix: Optional[str] = None) -> str:
    """Generate a deterministic rule name from the full notification config.

    Hashes the topic ARN + events + prefix so that:
    - Identical configs -> same rule name
    - Any config difference -> different rule name

    Args:
        sns_topic_arn: The SNS topic ARN.
        events: Normalized list of event statuses.
        job_name_prefix: Optional job name prefix filter.

    Returns:
        Rule name like "sm-pysdk-job-notif-a3f8b2c1".
    """
    config_str = f"{sns_topic_arn}|{','.join(sorted(events))}|{job_name_prefix or ''}"
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    return f"{_RULE_NAME_PREFIX}-{config_hash}"


def _normalize_events(events: Optional[List[str]]) -> List[str]:
    """Normalize and validate event status values.

    Args:
        events: List of event names (e.g., ["completed", "failed"]).
            If None, returns all default events.

    Returns:
        List of capitalized event status strings.

    Raises:
        ValueError: If an invalid event name is provided.
    """
    if not events:
        return _DEFAULT_EVENTS.copy()

    normalized = []
    for event in events:
        capitalized = event.capitalize()
        if capitalized == "Inprogress":
            capitalized = "InProgress"
        if capitalized not in _VALID_EVENTS:
            raise ValueError(
                f"Invalid notification event: '{event}'. "
                f"Valid events: {sorted(_VALID_EVENTS)}"
            )
        normalized.append(capitalized)

    return normalized


def _build_event_pattern(
    events: List[str],
    job_name_prefix: Optional[str] = None,
) -> str:
    """Build the EventBridge event pattern JSON for SMTJ job status changes.

    Args:
        events: List of TrainingJobStatus values to match.
        job_name_prefix: Optional job name prefix filter.

    Returns:
        JSON string of the event pattern.
    """
    pattern: Dict = {
        "source": ["aws.sagemaker"],
        "detail-type": ["SageMaker Training Job State Change"],
        "detail": {
            "TrainingJobStatus": events,
        },
    }

    if job_name_prefix:
        pattern["detail"]["TrainingJobName"] = [{"prefix": job_name_prefix}]

    return json.dumps(pattern)


def _validate_notifications_permissions(events_client) -> None:
    """Validate the caller has permissions to manage EventBridge rules.

    Args:
        events_client: boto3 EventBridge client.

    Raises:
        PermissionError: If the caller lacks required EventBridge permissions.
    """
    try:
        events_client.list_rules(NamePrefix=_RULE_NAME_PREFIX, Limit=1)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        if error_code in ("AccessDeniedException", "AccessDenied"):
            raise PermissionError(
                "Missing permissions to manage EventBridge rules. "
                "Ensure your caller identity has: "
                "events:PutRule, events:PutTargets, events:ListRules, "
                "events:RemoveTargets, events:DeleteRule."
            ) from e
        raise


def _validate_sns_topic(sns_client, topic_arn: str) -> None:
    """Validate that the SNS topic exists and is accessible.

    Args:
        sns_client: boto3 SNS client.
        topic_arn: The SNS topic ARN to validate.

    Raises:
        ValueError: If the topic doesn't exist or isn't accessible.
    """
    try:
        sns_client.get_topic_attributes(TopicArn=topic_arn)
    except Exception as e:
        error_msg = str(e)
        if "NotFound" in error_msg or "does not exist" in error_msg.lower():
            raise ValueError(
                f"SNS topic not found: {topic_arn}. "
                "Ensure the topic exists and you have sns:GetTopicAttributes permission."
            ) from e
        if "AuthorizationError" in error_msg or "AccessDenied" in error_msg:
            raise PermissionError(
                f"Cannot access SNS topic: {topic_arn}. "
                "Ensure you have sns:GetTopicAttributes permission."
            ) from e
        raise


def enable_notifications(
    sns_topic_arn: str,
    sagemaker_session,
    events: Optional[List[str]] = None,
    event_bus_arn: Optional[str] = None,
    job_name_prefix: Optional[str] = None,
) -> str:
    """Create or update an EventBridge rule for training job notifications.

    Args:
        sns_topic_arn: ARN of the SNS topic to receive notifications.
        sagemaker_session: SageMaker session (provides boto_session).
        events: List of job statuses to notify on. Defaults to
            ["Completed", "Failed", "Stopped"].
        event_bus_arn: Optional EventBridge bus ARN. Defaults to the
            account's default bus.
        job_name_prefix: Optional job name prefix to filter notifications.

    Returns:
        The ARN of the created/updated EventBridge rule.

    Raises:
        ValueError: If sns_topic_arn is invalid or topic doesn't exist.
        PermissionError: If caller lacks required permissions.
    """
    if not sns_topic_arn or not re.match(r"^arn:aws[a-z\-]*:sns:[a-z0-9\-]+:\d{12}:.+$", sns_topic_arn):
        raise ValueError(
            f"Invalid SNS topic ARN: '{sns_topic_arn}'. "
            "Must be a valid ARN like 'arn:aws:sns:us-east-1:012345678910:my-topic'."
        )

    region_name = sagemaker_session.boto_session.region_name
    events_client = sagemaker_session.boto_session.client("events", region_name=region_name)
    sns_client = sagemaker_session.boto_session.client("sns", region_name=region_name)

    _validate_notifications_permissions(events_client)
    _validate_sns_topic(sns_client, sns_topic_arn)

    normalized_events = _normalize_events(events)
    rule_name = _get_rule_name(sns_topic_arn, normalized_events, job_name_prefix)
    event_pattern = _build_event_pattern(normalized_events, job_name_prefix)

    put_rule_kwargs = {
        "Name": rule_name,
        "EventPattern": event_pattern,
        "State": "ENABLED",
        "Description": f"SageMaker PySDK training job notifications -> {sns_topic_arn}",
    }
    if event_bus_arn:
        put_rule_kwargs["EventBusName"] = event_bus_arn

    response = events_client.put_rule(**put_rule_kwargs)
    rule_arn = response["RuleArn"]
    logger.info(f"EventBridge rule created/updated: {rule_name} ({rule_arn})")

    # Add SNS topic as target with formatted message
    target_id = f"{rule_name}-sns-target"
    # Use a JSON object template
    input_template = (
        '{"Job": "<job_name>",'
        ' "Status": "<status>",'
        ' "Time": "<time>",'
        ' "Region": "<region>",'
        ' "Account": "<account>",'
        ' "Failure Reason": "<failure_reason>",'
        ' "Console": "https://<region>.console.aws.amazon.com/sagemaker/home?region=<region>#/jobs/<job_name>"}'
    )
    put_targets_kwargs = {
        "Rule": rule_name,
        "Targets": [{
            "Id": target_id,
            "Arn": sns_topic_arn,
            "InputTransformer": {
                "InputPathsMap": {
                    "job_name": "$.detail.TrainingJobName",
                    "status": "$.detail.TrainingJobStatus",
                    "time": "$.time",
                    "region": "$.region",
                    "failure_reason": "$.detail.FailureReason",
                    "account": "$.account",
                },
                "InputTemplate": input_template,
            },
        }],
    }
    if event_bus_arn:
        put_targets_kwargs["EventBusName"] = event_bus_arn

    try:
        events_client.put_targets(**put_targets_kwargs)
    except Exception as e:
        raise ValueError(
            f"Failed to attach SNS topic to EventBridge rule. "
            f"Ensure your SNS topic has a resource policy allowing "
            f"EventBridge to publish to it. Add this to the topic's access policy:\n"
            f'{{"Effect": "Allow", "Principal": {{"Service": "events.amazonaws.com"}}, '
            f'"Action": "sns:Publish", "Resource": "{sns_topic_arn}"}}'
        ) from e

    logger.info(f"Notifications enabled: {normalized_events} -> {sns_topic_arn}")
    return rule_arn


def delete_notification_rule(
    sagemaker_session,
    rule_arn: str,
    event_bus_arn: Optional[str] = None,
) -> str:
    """Delete an SDK-created EventBridge notification rule.

    Args:
        sagemaker_session: SageMaker session (provides boto_session).
        rule_arn: The ARN of the rule to delete.
        event_bus_arn: Optional EventBridge bus ARN. Defaults to "default".

    Returns:
        The name of the deleted rule.
    """
    region_name = sagemaker_session.boto_session.region_name
    events_client = sagemaker_session.boto_session.client("events", region_name=region_name)

    event_bus_name = event_bus_arn or "default"
    rule_name = rule_arn.rsplit("/", 1)[-1] if "/" in rule_arn else rule_arn

    # Remove SNS targets from the EB rule
    try:
        targets_response = events_client.list_targets_by_rule(
            Rule=rule_name, EventBusName=event_bus_name
        )
        target_ids = [t["Id"] for t in targets_response.get("Targets", [])]
        if target_ids:
            events_client.remove_targets(
                Rule=rule_name, Ids=target_ids, EventBusName=event_bus_name
            )
    except Exception as e:
        logger.debug(f"Error removing targets for rule {rule_name}: {e}")

    # Delete the rule
    try:
        events_client.delete_rule(Name=rule_name, EventBusName=event_bus_name)
    except Exception as e:
        if "ResourceNotFoundException" in str(type(e).__name__) or "does not exist" in str(e).lower():
            raise ValueError(
                f"Rule '{rule_name}' not found on event bus '{event_bus_name}'. "
                "If the rule was created on a custom event bus, pass the same "
                "event_bus_arn to delete_notification_rule()."
            ) from e
        logger.warning(f"Failed to delete rule {rule_name}: {e}")

    logger.info(f"Deleted notification rule: {rule_name}")
    return rule_name


def list_notification_rules(
    sagemaker_session,
    event_bus_arn: Optional[str] = None,
) -> List[Dict[str, str]]:
    """List all SDK-created EventBridge notification rules (identified by the sm-pysdk-job-notif prefix).

    Args:
        sagemaker_session: SageMaker session (provides boto_session).
        event_bus_arn: Optional EventBridge bus ARN. Defaults to "default".

    Returns:
        List of dicts with 'name', 'arn', and 'state' for each rule.
    """
    region_name = sagemaker_session.boto_session.region_name
    events_client = sagemaker_session.boto_session.client("events", region_name=region_name)

    event_bus_name = event_bus_arn or "default"
    rules = []

    paginator = events_client.get_paginator("list_rules")
    for page in paginator.paginate(NamePrefix=_RULE_NAME_PREFIX, EventBusName=event_bus_name):
        for rule in page.get("Rules", []):
            rules.append({
                "name": rule["Name"],
                "arn": rule["Arn"],
                "state": rule.get("State", "UNKNOWN"),
            })

    return rules



