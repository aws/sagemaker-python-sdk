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
"""Integration test for training job notifications (EventBridge + SNS).

Verifies the full end-to-end notification flow:
1. Creating a trainer with `notifications` config creates an EventBridge rule
2. The rule targets the correct SNS topic
3. Stopping the job delivers a notification message through EventBridge → SNS → SQS
4. The message contains the correct job name and status
5. Cleanup removes the rule and temporary SQS queue

Prerequisites:
    - Active AWS credentials in us-east-1
    - SNS topic: arn:aws:sns:us-east-1:784379639078:fine-tune-integ-test-job-notification
      with EventBridge publish access policy already attached
    - IAM permissions: events:PutRule, events:PutTargets, events:ListRules,
      events:ListTargetsByRule, events:RemoveTargets, events:DeleteRule,
      sqs:CreateQueue, sqs:DeleteQueue, sqs:GetQueueAttributes,
      sqs:ReceiveMessage, sqs:SetQueueAttributes, sns:Subscribe, sns:Unsubscribe

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_notifications.py -v -s
"""
from __future__ import absolute_import

import json
import logging
import os
import time
import random

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.train import SFTTrainer
from sagemaker.train.common import TrainingType

logger = logging.getLogger(__name__)

# Test configuration
REGION = "us-east-1"
SNS_TOPIC_ARN = "arn:aws:sns:us-east-1:784379639078:fine-tune-integ-test-job-notification"
ACCOUNT_ID = "784379639078"
DATA_PREFIX = "notifications-integ"
DATA_S3_KEY = f"{DATA_PREFIX}/sft_sample_data.jsonl"

# Local sample data file (reuse existing test data)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")
LOCAL_TRAINING_DATA = os.path.join(DATA_DIR, "sft_smtj_sample_data.jsonl")


@pytest.fixture(scope="module")
def sm_session():
    """Create a SageMaker session in us-east-1."""
    boto_session = boto3.Session(region_name=REGION)
    return Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def training_data_uri(sm_session):
    """Upload training data to S3 if not present, return the S3 URI."""
    bucket = sm_session.default_bucket()
    s3_client = sm_session.boto_session.client("s3")

    s3_uri = f"s3://{bucket}/{DATA_S3_KEY}"
    try:
        s3_client.head_object(Bucket=bucket, Key=DATA_S3_KEY)
        logger.info(f"Training data already at {s3_uri}")
    except s3_client.exceptions.ClientError:
        logger.info(f"Uploading training data to {s3_uri}")
        s3_client.upload_file(LOCAL_TRAINING_DATA, bucket, DATA_S3_KEY)

    return s3_uri


@pytest.fixture(scope="module")
def sqs_subscriber(sm_session):
    """Create a temporary SQS queue subscribed to the SNS topic for verification.

    Yields a dict with queue_url and subscription_arn.
    Cleans up the queue and subscription after the test module.
    """
    sqs_client = sm_session.boto_session.client("sqs", region_name=REGION)
    sns_client = sm_session.boto_session.client("sns", region_name=REGION)

    queue_name = f"notif-integ-test-{int(time.time())}-{random.randint(1000, 9999)}"

    # Create SQS queue
    queue_response = sqs_client.create_queue(
        QueueName=queue_name,
        Attributes={"MessageRetentionPeriod": "300"},  # 5 min retention
    )
    queue_url = queue_response["QueueUrl"]
    logger.info(f"Created SQS queue: {queue_url}")

    # Get queue ARN
    attrs = sqs_client.get_queue_attributes(
        QueueUrl=queue_url, AttributeNames=["QueueArn"]
    )
    queue_arn = attrs["Attributes"]["QueueArn"]

    # Allow SNS to send messages to this queue
    policy = json.dumps({
        "Version": "2012-10-17",
        "Statement": [{
            "Sid": "AllowSNSPublish",
            "Effect": "Allow",
            "Principal": {"Service": "sns.amazonaws.com"},
            "Action": "sqs:SendMessage",
            "Resource": queue_arn,
            "Condition": {
                "ArnEquals": {"aws:SourceArn": SNS_TOPIC_ARN}
            },
        }],
    })
    sqs_client.set_queue_attributes(
        QueueUrl=queue_url,
        Attributes={"Policy": policy},
    )

    # Subscribe queue to SNS topic
    sub_response = sns_client.subscribe(
        TopicArn=SNS_TOPIC_ARN,
        Protocol="sqs",
        Endpoint=queue_arn,
        Attributes={"RawMessageDelivery": "true"},
    )
    subscription_arn = sub_response["SubscriptionArn"]
    logger.info(f"Subscribed SQS to SNS: {subscription_arn}")

    yield {
        "queue_url": queue_url,
        "queue_arn": queue_arn,
        "subscription_arn": subscription_arn,
    }

    # Cleanup
    try:
        sns_client.unsubscribe(SubscriptionArn=subscription_arn)
        logger.info(f"Unsubscribed: {subscription_arn}")
    except Exception as e:
        logger.warning(f"Failed to unsubscribe: {e}")

    try:
        sqs_client.delete_queue(QueueUrl=queue_url)
        logger.info(f"Deleted queue: {queue_url}")
    except Exception as e:
        logger.warning(f"Failed to delete queue: {e}")


@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_notifications_creates_eventbridge_rule_and_cleanup(
    sm_session, training_data_uri, sqs_subscriber
):
    """Test end-to-end notification flow: EventBridge rule → SNS → SQS message.

    Flow:
    1. Create SFTTrainer with notifications config
    2. Verify EventBridge rule was created with correct SNS target
    3. Submit a serverless training job (non-blocking)
    4. Stop the job to trigger a "Stopped" event
    5. Poll SQS queue for the notification message
    6. Assert message contains job name and status
    7. Clean up the EventBridge rule
    """
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"
    job_name_prefix = f"notif-integ-{unique_id}"

    bucket = sm_session.default_bucket()

    sft_trainer = SFTTrainer(
        model="amazon.nova-micro-v1",
        training_type=TrainingType.LORA,
        training_dataset=training_data_uri,
        s3_output_path=f"s3://{bucket}/{DATA_PREFIX}/output/",
        model_package_group="sdk-test-finetuned-models",
        sagemaker_session=sm_session,
        notifications={
            "sns_topic_arn": SNS_TOPIC_ARN,
            "events": ["Completed", "Failed", "Stopped"],
            "job_name_prefix": job_name_prefix,
        },
        base_job_name=job_name_prefix,
    )

    # Verify notification rule ARN was set
    assert sft_trainer.notification_rule_arn is not None, (
        "Expected notification_rule_arn to be set after trainer construction"
    )
    rule_arn = sft_trainer.notification_rule_arn
    logger.info(f"EventBridge rule created: {rule_arn}")

    # Verify the rule exists via EventBridge API
    events_client = sm_session.boto_session.client("events", region_name=REGION)
    rule_name = rule_arn.rsplit("/", 1)[-1] if "/" in rule_arn else rule_arn.rsplit(":", 1)[-1]

    # Try extracting rule name from ARN format: arn:aws:events:region:account:rule/rule-name
    if "/rule/" in rule_arn:
        rule_name = rule_arn.split("/rule/")[-1]

    rules_response = events_client.list_rules(NamePrefix="sm-pysdk-job-notif")
    rule_names = [r["Name"] for r in rules_response["Rules"]]
    logger.info(f"Found rules: {rule_names}")

    # Find our rule
    matching_rules = [r for r in rules_response["Rules"] if r["Arn"] == rule_arn]
    assert len(matching_rules) == 1, (
        f"Expected exactly 1 rule matching ARN {rule_arn}, found {len(matching_rules)}"
    )
    rule = matching_rules[0]
    assert rule["State"] == "ENABLED"
    logger.info(f"Rule verified: {rule['Name']} (State={rule['State']})")

    # Verify the rule targets our SNS topic
    targets_response = events_client.list_targets_by_rule(Rule=rule["Name"])
    targets = targets_response["Targets"]
    assert len(targets) >= 1, "Expected at least 1 target on the rule"

    sns_targets = [t for t in targets if t["Arn"] == SNS_TOPIC_ARN]
    assert len(sns_targets) == 1, (
        f"Expected SNS topic {SNS_TOPIC_ARN} as target, got: {[t['Arn'] for t in targets]}"
    )
    logger.info(f"Target verified: {sns_targets[0]['Arn']}")

    # Submit a training job (serverless, non-blocking)
    training_job = sft_trainer.train(wait=False)
    assert training_job is not None
    logger.info(f"Training job submitted: {training_job.training_job_name}")

    # Wait briefly for the job to start, then stop it
    time.sleep(30)
    sm_client = sm_session.boto_session.client("sagemaker", region_name=REGION)

    try:
        sm_client.stop_training_job(TrainingJobName=training_job.training_job_name)
        logger.info(f"Stop requested for: {training_job.training_job_name}")
    except Exception as e:
        logger.warning(f"Could not stop job (may already be terminal): {e}")

    # Poll until terminal
    max_wait = 300  # 5 minutes
    start = time.time()
    while time.time() - start < max_wait:
        training_job.refresh()
        status = training_job.training_job_status
        if status in ("Completed", "Failed", "Stopped"):
            break
        logger.info(f"Status: {status} ({int(time.time() - start)}s)")
        time.sleep(15)

    logger.info(
        f"Job final status: {training_job.training_job_status} "
        f"(expected 'Stopped' or 'Failed')"
    )
    # The job should be Stopped (or Failed if it never started)
    assert training_job.training_job_status in ("Stopped", "Failed"), (
        f"Unexpected final status: {training_job.training_job_status}"
    )

    # Poll SQS queue for the notification message
    sqs_client = sm_session.boto_session.client("sqs", region_name=REGION)
    queue_url = sqs_subscriber["queue_url"]

    notification_received = False
    poll_start = time.time()
    poll_timeout = 120  # 2 minutes for event propagation

    while time.time() - poll_start < poll_timeout:
        response = sqs_client.receive_message(
            QueueUrl=queue_url,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=10,
        )

        messages = response.get("Messages", [])
        for msg in messages:
            body = msg["Body"]
            logger.info(f"Received SQS message: {body}")

            # Parse the notification payload
            try:
                payload = json.loads(body)
            except json.JSONDecodeError:
                # Might be a raw string
                payload = {"raw": body}

            # The notification format from InputTransformer has Job, Status, etc.
            job_name = payload.get("Job", "")
            status = payload.get("Status", "")

            if training_job.training_job_name in body:
                notification_received = True
                logger.info(
                    f"Notification matched! Job={job_name}, Status={status}"
                )

                # Verify the message content
                assert training_job.training_job_name == job_name or \
                    training_job.training_job_name in body, (
                    f"Expected job name '{training_job.training_job_name}' in message"
                )
                assert status in ("Stopped", "Failed") or \
                    "Stopped" in body or "Failed" in body, (
                    f"Expected 'Stopped' or 'Failed' status in message, got: {body}"
                )
                break

            # Delete processed message
            sqs_client.delete_message(
                QueueUrl=queue_url,
                ReceiptHandle=msg["ReceiptHandle"],
            )

        if notification_received:
            break

    assert notification_received, (
        f"No notification received for job {training_job.training_job_name} "
        f"within {poll_timeout}s. The EventBridge → SNS → SQS pipeline did not deliver."
    )
    logger.info("End-to-end notification delivery verified!")

    # Clean up: delete the EventBridge rule
    deleted_name = sft_trainer.delete_notification_rule(rule_arn=rule_arn)
    logger.info(f"Deleted rule: {deleted_name}")

    # Verify rule is gone
    rules_after = events_client.list_rules(NamePrefix="sm-pysdk-job-notif")
    remaining_arns = [r["Arn"] for r in rules_after["Rules"]]
    assert rule_arn not in remaining_arns, (
        f"Rule {rule_arn} should have been deleted but still exists"
    )
    logger.info("Cleanup verified: rule no longer exists")
