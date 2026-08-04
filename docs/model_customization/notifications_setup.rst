Setting Up Training Job Notifications
========================================

Get notified when your training jobs complete, fail, or stop via SNS email/SMS
alerts. This guide walks through creating the prerequisite SNS topic and
configuring your trainer to send notifications.

Architecture
-------------

.. code-block:: text

   SageMaker Training Job ──► EventBridge Rule ──► SNS Topic ──► Email/SMS/Slack

The SDK creates an EventBridge rule that listens for training job status changes
and routes them to your SNS topic. You provide the topic; the SDK handles the
wiring.

Prerequisites
--------------

You need:

1. An SNS topic with a policy allowing EventBridge to publish to it
2. A subscription on that topic (email, SMS, Slack, etc.)
3. IAM permissions for EventBridge rule management (see below)

Step 1: Create an SNS Topic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**AWS Console**

1. Go to **Amazon SNS** → **Topics** → **Create topic**
2. Choose **Standard** type
3. Name it (e.g., ``my-training-alerts``)
4. Click **Create topic**
5. Note the **Topic ARN** (e.g., ``arn:aws:sns:us-east-1:123456789012:my-training-alerts``)

**AWS CLI**

.. code-block:: bash

   aws sns create-topic --name my-training-alerts

Step 2: Allow EventBridge to Publish
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The topic needs a resource policy granting EventBridge publish access.

**AWS Console**

1. Go to **Amazon SNS** → **Topics** → open your topic → **Access policy** tab → **Edit**
2. Add this statement to the policy's ``Statement`` array (replace the ARN and
   account ID with your own SNS topic ARN and AWS account ID):

.. code-block:: json

   {
       "Sid": "AllowEventBridgePublish",
       "Effect": "Allow",
       "Principal": {"Service": "events.amazonaws.com"},
       "Action": "SNS:Publish",
       "Resource": "arn:aws:sns:us-east-1:123456789012:my-training-alerts",
       "Condition": {
           "StringEquals": {"AWS:SourceAccount": "123456789012"}
       }
   }

.. note::

   Replace ``arn:aws:sns:us-east-1:123456789012:my-training-alerts`` with your
   actual SNS topic ARN, and ``123456789012`` with your AWS account ID. These
   must match so that only EventBridge in your account can publish to your topic.

**AWS CLI**

.. code-block:: bash

   TOPIC_ARN="arn:aws:sns:us-east-1:123456789012:my-training-alerts"
   ACCOUNT_ID="123456789012"

   aws sns set-topic-attributes \
       --topic-arn $TOPIC_ARN \
       --attribute-name Policy \
       --attribute-value '{
           "Version": "2008-10-17",
           "Statement": [{
               "Sid": "AllowEventBridgePublish",
               "Effect": "Allow",
               "Principal": {"Service": "events.amazonaws.com"},
               "Action": "SNS:Publish",
               "Resource": "'$TOPIC_ARN'",
               "Condition": {"StringEquals": {"AWS:SourceAccount": "'$ACCOUNT_ID'"}}
           }]
       }'

Step 3: Subscribe to the Topic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Email**

.. code-block:: bash

   aws sns subscribe \
       --topic-arn arn:aws:sns:us-east-1:123456789012:my-training-alerts \
       --protocol email \
       --notification-endpoint you@example.com

Check your inbox and confirm the subscription.

**SMS**

.. code-block:: bash

   aws sns subscribe \
       --topic-arn arn:aws:sns:us-east-1:123456789012:my-training-alerts \
       --protocol sms \
       --notification-endpoint +15551234567

Step 4: Use with the SDK
~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the topic ARN in the ``notifications`` config when constructing your trainer:

.. code-block:: python

   from sagemaker.train import SFTTrainer
   from sagemaker.train.common import TrainingType
   from sagemaker.core.training.configs import TrainingJobCompute

   trainer = SFTTrainer(
       model="amazon.nova-2-lite-v1",
       training_type=TrainingType.LORA,
       training_dataset="s3://my-bucket/data/train.jsonl",
       compute=TrainingJobCompute(instance_type="ml.p4d.24xlarge"),
       notifications={
           "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:my-training-alerts",
       },
   )

   trainer.train()

Configuration Options
~~~~~~~~~~~~~~~~~~~~~~

The ``notifications`` dict supports:

.. list-table::
   :header-rows: 1
   :widths: 25 10 65

   * - Key
     - Required
     - Description
   * - ``sns_topic_arn``
     - Yes
     - ARN of your SNS topic
   * - ``events``
     - No
     - List of statuses to notify on. Default: ``["Completed", "Failed", "Stopped"]``.
       Valid values: ``Completed``, ``Failed``, ``Stopped``, ``InProgress``.
   * - ``event_bus_arn``
     - No
     - Custom EventBridge bus ARN. Defaults to the account's default event bus.
   * - ``job_name_prefix``
     - No
     - Only notify for jobs whose name starts with this prefix.

Example: notify only on failures for jobs matching a prefix:

.. code-block:: python

   notifications={
       "sns_topic_arn": "arn:aws:sns:us-east-1:123456789012:my-training-alerts",
       "events": ["Failed"],
       "job_name_prefix": "prod-sft-",
   }

Managing Notification Rules
-----------------------------

List active rules:

.. code-block:: python

   rules = trainer.list_notification_rules()
   for rule in rules:
       print(f"{rule['name']} ({rule['state']})")

Delete a rule:

.. code-block:: python

   trainer.delete_notification_rule(rule_arn="arn:aws:events:us-east-1:123456789012:rule/sm-pysdk-job-notif-abc123")

Required IAM Permissions
--------------------------

The caller (your IAM role or user) needs these permissions. Replace the SNS
resource ARN with your actual topic ARN:

.. code-block:: json

   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "events:PutRule",
                   "events:PutTargets",
                   "events:ListRules",
                   "events:ListTargetsByRule",
                   "events:RemoveTargets",
                   "events:DeleteRule"
               ],
               "Resource": "arn:aws:events:*:*:rule/sm-pysdk-job-notif-*"
           },
           {
               "Effect": "Allow",
               "Action": "sns:GetTopicAttributes",
               "Resource": "arn:aws:sns:*:*:my-training-alerts"
           }
       ]
   }

.. note::

   The ``sns:GetTopicAttributes`` resource must match the SNS topic you created
   in Step 1. You can use a wildcard (``arn:aws:sns:*:*:*``) for broader access
   or scope it to your specific topic ARN for least privilege.

Troubleshooting
-----------------

**PermissionError: Missing permissions to manage EventBridge rules**

Your IAM identity needs ``events:PutRule`` and ``events:PutTargets``. Ask your
admin to attach the policy above.

**ValueError: SNS topic not found**

Verify the topic ARN is correct and exists in the same region as your
SageMaker session. Ensure you have ``sns:GetTopicAttributes`` permission.

**Not receiving notifications**

1. Confirm the SNS subscription is in ``Confirmed`` state (check in the Console)
2. Verify the topic policy allows EventBridge to publish (Step 2 above)
3. Check that the ``events`` list includes the status you're waiting for
