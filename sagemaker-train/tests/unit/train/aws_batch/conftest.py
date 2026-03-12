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
"""Test constants for AWS Batch unit tests"""

# Job identifiers
JOB_NAME = "test-training-job"
JOB_ARN = "arn:aws:batch:us-west-2:123456789012:job/test-job-id"
JOB_ID = "test-job-id"
JOB_QUEUE = "test-queue"

# Training job identifiers
TRAINING_JOB_NAME = "training-job-20251211"
TRAINING_JOB_ARN = "arn:aws:sagemaker:us-west-2:123456789012:training-job/training-job-20251211"

# Job statuses
JOB_STATUS_SUBMITTED = "SUBMITTED"
JOB_STATUS_PENDING = "PENDING"
JOB_STATUS_RUNNABLE = "RUNNABLE"
JOB_STATUS_STARTING = "STARTING"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_SUCCEEDED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"

# Configuration values
INSTANCE_TYPE = "ml.m5.xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_IN_GB = 30
MAX_RUNTIME_IN_SECONDS = 3600
TRAINING_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5-gpu-py311"
EXECUTION_ROLE = "arn:aws:iam::123456789012:role/SageMakerRole"
S3_OUTPUT_PATH = "s3://my-bucket/output"

# Batch configuration
SCHEDULING_PRIORITY = 1
SHARE_IDENTIFIER = "test-share-id"
ATTEMPT_DURATION_IN_SECONDS = 86400
REASON = "Test termination reason"
NEXT_TOKEN = "test-next-token"

# Tags
BATCH_TAGS = {"batch-key": "batch-value", "environment": "test"}
TRAINING_TAGS = [
    {"Key": "training-key", "Value": "training-value"},
    {"Key": "project", "Value": "test-project"},
]
TRAINING_TAGS_CONVERTED = {"training-key": "training-value", "project": "test-project"}
MERGED_TAGS = {**BATCH_TAGS, **TRAINING_TAGS_CONVERTED}

# Retry configuration
DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG = {
    "attempts": 1,
    "evaluateOnExit": [
        {
            "action": "RETRY",
            "onStatusReason": "Received status from SageMaker:InternalServerError",
        },
        {"action": "EXIT", "onStatusReason": "*"},
    ],
}

# Timeout configuration
TIMEOUT_CONFIG = {"attemptDurationSeconds": ATTEMPT_DURATION_IN_SECONDS}

# API responses
SUBMIT_SERVICE_JOB_RESP = {
    "jobArn": JOB_ARN,
    "jobName": JOB_NAME,
    "jobId": JOB_ID,
}

DESCRIBE_SERVICE_JOB_RESP_RUNNING = {
    "jobId": JOB_ID,
    "jobName": JOB_NAME,
    "jobArn": JOB_ARN,
    "jobQueue": JOB_QUEUE,
    "status": JOB_STATUS_RUNNING,
    "createdAt": 1702300000,
    "startedAt": 1702300100,
}

DESCRIBE_SERVICE_JOB_RESP_SUCCEEDED = {
    "jobId": JOB_ID,
    "jobName": JOB_NAME,
    "jobArn": JOB_ARN,
    "jobQueue": JOB_QUEUE,
    "status": JOB_STATUS_SUCCEEDED,
    "createdAt": 1702300000,
    "startedAt": 1702300100,
    "stoppedAt": 1702300400,
    "latestAttempt": {
        "serviceResourceId": {
            "name": "trainingJobArn",
            "value": TRAINING_JOB_ARN,
        },
        "startedAt": 1702300100,
        "stoppedAt": 1702300400,
    },
}

DESCRIBE_SERVICE_JOB_RESP_FAILED = {
    "jobId": JOB_ID,
    "jobName": JOB_NAME,
    "jobArn": JOB_ARN,
    "jobQueue": JOB_QUEUE,
    "status": JOB_STATUS_FAILED,
    "statusReason": "Task failed",
    "createdAt": 1702300000,
    "startedAt": 1702300100,
    "stoppedAt": 1702300200,
}

DESCRIBE_SERVICE_JOB_RESP_PENDING = {
    "jobId": JOB_ID,
    "jobName": JOB_NAME,
    "jobArn": JOB_ARN,
    "jobQueue": JOB_QUEUE,
    "status": JOB_STATUS_PENDING,
    "createdAt": 1702300000,
}

LIST_SERVICE_JOB_RESP_EMPTY = {
    "jobSummaryList": [],
    "nextToken": None,
}

LIST_SERVICE_JOB_RESP_WITH_JOBS = {
    "jobSummaryList": [
        {"jobName": JOB_NAME, "jobArn": JOB_ARN, "jobId": JOB_ID},
        {"jobName": "another-job", "jobArn": "arn:aws:batch:us-west-2:123456789012:job/another-id", "jobId": "another-id"},
    ],
    "nextToken": None,
}

LIST_SERVICE_JOB_BY_SHARE_RESP_WITH_JOBS = {
    "jobSummaryList": [
        {
            "jobName": JOB_NAME,
            "jobArn": JOB_ARN,
            "jobId": JOB_ID,
            "shareIdentifier": SHARE_IDENTIFIER,
        },
        {
            "jobName": "another-job",
            "jobArn": "arn:aws:batch:us-west-2:123456789012:job/another-id",
            "jobId": "another-id",
            "shareIdentifier": "another-share-identifier",
        },
    ],
    "nextToken": None,
}

LIST_SERVICE_JOB_RESP_WITH_NEXT_TOKEN = {
    "jobSummaryList": [
        {"jobName": JOB_NAME, "jobArn": JOB_ARN, "jobId": JOB_ID},
    ],
    "nextToken": NEXT_TOKEN,
}

# Training payload
TRAINING_JOB_PAYLOAD = {
    "TrainingJobName": TRAINING_JOB_NAME,
    "RoleArn": EXECUTION_ROLE,
    "OutputDataConfig": {"S3OutputPath": S3_OUTPUT_PATH},
    "ResourceConfig": {
        "InstanceType": INSTANCE_TYPE,
        "InstanceCount": INSTANCE_COUNT,
        "VolumeSizeInGB": VOLUME_SIZE_IN_GB,
    },
    "StoppingCondition": {"MaxRuntimeInSeconds": MAX_RUNTIME_IN_SECONDS},
    "AlgorithmSpecification": {
        "TrainingImage": TRAINING_IMAGE,
        "TrainingInputMode": "File",
    },
}
