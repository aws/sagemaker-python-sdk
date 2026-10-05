from __future__ import absolute_import


TRAINING_JOB_NAME = "my-training-job"
TRAINING_IMAGE = "763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.8.0-cpu-py3"
TRAINING_INPUT_MODE = "File"
CONTAINER_ENTRYPOINT = ["echo", "hello"]
EXECUTION_ROLE = "myrole"
S3_OUTPUT_PATH = "s3://output"
INSTANCE_TYPE = "ml.m4.xlarge"
INSTANCE_COUNT = 1
VOLUME_SIZE_IN_GB = 1
MAX_RUNTIME_IN_SECONDS = 600
TRAINING_JOB_ARN = "arn:aws:sagemaker:us-west-2:476748761737:training-job/jobName"
JOB_NAME = "jobName"
JOB_NAME_IN_PAYLOAD = "jobNameInPayload"
JOB_ID = "123"
JOB_ARN = "arn:batch:job"
JOB_QUEUE = "testQueue"
JOB_STATUS_RUNNABLE = "RUNNABLE"
JOB_STATUS_RUNNING = "RUNNING"
JOB_STATUS_COMPLETED = "SUCCEEDED"
JOB_STATUS_FAILED = "FAILED"
NEXT_TOKEN = "SomeNextToken"
SCHEDULING_PRIORITY = 1
ATTEMPT_DURATION_IN_SECONDS = 100
REASON = "killed by Batch API"
SHARE_IDENTIFIER = "shareId"
BATCH_TAGS = {"batch_k": "batch_v"}
TRAINING_TAGS = [{"Key": "training_k", "Value": "training_v"}]
TRAINING_TAGS_DUPLICATING_BATCH_TAGS = [
    *TRAINING_TAGS,
    {"Key": "batch_k", "Value": "this value should win"},
]
TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS = {"training_k": "training_v"}
MERGED_TAGS = {**BATCH_TAGS, **TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS}
MERGED_TAGS_TRAINING_OVERRIDE = {
    **TRAINING_TAGS_CONVERTED_TO_BATCH_TAGS,
    "batch_k": "this value should win",
}
EXPERIMENT_CONFIG_EMPTY = {}

TRAINING_JOB_PAYLOAD_IN_PASCALCASE = {"TrainingJobName": JOB_NAME_IN_PAYLOAD}
TIMEOUT_CONFIG = {"attemptDurationSeconds": ATTEMPT_DURATION_IN_SECONDS}
SUBMIT_SERVICE_JOB_RESP = {"jobArn": JOB_ARN, "jobName": JOB_NAME, "jobId": JOB_ID}
FIRST_LIST_SERVICE_JOB_RESP = {
    "jobSummaryList": [{"jobName": JOB_NAME, "jobArn": JOB_ARN}],
    "nextToken": NEXT_TOKEN,
}
SECOND_LIST_SERVICE_JOB_RESP = {
    "jobSummaryList": [
        {"jobName": JOB_NAME, "jobArn": JOB_ARN},
        {"jobName": JOB_NAME, "jobArn": JOB_ARN},
    ],
    "nextToken": NEXT_TOKEN,
}
INCORRECT_FIRST_LIST_SERVICE_JOB_RESP = {
    "jobSummaryList": [{"jobName": JOB_NAME}],
    "nextToken": NEXT_TOKEN,
}
EMPTY_LIST_SERVICE_JOB_RESP = {"jobSummaryList": [], "nextToken": None}
DEFAULT_SAGEMAKER_TRAINING_RETRY_CONFIG = {
    "attempts": 1,
    "evaluateOnExit": [
        {
            "action": "RETRY",
            "onStatusReason": "Received status from SageMaker:InternalServerError: "
            "We encountered an internal error. Please try again.",
        },
        {"action": "EXIT", "onStatusReason": "*"},
    ],
}
