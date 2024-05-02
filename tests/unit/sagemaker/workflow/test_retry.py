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
# language governing permissions and limitations under the License.
from __future__ import absolute_import


from sagemaker.workflow.retry import (
    RetryPolicy,
    StepRetryPolicy,
    SageMakerJobStepRetryPolicy,
    StepExceptionTypeEnum,
    SageMakerJobExceptionTypeEnum,
)


def test_valid_step_retry_policy():
    retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT, StepExceptionTypeEnum.THROTTLING],
        interval_seconds=5,
        max_attempts=3,
    )
    assert retry_policy.to_request() == {
        "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
        "IntervalSeconds": 5,
        "BackoffRate": 2.0,
        "MaxAttempts": 3,
    }

    retry_policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT, StepExceptionTypeEnum.THROTTLING],
        interval_seconds=5,
        backoff_rate=2.0,
        expire_after_mins=30,
    )
    assert retry_policy.to_request() == {
        "ExceptionType": ["Step.SERVICE_FAULT", "Step.THROTTLING"],
        "IntervalSeconds": 5,
        "BackoffRate": 2.0,
        "ExpireAfterMin": 30,
    }


def test_invalid_step_retry_policy():
    try:
        StepRetryPolicy(
            exception_types=[SageMakerJobExceptionTypeEnum.INTERNAL_ERROR],
            interval_seconds=5,
            max_attempts=3,
        )
        assert False
    except Exception:
        assert True


def test_valid_sagemaker_job_step_retry_policy():
    retry_policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        failure_reason_types=[
            SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
            SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
        ],
        interval_seconds=5,
        max_attempts=3,
    )
    assert retry_policy.to_request() == {
        "ExceptionType": [
            "SageMaker.RESOURCE_LIMIT",
            "SageMaker.JOB_INTERNAL_ERROR",
            "SageMaker.CAPACITY_ERROR",
        ],
        "IntervalSeconds": 5,
        "BackoffRate": 2.0,
        "MaxAttempts": 3,
    }

    retry_policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.RESOURCE_LIMIT],
        failure_reason_types=[
            SageMakerJobExceptionTypeEnum.INTERNAL_ERROR,
            SageMakerJobExceptionTypeEnum.CAPACITY_ERROR,
        ],
        interval_seconds=5,
        max_attempts=3,
    )
    assert retry_policy.to_request() == {
        "ExceptionType": [
            "SageMaker.RESOURCE_LIMIT",
            "SageMaker.JOB_INTERNAL_ERROR",
            "SageMaker.CAPACITY_ERROR",
        ],
        "IntervalSeconds": 5,
        "BackoffRate": 2.0,
        "MaxAttempts": 3,
    }


def test_invalid_retry_policy():
    retry_policies = [
        (-5, 2.0, 3, None),
        (5, -2.0, 3, None),
        (5, 2.0, -3, None),
        (5, 2.0, 21, None),
        (5, 2.0, None, -1),
        (5, 2.0, None, 14401),
        (5, 2.0, 10, 30),
    ]

    for interval_sec, backoff_rate, max_attempts, expire_after in retry_policies:
        try:
            RetryPolicy(
                interval_seconds=interval_sec,
                backoff_rate=backoff_rate,
                max_attempts=max_attempts,
                expire_after_mins=expire_after,
            ).to_request()
            assert False
        except Exception:
            assert True
