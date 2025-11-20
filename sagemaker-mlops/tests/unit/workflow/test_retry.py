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
"""Unit tests for workflow retry."""
from __future__ import absolute_import

import pytest

from sagemaker.mlops.workflow.retry import (
    RetryPolicy, StepRetryPolicy, SageMakerJobStepRetryPolicy,
    StepExceptionTypeEnum, SageMakerJobExceptionTypeEnum
)


def test_retry_policy_max_attempts():
    policy = RetryPolicy(max_attempts=5)
    request = policy.to_request()
    assert request["MaxAttempts"] == 5
    assert request["BackoffRate"] == 2.0


def test_retry_policy_expire_after_mins():
    policy = RetryPolicy(expire_after_mins=60)
    request = policy.to_request()
    assert request["ExpireAfterMin"] == 60


def test_retry_policy_validation():
    policy = RetryPolicy(max_attempts=5)
    request = policy.to_request()
    assert "MaxAttempts" in request


def test_step_retry_policy():
    policy = StepRetryPolicy(
        exception_types=[StepExceptionTypeEnum.SERVICE_FAULT],
        max_attempts=3
    )
    request = policy.to_request()
    assert request["MaxAttempts"] == 3
    assert "Step.SERVICE_FAULT" in request["ExceptionType"]


def test_sagemaker_job_retry_policy():
    policy = SageMakerJobStepRetryPolicy(
        exception_types=[SageMakerJobExceptionTypeEnum.CAPACITY_ERROR],
        max_attempts=5
    )
    request = policy.to_request()
    assert request["MaxAttempts"] == 5
    assert "SageMaker.CAPACITY_ERROR" in request["ExceptionType"]


def test_sagemaker_job_retry_policy_no_types_raises_error():
    with pytest.raises(ValueError, match="At least one of"):
        SageMakerJobStepRetryPolicy(max_attempts=5)
