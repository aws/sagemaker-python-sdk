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

from sagemaker.workflow.retry import RetryPolicy, RetryExceptionTypeEnum, get_default_throttling_retry_policy


def test_valid_retry_policy():
    retry_policy = RetryPolicy(RetryExceptionTypeEnum.ALL, interval_seconds=5, max_attempts=3)
    assert retry_policy.to_request() == {
        "ALL": {
            "IntervalSeconds": 5,
            "BackoffRate": 0.0,
            "RetryUntil": {
                "MetricType": "MAX_ATTEMPTS",
                "MetricValue": 3
            }
        }
    }

    retry_policy = RetryPolicy(RetryExceptionTypeEnum.SERVICE_FAULT, interval_seconds=5, backoff_rate=2.0,
                               expire_after_mins=30)
    assert retry_policy.to_request() == {
        "SERVICE_FAULT": {
            "IntervalSeconds": 5,
            "BackoffRate": 2.0,
            "RetryUntil": {
                "MetricType": "EXPIRE_AFTER_MIN",
                "MetricValue": 30
            }
        }
    }

    retry_policy = RetryPolicy(expire_after_mins=30)
    assert retry_policy.to_request() == {
        "ALL": {
            "IntervalSeconds": 1,
            "BackoffRate": 0.0,
            "RetryUntil": {
                "MetricType": "EXPIRE_AFTER_MIN",
                "MetricValue": 30
            }
        }
    }

    retry_policy = RetryPolicy(max_attempts=10)
    assert retry_policy.to_request() == {
        "ALL": {
            "IntervalSeconds": 1,
            "BackoffRate": 0.0,
            "RetryUntil": {
                "MetricType": "MAX_ATTEMPTS",
                "MetricValue": 10
            }
        }
    }


def test_invalid_retry_policy():
    retry_policies = [
        (RetryExceptionTypeEnum.ALL, -5, 2.0, 3, None),
        (RetryExceptionTypeEnum.ALL, 5, -2.0, 3, None),
        (RetryExceptionTypeEnum.ALL, 5, 2.0, -3, None),
        (RetryExceptionTypeEnum.ALL, 5, 2.0, 21, None),
        (RetryExceptionTypeEnum.ALL, 5, 2.0, None, -1),
        (RetryExceptionTypeEnum.ALL, 5, 2.0, None, 14401),
        (RetryExceptionTypeEnum.SERVICE_FAULT, 5, 2.0, 10, 30),
    ]

    for (ret, interval_sec, backoff_rate, max_attempts, expire_after) in retry_policies:
        try:
            RetryPolicy(
                retry_exception_type=ret,
                interval_seconds=interval_sec,
                backoff_rate=backoff_rate,
                max_attempts=max_attempts,
                expire_after_mins=expire_after
            ).to_request()
            assert False
        except Exception:
            assert True


def test_default_throttling_retry_policy():
    assert get_default_throttling_retry_policy().to_request() == {
         "THROTTLING": {
            "IntervalSeconds": 1,
            "BackoffRate": 2.0,
            "RetryUntil": {
                "MetricType": "MAX_ATTEMPTS",
                "MetricValue": 10
            }
        }
    }


