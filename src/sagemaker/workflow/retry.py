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
"""Pipeline parameters and conditions for workflow."""
from __future__ import absolute_import

from enum import Enum

from sagemaker.workflow.entities import Entity, DefaultEnumMeta, RequestType

import attr

MAX_ATTEMPTS_CAP = 20
MAX_EXPIRE_AFTER_MIN = 14400


class RetryExceptionTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Parameter type enum."""

    ALL = "ALL"
    SERVICE_FAULT = "SERVICE_FAULT"
    THROTTLING = "THROTTLING"
    RESOURCE_LIMIT = "RESOURCE_LIMIT"
    CAPACITY_ERROR = "CAPACITY_ERROR"


@attr.s
class RetryPolicy(Entity):
    """RetryPolicy for workflow pipeline execution step.

    Attributes:
        retry_exception_type (RetryExceptionTypeEnum): The exception type to
            initiate the retry. (default: RetryExceptionTypeEnum.ALL)
        interval_seconds (int): An integer that represents the number of seconds before the
            first retry attempt (default: 5)
        backoff_rate (float): The multiplier by which the retry interval increases during each attempt,
            the default 0.0 is equivalent to linear backoff (default: 0.0)
        max_attempts (int): A positive integer that represents the maximum
            number of retry attempts. (default: None)
        expire_after_mins (int): A positive integer that represents the maximum minute
            to expire any further retry attempt (default: None)
    """
    retry_exception_type: RetryExceptionTypeEnum = attr.ib(factory=RetryExceptionTypeEnum.factory)
    backoff_rate: float = attr.ib(default=0.0)
    interval_seconds: int = attr.ib(default=1.0)
    max_attempts: int = attr.ib(default=None)
    expire_after_mins: int = attr.ib(default=None)

    @retry_exception_type.validator
    def validate_retry_exception_type(self, attribute, value):
        assert isinstance(value, RetryExceptionTypeEnum), \
            f"retry_exception_type should be of type RetryExceptionTypeEnum"

    @backoff_rate.validator
    def validate_backoff_rate(self, attribute, value):
        assert value >= 0.0, \
            f"backoff_rate should be non-negative"

    @interval_seconds.validator
    def validate_interval_seconds(self, attribute, value):
        assert value >= 0.0, \
            f"interval_seconds rate should be non-negative"

    @max_attempts.validator
    def validate_max_attempts(self, attribute, value):
        if value:
            assert MAX_ATTEMPTS_CAP >= value >= 1, \
                f"max_attempts must in range of (0, {MAX_ATTEMPTS_CAP}] attempts"

    @expire_after_mins.validator
    def validate_expire_after_mins(self, attribute, value):
        if value:
            assert MAX_EXPIRE_AFTER_MIN >= value >= 0, \
                f"expire_after_mins must in range of (0, {MAX_EXPIRE_AFTER_MIN}] minutes"

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        if not ((self.max_attempts is None) ^ (self.expire_after_mins is None)):
            raise ValueError("Only one of [max_attempts] and [expire_after_mins] can be given.")

        return {
            self.retry_exception_type.value: {
                "IntervalSeconds": self.interval_seconds,
                "BackoffRate": self.backoff_rate,
                "RetryUntil": {
                    "MetricType": "MAX_ATTEMPTS" if self.max_attempts is not None else "EXPIRE_AFTER_MIN",
                    "MetricValue": self.max_attempts if self.max_attempts is not None else self.expire_after_mins
                }
            }
        }


def get_default_throttling_retry_policy() -> RetryPolicy:
    return RetryPolicy(
        retry_exception_type=RetryExceptionTypeEnum.THROTTLING,
        interval_seconds=1,
        backoff_rate=2.0,
        max_attempts=10,
    )
