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
from __future__ import absolute_import

from sagemaker.async_inference.waiter_config import WaiterConfig

DEFAULT_DELAY = 15
DEFAULT_MAX_ATTEMPTS = 60
DEFAULT_WAITER_DICT = {
    "Delay": DEFAULT_DELAY,
    "MaxAttempts": DEFAULT_MAX_ATTEMPTS,
}

DELAY = 10
MAX_ATTEMPTS = 10


def test_init():
    waiter_config = WaiterConfig()

    assert waiter_config.delay == DEFAULT_DELAY
    assert waiter_config.max_attempts == DEFAULT_MAX_ATTEMPTS

    waiter_config_self_defined = WaiterConfig(
        max_attempts=DELAY,
        delay=MAX_ATTEMPTS,
    )

    assert waiter_config_self_defined.delay == DELAY
    assert waiter_config_self_defined.max_attempts == MAX_ATTEMPTS


def test_to_dict():
    waiter_config = WaiterConfig()

    assert waiter_config._to_request_dict() == DEFAULT_WAITER_DICT
