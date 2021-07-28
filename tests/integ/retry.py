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

import time

DEFAULT_SLEEP_TIME_SECONDS = 10


def retries(max_retry_count, exception_message_prefix, seconds_to_sleep=DEFAULT_SLEEP_TIME_SECONDS):
    for i in range(max_retry_count):
        yield i
        time.sleep(seconds_to_sleep)

    raise Exception(
        "{} has reached the maximum retry count {}".format(
            exception_message_prefix, max_retry_count
        )
    )
