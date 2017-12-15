# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import signal
from contextlib import contextmanager
import logging

from botocore.exceptions import ClientError

LOGGER = logging.getLogger('timeout')


class TimeoutError(Exception):
    pass


@contextmanager
def timeout(seconds=0, minutes=0, hours=0):
    """
    Add a signal-based timeout to any block of code.
    If multiple time units are specified, they will be added together to determine time limit.

    Usage:

    with timeout(seconds=5):
        my_slow_function(...)


    Args:
        - seconds: The time limit, in seconds.
        - minutes: The time limit, in minutes.
        - hours: The time limit, in hours.
    """

    limit = seconds + 60 * minutes + 3600 * hours

    def handler(signum, frame):
        raise TimeoutError('timed out after {} seconds'.format(limit))

    try:
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(limit)

        yield
    finally:
        signal.alarm(0)


@contextmanager
def timeout_and_delete_endpoint(estimator, seconds=0, minutes=0, hours=0):
    with timeout(seconds=seconds, minutes=minutes, hours=hours) as t:
        try:
            yield [t]
        finally:
            try:
                estimator.delete_endpoint()
                LOGGER.info('deleted endpoint')
            except ClientError as ce:
                if ce.response['Error']['Code'] == 'ValidationException':
                    # avoids the inner exception to be overwritten
                    pass


@contextmanager
def timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, seconds=0, minutes=0, hours=0):
    with timeout(seconds=seconds, minutes=minutes, hours=hours) as t:
        try:
            yield [t]
        finally:
            try:
                sagemaker_session.delete_endpoint(endpoint_name)
                LOGGER.info('deleted endpoint {}'.format(endpoint_name))
            except ClientError as ce:
                if ce.response['Error']['Code'] == 'ValidationException':
                    # avoids the inner exception to be overwritten
                    pass
