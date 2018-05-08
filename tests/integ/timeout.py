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
from awslogs.core import AWSLogs
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
def timeout_and_delete_endpoint_by_name(endpoint_name, sagemaker_session, seconds=0, minutes=35, hours=0):
    with timeout(seconds=seconds, minutes=minutes, hours=hours) as t:
        no_errors = False
        try:
            yield [t]
            no_errors = True
        finally:
            try:
                sagemaker_session.delete_endpoint(endpoint_name)
                LOGGER.info('deleted endpoint {}'.format(endpoint_name))

                _show_endpoint_logs(endpoint_name, sagemaker_session)
                if no_errors:
                    _cleanup_endpoint_logs(endpoint_name, sagemaker_session)
            except ClientError as ce:
                if ce.response['Error']['Code'] == 'ValidationException':
                    # avoids the inner exception to be overwritten
                    pass


def _show_endpoint_logs(endpoint_name, sagemaker_session):
    log_group = '/aws/sagemaker/Endpoints/{}'.format(endpoint_name)
    try:
        # print out logs before deletion for debuggability
        LOGGER.info('cloudwatch logs for log group {}:'.format(log_group))
        logs = AWSLogs(log_group_name=log_group, log_stream_name='ALL', start='1d',
                       aws_region=sagemaker_session.boto_session.region_name)
        logs.list_logs()
    except Exception:
        LOGGER.exception('Failure occurred while listing cloudwatch log group %s. ' +
                         'Swallowing exception but printing stacktrace for debugging.', log_group)


def _cleanup_endpoint_logs(endpoint_name, sagemaker_session):
    log_group = '/aws/sagemaker/Endpoints/{}'.format(endpoint_name)
    try:
        # print out logs before deletion for debuggability
        LOGGER.info('deleting cloudwatch log group {}:'.format(log_group))
        cwl_client = sagemaker_session.boto_session.client('logs')
        cwl_client.delete_log_group(logGroupName=log_group)
        LOGGER.info('deleted cloudwatch log group: {}'.format(log_group))
    except Exception:
        LOGGER.exception('Failure occurred while cleaning up cloudwatch log group %s. ' +
                         'Swallowing exception but printing stacktrace for debugging.', log_group)
