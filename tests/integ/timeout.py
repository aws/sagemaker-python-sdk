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

from contextlib import contextmanager
import logging
from time import sleep

from awslogs.core import AWSLogs
from botocore.exceptions import ClientError
import stopit

from sagemaker import Predictor
from tests.integ.retry import retries

# Setting LOGGER for backward compatibility, in case users import it...
logger = LOGGER = logging.getLogger("timeout")


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

    with stopit.ThreadingTimeout(limit, swallow_exc=False) as t:
        yield [t]


@contextmanager
def timeout_and_delete_endpoint_by_name(
    endpoint_name,
    sagemaker_session,
    seconds=0,
    minutes=45,
    hours=0,
    sleep_between_cleanup_attempts=10,
    exponential_sleep=False,
):
    limit = seconds + 60 * minutes + 3600 * hours

    with stopit.ThreadingTimeout(limit, swallow_exc=False) as t:
        no_errors = False
        try:
            yield [t]
            no_errors = True
        finally:
            attempts = 3

            while attempts > 0:
                attempts -= 1
                try:
                    _delete_schedules_associated_with_endpoint(
                        sagemaker_session=sagemaker_session, endpoint_name=endpoint_name
                    )
                    sagemaker_session.delete_endpoint(endpoint_name)
                    logger.info("deleted endpoint %s", endpoint_name)

                    _show_logs(endpoint_name, "Endpoints", sagemaker_session)
                    if no_errors:
                        _cleanup_logs(endpoint_name, "Endpoints", sagemaker_session)
                    break
                except ClientError as ce:
                    if ce.response["Error"]["Code"] == "ValidationException":
                        # avoids the inner exception to be overwritten
                        pass
                # trying to delete the resource again in 10 seconds
                if exponential_sleep:
                    _sleep_between_cleanup_attempts = sleep_between_cleanup_attempts * (
                        3 - attempts
                    )
                else:
                    _sleep_between_cleanup_attempts = sleep_between_cleanup_attempts
                sleep(_sleep_between_cleanup_attempts)


@contextmanager
def timeout_and_delete_model_with_transformer(
    transformer, sagemaker_session, seconds=0, minutes=0, hours=0, sleep_between_cleanup_attempts=10
):
    limit = seconds + 60 * minutes + 3600 * hours

    with stopit.ThreadingTimeout(limit, swallow_exc=False) as t:
        no_errors = False
        try:
            yield [t]
            no_errors = True
        finally:
            attempts = 3

            while attempts > 0:
                attempts -= 1
                try:
                    transformer.delete_model()
                    logger.info("deleted SageMaker model %s", transformer.model_name)

                    _show_logs(transformer.model_name, "Models", sagemaker_session)
                    if no_errors:
                        _cleanup_logs(transformer.model_name, "Models", sagemaker_session)
                    break
                except ClientError as ce:
                    if ce.response["Error"]["Code"] == "ValidationException":
                        pass
                sleep(sleep_between_cleanup_attempts)


@contextmanager
def timeout_and_delete_model_by_name(
    model_name,
    sagemaker_session,
    seconds=0,
    minutes=45,
    hours=0,
    sleep_between_cleanup_attempts=10,
    exponential_sleep=False,
):
    limit = seconds + 60 * minutes + 3600 * hours

    with stopit.ThreadingTimeout(limit, swallow_exc=False) as t:
        no_errors = False
        try:
            yield [t]
            no_errors = True
        finally:
            attempts = 3

            while attempts > 0:
                attempts -= 1
                try:
                    sagemaker_session.delete_model(model_name)
                    logger.info("deleted model %s", model_name)

                    _show_logs(model_name, "Models", sagemaker_session)
                    if no_errors:
                        _cleanup_logs(model_name, "Models", sagemaker_session)
                    break
                except ClientError as ce:
                    if ce.response["Error"]["Code"] == "ValidationException":
                        # avoids the inner exception to be overwritten
                        pass
                # trying to delete the resource again in 10 seconds
                if exponential_sleep:
                    _sleep_between_cleanup_attempts = sleep_between_cleanup_attempts * (
                        3 - attempts
                    )
                else:
                    _sleep_between_cleanup_attempts = sleep_between_cleanup_attempts
                sleep(_sleep_between_cleanup_attempts)


def _delete_schedules_associated_with_endpoint(sagemaker_session, endpoint_name):
    """Deletes schedules associated with a given endpoint. Per latest validation, ensures the
    schedule is stopped and no executions are running, before deleting (otherwise latest
    server-side validations will prevent deletes).

    Args:
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions (default: None). If not
            specified, one is created using the default AWS configuration
            chain.
        endpoint_name (str): The name of the endpoint to delete schedules from.

    """
    predictor = Predictor(endpoint_name=endpoint_name, sagemaker_session=sagemaker_session)
    monitors = predictor.list_monitors()
    for monitor in monitors:
        try:
            monitor._wait_for_schedule_changes_to_apply()
            # Stop the schedules to prevent new executions from triggering.
            monitor.stop_monitoring_schedule()
            executions = monitor.list_executions()
            for execution in executions:
                execution.stop()
            # Wait for all executions to completely stop.
            # Schedules can't be deleted with running executions.
            for execution in executions:
                for _ in retries(60, "Waiting for executions to stop", seconds_to_sleep=5):
                    status = execution.describe()["ProcessingJobStatus"]
                    if status == "Stopped":
                        break
            # Delete schedules.
            monitor.delete_monitoring_schedule()
        except Exception as e:
            logger.warning(
                "Failed to delete monitor %s,\nError: %s",
                monitor.monitoring_schedule_name,
                e,
            )


def _show_logs(resource_name, resource_type, sagemaker_session):
    log_group = "/aws/sagemaker/{}/{}".format(resource_type, resource_name)
    try:
        # print out logs before deletion for debuggability
        logger.info("cloudwatch logs for log group %s:", log_group)
        logs = AWSLogs(
            log_group_name=log_group,
            log_stream_name="ALL",
            start="1d",
            aws_region=sagemaker_session.boto_session.region_name,
        )
        logs.list_logs()
    except Exception:
        logger.exception(
            "Failure occurred while listing cloudwatch log group %s. Swallowing exception but printing "
            "stacktrace for debugging.",
            log_group,
        )


def _cleanup_logs(resource_name, resource_type, sagemaker_session):
    log_group = "/aws/sagemaker/{}/{}".format(resource_type, resource_name)
    try:
        # print out logs before deletion for debuggability
        logger.info("deleting cloudwatch log group %s:", log_group)
        cwl_client = sagemaker_session.boto_session.client("logs")
        cwl_client.delete_log_group(logGroupName=log_group)
        logger.info("deleted cloudwatch log group: %s", log_group)
    except Exception:
        logger.exception(
            "Failure occurred while cleaning up cloudwatch log group %s. "
            "Swallowing exception but printing stacktrace for debugging.",
            log_group,
        )
