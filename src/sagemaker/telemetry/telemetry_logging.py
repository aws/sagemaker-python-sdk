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
"""Telemetry module for SageMaker Python SDK to collect usage data and metrics."""
from __future__ import absolute_import
import logging
import platform
import sys
from time import perf_counter
from typing import List
import functools
import requests

import boto3
from sagemaker.session import Session
from sagemaker.utils import resolve_value_from_config
from sagemaker.config.config_schema import TELEMETRY_OPT_OUT_PATH
from sagemaker.telemetry.constants import (
    Feature,
    Status,
    DEFAULT_AWS_REGION,
)
from sagemaker.user_agent import SDK_VERSION, process_studio_metadata_file

logger = logging.getLogger(__name__)

OS_NAME = platform.system() or "UnresolvedOS"
OS_VERSION = platform.release() or "UnresolvedOSVersion"
OS_NAME_VERSION = "{}/{}".format(OS_NAME, OS_VERSION)
PYTHON_VERSION = "{}.{}.{}".format(
    sys.version_info.major, sys.version_info.minor, sys.version_info.micro
)

TELEMETRY_OPT_OUT_MESSAGING = (
    "SageMaker Python SDK will collect telemetry to help us better understand our user's needs, "
    "diagnose issues, and deliver additional features.\n"
    "To opt out of telemetry, please disable via TelemetryOptOut parameter in SDK defaults config. "
    "For more information, refer to https://sagemaker.readthedocs.io/en/stable/overview.html"
    "#configuring-and-using-defaults-with-the-sagemaker-python-sdk."
)

FEATURE_TO_CODE = {
    str(Feature.SDK_DEFAULTS): 1,
    str(Feature.LOCAL_MODE): 2,
    str(Feature.REMOTE_FUNCTION): 3,
}

STATUS_TO_CODE = {
    str(Status.SUCCESS): 1,
    str(Status.FAILURE): 0,
}


def _telemetry_emitter(feature: str, func_name: str):
    """Decorator to emit telemetry logs for SageMaker Python SDK functions"""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sagemaker_session = None
            if len(args) > 0 and hasattr(args[0], "sagemaker_session"):
                # Get the sagemaker_session from the instance method args
                sagemaker_session = args[0].sagemaker_session
            elif feature == Feature.REMOTE_FUNCTION:
                # Get the sagemaker_session from the function keyword arguments for remote function
                sagemaker_session = kwargs.get(
                    "sagemaker_session", _get_default_sagemaker_session()
                )

            if sagemaker_session:
                logger.debug("sagemaker_session found, preparing to emit telemetry...")
                logger.info(TELEMETRY_OPT_OUT_MESSAGING)
                response = None
                caught_ex = None
                studio_app_type = process_studio_metadata_file()

                # Check if telemetry is opted out
                telemetry_opt_out_flag = resolve_value_from_config(
                    direct_input=None,
                    config_path=TELEMETRY_OPT_OUT_PATH,
                    default_value=False,
                    sagemaker_session=sagemaker_session,
                )
                logger.debug("TelemetryOptOut flag is set to: %s", telemetry_opt_out_flag)

                # Construct the feature list to track feature combinations
                feature_list: List[int] = [FEATURE_TO_CODE[str(feature)]]

                if sagemaker_session.sagemaker_config and feature != Feature.SDK_DEFAULTS:
                    feature_list.append(FEATURE_TO_CODE[str(Feature.SDK_DEFAULTS)])

                if sagemaker_session.local_mode and feature != Feature.LOCAL_MODE:
                    feature_list.append(FEATURE_TO_CODE[str(Feature.LOCAL_MODE)])

                # Construct the extra info to track platform and environment usage metadata
                extra = (
                    f"{func_name}"
                    f"&x-sdkVersion={SDK_VERSION}"
                    f"&x-env={PYTHON_VERSION}"
                    f"&x-sys={OS_NAME_VERSION}"
                    f"&x-platform={studio_app_type}"
                )

                # Add endpoint ARN to the extra info if available
                if sagemaker_session.endpoint_arn:
                    extra += f"&x-endpointArn={sagemaker_session.endpoint_arn}"

                start_timer = perf_counter()
                try:
                    # Call the original function
                    response = func(*args, **kwargs)
                    stop_timer = perf_counter()
                    elapsed = stop_timer - start_timer
                    extra += f"&x-latency={round(elapsed, 2)}"
                    if not telemetry_opt_out_flag:
                        _send_telemetry_request(
                            STATUS_TO_CODE[str(Status.SUCCESS)],
                            feature_list,
                            sagemaker_session,
                            None,
                            None,
                            extra,
                        )
                except Exception as e:  # pylint: disable=W0703
                    stop_timer = perf_counter()
                    elapsed = stop_timer - start_timer
                    extra += f"&x-latency={round(elapsed, 2)}"
                    if not telemetry_opt_out_flag:
                        _send_telemetry_request(
                            STATUS_TO_CODE[str(Status.FAILURE)],
                            feature_list,
                            sagemaker_session,
                            str(e),
                            e.__class__.__name__,
                            extra,
                        )
                    caught_ex = e
                finally:
                    if caught_ex:
                        raise caught_ex
                    return response  # pylint: disable=W0150
            else:
                logger.debug(
                    "Unable to send telemetry for function %s. "
                    "sagemaker_session is not provided or not valid.",
                    func_name,
                )
                return func(*args, **kwargs)

        return wrapper

    return decorator


def _send_telemetry_request(
    status: int,
    feature_list: List[int],
    session: Session,
    failure_reason: str = None,
    failure_type: str = None,
    extra_info: str = None,
) -> None:
    """Make GET request to an empty object in S3 bucket"""
    try:
        accountId = _get_accountId(session)
        region = _get_region_or_default(session)
        url = _construct_url(
            accountId,
            region,
            str(status),
            str(
                ",".join(map(str, feature_list))
            ),  # Remove brackets and quotes to cut down on length
            failure_reason,
            failure_type,
            extra_info,
        )
        # Send the telemetry request
        logger.debug("Sending telemetry request to [%s]", url)
        _requests_helper(url, 2)
        logger.debug("SageMaker Python SDK telemetry successfully emitted.")
    except Exception:  # pylint: disable=W0703
        logger.debug("SageMaker Python SDK telemetry not emitted!")


def _construct_url(
    accountId: str,
    region: str,
    status: str,
    feature: str,
    failure_reason: str,
    failure_type: str,
    extra_info: str,
) -> str:
    """Construct the URL for the telemetry request"""

    base_url = (
        f"https://sm-pysdk-t-{region}.s3.{region}.amazonaws.com/telemetry?"
        f"x-accountId={accountId}"
        f"&x-status={status}"
        f"&x-feature={feature}"
    )
    logger.debug("Failure reason: %s", failure_reason)
    if failure_reason:
        base_url += f"&x-failureReason={failure_reason}"
        base_url += f"&x-failureType={failure_type}"
    if extra_info:
        base_url += f"&x-extra={extra_info}"
    return base_url


def _requests_helper(url, timeout):
    """Make a GET request to the given URL"""

    response = None
    try:
        response = requests.get(url, timeout)
    except requests.exceptions.RequestException as e:
        logger.exception("Request exception: %s", str(e))
    return response


def _get_accountId(session):
    """Return the account ID from the boto session"""

    try:
        sts = session.boto_session.client("sts")
        return sts.get_caller_identity()["Account"]
    except Exception:  # pylint: disable=W0703
        return None


def _get_region_or_default(session):
    """Return the region name from the boto session or default to us-west-2"""

    try:
        return session.boto_session.region_name
    except Exception:  # pylint: disable=W0703
        return DEFAULT_AWS_REGION


def _get_default_sagemaker_session():
    """Return the default sagemaker session"""
    boto_session = boto3.Session(region_name=DEFAULT_AWS_REGION)
    sagemaker_session = Session(boto_session=boto_session)

    return sagemaker_session
