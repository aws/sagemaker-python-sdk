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
import os
import platform
import sys
from time import perf_counter
from typing import List
import functools
import requests
from urllib.parse import quote

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.telemetry.attribution import _CREATED_BY_ENV_VAR
from sagemaker.core.telemetry.resource_creation import get_resource_arn
from sagemaker.core.common_utils import resolve_value_from_config
from sagemaker.core.config.config_schema import TELEMETRY_OPT_OUT_PATH
from sagemaker.core.telemetry.constants import (
    Feature,
    Status,
    Region,
    DEFAULT_AWS_REGION,
)
from sagemaker.core.user_agent import SDK_VERSION, process_studio_metadata_file

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
    str(Feature.SDK_DEFAULTS): 11,
    str(Feature.LOCAL_MODE): 12,
    str(Feature.REMOTE_FUNCTION): 13,
    str(Feature.MODEL_TRAINER): 14,
    str(Feature.MODEL_CUSTOMIZATION): 15,
    str(Feature.MLOPS): 16,
    str(Feature.FEATURE_STORE): 17,
    str(Feature.PROCESSING): 18,
    str(Feature.MODEL_CUSTOMIZATION_NOVA): 19,
    str(Feature.MODEL_CUSTOMIZATION_OSS): 20,
}

STATUS_TO_CODE = {
    str(Status.SUCCESS): 1,
    str(Status.FAILURE): 0,
}


def _classify_error(e: Exception) -> str:
    """Classify an exception into an actionable error category."""
    error_type = type(e).__name__
    error_msg = str(e).lower()

    if "validation" in error_type.lower() or "invalid" in error_msg or "must be" in error_msg:
        return "validation_error"
    if "accessdenied" in error_msg or "not authorized" in error_msg or "forbidden" in error_msg:
        return "auth_error"
    if "capacity" in error_msg or "insufficientcapacity" in error_msg or "resourcelimitexceeded" in error_msg:
        return "capacity_error"
    if "timeout" in error_type.lower() or "timed out" in error_msg or "timeout" in error_msg:
        return "timeout_error"
    if "not found" in error_msg or "does not exist" in error_msg or "could not find" in error_msg:
        return "resource_not_found"
    if "eula" in error_msg or "accept_eula" in error_msg:
        return "eula_error"
    if "throttl" in error_msg or "rate exceeded" in error_msg or "too many requests" in error_msg:
        return "throttling_error"
    if "connection" in error_msg or "network" in error_msg or "unreachable" in error_msg:
        return "network_error"
    return "unknown"


def _attr_to_key(attr: str) -> str:
    """Convert attribute name to camelCase telemetry key.

    Examples: '_model_name' -> 'modelName', 'training_type' -> 'trainingType'
    """
    attr = attr.lstrip("_")
    parts = attr.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


class TelemetryParamType:
    """Constants for telemetry parameter extraction types.

    Used in the `telemetry_params` list passed to @_telemetry_emitter decorator.
    Each entry in telemetry_params is a tuple of (name, type) or (name, type, value).

    To add a new telemetry signal to any class:
    1. Identify what you want to track (instance attribute, method return, or kwarg).
    2. Pick the appropriate type constant below.
    3. Add a tuple to the `telemetry_params` list on the decorator.

    Example:
        @_telemetry_emitter(
            feature=Feature.MODEL_CUSTOMIZATION,
            func_name="MyClass.my_method",
            telemetry_params=[
                ("model_name", TelemetryParamType.ATTR_VALUE),       # emits x-modelName=<value>
                ("networking", TelemetryParamType.ATTR_EXISTS),       # emits x-hasNetworking=true/false
                ("_is_fine_tuned", TelemetryParamType.ATTR_CALL),    # emits x-isFineTuned=True/False
                ("instance_type", TelemetryParamType.KWARG_VALUE),   # emits x-instanceType=<kwarg value>
                ("kms_key_id", TelemetryParamType.KWARG_EXISTS),     # emits x-hasKmsKeyId=true/false
            ],
        )
    """

    # Reads self.<name> and emits the actual value.
    # Use for: model names, training types, modes — values useful for analytics.
    # Emits nothing if the attribute is None.
    ATTR_VALUE = "attr_value"

    # Reads self.<name> and emits true/false based on whether it's set (not None).
    # Use for: sensitive configs (KMS, VPC, MLflow) where you only need to know
    # if the customer configured it, without exposing the actual value.
    ATTR_EXISTS = "attr_exists"

    # Calls self.<name>() and emits the return value.
    # Use for: computed/derived values like _is_model_customization(), _is_nova_model().
    # Silently skipped if the method raises an exception.
    ATTR_CALL = "attr_call"

    # Reads kwargs[<name>] from the decorated method's keyword arguments and emits the value.
    # Use for: method parameters not stored on self (e.g., instance_type passed to deploy()).
    # Emits nothing if the kwarg is None or not provided.
    KWARG_VALUE = "kwarg_value"

    # Reads kwargs[<name>] and emits true/false based on whether it's provided and truthy.
    # Use for: optional method parameters where you only need presence info
    # (e.g., update_endpoint, imported_model_kms_key_id).
    KWARG_EXISTS = "kwarg_exists"


def _extract_telemetry_params(instance, kwargs, telemetry_params=None) -> str:
    """Extract telemetry params from instance/kwargs based on telemetry_params list.

    Args:
        instance: The class instance (args[0]).
        kwargs: The kwargs dict from the decorated function call.
        telemetry_params: List of tuples defining what to extract.
            - ("attr_name", ATTR_VALUE) → emit self.attr value
            - ("attr_name", ATTR_EXISTS) → emit true/false
            - ("method_name", ATTR_CALL) → call self.method(), emit return value
            - ("kwarg_name", KWARG_VALUE) → emit kwargs value
            - ("kwarg_name", KWARG_EXISTS) → emit true/false

    Returns:
        str: URL query params string.
    """
    if not telemetry_params:
        return ""
    parts = []
    T = TelemetryParamType
    for param in telemetry_params:
        name, kind = param[0], param[1]
        key = _attr_to_key(name)
        if kind == T.ATTR_VALUE:
            value = getattr(instance, name, None)
            if value is not None:
                parts.append(f"&x-{key}={value}")
        elif kind == T.ATTR_EXISTS:
            value = getattr(instance, name, None)
            parts.append(f"&x-has{key[0].upper()}{key[1:]}={'true' if value else 'false'}")
        elif kind == T.ATTR_CALL:
            method = getattr(instance, name, None)
            if callable(method):
                try:
                    parts.append(f"&x-{key}={method()}")
                except Exception:
                    pass
        elif kind == T.KWARG_VALUE:
            value = kwargs.get(name) if kwargs else None
            if value is not None:
                parts.append(f"&x-{key}={value}")
        elif kind == T.KWARG_EXISTS:
            value = kwargs.get(name) if kwargs else None
            parts.append(f"&x-has{key[0].upper()}{key[1:]}={'true' if value else 'false'}")
    return "".join(parts)


def _telemetry_emitter(feature: str, func_name: str, telemetry_params=None):
    """Telemetry Emitter

    Decorator to emit telemetry logs for SageMaker Python SDK functions. This class needs
    sagemaker_session object as a member. Default session object is a pysdk v2 Session object
    in this repo. When collecting telemetry for classes using sagemaker-core Session object,
    we should be aware of its differences, such as sagemaker_session.sagemaker_config does not
    exist in new Session class.

    Args:
        feature: The Feature enum value for this telemetry event.
        func_name: Human-readable function name for tracking.
        telemetry_params: Optional list of tuples defining granular params to extract.
            See TelemetryParamType for available types.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            sagemaker_session = None
            if len(args) > 0 and hasattr(args[0], "sagemaker_session"):
                # Get the sagemaker_session from the instance method args
                sagemaker_session = args[0].sagemaker_session
            elif len(args) > 0 and hasattr(args[0], "_sagemaker_session"):
                # Get the sagemaker_session from the instance method args (private attribute)
                sagemaker_session = args[0]._sagemaker_session
            elif feature == Feature.REMOTE_FUNCTION:
                # Get the sagemaker_session from the function keyword arguments for remote function
                sagemaker_session = kwargs.get(
                    "sagemaker_session", _get_default_sagemaker_session()
                )

            # Fallback: check kwargs for sagemaker_session (e.g., classmethods where
            # args[0] is the class and the session is passed as a keyword argument)
            if not sagemaker_session:
                sagemaker_session = kwargs.get("sagemaker_session") or (
                    _get_default_sagemaker_session()
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

                # For MODEL_CUSTOMIZATION, append NOVA or OSS sub-feature
                # based on the instance's _is_nova_model_for_telemetry() method
                if feature == Feature.MODEL_CUSTOMIZATION and len(args) > 0:
                    instance = args[0]
                    try:
                        if hasattr(instance, "_is_nova_model_for_telemetry"):
                            if instance._is_nova_model_for_telemetry():
                                feature_list.append(
                                    FEATURE_TO_CODE[str(Feature.MODEL_CUSTOMIZATION_NOVA)]
                                )
                            else:
                                feature_list.append(
                                    FEATURE_TO_CODE[str(Feature.MODEL_CUSTOMIZATION_OSS)]
                                )
                    except Exception:  # pylint: disable=W0703
                        logger.debug(
                            "Unable to determine NOVA/OSS model type for telemetry."
                        )

                if (
                    hasattr(sagemaker_session, "sagemaker_config")
                    and sagemaker_session.sagemaker_config
                    and feature != Feature.SDK_DEFAULTS
                ):
                    feature_list.append(FEATURE_TO_CODE[str(Feature.SDK_DEFAULTS)])

                if (
                    hasattr(sagemaker_session, "local_mode")
                    and sagemaker_session.local_mode
                    and feature != Feature.LOCAL_MODE
                ):
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
                if hasattr(sagemaker_session, "endpoint_arn") and sagemaker_session.endpoint_arn:
                    extra += f"&x-endpointArn={sagemaker_session.endpoint_arn}"

                # Add created_by from environment variable if available
                created_by = os.environ.get(_CREATED_BY_ENV_VAR, "")
                if created_by:
                    extra += f"&x-createdBy={quote(created_by, safe='')}"

                # Extract granular telemetry params from the instance
                if telemetry_params and len(args) > 0:
                    extra += _extract_telemetry_params(args[0], kwargs, telemetry_params)

                start_timer = perf_counter()
                try:
                    # Call the original function
                    response = func(*args, **kwargs)
                    stop_timer = perf_counter()
                    elapsed = stop_timer - start_timer
                    extra += f"&x-latency={round(elapsed, 2)}"
                    # For specified response types (e.g., TrainingJob), obtain the ARN of the
                    # resource created if present so that it can be included.
                    resource_arn = get_resource_arn(response)
                    if resource_arn:
                        extra += f"&x-resourceArn={resource_arn}"
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
                    extra += f"&x-errorCategory={_classify_error(e)}"
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
        accountId = _get_accountId(session) if session else "NotAvailable"
        region = _get_region_or_default(session)

        try:
            Region(region)  # Validate the region
        except ValueError:
            logger.warning(
                "Region not found in supported regions. Telemetry request will not be emitted."
            )
            return

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
