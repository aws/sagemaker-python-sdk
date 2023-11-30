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
"""Placeholder docstring"""
from __future__ import absolute_import
import logging
import requests

from sagemaker import Session
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.utils.exceptions import ModelBuilderException
from sagemaker.serve.utils.types import ModelServer

logger = logging.getLogger(__name__)

TELEMETRY_OPT_OUT_MESSAGING = (
    "ModelBuilder will collect telemetry to help us better understand our user's needs, "
    "diagnose issues, and deliver additional features. "
    "To opt out of telemetry, please disable "
    "via TelemetryOptOut in intelligent defaults. See "
    "https://sagemaker.readthedocs.io/en/stable/overview.html#"
    "configuring-and-using-defaults-with-the-sagemaker-python-sdk "
    "for more info."
)

MODE_TO_CODE = {
    str(Mode.IN_PROCESS): 1,
    str(Mode.LOCAL_CONTAINER): 2,
    str(Mode.SAGEMAKER_ENDPOINT): 3,
}

MODEL_SERVER_TO_CODE = {
    str(ModelServer.TORCHSERVE): 1,
    str(ModelServer.MMS): 2,
    str(ModelServer.TENSORFLOW_SERVING): 3,
    str(ModelServer.DJL_SERVING): 4,
    str(ModelServer.TRITON): 5,
    str(ModelServer.TGI): 6,
}


def _capture_telemetry(func_name: str):
    """Placeholder docstring"""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Call the original function
            logger.info(TELEMETRY_OPT_OUT_MESSAGING)
            response = None
            caught_ex = None

            image_uri_tail = self.image_uri.split("/")[1]
            extra = f"{func_name}&{MODEL_SERVER_TO_CODE[str(self.model_server)]}&{image_uri_tail}"

            if self.model_server == ModelServer.DJL_SERVING or self.model_server == ModelServer.TGI:
                extra += f"&{self.model}"

            try:
                response = func(self, *args, **kwargs)
                if not self.serve_settings.telemetry_opt_out:
                    _send_telemetry(
                        "1", MODE_TO_CODE[str(self.mode)], self.sagemaker_session, None, extra
                    )
            except ModelBuilderException as e:
                if not self.serve_settings.telemetry_opt_out:
                    _send_telemetry(
                        "0", MODE_TO_CODE[str(self.mode)], self.sagemaker_session, str(e), extra
                    )
                caught_ex = e
            except Exception as e:  # pylint: disable=W0703
                caught_ex = e
            finally:
                if caught_ex:
                    raise caught_ex
                return response  # pylint: disable=W0150

        return wrapper

    return decorator


def _send_telemetry(
    status: str,
    mode: int,
    session: Session,
    failure_reason: str = None,
    extra_info: str = None,
) -> None:
    """Make GET request to an empty object in S3 bucket"""
    try:
        accountId = _get_accountId(session)
        region = _get_region_or_default(session)
        url = _construct_url(accountId, str(mode), status, failure_reason, extra_info, region)
        _requests_helper(url, 2)
        logger.debug("ModelBuilder metrics emitted.")
    except Exception:  # pylint: disable=W0703
        logger.debug("ModelBuilder metrics not emitted")


def _construct_url(
    accountId: str,
    mode: str,
    status: str,
    failure_reason: str,
    extra_info: str,
    region: str,
) -> str:
    """Placeholder docstring"""

    base_url = (
        f"https://dev-exp-t-{region}.s3.{region}.amazonaws.com/telemetry?"
        f"x-accountId={accountId}"
        f"&x-mode={mode}"
        f"&x-status={status}"
    )
    if failure_reason:
        base_url += f"&x-failureReason={failure_reason}"
    if extra_info:
        base_url += f"&x-extra={extra_info}"
    return base_url


def _requests_helper(url, timeout):
    """Placeholder docstring"""

    response = None
    try:
        response = requests.get(url, timeout)
    except requests.exceptions.RequestException as e:
        logger.debug("Request exception: %s", str(e))
    return response


def _get_accountId(session):
    """Placeholder docstring"""

    try:
        sts = session.boto_session.client("sts")
        return sts.get_caller_identity()["Account"]
    except Exception:  # pylint: disable=W0703
        return None


def _get_region_or_default(session):
    """Placeholder docstring"""

    try:
        return session.boto_session.region_name
    except Exception:  # pylint: disable=W0703
        return "us-west-2"
