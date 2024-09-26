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
from time import perf_counter

import requests

from sagemaker import Session, exceptions
from sagemaker.serve.mode.function_pointers import Mode
from sagemaker.serve.model_format.mlflow.constants import MLFLOW_MODEL_PATH
from sagemaker.serve.utils.exceptions import ModelBuilderException
from sagemaker.serve.utils.lineage_constants import (
    MLFLOW_LOCAL_PATH,
    MLFLOW_S3_PATH,
    MLFLOW_MODEL_PACKAGE_PATH,
    MLFLOW_RUN_ID,
    MLFLOW_REGISTRY_PATH,
)
from sagemaker.serve.utils.lineage_utils import _get_mlflow_model_path_type
from sagemaker.serve.utils.types import (
    ModelServer,
    ImageUriOption,
    ModelHub,
    SpeculativeDecodingDraftModelSource,
)
from sagemaker.serve.validations.check_image_uri import is_1p_image_uri
from sagemaker.user_agent import SDK_VERSION

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
    str(ModelServer.TEI): 7,
}

MLFLOW_MODEL_PATH_CODE = {
    MLFLOW_LOCAL_PATH: 1,
    MLFLOW_S3_PATH: 2,
    MLFLOW_MODEL_PACKAGE_PATH: 3,
    MLFLOW_RUN_ID: 4,
    MLFLOW_REGISTRY_PATH: 5,
}

MODEL_HUB_TO_CODE = {
    str(ModelHub.JUMPSTART): 1,
    str(ModelHub.HUGGINGFACE): 2,
}

SD_DRAFT_MODEL_SOURCE_TO_CODE = {
    str(SpeculativeDecodingDraftModelSource.SAGEMAKER): 1,
    str(SpeculativeDecodingDraftModelSource.CUSTOM): 2,
}


def _capture_telemetry(func_name: str):
    """Placeholder docstring"""

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Call the original function
            logger.info(TELEMETRY_OPT_OUT_MESSAGING)
            response = None
            caught_ex = None
            status = "1"
            failure_reason = None
            failure_type = None
            extra = f"{func_name}"

            start_timer = perf_counter()
            try:
                response = func(self, *args, **kwargs)
            except (
                ModelBuilderException,
                exceptions.CapacityError,
                exceptions.UnexpectedStatusException,
                exceptions.AsyncInferenceError,
            ) as e:
                status = "0"
                caught_ex = e
                failure_reason = str(e)
                failure_type = e.__class__.__name__
            except Exception as e:  # pylint: disable=W0703
                raise e

            stop_timer = perf_counter()
            elapsed = stop_timer - start_timer

            if self.model_server:
                extra += f"&x-modelServer={MODEL_SERVER_TO_CODE[str(self.model_server)]}"

            if self.image_uri:
                image_uri_tail = self.image_uri.split("/")[1]
                image_uri_option = _get_image_uri_option(
                    self.image_uri, getattr(self, "_is_custom_image_uri", False)
                )

            if self.image_uri:
                extra += f"&x-imageTag={image_uri_tail}"

            extra += f"&x-sdkVersion={SDK_VERSION}"

            if self.image_uri:
                extra += f"&x-defaultImageUsage={image_uri_option}"

            if self.model_server == ModelServer.DJL_SERVING or self.model_server == ModelServer.TGI:
                extra += f"&x-modelName={self.model}"

            if self.sagemaker_session and self.sagemaker_session.endpoint_arn:
                extra += f"&x-endpointArn={self.sagemaker_session.endpoint_arn}"

            if getattr(self, "_is_mlflow_model", False):
                mlflow_model_path = self.model_metadata[MLFLOW_MODEL_PATH]
                mlflow_model_path_type = _get_mlflow_model_path_type(mlflow_model_path)
                extra += f"&x-mlflowModelPathType={MLFLOW_MODEL_PATH_CODE[mlflow_model_path_type]}"

            if getattr(self, "model_hub", False):
                extra += f"&x-modelHub={MODEL_HUB_TO_CODE[str(self.model_hub)]}"

            if getattr(self, "is_fine_tuned", False):
                extra += "&x-fineTuned=1"

            if getattr(self, "is_compiled", False):
                extra += "&x-compiled=1"
            if getattr(self, "is_quantized", False):
                extra += "&x-quantized=1"
            if getattr(self, "speculative_decoding_draft_model_source", False):
                model_provider_enum = (
                    SpeculativeDecodingDraftModelSource.SAGEMAKER
                    if self.speculative_decoding_draft_model_source == "sagemaker"
                    else SpeculativeDecodingDraftModelSource.CUSTOM
                )
                model_provider_value = SD_DRAFT_MODEL_SOURCE_TO_CODE[str(model_provider_enum)]
                extra += f"&x-sdDraftModelSource={model_provider_value}"

            if getattr(self, "deployment_config_name", False):
                config_name_code = self.deployment_config_name.lower()
                extra += f"&x-configName={config_name_code}"

            extra += f"&x-latency={round(elapsed, 2)}"

            if not self.serve_settings.telemetry_opt_out:
                _send_telemetry(
                    status,
                    MODE_TO_CODE[str(self.mode)],
                    self.sagemaker_session,
                    failure_reason,
                    failure_type,
                    extra,
                )

            if caught_ex:
                raise caught_ex

            return response

        return wrapper

    return decorator


def _send_telemetry(
    status: str,
    mode: int,
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
            str(mode),
            status,
            failure_reason,
            failure_type,
            extra_info,
            region,
        )
        _requests_helper(url, 2)
        logger.debug("ModelBuilder metrics emitted.")
    except Exception:  # pylint: disable=W0703
        logger.debug("ModelBuilder metrics not emitted")


def _construct_url(
    accountId: str,
    mode: str,
    status: str,
    failure_reason: str,
    failure_type: str,
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
        base_url += f"&x-failureType={failure_type}"
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


def _get_image_uri_option(image_uri: str, is_custom_image: bool) -> int:
    """Detect whether default values are used for ModelBuilder

    Args:
        image_uri (str): Image uri used by ModelBuilder.
        is_custom_image: (bool): Boolean indicating whether customer provides with custom image.
    Returns:
        bool: Integer code of image option types.
    """

    if not is_custom_image:
        return ImageUriOption.DEFAULT_IMAGE.value

    if is_1p_image_uri(image_uri):
        return ImageUriOption.CUSTOM_1P_IMAGE.value

    return ImageUriOption.CUSTOM_IMAGE.value
