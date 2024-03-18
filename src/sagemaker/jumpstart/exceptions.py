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
"""This module stores exceptions related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import List, Optional

from botocore.exceptions import ClientError

from sagemaker.jumpstart.constants import (
    MODEL_ID_LIST_WEB_URL,
    JumpStartScriptScope,
)

NO_AVAILABLE_INSTANCES_ERROR_MSG = (
    "No instances available in {region} that can support model ID '{model_id}'. "
    "Please try another region."
)

NO_AVAILABLE_RESOURCE_REQUIREMENT_RECOMMENDATION_ERROR_MSG = (
    "No available compute resource requirement recommendation for model ID '{model_id}'. "
    "Provide the resource requirements in the deploy method."
)

INVALID_MODEL_ID_ERROR_MSG = (
    "Invalid model ID: '{model_id}'. Please visit "
    f"{MODEL_ID_LIST_WEB_URL} for a list of valid model IDs. "
    "The module `sagemaker.jumpstart.notebook_utils` contains utilities for "
    "fetching model IDs. We recommend upgrading to the latest version of sagemaker "
    "to get access to the most models."
)


_MAJOR_VERSION_WARNING_MSG = (
    "Note that models may have different input/output signatures after a major version upgrade."
)

_VULNERABLE_DEPRECATED_ERROR_RECOMMENDATION = (
    "We recommend that you specify a more recent "
    "model version or choose a different model. To access the latest models "
    "and model versions, be sure to upgrade to the latest version of the SageMaker Python SDK."
)


def get_wildcard_model_version_msg(
    model_id: str, wildcard_model_version: str, full_model_version: str
) -> str:
    """Returns customer-facing message for using a model version with a wildcard character."""

    return (
        f"Using model '{model_id}' with wildcard version identifier '{wildcard_model_version}'. "
        f"You can pin to version '{full_model_version}' "
        f"for more stable results. {_MAJOR_VERSION_WARNING_MSG}"
    )


def get_proprietary_model_subscription_msg(
    model_id: str,
    subscription_link: str,
) -> str:
    """Returns customer-facing message for using a proprietary model."""

    return (
        f"INFO: Using proprietary model '{model_id}'. "
        f"To subscribe to this model in AWS Marketplace, see {subscription_link}"
    )


def get_wildcard_proprietary_model_version_msg(
    model_id: str, wildcard_model_version: str, available_versions: List[str]
) -> str:
    """Returns customer-facing message for passing wildcard version to proprietary models."""
    msg = (
        f"Proprietary model '{model_id}' does not support "
        f"wildcard version identifier '{wildcard_model_version}'. "
    )
    if len(available_versions) > 0:
        msg += f"You can pin to version '{available_versions[0]}'. "
    msg += f"{MODEL_ID_LIST_WEB_URL} for a list of valid model IDs. "
    return msg


def get_old_model_version_msg(
    model_id: str, current_model_version: str, latest_model_version: str
) -> str:
    """Returns customer-facing message associated with using an old model version."""

    return (
        f"Using model '{model_id}' with version '{current_model_version}'. "
        f"You can upgrade to version '{latest_model_version}' to get the latest model "
        f"specifications. {_MAJOR_VERSION_WARNING_MSG}"
    )


def get_proprietary_model_subscription_error(error: ClientError, subscription_link: str) -> None:
    """Returns customer-facing message associated with a Marketplace subscription error."""

    error_code = error.response["Error"]["Code"]
    error_message = error.response["Error"]["Message"]
    if error_code == "ValidationException" and "not subscribed" in error_message:
        raise MarketplaceModelSubscriptionError(subscription_link)


class JumpStartHyperparametersError(ValueError):
    """Exception raised for bad hyperparameters of a JumpStart model."""

    def __init__(
        self,
        message: Optional[str] = None,
    ):
        self.message = message

        super().__init__(self.message)


class VulnerableJumpStartModelError(ValueError):
    """Exception raised when trying to access a JumpStart model specs flagged as vulnerable.

    Raise this exception only if the scope of attributes accessed in the specifications have
    vulnerabilities. For example, a model training script may have vulnerabilities, but not
    the hosting scripts. In such a case, raise a ``VulnerableJumpStartModelError`` only when
    accessing the training specifications.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        vulnerabilities: Optional[List[str]] = None,
        scope: Optional[JumpStartScriptScope] = None,
        message: Optional[str] = None,
    ):
        """Instantiates VulnerableJumpStartModelError exception.

        Args:
            model_id (Optional[str]): model ID of vulnerable JumpStart model.
                (Default: None).
            version (Optional[str]): version of vulnerable JumpStart model.
                (Default: None).
            vulnerabilities (Optional[List[str]]): vulnerabilities associated with
                model. (Default: None).

        """
        if message:
            self.message = message
        else:
            if None in [model_id, version, vulnerabilities, scope]:
                raise RuntimeError(
                    "Must specify `model_id`, `version`, `vulnerabilities`, " "and scope arguments."
                )
            if scope == JumpStartScriptScope.INFERENCE:
                self.message = (
                    f"Version '{version}' of JumpStart model '{model_id}' "  # type: ignore
                    "has at least 1 vulnerable dependency in the inference script. "
                    f"{_VULNERABLE_DEPRECATED_ERROR_RECOMMENDATION} "
                    "List of vulnerabilities: "
                    f"{', '.join(vulnerabilities)}"  # type: ignore
                )
            elif scope == JumpStartScriptScope.TRAINING:
                self.message = (
                    f"Version '{version}' of JumpStart model '{model_id}' "  # type: ignore
                    "has at least 1 vulnerable dependency in the training script. "
                    f"{_VULNERABLE_DEPRECATED_ERROR_RECOMMENDATION} "
                    "List of vulnerabilities: "
                    f"{', '.join(vulnerabilities)}"  # type: ignore
                )
            else:
                raise NotImplementedError(
                    "Unsupported scope for VulnerableJumpStartModelError: "  # type: ignore
                    f"'{scope.value}'"
                )

        super().__init__(self.message)


class DeprecatedJumpStartModelError(ValueError):
    """Exception raised when trying to access a JumpStart model deprecated specifications.

    A deprecated specification for a JumpStart model does not mean the whole model is
    deprecated. There may be more recent specifications available for this model. For
    example, all specification before version ``2.0.0`` may be deprecated, in such a
    case, the SDK would raise this exception only when specifications ``1.*`` are
    accessed.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        version: Optional[str] = None,
        message: Optional[str] = None,
    ):
        if message:
            self.message = message
        else:
            if None in [model_id, version]:
                raise RuntimeError("Must specify `model_id` and `version` arguments.")
            self.message = (
                f"Version '{version}' of JumpStart model '{model_id}' is deprecated. "
                f"{_VULNERABLE_DEPRECATED_ERROR_RECOMMENDATION}"
            )

        super().__init__(self.message)


class MarketplaceModelSubscriptionError(ValueError):
    """Exception raised when trying to deploy a JumpStart Marketplace model.

    A caller is required to subscribe to the Marketplace product in order to deploy.
    This exception is raised when a caller tries to deploy a JumpStart Marketplace model
    but the caller is not subscribed to the model.
    """

    def __init__(
        self,
        model_subscription_link: Optional[str] = None,
        message: Optional[str] = None,
    ):
        if message:
            self.message = message
        else:
            self.message = (
                "To use a proprietary JumpStart model, "
                "you must first subscribe to the model in AWS Marketplace. "
            )
            if model_subscription_link:
                self.message += f"To subscribe to this model, see {model_subscription_link}"

        super().__init__(self.message)
