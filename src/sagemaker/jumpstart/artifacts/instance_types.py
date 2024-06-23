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
"""This module contains functions for obtaining JumpStart instance types."""
from __future__ import absolute_import

from typing import List, Optional

from sagemaker.jumpstart.exceptions import NO_AVAILABLE_INSTANCES_ERROR_MSG
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    JumpStartModelType,
)
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_default_instance_type(
    model_id: str,
    model_version: str,
    scope: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    training_instance_type: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> str:
    """Retrieves the default instance type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default instance type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default instance type.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve default instance type.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        training_instance_type (str): In the case of a model fine-tuned on SageMaker, the training
            instance type used for the training job that produced the fine-tuned weights.
            Optionally supply this to get a inference instance type conditioned
            on the training instance, to ensure compatability of training artifact to inference
            instance. (Default: None).
    Returns:
        str: the default instance type to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of supported computing instances.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        model_type=model_type,
        sagemaker_session=sagemaker_session,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        instance_specific_default_instance_type = (
            (
                model_specs.training_instance_type_variants.get_instance_specific_default_inference_instance_type(  # pylint: disable=C0301 # noqa: E501
                    training_instance_type
                )
            )
            if training_instance_type is not None
            and getattr(model_specs, "training_instance_type_variants", None) is not None
            else None
        )
        default_instance_type = (
            instance_specific_default_instance_type
            if instance_specific_default_instance_type is not None
            else model_specs.default_inference_instance_type
        )
    elif scope == JumpStartScriptScope.TRAINING:
        default_instance_type = model_specs.default_training_instance_type
    else:
        raise NotImplementedError(
            f"Unsupported script scope for retrieving default instance type: '{scope}'"
        )

    if default_instance_type in {None, ""}:
        raise ValueError(NO_AVAILABLE_INSTANCES_ERROR_MSG.format(model_id=model_id, region=region))
    return default_instance_type


def _retrieve_instance_types(
    model_id: str,
    model_version: str,
    scope: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    training_instance_type: Optional[str] = None,
) -> List[str]:
    """Retrieves the supported instance types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported instance types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported instance types.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve supported instance types.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        training_instance_type (str): In the case of a model fine-tuned on SageMaker, the training
            instance type used for the training job that produced the fine-tuned weights.
            Optionally supply this to get a inference instance type conditioned
            on the training instance, to ensure compatability of training artifact to inference
            instance. (Default: None).
    Returns:
        list: the supported instance types to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of supported computing instances.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        default_instance_types = model_specs.supported_inference_instance_types or []
        instance_specific_instance_types = (
            model_specs.training_instance_type_variants.get_instance_specific_supported_inference_instance_types(  # pylint: disable=C0301 # noqa: E501
                training_instance_type
            )
            if training_instance_type is not None
            and getattr(model_specs, "training_instance_type_variants", None) is not None
            else []
        )
        instance_types = (
            instance_specific_instance_types
            if len(instance_specific_instance_types) > 0
            else default_instance_types
        )

    elif scope == JumpStartScriptScope.TRAINING:
        if training_instance_type is not None:
            raise ValueError("Cannot use `training_instance_type` argument with training scope.")
        instance_types = model_specs.supported_training_instance_types
    else:
        raise NotImplementedError(
            f"Unsupported script scope for retrieving supported instance types: '{scope}'"
        )

    if instance_types is None or len(instance_types) == 0:
        raise ValueError(NO_AVAILABLE_INSTANCES_ERROR_MSG.format(model_id=model_id, region=region))

    return instance_types
