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
"""This module contains functions for obtaining JumpStart image uris."""
from __future__ import absolute_import

from typing import Optional
from sagemaker.core.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.core.jumpstart.enums import (
    JumpStartModelType,
    JumpStartScriptScope,
)
from sagemaker.core.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.core.helper.session_helper import Session


def _retrieve_image_uri(
    model_id: str,
    model_version: str,
    image_scope: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    instance_type: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
):
    """Retrieves the container image URI for JumpStart models.

    Only `model_id`, `model_version`, and `image_scope` are required;
    the rest of the fields are auto-populated.


    Args:
        model_id (str): JumpStart model ID for which to retrieve image URI.
        model_version (str): Version of the JumpStart model for which to retrieve
            the image URI.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "eia". If ``accelerator_type`` is set,
            ``image_scope`` is ignored.
        region (str): The AWS region. (Default: None).
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.
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
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
        model_type (JumpStartModelType): The type of the model, can be open weights model
            or proprietary model. (Default: JumpStartModelType.OPEN_WEIGHTS).
    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=image_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
        model_type=model_type,
    )

    if image_scope == JumpStartScriptScope.INFERENCE:
        hosting_instance_type_variants = model_specs.hosting_instance_type_variants
        if hosting_instance_type_variants:
            image_uri = hosting_instance_type_variants.get_image_uri(
                instance_type=instance_type, region=region
            )
            if image_uri is not None:
                return image_uri
        if hub_arn:
            ecr_uri = model_specs.hosting_ecr_uri
            return ecr_uri

        raise ValueError(
            f"No inference ECR configuration found for JumpStart model ID '{model_id}' "
            f"with {instance_type} instance type in {region}. "
            "Please try another instance type or region."
        )
    if image_scope == JumpStartScriptScope.TRAINING:
        training_instance_type_variants = model_specs.training_instance_type_variants
        if training_instance_type_variants:
            image_uri = training_instance_type_variants.get_image_uri(
                instance_type=instance_type, region=region
            )
            if image_uri is not None:
                return image_uri
        if hub_arn:
            ecr_uri = model_specs.training_ecr_uri
            return ecr_uri

        raise ValueError(
            f"No training ECR configuration found for JumpStart model ID '{model_id}' "
            f"with {instance_type} instance type in {region}. "
            "Please try another instance type or region."
        )

    raise ValueError(f"Invalid scope: {image_scope}")
