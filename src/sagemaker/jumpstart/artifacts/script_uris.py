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
"""This module contains functions for obtaining JumpStart script uris."""
from __future__ import absolute_import
import os
from typing import Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE,
)
from sagemaker.jumpstart.enums import (
    JumpStartModelType,
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    get_jumpstart_content_bucket,
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_script_uri(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str] = None,
    script_scope: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
):
    """Retrieves the script S3 URI associated with the model matching the given arguments.

    Optionally uses a bucket override specified by environment variable.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the script S3 URI.
        model_version (str): Version of the JumpStart model for which to
            retrieve the model script S3 URI.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        script_scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (str): Region for which to retrieve model script S3 URI.
            (Default: None)
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
        str: the model script URI for the corresponding model.

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
        scope=script_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
        model_type=model_type,
    )

    if script_scope == JumpStartScriptScope.INFERENCE:
        model_script_key = model_specs.hosting_script_key
    elif script_scope == JumpStartScriptScope.TRAINING:
        model_script_key = (
            getattr(model_specs, "training_prepacked_script_key") or model_specs.training_script_key
        )

    bucket = os.environ.get(
        ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE
    ) or get_jumpstart_content_bucket(region)

    script_s3_uri = f"s3://{bucket}/{model_script_key}"

    return script_s3_uri


def _model_supports_inference_script_uri(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
    model_type: Optional[str] = None,
) -> bool:
    """Returns True if the model supports inference with script uri field.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the support status for script uri with inference.
        model_version (str): Version of the JumpStart model for which to retrieve the
            support status for script uri with inference.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (Optional[str]): Region for which to retrieve the
            support status for script uri with inference.
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
        model_type (JumpStartModelType): The type of the model, can be open weights model
            or proprietary model. (Default: JumpStartModelType.OPEN_WEIGHTS).
    Returns:
        bool: the support status for script uri with inference.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
        model_type=model_type,
    )

    return model_specs.use_inference_script_uri()
