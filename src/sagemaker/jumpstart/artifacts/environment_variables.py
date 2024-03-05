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
"""This module contains functions for obtaining JumpStart environment variables."""
from __future__ import absolute_import
from typing import Dict, Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
    SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    get_jumpstart_gated_content_bucket,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_default_environment_variables(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    include_aws_sdk_env_vars: bool = True,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    instance_type: Optional[str] = None,
    script: JumpStartScriptScope = JumpStartScriptScope.INFERENCE,
) -> Dict[str, str]:
    """Retrieves the inference environment variables for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default environment variables.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default environment variables.
        region (Optional[str]): Region for which to retrieve default environment variables.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        include_aws_sdk_env_vars (bool): True if environment variables for low-level AWS API call
            should be included. The `Model` class of the SageMaker Python SDK inserts environment
            variables that would be required when making the low-level AWS API call.
            (Default: True).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        instance_type (str): An instance type to optionally supply in order to get
            environment variables specific for the instance type.
        script (JumpStartScriptScope): The JumpStart script for which to retrieve
            environment variables.
    Returns:
        dict: the inference environment variables to use for the model.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=script,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    default_environment_variables: Dict[str, str] = {}
    if script == JumpStartScriptScope.INFERENCE:
        for environment_variable in model_specs.inference_environment_variables:
            if include_aws_sdk_env_vars or environment_variable.required_for_model_class:
                default_environment_variables[environment_variable.name] = str(
                    environment_variable.default
                )

    if instance_type:
        if script == JumpStartScriptScope.INFERENCE and getattr(
            model_specs, "hosting_instance_type_variants", None
        ):
            default_environment_variables.update(
                model_specs.hosting_instance_type_variants.get_instance_specific_environment_variables(  # noqa E501  # pylint: disable=c0301
                    instance_type
                )
            )
        elif script == JumpStartScriptScope.TRAINING and getattr(
            model_specs, "training_instance_type_variants", None
        ):
            instance_specific_environment_variables = model_specs.training_instance_type_variants.get_instance_specific_environment_variables(  # noqa E501  # pylint: disable=c0301
                instance_type
            )

            default_environment_variables.update(instance_specific_environment_variables)

            gated_model_env_var: Optional[str] = _retrieve_gated_model_uri_env_var_value(
                model_id=model_id,
                model_version=model_version,
                region=region,
                tolerate_vulnerable_model=tolerate_vulnerable_model,
                tolerate_deprecated_model=tolerate_deprecated_model,
                sagemaker_session=sagemaker_session,
                instance_type=instance_type,
            )

            if gated_model_env_var is not None:
                default_environment_variables.update(
                    {SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY: gated_model_env_var}
                )

    return default_environment_variables


def _retrieve_gated_model_uri_env_var_value(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    instance_type: Optional[str] = None,
) -> Optional[str]:
    """Retrieves the gated model env var URI matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the gated model env var URI.
        model_version (str): Version of the JumpStart model for which to retrieve the
            gated model env var URI.
        region (Optional[str]): Region for which to retrieve the gated model env var URI.
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
        instance_type (str): An instance type to optionally supply in order to get
            environment variables specific for the instance type.

    Returns:
        Optional[str]: the s3 URI to use for the environment variable, or None if the model does not
            have gated training artifacts.

    Raises:
        ValueError: If the model specs specified are invalid.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.TRAINING,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    s3_key: Optional[
        str
    ] = model_specs.training_instance_type_variants.get_instance_specific_gated_model_key_env_var_value(  # noqa E501  # pylint: disable=c0301
        instance_type
    )
    if s3_key is None:
        return None

    return f"s3://{get_jumpstart_gated_content_bucket(region)}/{s3_key}"
