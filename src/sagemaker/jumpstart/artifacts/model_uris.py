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
"""This module contains functions for obtaining JumpStart model uris."""
from __future__ import absolute_import
import os
from typing import Optional

from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    get_jumpstart_content_bucket,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session
from sagemaker.jumpstart.types import JumpStartModelSpecs


def _retrieve_hosting_prepacked_artifact_key(
    model_specs: JumpStartModelSpecs, instance_type: str
) -> str:
    """Returns instance specific hosting prepacked artifact key or default one as fallback."""
    instance_specific_prepacked_hosting_artifact_key: Optional[str] = (
        model_specs.hosting_instance_type_variants.get_instance_specific_prepacked_artifact_key(
            instance_type=instance_type
        )
        if instance_type
        and getattr(model_specs, "hosting_instance_type_variants", None) is not None
        else None
    )

    default_prepacked_hosting_artifact_key: Optional[str] = getattr(
        model_specs, "hosting_prepacked_artifact_key"
    )

    return (
        instance_specific_prepacked_hosting_artifact_key or default_prepacked_hosting_artifact_key
    )


def _retrieve_hosting_artifact_key(model_specs: JumpStartModelSpecs, instance_type: str) -> str:
    """Returns instance specific hosting artifact key or default one as fallback."""
    instance_specific_hosting_artifact_key: Optional[str] = (
        model_specs.hosting_instance_type_variants.get_instance_specific_artifact_key(
            instance_type=instance_type
        )
        if instance_type
        and getattr(model_specs, "hosting_instance_type_variants", None) is not None
        else None
    )

    default_hosting_artifact_key: str = model_specs.hosting_artifact_key

    return instance_specific_hosting_artifact_key or default_hosting_artifact_key


def _retrieve_training_artifact_key(model_specs: JumpStartModelSpecs, instance_type: str) -> str:
    """Returns instance specific training artifact key or default one as fallback."""
    instance_specific_training_artifact_key: Optional[str] = (
        model_specs.training_instance_type_variants.get_instance_specific_artifact_key(
            instance_type=instance_type
        )
        if instance_type
        and getattr(model_specs, "training_instance_type_variants", None) is not None
        else None
    )

    default_training_artifact_key: str = model_specs.training_artifact_key

    return instance_specific_training_artifact_key or default_training_artifact_key


def _retrieve_model_uri(
    model_id: str,
    model_version: str,
    model_scope: Optional[str] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
):
    """Retrieves the model artifact S3 URI for the model matching the given arguments.

    Optionally uses a bucket override specified by environment variable.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to retrieve
            the model artifact S3 URI.
        model_version (str): Version of the JumpStart model for which to retrieve the model
            artifact S3 URI.
        model_scope (str): The model type, i.e. what it is used for.
            Valid values: "training" and "inference".
        instance_type (str): The ML compute instance type for the specified scope. (Default: None).
        region (str): Region for which to retrieve model S3 URI. (Default: None).
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
    Returns:
        str: the model artifact S3 URI for the corresponding model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=model_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    model_artifact_key: str

    if model_scope == JumpStartScriptScope.INFERENCE:

        is_prepacked = not model_specs.use_inference_script_uri()

        model_artifact_key = (
            _retrieve_hosting_prepacked_artifact_key(model_specs, instance_type)
            if is_prepacked
            else _retrieve_hosting_artifact_key(model_specs, instance_type)
        )

    elif model_scope == JumpStartScriptScope.TRAINING:

        model_artifact_key = _retrieve_training_artifact_key(model_specs, instance_type)

    bucket = os.environ.get(
        ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE
    ) or get_jumpstart_content_bucket(region)

    model_s3_uri = f"s3://{bucket}/{model_artifact_key}"

    return model_s3_uri


def _model_supports_training_model_uri(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> bool:
    """Returns True if the model supports training with model uri field.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the support status for model uri with training.
        model_version (str): Version of the JumpStart model for which to retrieve the
            support status for model uri with training.
        region (Optional[str]): Region for which to retrieve the
            support status for model uri with training.
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
    Returns:
        bool: the support status for model uri with training.
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

    return model_specs.use_training_model_artifact()
