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
"""This module contains functions for obtaining JumpStart ECR and S3 URIs."""
from __future__ import absolute_import
from copy import deepcopy
import os
from typing import Dict, List, Optional
from sagemaker import image_uris
from sagemaker.jumpstart.exceptions import NO_AVAILABLE_INSTANCES_ERROR_MSG
from sagemaker.jumpstart.constants import (
    ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE,
    ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    ModelFramework,
    VariableScope,
)
from sagemaker.jumpstart.utils import (
    get_jumpstart_content_bucket,
    verify_model_region_and_return_specs,
)


def _retrieve_image_uri(
    model_id: str,
    model_version: str,
    image_scope: str,
    framework: Optional[str] = None,
    region: Optional[str] = None,
    version: Optional[str] = None,
    py_version: Optional[str] = None,
    instance_type: Optional[str] = None,
    accelerator_type: Optional[str] = None,
    container_version: Optional[str] = None,
    distribution: Optional[str] = None,
    base_framework_version: Optional[str] = None,
    training_compiler_config: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
):
    """Retrieves the container image URI for JumpStart models.

    Only `model_id`, `model_version`, and `image_scope` are required;
    the rest of the fields are auto-populated.


    Args:
        model_id (str): JumpStart model ID for which to retrieve image URI.
        model_version (str): Version of the JumpStart model for which to retrieve
            the image URI.
        image_scope (str): The image type, i.e. what it is used for.
            Valid values: "training", "inference", "eia". If ``accelerator_type`` is set,
            ``image_scope`` is ignored.
        framework (str): The name of the framework or algorithm.
        region (str): The AWS region. (Default: None).
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
            (Default: None).
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.
            (Default: None).
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
            (Default: None).
        container_version (str): the version of docker image.
            Ideally the value of parameter should be created inside the framework.
            For custom use, see the list of supported container versions:
            https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
            (Default: None).
        distribution (dict): A dictionary with information on how to run distributed training.
            (Default: None).
        training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
            A configuration class for the SageMaker Training Compiler.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

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
        scope=image_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    if image_scope == JumpStartScriptScope.INFERENCE:
        ecr_specs = model_specs.hosting_ecr_specs
    elif image_scope == JumpStartScriptScope.TRAINING:
        ecr_specs = model_specs.training_ecr_specs

    if framework is not None and framework != ecr_specs.framework:
        raise ValueError(
            f"Incorrect container framework '{framework}' for JumpStart model ID '{model_id}' "
            f"and version '{model_version}'."
        )

    if version is not None and version != ecr_specs.framework_version:
        raise ValueError(
            f"Incorrect container framework version '{version}' for JumpStart model ID "
            f"'{model_id}' and version '{model_version}'."
        )

    if py_version is not None and py_version != ecr_specs.py_version:
        raise ValueError(
            f"Incorrect python version '{py_version}' for JumpStart model ID '{model_id}' "
            f"and version '{model_version}'."
        )

    base_framework_version_override: Optional[str] = None
    version_override: Optional[str] = None
    if ecr_specs.framework == ModelFramework.HUGGINGFACE:
        base_framework_version_override = ecr_specs.framework_version
        version_override = ecr_specs.huggingface_transformers_version

    if image_scope == JumpStartScriptScope.TRAINING:
        return image_uris.get_training_image_uri(
            region=region,
            framework=ecr_specs.framework,
            framework_version=version_override or ecr_specs.framework_version,
            py_version=ecr_specs.py_version,
            image_uri=None,
            distribution=None,
            compiler_config=None,
            tensorflow_version=None,
            pytorch_version=base_framework_version_override or base_framework_version,
            instance_type=instance_type,
        )
    if base_framework_version_override is not None:
        base_framework_version_override = f"pytorch{base_framework_version_override}"

    return image_uris.retrieve(
        framework=ecr_specs.framework,
        region=region,
        version=version_override or ecr_specs.framework_version,
        py_version=ecr_specs.py_version,
        instance_type=instance_type,
        accelerator_type=accelerator_type,
        image_scope=image_scope,
        container_version=container_version,
        distribution=distribution,
        base_framework_version=base_framework_version_override or base_framework_version,
        training_compiler_config=training_compiler_config,
    )


def _retrieve_model_uri(
    model_id: str,
    model_version: str,
    model_scope: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
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
        region (str): Region for which to retrieve model S3 URI. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
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
    )

    if model_scope == JumpStartScriptScope.INFERENCE:
        model_artifact_key = (
            getattr(model_specs, "hosting_prepacked_artifact_key", None)
            or model_specs.hosting_artifact_key
        )

    elif model_scope == JumpStartScriptScope.TRAINING:
        model_artifact_key = model_specs.training_artifact_key

    bucket = os.environ.get(
        ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE
    ) or get_jumpstart_content_bucket(region)

    model_s3_uri = f"s3://{bucket}/{model_artifact_key}"

    return model_s3_uri


def _retrieve_script_uri(
    model_id: str,
    model_version: str,
    script_scope: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
):
    """Retrieves the script S3 URI associated with the model matching the given arguments.

    Optionally uses a bucket override specified by environment variable.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the script S3 URI.
        model_version (str): Version of the JumpStart model for which to
            retrieve the model script S3 URI.
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
    Returns:
        str: the model script URI for the corresponding model.

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
        scope=script_scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
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


def _retrieve_default_hyperparameters(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    include_container_hyperparameters: bool = False,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
):
    """Retrieves the training hyperparameters for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default hyperparameters.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default hyperparameters.
        region (str): Region for which to retrieve default hyperparameters.
            (Default: None).
        include_container_hyperparameters (bool): True if container hyperparameters
            should be returned as well. Container hyperparameters are not used to tune
            the specific algorithm, but rather by SageMaker Training to setup
            the training container environment. For example, there is a container hyperparameter
            that indicates the entrypoint script to use. These hyperparameters may be required
            when creating a training job with boto3, however the ``Estimator`` classes
            should take care of adding container hyperparameters to the job. (Default: False).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
    Returns:
        dict: the hyperparameters to use for the model.
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
    )

    default_hyperparameters: Dict[str, str] = {}
    for hyperparameter in model_specs.hyperparameters:
        if (
            include_container_hyperparameters and hyperparameter.scope == VariableScope.CONTAINER
        ) or hyperparameter.scope == VariableScope.ALGORITHM:
            default_hyperparameters[hyperparameter.name] = str(hyperparameter.default)
    return default_hyperparameters


def _retrieve_default_environment_variables(
    model_id: str,
    model_version: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
):
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

    Returns:
        dict: the inference environment variables to use for the model.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    default_environment_variables: Dict[str, str] = {}
    for environment_variable in model_specs.inference_environment_variables:
        default_environment_variables[environment_variable.name] = str(environment_variable.default)
    return default_environment_variables


def _retrieve_default_instance_type(
    model_id: str,
    model_version: str,
    scope: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> str:
    """Retrieves the default instance type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default instance type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default instance type.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (Optional[str]): Region for which to retrieve default instance type.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
    Returns:
        str: the default instance type to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of supported computing instances.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        default_instance_type = model_specs.default_inference_instance_type
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
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> List[str]:
    """Retrieves the supported instance types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported instance types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported instance types.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (Optional[str]): Region for which to retrieve supported instance types.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
    Returns:
        list: the supported instance types to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of supported computing instances.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        instance_types = model_specs.supported_inference_instance_types
    elif scope == JumpStartScriptScope.TRAINING:
        instance_types = model_specs.supported_training_instance_types
    else:
        raise NotImplementedError(
            f"Unsupported script scope for retrieving supported instance types: '{scope}'"
        )

    if instance_types is None or len(instance_types) == 0:
        raise ValueError(NO_AVAILABLE_INSTANCES_ERROR_MSG.format(model_id=model_id, region=region))

    return instance_types


def _retrieve_default_training_metric_definitions(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> Optional[List[Dict[str, str]]]:
    """Retrieves the default training metric definitions for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default training metric definitions.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default training metric definitions.
        region (Optional[str]): Region for which to retrieve default training metric
            definitions.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        list: the default training metric definitions to use for the model or None.
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
    )

    return deepcopy(model_specs.metrics) if model_specs.metrics else None
