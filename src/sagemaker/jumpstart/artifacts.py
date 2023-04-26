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
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.jumpstart.exceptions import NO_AVAILABLE_INSTANCES_ERROR_MSG
from sagemaker.jumpstart.constants import (
    ACCEPT_TYPE_TO_DESERIALIZER_TYPE_MAP,
    CONTENT_TYPE_TO_SERIALIZER_TYPE_MAP,
    DESERIALIZER_TYPE_TO_CLASS_MAP,
    ENV_VARIABLE_JUMPSTART_MODEL_ARTIFACT_BUCKET_OVERRIDE,
    ENV_VARIABLE_JUMPSTART_SCRIPT_ARTIFACT_BUCKET_OVERRIDE,
    JUMPSTART_DEFAULT_REGION_NAME,
    SERIALIZER_TYPE_TO_CLASS_MAP,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    KwargUseCase,
    MIMEType,
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
    include_aws_sdk_env_vars: bool = True,
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
        if include_aws_sdk_env_vars or (
            not include_aws_sdk_env_vars and environment_variable.required_for_model_class is True
        ):
            default_environment_variables[environment_variable.name] = str(
                environment_variable.default
            )
    return default_environment_variables


def _retrieve_kwargs(
    model_id: str,
    model_version: str,
    use_case: KwargUseCase,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> dict:
    """Retrieves kwargs for `Model`, `Estimator, `Estimator.fit`, and `Model.deploy`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        use_case (KwargUseCase): The use case for which to retrieve kwargs.
        region (Optional[str]): Region for which to retrieve kwargs.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    if use_case in {KwargUseCase.MODEL, KwargUseCase.MODEL_DEPLOY}:
        scope = JumpStartScriptScope.INFERENCE
    elif use_case in {KwargUseCase.ESTIMATOR, KwargUseCase.ESTIMATOR_FIT}:
        scope = JumpStartScriptScope.TRAINING
    else:
        raise ValueError(f"Unsupported named-argument use case: {use_case}")

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    if use_case == KwargUseCase.MODEL:
        return model_specs.model_kwargs

    if use_case == KwargUseCase.MODEL_DEPLOY:
        return model_specs.deploy_kwargs

    if use_case == KwargUseCase.ESTIMATOR:
        return model_specs.estimator_kwargs

    if use_case == KwargUseCase.ESTIMATOR_FIT:
        return model_specs.fit_kwargs

    raise ValueError(f"Unsupported named-argument use case: {use_case}")


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


def _model_supports_prepacked_inference(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> bool:
    """Returns True if the model supports prepacked inference.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the support status for prepacked inference.
        model_version (str): Version of the JumpStart model for which to retrieve the
            support status for prepacked inference.
        region (Optional[str]): Region for which to retrieve the
            support status for prepacked inference.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
    Returns:
        bool: the support status for prepacked inference.
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

    return model_specs.supports_prepacked_inference()


def _retrieve_serializer_from_content_type(
    content_type: MIMEType,
) -> BaseDeserializer:
    """Returns serializer object to use for content type."""

    serializer_type = CONTENT_TYPE_TO_SERIALIZER_TYPE_MAP.get(content_type)

    if serializer_type is None:
        raise RuntimeError(f"Unrecognized content type: {content_type}")

    serializer_handle = SERIALIZER_TYPE_TO_CLASS_MAP.get(serializer_type)

    if serializer_handle is None:
        raise RuntimeError(f"Unrecognized serializer type: {serializer_type}")

    return serializer_handle.__call__()


def _retrieve_deserializer_from_accept_type(
    accept_type: MIMEType,
) -> BaseDeserializer:
    """Returns deserializer object to use for accept type."""

    deserializer_type = ACCEPT_TYPE_TO_DESERIALIZER_TYPE_MAP.get(accept_type)

    if deserializer_type is None:
        raise RuntimeError(f"Unrecognized accept type: {accept_type}")

    deserializer_handle = DESERIALIZER_TYPE_TO_CLASS_MAP.get(deserializer_type)

    if deserializer_handle is None:
        raise RuntimeError(f"Unrecognized deserializer type: {deserializer_type}")

    return deserializer_handle.__call__()


def _retrieve_default_deserializer(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> BaseDeserializer:
    """Retrieves the default deserializer for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default deserializer.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default deserializer.
        region (Optional[str]): Region for which to retrieve default deserializer.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        BaseDeserializer: the default deserializer to use for the model.
    """

    default_accept_type = _retrieve_default_accept_type(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    return _retrieve_deserializer_from_accept_type(MIMEType.from_suffixed_type(default_accept_type))


def _retrieve_default_serializer(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> BaseSerializer:
    """Retrieves the default serializer for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default serializer.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default serializer.
        region (Optional[str]): Region for which to retrieve default serializer.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        BaseSerializer: the default serializer to use for the model.
    """

    default_content_type = _retrieve_default_content_type(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    return _retrieve_serializer_from_content_type(MIMEType.from_suffixed_type(default_content_type))


def _retrieve_deserializer_options(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> List[BaseDeserializer]:
    """Retrieves the supported deserializers for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported deserializers.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported deserializers.
        region (Optional[str]): Region for which to retrieve deserializer options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        List[BaseDeserializer]: the supported deserializers to use for the model.
    """

    supported_accept_types = _retrieve_supported_accept_types(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    seen_classes = set()

    deserializers_with_duplicates = [
        _retrieve_deserializer_from_accept_type(MIMEType.from_suffixed_type(accept_type))
        for accept_type in supported_accept_types
    ]

    deserializers = []

    for deserializer in deserializers_with_duplicates:
        if type(deserializer) not in seen_classes:
            seen_classes.add(type(deserializer))
            deserializers.append(deserializer)

    return deserializers


def _retrieve_serializer_options(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> List[BaseSerializer]:
    """Retrieves the supported serializers for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported serializers.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported serializers.
        region (Optional[str]): Region for which to retrieve serializer options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        List[BaseSerializer]: the supported serializers to use for the model.
    """

    supported_content_types = _retrieve_supported_content_types(
        model_id=model_id,
        model_version=model_version,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    seen_classes = set()

    serializers_with_duplicates = [
        _retrieve_serializer_from_content_type(MIMEType.from_suffixed_type(content_type))
        for content_type in supported_content_types
    ]

    serializers = []

    for serializer in serializers_with_duplicates:
        if type(serializer) not in seen_classes:
            seen_classes.add(type(serializer))
            serializers.append(serializer)

    return serializers


def _retrieve_default_content_type(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> str:
    """Retrieves the default content type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default content type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default content type.
        region (Optional[str]): Region for which to retrieve default content type.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        str: the default content type to use for the model.
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

    default_content_type = model_specs.predictor_specs.default_content_type
    return default_content_type


def _retrieve_default_accept_type(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> str:
    """Retrieves the default accept type for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default accept type.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default accept type.
        region (Optional[str]): Region for which to retrieve default accept type.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        str: the default accept type to use for the model.
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

    default_accept_type = model_specs.predictor_specs.default_accept_type

    return default_accept_type


def _retrieve_supported_accept_types(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> List[str]:
    """Retrieves the supported accept types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported accept types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported accept types.
        region (Optional[str]): Region for which to retrieve accept type options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        list: the supported accept types to use for the model.
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

    supported_accept_types = model_specs.predictor_specs.supported_accept_types

    return supported_accept_types


def _retrieve_supported_content_types(
    model_id: str,
    model_version: str,
    region: Optional[str],
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> List[str]:
    """Retrieves the supported content types for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the supported content types.
        model_version (str): Version of the JumpStart model for which to retrieve the
            supported content types.
        region (Optional[str]): Region for which to retrieve content type options.
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        list: the supported content types to use for the model.
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

    supported_content_types = model_specs.predictor_specs.supported_content_types

    return supported_content_types
