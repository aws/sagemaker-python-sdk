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
from typing import Dict, Optional
from sagemaker import image_uris
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
    INFERENCE,
    TRAINING,
    SUPPORTED_JUMPSTART_SCOPES,
    ModelFramework,
    VariableScope,
)
from sagemaker.jumpstart.utils import get_jumpstart_content_bucket
from sagemaker.jumpstart import accessors as jumpstart_accessors


def _retrieve_image_uri(
    model_id: str,
    model_version: str,
    image_scope: str,
    framework: Optional[str],
    region: Optional[str],
    version: Optional[str],
    py_version: Optional[str],
    instance_type: Optional[str],
    accelerator_type: Optional[str],
    container_version: Optional[str],
    distribution: Optional[str],
    base_framework_version: Optional[str],
    training_compiler_config: Optional[str],
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
        region (str): The AWS region.
        version (str): The framework or algorithm version. This is required if there is
            more than one supported version for the given framework or algorithm.
        py_version (str): The Python version. This is required if there is
            more than one supported Python version for the given framework version.
        instance_type (str): The SageMaker instance type. For supported types, see
            https://aws.amazon.com/sagemaker/pricing/instance-types. This is required if
            there are different images for different processor types.
        accelerator_type (str): Elastic Inference accelerator type. For more, see
            https://docs.aws.amazon.com/sagemaker/latest/dg/ei.html.
        container_version (str): the version of docker image.
            Ideally the value of parameter should be created inside the framework.
            For custom use, see the list of supported container versions:
            https://github.com/aws/deep-learning-containers/blob/master/available_images.md.
        distribution (dict): A dictionary with information on how to run distributed training
        training_compiler_config (:class:`~sagemaker.training_compiler.TrainingCompilerConfig`):
            A configuration class for the SageMaker Training Compiler.

    Returns:
        str: the ECR URI for the corresponding SageMaker Docker image.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    assert region is not None

    if image_scope is None:
        raise ValueError(
            "Must specify `image_scope` argument to retrieve image uri for JumpStart models."
        )
    if image_scope not in SUPPORTED_JUMPSTART_SCOPES:
        raise ValueError(
            f"JumpStart models only support scopes: {', '.join(SUPPORTED_JUMPSTART_SCOPES)}."
        )

    model_specs = jumpstart_accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )

    if image_scope == INFERENCE:
        ecr_specs = model_specs.hosting_ecr_specs
    elif image_scope == TRAINING:
        if not model_specs.training_supported:
            raise ValueError(
                f"JumpStart model ID '{model_id}' and version '{model_version}' "
                "does not support training."
            )
        assert model_specs.training_ecr_specs is not None
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
    if ecr_specs.framework == ModelFramework.HUGGINGFACE.value:
        base_framework_version_override = ecr_specs.framework_version
        version_override = ecr_specs.huggingface_transformers_version

    if image_scope == TRAINING:
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
    model_scope: Optional[str],
    region: Optional[str],
):
    """Retrieves the model artifact S3 URI for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to retrieve
            the model artifact S3 URI.
        model_version (str): Version of the JumpStart model for which to retrieve the model
            artifact S3 URI.
        model_scope (str): The model type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (str): Region for which to retrieve model S3 URI.
    Returns:
        str: the model artifact S3 URI for the corresponding model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    assert region is not None

    if model_scope is None:
        raise ValueError(
            "Must specify `model_scope` argument to retrieve model "
            "artifact uri for JumpStart models."
        )

    if model_scope not in SUPPORTED_JUMPSTART_SCOPES:
        raise ValueError(
            f"JumpStart models only support scopes: {', '.join(SUPPORTED_JUMPSTART_SCOPES)}."
        )

    model_specs = jumpstart_accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )
    if model_scope == INFERENCE:
        model_artifact_key = model_specs.hosting_artifact_key
    elif model_scope == TRAINING:
        if not model_specs.training_supported:
            raise ValueError(
                f"JumpStart model ID '{model_id}' and version '{model_version}' "
                "does not support training."
            )
        assert model_specs.training_artifact_key is not None
        model_artifact_key = model_specs.training_artifact_key

    bucket = get_jumpstart_content_bucket(region)

    model_s3_uri = f"s3://{bucket}/{model_artifact_key}"

    return model_s3_uri


def _retrieve_script_uri(
    model_id: str,
    model_version: str,
    script_scope: Optional[str],
    region: Optional[str],
):
    """Retrieves the script S3 URI associated with the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the script S3 URI.
        model_version (str): Version of the JumpStart model for which to
            retrieve the model script S3 URI.
        script_scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (str): Region for which to retrieve model script S3 URI.
    Returns:
        str: the model script URI for the corresponding model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    assert region is not None

    if script_scope is None:
        raise ValueError(
            "Must specify `script_scope` argument to retrieve model script uri for "
            "JumpStart models."
        )

    if script_scope not in SUPPORTED_JUMPSTART_SCOPES:
        raise ValueError(
            f"JumpStart models only support scopes: {', '.join(SUPPORTED_JUMPSTART_SCOPES)}."
        )

    model_specs = jumpstart_accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )
    if script_scope == INFERENCE:
        model_script_key = model_specs.hosting_script_key
    elif script_scope == TRAINING:
        if not model_specs.training_supported:
            raise ValueError(
                f"JumpStart model ID '{model_id}' and version '{model_version}' "
                "does not support training."
            )
        assert model_specs.training_script_key is not None
        model_script_key = model_specs.training_script_key

    bucket = get_jumpstart_content_bucket(region)

    script_s3_uri = f"s3://{bucket}/{model_script_key}"

    return script_s3_uri


def _retrieve_default_hyperparameters(
    model_id: str,
    model_version: str,
    region: Optional[str],
    include_container_hyperparameters: bool = False,
):
    """Retrieves the training hyperparameters for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default hyperparameters.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default hyperparameters.
        region (str): Region for which to retrieve default hyperparameters.
        include_container_hyperparameters (bool): True if container hyperparameters
            should be returned as well. Container hyperparameters are not used to tune
            the specific algorithm, but rather by SageMaker Training to setup
            the training container environment. For example, there is a container hyperparameter
            that indicates the entrypoint script to use. These hyperparameters may be required
            when creating a training job with boto3, however the ``Estimator`` classes
            should take care of adding container hyperparameters to the job. (Default: False).
    Returns:
        dict: the hyperparameters to use for the model.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    assert region is not None

    model_specs = jumpstart_accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
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
    region: Optional[str],
):
    """Retrieves the inference environment variables for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default environment variables.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default environment variables.
        region (Optional[str]): Region for which to retrieve default environment variables.

    Returns:
        dict: the inference environment variables to use for the model.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = jumpstart_accessors.JumpStartModelsAccessor.get_model_specs(
        region=region, model_id=model_id, version=model_version
    )

    default_environment_variables: Dict[str, str] = {}
    for environment_variable in model_specs.inference_environment_variables:
        default_environment_variables[environment_variable.name] = str(environment_variable.default)
    return default_environment_variables
