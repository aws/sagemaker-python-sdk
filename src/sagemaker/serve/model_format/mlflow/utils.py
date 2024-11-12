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
"""Holds the util functions used for MLflow model format"""
from __future__ import absolute_import

from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
import logging
import shutil
import os

from sagemaker import Session, image_uris
from sagemaker.serve.utils.types import ModelServer
from sagemaker.serve.detector.image_detector import _cast_to_compatible_version
from sagemaker.serve.model_format.mlflow.constants import (
    MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP,
    MLFLOW_PYFUNC,
    FLAVORS_WITH_FRAMEWORK_SPECIFIC_DLC_SUPPORT,
    DEFAULT_FW_USED_FOR_DEFAULT_IMAGE,
    DEFAULT_PYTORCH_VERSION,
    MLFLOW_METADATA_FILE,
    MLFLOW_PIP_DEPENDENCY_FILE,
    FLAVORS_DEFAULT_WITH_TF_SERVING,
    TENSORFLOW_SAVED_MODEL_NAME,
)

logger = logging.getLogger(__name__)


def _get_default_model_server_for_mlflow(deployment_flavor: str) -> ModelServer:
    """Get the default model server for mlflow based on deployment flavor.

    Args:
        deployment_flavor (str): The flavor mlflow model will be deployed with.

    Returns:
        str: The model server chosen for given model flavor.
    """
    if deployment_flavor in FLAVORS_DEFAULT_WITH_TF_SERVING:
        return ModelServer.TENSORFLOW_SERVING
    return ModelServer.TORCHSERVE


def _get_default_image_for_mlflow(python_version: str, region: str, instance_type: str) -> str:
    """Retrieves the default Docker image URI for MLflow deployments based on the specified Python

    version, AWS region, and instance type.

    Args:
        python_version (str): The Python version used in the environment,
            in the format 'major.minor.patch' (e.g., '3.8.6').
        region (str): The AWS region where the deployment is intended (e.g., 'us-east-1').
        instance_type (str): The type of AWS instance on which the deployment is intended.

    Returns:
        str: The URI of the default Docker image that matches the specified criteria.

    Raises:
        ValueError: If the function fails to retrieve a default Docker image URI based on the
            provided arguments.
    """
    major, minor, _ = python_version.split(".")
    shortened_py_version = f"py{major}{minor}"
    default_dlc = None
    # TODO: Dynamically getting fw version after beta
    try:
        default_dlc = image_uris.retrieve(
            framework=DEFAULT_FW_USED_FOR_DEFAULT_IMAGE,
            version=DEFAULT_PYTORCH_VERSION.get(shortened_py_version),
            region=region,
            image_scope="inference",
            py_version=shortened_py_version,
            instance_type=instance_type,
        )
    except ValueError:
        pass

    if default_dlc:
        logger.info(
            "Using default image %s based on inferred python version."
            " Proceeding with the the deployment.",
            default_dlc,
        )
        return default_dlc

    raise ValueError(
        f"Unable to find default image based on "
        f"python version {python_version} and region {region}"
    )


def _generate_mlflow_artifact_path(src_folder: str, artifact_name: str) -> str:
    """Generates the path to a specific MLflow model artifacts based on the source folder and

    convention.

    Args:
    - src_folder (str): The source folder where the MLflow model artifacts are stored.
    - artifact_name (str): The artifact name.

    Returns:
    - str: The path to the MLmodel file.
    """
    artifact_path = os.path.join(src_folder, artifact_name)

    if not os.path.isfile(artifact_path):
        raise FileNotFoundError(
            f"The {artifact_name} file does not exist in the specified "
            f"source folder: {src_folder}"
        )
    return artifact_path


def _get_all_flavor_metadata(mlmodel_path: str) -> Optional[Dict[str, Any]]:
    """Validates whether an MLflow MLmodel file is in YAML format and parses the supported flavors.

    Args:
    - mlmodel_path (str): Path to the MLmodel file.

    Returns:
    - Optional[List[str]]: A list of mlflow flavors model was saved with.

    Raises (ValueError): raises ValueError when any of the following scenario happen:
       1. file does not exist
       2. file cannot be parsed as YAML format.
       3. file does not have flavors as key.
    """
    if not os.path.isfile(mlmodel_path):
        raise ValueError(f"File does not exist: {mlmodel_path}")

    try:
        with open(mlmodel_path, "r") as file:
            mlmodel_content = yaml.safe_load(file)

            if "flavors" in mlmodel_content:
                # Extract and return the flavors as a list of keys
                return mlmodel_content["flavors"]
            else:
                raise ValueError("The 'flavors' key is missing in the MLmodel file.")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing the file as YAML: {e}")


def _get_framework_version_from_requirements(flavor: str, requirements_path: str) -> Optional[str]:
    """Checks the version of a given MLflow flavor from a requirements.txt file.

    Args:
    - flavor (str): The MLflow flavor to look for (e.g., "tensorflow", "scikit-learn").
    - requirements_path (str): Path to the requirements.txt file.

    Returns:
    - Optional[str]: The version of the framework if found, None otherwise.

    Raises:
    - ValueError: If the requirements.txt file is not found.
    """

    python_package = MLFLOW_FLAVOR_TO_PYTHON_PACKAGE_MAP.get(flavor)
    if python_package is not None:
        try:
            with open(requirements_path, "r") as file:
                for line in file:
                    if line.startswith(python_package):
                        for separator in ("==", ">=", "<="):
                            if separator in line:
                                version = line.split(separator)[1].strip()
                                return version
                        logger.warning(f"Dependency of flavor {flavor} is not found.")
                        return None
        except FileNotFoundError:
            raise ValueError(f"File not found: {requirements_path}")

    logger.warning(f"Dependency of flavor {flavor} is not found.")
    return None


def _get_deployment_flavor(flavor_metadata: Optional[Dict[str, Any]]) -> str:
    """Get deployment flavor from all parsed flavor metadata.

    Args:
    - flavor_metadata (Optional[Dict[str, Any]]): Dictionary contains mlflow flavors and
        their metadata.

    Returns:
    - str: The flavor mlflow model is saved with. Default to pyfunc if no other flavor is found.
    """

    if not flavor_metadata:
        raise ValueError("Flavor metadata is not found")

    deployment_flavor = MLFLOW_PYFUNC
    for flavor in flavor_metadata:
        if flavor != MLFLOW_PYFUNC:
            deployment_flavor = flavor
            return deployment_flavor
    return deployment_flavor


def _get_python_version_from_parsed_mlflow_model_file(
    parsed_metadata: Dict[str, Any]
) -> Optional[str]:
    """Checks the python version of a given parsed MLflow model file.

    Args:
    - parsed_metadata (Dict[str, Any]): The parsed metadata of a MLflow model.

    Returns:
    - Optional[str]: Python version

    Raises:
    - ValueError: If python_function flavor is not found in metadata.
    """
    pyfunc_metadata = parsed_metadata.get(MLFLOW_PYFUNC)
    if pyfunc_metadata is not None:
        return pyfunc_metadata.get("python_version")
    raise ValueError(f"{MLFLOW_PYFUNC} cannot be found in MLmodel file.")


def _download_s3_artifacts(s3_path: str, dst_path: str, session: Session) -> None:
    """Downloads all artifacts from a specified S3 path to a local destination path.

    Args:
    - s3_path (str): The S3 path to download artifacts from (format: s3://bucket/key).
    - dst_path (str): The local file system path where artifacts should be downloaded.
    - session (Session): A boto3 Session object with configuration.
    """
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path provided. Must start with 's3://'")

    s3_bucket, s3_key = s3_path.replace("s3://", "").split("/", 1)

    s3 = session.boto_session.client("s3")

    os.makedirs(dst_path, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=s3_bucket, Prefix=s3_key):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel_path = os.path.relpath(key, s3_key)
            local_file_path = os.path.join(dst_path, rel_path)

            if not key.endswith("/"):
                local_file_dir = os.path.dirname(local_file_path)
                os.makedirs(local_file_dir, exist_ok=True)

                # Download the file
                logger.info(f"Downloading {key} to {local_file_path}")
                s3.download_file(s3_bucket, key, local_file_path)


def _copy_directory_contents(src_dir, dest_dir) -> None:
    """Copy everything under src_dir to dest_dir preserving the directory structure.

    Args:
        src_dir (str): The source directory path from which to copy the contents.
        dest_dir (str): The destination directory path where the contents of src_dir will be copied.
    """
    src_dir = os.path.normpath(src_dir)
    dest_dir = os.path.normpath(dest_dir)

    if src_dir == dest_dir:
        logger.info("Source and destination directories are the same. No action taken.")
        return

    for root, dirs, files in os.walk(src_dir):
        relative_path = os.path.relpath(root, src_dir)
        dest_path = os.path.join(dest_dir, relative_path)
        normalized_dest_path = os.path.normpath(dest_path)
        os.makedirs(normalized_dest_path, exist_ok=True)

        for file in files:
            file_src_path = os.path.join(root, file)
            file_dest_path = os.path.join(dest_path, file)
            normalized_file_dest_path = os.path.normpath(file_dest_path)
            shutil.copy2(file_src_path, normalized_file_dest_path)


def _select_container_for_mlflow_model(
    mlflow_model_src_path: str,
    deployment_flavor: str,
    region: str,
    instance_type: str,
) -> str:
    """Select framework specific DLC for mlflow model based on flavor, region and isntance type.

    Args:
        - mlflow_model_src_path (str): The local path to mlflow model artifacts.
        - deployment_flavor (str): The flavor mlflow model will be deployed with.
        - region (str): Current region.
        - instance_type (str): The type of AWS instance on which the deployment is intended.

    Returns:
        str: The image uri chosen for mlflow model.
    """
    # TODO: Extend this for model server specific DLC as well, ie. DJL, Triton
    requirement_path = _generate_mlflow_artifact_path(
        mlflow_model_src_path, MLFLOW_PIP_DEPENDENCY_FILE
    )
    mlflow_model_metadata_path = _generate_mlflow_artifact_path(
        mlflow_model_src_path, MLFLOW_METADATA_FILE
    )
    flavor_metadata = _get_all_flavor_metadata(mlflow_model_metadata_path)
    python_version = _get_python_version_from_parsed_mlflow_model_file(flavor_metadata)
    major_python_version, minor_python_version, _ = python_version.split(".")
    casted_python_version = (
        f"py{major_python_version}{minor_python_version}"
        if deployment_flavor == "pytorch"
        else f"py{major_python_version}"
    )

    if deployment_flavor not in FLAVORS_WITH_FRAMEWORK_SPECIFIC_DLC_SUPPORT:
        logger.warning(
            f"{deployment_flavor} flavor currently doesn't have optimized framework "
            f"specific DLC support. Defaulting to generic image..."
        )
        return _get_default_image_for_mlflow(python_version, region, instance_type)

    framework_to_use = FLAVORS_WITH_FRAMEWORK_SPECIFIC_DLC_SUPPORT.get(deployment_flavor)
    framework_version = _get_framework_version_from_requirements(framework_to_use, requirement_path)

    logger.info("Auto-detected deployment flavor is %s", deployment_flavor)
    logger.info("Auto-detected framework to use is %s", framework_to_use)
    logger.info("Auto-detected framework version is %s", framework_version)

    if framework_version is None:
        raise ValueError(
            (
                "Unable to auto detect framework version. Please provide framework %s as part of the "
                "requirements.txt file for deployment flavor %s"
            )
            % (framework_to_use, deployment_flavor)
        )

    casted_versions = (
        _cast_to_compatible_version(framework_to_use, framework_version)
        if framework_version
        else (None,)
    )

    image_uri = None
    for casted_version in casted_versions:
        try:
            image_uri = image_uris.retrieve(
                framework=framework_to_use,
                region=region,
                version=casted_version,
                image_scope="inference",
                py_version=casted_python_version,
                instance_type=instance_type,
            )
            break
        except ValueError:
            pass

    if image_uri:
        logger.info("Auto detected %s. Proceeding with the the deployment.", image_uri)
        if deployment_flavor != "pytorch" and minor_python_version != "8":
            logger.warning(
                f"Image {image_uri} uses python version 3.8. Your python version used "
                f"for model is {python_version}. It is recommended to use the same "
                f"python version to avoid incompatibility."
            )
        return image_uri

    raise ValueError(
        (
            "Unable to auto detect a DLC for framework %s, framework version %s "
            "and python version %s. "
            "Please manually provide image_uri to ModelBuilder()"
        )
        % (deployment_flavor, framework_version, f"py{major_python_version}{minor_python_version}")
    )


def _validate_input_for_mlflow(model_server: ModelServer, deployment_flavor: str) -> None:
    """Validates arguments provided with mlflow models.

    Args:
        - model_server (ModelServer): Model server used for orchestrating mlflow model.
        - deployment_flavor (str): The flavor mlflow model will be deployed with.

    Raises:
    - ValueError: If model server is not torchserve.
    """
    if model_server != ModelServer.TORCHSERVE and model_server != ModelServer.TENSORFLOW_SERVING:
        raise ValueError(
            f"{model_server} is currently not supported for MLflow Model. "
            f"Please choose another model server."
        )
    if (
        model_server == ModelServer.TENSORFLOW_SERVING
        and deployment_flavor not in FLAVORS_DEFAULT_WITH_TF_SERVING
    ):
        raise ValueError(
            "Tensorflow Serving is currently only supported for the following "
            "deployment flavors: {}".format(FLAVORS_DEFAULT_WITH_TF_SERVING)
        )


def _get_saved_model_path_for_tensorflow_and_keras_flavor(model_path: str) -> Optional[str]:
    """Recursively searches for tensorflow saved model.

    Args:
        model_path (str): The root directory to start the search from.

    Returns:
        Optional[str]: The absolute path to the directory containing 'saved_model.pb'.
    """
    for dirpath, dirnames, filenames in os.walk(model_path):
        if TENSORFLOW_SAVED_MODEL_NAME in filenames:
            return os.path.abspath(dirpath)

    return None


def _move_contents(src_dir: Union[str, Path], dest_dir: Union[str, Path]) -> None:
    """Moves all contents of a source directory to a specified destination directory.

    Args:
        src_dir (Union[str, Path]): The path to the source directory.
        dest_dir (Union[str, Path]): The path to the destination directory.

    """
    _src_dir = Path(os.path.normpath(src_dir))
    _dest_dir = Path(os.path.normpath(dest_dir))

    _dest_dir.mkdir(parents=True, exist_ok=True)

    for item in _src_dir.iterdir():
        _dest_path = _dest_dir / item.name
        shutil.move(str(item), str(_dest_path))
