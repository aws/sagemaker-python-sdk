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
"""Utility function to capture local environment"""
from __future__ import absolute_import

import logging
import subprocess
import sys
from typing import Optional

import boto3
import docker
import yaml

logger = logging.getLogger(__name__)

REQUIREMENT_TXT_PATH = "/tmp/requirements.txt"
ENVIRONMENT_YML_PATH = "/tmp/environment.yml"
DOCKERFILE_PATH = "/tmp/Dockerfile"

CONDA_DOCKERFILE_TEMPLATE = """
FROM {base_image_name}
ADD environment.yml .

# Install prerequisites for conda
RUN apt-get update && \
    apt-get install -y wget bzip2 ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 && \
    apt-get clean

# Download and install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Initialize Conda
ENV PATH=/opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda && \
    conda config --add channels conda-forge

# Create a conda environment from the environment.yml file
RUN conda env create -f environment.yml -n {env_name}

# Activate the conda environment
RUN conda run -n {env_name}
"""

PIP_DOCKERFILE_TEMPLATE = """
FROM {base_image_name}
ADD requirements.txt .

# Create a virtual environment
RUN python -m venv {env_name}

# Activate the virtual environment
RUN . {env_name}/bin/activate

RUN pip install --no-cache-dir -r requirements.txt
"""


def capture_local_environment(
    image_name: str = "sm-local-capture",
    env_name: str = "saved_local_env",
    package_manager: str = "pip",
    deploy_to_ecr: bool = False,
    base_image_name: Optional[str] = None,
    job_conda_env: Optional[str] = None,
    additional_dependencies: Optional[str] = None,
    ecr_repo_name: Optional[str] = None,
    boto_session: Optional[boto3.Session] = None,
):
    """Capture all dependency packages installed in the local environment and build a docker image.

    When using this utility method, the docker daemon must be active in the environment.
    Please note that this is an experimental feature. This utility function is not be able to
    detect the package compatability between platforms. It is also not able to detect dependency
    conflicts between the local environment and the additional dependencies.

    Args:
        image_name (str): The name of the docker image.
        env_name (str): The name of the virtual environment to be activated in the image,
            defaults to "saved_local_env".
        package_manager (str): The package manager, must be one of "conda" or "pip".
        deploy_to_ecr (bool): Whether to deploy the docker image to AWS ECR, defaults to False.
            If set to True, the AWS credentials must be configured in the environment.
        base_image_name (Optional[str]): If provided will be used as the base image, else the
            utility will evaluate from local environment in following manner:
                1. If package manager is conda, it will use ubuntu:latest.
                2. If package manager is pip, it is resolved to base python image with the same
                    python version as the environment running the local code.
        job_conda_env (Optional[str]): If set, the dependencies will be captured from this specific
            conda Env, otherwise the dependencies will be the installed packages in the current
            active environment. This parameter is only valid when the package manager is conda.
        additional_dependencies (Optional[str]): Either the path to a dependencies file (conda
            environment.yml OR pip requirements.txt file). Regardless of this setting utility will
            automatically generate the dependencies file corresponding to the current active
            environment’s snapshot. In addition to this, additional dependencies is configurable.
        ecr_repo_name (Optional[str]): The AWS ECR repo to push the docker image. If not specified,
            it will use image_name as the ECR repo name. This parameter is only valid when
            deploy_to_ecr is True.
        boto_session (Optional[boto3.Session]): The boto3 session with AWS account info. If not
            provided, a new boto session will be created.

    Exceptions:
        docker.errors.DockerException: Error while fetching server API version:
            The docker engine is not running in your environment.
        docker.errors.BuildError: The docker failed to build the image. The most likely reason is:
            1) Some packages are not supported in the base image. 2) There are dependency conflicts
            between your local environment and additional dependencies.
        botocore.exceptions.ClientError: AWS credentials are not configured.
    """

    if package_manager == "conda":
        if job_conda_env:
            subprocess.run(
                f"conda env export -n {job_conda_env} > {ENVIRONMENT_YML_PATH} --no-builds",
                shell=True,
                check=True,
            )
        else:
            subprocess.run(
                f"conda env export > {ENVIRONMENT_YML_PATH} --no-builds", shell=True, check=True
            )

        if additional_dependencies:
            if not additional_dependencies.endswith(
                ".yml"
            ) and not additional_dependencies.endswith(".txt"):
                raise ValueError(
                    "When package manager is conda, additional dependencies "
                    "file must be a yml file or a txt file."
                )
            if additional_dependencies.endswith(".yml"):
                _merge_environment_ymls(
                    env_name,
                    ENVIRONMENT_YML_PATH,
                    additional_dependencies,
                    ENVIRONMENT_YML_PATH,
                )
            elif additional_dependencies.endswith(".txt"):
                _merge_environment_yml_with_requirement_txt(
                    env_name,
                    ENVIRONMENT_YML_PATH,
                    additional_dependencies,
                    ENVIRONMENT_YML_PATH,
                )

        if not base_image_name:
            base_image_name = "ubuntu:latest"
        dockerfile_contents = CONDA_DOCKERFILE_TEMPLATE.format(
            base_image_name=base_image_name,
            env_name=env_name,
        )
    elif package_manager == "pip":
        subprocess.run(f"pip list --format=freeze > {REQUIREMENT_TXT_PATH}", shell=True, check=True)

        if additional_dependencies:
            if not additional_dependencies.endswith(".txt"):
                raise ValueError(
                    "When package manager is pip, additional dependencies file must be a txt file."
                )
            with open(additional_dependencies, "r") as f:
                additional_requirements = f.read()
            with open(REQUIREMENT_TXT_PATH, "a") as f:
                f.write(additional_requirements)
                logger.info("Merged requirements file saved to %s", REQUIREMENT_TXT_PATH)

            if not base_image_name:
                version = sys.version_info
                base_image_name = f"python:{version.major}.{version.minor}.{version.micro}"
            dockerfile_contents = PIP_DOCKERFILE_TEMPLATE.format(
                base_image_name=base_image_name,
                env_name=env_name,
            )

    else:
        raise ValueError(
            "The provided package manager is not supported. "
            "Use conda or pip as the package manager."
        )

    # Create the Dockerfile
    with open(DOCKERFILE_PATH, "w") as f:
        f.write(dockerfile_contents)

    client = docker.from_env()
    _, logs = client.images.build(
        path="/tmp",
        dockerfile=DOCKERFILE_PATH,
        rm=True,
        tag=image_name,
    )
    for log in logs:
        logger.info(log.get("stream", "").strip())
    logger.info("Docker image %s built successfully", image_name)

    if deploy_to_ecr:
        if boto_session is None:
            boto_session = boto3.Session()
        _push_image_to_ecr(image_name, ecr_repo_name, boto_session)


def _merge_environment_ymls(env_name: str, env_file1: str, env_file2: str, output_file: str):
    """Merge two environment.yml files and save to a new environment.yml file.

    Args:
        env_name (str): The name of the virtual environment to be activated in the image.
        env_file1 (str): The path of the first environment.yml file.
        env_file2 (str): The path of the second environment.yml file.
        output_file (str): The path of the output environment.yml file.
    """

    # Load the YAML files
    with open(env_file1, "r") as f:
        env1 = yaml.safe_load(f)
    with open(env_file2, "r") as f:
        env2 = yaml.safe_load(f)

    # Combine dependencies and channels from both files
    dependencies = []
    pip_dependencies = []
    channels = set()

    for env in [env1, env2]:
        if "dependencies" in env:
            for dep in env["dependencies"]:
                if isinstance(dep, str):
                    # Conda package, e.g., 'python=3.7'
                    dependencies.append(dep)
                elif isinstance(dep, dict):
                    # Pip package list, e.g., {'pip': ['requests>=2.22.0']}
                    for pip_package in dep.get("pip", []):
                        pip_dependencies.append(pip_package)
        if "channels" in env:
            channels.update(env["channels"])

    if pip_dependencies:
        dependencies.append({"pip": pip_dependencies})
    # Create the merged environment file
    merged_env = {"name": env_name, "channels": list(channels), "dependencies": dependencies}

    with open(output_file, "w") as f:
        yaml.dump(merged_env, f, sort_keys=False)

    logger.info("Merged environment file saved to '%s'", output_file)


def _merge_environment_yml_with_requirement_txt(
    env_name: str, env_file: str, req_txt: str, output_file: str
):
    """Merge an environment.yml file with a requirements.txt file.

    Args:
        env_name (str): The name of the virtual environment to be activated in the image.
        env_file (str): The path of the environment.yml file.
        req_txt (str): The path of the requirements.txt file.
        output_file (str): The path of the output environment.yml file.
    """
    # Load the files
    with open(env_file, "r") as f:
        env = yaml.safe_load(f)
    with open(req_txt, "r") as f:
        requirements = f.read().splitlines()
    # Combine pip dependencies from both files
    dependencies = []
    pip_dependencies = []

    if "dependencies" in env:
        for dep in env["dependencies"]:
            if isinstance(dep, str):
                # Conda package, e.g., 'python=3.7'
                dependencies.append(dep)
            elif isinstance(dep, dict):
                # Pip package list, e.g., {'pip': ['requests>=2.22.0']}
                for pip_package in dep.get("pip", []):
                    pip_dependencies.append(pip_package)

    for req in requirements:
        if req and not req.startswith("#"):
            pip_dependencies.append(req)

    if pip_dependencies:
        dependencies.append({"pip": pip_dependencies})
    # Create the merged environment file
    merged_env = {"name": env_name, "channels": env["channels"], "dependencies": dependencies}

    with open(output_file, "w") as f:
        yaml.dump(merged_env, f, sort_keys=False)

    logger.info("Merged environment file saved to '%s'", output_file)


def _push_image_to_ecr(image_name: str, ecr_repo_name: str, boto_session: Optional[boto3.Session]):
    """Push the docker image to AWS ECR.

    Args:
        image_name (str): The name of the docker image.
        ecr_repo_name (str): The AWS ECR repo to push the docker image.
    """
    region = boto_session.region_name
    aws_account_id = boto_session.client("sts", region_name=region).get_caller_identity()["Account"]
    ecr_client = boto3.client("ecr")

    # Authenticate Docker with ECR
    registry_url = f"{aws_account_id}.dkr.ecr.{region}.amazonaws.com"
    docker_login_cmd = (
        f"aws ecr get-login-password --region {region} "
        f"| docker login --username AWS --password-stdin {aws_account_id}.dkr.ecr.{region}.amazonaws.com"
    )
    subprocess.run(docker_login_cmd, shell=True, check=True)

    # Create a new ECR repository (if it doesn't already exist)
    ecr_repo_name = ecr_repo_name or image_name
    try:
        ecr_client.create_repository(repositoryName=ecr_repo_name)
    except ecr_client.exceptions.RepositoryAlreadyExistsException:
        pass

    # Tag the local Docker image
    ecr_image_uri = f"{registry_url}/{ecr_repo_name}:latest"
    docker_tag_cmd = f"docker tag {image_name}:latest {ecr_image_uri}"
    subprocess.run(docker_tag_cmd, shell=True, check=True)

    # Push the Docker image to ECR
    docker_push_cmd = f"docker push {ecr_image_uri}"
    subprocess.run(docker_push_cmd, shell=True, check=True)

    logger.info("Image %s pushed to %s", image_name, ecr_image_uri)
