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
from __future__ import absolute_import

import base64
import os
import subprocess
import shutil
import pytest
import docker

from sagemaker.utils import sagemaker_timestamp, _tmpdir, sts_regional_endpoint

REPO_ACCOUNT_ID = "033110030271"

REPO_NAME = "remote-function-dummy-container"

DOCKERFILE_TEMPLATE = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' \
        && unzip awscliv2.zip \
        && ./aws/install\n\n"
    "COPY {source_archive} ./\n"
    "RUN pip3 install '{source_archive}'\n"
    "RUN rm {source_archive}\n"
)

DOCKERFILE_TEMPLATE_WITH_CONDA = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    'SHELL ["/bin/bash", "-c"]\n'
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl -L -O 'https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh' \
        && bash Mambaforge-Linux-x86_64.sh -b -p '/opt/conda' \
        && /opt/conda/bin/conda init bash\n\n"
    "ENV PATH $PATH:/opt/conda/bin\n"
    "RUN mamba create -n integ_test_env python={py_version} -y \
        && mamba create -n default_env python={py_version} -y\n"
    "COPY {source_archive} ./\n"
    "RUN pip install '{source_archive}' \
        && mamba run -n base pip install '{source_archive}' \
        && mamba run -n default_env pip install '{source_archive}' \
        && mamba run -n integ_test_env pip install '{source_archive}'\n"
    "ENV SHELL=/bin/bash\n"
    "ENV SAGEMAKER_JOB_CONDA_ENV=default_env\n"
)

CONDA_YML_FILE_TEMPLATE = (
    "name: integ_test_env\n"
    "channels:\n"
    "  - defaults\n"
    "dependencies:\n"
    "  - scipy=1.7.3\n"
    "  - pip:\n"
    "    - /sagemaker-{sagemaker_version}.tar.gz\n"
    "prefix: /opt/conda/bin/conda\n"
)

CONDA_YML_FILE_WITH_SM_FROM_INPUT_CHANNEL = (
    "name: integ_test_env\n"
    "channels:\n"
    "  - defaults\n"
    "dependencies:\n"
    "  - scipy=1.7.3\n"
    "  - pip:\n"
    "    - sagemaker-2.132.1.dev0-py2.py3-none-any.whl\n"
    "prefix: /opt/conda/bin/conda\n"
)


@pytest.fixture(scope="package")
def dummy_container_without_error(sagemaker_session):
    # TODO: the python version should be dynamically specified instead of hardcoding
    ecr_uri = _build_container(sagemaker_session, "3.7", DOCKERFILE_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="package")
def dummy_container_incompatible_python_runtime(sagemaker_session):
    ecr_uri = _build_container(sagemaker_session, "3.10", DOCKERFILE_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="package")
def dummy_container_with_conda(sagemaker_session):
    ecr_uri = _build_container(sagemaker_session, "3.7", DOCKERFILE_TEMPLATE_WITH_CONDA)
    return ecr_uri


@pytest.fixture(scope="package")
def conda_env_yml():
    """Write conda yml file needed for tests"""

    conda_yml_file_name = "conda_env.yml"
    with open(os.path.join(os.getcwd(), "VERSION"), "r") as version_file:
        sagemaker_version = version_file.readline().strip()
    conda_file_path = os.path.join(os.getcwd(), conda_yml_file_name)
    with open(conda_file_path, "w") as yml_file:
        yml_file.writelines(CONDA_YML_FILE_TEMPLATE.format(sagemaker_version=sagemaker_version))
    yield conda_file_path

    # cleanup
    if os.path.isfile(conda_yml_file_name):
        os.remove(conda_yml_file_name)


@pytest.fixture(scope="package")
def conda_yml_file_sm_from_input_channel():
    """Write conda yml file needed for tests"""

    conda_yml_file_name = "conda_env_sm_from_input_channel.yml"
    conda_file_path = os.path.join(os.getcwd(), conda_yml_file_name)

    with open(conda_file_path, "w") as yml_file:
        yml_file.writelines(CONDA_YML_FILE_WITH_SM_FROM_INPUT_CHANNEL)
    yield conda_file_path

    # cleanup
    if os.path.isfile(conda_yml_file_name):
        os.remove(conda_yml_file_name)


def _build_container(sagemaker_session, py_version, docker_templete):
    """Build a dummy test container locally and push a container to an ecr repo"""

    region = sagemaker_session.boto_region_name
    image_tag = f"{py_version.replace('.', '-')}-{sagemaker_timestamp()}"
    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    with _tmpdir() as tmpdir:
        print("building docker image locally in ", tmpdir)
        print("building source archive...")
        source_archive = _generate_and_move_sagemaker_sdk_tar(tmpdir)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as file:
            file.writelines(
                docker_templete.format(py_version=py_version, source_archive=source_archive)
            )

        docker_client = docker.from_env()

        print("building docker image...")
        image, build_logs = docker_client.images.build(path=tmpdir, tag=REPO_NAME, rm=True)

    if _is_repository_exists(ecr_client, REPO_NAME):
        sts_client = sagemaker_session.boto_session.client(
            "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
        )
        account_id = sts_client.get_caller_identity()["Account"]
        # When the test is run locally, repo will exist in same account whose credentials are used to run the test
        ecr_image = _ecr_image_uri(
            account_id, sagemaker_session.boto_region_name, REPO_NAME, image_tag
        )
    else:
        ecr_image = _ecr_image_uri(
            REPO_ACCOUNT_ID,
            sagemaker_session.boto_region_name,
            REPO_NAME,
            image_tag,
        )

    print("pushing image...")
    image.tag(ecr_image, tag=image_tag)
    docker_client.images.push(ecr_image, auth_config={"username": username, "password": password})

    return ecr_image


def _is_repository_exists(ecr_client, repo_name):
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        return False


def _ecr_login(ecr_client):
    """Get a login credentials for an ecr client."""
    login = ecr_client.get_authorization_token()
    b64token = login["authorizationData"][0]["authorizationToken"].encode("utf-8")
    username, password = base64.b64decode(b64token).decode("utf-8").split(":")
    return username, password


def _ecr_image_uri(account, region, image_name, tag):
    """Build an ECR image URI based in account, region and container name"""
    return "{}.dkr.ecr.{}.amazonaws.com/{}:{}".format(account, region, image_name, tag)


def _generate_and_move_sagemaker_sdk_tar(destination_folder):
    """
    Run setup.py sdist to generate the PySDK tar file and
    copy it to appropriate test data folder
    """
    subprocess.run("python3 setup.py sdist", shell=True)
    dist_dir = "dist"
    source_archive = os.listdir(dist_dir)[0]
    source_path = os.path.join(dist_dir, source_archive)
    destination_path = os.path.join(destination_folder, source_archive)
    shutil.copy2(source_path, destination_path)

    return source_archive
