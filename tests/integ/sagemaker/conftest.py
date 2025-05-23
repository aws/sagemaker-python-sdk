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
import re
import sys

from docker.errors import BuildError

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
    "RUN curl -L -O 'https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/Miniforge3-Linux-x86_64.sh' \
        && bash Miniforge3-Linux-x86_64.sh -b -p '/opt/conda' \
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

DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip' -o 'awscliv2.zip' \
        && unzip awscliv2.zip \
        && ./aws/install\n\n"
    "RUN apt install sudo\n"
    "RUN useradd -ms /bin/bash integ-test-user\n"
    # Add the user to sudo group
    "RUN usermod -aG sudo integ-test-user\n"
    # Ensure passwords are not required for sudo group users
    "RUN echo '%sudo ALL= (ALL) NOPASSWD:ALL' >> /etc/sudoers\n"
    "USER integ-test-user\n"
    "WORKDIR /home/integ-test-user\n"
    "COPY {source_archive} ./\n"
    "RUN pip install '{source_archive}'\n"
    "RUN rm {source_archive}\n"
)

AUTO_CAPTURE_CLIENT_DOCKER_TEMPLATE = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    'SHELL ["/bin/bash", "-c"]\n'
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl -L -O 'https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh' \
        && bash Miniforge3-Miniforge3-Linux-x86_64.sh -b -p '/opt/conda' \
        && /opt/conda/bin/conda init bash\n\n"
    "ENV PATH $PATH:/opt/conda/bin\n"
    "COPY {source_archive} ./\n"
    "RUN mamba create -n auto_capture_client python={py_version} -y \
        && mamba run -n auto_capture_client pip install '{source_archive}' awscli boto3\n"
    "COPY test_auto_capture.py .\n"
    'CMD ["mamba", "run", "-n", "auto_capture_client", "python", "test_auto_capture.py"]\n'
)

CONDA_YML_FILE_TEMPLATE = (
    "name: integ_test_env\n"
    "channels:\n"
    "  - defaults\n"
    "dependencies:\n"
    "  - requests=2.32.3\n"
    "  - charset-normalizer=3.3.2\n"
    "  - scipy=1.13.1\n"
    "  - pip:\n"
    "    - /sagemaker-{sagemaker_version}.tar.gz\n"
    "prefix: /opt/conda/bin/conda\n"
)


@pytest.fixture(scope="session")
def compatible_python_version():
    return "{}.{}".format(sys.version_info.major, sys.version_info.minor)


@pytest.fixture(scope="session")
def incompatible_python_version():
    return "{}.{}".format(sys.version_info.major, sys.version_info.minor + 1)


@pytest.fixture(scope="session")
def dummy_container_without_error(sagemaker_session, compatible_python_version):
    ecr_uri = _build_container(sagemaker_session, compatible_python_version, DOCKERFILE_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="session")
def dummy_container_with_user_and_workdir(sagemaker_session, compatible_python_version):
    ecr_uri = _build_container(
        sagemaker_session, compatible_python_version, DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR
    )
    return ecr_uri


@pytest.fixture(scope="session")
def dummy_container_incompatible_python_runtime(sagemaker_session, incompatible_python_version):
    ecr_uri = _build_container(sagemaker_session, incompatible_python_version, DOCKERFILE_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="session")
def dummy_container_with_conda(sagemaker_session, compatible_python_version):
    ecr_uri = _build_container(
        sagemaker_session, compatible_python_version, DOCKERFILE_TEMPLATE_WITH_CONDA
    )
    return ecr_uri


@pytest.fixture(scope="session")
def auto_capture_test_container(sagemaker_session):
    ecr_uri = _build_auto_capture_client_container("3.10", AUTO_CAPTURE_CLIENT_DOCKER_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="session")
def spark_test_container(sagemaker_session):
    ecr_uri = _build_container("3.9", DOCKERFILE_TEMPLATE)
    return ecr_uri


@pytest.fixture(scope="session")
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


def _build_container(sagemaker_session, py_version, docker_template):
    """Build a dummy test container locally and push a container to an ecr repo"""

    region = sagemaker_session.boto_region_name
    image_tag = f"{py_version.replace('.', '-')}-{sagemaker_timestamp()}"
    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    with _tmpdir() as tmpdir:
        print("building docker image locally in ", tmpdir)
        print("building source archive...")
        source_archive = _generate_sagemaker_sdk_tar(tmpdir)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as file:
            content = docker_template.format(py_version=py_version, source_archive=source_archive)
            print(f"Dockerfile contents: \n{content}\n")
            file.writelines(content)

        docker_client = docker.from_env()

        print("building docker image...")
        # platform is provided to make sure that the image builds correctly across different OS platforms
        try:
            image, build_logs = docker_client.images.build(
                path=tmpdir, tag=REPO_NAME, rm=True, platform="linux/amd64"
            )
        except BuildError as e:
            print("docker build failed!")
            for line in e.build_log:
                if "stream" in line:
                    print(line["stream"].strip())
            raise

    if _is_repository_exists(ecr_client, REPO_NAME):
        print("pushing to session configured account id!")
        sts_client = sagemaker_session.boto_session.client(
            "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
        )
        account_id = sts_client.get_caller_identity()["Account"]
        # When the test is run locally, repo will exist in same account whose credentials are used to run the test
        ecr_image = _ecr_image_uri(
            account_id, sagemaker_session.boto_region_name, REPO_NAME, image_tag
        )
    else:
        print(f"pushing to account id: {REPO_ACCOUNT_ID}")
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


def _build_auto_capture_client_container(py_version, docker_template):
    """Build a test docker container that will act as a client for auto_capture tests"""
    with _tmpdir() as tmpdir:
        print("building docker image locally in ", tmpdir)
        print("building source archive...")
        source_archive = _generate_sdk_tar_with_public_version(tmpdir)
        _move_auto_capture_test_file(tmpdir)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as file:
            content = docker_template.format(py_version=py_version, source_archive=source_archive)
            print(f"Dockerfile contents: \n{content}\n")
            file.writelines(content)

        docker_client = docker.from_env()

        print("building docker image...")
        image, build_logs = docker_client.images.build(path=tmpdir, tag=REPO_NAME, rm=True)
        return image.id


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


def _generate_sagemaker_sdk_tar(destination_folder):
    """
    Run setup.py sdist to generate the PySDK tar file
    """
    command = f"python -m build --sdist -o {destination_folder}"
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, check=True, capture_output=True)
    if result.returncode != 0:
        print(f"Command failed with return code: {result.returncode}")

    print(f"Standard output: {result.stdout.decode()}")
    print(f"Standard error: {result.stderr.decode()}")
    destination_folder_contents = os.listdir(destination_folder)
    source_archive = [file for file in destination_folder_contents if file.endswith("tar.gz")][0]

    return source_archive


def _generate_sdk_tar_with_public_version(destination_folder):
    """
    This function is used for auto capture integ tests. This test need the sagemaker version
    that is already published to PyPI. So we manipulate the current local dev version to change
    latest released SDK version.

    It does the following
    1. Change the dev version of the SDK to the latest published version
    2. Generate SDK tar using that version
    3. Move tar file to the folder when docker file is present
    3. Update the version back to the dev version
    """
    dist_folder_path = "dist"

    with open(os.path.join(os.getcwd(), "VERSION"), "r+") as version_file:
        dev_sagemaker_version = version_file.readline().strip()
        public_sagemaker_version = re.sub("1.dev0", "0", dev_sagemaker_version)
        version_file.seek(0)
        version_file.write(public_sagemaker_version)
        version_file.truncate()
    if os.path.exists(dist_folder_path):
        shutil.rmtree(dist_folder_path)

    source_archive = _generate_sagemaker_sdk_tar(destination_folder)

    with open(os.path.join(os.getcwd(), "VERSION"), "r+") as version_file:
        version_file.seek(0)
        version_file.write(dev_sagemaker_version)
        version_file.truncate()
    if os.path.exists(dist_folder_path):
        shutil.rmtree(dist_folder_path)

    return source_archive


def _move_auto_capture_test_file(destination_folder):
    """
    Move the test file for autocapture tests to a temp folder along with the docker file.
    """

    test_file_name = "remote_function/test_auto_capture.py"
    source_path = os.path.join(
        os.getcwd(), "tests", "integ", "sagemaker", "remote_function", test_file_name
    )
    destination_path = os.path.join(destination_folder, test_file_name)
    shutil.copy2(source_path, destination_path)
