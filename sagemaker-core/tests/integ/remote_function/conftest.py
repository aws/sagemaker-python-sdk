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
import re
import shutil
import subprocess
import sys
from contextlib import contextmanager

import docker
import filelock
import pytest
from docker.errors import BuildError

from sagemaker.core.common_utils import sagemaker_timestamp

REPO_ACCOUNT_ID = "033110030271"

REPO_NAME = "remote-function-dummy-container"

DOCKERFILE_TEMPLATE = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip'"
    " -o 'awscliv2.zip' \
        && unzip awscliv2.zip \
        && ./aws/install\n\n"
    "COPY {source_archive} ./\n"
    "RUN pip3 install --no-cache-dir '{source_archive}'\n"
    "RUN rm {source_archive}\n"
)

DOCKERFILE_TEMPLATE_WITH_CONDA = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    'SHELL ["/bin/bash", "-c"]\n'
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl -L -O "
    "'https://github.com/conda-forge/miniforge/releases/download/24.11.3-2/"
    "Miniforge3-Linux-x86_64.sh' \
        && bash Miniforge3-Linux-x86_64.sh -b -p '/opt/conda' \
        && /opt/conda/bin/conda init bash\n\n"
    "ENV PATH $PATH:/opt/conda/bin\n"
    "RUN mamba create -n integ_test_env python={py_version} -y \
        && mamba create -n default_env python={py_version} -y\n"
    "COPY {source_archive} ./\n"
    "RUN pip install --no-cache-dir '{source_archive}' \
        && mamba run -n base pip install --no-cache-dir '{source_archive}' \
        && mamba run -n default_env pip install --no-cache-dir '{source_archive}' \
        && mamba run -n integ_test_env pip install --no-cache-dir '{source_archive}'\n"
    "ENV SHELL=/bin/bash\n"
    "ENV SAGEMAKER_JOB_CONDA_ENV=default_env\n"
)

DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip'"
    " -o 'awscliv2.zip' \
        && unzip awscliv2.zip \
        && ./aws/install\n\n"
    "RUN apt install -y sudo\n"
    "RUN useradd -ms /bin/bash integ-test-user\n"
    "RUN usermod -aG sudo integ-test-user\n"
    "RUN echo '%sudo ALL= (ALL) NOPASSWD:ALL' >> /etc/sudoers\n"
    "USER integ-test-user\n"
    "WORKDIR /home/integ-test-user\n"
    "COPY {source_archive} ./\n"
    "RUN pip install --no-cache-dir '{source_archive}'\n"
    "RUN rm {source_archive}\n"
)

AUTO_CAPTURE_CLIENT_DOCKER_TEMPLATE = (
    "FROM public.ecr.aws/docker/library/python:{py_version}-slim\n\n"
    'SHELL ["/bin/bash", "-c"]\n'
    "RUN apt-get update -y \
        && apt-get install -y unzip curl\n\n"
    "RUN curl -L -O "
    "'https://github.com/conda-forge/miniforge/releases/latest/download/"
    "Miniforge3-Linux-x86_64.sh' \
        && bash Miniforge3-Linux-x86_64.sh -b -p '/opt/conda' \
        && /opt/conda/bin/conda init bash\n\n"
    "ENV PATH $PATH:/opt/conda/bin\n"
    "COPY {source_archive} ./\n"
    "RUN mamba create -n auto_capture_client python={py_version} -y \
        && mamba run -n auto_capture_client pip install --no-cache-dir '{source_archive}' awscli boto3\n"
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
    "    - /sagemaker_core-{sagemaker_version}.tar.gz\n"
    "prefix: /opt/conda/bin/conda\n"
)


@pytest.fixture(scope="session")
def compatible_python_version():
    return "{}.{}".format(sys.version_info.major, sys.version_info.minor)


@pytest.fixture(scope="session")
def incompatible_python_version():
    return "{}.{}".format(sys.version_info.major, sys.version_info.minor + 1)


@pytest.fixture(scope="session")
def sagemaker_session():
    import boto3
    from sagemaker.core.helper.session_helper import Session

    boto_session = boto3.session.Session(
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    )
    return Session(boto_session=boto_session)


@pytest.fixture(scope="session")
def cpu_instance_type():
    return "ml.m5.large"


@pytest.fixture(scope="session")
def gpu_instance_type():
    return "ml.g4dn.xlarge"


@pytest.fixture(scope="session")
def dummy_container_without_error(sagemaker_session, compatible_python_version, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "dummy_container_without_error", sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_with_user_and_workdir(sagemaker_session, compatible_python_version, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "dummy_container_with_user_and_workdir", sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_incompatible_python_runtime(sagemaker_session, incompatible_python_version, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "dummy_container_incompatible_python_runtime", sagemaker_session, incompatible_python_version,
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_with_conda(sagemaker_session, compatible_python_version, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "dummy_container_with_conda", sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE_WITH_CONDA, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def auto_capture_test_container(sagemaker_session, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "auto_capture_test_container", sagemaker_session, "3.10",
        AUTO_CAPTURE_CLIENT_DOCKER_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
        is_auto_capture=True,
    )


@pytest.fixture(scope="session")
def spark_test_container(sagemaker_session, sagemaker_sdk_tar_path, tmp_path_factory):
    return _build_container_once(
        "spark_test_container", sagemaker_session, "3.9",
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def conda_env_yml():
    """Write conda yml file needed for tests."""
    conda_yml_file_name = "conda_env.yml"
    version_path = os.path.join(os.getcwd(), "VERSION")
    if os.path.exists(version_path):
        with open(version_path, "r") as version_file:
            sagemaker_version = version_file.readline().strip()
    else:
        sagemaker_version = "0.0.0"
    conda_file_path = os.path.join(os.getcwd(), conda_yml_file_name)
    with open(conda_file_path, "w") as yml_file:
        yml_file.writelines(CONDA_YML_FILE_TEMPLATE.format(sagemaker_version=sagemaker_version))
    yield conda_file_path
    if os.path.isfile(conda_yml_file_name):
        os.remove(conda_yml_file_name)


@pytest.fixture(scope="session")
def sagemaker_sdk_tar_path(tmp_path_factory):
    """Build the sagemaker-core sdist once and share it across all xdist workers.

    Uses a file lock so only one worker runs the build; others wait and reuse
    the already-built tar.gz from the shared temp directory.
    """
    # tmp_path_factory.getbasetemp().parent is shared across all xdist workers
    root_tmp = tmp_path_factory.getbasetemp().parent
    tar_dir = root_tmp / "sagemaker_sdk_tar"
    tar_dir.mkdir(exist_ok=True)
    lock_file = root_tmp / "sagemaker_sdk_tar.lock"

    with filelock.FileLock(str(lock_file)):
        existing = list(tar_dir.glob("*.tar.gz"))
        if not existing:
            _generate_sagemaker_sdk_tar(str(tar_dir))
            existing = list(tar_dir.glob("*.tar.gz"))
    return str(existing[0])


def _tmpdir():
    """Create a temporary directory context manager."""
    import tempfile

    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


_tmpdir = contextmanager(_tmpdir)


def _build_container_once(
    fixture_name, sagemaker_session, py_version, docker_template, sdk_tar_path,
    tmp_path_factory, is_auto_capture=False,
):
    """Build and push a container image exactly once across all xdist workers.

    Uses a file lock keyed by fixture_name so parallel workers wait for the
    first worker to finish, then reuse the ECR URI written to a shared file.
    """
    root_tmp = tmp_path_factory.getbasetemp().parent
    uri_file = root_tmp / f"{fixture_name}.ecr_uri"
    lock_file = root_tmp / f"{fixture_name}.lock"

    with filelock.FileLock(str(lock_file)):
        if uri_file.exists():
            return uri_file.read_text().strip()
        if is_auto_capture:
            ecr_uri = _build_auto_capture_client_container(
                py_version, docker_template, sdk_tar_path
            )
        else:
            ecr_uri = _build_container(sagemaker_session, py_version, docker_template, sdk_tar_path)
        uri_file.write_text(ecr_uri)
    return ecr_uri


def _build_container(sagemaker_session, py_version, docker_template, sdk_tar_path):
    """Build a dummy test container locally and push to ECR."""
    region = sagemaker_session.boto_region_name
    image_tag = f"{py_version.replace('.', '-')}-{sagemaker_timestamp()}"
    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    with _tmpdir() as tmpdir:
        print("building docker image locally in ", tmpdir)
        source_archive = os.path.basename(sdk_tar_path)
        shutil.copy2(sdk_tar_path, os.path.join(tmpdir, source_archive))
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as file:
            content = docker_template.format(py_version=py_version, source_archive=source_archive)
            print(f"Dockerfile contents: \n{content}\n")
            file.writelines(content)

        docker_client = docker.from_env()
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
        from sagemaker.core.common_utils import sts_regional_endpoint

        sts_client = sagemaker_session.boto_session.client(
            "sts",
            region_name=region,
            endpoint_url=sts_regional_endpoint(region),
        )
        account_id = sts_client.get_caller_identity()["Account"]
        ecr_image = _ecr_image_uri(account_id, region, REPO_NAME, image_tag)
    else:
        ecr_image = _ecr_image_uri(REPO_ACCOUNT_ID, region, REPO_NAME, image_tag)

    image.tag(ecr_image, tag=image_tag)
    docker_client.images.push(ecr_image, auth_config={"username": username, "password": password})
    return ecr_image


def _build_auto_capture_client_container(py_version, docker_template, sdk_tar_path):
    """Build a test docker container for auto_capture tests."""
    with _tmpdir() as tmpdir:
        source_archive = os.path.basename(sdk_tar_path)
        shutil.copy2(sdk_tar_path, os.path.join(tmpdir, source_archive))
        _move_auto_capture_test_file(tmpdir)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as file:
            content = docker_template.format(py_version=py_version, source_archive=source_archive)
            file.writelines(content)
        docker_client = docker.from_env()
        image, build_logs = docker_client.images.build(path=tmpdir, tag=REPO_NAME, rm=True)
        return image.id


def _is_repository_exists(ecr_client, repo_name):
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        return True
    except ecr_client.exceptions.RepositoryNotFoundException:
        return False


def _ecr_login(ecr_client):
    """Get login credentials for an ECR client."""
    login = ecr_client.get_authorization_token()
    b64token = login["authorizationData"][0]["authorizationToken"].encode("utf-8")
    username, password = base64.b64decode(b64token).decode("utf-8").split(":")
    return username, password


def _ecr_image_uri(account, region, image_name, tag):
    """Build an ECR image URI."""
    return "{}.dkr.ecr.{}.amazonaws.com/{}:{}".format(account, region, image_name, tag)


def _generate_sagemaker_sdk_tar(destination_folder):
    """Run build to generate the SDK tar file."""
    command = f"python -m build --sdist -o {destination_folder}"
    try:
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error when building sagemaker-core sdist: {e.stderr}")
        raise
    destination_folder_contents = os.listdir(destination_folder)
    source_archive = [f for f in destination_folder_contents if f.endswith("tar.gz")][0]
    return source_archive


def _generate_sdk_tar_with_public_version(destination_folder):
    """Generate SDK tar with public version for auto capture tests."""
    dist_folder_path = "dist"
    version_path = os.path.join(os.getcwd(), "VERSION")
    if not os.path.exists(version_path):
        return _generate_sagemaker_sdk_tar(destination_folder)

    with open(version_path, "r+") as version_file:
        dev_sagemaker_version = version_file.readline().strip()
        public_sagemaker_version = re.sub("1.dev0", "0", dev_sagemaker_version)
        version_file.seek(0)
        version_file.write(public_sagemaker_version)
        version_file.truncate()
    if os.path.exists(dist_folder_path):
        shutil.rmtree(dist_folder_path)

    source_archive = _generate_sagemaker_sdk_tar(destination_folder)

    with open(version_path, "r+") as version_file:
        version_file.seek(0)
        version_file.write(dev_sagemaker_version)
        version_file.truncate()
    if os.path.exists(dist_folder_path):
        shutil.rmtree(dist_folder_path)
    return source_archive


def _move_auto_capture_test_file(destination_folder):
    """Move the auto capture test file to the build folder."""
    source_path = os.path.join(os.path.dirname(__file__), "test_auto_capture.py")
    destination_path = os.path.join(destination_folder, "test_auto_capture.py")
    shutil.copy2(source_path, destination_path)
