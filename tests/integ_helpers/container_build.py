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
"""Shared helpers for building and pushing ECR containers in integration tests.

Both sagemaker-core and sagemaker-mlops integration test suites need to build
the same dummy Docker images and push them to ECR.  This module centralises
that logic so the two conftest.py files can share it without duplication.

Parallel-safety
---------------
When tests are run with pytest-xdist (multiple workers), several workers may
reach the same fixture concurrently.  We use two layers of file locking
(provided by the ``filelock`` package) to ensure:

1. The SDK sdist is built exactly once and the resulting tar.gz is reused by
   all workers (``build_sdk_tar_once``).
2. Each Docker image is built and pushed exactly once; subsequent workers read
   the ECR URI from a small text file written by the first worker
   (``build_container_once``).

Both helpers rely on ``tmp_path_factory.getbasetemp().parent``, which
pytest-xdist guarantees is the *same* directory for every worker in a session.
"""
from __future__ import absolute_import

import base64
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager

import docker
import filelock
from docker.errors import BuildError

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ACCOUNT_ID = "033110030271"
REPO_NAME = "remote-function-dummy-container"

# ---------------------------------------------------------------------------
# Dockerfile templates (identical to those used in both conftest files)
# ---------------------------------------------------------------------------

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
    "RUN echo 'Acquire::Retries \"5\";' > /etc/apt/apt.conf.d/80-retries\n"
    "RUN apt-get update -y \
        && apt-get install -y unzip curl sudo\n\n"
    "RUN curl 'https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip'"
    " -o 'awscliv2.zip' \
        && unzip awscliv2.zip \
        && ./aws/install\n\n"
    "RUN useradd -ms /bin/bash integ-test-user\n"
    "RUN usermod -aG sudo integ-test-user\n"
    "RUN echo '%sudo ALL= (ALL) NOPASSWD:ALL' >> /etc/sudoers\n"
    "USER integ-test-user\n"
    "WORKDIR /home/integ-test-user\n"
    "COPY {source_archive} ./\n"
    "RUN pip install --no-cache-dir '{source_archive}'\n"
    "RUN rm {source_archive}\n"
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_sdk_tar_once(tmp_path_factory):
    """Build the SDK sdist exactly once across all xdist workers.

    Uses a file lock so only the first worker runs ``python -m build``; all
    others wait and then reuse the already-built tar.gz.

    Args:
        tmp_path_factory: pytest's ``tmp_path_factory`` fixture.

    Returns:
        str: Absolute path to the built ``.tar.gz`` file.
    """
    root_tmp = tmp_path_factory.getbasetemp().parent
    tar_dir = root_tmp / "sagemaker_sdk_tar"
    tar_dir.mkdir(exist_ok=True)
    lock_file = root_tmp / "sagemaker_sdk_tar.lock"

    with filelock.FileLock(str(lock_file)):
        existing = list(tar_dir.glob("*.tar.gz"))
        if not existing:
            _generate_sdk_tar(str(tar_dir))
            existing = list(tar_dir.glob("*.tar.gz"))

    return str(existing[0])


def build_container_once(fixture_name, sagemaker_session, py_version, docker_template,
                         sdk_tar_path, tmp_path_factory, is_auto_capture=False,
                         extra_files_hook=None):
    """Build and push a Docker image exactly once across all xdist workers.

    The first worker to acquire the lock builds the image and writes the ECR
    URI to a small text file.  All subsequent workers simply read that file.

    Args:
        fixture_name (str): A unique key for this image (used for lock/URI files).
        sagemaker_session: A SageMaker ``Session`` (or compatible) object.
        py_version (str): Python version string, e.g. ``"3.10"``.
        docker_template (str): Dockerfile template string with ``{py_version}``
            and ``{source_archive}`` placeholders.
        sdk_tar_path (str): Path to the pre-built SDK ``.tar.gz``.
        tmp_path_factory: pytest's ``tmp_path_factory`` fixture.
        is_auto_capture (bool): If True, build a local-only image (no ECR push)
            and return the local image ID instead of an ECR URI.
        extra_files_hook (callable | None): Optional ``fn(tmpdir: str)`` called
            after the SDK tar is copied into the build context but before the
            Docker build runs.  Use this to copy test-specific files (e.g.
            ``test_auto_capture.py``) into the build context.

    Returns:
        str: ECR image URI (or local Docker image ID when ``is_auto_capture=True``).
    """
    root_tmp = tmp_path_factory.getbasetemp().parent
    uri_file = root_tmp / f"{fixture_name}.ecr_uri"
    lock_file = root_tmp / f"{fixture_name}.lock"

    with filelock.FileLock(str(lock_file)):
        if uri_file.exists():
            return uri_file.read_text().strip()
        if is_auto_capture:
            ecr_uri = _build_auto_capture(py_version, docker_template, sdk_tar_path,
                                          extra_files_hook)
        else:
            ecr_uri = _build_and_push(sagemaker_session, py_version, docker_template, sdk_tar_path)
        uri_file.write_text(ecr_uri)

    return ecr_uri


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

@contextmanager
def _tmpdir():
    tmpdir = tempfile.mkdtemp()
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)


def _build_auto_capture(py_version, docker_template, sdk_tar_path, extra_files_hook=None):
    """Build a local-only Docker image for auto-capture tests and return its ID."""
    with _tmpdir() as tmpdir:
        source_archive = os.path.basename(sdk_tar_path)
        shutil.copy2(sdk_tar_path, os.path.join(tmpdir, source_archive))
        if extra_files_hook:
            extra_files_hook(tmpdir)
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(docker_template.format(py_version=py_version, source_archive=source_archive))
        docker_client = docker.from_env()
        image, _ = docker_client.images.build(path=tmpdir, tag=REPO_NAME, rm=True)
        return image.id


def _build_and_push(sagemaker_session, py_version, docker_template, sdk_tar_path):
    """Build a Docker image locally and push it to ECR."""
    region = sagemaker_session.boto_region_name
    ecr_client = sagemaker_session.boto_session.client("ecr")
    username, password = _ecr_login(ecr_client)

    # Import lazily to support both sagemaker-core and sagemaker-mlops layouts.
    try:
        from sagemaker.core.common_utils import sagemaker_timestamp
    except ImportError:
        from sagemaker.utils import sagemaker_timestamp

    image_tag = f"{py_version.replace('.', '-')}-{sagemaker_timestamp()}"

    with _tmpdir() as tmpdir:
        source_archive = os.path.basename(sdk_tar_path)
        shutil.copy2(sdk_tar_path, os.path.join(tmpdir, source_archive))
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(docker_template.format(py_version=py_version, source_archive=source_archive))

        docker_client = docker.from_env()
        try:
            image, _ = docker_client.images.build(
                path=tmpdir, tag=REPO_NAME, rm=True, platform="linux/amd64", pull=True
            )
        except BuildError as e:
            for line in e.build_log:
                if "stream" in line:
                    print(line["stream"].strip())
            raise

    # Resolve the ECR account: use the caller's account if the repo exists
    # there, otherwise fall back to the shared REPO_ACCOUNT_ID.
    if _is_repository_exists(ecr_client, REPO_NAME):
        try:
            from sagemaker.core.common_utils import sts_regional_endpoint
        except ImportError:
            from sagemaker.utils import sts_regional_endpoint

        sts_client = sagemaker_session.boto_session.client(
            "sts", region_name=region, endpoint_url=sts_regional_endpoint(region)
        )
        account_id = sts_client.get_caller_identity()["Account"]
    else:
        account_id = REPO_ACCOUNT_ID

    ecr_image = _ecr_image_uri(account_id, region, REPO_NAME, image_tag)
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
    login = ecr_client.get_authorization_token()
    b64token = login["authorizationData"][0]["authorizationToken"].encode("utf-8")
    username, password = base64.b64decode(b64token).decode("utf-8").split(":")
    return username, password


def _ecr_image_uri(account, region, image_name, tag):
    return "{}.dkr.ecr.{}.amazonaws.com/{}:{}".format(account, region, image_name, tag)


def _generate_sdk_tar(destination_folder):
    """Run ``python -m build --sdist`` and return the archive filename."""
    command = f"python -m build --sdist -o {destination_folder}"
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(result.stderr)
    except subprocess.CalledProcessError as e:
        print(f"SDK sdist build failed:\n{e.stderr}")
        raise
    archives = [f for f in os.listdir(destination_folder) if f.endswith(".tar.gz")]
    return archives[0]
