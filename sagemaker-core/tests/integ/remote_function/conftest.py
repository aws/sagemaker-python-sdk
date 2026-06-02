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

import os
import re
import shutil
import sys
import importlib.util as _importlib_util
import os as _os

import pytest

# ---------------------------------------------------------------------------
# Shared container-build helpers (file-locked, xdist-safe)
# ---------------------------------------------------------------------------
_container_build_path = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "..", "tests", "integ_helpers", "container_build.py")
)
_spec = _importlib_util.spec_from_file_location("integ_helpers.container_build", _container_build_path)
_container_build = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_container_build)

DOCKERFILE_TEMPLATE = _container_build.DOCKERFILE_TEMPLATE
DOCKERFILE_TEMPLATE_WITH_CONDA = _container_build.DOCKERFILE_TEMPLATE_WITH_CONDA
DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR = _container_build.DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR
build_sdk_tar_once = _container_build.build_sdk_tar_once
build_container_once = _container_build.build_container_once

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
def sagemaker_sdk_tar_path(tmp_path_factory):
    """Build the sagemaker-core sdist exactly once across all xdist workers."""
    return build_sdk_tar_once(tmp_path_factory)


@pytest.fixture(scope="session")
def dummy_container_without_error(sagemaker_session, compatible_python_version,
                                   sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "dummy_container_without_error",
        sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_with_user_and_workdir(sagemaker_session, compatible_python_version,
                                           sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "dummy_container_with_user_and_workdir",
        sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_incompatible_python_runtime(sagemaker_session, incompatible_python_version,
                                                 sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "dummy_container_incompatible_python_runtime",
        sagemaker_session, incompatible_python_version,
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def dummy_container_with_conda(sagemaker_session, compatible_python_version,
                                sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "dummy_container_with_conda",
        sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE_WITH_CONDA, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def auto_capture_test_container(sagemaker_session, sagemaker_sdk_tar_path, tmp_path_factory):
    def _copy_auto_capture_test_file(tmpdir):
        source_path = os.path.join(os.path.dirname(__file__), "test_auto_capture.py")
        shutil.copy2(source_path, os.path.join(tmpdir, "test_auto_capture.py"))

    return build_container_once(
        "auto_capture_test_container",
        sagemaker_session, "3.10",
        AUTO_CAPTURE_CLIENT_DOCKER_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
        is_auto_capture=True,
        extra_files_hook=_copy_auto_capture_test_file,
    )


@pytest.fixture(scope="session")
def spark_test_container(sagemaker_session, sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "spark_test_container",
        sagemaker_session, "3.9",
        DOCKERFILE_TEMPLATE, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def spark_pre_execution_commands(sagemaker_session):
    """Build sagemaker-core wheel, upload to S3, and return pre-execution install commands.

    This mirrors the pattern used in sagemaker-mlops feature_processor integ tests.
    The Spark processing image does not have sagemaker-core pre-installed, so we must
    build the local dev wheel and install it in the container via pre_execution_commands.
    """
    import subprocess
    import glob
    import tempfile
    from sagemaker.core.s3 import S3Uploader

    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    )
    core_dir = os.path.join(repo_root, "sagemaker-core")

    with tempfile.TemporaryDirectory() as dist_dir:
        subprocess.run(
            f"python -m build --wheel --outdir {dist_dir}",
            shell=True,
            cwd=core_dir,
            check=True,
        )
        wheels = glob.glob(os.path.join(dist_dir, "sagemaker_core-*.whl"))
        if not wheels:
            raise FileNotFoundError(f"No sagemaker-core wheel found in {dist_dir}")
        wheel_path = wheels[0]
        wheel_name = os.path.basename(wheel_path)

        s3_prefix = "s3://{}/spark-integ-test/wheels".format(
            sagemaker_session.default_bucket()
        )
        S3Uploader.upload(wheel_path, s3_prefix, sagemaker_session=sagemaker_session)

    PIP = "python3 -m pip install --root-user-action=ignore"
    AWS = "python3 -m awscli"
    cmds = [
        f"{PIP} awscli",
        f"{AWS} s3 cp {s3_prefix}/{wheel_name} /tmp/{wheel_name}",
        f"{PIP} /tmp/{wheel_name}",
    ]
    return cmds


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



