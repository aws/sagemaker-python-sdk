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
"""Shared pytest fixtures for sagemaker-mlops integration tests."""
from __future__ import absolute_import

import json
import os
import os as _os
import sys

import boto3
import pytest
from botocore.config import Config

from sagemaker.core.helper.session_helper import Session, expand_role
from sagemaker.core import image_uris
from sagemaker.core.workflow.pipeline_context import PipelineSession

# Shared container-build helpers (file-locked, xdist-safe).
# Loaded via importlib to avoid collision with the local `tests` package.
import importlib.util as _importlib_util

_container_build_path = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), "..", "..", "..", "tests", "integ_helpers", "container_build.py")
)
_spec = _importlib_util.spec_from_file_location("integ_helpers.container_build", _container_build_path)
_container_build = _importlib_util.module_from_spec(_spec)
_spec.loader.exec_module(_container_build)

DOCKERFILE_TEMPLATE = _container_build.DOCKERFILE_TEMPLATE
DOCKERFILE_TEMPLATE_WITH_CONDA = _container_build.DOCKERFILE_TEMPLATE_WITH_CONDA
DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR = _container_build.DOCKERFILE_TEMPLATE_WITH_USER_AND_WORKDIR
build_sdk_tar_once = _container_build.build_sdk_tar_once
build_container_once = _container_build.build_container_once

DEFAULT_REGION = "us-east-1"
CUSTOM_S3_OBJECT_KEY_PREFIX = "session-default-prefix"

CONDA_YML_FILE_TEMPLATE = (
    "name: integ_test_env\n"
    "channels:\n"
    "  - defaults\n"
    "dependencies:\n"
    "  - requests=2.32.3\n"
    "  - charset-normalizer=3.3.2\n"
    "  - scipy=1.13.1\n"
    "prefix: /opt/conda/bin/conda\n"
)


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption("--sagemaker-client-config", action="store", default=None)
    parser.addoption("--boto-config", action="store", default=None)


def pytest_configure(config):
    bc = config.getoption("--boto-config")
    parsed = json.loads(bc) if bc else {}
    region = parsed.get("region_name", boto3.session.Session().region_name)
    if region:
        os.environ["TEST_AWS_REGION_NAME"] = region


# ---------------------------------------------------------------------------
# Core session fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sagemaker_client_config(request):
    config = request.config.getoption("--sagemaker-client-config")
    return json.loads(config) if config else dict()


@pytest.fixture(scope="session")
def boto_session(request):
    config = request.config.getoption("--boto-config")
    if config:
        return boto3.Session(**json.loads(config))
    else:
        return boto3.Session(region_name=os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION))


@pytest.fixture(scope="session")
def sagemaker_session(sagemaker_client_config, boto_session):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_config={},
        default_bucket_prefix=CUSTOM_S3_OBJECT_KEY_PREFIX,
    )


@pytest.fixture(scope="session")
def pipeline_session(boto_session):
    return PipelineSession(boto_session=boto_session)


# ---------------------------------------------------------------------------
# Workflow-scoped session (isolated to prevent race conditions with other tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sagemaker_session_for_pipeline(sagemaker_client_config, boto_session):
    """Separate SageMaker session scoped to the module to avoid settings race conditions."""
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_config={},
        default_bucket_prefix=CUSTOM_S3_OBJECT_KEY_PREFIX,
    )


@pytest.fixture(scope="module")
def role(sagemaker_session_for_pipeline):
    return expand_role(sagemaker_session_for_pipeline, "SageMakerRole")


@pytest.fixture(scope="module")
def region_name(sagemaker_session_for_pipeline):
    return sagemaker_session_for_pipeline.boto_session.region_name


# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_data_dir():
    return os.path.join(os.path.dirname(__file__), "data")


@pytest.fixture(scope="session")
def test_code_dir():
    return os.path.join(os.path.dirname(__file__), "code")


# ---------------------------------------------------------------------------
# Framework version fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sklearn_latest_version():
    """Return the latest SKLearn framework version available.

    Some image_uri config version keys are not PEP 440 compliant (for
    example "1.4-2-py312", which encodes a Python scope in the version
    string). Those keys cannot be parsed by ``packaging.version.Version``
    and would raise ``InvalidVersion`` during sorting, so they are skipped.
    """
    from packaging.version import InvalidVersion, Version

    config = image_uris.config_for_framework("sklearn")
    if "versions" not in config:
        config = next(iter(config.values()))

    parseable_versions = []
    for version in config["versions"].keys():
        try:
            parseable_versions.append((Version(version), version))
        except InvalidVersion:
            continue

    if not parseable_versions:
        raise ValueError("No PEP 440 compliant SKLearn versions found in config")

    return max(parseable_versions, key=lambda item: item[0])[1]


# ---------------------------------------------------------------------------
# Python version fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def compatible_python_version():
    return "{}.{}".format(sys.version_info.major, sys.version_info.minor)


# ---------------------------------------------------------------------------
# SDK tar — built once, shared across all xdist workers via file lock
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sagemaker_sdk_tar_path(tmp_path_factory):
    """Build the sagemaker-mlops sdist exactly once across all xdist workers."""
    return build_sdk_tar_once(tmp_path_factory)


# ---------------------------------------------------------------------------
# Container fixtures — each image built & pushed once, ECR URI cached on disk
# ---------------------------------------------------------------------------

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
def dummy_container_with_conda(sagemaker_session, compatible_python_version,
                                sagemaker_sdk_tar_path, tmp_path_factory):
    return build_container_once(
        "dummy_container_with_conda",
        sagemaker_session, compatible_python_version,
        DOCKERFILE_TEMPLATE_WITH_CONDA, sagemaker_sdk_tar_path, tmp_path_factory,
    )


@pytest.fixture(scope="session")
def conda_env_yml():
    """Write a temporary conda yml file and yield its path; clean up afterwards."""
    conda_yml_file_name = "conda_env.yml"
    conda_file_path = os.path.join(os.getcwd(), conda_yml_file_name)
    with open(conda_file_path, "w") as yml_file:
        yml_file.writelines(CONDA_YML_FILE_TEMPLATE)
    yield conda_file_path
    if os.path.isfile(conda_yml_file_name):
        os.remove(conda_yml_file_name)
