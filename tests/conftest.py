# Copyright 2017-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import json
import os

import boto3
import pytest
import tests.integ
from botocore.config import Config
from packaging.version import Version

from sagemaker import Session, image_uris, utils
from sagemaker.local import LocalSession
from sagemaker.rl import RLEstimator

DEFAULT_REGION = "us-west-2"
CUSTOM_BUCKET_NAME_PREFIX = "sagemaker-custom-bucket"

NO_M4_REGIONS = [
    "eu-west-3",
    "eu-north-1",
    "ap-east-1",
    "ap-northeast-1",  # it has m4.xl, but not enough in all AZs
    "sa-east-1",
    "me-south-1",
]

NO_T2_REGIONS = ["eu-north-1", "ap-east-1", "me-south-1"]


def pytest_addoption(parser):
    parser.addoption("--sagemaker-client-config", action="store", default=None)
    parser.addoption("--sagemaker-runtime-config", action="store", default=None)
    parser.addoption("--boto-config", action="store", default=None)


def pytest_configure(config):
    bc = config.getoption("--boto-config")
    parsed = json.loads(bc) if bc else {}
    region = parsed.get("region_name", boto3.session.Session().region_name)
    if region:
        os.environ["TEST_AWS_REGION_NAME"] = region


@pytest.fixture(scope="session")
def sagemaker_client_config(request):
    config = request.config.getoption("--sagemaker-client-config")
    return json.loads(config) if config else dict()


@pytest.fixture(scope="session")
def sagemaker_runtime_config(request):
    config = request.config.getoption("--sagemaker-runtime-config")
    return json.loads(config) if config else None


@pytest.fixture(scope="session")
def boto_session(request):
    config = request.config.getoption("--boto-config")
    if config:
        return boto3.Session(**json.loads(config))
    else:
        return boto3.Session(region_name=DEFAULT_REGION)


@pytest.fixture(scope="session")
def sagemaker_session(sagemaker_client_config, sagemaker_runtime_config, boto_session):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    runtime_client = (
        boto_session.client("sagemaker-runtime", **sagemaker_runtime_config)
        if sagemaker_runtime_config
        else None
    )

    return Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
    )


@pytest.fixture(scope="session")
def sagemaker_local_session(boto_session):
    return LocalSession(boto_session=boto_session)


@pytest.fixture(scope="module")
def custom_bucket_name(boto_session):
    region = boto_session.region_name
    account = boto_session.client(
        "sts", region_name=region, endpoint_url=utils.sts_regional_endpoint(region)
    ).get_caller_identity()["Account"]
    return "{}-{}-{}".format(CUSTOM_BUCKET_NAME_PREFIX, region, account)


@pytest.fixture(scope="module", params=["py2", "py3"])
def chainer_py_version(request):
    return request.param


# TODO: current version fixtures are legacy fixtures that aren't useful
# and no longer verify whether images are valid
@pytest.fixture(
    scope="module",
    params=[
        "0.12",
        "0.12.1",
        "1.0",
        "1.0.0",
        "1.1",
        "1.1.0",
        "1.2",
        "1.2.1",
        "1.3",
        "1.3.0",
        "1.4",
        "1.4.0",
        "1.4.1",
    ],
)
def mxnet_version(request):
    return request.param


@pytest.fixture(scope="module", params=["py2", "py3"])
def mxnet_py_version(request):
    return request.param


@pytest.fixture(scope="module", params=["0.4", "0.4.0", "1.0", "1.0.0"])
def pytorch_version(request):
    return request.param


@pytest.fixture(scope="module", params=["py2", "py3"])
def pytorch_py_version(request):
    return request.param


@pytest.fixture(scope="module", params=["0.20.0"])
def sklearn_version(request):
    return request.param


@pytest.fixture(scope="module")
def xgboost_framework_version(xgboost_version):
    if xgboost_version in ("1", "latest"):
        pytest.skip("Skipping XGBoost algorithm version.")
    return xgboost_version


@pytest.fixture(scope="module")
def tf_version(tensorflow_training_version):
    # TODO: remove this fixture and update tests
    if tensorflow_training_version in ("1.13.1", "2.2", "2.2.0"):
        pytest.skip("version isn't compatible with both training and inference.")
    return tensorflow_training_version


@pytest.fixture(scope="module", params=["py2", "py3"])
def tf_py_version(tensorflow_training_version, request):
    version = Version(tensorflow_training_version)
    if version < Version("1.11"):
        return "py2"
    if version < Version("2.2"):
        return request.param
    return "py37"


@pytest.fixture(scope="module", params=["0.10.1", "0.10.1", "0.11", "0.11.0", "0.11.1"])
def rl_coach_tf_version(request):
    return request.param


@pytest.fixture(scope="module", params=["0.11", "0.11.0"])
def rl_coach_mxnet_version(request):
    return request.param


@pytest.fixture(scope="module", params=["0.5", "0.5.3", "0.6", "0.6.5"])
def rl_ray_version(request):
    return request.param


@pytest.fixture(scope="module")
def mxnet_full_version():
    return "1.6.0"


@pytest.fixture(scope="module")
def mxnet_full_py_version():
    return "py3"


@pytest.fixture(scope="module")
def ei_mxnet_full_version():
    return "1.5.1"


@pytest.fixture(scope="module")
def pytorch_full_version():
    return "1.5.0"


@pytest.fixture(scope="module")
def pytorch_full_py_version():
    return "py3"


@pytest.fixture(scope="module")
def pytorch_full_ei_version():
    return "1.3.1"


@pytest.fixture(scope="module")
def rl_coach_mxnet_full_version():
    return RLEstimator.COACH_LATEST_VERSION_MXNET


@pytest.fixture(scope="module")
def rl_coach_tf_full_version():
    return RLEstimator.COACH_LATEST_VERSION_TF


@pytest.fixture(scope="module")
def rl_ray_full_version():
    return RLEstimator.RAY_LATEST_VERSION


@pytest.fixture(scope="module")
def sklearn_full_version():
    return "0.20.0"


@pytest.fixture(scope="module")
def sklearn_full_py_version():
    return "py3"


@pytest.fixture(scope="module")
def tf_training_latest_version():
    return "2.2.0"


@pytest.fixture(scope="module")
def tf_training_latest_py_version():
    return "py37"


@pytest.fixture(scope="module")
def tf_serving_latest_version():
    return "2.1.0"


@pytest.fixture(scope="module")
def tf_full_version(tf_training_latest_version, tf_serving_latest_version):
    """Fixture for TF tests that test both training and inference.

    Fixture exists as such, since TF training and TFS have different latest versions.
    Otherwise, this would simply be a single latest version.
    """
    return str(min(Version(tf_training_latest_version), Version(tf_serving_latest_version)))


@pytest.fixture(scope="module")
def tf_full_py_version(tf_full_version):
    """Fixture to match tf_full_version

    Fixture exists as such, since TF training and TFS have different latest versions.
    Otherwise, this would simply be py37 to match the latest version support.
    """
    version = Version(tf_full_version)
    if version < Version("1.11"):
        return "py2"
    if version < Version("2.2"):
        return "py3"
    return "py37"


@pytest.fixture(scope="module")
def ei_tf_full_version():
    return "2.0.0"


@pytest.fixture(scope="module")
def xgboost_full_version():
    return "1.0-1"


@pytest.fixture(scope="module")
def xgboost_full_py_version():
    return "py3"


@pytest.fixture(scope="session")
def cpu_instance_type(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_M4_REGIONS:
        return "ml.m5.xlarge"
    else:
        return "ml.m4.xlarge"


@pytest.fixture(scope="session")
def inf_instance_type(sagemaker_session, request):
    return "ml.inf1.xlarge"


@pytest.fixture(scope="session")
def ec2_instance_type(cpu_instance_type):
    return cpu_instance_type[3:]


@pytest.fixture(scope="session")
def alternative_cpu_instance_type(sagemaker_session, request):
    region = sagemaker_session.boto_session.region_name
    if region in NO_T2_REGIONS:
        # T3 is not supported by hosting yet
        return "ml.c5.xlarge"
    else:
        return "ml.t2.medium"


@pytest.fixture(scope="session")
def cpu_instance_family(cpu_instance_type):
    return "_".join(cpu_instance_type.split(".")[0:2])


@pytest.fixture(scope="session")
def inf_instance_family(inf_instance_type):
    return "_".join(inf_instance_type.split(".")[0:2])


def pytest_generate_tests(metafunc):
    if "instance_type" in metafunc.fixturenames:
        boto_config = metafunc.config.getoption("--boto-config")
        parsed_config = json.loads(boto_config) if boto_config else {}
        region = parsed_config.get("region_name", DEFAULT_REGION)
        cpu_instance_type = "ml.m5.xlarge" if region in NO_M4_REGIONS else "ml.m4.xlarge"

        params = [cpu_instance_type]
        if not (
            region in tests.integ.HOSTING_NO_P2_REGIONS
            or region in tests.integ.TRAINING_NO_P2_REGIONS
        ):
            params.append("ml.p2.xlarge")
        metafunc.parametrize("instance_type", params, scope="session")

    _generate_all_framework_version_fixtures(metafunc)


def _generate_all_framework_version_fixtures(metafunc):
    for fw in ("chainer", "tensorflow", "xgboost"):
        config = image_uris.config_for_framework(fw)
        if "scope" in config:
            _parametrize_framework_version_fixtures(metafunc, fw, config)
        else:
            for image_scope in config.keys():
                _parametrize_framework_version_fixtures(
                    metafunc, "{}_{}".format(fw, image_scope), config[image_scope]
                )


def _parametrize_framework_version_fixtures(metafunc, fixture_prefix, config):
    fixture_name = "{}_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        versions = list(config["versions"].keys()) + list(config.get("version_aliases", {}).keys())
        metafunc.parametrize(fixture_name, versions, scope="session")

    latest_version = sorted(config["versions"].keys(), key=lambda v: Version(v))[-1]

    fixture_name = "{}_latest_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        metafunc.parametrize(fixture_name, (latest_version,), scope="session")

    fixture_name = "{}_latest_py_version".format(fixture_prefix)
    if fixture_name in metafunc.fixturenames:
        config = config["versions"]
        py_versions = config[latest_version].get("py_versions", config[latest_version].keys())
        metafunc.parametrize(fixture_name, (sorted(py_versions)[-1],), scope="session")
