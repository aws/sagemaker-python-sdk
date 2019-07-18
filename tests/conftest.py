# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
from botocore.config import Config

from sagemaker import Session
from sagemaker.chainer import Chainer
from sagemaker.local import LocalSession
from sagemaker.mxnet import MXNet
from sagemaker.pytorch import PyTorch
from sagemaker.rl import RLEstimator
from sagemaker.sklearn.defaults import SKLEARN_VERSION
from sagemaker.tensorflow.estimator import TensorFlow

DEFAULT_REGION = "us-west-2"


def pytest_addoption(parser):
    parser.addoption("--sagemaker-client-config", action="store", default=None)
    parser.addoption("--sagemaker-runtime-config", action="store", default=None)
    parser.addoption("--boto-config", action="store", default=None)
    parser.addoption("--chainer-full-version", action="store", default=Chainer.LATEST_VERSION)
    parser.addoption("--mxnet-full-version", action="store", default=MXNet.LATEST_VERSION)
    parser.addoption("--ei-mxnet-full-version", action="store", default=MXNet.LATEST_VERSION)
    parser.addoption("--pytorch-full-version", action="store", default=PyTorch.LATEST_VERSION)
    parser.addoption(
        "--rl-coach-mxnet-full-version",
        action="store",
        default=RLEstimator.COACH_LATEST_VERSION_MXNET,
    )
    parser.addoption(
        "--rl-coach-tf-full-version", action="store", default=RLEstimator.COACH_LATEST_VERSION_TF
    )
    parser.addoption(
        "--rl-ray-full-version", action="store", default=RLEstimator.RAY_LATEST_VERSION
    )
    parser.addoption("--sklearn-full-version", action="store", default=SKLEARN_VERSION)
    parser.addoption("--tf-full-version", action="store", default=TensorFlow.LATEST_VERSION)
    parser.addoption("--ei-tf-full-version", action="store", default=TensorFlow.LATEST_VERSION)


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
def boto_config(request):
    config = request.config.getoption("--boto-config")
    return json.loads(config) if config else None


@pytest.fixture(scope="session")
def sagemaker_session(sagemaker_client_config, sagemaker_runtime_config, boto_config):
    boto_session = (
        boto3.Session(**boto_config) if boto_config else boto3.Session(region_name=DEFAULT_REGION)
    )
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
def sagemaker_local_session(boto_config):
    if boto_config:
        boto_session = boto3.Session(**boto_config)
    else:
        boto_session = boto3.Session(region_name=DEFAULT_REGION)
    return LocalSession(boto_session=boto_session)


@pytest.fixture(scope="module", params=["4.0", "4.0.0", "4.1", "4.1.0", "5.0", "5.0.0"])
def chainer_version(request):
    return request.param


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


@pytest.fixture(scope="module", params=["0.4", "0.4.0", "1.0", "1.0.0"])
def pytorch_version(request):
    return request.param


@pytest.fixture(scope="module", params=["0.20.0"])
def sklearn_version(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=[
        "1.4",
        "1.4.1",
        "1.5",
        "1.5.0",
        "1.6",
        "1.6.0",
        "1.7",
        "1.7.0",
        "1.8",
        "1.8.0",
        "1.9",
        "1.9.0",
        "1.10",
        "1.10.0",
        "1.11",
        "1.11.0",
        "1.12",
        "1.12.0",
    ],
)
def tf_version(request):
    return request.param


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
def chainer_full_version(request):
    return request.config.getoption("--chainer-full-version")


@pytest.fixture(scope="module")
def mxnet_full_version(request):
    return request.config.getoption("--mxnet-full-version")


@pytest.fixture(scope="module")
def ei_mxnet_full_version(request):
    return request.config.getoption("--ei-mxnet-full-version")


@pytest.fixture(scope="module")
def pytorch_full_version(request):
    return request.config.getoption("--pytorch-full-version")


@pytest.fixture(scope="module")
def rl_coach_mxnet_full_version(request):
    return request.config.getoption("--rl-coach-mxnet-full-version")


@pytest.fixture(scope="module")
def rl_coach_tf_full_version(request):
    return request.config.getoption("--rl-coach-tf-full-version")


@pytest.fixture(scope="module")
def rl_ray_full_version(request):
    return request.config.getoption("--rl-ray-full-version")


@pytest.fixture(scope="module")
def sklearn_full_version(request):
    return request.config.getoption("--sklearn-full-version")


@pytest.fixture(scope="module")
def tf_full_version(request):
    return request.config.getoption("--tf-full-version")


@pytest.fixture(scope="module")
def ei_tf_full_version(request):
    return request.config.getoption("--ei-tf-full-version")
