# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import boto3
import pytest

from sagemaker import Session

DEFAULT_REGION = 'us-west-2'


def pytest_addoption(parser):
    parser.addoption('--sagemaker-client-config', action='store', default=None)
    parser.addoption('--sagemaker-runtime-config', action='store', default=None)
    parser.addoption('--boto-config', action='store', default=None)


@pytest.fixture(scope='session')
def sagemaker_client_config(request):
    config = request.config.getoption('--sagemaker-client-config')
    return json.loads(config) if config else None


@pytest.fixture(scope='session')
def sagemaker_runtime_config(request):
    config = request.config.getoption('--sagemaker-runtime-config')
    return json.loads(config) if config else None


@pytest.fixture(scope='session')
def boto_config(request):
    config = request.config.getoption('--boto-config')
    return json.loads(config) if config else None


@pytest.fixture(scope='session')
def sagemaker_session(sagemaker_client_config, sagemaker_runtime_config, boto_config):
    boto_session = boto3.Session(**boto_config) if boto_config else boto3.Session(region_name=DEFAULT_REGION)
    sagemaker_client = boto_session.client('sagemaker', **sagemaker_client_config) if sagemaker_client_config else None
    runtime_client = (boto_session.client('sagemaker-runtime', **sagemaker_runtime_config) if sagemaker_runtime_config
                      else None)

    return Session(boto_session=boto_session,
                   sagemaker_client=sagemaker_client,
                   sagemaker_runtime_client=runtime_client)


@pytest.fixture(scope='module', params=['1.4', '1.4.1', '1.5', '1.5.0', '1.6', '1.6.0'])
def tf_version(request):
    return request.param


@pytest.fixture(scope='module', params=['0.12', '0.12.1', '1.0', '1.0.0', '1.1', '1.1.0'])
def mxnet_version(request):
    return request.param


@pytest.fixture(scope='module', params=['1.4.1', '1.5.0', '1.6.0'])
def tf_full_version(request):
    return request.param


@pytest.fixture(scope='module', params=['0.12.1', '1.0.0', '1.1.0'])
def mxnet_full_version(request):
    return request.param
