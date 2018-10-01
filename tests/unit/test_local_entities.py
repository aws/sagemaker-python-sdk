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

import pytest

from mock import patch, Mock

import sagemaker.local


@pytest.fixture(scope='session')
def local_transform_job(sagemaker_local_session):
    with patch('sagemaker.local.local_session.LocalSagemakerClient.describe_model') as describe_model:
        describe_model.return_value = {'PrimaryContainer': {'Environment': {}}}
        job = sagemaker.local.entities._LocalTransformJob('my-transform-job', 'some-model', sagemaker_local_session)
        return job


@patch('sagemaker.local.local_session.LocalSagemakerClient.describe_model', Mock(return_value={'PrimaryContainer': {}}))
def test_local_transform_job_init(sagemaker_local_session):
    job = sagemaker.local.entities._LocalTransformJob('my-transform-job', 'some-model', sagemaker_local_session)
    assert job.name == 'my-transform-job'
    assert job.state == sagemaker.local.entities._LocalTransformJob._CREATING


def test_local_transform_job_container_environment(local_transform_job):
    transform_kwargs = {
        'MaxPayloadInMB': 3,
        'BatchStrategy': 'SingleRecord',
    }
    container_env = local_transform_job._get_container_environment(**transform_kwargs)

    assert 'SAGEMAKER_BATCH' in container_env
    assert 'SAGEMAKER_MAX_PAYLOAD_IN_MB' in container_env
    assert 'SAGEMAKER_BATCH_STRATEGY' in container_env
    assert 'SAGEMAKER_MAX_CONCURRENT_TRANSFORMS' in container_env

    transform_kwargs = {
        'Environment': {'MY_ENV': 3}
    }

    container_env = local_transform_job._get_container_environment(**transform_kwargs)

    assert 'SAGEMAKER_BATCH' in container_env
    assert 'SAGEMAKER_MAX_PAYLOAD_IN_MB' not in container_env
    assert 'SAGEMAKER_BATCH_STRATEGY' not in container_env
    assert 'SAGEMAKER_MAX_CONCURRENT_TRANSFORMS' in container_env
    assert 'MY_ENV' in container_env
