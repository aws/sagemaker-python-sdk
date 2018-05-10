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

import pytest
import urllib3

from botocore.exceptions import ClientError
from mock import patch

import sagemaker


OK_RESPONSE = urllib3.HTTPResponse()
OK_RESPONSE.status = 200

BAD_RESPONSE = urllib3.HTTPResponse()
BAD_RESPONSE.status = 502


@patch('sagemaker.local.image._SageMakerContainer.train', return_value="/some/path/to/model")
@patch('sagemaker.local.local_session.LocalSession')
def test_create_training_job(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {'TrainingImage': image}
    input_data_config = [
        {
            'ChannelName': 'a',
            'DataSource': {
                'S3DataSource': {
                    'S3DataDistributionType': 'FullyReplicated',
                    'S3Uri': 's3://my_bucket/tmp/source1'
                }
            }
        },
        {
            'ChannelName': 'b',
            'DataSource': {
                'FileDataSource': {
                    'FileDataDistributionType': 'FullyReplicated',
                    'FileUri': 'file:///tmp/source1'
                }
            }
        }
    ]
    output_data_config = {}
    resource_config = {'InstanceType': 'local', 'InstanceCount': instance_count}
    hyperparameters = {'a': 1, 'b': 'bee'}

    local_sagemaker_client.create_training_job("my-training-job", algo_spec, 'arn:my-role', input_data_config,
                                               output_data_config, resource_config, None, hyperparameters)

    train_container = local_sagemaker_client.train_container
    assert train_container is not None
    assert train_container.image == image
    assert train_container.instance_count == instance_count

    expected = {
        'ResourceConfig': {'InstanceCount': instance_count},
        'TrainingJobStatus': 'Completed',
        'ModelArtifacts': {'S3ModelArtifacts': "/some/path/to/model"}
    }

    response = local_sagemaker_client.describe_training_job("my-training-job")

    assert response['TrainingJobStatus'] == expected['TrainingJobStatus']
    assert response['ResourceConfig']['InstanceCount'] == expected['ResourceConfig']['InstanceCount']
    assert response['ModelArtifacts']['S3ModelArtifacts'] == expected['ModelArtifacts']['S3ModelArtifacts']


@patch('sagemaker.local.image._SageMakerContainer.train', return_value="/some/path/to/model")
@patch('sagemaker.local.local_session.LocalSession')
def test_create_training_job_invalid_data_source(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {'TrainingImage': image}

    # InvalidDataSource is not supported. S3DataSource and FileDataSource are currently the only
    # valid Data Sources. We expect a ValueError if we pass this input data config.
    input_data_config = [{
        'ChannelName': 'a',
        'DataSource': {
            'InvalidDataSource': {
                'FileDataDistributionType': 'FullyReplicated',
                'FileUri': 'ftp://myserver.com/tmp/source1'
            }
        }
    }]

    output_data_config = {}
    resource_config = {'InstanceType': 'local', 'InstanceCount': instance_count}
    hyperparameters = {'a': 1, 'b': 'bee'}

    with pytest.raises(ValueError):
        local_sagemaker_client.create_training_job("my-training-job", algo_spec, 'arn:my-role', input_data_config,
                                                   output_data_config, resource_config, None, hyperparameters)


@patch('sagemaker.local.image._SageMakerContainer.train', return_value="/some/path/to/model")
@patch('sagemaker.local.local_session.LocalSession')
def test_create_training_job_not_fully_replicated(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {'TrainingImage': image}

    # Local Mode only supports FullyReplicated as Data Distribution type.
    input_data_config = [{
        'ChannelName': 'a',
        'DataSource': {
            'S3DataSource': {
                'S3DataDistributionType': 'ShardedByS3Key',
                'S3Uri': 's3://my_bucket/tmp/source1'
            }
        }
    }]

    output_data_config = {}
    resource_config = {'InstanceType': 'local', 'InstanceCount': instance_count}
    hyperparameters = {'a': 1, 'b': 'bee'}

    with pytest.raises(RuntimeError):
        local_sagemaker_client.create_training_job("my-training-job", algo_spec, 'arn:my-role', input_data_config,
                                                   output_data_config, resource_config, None, hyperparameters)


@patch('sagemaker.local.local_session.LocalSession')
def test_create_model(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    model_name = "my-model"
    primary_container = {'ModelDataUrl': '/some/model/path', 'Environment': {'env1': 1, 'env2': 'b'}}
    execution_role_arn = 'arn:aws:iam::111111111111:role/ExpandedRole'

    local_sagemaker_client.create_model(model_name, primary_container, execution_role_arn)

    assert local_sagemaker_client.model_name == model_name
    assert local_sagemaker_client.primary_container == primary_container
    assert local_sagemaker_client.role_arn == execution_role_arn


@patch('sagemaker.local.local_session.LocalSession')
def test_describe_endpoint_config(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    # No Endpoint Config Created
    with pytest.raises(ClientError):
        local_sagemaker_client.describe_endpoint_config('my-endpoint-config')

    local_sagemaker_client.created_endpoint = True
    assert local_sagemaker_client.describe_endpoint_config('my-endpoint-config')


@patch('sagemaker.local.local_session.LocalSession')
def test_create_endpoint_config(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    production_variants = [{'InstanceType': 'ml.c4.99xlarge', 'InitialInstanceCount': 10}]
    local_sagemaker_client.create_endpoint_config('my-endpoint-config', production_variants)

    assert local_sagemaker_client.variants == production_variants


@patch('sagemaker.local.local_session.LocalSession')
def test_describe_endpoint(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    response = local_sagemaker_client.describe_endpoint('my-endpoint')
    assert 'EndpointStatus' in response


@patch('sagemaker.local.image._SageMakerContainer.serve')
@patch('urllib3.PoolManager.request', return_value=OK_RESPONSE)
@patch('sagemaker.local.local_session.LocalSession')
def test_create_endpoint(serve, request, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    local_sagemaker_client.variants = [{'InstanceType': 'ml.c4.99xlarge', 'InitialInstanceCount': 10}]
    local_sagemaker_client.primary_container = {'ModelDataUrl': '/some/model/path',
                                                'Environment': {'env1': 1, 'env2': 'b'},
                                                'Image': 'my-image:1.0'}

    local_sagemaker_client.create_endpoint('my-endpoint', 'some-endpoint-config')

    assert local_sagemaker_client.created_endpoint


@patch('sagemaker.local.image._SageMakerContainer.serve')
@patch('urllib3.PoolManager.request', return_value=BAD_RESPONSE)
@patch('sagemaker.local.local_session.LocalSession')
def test_create_endpoint_fails(serve, request, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    local_sagemaker_client.variants = [{'InstanceType': 'ml.c4.99xlarge', 'InitialInstanceCount': 10}]
    local_sagemaker_client.primary_container = {'ModelDataUrl': '/some/model/path',
                                                'Environment': {'env1': 1, 'env2': 'b'},
                                                'Image': 'my-image:1.0'}

    with pytest.raises(RuntimeError):
        local_sagemaker_client.create_endpoint('my-endpoint', 'some-endpoint-config')


def test_file_input_all_defaults():
    prefix = 'pre'
    actual = sagemaker.local.local_session.file_input(fileUri=prefix)
    expected = \
        {
            'DataSource': {
                'FileDataSource': {
                    'FileDataDistributionType': 'FullyReplicated',
                    'FileUri': prefix
                }
            }
        }
    assert actual.config == expected


def test_file_input_content_type():
    prefix = 'pre'
    actual = sagemaker.local.local_session.file_input(fileUri=prefix, content_type='text/csv')
    expected = \
        {
            'DataSource': {
                'FileDataSource': {
                    'FileDataDistributionType': 'FullyReplicated',
                    'FileUri': prefix
                }
            },
            'ContentType': 'text/csv'
        }
    assert actual.config == expected
