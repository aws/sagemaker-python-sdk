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
import urllib3

from botocore.exceptions import ClientError
from mock import Mock, patch

import sagemaker


OK_RESPONSE = urllib3.HTTPResponse()
OK_RESPONSE.status = 200

BAD_RESPONSE = urllib3.HTTPResponse()
BAD_RESPONSE.status = 502

ENDPOINT_CONFIG_NAME = "test-endpoint-config"
PRODUCTION_VARIANTS = [{"InstanceType": "ml.c4.99xlarge", "InitialInstanceCount": 10}]

MODEL_NAME = "test-model"
PRIMARY_CONTAINER = {"ModelDataUrl": "/some/model/path", "Environment": {"env1": 1, "env2": "b"}}


@patch("sagemaker.local.image._SageMakerContainer.train", return_value="/some/path/to/model")
@patch("sagemaker.local.local_session.LocalSession")
def test_create_training_job(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {"TrainingImage": image}
    input_data_config = [
        {
            "ChannelName": "a",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "FullyReplicated",
                    "S3Uri": "s3://my_bucket/tmp/source1",
                }
            },
        },
        {
            "ChannelName": "b",
            "DataSource": {
                "FileDataSource": {
                    "FileDataDistributionType": "FullyReplicated",
                    "FileUri": "file:///tmp/source1",
                }
            },
        },
    ]
    output_data_config = {}
    resource_config = {"InstanceType": "local", "InstanceCount": instance_count}
    hyperparameters = {"a": 1, "b": "bee"}

    local_sagemaker_client.create_training_job(
        "my-training-job",
        algo_spec,
        output_data_config,
        resource_config,
        InputDataConfig=input_data_config,
        HyperParameters=hyperparameters,
    )

    expected = {
        "ResourceConfig": {"InstanceCount": instance_count},
        "TrainingJobStatus": "Completed",
        "ModelArtifacts": {"S3ModelArtifacts": "/some/path/to/model"},
    }

    response = local_sagemaker_client.describe_training_job("my-training-job")

    assert response["TrainingJobStatus"] == expected["TrainingJobStatus"]
    assert (
        response["ResourceConfig"]["InstanceCount"] == expected["ResourceConfig"]["InstanceCount"]
    )
    assert (
        response["ModelArtifacts"]["S3ModelArtifacts"]
        == expected["ModelArtifacts"]["S3ModelArtifacts"]
    )


@patch("sagemaker.local.local_session.LocalSession")
def test_describe_invalid_training_job(*args):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    with pytest.raises(ClientError):
        local_sagemaker_client.describe_training_job("i-havent-created-this-job")


@patch("sagemaker.local.image._SageMakerContainer.train", return_value="/some/path/to/model")
@patch("sagemaker.local.local_session.LocalSession")
def test_create_training_job_invalid_data_source(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {"TrainingImage": image}

    # InvalidDataSource is not supported. S3DataSource and FileDataSource are currently the only
    # valid Data Sources. We expect a ValueError if we pass this input data config.
    input_data_config = [
        {
            "ChannelName": "a",
            "DataSource": {
                "InvalidDataSource": {
                    "FileDataDistributionType": "FullyReplicated",
                    "FileUri": "ftp://myserver.com/tmp/source1",
                }
            },
        }
    ]

    output_data_config = {}
    resource_config = {"InstanceType": "local", "InstanceCount": instance_count}
    hyperparameters = {"a": 1, "b": "bee"}

    with pytest.raises(ValueError):
        local_sagemaker_client.create_training_job(
            "my-training-job",
            algo_spec,
            output_data_config,
            resource_config,
            InputDataConfig=input_data_config,
            HyperParameters=hyperparameters,
        )


@patch("sagemaker.local.image._SageMakerContainer.train", return_value="/some/path/to/model")
@patch("sagemaker.local.local_session.LocalSession")
def test_create_training_job_not_fully_replicated(train, LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    instance_count = 2
    image = "my-docker-image:1.0"

    algo_spec = {"TrainingImage": image}

    # Local Mode only supports FullyReplicated as Data Distribution type.
    input_data_config = [
        {
            "ChannelName": "a",
            "DataSource": {
                "S3DataSource": {
                    "S3DataDistributionType": "ShardedByS3Key",
                    "S3Uri": "s3://my_bucket/tmp/source1",
                }
            },
        }
    ]

    output_data_config = {}
    resource_config = {"InstanceType": "local", "InstanceCount": instance_count}
    hyperparameters = {"a": 1, "b": "bee"}

    with pytest.raises(RuntimeError):
        local_sagemaker_client.create_training_job(
            "my-training-job",
            algo_spec,
            output_data_config,
            resource_config,
            InputDataConfig=input_data_config,
            HyperParameters=hyperparameters,
        )


@patch("sagemaker.local.local_session.LocalSession")
def test_create_model(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    local_sagemaker_client.create_model(MODEL_NAME, PRIMARY_CONTAINER)

    assert MODEL_NAME in sagemaker.local.local_session.LocalSagemakerClient._models


@patch("sagemaker.local.local_session.LocalSession")
def test_delete_model(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    local_sagemaker_client.create_model(MODEL_NAME, PRIMARY_CONTAINER)
    assert MODEL_NAME in sagemaker.local.local_session.LocalSagemakerClient._models

    local_sagemaker_client.delete_model(MODEL_NAME)
    assert MODEL_NAME not in sagemaker.local.local_session.LocalSagemakerClient._models


@patch("sagemaker.local.local_session.LocalSession")
def test_describe_model(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    with pytest.raises(ClientError):
        local_sagemaker_client.describe_model("model-does-not-exist")

    local_sagemaker_client.create_model(MODEL_NAME, PRIMARY_CONTAINER)
    response = local_sagemaker_client.describe_model(MODEL_NAME)

    assert response["ModelName"] == "test-model"
    assert response["PrimaryContainer"]["ModelDataUrl"] == "/some/model/path"


@patch("sagemaker.local.local_session._LocalTransformJob")
@patch("sagemaker.local.local_session.LocalSession")
def test_create_transform_job(LocalSession, _LocalTransformJob):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    local_sagemaker_client.create_transform_job("transform-job", "some-model", None, None, None)
    _LocalTransformJob().start.assert_called_with(None, None, None)

    local_sagemaker_client.describe_transform_job("transform-job")
    _LocalTransformJob().describe.assert_called()


@patch("sagemaker.local.local_session._LocalTransformJob")
@patch("sagemaker.local.local_session.LocalSession")
def test_describe_transform_job_does_not_exist(LocalSession, _LocalTransformJob):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    with pytest.raises(ClientError):
        local_sagemaker_client.describe_transform_job("transform-job-does-not-exist")


@patch("sagemaker.local.local_session.LocalSession")
def test_describe_endpoint_config(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    # No Endpoint Config Created
    with pytest.raises(ClientError):
        local_sagemaker_client.describe_endpoint_config("some-endpoint-config")

    production_variants = [{"InstanceType": "ml.c4.99xlarge", "InitialInstanceCount": 10}]
    local_sagemaker_client.create_endpoint_config("test-endpoint-config", production_variants)

    response = local_sagemaker_client.describe_endpoint_config("test-endpoint-config")
    assert response["EndpointConfigName"] == "test-endpoint-config"
    assert response["ProductionVariants"][0]["InstanceType"] == "ml.c4.99xlarge"


@patch("sagemaker.local.local_session.LocalSession")
def test_create_endpoint_config(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    local_sagemaker_client.create_endpoint_config(ENDPOINT_CONFIG_NAME, PRODUCTION_VARIANTS)

    assert (
        ENDPOINT_CONFIG_NAME in sagemaker.local.local_session.LocalSagemakerClient._endpoint_configs
    )


@patch("sagemaker.local.local_session.LocalSession")
def test_delete_endpoint_config(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    local_sagemaker_client.create_endpoint_config(ENDPOINT_CONFIG_NAME, PRODUCTION_VARIANTS)
    assert (
        ENDPOINT_CONFIG_NAME in sagemaker.local.local_session.LocalSagemakerClient._endpoint_configs
    )

    local_sagemaker_client.delete_endpoint_config(ENDPOINT_CONFIG_NAME)
    assert (
        ENDPOINT_CONFIG_NAME
        not in sagemaker.local.local_session.LocalSagemakerClient._endpoint_configs
    )


@patch("sagemaker.local.image._SageMakerContainer.serve")
@patch("sagemaker.local.local_session.LocalSession")
@patch("urllib3.PoolManager.request")
@patch("sagemaker.local.local_session.LocalSagemakerClient.describe_endpoint_config")
@patch("sagemaker.local.local_session.LocalSagemakerClient.describe_model")
def test_describe_endpoint(describe_model, describe_endpoint_config, request, *args):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    request.return_value = OK_RESPONSE
    describe_endpoint_config.return_value = {
        "EndpointConfigName": "name",
        "EndpointConfigArn": "local:arn-does-not-matter",
        "CreationTime": "00:00:00",
        "ProductionVariants": [
            {
                "InitialVariantWeight": 1.0,
                "ModelName": "my-model",
                "VariantName": "AllTraffic",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ],
    }

    describe_model.return_value = {
        "ModelName": "my-model",
        "CreationTime": "00:00;00",
        "ExecutionRoleArn": "local:arn-does-not-matter",
        "ModelArn": "local:arn-does-not-matter",
        "PrimaryContainer": {
            "Environment": {"SAGEMAKER_REGION": "us-west-2"},
            "Image": "123.dkr.ecr-us-west-2.amazonaws.com/sagemaker-container:1.0",
            "ModelDataUrl": "s3://sagemaker-us-west-2/some/model.tar.gz",
        },
    }

    with pytest.raises(ClientError):
        local_sagemaker_client.describe_endpoint("non-existing-endpoint")

    local_sagemaker_client.create_endpoint("test-endpoint", "some-endpoint-config")
    response = local_sagemaker_client.describe_endpoint("test-endpoint")

    assert response["EndpointName"] == "test-endpoint"


@patch("sagemaker.local.image._SageMakerContainer.serve")
@patch("sagemaker.local.local_session.LocalSession")
@patch("urllib3.PoolManager.request")
@patch("sagemaker.local.local_session.LocalSagemakerClient.describe_endpoint_config")
@patch("sagemaker.local.local_session.LocalSagemakerClient.describe_model")
def test_create_endpoint(describe_model, describe_endpoint_config, request, *args):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()

    request.return_value = OK_RESPONSE
    describe_endpoint_config.return_value = {
        "EndpointConfigName": "name",
        "EndpointConfigArn": "local:arn-does-not-matter",
        "CreationTime": "00:00:00",
        "ProductionVariants": [
            {
                "InitialVariantWeight": 1.0,
                "ModelName": "my-model",
                "VariantName": "AllTraffic",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ],
    }

    describe_model.return_value = {
        "ModelName": "my-model",
        "CreationTime": "00:00;00",
        "ExecutionRoleArn": "local:arn-does-not-matter",
        "ModelArn": "local:arn-does-not-matter",
        "PrimaryContainer": {
            "Environment": {"SAGEMAKER_REGION": "us-west-2"},
            "Image": "123.dkr.ecr-us-west-2.amazonaws.com/sagemaker-container:1.0",
            "ModelDataUrl": "s3://sagemaker-us-west-2/some/model.tar.gz",
        },
    }

    local_sagemaker_client.create_endpoint("my-endpoint", "some-endpoint-config")

    assert "my-endpoint" in sagemaker.local.local_session.LocalSagemakerClient._endpoints


@patch("sagemaker.local.local_session.LocalSession")
def test_update_endpoint(LocalSession):
    local_sagemaker_client = sagemaker.local.local_session.LocalSagemakerClient()
    endpoint_name = "my-endpoint"
    endpoint_config = "my-endpoint-config"
    expected_error_message = "Update endpoint name is not supported in local session."
    with pytest.raises(NotImplementedError, match=expected_error_message):
        local_sagemaker_client.update_endpoint(endpoint_name, endpoint_config)


@patch("sagemaker.local.image._SageMakerContainer.serve")
@patch("urllib3.PoolManager.request")
def test_serve_endpoint_with_correct_accelerator(request, *args):
    mock_session = Mock(name="sagemaker_session")
    mock_session.return_value.sagemaker_client = Mock(name="sagemaker_client")
    mock_session.config = None

    request.return_value = OK_RESPONSE
    mock_session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [
            {
                "ModelName": "my-model",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
                "AcceleratorType": "local_sagemaker_notebook",
            }
        ]
    }

    mock_session.sagemaker_client.describe_model.return_value = {
        "PrimaryContainer": {
            "Environment": {},
            "Image": "123.dkr.ecr-us-west-2.amazonaws.com/sagemaker-container:1.0",
            "ModelDataUrl": "s3://sagemaker-us-west-2/some/model.tar.gz",
        }
    }

    endpoint = sagemaker.local.local_session._LocalEndpoint(
        "my-endpoint", "some-endpoint-config", local_session=mock_session
    )
    endpoint.serve()

    assert (
        endpoint.primary_container["Environment"]["SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"]
        == "true"
    )


@patch("sagemaker.local.image._SageMakerContainer.serve")
@patch("urllib3.PoolManager.request")
def test_serve_endpoint_with_incorrect_accelerator(request, *args):
    mock_session = Mock(name="sagemaker_session")
    mock_session.return_value.sagemaker_client = Mock(name="sagemaker_client")
    mock_session.config = None

    request.return_value = OK_RESPONSE
    mock_session.sagemaker_client.describe_endpoint_config.return_value = {
        "ProductionVariants": [
            {
                "ModelName": "my-model",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
                "AcceleratorType": "local",
            }
        ]
    }

    mock_session.sagemaker_client.describe_model.return_value = {
        "PrimaryContainer": {
            "Environment": {},
            "Image": "123.dkr.ecr-us-west-2.amazonaws.com/sagemaker-container:1.0",
            "ModelDataUrl": "s3://sagemaker-us-west-2/some/model.tar.gz",
        }
    }

    endpoint = sagemaker.local.local_session._LocalEndpoint(
        "my-endpoint", "some-endpoint-config", local_session=mock_session
    )
    endpoint.serve()

    with pytest.raises(KeyError):
        assert (
            endpoint.primary_container["Environment"]["SAGEMAKER_INFERENCE_ACCELERATOR_PRESENT"]
            == "true"
        )


def test_file_input_all_defaults():
    prefix = "pre"
    actual = sagemaker.local.local_session.file_input(fileUri=prefix)
    expected = {
        "DataSource": {
            "FileDataSource": {"FileDataDistributionType": "FullyReplicated", "FileUri": prefix}
        }
    }
    assert actual.config == expected


def test_file_input_content_type():
    prefix = "pre"
    actual = sagemaker.local.local_session.file_input(fileUri=prefix, content_type="text/csv")
    expected = {
        "DataSource": {
            "FileDataSource": {"FileDataDistributionType": "FullyReplicated", "FileUri": prefix}
        },
        "ContentType": "text/csv",
    }
    assert actual.config == expected


def test_local_session_is_set_to_local_mode():
    boto_session = Mock(region_name="us-west-2")
    local_session = sagemaker.local.local_session.LocalSession(boto_session=boto_session)
    assert local_session.local_mode
