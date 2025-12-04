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

"""Unit tests for sagemaker.core.local.entities module"""

import pytest
import datetime
import os
import tempfile
import urllib3
from unittest.mock import Mock, patch, MagicMock, call

from sagemaker.core.local.entities import (
    _LocalProcessingJob,
    _LocalTrainingJob,
    _LocalTransformJob,
    _LocalModel,
    _LocalEndpointConfig,
    _LocalEndpoint,
    _wait_for_serving_container,
    _perform_request,
    HEALTH_CHECK_TIMEOUT_LIMIT,
)


class TestLocalProcessingJob:
    """Test cases for _LocalProcessingJob"""

    def test_processing_job_creation(self):
        """Test processing job creation"""
        mock_container = Mock()
        job = _LocalProcessingJob(mock_container)

        assert job.container == mock_container
        assert job.state == "Created"
        assert job.start_time is None

    def test_processing_job_start_basic(self):
        """Test starting a basic processing job"""
        mock_container = Mock()
        job = _LocalProcessingJob(mock_container)

        processing_inputs = [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "s3://bucket/input",
                    "LocalPath": "/opt/ml/processing/input",
                    "S3InputMode": "File",
                    "S3DataDistributionType": "FullyReplicated",
                    "S3CompressionType": "None",
                },
                "DataUri": "s3://bucket/input",
            }
        ]

        processing_output_config = {
            "Outputs": [
                {
                    "OutputName": "output-1",
                    "S3Output": {
                        "S3Uri": "s3://bucket/output",
                        "LocalPath": "/opt/ml/processing/output",
                        "S3UploadMode": "EndOfJob",
                    },
                }
            ]
        }

        environment = {"ENV_VAR": "value"}
        job_name = "test-processing-job"

        job.start(processing_inputs, processing_output_config, environment, job_name)

        assert job.state == job._COMPLETED
        assert job.processing_job_name == job_name
        mock_container.process.assert_called_once()

    def test_processing_job_with_dataset_definition_raises(self):
        """Test that DatasetDefinition raises error"""
        mock_container = Mock()
        job = _LocalProcessingJob(mock_container)

        processing_inputs = [
            {
                "InputName": "input-1",
                "DatasetDefinition": {"DatasetName": "test"},
            }
        ]

        with pytest.raises(RuntimeError, match="DatasetDefinition is not currently supported"):
            job.start(processing_inputs, {}, {}, "job-name")

    def test_processing_job_with_invalid_s3_input_mode(self):
        """Test that invalid S3InputMode raises error"""
        mock_container = Mock()
        job = _LocalProcessingJob(mock_container)

        processing_inputs = [
            {
                "InputName": "input-1",
                "S3Input": {
                    "S3Uri": "s3://bucket/input",
                    "S3InputMode": "Pipe",
                },
                "DataUri": "s3://bucket/input",
            }
        ]

        with pytest.raises(RuntimeError, match="S3InputMode.*not currently supported"):
            job.start(processing_inputs, {}, {}, "job-name")

    def test_processing_job_describe(self):
        """Test describing a processing job"""
        mock_container = Mock()
        mock_container.image = "test-image:latest"
        mock_container.container_entrypoint = ["/bin/bash"]
        mock_container.container_arguments = ["script.sh"]
        mock_container.instance_count = 1
        mock_container.instance_type = "local"

        job = _LocalProcessingJob(mock_container)
        job.processing_job_name = "test-job"
        job.environment = {"KEY": "value"}
        job.processing_inputs = []
        job.processing_output_config = {}
        job.state = job._COMPLETED
        job.start_time = datetime.datetime.now()
        job.end_time = datetime.datetime.now()

        description = job.describe()

        assert description["ProcessingJobName"] == "test-job"
        assert description["ProcessingJobStatus"] == job._COMPLETED
        assert "AppSpecification" in description
        assert description["AppSpecification"]["ImageUri"] == "test-image:latest"


class TestLocalTrainingJob:
    """Test cases for _LocalTrainingJob"""

    def test_training_job_creation(self):
        """Test training job creation"""
        mock_container = Mock()
        job = _LocalTrainingJob(mock_container)

        assert job.container == mock_container
        assert job.state == "created"
        assert job.model_artifacts is None

    def test_training_job_start(self):
        """Test starting a training job"""
        mock_container = Mock()
        mock_container.train.return_value = "s3://bucket/model.tar.gz"

        job = _LocalTrainingJob(mock_container)

        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/training",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "DataUri": "s3://bucket/training",
            }
        ]

        output_data_config = {"S3OutputPath": "s3://bucket/output"}
        hyperparameters = {"epochs": "10"}
        environment = {"ENV": "value"}
        job_name = "test-training-job"

        job.start(input_data_config, output_data_config, hyperparameters, environment, job_name)

        assert job.state == job._COMPLETED
        assert job.training_job_name == job_name
        assert job.model_artifacts == "s3://bucket/model.tar.gz"

    def test_training_job_with_file_data_source(self):
        """Test training job with FileDataSource"""
        mock_container = Mock()
        mock_container.train.return_value = "file:///path/model.tar.gz"

        job = _LocalTrainingJob(mock_container)

        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "FileDataSource": {
                        "FileUri": "file:///data/training",
                        "FileDataDistributionType": "FullyReplicated",
                    }
                },
                "DataUri": "file:///data/training",
            }
        ]

        output_data_config = {"S3OutputPath": "file:///output"}

        job.start(input_data_config, output_data_config, {}, {}, "job-name")

        assert job.state == job._COMPLETED

    def test_training_job_invalid_data_distribution(self):
        """Test that invalid data distribution raises error"""
        mock_container = Mock()
        job = _LocalTrainingJob(mock_container)

        input_data_config = [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3Uri": "s3://bucket/training",
                        "S3DataDistributionType": "ShardedByS3Key",
                    }
                },
                "DataUri": "s3://bucket/training",
            }
        ]

        with pytest.raises(RuntimeError, match="Invalid DataDistribution"):
            job.start(input_data_config, {}, {}, {}, "job-name")

    def test_training_job_describe(self):
        """Test describing a training job"""
        mock_container = Mock()
        mock_container.instance_count = 1
        mock_container.container_entrypoint = ["/bin/bash"]

        job = _LocalTrainingJob(mock_container)
        job.training_job_name = "test-job"
        job.state = job._COMPLETED
        job.start_time = datetime.datetime.now()
        job.end_time = datetime.datetime.now()
        job.model_artifacts = "s3://bucket/model.tar.gz"
        job.output_data_config = {"S3OutputPath": "s3://bucket/output"}
        job.environment = {}

        description = job.describe()

        assert description["TrainingJobName"] == "test-job"
        assert description["TrainingJobStatus"] == job._COMPLETED
        assert description["ModelArtifacts"]["S3ModelArtifacts"] == "s3://bucket/model.tar.gz"


class TestLocalTransformJob:
    """Test cases for _LocalTransformJob"""

    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_transform_job_creation(self, mock_session_class):
        """Test transform job creation"""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }
        mock_session.sagemaker_client = mock_client

        job = _LocalTransformJob("test-job", "test-model", mock_session)

        assert job.name == "test-job"
        assert job.model_name == "test-model"
        assert job.state == job._CREATING

    @patch("sagemaker.core.local.local_session.LocalSession")
    @patch("sagemaker.core.local.entities._wait_for_serving_container")
    @patch("sagemaker.core.local.entities._SageMakerContainer")
    def test_transform_job_start(self, mock_container_class, mock_wait, mock_session_class):
        """Test starting a transform job"""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }
        mock_session.sagemaker_client = mock_client
        mock_session.config = {}

        mock_container = Mock()
        mock_container_class.return_value = mock_container

        job = _LocalTransformJob("test-job", "test-model", mock_session)

        input_data = {
            "DataSource": {
                "S3DataSource": {
                    "S3Uri": "s3://bucket/input",
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        }

        output_data = {
            "S3OutputPath": "s3://bucket/output",
            "Accept": "text/csv",
        }

        transform_resources = {
            "InstanceType": "local",
            "InstanceCount": 1,
        }

        job.start(input_data, output_data, transform_resources)

        assert job.state == job._COMPLETED
        mock_container.serve.assert_called_once()

    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_transform_job_describe(self, mock_session_class):
        """Test describing a transform job"""
        mock_session = Mock()
        mock_client = Mock()
        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }
        mock_session.sagemaker_client = mock_client

        job = _LocalTransformJob("test-job", "test-model", mock_session)
        job.state = job._COMPLETED
        job.start_time = datetime.datetime.now()
        job.end_time = datetime.datetime.now()
        job.batch_strategy = "MultiRecord"

        description = job.describe()

        assert description["TransformJobName"] == "test-job"
        assert description["TransformJobStatus"] == job._COMPLETED
        assert description["ModelName"] == "test-model"


class TestLocalModel:
    """Test cases for _LocalModel"""

    def test_model_creation(self):
        """Test model creation"""
        primary_container = {
            "Image": "test-image:latest",
            "ModelDataUrl": "s3://bucket/model.tar.gz",
            "Environment": {"KEY": "value"},
        }

        model = _LocalModel("test-model", primary_container)

        assert model.model_name == "test-model"
        assert model.primary_container == primary_container
        assert model.creation_time is not None

    def test_model_describe(self):
        """Test describing a model"""
        primary_container = {
            "Image": "test-image:latest",
            "ModelDataUrl": "s3://bucket/model.tar.gz",
        }

        model = _LocalModel("test-model", primary_container)
        description = model.describe()

        assert description["ModelName"] == "test-model"
        assert description["PrimaryContainer"] == primary_container
        assert "CreationTime" in description


class TestLocalEndpointConfig:
    """Test cases for _LocalEndpointConfig"""

    def test_endpoint_config_creation(self):
        """Test endpoint config creation"""
        production_variants = [
            {
                "VariantName": "AllTraffic",
                "ModelName": "test-model",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ]

        config = _LocalEndpointConfig("test-config", production_variants)

        assert config.name == "test-config"
        assert config.production_variants == production_variants
        assert config.creation_time is not None

    def test_endpoint_config_with_tags(self):
        """Test endpoint config with tags"""
        production_variants = []
        tags = [{"Key": "Environment", "Value": "test"}]

        config = _LocalEndpointConfig("test-config", production_variants, tags)

        assert len(config.tags) == 1

    def test_endpoint_config_describe(self):
        """Test describing endpoint config"""
        production_variants = [
            {
                "VariantName": "AllTraffic",
                "ModelName": "test-model",
            }
        ]

        config = _LocalEndpointConfig("test-config", production_variants)
        description = config.describe()

        assert description["EndpointConfigName"] == "test-config"
        assert description["ProductionVariants"] == production_variants


class TestLocalEndpoint:
    """Test cases for _LocalEndpoint"""

    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_endpoint_creation(self, mock_session_class):
        """Test endpoint creation"""
        mock_session = Mock()
        mock_client = Mock()

        # Mock endpoint config
        mock_client.describe_endpoint_config.return_value = {
            "EndpointConfigName": "test-config",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "test-model",
                    "InitialInstanceCount": 1,
                    "InstanceType": "local",
                }
            ],
        }

        # Mock model
        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }

        mock_session.sagemaker_client = mock_client

        endpoint = _LocalEndpoint("test-endpoint", "test-config", None, mock_session)

        assert endpoint.name == "test-endpoint"
        assert endpoint.state == endpoint._CREATING

    @patch("sagemaker.core.local.local_session.LocalSession")
    @patch("sagemaker.core.local.entities._wait_for_serving_container")
    @patch("sagemaker.core.local.entities._SageMakerContainer")
    def test_endpoint_serve(self, mock_container_class, mock_wait, mock_session_class):
        """Test serving an endpoint"""
        mock_session = Mock()
        mock_client = Mock()

        mock_client.describe_endpoint_config.return_value = {
            "EndpointConfigName": "test-config",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "test-model",
                    "InitialInstanceCount": 1,
                    "InstanceType": "local",
                }
            ],
        }

        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }

        mock_session.sagemaker_client = mock_client
        mock_session.config = {}

        mock_container = Mock()
        mock_container_class.return_value = mock_container

        endpoint = _LocalEndpoint("test-endpoint", "test-config", None, mock_session)
        endpoint.serve()

        assert endpoint.state == endpoint._IN_SERVICE
        mock_container.serve.assert_called_once()

    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_endpoint_stop(self, mock_session_class):
        """Test stopping an endpoint"""
        mock_session = Mock()
        mock_client = Mock()

        mock_client.describe_endpoint_config.return_value = {
            "ProductionVariants": [
                {
                    "ModelName": "test-model",
                    "InitialInstanceCount": 1,
                    "InstanceType": "local",
                }
            ],
        }

        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }

        mock_session.sagemaker_client = mock_client

        endpoint = _LocalEndpoint("test-endpoint", "test-config", None, mock_session)
        endpoint.container = Mock()

        endpoint.stop()

        endpoint.container.stop_serving.assert_called_once()

    @patch("sagemaker.core.local.local_session.LocalSession")
    def test_endpoint_describe(self, mock_session_class):
        """Test describing an endpoint"""
        mock_session = Mock()
        mock_client = Mock()

        mock_client.describe_endpoint_config.return_value = {
            "EndpointConfigName": "test-config",
            "ProductionVariants": [
                {
                    "ModelName": "test-model",
                }
            ],
        }

        mock_client.describe_model.return_value = {
            "PrimaryContainer": {
                "Image": "test-image:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
                "Environment": {},
            }
        }

        mock_session.sagemaker_client = mock_client

        endpoint = _LocalEndpoint("test-endpoint", "test-config", None, mock_session)
        endpoint.state = endpoint._IN_SERVICE
        endpoint.create_time = datetime.datetime.now()

        description = endpoint.describe()

        assert description["EndpointName"] == "test-endpoint"
        assert description["EndpointStatus"] == endpoint._IN_SERVICE


class TestWaitForServingContainer:
    """Test cases for _wait_for_serving_container"""

    @patch("sagemaker.core.local.entities._perform_request")
    @patch("sagemaker.core.local.entities.get_docker_host")
    @patch("time.sleep")
    def test_wait_success(self, mock_sleep, mock_get_host, mock_perform_request):
        """Test successful wait"""
        mock_get_host.return_value = "localhost"
        mock_perform_request.return_value = (Mock(), 200)

        _wait_for_serving_container(8080)

        mock_perform_request.assert_called()

    @patch("sagemaker.core.local.entities._perform_request")
    @patch("sagemaker.core.local.entities.get_docker_host")
    @patch("time.sleep")
    def test_wait_timeout(self, mock_sleep, mock_get_host, mock_perform_request):
        """Test timeout"""
        mock_get_host.return_value = "localhost"
        mock_perform_request.return_value = (None, 500)

        with pytest.raises(RuntimeError, match="Giving up"):
            _wait_for_serving_container(8080)


class TestPerformRequest:
    """Test cases for _perform_request"""

    @patch("urllib3.PoolManager")
    def test_perform_request_success(self, mock_pool_manager_class):
        """Test successful request"""
        mock_pool = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_pool.request.return_value = mock_response
        mock_pool_manager_class.return_value = mock_pool

        response, code = _perform_request("http://localhost:8080/ping")

        assert code == 200
        assert response == mock_response

    @patch("urllib3.PoolManager")
    def test_perform_request_error(self, mock_pool_manager_class):
        """Test request error"""
        mock_pool = Mock()
        mock_pool.request.side_effect = urllib3.exceptions.RequestError(
            mock_pool, "http://localhost:8080/ping", "Connection error"
        )
        mock_pool_manager_class.return_value = mock_pool

        response, code = _perform_request("http://localhost:8080/ping")

        assert code == -1
        assert response is None
