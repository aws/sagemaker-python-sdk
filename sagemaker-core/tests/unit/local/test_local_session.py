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

"""Unit tests for sagemaker.core.local.local_session module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from botocore.exceptions import ClientError

from sagemaker.core.local.local_session import (
    LocalSagemakerClient,
    LocalSagemakerRuntimeClient,
    LocalSession,
    FileInput,
)


class TestLocalSagemakerClient:
    """Test cases for LocalSagemakerClient"""

    def test_client_creation(self):
        """Test client creation"""
        client = LocalSagemakerClient()
        
        assert client.sagemaker_session is not None

    def test_create_processing_job(self):
        """Test creating a processing job"""
        mock_session = Mock()
        mock_session.sagemaker_config = {}
        client = LocalSagemakerClient(mock_session)
        
        with patch("sagemaker.core.local.local_session._SageMakerContainer") as mock_container_class:
            with patch("sagemaker.core.local.local_session._LocalProcessingJob") as mock_job_class:
                mock_job = Mock()
                mock_job_class.return_value = mock_job
                
                client.create_processing_job(
                    ProcessingJobName="test-job",
                    AppSpecification={"ImageUri": "test-image:latest"},
                    ProcessingResources={
                        "ClusterConfig": {
                            "InstanceType": "local",
                            "InstanceCount": 1,
                        }
                    },
                )
                
                mock_job.start.assert_called_once()

    def test_describe_processing_job_exists(self):
        """Test describing existing processing job"""
        client = LocalSagemakerClient()
        
        mock_job = Mock()
        mock_job.describe.return_value = {"ProcessingJobName": "test-job"}
        LocalSagemakerClient._processing_jobs["test-job"] = mock_job
        
        try:
            description = client.describe_processing_job("test-job")
            assert description["ProcessingJobName"] == "test-job"
        finally:
            del LocalSagemakerClient._processing_jobs["test-job"]

    def test_describe_processing_job_not_found(self):
        """Test describing non-existent processing job"""
        client = LocalSagemakerClient()
        
        with pytest.raises(ClientError, match="Could not find local processing job"):
            client.describe_processing_job("non-existent-job")

    def test_create_training_job(self):
        """Test creating a training job"""
        mock_session = Mock()
        mock_session.sagemaker_config = {}
        client = LocalSagemakerClient(mock_session)
        
        with patch("sagemaker.core.local.local_session._SageMakerContainer") as mock_container_class:
            with patch("sagemaker.core.local.local_session._LocalTrainingJob") as mock_job_class:
                mock_job = Mock()
                mock_job_class.return_value = mock_job
                
                client.create_training_job(
                    TrainingJobName="test-job",
                    AlgorithmSpecification={"TrainingImage": "test-image:latest"},
                    OutputDataConfig={"S3OutputPath": "s3://bucket/output"},
                    ResourceConfig={
                        "InstanceType": "local",
                        "InstanceCount": 1,
                    },
                )
                
                mock_job.start.assert_called_once()

    def test_describe_training_job_exists(self):
        """Test describing existing training job"""
        client = LocalSagemakerClient()
        
        mock_job = Mock()
        mock_job.describe.return_value = {"TrainingJobName": "test-job"}
        LocalSagemakerClient._training_jobs["test-job"] = mock_job
        
        try:
            description = client.describe_training_job("test-job")
            assert description["TrainingJobName"] == "test-job"
        finally:
            del LocalSagemakerClient._training_jobs["test-job"]

    def test_describe_training_job_not_found(self):
        """Test describing non-existent training job"""
        client = LocalSagemakerClient()
        
        with pytest.raises(ClientError, match="Could not find local training job"):
            client.describe_training_job("non-existent-job")

    def test_create_transform_job(self):
        """Test creating a transform job"""
        mock_session = Mock()
        mock_session.sagemaker_config = {}
        client = LocalSagemakerClient(mock_session)
        
        with patch("sagemaker.core.local.local_session._LocalTransformJob") as mock_job_class:
            mock_job = Mock()
            mock_job_class.return_value = mock_job
            
            client.create_transform_job(
                TransformJobName="test-job",
                ModelName="test-model",
                TransformInput={"DataSource": {"S3DataSource": {"S3Uri": "s3://bucket/input"}}},
                TransformOutput={"S3OutputPath": "s3://bucket/output"},
                TransformResources={"InstanceType": "local", "InstanceCount": 1},
            )
            
            mock_job.start.assert_called_once()

    def test_describe_transform_job_exists(self):
        """Test describing existing transform job"""
        client = LocalSagemakerClient()
        
        mock_job = Mock()
        mock_job.describe.return_value = {"TransformJobName": "test-job"}
        LocalSagemakerClient._transform_jobs["test-job"] = mock_job
        
        try:
            description = client.describe_transform_job("test-job")
            assert description["TransformJobName"] == "test-job"
        finally:
            del LocalSagemakerClient._transform_jobs["test-job"]

    def test_create_model(self):
        """Test creating a model"""
        client = LocalSagemakerClient()
        
        primary_container = {
            "Image": "test-image:latest",
            "ModelDataUrl": "s3://bucket/model.tar.gz",
        }
        
        client.create_model("test-model", primary_container)
        
        try:
            assert "test-model" in LocalSagemakerClient._models
        finally:
            if "test-model" in LocalSagemakerClient._models:
                del LocalSagemakerClient._models["test-model"]

    def test_describe_model_exists(self):
        """Test describing existing model"""
        client = LocalSagemakerClient()
        
        mock_model = Mock()
        mock_model.describe.return_value = {"ModelName": "test-model"}
        LocalSagemakerClient._models["test-model"] = mock_model
        
        try:
            description = client.describe_model("test-model")
            assert description["ModelName"] == "test-model"
        finally:
            del LocalSagemakerClient._models["test-model"]

    def test_describe_model_not_found(self):
        """Test describing non-existent model"""
        client = LocalSagemakerClient()
        
        with pytest.raises(ClientError, match="Could not find local model"):
            client.describe_model("non-existent-model")

    def test_create_endpoint_config(self):
        """Test creating endpoint config"""
        client = LocalSagemakerClient()
        
        production_variants = [
            {
                "VariantName": "AllTraffic",
                "ModelName": "test-model",
                "InitialInstanceCount": 1,
                "InstanceType": "local",
            }
        ]
        
        client.create_endpoint_config("test-config", production_variants)
        
        try:
            assert "test-config" in LocalSagemakerClient._endpoint_configs
        finally:
            if "test-config" in LocalSagemakerClient._endpoint_configs:
                del LocalSagemakerClient._endpoint_configs["test-config"]

    def test_describe_endpoint_config_exists(self):
        """Test describing existing endpoint config"""
        client = LocalSagemakerClient()
        
        mock_config = Mock()
        mock_config.describe.return_value = {"EndpointConfigName": "test-config"}
        LocalSagemakerClient._endpoint_configs["test-config"] = mock_config
        
        try:
            description = client.describe_endpoint_config("test-config")
            assert description["EndpointConfigName"] == "test-config"
        finally:
            del LocalSagemakerClient._endpoint_configs["test-config"]

    def test_create_endpoint(self):
        """Test creating endpoint"""
        mock_session = Mock()
        mock_session.sagemaker_config = {}
        client = LocalSagemakerClient(mock_session)
        
        with patch("sagemaker.core.local.local_session._LocalEndpoint") as mock_endpoint_class:
            mock_endpoint = Mock()
            mock_endpoint_class.return_value = mock_endpoint
            
            client.create_endpoint("test-endpoint", "test-config")
            
            mock_endpoint.serve.assert_called_once()

    def test_delete_endpoint(self):
        """Test deleting endpoint"""
        client = LocalSagemakerClient()
        
        mock_endpoint = Mock()
        LocalSagemakerClient._endpoints["test-endpoint"] = mock_endpoint
        
        try:
            client.delete_endpoint("test-endpoint")
            mock_endpoint.stop.assert_called_once()
        finally:
            if "test-endpoint" in LocalSagemakerClient._endpoints:
                del LocalSagemakerClient._endpoints["test-endpoint"]

    def test_delete_endpoint_config(self):
        """Test deleting endpoint config"""
        client = LocalSagemakerClient()
        
        LocalSagemakerClient._endpoint_configs["test-config"] = Mock()
        
        client.delete_endpoint_config("test-config")
        
        assert "test-config" not in LocalSagemakerClient._endpoint_configs

    def test_delete_model(self):
        """Test deleting model"""
        client = LocalSagemakerClient()
        
        LocalSagemakerClient._models["test-model"] = Mock()
        
        client.delete_model("test-model")
        
        assert "test-model" not in LocalSagemakerClient._models


class TestLocalSagemakerRuntimeClient:
    """Test cases for LocalSagemakerRuntimeClient"""

    def test_runtime_client_creation(self):
        """Test runtime client creation"""
        client = LocalSagemakerRuntimeClient()
        
        assert client.serving_port == 8080

    def test_runtime_client_with_config(self):
        """Test runtime client with custom config"""
        config = {"local": {"serving_port": 9090}}
        client = LocalSagemakerRuntimeClient(config)
        
        assert client.serving_port == 9090

    @patch("sagemaker.core.local.local_session.get_docker_host")
    @patch("urllib3.PoolManager")
    def test_invoke_endpoint_basic(self, mock_pool_manager_class, mock_get_host):
        """Test basic endpoint invocation"""
        mock_get_host.return_value = "localhost"
        
        mock_pool = Mock()
        mock_response = Mock()
        mock_response.status = 200
        mock_pool.request.return_value = mock_response
        mock_pool_manager_class.return_value = mock_pool
        
        client = LocalSagemakerRuntimeClient()
        
        response = client.invoke_endpoint(
            Body=b"test data",
            EndpointName="test-endpoint",
        )
        
        assert response["Body"] == mock_response
        mock_pool.request.assert_called_once()

    @patch("sagemaker.core.local.local_session.get_docker_host")
    @patch("urllib3.PoolManager")
    def test_invoke_endpoint_with_headers(self, mock_pool_manager_class, mock_get_host):
        """Test endpoint invocation with custom headers"""
        mock_get_host.return_value = "localhost"
        
        mock_pool = Mock()
        mock_response = Mock()
        mock_pool.request.return_value = mock_response
        mock_pool_manager_class.return_value = mock_pool
        
        client = LocalSagemakerRuntimeClient()
        
        client.invoke_endpoint(
            Body=b"test data",
            EndpointName="test-endpoint",
            ContentType="application/json",
            Accept="application/json",
            CustomAttributes="attr1=value1",
            TargetModel="model1",
            TargetVariant="variant1",
            InferenceId="inference-123",
        )
        
        call_args = mock_pool.request.call_args
        headers = call_args[1]["headers"]
        
        assert headers["Content-type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["X-Amzn-SageMaker-Custom-Attributes"] == "attr1=value1"

    @patch("sagemaker.core.local.local_session.get_docker_host")
    @patch("urllib3.PoolManager")
    def test_invoke_endpoint_with_string_body(self, mock_pool_manager_class, mock_get_host):
        """Test endpoint invocation with string body"""
        mock_get_host.return_value = "localhost"
        
        mock_pool = Mock()
        mock_response = Mock()
        mock_pool.request.return_value = mock_response
        mock_pool_manager_class.return_value = mock_pool
        
        client = LocalSagemakerRuntimeClient()
        
        client.invoke_endpoint(
            Body="test string data",
            EndpointName="test-endpoint",
        )
        
        call_args = mock_pool.request.call_args
        body = call_args[1]["body"]
        
        # String should be encoded to bytes
        assert isinstance(body, bytes)


class TestLocalSession:
    """Test cases for LocalSession"""

    @patch("boto3.Session")
    def test_local_session_creation(self, mock_boto_session_class):
        """Test local session creation"""
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session_class.return_value = mock_boto_session
        
        with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
            with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                session = LocalSession()
                
                assert session.local_mode is True
                assert session._region_name == "us-west-2"

    def test_local_session_with_s3_endpoint(self):
        """Test local session with custom S3 endpoint"""
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session.resource = Mock()
        mock_boto_session.client = Mock()
        
        with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
            with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                session = LocalSession(boto_session=mock_boto_session, s3_endpoint_url="http://localhost:9000")
                
                assert session.s3_endpoint_url == "http://localhost:9000"

    @patch("boto3.Session")
    def test_local_session_no_region_raises(self, mock_boto_session_class):
        """Test that missing region raises error"""
        mock_boto_session = Mock()
        mock_boto_session.region_name = None
        mock_boto_session_class.return_value = mock_boto_session
        
        with pytest.raises(ValueError, match="Must setup local AWS configuration"):
            LocalSession()

    @patch("boto3.Session")
    @patch("platform.system")
    def test_local_session_windows_warning(self, mock_platform, mock_boto_session_class):
        """Test Windows warning"""
        mock_platform.return_value = "Windows"
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session_class.return_value = mock_boto_session
        
        with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
            with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                with patch("sagemaker.core.local.local_session.logger") as mock_logger:
                    session = LocalSession()
                    
                    mock_logger.warning.assert_called()

    def test_logs_for_job_noop(self):
        """Test that logs_for_job is a no-op"""
        with patch("boto3.Session") as mock_boto_session_class:
            mock_boto_session = Mock()
            mock_boto_session.region_name = "us-west-2"
            mock_boto_session_class.return_value = mock_boto_session
            
            with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
                with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                    session = LocalSession()
                    
                    # Should not raise any errors
                    session.logs_for_job("test-job")

    def test_logs_for_processing_job_noop(self):
        """Test that logs_for_processing_job is a no-op"""
        with patch("boto3.Session") as mock_boto_session_class:
            mock_boto_session = Mock()
            mock_boto_session.region_name = "us-west-2"
            mock_boto_session_class.return_value = mock_boto_session
            
            with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
                with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                    session = LocalSession()
                    
                    # Should not raise any errors
                    session.logs_for_processing_job("test-job")

    @patch("boto3.Session")
    @patch("jsonschema.validate")
    def test_config_setter_validates(self, mock_validate, mock_boto_session_class):
        """Test that config setter validates schema"""
        mock_boto_session = Mock()
        mock_boto_session.region_name = "us-west-2"
        mock_boto_session_class.return_value = mock_boto_session
        
        with patch("sagemaker.core.local.local_session.load_sagemaker_config", return_value={}):
            with patch("sagemaker.core.local.local_session.load_local_mode_config", return_value={"local": {}}):
                session = LocalSession()
                
                new_config = {"local": {"container_root": "/tmp"}}
                session.config = new_config
                
                mock_validate.assert_called()


class TestFileInput:
    """Test cases for FileInput"""

    def test_file_input_creation(self):
        """Test FileInput creation"""
        file_input = FileInput("file:///path/to/data")
        
        assert "DataSource" in file_input.config
        assert "FileDataSource" in file_input.config["DataSource"]
        assert file_input.config["DataSource"]["FileDataSource"]["FileUri"] == "file:///path/to/data"

    def test_file_input_with_content_type(self):
        """Test FileInput with content type"""
        file_input = FileInput("file:///path/to/data", content_type="text/csv")
        
        assert file_input.config["ContentType"] == "text/csv"

    def test_file_input_distribution_type(self):
        """Test FileInput distribution type"""
        file_input = FileInput("file:///path/to/data")
        
        assert file_input.config["DataSource"]["FileDataSource"]["FileDataDistributionType"] == "FullyReplicated"
