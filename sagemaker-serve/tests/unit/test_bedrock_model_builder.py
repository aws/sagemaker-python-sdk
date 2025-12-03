"""Unit tests for BedrockModelBuilder."""

import pytest
from unittest.mock import Mock, patch
from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder


class TestBedrockModelBuilder:
    """Test suite for BedrockModelBuilder."""

    @pytest.fixture
    def mock_model_package(self):
        """Create a mock model package."""
        mock_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = "llama"
        mock_base_model.hub_content_name = "llama-model"
        mock_container.base_model = mock_base_model
        mock_container.model_data_source = None
        mock_package.inference_specification.containers = [mock_container]
        return mock_package

    @pytest.fixture
    def mock_training_job(self):
        """Create a mock training job."""
        mock_job = Mock()
        mock_job.output_model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package"
        return mock_job

    def test_init_with_training_job(self, mock_training_job):
        """Test initialization with TrainingJob."""        
        mock_model_package = Mock()
        
        with patch.object(BedrockModelBuilder, '_fetch_model_package', return_value=mock_model_package), \
             patch.object(BedrockModelBuilder, '_get_s3_artifacts', return_value=None):
            builder = BedrockModelBuilder(model=mock_training_job)
        
        assert builder.model == mock_training_job

    def test_init_with_model_package(self):
        """Test initialization with ModelPackage."""        
        mock_model_package = Mock()
        
        with patch.object(BedrockModelBuilder, '_fetch_model_package', return_value=mock_model_package), \
             patch.object(BedrockModelBuilder, '_get_s3_artifacts', return_value=None):
            builder = BedrockModelBuilder(model=mock_model_package)
        
        assert builder.model == mock_model_package

    def test_get_s3_artifacts_success(self):
        """Test successful S3 artifacts retrieval."""
        
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = "llama"
        mock_base_model.hub_content_name = "llama-model"
        mock_container.base_model = mock_base_model
        mock_model_data_source = Mock()
        mock_s3_data_source = Mock()
        mock_s3_data_source.s3_uri = "s3://bucket/model.tar.gz"
        mock_model_data_source.s3_data_source = mock_s3_data_source
        mock_container.model_data_source = mock_model_data_source
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        result = builder._get_s3_artifacts()
        
        assert result == "s3://bucket/model.tar.gz"

    def test_get_s3_artifacts_none(self):
        """Test S3 artifacts retrieval returns None when no model package."""        
        builder = BedrockModelBuilder(model=None)
        result = builder._get_s3_artifacts()
        
        assert result is None

    def test_deploy_non_nova_model(self):
        """Test deploy method for non-Nova model."""        
        mock_bedrock_client = Mock()
        mock_bedrock_client.create_model_import_job.return_value = {"jobArn": "test-job-arn"}
        
        mock_model_package = Mock()
        mock_container = Mock()
        mock_container.base_model = None
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        builder.s3_model_artifacts = "s3://bucket/model.tar.gz"
        builder._bedrock_client = mock_bedrock_client
        
        result = builder.deploy(
            job_name="test-job",
            imported_model_name="test-model",
                role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        
        assert result == {"jobArn": "test-job-arn"}

    def test_deploy_nova_model(self):
        """Test deploy method for Nova model."""        
        mock_bedrock_client = Mock()
        mock_bedrock_client.create_custom_model.return_value = {"modelArn": "test-model-arn"}
        
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = "nova-micro"
        mock_base_model.hub_content_name = "nova-model"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        builder.s3_model_artifacts = "s3://bucket/checkpoint"
        builder._bedrock_client = mock_bedrock_client
        
        result = builder.deploy(
            custom_model_name="test-nova-model",
            role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        
        assert result == {"modelArn": "test-model-arn"}
        mock_bedrock_client.create_custom_model.assert_called_once()

    def test_deploy_nova_model_with_hub_content_name(self):
        """Test deploy for Nova model detected via hub_content_name."""
        mock_bedrock_client = Mock()
        mock_bedrock_client.create_custom_model.return_value = {"modelArn": "test-model-arn"}
        
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = None
        mock_base_model.hub_content_name = "amazon-nova-lite"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        builder.s3_model_artifacts = "s3://bucket/checkpoint"
        builder._bedrock_client = mock_bedrock_client
        
        result = builder.deploy(
            custom_model_name="test-nova-model",
            role_arn="arn:aws:iam::123456789012:role/test-role"
        )
        
        assert result == {"modelArn": "test-model-arn"}
        mock_bedrock_client.create_custom_model.assert_called_once()

    def test_get_checkpoint_uri_from_manifest(self):
        """Test checkpoint URI extraction from manifest.json."""
        import json
        from unittest.mock import MagicMock
        from sagemaker.core.resources import TrainingJob
        
        mock_training_job = Mock()
        mock_training_job.model_artifacts.s3_model_artifacts = "s3://bucket/path/output/model.tar.gz"
        
        mock_s3_client = Mock()
        mock_response = Mock()
        manifest_data = {"checkpoint_s3_bucket": "s3://bucket/checkpoint/step_4"}
        mock_response.__getitem__ = lambda self, key: MagicMock(read=lambda: json.dumps(manifest_data).encode())
        mock_s3_client.get_object.return_value = mock_response
        
        mock_boto_session = Mock()
        mock_boto_session.client.return_value = mock_s3_client
        
        builder = BedrockModelBuilder(model=None)
        builder.model = mock_training_job
        builder.boto_session = mock_boto_session
        
        with patch('sagemaker.serve.bedrock_model_builder.isinstance', return_value=True):
            result = builder._get_checkpoint_uri_from_manifest()
        
        assert result == "s3://bucket/checkpoint/step_4"
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="bucket",
            Key="path/output/output/manifest.json"
        )

    def test_get_checkpoint_uri_manifest_not_found(self):
        """Test error when manifest.json not found."""
        from botocore.exceptions import ClientError
        
        mock_training_job = Mock()
        mock_training_job.model_artifacts.s3_model_artifacts = "s3://bucket/path/output/model.tar.gz"
        
        mock_s3_client = Mock()
        mock_s3_client.exceptions.NoSuchKey = ClientError
        mock_s3_client.get_object.side_effect = ClientError(
            {"Error": {"Code": "NoSuchKey"}}, "GetObject"
        )
        
        mock_boto_session = Mock()
        mock_boto_session.client.return_value = mock_s3_client
        
        builder = BedrockModelBuilder(model=None)
        builder.model = mock_training_job
        builder.boto_session = mock_boto_session
        
        with patch('sagemaker.serve.bedrock_model_builder.isinstance', return_value=True), \
             pytest.raises(ValueError, match="manifest.json not found"):
            builder._get_checkpoint_uri_from_manifest()

    def test_is_nova_detection_recipe_name(self):
        """Test Nova model detection via recipe_name."""
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = "amazon-nova-micro-v1"
        mock_base_model.hub_content_name = "other-model"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        
        container = mock_model_package.inference_specification.containers[0]
        is_nova = "nova" in container.base_model.recipe_name.lower()
        
        assert is_nova is True

    def test_is_nova_detection_hub_content_name(self):
        """Test Nova model detection via hub_content_name."""
        mock_model_package = Mock()
        mock_container = Mock()
        mock_base_model = Mock()
        mock_base_model.recipe_name = None
        mock_base_model.hub_content_name = "amazon-nova-lite"
        mock_container.base_model = mock_base_model
        mock_model_package.inference_specification.containers = [mock_container]
        
        builder = BedrockModelBuilder(model=None)
        builder.model_package = mock_model_package
        
        container = mock_model_package.inference_specification.containers[0]
        is_nova = "nova" in container.base_model.hub_content_name.lower()
        
        assert is_nova is True
