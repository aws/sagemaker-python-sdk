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

import pytest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import CustomizationTechnique, DataSetMethod, DataSetHubContentDocument
from sagemaker.ai_registry.air_constants import (
    HubContentStatus, RESPONSE_KEY_HUB_CONTENT_ARN, RESPONSE_KEY_HUB_CONTENT_VERSION,
    DATASET_MAX_FILE_SIZE_BYTES, DATASET_SUPPORTED_EXTENSIONS
)


class TestDataSetHubContentDocument:
    def test_to_json(self):
        doc = DataSetHubContentDocument(
            dataset_type="CUSTOMER_PROVIDED",
            dataset_role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
            dataset_s3_bucket="test-bucket",
            dataset_s3_prefix="test-prefix",
            dependencies=["dep1", "dep2"]
        )
        
        result = doc.to_json()
        parsed = json.loads(result)
        
        assert parsed["DatasetType"] == "CUSTOMER_PROVIDED"
        assert parsed["DatasetRoleArn"] == "arn:aws:iam::123456789012:role/SageMakerRole"
        assert parsed["DatasetS3Bucket"] == "test-bucket"
        assert parsed["DatasetS3Prefix"] == "test-prefix"
        assert parsed["Dependencies"] == ["dep1", "dep2"]


class TestDataSetValidation:
    """Test class for dataset file validation functionality."""
    
    def test_validate_dataset_file_supported_extension(self):
        """Test validation passes for supported file extensions."""
        # Test .jsonl extension (supported)
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write(b'{"test": "data"}')
            f.flush()
            temp_file = f.name
        
        try:
            # Should not raise any exception
            DataSet._validate_dataset_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_dataset_file_unsupported_extension(self):
        """Test validation fails for unsupported file extensions."""
        # Test various unsupported extensions
        unsupported_extensions = ['.csv', '.txt', '.json', '.parquet', '.xlsx']
        
        for ext in unsupported_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(b'test content')
                f.flush()
                temp_file = f.name
            
            try:
                with pytest.raises(ValueError, match=f"Unsupported file extension: {ext}"):
                    DataSet._validate_dataset_file(temp_file)
            finally:
                os.unlink(temp_file)
    
    def test_validate_dataset_file_no_extension(self):
        """Test validation fails for files without extension."""
        with tempfile.NamedTemporaryFile(suffix='', delete=False) as f:
            f.write(b'test content')
            f.flush()
            temp_file = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported file extension: "):
                DataSet._validate_dataset_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_dataset_file_size_within_limit(self):
        """Test validation passes for files within size limit."""
        # Create a small file (well under 1GB limit)
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write(b'{"test": "data"}\n' * 1000)  # Small file
            f.flush()
            temp_file = f.name
        
        try:
            # Should not raise any exception
            DataSet._validate_dataset_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_dataset_file_size_exceeds_limit(self):
        """Test validation fails for files exceeding size limit."""
        # Mock os.path.getsize to return a size larger than the limit
        with tempfile.NamedTemporaryFile(suffix='.jsonl', delete=False) as f:
            f.write(b'{"test": "data"}')
            f.flush()
            temp_file = f.name
        
        try:
            with patch('os.path.getsize', return_value=DATASET_MAX_FILE_SIZE_BYTES + 1):
                with pytest.raises(ValueError, match="File size .* MB exceeds maximum allowed size"):
                    DataSet._validate_dataset_file(temp_file)
        finally:
            os.unlink(temp_file)
    
    def test_validate_dataset_file_s3_path_extension_only(self):
        """Test validation for S3 paths only checks extension, not size."""
        # S3 paths should only have extension validation, not size validation
        s3_paths = [
            "s3://bucket/data.jsonl",  # Should pass
            "s3://bucket/data.csv",    # Should fail
            "s3://bucket/path/to/data.jsonl",  # Should pass
        ]
        
        # Test supported S3 path
        DataSet._validate_dataset_file(s3_paths[0])  # Should not raise
        
        # Test unsupported S3 path
        with pytest.raises(ValueError, match="Unsupported file extension: .csv"):
            DataSet._validate_dataset_file(s3_paths[1])
        
        # Test supported S3 path with subdirectories
        DataSet._validate_dataset_file(s3_paths[2])  # Should not raise
    
    def test_validate_dataset_file_nonexistent_local_file(self):
        """Test validation handles non-existent local files gracefully."""
        # For non-existent local files, only extension validation should occur
        # Size validation should be skipped since file doesn't exist
        non_existent_file = "/path/to/nonexistent/file.jsonl"
        
        # Should only validate extension, not size (since file doesn't exist)
        DataSet._validate_dataset_file(non_existent_file)  # Should not raise
        
        # Test with unsupported extension
        non_existent_csv = "/path/to/nonexistent/file.csv"
        with pytest.raises(ValueError, match="Unsupported file extension: .csv"):
            DataSet._validate_dataset_file(non_existent_csv)


class TestDataSet:
    @patch('boto3.client')
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.train.common_utils.finetune_utils._get_current_domain_id')
    @patch('sagemaker.ai_registry.dataset.DataSet._validate_dataset_file')
    @patch('sagemaker.ai_registry.dataset.validate_dataset')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_create_with_s3_location(self, mock_air_hub, mock_validate, mock_validate_file, mock_get_domain_id, mock_session, mock_boto_client):
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_session.return_value = mock_session_instance
        
        # Mock boto3 STS client
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {'Account': '123456789012'}
        mock_boto_client.return_value = mock_sts_client
        
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate.return_value = None
        mock_validate_file.return_value = None
        
        def mock_exists(path):
            # Only return True for the temp file, False for metadata files
            return path == '/tmp/test_file.jsonl'
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('os.path.exists', side_effect=mock_exists), \
             patch('os.remove'), \
             patch('sagemaker.ai_registry.dataset.DataSet.wait'):
            
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test_file.jsonl'
            
            dataset = DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data/file.jsonl",
                customization_technique=CustomizationTechnique.SFT,
                sagemaker_session=mock_session_instance,
                wait=False
            )
            
            assert dataset.name == "test-dataset"
            assert dataset.arn == "test-arn"
            assert dataset.version == "1.0.0"
            assert dataset.status == HubContentStatus.IMPORTING
            mock_air_hub.import_hub_content.assert_called_once()
            mock_validate_file.assert_called_once_with("s3://test-bucket/data/file.jsonl")

    @patch('boto3.client')
    @patch('sagemaker.core.helper.session_helper.Session')
    @patch('sagemaker.train.common_utils.finetune_utils._get_current_domain_id')
    @patch('sagemaker.ai_registry.dataset.DataSet._validate_dataset_file')
    @patch('sagemaker.ai_registry.dataset.validate_dataset')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_create_with_s3_location_preserves_full_path(self, mock_air_hub, mock_validate, mock_validate_file, mock_get_domain_id, mock_session, mock_boto_client):
        """Test that S3 path includes filename, not just directory."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = "arn:aws:iam::123456789012:role/SageMakerRole"
        mock_session.return_value = mock_session_instance
        
        # Mock boto3 STS client
        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {'Account': '123456789012'}
        mock_boto_client.return_value = mock_sts_client
        
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate.return_value = None
        mock_validate_file.return_value = None
        
        def mock_exists(path):
            # Only return True for the temp file, False for metadata files
            return path == '/tmp/test_file.jsonl'
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp, \
             patch('os.path.exists', side_effect=mock_exists), \
             patch('os.remove'), \
             patch('sagemaker.ai_registry.dataset.DataSet.wait'):
            
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test_file.jsonl'
            
            dataset = DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/path/to/dataset.jsonl",
                customization_technique=CustomizationTechnique.SFT,
                sagemaker_session=mock_session_instance,
                wait=False
            )
            
            # Verify import_hub_content was called
            assert mock_air_hub.import_hub_content.called
            
            # Get the hub_content_document argument
            call_args = mock_air_hub.import_hub_content.call_args
            document_str = call_args[1]['hub_content_document']
            document = json.loads(document_str)
            
            # Verify the S3 prefix includes the full path with filename
            assert document['DatasetS3Prefix'] == 'path/to/dataset.jsonl'
            assert document['DatasetS3Bucket'] == 'test-bucket'

    @patch('sagemaker.ai_registry.dataset.DataSet._validate_dataset_file')
    @patch('sagemaker.ai_registry.dataset.validate_dataset')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_create_with_local_file(self, mock_air_hub, mock_validate, mock_validate_file):
        mock_air_hub.upload_to_s3.return_value = "s3://bucket/path"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        mock_validate.return_value = None
        mock_validate_file.return_value = None
        
        dataset = DataSet.create(
            name="test-dataset",
            source="/local/path/file.jsonl",
            customization_technique=CustomizationTechnique.DPO,
            wait=False
        )
        
        assert dataset.name == "test-dataset"
        assert dataset.method == DataSetMethod.UPLOADED
        mock_air_hub.upload_to_s3.assert_called_once()
        mock_validate.assert_called_once_with("/local/path/file.jsonl", "dpo")
        mock_validate_file.assert_called_once_with("/local/path/file.jsonl")

    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_get(self, mock_air_hub):
        mock_air_hub.describe_hub_content.return_value = {
            "HubContentName": "test-dataset",
            "HubContentArn": "test-arn",
            "HubContentVersion": "1.0.0",
            "HubContentStatus": "Available",
            "HubContentDocument": json.dumps({"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}),
            "HubContentDescription": "Test description",
            "HubContentSearchKeywords": ["customization_technique:sft", "method:generated"],
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        
        dataset = DataSet.get("test-dataset")
        
        assert dataset.name == "test-dataset"
        assert dataset.arn == "test-arn"
        assert dataset.customization_technique == CustomizationTechnique.SFT

    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_get_versions(self, mock_air_hub):
        mock_air_hub.list_hub_content_versions.return_value = [
            {"HubContentVersion": "1.0.0"},
            {"HubContentVersion": "2.0.0"}
        ]
        mock_air_hub.describe_hub_content.return_value = {
            "HubContentName": "test-dataset",
            "HubContentArn": "test-arn",
            "HubContentVersion": "1.0.0",
            "HubContentStatus": "Available",
            "HubContentDocument": json.dumps({"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}),
            "HubContentDescription": "Test",
            "HubContentSearchKeywords": ["customization_technique:sft", "method:generated"],
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01"
        }
        
        dataset = DataSet("test", "arn", "1.0.0", "s3://bucket/prefix", HubContentStatus.AVAILABLE, "desc", CustomizationTechnique.SFT)
        versions = dataset.get_versions()
        
        assert len(versions) == 2

    @patch('sagemaker.ai_registry.dataset.DataSet.create')
    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_create_version_success(self, mock_air_hub, mock_create):
        mock_air_hub.describe_hub_content.return_value = {
            "HubContentDocument": "{}",
            "HubContentSearchKeywords": ["customization_technique:sft", "method:generated"]
        }
        mock_create.return_value = Mock()
        
        dataset = DataSet("test", "arn", "1.0.0", "s3://bucket/prefix", HubContentStatus.AVAILABLE, "desc", CustomizationTechnique.SFT)
        result = dataset.create_version("s3://bucket/new-data")
        
        assert result is True
        mock_create.assert_called_once()

    @patch('sagemaker.ai_registry.dataset.AIRHub')
    def test_create_version_failure(self, mock_air_hub):
        mock_air_hub.describe_hub_content.side_effect = Exception("Error")
        
        dataset = DataSet("test", "arn", "1.0.0", "s3://bucket/prefix", HubContentStatus.AVAILABLE, "desc", CustomizationTechnique.SFT)
        result = dataset.create_version("s3://bucket/new-data")
        
        assert result is False
