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
from unittest.mock import patch, MagicMock

from sagemaker.ai_registry.air_constants import AIR_DEFAULT_PAGE_SIZE
from sagemaker.ai_registry.air_hub import AIRHub


class TestAIRHub:
    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_import_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_client.describe_hub.return_value = {"HubName": "test-hub"}
        
        # Reset the class variable to use our mock
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        result = AIRHub.import_hub_content(
            hub_content_type="DataSet",
            hub_content_name="test-dataset",
            document_schema_version="1.0.0",
            hub_content_document='{"test": "document"}'
        )
        
        assert result["HubContentArn"] == "test-arn"
        mock_client.import_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion='1.0.0',
            DocumentSchemaVersion="1.0.0",
            HubContentDocument='{"test": "document"}'
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_list_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.list_hub_contents.return_value = {"HubContentSummaries": []}
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}
        
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        result = AIRHub.list_hub_content("DataSet")
        
        assert "items" in result
        assert "next_token" in result
        mock_client.list_hub_contents.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            MaxResults=AIR_DEFAULT_PAGE_SIZE
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_describe_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}
        
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        result = AIRHub.describe_hub_content("DataSet", "test-dataset")
        
        assert result["HubContentName"] == "test"
        mock_client.describe_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset"
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_describe_hub_content_with_version(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}
        
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        AIRHub.describe_hub_content("DataSet", "test-dataset", "1.0.0")
        
        mock_client.describe_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion="1.0.0"
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_list_hub_content_versions(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.list_hub_content_versions.return_value = {"HubContentSummaries": []}
        
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        result = AIRHub.list_hub_content_versions("DataSet", "test-dataset")
        
        assert isinstance(result, list)
        mock_client.list_hub_content_versions.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset"
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_delete_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.delete_hub_content.return_value = {}
        
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"
        
        result = AIRHub.delete_hub_content("DataSet", "test-dataset", "1.0.0")
        
        mock_client.delete_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion="1.0.0"
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_upload_to_s3(self, mock_boto3):
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        AIRHub._s3_client = mock_s3_client
        
        result = AIRHub.upload_to_s3("test-bucket", "test/key", "/local/path")
        
        assert result == "s3://test-bucket/test/key"
        mock_s3_client.upload_file.assert_called_once_with("/local/path", "test-bucket", "test/key")

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_download_from_s3(self, mock_boto3):
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        AIRHub._s3_client = mock_s3_client
        
        AIRHub.download_from_s3("s3://test-bucket/test/key", "/local/path")
        
        mock_s3_client.download_file.assert_called_once_with("test-bucket", "test/key", "/local/path")

    def test_generate_hub_names_no_padding(self):
        """Test that generated hub names don't contain = padding characters."""
        # Clear any existing hubName to ensure clean test
        if hasattr(AIRHub, 'hubName'):
            delattr(AIRHub, 'hubName')
        if hasattr(AIRHub, 'hubDisplayName'):
            delattr(AIRHub, 'hubDisplayName')
        
        # Generate hub names
        AIRHub._generate_hub_names("us-west-2", "123456789012")
        
        # Verify hubName doesn't contain = padding
        assert "=" not in AIRHub.hubName, f"Hub name should not contain '=' padding: {AIRHub.hubName}"
        
        # Verify hubDisplayName is correctly formatted
        assert AIRHub.hubDisplayName == "AiRegistry-us-west-2-123456789012"
        
        # Verify hubName is not empty and is a valid base32 string
        assert len(AIRHub.hubName) > 0
        assert AIRHub.hubName.isalnum(), f"Hub name should only contain alphanumeric characters: {AIRHub.hubName}"