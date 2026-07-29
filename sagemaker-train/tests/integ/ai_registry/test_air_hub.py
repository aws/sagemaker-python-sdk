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

"""Integration tests for AIRHub."""
import os
import tempfile

import pytest
from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_constants import DATASET_HUB_CONTENT_TYPE


class TestAIRHubIntegration:
    """Integration tests for AIRHub operations."""

    def test_get_hub_name(self):
        """Test hub name generation."""
        hub_name = AIRHub.get_hub_name()
        assert hub_name is not None
        assert len(hub_name) > 0
        assert hub_name == AIRHub.get_hub_name()  # Should be consistent

    def test_hub_name_initialization(self):
        """Test hub name is properly initialized."""
        AIRHub._ensure_hub_name_initialized()
        assert hasattr(AIRHub, 'hubName')
        assert hasattr(AIRHub, 'hubDisplayName')

    def test_import_hub_content(self, unique_name, sample_hub_content_document):
        """Test importing hub content."""
        response = AIRHub.import_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE,
            hub_content_name=unique_name,
            document_schema_version="2.0.0",
            hub_content_document=sample_hub_content_document,
        )
        assert response is not None
        assert 'HubContentArn' in response

    def test_describe_hub_content(self, unique_name, sample_hub_content_document):
        """Test describing hub content."""
        AIRHub.import_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE,
            hub_content_name=unique_name,
            document_schema_version="2.0.0",
            hub_content_document=sample_hub_content_document,
        )
        response = AIRHub.describe_hub_content(DATASET_HUB_CONTENT_TYPE, unique_name)
        assert response['HubContentName'] == unique_name
        assert 'HubContentArn' in response
        assert 'HubContentVersion' in response

    def test_list_hub_content(self):
        """Test listing hub content."""
        result = AIRHub.list_hub_content(DATASET_HUB_CONTENT_TYPE, max_results=5)
        assert 'items' in result
        assert isinstance(result['items'], list)

    def test_list_hub_content_versions(self, unique_name, sample_hub_content_document):
        """Test listing hub content versions."""
        AIRHub.import_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE,
            hub_content_name=unique_name,
            document_schema_version="2.0.0",
            hub_content_document=sample_hub_content_document,
        )
        versions = AIRHub.list_hub_content_versions(DATASET_HUB_CONTENT_TYPE, unique_name)
        assert isinstance(versions, list)
        assert len(versions) >= 1

    def test_delete_hub_content(self, unique_name, sample_hub_content_document):
        """Test deleting hub content."""
        response = AIRHub.import_hub_content(
            hub_content_type=DATASET_HUB_CONTENT_TYPE,
            hub_content_name=unique_name,
            document_schema_version="2.0.0",
            hub_content_document=sample_hub_content_document,
        )
        version = AIRHub.describe_hub_content(DATASET_HUB_CONTENT_TYPE, unique_name)['HubContentVersion']
        AIRHub.delete_hub_content(DATASET_HUB_CONTENT_TYPE, unique_name, version)

    def test_upload_to_s3(self, sample_jsonl_file, test_bucket):
        """Test uploading file to S3."""
        s3_uri = AIRHub.upload_to_s3(test_bucket, "test/file.jsonl", sample_jsonl_file)
        assert s3_uri.startswith("s3://")
        assert test_bucket in s3_uri

    def test_download_from_s3(self, test_bucket, sample_jsonl_file):
        """Test downloading file from S3."""
        s3_uri = AIRHub.upload_to_s3(test_bucket, "test/download.jsonl", sample_jsonl_file)
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            AIRHub.download_from_s3(s3_uri, tmp.name)
            assert os.path.exists(tmp.name)
            os.unlink(tmp.name)
