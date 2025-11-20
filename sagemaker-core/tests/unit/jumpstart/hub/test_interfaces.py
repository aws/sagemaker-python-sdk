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
"""Unit tests for sagemaker.core.jumpstart.hub.interfaces module."""
from __future__ import absolute_import

import json
import datetime
import pytest
from unittest.mock import Mock, patch

from sagemaker.core.jumpstart.hub.interfaces import (
    CreateHubResponse,
    HubContentDependency,
    DescribeHubContentResponse,
    HubS3StorageConfig,
    DescribeHubResponse,
    ImportHubResponse,
    HubSummary,
    ListHubsResponse,
    EcrUri,
    NotebookLocationUris,
    HubModelDocument,
    HubNotebookDocument,
    HubContentInfo,
    ListHubContentsResponse,
)
from sagemaker.core.jumpstart.types import HubContentType


class TestCreateHubResponse:
    """Test CreateHubResponse class."""

    def test_create_hub_response_init(self):
        """Test CreateHubResponse initialization."""
        json_obj = {"HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"}
        
        response = CreateHubResponse(json_obj)
        
        assert response.hub_arn == "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"

    def test_create_hub_response_to_json(self):
        """Test CreateHubResponse to_json method."""
        json_obj = {"HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"}
        response = CreateHubResponse(json_obj)
        
        result = response.to_json()
        
        assert "hub_arn" in result
        assert result["hub_arn"] == "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"


class TestHubContentDependency:
    """Test HubContentDependency class."""

    def test_hub_content_dependency_init(self):
        """Test HubContentDependency initialization."""
        json_obj = {
            "DependencyCopyPath": "s3://bucket/copy/path",
            "DependencyOriginPath": "s3://bucket/origin/path",
            "DependencyType": "MODEL"
        }
        
        dependency = HubContentDependency(json_obj)
        
        assert dependency.dependency_copy_path == "s3://bucket/copy/path"
        assert dependency.dependency_origin_path == "s3://bucket/origin/path"
        assert dependency.dependency_type == "MODEL"

    def test_hub_content_dependency_empty(self):
        """Test HubContentDependency with empty values."""
        json_obj = {}
        
        dependency = HubContentDependency(json_obj)
        
        assert dependency.dependency_copy_path == ""
        assert dependency.dependency_origin_path == ""
        assert dependency.dependency_type == ""


class TestDescribeHubContentResponse:
    """Test DescribeHubContentResponse class."""

    def test_describe_hub_content_response_model(self):
        """Test DescribeHubContentResponse with model content."""
        json_obj = {
            "CreationTime": datetime.datetime(2023, 1, 1),
            "DocumentSchemaVersion": "1.0",
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1",
            "HubContentDescription": "Test model",
            "HubContentDisplayName": "My Model",
            "HubContentType": "Model",
            "HubContentDocument": json.dumps({
                "Url": "https://example.com",
                "MinSdkVersion": "2.0.0",
                "TrainingSupported": True,
                "HostingEcrUri": "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
            }),
            "HubContentName": "my-model",
            "HubContentStatus": "Available",
            "HubContentVersion": "1.0",
            "HubName": "my-hub"
        }
        
        response = DescribeHubContentResponse(json_obj)
        
        assert response.hub_content_name == "my-model"
        assert response.hub_content_type == "Model"
        assert isinstance(response.hub_content_document, HubModelDocument)

    def test_describe_hub_content_response_notebook(self):
        """Test DescribeHubContentResponse with notebook content."""
        json_obj = {
            "CreationTime": datetime.datetime(2023, 1, 1),
            "DocumentSchemaVersion": "1.0",
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Notebook/my-notebook/1",
            "HubContentType": "Notebook",
            "HubContentDocument": json.dumps({
                "NotebookLocation": "s3://bucket/notebook.ipynb",
                "Dependencies": []
            }),
            "HubContentName": "my-notebook",
            "HubContentStatus": "Available",
            "HubContentVersion": "1.0",
            "HubName": "my-hub"
        }
        
        response = DescribeHubContentResponse(json_obj)
        
        assert response.hub_content_name == "my-notebook"
        assert response.hub_content_type == "Notebook"
        assert isinstance(response.hub_content_document, HubNotebookDocument)

    def test_describe_hub_content_response_invalid_type(self):
        """Test DescribeHubContentResponse with invalid content type."""
        json_obj = {
            "CreationTime": datetime.datetime(2023, 1, 1),
            "DocumentSchemaVersion": "1.0",
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Invalid/test/1",
            "HubContentType": "InvalidType",
            "HubContentDocument": json.dumps({}),
            "HubContentName": "test",
            "HubContentStatus": "Available",
            "HubContentVersion": "1.0",
            "HubName": "my-hub"
        }
        
        with pytest.raises(ValueError, match="not a valid HubContentType"):
            DescribeHubContentResponse(json_obj)

    def test_describe_hub_content_response_get_hub_region(self):
        """Test get_hub_region method."""
        json_obj = {
            "CreationTime": datetime.datetime(2023, 1, 1),
            "DocumentSchemaVersion": "1.0",
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1",
            "HubContentType": "Model",
            "HubContentDocument": json.dumps({
                "Url": "https://example.com",
                "TrainingSupported": False
            }),
            "HubContentName": "my-model",
            "HubContentStatus": "Available",
            "HubContentVersion": "1.0",
            "HubName": "my-hub"
        }
        
        response = DescribeHubContentResponse(json_obj)
        
        assert response.get_hub_region() == "us-west-2"


class TestHubS3StorageConfig:
    """Test HubS3StorageConfig class."""

    def test_hub_s3_storage_config_init(self):
        """Test HubS3StorageConfig initialization."""
        json_obj = {"S3OutputPath": "s3://bucket/output"}
        
        config = HubS3StorageConfig(json_obj)
        
        assert config.s3_output_path == "s3://bucket/output"

    def test_hub_s3_storage_config_empty(self):
        """Test HubS3StorageConfig with empty values."""
        json_obj = {}
        
        config = HubS3StorageConfig(json_obj)
        
        assert config.s3_output_path == ""


class TestDescribeHubResponse:
    """Test DescribeHubResponse class."""

    @patch('sagemaker.core.jumpstart.hub.interfaces.datetime')
    def test_describe_hub_response_init(self, mock_datetime):
        """Test DescribeHubResponse initialization."""
        mock_dt = Mock()
        mock_datetime.datetime.return_value = mock_dt
        
        json_obj = {
            "CreationTime": 2023,
            "FailureReason": "",
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubDescription": "Test hub",
            "HubDisplayName": "My Hub",
            "HubName": "my-hub",
            "HubSearchKeywords": ["ml", "test"],
            "HubStatus": "InService",
            "LastModifiedTime": 2023,
            "S3StorageConfig": {"S3OutputPath": "s3://bucket/output"}
        }
        
        response = DescribeHubResponse(json_obj)
        
        assert response.hub_name == "my-hub"
        assert response.hub_status == "InService"
        assert isinstance(response.s3_storage_config, HubS3StorageConfig)

    @patch('sagemaker.core.jumpstart.hub.interfaces.datetime')
    def test_describe_hub_response_get_hub_region(self, mock_datetime):
        """Test get_hub_region method."""
        mock_dt = Mock()
        mock_datetime.datetime.return_value = mock_dt
        
        json_obj = {
            "CreationTime": 2023,
            "FailureReason": "",
            "HubArn": "arn:aws:sagemaker:us-east-1:123456789012:hub/my-hub",
            "HubDescription": "Test hub",
            "HubDisplayName": "My Hub",
            "HubName": "my-hub",
            "HubSearchKeywords": [],
            "HubStatus": "InService",
            "LastModifiedTime": 2023,
            "S3StorageConfig": {"S3OutputPath": "s3://bucket/output"}
        }
        
        response = DescribeHubResponse(json_obj)
        
        assert response.get_hub_region() == "us-east-1"


class TestImportHubResponse:
    """Test ImportHubResponse class."""

    def test_import_hub_response_init(self):
        """Test ImportHubResponse initialization."""
        json_obj = {
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1"
        }
        
        response = ImportHubResponse(json_obj)
        
        assert response.hub_arn == "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub"
        assert "hub-content" in response.hub_content_arn


class TestHubSummary:
    """Test HubSummary class."""

    @patch('sagemaker.core.jumpstart.hub.interfaces.datetime')
    def test_hub_summary_init(self, mock_datetime):
        """Test HubSummary initialization."""
        mock_dt = Mock()
        mock_datetime.datetime.return_value = mock_dt
        
        json_obj = {
            "CreationTime": 2023,
            "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/my-hub",
            "HubDescription": "Test hub",
            "HubDisplayName": "My Hub",
            "HubName": "my-hub",
            "HubSearchKeywords": ["ml"],
            "HubStatus": "InService",
            "LastModifiedTime": 2023
        }
        
        summary = HubSummary(json_obj)
        
        assert summary.hub_name == "my-hub"
        assert summary.hub_status == "InService"


class TestListHubsResponse:
    """Test ListHubsResponse class."""

    @patch('sagemaker.core.jumpstart.hub.interfaces.datetime')
    def test_list_hubs_response_init(self, mock_datetime):
        """Test ListHubsResponse initialization."""
        mock_dt = Mock()
        mock_datetime.datetime.return_value = mock_dt
        
        json_obj = {
            "HubSummaries": [
                {
                    "CreationTime": 2023,
                    "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/hub1",
                    "HubDescription": "Hub 1",
                    "HubDisplayName": "Hub 1",
                    "HubName": "hub1",
                    "HubSearchKeywords": [],
                    "HubStatus": "InService",
                    "LastModifiedTime": 2023
                },
                {
                    "CreationTime": 2023,
                    "HubArn": "arn:aws:sagemaker:us-west-2:123456789012:hub/hub2",
                    "HubDescription": "Hub 2",
                    "HubDisplayName": "Hub 2",
                    "HubName": "hub2",
                    "HubSearchKeywords": [],
                    "HubStatus": "InService",
                    "LastModifiedTime": 2023
                }
            ],
            "NextToken": "next-token"
        }
        
        response = ListHubsResponse(json_obj)
        
        assert len(response.hub_summaries) == 2
        assert all(isinstance(s, HubSummary) for s in response.hub_summaries)
        assert response.next_token == "next-token"


class TestEcrUri:
    """Test EcrUri class."""

    def test_ecr_uri_parse(self):
        """Test parsing ECR URI."""
        uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-repo:latest"
        
        ecr_uri = EcrUri(uri)
        
        assert ecr_uri.account == "123456789012"
        assert ecr_uri.region_name == "us-west-2"
        assert ecr_uri.repository == "my-repo"
        assert ecr_uri.tag == "latest"

    def test_ecr_uri_parse_nested_repo(self):
        """Test parsing ECR URI with nested repository."""
        uri = "123456789012.dkr.ecr.us-west-2.amazonaws.com/org/team/my-repo:v1.0"
        
        ecr_uri = EcrUri(uri)
        
        assert ecr_uri.account == "123456789012"
        assert ecr_uri.region_name == "us-west-2"
        assert ecr_uri.repository == "org/team/my-repo"
        assert ecr_uri.tag == "v1.0"


class TestNotebookLocationUris:
    """Test NotebookLocationUris class."""

    def test_notebook_location_uris_init(self):
        """Test NotebookLocationUris initialization."""
        json_obj = {
            "demo_notebook": "s3://bucket/demo.ipynb",
            "model_fit": "s3://bucket/fit.ipynb",
            "model_deploy": "s3://bucket/deploy.ipynb"
        }
        
        uris = NotebookLocationUris(json_obj)
        
        assert uris.demo_notebook == "s3://bucket/demo.ipynb"
        assert uris.model_fit == "s3://bucket/fit.ipynb"
        assert uris.model_deploy == "s3://bucket/deploy.ipynb"

    def test_notebook_location_uris_partial(self):
        """Test NotebookLocationUris with partial data."""
        json_obj = {"demo_notebook": "s3://bucket/demo.ipynb"}
        
        uris = NotebookLocationUris(json_obj)
        
        assert uris.demo_notebook == "s3://bucket/demo.ipynb"
        assert uris.model_fit is None
        assert uris.model_deploy is None


class TestHubModelDocument:
    """Test HubModelDocument class."""

    def test_hub_model_document_init_inference_only(self):
        """Test HubModelDocument initialization for inference-only model."""
        json_obj = {
            "Url": "https://example.com",
            "MinSdkVersion": "2.0.0",
            "TrainingSupported": False,
            "HostingEcrUri": "123.dkr.ecr.us-west-2.amazonaws.com/model:latest",
            "HostingArtifactUri": "s3://bucket/model.tar.gz",
            "DefaultInferenceInstanceType": "ml.m5.xlarge",
            "SupportedInferenceInstanceTypes": ["ml.m5.xlarge", "ml.m5.2xlarge"]
        }
        
        doc = HubModelDocument(json_obj, region="us-west-2")
        
        assert doc.url == "https://example.com"
        assert doc.min_sdk_version == "2.0.0"
        assert doc.training_supported is False
        assert doc.hosting_ecr_uri == "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"

    def test_hub_model_document_init_with_training(self):
        """Test HubModelDocument initialization with training support."""
        json_obj = {
            "Url": "https://example.com",
            "MinSdkVersion": "2.0.0",
            "TrainingSupported": True,
            "HostingEcrUri": "123.dkr.ecr.us-west-2.amazonaws.com/model:latest",
            "TrainingEcrUri": "123.dkr.ecr.us-west-2.amazonaws.com/training:latest",
            "DefaultTrainingInstanceType": "ml.p3.2xlarge",
            "SupportedTrainingInstanceTypes": ["ml.p3.2xlarge"],
            "Hyperparameters": [
                {"Name": "learning_rate", "Type": "float", "Default": "0.001", "Scope": "algorithm"}
            ]
        }
        
        doc = HubModelDocument(json_obj, region="us-west-2")
        
        assert doc.training_supported is True
        assert doc.training_ecr_uri == "123.dkr.ecr.us-west-2.amazonaws.com/training:latest"
        assert len(doc.hyperparameters) == 1

    def test_hub_model_document_get_schema_version(self):
        """Test get_schema_version method."""
        json_obj = {
            "Url": "https://example.com",
            "TrainingSupported": False
        }
        
        doc = HubModelDocument(json_obj, region="us-west-2")
        
        assert doc.get_schema_version() == "2.3.0"

    def test_hub_model_document_get_region(self):
        """Test get_region method."""
        json_obj = {
            "Url": "https://example.com",
            "TrainingSupported": False
        }
        
        doc = HubModelDocument(json_obj, region="us-east-1")
        
        assert doc.get_region() == "us-east-1"

    def test_hub_model_document_to_json(self):
        """Test to_json method."""
        json_obj = {
            "Url": "https://example.com",
            "MinSdkVersion": "2.0.0",
            "TrainingSupported": False,
            "HostingEcrUri": "123.dkr.ecr.us-west-2.amazonaws.com/model:latest"
        }
        
        doc = HubModelDocument(json_obj, region="us-west-2")
        result = doc.to_json()
        
        assert "url" in result
        assert "min_sdk_version" in result
        assert "training_supported" in result


class TestHubNotebookDocument:
    """Test HubNotebookDocument class."""

    def test_hub_notebook_document_init(self):
        """Test HubNotebookDocument initialization."""
        json_obj = {
            "NotebookLocation": "s3://bucket/notebook.ipynb",
            "Dependencies": [
                {
                    "DependencyCopyPath": "s3://bucket/copy",
                    "DependencyOriginPath": "s3://bucket/origin",
                    "DependencyType": "NOTEBOOK"
                }
            ]
        }
        
        doc = HubNotebookDocument(json_obj, region="us-west-2")
        
        assert doc.notebook_location == "s3://bucket/notebook.ipynb"
        assert len(doc.dependencies) == 1
        assert isinstance(doc.dependencies[0], HubContentDependency)

    def test_hub_notebook_document_get_schema_version(self):
        """Test get_schema_version method."""
        json_obj = {
            "NotebookLocation": "s3://bucket/notebook.ipynb",
            "Dependencies": []
        }
        
        doc = HubNotebookDocument(json_obj, region="us-west-2")
        
        assert doc.get_schema_version() == "1.0.0"

    def test_hub_notebook_document_get_region(self):
        """Test get_region method."""
        json_obj = {
            "NotebookLocation": "s3://bucket/notebook.ipynb",
            "Dependencies": []
        }
        
        doc = HubNotebookDocument(json_obj, region="ap-south-1")
        
        assert doc.get_region() == "ap-south-1"


class TestHubContentInfo:
    """Test HubContentInfo class."""

    def test_hub_content_info_init(self):
        """Test HubContentInfo initialization."""
        json_obj = {
            "CreationTime": "2023-01-01T00:00:00Z",
            "DocumentSchemaVersion": "1.0",
            "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/my-model/1",
            "HubContentName": "my-model",
            "HubContentStatus": "Available",
            "HubContentType": "Model",
            "HubContentVersion": "1.0",
            "HubContentDescription": "Test model",
            "HubContentDisplayName": "My Model",
            "HubContentSearchKeywords": ["ml", "test"]
        }
        
        info = HubContentInfo(json_obj)
        
        assert info.hub_content_name == "my-model"
        assert info.hub_content_type == HubContentType.MODEL
        assert info.hub_content_status == "Available"

    def test_hub_content_info_get_hub_region(self):
        """Test get_hub_region method."""
        json_obj = {
            "CreationTime": "2023-01-01T00:00:00Z",
            "DocumentSchemaVersion": "1.0",
            "HubContentArn": "arn:aws:sagemaker:eu-west-1:123456789012:hub-content/my-hub/Model/my-model/1",
            "HubContentName": "my-model",
            "HubContentStatus": "Available",
            "HubContentType": "Model",
            "HubContentVersion": "1.0"
        }
        
        info = HubContentInfo(json_obj)
        
        assert info.get_hub_region() == "eu-west-1"


class TestListHubContentsResponse:
    """Test ListHubContentsResponse class."""

    def test_list_hub_contents_response_init(self):
        """Test ListHubContentsResponse initialization."""
        json_obj = {
            "HubContentSummaries": [
                {
                    "CreationTime": "2023-01-01T00:00:00Z",
                    "DocumentSchemaVersion": "1.0",
                    "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Model/model1/1",
                    "HubContentName": "model1",
                    "HubContentStatus": "Available",
                    "HubContentType": "Model",
                    "HubContentVersion": "1.0"
                },
                {
                    "CreationTime": "2023-01-01T00:00:00Z",
                    "DocumentSchemaVersion": "1.0",
                    "HubContentArn": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/my-hub/Notebook/notebook1/1",
                    "HubContentName": "notebook1",
                    "HubContentStatus": "Available",
                    "HubContentType": "Notebook",
                    "HubContentVersion": "1.0"
                }
            ],
            "NextToken": "next-token"
        }
        
        response = ListHubContentsResponse(json_obj)
        
        assert len(response.hub_content_summaries) == 2
        assert all(isinstance(s, HubContentInfo) for s in response.hub_content_summaries)
        assert response.next_token == "next-token"

    def test_list_hub_contents_response_empty(self):
        """Test ListHubContentsResponse with empty list."""
        json_obj = {
            "HubContentSummaries": [],
            "NextToken": ""
        }
        
        response = ListHubContentsResponse(json_obj)
        
        assert len(response.hub_content_summaries) == 0
        assert response.next_token == ""
