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
"""AI Registry Hub client for managing hub content operations."""
from __future__ import annotations

import hashlib
from typing import Optional
from urllib.parse import urlparse
from sagemaker.ai_registry.utils import base32_encode
import boto3
from sagemaker.core.helper.session_helper import Session

from sagemaker.ai_registry.air_constants import AIR_DEFAULT_PAGE_SIZE, AIR_HUB_CONTENT_DEFAULT_VERSION
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

class AIRHub:
    """AI Registry Hub class for managing hub content operations."""
    
    # Use production SageMaker endpoint (default)
    _sagemaker_client = boto3.client("sagemaker")
    _s3_client = boto3.client("s3")

    @classmethod
    def _generate_hub_names(cls, region: str, account_id: str) -> None:
        """Generate hub name and display name based on region and account ID.
        
        Args:
            region: AWS region name
            account_id: AWS account ID
        """
        hub_name_base = f"AiRegistry-{region}-{account_id}"
        hash_bytes = hashlib.sha256(hub_name_base.encode()).digest()
        
        cls.hubName = base32_encode(hash_bytes).strip('=')
        cls.hubDisplayName = hub_name_base

    @classmethod
    def _ensure_hub_name_initialized(cls) -> None:
        """Ensure hubName is initialized."""
        if not hasattr(cls, 'hubName'):
            sts_client = boto3.client("sts")
            account_id = sts_client.get_caller_identity()['Account']
            region = boto3.session.Session().region_name
            cls._generate_hub_names(region, account_id)

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.get_hub_name")
    def get_hub_name(cls) -> str:
        """Get hub name, initializing it if not yet initialized.
        
        Returns:
            Hub name string
        """
        cls._ensure_hub_name_initialized()
        return cls.hubName

    @classmethod
    def _create_airegistry_hub_if_not_exists(cls, client=None) -> None:
        """Create AI Registry hub if it doesn't exist."""
        cls._ensure_hub_name_initialized()
        
        if client is None:
            client = cls._sagemaker_client

        try:
            client.describe_hub(HubName=cls.hubName)
        except client.exceptions.ResourceNotFound:
            client.create_hub(
                HubName=cls.hubName,
                HubDisplayName=cls.hubDisplayName,
                HubDescription="AI Registry Hub"
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to create AI Registry hub '{cls.hubName}'. "
                f"Ensure you have the necessary IAM permissions (sagemaker:CreateHub, sagemaker:DescribeHub). "
                f"Error: {str(e)}"
            )

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.import_hub_content")
    def import_hub_content(
        cls,
        hub_content_type: str,
        hub_content_name: str,
        document_schema_version: str,
        hub_content_document: str,
        hub_content_version: str = AIR_HUB_CONTENT_DEFAULT_VERSION,
        tags: Optional[tuple] = None,
        session: Optional[Session] = None,
    ):
        """Import hub content into the AI Registry hub.
        
        Args:
            hub_content_type: Type of hub content
            hub_content_name: Name of the hub content
            document_schema_version: Schema version of the document
            hub_content_document: JSON document content
            tags: Optional tuple of tags
            session: Boto3 session

        Returns:
            Response from import_hub_content API call
        """
        client = session.sagemaker_client if session is not None else cls._sagemaker_client

        cls._create_airegistry_hub_if_not_exists(client)

        request = {
            "HubName": cls.hubName,
            "HubContentType": hub_content_type,
            "HubContentName": hub_content_name,
            "HubContentVersion": hub_content_version,
            "DocumentSchemaVersion": document_schema_version,
            "HubContentDocument": hub_content_document,
        }
        if tags:
            request["HubContentSearchKeywords"] = [f"{tag[0]}:{tag[1]}" for tag in tags]
        return client.import_hub_content(**request)

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.list_hub_content")
    def list_hub_content(
        cls, 
        hub_content_type: str, 
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
        session: Optional[Session] = None,
    ):
        """List hub content with detailed information.
        
        Args:
            hub_content_type: Type of hub content to list
            max_results: Maximum number of results to return
            next_token: Token for pagination
            session: Boto3 session

        Returns:
            Dictionary containing items list and next_token
        """
        cls._ensure_hub_name_initialized()

        client = session.sagemaker_client if session is not None else cls._sagemaker_client
        
        request = {
            "HubName": cls.hubName,
            "HubContentType": hub_content_type,
            "MaxResults": max_results or AIR_DEFAULT_PAGE_SIZE,
        }

        if next_token:
            request["NextToken"] = next_token

        response = client.list_hub_contents(**request)
        summaries = response.get("HubContentSummaries", [])

        items = []
        for summary in summaries:
            hub_content_name = summary.get("HubContentName")
            detailed_response = cls.describe_hub_content(hub_content_type, hub_content_name, session=session)
            items.append(detailed_response)

        return {
            "items": items,
            "next_token": response.get("NextToken")
        }

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.describe_hub_content")
    def describe_hub_content(
        cls, 
        hub_content_type: str, 
        hub_content_name: str,
        hub_content_version: Optional[str] = None,
        session: Optional[Session] = None
    ):
        """Describe hub content details.
        
        Args:
            hub_content_type: Type of hub content
            hub_content_name: Name of the hub content
            hub_content_version: Optional version of the hub content
            session: Boto3 session

        Returns:
            Hub content description response
        """
        cls._ensure_hub_name_initialized()

        client = session.sagemaker_client if session is not None else cls._sagemaker_client
        
        request = {
            "HubName": cls.hubName,
            "HubContentType": hub_content_type,
            "HubContentName": hub_content_name,
        }
        if hub_content_version:
            request["HubContentVersion"] = hub_content_version
        return client.describe_hub_content(**request)

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.list_hub_content_versions")
    def list_hub_content_versions(cls, hub_content_type: str, hub_content_name: str, session: Optional[Session] = None):
        """List all versions of a hub content.
        
        Args:
            hub_content_type: Type of hub content
            hub_content_name: Name of the hub content
            session: Boto3 session

        Returns:
            List of hub content version summaries
        """
        cls._ensure_hub_name_initialized()
        
        client = session.sagemaker_client if session is not None else cls._sagemaker_client
        
        request = {
            "HubName": cls.hubName,
            "HubContentType": hub_content_type,
            "HubContentName": hub_content_name,
        }
        return client.list_hub_content_versions(**request).get("HubContentSummaries", [])

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.delete_hub_content")
    def delete_hub_content(cls, hub_content_type: str, hub_content_name: str, hub_content_version: str, session: Optional[Session] = None):
        """Delete a specific version of hub content.
        
        Args:
            hub_content_type: Type of hub content
            hub_content_name: Name of the hub content
            hub_content_version: Version of the hub content to delete
            session: Boto3 session

        Returns:
            Delete response
        """
        cls._ensure_hub_name_initialized()
        
        client = session.sagemaker_client if session is not None else cls._sagemaker_client
        
        request = {
            "HubName": cls.hubName,
            "HubContentType": hub_content_type,
            "HubContentName": hub_content_name,
            "HubContentVersion": hub_content_version
        }
        return client.delete_hub_content(**request)

    @staticmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.upload_to_s3")
    def upload_to_s3(bucket: str, prefix: str, local_file_path: str) -> str:
        """Upload a local file to S3.
        
        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix
            local_file_path: Path to local file
            
        Returns:
            S3 URI of uploaded file
        """
        AIRHub._s3_client.upload_file(local_file_path, bucket, prefix)
        return f"s3://{bucket}/{prefix}"

    @staticmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="AIRHub.download_from_s3")
    def download_from_s3(s3_uri: str, local_path: str) -> None:
        """Download a file from S3 to local path.
        
        Args:
            s3_uri: S3 URI of the file
            local_path: Local path to save the file
        """
        parsed = urlparse(s3_uri)
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        AIRHub._s3_client.download_file(bucket, key, local_path)
