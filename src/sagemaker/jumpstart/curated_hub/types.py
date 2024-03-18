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
"""This module stores types related to SageMaker JumpStart CuratedHub."""
from __future__ import absolute_import
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

from sagemaker.jumpstart.types import (
    JumpStartDataHolderType,
    JumpStartModelSpecs,
    HubContentType,
)


class CuratedHubUnsupportedFlag(str, Enum):
    """Enum class for Curated Hub tag names."""

    DEPRECATED_VERSIONS = "deprecated_versions"
    TRAINING_VULNERABLE_VERSIONS = "training_vulnerable_versions"
    INFERENCE_VULNERABLE_VERSIONS = "inference_vulnerable_versions"


@dataclass
class HubContentSummary:
    """Dataclass to store HubContentSummary from List APIs."""

    hub_content_arn: str
    hub_content_name: str
    hub_content_version: str
    hub_content_type: HubContentType
    document_schema_version: str
    hub_content_status: str
    creation_time: datetime
    hub_content_display_name: str = None
    hub_content_description: str = None
    hub_content_search_keywords: List[str] = None


def summary_from_list_api_response(hub_content_summary: Dict[str, Any]) -> HubContentSummary:
    """Creates a single HubContentSummary.
    
    This is based on the ListHubContent or ListHubContentVersions API response."""
    return HubContentSummary(
        hub_content_arn=hub_content_summary.get("HubContentArn"),
        hub_content_name=hub_content_summary.get("HubContentName"),
        hub_content_version=hub_content_summary.get("HubContentVersion"),
        hub_content_type=hub_content_summary.get("HubContentType"),
        document_schema_version=hub_content_summary.get("DocumentSchemaVersion"),
        hub_content_status=hub_content_summary.get("HubContentStatus"),
        hub_content_display_name=hub_content_summary.get("HubContentDisplayName"),
        hub_content_description=hub_content_summary.get("HubContentDescription"),
        hub_content_search_keywords=hub_content_summary.get("HubContentSearchKeywords"),
        creation_time=hub_content_summary.get("CreationTime"),
    )


def summary_list_from_list_api_response(
    list_hub_contents_response: Dict[str, Any]
) -> List[HubContentSummary]:
    """Creates a HubContentSummary list.
    
    This is based on the ListHubContent or ListHubContentVersions API response."""
    return list(
        map(
            summary_from_list_api_response,
            list_hub_contents_response["HubContentSummaries"],
        )
    )


@dataclass
class S3ObjectLocation:
    """Helper class for S3 object references."""

    bucket: str
    key: str

    def format_for_s3_copy(self) -> Dict[str, str]:
        """Returns a dict formatted for S3 copy calls"""
        return {
            "Bucket": self.bucket,
            "Key": self.key,
        }

    def get_uri(self) -> str:
        """Returns the s3 URI"""
        return f"s3://{self.bucket}/{self.key}"


@dataclass
class JumpStartModelInfo:
    """Helper class for storing JumpStart model info."""

    model_id: str
    version: str


class HubContentDependencyType(str, Enum):
    """Enum class for HubContent dependency names"""

    INFERENCE_ARTIFACT = "inference_artifact_s3_reference"
    TRAINING_ARTIFACT = "training_artifact_s3_reference"
    INFERENCE_SCRIPT = "inference_script_s3_reference"
    TRAINING_SCRIPT = "training_script_s3_reference"
    DEFAULT_TRAINING_DATASET = "default_training_dataset_s3_reference"
    DEMO_NOTEBOOK = "demo_notebook_s3_reference"
    MARKDOWN = "markdown_s3_reference"


class FileInfo(JumpStartDataHolderType):
    """Data class for additional S3 file info."""

    location: S3ObjectLocation

    def __init__(
        self,
        bucket: str,
        key: str,
        size: Optional[bytes],
        last_updated: Optional[datetime],
        dependecy_type: Optional[HubContentDependencyType] = None,
    ):
        self.location = S3ObjectLocation(bucket, key)
        self.size = size
        self.last_updated = last_updated
        self.dependecy_type = dependecy_type


class HubContentDocument_v2(JumpStartDataHolderType):
    """Data class for HubContentDocument v2.0.0"""

    SCHEMA_VERSION = "2.0.0"

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a HubContentDocument_v2 object from JumpStart model specs.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_specs(spec)

    def from_specs(self, model_specs: JumpStartModelSpecs) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representatino of spec.
        """
        # TODO: Implement
        self.Url: str = model_specs.url
