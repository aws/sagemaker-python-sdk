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
from enum import Enum
from typing import Any, Dict, List, Optional

from sagemaker.jumpstart.types import JumpStartDataHolderType


class HubContentType(str, Enum):
    """Enum for Hub data storage objects."""

    HUB = "Hub"
    MODEL = "Model"
    NOTEBOOK = "Notebook"

    @classmethod
    @property
    def content_only(cls):
        """Subset of HubContentType defined by SageMaker API Documentation"""
        return cls.MODEL, cls.NOTEBOOK


class HubArnExtractedInfo(JumpStartDataHolderType):
    """Data class for info extracted from Hub arn."""

    __slots__ = [
        "partition",
        "region",
        "account_id",
        "hub_name",
        "hub_content_type",
        "hub_content_name",
        "hub_content_version",
    ]

    def __init__(
        self,
        partition: str,
        region: str,
        account_id: str,
        hub_name: str,
        hub_content_type: Optional[str] = None,
        hub_content_name: Optional[str] = None,
        hub_content_version: Optional[str] = None,
    ) -> None:
        """Instantiates HubArnExtractedInfo object."""

        self.partition = partition
        self.region = region
        self.account_id = account_id
        self.hub_name = hub_name
        self.hub_content_type = hub_content_type
        self.hub_content_name = hub_content_name
        self.hub_content_version = hub_content_version


class HubContentDependency(JumpStartDataHolderType):
    """Data class for any dependencies related to hub content.

    Content can be scripts, model artifacts, datasets, or notebooks.
    """

    __slots__ = ["dependency_copy_path", "dependency_origin_path"]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates HubContentDependency object

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """

        self.dependency_copy_path: Optional[str] = json_obj.get("dependency_copy_path", "")
        self.dependency_origin_path: Optional[str] = json_obj.get("dependency_origin_path", "")

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of HubContentDependency object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class DescribeHubContentsResponse(JumpStartDataHolderType):
    """Data class for the Hub Content from session.describe_hub_contents()"""

    __slots__ = [
        "creation_time",
        "document_schema_version",
        "failure_reason",
        "hub_arn",
        "hub_content_arn",
        "hub_content_dependencies",
        "hub_content_description",
        "hub_content_display_name",
        "hub_content_document",
        "hub_content_markdown",
        "hub_content_name",
        "hub_content_search_keywords",
        "hub_content_status",
        "hub_content_type",
        "hub_content_version",
        "hub_name",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates DescribeHubContentsResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.creation_time: int = int(json_obj["creation_time"])
        self.document_schema_version: str = json_obj["document_schema_version"]
        self.failure_reason: str = json_obj["failure_reason"]
        self.hub_arn: str = json_obj["hub_arn"]
        self.hub_content_arn: str = json_obj["hub_content_arn"]
        self.hub_content_dependencies: List[HubContentDependency] = [
            HubContentDependency(dep) for dep in json_obj["hub_content_dependencies"]
        ]
        self.hub_content_description: str = json_obj["hub_content_description"]
        self.hub_content_display_name: str = json_obj["hub_content_display_name"]
        self.hub_content_document: str = json_obj["hub_content_document"]
        self.hub_content_markdown: str = json_obj["hub_content_markdown"]
        self.hub_content_name: str = json_obj["hub_content_name"]
        self.hub_content_search_keywords: str = json_obj["hub_content_search_keywords"]
        self.hub_content_status: str = json_obj["hub_content_status"]
        self.hub_content_type: HubContentType.content_only = json_obj["hub_content_type"]
        self.hub_content_version: str = json_obj["hub_content_version"]
        self.hub_name: str = json_obj["hub_name"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of DescribeHubContentsResponse object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                if issubclass(type(cur_val), JumpStartDataHolderType):
                    json_obj[att] = cur_val.to_json()
                elif isinstance(cur_val, list):
                    json_obj[att] = []
                    for obj in cur_val:
                        if issubclass(type(obj), JumpStartDataHolderType):
                            json_obj[att].append(obj.to_json())
                        else:
                            json_obj[att].append(obj)
                else:
                    json_obj[att] = cur_val
        return json_obj


class HubS3StorageConfig(JumpStartDataHolderType):
    """Data class for any dependencies related to hub content.

    Includes scripts, model artifacts, datasets, or notebooks."""

    __slots__ = ["s3_output_path"]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates HubS3StorageConfig object

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """

        self.s3_output_path: Optional[str] = json_obj.get("s3_output_path", "")

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of HubS3StorageConfig object."""
        return {"s3_output_path": self.s3_output_path}


class DescribeHubResponse(JumpStartDataHolderType):
    """Data class for the Hub from session.describe_hub()"""

    __slots__ = [
        "creation_time",
        "failure_reason",
        "hub_arn",
        "hub_description",
        "hub_display_name",
        "hub_name",
        "hub_search_keywords",
        "hub_status",
        "last_modified_time",
        "s3_storage_config",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates DescribeHubResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub  description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """

        self.creation_time: int = int(json_obj["creation_time"])
        self.failure_reason: str = json_obj["failure_reason"]
        self.hub_arn: str = json_obj["hub_arn"]
        self.hub_description: str = json_obj["hub_description"]
        self.hub_display_name: str = json_obj["hub_display_name"]
        self.hub_name: str = json_obj["hub_name"]
        self.hub_search_keywords: List[str] = json_obj["hub_search_keywords"]
        self.hub_status: str = json_obj["hub_status"]
        self.last_modified_time: int = int(json_obj["last_modified_time"])
        self.s3_storage_config: HubS3StorageConfig = HubS3StorageConfig(
            json_obj["s3_storage_config"]
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of DescribeHubContentsResponse object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                if issubclass(type(cur_val), JumpStartDataHolderType):
                    json_obj[att] = cur_val.to_json()
                elif isinstance(cur_val, list):
                    json_obj[att] = []
                    for obj in cur_val:
                        if issubclass(type(obj), JumpStartDataHolderType):
                            json_obj[att].append(obj.to_json())
                        else:
                            json_obj[att].append(obj)
                else:
                    json_obj[att] = cur_val
        return json_obj
