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
"""This module stores types related to SageMaker JumpStart."""
from __future__ import absolute_import
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class JumpStartDataHolderType:
    """Base class for many JumpStart types.

    Allows objects to be added to dicts and sets,
    and improves string representation. This class allows different objects with the same
    attributes and types to have equality.
    """

    __slots__: List[str] = []

    def __eq__(self, other: Any) -> bool:
        """Returns True if ``other`` is of the same type and has all attributes equal."""

        if not isinstance(other, type(self)):
            return False
        for attribute in self.__slots__:
            if getattr(self, attribute) != getattr(other, attribute):
                return False
        return True

    def __hash__(self) -> int:
        """Makes hash of object.

        Maps object to unique tuple, which then gets hashed.
        """

        return hash((type(self),) + tuple([getattr(self, att) for att in self.__slots__]))

    def __str__(self) -> str:
        """Returns string representation of object. Example:

        "JumpStartLaunchedRegionInfo:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {att: getattr(self, att) for att in self.__slots__}
        return f"{type(self).__name__}: {str(att_dict)}"

    def __repr__(self) -> str:
        """Returns ``__repr__`` string of object. Example:

        "JumpStartLaunchedRegionInfo at 0x7f664529efa0:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {att: getattr(self, att) for att in self.__slots__}
        return f"{type(self).__name__} at {hex(id(self))}: {str(att_dict)}"


class JumpStartS3FileType(str, Enum):
    """Simple enum for classifying S3 file type."""

    MANIFEST = "manifest"
    SPECS = "specs"


class JumpStartLaunchedRegionInfo(JumpStartDataHolderType):
    """Data class for launched region info."""

    __slots__ = ["content_bucket", "region_name"]

    def __init__(self, content_bucket: str, region_name: str):
        self.content_bucket = content_bucket
        self.region_name = region_name


class JumpStartModelHeader(JumpStartDataHolderType):
    """Data class JumpStart model header."""

    __slots__ = ["model_id", "version", "min_version", "spec_key"]

    def __init__(self, header: Dict[str, str]):
        """Initializes a JumpStartModelHeader object from its json representation."""
        self.from_json(header)

    def to_json(self) -> Dict[str, str]:
        """Returns json representation of JumpStartModelHeader object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__}
        return json_obj

    def from_json(self, json_obj: Dict[str, str]) -> None:
        """Sets fields in object based on json of header."""
        self.model_id: str = json_obj["model_id"]
        self.version: str = json_obj["version"]
        self.min_version: str = json_obj["min_version"]
        self.spec_key: str = json_obj["spec_key"]


class JumpStartModelSpecs(JumpStartDataHolderType):
    """Data class JumpStart model specs."""

    __slots__ = [
        "model_id",
        "version",
        "min_sdk_version",
        "incremental_training_supported",
        "hosting_ecr_specs",
        "hosting_artifact_uri",
        "hosting_script_uri",
        "training_supported",
        "training_ecr_specs",
        "training_artifact_uri",
        "training_script_uri",
        "hyperparameters",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartModelSpecs object from its json representation."""
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json of header."""
        self.model_id: str = json_obj["model_id"]
        self.version: str = json_obj["version"]
        self.min_sdk_version: str = json_obj["min_sdk_version"]
        self.incremental_training_supported: bool = bool(json_obj["incremental_training_supported"])
        self.hosting_ecr_specs: dict = json_obj["hosting_ecr_specs"]
        self.hosting_artifact_uri: str = json_obj["hosting_artifact_uri"]
        self.hosting_script_uri: str = json_obj["hosting_script_uri"]
        self.training_supported: bool = bool(json_obj["training_supported"])
        if self.training_supported:
            self.training_ecr_specs: Optional[dict] = json_obj["training_ecr_specs"]
            self.training_artifact_uri: Optional[str] = json_obj["training_artifact_uri"]
            self.training_script_uri: Optional[str] = json_obj["training_script_uri"]
            self.hyperparameters: Optional[dict] = json_obj["hyperparameters"]
        else:
            self.training_ecr_specs = (
                self.training_artifact_uri
            ) = self.training_script_uri = self.hyperparameters = None

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartModelSpecs object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__}
        return json_obj


class JumpStartVersionedModelId(JumpStartDataHolderType):
    """Data class for versioned model ids."""

    __slots__ = ["model_id", "version"]

    def __init__(
        self,
        model_id: str,
        version: str,
    ) -> None:
        self.model_id = model_id
        self.version = version


class JumpStartCachedS3ContentKey(JumpStartDataHolderType):
    """Data class for the s3 cached content keys."""

    __slots__ = ["file_type", "s3_key"]

    def __init__(
        self,
        file_type: JumpStartS3FileType,
        s3_key: str,
    ) -> None:
        self.file_type = file_type
        self.s3_key = s3_key


class JumpStartCachedS3ContentValue(JumpStartDataHolderType):
    """Data class for the s3 cached content values."""

    __slots__ = ["formatted_file_content", "md5_hash"]

    def __init__(
        self,
        formatted_file_content: Union[
            Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            List[JumpStartModelSpecs],
        ],
        md5_hash: Optional[str] = None,
    ) -> None:
        self.formatted_file_content = formatted_file_content
        self.md5_hash = md5_hash
