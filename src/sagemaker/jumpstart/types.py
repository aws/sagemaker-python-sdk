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
    and improves string representation. This class overrides the ``__eq__``
    and ``__hash__`` methods so that different objects with the same attributes/types
    can be compared.
    """

    __slots__: List[str] = []

    def __eq__(self, other: Any) -> bool:
        """Returns True if ``other`` is of the same type and has all attributes equal.

        Args:
            other (Any): Other object to which to compare this object.
        """

        if not isinstance(other, type(self)):
            return False
        if getattr(other, "__slots__", None) is None:
            return False
        if self.__slots__ != other.__slots__:
            return False
        for attribute in self.__slots__:
            if (hasattr(self, attribute) and not hasattr(other, attribute)) or (
                hasattr(other, attribute) and not hasattr(self, attribute)
            ):
                return False
            if hasattr(self, attribute) and hasattr(other, attribute):
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

        att_dict = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return f"{type(self).__name__}: {str(att_dict)}"

    def __repr__(self) -> str:
        """Returns ``__repr__`` string of object. Example:

        "JumpStartLaunchedRegionInfo at 0x7f664529efa0:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return f"{type(self).__name__} at {hex(id(self))}: {str(att_dict)}"


class JumpStartS3FileType(str, Enum):
    """Type of files published in JumpStart S3 distribution buckets."""

    MANIFEST = "manifest"
    SPECS = "specs"


class JumpStartLaunchedRegionInfo(JumpStartDataHolderType):
    """Data class for launched region info."""

    __slots__ = ["content_bucket", "region_name"]

    def __init__(self, content_bucket: str, region_name: str):
        """Instantiates JumpStartLaunchedRegionInfo object.

        Args:
            content_bucket (str): Name of JumpStart s3 content bucket associated with region.
            region_name (str): Name of JumpStart launched region.
        """
        self.content_bucket = content_bucket
        self.region_name = region_name


class JumpStartModelHeader(JumpStartDataHolderType):
    """Data class JumpStart model header."""

    __slots__ = ["model_id", "version", "min_version", "spec_key"]

    def __init__(self, header: Dict[str, str]):
        """Initializes a JumpStartModelHeader object from its json representation.

        Args:
            header (Dict[str, str]): Dictionary representation of header.
        """
        self.from_json(header)

    def to_json(self) -> Dict[str, str]:
        """Returns json representation of JumpStartModelHeader object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj

    def from_json(self, json_obj: Dict[str, str]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, str]): Dictionary representation of header.
        """
        self.model_id: str = json_obj["model_id"]
        self.version: str = json_obj["version"]
        self.min_version: str = json_obj["min_version"]
        self.spec_key: str = json_obj["spec_key"]


class JumpStartECRSpecs(JumpStartDataHolderType):
    """Data class for JumpStart ECR specs."""

    __slots__ = [
        "framework",
        "framework_version",
        "py_version",
        "huggingface_transformers_version",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartECRSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """

        self.framework = json_obj["framework"]
        self.framework_version = json_obj["framework_version"]
        self.py_version = json_obj["py_version"]
        huggingface_transformers_version = json_obj.get("huggingface_transformers_version")
        if huggingface_transformers_version is not None:
            self.huggingface_transformers_version = huggingface_transformers_version

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartECRSpecs object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartHyperparameter(JumpStartDataHolderType):
    """Data class for JumpStart hyperparameter definition in the training container."""

    __slots__ = [
        "name",
        "type",
        "options",
        "default",
        "scope",
        "min",
        "max",
        "exclusive_min",
        "exclusive_max",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartHyperparameter object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of hyperparameter.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hyperparameter.
        """

        self.name = json_obj["name"]
        self.type = json_obj["type"]
        self.default = json_obj["default"]
        self.scope = json_obj["scope"]

        options = json_obj.get("options")
        if options is not None:
            self.options = options

        min_val = json_obj.get("min")
        if min_val is not None:
            self.min = min_val

        max_val = json_obj.get("max")
        if max_val is not None:
            self.max = max_val

        exclusive_min_val = json_obj.get("exclusive_min")
        if exclusive_min_val is not None:
            self.exclusive_min = exclusive_min_val

        exclusive_max_val = json_obj.get("exclusive_max")
        if exclusive_max_val is not None:
            self.exclusive_max = exclusive_max_val

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartHyperparameter object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartEnvironmentVariable(JumpStartDataHolderType):
    """Data class for JumpStart environment variable definitions in the hosting container."""

    __slots__ = [
        "name",
        "type",
        "default",
        "scope",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartEnvironmentVariable object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of environment variable.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of environment variable.
        """

        self.name = json_obj["name"]
        self.type = json_obj["type"]
        self.default = json_obj["default"]
        self.scope = json_obj["scope"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartEnvironmentVariable object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartModelSpecs(JumpStartDataHolderType):
    """Data class JumpStart model specs."""

    __slots__ = [
        "model_id",
        "url",
        "version",
        "min_sdk_version",
        "incremental_training_supported",
        "hosting_ecr_specs",
        "hosting_artifact_key",
        "hosting_script_key",
        "training_supported",
        "training_ecr_specs",
        "training_artifact_key",
        "training_script_key",
        "hyperparameters",
        "inference_environment_variables",
        "inference_vulnerable",
        "inference_dependencies",
        "inference_vulnerabilities",
        "training_vulnerable",
        "training_dependencies",
        "training_vulnerabilities",
        "deprecated",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartModelSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """
        self.model_id: str = json_obj["model_id"]
        self.url: str = json_obj["url"]
        self.version: str = json_obj["version"]
        self.min_sdk_version: str = json_obj["min_sdk_version"]
        self.incremental_training_supported: bool = bool(json_obj["incremental_training_supported"])
        self.hosting_ecr_specs: JumpStartECRSpecs = JumpStartECRSpecs(json_obj["hosting_ecr_specs"])
        self.hosting_artifact_key: str = json_obj["hosting_artifact_key"]
        self.hosting_script_key: str = json_obj["hosting_script_key"]
        self.training_supported: bool = bool(json_obj["training_supported"])
        self.inference_environment_variables = [
            JumpStartEnvironmentVariable(env_variable)
            for env_variable in json_obj["inference_environment_variables"]
        ]
        self.inference_vulnerable: bool = bool(json_obj["inference_vulnerable"])
        self.inference_dependencies: List[str] = json_obj["inference_dependencies"]
        self.inference_vulnerabilities: List[str] = json_obj["inference_vulnerabilities"]
        self.training_vulnerable: bool = bool(json_obj["training_vulnerable"])
        self.training_dependencies: List[str] = json_obj["training_dependencies"]
        self.training_vulnerabilities: List[str] = json_obj["training_vulnerabilities"]
        self.deprecated: bool = bool(json_obj["deprecated"])

        if self.training_supported:
            self.training_ecr_specs: JumpStartECRSpecs = JumpStartECRSpecs(
                json_obj["training_ecr_specs"]
            )
            self.training_artifact_key: str = json_obj["training_artifact_key"]
            self.training_script_key: str = json_obj["training_script_key"]
            hyperparameters: Any = json_obj.get("hyperparameters")
            if hyperparameters is not None:
                self.hyperparameters = [
                    JumpStartHyperparameter(hyperparameter) for hyperparameter in hyperparameters
                ]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartModelSpecs object."""
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


class JumpStartVersionedModelId(JumpStartDataHolderType):
    """Data class for versioned model IDs."""

    __slots__ = ["model_id", "version"]

    def __init__(
        self,
        model_id: str,
        version: str,
    ) -> None:
        """Instantiates JumpStartVersionedModelId object.

        Args:
            model_id (str): JumpStart model ID.
            version (str): JumpStart model version.
        """
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
        """Instantiates JumpStartCachedS3ContentKey object.

        Args:
            file_type (JumpStartS3FileType): JumpStart file type.
            s3_key (str): object key in s3.
        """
        self.file_type = file_type
        self.s3_key = s3_key


class JumpStartCachedS3ContentValue(JumpStartDataHolderType):
    """Data class for the s3 cached content values."""

    __slots__ = ["formatted_content", "md5_hash"]

    def __init__(
        self,
        formatted_content: Union[
            Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            JumpStartModelSpecs,
        ],
        md5_hash: Optional[str] = None,
    ) -> None:
        """Instantiates JumpStartCachedS3ContentValue object.

        Args:
            formatted_content (Union[Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            JumpStartModelSpecs]):
                Formatted content for model specs and mappings from
                versioned model IDs to specs.
            md5_hash (str): md5_hash for stored file content from s3.
        """
        self.formatted_content = formatted_content
        self.md5_hash = md5_hash
