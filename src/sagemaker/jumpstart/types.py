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
# pylint: skip-file
"""This module stores types related to SageMaker JumpStart."""
from __future__ import absolute_import
import re
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from sagemaker.model_card.model_card import ModelCard, ModelPackageModelCard
from sagemaker.utils import (
    S3_PREFIX,
    get_instance_type_family,
    format_tags,
    Tags,
    deep_override_dict,
)
from sagemaker.model_metrics import ModelMetrics
from sagemaker.metadata_properties import MetadataProperties
from sagemaker.drift_check_baselines import DriftCheckBaselines
from sagemaker.jumpstart.enums import (
    JumpStartModelType,
    JumpStartScriptScope,
    JumpStartConfigRankingName,
)

from sagemaker.session import Session
from sagemaker.workflow.entities import PipelineVariable
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.enums import EndpointType
from sagemaker.jumpstart.hub.parser_utils import (
    camel_to_snake,
    walk_and_apply_json,
)


class JumpStartDataHolderType:
    """Base class for many JumpStart types.

    Allows objects to be added to dicts and sets,
    and improves string representation. This class overrides the ``__eq__``
    and ``__hash__`` methods so that different objects with the same attributes/types
    can be compared.
    """

    __slots__: List[str] = []

    _non_serializable_slots: List[str] = []

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

        att_dict = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in self._non_serializable_slots
        }
        return f"{type(self).__name__}: {str(att_dict)}"

    def __repr__(self) -> str:
        """Returns ``__repr__`` string of object. Example:

        "JumpStartLaunchedRegionInfo at 0x7f664529efa0:
        {'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in self._non_serializable_slots
        }
        return f"{type(self).__name__} at {hex(id(self))}: {str(att_dict)}"


class JumpStartS3FileType(str, Enum):
    """Type of files published in JumpStart S3 distribution buckets."""

    OPEN_WEIGHT_MANIFEST = "manifest"
    OPEN_WEIGHT_SPECS = "specs"
    PROPRIETARY_MANIFEST = "proprietary_manifest"
    PROPRIETARY_SPECS = "proprietary_specs"


class HubType(str, Enum):
    """Enum for Hub objects."""

    HUB = "Hub"


class HubContentType(str, Enum):
    """Enum for Hub content objects."""

    MODEL = "Model"
    NOTEBOOK = "Notebook"
    MODEL_REFERENCE = "ModelReference"


JumpStartContentDataType = Union[JumpStartS3FileType, HubType, HubContentType]


class JumpStartLaunchedRegionInfo(JumpStartDataHolderType):
    """Data class for launched region info."""

    __slots__ = ["content_bucket", "region_name", "gated_content_bucket", "neo_content_bucket"]

    def __init__(
        self,
        content_bucket: str,
        region_name: str,
        gated_content_bucket: Optional[str] = None,
        neo_content_bucket: Optional[str] = None,
    ):
        """Instantiates JumpStartLaunchedRegionInfo object.

        Args:
            content_bucket (str): Name of JumpStart s3 content bucket associated with region.
            region_name (str): Name of JumpStart launched region.
            gated_content_bucket (Optional[str[]): Name of JumpStart gated s3 content bucket
                optionally associated with region.
            neo_content_bucket (Optional[str]): Name of Neo service s3 content bucket
                optionally associated with region.
        """
        self.content_bucket = content_bucket
        self.gated_content_bucket = gated_content_bucket
        self.region_name = region_name
        self.neo_content_bucket = neo_content_bucket


class JumpStartModelHeader(JumpStartDataHolderType):
    """Data class JumpStart model header."""

    __slots__ = ["model_id", "version", "min_version", "spec_key", "search_keywords"]

    def __init__(self, header: Dict[str, str]):
        """Initializes a JumpStartModelHeader object from its json representation.

        Args:
            header (Dict[str, str]): Dictionary representation of header.
        """
        self.from_json(header)

    def to_json(self) -> Dict[str, str]:
        """Returns json representation of JumpStartModelHeader object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if getattr(self, att, None) is not None
        }
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
        self.search_keywords: Optional[List[str]] = json_obj.get("search_keywords")


class JumpStartECRSpecs(JumpStartDataHolderType):
    """Data class for JumpStart ECR specs."""

    __slots__ = [
        "framework",
        "framework_version",
        "py_version",
        "huggingface_transformers_version",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, spec: Dict[str, Any], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartECRSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self._is_hub_content = is_hub_content
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """

        if not json_obj:
            return

        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)

        self.framework = json_obj.get("framework")
        self.framework_version = json_obj.get("framework_version")
        self.py_version = json_obj.get("py_version")
        huggingface_transformers_version = json_obj.get("huggingface_transformers_version")
        if huggingface_transformers_version is not None:
            self.huggingface_transformers_version = huggingface_transformers_version

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartECRSpecs object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", [])
        }
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
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, spec: Dict[str, Any], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartHyperparameter object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of hyperparameter.
        """
        self._is_hub_content = is_hub_content
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hyperparameter.
        """

        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)
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

        # HubContentDocument model schema does not allow exclusive min/max.
        if self._is_hub_content:
            return

        exclusive_min_val = json_obj.get("exclusive_min")
        exclusive_max_val = json_obj.get("exclusive_max")
        if exclusive_min_val is not None:
            self.exclusive_min = exclusive_min_val
        if exclusive_max_val is not None:
            self.exclusive_max = exclusive_max_val

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartHyperparameter object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", [])
        }
        return json_obj


class JumpStartEnvironmentVariable(JumpStartDataHolderType):
    """Data class for JumpStart environment variable definitions in the hosting container."""

    __slots__ = [
        "name",
        "type",
        "default",
        "scope",
        "required_for_model_class",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, spec: Dict[str, Any], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartEnvironmentVariable object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of environment variable.
        """
        self._is_hub_content = is_hub_content
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of environment variable.
        """
        json_obj = walk_and_apply_json(json_obj, camel_to_snake)
        self.name = json_obj["name"]
        self.type = json_obj["type"]
        self.default = json_obj["default"]
        self.scope = json_obj["scope"]
        self.required_for_model_class: bool = json_obj.get("required_for_model_class", False)

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartEnvironmentVariable object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", [])
        }
        return json_obj


class JumpStartPredictorSpecs(JumpStartDataHolderType):
    """Data class for JumpStart Predictor specs."""

    __slots__ = [
        "default_content_type",
        "supported_content_types",
        "default_accept_type",
        "supported_accept_types",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, spec: Optional[Dict[str, Any]], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartPredictorSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of predictor specs.
        """
        self._is_hub_content = is_hub_content
        self.from_json(spec)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of predictor specs.
        """

        if json_obj is None:
            return

        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)
        self.default_content_type = json_obj["default_content_type"]
        self.supported_content_types = json_obj["supported_content_types"]
        self.default_accept_type = json_obj["default_accept_type"]
        self.supported_accept_types = json_obj["supported_accept_types"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartPredictorSpecs object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", [])
        }
        return json_obj


class JumpStartSerializablePayload(JumpStartDataHolderType):
    """Data class for JumpStart serialized payload specs."""

    __slots__ = [
        "raw_payload",
        "content_type",
        "accept",
        "body",
        "prompt_key",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["raw_payload", "prompt_key", "_is_hub_content"]

    def __init__(self, spec: Optional[Dict[str, Any]], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartSerializablePayload object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of payload specs.
        """
        self._is_hub_content = is_hub_content
        self.from_json(spec)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of serializable
                payload specs.

        Raises:
            KeyError: If the dictionary is missing keys.
        """

        if json_obj is None:
            return

        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)
        self.raw_payload = json_obj
        self.content_type = json_obj["content_type"]
        self.body = json_obj.get("body")
        accept = json_obj.get("accept")
        self.prompt_key = json_obj.get("prompt_key")
        if accept:
            self.accept = accept

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartSerializablePayload object."""
        return deepcopy(self.raw_payload)


class JumpStartInstanceTypeVariants(JumpStartDataHolderType):
    """Data class for JumpStart instance type variants."""

    __slots__ = [
        "regional_aliases",
        "aliases",
        "variants",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, spec: Optional[Dict[str, Any]], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartInstanceTypeVariants object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of instance type variants.
        """

        self._is_hub_content = is_hub_content

        if self._is_hub_content:
            self.from_describe_hub_content_response(spec)
        else:
            self.from_json(spec)

    def from_json(self, json_obj: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of instance type variants.
        """

        if json_obj is None:
            return

        self.aliases = None
        self.regional_aliases: Optional[dict] = json_obj.get("regional_aliases")
        self.variants: Optional[dict] = json_obj.get("variants")

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartInstance object."""
        json_obj = {
            att: getattr(self, att)
            for att in self.__slots__
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", [])
        }
        return json_obj

    def from_describe_hub_content_response(self, response: Optional[Dict[str, Any]]) -> None:
        """Sets fields in object based on DescribeHubContent response.

        Args:
            response (Dict[str, Any]): Dictionary representation of instance type variants.
        """

        if response is None:
            return

        response = walk_and_apply_json(response, camel_to_snake)
        self.aliases: Optional[dict] = response.get("aliases")
        self.regional_aliases = None
        self.variants: Optional[dict] = response.get("variants")

    def regionalize(  # pylint: disable=inconsistent-return-statements
        self, region: str
    ) -> Optional[Dict[str, Any]]:
        """Returns regionalized instance type variants."""

        if self.regional_aliases is None or self.aliases is not None:
            return
        aliases = self.regional_aliases.get(region, {})
        variants = {}
        for instance_name, properties in self.variants.items():
            if properties.get("regional_properties") is not None:
                variants.update({instance_name: properties.get("regional_properties")})
            if properties.get("properties") is not None:
                variants.update({instance_name: properties.get("properties")})
        return {"Aliases": aliases, "Variants": variants}

    def get_instance_specific_metric_definitions(
        self, instance_type: str
    ) -> List[JumpStartHyperparameter]:
        """Returns instance specific metric definitions.

        Returns empty list if a model, instance type tuple does not have specific
        metric definitions.
        """

        if self.variants is None:
            return []

        instance_specific_metric_definitions: List[Dict[str, Union[str, Any]]] = (
            self.variants.get(instance_type, {}).get("properties", {}).get("metrics", [])
        )

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_metric_definitions: List[Dict[str, Union[str, Any]]] = (
            self.variants.get(instance_type_family, {}).get("properties", {}).get("metrics", [])
            if instance_type_family not in {"", None}
            else []
        )

        instance_specific_metric_names = {
            metric_definition["Name"] for metric_definition in instance_specific_metric_definitions
        }

        metric_definitions_to_return = deepcopy(instance_specific_metric_definitions)

        for instance_family_metric_definition in instance_family_metric_definitions:
            if instance_family_metric_definition["Name"] not in instance_specific_metric_names:
                metric_definitions_to_return.append(instance_family_metric_definition)

        return metric_definitions_to_return

    def get_instance_specific_prepacked_artifact_key(self, instance_type: str) -> Optional[str]:
        """Returns instance specific model artifact key.

        Returns None if a model, instance type tuple does not have specific
        artifact key.
        """

        return self._get_instance_specific_property(
            instance_type=instance_type, property_name="prepacked_artifact_key"
        )

    def get_instance_specific_artifact_key(self, instance_type: str) -> Optional[str]:
        """Returns instance specific model artifact key.

        Returns None if a model, instance type tuple does not have specific
        artifact key.
        """

        return self._get_instance_specific_property(
            instance_type=instance_type, property_name="artifact_key"
        )

    def get_instance_specific_resource_requirements(self, instance_type: str) -> Optional[str]:
        """Returns instance specific resource requirements.

        If a value exists for both the instance family and instance type, the instance type value
        is chosen.
        """

        instance_specific_resource_requirements: dict = (
            self.variants.get(instance_type, {})
            .get("properties", {})
            .get("resource_requirements", {})
        )

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_resource_requirements: dict = (
            self.variants.get(instance_type_family, {})
            .get("properties", {})
            .get("resource_requirements", {})
        )

        return {**instance_family_resource_requirements, **instance_specific_resource_requirements}

    def _get_instance_specific_property(
        self, instance_type: str, property_name: str
    ) -> Optional[str]:
        """Returns instance specific property.

        If a value exists for both the instance family and instance type,
        the instance type value is chosen.

        Returns None if a (model, instance type, property name) tuple does not have
        specific prepacked artifact key.
        """

        if self.variants is None:
            return None

        instance_specific_property: Optional[str] = (
            self.variants.get(instance_type, {}).get("properties", {}).get(property_name, None)
        )

        if instance_specific_property:
            return instance_specific_property

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_property: Optional[str] = (
            self.variants.get(instance_type_family, {})
            .get("properties", {})
            .get(property_name, None)
            if instance_type_family not in {"", None}
            else None
        )

        return instance_family_property

    def get_instance_specific_hyperparameters(
        self, instance_type: str
    ) -> List[JumpStartHyperparameter]:
        """Returns instance specific hyperparameters.

        Returns empty list if a model, instance type tuple does not have specific
        hyperparameters.
        """

        if self.variants is None:
            return []

        instance_specific_hyperparameters: List[JumpStartHyperparameter] = [
            JumpStartHyperparameter(json)
            for json in self.variants.get(instance_type, {})
            .get("properties", {})
            .get("hyperparameters", [])
        ]

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_hyperparameters: List[JumpStartHyperparameter] = [
            JumpStartHyperparameter(json)
            for json in (
                self.variants.get(instance_type_family, {})
                .get("properties", {})
                .get("hyperparameters", [])
                if instance_type_family not in {"", None}
                else []
            )
        ]

        instance_specific_hyperparameter_names = {
            hyperparameter.name for hyperparameter in instance_specific_hyperparameters
        }

        hyperparams_to_return = deepcopy(instance_specific_hyperparameters)

        for hyperparameter in instance_family_hyperparameters:
            if hyperparameter.name not in instance_specific_hyperparameter_names:
                hyperparams_to_return.append(hyperparameter)

        return hyperparams_to_return

    def get_instance_specific_environment_variables(self, instance_type: str) -> Dict[str, str]:
        """Returns instance specific environment variables.

        Returns empty dict if a model, instance type tuple does not have specific
        environment variables.
        """

        if self.variants is None:
            return {}

        instance_specific_environment_variables: Dict[str, str] = (
            self.variants.get(instance_type, {})
            .get("properties", {})
            .get("environment_variables", {})
        )

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_environment_variables: dict = (
            self.variants.get(instance_type_family, {})
            .get("properties", {})
            .get("environment_variables", {})
            if instance_type_family not in {"", None}
            else {}
        )

        instance_family_environment_variables.update(instance_specific_environment_variables)

        return instance_family_environment_variables

    def get_instance_specific_gated_model_key_env_var_value(
        self, instance_type: str
    ) -> Optional[str]:
        """Returns instance specific gated model env var s3 key.

        Returns None if a model, instance type tuple does not have instance
        specific property.
        """

        gated_model_key_env_var_value = (
            "gated_model_env_var_uri" if self._is_hub_content else "gated_model_key_env_var_value"
        )

        return self._get_instance_specific_property(instance_type, gated_model_key_env_var_value)

    def get_instance_specific_default_inference_instance_type(
        self, instance_type: str
    ) -> Optional[str]:
        """Returns instance specific default inference instance type.

        Returns None if a model, instance type tuple does not have instance
        specific inference instance types.
        """

        return self._get_instance_specific_property(
            instance_type, "default_inference_instance_type"
        )

    def get_instance_specific_supported_inference_instance_types(
        self, instance_type: str
    ) -> List[str]:
        """Returns instance specific supported inference instance types.

        Returns empty list if a model, instance type tuple does not have instance
        specific inference instance types.
        """

        if self.variants is None:
            return []

        instance_specific_inference_instance_types: List[str] = (
            self.variants.get(instance_type, {})
            .get("properties", {})
            .get("supported_inference_instance_types", [])
        )

        instance_type_family = get_instance_type_family(instance_type)

        instance_family_inference_instance_types: List[str] = (
            self.variants.get(instance_type_family, {})
            .get("properties", {})
            .get("supported_inference_instance_types", [])
            if instance_type_family not in {"", None}
            else []
        )

        return sorted(
            list(
                set(
                    instance_specific_inference_instance_types
                    + instance_family_inference_instance_types
                )
            )
        )

    def get_image_uri(self, instance_type: str, region: Optional[str] = None) -> Optional[str]:
        """Returns image uri from instance type and region.

        Returns None if no instance type is available or found.
        None is also returned if the metadata is improperly formatted.
        """
        return self._get_regional_property(
            instance_type=instance_type, region=region, property_name="image_uri"
        )

    def get_model_package_arn(self, instance_type: str, region: str) -> Optional[str]:
        """Returns model package arn from instance type and region.

        Returns None if no instance type is available or found.
        None is also returned if the metadata is improperly formatted.
        """
        return self._get_regional_property(
            instance_type=instance_type, region=region, property_name="model_package_arn"
        )

    def _get_regional_property(
        self, instance_type: str, region: Optional[str], property_name: str
    ) -> Optional[str]:
        """Returns regional property from instance type and region.

        Returns None if no instance type is available or found.
        None is also returned if the metadata is improperly formatted.
        """
        # pylint: disable=too-many-return-statements
        # if self.variants is None or (self.aliases is None and self.regional_aliases is None):
        #    return None

        if self.variants is None:
            return None

        if region is None and self.regional_aliases is not None:
            return None

        regional_property_alias: Optional[str] = None
        regional_property_value: Optional[str] = None

        if self.regional_aliases:
            regional_property_alias = (
                self.variants.get(instance_type, {})
                .get("regional_properties", {})
                .get(property_name)
            )
        else:
            regional_property_value = (
                self.variants.get(instance_type, {}).get("properties", {}).get(property_name)
            )

        if regional_property_alias is None and regional_property_value is None:
            instance_type_family = get_instance_type_family(instance_type)
            if instance_type_family in {"", None}:
                return None
            if self.regional_aliases:
                regional_property_alias = (
                    self.variants.get(instance_type_family, {})
                    .get("regional_properties", {})
                    .get(property_name)
                )
            else:
                # if reading from HubContent, aliases are already regionalized
                regional_property_value = (
                    self.variants.get(instance_type_family, {})
                    .get("properties", {})
                    .get(property_name)
                )

        if (regional_property_alias is None or len(regional_property_alias) == 0) and (
            regional_property_value is None or len(regional_property_value) == 0
        ):
            return None

        if regional_property_alias and not regional_property_alias.startswith("$"):
            # No leading '$' indicates bad metadata.
            # There are tests to ensure this never happens.
            # However, to allow for fallback options in the unlikely event
            # of a regression, we do not raise an exception here.
            # We return None, indicating the field does not exist.
            return None

        if self.regional_aliases and region not in self.regional_aliases:
            return None

        if self.regional_aliases:
            alias_value = self.regional_aliases[region].get(regional_property_alias[1:], None)
            return alias_value
        return regional_property_value


class JumpStartAdditionalDataSources(JumpStartDataHolderType):
    """Data class of additional data sources."""

    __slots__ = ["speculative_decoding", "scripts"]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a AdditionalDataSources object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of data source.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        self.speculative_decoding: Optional[List[JumpStartModelDataSource]] = (
            [
                JumpStartModelDataSource(data_source)
                for data_source in json_obj["speculative_decoding"]
            ]
            if json_obj.get("speculative_decoding")
            else None
        )
        self.scripts: Optional[List[JumpStartModelDataSource]] = (
            [JumpStartModelDataSource(data_source) for data_source in json_obj["scripts"]]
            if json_obj.get("scripts")
            else None
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of AdditionalDataSources object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                if isinstance(cur_val, list):
                    json_obj[att] = []
                    for obj in cur_val:
                        if issubclass(type(obj), JumpStartDataHolderType):
                            json_obj[att].append(obj.to_json())
                        else:
                            json_obj[att].append(obj)
                else:
                    json_obj[att] = cur_val
        return json_obj


class ModelAccessConfig(JumpStartDataHolderType):
    """Data class of model access config that mirrors CreateModel API."""

    __slots__ = ["accept_eula"]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a ModelAccessConfig object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of data source.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        self.accept_eula: bool = json_obj["accept_eula"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of ModelAccessConfig object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class HubAccessConfig(JumpStartDataHolderType):
    """Data class of model access config that mirrors CreateModel API."""

    __slots__ = ["hub_content_arn"]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a HubAccessConfig object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of data source.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        self.hub_content_arn: bool = json_obj["accept_eula"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of ModelAccessConfig object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class S3DataSource(JumpStartDataHolderType):
    """Data class of S3 data source that mirrors CreateModel API."""

    __slots__ = [
        "compression_type",
        "s3_data_type",
        "s3_uri",
        "model_access_config",
        "hub_access_config",
    ]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a S3DataSource object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of data source.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        self.compression_type: str = json_obj["compression_type"]
        self.s3_data_type: str = json_obj["s3_data_type"]
        self.s3_uri: str = json_obj["s3_uri"]
        self.model_access_config: ModelAccessConfig = (
            ModelAccessConfig(json_obj["model_access_config"])
            if json_obj.get("model_access_config")
            else None
        )
        self.hub_access_config: HubAccessConfig = (
            HubAccessConfig(json_obj["hub_access_config"])
            if json_obj.get("hub_access_config")
            else None
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of S3DataSource object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                if issubclass(type(cur_val), JumpStartDataHolderType):
                    json_obj[att] = cur_val.to_json()
                elif cur_val:
                    json_obj[att] = cur_val
        return json_obj

    def set_bucket(self, bucket: str) -> None:
        """Sets bucket name from S3 URI."""

        if self.s3_uri.startswith(S3_PREFIX):
            s3_path = self.s3_uri[len(S3_PREFIX) :]
            old_bucket = s3_path.split("/")[0]
            key = s3_path[len(old_bucket) :]
            self.s3_uri = f"{S3_PREFIX}{bucket}{key}"  # pylint: disable=W0201
            return

        if not bucket.endswith("/"):
            bucket += "/"

        self.s3_uri = f"{S3_PREFIX}{bucket}{self.s3_uri}"  # pylint: disable=W0201


class AdditionalModelDataSource(JumpStartDataHolderType):
    """Data class of additional model data source mirrors CreateModel API."""

    SERIALIZATION_EXCLUSION_SET: Set[str] = set()

    __slots__ = ["channel_name", "s3_data_source"]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a AdditionalModelDataSource object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of data source.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        self.channel_name: str = json_obj["channel_name"]
        self.s3_data_source: S3DataSource = S3DataSource(json_obj["s3_data_source"])

    def to_json(self, exclude_keys=True) -> Dict[str, Any]:
        """Returns json representation of AdditionalModelDataSource object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                if exclude_keys and att not in self.SERIALIZATION_EXCLUSION_SET or not exclude_keys:
                    cur_val = getattr(self, att)
                    if issubclass(type(cur_val), JumpStartDataHolderType):
                        json_obj[att] = cur_val.to_json()
                    else:
                        json_obj[att] = cur_val
        return json_obj


class JumpStartModelDataSource(AdditionalModelDataSource):
    """Data class JumpStart additional model data source."""

    SERIALIZATION_EXCLUSION_SET = {"artifact_version"}

    __slots__ = list(SERIALIZATION_EXCLUSION_SET) + AdditionalModelDataSource.__slots__

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of data source.
        """
        super().from_json(json_obj)
        self.artifact_version: str = json_obj["artifact_version"]


class JumpStartBenchmarkStat(JumpStartDataHolderType):
    """Data class JumpStart benchmark stat."""

    __slots__ = ["name", "value", "unit", "concurrency"]

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a JumpStartBenchmarkStat object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of benchmark stat.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of benchmark stats.
        """
        self.name: str = json_obj["name"]
        self.value: str = json_obj["value"]
        self.unit: Union[int, str] = json_obj["unit"]
        self.concurrency: Union[int, str] = json_obj["concurrency"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartBenchmarkStat object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartConfigRanking(JumpStartDataHolderType):
    """Data class JumpStart config ranking."""

    __slots__ = ["description", "rankings"]

    def __init__(self, spec: Optional[Dict[str, Any]], is_hub_content=False):
        """Initializes a JumpStartConfigRanking object.

        Args:
            spec (Dict[str, Any]): Dictionary representation of training config ranking.
        """
        if is_hub_content:
            spec = walk_and_apply_json(spec, camel_to_snake)
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of config ranking.
        """
        self.description: str = json_obj["description"]
        self.rankings: List[str] = json_obj["rankings"]

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartConfigRanking object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj


class JumpStartMetadataBaseFields(JumpStartDataHolderType):
    """Data class JumpStart metadata base fields that can be overridden."""

    __slots__ = [
        "model_id",
        "url",
        "version",
        "min_sdk_version",
        "incremental_training_supported",
        "hosting_ecr_specs",
        "hosting_ecr_uri",
        "hosting_artifact_uri",
        "hosting_artifact_key",
        "hosting_script_key",
        "training_supported",
        "training_ecr_specs",
        "training_ecr_uri",
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
        "usage_info_message",
        "deprecated_message",
        "deprecate_warn_message",
        "default_inference_instance_type",
        "supported_inference_instance_types",
        "dynamic_container_deployment_supported",
        "hosting_resource_requirements",
        "default_training_instance_type",
        "supported_training_instance_types",
        "metrics",
        "training_prepacked_script_key",
        "training_prepacked_script_version",
        "hosting_prepacked_artifact_key",
        "hosting_prepacked_artifact_version",
        "model_kwargs",
        "deploy_kwargs",
        "estimator_kwargs",
        "fit_kwargs",
        "predictor_specs",
        "inference_volume_size",
        "training_volume_size",
        "inference_enable_network_isolation",
        "training_enable_network_isolation",
        "resource_name_base",
        "hosting_eula_key",
        "hosting_model_package_arns",
        "training_model_package_artifact_uris",
        "hosting_use_script_uri",
        "hosting_instance_type_variants",
        "training_instance_type_variants",
        "default_payloads",
        "gated_bucket",
        "model_subscription_link",
        "hosting_additional_data_sources",
        "hosting_neuron_model_id",
        "hosting_neuron_model_version",
        "hub_content_type",
        "_is_hub_content",
    ]

    _non_serializable_slots = ["_is_hub_content"]

    def __init__(self, fields: Dict[str, Any], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartMetadataFields object.

        Args:
            fields (Dict[str, Any]): Dictionary representation of metadata fields.
        """
        self._is_hub_content = is_hub_content
        self.from_json(fields)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """
        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)
        self.model_id: str = json_obj.get("model_id")
        self.url: str = json_obj.get("url")
        self.version: str = json_obj.get("version")
        self.min_sdk_version: str = json_obj.get("min_sdk_version")
        self.incremental_training_supported: bool = bool(
            json_obj.get("incremental_training_supported", False)
        )
        if self._is_hub_content:
            self.hosting_ecr_uri: Optional[str] = json_obj.get("hosting_ecr_uri")
            self._non_serializable_slots.append("hosting_ecr_specs")
        else:
            self.hosting_ecr_specs: Optional[JumpStartECRSpecs] = (
                JumpStartECRSpecs(
                    json_obj["hosting_ecr_specs"], is_hub_content=self._is_hub_content
                )
                if "hosting_ecr_specs" in json_obj
                else None
            )
            self._non_serializable_slots.append("hosting_ecr_uri")
        self.hosting_artifact_key: Optional[str] = json_obj.get("hosting_artifact_key")
        self.hosting_artifact_uri: Optional[str] = json_obj.get("hosting_artifact_uri")
        self.hosting_script_key: Optional[str] = json_obj.get("hosting_script_key")
        self.training_supported: Optional[bool] = bool(json_obj.get("training_supported", False))
        self.inference_environment_variables = [
            JumpStartEnvironmentVariable(env_variable, is_hub_content=self._is_hub_content)
            for env_variable in json_obj.get("inference_environment_variables", [])
        ]
        self.inference_vulnerable: bool = bool(json_obj.get("inference_vulnerable", False))
        self.inference_dependencies: List[str] = json_obj.get("inference_dependencies", [])
        self.inference_vulnerabilities: List[str] = json_obj.get("inference_vulnerabilities", [])
        self.training_vulnerable: bool = bool(json_obj.get("training_vulnerable", False))
        self.training_dependencies: List[str] = json_obj.get("training_dependencies", [])
        self.training_vulnerabilities: List[str] = json_obj.get("training_vulnerabilities", [])
        self.deprecated: bool = bool(json_obj.get("deprecated", False))
        self.deprecated_message: Optional[str] = json_obj.get("deprecated_message")
        self.deprecate_warn_message: Optional[str] = json_obj.get("deprecate_warn_message")
        self.usage_info_message: Optional[str] = json_obj.get("usage_info_message")
        self.default_inference_instance_type: Optional[str] = json_obj.get(
            "default_inference_instance_type"
        )
        self.default_training_instance_type: Optional[str] = json_obj.get(
            "default_training_instance_type"
        )
        self.supported_inference_instance_types: Optional[List[str]] = json_obj.get(
            "supported_inference_instance_types"
        )
        self.supported_training_instance_types: Optional[List[str]] = json_obj.get(
            "supported_training_instance_types"
        )
        self.dynamic_container_deployment_supported: Optional[bool] = bool(
            json_obj.get("dynamic_container_deployment_supported")
        )
        self.hosting_resource_requirements: Optional[Dict[str, int]] = json_obj.get(
            "hosting_resource_requirements", None
        )
        self.metrics: Optional[List[Dict[str, str]]] = json_obj.get("metrics", None)
        self.training_prepacked_script_key: Optional[str] = json_obj.get(
            "training_prepacked_script_key", None
        )
        self.hosting_prepacked_artifact_key: Optional[str] = json_obj.get(
            "hosting_prepacked_artifact_key", None
        )
        # New fields required for Hub model.
        if self._is_hub_content:
            self.training_prepacked_script_version: Optional[str] = json_obj.get(
                "training_prepacked_script_version"
            )
            self.hosting_prepacked_artifact_version: Optional[str] = json_obj.get(
                "hosting_prepacked_artifact_version"
            )
        self.model_kwargs = deepcopy(json_obj.get("model_kwargs", {}))
        self.deploy_kwargs = deepcopy(json_obj.get("deploy_kwargs", {}))
        self.predictor_specs: Optional[JumpStartPredictorSpecs] = (
            JumpStartPredictorSpecs(
                json_obj["predictor_specs"], is_hub_content=self._is_hub_content
            )
            if "predictor_specs" in json_obj
            else None
        )
        self.default_payloads: Optional[Dict[str, JumpStartSerializablePayload]] = (
            {
                alias: JumpStartSerializablePayload(payload, is_hub_content=self._is_hub_content)
                for alias, payload in json_obj["default_payloads"].items()
            }
            if json_obj.get("default_payloads")
            else None
        )
        self.gated_bucket = json_obj.get("gated_bucket", False)
        self.inference_volume_size: Optional[int] = json_obj.get("inference_volume_size")
        self.inference_enable_network_isolation: bool = json_obj.get(
            "inference_enable_network_isolation", False
        )
        self.resource_name_base: bool = json_obj.get("resource_name_base")

        self.hosting_eula_key: Optional[str] = json_obj.get("hosting_eula_key")

        model_package_arns = json_obj.get("hosting_model_package_arns")
        self.hosting_model_package_arns: Optional[Dict] = (
            model_package_arns if model_package_arns is not None else {}
        )
        self.hosting_use_script_uri: bool = json_obj.get("hosting_use_script_uri", True)

        self.hosting_instance_type_variants: Optional[JumpStartInstanceTypeVariants] = (
            JumpStartInstanceTypeVariants(
                json_obj["hosting_instance_type_variants"], self._is_hub_content
            )
            if json_obj.get("hosting_instance_type_variants")
            else None
        )
        self.hosting_additional_data_sources: Optional[JumpStartAdditionalDataSources] = (
            JumpStartAdditionalDataSources(json_obj["hosting_additional_data_sources"])
            if json_obj.get("hosting_additional_data_sources")
            else None
        )
        self.hosting_neuron_model_id: Optional[str] = json_obj.get("hosting_neuron_model_id")
        self.hosting_neuron_model_version: Optional[str] = json_obj.get(
            "hosting_neuron_model_version"
        )

        if self.training_supported:
            if self._is_hub_content:
                self.training_ecr_uri: Optional[str] = json_obj.get("training_ecr_uri")
                self._non_serializable_slots.append("training_ecr_specs")
            else:
                self.training_ecr_specs: Optional[JumpStartECRSpecs] = (
                    JumpStartECRSpecs(json_obj["training_ecr_specs"])
                    if "training_ecr_specs" in json_obj
                    else None
                )
                self._non_serializable_slots.append("training_ecr_uri")
            self.training_artifact_key: str = json_obj["training_artifact_key"]
            self.training_script_key: str = json_obj["training_script_key"]
            hyperparameters: Any = json_obj.get("hyperparameters")
            self.hyperparameters: List[JumpStartHyperparameter] = []
            if hyperparameters is not None:
                self.hyperparameters.extend(
                    [
                        JumpStartHyperparameter(hyperparameter, is_hub_content=self._is_hub_content)
                        for hyperparameter in hyperparameters
                    ]
                )
            self.estimator_kwargs = deepcopy(json_obj.get("estimator_kwargs", {}))
            self.fit_kwargs = deepcopy(json_obj.get("fit_kwargs", {}))
            self.training_volume_size: Optional[int] = json_obj.get("training_volume_size")
            self.training_enable_network_isolation: bool = json_obj.get(
                "training_enable_network_isolation", False
            )
            self.training_model_package_artifact_uris: Optional[Dict] = json_obj.get(
                "training_model_package_artifact_uris"
            )
            self.training_instance_type_variants: Optional[JumpStartInstanceTypeVariants] = (
                JumpStartInstanceTypeVariants(
                    json_obj["training_instance_type_variants"], is_hub_content=self._is_hub_content
                )
                if json_obj.get("training_instance_type_variants")
                else None
            )
        self.model_subscription_link = json_obj.get("model_subscription_link")

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartMetadataBaseFields object."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att) and att not in getattr(self, "_non_serializable_slots", []):
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
                elif isinstance(cur_val, dict):
                    json_obj[att] = {}
                    for key, val in cur_val.items():
                        if issubclass(type(val), JumpStartDataHolderType):
                            json_obj[att][key] = val.to_json()
                        else:
                            json_obj[att][key] = val
                else:
                    json_obj[att] = cur_val
        return json_obj

    def set_hub_content_type(self, hub_content_type: HubContentType) -> None:
        """Sets the hub content type."""
        if self._is_hub_content:
            self.hub_content_type = hub_content_type


class JumpStartConfigComponent(JumpStartMetadataBaseFields):
    """Data class of JumpStart config component."""

    slots = ["component_name"]

    # List of fields that is not allowed to override to JumpStartMetadataBaseFields
    OVERRIDING_DENY_LIST = [
        "model_id",
        "url",
        "version",
        "min_sdk_version",
        "deprecated",
        "deprecated_message",
        "deprecate_warn_message",
        "resource_name_base",
        "gated_bucket",
        "training_supported",
        "incremental_training_supported",
    ]

    __slots__ = slots + JumpStartMetadataBaseFields.__slots__

    def __init__(
        self, component_name: str, component: Optional[Dict[str, Any]], is_hub_content=False
    ):
        """Initializes a JumpStartConfigComponent object from its json representation.

        Args:
            component_name (str): Name of the component.
            component (Dict[str, Any]):
                Dictionary representation of the config component.
        Raises:
            ValueError: If the component field is invalid.
        """
        if is_hub_content:
            component = walk_and_apply_json(component, camel_to_snake)
        self.component_name = component_name
        super().__init__(component, is_hub_content)
        self.from_json(component)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Initializes a JumpStartConfigComponent object from its json representation.

        Args:
            json_obj (Dict[str, Any]):
                Dictionary representation of the config component.
        """
        for field in json_obj.keys():
            if field in self.__slots__:
                setattr(self, field, json_obj[field])


class JumpStartMetadataConfig(JumpStartDataHolderType):
    """Data class of JumpStart metadata config."""

    __slots__ = [
        "base_fields",
        "benchmark_metrics",
        "acceleration_configs",
        "config_components",
        "resolved_metadata_config",
        "config_name",
        "default_inference_config",
        "default_incremental_training_config",
        "supported_inference_configs",
        "supported_incremental_training_configs",
    ]

    def __init__(
        self,
        config_name: str,
        config: Dict[str, Any],
        base_fields: Dict[str, Any],
        config_components: Dict[str, JumpStartConfigComponent],
        is_hub_content=False,
    ):
        """Initializes a JumpStartMetadataConfig object from its json representation.

        Args:
            config_name (str): Name of the config,
            config (Dict[str, Any]):
                Dictionary representation of the config.
            base_fields (Dict[str, Any]):
                The default base fields that are used to construct the resolved config.
            config_components (Dict[str, JumpStartConfigComponent]):
                The list of components that are used to construct the resolved config.
        """
        if is_hub_content:
            config = walk_and_apply_json(config, camel_to_snake)
            base_fields = walk_and_apply_json(base_fields, camel_to_snake)
        self.base_fields = base_fields
        self.config_components: Dict[str, JumpStartConfigComponent] = config_components
        self.benchmark_metrics: Dict[str, List[JumpStartBenchmarkStat]] = (
            {
                stat_name: [JumpStartBenchmarkStat(stat) for stat in stats]
                for stat_name, stats in config.get("benchmark_metrics").items()
            }
            if config and config.get("benchmark_metrics")
            else None
        )
        self.acceleration_configs = config.get("acceleration_configs")
        self.resolved_metadata_config: Optional[Dict[str, Any]] = None
        self.config_name: Optional[str] = config_name
        self.default_inference_config: Optional[str] = config.get("default_inference_config")
        self.default_incremental_training_config: Optional[str] = config.get(
            "default_incremental_training_config"
        )
        self.supported_inference_configs: Optional[List[str]] = config.get(
            "supported_inference_configs"
        )
        self.supported_incremental_training_configs: Optional[List[str]] = config.get(
            "supported_incremental_training_configs"
        )

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartMetadataConfig object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj

    @property
    def resolved_config(self) -> Dict[str, Any]:
        """Returns the final config that is resolved from the components map.

        Construct the final config by applying the list of configs from list index,
        and apply to the base default fields in the current model specs.
        """
        if self.resolved_metadata_config:
            return self.resolved_metadata_config

        resolved_config = JumpStartMetadataBaseFields(self.base_fields)
        for component in self.config_components.values():
            resolved_config = deep_override_dict(
                deepcopy(resolved_config.to_json()),
                deepcopy(component.to_json()),
                component.OVERRIDING_DENY_LIST,
            )

        # Remove environment variables from resolved config if using model packages
        hosting_model_pacakge_arns = resolved_config.get("hosting_model_package_arns")
        if hosting_model_pacakge_arns is not None and hosting_model_pacakge_arns != {}:
            resolved_config["inference_environment_variables"] = []

        self.resolved_metadata_config = resolved_config

        return resolved_config


class JumpStartMetadataConfigs(JumpStartDataHolderType):
    """Data class to hold the set of JumpStart Metadata configs."""

    __slots__ = ["configs", "config_rankings", "scope"]

    def __init__(
        self,
        configs: Optional[Dict[str, JumpStartMetadataConfig]],
        config_rankings: Optional[Dict[str, JumpStartConfigRanking]],
        scope: JumpStartScriptScope = JumpStartScriptScope.INFERENCE,
    ):
        """Initializes a JumpStartMetadataConfigs object.

        Args:
            configs (Dict[str, JumpStartMetadataConfig]):
                The map of JumpStartMetadataConfig object, with config name being the key.
            config_rankings (JumpStartConfigRanking):
                Config ranking class represents the ranking of the configs in the model.
            scope (JumpStartScriptScope):
                The scope of the current config (inference or training)
        """
        self.configs = configs
        self.config_rankings = config_rankings
        self.scope = scope

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of JumpStartMetadataConfigs object."""
        json_obj = {att: getattr(self, att) for att in self.__slots__ if hasattr(self, att)}
        return json_obj

    def get_top_config_from_ranking(
        self,
        ranking_name: str = JumpStartConfigRankingName.DEFAULT,
        instance_type: Optional[str] = None,
    ) -> Optional[JumpStartMetadataConfig]:
        """Gets the best the config based on config ranking.

        Fallback to use the ordering in config names if
        ranking is not available.
        Args:
            ranking_name (str):
                The ranking name that config priority is based on.
            instance_type (Optional[str]):
                The instance type which the config selection is based on.

        Raises:
            NotImplementedError: If the scope is unrecognized.
        """

        if self.scope == JumpStartScriptScope.INFERENCE:
            instance_type_attribute = "supported_inference_instance_types"
        elif self.scope == JumpStartScriptScope.TRAINING:
            instance_type_attribute = "supported_training_instance_types"
        else:
            raise NotImplementedError(f"Unknown script scope {self.scope}")

        if self.configs and (
            not self.config_rankings or not self.config_rankings.get(ranking_name)
        ):
            ranked_config_names = sorted(list(self.configs.keys()))
        else:
            rankings = self.config_rankings.get(ranking_name)
            ranked_config_names = rankings.rankings
        for config_name in ranked_config_names:
            resolved_config = self.configs[config_name].resolved_config
            if instance_type and instance_type not in getattr(
                resolved_config, instance_type_attribute
            ):
                continue
            return self.configs[config_name]

        return None


class JumpStartModelSpecs(JumpStartMetadataBaseFields):
    """Data class JumpStart model specs."""

    slots = [
        "inference_configs",
        "inference_config_components",
        "inference_config_rankings",
        "training_configs",
        "training_config_components",
        "training_config_rankings",
    ]

    __slots__ = JumpStartMetadataBaseFields.__slots__ + slots

    def __init__(self, spec: Dict[str, Any], is_hub_content: Optional[bool] = False):
        """Initializes a JumpStartModelSpecs object from its json representation.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
            is_hub_content (Optional[bool]): Whether the model is from a private hub.
        """
        super().__init__(spec, is_hub_content)
        self.from_json(spec)
        if self.inference_configs and self.inference_configs.get_top_config_from_ranking():
            super().from_json(self.inference_configs.get_top_config_from_ranking().resolved_config)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json of header.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of spec.
        """
        super().from_json(json_obj)
        if self._is_hub_content:
            json_obj = walk_and_apply_json(json_obj, camel_to_snake)
        self.inference_config_components: Optional[Dict[str, JumpStartConfigComponent]] = (
            {
                component_name: JumpStartConfigComponent(component_name, component)
                for component_name, component in json_obj["inference_config_components"].items()
            }
            if json_obj.get("inference_config_components")
            else None
        )
        self.inference_config_rankings: Optional[Dict[str, JumpStartConfigRanking]] = (
            {
                alias: JumpStartConfigRanking(ranking, is_hub_content=self._is_hub_content)
                for alias, ranking in json_obj["inference_config_rankings"].items()
            }
            if json_obj.get("inference_config_rankings")
            else None
        )

        if self._is_hub_content:
            inference_configs_dict: Optional[Dict[str, JumpStartMetadataConfig]] = (
                {
                    alias: JumpStartMetadataConfig(
                        alias,
                        config,
                        json_obj,
                        config.config_components,
                        is_hub_content=self._is_hub_content,
                    )
                    for alias, config in json_obj["inference_configs"]["configs"].items()
                }
                if json_obj.get("inference_configs")
                else None
            )
        else:
            inference_configs_dict: Optional[Dict[str, JumpStartMetadataConfig]] = (
                {
                    alias: JumpStartMetadataConfig(
                        alias,
                        config,
                        json_obj,
                        (
                            {
                                component_name: self.inference_config_components.get(component_name)
                                for component_name in config.get("component_names")
                            }
                            if config and config.get("component_names")
                            else None
                        ),
                    )
                    for alias, config in json_obj["inference_configs"].items()
                }
                if json_obj.get("inference_configs")
                else None
            )

        self.inference_configs: Optional[JumpStartMetadataConfigs] = (
            JumpStartMetadataConfigs(
                inference_configs_dict,
                self.inference_config_rankings,
            )
            if json_obj.get("inference_configs")
            else None
        )

        if self.training_supported:
            self.training_config_components: Optional[Dict[str, JumpStartConfigComponent]] = (
                {
                    alias: JumpStartConfigComponent(alias, component)
                    for alias, component in json_obj["training_config_components"].items()
                }
                if json_obj.get("training_config_components")
                else None
            )
            self.training_config_rankings: Optional[Dict[str, JumpStartConfigRanking]] = (
                {
                    alias: JumpStartConfigRanking(ranking)
                    for alias, ranking in json_obj["training_config_rankings"].items()
                }
                if json_obj.get("training_config_rankings")
                else None
            )

            if self._is_hub_content:
                training_configs_dict: Optional[Dict[str, JumpStartMetadataConfig]] = (
                    {
                        alias: JumpStartMetadataConfig(
                            alias,
                            config,
                            json_obj,
                            config.config_components,
                            is_hub_content=self._is_hub_content,
                        )
                        for alias, config in json_obj["training_configs"]["configs"].items()
                    }
                    if json_obj.get("training_configs")
                    else None
                )
            else:
                training_configs_dict: Optional[Dict[str, JumpStartMetadataConfig]] = (
                    {
                        alias: JumpStartMetadataConfig(
                            alias,
                            config,
                            json_obj,
                            (
                                {
                                    component_name: self.training_config_components.get(
                                        component_name
                                    )
                                    for component_name in config.get("component_names")
                                }
                                if config and config.get("component_names")
                                else None
                            ),
                        )
                        for alias, config in json_obj["training_configs"].items()
                    }
                    if json_obj.get("training_configs")
                    else None
                )

            self.training_configs: Optional[JumpStartMetadataConfigs] = (
                JumpStartMetadataConfigs(
                    training_configs_dict,
                    self.training_config_rankings,
                    JumpStartScriptScope.TRAINING,
                )
                if json_obj.get("training_configs")
                else None
            )
        self.model_subscription_link = json_obj.get("model_subscription_link")

    def set_config(
        self, config_name: str, scope: JumpStartScriptScope = JumpStartScriptScope.INFERENCE
    ) -> None:
        """Apply the seleted config and resolve to the current model spec.

        Args:
            config_name (str): Name of the config.
            scope (JumpStartScriptScope, optional):
                Scope of the config. Defaults to JumpStartScriptScope.INFERENCE.

        Raises:
            ValueError: If the scope is not supported, or cannot find config name.
        """
        if scope == JumpStartScriptScope.INFERENCE:
            metadata_configs = self.inference_configs
        elif scope == JumpStartScriptScope.TRAINING and self.training_supported:
            metadata_configs = self.training_configs
        else:
            raise ValueError(f"Unknown Jumpstart script scope {scope}.")

        config_object = metadata_configs.configs.get(config_name)
        if not config_object:
            error_msg = f"Cannot find Jumpstart config name {config_name}. "
            config_names = list(metadata_configs.configs.keys())
            if config_names:
                error_msg += f"List of config names that is supported by the model: {config_names}"
            raise ValueError(error_msg)

        super().from_json(config_object.resolved_config)

    def supports_prepacked_inference(self) -> bool:
        """Returns True if the model has a prepacked inference artifact."""
        return getattr(self, "hosting_prepacked_artifact_key", None) is not None

    def use_inference_script_uri(self) -> bool:
        """Returns True if the model should use a script uri when deploying inference model."""
        if self.supports_prepacked_inference():
            return False
        return self.hosting_use_script_uri

    def use_training_model_artifact(self) -> bool:
        """Returns True if the model should use a model uri when kicking off training job."""
        # gated model never use training model artifact
        if self.gated_bucket:
            return False

        # otherwise, return true is a training model package is not set
        return len(self.training_model_package_artifact_uris or {}) == 0

    def is_gated_model(self) -> bool:
        """Returns True if the model has a EULA key or the model bucket is gated."""
        return self.gated_bucket or self.hosting_eula_key is not None

    def supports_incremental_training(self) -> bool:
        """Returns True if the model supports incremental training."""
        return self.incremental_training_supported

    def get_speculative_decoding_s3_data_sources(self) -> List[JumpStartModelDataSource]:
        """Returns data sources for speculative decoding."""
        if not self.hosting_additional_data_sources:
            return []
        return self.hosting_additional_data_sources.speculative_decoding or []

    def get_additional_s3_data_sources(self) -> List[JumpStartAdditionalDataSources]:
        """Returns a list of the additional S3 data sources for use by the model."""
        additional_data_sources = []
        if self.hosting_additional_data_sources:
            for data_source in self.hosting_additional_data_sources.to_json():
                data_sources = getattr(self.hosting_additional_data_sources, data_source) or []
                additional_data_sources.extend(data_sources)
        return additional_data_sources


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


class JumpStartCachedContentKey(JumpStartDataHolderType):
    """Data class for the cached content keys."""

    __slots__ = ["data_type", "id_info"]

    def __init__(
        self,
        data_type: JumpStartContentDataType,
        id_info: str,
    ) -> None:
        """Instantiates JumpStartCachedContentKey object.

        Args:
            data_type (JumpStartContentDataType): JumpStart content data type.
            id_info (str): if S3Content, object key in s3. if HubContent, hub content arn.
        """
        self.data_type = data_type
        self.id_info = id_info


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
        self.hub_content_name = hub_content_name
        self.hub_content_type = hub_content_type
        self.hub_content_version = hub_content_version

    @staticmethod
    def extract_region_from_arn(arn: str) -> Optional[str]:
        """Extracts hub_name, content_name, and content_version from a HubContentArn"""

        HUB_CONTENT_ARN_REGEX = (
            r"arn:(.*?):sagemaker:(.*?):(.*?):hub-content/(.*?)/(.*?)/(.*?)/(.*?)$"
        )
        HUB_ARN_REGEX = r"arn:(.*?):sagemaker:(.*?):(.*?):hub/(.*?)$"

        match = re.match(HUB_CONTENT_ARN_REGEX, arn)
        hub_region = None
        if match:
            hub_region = match.group(2)
            return hub_region

        match = re.match(HUB_ARN_REGEX, arn)
        if match:
            hub_region = match.group(2)
            return hub_region

        return hub_region


class JumpStartCachedContentValue(JumpStartDataHolderType):
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
        """Instantiates JumpStartCachedContentValue object.

        Args:
            formatted_content (Union[Dict[JumpStartVersionedModelId, JumpStartModelHeader],
            JumpStartModelSpecs]):
                Formatted content for model specs and mappings from
                versioned model IDs to specs.
            md5_hash (str): md5_hash for stored file content from s3.
        """
        self.formatted_content = formatted_content
        self.md5_hash = md5_hash


class JumpStartKwargs(JumpStartDataHolderType):
    """Data class for JumpStart object kwargs."""

    BASE_SERIALIZATION_EXCLUSION_SET: Set[str] = ["specs"]
    SERIALIZATION_EXCLUSION_SET: Set[str] = set()

    def to_kwargs_dict(self, exclude_keys: bool = True):
        """Serializes object to dictionary to be used for kwargs for method arguments."""
        kwargs_dict = {}
        for field in self.__slots__:
            if (
                exclude_keys
                and field
                not in self.SERIALIZATION_EXCLUSION_SET.union(self.BASE_SERIALIZATION_EXCLUSION_SET)
                or not exclude_keys
            ):
                att_value = getattr(self, field, None)
                if att_value is not None:
                    kwargs_dict[field] = getattr(self, field)
        return kwargs_dict


class JumpStartModelInitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartModel.__init__` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "instance_type",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "image_uri",
        "model_data",
        "source_dir",
        "entry_point",
        "env",
        "predictor_cls",
        "role",
        "name",
        "vpc_config",
        "sagemaker_session",
        "enable_network_isolation",
        "model_kms_key",
        "image_config",
        "code_location",
        "container_log_level",
        "dependencies",
        "git_config",
        "model_package_arn",
        "training_instance_type",
        "resources",
        "config_name",
        "additional_model_data_sources",
        "hub_content_type",
        "model_reference_arn",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "instance_type",
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_package_arn",
        "training_instance_type",
        "config_name",
        "hub_content_type",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
        region: Optional[str] = None,
        instance_type: Optional[str] = None,
        image_uri: Optional[Union[str, Any]] = None,
        model_data: Optional[Union[str, Any, dict]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, Any]]] = None,
        name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, Any]]]] = None,
        sagemaker_session: Optional[Any] = None,
        enable_network_isolation: Union[bool, Any] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, Any]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        model_package_arn: Optional[str] = None,
        training_instance_type: Optional[str] = None,
        resources: Optional[ResourceRequirements] = None,
        config_name: Optional[str] = None,
        additional_model_data_sources: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiates JumpStartModelInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.model_type = model_type
        self.instance_type = instance_type
        self.region = region
        self.image_uri = image_uri
        self.model_data = deepcopy(model_data)
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.env = deepcopy(env)
        self.predictor_cls = predictor_cls
        self.role = role
        self.name = name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key
        self.image_config = image_config
        self.code_location = code_location
        self.container_log_level = container_log_level
        self.dependencies = dependencies
        self.git_config = git_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.model_package_arn = model_package_arn
        self.training_instance_type = training_instance_type
        self.resources = resources
        self.config_name = config_name
        self.additional_model_data_sources = additional_model_data_sources


class JumpStartModelDeployKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartModel.deploy` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "initial_instance_count",
        "instance_type",
        "region",
        "serializer",
        "deserializer",
        "accelerator_type",
        "endpoint_name",
        "inference_component_name",
        "tags",
        "kms_key",
        "wait",
        "data_capture_config",
        "async_inference_config",
        "serverless_inference_config",
        "volume_size",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "inference_recommendation_id",
        "explainer_config",
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "sagemaker_session",
        "training_instance_type",
        "accept_eula",
        "model_reference_arn",
        "endpoint_logging",
        "resources",
        "endpoint_type",
        "config_name",
        "routing_config",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "model_id",
        "model_version",
        "model_type",
        "hub_arn",
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
        "training_instance_type",
        "config_name",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
        region: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[Any] = None,
        deserializer: Optional[Any] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        inference_component_name: Optional[str] = None,
        tags: Optional[Tags] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = None,
        data_capture_config: Optional[Any] = None,
        async_inference_config: Optional[Any] = None,
        serverless_inference_config: Optional[Any] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[Any] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        sagemaker_session: Optional[Session] = None,
        training_instance_type: Optional[str] = None,
        accept_eula: Optional[bool] = None,
        model_reference_arn: Optional[str] = None,
        endpoint_logging: Optional[bool] = None,
        resources: Optional[ResourceRequirements] = None,
        endpoint_type: Optional[EndpointType] = None,
        config_name: Optional[str] = None,
        routing_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Instantiates JumpStartModelDeployKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.model_type = model_type
        self.initial_instance_count = initial_instance_count
        self.instance_type = instance_type
        self.region = region
        self.serializer = serializer
        self.deserializer = deserializer
        self.accelerator_type = accelerator_type
        self.endpoint_name = endpoint_name
        self.inference_component_name = inference_component_name
        self.tags = format_tags(tags)
        self.kms_key = kms_key
        self.wait = wait
        self.data_capture_config = data_capture_config
        self.async_inference_config = async_inference_config
        self.serverless_inference_config = serverless_inference_config
        self.volume_size = volume_size
        self.model_data_download_timeout = model_data_download_timeout
        self.container_startup_health_check_timeout = container_startup_health_check_timeout
        self.inference_recommendation_id = inference_recommendation_id
        self.explainer_config = explainer_config
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.sagemaker_session = sagemaker_session
        self.training_instance_type = training_instance_type
        self.accept_eula = accept_eula
        self.model_reference_arn = model_reference_arn
        self.endpoint_logging = endpoint_logging
        self.resources = resources
        self.endpoint_type = endpoint_type
        self.config_name = config_name
        self.routing_config = routing_config


class JumpStartEstimatorInitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.__init__` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "instance_type",
        "instance_count",
        "region",
        "image_uri",
        "model_uri",
        "source_dir",
        "entry_point",
        "hyperparameters",
        "metric_definitions",
        "role",
        "keep_alive_period_in_seconds",
        "volume_size",
        "volume_kms_key",
        "max_run",
        "input_mode",
        "output_path",
        "output_kms_key",
        "base_job_name",
        "sagemaker_session",
        "tags",
        "subnets",
        "security_group_ids",
        "model_channel_name",
        "encrypt_inter_container_traffic",
        "use_spot_instances",
        "max_wait",
        "checkpoint_s3_uri",
        "checkpoint_local_path",
        "enable_network_isolation",
        "rules",
        "debugger_hook_config",
        "tensorboard_output_config",
        "enable_sagemaker_metrics",
        "profiler_config",
        "disable_profiler",
        "environment",
        "max_retry_attempts",
        "git_config",
        "container_log_level",
        "code_location",
        "dependencies",
        "instance_groups",
        "training_repository_access_mode",
        "training_repository_credentials_provider_arn",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "container_entry_point",
        "container_arguments",
        "disable_output_compression",
        "enable_infra_check",
        "enable_remote_debug",
        "config_name",
        "enable_session_tag_chaining",
        "hub_content_type",
        "model_reference_arn",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "hub_content_type",
        "config_name",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
        region: Optional[str] = None,
        image_uri: Optional[Union[str, Any]] = None,
        role: Optional[str] = None,
        instance_count: Optional[Union[int, Any]] = None,
        instance_type: Optional[Union[str, Any]] = None,
        keep_alive_period_in_seconds: Optional[Union[int, Any]] = None,
        volume_size: Optional[Union[int, Any]] = None,
        volume_kms_key: Optional[Union[str, Any]] = None,
        max_run: Optional[Union[int, Any]] = None,
        input_mode: Optional[Union[str, Any]] = None,
        output_path: Optional[Union[str, Any]] = None,
        output_kms_key: Optional[Union[str, Any]] = None,
        base_job_name: Optional[str] = None,
        sagemaker_session: Optional[Any] = None,
        hyperparameters: Optional[Dict[str, Union[str, Any]]] = None,
        tags: Optional[Tags] = None,
        subnets: Optional[List[Union[str, Any]]] = None,
        security_group_ids: Optional[List[Union[str, Any]]] = None,
        model_uri: Optional[str] = None,
        model_channel_name: Optional[Union[str, Any]] = None,
        metric_definitions: Optional[List[Dict[str, Union[str, Any]]]] = None,
        encrypt_inter_container_traffic: Union[bool, Any] = None,
        use_spot_instances: Optional[Union[bool, Any]] = None,
        max_wait: Optional[Union[int, Any]] = None,
        checkpoint_s3_uri: Optional[Union[str, Any]] = None,
        checkpoint_local_path: Optional[Union[str, Any]] = None,
        enable_network_isolation: Union[bool, Any] = None,
        rules: Optional[List[Any]] = None,
        debugger_hook_config: Optional[Union[Any, bool]] = None,
        tensorboard_output_config: Optional[Any] = None,
        enable_sagemaker_metrics: Optional[Union[bool, Any]] = None,
        profiler_config: Optional[Any] = None,
        disable_profiler: Optional[bool] = None,
        environment: Optional[Dict[str, Union[str, Any]]] = None,
        max_retry_attempts: Optional[Union[int, Any]] = None,
        source_dir: Optional[Union[str, Any]] = None,
        git_config: Optional[Dict[str, str]] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[Union[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        instance_groups: Optional[List[Any]] = None,
        training_repository_access_mode: Optional[Union[str, Any]] = None,
        training_repository_credentials_provider_arn: Optional[Union[str, Any]] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        container_entry_point: Optional[List[str]] = None,
        container_arguments: Optional[List[str]] = None,
        disable_output_compression: Optional[bool] = None,
        enable_infra_check: Optional[Union[bool, PipelineVariable]] = None,
        enable_remote_debug: Optional[Union[bool, PipelineVariable]] = None,
        config_name: Optional[str] = None,
        enable_session_tag_chaining: Optional[Union[bool, PipelineVariable]] = None,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.model_type = model_type
        self.instance_type = instance_type
        self.instance_count = instance_count
        self.region = region
        self.image_uri = image_uri
        self.model_uri = model_uri
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.hyperparameters = deepcopy(hyperparameters)
        self.metric_definitions = deepcopy(metric_definitions)
        self.role = role
        self.keep_alive_period_in_seconds = keep_alive_period_in_seconds
        self.volume_size = volume_size
        self.volume_kms_key = volume_kms_key
        self.max_run = max_run
        self.input_mode = input_mode
        self.output_path = output_path
        self.output_kms_key = output_kms_key
        self.base_job_name = base_job_name
        self.sagemaker_session = sagemaker_session
        self.tags = format_tags(tags)
        self.subnets = subnets
        self.security_group_ids = security_group_ids
        self.model_channel_name = model_channel_name
        self.encrypt_inter_container_traffic = encrypt_inter_container_traffic
        self.use_spot_instances = use_spot_instances
        self.max_wait = max_wait
        self.checkpoint_s3_uri = checkpoint_s3_uri
        self.checkpoint_local_path = checkpoint_local_path
        self.enable_network_isolation = enable_network_isolation
        self.rules = rules
        self.debugger_hook_config = debugger_hook_config
        self.tensorboard_output_config = tensorboard_output_config
        self.enable_sagemaker_metrics = enable_sagemaker_metrics
        self.profiler_config = profiler_config
        self.disable_profiler = disable_profiler
        self.environment = deepcopy(environment)
        self.max_retry_attempts = max_retry_attempts
        self.git_config = git_config
        self.container_log_level = container_log_level
        self.code_location = code_location
        self.dependencies = dependencies
        self.instance_groups = instance_groups
        self.training_repository_access_mode = training_repository_access_mode
        self.training_repository_credentials_provider_arn = (
            training_repository_credentials_provider_arn
        )
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.container_entry_point = container_entry_point
        self.container_arguments = container_arguments
        self.disable_output_compression = disable_output_compression
        self.enable_infra_check = enable_infra_check
        self.enable_remote_debug = enable_remote_debug
        self.config_name = config_name
        self.enable_session_tag_chaining = enable_session_tag_chaining


class JumpStartEstimatorFitKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.fit` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "region",
        "inputs",
        "wait",
        "logs",
        "job_name",
        "experiment_config",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
        "config_name",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "model_id",
        "model_version",
        "hub_arn",
        "model_type",
        "region",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "sagemaker_session",
        "config_name",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
        region: Optional[str] = None,
        inputs: Optional[Union[str, Dict, Any, Any]] = None,
        wait: Optional[bool] = None,
        logs: Optional[str] = None,
        job_name: Optional[str] = None,
        experiment_config: Optional[Dict[str, str]] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        sagemaker_session: Optional[Session] = None,
        config_name: Optional[str] = None,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.model_type = model_type
        self.region = region
        self.inputs = inputs
        self.wait = wait
        self.logs = logs
        self.job_name = job_name
        self.experiment_config = experiment_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.sagemaker_session = sagemaker_session
        self.config_name = config_name


class JumpStartEstimatorDeployKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.deploy` method."""

    __slots__ = [
        "model_id",
        "model_version",
        "hub_arn",
        "instance_type",
        "initial_instance_count",
        "region",
        "image_uri",
        "source_dir",
        "entry_point",
        "env",
        "predictor_cls",
        "serializer",
        "deserializer",
        "accelerator_type",
        "endpoint_name",
        "tags",
        "kms_key",
        "wait",
        "data_capture_config",
        "async_inference_config",
        "serverless_inference_config",
        "volume_size",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "inference_recommendation_id",
        "explainer_config",
        "role",
        "vpc_config",
        "sagemaker_session",
        "enable_network_isolation",
        "model_kms_key",
        "image_config",
        "code_location",
        "container_log_level",
        "dependencies",
        "git_config",
        "tolerate_deprecated_model",
        "tolerate_vulnerable_model",
        "model_name",
        "use_compiled_model",
        "config_name",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_id",
        "model_version",
        "hub_arn",
        "sagemaker_session",
        "config_name",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        region: Optional[str] = None,
        initial_instance_count: Optional[int] = None,
        instance_type: Optional[str] = None,
        serializer: Optional[Any] = None,
        deserializer: Optional[Any] = None,
        accelerator_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        tags: Optional[Tags] = None,
        kms_key: Optional[str] = None,
        wait: Optional[bool] = None,
        data_capture_config: Optional[Any] = None,
        async_inference_config: Optional[Any] = None,
        serverless_inference_config: Optional[Any] = None,
        volume_size: Optional[int] = None,
        model_data_download_timeout: Optional[int] = None,
        container_startup_health_check_timeout: Optional[int] = None,
        inference_recommendation_id: Optional[str] = None,
        explainer_config: Optional[Any] = None,
        image_uri: Optional[Union[str, Any]] = None,
        role: Optional[str] = None,
        predictor_cls: Optional[callable] = None,
        env: Optional[Dict[str, Union[str, Any]]] = None,
        model_name: Optional[str] = None,
        vpc_config: Optional[Dict[str, List[Union[str, Any]]]] = None,
        sagemaker_session: Optional[Any] = None,
        enable_network_isolation: Union[bool, Any] = None,
        model_kms_key: Optional[str] = None,
        image_config: Optional[Dict[str, Union[str, Any]]] = None,
        source_dir: Optional[str] = None,
        code_location: Optional[str] = None,
        entry_point: Optional[str] = None,
        container_log_level: Optional[Union[int, Any]] = None,
        dependencies: Optional[List[str]] = None,
        git_config: Optional[Dict[str, str]] = None,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        use_compiled_model: bool = False,
        config_name: Optional[str] = None,
    ) -> None:
        """Instantiates JumpStartEstimatorInitKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.instance_type = instance_type
        self.initial_instance_count = initial_instance_count
        self.region = region
        self.image_uri = image_uri
        self.source_dir = source_dir
        self.entry_point = entry_point
        self.env = deepcopy(env)
        self.predictor_cls = predictor_cls
        self.serializer = serializer
        self.deserializer = deserializer
        self.accelerator_type = accelerator_type
        self.endpoint_name = endpoint_name
        self.tags = format_tags(tags)
        self.kms_key = kms_key
        self.wait = wait
        self.data_capture_config = data_capture_config
        self.async_inference_config = async_inference_config
        self.serverless_inference_config = serverless_inference_config
        self.volume_size = volume_size
        self.model_data_download_timeout = model_data_download_timeout
        self.container_startup_health_check_timeout = container_startup_health_check_timeout
        self.inference_recommendation_id = inference_recommendation_id
        self.explainer_config = explainer_config
        self.role = role
        self.model_name = model_name
        self.vpc_config = vpc_config
        self.sagemaker_session = sagemaker_session
        self.enable_network_isolation = enable_network_isolation
        self.model_kms_key = model_kms_key
        self.image_config = image_config
        self.code_location = code_location
        self.container_log_level = container_log_level
        self.dependencies = dependencies
        self.git_config = git_config
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.use_compiled_model = use_compiled_model
        self.config_name = config_name


class JumpStartModelRegisterKwargs(JumpStartKwargs):
    """Data class for the inputs to `JumpStartEstimator.deploy` method."""

    __slots__ = [
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_id",
        "model_type",
        "model_version",
        "hub_arn",
        "sagemaker_session",
        "content_types",
        "response_types",
        "inference_instances",
        "transform_instances",
        "model_package_group_name",
        "image_uri",
        "model_metrics",
        "metadata_properties",
        "approval_status",
        "description",
        "drift_check_baselines",
        "customer_metadata_properties",
        "validation_specification",
        "domain",
        "task",
        "sample_payload_url",
        "framework",
        "framework_version",
        "nearest_model_name",
        "data_input_configuration",
        "skip_model_validation",
        "source_uri",
        "config_name",
        "model_card",
        "accept_eula",
        "specs",
    ]

    SERIALIZATION_EXCLUSION_SET = {
        "tolerate_vulnerable_model",
        "tolerate_deprecated_model",
        "region",
        "model_id",
        "model_version",
        "hub_arn",
        "sagemaker_session",
        "config_name",
    }

    def __init__(
        self,
        model_id: str,
        model_version: Optional[str] = None,
        hub_arn: Optional[str] = None,
        region: Optional[str] = None,
        model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
        tolerate_deprecated_model: Optional[bool] = None,
        tolerate_vulnerable_model: Optional[bool] = None,
        sagemaker_session: Optional[Any] = None,
        content_types: List[str] = None,
        response_types: List[str] = None,
        inference_instances: Optional[List[str]] = None,
        transform_instances: Optional[List[str]] = None,
        model_package_group_name: Optional[str] = None,
        image_uri: Optional[str] = None,
        model_metrics: Optional[ModelMetrics] = None,
        metadata_properties: Optional[MetadataProperties] = None,
        approval_status: Optional[str] = None,
        description: Optional[str] = None,
        drift_check_baselines: Optional[DriftCheckBaselines] = None,
        customer_metadata_properties: Optional[Dict[str, str]] = None,
        validation_specification: Optional[str] = None,
        domain: Optional[str] = None,
        task: Optional[str] = None,
        sample_payload_url: Optional[str] = None,
        framework: Optional[str] = None,
        framework_version: Optional[str] = None,
        nearest_model_name: Optional[str] = None,
        data_input_configuration: Optional[str] = None,
        skip_model_validation: Optional[str] = None,
        source_uri: Optional[str] = None,
        config_name: Optional[str] = None,
        model_card: Optional[Dict[ModelCard, ModelPackageModelCard]] = None,
        accept_eula: Optional[bool] = None,
    ) -> None:
        """Instantiates JumpStartModelRegisterKwargs object."""

        self.model_id = model_id
        self.model_version = model_version
        self.hub_arn = hub_arn
        self.model_type = model_type
        self.region = region
        self.image_uri = image_uri
        self.sagemaker_session = sagemaker_session
        self.tolerate_deprecated_model = tolerate_deprecated_model
        self.tolerate_vulnerable_model = tolerate_vulnerable_model
        self.content_types = content_types
        self.response_types = response_types
        self.inference_instances = inference_instances
        self.transform_instances = transform_instances
        self.model_package_group_name = model_package_group_name
        self.image_uri = image_uri
        self.model_metrics = model_metrics
        self.metadata_properties = metadata_properties
        self.approval_status = approval_status
        self.description = description
        self.drift_check_baselines = drift_check_baselines
        self.customer_metadata_properties = customer_metadata_properties
        self.validation_specification = validation_specification
        self.domain = domain
        self.task = task
        self.sample_payload_url = sample_payload_url
        self.framework = framework
        self.framework_version = framework_version
        self.nearest_model_name = nearest_model_name
        self.data_input_configuration = data_input_configuration
        self.skip_model_validation = skip_model_validation
        self.source_uri = source_uri
        self.config_name = config_name
        self.model_card = model_card
        self.accept_eula = accept_eula


class BaseDeploymentConfigDataHolder(JumpStartDataHolderType):
    """Base class for Deployment Config Data."""

    def _convert_to_pascal_case(self, attr_name: str) -> str:
        """Converts a snake_case attribute name into a camelCased string.

        Args:
            attr_name (str): The snake_case attribute name.
        Returns:
            str: The PascalCased attribute name.
        """
        return attr_name.replace("_", " ").title().replace(" ", "")

    def to_json(self) -> Dict[str, Any]:
        """Represents ``This`` object as JSON."""
        json_obj = {}
        for att in self.__slots__:
            if hasattr(self, att):
                cur_val = getattr(self, att)
                att = self._convert_to_pascal_case(att)
                json_obj[att] = self._val_to_json(cur_val)
        return json_obj

    def _val_to_json(self, val: Any) -> Any:
        """Converts the given value to JSON.

        Args:
            val (Any): The value to convert.
        Returns:
            Any: The converted json value.
        """
        if issubclass(type(val), JumpStartDataHolderType):
            if isinstance(val, JumpStartBenchmarkStat):
                val.name = val.name.replace("_", " ").title()
            return val.to_json()
        if isinstance(val, list):
            list_obj = []
            for obj in val:
                list_obj.append(self._val_to_json(obj))
            return list_obj
        if isinstance(val, dict):
            dict_obj = {}
            for k, v in val.items():
                if isinstance(v, JumpStartDataHolderType):
                    dict_obj[self._convert_to_pascal_case(k)] = self._val_to_json(v)
                else:
                    dict_obj[k] = self._val_to_json(v)
            return dict_obj
        return val


class DeploymentArgs(BaseDeploymentConfigDataHolder):
    """Dataclass representing a Deployment Args."""

    __slots__ = [
        "image_uri",
        "model_data",
        "model_package_arn",
        "environment",
        "instance_type",
        "compute_resource_requirements",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "additional_data_sources",
    ]

    def __init__(
        self,
        init_kwargs: Optional[JumpStartModelInitKwargs] = None,
        deploy_kwargs: Optional[JumpStartModelDeployKwargs] = None,
        resolved_config: Optional[Dict[str, Any]] = None,
    ):
        """Instantiates DeploymentArgs object."""
        if init_kwargs is not None:
            self.image_uri = init_kwargs.image_uri
            self.model_data = init_kwargs.model_data
            self.model_package_arn = init_kwargs.model_package_arn
            self.instance_type = init_kwargs.instance_type
            self.environment = init_kwargs.env
            if init_kwargs.resources is not None:
                self.compute_resource_requirements = (
                    init_kwargs.resources.get_compute_resource_requirements()
                )
        if deploy_kwargs is not None:
            self.model_data_download_timeout = deploy_kwargs.model_data_download_timeout
            self.container_startup_health_check_timeout = (
                deploy_kwargs.container_startup_health_check_timeout
            )
        if resolved_config is not None:
            self.default_instance_type = resolved_config.get("default_inference_instance_type")
            self.supported_instance_types = resolved_config.get(
                "supported_inference_instance_types"
            )
            self.additional_data_sources = resolved_config.get("hosting_additional_data_sources")


class DeploymentConfigMetadata(BaseDeploymentConfigDataHolder):
    """Dataclass representing a Deployment Config Metadata"""

    __slots__ = [
        "deployment_config_name",
        "deployment_args",
        "acceleration_configs",
        "benchmark_metrics",
    ]

    def __init__(
        self,
        config_name: Optional[str] = None,
        metadata_config: Optional[JumpStartMetadataConfig] = None,
        init_kwargs: Optional[JumpStartModelInitKwargs] = None,
        deploy_kwargs: Optional[JumpStartModelDeployKwargs] = None,
    ):
        """Instantiates DeploymentConfigMetadata object."""
        self.deployment_config_name = config_name
        self.deployment_args = DeploymentArgs(
            init_kwargs, deploy_kwargs, metadata_config.resolved_config
        )
        self.benchmark_metrics = metadata_config.benchmark_metrics
        self.acceleration_configs = metadata_config.acceleration_configs
