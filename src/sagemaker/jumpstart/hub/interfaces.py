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
"""This module stores types related to SageMaker JumpStart HubAPI requests and responses."""
from __future__ import absolute_import

import re
import json
import datetime

from typing import Any, Dict, List, Union, Optional
from sagemaker.jumpstart.types import (
    HubContentType,
    HubArnExtractedInfo,
    JumpStartPredictorSpecs,
    JumpStartHyperparameter,
    JumpStartDataHolderType,
    JumpStartEnvironmentVariable,
    JumpStartSerializablePayload,
    JumpStartInstanceTypeVariants,
)
from sagemaker.jumpstart.hub.parser_utils import (
    snake_to_upper_camel,
    walk_and_apply_json,
)


class HubDataHolderType(JumpStartDataHolderType):
    """Base class for many Hub API interfaces."""

    def to_json(self) -> Dict[str, Any]:
        """Returns json representation of object."""
        json_obj = {}
        for att in self.__slots__:
            if att in self._non_serializable_slots:
                continue
            if hasattr(self, att):
                cur_val = getattr(self, att)
                # Do not serialize null values.
                if cur_val is None:
                    continue
                if issubclass(type(cur_val), JumpStartDataHolderType):
                    json_obj[att] = cur_val.to_json()
                elif isinstance(cur_val, list):
                    json_obj[att] = []
                    for obj in cur_val:
                        if issubclass(type(obj), JumpStartDataHolderType):
                            json_obj[att].append(obj.to_json())
                        else:
                            json_obj[att].append(obj)
                elif isinstance(cur_val, datetime.datetime):
                    json_obj[att] = str(cur_val)
                else:
                    json_obj[att] = cur_val
        return json_obj

    def __str__(self) -> str:
        """Returns string representation of object.

        Example: "{'content_bucket': 'bucket', 'region_name': 'us-west-2'}"
        """

        att_dict = walk_and_apply_json(self.to_json(), snake_to_upper_camel)
        return f"{json.dumps(att_dict, default=lambda o: o.to_json())}"


class CreateHubResponse(HubDataHolderType):
    """Data class for the Hub from session.create_hub()"""

    __slots__ = [
        "hub_arn",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates CreateHubResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of session.create_hub() response.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.hub_arn: str = json_obj["HubArn"]


class HubContentDependency(HubDataHolderType):
    """Data class for any dependencies related to hub content.

    Content can be scripts, model artifacts, datasets, or notebooks.
    """

    __slots__ = ["dependency_copy_path", "dependency_origin_path", "dependency_type"]

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

        self.dependency_copy_path: Optional[str] = json_obj.get("DependencyCopyPath", "")
        self.dependency_origin_path: Optional[str] = json_obj.get("DependencyOriginPath", "")
        self.dependency_type: Optional[str] = json_obj.get("DependencyType", "")


class DescribeHubContentResponse(HubDataHolderType):
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
        "reference_min_version",
        "hub_name",
        "_region",
    ]

    _non_serializable_slots = ["_region"]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates DescribeHubContentResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.creation_time: datetime.datetime = json_obj["CreationTime"]
        self.document_schema_version: str = json_obj["DocumentSchemaVersion"]
        self.failure_reason: Optional[str] = json_obj.get("FailureReason")
        self.hub_arn: str = json_obj["HubArn"]
        self.hub_content_arn: str = json_obj["HubContentArn"]
        self.hub_content_dependencies = []
        if "Dependencies" in json_obj:
            self.hub_content_dependencies: Optional[List[HubContentDependency]] = [
                HubContentDependency(dep) for dep in json_obj.get(["Dependencies"])
            ]
        self.hub_content_description: str = json_obj.get("HubContentDescription")
        self.hub_content_display_name: str = json_obj.get("HubContentDisplayName")
        hub_region: Optional[str] = HubArnExtractedInfo.extract_region_from_arn(self.hub_arn)
        self._region = hub_region
        self.hub_content_type: str = json_obj.get("HubContentType")
        hub_content_document = json.loads(json_obj["HubContentDocument"])
        if self.hub_content_type == HubContentType.MODEL:
            self.hub_content_document: HubContentDocument = HubModelDocument(
                json_obj=hub_content_document,
                region=self._region,
                dependencies=self.hub_content_dependencies,
            )
        elif self.hub_content_type == HubContentType.MODEL_REFERENCE:
            self.hub_content_document: HubContentDocument = HubModelDocument(
                json_obj=hub_content_document,
                region=self._region,
                dependencies=self.hub_content_dependencies,
            )
        elif self.hub_content_type == HubContentType.NOTEBOOK:
            self.hub_content_document: HubContentDocument = HubNotebookDocument(
                json_obj=hub_content_document, region=self._region
            )
        else:
            raise ValueError(
                f"[{self.hub_content_type}] is not a valid HubContentType."
                f"Should be one of: {[item.name for item in HubContentType]}."
            )

        self.hub_content_markdown: str = json_obj.get("HubContentMarkdown")
        self.hub_content_name: str = json_obj["HubContentName"]
        self.hub_content_search_keywords: List[str] = json_obj.get("HubContentSearchKeywords")
        self.hub_content_status: str = json_obj["HubContentStatus"]
        self.hub_content_version: str = json_obj["HubContentVersion"]
        self.hub_name: str = json_obj["HubName"]

    def get_hub_region(self) -> Optional[str]:
        """Returns the region hub is in."""
        return self._region


class HubS3StorageConfig(HubDataHolderType):
    """Data class for any dependencies related to hub content.

    Includes scripts, model artifacts, datasets, or notebooks.
    """

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

        self.s3_output_path: Optional[str] = json_obj.get("S3OutputPath", "")


class DescribeHubResponse(HubDataHolderType):
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
        "_region",
    ]

    _non_serializable_slots = ["_region"]

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

        self.creation_time: datetime.datetime = datetime.datetime(json_obj["CreationTime"])
        self.failure_reason: str = json_obj["FailureReason"]
        self.hub_arn: str = json_obj["HubArn"]
        hub_region: Optional[str] = HubArnExtractedInfo.extract_region_from_arn(self.hub_arn)
        self._region = hub_region
        self.hub_description: str = json_obj["HubDescription"]
        self.hub_display_name: str = json_obj["HubDisplayName"]
        self.hub_name: str = json_obj["HubName"]
        self.hub_search_keywords: List[str] = json_obj["HubSearchKeywords"]
        self.hub_status: str = json_obj["HubStatus"]
        self.last_modified_time: datetime.datetime = datetime.datetime(json_obj["LastModifiedTime"])
        self.s3_storage_config: HubS3StorageConfig = HubS3StorageConfig(json_obj["S3StorageConfig"])

    def get_hub_region(self) -> Optional[str]:
        """Returns the region hub is in."""
        return self._region


class ImportHubResponse(HubDataHolderType):
    """Data class for the Hub from session.import_hub()"""

    __slots__ = [
        "hub_arn",
        "hub_content_arn",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates ImportHubResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.hub_arn: str = json_obj["HubArn"]
        self.hub_content_arn: str = json_obj["HubContentArn"]


class HubSummary(HubDataHolderType):
    """Data class for the HubSummary from session.list_hubs()"""

    __slots__ = [
        "creation_time",
        "hub_arn",
        "hub_description",
        "hub_display_name",
        "hub_name",
        "hub_search_keywords",
        "hub_status",
        "last_modified_time",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates HubSummary object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.creation_time: datetime.datetime = datetime.datetime(json_obj["CreationTime"])
        self.hub_arn: str = json_obj["HubArn"]
        self.hub_description: str = json_obj["HubDescription"]
        self.hub_display_name: str = json_obj["HubDisplayName"]
        self.hub_name: str = json_obj["HubName"]
        self.hub_search_keywords: List[str] = json_obj["HubSearchKeywords"]
        self.hub_status: str = json_obj["HubStatus"]
        self.last_modified_time: datetime.datetime = datetime.datetime(json_obj["LastModifiedTime"])


class ListHubsResponse(HubDataHolderType):
    """Data class for the Hub from session.list_hubs()"""

    __slots__ = [
        "hub_summaries",
        "next_token",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates ListHubsResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of session.list_hubs() response.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of session.list_hubs() response.
        """
        self.hub_summaries: List[HubSummary] = [
            HubSummary(item) for item in json_obj["HubSummaries"]
        ]
        self.next_token: str = json_obj["NextToken"]


class EcrUri(HubDataHolderType):
    """Data class for ECR image uri."""

    __slots__ = ["account", "region_name", "repository", "tag"]

    def __init__(self, uri: str):
        """Instantiates EcrUri object."""
        self.from_ecr_uri(uri)

    def from_ecr_uri(self, uri: str) -> None:
        """Parse a given aws ecr image uri into its various components."""
        uri_regex = (
            r"^(?:(?P<account_id>[a-zA-Z0-9][\w-]*)\.dkr\.ecr\.(?P<region>[a-zA-Z0-9][\w-]*)"
            r"\.(?P<tld>[a-zA-Z0-9\.-]+))\/(?P<repository_name>([a-z0-9]+"
            r"(?:[._-][a-z0-9]+)*\/)*[a-z0-9]+(?:[._-][a-z0-9]+)*)(:*)(?P<image_tag>.*)?"
        )

        parsed_image_uri = re.compile(uri_regex).match(uri)

        account = parsed_image_uri.group("account_id")
        region = parsed_image_uri.group("region")
        repository = parsed_image_uri.group("repository_name")
        tag = parsed_image_uri.group("image_tag")

        self.account = account
        self.region_name = region
        self.repository = repository
        self.tag = tag


class NotebookLocationUris(HubDataHolderType):
    """Data class for Notebook Location uri."""

    __slots__ = ["demo_notebook", "model_fit", "model_deploy"]

    def __init__(self, json_obj: Dict[str, Any]):
        """Instantiates EcrUri object."""
        self.from_json(json_obj)

    def from_json(self, json_obj: str) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.demo_notebook = json_obj.get("demo_notebook")
        self.model_fit = json_obj.get("model_fit")
        self.model_deploy = json_obj.get("model_deploy")


class HubModelDocument(HubDataHolderType):
    """Data class for model type HubContentDocument from session.describe_hub_content()."""

    SCHEMA_VERSION = "2.2.0"

    __slots__ = [
        "url",
        "min_sdk_version",
        "training_supported",
        "incremental_training_supported",
        "dynamic_container_deployment_supported",
        "hosting_ecr_uri",
        "hosting_artifact_s3_data_type",
        "hosting_artifact_compression_type",
        "hosting_artifact_uri",
        "hosting_prepacked_artifact_uri",
        "hosting_prepacked_artifact_version",
        "hosting_script_uri",
        "hosting_use_script_uri",
        "hosting_eula_uri",
        "hosting_model_package_arn",
        "training_artifact_s3_data_type",
        "training_artifact_compression_type",
        "training_model_package_artifact_uri",
        "hyperparameters",
        "inference_environment_variables",
        "training_script_uri",
        "training_prepacked_script_uri",
        "training_prepacked_script_version",
        "training_ecr_uri",
        "training_metrics",
        "training_artifact_uri",
        "inference_dependencies",
        "training_dependencies",
        "default_inference_instance_type",
        "supported_inference_instance_types",
        "default_training_instance_type",
        "supported_training_instance_types",
        "sage_maker_sdk_predictor_specifications",
        "inference_volume_size",
        "training_volume_size",
        "inference_enable_network_isolation",
        "training_enable_network_isolation",
        "fine_tuning_supported",
        "validation_supported",
        "default_training_dataset_uri",
        "resource_name_base",
        "gated_bucket",
        "default_payloads",
        "hosting_resource_requirements",
        "hosting_instance_type_variants",
        "training_instance_type_variants",
        "notebook_location_uris",
        "model_provider_icon_uri",
        "task",
        "framework",
        "datatype",
        "license",
        "contextual_help",
        "model_data_download_timeout",
        "container_startup_health_check_timeout",
        "encrypt_inter_container_traffic",
        "max_runtime_in_seconds",
        "disable_output_compression",
        "model_dir",
        "dependencies",
        "_region",
    ]

    _non_serializable_slots = ["_region"]

    def __init__(
        self,
        json_obj: Dict[str, Any],
        region: str,
        dependencies: List[HubContentDependency] = None,
    ) -> None:
        """Instantiates HubModelDocument object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content document.

        Raises:
            ValueError: When one of (json_obj) or (model_specs and studio_specs) is not provided.
        """
        self._region = region
        self.dependencies = dependencies or []
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub model document.
        """
        self.url: str = json_obj["Url"]
        self.min_sdk_version: str = json_obj["MinSdkVersion"]
        self.hosting_ecr_uri: Optional[str] = json_obj["HostingEcrUri"]
        self.hosting_artifact_uri = json_obj["HostingArtifactUri"]
        self.hosting_script_uri = json_obj["HostingScriptUri"]
        self.inference_dependencies: List[str] = json_obj["InferenceDependencies"]
        self.inference_environment_variables: List[JumpStartEnvironmentVariable] = [
            JumpStartEnvironmentVariable(env_variable, is_hub_content=True)
            for env_variable in json_obj["InferenceEnvironmentVariables"]
        ]
        self.training_supported: bool = bool(json_obj["TrainingSupported"])
        self.incremental_training_supported: bool = bool(json_obj["IncrementalTrainingSupported"])
        self.dynamic_container_deployment_supported: Optional[bool] = (
            bool(json_obj.get("DynamicContainerDeploymentSupported"))
            if json_obj.get("DynamicContainerDeploymentSupported")
            else None
        )
        self.hosting_artifact_s3_data_type: Optional[str] = json_obj.get(
            "HostingArtifactS3DataType"
        )
        self.hosting_artifact_compression_type: Optional[str] = json_obj.get(
            "HostingArtifactCompressionType"
        )
        self.hosting_prepacked_artifact_uri: Optional[str] = json_obj.get(
            "HostingPrepackedArtifactUri"
        )
        self.hosting_prepacked_artifact_version: Optional[str] = json_obj.get(
            "HostingPrepackedArtifactVersion"
        )
        self.hosting_use_script_uri: Optional[bool] = (
            bool(json_obj.get("HostingUseScriptUri"))
            if json_obj.get("HostingUseScriptUri") is not None
            else None
        )
        self.hosting_eula_uri: Optional[str] = json_obj.get("HostingEulaUri")
        self.hosting_model_package_arn: Optional[str] = json_obj.get("HostingModelPackageArn")
        self.default_inference_instance_type: Optional[str] = json_obj.get(
            "DefaultInferenceInstanceType"
        )
        self.supported_inference_instance_types: Optional[str] = json_obj.get(
            "SupportedInferenceInstanceTypes"
        )
        self.sage_maker_sdk_predictor_specifications: Optional[JumpStartPredictorSpecs] = (
            JumpStartPredictorSpecs(
                json_obj.get("SageMakerSdkPredictorSpecifications"),
                is_hub_content=True,
            )
            if json_obj.get("SageMakerSdkPredictorSpecifications")
            else None
        )
        self.inference_volume_size: Optional[int] = json_obj.get("InferenceVolumeSize")
        self.inference_enable_network_isolation: Optional[str] = json_obj.get(
            "InferenceEnableNetworkIsolation", False
        )
        self.fine_tuning_supported: Optional[bool] = (
            bool(json_obj.get("FineTuningSupported"))
            if json_obj.get("FineTuningSupported")
            else None
        )
        self.validation_supported: Optional[bool] = (
            bool(json_obj.get("ValidationSupported"))
            if json_obj.get("ValidationSupported")
            else None
        )
        self.default_training_dataset_uri: Optional[str] = json_obj.get("DefaultTrainingDatasetUri")
        self.resource_name_base: Optional[str] = json_obj.get("ResourceNameBase")
        self.gated_bucket: bool = bool(json_obj.get("GatedBucket", False))
        self.default_payloads: Optional[Dict[str, JumpStartSerializablePayload]] = (
            {
                alias: JumpStartSerializablePayload(payload, is_hub_content=True)
                for alias, payload in json_obj.get("DefaultPayloads").items()
            }
            if json_obj.get("DefaultPayloads")
            else None
        )
        self.hosting_resource_requirements: Optional[Dict[str, int]] = json_obj.get(
            "HostingResourceRequirements", None
        )
        self.hosting_instance_type_variants: Optional[JumpStartInstanceTypeVariants] = (
            JumpStartInstanceTypeVariants(
                json_obj.get("HostingInstanceTypeVariants"),
                is_hub_content=True,
            )
            if json_obj.get("HostingInstanceTypeVariants")
            else None
        )
        self.notebook_location_uris: Optional[NotebookLocationUris] = (
            NotebookLocationUris(json_obj.get("NotebookLocationUris"))
            if json_obj.get("NotebookLocationUris")
            else None
        )
        self.model_provider_icon_uri: Optional[str] = None  # Not needed for private beta
        self.task: Optional[str] = json_obj.get("Task")
        self.framework: Optional[str] = json_obj.get("Framework")
        self.datatype: Optional[str] = json_obj.get("Datatype")
        self.license: Optional[str] = json_obj.get("License")
        self.contextual_help: Optional[str] = json_obj.get("ContextualHelp")
        self.model_dir: Optional[str] = json_obj.get("ModelDir")
        # Deploy kwargs
        self.model_data_download_timeout: Optional[str] = json_obj.get("ModelDataDownloadTimeout")
        self.container_startup_health_check_timeout: Optional[str] = json_obj.get(
            "ContainerStartupHealthCheckTimeout"
        )

        if self.training_supported:
            self.training_model_package_artifact_uri: Optional[str] = json_obj.get(
                "TrainingModelPackageArtifactUri"
            )
            self.training_artifact_compression_type: Optional[str] = json_obj.get(
                "TrainingArtifactCompressionType"
            )
            self.training_artifact_s3_data_type: Optional[str] = json_obj.get(
                "TrainingArtifactS3DataType"
            )
            self.hyperparameters: List[JumpStartHyperparameter] = []
            hyperparameters: Any = json_obj.get("Hyperparameters")
            if hyperparameters is not None:
                self.hyperparameters.extend(
                    [
                        JumpStartHyperparameter(hyperparameter, is_hub_content=True)
                        for hyperparameter in hyperparameters
                    ]
                )

            self.training_script_uri: Optional[str] = json_obj.get("TrainingScriptUri")
            self.training_prepacked_script_uri: Optional[str] = json_obj.get(
                "TrainingPrepackedScriptUri"
            )
            self.training_prepacked_script_version: Optional[str] = json_obj.get(
                "TrainingPrepackedScriptVersion"
            )
            self.training_ecr_uri: Optional[str] = json_obj.get("TrainingEcrUri")
            self._non_serializable_slots.append("training_ecr_specs")
            self.training_metrics: Optional[List[Dict[str, str]]] = json_obj.get(
                "TrainingMetrics", None
            )
            self.training_artifact_uri: Optional[str] = json_obj.get("TrainingArtifactUri")
            self.training_dependencies: Optional[str] = json_obj.get("TrainingDependencies")
            self.default_training_instance_type: Optional[str] = json_obj.get(
                "DefaultTrainingInstanceType"
            )
            self.supported_training_instance_types: Optional[str] = json_obj.get(
                "SupportedTrainingInstanceTypes"
            )
            self.training_volume_size: Optional[int] = json_obj.get("TrainingVolumeSize")
            self.training_enable_network_isolation: Optional[str] = json_obj.get(
                "TrainingEnableNetworkIsolation", False
            )
            self.training_instance_type_variants: Optional[JumpStartInstanceTypeVariants] = (
                JumpStartInstanceTypeVariants(
                    json_obj.get("TrainingInstanceTypeVariants"),
                    is_hub_content=True,
                )
                if json_obj.get("TrainingInstanceTypeVariants")
                else None
            )
            # Estimator kwargs
            self.encrypt_inter_container_traffic: Optional[bool] = (
                bool(json_obj.get("EncryptInterContainerTraffic"))
                if json_obj.get("EncryptInterContainerTraffic")
                else None
            )
            self.max_runtime_in_seconds: Optional[str] = json_obj.get("MaxRuntimeInSeconds")
            self.disable_output_compression: Optional[bool] = (
                bool(json_obj.get("DisableOutputCompression"))
                if json_obj.get("DisableOutputCompression")
                else None
            )

    def get_schema_version(self) -> str:
        """Returns schema version."""
        return self.SCHEMA_VERSION

    def get_region(self) -> str:
        """Returns hub region."""
        return self._region


class HubNotebookDocument(HubDataHolderType):
    """Data class for notebook type HubContentDocument from session.describe_hub_content()."""

    SCHEMA_VERSION = "1.0.0"

    __slots__ = ["notebook_location", "dependencies", "_region"]

    _non_serializable_slots = ["_region"]

    def __init__(self, json_obj: Dict[str, Any], region: str) -> None:
        """Instantiates HubNotebookDocument object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content document.
        """
        self._region = region
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.notebook_location = json_obj["NotebookLocation"]
        self.dependencies: List[HubContentDependency] = [
            HubContentDependency(dep) for dep in json_obj["Dependencies"]
        ]

    def get_schema_version(self) -> str:
        """Returns schema version."""
        return self.SCHEMA_VERSION

    def get_region(self) -> str:
        """Returns hub region."""
        return self._region


HubContentDocument = Union[HubModelDocument, HubNotebookDocument]


class HubContentInfo(HubDataHolderType):
    """Data class for the HubContentInfo from session.list_hub_contents()."""

    __slots__ = [
        "creation_time",
        "document_schema_version",
        "hub_content_arn",
        "hub_content_name",
        "hub_content_status",
        "hub_content_type",
        "hub_content_version",
        "hub_content_description",
        "hub_content_display_name",
        "hub_content_search_keywords",
        "_region",
    ]

    _non_serializable_slots = ["_region"]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates HubContentInfo object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub content description.
        """
        self.creation_time: str = json_obj["CreationTime"]
        self.document_schema_version: str = json_obj["DocumentSchemaVersion"]
        self.hub_content_arn: str = json_obj["HubContentArn"]
        self.hub_content_name: str = json_obj["HubContentName"]
        self.hub_content_status: str = json_obj["HubContentStatus"]
        self.hub_content_type: HubContentType = HubContentType(json_obj["HubContentType"])
        self.hub_content_version: str = json_obj["HubContentVersion"]
        self.hub_content_description: Optional[str] = json_obj.get("HubContentDescription")
        self.hub_content_display_name: Optional[str] = json_obj.get("HubContentDisplayName")
        self._region: Optional[str] = HubArnExtractedInfo.extract_region_from_arn(
            self.hub_content_arn
        )
        self.hub_content_search_keywords: Optional[List[str]] = json_obj.get(
            "HubContentSearchKeywords"
        )

    def get_hub_region(self) -> Optional[str]:
        """Returns the region hub is in."""
        return self._region


class ListHubContentsResponse(HubDataHolderType):
    """Data class for the Hub from session.list_hub_contents()"""

    __slots__ = [
        "hub_content_summaries",
        "next_token",
    ]

    def __init__(self, json_obj: Dict[str, Any]) -> None:
        """Instantiates ImportHubResponse object.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub  description.
        """
        self.from_json(json_obj)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representation of hub description.
        """
        self.hub_content_summaries: List[HubContentInfo] = [
            HubContentInfo(item) for item in json_obj["HubContentSummaries"]
        ]
        self.next_token: str = json_obj["NextToken"]
