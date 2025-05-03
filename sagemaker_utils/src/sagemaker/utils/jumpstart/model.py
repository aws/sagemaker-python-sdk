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
"""This module contains the model for JumpStart HubContentDocument."""
from __future__ import absolute_import, annotations

from enum import StrEnum
from typing import Any, Dict, List, Optional, Union
from pydantic import Field, ConfigDict

from sagemaker.utils.jumpstart.configs import BaseConfig


class ModelTypeEnum(StrEnum):
    PROPRIETARY = "PROPRIETARY"
    OPEN_WEIGHTS = "OPEN_WEIGHTS"


class ContextualHelpModel(BaseConfig):
    HubFormatTrainData: Optional[List[str]] = None
    HubDefaultTrainData: Optional[List[str]] = None


class DependencyTypeEnum(StrEnum):
    SCRIPT = "SCRIPT"
    ARTIFACT = "ARTIFACT"
    DATASET = "DATASET"
    NOTEBOOK = "NOTEBOOK"
    OTHER = "OTHER"


class ResourceModel(BaseConfig):
    DisplayName: Optional[str] = None
    Url: Optional[str] = None


class HubContentDependencyModel(BaseConfig):
    HubContentArn: str


class CompressionTypeEnum(StrEnum):
    NONE = "None"
    GZIP = "Gzip"


class S3DataTypeEnum(StrEnum):
    S3PREFIX = "S3Prefix"
    S3OBJECT = "S3Object"


class HubAccessConfigModel(BaseConfig):
    HubContentArn: Optional[str] = None


class S3DataSourceModel(BaseConfig):
    CompressionType: Optional[CompressionTypeEnum] = None
    S3DataType: Optional[S3DataTypeEnum] = None
    S3Uri: Optional[str] = None
    HubAccessConfig: Optional[HubAccessConfigModel] = None


class ProviderModel(BaseConfig):
    Name: str
    Classification: str


class HostingAdditionalDataSourceModel(BaseConfig):
    ChannelName: str
    ArtifactVersion: Optional[str] = None
    S3DataSource: S3DataSourceModel = None
    HostingEulaUri: Optional[str] = None
    Provider: Optional[ProviderModel] = None


class HostingArtifactS3DataTypeEnum(StrEnum):
    S3PREFIX = "S3Prefix"
    S3OBJECT = "S3Object"


class HostingArtifactCompressionTypeEnum(StrEnum):
    NONE = "None"
    GZIP = "Gzip"


class InferenceAmiVersionEnum(StrEnum):
    AL2_AMI_SAGEMAKER_INFERENCE_GPU_2 = "al2-ami-sagemaker-inference-gpu-2"


class ScopeEnum(StrEnum):
    ALGORITHM = "algorithm"
    CONTAINER = "container"
    HYPER = "hyper"


class TypeEnum(StrEnum):
    INT = "int"
    FLOAT = "float"
    TEXT = "text"
    BOOL = "bool"


class InferenceEnvironmentVariablesModel(BaseConfig):
    Name: Optional[str] = None
    Scope: Optional[ScopeEnum] = Field(default=None, alias="Scope")
    Default: Optional[Union[int, float, str]] = None
    Min: Optional[Union[int, float, str]] = None
    Max: Optional[Union[int, float, str]] = None
    Type: Optional[TypeEnum] = Field(default=None, alias="Type")
    RequiredForModelClass: Optional[bool] = None


class SageMakerSdkPredictorSpecificationsModel(BaseConfig):
    DefaultContentType: str
    SupportedContentTypes: List[str]
    DefaultAcceptType: str
    SupportedAcceptTypes: List[str]


class OutputKeysModel(BaseConfig):
    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    generated_text: Optional[str] = None
    input_logprobs: Optional[str] = None
    label: Optional[str] = None


class DefaultPayloadsModel(BaseConfig):
    ContentType: str
    PromptKey: Optional[str] = None
    OutputKeys: Optional[OutputKeysModel] = None
    Body: Union[str, Dict[str, Any]]


class HostingResourceRequirementsModel(BaseConfig):
    NumAccelerators: Optional[int] = None
    NumCpus: Optional[int] = None
    MinMemoryMb: int


class HostingEcrSpecsModel(BaseConfig):
    Framework: str
    FrameworkVersion: str
    PyVersion: str
    HuggingfaceTransformersVersion: Optional[str] = None


class HostingVariantPropertiesModel(BaseConfig):
    ImageUri: Optional[str] = None
    EnvironmentVariables: Optional[Dict[str, str]] = None
    ModelPackageArn: Optional[str] = None
    ListingId: Optional[str] = None
    ProductId: Optional[str] = None
    ResourceRequirements: Optional[HostingResourceRequirementsModel] = None


class HostingVariantModel(BaseConfig):
    Properties: Optional[HostingVariantPropertiesModel] = None


class HostingInstanceTypeVariantsModel(BaseConfig):
    Aliases: Optional[Dict[str, str]] = None
    Variants: Optional[Dict[str, HostingVariantModel]] = None


class TrainingArtifactS3DataTypeEnum(StrEnum):
    S3PREFIX = "S3Prefix"
    S3OBJECT = "S3Object"


class TrainingArtifactCompressionTypeEnum(StrEnum):
    NONE = "None"
    GZIP = "Gzip"


class ValidatorEnum(StrEnum):
    RESOURCENAME = "resourceName"
    RESOURCETAG = "resourceTag"


class HyperparameterUpperModel(BaseConfig):
    Name: str = None
    Label: Optional[str] = None
    Description: Optional[str] = None
    Scope: ScopeEnum = None
    Validators: Optional[List[Union[str, ValidatorEnum]]] = None
    Default: Union[int, float, str] = None
    Min: Optional[Union[int, float, str]] = None
    Max: Optional[Union[int, float, str]] = None
    Type: TypeEnum = None
    Options: Optional[List[str]] = None


class HyperparameterLowerModel(BaseConfig):
    name: str = None
    label: Optional[str] = None
    description: Optional[str] = None
    scope: ScopeEnum = None
    validators: Optional[List[Union[str, ValidatorEnum]]] = None
    default: Union[int, float, str] = None
    min: Optional[Union[int, float, str]] = None
    max: Optional[Union[int, float, str]] = None
    type: TypeEnum = None
    options: Optional[List[str]] = None


class TrainingMetricModel(BaseConfig):
    Name: str
    Regex: str


class TrainingVarientPropertiesModel(BaseConfig):
    ImageUri: Optional[str] = None
    GatedModelEnvVarUri: Optional[str] = None
    TrainingArtifactUri: Optional[str] = None
    EnvironmentVariables: Optional[Dict[str, str]] = None
    Hyperparameters: Optional[List[HyperparameterLowerModel]] = None


class TrainingVariantModel(BaseConfig):
    Properties: Optional[TrainingVarientPropertiesModel] = None


class TrainingInstanceTypeVariantsModel(BaseConfig):
    Aliases: Optional[Dict[str, str]] = None
    Variants: Optional[Dict[str, TrainingVariantModel]] = None


class TrainingComponentsModel(BaseConfig):
    TrainingArtifactS3DataType: Optional[TrainingArtifactS3DataTypeEnum] = None
    TrainingArtifactCompressionType: Optional[TrainingArtifactCompressionTypeEnum] = None
    TrainingModelPackageArtifactUri: Optional[str] = None
    Hyperparameters: Optional[List[HyperparameterUpperModel]] = None
    TrainingScriptUri: Optional[str] = None
    TrainingEcrUri: Optional[str] = None
    TrainingMetrics: Optional[List[TrainingMetricModel]] = None
    TrainingArtifactUri: Optional[str] = None
    TrainingDependencies: Optional[List[str]] = None
    DefaultTrainingInstanceType: Optional[str] = None
    SupportedTrainingInstanceTypes: Optional[List[str]] = None
    TrainingVolumeSize: Optional[int] = None
    TrainingEnableNetworkIsolation: Optional[bool] = None
    FineTuningSupported: Optional[bool] = None
    ValidationSupported: Optional[bool] = None
    DefaultTrainingDatasetUri: Optional[str] = None
    EncryptInterContainerTraffic: Optional[bool] = None
    MaxRuntimeInSeconds: Optional[int] = None
    DisableOutputCompression: Optional[bool] = None
    ModelDir: Optional[str] = None
    TrainingInstanceTypeVariants: Optional[TrainingInstanceTypeVariantsModel] = None


class BenchmarkMetricModel(BaseConfig):
    Name: str
    Value: str
    Unit: str
    DisplayText: Optional[str] = None
    Concurrency: Optional[str] = None


class SpecModel(BaseConfig):
    Compiler: Optional[str] = None
    Version: Optional[str] = None


class DiyWorkflowOverridesModel(BaseConfig):
    enabled: Optional[bool] = None
    reason: Optional[str] = None


class AccelerationConfigModel(BaseConfig):
    Type: str
    Enabled: bool
    Spec: Optional[SpecModel] = None
    DiyWorkflowOverrides: Optional[Dict[str, DiyWorkflowOverridesModel]] = None


class ConfigSchemaModel(BaseConfig):
    ComponentNames: List[str]
    HubContentDependencies: Optional[List[HubContentDependencyModel]] = None
    BenchmarkMetrics: Optional[Dict[str, List[BenchmarkMetricModel]]] = None
    AccelerationConfigs: Optional[List[AccelerationConfigModel]] = None


class RankingSchemaModel(BaseConfig):
    Description: Optional[str] = None
    Rankings: Optional[List[str]] = None


class CapabilityEnum(StrEnum):
    BEDROCK_CONSOLE = "BEDROCK_CONSOLE"
    HYPERPOD_DEPLOYMENT = "HYPERPOD_DEPLOYMENT"
    SAGEMAKER_EVALUATE = "SAGEMAKER_EVALUATE"
    SAGEMAKER_MODEL_OPTIMIZE = "SAGEMAKER_MODEL_OPTIMIZE"


class DemoNotebookModel(BaseConfig):
    Title: str
    S3Uri: str = None
    IsDefault: Optional[bool] = None


class NotebookLocationsModel(BaseConfig):
    DemoNotebook: Optional[str] = None
    DemoNotebooks: Optional[List[DemoNotebookModel]] = None
    ModelFit: Optional[str] = None
    ModelDeploy: Optional[str] = None


class DependencyModel(BaseConfig):
    DependencyOriginPath: Optional[str] = None
    DependencyCopyPath: Optional[str] = None
    DependencyType: Optional[DependencyTypeEnum] = None


class HostingComponentsModel(BaseConfig):

    model_config = ConfigDict(
        validate_assignment=True,
        extra="allow",
    )

    DynamicContainerDeploymentSupported: Optional[bool] = None
    HostingAdditionalDataSources: Optional[Dict[str, List[HostingAdditionalDataSourceModel]]] = None
    BedrockIOMappingId: Optional[str] = None
    Capabilities: Optional[List[CapabilityEnum]] = None
    HostingEcrUri: Optional[str] = None
    HostingArtifactS3DataType: Optional[HostingArtifactS3DataTypeEnum] = None
    HostingArtifactCompressionType: Optional[HostingArtifactCompressionTypeEnum] = None
    HostingArtifactUri: Optional[str] = None
    HostingScriptUri: Optional[str] = None
    HostingUseScriptUri: Optional[bool] = None
    HostingEulaUri: Optional[str] = None
    HostingEulaExternalLink: Optional[str] = None
    ModelSubscriptionLink: Optional[str] = None
    ListingId: Optional[str] = None
    ProductId: Optional[str] = None
    HostingModelPackageArn: Optional[str] = None
    ModelDataDownloadTimeout: Optional[int] = None
    ContainerStartupHealthCheckTimeout: Optional[int] = None
    InferenceAmiVersion: Optional[InferenceAmiVersionEnum] = None
    InferenceEnvironmentVariables: Optional[List[InferenceEnvironmentVariablesModel]] = None
    InferenceDependencies: Optional[List[str]] = None
    DefaultInferenceInstanceType: Optional[str] = None
    SupportedInferenceInstanceTypes: Optional[List[str]] = None
    SageMakerSdkPredictorSpecifications: Optional[SageMakerSdkPredictorSpecificationsModel] = None
    InferenceVolumeSize: Optional[int] = None
    InferenceEnableNetworkIsolation: Optional[bool] = None
    DefaultPayloads: Optional[Dict[str, DefaultPayloadsModel]] = None
    HostingResourceRequirements: Optional[HostingResourceRequirementsModel] = None
    HostingEcrSpecs: Optional[HostingEcrSpecsModel] = None
    HostingInstanceTypeVariants: Optional[HostingInstanceTypeVariantsModel] = None


class HubContentDocument(HostingComponentsModel, TrainingComponentsModel):
    ModelTypes: List[ModelTypeEnum]
    Url: str
    MinSdkVersion: Optional[str] = None
    TrainingSupported: bool
    IncrementalTrainingSupported: bool
    ResourceNameBase: Optional[str] = None
    GatedBucket: Optional[bool] = None
    MarketplaceVersion: Optional[str] = None
    NotebookLocations: Optional[NotebookLocationsModel] = None
    ModelProviderIconUri: Optional[str] = None
    Task: Optional[str] = None
    Framework: Optional[str] = None
    Provider: Optional[str] = None
    DataType: Optional[str] = None
    License: Optional[str] = None
    ContextualHelp: Optional[ContextualHelpModel] = None
    Dependencies: Optional[List[DependencyModel]] = None
    InferenceConfigs: Optional[Dict[str, ConfigSchemaModel]] = None
    InferenceConfigComponents: Optional[Dict[str, HostingComponentsModel]] = None
    InferenceConfigRankings: Optional[Dict[str, RankingSchemaModel]] = None
    TrainingConfigs: Optional[Dict[str, ConfigSchemaModel]] = None
    TrainingConfigComponents: Optional[Dict[str, TrainingComponentsModel]] = None
    TrainingConfigRankings: Optional[Dict[str, RankingSchemaModel]] = None
    Resources: Optional[List[ResourceModel]] = None
    Highlights: Optional[List[str]] = None
    Capabilities: List[CapabilityEnum] = None
