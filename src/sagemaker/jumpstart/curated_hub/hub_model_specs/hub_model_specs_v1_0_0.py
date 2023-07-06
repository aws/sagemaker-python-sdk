from dataclasses import dataclass
from typing import Any, Dict, Union, Optional, List, Mapping


@dataclass
class Origin:
    OriginTrainingJobArn: Optional[str]
    OriginEndpointArn: Optional[str]


@dataclass
class Dependencies:
    DependencyOriginPath: str
    DependencyCopyPath: str
    DependencyType: str  # enum?


@dataclass
class DatasetConfig:
    TrainingDatasetLocation: Optional[str]  # S3 validation?
    ValidationDatasetLocation: Optional[str]  # S3 validation?
    DataFormatLocation: Optional[str]  # S3 validation?
    PredictColumn: Optional[str]


@dataclass
class Metric:
    Name: str
    Regex: str


@dataclass
class SdkEstimatorArgs:
    EntryPoint: Optional[str]
    EnableNetworkIsolation: Optional[bool]
    Environment: Mapping[str, str]
    Metrics: List[Metric]
    OutputPath: Optional[str]


@dataclass
class SdkArgs:
    MinSdkVersion: Optional[str]
    SdkEstimatorArgs: Optional[SdkEstimatorArgs]


@dataclass
class CustomImageConfig:
    ImageLocation: str  # ecr address


@dataclass
class FrameworkImageConfig:
    Framework: str
    FrameworkVersion: str
    PythonVersion: str
    TransformersVersion: Optional[str]
    BaseFramework: Optional[str]


@dataclass
class ModelArtifactConfig:
    ArtifactLocation: str  # s3 address


@dataclass
class ScriptConfig:
    ScriptLocation: str  # s3 address


@dataclass
class InstanceConfig:
    DefaultInstanceType: str
    InstanceTypeOptions: List[str]


@dataclass
class Hyperparameters:
    Name: str
    DefaultValue: Optional[str]
    Type: str  # enum
    Options: List[Optional[str]]
    Label: Optional[str]
    Description: Optional[str]
    Regex: Optional[str]
    Min: Optional[str]
    Max: Optional[str]


@dataclass
class ExtraChannels:
    ChannelName: str
    ChannelDataLocation: str  # s3 uri (prefix or key)


@dataclass
class DefaultTrainingConfig:
    SdkArgs: Optional[SdkArgs]
    CustomImageConfig: Optional[CustomImageConfig]
    FrameworkImageConfig: Optional[FrameworkImageConfig]
    ModelArtifactConfig: Optional[ModelArtifactConfig]
    ScriptConfig: Optional[ScriptConfig]
    InstanceConfig: Optional[InstanceConfig]
    Hyperparameters: List[Hyperparameters]
    ExtraChannels: List[ExtraChannels]


@dataclass
class InferenceNotebookConfig:
    NotebookLocation: str  # s3 uri,


@dataclass
class DefaultDeploymentConfig:
    SdkArgs: Optional[SdkArgs]
    CustomImageConfig: Optional[CustomImageConfig]
    FrameworkImageConfig: Optional[FrameworkImageConfig]
    ModelArtifactConfig: Optional[ModelArtifactConfig]
    ScriptConfig: Optional[ScriptConfig]
    InstanceConfig: Optional[InstanceConfig]
    InferenceNotebookConfig: Optional[InferenceNotebookConfig]


@dataclass
class HubModelSpec_v1_0_0:
    capabilities: List[str]  # enum?
    DataType: str
    MlTask: str
    Framework: Optional[str]
    Origin: Optional[Origin]
    Dependencies: List[Dependencies]
    DatasetConfig: Optional[DatasetConfig]
    DefaultTrainingConfig: Optional[DefaultTrainingConfig]
    DefaultDeploymentConfig: Optional[DefaultDeploymentConfig]
