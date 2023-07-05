
from typing import Any, Dict, Union, Optional, List, Mapping

@dataclass
class Origin:
      OriginTrainingJobArn: Optional[str]
      OriginEndpointArn: Optional[str]

@dataclass
class Dependencies:
    DependencyOriginPath: str
    DependencyCopyPath: str
    DependencyType: str #enum?

@dataclass
class DatasetConfig:
    TrainingDatasetLocation: Optional[str] # S3 validation?
    ValidationDatasetLocation: Optional[str] # S3 validation?
    DataFormatLocation: Optional[str] # S3 validation?
    PredictColumn: Optional[str]

@dataclass
class Metric:
    name: str
    regex: str

@dataclass
class SdkEstimatorArgs:
        EntryPoint: Optional[str]
        EnableNetworkIsolation: Optional[bool]
        Environment: Mapping[str, str]
        Metrics: List[Metric]
        OutputPath: Optional[str]

@dataclass
class SdkArgs
      MinSdkVersion: Optional[str]
      SdkEstimatorArgs?: {
          EntryPoint?: string,
          EnableNetworkIsolation?: boolean,
          Environment?: {<key>: <value>},
          Metrics?: {"Name": str, "Regex": str}[],
          OutputPath?: string,
      }, 

@dataclass
class DefaultTrainingConfig:
        SdkArgs?: {
            MinSdkVersion?: string,
            SdkEstimatorArgs?: {
                EntryPoint?: string,
                EnableNetworkIsolation?: boolean,
                Environment?: {<key>: <value>},
                Metrics?: {"Name": str, "Regex": str}[],
                OutputPath?: string,
            }, 
        },
        CustomImageConfig?: {
            ImageLocation: string, // ecr address
        },
        FrameworkImageConfig?: {
            Framework: string,
            FrameworkVersion: string,
            PythonVersion: string,
            TransformersVersion?: string,
            BaseFramework?: string,
        },
        ModelArtifactConfig?: {
            ArtifactLocation: string, // s3 address
        },
        ScriptConfig?: {
            ScriptLocation: string, //s3 address
        },
        InstanceConfig?: {
            DefaultInstanceType: string,
            InstanceTypeOptions: string[],
        },
        Hyperparameters: {
            Name: string,
            DefaultValue?: string,
            Type: [TEXT | INTEGER | FLOAT | LIST],
            Options?: string[],
            Label?: string,
            Description?: string,
            Regex?: string,
            Min?: string,
            Max?: string,
        }[],
        ExtraChannels: {
            ChannelName: string,
            ChannelDataLocation: string, // s3 uri (prefix or key)
        }[]

@dataclass
class HubModelSpec_v1_0_0:
    capabilities: List[str] #enum?
    DataType: str
    MlTask: str
    Framework: Optional[str]
    Origin: Optional[Origin]
    Dependencies: List[Dependencies]
    DatasetConfig: Optional[DatasetConfig]
    DefaultTrainingConfig: Optional[DefaultTrainingConfig]
