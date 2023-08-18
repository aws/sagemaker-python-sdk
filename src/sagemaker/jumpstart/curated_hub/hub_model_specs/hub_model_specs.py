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
"""This module data structures for Private Hub content."""
from __future__ import absolute_import

from dataclasses import dataclass
from typing import Optional, List, Mapping
from enum import Enum


class DependencyType(str, Enum):
    """Type of dependency for Private Hub model"""

    SCRIPT = "Script"
    ARTIFACT = "Artifact"
    DATASET = "Dataset"
    NOTEBOOK = "Notebook"
    OTHER = "Other"


@dataclass
class Origin:
    """Original location for pre-trained models for Private Hub model"""

    OriginTrainingJobArn: Optional[str]
    OriginEndpointArn: Optional[str]


@dataclass
class Dependency:
    """A model dependency for Private Hub model"""

    DependencyOriginPath: str
    DependencyCopyPath: str
    DependencyType: DependencyType


@dataclass
class DatasetConfig:
    """Dataset locations for Private Hub model"""

    TrainingDatasetLocation: Optional[str]  
    ValidationDatasetLocation: Optional[str]  
    DataFormatLocation: Optional[str]  
    PredictColumn: Optional[str]


@dataclass
class Metric:
    """A metric for Private Hub model"""

    Name: str
    Regex: str


@dataclass
class SdkArgs:
    """Arguments for JumpStart Python SDK for Private Hub model"""

    EntryPoint: Optional[str]
    EnableNetworkIsolation: Optional[bool]
    Environment: Mapping[str, str]


@dataclass
class SdkEstimatorArgs(SdkArgs):
    """Estimator arguments for Private Hub model"""

    Metrics: List[Metric]
    OutputPath: Optional[str]


@dataclass
class DefaultTrainingSdkArgs:
    """Default training arguments for Private Hub model"""

    MinSdkVersion: Optional[str]
    SdkEstimatorArgs: Optional[SdkEstimatorArgs]


@dataclass
class DefaultDeploymentSdkArgs:
    """Default deployment arguments for Private Hub model"""

    MinSdkVersion: Optional[str]
    SdkModelArgs: Optional[SdkArgs]


@dataclass
class CustomImageConfig:
    """BYOC image for Private Hub model"""

    ImageLocation: str  # ecr address


@dataclass
class FrameworkImageConfig:
    """Frameowrk image for Private Hub model"""

    Framework: str
    FrameworkVersion: str
    PythonVersion: str
    TransformersVersion: Optional[str]
    BaseFramework: Optional[str]


@dataclass
class ModelArtifactConfig:
    """Model artifact location for Private Hub model"""

    ArtifactLocation: str  # s3 address


@dataclass
class ScriptConfig:
    """Script location for Private Hub model"""

    ScriptLocation: str  # s3 address


@dataclass
class InstanceConfig:
    """Instance configuration for Private Hub model"""

    DefaultInstanceType: str
    InstanceTypeOptions: List[str]


@dataclass
class Hyperparameter:
    """Hyperparameter for Private Hub model"""

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
    """Extra channels for Private Hub model"""

    ChannelName: str
    ChannelDataLocation: str  # s3 uri (prefix or key)


@dataclass
class DefaultTrainingConfig:
    """Default training configuration arguments for Private Hub model"""

    SdkArgs: Optional[DefaultTrainingSdkArgs]
    CustomImageConfig: Optional[CustomImageConfig]
    FrameworkImageConfig: Optional[FrameworkImageConfig]
    ModelArtifactConfig: Optional[ModelArtifactConfig]
    ScriptConfig: Optional[ScriptConfig]
    InstanceConfig: Optional[InstanceConfig]
    Hyperparameters: List[Hyperparameter]
    ExtraChannels: List[ExtraChannels]


@dataclass
class InferenceNotebookConfig:
    """Jupyter notebook location for Private Hub model"""

    NotebookLocation: str  # s3 uri,


@dataclass
class DefaultDeploymentConfig:
    """Default deployment configuration arguments for Private Hub model"""

    SdkArgs: Optional[DefaultDeploymentSdkArgs]
    CustomImageConfig: Optional[CustomImageConfig]
    FrameworkImageConfig: Optional[FrameworkImageConfig]
    ModelArtifactConfig: Optional[ModelArtifactConfig]
    ScriptConfig: Optional[ScriptConfig]
    InstanceConfig: Optional[InstanceConfig]
    InferenceNotebookConfig: Optional[InferenceNotebookConfig]


class ModelCapabilities(str, Enum):
    """Model capabilities of a Private Hub model"""

    TRAINING = "Training"
    INCREMENTAL_TRAINING = "IncrementalTraining"
    VALIDATION = "Validation"


@dataclass
class HubModelSpec_v1_0_0:
    """Dataclass for Private Hub Model document"""

    Capabilities: List[ModelCapabilities]
    DataType: str
    MlTask: str
    Framework: Optional[str]
    Origin: Optional[Origin]
    Dependencies: List[Dependency]
    DatasetConfig: Optional[DatasetConfig]
    DefaultTrainingConfig: Optional[DefaultTrainingConfig]
    DefaultDeploymentConfig: Optional[DefaultDeploymentConfig]
