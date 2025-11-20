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
"""This module contains JumpStart utilities for the SageMaker Python SDK."""
from __future__ import absolute_import

# Core JumpStart Accessors
from sagemaker.core.jumpstart.accessors import (  # noqa: F401
    JumpStartModelsAccessor,
    JumpStartS3PayloadAccessor,
    SageMakerSettings,
)

# Core JumpStart Configuration
from sagemaker.core.jumpstart.configs import JumpStartConfig  # noqa: F401

# NOTE: JumpStartEstimator is in sagemaker.train.jumpstart.estimator
# NOTE: JumpStartPredictor does not exist as a standalone class.
# Predictor functionality is provided through JumpStartPredictorSpecs in types module.

# Enums (alphabetically ordered)
from sagemaker.core.jumpstart.enums import (  # noqa: F401
    DeserializerType,
    HubContentCapability,
    HyperparameterValidationMode,
    JumpStartConfigRankingName,
    JumpStartModelType,
    JumpStartScriptScope,
    JumpStartTag,
    MIMEType,
    ModelFramework,
    ModelSpecKwargType,
    NamingConventionType,
    SerializerType,
    VariableScope,
    VariableTypes,
)

# Types (alphabetically ordered)
from sagemaker.core.jumpstart.types import (  # noqa: F401
    AdditionalModelDataSource,
    BaseDeploymentConfigDataHolder,
    DeploymentArgs,
    DeploymentConfigMetadata,
    HubAccessConfig,
    HubArnExtractedInfo,
    HubContentType,
    HubType,
    JumpStartAdditionalDataSources,
    JumpStartBenchmarkStat,
    JumpStartCachedContentKey,
    JumpStartCachedContentValue,
    JumpStartConfigComponent,
    JumpStartConfigRanking,
    JumpStartDataHolderType,
    JumpStartECRSpecs,
    JumpStartEnvironmentVariable,
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartHyperparameter,
    JumpStartInstanceTypeVariants,
    JumpStartKwargs,
    JumpStartLaunchedRegionInfo,
    JumpStartMetadataBaseFields,
    JumpStartMetadataConfig,
    JumpStartMetadataConfigs,
    JumpStartModelDataSource,
    JumpStartModelDeployKwargs,
    JumpStartModelHeader,
    JumpStartModelInitKwargs,
    JumpStartModelRegisterKwargs,
    JumpStartModelSpecs,
    JumpStartPredictorSpecs,
    JumpStartS3FileType,
    JumpStartSerializablePayload,
    JumpStartVersionedModelId,
    ModelAccessConfig,
    S3DataSource,
)

# NOTE: ModelDataType does not exist in the JumpStart module.
# This may be a reference to a type that exists elsewhere or is planned for future implementation.

__all__ = [
    # Accessors
    "JumpStartModelsAccessor",
    "JumpStartS3PayloadAccessor",
    "SageMakerSettings",
    # Configuration
    "JumpStartConfig",
    # Enums
    "DeserializerType",
    "HubContentCapability",
    "HyperparameterValidationMode",
    "JumpStartConfigRankingName",
    "JumpStartModelType",
    "JumpStartScriptScope",
    "JumpStartTag",
    "MIMEType",
    "ModelFramework",
    "ModelSpecKwargType",
    "NamingConventionType",
    "SerializerType",
    "VariableScope",
    "VariableTypes",
    # Types
    "AdditionalModelDataSource",
    "BaseDeploymentConfigDataHolder",
    "DeploymentArgs",
    "DeploymentConfigMetadata",
    "HubAccessConfig",
    "HubArnExtractedInfo",
    "HubContentType",
    "HubType",
    "JumpStartAdditionalDataSources",
    "JumpStartBenchmarkStat",
    "JumpStartCachedContentKey",
    "JumpStartCachedContentValue",
    "JumpStartConfigComponent",
    "JumpStartConfigRanking",
    "JumpStartDataHolderType",
    "JumpStartECRSpecs",
    "JumpStartEnvironmentVariable",
    "JumpStartEstimatorDeployKwargs",
    "JumpStartEstimatorFitKwargs",
    "JumpStartEstimatorInitKwargs",
    "JumpStartHyperparameter",
    "JumpStartInstanceTypeVariants",
    "JumpStartKwargs",
    "JumpStartLaunchedRegionInfo",
    "JumpStartMetadataBaseFields",
    "JumpStartMetadataConfig",
    "JumpStartMetadataConfigs",
    "JumpStartModelDataSource",
    "JumpStartModelDeployKwargs",
    "JumpStartModelHeader",
    "JumpStartModelInitKwargs",
    "JumpStartModelRegisterKwargs",
    "JumpStartModelSpecs",
    "JumpStartPredictorSpecs",
    "JumpStartS3FileType",
    "JumpStartSerializablePayload",
    "JumpStartVersionedModelId",
    "ModelAccessConfig",
    "S3DataSource",
]
