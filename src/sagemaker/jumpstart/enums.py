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
"""This module stores enums related to SageMaker JumpStart."""

from __future__ import absolute_import

import re
from enum import Enum
from typing import List, Dict, Any


class ModelFramework(str, Enum):
    """Enum class for JumpStart model framework.

    The ML framework as referenced in the prefix of the model ID.
    This value does not necessarily correspond to the container name.
    """

    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    MXNET = "mxnet"
    HUGGINGFACE = "huggingface"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"


class VariableScope(str, Enum):
    """Possible value of the ``scope`` attribute for a hyperparameter or environment variable.

    Used for hosting environment variables and training hyperparameters.
    """

    CONTAINER = "container"
    ALGORITHM = "algorithm"


class JumpStartScriptScope(str, Enum):
    """Enum class for JumpStart script scopes."""

    INFERENCE = "inference"
    TRAINING = "training"


class HyperparameterValidationMode(str, Enum):
    """Possible modes for validating hyperparameters."""

    VALIDATE_PROVIDED = "validate_provided"
    VALIDATE_ALGORITHM = "validate_algorithm"
    VALIDATE_ALL = "validate_all"


class VariableTypes(str, Enum):
    """Possible types for hyperparameters and environment variables."""

    TEXT = "text"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"


class JumpStartTag(str, Enum):
    """Enum class for tag keys to apply to JumpStart models."""

    INFERENCE_MODEL_URI = "aws-jumpstart-inference-model-uri"
    INFERENCE_SCRIPT_URI = "aws-jumpstart-inference-script-uri"
    TRAINING_MODEL_URI = "aws-jumpstart-training-model-uri"
    TRAINING_SCRIPT_URI = "aws-jumpstart-training-script-uri"

    MODEL_ID = "sagemaker-sdk:jumpstart-model-id"
    MODEL_VERSION = "sagemaker-sdk:jumpstart-model-version"

    HUB_ARN = "sagemaker-hub:hub-arn"


class SerializerType(str, Enum):
    """Enum class for serializers associated with JumpStart models."""

    TEXT = "text"
    JSON = "json"
    CSV = "csv"
    RAW_BYTES = "raw_bytes"


class DeserializerType(str, Enum):
    """Enum class for deserializers associated with JumpStart models."""

    JSON = "json"


class MIMEType(str, Enum):
    """Enum class for MIME types associated with JumpStart models."""

    X_IMAGE = "application/x-image"
    LIST_TEXT = "application/list-text"
    X_TEXT = "application/x-text"
    JSON = "application/json"
    CSV = "text/csv"
    WAV = "audio/wav"

    @staticmethod
    def from_suffixed_type(mime_type_with_suffix: str) -> "MIMEType":
        """Removes suffix from type and instantiates enum."""
        base_type, _, _ = mime_type_with_suffix.partition(";")
        return MIMEType(base_type)


class NamingConventionType(str, Enum):
    """Enum class for naming conventions."""

    SNAKE_CASE = "snake_case"
    UPPER_CAMEL_CASE = "upper_camel_case"
    DEFAULT = UPPER_CAMEL_CASE

    @staticmethod
    def upper_camel_to_snake(upper_camel_case_string: str):
        """Converts UpperCamelCaseString to snake_case_string."""
        snake_case_string = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", upper_camel_case_string)
        return re.sub("([a-z0-9])([A-Z])", r"\1_\2", snake_case_string).lower()

    @staticmethod
    def snake_to_upper_camel(snake_case_string: str):
        """Converts snake_case_string to UpperCamelCaseString."""
        upper_camel_case_string = "".join(word.title() for word in snake_case_string.split("_"))
        return upper_camel_case_string


class ModelSpecKwargType(str, Enum):
    """Enum class for types of kwargs for model hub content document and model specs."""

    FIT = "fit_kwargs"
    MODEL = "model_kwargs"
    ESTIMATOR = "estimator_kwargs"
    DEPLOY = "deploy_kwargs"

    @classmethod
    def kwargs(cls) -> List[str]:
        """Returns a list of kwargs keys that each type can have"""
        return [member.value for member in cls]

    @staticmethod
    def get_model_spec_kwarg_keys(
        kwarg_type: "ModelSpecKwargType",
        naming_convention: NamingConventionType = NamingConventionType.DEFAULT,
    ) -> List[str]:
        kwargs = []
        if kwarg_type == ModelSpecKwargType.DEPLOY:
            kwargs = ["ModelDataDownloadTimeout", "ContainerStartupHealthCheckTimeout"]
        elif kwarg_type == ModelSpecKwargType.ESTIMATOR:
            kwargs = ["EncryptInterContainerTraffic", "MaxRun", "DisableOutputCompression"]
        elif kwarg_type == ModelSpecKwargType.MODEL:
            kwargs = []
        elif kwarg_type == ModelSpecKwargType.FIT:
            kwargs = []
        if naming_convention == NamingConventionType.SNAKE_CASE:
            return NamingConventionType.upper_camel_to_snake(kwargs)
        elif naming_convention == NamingConventionType.UPPER_CAMEL_CASE:
            return kwargs
        else:
            raise ValueError("Please provide a valid naming convention.")

    @staticmethod
    def get_model_spec_kwargs_from_hub_content_document(
        kwarg_type: "ModelSpecKwargType",
        hub_content_document: Dict[str, Any],
        naming_convention: NamingConventionType = NamingConventionType.UPPER_CAMEL_CASE,
    ):
        kwargs = dict()
        keys = ModelSpecKwargType.get_model_spec_kwarg_keys(
            kwarg_type, naming_convention=naming_convention
        )
        for k in keys:
            kwarg_value = hub_content_document.get(k, None)
            if kwarg_value is not None:
                kwargs[k] = kwarg_value
        return kwargs
