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
"""Holds the util functions used for the optimize function"""
from __future__ import absolute_import

import re
import logging
from typing import Dict, Any, Optional, Union

from sagemaker import Model
from sagemaker.enums import Tag


logger = logging.getLogger(__name__)


def _is_inferentia_or_trainium(instance_type: Optional[str]) -> bool:
    """Checks whether an instance is compatible with Inferentia.

    Args:
        instance_type (str): The instance type used for the compilation job.

    Returns:
        bool: Whether the given instance type is Inferentia or Trainium.
    """
    if isinstance(instance_type, str):
        match = re.match(r"^ml[\._]([a-z\d]+)\.?\w*$", instance_type)
        if match:
            if match[1].startswith("inf") or match[1].startswith("trn"):
                return True
    return False


def _is_image_compatible_with_optimization_job(image_uri: Optional[str]) -> bool:
    """Checks whether an instance is compatible with an optimization job.

    Args:
        image_uri (str): The image URI of the optimization job.

    Returns:
        bool: Whether the given instance type is compatible with an optimization job.
    """
    # TODO: Use specific container type instead.
    if image_uri is None:
        return True
    return "djl-inference:" in image_uri and ("-lmi" in image_uri or "-neuronx-" in image_uri)


def _generate_optimized_model(pysdk_model: Model, optimization_response: dict) -> Model:
    """Generates a new optimization model.

    Args:
        pysdk_model (Model): A PySDK model.
        optimization_response (dict): The optimization response.

    Returns:
        Model: A deployable optimized model.
    """
    pysdk_model.image_uri = optimization_response["RecommendedInferenceImage"]
    pysdk_model.env = optimization_response["OptimizationEnvironment"]
    pysdk_model.model_data["S3DataSource"]["S3Uri"] = optimization_response["ModelSource"]["S3"]
    pysdk_model.instance_type = optimization_response["DeploymentInstanceType"]
    pysdk_model.add_tags(
        {"key": Tag.OPTIMIZATION_JOB_NAME, "value": optimization_response["OptimizationJobName"]}
    )

    return pysdk_model


def _extract_model_source(
    model_data: Optional[Union[Dict[str, Any], str]], accept_eula: Optional[bool]
) -> Optional[Dict[str, Any]]:
    """Extracts model source from model data.

    Args:
        model_data (Optional[Union[Dict[str, Any], str]]): A model data.

    Returns:
        Optional[Dict[str, Any]]: Model source data.
    """
    if model_data is None:
        raise ValueError("Model Optimization Job only supports model with S3 data source.")

    s3_uri = model_data
    if isinstance(s3_uri, dict):
        s3_uri = s3_uri.get("S3DataSource").get("S3Uri")

    model_source = {"S3": {"S3Uri": s3_uri}}
    if accept_eula:
        model_source["S3"]["ModelAccessConfig"] = {"AcceptEula": True}
    return model_source


def _update_environment_variables(
    env: Optional[Dict[str, str]], new_env: Optional[Dict[str, str]]
) -> Optional[Dict[str, str]]:
    """Updates environment variables based on environment variables.

    Args:
        env (Optional[Dict[str, str]]): The environment variables.
        new_env (Optional[Dict[str, str]]): The new environment variables.

    Returns:
        Optional[Dict[str, str]]: The updated environment variables.
    """
    if new_env:
        if env:
            env.update(new_env)
        else:
            env = new_env
    return env


def _extract_speculative_draft_model_provider(
    speculative_decoding_config: Optional[Dict] = None,
) -> Optional[str]:
    """Extracts speculative draft model provider from speculative decoding config.

    Args:
        speculative_decoding_config (Optional[Dict]): A speculative decoding config.

    Returns:
        Optional[str]: The speculative draft model provider.
    """
    if speculative_decoding_config is None:
        return None

    if speculative_decoding_config.get(
        "ModelProvider"
    ) == "Custom" or speculative_decoding_config.get("ModelSource"):
        return "custom"

    return "sagemaker"


def _validate_optimization_inputs(
    output_path: Optional[str] = None,
    instance_type: Optional[str] = None,
    quantization_config: Optional[Dict] = None,
    compilation_config: Optional[Dict] = None,
) -> None:
    """Validates optimization inputs.

    Args:
        output_path (Optional[str]): The output path.
        instance_type (Optional[str]): The instance type.
        quantization_config (Optional[Dict]): The quantization config.
        compilation_config (Optional[Dict]): The compilation config.

    Raises:
        ValueError: If an optimization input is invalid.
    """
    if quantization_config and compilation_config:
        raise ValueError("Quantization config and compilation config are mutually exclusive.")

    instance_type_msg = "Please provide an instance type for %s optimization job."
    output_path_msg = "Please provide an output path for %s optimization job."

    if quantization_config:
        if not instance_type:
            raise ValueError(instance_type_msg.format("quantization"))
        if not output_path:
            raise ValueError(output_path_msg.format("quantization"))

    if compilation_config:
        if not instance_type:
            raise ValueError(instance_type_msg.format("compilation"))
        if not output_path:
            raise ValueError(output_path_msg.format("compilation"))
