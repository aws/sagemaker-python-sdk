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
from typing import Dict, Any, Optional, Union, List, Tuple

from sagemaker import Model
from sagemaker.enums import Tag

logger = logging.getLogger(__name__)


SPECULATIVE_DRAFT_MODEL = "/opt/ml/additional-model-data-sources"


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
    recommended_image_uri = optimization_response.get("OptimizationOutput", {}).get(
        "RecommendedInferenceImage"
    )
    s3_uri = optimization_response.get("OutputConfig", {}).get("S3OutputLocation")
    deployment_instance_type = optimization_response.get("DeploymentInstanceType")

    if recommended_image_uri:
        pysdk_model.image_uri = recommended_image_uri
    if s3_uri:
        pysdk_model.model_data["S3DataSource"]["S3Uri"] = s3_uri
    if deployment_instance_type:
        pysdk_model.instance_type = deployment_instance_type

    pysdk_model.add_tags(
        {"Key": Tag.OPTIMIZATION_JOB_NAME, "Value": optimization_response["OptimizationJobName"]}
    )
    return pysdk_model


def _is_optimized(pysdk_model: Model) -> bool:
    """Checks whether an optimization model is optimized.

    Args:
        pysdk_model (Model): A PySDK model.

    Return:
        bool: Whether the given model type is optimized.
    """
    optimized_tags = [Tag.OPTIMIZATION_JOB_NAME, Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER]
    if hasattr(pysdk_model, "_tags") and pysdk_model._tags:
        if isinstance(pysdk_model._tags, dict):
            return pysdk_model._tags.get("Key") in optimized_tags
        for tag in pysdk_model._tags:
            if tag.get("Key") in optimized_tags:
                return True
    return False


def _generate_model_source(
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


def _extracts_and_validates_speculative_model_source(
    speculative_decoding_config: Dict,
) -> str:
    """Extracts model source from speculative decoding config.

    Args:
        speculative_decoding_config (Optional[Dict]): A speculative decoding config.

    Returns:
        str: Model source.

    Raises:
        ValueError: If model source is none.
    """
    model_source: str = speculative_decoding_config.get("ModelSource")

    if not model_source:
        raise ValueError("ModelSource must be provided in speculative decoding config.")
    return model_source


def _generate_channel_name(additional_model_data_sources: Optional[List[Dict]]) -> str:
    """Generates a channel name.

    Args:
        additional_model_data_sources (Optional[List[Dict]]): The additional model data sources.

    Returns:
        str: The channel name.
    """
    channel_name = "draft_model"
    if additional_model_data_sources and len(additional_model_data_sources) > 0:
        channel_name = additional_model_data_sources[0].get("ChannelName", channel_name)

    return channel_name


def _generate_additional_model_data_sources(
    model_source: str,
    channel_name: str,
    accept_eula: bool = False,
    s3_data_type: Optional[str] = "S3Prefix",
    compression_type: Optional[str] = "None",
) -> List[Dict]:
    """Generates additional model data sources.

    Args:
        model_source (Optional[str]): The model source.
        channel_name (Optional[str]): The channel name.
        accept_eula (Optional[bool]): Whether to accept eula or not.
        s3_data_type (Optional[str]): The S3 data type, defaults to 'S3Prefix'.
        compression_type (Optional[str]): The compression type, defaults to None.

    Returns:
        List[Dict]: The additional model data sources.
    """

    additional_model_data_source = {
        "ChannelName": channel_name,
        "S3DataSource": {
            "S3Uri": model_source,
            "S3DataType": s3_data_type,
            "CompressionType": compression_type,
        },
    }
    if accept_eula:
        additional_model_data_source["S3DataSource"]["ModelAccessConfig"] = {"ACCEPT_EULA": True}

    return [additional_model_data_source]


def _is_s3_uri(s3_uri: Optional[str]) -> bool:
    """Checks whether an S3 URI is valid.

    Args:
        s3_uri (Optional[str]): The S3 URI.

    Returns:
        bool: Whether the S3 URI is valid.
    """
    if s3_uri is None:
        return False

    return re.match("^s3://([^/]+)/?(.*)$", s3_uri) is not None


def _extract_optimization_config_and_env(
    quantization_config: Optional[Dict] = None, compilation_config: Optional[Dict] = None
) -> Optional[Tuple[Optional[Dict], Optional[Dict]]]:
    """Extracts optimization config and environment variables.

    Args:
        quantization_config (Optional[Dict]): The quantization config.
        compilation_config (Optional[Dict]): The compilation config.

    Returns:
        Optional[Tuple[Optional[Dict], Optional[Dict]]]:
            The optimization config and environment variables.
    """
    if quantization_config:
        return {"ModelQuantizationConfig": quantization_config}, quantization_config.get(
            "OverrideEnvironment"
        )
    if compilation_config:
        return {"ModelCompilationConfig": compilation_config}, compilation_config.get(
            "OverrideEnvironment"
        )
    return None, None


def _custom_speculative_decoding(
    model: Model,
    speculative_decoding_config: Optional[Dict],
    accept_eula: Optional[bool] = False,
) -> Model:
    """Modifies the given model for speculative decoding config with custom provider.

    Args:
        model (Model): The model.
        speculative_decoding_config (Optional[Dict]): The speculative decoding config.
        accept_eula (Optional[bool]): Whether to accept eula or not.
    """

    if speculative_decoding_config:
        additional_model_source = _extracts_and_validates_speculative_model_source(
            speculative_decoding_config
        )

        if _is_s3_uri(additional_model_source):
            channel_name = _generate_channel_name(model.additional_model_data_sources)
            speculative_draft_model = f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}"

            model.additional_model_data_sources = _generate_additional_model_data_sources(
                additional_model_source, channel_name, accept_eula
            )
        else:
            speculative_draft_model = additional_model_source

        model.env.update({"OPTION_SPECULATIVE_DRAFT_MODEL": speculative_draft_model})
        model.add_tags(
            {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "custom"},
        )

    return model
