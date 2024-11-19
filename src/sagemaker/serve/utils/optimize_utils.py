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

from sagemaker import Model, Session
from sagemaker.enums import Tag
from sagemaker.jumpstart.utils import accessors, get_eula_message


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


def _deployment_config_contains_draft_model(deployment_config: Optional[Dict]) -> bool:
    """Checks whether a deployment config contains a speculative decoding draft model.

    Args:
        deployment_config (Dict): The deployment config to check.

    Returns:
        bool: Whether the deployment config contains a draft model or not.
    """
    if deployment_config is None:
        return False
    deployment_args = deployment_config.get("DeploymentArgs", {})
    additional_data_sources = deployment_args.get("AdditionalDataSources")

    return "speculative_decoding" in additional_data_sources if additional_data_sources else False


def _is_draft_model_jumpstart_provided(deployment_config: Optional[Dict]) -> bool:
    """Checks whether a deployment config's draft model is provided by JumpStart.

    Args:
        deployment_config (Dict): The deployment config to check.

    Returns:
        bool: Whether the draft model is provided by JumpStart or not.
    """
    if deployment_config is None:
        return False

    additional_model_data_sources = deployment_config.get("DeploymentArgs", {}).get(
        "AdditionalDataSources"
    )
    for source in additional_model_data_sources.get("speculative_decoding", []):
        if source["channel_name"] == "draft_model":
            if source.get("provider", {}).get("name") == "JumpStart":
                return True
            continue
    return False


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

    model_provider = speculative_decoding_config.get("ModelProvider", "").lower()

    if model_provider == "jumpstart":
        return "jumpstart"

    if model_provider == "custom" or speculative_decoding_config.get("ModelSource"):
        return "custom"

    if model_provider == "sagemaker":
        return "sagemaker"

    return "auto"


def _extract_additional_model_data_source_s3_uri(
    additional_model_data_source: Optional[Dict] = None,
) -> Optional[str]:
    """Extracts model data source s3 uri from a model data source in Pascal case.

    Args:
        additional_model_data_source (Optional[Dict]): A model data source.

    Returns:
        str: S3 uri of the model resources.
    """
    if (
        additional_model_data_source is None
        or additional_model_data_source.get("S3DataSource", None) is None
    ):
        return None

    return additional_model_data_source.get("S3DataSource").get("S3Uri")


def _extract_deployment_config_additional_model_data_source_s3_uri(
    additional_model_data_source: Optional[Dict] = None,
) -> Optional[str]:
    """Extracts model data source s3 uri from a model data source in snake case.

    Args:
        additional_model_data_source (Optional[Dict]): A model data source.

    Returns:
        str: S3 uri of the model resources.
    """
    if (
        additional_model_data_source is None
        or additional_model_data_source.get("s3_data_source", None) is None
    ):
        return None

    return additional_model_data_source.get("s3_data_source").get("s3_uri", None)


def _is_draft_model_gated(
    draft_model_config: Optional[Dict] = None,
) -> bool:
    """Extracts model gated-ness from draft model data source.

    Args:
        draft_model_config (Optional[Dict]): A model data source.

    Returns:
        bool: Whether the draft model is gated or not.
    """
    return "hosting_eula_key" in draft_model_config if draft_model_config else False


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
        additional_model_data_source["S3DataSource"]["ModelAccessConfig"] = {"AcceptEula": True}

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
    quantization_config: Optional[Dict] = None,
    compilation_config: Optional[Dict] = None,
    sharding_config: Optional[Dict] = None,
) -> Optional[Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]]:
    """Extracts optimization config and environment variables.

    Args:
        quantization_config (Optional[Dict]): The quantization config.
        compilation_config (Optional[Dict]): The compilation config.
        sharding_config (Optional[Dict]): The sharding config.

    Returns:
        Optional[Tuple[Optional[Dict], Optional[Dict], Optional[Dict], Optional[Dict]]]:
            The optimization config and environment variables.
    """
    optimization_config = {}
    quantization_override_env = (
        quantization_config.get("OverrideEnvironment") if quantization_config else None
    )
    compilation_override_env = (
        compilation_config.get("OverrideEnvironment") if compilation_config else None
    )
    sharding_override_env = sharding_config.get("OverrideEnvironment") if sharding_config else None

    if quantization_config is not None:
        optimization_config["ModelQuantizationConfig"] = quantization_config

    if compilation_config is not None:
        optimization_config["ModelCompilationConfig"] = compilation_config

    if sharding_config is not None:
        optimization_config["ModelShardingConfig"] = sharding_config

    # Return optimization config dict and environment variables if either is present
    if optimization_config:
        return (
            optimization_config,
            quantization_override_env,
            compilation_override_env,
            sharding_override_env,
        )

    return None, None, None, None


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

        accept_eula = speculative_decoding_config.get("AcceptEula", accept_eula)

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


def _jumpstart_speculative_decoding(
    model=Model,
    speculative_decoding_config: Optional[Dict[str, Any]] = None,
    sagemaker_session: Optional[Session] = None,
):
    """Modifies the given model for speculative decoding config with JumpStart provider.

    Args:
        model (Model): The model.
        speculative_decoding_config (Optional[Dict]): The speculative decoding config.
        sagemaker_session (Optional[Session]): Sagemaker session for execution.
    """
    if speculative_decoding_config:
        js_id = speculative_decoding_config.get("ModelID")
        if not js_id:
            raise ValueError(
                "`ModelID` is a required field in `speculative_decoding_config` when "
                "using JumpStart as draft model provider."
            )
        model_version = speculative_decoding_config.get("ModelVersion", "*")
        accept_eula = speculative_decoding_config.get("AcceptEula", False)
        channel_name = _generate_channel_name(model.additional_model_data_sources)

        model_specs = accessors.JumpStartModelsAccessor.get_model_specs(
            model_id=js_id,
            version=model_version,
            region=sagemaker_session.boto_region_name,
            sagemaker_session=sagemaker_session,
        )
        model_spec_json = model_specs.to_json()

        js_bucket = accessors.JumpStartModelsAccessor.get_jumpstart_content_bucket()

        if model_spec_json.get("gated_bucket", False):
            if not accept_eula:
                eula_message = get_eula_message(
                    model_specs=model_specs, region=sagemaker_session.boto_region_name
                )
                raise ValueError(
                    f"{eula_message} Set `AcceptEula`=True in "
                    f"speculative_decoding_config once acknowledged."
                )
            js_bucket = accessors.JumpStartModelsAccessor.get_jumpstart_gated_content_bucket()

        key_prefix = model_spec_json.get("hosting_prepacked_artifact_key")
        model.additional_model_data_sources = _generate_additional_model_data_sources(
            f"s3://{js_bucket}/{key_prefix}",
            channel_name,
            accept_eula,
        )

        model.env.update(
            {"OPTION_SPECULATIVE_DRAFT_MODEL": f"{SPECULATIVE_DRAFT_MODEL}/{channel_name}/"}
        )
        model.add_tags(
            {"Key": Tag.SPECULATIVE_DRAFT_MODEL_PROVIDER, "Value": "jumpstart"},
        )
