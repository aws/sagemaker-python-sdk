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
# pylint: skip-file
"""This module stores Hub converter utilities for JumpStart."""
from __future__ import absolute_import

from typing import Any, Dict, List
from sagemaker.jumpstart.enums import ModelSpecKwargType, NamingConventionType
from sagemaker.s3 import parse_s3_url
from sagemaker.jumpstart.types import (
    JumpStartModelSpecs,
    HubContentType,
    JumpStartDataHolderType,
)
from sagemaker.jumpstart.hub.interfaces import (
    DescribeHubContentResponse,
    HubModelDocument,
)
from sagemaker.jumpstart.hub.parser_utils import (
    camel_to_snake,
    snake_to_upper_camel,
    walk_and_apply_json,
)


def _to_json(dictionary: Dict[Any, Any]) -> Dict[Any, Any]:
    """Convert a nested dictionary of JumpStartDataHolderType into json with UpperCamelCase keys"""
    for key, value in dictionary.items():
        if issubclass(type(value), JumpStartDataHolderType):
            dictionary[key] = walk_and_apply_json(value.to_json(), snake_to_upper_camel)
        elif isinstance(value, list):
            new_value = []
            for value_in_list in value:
                new_value_in_list = value_in_list
                if issubclass(type(value_in_list), JumpStartDataHolderType):
                    new_value_in_list = walk_and_apply_json(
                        value_in_list.to_json(), snake_to_upper_camel
                    )
                new_value.append(new_value_in_list)
            dictionary[key] = new_value
        elif isinstance(value, dict):
            for key_in_dict, value_in_dict in value.items():
                if issubclass(type(value_in_dict), JumpStartDataHolderType):
                    value[key_in_dict] = walk_and_apply_json(
                        value_in_dict.to_json(), snake_to_upper_camel
                    )
    return dictionary


def get_model_spec_arg_keys(
    arg_type: ModelSpecKwargType,
    naming_convention: NamingConventionType = NamingConventionType.DEFAULT,
) -> List[str]:
    """Returns a list of arg keys for a specific model spec arg type.

    Args:
        arg_type (ModelSpecKwargType): Type of the model spec's kwarg.
        naming_convention (NamingConventionType): Type of naming convention to return.

    Raises:
        ValueError: If the naming convention is not valid.
    """
    arg_keys: List[str] = []
    if arg_type == ModelSpecKwargType.DEPLOY:
        arg_keys = ["ModelDataDownloadTimeout", "ContainerStartupHealthCheckTimeout"]
    elif arg_type == ModelSpecKwargType.ESTIMATOR:
        arg_keys = [
            "EncryptInterContainerTraffic",
            "MaxRuntimeInSeconds",
            "DisableOutputCompression",
            "ModelDir",
        ]
    elif arg_type == ModelSpecKwargType.MODEL:
        arg_keys = []
    elif arg_type == ModelSpecKwargType.FIT:
        arg_keys = []

    if naming_convention == NamingConventionType.SNAKE_CASE:
        arg_keys = [camel_to_snake(key) for key in arg_keys]
    elif naming_convention == NamingConventionType.UPPER_CAMEL_CASE:
        return arg_keys
    else:
        raise ValueError("Please provide a valid naming convention.")
    return arg_keys


def get_model_spec_kwargs_from_hub_model_document(
    arg_type: ModelSpecKwargType,
    hub_content_document: Dict[str, Any],
    naming_convention: NamingConventionType = NamingConventionType.UPPER_CAMEL_CASE,
) -> Dict[str, Any]:
    """Returns a map of arg type to arg keys for a given hub content document.

    Args:
        arg_type (ModelSpecKwargType): Type of the model spec's kwarg.
        hub_content_document: A dictionary representation of hub content document.
        naming_convention (NamingConventionType): Type of naming convention to return.

    """
    kwargs = dict()
    keys = get_model_spec_arg_keys(arg_type, naming_convention=naming_convention)
    for k in keys:
        kwarg_value = hub_content_document.get(k)
        if kwarg_value is not None:
            kwargs[k] = kwarg_value
    return kwargs


def make_model_specs_from_describe_hub_content_response(
    response: DescribeHubContentResponse,
) -> JumpStartModelSpecs:
    """Sets fields in JumpStartModelSpecs based on values in DescribeHubContentResponse

    Args:
        response (Dict[str, any]): parsed DescribeHubContentResponse returned
            from SageMaker:DescribeHubContent
    """
    if response.hub_content_type not in {HubContentType.MODEL, HubContentType.MODEL_REFERENCE}:
        raise AttributeError(
            "Invalid content type, use either HubContentType.MODEL or HubContentType.MODEL_REFERENCE."
        )
    region = response.get_hub_region()
    specs = {}
    model_id = response.hub_content_name
    specs["model_id"] = model_id
    specs["version"] = response.hub_content_version
    hub_model_document: HubModelDocument = response.hub_content_document
    specs["url"] = hub_model_document.url
    specs["min_sdk_version"] = hub_model_document.min_sdk_version
    specs["training_supported"] = bool(hub_model_document.training_supported)
    specs["incremental_training_supported"] = bool(
        hub_model_document.incremental_training_supported
    )
    specs["hosting_ecr_uri"] = hub_model_document.hosting_ecr_uri
    specs["inference_configs"] = hub_model_document.inference_configs
    specs["inference_config_components"] = hub_model_document.inference_config_components
    specs["inference_config_rankings"] = hub_model_document.inference_config_rankings

    hosting_artifact_bucket, hosting_artifact_key = parse_s3_url(  # pylint: disable=unused-variable
        hub_model_document.hosting_artifact_uri
    )
    specs["hosting_artifact_key"] = hosting_artifact_key
    specs["hosting_artifact_uri"] = hub_model_document.hosting_artifact_uri
    hosting_script_bucket, hosting_script_key = parse_s3_url(  # pylint: disable=unused-variable
        hub_model_document.hosting_script_uri
    )
    specs["hosting_script_key"] = hosting_script_key
    specs["inference_environment_variables"] = hub_model_document.inference_environment_variables
    specs["inference_vulnerable"] = False
    specs["inference_dependencies"] = hub_model_document.inference_dependencies
    specs["inference_vulnerabilities"] = []
    specs["training_vulnerable"] = False
    specs["training_vulnerabilities"] = []
    specs["deprecated"] = False
    specs["deprecated_message"] = None
    specs["deprecate_warn_message"] = None
    specs["usage_info_message"] = None
    specs["default_inference_instance_type"] = hub_model_document.default_inference_instance_type
    specs["supported_inference_instance_types"] = (
        hub_model_document.supported_inference_instance_types
    )
    specs["dynamic_container_deployment_supported"] = (
        hub_model_document.dynamic_container_deployment_supported
    )
    specs["hosting_resource_requirements"] = hub_model_document.hosting_resource_requirements

    specs["hosting_prepacked_artifact_key"] = None
    if hub_model_document.hosting_prepacked_artifact_uri is not None:
        (
            hosting_prepacked_artifact_bucket,  # pylint: disable=unused-variable
            hosting_prepacked_artifact_key,
        ) = parse_s3_url(hub_model_document.hosting_prepacked_artifact_uri)
        specs["hosting_prepacked_artifact_key"] = hosting_prepacked_artifact_key

    hub_content_document_dict: Dict[str, Any] = hub_model_document.to_json()

    specs["fit_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.FIT, hub_content_document_dict
    )
    specs["model_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.MODEL, hub_content_document_dict
    )
    specs["deploy_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.DEPLOY, hub_content_document_dict
    )
    specs["estimator_kwargs"] = get_model_spec_kwargs_from_hub_model_document(
        ModelSpecKwargType.ESTIMATOR, hub_content_document_dict
    )

    specs["predictor_specs"] = hub_model_document.sage_maker_sdk_predictor_specifications
    default_payloads: Dict[str, Any] = {}
    if hub_model_document.default_payloads is not None:
        for alias, payload in hub_model_document.default_payloads.items():
            default_payloads[alias] = walk_and_apply_json(payload.to_json(), camel_to_snake)
        specs["default_payloads"] = default_payloads
    specs["gated_bucket"] = hub_model_document.gated_bucket
    specs["inference_volume_size"] = hub_model_document.inference_volume_size
    specs["inference_enable_network_isolation"] = (
        hub_model_document.inference_enable_network_isolation
    )
    specs["resource_name_base"] = hub_model_document.resource_name_base

    specs["hosting_eula_key"] = None
    if hub_model_document.hosting_eula_uri is not None:
        hosting_eula_bucket, hosting_eula_key = parse_s3_url(  # pylint: disable=unused-variable
            hub_model_document.hosting_eula_uri
        )
        specs["hosting_eula_key"] = hosting_eula_key

    if hub_model_document.hosting_model_package_arn:
        specs["hosting_model_package_arns"] = {region: hub_model_document.hosting_model_package_arn}

    specs["hosting_use_script_uri"] = hub_model_document.hosting_use_script_uri

    specs["hosting_instance_type_variants"] = hub_model_document.hosting_instance_type_variants

    if specs["training_supported"]:
        specs["training_ecr_uri"] = hub_model_document.training_ecr_uri
        (
            training_artifact_bucket,  # pylint: disable=unused-variable
            training_artifact_key,
        ) = parse_s3_url(hub_model_document.training_artifact_uri)
        specs["training_artifact_key"] = training_artifact_key
        (
            training_script_bucket,  # pylint: disable=unused-variable
            training_script_key,
        ) = parse_s3_url(hub_model_document.training_script_uri)
        specs["training_script_key"] = training_script_key

        specs["training_configs"] = hub_model_document.training_configs
        specs["training_config_components"] = hub_model_document.training_config_components
        specs["training_config_rankings"] = hub_model_document.training_config_rankings

        specs["training_dependencies"] = hub_model_document.training_dependencies
        specs["default_training_instance_type"] = hub_model_document.default_training_instance_type
        specs["supported_training_instance_types"] = (
            hub_model_document.supported_training_instance_types
        )
        specs["metrics"] = hub_model_document.training_metrics
        specs["training_prepacked_script_key"] = None
        if hub_model_document.training_prepacked_script_uri is not None:
            (
                training_prepacked_script_bucket,  # pylint: disable=unused-variable
                training_prepacked_script_key,
            ) = parse_s3_url(hub_model_document.training_prepacked_script_uri)
            specs["training_prepacked_script_key"] = training_prepacked_script_key

        specs["hyperparameters"] = hub_model_document.hyperparameters
        specs["training_volume_size"] = hub_model_document.training_volume_size
        specs["training_enable_network_isolation"] = (
            hub_model_document.training_enable_network_isolation
        )
        if hub_model_document.training_model_package_artifact_uri:
            specs["training_model_package_artifact_uris"] = {
                region: hub_model_document.training_model_package_artifact_uri
            }
        specs["training_instance_type_variants"] = (
            hub_model_document.training_instance_type_variants
        )
    return JumpStartModelSpecs(_to_json(specs), is_hub_content=True)
