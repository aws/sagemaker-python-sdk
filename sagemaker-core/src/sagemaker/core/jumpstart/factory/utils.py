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
"""This module stores JumpStart factory utilities."""

from __future__ import absolute_import
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from sagemaker.core.shapes import ModelAccessConfig
from sagemaker.core import (
    environment_variables,
    image_uris,
    instance_types,
    model_uris,
    script_uris,
)
from sagemaker.serve.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.core.deserializers.base import BaseDeserializer
from sagemaker.core.serializers.base import BaseSerializer
from sagemaker.core.explainer.explainer_config import ExplainerConfig
from sagemaker.core.jumpstart.artifacts import (
    _model_supports_inference_script_uri,
    _retrieve_model_init_kwargs,
    _retrieve_model_deploy_kwargs,
    _retrieve_model_package_arn,
)
from sagemaker.core.jumpstart.artifacts.resource_names import _retrieve_resource_name_base
from sagemaker.core.jumpstart.constants import (
    INFERENCE_ENTRY_POINT_SCRIPT_NAME,
    JUMPSTART_DEFAULT_REGION_NAME,
    JUMPSTART_LOGGER,
)

from sagemaker.core.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.core.jumpstart.hub.utils import (
    construct_hub_model_arn_from_inputs,
    construct_hub_model_reference_arn_from_inputs,
)

from sagemaker.core.jumpstart.enums import (
    JumpStartScriptScope,
    JumpStartModelType,
    HubContentCapability,
)
from sagemaker.core.jumpstart.types import (
    HubContentType,
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
    JumpStartModelSpecs,
)
from sagemaker.core.jumpstart.utils import (
    add_hub_content_arn_tags,
    add_jumpstart_model_info_tags,
    add_bedrock_store_tags,
    get_default_jumpstart_session_with_user_agent_suffix,
    get_top_ranked_config_name,
    update_dict_if_key_not_present,
    resolve_model_sagemaker_config_field,
    verify_model_region_and_return_specs,
    get_draft_model_content_bucket,
)

from sagemaker.core.model_monitor.data_capture_config import DataCaptureConfig

from sagemaker.serve.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.common_utils import (
    camel_case_to_pascal_case,
    name_from_base,
    format_tags,
    Tags,
)
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.serve.compute_resource_requirements.resource_requirements import ResourceRequirements
from sagemaker.core import resource_requirements
from sagemaker.core.enums import EndpointType


KwargsType = Union[
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartEstimatorDeployKwargs,
]


def get_model_info_default_kwargs(
    kwargs: KwargsType,
    include_config_name: bool = True,
    include_model_version: bool = True,
    include_tolerate_flags: bool = True,
) -> dict:
    """Returns a dictionary of model info kwargs to use with JumpStart APIs."""

    kwargs_dict = {
        "model_id": kwargs.model_id,
        "hub_arn": kwargs.hub_arn,
        "region": kwargs.region,
        "sagemaker_session": kwargs.sagemaker_session,
        "model_type": kwargs.model_type,
    }
    if include_config_name:
        kwargs_dict.update({"config_name": kwargs.config_name})

    if include_model_version:
        kwargs_dict.update({"model_version": kwargs.model_version})

    if include_tolerate_flags:
        kwargs_dict.update(
            {
                "tolerate_deprecated_model": kwargs.tolerate_deprecated_model,
                "tolerate_vulnerable_model": kwargs.tolerate_vulnerable_model,
            }
        )

    return kwargs_dict


def _set_temp_sagemaker_session_if_not_set(kwargs: KwargsType) -> Tuple[KwargsType, Session]:
    """Sets a temporary sagemaker session if one is not set, and returns original session.

    We need to create a default JS session (without custom user agent)
    in order to retrieve config name info.
    """

    orig_session = kwargs.sagemaker_session
    if kwargs.sagemaker_session is None:
        kwargs.sagemaker_session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
    return kwargs, orig_session


def _add_region_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets region kwargs based on default or override, returns full kwargs."""

    kwargs.region = (
        kwargs.region or kwargs.sagemaker_session.boto_region_name or JUMPSTART_DEFAULT_REGION_NAME
    )

    return kwargs


def _add_sagemaker_session_with_custom_user_agent_to_kwargs(
    kwargs: Union[JumpStartModelInitKwargs, JumpStartModelDeployKwargs],
    orig_session: Optional[Session],
) -> JumpStartModelInitKwargs:
    """Sets session in kwargs based on default or override, returns full kwargs."""

    kwargs.sagemaker_session = orig_session or get_default_jumpstart_session_with_user_agent_suffix(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        config_name=kwargs.config_name,
        is_hub_content=kwargs.hub_arn is not None,
    )

    return kwargs


def _add_role_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets role based on default or override, returns full kwargs."""

    kwargs.role = resolve_model_sagemaker_config_field(
        field_name="role",
        field_val=kwargs.role,
        sagemaker_session=kwargs.sagemaker_session,
        default_value=kwargs.role,
    )

    return kwargs


def _add_model_version_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets model version based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    if kwargs.hub_arn:
        hub_content_version = kwargs.specs.version
        kwargs.model_version = hub_content_version

    return kwargs


def _add_vulnerable_and_deprecated_status_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets deprecated and vulnerability check status, returns full kwargs."""

    kwargs.tolerate_deprecated_model = kwargs.tolerate_deprecated_model or False
    kwargs.tolerate_vulnerable_model = kwargs.tolerate_vulnerable_model or False

    return kwargs


def _add_instance_type_to_kwargs(
    kwargs: JumpStartModelInitKwargs, disable_instance_type_logging: bool = False
) -> JumpStartModelInitKwargs:
    """Sets instance type based on default or override, returns full kwargs."""

    orig_instance_type = kwargs.instance_type
    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        **get_model_info_default_kwargs(kwargs),
        scope=JumpStartScriptScope.INFERENCE,
        training_instance_type=kwargs.training_instance_type,
    )

    if not disable_instance_type_logging and orig_instance_type is None:
        JUMPSTART_LOGGER.info(
            "No instance type selected for inference hosting endpoint. Defaulting to %s.",
            kwargs.instance_type,
        )

    specs = kwargs.specs

    if specs.inference_configs and kwargs.config_name not in specs.inference_configs.configs:
        return kwargs

    resolved_config = (
        specs.inference_configs.configs[kwargs.config_name].resolved_config
        if specs.inference_configs
        else None
    )
    if resolved_config is None:
        return kwargs
    supported_instance_types = resolved_config.get("supported_inference_instance_types", [])
    if kwargs.instance_type not in supported_instance_types:
        JUMPSTART_LOGGER.warning("Overriding instance type to %s", kwargs.instance_type)

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets image uri based on default or override, returns full kwargs.
    Uses placeholder image uri for JumpStart proprietary models that uses ModelPackages
    """

    if kwargs.model_type == JumpStartModelType.PROPRIETARY:
        kwargs.image_uri = None
        return kwargs

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        **get_model_info_default_kwargs(kwargs),
        framework=None,
        image_scope=JumpStartScriptScope.INFERENCE,
        instance_type=kwargs.instance_type,
    )

    return kwargs


def _add_model_reference_arn_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets Model Reference ARN if the hub content type is Model Reference, returns full kwargs."""

    hub_content_type = kwargs.specs.hub_content_type
    kwargs.hub_content_type = hub_content_type if kwargs.hub_arn else None

    if hub_content_type == HubContentType.MODEL_REFERENCE:
        kwargs.model_reference_arn = construct_hub_model_reference_arn_from_inputs(
            hub_arn=kwargs.hub_arn, model_name=kwargs.model_id, version=kwargs.model_version
        )
    else:
        kwargs.model_reference_arn = None
    return kwargs


def _add_model_data_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets model data based on default or override, returns full kwargs."""

    if kwargs.model_type == JumpStartModelType.PROPRIETARY:
        kwargs.model_data = None
        return kwargs

    model_info_kwargs = get_model_info_default_kwargs(kwargs)
    model_data: Union[str, dict] = kwargs.model_data or model_uris.retrieve(
        **model_info_kwargs,
        model_scope=JumpStartScriptScope.INFERENCE,
        instance_type=kwargs.instance_type,
    )

    if isinstance(model_data, str) and model_data.startswith("s3://") and model_data.endswith("/"):
        old_model_data_str = model_data
        model_data = {
            "S3DataSource": {
                "S3Uri": model_data,
                "S3DataType": "S3Prefix",
                "CompressionType": "None",
            }
        }
        if kwargs.model_data:
            JUMPSTART_LOGGER.info(
                "S3 prefix model_data detected for JumpStartModel: '%s'. "
                "Converting to S3DataSource dictionary: '%s'.",
                old_model_data_str,
                json.dumps(model_data),
            )

    kwargs.model_data = model_data

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets source dir based on default or override, returns full kwargs."""

    if kwargs.model_type == JumpStartModelType.PROPRIETARY:
        kwargs.source_dir = None
        return kwargs

    source_dir = kwargs.source_dir

    if _model_supports_inference_script_uri(**get_model_info_default_kwargs(kwargs)):
        source_dir = source_dir or script_uris.retrieve(
            **get_model_info_default_kwargs(kwargs), script_scope=JumpStartScriptScope.INFERENCE
        )

    kwargs.source_dir = source_dir

    return kwargs


def _add_entry_point_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets entry point based on default or override, returns full kwargs."""

    if kwargs.model_type == JumpStartModelType.PROPRIETARY:
        kwargs.entry_point = None
        return kwargs

    entry_point = kwargs.entry_point

    if _model_supports_inference_script_uri(**get_model_info_default_kwargs(kwargs)):

        entry_point = entry_point or INFERENCE_ENTRY_POINT_SCRIPT_NAME

    kwargs.entry_point = entry_point

    return kwargs


def _add_env_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets env based on default or override, returns full kwargs."""

    if kwargs.model_type == JumpStartModelType.PROPRIETARY:
        kwargs.env = None
        return kwargs

    env = kwargs.env

    if env is None:
        env = {}

    extra_env_vars = environment_variables.retrieve_default(
        **get_model_info_default_kwargs(kwargs),
        include_aws_sdk_env_vars=False,
        script=JumpStartScriptScope.INFERENCE,
        instance_type=kwargs.instance_type,
    )

    for key, value in extra_env_vars.items():
        update_dict_if_key_not_present(
            env,
            key,
            value,
        )

    if env == {}:
        env = None

    kwargs.env = env

    return kwargs


def _add_model_package_arn_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets model package arn based on default or override, returns full kwargs."""

    model_package_arn = kwargs.model_package_arn or _retrieve_model_package_arn(
        **get_model_info_default_kwargs(kwargs),
        instance_type=kwargs.instance_type,
        scope=JumpStartScriptScope.INFERENCE,
    )

    kwargs.model_package_arn = model_package_arn
    return kwargs


def _add_extra_model_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    model_kwargs_to_add = _retrieve_model_init_kwargs(**get_model_info_default_kwargs(kwargs))

    for key, value in model_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            resolved_value = resolve_model_sagemaker_config_field(
                field_name=key,
                field_val=value,
                sagemaker_session=kwargs.sagemaker_session,
            )
            setattr(kwargs, key, resolved_value)

    return kwargs


def _add_endpoint_name_to_kwargs(
    kwargs: Optional[JumpStartModelDeployKwargs],
) -> JumpStartModelDeployKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_endpoint_name = _retrieve_resource_name_base(**get_model_info_default_kwargs(kwargs))

    kwargs.endpoint_name = kwargs.endpoint_name or (
        name_from_base(default_endpoint_name) if default_endpoint_name is not None else None
    )

    return kwargs


def _add_model_name_to_kwargs(
    kwargs: Optional[JumpStartModelInitKwargs],
) -> JumpStartModelInitKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_model_name = _retrieve_resource_name_base(**get_model_info_default_kwargs(kwargs))

    kwargs.name = kwargs.name or (
        name_from_base(default_model_name) if default_model_name is not None else None
    )

    return kwargs


def _add_tags_to_kwargs(kwargs: JumpStartModelDeployKwargs) -> Dict[str, Any]:
    """Sets tags based on default or override, returns full kwargs."""

    full_model_version = kwargs.specs.version

    if kwargs.sagemaker_session.settings.include_jumpstart_tags:
        kwargs.tags = add_jumpstart_model_info_tags(
            kwargs.tags,
            kwargs.model_id,
            full_model_version,
            kwargs.model_type,
            config_name=kwargs.config_name,
            scope=JumpStartScriptScope.INFERENCE,
        )

    if kwargs.hub_arn:
        if kwargs.model_reference_arn:
            hub_content_arn = construct_hub_model_reference_arn_from_inputs(
                kwargs.hub_arn, kwargs.model_id, kwargs.model_version
            )
        else:
            hub_content_arn = construct_hub_model_arn_from_inputs(
                kwargs.hub_arn, kwargs.model_id, kwargs.model_version
            )
        kwargs.tags = add_hub_content_arn_tags(kwargs.tags, hub_content_arn=hub_content_arn)

    if hasattr(kwargs.specs, "capabilities") and kwargs.specs.capabilities is not None:
        if HubContentCapability.BEDROCK_CONSOLE in kwargs.specs.capabilities:
            kwargs.tags = add_bedrock_store_tags(kwargs.tags, compatibility="compatible")

    return kwargs


def _add_deploy_extra_kwargs(kwargs: JumpStartModelInitKwargs) -> Dict[str, Any]:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    deploy_kwargs_to_add = _retrieve_model_deploy_kwargs(
        **get_model_info_default_kwargs(kwargs), instance_type=kwargs.instance_type
    )

    for key, value in deploy_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            setattr(kwargs, key, value)

    return kwargs


def _add_resources_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets the resource requirements based on the default or an override. Returns full kwargs."""

    kwargs.resources = kwargs.resources or resource_requirements.retrieve_default(
        **get_model_info_default_kwargs(kwargs),
        scope=JumpStartScriptScope.INFERENCE,
        instance_type=kwargs.instance_type,
    )

    return kwargs


def _select_inference_config_from_training_config(
    specs: JumpStartModelSpecs, training_config_name: str
) -> Optional[str]:
    """Selects the inference config from the training config.
    Args:
        specs (JumpStartModelSpecs): The specs for the model.
        training_config_name (str): The name of the training config.
    Returns:
        str: The name of the inference config.
    """
    if specs.training_configs:
        resolved_training_config = specs.training_configs.configs.get(training_config_name)
        if resolved_training_config:
            return resolved_training_config.default_inference_config

    return None


def _add_config_name_to_init_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets default config name to the kwargs. Returns full kwargs.
    Raises:
        ValueError: If the instance_type is not supported with the current config.
    """

    kwargs.config_name = kwargs.config_name or get_top_ranked_config_name(
        **get_model_info_default_kwargs(kwargs, include_config_name=False),
        scope=JumpStartScriptScope.INFERENCE,
    )

    if kwargs.config_name is None:
        return kwargs

    return kwargs


def _add_additional_model_data_sources_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets default additional model data sources to init kwargs"""

    specs = kwargs.specs
    # Append speculative decoding data source from metadata
    speculative_decoding_data_sources = specs.get_speculative_decoding_s3_data_sources()
    for data_source in speculative_decoding_data_sources:
        data_source.s3_data_source.set_bucket(
            get_draft_model_content_bucket(provider=data_source.provider, region=kwargs.region)
        )
    api_shape_additional_model_data_sources = (
        [
            camel_case_to_pascal_case(data_source.to_json())
            for data_source in speculative_decoding_data_sources
        ]
        if specs.get_speculative_decoding_s3_data_sources()
        else None
    )

    kwargs.additional_model_data_sources = (
        kwargs.additional_model_data_sources or api_shape_additional_model_data_sources
    )

    return kwargs


def _add_config_name_to_deploy_kwargs(
    kwargs: JumpStartModelDeployKwargs, training_config_name: Optional[str] = None
) -> JumpStartModelInitKwargs:
    """Sets default config name to the kwargs. Returns full kwargs.
    If a training_config_name is passed, then choose the inference config
    based on the supported inference configs in that training config.
    Raises:
        ValueError: If the instance_type is not supported with the current config.
    """

    if training_config_name:

        specs = kwargs.specs
        default_config_name = _select_inference_config_from_training_config(
            specs=specs, training_config_name=training_config_name
        )

    else:
        default_config_name = kwargs.config_name or get_top_ranked_config_name(
            **get_model_info_default_kwargs(kwargs, include_config_name=False),
            scope=JumpStartScriptScope.INFERENCE,
        )

    kwargs.config_name = kwargs.config_name or default_config_name

    return kwargs


def get_deploy_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    region: Optional[str] = None,
    initial_instance_count: Optional[int] = None,
    instance_type: Optional[str] = None,
    serializer: Optional[BaseSerializer] = None,
    deserializer: Optional[BaseDeserializer] = None,
    accelerator_type: Optional[str] = None,
    endpoint_name: Optional[str] = None,
    inference_component_name: Optional[str] = None,
    tags: Optional[Tags] = None,
    kms_key: Optional[str] = None,
    wait: Optional[bool] = None,
    data_capture_config: Optional[DataCaptureConfig] = None,
    async_inference_config: Optional[AsyncInferenceConfig] = None,
    serverless_inference_config: Optional[ServerlessInferenceConfig] = None,
    volume_size: Optional[int] = None,
    model_data_download_timeout: Optional[int] = None,
    container_startup_health_check_timeout: Optional[int] = None,
    inference_recommendation_id: Optional[str] = None,
    explainer_config: Optional[ExplainerConfig] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    sagemaker_session: Optional[Session] = None,
    accept_eula: Optional[bool] = None,
    model_reference_arn: Optional[str] = None,
    endpoint_logging: Optional[bool] = None,
    resources: Optional[ResourceRequirements] = None,
    managed_instance_scaling: Optional[str] = None,
    endpoint_type: Optional[EndpointType] = None,
    training_config_name: Optional[str] = None,
    config_name: Optional[str] = None,
    routing_config: Optional[Dict[str, Any]] = None,
    model_access_configs: Optional[Dict[str, ModelAccessConfig]] = None,
    inference_ami_version: Optional[str] = None,
) -> JumpStartModelDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Model` object."""

    deploy_kwargs: JumpStartModelDeployKwargs = JumpStartModelDeployKwargs(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        model_type=model_type,
        region=region,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        serializer=serializer,
        deserializer=deserializer,
        accelerator_type=accelerator_type,
        endpoint_name=endpoint_name,
        inference_component_name=inference_component_name,
        tags=format_tags(tags),
        kms_key=kms_key,
        wait=wait,
        data_capture_config=data_capture_config,
        async_inference_config=async_inference_config,
        serverless_inference_config=serverless_inference_config,
        volume_size=volume_size,
        model_data_download_timeout=model_data_download_timeout,
        container_startup_health_check_timeout=container_startup_health_check_timeout,
        inference_recommendation_id=inference_recommendation_id,
        explainer_config=explainer_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
        accept_eula=accept_eula,
        model_reference_arn=model_reference_arn,
        endpoint_logging=endpoint_logging,
        resources=resources,
        config_name=config_name,
        routing_config=routing_config,
        model_access_configs=model_access_configs,
        inference_ami_version=inference_ami_version,
    )
    deploy_kwargs, orig_session = _set_temp_sagemaker_session_if_not_set(kwargs=deploy_kwargs)
    deploy_kwargs.specs = verify_model_region_and_return_specs(
        **get_model_info_default_kwargs(
            deploy_kwargs, include_model_version=False, include_tolerate_flags=False
        ),
        version=deploy_kwargs.model_version or "*",
        scope=JumpStartScriptScope.INFERENCE,
        # We set these flags to True to retrieve the json specs.
        # Exceptions will be thrown later if these are not tolerated.
        tolerate_deprecated_model=True,
        tolerate_vulnerable_model=True,
    )

    deploy_kwargs = _add_config_name_to_deploy_kwargs(
        kwargs=deploy_kwargs, training_config_name=training_config_name
    )

    deploy_kwargs = _add_model_version_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_sagemaker_session_with_custom_user_agent_to_kwargs(
        kwargs=deploy_kwargs, orig_session=orig_session
    )

    deploy_kwargs = _add_endpoint_name_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_instance_type_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs.initial_instance_count = initial_instance_count or 1

    deploy_kwargs = _add_deploy_extra_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_tags_to_kwargs(kwargs=deploy_kwargs)

    if endpoint_type == EndpointType.INFERENCE_COMPONENT_BASED:
        deploy_kwargs = _add_resources_to_kwargs(kwargs=deploy_kwargs)
        deploy_kwargs.endpoint_type = endpoint_type
        deploy_kwargs.managed_instance_scaling = managed_instance_scaling

    return deploy_kwargs


def get_init_kwargs(
    model_id: str,
    model_from_estimator: bool = False,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    model_data: Optional[Union[str, PipelineVariable, dict]] = None,
    role: Optional[str] = None,
    env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    name: Optional[str] = None,
    vpc_config: Optional[Dict[str, List[Union[str, PipelineVariable]]]] = None,
    sagemaker_session: Optional[Session] = None,
    enable_network_isolation: Union[bool, PipelineVariable] = None,
    model_kms_key: Optional[str] = None,
    image_config: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    source_dir: Optional[str] = None,
    code_location: Optional[str] = None,
    entry_point: Optional[str] = None,
    container_log_level: Optional[Union[int, PipelineVariable]] = None,
    dependencies: Optional[List[str]] = None,
    git_config: Optional[Dict[str, str]] = None,
    model_package_arn: Optional[str] = None,
    training_instance_type: Optional[str] = None,
    disable_instance_type_logging: bool = False,
    resources: Optional[ResourceRequirements] = None,
    config_name: Optional[str] = None,
    additional_model_data_sources: Optional[Dict[str, Any]] = None,
) -> JumpStartModelInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Model` object."""

    model_init_kwargs: JumpStartModelInitKwargs = JumpStartModelInitKwargs(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        model_type=model_type,
        instance_type=instance_type,
        region=region,
        image_uri=image_uri,
        model_data=model_data,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        role=role,
        name=name,
        vpc_config=vpc_config,
        sagemaker_session=sagemaker_session,
        enable_network_isolation=enable_network_isolation,
        model_kms_key=model_kms_key,
        image_config=image_config,
        code_location=code_location,
        container_log_level=container_log_level,
        dependencies=dependencies,
        git_config=git_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        model_package_arn=model_package_arn,
        training_instance_type=training_instance_type,
        resources=resources,
        config_name=config_name,
        additional_model_data_sources=additional_model_data_sources,
    )
    model_init_kwargs, orig_session = _set_temp_sagemaker_session_if_not_set(
        kwargs=model_init_kwargs
    )
    model_init_kwargs.specs = verify_model_region_and_return_specs(
        **get_model_info_default_kwargs(
            model_init_kwargs, include_model_version=False, include_tolerate_flags=False
        ),
        version=model_init_kwargs.model_version or "*",
        scope=JumpStartScriptScope.INFERENCE,
        # We set these flags to True to retrieve the json specs.
        # Exceptions will be thrown later if these are not tolerated.
        tolerate_deprecated_model=True,
        tolerate_vulnerable_model=True,
    )

    model_init_kwargs = _add_vulnerable_and_deprecated_status_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_model_version_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_config_name_to_init_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_sagemaker_session_with_custom_user_agent_to_kwargs(
        kwargs=model_init_kwargs, orig_session=orig_session
    )
    model_init_kwargs = _add_region_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_model_name_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_instance_type_to_kwargs(
        kwargs=model_init_kwargs, disable_instance_type_logging=disable_instance_type_logging
    )

    model_init_kwargs = _add_image_uri_to_kwargs(kwargs=model_init_kwargs)

    if hub_arn:
        model_init_kwargs = _add_model_reference_arn_to_kwargs(kwargs=model_init_kwargs)
    else:
        model_init_kwargs.model_reference_arn = None
        model_init_kwargs.hub_content_type = None

    # we use the model artifact from the training job output
    if not model_from_estimator:
        model_init_kwargs = _add_model_data_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_source_dir_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_entry_point_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_env_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_extra_model_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_role_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_model_package_arn_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_resources_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_additional_model_data_sources_to_kwargs(kwargs=model_init_kwargs)

    return model_init_kwargs
