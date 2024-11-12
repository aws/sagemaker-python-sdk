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
"""This module stores JumpStart Estimator factory methods."""
from __future__ import absolute_import


from typing import Dict, List, Optional, Union
from sagemaker import (
    environment_variables,
    hyperparameters as hyperparameters_utils,
    image_uris,
    instance_types,
    metric_definitions as metric_definitions_utils,
    model_uris,
    script_uris,
)
from sagemaker.jumpstart.artifacts import (
    _model_supports_incremental_training,
    _retrieve_model_package_model_artifact_s3_uri,
)
from sagemaker.jumpstart.artifacts.resource_names import _retrieve_resource_name_base
from sagemaker.jumpstart.factory.utils import (
    _set_temp_sagemaker_session_if_not_set,
    get_model_info_default_kwargs,
)
from sagemaker.jumpstart.hub.utils import (
    construct_hub_model_arn_from_inputs,
    construct_hub_model_reference_arn_from_inputs,
)
from sagemaker.session import Session
from sagemaker.async_inference.async_inference_config import AsyncInferenceConfig
from sagemaker.base_deserializers import BaseDeserializer
from sagemaker.base_serializers import BaseSerializer
from sagemaker.debugger.debugger import DebuggerHookConfig, RuleBase, TensorBoardOutputConfig
from sagemaker.debugger.profiler_config import ProfilerConfig
from sagemaker.explainer.explainer_config import ExplainerConfig
from sagemaker.inputs import FileSystemInput, TrainingInput
from sagemaker.instance_group import InstanceGroup
from sagemaker.jumpstart.artifacts import (
    _retrieve_estimator_init_kwargs,
    _retrieve_estimator_fit_kwargs,
    _model_supports_training_model_uri,
)
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
    JUMPSTART_LOGGER,
    TRAINING_ENTRY_POINT_SCRIPT_NAME,
    SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope, JumpStartModelType
from sagemaker.jumpstart.factory import model
from sagemaker.jumpstart.types import (
    HubContentType,
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartKwargs,
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
)
from sagemaker.jumpstart.utils import (
    add_hub_content_arn_tags,
    add_jumpstart_model_info_tags,
    get_eula_message,
    get_default_jumpstart_session_with_user_agent_suffix,
    get_top_ranked_config_name,
    update_dict_if_key_not_present,
    resolve_estimator_sagemaker_config_field,
    verify_model_region_and_return_specs,
)


from sagemaker.model_monitor.data_capture_config import DataCaptureConfig
from sagemaker.serverless.serverless_inference_config import ServerlessInferenceConfig
from sagemaker.utils import name_from_base, format_tags, Tags
from sagemaker.workflow.entities import PipelineVariable


def get_init_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    region: Optional[str] = None,
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    role: Optional[str] = None,
    instance_count: Optional[Union[int, PipelineVariable]] = None,
    instance_type: Optional[Union[str, PipelineVariable]] = None,
    keep_alive_period_in_seconds: Optional[Union[int, PipelineVariable]] = None,
    volume_size: Optional[Union[int, PipelineVariable]] = None,
    volume_kms_key: Optional[Union[str, PipelineVariable]] = None,
    max_run: Optional[Union[int, PipelineVariable]] = None,
    input_mode: Optional[Union[str, PipelineVariable]] = None,
    output_path: Optional[Union[str, PipelineVariable]] = None,
    output_kms_key: Optional[Union[str, PipelineVariable]] = None,
    base_job_name: Optional[str] = None,
    sagemaker_session: Optional[Session] = None,
    hyperparameters: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    tags: Optional[Tags] = None,
    subnets: Optional[List[Union[str, PipelineVariable]]] = None,
    security_group_ids: Optional[List[Union[str, PipelineVariable]]] = None,
    model_uri: Optional[str] = None,
    model_channel_name: Optional[Union[str, PipelineVariable]] = None,
    metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
    encrypt_inter_container_traffic: Union[bool, PipelineVariable] = None,
    use_spot_instances: Optional[Union[bool, PipelineVariable]] = None,
    max_wait: Optional[Union[int, PipelineVariable]] = None,
    checkpoint_s3_uri: Optional[Union[str, PipelineVariable]] = None,
    checkpoint_local_path: Optional[Union[str, PipelineVariable]] = None,
    enable_network_isolation: Union[bool, PipelineVariable] = None,
    rules: Optional[List[RuleBase]] = None,
    debugger_hook_config: Optional[Union[DebuggerHookConfig, bool]] = None,
    tensorboard_output_config: Optional[TensorBoardOutputConfig] = None,
    enable_sagemaker_metrics: Optional[Union[bool, PipelineVariable]] = None,
    profiler_config: Optional[ProfilerConfig] = None,
    disable_profiler: Optional[bool] = None,
    environment: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
    max_retry_attempts: Optional[Union[int, PipelineVariable]] = None,
    source_dir: Optional[Union[str, PipelineVariable]] = None,
    git_config: Optional[Dict[str, str]] = None,
    container_log_level: Optional[Union[int, PipelineVariable]] = None,
    code_location: Optional[str] = None,
    entry_point: Optional[Union[str, PipelineVariable]] = None,
    dependencies: Optional[List[str]] = None,
    instance_groups: Optional[List[InstanceGroup]] = None,
    training_repository_access_mode: Optional[Union[str, PipelineVariable]] = None,
    training_repository_credentials_provider_arn: Optional[Union[str, PipelineVariable]] = None,
    container_entry_point: Optional[List[str]] = None,
    container_arguments: Optional[List[str]] = None,
    disable_output_compression: Optional[bool] = None,
    enable_infra_check: Optional[Union[bool, PipelineVariable]] = None,
    enable_remote_debug: Optional[Union[bool, PipelineVariable]] = None,
    config_name: Optional[str] = None,
    enable_session_tag_chaining: Optional[Union[bool, PipelineVariable]] = None,
) -> JumpStartEstimatorInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Estimator` object."""

    estimator_init_kwargs: JumpStartEstimatorInitKwargs = JumpStartEstimatorInitKwargs(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        model_type=model_type,
        role=role,
        region=region,
        instance_count=instance_count,
        instance_type=instance_type,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        volume_size=volume_size,
        volume_kms_key=volume_kms_key,
        max_run=max_run,
        input_mode=input_mode,
        output_path=output_path,
        output_kms_key=output_kms_key,
        base_job_name=base_job_name,
        sagemaker_session=sagemaker_session,
        tags=format_tags(tags),
        subnets=subnets,
        security_group_ids=security_group_ids,
        model_uri=model_uri,
        model_channel_name=model_channel_name,
        metric_definitions=metric_definitions,
        encrypt_inter_container_traffic=encrypt_inter_container_traffic,
        use_spot_instances=use_spot_instances,
        max_wait=max_wait,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        rules=rules,
        debugger_hook_config=debugger_hook_config,
        tensorboard_output_config=tensorboard_output_config,
        enable_sagemaker_metrics=enable_sagemaker_metrics,
        enable_network_isolation=enable_network_isolation,
        profiler_config=profiler_config,
        disable_profiler=disable_profiler,
        environment=environment,
        max_retry_attempts=max_retry_attempts,
        source_dir=source_dir,
        git_config=git_config,
        hyperparameters=hyperparameters,
        container_log_level=container_log_level,
        code_location=code_location,
        entry_point=entry_point,
        dependencies=dependencies,
        instance_groups=instance_groups,
        training_repository_access_mode=training_repository_access_mode,
        training_repository_credentials_provider_arn=training_repository_credentials_provider_arn,
        image_uri=image_uri,
        container_entry_point=container_entry_point,
        container_arguments=container_arguments,
        disable_output_compression=disable_output_compression,
        enable_infra_check=enable_infra_check,
        enable_remote_debug=enable_remote_debug,
        config_name=config_name,
        enable_session_tag_chaining=enable_session_tag_chaining,
    )

    estimator_init_kwargs, orig_session = _set_temp_sagemaker_session_if_not_set(
        kwargs=estimator_init_kwargs
    )
    estimator_init_kwargs.specs = verify_model_region_and_return_specs(
        **get_model_info_default_kwargs(
            estimator_init_kwargs, include_model_version=False, include_tolerate_flags=False
        ),
        version=estimator_init_kwargs.model_version or "*",
        scope=JumpStartScriptScope.TRAINING,
        # We set these flags to True to retrieve the json specs.
        # Exceptions will be thrown later if these are not tolerated.
        tolerate_deprecated_model=True,
        tolerate_vulnerable_model=True,
    )

    estimator_init_kwargs = _add_model_version_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_vulnerable_and_deprecated_status_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_sagemaker_session_with_custom_user_agent_to_kwargs(
        estimator_init_kwargs, orig_session
    )
    estimator_init_kwargs = _add_region_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_instance_type_and_count_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_image_uri_to_kwargs(estimator_init_kwargs)
    if hub_arn:
        estimator_init_kwargs = _add_model_reference_arn_to_kwargs(kwargs=estimator_init_kwargs)
    else:
        estimator_init_kwargs.model_reference_arn = None
        estimator_init_kwargs.hub_content_type = None
    estimator_init_kwargs = _add_model_uri_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_source_dir_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_entry_point_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_hyperparameters_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_metric_definitions_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_estimator_extra_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_role_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_env_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_tags_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_config_name_to_kwargs(estimator_init_kwargs)

    return estimator_init_kwargs


def get_fit_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    inputs: Optional[Union[str, Dict, TrainingInput, FileSystemInput]] = None,
    wait: Optional[bool] = None,
    logs: Optional[str] = None,
    job_name: Optional[str] = None,
    experiment_config: Optional[Dict[str, str]] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    tolerate_deprecated_model: Optional[bool] = None,
    sagemaker_session: Optional[Session] = None,
    config_name: Optional[str] = None,
) -> JumpStartEstimatorFitKwargs:
    """Returns kwargs required call `fit` on `sagemaker.estimator.Estimator` object."""

    estimator_fit_kwargs: JumpStartEstimatorFitKwargs = JumpStartEstimatorFitKwargs(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        inputs=inputs,
        wait=wait,
        logs=logs,
        job_name=job_name,
        experiment_config=experiment_config,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
    )

    estimator_fit_kwargs, _ = _set_temp_sagemaker_session_if_not_set(kwargs=estimator_fit_kwargs)
    estimator_fit_kwargs.specs = verify_model_region_and_return_specs(
        **get_model_info_default_kwargs(
            estimator_fit_kwargs, include_model_version=False, include_tolerate_flags=False
        ),
        version=estimator_fit_kwargs.model_version or "*",
        scope=JumpStartScriptScope.TRAINING,
        # We set these flags to True to retrieve the json specs.
        # Exceptions will be thrown later if these are not tolerated.
        tolerate_deprecated_model=True,
        tolerate_vulnerable_model=True,
    )

    estimator_fit_kwargs = _add_model_version_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_region_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_training_job_name_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_fit_extra_kwargs(estimator_fit_kwargs)

    return estimator_fit_kwargs


def get_deploy_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
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
    image_uri: Optional[Union[str, PipelineVariable]] = None,
    role: Optional[str] = None,
    predictor_cls: Optional[callable] = None,
    env: Optional[Dict[str, Union[str, PipelineVariable]]] = None,
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
    tolerate_deprecated_model: Optional[bool] = None,
    tolerate_vulnerable_model: Optional[bool] = None,
    use_compiled_model: Optional[bool] = None,
    model_name: Optional[str] = None,
    training_instance_type: Optional[str] = None,
    training_config_name: Optional[str] = None,
    inference_config_name: Optional[str] = None,
) -> JumpStartEstimatorDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Estimator` object."""

    model_deploy_kwargs: JumpStartModelDeployKwargs = model.get_deploy_kwargs(
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
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
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        training_config_name=training_config_name,
        config_name=inference_config_name,
    )

    model_init_kwargs: JumpStartModelInitKwargs = model.get_init_kwargs(
        model_id=model_id,
        model_from_estimator=True,
        model_version=model_version,
        hub_arn=hub_arn,
        instance_type=(
            model_deploy_kwargs.instance_type
            if training_instance_type is None
            or instance_type is not None  # always use supplied inference instance type
            else None
        ),
        region=region,
        image_uri=image_uri,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        predictor_cls=predictor_cls,
        role=role,
        name=model_name,
        vpc_config=vpc_config,
        sagemaker_session=model_deploy_kwargs.sagemaker_session,
        enable_network_isolation=enable_network_isolation,
        model_kms_key=model_kms_key,
        image_config=image_config,
        code_location=code_location,
        container_log_level=container_log_level,
        dependencies=dependencies,
        git_config=git_config,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        training_instance_type=training_instance_type,
        disable_instance_type_logging=True,
        config_name=model_deploy_kwargs.config_name,
    )

    estimator_deploy_kwargs: JumpStartEstimatorDeployKwargs = JumpStartEstimatorDeployKwargs(
        model_id=model_init_kwargs.model_id,
        model_version=model_init_kwargs.model_version,
        hub_arn=hub_arn,
        instance_type=model_init_kwargs.instance_type,
        initial_instance_count=model_deploy_kwargs.initial_instance_count,
        region=model_init_kwargs.region,
        image_uri=model_init_kwargs.image_uri,
        source_dir=model_init_kwargs.source_dir,
        entry_point=model_init_kwargs.entry_point,
        env=model_init_kwargs.env,
        predictor_cls=model_init_kwargs.predictor_cls,
        serializer=model_deploy_kwargs.serializer,
        deserializer=model_deploy_kwargs.deserializer,
        accelerator_type=model_deploy_kwargs.accelerator_type,
        endpoint_name=model_deploy_kwargs.endpoint_name,
        tags=model_deploy_kwargs.tags,
        kms_key=model_deploy_kwargs.kms_key,
        wait=model_deploy_kwargs.wait,
        data_capture_config=model_deploy_kwargs.data_capture_config,
        async_inference_config=model_deploy_kwargs.async_inference_config,
        serverless_inference_config=model_deploy_kwargs.serverless_inference_config,
        volume_size=model_deploy_kwargs.volume_size,
        model_data_download_timeout=model_deploy_kwargs.model_data_download_timeout,
        container_startup_health_check_timeout=(
            model_deploy_kwargs.container_startup_health_check_timeout
        ),
        inference_recommendation_id=model_deploy_kwargs.inference_recommendation_id,
        explainer_config=model_deploy_kwargs.explainer_config,
        role=model_init_kwargs.role,
        model_name=model_init_kwargs.name,
        vpc_config=model_init_kwargs.vpc_config,
        sagemaker_session=model_init_kwargs.sagemaker_session,
        enable_network_isolation=model_init_kwargs.enable_network_isolation,
        model_kms_key=model_init_kwargs.model_kms_key,
        image_config=model_init_kwargs.image_config,
        code_location=model_init_kwargs.code_location,
        container_log_level=model_init_kwargs.container_log_level,
        dependencies=model_init_kwargs.dependencies,
        git_config=model_init_kwargs.git_config,
        tolerate_vulnerable_model=model_init_kwargs.tolerate_vulnerable_model,
        tolerate_deprecated_model=model_init_kwargs.tolerate_deprecated_model,
        use_compiled_model=use_compiled_model,
        config_name=model_deploy_kwargs.config_name,
    )

    return estimator_deploy_kwargs


def _add_region_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets region in kwargs based on default or override, returns full kwargs."""
    kwargs.region = (
        kwargs.region or kwargs.sagemaker_session.boto_region_name or JUMPSTART_DEFAULT_REGION_NAME
    )
    return kwargs


def _add_sagemaker_session_with_custom_user_agent_to_kwargs(
    kwargs: JumpStartKwargs, orig_session: Optional[Session]
) -> JumpStartKwargs:
    """Sets session in kwargs based on default or override, returns full kwargs."""
    kwargs.sagemaker_session = orig_session or get_default_jumpstart_session_with_user_agent_suffix(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        config_name=None,
        is_hub_content=kwargs.hub_arn is not None,
    )
    return kwargs


def _add_model_version_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets model version in kwargs based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    if kwargs.hub_arn:
        hub_content_version = kwargs.specs.version
        kwargs.model_version = hub_content_version

    return kwargs


def _add_role_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets role based on default or override, returns full kwargs."""

    kwargs.role = resolve_estimator_sagemaker_config_field(
        field_name="role",
        field_val=kwargs.role,
        sagemaker_session=kwargs.sagemaker_session,
        default_value=kwargs.role,
    )

    return kwargs


def _add_instance_type_and_count_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets instance type and count in kwargs based on default or override, returns full kwargs."""

    orig_instance_type = kwargs.instance_type

    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        **get_model_info_default_kwargs(kwargs), scope=JumpStartScriptScope.TRAINING
    )

    kwargs.instance_count = kwargs.instance_count or 1

    if orig_instance_type is None:
        JUMPSTART_LOGGER.info(
            "No instance type selected for training job. Defaulting to %s.", kwargs.instance_type
        )

    return kwargs


def _add_tags_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets tags in kwargs based on default or override, returns full kwargs."""

    full_model_version = kwargs.specs.version

    if kwargs.sagemaker_session.settings.include_jumpstart_tags:
        kwargs.tags = add_jumpstart_model_info_tags(
            kwargs.tags,
            kwargs.model_id,
            full_model_version,
            config_name=kwargs.config_name,
            scope=JumpStartScriptScope.TRAINING,
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

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets image uri in kwargs based on default or override, returns full kwargs."""

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        **get_model_info_default_kwargs(kwargs),
        instance_type=kwargs.instance_type,
        framework=None,
        image_scope=JumpStartScriptScope.TRAINING,
    )

    return kwargs


def _add_model_reference_arn_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
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


def _add_model_uri_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets model uri in kwargs based on default or override, returns full kwargs."""

    if _model_supports_training_model_uri(**get_model_info_default_kwargs(kwargs)):
        default_model_uri = model_uris.retrieve(
            model_scope=JumpStartScriptScope.TRAINING,
            instance_type=kwargs.instance_type,
            **get_model_info_default_kwargs(kwargs),
        )

        if (
            kwargs.model_uri is not None
            and kwargs.model_uri != default_model_uri
            and not _model_supports_incremental_training(**get_model_info_default_kwargs(kwargs))
        ):
            JUMPSTART_LOGGER.warning(
                "'%s' does not support incremental training but is being trained with"
                " non-default model artifact.",
                kwargs.model_id,
            )

        kwargs.model_uri = kwargs.model_uri or default_model_uri

    return kwargs


def _add_vulnerable_and_deprecated_status_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets deprecated and vulnerability check status, returns full kwargs."""

    kwargs.tolerate_deprecated_model = kwargs.tolerate_deprecated_model or False
    kwargs.tolerate_vulnerable_model = kwargs.tolerate_vulnerable_model or False

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartEstimatorInitKwargs) -> JumpStartEstimatorInitKwargs:
    """Sets source dir in kwargs based on default or override, returns full kwargs."""

    kwargs.source_dir = kwargs.source_dir or script_uris.retrieve(
        script_scope=JumpStartScriptScope.TRAINING, **get_model_info_default_kwargs(kwargs)
    )

    return kwargs


def _add_env_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets environment in kwargs based on default or override, returns full kwargs."""

    extra_env_vars = environment_variables.retrieve_default(
        **get_model_info_default_kwargs(kwargs),
        script=JumpStartScriptScope.TRAINING,
        instance_type=kwargs.instance_type,
        include_aws_sdk_env_vars=False,
    )

    model_package_artifact_uri = _retrieve_model_package_model_artifact_s3_uri(
        **get_model_info_default_kwargs(kwargs),
        scope=JumpStartScriptScope.TRAINING,
    )

    if model_package_artifact_uri:
        extra_env_vars.update(
            {SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY: model_package_artifact_uri}
        )

    for key, value in extra_env_vars.items():
        kwargs.environment = update_dict_if_key_not_present(
            kwargs.environment,
            key,
            value,
        )

    environment = getattr(kwargs, "environment", {}) or {}
    if (
        environment.get(SAGEMAKER_GATED_MODEL_S3_URI_TRAINING_ENV_VAR_KEY)
        and str(environment.get("accept_eula", "")).lower() != "true"
    ):
        model_specs = kwargs.specs
        if model_specs.is_gated_model():
            raise ValueError(
                "Need to define ‘accept_eula'='true' within Environment. "
                f"{get_eula_message(model_specs, kwargs.region)}"
            )

    return kwargs


def _add_entry_point_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets entry point in kwargs based on default or override, returns full kwargs."""

    kwargs.entry_point = kwargs.entry_point or TRAINING_ENTRY_POINT_SCRIPT_NAME

    return kwargs


def _add_training_job_name_to_kwargs(
    kwargs: Optional[JumpStartEstimatorFitKwargs],
) -> JumpStartEstimatorFitKwargs:
    """Sets resource name based on default or override, returns full kwargs."""

    default_training_job_name = _retrieve_resource_name_base(
        **get_model_info_default_kwargs(kwargs),
        scope=JumpStartScriptScope.TRAINING,
    )

    kwargs.job_name = kwargs.job_name or (
        name_from_base(default_training_job_name) if default_training_job_name is not None else None
    )

    return kwargs


def _add_hyperparameters_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets hyperparameters in kwargs based on default or override, returns full kwargs."""

    kwargs.hyperparameters = (
        kwargs.hyperparameters.copy() if kwargs.hyperparameters is not None else {}
    )

    default_hyperparameters = hyperparameters_utils.retrieve_default(
        **get_model_info_default_kwargs(kwargs),
        instance_type=kwargs.instance_type,
    )

    for key, value in default_hyperparameters.items():
        kwargs.hyperparameters = update_dict_if_key_not_present(
            kwargs.hyperparameters,
            key,
            value,
        )

    if kwargs.hyperparameters == {}:
        kwargs.hyperparameters = None

    return kwargs


def _add_metric_definitions_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets metric definitions in kwargs based on default or override, returns full kwargs."""

    kwargs.metric_definitions = (
        kwargs.metric_definitions.copy() if kwargs.metric_definitions is not None else []
    )

    default_metric_definitions = (
        metric_definitions_utils.retrieve_default(
            **get_model_info_default_kwargs(kwargs),
            instance_type=kwargs.instance_type,
        )
        or []
    )

    for metric_definition in default_metric_definitions:
        if metric_definition["Name"] not in {
            definition["Name"] for definition in kwargs.metric_definitions
        }:
            kwargs.metric_definitions.append(metric_definition)

    if kwargs.metric_definitions == []:
        kwargs.metric_definitions = None

    return kwargs


def _add_estimator_extra_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    estimator_kwargs_to_add = _retrieve_estimator_init_kwargs(
        **get_model_info_default_kwargs(kwargs), instance_type=kwargs.instance_type
    )

    for key, value in estimator_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            resolved_value = resolve_estimator_sagemaker_config_field(
                field_name=key,
                field_val=value,
                sagemaker_session=kwargs.sagemaker_session,
            )
            setattr(kwargs, key, resolved_value)

    return kwargs


def _add_fit_extra_kwargs(kwargs: JumpStartEstimatorFitKwargs) -> JumpStartEstimatorFitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    fit_kwargs_to_add = _retrieve_estimator_fit_kwargs(**get_model_info_default_kwargs(kwargs))

    for key, value in fit_kwargs_to_add.items():
        if getattr(kwargs, key) is None:
            setattr(kwargs, key, value)

    return kwargs


def _add_config_name_to_kwargs(
    kwargs: JumpStartEstimatorInitKwargs,
) -> JumpStartEstimatorInitKwargs:
    """Sets tags in kwargs based on default or override, returns full kwargs."""

    kwargs.config_name = kwargs.config_name or get_top_ranked_config_name(
        scope=JumpStartScriptScope.TRAINING,
        **get_model_info_default_kwargs(kwargs, include_config_name=False),
    )

    return kwargs
