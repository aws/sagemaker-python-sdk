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
"""This module stores JumpStart Model factory methods."""
from __future__ import absolute_import


from typing import Any, Dict, Optional
from sagemaker import environment_variables, image_uris, instance_types, model_uris, script_uris
from sagemaker.jumpstart.artifacts import _model_supports_prepacked_inference, _retrieve_kwargs
from sagemaker.jumpstart.constants import (
    INFERENCE_ENTRY_POINT_SCRIPT_NAME,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import EnvVariableUseCase, JumpStartScriptScope, KwargUseCase
from sagemaker.jumpstart.predictor import JumpStartPredictor
from sagemaker.jumpstart.types import JumpStartModelDeployKwargs, JumpStartModelInitKwargs
from sagemaker.jumpstart.utils import update_dict_if_key_not_present

from sagemaker.predictor import Predictor


def _add_region_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets region kwargs based on default or override, returns full kwargs."""

    kwargs.region = kwargs.region or JUMPSTART_DEFAULT_REGION_NAME

    return kwargs


def _add_model_version_to_kwargs(
    kwargs: JumpStartModelInitKwargs,
) -> JumpStartModelInitKwargs:
    """Sets model version based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    return kwargs


def _add_instance_type_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets instance type based on default or override, returns full kwargs."""

    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        region=kwargs.region,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        scope=JumpStartScriptScope.INFERENCE,
    )

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets image uri based on default or override, returns full kwargs."""

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        region=kwargs.region,
        framework=None,
        image_scope=JumpStartScriptScope.INFERENCE,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
    )

    return kwargs


def _add_model_data_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets model data based on default or override, returns full kwargs."""

    model_data = kwargs.model_data

    kwargs.model_data = model_data or model_uris.retrieve(
        model_scope=JumpStartScriptScope.INFERENCE,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
    )

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets source dir based on default or override, returns full kwargs."""

    source_dir = kwargs.source_dir

    if not _model_supports_prepacked_inference(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
    ):
        source_dir = source_dir or script_uris.retrieve(
            script_scope=JumpStartScriptScope.INFERENCE,
            model_id=kwargs.model_id,
            model_version=kwargs.model_version,
            region=kwargs.region,
        )

    kwargs.source_dir = source_dir

    return kwargs


def _add_entry_point_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets entry point based on default or override, returns full kwargs."""

    entry_point = kwargs.entry_point

    if not _model_supports_prepacked_inference(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
    ):

        entry_point = entry_point or INFERENCE_ENTRY_POINT_SCRIPT_NAME

    kwargs.entry_point = entry_point

    return kwargs


def _add_env_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets env based on default or override, returns full kwargs."""

    env = kwargs.env

    if env is None:
        env = {}

    extra_env_vars = environment_variables.retrieve_default(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        use_case=EnvVariableUseCase.SAGEMAKER_PYTHON_SDK,
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


def _add_extra_model_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    model_kwargs_to_add = _retrieve_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        use_case=KwargUseCase.MODEL,
    )

    for key, value in model_kwargs_to_add.items():
        if hasattr(kwargs, key) and getattr(kwargs, key) is None:
            setattr(kwargs, key, value)
        else:
            update_dict_if_key_not_present(
                kwargs.kwargs,
                key,
                value,
            )

    return kwargs


def _add_predictor_cls_to_kwargs(kwargs: JumpStartModelInitKwargs) -> JumpStartModelInitKwargs:
    """Sets predictor class kwargs based on on default or override, returns full kwargs."""

    predictor_cls = kwargs.predictor_cls or JumpStartPredictor

    kwargs.predictor_cls = predictor_cls
    return kwargs


def _add_deploy_extra_kwargs(kwargs: JumpStartModelInitKwargs) -> Dict[str, Any]:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    deploy_kwargs_to_add = _retrieve_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        use_case=KwargUseCase.MODEL_DEPLOY,
    )

    for key, value in deploy_kwargs_to_add.items():
        if hasattr(kwargs, key) and getattr(kwargs, key) is None:
            setattr(kwargs, key, value)
        else:
            update_dict_if_key_not_present(
                kwargs.kwargs,
                key,
                value,
            )

    return kwargs


def get_deploy_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    region: Optional[str] = None,
    initial_instance_count: Optional[int] = None,
    instance_type: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> JumpStartModelDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Model` object."""

    deploy_kwargs: JumpStartModelDeployKwargs = JumpStartModelDeployKwargs(
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        model_id=model_id,
        model_version=model_version,
        region=region,
        kwargs=kwargs,
    )

    deploy_kwargs = _add_model_version_to_kwargs(kwargs=deploy_kwargs)

    deploy_kwargs = _add_instance_type_to_kwargs(
        kwargs=deploy_kwargs,
    )

    deploy_kwargs.initial_instance_count = initial_instance_count or 1

    deploy_kwargs = _add_deploy_extra_kwargs(kwargs=deploy_kwargs)

    return deploy_kwargs


def get_init_kwargs(
    model_id: str,
    model_from_estimator: bool = False,
    model_version: Optional[str] = None,
    instance_type: Optional[str] = None,
    region: Optional[str] = None,
    image_uri: Optional[str] = None,
    model_data: Optional[str] = None,
    source_dir: Optional[str] = None,
    entry_point: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    predictor_cls: Optional[Predictor] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> JumpStartModelInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Model` object."""

    model_init_kwargs: JumpStartModelInitKwargs = JumpStartModelInitKwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        region=region,
        image_uri=image_uri,
        model_data=model_data,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        predictor_cls=predictor_cls,
        kwargs=kwargs,
    )

    model_init_kwargs = _add_model_version_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_region_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_instance_type_to_kwargs(
        kwargs=model_init_kwargs,
    )

    model_init_kwargs = _add_image_uri_to_kwargs(kwargs=model_init_kwargs)

    # we use the model artifact from the training job output
    if not model_from_estimator:
        model_init_kwargs = _add_model_data_to_kwargs(kwargs=model_init_kwargs)

    model_init_kwargs = _add_source_dir_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_entry_point_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_env_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_predictor_cls_to_kwargs(kwargs=model_init_kwargs)
    model_init_kwargs = _add_extra_model_kwargs(kwargs=model_init_kwargs)

    return model_init_kwargs
