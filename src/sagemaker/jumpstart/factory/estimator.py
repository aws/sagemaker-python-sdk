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


from typing import Any, Dict, List, Optional
from sagemaker import (
    hyperparameters as hyperparameters_utils,
    image_uris,
    instance_types,
    metric_definitions as metric_definitions_utils,
    model_uris,
    script_uris,
)
from sagemaker.jumpstart.artifacts import _retrieve_kwargs
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
    TRAINING_ENTRY_POINT_SCRIPT_NAME,
)
from sagemaker.jumpstart.enums import JumpStartScriptScope, KwargUseCase
from sagemaker.jumpstart.factory import model
from sagemaker.jumpstart.types import (
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartKwargs,
)
from sagemaker.jumpstart.utils import update_dict_if_key_not_present
from sagemaker.predictor import Predictor


def get_init_kwargs(
    model_id: str,
    model_version: Optional[str] = None,
    instance_type: Optional[str] = None,
    instance_count: Optional[int] = None,
    region: Optional[str] = None,
    image_uri: Optional[str] = None,
    model_uri: Optional[str] = None,
    source_dir: Optional[str] = None,
    entry_point: Optional[str] = None,
    hyperparameters: Optional[dict] = None,
    metric_definitions: Optional[List[dict]] = None,
    kwargs: Optional[dict] = None,
) -> JumpStartEstimatorInitKwargs:
    """Returns kwargs required to instantiate `sagemaker.estimator.Estimator` object."""

    estimator_init_kwargs: JumpStartEstimatorInitKwargs = JumpStartEstimatorInitKwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        region=region,
        image_uri=image_uri,
        model_uri=model_uri,
        source_dir=source_dir,
        entry_point=entry_point,
        instance_count=instance_count,
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
        kwargs=kwargs,
    )

    estimator_init_kwargs = _add_model_version_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_region_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_instance_type_and_count_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_image_uri_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_model_uri_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_source_dir_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_entry_point_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_hyperparameters_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_metric_definitions_to_kwargs(estimator_init_kwargs)
    estimator_init_kwargs = _add_estimator_extra_kwargs(estimator_init_kwargs)

    return estimator_init_kwargs


def get_fit_kwargs(
    model_id: str,
    model_version: Optional[str],
    instance_type: Optional[str],
    instance_count: Optional[int],
    region: Optional[str],
    kwargs: Any,
) -> JumpStartEstimatorFitKwargs:
    """Returns kwargs required call `fit` on `sagemaker.estimator.Estimator` object."""

    estimator_fit_kwargs: JumpStartEstimatorFitKwargs = JumpStartEstimatorFitKwargs(
        model_id=model_id,
        model_version=model_version,
        instance_type=instance_type,
        region=region,
        instance_count=instance_count,
        kwargs=kwargs,
    )

    estimator_fit_kwargs = _add_model_version_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_region_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_instance_type_and_count_to_kwargs(estimator_fit_kwargs)
    estimator_fit_kwargs = _add_fit_extra_kwargs(estimator_fit_kwargs)

    return estimator_fit_kwargs


def get_deploy_kwargs(
    model_id: str,
    model_version: Optional[str],
    region: Optional[str] = None,
    image_uri: Optional[str] = None,
    source_dir: Optional[str] = None,
    entry_point: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    predictor_cls: Optional[Predictor] = None,
    initial_instance_count: Optional[int] = None,
    instance_type: Optional[str] = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> JumpStartEstimatorDeployKwargs:
    """Returns kwargs required to call `deploy` on `sagemaker.estimator.Estimator` object."""

    model_deploy_kwargs = model.get_deploy_kwargs(
        model_id=model_id,
        model_version=model_version,
        region=region,
        initial_instance_count=initial_instance_count,
        instance_type=instance_type,
        kwargs=kwargs,
    )

    model_init_kwargs = model.get_init_kwargs(
        model_id=model_id,
        model_version=model_version,
        region=region,
        instance_type=instance_type,
        image_uri=image_uri,
        source_dir=source_dir,
        entry_point=entry_point,
        env=env,
        predictor_cls=predictor_cls,
        model_from_estimator=True,
        kwargs=kwargs,
    )

    all_extra_kwargs = {**model_deploy_kwargs.kwargs, **model_init_kwargs.kwargs}

    estimator_fit_kwargs: JumpStartEstimatorDeployKwargs = JumpStartEstimatorDeployKwargs(
        kwargs=all_extra_kwargs,
        model_id=model_id,
        model_version=model_init_kwargs.model_version,
        region=model_init_kwargs.region,
        initial_instance_count=model_deploy_kwargs.initial_instance_count,
        instance_type=model_init_kwargs.instance_type,
        image_uri=model_init_kwargs.image_uri,
        source_dir=model_init_kwargs.source_dir,
        entry_point=model_init_kwargs.entry_point,
        env=model_init_kwargs.env,
        predictor_cls=model_init_kwargs.predictor_cls,
    )

    return estimator_fit_kwargs


def _add_region_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets region in kwargs based on default or override, returns full kwargs."""
    kwargs.region = kwargs.region or JUMPSTART_DEFAULT_REGION_NAME
    return kwargs


def _add_model_version_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets model version in kwargs based on default or override, returns full kwargs."""

    kwargs.model_version = kwargs.model_version or "*"

    return kwargs


def _add_instance_type_and_count_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets instance type and count in kwargs based on default or override, returns full kwargs."""

    kwargs.instance_type = kwargs.instance_type or instance_types.retrieve_default(
        region=kwargs.region,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        scope=JumpStartScriptScope.TRAINING,
    )

    kwargs.instance_count = kwargs.instance_count or 1

    return kwargs


def _add_image_uri_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets image uri in kwargs based on default or override, returns full kwargs."""

    kwargs.image_uri = kwargs.image_uri or image_uris.retrieve(
        region=None,
        framework=None,
        image_scope=JumpStartScriptScope.TRAINING,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        instance_type=kwargs.instance_type,
    )

    return kwargs


def _add_model_uri_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets model uri in kwargs based on default or override, returns full kwargs."""

    kwargs.model_uri = kwargs.model_uri or model_uris.retrieve(
        model_scope=JumpStartScriptScope.TRAINING,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
    )

    return kwargs


def _add_source_dir_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets source dir in kwargs based on default or override, returns full kwargs."""

    kwargs.source_dir = kwargs.source_dir or script_uris.retrieve(
        script_scope=JumpStartScriptScope.TRAINING,
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
    )

    return kwargs


def _add_entry_point_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets entry point in kwargs based on default or override, returns full kwargs."""

    kwargs.entry_point = kwargs.entry_point or TRAINING_ENTRY_POINT_SCRIPT_NAME

    return kwargs


def _add_hyperparameters_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets hyperparameters in kwargs based on default or override, returns full kwargs."""

    kwargs.hyperparameters = (
        kwargs.hyperparameters.copy() if kwargs.hyperparameters is not None else {}
    )

    default_hyperparameters = hyperparameters_utils.retrieve_default(
        region=kwargs.region, model_id=kwargs.model_id, model_version=kwargs.model_version
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


def _add_metric_definitions_to_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets metric definitions in kwargs based on default or override, returns full kwargs."""

    kwargs.metric_definitions = (
        kwargs.metric_definitions.copy() if kwargs.metric_definitions is not None else []
    )

    default_metric_definitions = metric_definitions_utils.retrieve_default(
        region=kwargs.region, model_id=kwargs.model_id, model_version=kwargs.model_version
    )

    for metric_definition in default_metric_definitions:
        if metric_definition["Name"] not in {
            definition["Name"] for definition in kwargs.metric_definitions
        }:
            kwargs.metric_definitions.append(metric_definition)

    if kwargs.metric_definitions == []:
        kwargs.metric_definitions = None

    return kwargs


def _add_estimator_extra_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    estimator_kwargs_to_add = _retrieve_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        use_case=KwargUseCase.ESTIMATOR,
    )

    for key, value in estimator_kwargs_to_add.items():
        if hasattr(kwargs, key) and getattr(kwargs, key) is None:
            setattr(kwargs, key, value)
        else:
            update_dict_if_key_not_present(
                kwargs.kwargs,
                key,
                value,
            )

    return kwargs


def _add_fit_extra_kwargs(kwargs: JumpStartKwargs) -> JumpStartKwargs:
    """Sets extra kwargs based on default or override, returns full kwargs."""

    fit_kwargs_to_add = _retrieve_kwargs(
        model_id=kwargs.model_id,
        model_version=kwargs.model_version,
        region=kwargs.region,
        use_case=KwargUseCase.ESTIMATOR_FIT,
    )

    for key, value in fit_kwargs_to_add.items():
        if hasattr(kwargs, key) and getattr(kwargs, key) is None:
            setattr(kwargs, key, value)
        else:
            update_dict_if_key_not_present(
                kwargs.kwargs,
                key,
                value,
            )

    return kwargs
