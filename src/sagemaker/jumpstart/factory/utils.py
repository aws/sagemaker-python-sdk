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
from typing import Tuple, Union

from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.types import (
    JumpStartEstimatorDeployKwargs,
    JumpStartEstimatorFitKwargs,
    JumpStartEstimatorInitKwargs,
    JumpStartModelDeployKwargs,
    JumpStartModelInitKwargs,
)
from sagemaker.session import Session

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
