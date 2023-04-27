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
"""This module contains functions for obtaining JumpStart kwargs."""
from __future__ import absolute_import
from typing import Optional
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    _KwargUseCase,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)


def _retrieve_kwargs(
    model_id: str,
    model_version: str,
    use_case: _KwargUseCase,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> dict:
    """Retrieves kwargs for `Model`, `Estimator, `Estimator.fit`, and `Model.deploy`.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the kwargs.
        model_version (str): Version of the JumpStart model for which to retrieve the
            kwargs.
        use_case (_KwargUseCase): The use case for which to retrieve kwargs.
        region (Optional[str]): Region for which to retrieve kwargs.
            (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).

    Returns:
        dict: the kwargs to use for the use case.
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    if use_case in {_KwargUseCase.MODEL, _KwargUseCase.MODEL_DEPLOY}:
        scope = JumpStartScriptScope.INFERENCE
    elif use_case in {_KwargUseCase.ESTIMATOR, _KwargUseCase.ESTIMATOR_FIT}:
        scope = JumpStartScriptScope.TRAINING
    else:
        raise ValueError(f"Unsupported named-argument use case: {use_case}")

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
    )

    if use_case == _KwargUseCase.MODEL:
        return model_specs.model_kwargs

    if use_case == _KwargUseCase.MODEL_DEPLOY:
        return model_specs.deploy_kwargs

    if use_case == _KwargUseCase.ESTIMATOR:
        return model_specs.estimator_kwargs

    if use_case == _KwargUseCase.ESTIMATOR_FIT:
        return model_specs.fit_kwargs

    raise ValueError(f"Unsupported named-argument use case: {use_case}")
