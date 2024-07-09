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
"""This module contains functions for obtaining JumpStart incremental training status."""
from __future__ import absolute_import
from typing import Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _model_supports_incremental_training(
    model_id: str,
    model_version: str,
    region: Optional[str],
    hub_arn: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    config_name: Optional[str] = None,
) -> bool:
    """Returns True if the model supports incremental training.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the support status for incremental training.
        model_version (str): Version of the JumpStart model for which to retrieve the
            support status for incremental training.
        region (Optional[str]): Region for which to retrieve the
            support status for incremental training.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated versions of model
            specifications should be tolerated (exception not raised). If False, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
    Returns:
        bool: the support status for incremental training.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.TRAINING,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        config_name=config_name,
    )

    return model_specs.supports_incremental_training()
