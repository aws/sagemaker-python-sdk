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
"""This module contains functions to obtain JumpStart model payloads."""
from __future__ import absolute_import
from copy import deepcopy
from typing import Dict, Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    JumpStartModelType,
)
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_example_payloads(
    model_id: str,
    model_version: str,
    region: Optional[str],
    hub_arn: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> Optional[Dict[str, JumpStartSerializablePayload]]:
    """Returns example payloads.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            get example payloads.
        model_version (str): Version of the JumpStart model for which to retrieve the
            example payloads.
        region (Optional[str]): Region for which to retrieve the
            example payloads.
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
    Returns:
        Optional[Dict[str, JumpStartSerializablePayload]]: dictionary mapping payload aliases
            to the serializable payload object.
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=JumpStartScriptScope.INFERENCE,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
    )

    default_payloads = model_specs.default_payloads

    if default_payloads:
        for payload in default_payloads.values():
            payload.accept = getattr(
                payload, "accept", model_specs.predictor_specs.default_accept_type
            )

    return deepcopy(default_payloads) if default_payloads else None
