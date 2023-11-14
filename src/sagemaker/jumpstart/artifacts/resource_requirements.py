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
"""This module contains functions for obtaining JumpStart resoure requirements."""
from __future__ import absolute_import

from typing import Optional

from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_REGION_NAME,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements


def _retrieve_default_resources(
    model_id: str,
    model_version: str,
    scope: str,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> ResourceRequirements:
    """Retrieves the default resource requirements for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default resource requirements.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default resource requirements.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        region (Optional[str]): Region for which to retrieve default resource requirements.
            (Default: None).
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
        str: The default resource requirements to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of pre-defined resource requirements.
        NotImplementedError: If ScriptScope is not Inference, then we cannot
            retrieve default resource requirements
    """

    if region is None:
        region = JUMPSTART_DEFAULT_REGION_NAME

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        is_dynamic_container_deployment_supported = (
            model_specs.dynamic_container_deployment_supported
        )
        default_resource_requirements = model_specs.hosting_resource_requirements
    else:
        raise NotImplementedError(
            f"Unsupported script scope for retrieving default resource requirements: '{scope}'"
        )

    if is_dynamic_container_deployment_supported:
        requests = {}
        if "num_accelerators" in default_resource_requirements:
            requests["num_accelerators"] = default_resource_requirements["num_accelerators"]
        if "min_memory_mb" in default_resource_requirements:
            requests["memory"] = default_resource_requirements["min_memory_mb"]
        if "num_cpus" in default_resource_requirements:
            requests["num_cpus"] = default_resource_requirements["num_cpus"]

        limits = {}
        if "max_memory_mb" in default_resource_requirements:
            limits["memory"] = default_resource_requirements["max_memory_mb"]
        return ResourceRequirements(requests=requests, limits=limits)
    return None
