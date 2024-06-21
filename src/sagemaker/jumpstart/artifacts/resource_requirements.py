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

from typing import Dict, Optional, Tuple

from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
    JumpStartModelType,
)
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session
from sagemaker.compute_resource_requirements.resource_requirements import ResourceRequirements

REQUIREMENT_TYPE_TO_SPEC_FIELD_NAME_TO_RESOURCE_REQUIREMENT_NAME_MAP: Dict[
    str, Dict[str, Tuple[str, str]]
] = {
    "requests": {
        "num_accelerators": ("num_accelerators", "num_accelerators"),
        "num_cpus": ("num_cpus", "num_cpus"),
        "copies": ("copies", "copy_count"),
        "min_memory_mb": ("memory", "min_memory"),
    },
    "limits": {
        "max_memory_mb": ("memory", "max_memory"),
    },
}


def _retrieve_default_resources(
    model_id: str,
    model_version: str,
    scope: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    instance_type: Optional[str] = None,
) -> ResourceRequirements:
    """Retrieves the default resource requirements for the model.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default resource requirements.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default resource requirements.
        scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
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
        instance_type (str): An instance type to optionally supply in order to get
            host requirements specific for the instance type.
    Returns:
        str: The default resource requirements to use for the model or None.

    Raises:
        ValueError: If the model is not available in the
            specified region due to lack of pre-defined resource requirements.
        NotImplementedError: If ScriptScope is not Inference, then we cannot
            retrieve default resource requirements
    """

    region = region or get_region_fallback(
        sagemaker_session=sagemaker_session,
    )

    model_specs = verify_model_region_and_return_specs(
        model_id=model_id,
        version=model_version,
        hub_arn=hub_arn,
        scope=scope,
        region=region,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        model_type=model_type,
        sagemaker_session=sagemaker_session,
    )

    if scope == JumpStartScriptScope.INFERENCE:
        is_dynamic_container_deployment_supported = (
            model_specs.dynamic_container_deployment_supported
        )
        default_resource_requirements: Dict[str, int] = (
            model_specs.hosting_resource_requirements or {}
        )
    else:
        raise NotImplementedError(
            f"Unsupported script scope for retrieving default resource requirements: '{scope}'"
        )

    instance_specific_resource_requirements: Dict[str, int] = (
        model_specs.hosting_instance_type_variants.get_instance_specific_resource_requirements(
            instance_type
        )
        if instance_type
        and getattr(model_specs, "hosting_instance_type_variants", None) is not None
        else {}
    )

    default_resource_requirements = {
        **default_resource_requirements,
        **instance_specific_resource_requirements,
    }

    if is_dynamic_container_deployment_supported:

        all_resource_requirement_kwargs = {}

        for (
            requirement_type,
            spec_field_to_resource_requirement_map,
        ) in REQUIREMENT_TYPE_TO_SPEC_FIELD_NAME_TO_RESOURCE_REQUIREMENT_NAME_MAP.items():
            requirement_kwargs = {}
            for spec_field, resource_requirement in spec_field_to_resource_requirement_map.items():
                if spec_field in default_resource_requirements:
                    requirement_kwargs[resource_requirement[0]] = default_resource_requirements[
                        spec_field
                    ]

            all_resource_requirement_kwargs[requirement_type] = requirement_kwargs

        return ResourceRequirements(**all_resource_requirement_kwargs)
    return None
