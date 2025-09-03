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
"""This module contains functions for obtaining JumpStart hyperparameters."""
from __future__ import absolute_import
from typing import Dict, Optional
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.jumpstart.enums import (
    JumpStartModelType,
    JumpStartScriptScope,
    VariableScope,
)
from sagemaker.jumpstart.utils import (
    get_region_fallback,
    verify_model_region_and_return_specs,
)
from sagemaker.session import Session


def _retrieve_default_hyperparameters(
    model_id: str,
    model_version: str,
    hub_arn: Optional[str] = None,
    region: Optional[str] = None,
    include_container_hyperparameters: bool = False,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    instance_type: Optional[str] = None,
    config_name: Optional[str] = None,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
):
    """Retrieves the training hyperparameters for the model matching the given arguments.

    Args:
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default hyperparameters.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default hyperparameters.
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        region (str): Region for which to retrieve default hyperparameters.
            (Default: None).
        include_container_hyperparameters (bool): True if container hyperparameters
            should be returned as well. Container hyperparameters are not used to tune
            the specific algorithm, but rather by SageMaker Training to setup
            the training container environment. For example, there is a container hyperparameter
            that indicates the entrypoint script to use. These hyperparameters may be required
            when creating a training job with boto3, however the ``Estimator`` classes
            should take care of adding container hyperparameters to the job. (Default: False).
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
        instance_type (str): An instance type to optionally supply in order to get hyperparameters
            specific for the instance type.
        config_name (Optional[str]): Name of the JumpStart Model config to apply. (Default: None).
        model_type (JumpStartModelType): The type of the model, can be open weights model
            or proprietary model. (Default: JumpStartModelType.OPEN_WEIGHTS).
    Returns:
        dict: the hyperparameters to use for the model.
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
        model_type=model_type,
    )

    default_hyperparameters: Dict[str, str] = {}
    for hyperparameter in model_specs.hyperparameters:
        if (
            include_container_hyperparameters and hyperparameter.scope == VariableScope.CONTAINER
        ) or hyperparameter.scope == VariableScope.ALGORITHM:
            default_hyperparameters[hyperparameter.name] = str(hyperparameter.default)

    instance_specific_hyperparameters = (
        model_specs.training_instance_type_variants.get_instance_specific_hyperparameters(
            instance_type
        )
        if instance_type
        and getattr(model_specs, "training_instance_type_variants", None) is not None
        else []
    )

    for instance_specific_hyperparameter in instance_specific_hyperparameters:
        default_hyperparameters[instance_specific_hyperparameter.name] = str(
            instance_specific_hyperparameter.default
        )

    return default_hyperparameters
