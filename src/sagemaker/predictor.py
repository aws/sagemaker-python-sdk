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
"""Placeholder docstring"""
from __future__ import print_function, absolute_import

from typing import Optional
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.enums import JumpStartModelType

from sagemaker.jumpstart.factory.model import get_default_predictor
from sagemaker.jumpstart.session_utils import get_model_id_version_from_endpoint


from sagemaker.session import Session


# base_predictor was refactored from predictor.
# this import ensures backward compatibility.
from sagemaker.base_predictor import (  # noqa: F401 # pylint: disable=W0611
    Predictor,
    PredictorBase,
    RealTimePredictor,
)


def retrieve_default(
    endpoint_name: str,
    inference_component_name: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    model_type: JumpStartModelType = JumpStartModelType.OPEN_WEIGHTS,
) -> Predictor:
    """Retrieves the default predictor for the model matching the given arguments.

    Args:
        endpoint_name (str): Endpoint name for which to create a predictor.
        inference_component_name (str): Name of the Amazon SageMaker inference component
            from which to optionally create a predictor. (Default: None).
        sagemaker_session (Session): The SageMaker Session to attach to the predictor.
            (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
        region (str): The AWS Region for which to retrieve the default predictor.
            (Default: None).
        model_id (str): The model ID of the model for which to
            retrieve the default predictor. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default predictor. (Default: None).
        hub_arn (str): The arn of the SageMaker Hub for which to retrieve
            model details from. (Default: None).
        tolerate_vulnerable_model (bool): True if vulnerable versions of model
            specifications should be tolerated (exception not raised). If False, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): True if deprecated models should be tolerated
            (exception not raised). False if these models should raise an exception.
            (Default: False).
    Returns:
        Predictor: The default predictor to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported, or if a model ID or
            version cannot be inferred from the endpoint.
    """

    if model_id is None:
        (
            inferred_model_id,
            inferred_model_version,
            inferred_inference_component_name,
        ) = get_model_id_version_from_endpoint(
            endpoint_name, inference_component_name, sagemaker_session
        )

        if not inferred_model_id:
            raise ValueError(
                f"Cannot infer JumpStart model ID from endpoint '{endpoint_name}'. "
                "Please specify JumpStart `model_id` when retrieving default "
                "predictor for this endpoint."
            )

        model_id = inferred_model_id
        model_version = model_version or inferred_model_version or "*"
        inference_component_name = inference_component_name or inferred_inference_component_name
    else:
        model_version = model_version or "*"

    predictor = Predictor(
        endpoint_name=endpoint_name,
        component_name=inference_component_name,
        sagemaker_session=sagemaker_session,
    )

    return get_default_predictor(
        predictor=predictor,
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        region=region,
        tolerate_deprecated_model=tolerate_deprecated_model,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        sagemaker_session=sagemaker_session,
        model_type=model_type,
    )
