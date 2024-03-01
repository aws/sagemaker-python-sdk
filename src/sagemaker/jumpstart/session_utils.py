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
"""This module stores SageMaker Session utilities for JumpStart models."""

from __future__ import absolute_import

from typing import Optional, Tuple
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION

from sagemaker.jumpstart.utils import get_jumpstart_model_id_version_from_resource_arn
from sagemaker.session import Session
from sagemaker.utils import aws_partition


def get_model_id_version_from_endpoint(
    endpoint_name: str,
    inference_component_name: Optional[str] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Tuple[str, str, Optional[str]]:
    """Given an endpoint and optionally inference component names, return the model ID and version.

    Infers the model ID and version based on the resource tags. Returns a tuple of the model ID
    and version. A third string element is included in the tuple for any inferred inference
    component name, or 'None' if it's a model-based endpoint.

    JumpStart adds tags automatically to endpoints, models, endpoint configs, and inference
    components launched in SageMaker Studio and programmatically with the SageMaker Python SDK.

    Raises:
        ValueError: If model ID and version cannot be inferred from the endpoint.
    """
    if inference_component_name or sagemaker_session.is_inference_component_based_endpoint(
        endpoint_name
    ):
        if inference_component_name:
            (
                model_id,
                model_version,
            ) = _get_model_id_version_from_inference_component_endpoint_with_inference_component_name(  # noqa E501  # pylint: disable=c0301
                inference_component_name, sagemaker_session
            )

        else:
            (
                model_id,
                model_version,
                inference_component_name,
            ) = _get_model_id_version_from_inference_component_endpoint_without_inference_component_name(  # noqa E501  # pylint: disable=c0301
                endpoint_name, sagemaker_session
            )

    else:
        model_id, model_version = _get_model_id_version_from_model_based_endpoint(
            endpoint_name, inference_component_name, sagemaker_session
        )
    return model_id, model_version, inference_component_name


def _get_model_id_version_from_inference_component_endpoint_without_inference_component_name(
    endpoint_name: str, sagemaker_session: Session
) -> Tuple[str, str, str]:
    """Given an endpoint name, derives the model ID, version, and inferred inference component name.

    This function assumes the endpoint corresponds to an inference-component-based endpoint.
    An endpoint is inference-component-based if and only if the associated endpoint config
    has a role associated with it and no production variants with a ``ModelName`` field.

    Raises:
        ValueError: If there is not a single inference component associated with the endpoint.
    """
    inference_component_names = (
        sagemaker_session.list_and_paginate_inference_component_names_associated_with_endpoint(
            endpoint_name=endpoint_name
        )
    )

    if len(inference_component_names) == 0:
        raise ValueError(
            f"No inference component found for the following endpoint: {endpoint_name}. "
            "Use ``SageMaker.CreateInferenceComponent`` to add inference components to "
            "your endpoint."
        )
    if len(inference_component_names) > 1:
        raise ValueError(
            f"Multiple inference components found for the following endpoint: {endpoint_name}. "
            "Provide an 'inference_component_name' to retrieve the model ID and version "
            "associated with a particular inference component."
        )
    inference_component_name = inference_component_names[0]
    return (
        *_get_model_id_version_from_inference_component_endpoint_with_inference_component_name(
            inference_component_name, sagemaker_session
        ),
        inference_component_name,
    )


def _get_model_id_version_from_inference_component_endpoint_with_inference_component_name(
    inference_component_name: str, sagemaker_session: Session
):
    """Returns the model ID and version inferred from a SageMaker inference component.

    Raises:
        ValueError: If the inference component does not have tags from which the model ID
            and version can be inferred.
    """
    region: str = sagemaker_session.boto_region_name
    partition: str = aws_partition(region)
    account_id: str = sagemaker_session.account_id()

    inference_component_arn = (
        f"arn:{partition}:sagemaker:{region}:{account_id}:"
        f"inference-component/{inference_component_name}"
    )

    model_id, model_version = get_jumpstart_model_id_version_from_resource_arn(
        inference_component_arn, sagemaker_session
    )

    if not model_id:
        raise ValueError(
            "Cannot infer JumpStart model ID from inference component "
            f"'{inference_component_name}'. Please specify JumpStart `model_id` "
            "when retrieving default predictor for this inference component."
        )

    return model_id, model_version


def _get_model_id_version_from_model_based_endpoint(
    endpoint_name: str,
    inference_component_name: Optional[str],
    sagemaker_session: Session,
) -> Tuple[str, str]:
    """Returns the model ID and version inferred from a model-based endpoint.

    Raises:
        ValueError: If an inference component name is supplied, or if the endpoint does
            not have tags from which the model ID and version can be inferred.
    """

    if inference_component_name:
        raise ValueError("Cannot specify inference component name for model-based endpoints.")

    region: str = sagemaker_session.boto_region_name
    partition: str = aws_partition(region)
    account_id: str = sagemaker_session.account_id()

    # SageMaker Tagging requires endpoint names to be lower cased
    endpoint_name = endpoint_name.lower()

    endpoint_arn = f"arn:{partition}:sagemaker:{region}:{account_id}:endpoint/{endpoint_name}"

    model_id, model_version = get_jumpstart_model_id_version_from_resource_arn(
        endpoint_arn, sagemaker_session
    )

    if not model_id:
        raise ValueError(
            f"Cannot infer JumpStart model ID from endpoint '{endpoint_name}'. "
            "Please specify JumpStart `model_id` when retrieving default "
            "predictor for this endpoint."
        )

    return model_id, model_version


def get_model_id_version_from_training_job(
    training_job_name: str,
    sagemaker_session: Optional[Session] = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Tuple[str, str]:
    """Returns the model ID and version inferred from a training job.

    Raises:
        ValueError: If the training job does not have tags from which the model ID
            and version can be inferred. JumpStart adds tags automatically to training jobs
            launched in SageMaker Studio and programmatically with the SageMaker Python SDK.
    """
    region: str = sagemaker_session.boto_region_name
    partition: str = aws_partition(region)
    account_id: str = sagemaker_session.account_id()

    training_job_arn = (
        f"arn:{partition}:sagemaker:{region}:{account_id}:training-job/{training_job_name}"
    )

    model_id, inferred_model_version = get_jumpstart_model_id_version_from_resource_arn(
        training_job_arn, sagemaker_session
    )

    model_version = inferred_model_version or None

    if not model_id:
        raise ValueError(
            f"Cannot infer JumpStart model ID from training job '{training_job_name}'. "
            "Please specify JumpStart `model_id` when retrieving Estimator "
            "for this training job."
        )

    return model_id, model_version
