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
"""Utilities related to payloads of pretrained machine learning models."""
from __future__ import absolute_import

import logging
from typing import Dict, List, Optional

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.payload_utils import PayloadSerializer
from sagemaker.jumpstart.types import JumpStartSerializablePayload
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.session import Session


logger = logging.getLogger(__name__)


def retrieve_all_examples(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
    serialize: bool = False,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Optional[List[JumpStartSerializablePayload]]:
    """Retrieves the compatible payloads for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the Jumpstart model payloads.
        model_id (str): The model ID of the JumpStart model for which to retrieve
            the model payloads.
        model_version (str): The version of the JumpStart model for which to retrieve
            the model payloads.
        serialize (bool): Whether to serialize byte-stream valued payloads by downloading
            binary files from s3 and applying encoding, or to keep payload in pre-serialized
            state. Set this option to False if you want to avoid s3 downloads or if you
            want to inspect the payload in a human-readable form. (Default: False).
        tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): ``True`` if deprecated versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        Optional[List[JumpStartSerializablePayload]]: List of payloads or None.

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify JumpStart `model_id` and `model_version` when retrieving payloads."
        )

    unserialized_payload_dict: Optional[Dict[str, JumpStartSerializablePayload]] = (
        artifacts._retrieve_example_payloads(
            model_id=model_id,
            model_version=model_version,
            region=region,
            hub_arn=hub_arn,
            tolerate_vulnerable_model=tolerate_vulnerable_model,
            tolerate_deprecated_model=tolerate_deprecated_model,
            sagemaker_session=sagemaker_session,
            model_type=model_type,
        )
    )

    if unserialized_payload_dict is None:
        return None

    unserialized_payloads: List[JumpStartSerializablePayload] = list(
        unserialized_payload_dict.values()
    )

    if not serialize:
        return unserialized_payloads

    payload_serializer = PayloadSerializer(region=region, s3_client=sagemaker_session.s3_client)

    serialized_payloads: List[JumpStartSerializablePayload] = []

    for payload in unserialized_payloads:

        serialized_body = payload_serializer.serialize(payload)

        serialized_payloads.append(
            JumpStartSerializablePayload(
                {
                    "content_type": payload.content_type,
                    "body": serialized_body,
                    "accept": payload.accept,
                }
            )
        )

    return serialized_payloads


def retrieve_example(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
    hub_arn: Optional[str] = None,
    model_type: Optional[JumpStartModelType] = JumpStartModelType.OPEN_WEIGHTS,
    serialize: bool = False,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Optional[JumpStartSerializablePayload]:
    """Retrieves a single compatible payload for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the Jumpstart model payloads.
        model_id (str): The model ID of the JumpStart model for which to retrieve
            the model payload.
        model_version (str): The version of the JumpStart model for which to retrieve
            the model payload.
        model_type (str): The model type of the JumpStart model, either is open weight
            or proprietary.
        serialize (bool): Whether to serialize byte-stream valued payloads by downloading
            binary files from s3 and applying encoding, or to keep payload in pre-serialized
            state. Set this option to False if you want to avoid s3 downloads or if you
            want to inspect the payload in a human-readable form. (Default: False).
        tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): ``True`` if deprecated versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises
            an exception if the version of the model is deprecated. (Default: False).
        sagemaker_session (sagemaker.session.Session): A SageMaker Session
            object, used for SageMaker interactions. If not
            specified, one is created using the default AWS configuration
            chain. (Default: sagemaker.jumpstart.constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION).
    Returns:
        Optional[JumpStartSerializablePayload]: A single default payload or None.

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    example_payloads: Optional[List[JumpStartSerializablePayload]] = retrieve_all_examples(
        region=region,
        model_id=model_id,
        model_version=model_version,
        hub_arn=hub_arn,
        model_type=model_type,
        serialize=serialize,
        tolerate_vulnerable_model=tolerate_vulnerable_model,
        tolerate_deprecated_model=tolerate_deprecated_model,
        sagemaker_session=sagemaker_session,
    )

    if example_payloads is None or len(example_payloads) == 0:
        return None

    return example_payloads[0]
