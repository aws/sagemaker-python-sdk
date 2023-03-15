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
"""Accessors to retrieve the model artifact S3 URI of pretrained machine learning models."""
from __future__ import absolute_import

import logging
from typing import Optional

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts


logger = logging.getLogger(__name__)


def retrieve(
    region=None,
    model_id=None,
    model_version: Optional[str] = None,
    model_scope: Optional[str] = None,
    tolerate_vulnerable_model: bool = False,
    tolerate_deprecated_model: bool = False,
) -> str:
    """Retrieves the model artifact Amazon S3 URI for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the Jumpstart model S3 URI.
        model_id (str): The model ID of the JumpStart model for which to retrieve
            the model artifact S3 URI.
        model_version (str): The version of the JumpStart model for which to retrieve
            the model artifact S3 URI.
        model_scope (str): The model type.
            Valid values: "training" and "inference".
        tolerate_vulnerable_model (bool): ``True`` if vulnerable versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises an
            exception if the script used by this version of the model has dependencies with known
            security vulnerabilities. (Default: False).
        tolerate_deprecated_model (bool): ``True`` if deprecated versions of model
            specifications should be tolerated without raising an exception. If ``False``, raises
            an exception if the version of the model is deprecated. (Default: False).
    Returns:
        str: The model artifact S3 URI for the corresponding model.

    Raises:
        NotImplementedError: If the scope is not supported.
        ValueError: If the combination of arguments specified is not supported.
        VulnerableJumpStartModelError: If any of the dependencies required by the script have
            known security vulnerabilities.
        DeprecatedJumpStartModelError: If the version of the model is deprecated.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError("Must specify `model_id` and `model_version` when retrieving model URIs.")

    return artifacts._retrieve_model_uri(
        model_id,
        model_version,  # type: ignore
        model_scope,
        region,
        tolerate_vulnerable_model,
        tolerate_deprecated_model,
    )
