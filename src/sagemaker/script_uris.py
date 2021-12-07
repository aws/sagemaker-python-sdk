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
"""Functions for generating S3 model script URIs for pre-built SageMaker models."""
from __future__ import absolute_import

import json
import logging
import os
import re

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import accessors as jumpstart_accessors
from sagemaker.jumpstart import constants as jumpstart_constants

logger = logging.getLogger(__name__)


def retrieve(
    region=jumpstart_constants.JUMPSTART_DEFAULT_REGION_NAME,
    model_id=None,
    model_version=None,
    model_scope=None,
):
    """Retrieves the model script s3 URI for the model matching the given arguments.

    Args:
        region (str): Region for which to retrieve model script S3 URI.
        model_id (str): JumpStart model id for which to retrieve model script S3 URI.
        model_version (str): JumpStart model version for which to retrieve model script S3 URI.
        model_scope (str): The model type, i.e. what it is used for.
            Valid values: "training", "inference", "eia".
    Returns:
        str: the model script URI for the corresponding model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if model_id is None or model_version is None:
        raise ValueError(
            "Must specify `model_id` and `model_version` when getting model script uri for "
            "JumpStart models. "
        )
    model_specs = jumpstart_accessors.JumpStartModelsCache.get_model_specs(
        region, model_id, model_version
    )
    if model_scope is None:
        raise ValueError(
            "Must specify `model_scope` argument to retrieve model script uri for JumpStart models."
        )
    if model_scope == "inference":
        model_script_key = model_specs.hosting_script_key
    elif model_scope == "training":
        if not model_specs.training_supported:
            raise ValueError(f"JumpStart model id '{model_id}' does not support training.")
        model_script_key = model_specs.training_script_key
    else:
        raise ValueError("JumpStart models only support inference and training.")

    bucket = jumpstart_utils.get_jumpstart_content_bucket(region)

    script_s3_uri = f"s3://{bucket}/{model_script_key}"

    return script_s3_uri
