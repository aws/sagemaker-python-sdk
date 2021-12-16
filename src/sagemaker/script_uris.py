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

import logging

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import constants as jumpstart_constants
from sagemaker.jumpstart import artifacts

logger = logging.getLogger(__name__)


def retrieve(
    region=jumpstart_constants.JUMPSTART_DEFAULT_REGION_NAME,
    model_id=None,
    model_version=None,
    script_scope=None,
):
    """Retrieves the model script s3 URI for the model matching the given arguments.

    Args:
        region (str): Region for which to retrieve model script S3 URI.
        model_id (str): JumpStart model id for which to retrieve model script S3 URI.
        model_version (str): JumpStart model version for which to retrieve model script S3 URI.
        script_scope (str): The script type, i.e. what it is used for.
            Valid values: "training" and "inference".
    Returns:
        str: the model script URI for the corresponding model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError("Must specify `model_id` and `model_version` when retrieving script URIs.")

    assert model_id is not None
    assert model_version is not None
    return artifacts._retrieve_script_uri(model_id, model_version, script_scope, region)
