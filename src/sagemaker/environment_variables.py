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
"""Accessors to retrieve environment variables for hosting containers."""

from __future__ import absolute_import

import logging
from typing import Dict

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts

logger = logging.getLogger(__name__)


def retrieve_default(
    region=None,
    model_id=None,
    model_version=None,
) -> Dict[str, str]:
    """Retrieves the default container environment variables for the model matching the arguments.

    Args:
        region (str): Optional. The AWS Region for which to retrieve the default environment
             variables. (Default: None).
        model_id (str): Optional. The model ID of the model for which to
            retrieve the default environment variables. (Default: None).
        model_version (str): Optional. The version of the model for which to retrieve the
            default environment variables. (Default: None).
    Returns:
        dict: The variables to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify `model_id` and `model_version` when retrieving environment variables."
        )

    return artifacts._retrieve_default_environment_variables(model_id, model_version, region)
