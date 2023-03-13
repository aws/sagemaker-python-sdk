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
"""Accessors to retrieve metric definition for training jobs."""

from __future__ import absolute_import

import logging
from typing import Dict, Optional, List

from sagemaker.jumpstart import utils as jumpstart_utils
from sagemaker.jumpstart import artifacts

logger = logging.getLogger(__name__)


def retrieve_default(
    region: Optional[str] = None,
    model_id: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Optional[List[Dict[str, str]]]:
    """Retrieves the default training metric definitions for the model matching the given arguments.

    Args:
        region (str): The AWS Region for which to retrieve the default default training metric
            definitions. Defaults to ``None``.
        model_id (str): The model ID of the model for which to
            retrieve the default training metric definitions. (Default: None).
        model_version (str): The version of the model for which to retrieve the
            default training metric definitions. (Default: None).
    Returns:
        list: The default metric definitions to use for the model or None.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError(
            "Must specify `model_id` and `model_version` when retrieving default training "
            "metric definitions."
        )

    return artifacts._retrieve_default_training_metric_definitions(model_id, model_version, region)
