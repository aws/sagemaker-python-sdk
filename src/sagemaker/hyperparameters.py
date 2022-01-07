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
"""Accessors to retrieve hyperparameters for training jobs."""

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
    include_container_hyperparameters=False,
) -> Dict[str, str]:
    """Retrieves the default training hyperparameters for the model matching the given arguments.

    Args:
        region (str): Region for which to retrieve default hyperparameters.
        model_id (str): JumpStart model ID of the JumpStart model for which to
            retrieve the default hyperparameters.
        model_version (str): Version of the JumpStart model for which to retrieve the
            default hyperparameters.
        include_container_hyperparameters (bool): True if container hyperparameters
            should be returned as well. Container hyperparameters are not used to tune
            the specific algorithm, but rather by SageMaker Training to setup
            the training container environment. For example, there is a container hyperparameter
            that indicates the entrypoint script to use. These hyperparameters may be required
            when creating a training job with boto3, however the ``Estimator`` classes
            should take care of adding container hyperparameters to the job. (Default: False).
    Returns:
        dict: the hyperparameters to use for the model.

    Raises:
        ValueError: If the combination of arguments specified is not supported.
    """
    if not jumpstart_utils.is_jumpstart_model_input(model_id, model_version):
        raise ValueError("Must specify `model_id` and `model_version` when retrieving script URIs.")

    return artifacts._retrieve_default_hyperparameters(
        model_id, model_version, region, include_container_hyperparameters
    )
