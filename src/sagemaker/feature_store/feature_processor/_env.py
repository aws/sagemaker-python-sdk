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
"""Contains class that determines the current execution environment."""
from __future__ import absolute_import

from typing import Dict, Optional
import json
import logging

logger = logging.getLogger("sagemaker")


class EnvironmentHelper:
    """Helper class to check if the current environment is a training job."""

    def is_training_job(self) -> bool:
        """Determine if the current execution environment is inside a SageMaker Training Job"""
        return self.load_training_resource_config() is not None

    def get_instance_count(self) -> int:
        """Determine the number of instances for the current execution environment."""
        resource_config = self.load_training_resource_config()
        return len(resource_config["hosts"]) if resource_config else 1

    def load_training_resource_config(self) -> Optional[Dict]:
        """Load the contents of resourceconfig.json contents.

        Returns:
            Optional[Dict]: None if not found.
        """
        SM_TRAINING_CONFIG_FILE_PATH = "/opt/ml/input/config/resourceconfig.json"
        try:
            with open(SM_TRAINING_CONFIG_FILE_PATH, "r") as cfgfile:
                resource_config = json.load(cfgfile)
                logger.debug("Contents of %s: %s", SM_TRAINING_CONFIG_FILE_PATH, resource_config)
                return resource_config
        except FileNotFoundError:
            return None
