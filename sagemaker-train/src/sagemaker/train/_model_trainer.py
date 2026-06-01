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
"""ModelTrainer class for configuring and launching training jobs.

This module contains the networking resolution fix for ModelTrainer.
The _resolve_networking method is patched to properly handle VpcConfig
from sagemaker_config.
"""
from __future__ import absolute_import

import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from sagemaker.core.config.config_schema import (
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_SUBNETS_PATH,
    TRAINING_JOB_SECURITY_GROUP_IDS_PATH,
)
# Networking is defined in sagemaker.train.configs and provides:
# - enable_network_isolation: Optional[bool]
# - subnets: Optional[List[str]]
# - security_group_ids: Optional[List[str]]
from sagemaker.train.configs import Networking

logger = logging.getLogger(__name__)


class Mode(Enum):
    """Enum for ModelTrainer execution modes."""

    LOCAL_CONTAINER = "LOCAL_CONTAINER"
    SAGEMAKER_TRAINING_JOB = "SAGEMAKER_TRAINING_JOB"


class ModelTrainer:
    """Class for configuring and launching training jobs on SageMaker.

    This class handles the configuration and execution of training jobs,
    including resolution of networking settings from sagemaker_config.
    """

    def __init__(
        self,
        training_image: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        networking: Optional[Networking] = None,
        config_mgr: Optional[Any] = None,
    ):
        """Initialize ModelTrainer.

        Args:
            training_image: The training image URI.
            algorithm_name: The algorithm name.
            hyperparameters: Hyperparameters for the training job.
            networking: Networking configuration including VPC settings
                and network isolation.
            config_mgr: Configuration manager for resolving defaults
                from sagemaker_config. Expected to implement
                resolve_value_from_config(config_path) -> Optional[Any].
        """
        self.training_image = training_image
        self.algorithm_name = algorithm_name
        self.hyperparameters = hyperparameters
        self.networking = networking
        self.config_mgr = config_mgr

        # Populate intelligent defaults
        self._populate_intelligent_defaults_from_training_job_space()

    def _resolve_networking(self):
        """Resolve networking configuration from sagemaker_config.

        Checks sagemaker_config for VpcConfig when no explicit Networking is
        provided, or when Networking is provided without subnets/security_groups.
        This fixes two bugs:
        1) VpcConfig from sagemaker_config is ignored when Networking is not
           explicitly passed to ModelTrainer.
        2) When Networking(enable_network_isolation=True) is passed without
           subnets/security_groups, the VpcConfig from sagemaker_config should
           be merged but was not.
        """
        if self.config_mgr is None:
            return

        # Resolve enable_network_isolation from config
        config_enable_network_isolation = self.config_mgr.resolve_value_from_config(
            TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH
        )

        # Resolve subnets and security_group_ids from config
        config_subnets = self.config_mgr.resolve_value_from_config(
            TRAINING_JOB_SUBNETS_PATH
        )
        config_security_group_ids = self.config_mgr.resolve_value_from_config(
            TRAINING_JOB_SECURITY_GROUP_IDS_PATH
        )

        if self.networking is None:
            # When no Networking is provided, create one from sagemaker_config
            if (
                config_enable_network_isolation is not None
                or config_subnets is not None
                or config_security_group_ids is not None
            ):
                self.networking = Networking(
                    enable_network_isolation=(
                        config_enable_network_isolation
                        if config_enable_network_isolation is not None
                        else False
                    ),
                    subnets=config_subnets,
                    security_group_ids=config_security_group_ids,
                )
        else:
            # Merge VpcConfig from sagemaker_config when Networking
            # is provided without subnets/security_groups
            if self.networking.enable_network_isolation is None:
                if config_enable_network_isolation is not None:
                    self.networking.enable_network_isolation = (
                        config_enable_network_isolation
                    )

            if self.networking.subnets is None:
                if config_subnets is not None:
                    self.networking.subnets = config_subnets

            if self.networking.security_group_ids is None:
                if config_security_group_ids is not None:
                    self.networking.security_group_ids = config_security_group_ids

    def _populate_intelligent_defaults_from_training_job_space(self):
        """Populate intelligent defaults from the training job config space."""
        if self.config_mgr is None:
            return

        # Resolve networking from sagemaker_config
        self._resolve_networking()

    def train(self):
        """Launch the training job.

        Raises:
            NotImplementedError: This method must be implemented by subclasses
                or the full ModelTrainer implementation.
        """
        raise NotImplementedError(
            "train() is not yet implemented. This module currently provides "
            "only the networking resolution fix."
        )
