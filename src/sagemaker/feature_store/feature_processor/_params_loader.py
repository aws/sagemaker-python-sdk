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
"""Contains classes for loading the 'params' argument for the UDF."""
from __future__ import absolute_import

from typing import Dict, Union

import attr

from sagemaker.feature_store.feature_processor._data_source import FeatureGroupDataSource
from sagemaker.feature_store.feature_processor._env import EnvironmentHelper
from sagemaker.feature_store.feature_processor._feature_processor_config import (
    FeatureProcessorConfig,
)


@attr.s
class SystemParamsLoader:
    """Provides the fields for the params['system'] namespace.

    These are the parameters that the feature_processor automatically loads from various SageMaker
    resources.
    """

    _SYSTEM_PARAMS_KEY = "system"
    _SYSTEM_PARAMS_INPUTS_KEY = "inputs"
    _SYSTEM_PARAMS_EXECUTION_KEY = "execution"

    environment_helper: EnvironmentHelper = attr.ib()

    def get_system_args(self, fp_config: FeatureProcessorConfig) -> Dict[str, Union[str, Dict]]:
        """Generates the system generated parameters for the feature_processor wrapped function.

        Args:
            fp_config (FeatureProcessorConfig): The configuration values for the
                feature_processor decorator.

        Returns:
            Dict[str, Union[str, Dict]]: The system parameters.
        """
        return {
            self._SYSTEM_PARAMS_KEY: {
                **self._get_input_details(fp_config),
                **self._get_execution_details(),
            }
        }

    def _get_input_details(self, fp_config: FeatureProcessorConfig) -> Dict[str, Dict]:
        """Load additional details for FeatureGroupDataSources.

        Args:
            fp_config (FeatureProcessorConfig): The configuration values for the
                feature_processor decorator.

        Returns:
            Dict[str, Dict]: The details of the FeatureGroupDataSources.
        """

        def get_input_details():
            # TODO: Populate the fields.
            return {}

        feature_group_data_sources = [
            ds for ds in fp_config.inputs if isinstance(ds, FeatureGroupDataSource)
        ]

        return {
            self._SYSTEM_PARAMS_INPUTS_KEY: dict(
                zip(
                    [ds.name for ds in feature_group_data_sources],
                    [get_input_details() for ds in feature_group_data_sources],
                )
            )
        }

    def _get_execution_details(self) -> Dict[str, Union[str, Dict]]:
        """Loads additional details about the execution environment and pipeline."""
        # TODO: Populate the fields.
        execution_details: Dict[str, Union[str, Dict]] = {
            self._SYSTEM_PARAMS_EXECUTION_KEY: {
                "training_job_arn": "",
                "pipeline_arn": "",
                "pipeline_execution_arn": "",
                "scheduled_start_time": "",
            }
        }

        return execution_details


@attr.s
class ParamsLoader:
    """Provides 'params' argument for the FeatureProcessor."""

    _PARAMS_KEY = "params"

    system_parameters_arg_provider: SystemParamsLoader = attr.ib()

    def get_parameter_args(
        self,
        fp_config: FeatureProcessorConfig,
    ) -> Dict[str, Union[str, Dict]]:
        """Loads the 'params' argument for the FeatureProcessor.

        Args:
            fp_config (FeatureProcessorConfig): The configuration values for the
                feature_processor decorator.

        Returns:
            Dict[str, Union[str, Dict]]: A dictionary containin both user provided
                parameters (feature_processor argument) and system parameters.
        """
        return {
            self._PARAMS_KEY: {
                **(fp_config.parameters or {}),
                **self.system_parameters_arg_provider.get_system_args(fp_config),
            }
        }
