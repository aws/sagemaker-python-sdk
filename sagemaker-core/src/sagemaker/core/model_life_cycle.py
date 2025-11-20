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
"""This file contains code related to model life cycle."""
from __future__ import absolute_import

from typing import Optional, Union

from sagemaker.core.helper.pipeline_variable import PipelineVariable


class ModelLifeCycle(object):
    """Accepts ModelLifeCycle parameters for conversion to request dict."""

    def __init__(
        self,
        stage: Optional[Union[str, PipelineVariable]] = None,
        stage_status: Optional[Union[str, PipelineVariable]] = None,
        stage_description: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``ModelLifeCycle`` instance and turn parameters into dict.

        # TODO: flesh out docstrings
        Args:
            stage (str or PipelineVariable):
            stage_status (str or PipelineVariable):
            stage_description (str or PipelineVariable):
        """
        self.stage = stage
        self.stage_status = stage_status
        self.stage_description = stage_description

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        model_life_cycle_request = dict()
        if self.stage:
            model_life_cycle_request["Stage"] = self.stage
        if self.stage_status:
            model_life_cycle_request["StageStatus"] = self.stage_status
        if self.stage_description:
            model_life_cycle_request["StageDescription"] = self.stage_description
        return model_life_cycle_request
