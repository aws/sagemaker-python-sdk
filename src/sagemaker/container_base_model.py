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
# ANY KIND, either express oXr implied. See the License for the specific
# language governing permissions and limitations under the License.
"""This file contains code related to base model for containers."""
from __future__ import absolute_import

from typing import Optional, Union

from sagemaker.workflow.entities import PipelineVariable


class ContainerBaseModel(object):
    """Accepts Base Model parameters for conversion to request dict."""

    def __init__(
        self,
        hub_content_name: Union[str, PipelineVariable] = None,
        hub_content_version: Optional[Union[str, PipelineVariable]] = None,
        recipe_name: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Initialize a ``ContainerBaseModel`` instance and turn parameters into dict.

        Args:
            hub_content_name (str or PipelineVariable): The hub content name
            hub_content_version (str or PipelineVariable): The hub content version
                (default: None)
            recipe_name (str or PipelineVariable): The Recipe name
               (default: None)
        """
        self.hub_content_name = hub_content_name
        self.hub_content_version = hub_content_version
        self.recipe_name = recipe_name

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided to the class."""
        base_model_request = {}
        if self.hub_content_name is not None:
            base_model_request["HubContentName"] = self.hub_content_name
        if self.hub_content_version is not None:
            base_model_request["HubContentVersion"] = self.hub_content_version
        if self.recipe_name is not None:
            base_model_request["RecipeName"] = self.recipe_name
        return base_model_request
