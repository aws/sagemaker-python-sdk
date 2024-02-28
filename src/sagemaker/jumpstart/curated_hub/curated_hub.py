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
"""This module provides the JumpStart Curated Hub class."""
from __future__ import absolute_import


from typing import Any, Dict, Optional
from sagemaker.session import Session
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.types import (
    DescribeHubResponse,
    DescribeHubContentsResponse,
    HubContentType,
)
from sagemaker.jumpstart.curated_hub.utils import create_hub_bucket_if_it_does_not_exist


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    def __init__(
        self,
        hub_name: str,
        sagemaker_session: Optional[Session] = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ):
        """Instantiates a SageMaker ``CuratedHub``.

        Args:
            hub_name (str): The name of the Hub to create.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.
        """
        self.hub_name = hub_name
        self.region = sagemaker_session.boto_region_name
        self._sagemaker_session = sagemaker_session

    def create(
        self,
        description: str,
        display_name: Optional[str] = None,
        search_keywords: Optional[str] = None,
        bucket_name: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Dict[str, str]:
        """Creates a hub with the given description"""

        bucket_name = create_hub_bucket_if_it_does_not_exist(bucket_name, self._sagemaker_session)

        return self._sagemaker_session.create_hub(
            hub_name=self.hub_name,
            hub_description=description,
            hub_display_name=display_name,
            hub_search_keywords=search_keywords,
            hub_bucket_name=bucket_name,
            tags=tags,
        )

    def describe(self) -> DescribeHubResponse:
        """Returns descriptive information about the Hub"""

        hub_description: DescribeHubResponse = self._sagemaker_session.describe_hub(
            hub_name=self.hub_name
        )

        return hub_description

    def list_models(self, **kwargs) -> Dict[str, Any]:
        """Lists the models in this Curated Hub

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        # TODO: Validate kwargs and fast-fail?

        hub_content_summaries = self._sagemaker_session.list_hub_contents(
            hub_name=self.hub_name, hub_content_type=HubContentType.MODEL, **kwargs
        )
        # TODO: Handle pagination
        return hub_content_summaries

    def describe_model(
        self, model_name: str, model_version: str = "*"
    ) -> DescribeHubContentsResponse:
        """Returns descriptive information about the Hub Model"""

        hub_content_description: Dict[str, Any] = self._sagemaker_session.describe_hub_content(
            hub_name=self.hub_name,
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL,
        )

        return DescribeHubContentsResponse(hub_content_description)

    def delete_model(self, model_name: str, model_version: str = "*") -> None:
        """Deletes a model from this CuratedHub."""
        return self._sagemaker_session.delete_hub_content(
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL,
            hub_name=self.hub_name,
        )

    def delete(self) -> None:
        """Deletes this Curated Hub"""
        return self._sagemaker_session.delete_hub(self.hub_name)
