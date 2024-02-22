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

from typing import Optional, Dict, Any
import boto3
from sagemaker.session import Session
from sagemaker.jumpstart.constants import (
    JUMPSTART_DEFAULT_REGION_NAME,
)

from sagemaker.jumpstart.types import HubDataType
import sagemaker.jumpstart.curated_hub.utils as hubutils


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    def __init__(
        self,
        name: str,
        region: str = JUMPSTART_DEFAULT_REGION_NAME,
        session: Optional[Session] = None,
    ):
        self.name = name
        if session.boto_region_name != region:
            # TODO: Handle error
            pass
        self.region = region
        self._session = session or Session(boto3.Session(region_name=region))

    def create(
        self,
        description: str,
        display_name: Optional[str] = None,
        search_keywords: Optional[str] = None,
        bucket_name: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Dict[str, str]:
        """Creates a hub with the given description"""

        return hubutils.create_hub(
            hub_name=self.name,
            hub_description=description,
            hub_display_name=display_name,
            hub_search_keywords=search_keywords,
            hub_bucket_name=bucket_name,
            tags=tags,
            sagemaker_session=self._session,
        )

    def describe_model(self, model_name: str, model_version: str = "*") -> Dict[str, Any]:
        """Returns descriptive information about the Hub Model"""

        hub_content = hubutils.describe_hub_content(
            hub_name=self.name,
            content_name=model_name,
            content_type=HubDataType.MODEL,
            content_version=model_version,
            sagemaker_session=self._session,
        )

        return hub_content

    def describe(self) -> Dict[str, Any]:
        """Returns descriptive information about the Hub"""

        hub_info = hubutils.describe_hub(hub_name=self.name, sagemaker_session=self._session)

        return hub_info
