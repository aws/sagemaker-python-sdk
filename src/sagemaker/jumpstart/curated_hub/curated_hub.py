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
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION

from sagemaker.session import Session


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    def __init__(self, hub_name: str, region: str, session: Optional[Session] = DEFAULT_JUMPSTART_SAGEMAKER_SESSION):
        self.hub_name = hub_name
        self.region = region
        self.session = session
        self._sm_session = session

    def describe_model(self, model_name: str, model_version: str = "*") -> Dict[str, Any]:
        """Returns descriptive information about the Hub Model"""

        hub_content = self._sm_session.describe_hub_content(
            model_name, "Model", self.hub_name, model_version
        )

        # TODO: Parse HubContent
        # TODO: Parse HubContentDocument

        return hub_content

    def describe(self) -> Dict[str, Any]:
        """Returns descriptive information about the Hub"""

        hub_info = self._sm_session.describe_hub(hub_name=self.hub_name)

        # TODO: Validations?

        return hub_info
