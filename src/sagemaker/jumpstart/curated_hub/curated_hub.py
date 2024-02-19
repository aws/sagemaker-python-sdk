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
from typing import Optional, Dict, Any

import boto3

from sagemaker.session import Session

from sagemaker.jumpstart.curated_hub.constants import DEFAULT_CLIENT_CONFIG


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    def __init__(self, hub_name: str, region: str, session: Optional[Session]):
        self.hub_name = hub_name
        self.region = region
        self.session = session
        self._s3_client = self._get_s3_client()
        self._sm_session = session or Session()

    def _get_s3_client(self) -> Any:
        """Returns an S3 client."""
        return boto3.client("s3", region_name=self._region, config=DEFAULT_CLIENT_CONFIG)

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
