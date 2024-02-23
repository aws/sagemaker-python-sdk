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
"""This module stores types related to SageMaker JumpStart CuratedHub."""
from __future__ import absolute_import
from typing import Optional

from sagemaker.jumpstart.types import JumpStartDataHolderType

class HubArnExtractedInfo(JumpStartDataHolderType):
    """Data class for info extracted from Hub arn."""

    __slots__ = [
        "partition",
        "region",
        "account_id",
        "hub_name",
        "hub_content_type",
        "hub_content_name",
        "hub_content_version",
    ]

    def __init__(
        self,
        partition: str,
        region: str,
        account_id: str,
        hub_name: str,
        hub_content_type: Optional[str] = None,
        hub_content_name: Optional[str] = None,
        hub_content_version: Optional[str] = None,
    ) -> None:
        """Instantiates HubArnExtractedInfo object."""

        self.partition = partition
        self.region = region
        self.account_id = account_id
        self.hub_name = hub_name
        self.hub_content_type = hub_content_type
        self.hub_content_name = hub_content_name
        self.hub_content_version = hub_content_version
