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
"""This module stores types related to SageMaker JumpStart."""
from __future__ import absolute_import
from typing import Dict, Any

from sagemaker.jumpstart.types import JumpStartDataHolderType


class HubContentDocument_v2(JumpStartDataHolderType):
    """Data class for HubContentDocument v2.0.0"""

    SCHEMA_VERSION = "2.0.0"

    def __init__(self, spec: Dict[str, Any]):
        """Initializes a HubContentDocument_v2 object from JumpStart model specs.

        Args:
            spec (Dict[str, Any]): Dictionary representation of spec.
        """
        self.from_json(spec)

    def from_json(self, json_obj: Dict[str, Any]) -> None:
        """Sets fields in object based on json.

        Args:
            json_obj (Dict[str, Any]): Dictionary representatino of spec.
        """
        # TODO: Implement
        self.Url: str = json_obj["url"]
