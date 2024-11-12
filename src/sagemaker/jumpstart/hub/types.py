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
"""This module stores types related to SageMaker JumpStart Hub."""
from __future__ import absolute_import
from typing import Dict
from dataclasses import dataclass


@dataclass
class S3ObjectLocation:
    """Helper class for S3 object references."""

    bucket: str
    key: str

    def format_for_s3_copy(self) -> Dict[str, str]:
        """Returns a dict formatted for S3 copy calls"""
        return {
            "Bucket": self.bucket,
            "Key": self.key,
        }

    def get_uri(self) -> str:
        """Returns the s3 URI"""
        return f"s3://{self.bucket}/{self.key}"
