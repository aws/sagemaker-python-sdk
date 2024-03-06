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
"""This module utilites to assist S3 client calls for the Curated Hub."""
from __future__ import absolute_import
from dataclasses import dataclass
from typing import Dict

from sagemaker.s3_utils import parse_s3_url


@dataclass
class S3ObjectLocation:
    """Helper class for S3 object references"""

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


def create_s3_object_reference_from_uri(s3_uri: str) -> S3ObjectLocation:
    """Utiity to help generate an S3 object reference"""
    bucket, key = parse_s3_url(s3_uri)

    return S3ObjectLocation(
        bucket=bucket,
        key=key,
    )
