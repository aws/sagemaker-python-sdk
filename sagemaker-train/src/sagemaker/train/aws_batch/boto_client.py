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
"""The file provides helper function for getting Batch boto client."""
from __future__ import absolute_import

from typing import Optional
import boto3


def get_batch_boto_client(
    region: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> boto3.session.Session.client:
    """Helper function for getting Batch boto3 client.

    Args:
        region: Region specified
        endpoint: Batch API endpoint.

    Returns: Batch boto3 client.

    """
    return boto3.client("batch", region_name=region, endpoint_url=endpoint)
