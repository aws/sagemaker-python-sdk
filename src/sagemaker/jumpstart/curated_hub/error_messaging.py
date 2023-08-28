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
"""This module provides error message handling for the JumpStart Curated Hub."""
from __future__ import absolute_import

from typing import List

RESOURCE_NOT_FOUND_ERROR_CODE = "ResourceNotFound"
NO_SUCH_BUCKET_ERROR_CODE = "404"
ACCESS_DENIED_ERROR_CODE = "AccessDenied"


def get_hub_limit_exceeded_error(region: str, hubs_on_account: List[str]) -> ValueError:
    """Returns an error if the Hubs on an account exceed the account limit."""
    return ValueError(
        f"You have reached the limit of hubs on the account for {region}. "
        f"The current hubs you have are {hubs_on_account}. "
        "You can delete one of the hubs to free up space or change "
        "curated_hub_name to one of the above preexisting hubs."
    )


def get_hub_s3_bucket_permissions_error(hub_s3_bucket_name: str) -> PermissionError:
    """Returns an error that outlines necessary S3 permissions for the Curated Hub."""
    return PermissionError(
        f"You do not have permissions to the hub bucket {hub_s3_bucket_name}. "
        "Please add [s3:CreateBucket, s3:ListBucket, s3:GetObject*, "
        "s3:PutObject*, s3:DeleteObject*] permissions to our IAM role."
    )


def get_hub_creation_error_message(s3_bucket_name: str) -> str:
    """The Curated Hub creation creates a S3 bucket along with the Hub
    
    If the Hub creation fails but the S3 bucket succeeded, the S3 bucket
    will need to be manually deleted.
    """
    return (
        "ERROR: Exception occurred during hub Curated Hub Creation. "
        f"A S3 bucket {s3_bucket_name} has been created and must be manually deleted."
    )


def get_preexisting_hub_should_be_true_error(hub_name: str, region: str) -> ValueError:
    """Returns error for when a preexisting hub does exist but was expected not to."""
    return ValueError(
        f"Hub detected on account with name {hub_name} in {region}. "
        f"If you wish to use the hub as your Curated Hub, "
        "please pass in use_preexisting_hub=True"
    )


def get_preexisting_hub_should_be_false_error(hub_name: str, region: str) -> ValueError:
    """Returns error for when a preexisting hub does not exist but was expected to."""
    return ValueError(
        f"Attempted to use a preexisting hub but no hub with name {hub_name} "
        f"exists for this account in {region}. If you wish to create a new Curated Hub, "
        "please pass in use_preexisting_hub=False"
    )
