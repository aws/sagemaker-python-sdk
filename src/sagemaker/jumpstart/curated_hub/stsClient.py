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
"""This module contains an STS client helper for the Curated Hub."""
from __future__ import absolute_import

from typing import Any
from datetime import datetime

import boto3


def convert_iso_to_yyyymmdd_hhmm(iso_time: str) -> str:
    """Convert iso time string (generated from assign_timestamp) to 'YYYYMMDD-HHMM'-formatted time."""
    return datetime.fromisoformat(iso_time.rstrip("Z")).strftime("%Y%m%d-%H%M")


def assign_timestamp() -> str:
    """Return the current UTC timestamp in ISO Format."""
    return datetime.utcnow().isoformat() + "Z"


class StsClient:
    """Boto3 client to access STS."""

    def __init__(self, region: str = None) -> None:
        """Creates the boto3 client for STS."""
        self._client = boto3.client(service_name="sts", region_name=region)

    def get_region(self) -> str:
        """Return the AWS region from the client meta information."""
        return self._client.meta.region_name

    def get_account_id(self) -> str: 
        """Returns the AWS account id associated with the caller identity."""
        identity = self._client.get_caller_identity()
        return identity["Account"]

    def get_boto3_session_from_role_arn(
        self, role_arn: str, **assume_role_kwargs: Any
    ) -> boto3.Session:
        """Return boto3 session using sts.assume_role.

        kwarg arguments are passed to `assume_role` boto3 call.
        See: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/sts.html#STS.Client.assume_role
        """

        kwargs = {
            "RoleArn": role_arn,
            "RoleSessionName": "JumpStartModelHub-"
            + convert_iso_to_yyyymmdd_hhmm(assign_timestamp()),
        }
        kwargs.update(assume_role_kwargs)

        assumed_role_object = self._client.assume_role(**kwargs)

        credentials = assumed_role_object["Credentials"]

        return boto3.Session(
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )
