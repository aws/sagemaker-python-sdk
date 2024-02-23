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
"""Utilities to interact with Hub."""
from typing import Any, Dict
import boto3
from sagemaker.session import Session
from sagemaker.jumpstart.types import HubDescription, HubContentType, HubContentDescription


def describe(hub_name: str, region: str) -> HubDescription:
    """Returns descriptive information about the Hub."""

    sagemaker_session = Session(boto3.Session(region_name=region))
    hub_description = sagemaker_session.describe_hub(hub_name=hub_name)
    return HubDescription(hub_description)


def describe_model(
    hub_name: str, region: str, model_name: str, model_version: str = "*"
) -> HubContentDescription:
    """Returns descriptive information about the Hub model."""

    sagemaker_session = Session(boto3.Session(region_name=region))
    hub_content_description: Dict[str, Any] = sagemaker_session.describe_hub_content(
        hub_name=hub_name,
        hub_content_name=model_name,
        hub_content_version=model_version,
        hub_content_type=HubContentType.MODEL,
    )

    return HubContentDescription(hub_content_description)
