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

"""Utility functions for AI Registry Hub."""

import boto3

from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_constants import (
    RESPONSE_KEY_HUB_CONTENT_VERSION,
    AIR_HUB_CONTENT_DEFAULT_VERSION
)


def _determine_new_version(hub_content_type: str, hub_content_name: str, session=None) -> str:
    """Determine new version for hub content.
    
    Args:
        hub_content_type: Type of hub content
        hub_content_name: Name of hub content
        session: Optional SageMaker session
        
    Returns:
        New version string (e.g., "2.0.0" if current is "1.0.0", or default if doesn't exist)
    """
    try:
        response = AIRHub.describe_hub_content(
            hub_content_type=hub_content_type,
            hub_content_name=hub_content_name,
            session=session
        )
        current_version = response[RESPONSE_KEY_HUB_CONTENT_VERSION]
        major_version = int(current_version.split('.')[0]) + 1
        return f"{major_version}.0.0"
    except Exception:
        return AIR_HUB_CONTENT_DEFAULT_VERSION


def _get_default_bucket() -> str:
    """Get default S3 bucket name in format sagemaker-{region}-{account_id}."""
    sts_client = boto3.client("sts")
    account_id = sts_client.get_caller_identity()['Account']
    region = boto3.session.Session().region_name
    return f"sagemaker-{region}-{account_id}"
