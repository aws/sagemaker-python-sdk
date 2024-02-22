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
"""Mid-level wrappers to HubService API. These utilities handles parsing, custom
errors, and validations on top of the low-level HubService API calls in Session."""
from __future__ import absolute_import
from typing import Optional, Dict, Any, List

from sagemaker.jumpstart.types import HubDataType
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
)
from sagemaker.session import Session


# def _validate_hub_name(hub_name: str) -> bool:
#     """Validates hub_name to be either a name or a full ARN"""
#     pass


def _generate_default_hub_bucket_name(
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Return the name of the default bucket to use in relevant Amazon SageMaker interactions.

    This function will create the s3 bucket if it does not exist.

    Returns:
        str: The name of the default bucket. If the name was not explicitly specified through
            the Session or sagemaker_config, the bucket will take the form:
            ``sagemaker-hubs-{region}-{AWS account ID}``.
    """

    region: str = sagemaker_session.boto_region_name
    account_id: str = sagemaker_session.account_id()

    # TODO: Validate and fast fail

    return f"sagemaker-hubs-{region}-{account_id}"


def create_hub(
    hub_name: str,
    hub_description: str,
    hub_display_name: str = None,
    hub_search_keywords: Optional[List[str]] = None,
    hub_bucket_name: Optional[str] = None,
    tags: Optional[List[Dict[str, Any]]] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Creates a SageMaker Hub

    Returns:
        (str): Arn of the created hub.
    """

    if hub_bucket_name is None:
        hub_bucket_name = _generate_default_hub_bucket_name(sagemaker_session)
    s3_storage_config = {"S3OutputPath": hub_bucket_name}
    response = sagemaker_session.create_hub(
        hub_name, hub_description, hub_display_name, hub_search_keywords, s3_storage_config, tags
    )

    # TODO: Custom error message

    hub_arn = response["HubArn"]
    return hub_arn


def describe_hub(
    hub_name: str, sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION
) -> Dict[str, Any]:
    """Returns descriptive information about the Hub"""
    # TODO: hub_name validation and fast-fail

    response = sagemaker_session.describe_hub(hub_name=hub_name)

    # TODO: Make HubInfo and parse response?
    # TODO: Custom error message

    return response


def delete_hub(hub_name, sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION) -> None:
    """Deletes a SageMaker Hub"""
    response = sagemaker_session.delete_hub(hub_name=hub_name)

    # TODO: Custom error message

    return response


def import_hub_content(
    document_schema_version: str,
    hub_name: str,
    hub_content_name: str,
    hub_content_type: str,
    hub_content_document: str,
    hub_content_display_name: str = None,
    hub_content_description: str = None,
    hub_content_version: str = None,
    hub_content_markdown: str = None,
    hub_content_search_keywords: List[str] = None,
    tags: List[Dict[str, Any]] = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Dict[str, str]:
    """Imports a new HubContent into a SageMaker Hub

    Returns arns for the Hub and the HubContent where import was successful.
    """

    response = sagemaker_session.import_hub_content(
        document_schema_version,
        hub_name,
        hub_content_name,
        hub_content_type,
        hub_content_document,
        hub_content_display_name,
        hub_content_description,
        hub_content_version,
        hub_content_markdown,
        hub_content_search_keywords,
        tags,
    )
    return response


def list_hub_contents(
    hub_name: str,
    hub_content_type: HubDataType.MODEL or HubDataType.NOTEBOOK,
    creation_time_after: str = None,
    creation_time_before: str = None,
    max_results: int = None,
    max_schema_version: str = None,
    name_contains: str = None,
    next_token: str = None,
    sort_by: str = None,
    sort_order: str = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Dict[str, Any]:
    """List contents of a hub."""

    response = sagemaker_session.list_hub_contents(
        hub_name,
        hub_content_type,
        creation_time_after,
        creation_time_before,
        max_results,
        max_schema_version,
        name_contains,
        next_token,
        sort_by,
        sort_order,
    )
    return response


def describe_hub_content(
    hub_name: str,
    content_name: str,
    content_type: HubDataType,
    content_version: str = None,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> Dict[str, Any]:
    """Returns descriptive information about the content of a hub."""
    # TODO: hub_name validation and fast-fail

    hub_content: Dict[str, Any] = sagemaker_session.describe_hub_content(
        hub_content_name=content_name,
        hub_content_type=content_type,
        hub_name=hub_name,
        hub_content_version=content_version,
    )

    # TODO: Parse HubContent
    # TODO: Parse HubContentDocument

    return hub_content


def delete_hub_content(
    hub_content_name: str,
    hub_content_version: str,
    hub_content_type: str,
    hub_name: str,
    sagemaker_session: Session = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> None:
    """Deletes a given HubContent in a SageMaker Hub"""
    # TODO: Validate hub name

    response = sagemaker_session.delete_hub_content(
        hub_content_name, hub_content_version, hub_content_type, hub_name
    )
    return response
