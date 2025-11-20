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
# pylint: skip-file
"""This module contains utilities related to SageMaker JumpStart Hub."""
from __future__ import absolute_import
import re
from typing import Optional, List, Any
from sagemaker.core.helper.session_helper import Session
from sagemaker.core.common_utils import aws_partition
from sagemaker.core.jumpstart.types import HubContentType, HubArnExtractedInfo
from sagemaker.core.jumpstart import constants
from packaging.specifiers import SpecifierSet, InvalidSpecifier
from packaging import version

PROPRIETARY_VERSION_KEYWORD = "@marketplace-version:"


def _convert_str_to_optional(string: str) -> Optional[str]:
    if string == "None":
        string = None
    return string


def get_info_from_hub_resource_arn(
    arn: str,
) -> HubArnExtractedInfo:
    """Extracts descriptive information from a Hub or HubContent Arn."""

    match = re.match(constants.HUB_CONTENT_ARN_REGEX, arn)
    if match:
        partition = match.group(1)
        hub_region = match.group(2)
        account_id = match.group(3)
        hub_name = match.group(4)
        hub_content_type = match.group(5)
        hub_content_name = match.group(6)
        hub_content_version = _convert_str_to_optional(match.group(7))

        return HubArnExtractedInfo(
            partition=partition,
            region=hub_region,
            account_id=account_id,
            hub_name=hub_name,
            hub_content_type=hub_content_type,
            hub_content_name=hub_content_name,
            hub_content_version=hub_content_version,
        )

    match = re.match(constants.HUB_ARN_REGEX, arn)
    if match:
        partition = match.group(1)
        hub_region = match.group(2)
        account_id = match.group(3)
        hub_name = match.group(4)
        return HubArnExtractedInfo(
            partition=partition,
            region=hub_region,
            account_id=account_id,
            hub_name=hub_name,
        )


def construct_hub_arn_from_name(
    hub_name: str,
    region: Optional[str] = None,
    session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    account_id: Optional[str] = None,
) -> str:
    """Constructs a Hub arn from the Hub name using default Session values."""
    if session is None:
        # session is overridden to none by some callers
        session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION

    account_id = account_id or session.account_id()
    region = region or session.boto_region_name
    partition = aws_partition(region)

    return f"arn:{partition}:sagemaker:{region}:{account_id}:hub/{hub_name}"


def construct_hub_model_arn_from_inputs(hub_arn: str, model_name: str, version: str) -> str:
    """Constructs a HubContent model arn from the Hub name, model name, and model version."""

    info = get_info_from_hub_resource_arn(hub_arn)
    arn = (
        f"arn:{info.partition}:sagemaker:{info.region}:{info.account_id}:hub-content/"
        f"{info.hub_name}/{HubContentType.MODEL.value}/{model_name}/{version}"
    )

    return arn


def construct_hub_model_reference_arn_from_inputs(
    hub_arn: str, model_name: str, version: str
) -> str:
    """Constructs a HubContent model arn from the Hub name, model name, and model version."""

    info = get_info_from_hub_resource_arn(hub_arn)
    arn = (
        f"arn:{info.partition}:sagemaker:{info.region}:{info.account_id}:hub-content/"
        f"{info.hub_name}/{HubContentType.MODEL_REFERENCE.value}/{model_name}/{version}"
    )

    return arn


def generate_hub_arn_for_init_kwargs(
    hub_name: str, region: Optional[str] = None, session: Optional[Session] = None
):
    """Generates the Hub Arn for JumpStart class args from a HubName or Arn.

    Args:
        hub_name (str): HubName or HubArn from JumpStart class args
        region (str): Region from JumpStart class args
        session (Session): Custom SageMaker Session from JumpStart class args
    """

    hub_arn = None
    if hub_name:
        if hub_name == constants.JUMPSTART_MODEL_HUB_NAME:
            return None
        match = re.match(constants.HUB_ARN_REGEX, hub_name)
        if match:
            hub_arn = hub_name
        else:
            hub_arn = construct_hub_arn_from_name(hub_name=hub_name, region=region, session=session)
    return hub_arn


def is_gated_bucket(bucket_name: str) -> bool:
    """Returns true if the bucket name is the JumpStart gated bucket."""
    return bucket_name in constants.JUMPSTART_GATED_BUCKET_NAME_SET


def get_hub_model_version(
    hub_name: str,
    hub_model_name: str,
    hub_model_type: str,
    hub_model_version: Optional[str] = None,
    sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Returns available Jumpstart hub model version.

    It will attempt both a semantic HubContent version search and Marketplace version search.
    If the Marketplace version is also semantic, this function will default to HubContent version.

    Raises:
        ClientError: If the specified model is not found in the hub.
        KeyError: If the specified model version is not found.
    """
    if sagemaker_session is None:
        # sagemaker_session is overridden to none by some callers
        sagemaker_session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION

    try:
        hub_content_summaries = _list_hub_content_versions_helper(
            hub_name=hub_name,
            hub_content_name=hub_model_name,
            hub_content_type=hub_model_type,
            sagemaker_session=sagemaker_session,
        )
    except Exception as ex:
        raise Exception(f"Failed calling list_hub_content_versions: {str(ex)}")

    try:
        return _get_hub_model_version_for_open_weight_version(
            hub_content_summaries, hub_model_version
        )
    except KeyError:
        marketplace_hub_content_version = _get_hub_model_version_for_marketplace_version(
            hub_content_summaries, hub_model_version
        )
        if marketplace_hub_content_version:
            return marketplace_hub_content_version
        raise


def _list_hub_content_versions_helper(
    hub_name, hub_content_name, hub_content_type, sagemaker_session
):
    all_hub_content_summaries = []
    list_hub_content_versions_response = sagemaker_session.list_hub_content_versions(
        hub_name=hub_name, hub_content_name=hub_content_name, hub_content_type=hub_content_type
    )
    all_hub_content_summaries.extend(list_hub_content_versions_response.get("HubContentSummaries"))
    while "NextToken" in list_hub_content_versions_response:
        list_hub_content_versions_response = sagemaker_session.list_hub_content_versions(
            hub_name=hub_name,
            hub_content_name=hub_content_name,
            hub_content_type=hub_content_type,
            next_token=list_hub_content_versions_response["NextToken"],
        )
        all_hub_content_summaries.extend(
            list_hub_content_versions_response.get("HubContentSummaries")
        )
    return all_hub_content_summaries


def _get_hub_model_version_for_open_weight_version(
    hub_content_summaries: List[Any], hub_model_version: Optional[str] = None
) -> str:
    available_model_versions = [model.get("HubContentVersion") for model in hub_content_summaries]

    if hub_model_version == "*" or hub_model_version is None:
        return str(max(version.parse(v) for v in available_model_versions))

    try:
        spec = SpecifierSet(f"=={hub_model_version}")
    except InvalidSpecifier:
        raise KeyError(f"Bad semantic version: {hub_model_version}")
    available_versions_filtered = list(spec.filter(available_model_versions))
    if not available_versions_filtered:
        raise KeyError("Model version not available in the Hub")
    hub_model_version = str(max(available_versions_filtered))

    return hub_model_version


def _get_hub_model_version_for_marketplace_version(
    hub_content_summaries: List[Any], marketplace_version: str
) -> Optional[str]:
    """Returns the HubContent version associated with the Marketplace version.

    This function will check within the HubContentSearchKeywords for the proprietary version.
    """
    for model in hub_content_summaries:
        model_search_keywords = model.get("HubContentSearchKeywords", [])
        if _hub_search_keywords_contains_marketplace_version(
            model_search_keywords, marketplace_version
        ):
            return model.get("HubContentVersion")

    return None


def _hub_search_keywords_contains_marketplace_version(
    model_search_keywords: List[str], marketplace_version: str
) -> bool:
    proprietary_version_keyword = next(
        filter(lambda s: s.startswith(PROPRIETARY_VERSION_KEYWORD), model_search_keywords), None
    )

    if not proprietary_version_keyword:
        return False

    proprietary_version = proprietary_version_keyword.lstrip(PROPRIETARY_VERSION_KEYWORD)
    if proprietary_version == marketplace_version:
        return True

    return False
