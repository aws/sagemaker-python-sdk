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
"""This module contains utilities related to SageMaker JumpStart CuratedHub."""
from __future__ import absolute_import
import re
from typing import Optional, Dict, List, Any
from sagemaker.jumpstart.curated_hub.types import (
    FileInfo,
    HubContentReferenceType,
    S3ObjectLocation,
)
from sagemaker.s3_utils import parse_s3_url
from sagemaker.session import Session
from sagemaker.utils import aws_partition
from sagemaker.jumpstart.types import HubContentType, HubArnExtractedInfo
from sagemaker.jumpstart.curated_hub.types import (
    CuratedHubUnsupportedFlag,
    HubContentInfo,
    JumpStartModelInfo,
    summary_list_from_list_api_response,
)
from sagemaker.jumpstart import constants
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.curated_hub.constants import (
    JUMPSTART_CURATED_HUB_MODEL_TAG,
)
from sagemaker.utils import format_tags, TagsDict


def get_info_from_hub_resource_arn(
    arn: str,
) -> HubArnExtractedInfo:
    """Extracts descriptive information from a Hub or HubContent Arn."""

    match = re.match(constants.HUB_MODEL_ARN_REGEX, arn)
    if match:
        partition = match.group(1)
        hub_region = match.group(2)
        account_id = match.group(3)
        hub_name = match.group(4)
        hub_content_name = match.group(5)
        hub_content_version = match.group(6)

        return HubArnExtractedInfo(
            partition=partition,
            region=hub_region,
            account_id=account_id,
            hub_name=hub_name,
            hub_content_type=HubContentType.MODEL.value,
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

    return None


def construct_hub_arn_from_name(
    hub_name: str,
    region: Optional[str] = None,
    session: Optional[Session] = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Constructs a Hub arn from the Hub name using default Session values."""

    account_id = session.account_id()
    region = region or session.boto_region_name
    partition = aws_partition(region)

    return f"arn:{partition}:sagemaker:{region}:{account_id}:hub/{hub_name}"


def construct_hub_model_arn_from_inputs(hub_arn: str, model_name: str, version: str) -> str:
    """Constructs a HubContent model arn from the Hub name, model name, and model version."""

    info = get_info_from_hub_resource_arn(hub_arn)
    arn = (
        f"arn:{info.partition}:sagemaker:{info.region}:{info.account_id}:hub-content/"
        f"{info.hub_name}/{HubContentType.MODEL}/{model_name}/{version}"
    )

    return arn


# TODO: Update to recognize JumpStartHub hub_name
def generate_hub_arn_for_init_kwargs(
    hub_name: str, region: Optional[str] = None, session: Optional[Session] = None
):
    """Generates the Hub Arn for JumpStart class args from a HubName or Arn.

    Args:
        hub_name (str): HubName or HubArn from JumpStart class args
        region (str): Region from JumpStart class args
        session (Session): Custom SageMaker Session from JumpStart class args
    """

    if session is None:
        session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION

    hub_arn = None
    if hub_name:
        match = re.match(constants.HUB_ARN_REGEX, hub_name)
        if match:
            hub_arn = hub_name
        else:
            hub_arn = construct_hub_arn_from_name(hub_name=hub_name, region=region, session=session)
    return hub_arn


def generate_default_hub_bucket_name(
    sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Return the name of the default bucket to use in relevant Amazon SageMaker Hub interactions.

    Returns:
        str: The name of the default bucket. If the name was not explicitly specified through
            the Session or sagemaker_config, the bucket will take the form:
            ``sagemaker-hubs-{region}-{AWS account ID}``.
    """

    region: str = sagemaker_session.boto_region_name
    account_id: str = sagemaker_session.account_id()

    # TODO: Validate and fast fail

    return f"sagemaker-hubs-{region}-{account_id}"


def create_s3_object_reference_from_uri(s3_uri: Optional[str]) -> Optional[S3ObjectLocation]:
    """Utiity to help generate an S3 object reference"""
    if not s3_uri:
        return None

    bucket, key = parse_s3_url(s3_uri)

    return S3ObjectLocation(
        bucket=bucket,
        key=key,
    )


def create_hub_bucket_if_it_does_not_exist(
    bucket_name: Optional[str] = None,
    sagemaker_session: Session = constants.DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
) -> str:
    """Creates the default SageMaker Hub bucket if it does not exist.

    Returns:
        str: The name of the default bucket. Takes the form:
            ``sagemaker-hubs-{region}-{AWS account ID}``.
    """

    region: str = sagemaker_session.boto_region_name
    if bucket_name is None:
        bucket_name: str = generate_default_hub_bucket_name(sagemaker_session)

    sagemaker_session._create_s3_bucket_if_it_does_not_exist(
        bucket_name=bucket_name,
        region=region,
    )

    return bucket_name


def find_deprecated_vulnerable_flags_for_hub_content(
    hub_name: str, hub_content_name: str, region: str, session: Session
) -> List[TagsDict]:
    """Finds the JumpStart public hub model for a HubContent and calculates relevant tags.

    Since tags are the same for all versions of a HubContent,
    these tags will map from the key to a list of versions impacted.
    For example, if certain public hub model versions are deprecated,
    this utility will return a `deprecated` tag
    mapped to the deprecated versions for the HubContent.
    """
    list_versions_response = session.list_hub_content_versions(
        hub_name=hub_name,
        hub_content_type=HubContentType.MODEL,
        hub_content_name=hub_content_name,
    )
    hub_content_versions: List[HubContentInfo] = summary_list_from_list_api_response(
        list_versions_response
    )

    unsupported_hub_content_versions_map: Dict[str, List[str]] = {}
    version_to_tag_map = _get_tags_for_all_versions(hub_content_versions, region, session)
    unsupported_hub_content_versions_map = _convert_to_tag_to_versions_map(version_to_tag_map)

    return format_tags(unsupported_hub_content_versions_map)


def _get_tags_for_all_versions(
    hub_content_versions: List[HubContentInfo],
    region: str,
    session: Session,
) -> Dict[str, List[CuratedHubUnsupportedFlag]]:
    """Helper function to create mapping between HubContent version and associated tags."""
    version_to_tags_map: Dict[str, List[CuratedHubUnsupportedFlag]] = {}
    for hub_content_version_summary in hub_content_versions:
        if is_curated_jumpstart_model(hub_content_version_summary) is False:
            continue
        tag_names_to_add: List[
            CuratedHubUnsupportedFlag
        ] = find_unsupported_flags_for_model_version(
            model_id=hub_content_version_summary.hub_content_name,
            version=hub_content_version_summary.hub_content_version,
            region=region,
            session=session,
        )

        version_to_tags_map[hub_content_version_summary.hub_content_version] = tag_names_to_add
    return version_to_tags_map


def _convert_to_tag_to_versions_map(
    version_to_tags_map: Dict[str, List[CuratedHubUnsupportedFlag]]
) -> Dict[CuratedHubUnsupportedFlag, List[str]]:
    """Helper function to create tag to version map from a version to flag mapping."""
    unsupported_hub_content_versions_map: Dict[CuratedHubUnsupportedFlag, List[str]] = {}
    for version, tags in version_to_tags_map.items():
        for tag in tags:
            if tag not in unsupported_hub_content_versions_map:
                unsupported_hub_content_versions_map[tag.value] = []
            # Versions for a HubContent are unique
            unsupported_hub_content_versions_map[tag.value].append(version)

    return unsupported_hub_content_versions_map


def find_unsupported_flags_for_model_version(
    model_id: str, version: str, region: str, session: Session
) -> List[CuratedHubUnsupportedFlag]:
    """Finds relevant CuratedHubTags for a version of a JumpStart public hub model.

    For example, if the public hub model is deprecated,
    this utility will return a `deprecated` tag.
    Since tags are the same for all versions of a HubContent,
    these tags will map from the key to a list of versions impacted.
    """
    flags_to_add: List[CuratedHubUnsupportedFlag] = []
    jumpstart_model_specs = utils.verify_model_region_and_return_specs(
        model_id=model_id,
        version=version,
        region=region,
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=session,
    )

    if jumpstart_model_specs.deprecated:
        flags_to_add.append(CuratedHubUnsupportedFlag.DEPRECATED_VERSIONS)
    if jumpstart_model_specs.inference_vulnerable:
        flags_to_add.append(CuratedHubUnsupportedFlag.INFERENCE_VULNERABLE_VERSIONS)
    if jumpstart_model_specs.training_vulnerable:
        flags_to_add.append(CuratedHubUnsupportedFlag.TRAINING_VULNERABLE_VERSIONS)

    return flags_to_add


def is_curated_jumpstart_model(
    hub_content_summary: HubContentInfo,
) -> Optional[JumpStartModelInfo]:
    """Retrieves the JumpStart model id and version from the JumpStart tag."""
    is_curated_model = next(
        (
            tag
            for tag in hub_content_summary.hub_content_search_keywords
            if tag.startswith(JUMPSTART_CURATED_HUB_MODEL_TAG)
        ),
        None,
    )

    return is_curated_model is not None


def is_gated_bucket(bucket_name: str) -> bool:
    """Returns true if the bucket name is the JumpStart gated bucket."""
    return bucket_name in constants.JUMPSTART_GATED_BUCKET_NAME_SET


def get_data_location_uri(
    src_file: FileInfo, dest_location: S3ObjectLocation, is_gated: bool
) -> str:
    """Util to create data location uri"""
    file_location = src_file.location
    if is_gated and src_file.reference_type in [
        HubContentReferenceType.INFERENCE_ARTIFACT,
        HubContentReferenceType.TRAINING_ARTIFACT,
    ]:
        return file_location.get_uri()

    return f"s3://{dest_location.bucket}/{dest_location.key}/{file_location.key}"

def get_hub_content_arn_without_version(hub_content_arn: str) -> str:
    arn_split = hub_content_arn.split("/")
    return "/".join(arn_split[:-1])
