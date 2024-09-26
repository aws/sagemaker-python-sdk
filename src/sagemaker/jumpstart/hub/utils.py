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
from typing import Optional
from sagemaker.jumpstart.hub.types import S3ObjectLocation
from sagemaker.s3_utils import parse_s3_url
from sagemaker.session import Session
from sagemaker.utils import aws_partition
from sagemaker.jumpstart.types import HubContentType, HubArnExtractedInfo
from sagemaker.jumpstart import constants
from packaging.specifiers import SpecifierSet, InvalidSpecifier


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
        hub_content_version = match.group(7)

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
        f"{info.hub_name}/{HubContentType.MODEL_REFERENCE}/{model_name}/{version}"
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
    """Returns available Jumpstart hub model version

    Raises:
        ClientError: If the specified model is not found in the hub.
    """

    try:
        hub_content_summaries = sagemaker_session.list_hub_content_versions(
            hub_name=hub_name, hub_content_name=hub_model_name, hub_content_type=hub_model_type
        ).get("HubContentSummaries")
    except Exception as ex:
        raise Exception(f"Failed calling list_hub_content_versions: {str(ex)}")

    available_model_versions = [model.get("HubContentVersion") for model in hub_content_summaries]

    if hub_model_version == "*" or hub_model_version is None:
        return str(max(available_model_versions))

    try:
        spec = SpecifierSet(f"=={hub_model_version}")
    except InvalidSpecifier:
        raise KeyError(f"Bad semantic version: {hub_model_version}")
    available_versions_filtered = list(spec.filter(available_model_versions))
    if not available_versions_filtered:
        raise KeyError("Model version not available in the Hub")
    hub_model_version = str(max(available_versions_filtered))

    return hub_model_version
