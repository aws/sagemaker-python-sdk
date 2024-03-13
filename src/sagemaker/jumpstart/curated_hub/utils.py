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
from typing import Optional
from sagemaker.jumpstart.curated_hub.types import S3ObjectLocation
from sagemaker.s3_utils import parse_s3_url
from sagemaker.session import Session
from sagemaker.utils import aws_partition
from typing import Optional, Dict, List, Any, Set
from botocore.exceptions import ClientError
from sagemaker.jumpstart.types import (
    HubContentType,
    HubArnExtractedInfo,
)
from sagemaker.jumpstart.curated_hub.types import (
    Tag,
    CuratedHubTagName
)
from sagemaker.jumpstart import constants
from sagemaker.jumpstart import utils
from sagemaker.session import Session
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.curated_hub.constants import (
    JUMPSTART_HUB_MODEL_ID_TAG_PREFIX,
    JUMPSTART_HUB_MODEL_VERSION_TAG_PREFIX,
    TASK_TAG_PREFIX,
    FRAMEWORK_TAG_PREFIX,
)


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


def create_s3_object_reference_from_uri(s3_uri: str) -> S3ObjectLocation:
    """Utiity to help generate an S3 object reference"""
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

def tag_hub_content(hub_content_arn: str, tags: List[Tag], session: Session) -> List[Dict[str, List[Dict[str, str]]]]:
    responses = []          
    for tag in tags:
      responses.add(session.add_tags(
        ResourceArn=hub_content_arn,
        Tags=[
            {
                'Key': tag.key,
                'Value': tag.value
            },
        ]
      ))

    return responses
    
def find_all_tags_for_jumpstart_model(hub_name: str, hub_content_name: str, region: str, session: Session) -> List[Tag]:
    hub_content_versions = session.list_hub_content_versions(
        hub_name=hub_name,
        hub_content_type='Model',
        hub_content_name=hub_content_name
    )

    tag_name_to_versions_map: Dict[CuratedHubTagName, List[str]] = {}
    for hub_content_version_summary in hub_content_versions:
        jumpstart_model = get_jumpstart_model_and_version(hub_content_version_summary)
        tag_names_to_add: List[CuratedHubTagName] = find_tags_for_jumpstart_model_version(
            model_id=jumpstart_model["model_id"],
            version=jumpstart_model["version"],
            region=region,
            session=session
        )

        for tag_name in tag_names_to_add:
          if tag_name not in tag_name_to_versions_map:
              tag_name_to_versions_map[tag_name] = []
          tag_name_to_versions_map[tag_name].append(jumpstart_model["version"])
      
    tags: List[Tag] = []
    for tag_name, versions in tag_name_to_versions_map:
        tags.append(Tag(
            key=tag_name,
            versions=str(versions)
        ))

    return tags

    
    
def find_tags_for_jumpstart_model_version(model_id: str, version: str, region: str, session: Session) -> List[CuratedHubTagName]:
    tags_to_add: List[CuratedHubTagName] = []
    specs = utils.verify_model_region_and_return_specs(
        model_id=model_id, 
        version=version,
        region=region,
        scope=JumpStartScriptScope.INFERENCE,
        tolerate_vulnerable_model = True,
        tolerate_deprecated_model = True,
        sagemaker_session=session,
    )

    if (specs.deprecated):
          tags_to_add.add(CuratedHubTagName.DEPRECATED_VERSIONS_TAG)
    if (specs.inference_vulnerable):
        tags_to_add.add(CuratedHubTagName.INFERENCE_VULNERABLE_VERSIONS_TAG)
    if (specs.training_vulnerable):
        tags_to_add.add(CuratedHubTagName.TRAINING_VULNERABLE_VERSIONS_TAG)

    return tags_to_add

    

def get_jumpstart_model_and_version(hub_content_summary: Dict[str, Any]) -> Dict[str, Any]:
    jumpstart_model_id = next(
        (
            tag
            for tag in hub_content_summary["search_keywords"]
            if tag.startswith(JUMPSTART_HUB_MODEL_ID_TAG_PREFIX)
        ),
        None,
    )
    jumpstart_model_version = next(
        (
            tag
            for tag in hub_content_summary["search_keywords"]
            if tag.startswith(JUMPSTART_HUB_MODEL_VERSION_TAG_PREFIX)
        ),
        None,
    )

    return {"model_id": jumpstart_model_id, "version": jumpstart_model_version}