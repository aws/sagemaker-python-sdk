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
"""This module contains utilities related to SageMaker JumpStart."""
from __future__ import absolute_import
from functools import reduce
from typing import Dict, List, Optional
from urllib.parse import urlparse
from packaging.version import Version
import sagemaker
from sagemaker.jumpstart import constants
from sagemaker.jumpstart import accessors
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartVersionedModelId
from sagemaker.s3 import parse_s3_url


def get_jumpstart_launched_regions_message() -> str:
    """Returns formatted string indicating where JumpStart is launched."""
    if len(constants.JUMPSTART_REGION_NAME_SET) == 0:
        return "JumpStart is not available in any region."
    if len(constants.JUMPSTART_REGION_NAME_SET) == 1:
        region = list(constants.JUMPSTART_REGION_NAME_SET)[0]
        return f"JumpStart is available in {region} region."

    sorted_regions = sorted(list(constants.JUMPSTART_REGION_NAME_SET))
    if len(constants.JUMPSTART_REGION_NAME_SET) == 2:
        return f"JumpStart is available in {sorted_regions[0]} and {sorted_regions[1]} regions."

    formatted_launched_regions_list = []
    for i, region in enumerate(sorted_regions):
        region_prefix = "" if i < len(sorted_regions) - 1 else "and "
        formatted_launched_regions_list.append(region_prefix + region)
    formatted_launched_regions_str = ", ".join(formatted_launched_regions_list)
    return f"JumpStart is available in {formatted_launched_regions_str} regions."


def get_jumpstart_content_bucket(region: str) -> str:
    """Returns regionalized content bucket name for JumpStart.

    Raises:
        RuntimeError: If JumpStart is not launched in ``region``.
    """
    try:
        return constants.JUMPSTART_REGION_NAME_TO_LAUNCHED_REGION_DICT[region].content_bucket
    except KeyError:
        formatted_launched_regions_str = get_jumpstart_launched_regions_message()
        raise ValueError(
            f"Unable to get content bucket for JumpStart in {region} region. "
            f"{formatted_launched_regions_str}"
        )


def get_formatted_manifest(
    manifest: List[Dict],
) -> Dict[JumpStartVersionedModelId, JumpStartModelHeader]:
    """Returns formatted manifest dictionary from raw manifest.

    Keys are JumpStartVersionedModelId objects, values are
    ``JumpStartModelHeader`` objects.
    """
    manifest_dict = {}
    for header in manifest:
        header_obj = JumpStartModelHeader(header)
        manifest_dict[
            JumpStartVersionedModelId(header_obj.model_id, header_obj.version)
        ] = header_obj
    return manifest_dict


def get_sagemaker_version() -> str:
    """Returns sagemaker library version.

    If the sagemaker library version has not been set, this function
    calls ``parse_sagemaker_version`` to retrieve the version and set
    the constant.
    """
    if accessors.SageMakerSettings.get_sagemaker_version() == "":
        accessors.SageMakerSettings.set_sagemaker_version(parse_sagemaker_version())
    return accessors.SageMakerSettings.get_sagemaker_version()


def parse_sagemaker_version() -> str:
    """Returns sagemaker library version. This should only be called once.

    Function reads ``__version__`` variable in ``sagemaker`` module.
    In order to maintain compatibility with the ``packaging.version``
    library, versions with fewer than 2, or more than 3, periods are rejected.
    All versions that cannot be parsed with ``packaging.version`` are also
    rejected.

    Raises:
        RuntimeError: If the SageMaker version is not readable. An exception is also raised if
        the version cannot be parsed by ``packaging.version``.
    """
    version = sagemaker.__version__
    parsed_version = None

    num_periods = version.count(".")
    if num_periods == 2:
        parsed_version = version
    elif num_periods == 3:
        trailing_period_index = version.rfind(".")
        parsed_version = version[:trailing_period_index]
    else:
        raise RuntimeError(f"Bad value for SageMaker version: {sagemaker.__version__}")

    Version(parsed_version)

    return parsed_version


def is_jumpstart_model_input(model_id: Optional[str], version: Optional[str]) -> bool:
    """Determines if `model_id` and `version` input are for JumpStart.

    This method returns True if both arguments are not None, false if both arguments
    are None, and raises an exception if one argument is None but the other isn't.

    Args:
        model_id (str): Optional. Model ID of the JumpStart model.
        version (str): Optional. Version of the JumpStart model.

    Raises:
        ValueError: If only one of the two arguments is None.
    """
    if model_id is not None or version is not None:
        if model_id is None or version is None:
            raise ValueError(
                "Must specify `model_id` and `model_version` when getting specs for "
                "JumpStart models."
            )
        return True
    return False


def is_jumpstart_model_uri(uri: Optional[str]) -> bool:
    """Returns True if URI corresponds to a JumpStart-hosted model.

    Args:
        uri (Optional[str]): uri for inference/training job.
    """

    bucket = None
    if urlparse(uri).scheme == "s3":
        bucket, _ = parse_s3_url(uri)

    return bucket in constants.JUMPSTART_BUCKET_NAME_SET


def tag_key_in_array(tag_key: str, tag_array: List[Dict[str, str]]) -> bool:
    """Returns True if ``tag_key`` is in the ``tag_array``.

    Args:
        tag_key (str): the tag key to check if it's already in the ``tag_array``.
        tag_array (List[Dict[str, str]]): array of tags to check for ``tag_key``.
    """
    if len(tag_array) == 0:
        return False
    return tag_key in reduce(lambda a, b: set(a.keys()).union(set(b.keys())), tag_array)


def add_jumpstart_tags(
    tags: Optional[List[Dict[str, str]]],
    inference_model_uri: Optional[str],
    inference_script_uri: Optional[str],
) -> List[Dict[str, str]]:
    """Add custom tags to JumpStart models, return the updated tags.

    No-op if this is not a JumpStart model related resource.

    Args:
        tags (Optional[List[Dict[str,str]]): Current tags for JumpStart inference
        or training job.
        inference_model_uri (Optional[str]): S3 URI for inference model artifact.
        inference_script_uri (Optional[str]): S3 URI for inference script tarball.
    """

    if is_jumpstart_model_uri(inference_model_uri):
        if tags is None:
            tags = []
        if not tag_key_in_array(constants.JumpStartTag.INFERENCE_MODEL_URI.value, tags):
            tags.append(
                {
                    constants.JumpStartTag.INFERENCE_MODEL_URI.value: inference_model_uri,
                }
            )

    if is_jumpstart_model_uri(inference_script_uri):
        if tags is None:
            tags = []
        if not tag_key_in_array(constants.JumpStartTag.INFERENCE_SCRIPT_URI.value, tags):
            tags.append(
                {
                    constants.JumpStartTag.INFERENCE_SCRIPT_URI.value: inference_script_uri,
                }
            )

    return tags
