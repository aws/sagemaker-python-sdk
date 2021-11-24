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
from typing import Dict, List
import semantic_version
import sagemaker
from sagemaker.jumpstart import constants
from sagemaker.jumpstart.types import JumpStartModelHeader, JumpStartVersionedModelId


class SageMakerSettings(object):
    """Static class for storing the SageMaker settings."""

    _PARSED_SAGEMAKER_VERSION = ""

    @staticmethod
    def set_sagemaker_version(version: str) -> None:
        """Set SageMaker version."""
        SageMakerSettings._PARSED_SAGEMAKER_VERSION = version

    @staticmethod
    def get_sagemaker_version() -> str:
        """Return SageMaker version."""
        return SageMakerSettings._PARSED_SAGEMAKER_VERSION


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
    if SageMakerSettings.get_sagemaker_version() == "":
        SageMakerSettings.set_sagemaker_version(parse_sagemaker_version())
    return SageMakerSettings.get_sagemaker_version()


def parse_sagemaker_version() -> str:
    """Returns sagemaker library version. This should only be called once.

    Function reads ``__version__`` variable in ``sagemaker`` module.
    In order to maintain compatibility with the ``semantic_version``
    library, versions with fewer than 2, or more than 3, periods are rejected.
    All versions that cannot be parsed with ``semantic_version`` are also
    rejected.

    Raises:
        RuntimeError: If the SageMaker version is not readable. An exception is also raised if
        the version cannot be parsed by ``semantic_version``.
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

    semantic_version.Version(parsed_version)

    return parsed_version
