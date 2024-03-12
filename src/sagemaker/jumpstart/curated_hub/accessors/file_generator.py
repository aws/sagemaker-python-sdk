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
"""This module contains important utilities related to HubContent data files."""
from __future__ import absolute_import
from typing import Any, Dict, List

from botocore.client import BaseClient

from sagemaker.jumpstart.curated_hub.types import (
    FileInfo,
    HubContentDependencyType,
    S3ObjectLocation,
)
from sagemaker.jumpstart.curated_hub.accessors.public_model_data import PublicModelDataAccessor
from sagemaker.jumpstart.types import JumpStartModelSpecs


def generate_file_infos_from_s3_location(
    location: S3ObjectLocation, s3_client: BaseClient
) -> List[FileInfo]:
    """Lists objects from an S3 bucket and formats into FileInfo.

    Returns a list of ``FileInfo`` objects from the specified bucket location.
    """
    parameters = {"Bucket": location.bucket, "Prefix": location.key}
    response = s3_client.list_objects_v2(**parameters)
    contents = response.get("Contents")

    if not contents:
        return []

    files = []
    for s3_obj in contents:
        key = s3_obj.get("Key")
        size = s3_obj.get("Size")
        last_modified = s3_obj.get("LastModified")
        files.append(FileInfo(location.bucket, key, size, last_modified))
    return files


def generate_file_infos_from_model_specs(
    model_specs: JumpStartModelSpecs,
    studio_specs: Dict[str, Any],
    region: str,
    s3_client: BaseClient,
) -> List[FileInfo]:
    """Collects data locations from JumpStart public model specs and converts into `FileInfo`.

    Returns a list of `FileInfo` objects from dependencies found in the public
        model specs.
    """
    public_model_data_accessor = PublicModelDataAccessor(
        region=region, model_specs=model_specs, studio_specs=studio_specs
    )
    files = []
    for dependency in HubContentDependencyType:
        location: S3ObjectLocation = public_model_data_accessor.get_s3_reference(dependency)
        location_type = "prefix" if location.key.endswith("/") else "object"

        if location_type == "prefix":
            parameters = {"Bucket": location.bucket, "Prefix": location.key}
            response = s3_client.list_objects_v2(**parameters)
            contents = response.get("Contents")
            for s3_obj in contents:
                key = s3_obj.get("Key")
                size = s3_obj.get("Size")
                last_modified = s3_obj.get("LastModified")
                files.append(
                    FileInfo(
                        location.bucket,
                        key,
                        size,
                        last_modified,
                        dependency,
                    )
                )
        elif location_type == "object":
            parameters = {"Bucket": location.bucket, "Key": location.key}
            response = s3_client.head_object(**parameters)
            size = response.get("ContentLength")
            last_updated = response.get("LastModified")
            files.append(FileInfo(location.bucket, location.key, size, last_updated, dependency))
    return files
