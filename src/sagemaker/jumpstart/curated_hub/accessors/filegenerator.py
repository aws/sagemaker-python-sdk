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
from functools import singledispatchmethod
from typing import Any, Dict, List, Optional

from botocore.client import BaseClient

from sagemaker.jumpstart.curated_hub.accessors.fileinfo import FileInfo, HubContentDependencyType
from sagemaker.jumpstart.curated_hub.accessors.objectlocation import S3ObjectLocation
from sagemaker.jumpstart.curated_hub.accessors.public_model_data import PublicModelDataAccessor
from sagemaker.jumpstart.types import JumpStartModelSpecs


class FileGenerator:
    """Utility class to help format HubContent data files."""

    def __init__(
        self, region: str, s3_client: BaseClient, studio_specs: Optional[Dict[str, Any]] = None
    ):
        self.region = region
        self.s3_client = s3_client
        self.studio_specs = studio_specs

    @singledispatchmethod
    def format(self, file_input) -> List[FileInfo]:
        """Dispatch method that takes in an input of either ``S3ObjectLocation`` or
        ``JumpStartModelSpecs`` and is implemented in below registered functions.
        """
        # pylint: disable=W0107
        pass

    @format.register
    def _(self, file_input: S3ObjectLocation) -> List[FileInfo]:
        """Implements ``.format`` when the input is of type ``S3ObjectLocation``.

        Returns a list of ``FileInfo`` objects from the specified bucket location.
        """
        files = self.s3_format(file_input)
        return files

    @format.register
    def _(self, file_input: JumpStartModelSpecs) -> List[FileInfo]:
        """Implements ``.format`` when the input is of type ``JumpStartModelSpecs``.

        Returns a list of ``FileInfo`` objects from dependencies found in the public
            model specs.
        """
        files = self.specs_format(file_input, self.studio_specs)
        return files

    def s3_format(self, file_input: S3ObjectLocation) -> List[FileInfo]:
        """Retrieves data from a bucket and formats into FileInfo"""
        parameters = {"Bucket": file_input.bucket, "Prefix": file_input.key}
        response = self.s3_client.list_objects_v2(**parameters)
        contents = response.get("Contents", None)

        if not contents:
            print("Nothing to download")
            return []

        files = []
        for s3_obj in contents:
            key: str = s3_obj.get("Key")
            size: bytes = s3_obj.get("Size", None)
            last_modified: str = s3_obj.get("LastModified", None)
            files.append(FileInfo(key, size, last_modified))
        return files

    def specs_format(
        self, file_input: JumpStartModelSpecs, studio_specs: Dict[str, Any]
    ) -> List[FileInfo]:
        """
        Collects data locations from JumpStart public model specs and
            converts into FileInfo.
        """
        public_model_data_accessor = PublicModelDataAccessor(
            region=self.region, model_specs=file_input, studio_specs=studio_specs
        )
        files = []
        for dependency in HubContentDependencyType:
            location = public_model_data_accessor.get_s3_reference(dependency)
            parameters = {"Bucket": location.bucket, "Prefix": location.key}
            response = self.s3_client.head_object(**parameters)
            key: str = location.key
            size: bytes = response.get("ContentLength", None)
            last_updated: str = response.get("LastModified", None)
            dependency_type: HubContentDependencyType = dependency
            files.append(FileInfo(key, size, last_updated, dependency_type))
        return files
