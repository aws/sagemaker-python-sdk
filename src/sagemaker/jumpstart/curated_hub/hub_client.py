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
"""This module contains a client with helpers to access the Private Hub."""
from __future__ import absolute_import
import boto3
from botocore.exceptions import ClientError

from sagemaker.jumpstart.types import JumpStartModelSpecs


class CuratedHubClient:
    """Calls SageMaker Hub APIs for the curated hub."""

    def __init__(self, curated_hub_name: str, region: str) -> None:
        """Sets up region and underlying client."""
        self.curated_hub_name = curated_hub_name
        self._region = region
        self._sm_client = boto3.client("sagemaker", region_name=self._region)

    def create_hub(self, hub_name: str, hub_s3_bucket_name: str):
        """Creates a Private Hub."""
        hub_bucket_s3_uri = f"s3://{hub_s3_bucket_name}"
        self._sm_client.create_hub(
            HubName=hub_name,
            HubDescription="This is a curated hub.",  # TODO verify description
            HubDisplayName=hub_name,
            HubSearchKeywords=[],
            S3StorageConfig={
                "S3OutputPath": hub_bucket_s3_uri,
            },
            Tags=[],
        )

    def desribe_model(self, model_specs: JumpStartModelSpecs):
        """Describes a version of a model in the Private Hub."""
        return self._sm_client.describe_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
            HubContentType="Model",
            HubContentVersion=model_specs.version,
        )

    def delete_all_versions_of_model(self, model_specs: JumpStartModelSpecs):
        """Deletes all versions of a model in the Private Hub."""
        print(f"Deleting all versions of model {model_specs.model_id} from curated hub...")
        content_versions = self._list_hub_content_versions_no_content_noop(model_specs.model_id)

        print(
            f"Found {len(content_versions)} versions of {model_specs.model_id}. Deleting all versions..."
        )

        for content_version in content_versions:
            self.delete_version_of_model(model_specs)

        print(f"Deleting all versions of model {model_specs.model_id} from curated hub complete!")

    def delete_version_of_model(self, model_specs: JumpStartModelSpecs):
        print(f"Deleting version {model_specs.version} of model {model_specs.model_id} from curated hub...")

        self._sm_client.delete_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
            HubContentType="Model",
            HubContentVersion=model_specs.version,
        )

        print(f"Deleting version {model_specs.version} of model {model_specs.model_id} from curated hub complete!")

    def _list_hub_content_versions_no_content_noop(self, hub_content_name: str):
        content_versions = []
        try:
            response = self._sm_client.list_hub_content_versions(
                HubName=self.curated_hub_name,
                HubContentName=hub_content_name,
                HubContentType="Model",
            )
            content_versions = response.pop("HubContentSummaries")
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceNotFound":
                raise ex

        return content_versions