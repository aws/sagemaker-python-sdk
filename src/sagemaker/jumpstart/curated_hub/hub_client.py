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
from typing import Dict, Any, List, Optional
import time
import boto3
from botocore.exceptions import ClientError

from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.constants import (
    CURATED_HUB_DEFAULT_DESCRIPTION,
    HubContentType,
)
from sagemaker.jumpstart.curated_hub.accessors.s3_object_reference import (
    S3ObjectLocation,
)


class CuratedHubClient:
    """Calls SageMaker Hub APIs for the curated hub."""

    def __init__(self, curated_hub_name: str, region: str) -> None:
        """Sets up region and underlying client."""
        self.curated_hub_name = curated_hub_name
        self._region = region
        self._sm_client = boto3.client("sagemaker", region_name=self._region)

    def create_hub(
        self,
        hub_name: str,
        hub_s3_location: S3ObjectLocation = None,
        hub_description: str = CURATED_HUB_DEFAULT_DESCRIPTION,
    ) -> None:
        """Creates a Private Hub."""
        self._sm_client.create_hub(
            HubName=hub_name,
            HubDescription=hub_description,
            HubDisplayName=hub_name,
            HubSearchKeywords=[],
            S3StorageConfig={
                "S3OutputPath": hub_s3_location.get_uri(),
            },
            Tags=[],
        )

    def describe_model_version(self, model_specs: JumpStartModelSpecs) -> Dict[str, Any]:
        """Describes a version of a model in the Private Hub."""
        return self._sm_client.describe_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
            HubContentType=HubContentType.MODEL,
            HubContentVersion=model_specs.version,
        )

    def delete_all_versions_of_model(self, model_specs: JumpStartModelSpecs):
        """Deletes all versions of a model in the Private Hub."""
        print(f"Deleting all versions of model {model_specs.model_id} from curated hub...")
        content_versions = self._list_hub_content_versions_no_content_noop(model_specs.model_id)

        print(
            f"Found {len(content_versions)} versions of"
            f" {model_specs.model_id}. Deleting all versions..."
        )

        for content_version in content_versions:
            self.delete_version_of_model(
                model_specs.model_id, content_version.pop("HubContentVersion")
            )

        print(f"Deleting all versions of model {model_specs.model_id} from curated hub complete!")

    def delete_version_of_model(self, model_id: str, version: str) -> None:
        """Deletes specific version of a model"""
        print(f"Deleting version {version} of" f" model {model_id} from curated hub...")

        self._sm_client.delete_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_id,
            HubContentType=HubContentType.MODEL,
            HubContentVersion=version,
        )

        # Sleep for one second avoid being throttled
        time.sleep(1)

        print(f"Deleted version {version} of" f" model {model_id} from curated hub!")

    def _list_hub_content_versions_no_content_noop(
        self, hub_content_name: str
    ) -> List[Dict[str, Any]]:
        """Lists hub content versions, returns an empty list if the hub content does not exist."""
        content_versions = []
        try:
            response = self._sm_client.list_hub_content_versions(
                HubName=self.curated_hub_name,
                HubContentName=hub_content_name,
                HubContentType=HubContentType.MODEL,
            )
            content_versions = response["HubContentSummaries"]
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceNotFound":
                raise

        return content_versions

    def list_hub_names_on_account(self) -> List[str]:
        """Lists the Private Hubs on an AWS account for the region.

        This call handles the pagination.
        """
        hub_names: List[str] = []
        run_once: bool = True
        next_token: Optional[str] = None
        while next_token or run_once:
            run_once = False
            if next_token:
                res = self._sm_client.list_hubs(NextToken=next_token)
            else:
                res = self._sm_client.list_hubs()

            hub_names.extend(map(self._get_hub_name_from_hub_summary, res["HubSummaries"]))
            next_token = res["NextToken"]

        return hub_names

    def _get_hub_name_from_hub_summary(self, hub_summary: Dict[str, Any]) -> str:
        """Retrieves a hub name form a ListHubs HubSummary field."""
        return hub_summary["HubName"]
