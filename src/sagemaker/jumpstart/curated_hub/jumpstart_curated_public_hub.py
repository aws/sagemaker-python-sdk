from __future__ import absolute_import

import json
import time
from concurrent import futures
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

import boto3
from botocore.client import ClientError

from sagemaker.jumpstart.curated_hub.content_copy import ContentCopier
from sagemaker.jumpstart.curated_hub.hub_client import CuratedHubClient
from sagemaker.jumpstart.curated_hub.model_document import ModelDocumentCreator
from sagemaker.jumpstart.curated_hub.stsClient import StsClient
from sagemaker.jumpstart.curated_hub.filesystem.public_hub_s3_filesystem import PublicHubS3Filesystem
from sagemaker.jumpstart.curated_hub.filesystem.curated_hub_s3_filesystem import CuratedHubS3Filesystem
from sagemaker.jumpstart.curated_hub.utils import PublicModelId, \
    construct_s3_uri, get_studio_model_metadata_map_from_region
from sagemaker.jumpstart.enums import (
    JumpStartScriptScope,
)
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.utils import (
    verify_model_region_and_return_specs, )
from sagemaker.session import Session


class JumpStartCuratedPublicHub:
    """JumpStartCuratedPublicHub class.

    This class helps users create a new curated hub.
    If a hub already exists on the account, it will attempt to use that hub.
    """

    def __init__(self, curated_hub_name: str, import_to_preexisting_hub: bool = False, region: str = "us-west-2"):
        self._region = region
        self._s3_client = boto3.client("s3", region_name=self._region)
        self._sm_client = boto3.client("sagemaker", region_name=self._region)
        self._thread_pool_size = 20
        self._account_id = StsClient().get_account_id()

        # Finds the relevant hub and s3 locations
        self.curated_hub_name = curated_hub_name
        self.curated_hub_s3_bucket_name = f"{curated_hub_name}-{self._region}-{self._account_id}"
        preexisting_hub = self._get_curated_hub_and_curated_hub_s3_bucket_names(import_to_preexisting_hub)
        if (preexisting_hub):
            name_of_hub_already_on_account = preexisting_hub[0]

            if not import_to_preexisting_hub:
                raise Exception(f"Hub with name {name_of_hub_already_on_account} detected on account. The limit of hubs per account is 1. If you wish to use this hub as the curated hub, please set the flag `import_to_preexisting_hub` to True.")
            print(f"WARN: Hub with name {name_of_hub_already_on_account} detected on account. The limit of hubs per account is 1. `import_to_preexisting_hub` is set to true - defaulting to this hub.")

            self.curated_hub_name = name_of_hub_already_on_account
            self.curated_hub_s3_bucket_name = preexisting_hub[1]

        self._hub_client = CuratedHubClient(curated_hub_name=self.curated_hub_name, region=self._region)
        self._sagemaker_session = Session()
        self.studio_metadata_map = get_studio_model_metadata_map_from_region(region=self._region)

        self._src_s3_filesystem = PublicHubS3Filesystem(self._region)
        self._dst_s3_filesystem = CuratedHubS3Filesystem(self._region, self.curated_hub_s3_bucket_name)

        self._content_copier = ContentCopier(
            region=self._region,
            s3_client=self._s3_client,
            src_s3_filesystem=self._src_s3_filesystem,
            dst_s3_filesystem=self._dst_s3_filesystem
        )
        self._document_creator = ModelDocumentCreator(
            region=self._region, palatine_hub_s3_filesystem=self._dst_s3_filesystem, studio_metadata_map=self.studio_metadata_map
        )

    def _get_curated_hub_and_curated_hub_s3_bucket_names(self, import_to_preexisting_hub: bool) -> Optional[Tuple[str, str]]:
        res = self._sm_client.list_hubs().pop("HubSummaries")
        if (len(res) > 0):
            name_of_hub_already_on_account = res[0]["HubName"]
            hub_res = self._sm_client.describe_hub(HubName=name_of_hub_already_on_account)
            curated_hub_name = hub_res["HubName"]
            curated_hub_s3_bucket_name = hub_res.pop("S3StorageConfig")["S3OutputPath"].replace("s3://", "", 1).split("/")[0]
            print(f"Hub found on account in region {self._region} with name {curated_hub_name} and s3Config {curated_hub_s3_bucket_name}")
            return (curated_hub_name, curated_hub_s3_bucket_name)
        return None

    def get_or_create(self):
        """Creates a curated hub in the caller AWS account.

        If the S3 bucket does not exist, this will create a new one.
        If the curated hub does not exist, this will create a new one."""
        self._get_or_create_s3_bucket(self.curated_hub_s3_bucket_name)
        self._get_or_create_private_hub(self.curated_hub_name)

        print(f"HUB_BUCKET_NAME={self.curated_hub_s3_bucket_name}")

    def _get_or_create_private_hub(self, hub_name: str):
        try:
            return self._create_private_hub(hub_name)
        except ClientError as ex:
            if ex.response["Error"]["Code"] not in ["ResourceLimitExceeded", "ResourceInUse"]:
                raise ex

    def _create_private_hub(self, hub_name: str):
        hub_bucket_s3_uri = f"s3://{hub_name}"
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

    def _get_or_create_s3_bucket(self, bucket_name: str):
        try:
            return self._call_create_bucket(bucket_name)
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise ex

    def _call_create_bucket(self, bucket_name: str):
        # TODO make sure bucket policy permits PutObjectTagging so bucket-to-bucket copy will work
        self._s3_client.create_bucket(
            Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": self._region}
        )

    def import_models(self, model_ids: List[PublicModelId], parallel_import: bool = False):
        """Imports models in list to curated hub

        By default, this function imports models in parallel.
        If the model already exists in the curated hub, it will remove the version and replace it with the latest version."""


        print(f"Importing {len(model_ids)} models to curated private hub...")
        if not parallel_import:
          for model_id in model_ids:
              self._delete_and_import_model(model_id)
        else:
          tasks = []
          with futures.ThreadPoolExecutor(
              max_workers=self._thread_pool_size, thread_name_prefix="import-models-to-curated-hub"
          ) as deploy_executor:
              for model_id in model_ids:
                  task = deploy_executor.submit(self._delete_and_import_model, model_id)
                  tasks.append(task)

          results = futures.wait(tasks)
          failed_deployments: List[BaseException] = []
          for result in results.done:
              exception = result.exception()
              if exception:
                  failed_deployments.append(exception)
          if failed_deployments:
              raise RuntimeError(f"Failures when importing models to curated hub in parallel: {failed_deployments}")

    def _delete_and_import_model(self, model_id: PublicModelId):
        self._hub_client.delete_model(model_id) # TODO: Figure out why Studio terminal is passing in tags to import call
        self._import_model(model_id)

    def _import_model(self, public_js_model: PublicModelId) -> None:
        print(
            f"Importing model {public_js_model.id} version {public_js_model.version} to curated private hub..."
        )
        model_specs = verify_model_region_and_return_specs(
            model_id=public_js_model.id,
            version=public_js_model.version,
            scope=JumpStartScriptScope.INFERENCE,
            region=self._region,
        )

        self._content_copier.copy_hub_content_dependencies_to_hub_bucket(model_specs=model_specs)

        # self._import_public_model_to_hub_no_overwrite(model_specs=model_specs)
        self._import_public_model_to_hub(model_specs=model_specs)
        print(
            f"Importing model {public_js_model.id} version {public_js_model.version} to curated private hub complete!"
        )

    def _import_public_model_to_hub_no_overwrite(self, model_specs: JumpStartModelSpecs):
        try:
            self._import_public_model_to_hub(model_specs)
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceInUse":
                raise ex

    def _import_public_model_to_hub(self, model_specs: JumpStartModelSpecs):
        # TODO Several fields are not present in SDK specs as they are only in Studio specs right now (not urgent)
        hub_content_display_name = self.studio_metadata_map[model_specs.model_id]["name"]
        hub_content_description = f"This is a curated model based off the public JumpStart model {hub_content_display_name}" # TODO enable: self.studio_metadata_map[model_specs.model_id]["desc"]
        hub_content_markdown = self._dst_s3_filesystem.get_markdown_s3_reference(model_specs).get_uri()

        hub_content_document = self._document_creator.make_hub_content_document(model_specs=model_specs)

        self._sm_client.import_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
            HubContentType="Model",
            DocumentSchemaVersion="1.0.0",
            HubContentDisplayName=hub_content_display_name,
            HubContentDescription=hub_content_description,
            HubContentMarkdown=hub_content_markdown,
            HubContentDocument=hub_content_document
        )

    def delete_models(self, model_ids: List[PublicModelId]):
        for model_id in model_ids:
            self._hub_client.delete_model(model_id)
