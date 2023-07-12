from __future__ import absolute_import

from typing import List

import boto3
from botocore.client import ClientError

from sagemaker.jumpstart.curated_hub.content_copy import ContentCopier, dst_markdown_key
from sagemaker.jumpstart.curated_hub.hub_client import CuratedHubClient
from sagemaker.jumpstart.curated_hub.model_document import ModelDocumentCreator
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
    """

    def __init__(self, curated_hub_s3_prefix: str):
        self.curated_hub_s3_prefix = curated_hub_s3_prefix
        self.curated_hub_name = f"{curated_hub_s3_prefix}"  # TODO verify name

        self._region = "us-west-2"
        self._s3_client = boto3.client("s3", region_name=self._region)
        self._sm_client = boto3.client("sagemaker", region_name=self._region)
        self._hub_client = CuratedHubClient(curated_hub_name=self.curated_hub_name, region=self._region)
        self._sagemaker_session = Session()
        # self.studio_metadata_map = {}
        self.studio_metadata_map = get_studio_model_metadata_map_from_region(region=self._region)
        self._content_copier = ContentCopier(
            region=self._region,
            s3_client=self._s3_client,
            curated_hub_name=self.curated_hub_name,
            studio_metadata_map=self.studio_metadata_map,
        )
        self._document_creator = ModelDocumentCreator(region=self._region, content_copier=self._content_copier)

    def create(self):
        """Creates a curated hub in the caller AWS account.

        If the S3 bucket does not exist, this will create a new one.
        If the curated hub does not exist, this will create a new one."""
        self._get_or_create_s3_bucket(self.curated_hub_name)
        self._get_or_create_curated_hub()

    def _get_or_create_curated_hub(self):
        try:
            return self._create_curated_hub()
        except ClientError as ex:
            if ex.response["Error"]["Code"] != "ResourceInUse":
                raise ex

    def _get_curated_hub(self):
        self._sm_client.describe_hub(HubName=self.curated_hub_name)

    def _create_curated_hub(self):
        hub_bucket_s3_uri = f"s3://{self.curated_hub_name}"
        self._sm_client.create_hub(
            HubName=self.curated_hub_name,
            HubDescription="This is a curated hub.",  # TODO verify description
            HubDisplayName=self.curated_hub_name,
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

    def import_models(self, model_ids: List[PublicModelId]):
        """Imports models in list to curated hub

        If the model already exists in the curated hub, it will skip the upload."""
        print(f"Importing {len(model_ids)} models to curated private hub...")
        for model_id in model_ids:
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
        hub_content_display_name = model_specs.model_id
        hub_content_description = f"This is the very informative {model_specs.model_id} description"
        hub_content_markdown = construct_s3_uri(self._content_copier.dst_bucket(), dst_markdown_key(model_specs))

        hub_content_document = self._document_creator.make_hub_content_document(model_specs=model_specs)

        self._sm_client.import_hub_content(
            HubName=self.curated_hub_name,
            HubContentName=model_specs.model_id,
            HubContentType="Model",
            DocumentSchemaVersion="1.0.0",
            HubContentDisplayName=hub_content_display_name,
            HubContentDescription=hub_content_description,
            HubContentMarkdown=hub_content_markdown,
            HubContentDocument=hub_content_document,
            HubContentSearchKeywords=[],
            Tags=[],
        )

    def delete_models(self, model_ids: List[PublicModelId]):
        for model_id in model_ids:
            self._hub_client.delete_model(model_id)
