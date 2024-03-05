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
"""This module provides the JumpStart Curated Hub class."""
from __future__ import absolute_import
from concurrent import futures
import traceback
from typing import Optional, Dict, List, Any

import boto3
from botocore.client import BaseClient
from packaging.version import Version

from sagemaker.jumpstart.curated_hub.accessors.filegenerator import FileGenerator, ModelSpecsFileGenerator, S3PathFileGenerator
from sagemaker.jumpstart.curated_hub.accessors.objectlocation import S3ObjectLocation
from sagemaker.jumpstart.curated_hub.accessors.sync import FileSync
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.utils import verify_model_region_and_return_specs
from sagemaker.session import Session
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION
from sagemaker.jumpstart.types import (
    DescribeHubResponse,
    DescribeHubContentsResponse,
    HubContentType,
)
from sagemaker.jumpstart.curated_hub.utils import (
    create_hub_bucket_if_it_does_not_exist,
    generate_default_hub_bucket_name,
)
from sagemaker.jumpstart.curated_hub.types import HubContentDocument_v2


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    def __init__(
        self,
        hub_name: str,
        bucket_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ) -> None:
        """Instantiates a SageMaker ``CuratedHub``.

        Args:
            hub_name (str): The name of the Hub to create.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.
        """
        self.hub_name = hub_name
        self.region = sagemaker_session.boto_region_name
        self._sagemaker_session = sagemaker_session
        self._default_thread_pool_size = 20
        self.hub_bucket_name = bucket_name or self._fetch_hub_bucket_name()
        self._s3_client = self._get_s3_client()

    def _get_s3_client(self) -> BaseClient:
        """Returns an S3 client."""
        return boto3.client("s3", region_name=self.region)

    def _fetch_hub_bucket_name(self) -> str:
        """Retrieves hub bucket name from Hub config if exists"""
        try:
            hub_response = self._sagemaker_session.describe_hub(hub_name=self.hub_name)
            hub_bucket_prefix = hub_response["S3StorageConfig"]["S3OutputPath"]
            return hub_bucket_prefix  # TODO: Strip s3:// prefix
        except ValueError:
            hub_bucket_name = generate_default_hub_bucket_name(self._sagemaker_session)
            print(f"Hub bucket name is: {hub_bucket_name}")  # TODO: Better messaging
            return hub_bucket_name

    def create(
        self,
        description: str,
        display_name: Optional[str] = None,
        search_keywords: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Dict[str, str]:
        """Creates a hub with the given description"""

        bucket_name = create_hub_bucket_if_it_does_not_exist(
            self.hub_bucket_name, self._sagemaker_session
        )

        return self._sagemaker_session.create_hub(
            hub_name=self.hub_name,
            hub_description=description,
            hub_display_name=display_name,
            hub_search_keywords=search_keywords,
            hub_bucket_name=bucket_name,
            tags=tags,
        )

    def describe(self) -> DescribeHubResponse:
        """Returns descriptive information about the Hub"""

        hub_description: DescribeHubResponse = self._sagemaker_session.describe_hub(
            hub_name=self.hub_name
        )

        return hub_description

    def list_models(self, **kwargs) -> Dict[str, Any]:
        """Lists the models in this Curated Hub

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        # TODO: Validate kwargs and fast-fail?

        hub_content_summaries = self._sagemaker_session.list_hub_contents(
            hub_name=self.hub_name, hub_content_type=HubContentType.MODEL, **kwargs
        )
        # TODO: Handle pagination
        return hub_content_summaries

    def describe_model(
        self, model_name: str, model_version: str = "*"
    ) -> DescribeHubContentsResponse:
        """Returns descriptive information about the Hub Model"""

        hub_content_description: Dict[str, Any] = self._sagemaker_session.describe_hub_content(
            hub_name=self.hub_name,
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL,
        )

        return DescribeHubContentsResponse(hub_content_description)

    def delete_model(self, model_name: str, model_version: str = "*") -> None:
        """Deletes a model from this CuratedHub."""
        return self._sagemaker_session.delete_hub_content(
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL,
            hub_name=self.hub_name,
        )

    def delete(self) -> None:
        """Deletes this Curated Hub"""
        return self._sagemaker_session.delete_hub(self.hub_name)

    def _is_invalid_model_list_input(self, model_list: List[Dict[str, str]]) -> bool:
        """Determines if input args to ``sync`` is correct."""
        for obj in model_list:
            if not isinstance(obj.get("model_id"), str):
                return True
            if "version" in obj and not isinstance(obj["version"], str):
                return True
        return False

    def sync(self, model_list: List[Dict[str, str]]):
        """Syncs a list of JumpStart model ids and versions with a CuratedHub

        Args:
            model_list (List[Dict[str, str]]): List of `{ model_id: str, version: Optional[str] }`
                objects that should be synced into the Hub.
        """
        if self._is_invalid_model_list_input(model_list):
            raise ValueError(
                "Model list should be a list of objects with values 'model_id',",
                "and optional 'version'.",
            )

        # Fetch required information
        # self._get_studio_manifest_map()
        hub_models = self.list_models()

        # Retrieve latest version of unspecified JumpStart model versions
        model_version_list = []
        for model in model_list:
            version = model.get("version", "*")
            if not version or version == "*":
                model_specs = verify_model_region_and_return_specs(
                    model["model_id"], version, JumpStartScriptScope.INFERENCE, self.region
                )
                model["version"] = model_specs.version
            model_version_list.append(model)

        # Find synced JumpStart model versions in the Hub
        js_models_in_hub = []
        for hub_model in hub_models:
            # TODO: extract both in one pass
            jumpstart_model_id = next(
                (
                    tag
                    for tag in hub_model["search_keywords"]
                    if tag.startswith("@jumpstart-model-id")
                ),
                None,
            )
            jumpstart_model_version = next(
                (
                    tag
                    for tag in hub_model["search_keywords"]
                    if tag.startswith("@jumpstart-model-version")
                ),
                None,
            )

            if jumpstart_model_id and jumpstart_model_version:
                js_models_in_hub.append(hub_model)

        # Match inputted list of model versions with synced JumpStart model versions in the Hub
        models_to_sync = []
        for model in model_version_list:
            matched_model = next(
                (
                    hub_model
                    for hub_model in js_models_in_hub
                    if hub_model and hub_model["name"] == model["model_id"]
                ),
                None,
            )

            # Model does not exist in Hub, sync
            if not matched_model:
                models_to_sync.append(model)

            if matched_model:
                model_version = Version(model["version"])
                hub_model_version = Version(matched_model["version"])

                # 1. Model version exists in Hub, pass
                if hub_model_version == model_version:
                    pass

                # 2. Invalid model version exists in Hub, pass
                # This will only happen if something goes wrong in our metadata
                if hub_model_version > model_version:
                    pass

                # 3. Old model version exists in Hub, update
                if hub_model_version < model_version:
                    # Check minSDKVersion against current SDK version, emit log
                    models_to_sync.append(model)

        # Delete old models?

        # Copy content workflow + `SageMaker:ImportHubContent` for each model-to-sync in parallel
        tasks: List[futures.Future] = []
        with futures.ThreadPoolExecutor(
            max_workers=self._default_thread_pool_size,
            thread_name_prefix="import-models-to-curated-hub",
        ) as deploy_executor:
            for model in models_to_sync:
                task = deploy_executor.submit(self._sync_public_model_to_hub, model)
                tasks.append(task)

        # Handle failed imports
        results = futures.wait(tasks)
        failed_imports: List[Dict[str, Any]] = []
        for result in results.done:
            exception = result.exception()
            if exception:
                failed_imports.append(
                    {
                        "Exception": exception,
                        "Traceback": "".join(
                            traceback.TracebackException.from_exception(exception).format()
                        ),
                    }
                )
        if failed_imports:
            raise RuntimeError(
                f"Failures when importing models to curated hub in parallel: {failed_imports}"
            )

    def _sync_public_model_to_hub(self, model: Dict[str, str]):
        """Syncs a public JumpStart model version to the Hub. Runs in parallel."""
        model_name = model["name"]
        model_version = model["version"]

        model_specs = verify_model_region_and_return_specs(
            model_id=model_name,
            version=model_version,
            region=self.region,
            scope=JumpStartScriptScope.INFERENCE,
            sagemaker_session=self._sagemaker_session,
        )

        # TODO: Uncomment and implement
        # studio_specs = self.fetch_studio_specs(model_id=model_name, version=model_version)
        studio_specs = {}

        dest_location = S3ObjectLocation(
            bucket=self.hub_bucket_name, key=f"{model_name}/{model_version}"
        )
        # TODO: Validations? HeadBucket?

        src_files = ModelSpecsFileGenerator(self.region, self._s3_client, studio_specs).format(model_specs)
        dest_files = S3PathFileGenerator(self.region, self._s3_client).format(dest_location)

        files_to_copy = FileSync(src_files, dest_files, dest_location).call()

        if len(files_to_copy) > 0:
            # TODO: Copy files with MPU
            print("hi")

        # Tag model if specs say it is deprecated or training/inference vulnerable
        # Update tag of HubContent ARN without version. Versioned ARNs are not
        # onboarded to Tagris.
        tags = []

        hub_content_document = HubContentDocument_v2(spec=model_specs)

        self._sagemaker_session.import_hub_content(
            document_schema_version=HubContentDocument_v2.SCHEMA_VERSION,
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_name=self.hub_name,
            hub_content_document=hub_content_document,
            hub_content_type=HubContentType.MODEL,
            hub_content_display_name="",
            hub_content_description="",
            hub_content_markdown="",
            hub_content_search_keywords=[],
            tags=tags,
        )
