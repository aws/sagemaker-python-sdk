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
from datetime import datetime
import json
import traceback
from typing import Optional, Dict, List, Any

import boto3
from botocore import exceptions
from botocore.client import BaseClient
from packaging.version import Version

from sagemaker.jumpstart import utils
from sagemaker.jumpstart.curated_hub.accessors import file_generator
from sagemaker.jumpstart.curated_hub.accessors.multipartcopy import MultiPartCopyHandler
from sagemaker.jumpstart.curated_hub.constants import (
    JUMPSTART_HUB_MODEL_ID_TAG_PREFIX,
    JUMPSTART_HUB_MODEL_VERSION_TAG_PREFIX,
    TASK_TAG_PREFIX,
    FRAMEWORK_TAG_PREFIX,
)
from sagemaker.jumpstart.curated_hub.sync.comparator import SizeAndLastUpdatedComparator
from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequestFactory
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.session import Session
from sagemaker.jumpstart.constants import DEFAULT_JUMPSTART_SAGEMAKER_SESSION, JUMPSTART_LOGGER
from sagemaker.jumpstart.types import (
    DescribeHubResponse,
    DescribeHubContentsResponse,
    HubContentType,
    JumpStartModelSpecs,
)
from sagemaker.jumpstart.curated_hub.utils import (
    create_hub_bucket_if_it_does_not_exist,
    generate_default_hub_bucket_name,
    create_s3_object_reference_from_uri,
    tag_jumpstart_hub_content_on_spec_fields,
    get_jumpstart_model_and_version,
)
from sagemaker.jumpstart.curated_hub.types import (
    HubContentDocument_v2,
    JumpStartModelInfo,
    S3ObjectLocation,
)


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
        self._s3_client = self._get_s3_client()
        self.hub_storage_location = self._generate_hub_storage_location(bucket_name)

    def _get_s3_client(self) -> BaseClient:
        """Returns an S3 client used for creating a HubContentDocument."""
        return boto3.client("s3", region_name=self.region)

    def _fetch_hub_bucket_name(self) -> str:
        """Retrieves hub bucket name from Hub config if exists"""
        try:
            hub_response = self._sagemaker_session.describe_hub(hub_name=self.hub_name)
            hub_output_location = hub_response["S3StorageConfig"].get("S3OutputPath")
            if hub_output_location:
                location = create_s3_object_reference_from_uri(hub_output_location)
                return location.bucket
            default_bucket_name = generate_default_hub_bucket_name(self._sagemaker_session)
            JUMPSTART_LOGGER.warning(
                "There is not a Hub bucket associated with %s. Using %s",
                self.hub_name,
                default_bucket_name,
            )
            return default_bucket_name
        except exceptions.ClientError:
            hub_bucket_name = generate_default_hub_bucket_name(self._sagemaker_session)
            JUMPSTART_LOGGER.warning(
                "There is not a Hub bucket associated with %s. Using %s",
                self.hub_name,
                hub_bucket_name,
            )
            return hub_bucket_name

    def _generate_hub_storage_location(self, bucket_name: Optional[str] = None) -> None:
        """Generates an ``S3ObjectLocation`` given a Hub name."""
        hub_bucket_name = bucket_name or self._fetch_hub_bucket_name()
        curr_timestamp = datetime.now().timestamp()
        return S3ObjectLocation(bucket=hub_bucket_name, key=f"{self.hub_name}-{curr_timestamp}")

    def create(
        self,
        description: str,
        display_name: Optional[str] = None,
        search_keywords: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> Dict[str, str]:
        """Creates a hub with the given description"""

        create_hub_bucket_if_it_does_not_exist(
            self.hub_storage_location.bucket, self._sagemaker_session
        )

        return self._sagemaker_session.create_hub(
            hub_name=self.hub_name,
            hub_description=description,
            hub_display_name=display_name,
            hub_search_keywords=search_keywords,
            s3_storage_config={"S3OutputPath": self.hub_storage_location.get_uri()},
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
        """Determines if input args to ``sync`` is correct.

        `model_list` objects must have `model_id` (str) and optional `version` (str).
        """
        for obj in model_list:
            if not isinstance(obj.get("model_id"), str):
                return True
            if "version" in obj and not isinstance(obj["version"], str):
                return True
        return False

    def _populate_latest_model_version(self, model: Dict[str, str]) -> Dict[str, str]:
        """Populates the lastest version of a model from specs no matter what is passed.

        Returns model ({ model_id: str, version: str })
        """
        model_specs = utils.verify_model_region_and_return_specs(
            model["model_id"], "*", JumpStartScriptScope.INFERENCE, self.region
        )
        return {"model_id": model["model_id"], "version": model_specs.version}

    def _get_jumpstart_models_in_hub(self) -> List[Dict[str, Any]]:
        """Returns list of `HubContent` that have been created from a JumpStart model."""
        hub_models = self.list_models()

        js_models_in_hub = []
        for hub_model_summary in hub_models["HubContentSummaries"]:
            jumpstart_model = get_jumpstart_model_and_version(hub_model_summary)
            if jumpstart_model["model_id"] and jumpstart_model["version"]:
                js_models_in_hub.append(hub_model_summary)

        return js_models_in_hub

    def _determine_models_to_sync(
        self, model_list: List[JumpStartModelInfo], models_in_hub: Dict[str, Any]
    ) -> List[JumpStartModelInfo]:
        """Determines which models from `sync` params to sync into the CuratedHub.

        Algorithm:

            First, look for a match of model name in Hub. If no model is found, sync that model.

            Next, compare versions of model to sync and what's in the Hub. If version already
            in Hub, don't sync. If newer version in Hub, don't sync. If older version in Hub,
            sync that model.
        """
        models_to_sync = []
        for model in model_list:
            matched_model = models_in_hub.get(model.model_id)

            # Model does not exist in Hub, sync
            if not matched_model:
                models_to_sync.append(model)

            if matched_model:
                model_version = Version(model.version)
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

        return models_to_sync

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

        # Retrieve latest version of unspecified JumpStart model versions
        model_version_list = []
        for model in model_list:
            version = model.get("version", "*")
            if version == "*":
                model = self._populate_latest_model_version(model)
                JUMPSTART_LOGGER.warning(
                    "No version specified for model %s. Using version %s",
                    model["model_id"],
                    model["version"],
                )
            model_version_list.append(JumpStartModelInfo(model["model_id"], model["version"]))

        js_models_in_hub = self._get_jumpstart_models_in_hub()
        mapped_models_in_hub = {model["name"]: model for model in js_models_in_hub}

        models_to_sync = self._determine_models_to_sync(model_version_list, mapped_models_in_hub)
        JUMPSTART_LOGGER.warning(
            "Syncing the following models into Hub %s: %s", self.hub_name, models_to_sync
        )

        # Delete old models?

        # CopyContentWorkflow + `SageMaker:ImportHubContent` for each model-to-sync in parallel
        tasks: List[futures.Future] = []
        with futures.ThreadPoolExecutor(
            max_workers=self._default_thread_pool_size,
            thread_name_prefix="import-models-to-curated-hub",
        ) as import_executor:
            for thread_num, model in enumerate(models_to_sync):
                task = import_executor.submit(self._sync_public_model_to_hub, model, thread_num)
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

    def _sync_public_model_to_hub(self, model: JumpStartModelInfo, thread_num: int):
        """Syncs a public JumpStart model version to the Hub. Runs in parallel."""
        model_specs = utils.verify_model_region_and_return_specs(
            model_id=model.model_id,
            version=model.version,
            region=self.region,
            scope=JumpStartScriptScope.INFERENCE,
            sagemaker_session=self._sagemaker_session,
        )
        studio_specs = self._fetch_studio_specs(model_specs=model_specs)

        dest_location = S3ObjectLocation(
            bucket=self.hub_storage_location.bucket,
            key=f"{self.hub_storage_location.key}/curated_models/{model.model_id}/{model.version}",
        )
        src_files = file_generator.generate_file_infos_from_model_specs(
            model_specs, studio_specs, self.region, self._s3_client
        )
        dest_files = file_generator.generate_file_infos_from_s3_location(
            dest_location, self._s3_client
        )

        comparator = SizeAndLastUpdatedComparator()
        sync_request = HubSyncRequestFactory(
            src_files, dest_files, dest_location, comparator
        ).create()

        if len(sync_request.files) > 0:
            MultiPartCopyHandler(
                thread_num=thread_num,
                sync_request=sync_request,
                region=self.region,
                label=dest_location.key,
            ).execute()
        else:
            JUMPSTART_LOGGER.warning("Nothing to copy for %s v%s", model.model_id, model.version)

        # TODO: Tag model if specs say it is deprecated or training/inference
        # vulnerable. Update tag of HubContent ARN without version.
        # Versioned ARNs are not onboarded to Tagris.
        tags = []

        search_keywords = [
            f"{JUMPSTART_HUB_MODEL_ID_TAG_PREFIX}:{model.model_id}",
            f"{JUMPSTART_HUB_MODEL_VERSION_TAG_PREFIX}:{model.version}",
            f"{FRAMEWORK_TAG_PREFIX}:{model_specs.get_framework()}",
            f"{TASK_TAG_PREFIX}:TODO: pull from specs",
        ]

        hub_content_document = str(HubContentDocument_v2(spec=model_specs))

        self._sagemaker_session.import_hub_content(
            document_schema_version=HubContentDocument_v2.SCHEMA_VERSION,
            hub_content_name=model.model_id,
            hub_content_version=model.version,
            hub_name=self.hub_name,
            hub_content_document=hub_content_document,
            hub_content_type=HubContentType.MODEL,
            hub_content_display_name="",
            hub_content_description="",
            hub_content_markdown="",
            hub_content_search_keywords=search_keywords,
            tags=tags,
        )

    def _fetch_studio_specs(self, model_specs: JumpStartModelSpecs) -> Dict[str, Any]:
        """Fetches StudioSpecs given a model's SDK Specs."""
        model_id = model_specs.model_id
        model_version = model_specs.version

        key = utils.generate_studio_spec_file_prefix(model_id, model_version)
        response = self._s3_client.get_object(
            Bucket=utils.get_jumpstart_content_bucket(self.region), Key=key
        )
        return json.loads(response["Body"].read().decode("utf-8"))


    def scan_and_tag_models(self) -> None:
        """Scans the Hub for JumpStart models and tags the HubContent.
        
        If the scan detects a model is deprecated or vulnerable, it will tag the HubContent.
        The tags that will be added are based off the specifications in the JumpStart public hub:
        1. "deprecated_versions" -> If the public hub model is deprecated
        2. "inference_vulnerable_versions" -> If the public hub model has inference vulnerabilities
        3. "training_vulnerable_versions" -> If the public hub model has training vulnerabilities

        The tag value will be a list of versions in the Curated Hub that fall under those keys. 
        For example, if model_a version_a is deprecated and inference is vulnerable, the
        HubContent for `model_a` will have tags [{"deprecated_versions": [version_a]}, 
        {"inference_vulnerable_versions": [version_a]}]
        """
        JUMPSTART_LOGGER.info(
            "Tagging models in hub: %s", self.hub_name
        )
        models_in_hub: List[Dict[str, Any]] = self._get_jumpstart_models_in_hub()
        tags = tag_jumpstart_hub_content_on_spec_fields(models_in_hub, self.region, self._sagemaker_session)

        output_string = "No tags were added!"
        if len(tags) > 0:
          output_string = f"Added the following tags: {tags}"
        JUMPSTART_LOGGER.info(output_string)