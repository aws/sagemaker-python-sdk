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

from sagemaker.utils import TagsDict
from sagemaker.session import Session
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.curated_hub.accessors import file_generator
from sagemaker.jumpstart.curated_hub.accessors.multipartcopy import MultiPartCopyHandler
from sagemaker.jumpstart.curated_hub.constants import (
    JUMPSTART_CURATED_HUB_MODEL_TAG,
    LATEST_VERSION_WILDCARD,
)
from sagemaker.jumpstart.curated_hub.sync.comparator import SizeAndLastUpdatedComparator
from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequest, HubSyncRequestFactory
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_DEFAULT_STUDIO_MANIFEST_KEY,
    JUMPSTART_LOGGER,
    STUDIO_MODEL_ID_KEY,
    STUDIO_SPEC_PATH_KEY_IN_MANIFEST,
)
from sagemaker.jumpstart.types import (
    HubContentType,
)
from sagemaker.jumpstart.curated_hub.utils import (
    create_hub_bucket_if_it_does_not_exist,
    generate_default_hub_bucket_name,
    create_s3_object_reference_from_uri,
    find_deprecated_vulnerable_flags_for_hub_content,
    is_curated_jumpstart_model,
)
from sagemaker.jumpstart.curated_hub.interfaces import (
    DescribeHubResponse,
    DescribeHubContentResponse,
    HubModelDocument,
    HubContentInfo,
    HubContentDependency,
)
from sagemaker.jumpstart.curated_hub.types import (
    HubContentDependencyType,
    HubContentReferenceType,
    JumpStartModelInfo,
    S3ObjectLocation,
    summary_list_from_list_api_response,
)
from sagemaker.jumpstart.curated_hub.parsers import make_hub_model_document_from_specs


class CuratedHub:
    """Class for creating and managing a curated JumpStart hub"""

    _list_hubs_cache: Dict[str, Any] = None

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
        self.studio_manifest = self._fetch_manifest_from_s3(JUMPSTART_DEFAULT_STUDIO_MANIFEST_KEY)

    def _get_s3_client(self) -> BaseClient:
        """Returns an S3 client used for creating a HubContentDocument."""
        return boto3.client("s3", region_name=self.region)

    def _fetch_hub_storage_location(self) -> S3ObjectLocation:
        """Retrieves hub bucket name from Hub config if exists"""
        try:
            hub_response = self._sagemaker_session.describe_hub(hub_name=self.hub_name)
            hub_output_location = hub_response["S3StorageConfig"].get("S3OutputPath")

            if hub_output_location:
                location = create_s3_object_reference_from_uri(hub_output_location)
                return location
            default_bucket_name = generate_default_hub_bucket_name(self._sagemaker_session)
            curr_timestamp = datetime.now().timestamp()
            JUMPSTART_LOGGER.warning(
                "There is not a Hub bucket associated with %s. Using %s",
                self.hub_name,
                default_bucket_name,
            )
            return S3ObjectLocation(
                bucket=default_bucket_name, key=f"{self.hub_name}-{curr_timestamp}"
            )
        except exceptions.ClientError:
            hub_bucket_name = generate_default_hub_bucket_name(self._sagemaker_session)
            curr_timestamp = datetime.now().timestamp()
            JUMPSTART_LOGGER.warning(
                "There is not a Hub bucket associated with %s. Using %s",
                self.hub_name,
                hub_bucket_name,
            )
            return S3ObjectLocation(bucket=hub_bucket_name, key=f"{self.hub_name}-{curr_timestamp}")

    def _generate_hub_storage_location(self, bucket_name: Optional[str] = None) -> None:
        """Generates an ``S3ObjectLocation`` given a Hub name."""
        curr_timestamp = datetime.now().timestamp()
        return (
            S3ObjectLocation(bucket=bucket_name, key=f"{self.hub_name}-{curr_timestamp}")
            if bucket_name
            else self._fetch_hub_storage_location()
        )

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

    def list_models(self, clear_cache: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """Lists the models in this Curated Hub.

        This function caches the models in local memory

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        JUMPSTART_LOGGER.info("Listing models in %s", self.hub_name)
        if clear_cache:
            self._list_hubs_cache = None
        if self._list_hubs_cache is None:
            hub_content_summaries = self._sagemaker_session.list_hub_contents(
                hub_name=self.hub_name, hub_content_type=HubContentType.MODEL.value, **kwargs
            )
            self._list_hubs_cache = hub_content_summaries
        return self._list_hubs_cache

    def describe_model(
        self, model_name: str, model_version: Optional[str] = None
    ) -> DescribeHubContentResponse:
        """Returns descriptive information about the Hub Model"""
        if model_version == LATEST_VERSION_WILDCARD or model_version is None:
            model_version = self._get_latest_model_version(model_name)
        hub_content_description: Dict[str, Any] = self._sagemaker_session.describe_hub_content(
            hub_name=self.hub_name,
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL.value,
        )

        return DescribeHubContentResponse(hub_content_description)

    def delete_model(self, model_name: str, model_version: Optional[str] = None) -> None:
        """Deletes a model from this CuratedHub."""
        if model_version == LATEST_VERSION_WILDCARD or model_version is None:
            model_version = self._get_latest_model_version(model_name)
        return self._sagemaker_session.delete_hub_content(
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL.value,
            hub_name=self.hub_name,
        )

    def delete(self) -> None:
        """Deletes this Curated Hub"""
        return self._sagemaker_session.delete_hub(self.hub_name)

    def _is_invalid_model_list_input(self, model_list: List[Dict[str, str]]) -> bool:
        """Determines if input args to ``sync`` is correct.

        `model_list` objects must have `model_id` (str) and optional `version` (str).
        """
        if model_list is None:
            return True
        for obj in model_list:
            if not isinstance(obj.get("model_id"), str):
                return True
            if "version" in obj and not isinstance(obj["version"], str):
                return True
        return False

    def _get_latest_model_version(self, model_id: str) -> str:
        """Populates the lastest version of a model from specs no matter what is passed.

        Returns model ({ model_id: str, version: str })
        """
        model_specs = utils.verify_model_region_and_return_specs(
            model_id, LATEST_VERSION_WILDCARD, JumpStartScriptScope.INFERENCE, self.region
        )
        return model_specs.version

    def _populate_latest_model_version(self, model: Dict[str, str]) -> Dict[str, str]:
        """Populates the lastest version of a model from specs no matter what is passed.

        Returns model ({ model_id: str, version: str })
        """
        model_version = self._get_latest_model_version(model["model_id"])
        return {"model_id": model["model_id"], "version": model_version}

    def _get_jumpstart_models_in_hub(self) -> List[HubContentInfo]:
        """Retrieves all JumpStart models in a private Hub."""
        hub_models = summary_list_from_list_api_response(self.list_models())
        return [model for model in hub_models if is_curated_jumpstart_model(model) is True]

    def _determine_models_to_sync(
        self,
        model_list: List[JumpStartModelInfo],
        models_in_hub: Dict[str, HubContentInfo],
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
                hub_model_version = Version(matched_model.hub_content_version)

                # 1. Model version exists in Hub, pass
                if hub_model_version == model_version:
                    JUMPSTART_LOGGER.info(
                        "Model %s/%s already exists in your Hub %s and will not be synced.",
                        model.model_id,
                        model.version,
                        self.hub_name,
                    )
                    continue

                # 2. Invalid model version exists in Hub, pass
                # This will only happen if something goes wrong in our metadata
                if hub_model_version > model_version:
                    continue

                # 3. Old model version exists in Hub, update
                if hub_model_version < model_version:
                    # Check minSDKVersion against current SDK version, emit log
                    models_to_sync.append(model)

        return models_to_sync

    def _reference_type_to_dependency_type(
        self, reference_type: HubContentReferenceType
    ) -> Optional[HubContentDependencyType]:
        """Returns a corresponding DependencyType for a given ReferenceType."""

        dependency_type = None
        if reference_type in [
            HubContentReferenceType.INFERENCE_ARTIFACT,
            HubContentReferenceType.TRAINING_ARTIFACT,
        ]:
            dependency_type = HubContentDependencyType.ARTIFACT
        elif reference_type in [
            HubContentReferenceType.INFERENCE_SCRIPT,
            HubContentReferenceType.TRAINING_SCRIPT,
        ]:
            dependency_type = HubContentDependencyType.SCRIPT
        elif reference_type in [
            HubContentReferenceType.DEFAULT_TRAINING_DATASET,
        ]:
            dependency_type = HubContentDependencyType.DATASET
        elif reference_type in [
            HubContentReferenceType.INFERENCE_NOTEBOOK,
        ]:
            dependency_type = HubContentDependencyType.NOTEBOOK
        elif reference_type in [
            HubContentReferenceType.MARKDOWN,
        ]:
            dependency_type = HubContentDependencyType.OTHER
        return dependency_type

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
            version = model.get("version")
            if version == LATEST_VERSION_WILDCARD or version is None:
                model = self._populate_latest_model_version(model)
            model_version_list.append(JumpStartModelInfo(model["model_id"], model["version"]))

        # TODO: Flip this logic. We should 1/ get Hub models that align with inputted
        # name/version, then 2. Check if they are JumpStart models. Elsewhere, we can
        # check if the JumpStart models in Hub are deprecated/vulnerable
        js_models_in_hub = self._get_jumpstart_models_in_hub()
        mapped_models_in_hub = {model.hub_content_name: model for model in js_models_in_hub}

        models_to_sync = self._determine_models_to_sync(model_version_list, mapped_models_in_hub)
        JUMPSTART_LOGGER.warning(
            "Syncing the following models into Hub %s: %s",
            self.hub_name,
            models_to_sync,
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
                f"Failures when importing models to curated hub in parallel: {failed_imports}."
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
        studio_manifest_entry = self.studio_manifest.get(model.model_id)
        if not studio_manifest_entry:
            raise KeyError(f"Could not find model entry {model.model_id} in studio manifest.")
        studio_specs = self._fetch_studio_specs(
            studio_manifest_entry[STUDIO_SPEC_PATH_KEY_IN_MANIFEST]
        )

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
            JUMPSTART_LOGGER.info("Nothing to copy for model %s/%s.", model.model_id, model.version)

        # TODO: Tag model if specs say it is deprecated or training/inference
        # vulnerable. Update tag of HubContent ARN without version.
        # Versioned ARNs are not onboarded to Tagris.
        tags = []

        search_keywords = [JUMPSTART_CURATED_HUB_MODEL_TAG]

        dependencies = self._calculate_dependencies(sync_request)

        hub_content_document: HubModelDocument = make_hub_model_document_from_specs(
            model_specs=model_specs,
            studio_manifest_entry=studio_manifest_entry,
            studio_specs=studio_specs,
            files=src_files,
            dest_location=dest_location,
            hub_content_dependencies=dependencies,
            region=self.region,
        )

        JUMPSTART_LOGGER.info("Importing %s/%s...", model.model_id, model.version)

        self._sagemaker_session.import_hub_content(
            document_schema_version=hub_content_document.get_schema_version(),
            hub_content_name=model.model_id,
            hub_content_version=model.version,
            hub_name=self.hub_name,
            hub_content_document=str(hub_content_document),
            hub_content_type=HubContentType.MODEL,
            hub_content_display_name="",
            hub_content_description="",
            hub_content_markdown="",
            hub_content_search_keywords=search_keywords,
            tags=tags,
        )

    def _fetch_studio_specs(self, studio_spec_path: str) -> Dict[str, Any]:
        """Fetches StudioSpec given spec path."""

        response = self._s3_client.get_object(
            Bucket=utils.get_jumpstart_content_bucket(self.region), Key=studio_spec_path
        )
        return json.loads(response["Body"].read().decode("utf-8"))

    def _fetch_manifest_from_s3(self, key: str) -> Dict[str, Dict[str, Any]]:
        """Fetches Studio manifest from S3"""
        response = self._s3_client.get_object(
            Bucket=utils.get_jumpstart_content_bucket(self.region), Key=key
        )
        manifest_list = json.loads(response["Body"].read().decode("utf-8"))
        return {entry.get(STUDIO_MODEL_ID_KEY): entry for entry in manifest_list}

    def _calculate_dependencies(self, sync_request: HubSyncRequest) -> List[HubContentDependency]:
        """Calculates dependencies for HubContentDocument"""

        files = sync_request.files
        dest_location = sync_request.destination
        dependencies: List[HubContentDependency] = []
        for file in files:
            dependencies.append(
                HubContentDependency(
                    {
                        "dependency_origin_path": f"{file.location.bucket}/{file.location.key}",
                        "depenency_copy_path": f"{dest_location.bucket}/{dest_location.key}/{file.location.key}",
                        "dependency_type": self._reference_type_to_dependency_type(
                            file.reference_type
                        ),
                    }
                )
            )

        return dependencies

    def scan_and_tag_models(self, model_ids: List[str] = None) -> None:
        """Scans the Hub for JumpStart models and tags the HubContent.

        If the scan detects a model is deprecated or vulnerable, it will tag the HubContent.
        The tags that will be added are based off the specifications in the JumpStart public hub:
        1. "deprecated_versions" -> If the public hub model is deprecated
        2. "inference_vulnerable_versions" -> If the inference script has vulnerabilities
        3. "training_vulnerable_versions" -> If the training script has vulnerabilities

        The tag value will be a list of versions in the Curated Hub that fall under those keys.
        For example, if model_a version_a is deprecated and inference is vulnerable, the
        HubContent for `model_a` will have tags [{"deprecated_versions": [version_a]},
        {"inference_vulnerable_versions": [version_a]}]

        If models are passed in, this will only scan those models if they exist in the Curated Hub.
        """
        JUMPSTART_LOGGER.info("Tagging models in hub: %s", self.hub_name)
        model_ids = model_ids if model_ids is not None else []
        if self._is_invalid_model_list_input(model_ids):
            raise ValueError(
                "Model list should be a list of objects with values 'model_id',",
                "and optional 'version'.",
            )

        models_in_hub = summary_list_from_list_api_response(self.list_models(clear_cache=False))

        model_summaries_to_scan = models_in_hub
        if model_ids:
            model_summaries_to_scan = list(
                filter(
                    lambda model_summary: model_summary.hub_content_name in model_ids, models_in_hub
                )
            )

        js_models_in_hub = [
            model for model in model_summaries_to_scan if is_curated_jumpstart_model(model) is True
        ]
        for model in js_models_in_hub:
            tags_to_add: List[TagsDict] = find_deprecated_vulnerable_flags_for_hub_content(
                hub_name=self.hub_name,
                hub_content_name=model.hub_content_name,
                region=self.region,
                session=self._sagemaker_session,
            )
            self._sagemaker_session.add_tags(ResourceArn=model.hub_content_arn, Tags=tags_to_add)
            JUMPSTART_LOGGER.info(
                "Added tags to HubContentArn %s: %s", model.hub_content_arn, tags_to_add
            )
        JUMPSTART_LOGGER.info("Tagging complete!")
