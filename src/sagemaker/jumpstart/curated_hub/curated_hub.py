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
from sagemaker.s3 import s3_path_join
from sagemaker.jumpstart import utils
from sagemaker.jumpstart.curated_hub.accessors import file_generator
from sagemaker.jumpstart.curated_hub.accessors.multipartcopy import MultiPartCopyHandler
from sagemaker.jumpstart.curated_hub.constants import (
    JUMPSTART_CURATED_HUB_MODEL_TAG,
    LATEST_VERSION_WILDCARD,
)
from sagemaker.jumpstart.curated_hub.sync.comparator import SizeAndLastUpdatedComparator
from sagemaker.jumpstart.curated_hub.sync.request import HubSyncRequestFactory
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_LOGGER,
)
from sagemaker.jumpstart.types import (
    HubContentType,
    JumpStartModelSpecs,
)
from sagemaker.jumpstart.curated_hub.utils import (
    create_hub_bucket_if_it_does_not_exist,
    generate_default_hub_bucket_name,
    create_s3_object_reference_from_uri,
    get_jumpstart_model_and_version,
    find_deprecated_vulnerable_flags_for_hub_content,
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

    def list_models(self, clear_cache: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """Lists the models in this Curated Hub.

        This function caches the models in local memory

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
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

    def _fetch_studio_specs(self, model_specs: JumpStartModelSpecs) -> Dict[str, Any]:
        """Fetches StudioSpecs given a model's SDK Specs."""
        model_id = model_specs.model_id
        model_version = model_specs.version

        key = utils.generate_studio_spec_file_prefix(model_id, model_version)
        response = self._s3_client.get_object(
            Bucket=utils.get_jumpstart_content_bucket(self.region), Key=key
        )
        return json.loads(response["Body"].read().decode("utf-8"))

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
            model
            for model in model_summaries_to_scan
            if get_jumpstart_model_and_version(model) is not None
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
