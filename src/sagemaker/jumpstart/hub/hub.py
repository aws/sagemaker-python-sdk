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
from datetime import datetime
from typing import Optional, Dict, List, Any
from botocore import exceptions

from sagemaker.jumpstart.hub.constants import JUMPSTART_MODEL_HUB_NAME
from sagemaker.jumpstart.enums import JumpStartScriptScope
from sagemaker.session import Session

from sagemaker.jumpstart.constants import (
    DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    JUMPSTART_LOGGER,
)
from sagemaker.jumpstart.types import (
    HubContentType,
)
from sagemaker.jumpstart.hub.utils import (
    create_hub_bucket_if_it_does_not_exist,
    generate_default_hub_bucket_name,
    create_s3_object_reference_from_uri,
)

from sagemaker.jumpstart.hub.types import (
    S3ObjectLocation,
)
from sagemaker.jumpstart.hub.interfaces import (
    DescribeHubContentResponse,
)
from sagemaker.jumpstart.hub.constants import (
    LATEST_VERSION_WILDCARD,
)
from sagemaker.jumpstart import utils


class Hub:
    """Class for creating and managing a curated JumpStart hub"""

    _list_hubs_cache: Dict[str, Any] = None

    def __init__(
        self,
        hub_name: str,
        bucket_name: Optional[str] = None,
        sagemaker_session: Optional[Session] = DEFAULT_JUMPSTART_SAGEMAKER_SESSION,
    ) -> None:
        """Instantiates a SageMaker ``Hub``.

        Args:
            hub_name (str): The name of the Hub to create.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.
        """
        self.hub_name = hub_name
        self.region = sagemaker_session.boto_region_name
        self._sagemaker_session = sagemaker_session
        self.hub_storage_location = self._generate_hub_storage_location(bucket_name)

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
    
    def _get_latest_model_version(self, model_id: str) -> str:
            """Populates the lastest version of a model from specs no matter what is passed.

            Returns model ({ model_id: str, version: str })
            """
            model_specs = utils.verify_model_region_and_return_specs(
                model_id, LATEST_VERSION_WILDCARD, JumpStartScriptScope.INFERENCE, self.region
            )
            return model_specs.version

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

    def describe(self) -> Dict[str, Any]:
        """Returns descriptive information about the Hub"""

        hub_description = self._sagemaker_session.describe_hub(
            hub_name=self.hub_name
        )

        return hub_description
    
    def list_models(self, clear_cache: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """Lists the models and model references in this Curated Hub.

        This function caches the models in local memory

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        if clear_cache:
            self._list_hubs_cache = None
        if self._list_hubs_cache is None:
            hub_content_summaries = self._sagemaker_session.list_hub_contents(
                hub_name=self.hub_name, hub_content_type=HubContentType.MODEL_REFERENCE.value, **kwargs
            )
            hub_content_summaries.update(self._sagemaker_session.list_hub_contents(
                hub_name=self.hub_name, hub_content_type=HubContentType.MODEL.value, **kwargs
            ))
            self._list_hubs_cache = hub_content_summaries
        return self._list_hubs_cache
    
    # TODO: Update to use S3 source for listing the public models
    def list_jumpstart_service_hub_models(self, filter_name: Optional[str] = None, clear_cache: bool = True, **kwargs) -> List[Dict[str, Any]]:
        """Lists the models from AmazonSageMakerJumpStart Public Hub.

        This function caches the models in local memory

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        if clear_cache:
            self._list_hubs_cache = None
        if self._list_hubs_cache is None:
            hub_content_summaries = self._sagemaker_session.list_hub_contents(
                hub_name=JUMPSTART_MODEL_HUB_NAME, 
                hub_content_type=HubContentType.MODEL_REFERENCE.value, 
                name_contains=filter_name, 
                **kwargs
            )
            self._list_hubs_cache = hub_content_summaries
        return self._list_hubs_cache

    def delete(self) -> None:
        """Deletes this Curated Hub"""
        return self._sagemaker_session.delete_hub(self.hub_name)

    def create_model_reference(
        self, model_arn: str, model_name: Optional[str], min_version: Optional[str] = None
    ):
        """Adds model reference to this Curated Hub"""
        return self._sagemaker_session.create_hub_content_reference(
            hub_name=self.hub_name,
            source_hub_content_arn=model_arn,
            hub_content_name=model_name,
            min_version=min_version,
        )

    def delete_model_reference(self, model_name: str) -> None:
        """Deletes model reference from this Curated Hub"""
        return self._sagemaker_session.delete_hub_content_reference(
            hub_name=self.hub_name,
            hub_content_type=HubContentType.MODEL_REFERENCE.value,
            hub_content_name=model_name,
        )
    
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
    
    def describe_model_reference(
        self, model_name: str, model_version: Optional[str] = None
    ) -> DescribeHubContentResponse:
        """Returns descriptive information about the Hub Model"""
        if model_version == LATEST_VERSION_WILDCARD or model_version is None:
            model_version = self._get_latest_model_version(model_name)
        hub_content_description: Dict[str, Any] = self._sagemaker_session.describe_hub_content(
            hub_name=self.hub_name,
            hub_content_name=model_name,
            hub_content_version=model_version,
            hub_content_type=HubContentType.MODEL_REFERENCE.value,
        )

        return DescribeHubContentResponse(hub_content_description)