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
# pylint: skip-file
"""This module provides the JumpStart Hub class."""
from __future__ import absolute_import
from datetime import datetime
import logging
from typing import Optional, Dict, List, Any, Union

from sagemaker.core.jumpstart.constants import JUMPSTART_MODEL_HUB_NAME
from sagemaker.core.jumpstart.enums import JumpStartScriptScope
from sagemaker.core.helper.session_helper import Session

from sagemaker.core.jumpstart.types import (
    HubContentType,
)
from sagemaker.core.jumpstart.filters import Constant, Operator, BooleanValues
from sagemaker.core.jumpstart.hub.utils import (
    get_hub_model_version,
    get_info_from_hub_resource_arn,
    construct_hub_arn_from_name,
)

from sagemaker.core.jumpstart.notebook_utils import (
    list_jumpstart_models,
)

from sagemaker.core.jumpstart.hub.interfaces import (
    DescribeHubResponse,
    DescribeHubContentResponse,
)
from sagemaker.core.jumpstart.hub.constants import (
    LATEST_VERSION_WILDCARD,
)
from sagemaker.core.jumpstart import utils


class Hub:
    """Class for creating and managing a curated JumpStart hub"""

    # Setting LOGGER for backward compatibility, in case users import it...
    logger = LOGGER = logging.getLogger("sagemaker")

    _list_hubs_cache: List[Dict[str, Any]] = []

    def __init__(
        self,
        hub_name: str,
        sagemaker_session: Session,
        bucket_name: Optional[str] = None,
    ) -> None:
        """Instantiates a SageMaker ``Hub``.

        Args:
            hub_name (str): The name of the Hub to create.
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions.
        """
        self.hub_name = hub_name
        self.region = sagemaker_session.boto_region_name
        self.bucket_name = bucket_name
        self._sagemaker_session = (
            sagemaker_session
            or utils.get_default_jumpstart_session_with_user_agent_suffix(is_hub_content=True)
        )

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
        curr_timestamp = datetime.now().timestamp()

        request = {
            "hub_name": self.hub_name,
            "hub_description": description,
            "hub_display_name": display_name,
            "hub_search_keywords": search_keywords,
            "tags": tags,
        }

        if self.bucket_name:
            request["s3_storage_config"] = {
                "S3OutputPath": (f"s3://{self.bucket_name}/{self.hub_name}-{curr_timestamp}")
            }

        return self._sagemaker_session.create_hub(**request)

    def describe(self, hub_name: Optional[str] = None) -> DescribeHubResponse:
        """Returns descriptive information about the Hub"""

        hub_description: DescribeHubResponse = self._sagemaker_session.describe_hub(
            hub_name=self.hub_name if not hub_name else hub_name
        )

        return hub_description

    def _list_and_paginate_models(self, **kwargs) -> List[Dict[str, Any]]:
        """List and paginate models from Hub."""
        next_token: Optional[str] = None
        first_iteration: bool = True
        hub_model_summaries: List[Dict[str, Any]] = []

        while first_iteration or next_token:
            first_iteration = False
            list_hub_content_response = self._sagemaker_session.list_hub_contents(**kwargs)
            hub_model_summaries.extend(list_hub_content_response.get("HubContentSummaries", []))
            next_token = list_hub_content_response.get("NextToken")

        return hub_model_summaries

    def list_models(self, clear_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """Lists the models and model references in this SageMaker Hub.

        This function caches the models in local memory

        **kwargs: Passed to invocation of ``Session:list_hub_contents``.
        """
        response = {}

        if clear_cache:
            self._list_hubs_cache = None
        if self._list_hubs_cache is None:

            hub_model_reference_summaries = self._list_and_paginate_models(
                **{
                    "hub_name": self.hub_name,
                    "hub_content_type": HubContentType.MODEL_REFERENCE.value,
                    **kwargs,
                }
            )

            hub_model_summaries = self._list_and_paginate_models(
                **{
                    "hub_name": self.hub_name,
                    "hub_content_type": HubContentType.MODEL.value,
                    **kwargs,
                }
            )
            response["hub_content_summaries"] = hub_model_reference_summaries + hub_model_summaries
        response["next_token"] = None  # Temporary until pagination is implemented
        return response

    def list_sagemaker_public_hub_models(
        self,
        filter: Union[Operator, str] = Constant(BooleanValues.TRUE),
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Lists the models and model arns from AmazonSageMakerJumpStart Public Hub.

        Args:
        filter (Union[Operator, str]): Optional. The filter to apply to list models. This can be
            either an ``Operator`` type filter (e.g. ``And("task == ic", "framework == pytorch")``),
            or simply a string filter which will get serialized into an Identity filter.
            (e.g. ``"task == ic"``). If this argument is not supplied, all models will be listed.
            (Default: Constant(BooleanValues.TRUE)).
        next_token (str): Optional. A token to resume pagination of list_inference_components.
            This is currently not implemented.
        """

        response = {}

        jumpstart_public_hub_arn = construct_hub_arn_from_name(
            JUMPSTART_MODEL_HUB_NAME, self.region, self._sagemaker_session
        )

        hub_content_summaries = []
        models = list_jumpstart_models(filter=filter, list_versions=True)
        for model in models:
            if len(model) <= 63:
                info = get_info_from_hub_resource_arn(jumpstart_public_hub_arn)
                hub_model_arn = (
                    f"arn:{info.partition}:"
                    f"sagemaker:{info.region}:"
                    f"aws:hub-content/{info.hub_name}/"
                    f"{HubContentType.MODEL.value}/{model[0]}"
                )
                hub_content_summary = {
                    "hub_content_name": model[0],
                    "hub_content_arn": hub_model_arn,
                }
                hub_content_summaries.append(hub_content_summary)
        response["hub_content_summaries"] = hub_content_summaries

        response["next_token"] = None  # Temporary until pagination is implemented for this function

        return response

    def delete(self) -> None:
        """Deletes this SageMaker Hub."""
        return self._sagemaker_session.delete_hub(self.hub_name)

    def create_model_reference(
        self, model_arn: str, model_name: Optional[str] = None, min_version: Optional[str] = None
    ):
        """Adds model reference to this SageMaker Hub."""
        return self._sagemaker_session.create_hub_content_reference(
            hub_name=self.hub_name,
            source_hub_content_arn=model_arn,
            hub_content_name=model_name,
            min_version=min_version,
        )

    def delete_model_reference(self, model_name: str) -> None:
        """Deletes model reference from this SageMaker Hub."""
        return self._sagemaker_session.delete_hub_content_reference(
            hub_name=self.hub_name,
            hub_content_type=HubContentType.MODEL_REFERENCE.value,
            hub_content_name=model_name,
        )

    def describe_model(
        self, model_name: str, hub_name: Optional[str] = None, model_version: Optional[str] = None
    ) -> DescribeHubContentResponse:
        """Describe Model or ModelReference in a Hub."""
        hub_name = hub_name or self.hub_name

        # Users only input model id, not contentType, so first try to describe with ModelReference, then with Model
        try:
            model_version = get_hub_model_version(
                hub_model_name=model_name,
                hub_model_type=HubContentType.MODEL_REFERENCE.value,
                hub_name=hub_name,
                sagemaker_session=self._sagemaker_session,
                hub_model_version=model_version,
            )

            hub_content_description: Dict[str, Any] = self._sagemaker_session.describe_hub_content(
                hub_name=hub_name,
                hub_content_name=model_name,
                hub_content_version=model_version,
                hub_content_type=HubContentType.MODEL_REFERENCE.value,
            )

        except Exception as ex:
            logging.info(
                "Received exeption while calling APIs for ContentType ModelReference, retrying with ContentType Model: "
                + str(ex)
            )

            # Failed to describe ModelReference, try with Model
            try:
                model_version = get_hub_model_version(
                    hub_model_name=model_name,
                    hub_model_type=HubContentType.MODEL.value,
                    hub_name=hub_name,
                    sagemaker_session=self._sagemaker_session,
                    hub_model_version=model_version,
                )

                hub_content_description: Dict[str, Any] = (
                    self._sagemaker_session.describe_hub_content(
                        hub_name=hub_name,
                        hub_content_name=model_name,
                        hub_content_version=model_version,
                        hub_content_type=HubContentType.MODEL.value,
                    )
                )

            except Exception as ex:
                # Failed with both, throw a custom error message
                raise RuntimeError(
                    f"Cannot get details for {model_name} in Hub {hub_name}. \
                        {model_name} does not exist as a Model or ModelReference in {hub_name}: \n"
                    + str(ex)
                )

        return DescribeHubContentResponse(hub_content_description)
