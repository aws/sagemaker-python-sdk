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
"""This module contains utilites for JumpStart model metadata."""

from __future__ import absolute_import

import json
from typing import Optional, Tuple
from functools import lru_cache
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.utils.utils import logger
from sagemaker.core.resources import HubContent
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.jumpstart.models import HubContentDocument
from sagemaker.core.jumpstart.constants import SAGEMAKER_PUBLIC_HUB


@lru_cache(maxsize=128)
def get_hub_content_and_document(
    jumpstart_config: JumpStartConfig,
    sagemaker_session: Optional[Session] = None,
) -> Tuple[HubContent, HubContentDocument]:
    """Get model metadata for JumpStart.


    Args:
        jumpstart_config (JumpStartConfig): JumpStart configuration.
        sagemaker_session (Session, optional): SageMaker session.
            Defaults to None.

    Returns:
        HubContentDocument: Model metadata.
    """
    if sagemaker_session is None:
        sagemaker_session = Session()
        logger.debug("No sagemaker session provided. Using default session.")

    hub_name = jumpstart_config.hub_name if jumpstart_config.hub_name else SAGEMAKER_PUBLIC_HUB

    region = sagemaker_session.boto_region_name

    # The hub content may be filed under an alias that differs from the public
    # model_id, so honor hub_content_name when provided.
    hub_content_name = (
        jumpstart_config.hub_content_name
        if getattr(jumpstart_config, "hub_content_name", None)
        else jumpstart_config.model_id
    )

    # A private hub can contain either a ModelReference (a pointer to a public
    # JumpStart model) or a privately-owned Model authored directly into the
    # hub. We cannot tell which from the name alone, so probe: try
    # ModelReference first, then fall back to Model. The public hub only holds
    # Models. This mirrors ModelBuilder's resolution in accessors.py.
    if hub_name == SAGEMAKER_PUBLIC_HUB:
        content_types_to_try = ["Model"]
    else:
        content_types_to_try = ["ModelReference", "Model"]

    hub_content = None
    last_error: Optional[ClientError] = None
    for content_type in content_types_to_try:
        try:
            hub_content = HubContent.get(
                hub_name=hub_name,
                hub_content_name=hub_content_name,
                hub_content_version=jumpstart_config.model_version,
                hub_content_type=content_type,
                session=sagemaker_session.boto_session,
                region=region,
            )
            break
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFound":
                last_error = e
                continue
            raise e

    if hub_content is None:
        logger.error(
            f"Hub content {hub_content_name} not found in {hub_name} as any of "
            f"{content_types_to_try}.\n"
            "Please check that the Model ID (or hub_content_name) is available "
            "in the specified hub."
        )
        raise last_error

    logger.info(
        f"hub_content_name: {hub_content.hub_content_name}, "
        f"hub_content_version: {hub_content.hub_content_version}"
    )
    document_json = json.loads(hub_content.hub_content_document)
    return (hub_content, HubContentDocument(**document_json))
