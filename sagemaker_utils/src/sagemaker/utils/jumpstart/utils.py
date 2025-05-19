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
"""This module contains utils for JumpStart."""
from __future__ import absolute_import

from typing import Optional
from sagemaker_core.helper.session_helper import Session
from sagemaker.utils.jumpstart.models import HubContentDocument


def get_eula_url(document: HubContentDocument, sagemaker_session: Optional[Session] = None) -> str:
    """Get the EULA URL from the HubContentDocument.

    Args:
        document (HubContentDocument): The HubContentDocument object.
        sagemaker_session (Optional[Session]): SageMaker session. Defaults to None.
    Returns:
        str: The EULA URL.
    """
    if not document.HostingEulaUri:
        return ""
    if sagemaker_session is None:
        sagemaker_session = Session()

    path_parts = document.HostingEulaUri.replace("s3://", "").split("/")

    bucket = path_parts[0]
    key = "/".join(path_parts[1:])
    region = sagemaker_session.boto_region_name

    botocore_session = sagemaker_session.boto_session._session
    endpoint_resolver = botocore_session.get_component("endpoint_resolver")
    partition = endpoint_resolver.get_partition_for_region(region)
    dns_suffix = endpoint_resolver.get_partition_dns_suffix(partition)

    return f"https://{bucket}.s3.{region}.{dns_suffix}/{key}"
