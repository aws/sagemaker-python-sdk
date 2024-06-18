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


"""This module contains code related to the Mlflow Tracking Server."""

from __future__ import absolute_import
from typing import Optional, TYPE_CHECKING
from sagemaker.apiutils import _utils

if TYPE_CHECKING:
    from sagemaker import Session


def generate_mlflow_presigned_url(
    name: str,
    expires_in_seconds: Optional[int] = None,
    session_expiration_duration_in_seconds: Optional[int] = None,
    sagemaker_session: Optional["Session"] = None,
) -> str:
    """Generate a presigned url to acess the Mlflow UI.

    Args:
        name (str): Name of the Mlflow Tracking Server
        expires_in_seconds (int): Expiration time of the first usage
               of the presigned url in seconds.
        session_expiration_duration_in_seconds (int): Session duration of the presigned url in
               seconds after the first use.
        sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.
    Returns:
        (str): Authorized Url to acess the Mlflow UI.
    """
    session = sagemaker_session or _utils.default_session()
    api_response = session.create_presigned_mlflow_tracking_server_url(
        name, expires_in_seconds, session_expiration_duration_in_seconds
    )
    return api_response["AuthorizedUrl"]
