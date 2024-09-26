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

from __future__ import absolute_import
from sagemaker.mlflow.tracking_server import generate_mlflow_presigned_url


def test_generate_presigned_url(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_presigned_mlflow_tracking_server_url.return_value = {
        "AuthorizedUrl": "https://t-wo.example.com",
    }
    url = generate_mlflow_presigned_url(
        "w",
        expires_in_seconds=10,
        session_expiration_duration_in_seconds=5,
        sagemaker_session=sagemaker_session,
    )
    client.create_presigned_mlflow_tracking_server_url.assert_called_with(
        TrackingServerName="w", ExpiresInSeconds=10, SessionExpirationDurationInSeconds=5
    )
    assert url == "https://t-wo.example.com"


def test_generate_presigned_url_minimal(sagemaker_session):
    client = sagemaker_session.sagemaker_client
    client.create_presigned_mlflow_tracking_server_url.return_value = {
        "AuthorizedUrl": "https://t-wo.example.com",
    }
    url = generate_mlflow_presigned_url("w", sagemaker_session=sagemaker_session)
    client.create_presigned_mlflow_tracking_server_url.assert_called_with(TrackingServerName="w")
    assert url == "https://t-wo.example.com"
