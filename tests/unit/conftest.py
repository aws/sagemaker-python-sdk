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


import pytest

import sagemaker

from mock import Mock, PropertyMock

_ROLE = "DummyRole"
_REGION = "us-west-2"
_DEFAULT_BUCKET = "my-bucket"


@pytest.fixture(scope="session")
def client():
    """Mock client.

    Considerations when appropriate:

         * utilize botocore.stub.Stubber
         * separate runtime client from client
    """
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.8.5 Linux/5.4.0-42-generic Botocore/1.17.24 Resource"
    )
    return client_mock


@pytest.fixture(scope="session")
def boto_session(client):
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=_ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=_REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = client

    return session_mock


@pytest.fixture(scope="session")
def sagemaker_session(boto_session, client):
    # ideally this would mock Session instead of instantiating it
    # most unit tests do mock the session correctly
    session = sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=client,
        sagemaker_runtime_client=client,
        default_bucket=_DEFAULT_BUCKET,
        sagemaker_metrics_client=client,
    )
    return session
