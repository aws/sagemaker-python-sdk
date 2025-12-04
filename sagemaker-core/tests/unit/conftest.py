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
"""Shared test fixtures for unit tests."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, PropertyMock

_ROLE = "arn:aws:iam::012345678901:role/SageMakerRole"
_REGION = "us-west-2"
_DEFAULT_BUCKET = "my-test-bucket"
_ACCOUNT_ID = "012345678901"


@pytest.fixture(scope="session")
def aws_credentials():
    """Mock AWS credentials for testing."""
    return {
        "aws_access_key_id": "testing",
        "aws_secret_access_key": "testing",
        "aws_session_token": "testing",
    }


@pytest.fixture(scope="session")
def sagemaker_client():
    """Mock SageMaker client."""
    client_mock = Mock()
    client_mock._client_config.user_agent = (
        "Boto3/1.14.24 Python/3.12.11 Darwin/23.0.0 Botocore/1.17.24"
    )
    return client_mock


@pytest.fixture(scope="session")
def boto_session(sagemaker_client):
    """Mock boto3 session."""
    role_mock = Mock()
    type(role_mock).arn = PropertyMock(return_value=_ROLE)

    resource_mock = Mock()
    resource_mock.Role.return_value = role_mock

    session_mock = Mock(region_name=_REGION)
    session_mock.resource.return_value = resource_mock
    session_mock.client.return_value = sagemaker_client

    return session_mock


@pytest.fixture(scope="session")
def sagemaker_session(boto_session, sagemaker_client):
    """Mock SageMaker session."""
    from sagemaker.core.helper.session_helper import Session

    # Create a mock session with all necessary attributes
    session = Mock(spec=Session)
    session.boto_session = boto_session
    session.sagemaker_client = sagemaker_client
    session.sagemaker_runtime_client = sagemaker_client
    session.sagemaker_featurestore_runtime_client = sagemaker_client
    session.sagemaker_metrics_client = sagemaker_client
    session.boto_region_name = _REGION
    session._region_name = _REGION
    session.account_id.return_value = _ACCOUNT_ID
    session.default_bucket.return_value = _DEFAULT_BUCKET
    session._default_bucket = _DEFAULT_BUCKET
    session.s3_client = sagemaker_client
    session.s3_resource = boto_session.resource.return_value

    return session


@pytest.fixture
def role():
    """Return a mock IAM role ARN."""
    return _ROLE


@pytest.fixture
def region():
    """Return a mock AWS region."""
    return _REGION


@pytest.fixture
def default_bucket():
    """Return a mock default S3 bucket."""
    return _DEFAULT_BUCKET


@pytest.fixture
def account_id():
    """Return a mock AWS account ID."""
    return _ACCOUNT_ID
