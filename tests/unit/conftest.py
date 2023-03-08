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
from jsonschema.validators import validate
from mock.mock import MagicMock

import sagemaker

from mock import Mock, PropertyMock

from sagemaker.config import SageMakerConfig
from sagemaker.config.config_schema import SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA, SCHEMA_VERSION

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
    session.get_sagemaker_config_override = Mock(
        name="get_sagemaker_config_override",
        side_effect=lambda key, default_value=None: default_value,
    )
    return session


@pytest.fixture()
def sagemaker_config_session():
    """
    Returns: a sagemaker.Session to use for tests of injection of default parameters from the
    sagemaker_config.

    This session has a custom SageMakerConfig that allows us to set the sagemaker_config.config
    dict manually. This allows us to test in unit tests without tight coupling to the exact
    sagemaker_config related helpers/utils/methods used. (And those helpers/utils/methods should
    have their own separate and specific unit tests.)

    An alternative would be to mock each call to a sagemaker_config-related method, but that would
    be harder to maintain/update over time, and be less readable.
    """

    class SageMakerConfigWithSetter(SageMakerConfig):
        """
        Version of SageMakerConfig that allows the config to be set
        """

        def __init__(self):
            self._config = {}
            # no need to call super

        @property
        def config(self) -> dict:
            return self._config

        @config.setter
        def config(self, new_config):
            """Validates and sets a new config."""
            # Add schema version if not already there since that is required
            if SCHEMA_VERSION not in new_config:
                new_config[SCHEMA_VERSION] = "1.0"
            # Validate to make sure unit tests are not accidentally testing with a wrong config
            validate(new_config, SAGEMAKER_PYTHON_SDK_CONFIG_SCHEMA)
            self._config = new_config

    boto_mock = MagicMock(name="boto_session", region_name="us-west-2")
    session_with_custom_sagemaker_config = sagemaker.Session(
        boto_session=boto_mock,
        sagemaker_client=MagicMock(),
        sagemaker_config=SageMakerConfigWithSetter(),
    )
    return session_with_custom_sagemaker_config
