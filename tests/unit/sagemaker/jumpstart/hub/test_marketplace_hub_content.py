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
from unittest.mock import patch, MagicMock
from mock import Mock
from sagemaker.jumpstart.hub import utils as hub_utils
from sagemaker.jumpstart.enums import JumpStartModelType
from sagemaker.jumpstart.utils import _validate_hub_service_model_id_and_get_type

REGION = "us-east-1"
ACCOUNT_ID = "123456789123"
HUB_NAME = "mock-hub-name"

MOCK_MODEL_ID = "test-model-id"


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session")
    sagemaker_session_mock = Mock(
        name="sagemaker_session", boto_session=boto_mock, boto_region_name=REGION
    )
    sagemaker_session_mock._client_config.user_agent = (
        "Boto3/1.9.69 Python/3.6.5 Linux/4.14.77-70.82.amzn1.x86_64 Botocore/1.12.69 Resource"
    )
    sagemaker_session_mock.account_id.return_value = ACCOUNT_ID
    return sagemaker_session_mock


@pytest.mark.parametrize(
    "input_version, expected_version, expected_exception, expected_message",
    [
        ("1.0.0", "1.0.0", None, None),
        ("*", "3.2.0", None, None),
        (None, "3.2.0", None, None),
        ("1.*", "1.1.0", None, None),
        ("240612.4", "2.0.0", None, None),
        ("3.0.0", "3.0.0", None, None),
        ("4.0.0", "3.2.0", None, None),
        ("5.0.0", None, KeyError, "Model version not available in the Hub"),
        ("Blah", None, KeyError, "Bad semantic version"),
    ],
)
def test_proprietary_model(
    input_version, expected_version, expected_exception, expected_message, sagemaker_session
):
    sagemaker_session.list_hub_content_versions.return_value = {
        "HubContentSummaries": [
            {"HubContentVersion": "1.0.0", "HubContentSearchKeywords": []},
            {"HubContentVersion": "1.1.0", "HubContentSearchKeywords": []},
            {
                "HubContentVersion": "2.0.0",
                "HubContentSearchKeywords": ["@marketplace-version:240612.4"],
            },
            {
                "HubContentVersion": "3.0.0",
                "HubContentSearchKeywords": ["@marketplace-version:240612.5"],
            },
            {
                "HubContentVersion": "3.1.0",
                "HubContentSearchKeywords": ["@marketplace-version:3.0.0"],
            },
            {
                "HubContentVersion": "3.2.0",
                "HubContentSearchKeywords": ["@marketplace-version:4.0.0"],
            },
        ]
    }

    if expected_exception:
        with pytest.raises(expected_exception, match=expected_message):
            _test_proprietary_model(input_version, expected_version, sagemaker_session)
    else:
        _test_proprietary_model(input_version, expected_version, sagemaker_session)


def _test_proprietary_model(input_version, expected_version, sagemaker_session):
    result = hub_utils.get_hub_model_version(
        hub_model_name=MOCK_MODEL_ID,
        hub_model_type="Model",
        hub_name="blah",
        sagemaker_session=sagemaker_session,
        hub_model_version=input_version,
    )

    assert result == expected_version


@pytest.mark.parametrize(
    "get_model_specs_attr, get_model_specs_response, expected, expected_exception, expected_message",
    [
        (False, None, [], None, None),
        (True, None, [], None, None),
        (True, [], [], None, None),
        (True, ["OPEN_WEIGHTS"], [JumpStartModelType.OPEN_WEIGHTS], None, None),
        (
            True,
            ["OPEN_WEIGHTS", "PROPRIETARY"],
            [JumpStartModelType.OPEN_WEIGHTS, JumpStartModelType.PROPRIETARY],
            None,
            None,
        ),
    ],
)
@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_validate_hub_service_model_id_and_get_type(
    mock_get_model_specs,
    get_model_specs_attr,
    get_model_specs_response,
    expected,
    expected_exception,
    expected_message,
):
    mock_object = MagicMock()
    if get_model_specs_attr:
        mock_object.model_types = get_model_specs_response
    mock_get_model_specs.return_value = mock_object

    result = _validate_hub_service_model_id_and_get_type(model_id="blah", hub_arn="blah")

    assert result == expected
