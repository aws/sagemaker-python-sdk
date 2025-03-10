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
from datetime import datetime
from unittest.mock import patch, MagicMock
import pytest
from mock import Mock
from sagemaker.jumpstart.hub.hub import Hub


REGION = "us-east-1"
ACCOUNT_ID = "123456789123"
HUB_NAME = "mock-hub-name"

MODULE_PATH = "sagemaker.jumpstart.hub.hub.Hub"

FAKE_TIME = datetime(1997, 8, 14, 00, 00, 00)


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session")
    sagemaker_session_mock = Mock(
        name="sagemaker_session", boto_session=boto_mock, boto_region_name=REGION
    )
    sagemaker_session_mock._client_config.user_agent = (
        "Boto3/1.9.69 Python/3.6.5 Linux/4.14.77-70.82.amzn1.x86_64 Botocore/1.12.69 Resource"
    )
    sagemaker_session_mock.describe_hub.return_value = {
        "S3StorageConfig": {"S3OutputPath": "s3://mock-bucket-123"}
    }
    sagemaker_session_mock.account_id.return_value = ACCOUNT_ID
    return sagemaker_session_mock


@pytest.fixture
def mock_instance(sagemaker_session):
    mock_instance = MagicMock()
    mock_instance.hub_name = "test-hub"
    mock_instance._sagemaker_session = sagemaker_session
    return mock_instance


def test_instantiates(sagemaker_session):
    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    assert hub.hub_name == HUB_NAME
    assert hub.region == "us-east-1"
    assert hub._sagemaker_session == sagemaker_session


@pytest.mark.parametrize(
    ("hub_name,hub_description,,hub_display_name,hub_search_keywords,tags"),
    [
        pytest.param("MockHub1", "this is my sagemaker hub", None, None, None),
        pytest.param(
            "MockHub2",
            "this is my sagemaker hub two",
            "DisplayMockHub2",
            ["mock", "hub", "123"],
            [{"Key": "tag-key-1", "Value": "tag-value-1"}],
        ),
    ],
)
def test_create_with_no_bucket_name(
    sagemaker_session,
    hub_name,
    hub_description,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    hub = Hub(hub_name=hub_name, sagemaker_session=sagemaker_session)
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "s3_storage_config": {"S3OutputPath": None},
        "tags": tags,
    }
    response = hub.create(
        description=hub_description,
        display_name=hub_display_name,
        search_keywords=hub_search_keywords,
        tags=tags,
    )
    sagemaker_session.create_hub.assert_called_with(**request)
    assert response == {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}


@pytest.mark.parametrize(
    ("hub_name,hub_description,hub_bucket_name,hub_display_name,hub_search_keywords,tags"),
    [
        pytest.param("MockHub1", "this is my sagemaker hub", "mock-bucket-123", None, None, None),
        pytest.param(
            "MockHub2",
            "this is my sagemaker hub two",
            "mock-bucket-123",
            "DisplayMockHub2",
            ["mock", "hub", "123"],
            [{"Key": "tag-key-1", "Value": "tag-value-1"}],
        ),
    ],
)
@patch("sagemaker.jumpstart.hub.hub.datetime")
def test_create_with_bucket_name(
    mock_datetime,
    sagemaker_session,
    hub_name,
    hub_description,
    hub_bucket_name,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    mock_datetime.now.return_value = FAKE_TIME

    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    hub = Hub(hub_name=hub_name, sagemaker_session=sagemaker_session, bucket_name=hub_bucket_name)
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "s3_storage_config": {
            "S3OutputPath": f"s3://mock-bucket-123/{hub_name}-{FAKE_TIME.timestamp()}"
        },
        "tags": tags,
    }
    response = hub.create(
        description=hub_description,
        display_name=hub_display_name,
        search_keywords=hub_search_keywords,
        tags=tags,
    )
    sagemaker_session.create_hub.assert_called_with(**request)
    assert response == {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}


@patch("sagemaker.jumpstart.hub.interfaces.DescribeHubContentResponse.from_json")
def test_describe_model_success(mock_describe_hub_content_response, sagemaker_session):
    mock_describe_hub_content_response.return_value = Mock()
    mock_list_hub_content_versions = sagemaker_session.list_hub_content_versions
    mock_list_hub_content_versions.return_value = {
        "HubContentSummaries": [
            {"HubContentVersion": "1.0"},
            {"HubContentVersion": "2.0"},
            {"HubContentVersion": "3.0"},
        ]
    }

    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)

    with patch("sagemaker.jumpstart.hub.utils.get_hub_model_version") as mock_get_hub_model_version:
        mock_get_hub_model_version.return_value = "3.0"

        hub.describe_model("test-model")

        mock_list_hub_content_versions.assert_called_with(
            hub_name=HUB_NAME, hub_content_name="test-model", hub_content_type="ModelReference"
        )
        sagemaker_session.describe_hub_content.assert_called_with(
            hub_name=HUB_NAME,
            hub_content_name="test-model",
            hub_content_version="3.0",
            hub_content_type="ModelReference",
        )


@patch("sagemaker.jumpstart.hub.interfaces.DescribeHubContentResponse.from_json")
def test_describe_model_one_thrown_error(mock_describe_hub_content_response, sagemaker_session):
    mock_describe_hub_content_response.return_value = Mock()
    mock_list_hub_content_versions = sagemaker_session.list_hub_content_versions
    mock_list_hub_content_versions.return_value = {
        "HubContentSummaries": [
            {"HubContentVersion": "1.0"},
            {"HubContentVersion": "2.0"},
            {"HubContentVersion": "3.0"},
        ]
    }
    mock_describe_hub_content = sagemaker_session.describe_hub_content
    mock_describe_hub_content.side_effect = [
        Exception("Some exception"),
        {"HubContentName": "test-model", "HubContentVersion": "3.0"},
    ]

    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)

    with patch("sagemaker.jumpstart.hub.utils.get_hub_model_version") as mock_get_hub_model_version:
        mock_get_hub_model_version.return_value = "3.0"

        hub.describe_model("test-model")

        mock_describe_hub_content.asssert_called_times(2)
        mock_describe_hub_content.assert_called_with(
            hub_name=HUB_NAME,
            hub_content_name="test-model",
            hub_content_version="3.0",
            hub_content_type="Model",
        )


def test_create_hub_content_reference(sagemaker_session):
    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"
    min_version = "1.1.1"
    public_model_arn = (
        f"arn:aws:sagemaker:us-east-1:123456789123:hub-content/JumpStartHub/model/{model_name}"
    )
    create_hub_content_reference = {
        "HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{HUB_NAME}",
        "HubContentReferenceArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub-content/{HUB_NAME}/ModelRef/{model_name}",  # noqa: E501
    }
    sagemaker_session.create_hub_content_reference = Mock(return_value=create_hub_content_reference)

    request = {
        "hub_name": HUB_NAME,
        "source_hub_content_arn": public_model_arn,
        "hub_content_name": model_name,
        "min_version": min_version,
    }

    response = hub.create_model_reference(
        model_arn=public_model_arn, model_name=model_name, min_version=min_version
    )
    sagemaker_session.create_hub_content_reference.assert_called_with(**request)

    assert response == {
        "HubArn": "arn:aws:sagemaker:us-east-1:123456789123:hub/mock-hub-name",
        "HubContentReferenceArn": "arn:aws:sagemaker:us-east-1:123456789123:hub-content/mock-hub-name/ModelRef/mock-model-one-huggingface",  # noqa: E501
    }


def test_delete_hub_content_reference(sagemaker_session):
    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"

    hub.delete_model_reference(model_name)
    sagemaker_session.delete_hub_content_reference.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_type="ModelReference",
        hub_content_name="mock-model-one-huggingface",
    )
