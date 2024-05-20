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
from copy import deepcopy
import datetime
from unittest import mock
from unittest.mock import patch
import pytest
from mock import Mock
from sagemaker.jumpstart.types import JumpStartModelSpecs
from sagemaker.jumpstart.curated_hub.curated_hub import CuratedHub
from sagemaker.jumpstart.curated_hub.interfaces import HubContentInfo
from sagemaker.jumpstart.curated_hub.types import JumpStartModelInfo, S3ObjectLocation
from tests.unit.sagemaker.jumpstart.constants import BASE_SPEC
from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec


REGION = "us-east-1"
ACCOUNT_ID = "123456789123"
HUB_NAME = "mock-hub-name"

MODULE_PATH = "sagemaker.jumpstart.curated_hub.curated_hub.CuratedHub"

FAKE_TIME = datetime.datetime(1997, 8, 14, 00, 00, 00)


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


def test_instantiates(sagemaker_session):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    assert hub.hub_name == HUB_NAME
    assert hub.region == "us-east-1"
    assert hub._sagemaker_session == sagemaker_session


@pytest.mark.parametrize(
    ("hub_name,hub_description,hub_bucket_name,hub_display_name,hub_search_keywords,tags"),
    [
        pytest.param("MockHub1", "this is my sagemaker hub", None, None, None, None),
        pytest.param(
            "MockHub2",
            "this is my sagemaker hub two",
            None,
            "DisplayMockHub2",
            ["mock", "hub", "123"],
            [{"Key": "tag-key-1", "Value": "tag-value-1"}],
        ),
    ],
)
@patch("sagemaker.jumpstart.curated_hub.curated_hub.CuratedHub._generate_hub_storage_location")
def test_create_with_no_bucket_name(
    mock_generate_hub_storage_location,
    sagemaker_session,
    hub_name,
    hub_description,
    hub_bucket_name,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    storage_location = S3ObjectLocation(
        "sagemaker-hubs-us-east-1-123456789123", f"{hub_name}-{FAKE_TIME.timestamp()}"
    )
    mock_generate_hub_storage_location.return_value = storage_location
    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    sagemaker_session.describe_hub.return_value = {
        "S3StorageConfig": {"S3OutputPath": f"s3://{hub_bucket_name}/{storage_location.key}"}
    }
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "s3_storage_config": {
            "S3OutputPath": f"s3://sagemaker-hubs-us-east-1-123456789123/{storage_location.key}"
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
@patch("sagemaker.jumpstart.curated_hub.curated_hub.CuratedHub._generate_hub_storage_location")
def test_create_with_bucket_name(
    mock_generate_hub_storage_location,
    sagemaker_session,
    hub_name,
    hub_description,
    hub_bucket_name,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    storage_location = S3ObjectLocation(hub_bucket_name, f"{hub_name}-{FAKE_TIME.timestamp()}")
    mock_generate_hub_storage_location.return_value = storage_location
    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    hub = CuratedHub(
        hub_name=hub_name, sagemaker_session=sagemaker_session, bucket_name=hub_bucket_name
    )
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "s3_storage_config": {"S3OutputPath": f"s3://mock-bucket-123/{storage_location.key}"},
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


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_get_latest_model_version(mock_get_model_specs, sagemaker_session):
    mock_get_model_specs.return_value = JumpStartModelSpecs(deepcopy(BASE_SPEC))

    hub_name = "mock_hub_name"
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    res = hub._get_latest_model_version("pytorch-ic-mobilenet-v2")
    assert res == "1.0.0"


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_populate_latest_model_version(mock_get_model_specs, sagemaker_session):
    mock_get_model_specs.return_value = JumpStartModelSpecs(deepcopy(BASE_SPEC))

    hub_name = "mock_hub_name"
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    res = hub._populate_latest_model_version({"model_id": "mock-pytorch-model-one", "version": "*"})
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}

    res = hub._populate_latest_model_version({"model_id": "mock-pytorch-model-one"})
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}

    # Should take latest version from specs no matter what. Parent should responsibly call.
    res = hub._populate_latest_model_version(
        {"model_id": "mock-pytorch-model-one", "version": "2.0.0"}
    )
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}


@patch(f"{MODULE_PATH}._get_latest_model_version")
@patch("sagemaker.jumpstart.curated_hub.interfaces.DescribeHubContentResponse.from_json")
def test_describe_model_with_none_version(
    mock_describe_hub_content_response, mock_get_latest_model_version, sagemaker_session
):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"
    mock_get_latest_model_version.return_value = "1.1.1"
    mock_describe_hub_content_response.return_value = Mock()

    hub.describe_model(model_name, None)
    sagemaker_session.describe_hub_content.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_name="mock-model-one-huggingface",
        hub_content_version="1.1.1",
        hub_content_type="Model",
    )


@patch(f"{MODULE_PATH}._get_latest_model_version")
@patch("sagemaker.jumpstart.curated_hub.interfaces.DescribeHubContentResponse.from_json")
def test_describe_model_with_wildcard_version(
    mock_describe_hub_content_response, mock_get_latest_model_version, sagemaker_session
):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"
    mock_get_latest_model_version.return_value = "1.1.1"
    mock_describe_hub_content_response.return_value = Mock()

    hub.describe_model(model_name, "*")
    sagemaker_session.describe_hub_content.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_name="mock-model-one-huggingface",
        hub_content_version="1.1.1",
        hub_content_type="Model",
    )


@patch(f"{MODULE_PATH}._get_latest_model_version")
def test_delete_model_with_none_version(mock_get_latest_model_version, sagemaker_session):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"
    mock_get_latest_model_version.return_value = "1.1.1"

    hub.delete_model(model_name, None)
    sagemaker_session.delete_hub_content.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_name="mock-model-one-huggingface",
        hub_content_version="1.1.1",
        hub_content_type="Model",
    )


@patch(f"{MODULE_PATH}._get_latest_model_version")
def test_delete_model_with_wildcard_version(mock_get_latest_model_version, sagemaker_session):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"
    mock_get_latest_model_version.return_value = "1.1.1"

    hub.delete_model(model_name, "*")
    sagemaker_session.delete_hub_content.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_name="mock-model-one-huggingface",
        hub_content_version="1.1.1",
        hub_content_type="Model",
    )

def test_delete_hub_content_reference(sagemaker_session):
    hub = CuratedHub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)
    model_name = "mock-model-one-huggingface"

    hub.delete_model_reference(model_name)
    sagemaker_session.delete_hub_content_reference.assert_called_with(
        hub_name=HUB_NAME,
        hub_content_type="ModelReference",
        hub_content_name="mock-model-one-huggingface"
    )
