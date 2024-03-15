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
from sagemaker.jumpstart.curated_hub.curated_hub import CuratedHub
from sagemaker.jumpstart.curated_hub.types import JumpStartModelInfo, S3ObjectLocation, HubContentSummary
from sagemaker.jumpstart.types import JumpStartModelSpecs
from tests.unit.sagemaker.jumpstart.constants import BASE_SPEC
from tests.unit.sagemaker.jumpstart.utils import get_spec_from_base_spec, HubContentType


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
@patch(f"{MODULE_PATH}._sync_public_model_to_hub")
@patch(f"{MODULE_PATH}.list_models")
def test_sync_kicks_off_parallel_syncs(
    mock_list_models, mock_sync_public_models, mock_get_model_specs, sagemaker_session
):
    mock_get_model_specs.side_effect = get_spec_from_base_spec
    mock_list_models.return_value = {"HubContentSummaries": []}
    hub_name = "mock_hub_name"
    model_one = {"model_id": "mock-model-one-huggingface"}
    model_two = {"model_id": "mock-model-two-pytorch", "version": "1.0.2"}
    mock_sync_public_models.return_value = ""
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    hub.sync([model_one, model_two])

    # mock_get_model_specs.assert_called_once()
    mock_sync_public_models.assert_has_calls(
        [
            mock.call(JumpStartModelInfo("mock-model-one-huggingface", "*"), 0),
            mock.call(JumpStartModelInfo("mock-model-two-pytorch", "1.0.2"), 1),
        ]
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch(f"{MODULE_PATH}._sync_public_model_to_hub")
@patch(f"{MODULE_PATH}.list_models")
def test_sync_filters_models_that_exist_in_hub(
    mock_list_models, mock_sync_public_models, mock_get_model_specs, sagemaker_session
):
    mock_get_model_specs.side_effect = get_spec_from_base_spec
    mock_list_models.return_value = {
        "HubContentSummaries": [
            {
                "HubContentName": "mock-model-two-pytorch",
                "HubContentVersion": "1.0.2",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-two-pytorch",
                    "@jumpstart-model-version:1.0.2",
                ],
            },
            {"HubContentName": "mock-model-three-nonsense", "HubContentVersion": "1.0.2", "HubContentSearchKeywords": []},
            {
                "HubContentName": "mock-model-four-huggingface",
                "HubContentVersion": "2.0.2",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-four-huggingface",
                    "@jumpstart-model-version:2.0.2",
                ],
            },
        ]
    }
    hub_name = "mock_hub_name"
    model_one = {"model_id": "mock-model-one-huggingface"}
    model_two = {"model_id": "mock-model-two-pytorch", "version": "1.0.2"}
    mock_sync_public_models.return_value = ""
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    hub.sync([model_one, model_two])

    mock_sync_public_models.assert_called_once_with(
        JumpStartModelInfo("mock-model-one-huggingface", "*"), 0
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch(f"{MODULE_PATH}._sync_public_model_to_hub")
@patch(f"{MODULE_PATH}.list_models")
def test_sync_updates_old_models_in_hub(
    mock_list_models, mock_sync_public_models, mock_get_model_specs, sagemaker_session
):
    mock_get_model_specs.side_effect = get_spec_from_base_spec
    mock_list_models.return_value = {
        "HubContentSummaries": [
            {
                "HubContentName": "mock-model-two-pytorch",
                "HubContentVersion": "1.0.1",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-two-pytorch",
                    "@jumpstart-model-version:1.0.2",
                ],
            },
            {
                "HubContentName": "mock-model-three-nonsense",
                "HubContentVersion": "1.0.2",
                "HubContentSearchKeywords": ["tag-one", "tag-two"],
            },
            {
                "HubContentName": "mock-model-four-huggingface",
                "HubContentVersion": "2.0.2",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-four-huggingface",
                    "@jumpstart-model-version:2.0.2",
                ],
            },
        ]
    }
    hub_name = "mock_hub_name"
    model_one = {"model_id": "mock-model-one-huggingface"}
    model_two = {"model_id": "mock-model-two-pytorch", "version": "1.0.2"}
    mock_sync_public_models.return_value = ""
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    hub.sync([model_one, model_two])

    mock_sync_public_models.assert_has_calls(
        [
            mock.call(JumpStartModelInfo("mock-model-one-huggingface", "*"), 0),
            mock.call(JumpStartModelInfo("mock-model-two-pytorch", "1.0.2"), 1),
        ]
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
@patch(f"{MODULE_PATH}._sync_public_model_to_hub")
@patch(f"{MODULE_PATH}.list_models")
def test_sync_passes_newer_hub_models(
    mock_list_models, mock_sync_public_models, mock_get_model_specs, sagemaker_session
):
    mock_get_model_specs.side_effect = get_spec_from_base_spec
    mock_list_models.return_value = {
        "HubContentSummaries": [
            {
                "HubContentName": "mock-model-two-pytorch",
                "HubContentVersion": "1.0.3",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-two-pytorch",
                    "@jumpstart-model-version:1.0.2",
                ],
            },
            {
                "HubContentName": "mock-model-three-nonsense",
                "HubContentVersion": "1.0.2",
                "HubContentSearchKeywords": ["tag-one", "tag-two"],
            },
            {
                "HubContentName": "mock-model-four-huggingface",
                "HubContentVersion": "2.0.2",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-four-huggingface",
                    "@jumpstart-model-version:2.0.2",
                ],
            },
        ]
    }
    hub_name = "mock_hub_name"
    model_one = {"model_id": "mock-model-one-huggingface"}
    model_two = {"model_id": "mock-model-two-pytorch", "version": "1.0.2"}
    mock_sync_public_models.return_value = ""
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    hub.sync([model_one, model_two])

    mock_sync_public_models.assert_called_once_with(
        JumpStartModelInfo("mock-model-one-huggingface", "*"), 0
    )


@patch("sagemaker.jumpstart.accessors.JumpStartModelsAccessor.get_model_specs")
def test_populate_latest_model_version(mock_get_model_specs, sagemaker_session):
    mock_get_model_specs.return_value = JumpStartModelSpecs(deepcopy(BASE_SPEC))

    hub_name = "mock_hub_name"
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    res = hub._populate_latest_model_version({"model_id": "mock-pytorch-model-one", "version": "*"})
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}

    res = hub._populate_latest_model_version({"model_id": "mock-pytorch-model-one"})
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}

    # Should take latest version from specs no matter what. Parent should responsibly call
    res = hub._populate_latest_model_version(
        {"model_id": "mock-pytorch-model-one", "version": "2.0.0"}
    )
    assert res == {"model_id": "mock-pytorch-model-one", "version": "1.0.0"}


@patch(f"{MODULE_PATH}.list_models")
def test_get_jumpstart_models_in_hub(mock_list_models, sagemaker_session):
    mock_list_models.return_value = {
        "HubContentSummaries": [
            {
                "HubContentName": "mock-model-two-pytorch",
                "HubContentVersion": "1.0.3",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-two-pytorch",
                    "@jumpstart-model-version:1.0.3",
                ],
            },
            {
                "HubContentName": "mock-model-three-nonsense",
                "HubContentVersion": "1.0.2",
                "HubContentSearchKeywords": ["tag-one", "tag-two"],
            },
            {
                "HubContentName": "mock-model-four-huggingface",
                "HubContentVersion": "2.0.2",
                "HubContentSearchKeywords": [
                    "@jumpstart-model-id:model-four-huggingface",
                    "@jumpstart-model-version:2.0.2",
                ],
            },
        ]
    }

    hub_name = "mock_hub_name"
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    res = hub._get_jumpstart_models_in_hub()
    assert res == [
        HubContentSummary(
            hub_content_name="mock-model-two-pytorch",
            hub_content_version="1.0.3",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-two-pytorch",
                "@jumpstart-model-version:1.0.3",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
        HubContentSummary(
            hub_content_name="mock-model-four-huggingface",
            hub_content_version="2.0.2",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-four-huggingface",
                "@jumpstart-model-version:2.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
    ]

    mock_list_models.return_value = {"HubContentSummaries": []}

    res = hub._get_jumpstart_models_in_hub()
    assert res == []

    mock_list_models.return_value = {
        "HubContentSummaries": [
            {
                "HubContentName": "mock-model-three-nonsense",
                "HubContentVersion": "1.0.2",
                "HubContentSearchKeywords": ["tag-one", "tag-two"],
            },
        ]
    }

    res = hub._get_jumpstart_models_in_hub()
    assert res == []


def test_determine_models_to_sync(sagemaker_session):
    hub_name = "mock_hub_name"
    hub = CuratedHub(hub_name=hub_name, sagemaker_session=sagemaker_session)

    js_model_map = {
        "mock-model-two-pytorch": HubContentSummary(
            hub_content_name="mock-model-two-pytorch",
            hub_content_version="1.0.1",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-two-pytorch",
                "@jumpstart-model-version:1.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
        "mock-model-four-huggingface": HubContentSummary(
            hub_content_name="mock-model-four-huggingface",
            hub_content_version="2.0.2",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-four-huggingface",
                "@jumpstart-model-version:2.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        )
    }
    model_one = JumpStartModelInfo("mock-model-one-huggingface", "1.2.3")
    model_two = JumpStartModelInfo("mock-model-two-pytorch", "1.0.2")
    # No model_one, older model_two
    res = hub._determine_models_to_sync([model_one, model_two], js_model_map)
    assert res == [model_one, model_two]

    js_model_map = {
        "mock-model-two-pytorch": HubContentSummary(
            hub_content_name="mock-model-two-pytorch",
            hub_content_version="1.0.3",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-two-pytorch",
                "@jumpstart-model-version:1.0.3",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
        "mock-model-four-huggingface": HubContentSummary(
            hub_content_name="mock-model-four-huggingface",
            hub_content_version="2.0.2",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-four-huggingface",
                "@jumpstart-model-version:2.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        )
    }

    # No model_one, newer model_two
    res = hub._determine_models_to_sync([model_one, model_two], js_model_map)
    assert res == [model_one]

    js_model_map = {
        "mock-model-one-huggingface": HubContentSummary(
            hub_content_name="mock-model-one-huggingface",
            hub_content_version="1.2.3",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-one-huggingface",
                "@jumpstart-model-version:1.2.3",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
        "mock-model-two-pytorch": HubContentSummary(
            hub_content_name="mock-model-two-pytorch",
            hub_content_version="1.0.2",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-two-pytorch",
                "@jumpstart-model-version:1.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        )
    }
    # Same model_one, same model_two
    res = hub._determine_models_to_sync([model_one, model_two], js_model_map)
    assert res == []

    js_model_map = {
        "mock-model-one-huggingface": HubContentSummary(
            hub_content_name="mock-model-one-huggingface",
            hub_content_version="1.2.1",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-one-huggingface",
                "@jumpstart-model-version:1.2.1",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        ),
        "mock-model-two-pytorch": HubContentSummary(
            hub_content_name="mock-model-two-pytorch",
            hub_content_version="1.0.2",
            hub_content_search_keywords=[
                "@jumpstart-model-id:model-two-pytorch",
                "@jumpstart-model-version:1.0.2",
            ],
            hub_content_arn=None,
            hub_content_type=None,
            document_schema_version=None,
            hub_content_status=None,
            creation_time=None
        )
    }
    # Old model_one, same model_two
    res = hub._determine_models_to_sync([model_one, model_two], js_model_map)
    assert res == [model_one]