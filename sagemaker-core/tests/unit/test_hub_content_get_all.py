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
"""``HubContent.get_all`` lists a hub's contents as bound resources.

Before this method the SDK had no resource-level ``ListHubContents``: callers paged the
boto3 client by hand (six places in the SDK did the same), ``ResourceIterator`` dropped
``OriginalCreationTime`` because the ``HubContent`` class had no such field, and
``get_all_versions()`` built resources without the ``hub_name`` the list call used, so
the first ``refresh()`` failed with ``Missing required parameter in input: "HubName"``.
"""

from __future__ import absolute_import

import datetime
from unittest.mock import MagicMock, patch

import boto3
import pytest

from sagemaker.core.resources import Base, HubContent
from sagemaker.core.utils.utils import ResourceIterator, SageMakerClient

HUB = "SageMakerPublicHub"
CREATED = datetime.datetime(2024, 3, 1, tzinfo=datetime.timezone.utc)
ORIGINAL = datetime.datetime(2023, 6, 1, tzinfo=datetime.timezone.utc)


def _summary(name, version="1.0.0"):
    return {
        "HubContentName": name,
        "HubContentArn": f"arn:aws:sagemaker:us-west-2:aws:hub-content/{HUB}/Model/{name}/{version}",
        "HubContentVersion": version,
        "HubContentType": "Model",
        "DocumentSchemaVersion": "2.0.0",
        "HubContentStatus": "Available",
        "HubContentSearchKeywords": ["@task:text-generation", "@license:apache-2.0"],
        "CreationTime": CREATED,
        "OriginalCreationTime": ORIGINAL,
    }


def _describe(**kwargs):
    if "HubName" not in kwargs:
        raise ValueError('Missing required parameter in input: "HubName"')
    summary = _summary(kwargs["HubContentName"], kwargs.get("HubContentVersion", "1.0.0"))
    summary.pop("OriginalCreationTime")
    return {
        **summary,
        "HubName": kwargs["HubName"],
        "HubArn": f"arn:aws:sagemaker:us-west-2:aws:hub/{HUB}",
        "HubContentDocument": "{}",
    }


@pytest.fixture(autouse=True)
def _reset_client_cache():
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()


@pytest.fixture
def client():
    fake = MagicMock()
    fake.list_hub_contents.side_effect = [
        {"HubContentSummaries": [_summary("model-a")], "NextToken": "page-2"},
        {"HubContentSummaries": [_summary("model-b")]},
    ]
    fake.list_hub_content_versions.return_value = {
        "HubContentSummaries": [_summary("model-a", "1.0.0"), _summary("model-a", "2.0.0")]
    }
    fake.describe_hub_content.side_effect = _describe
    with patch.object(Base, "get_sagemaker_client", return_value=fake):
        yield fake


def test_get_all_pages_through_the_hub_and_binds_hub_name(client):
    contents = list(HubContent.get_all(hub_name=HUB, hub_content_type="Model"))

    assert [content.hub_content_name for content in contents] == ["model-a", "model-b"]
    assert all(content.hub_name == HUB for content in contents)
    assert client.list_hub_contents.call_args_list[0].kwargs == {
        "HubName": HUB,
        "HubContentType": "Model",
    }
    assert client.list_hub_contents.call_args_list[1].kwargs["NextToken"] == "page-2"


def test_get_all_forwards_the_list_filters(client):
    client.list_hub_contents.side_effect = [{"HubContentSummaries": []}]

    list(
        HubContent.get_all(
            hub_name=HUB,
            hub_content_type="Model",
            name_contains="llama",
            sort_by="HubContentName",
            sort_order="Ascending",
        )
    )

    assert client.list_hub_contents.call_args.kwargs == {
        "HubName": HUB,
        "HubContentType": "Model",
        "NameContains": "llama",
        "SortBy": "HubContentName",
        "SortOrder": "Ascending",
    }


def test_get_all_keeps_original_creation_time_from_the_summary(client):
    contents = list(HubContent.get_all(hub_name=HUB, hub_content_type="Model"))

    assert contents[0].original_creation_time == ORIGINAL
    assert contents[0].creation_time == CREATED
    assert contents[0].hub_content_search_keywords == [
        "@task:text-generation",
        "@license:apache-2.0",
    ]


def test_get_all_refreshes_each_resource_with_the_hub_name(client):
    list(HubContent.get_all(hub_name=HUB, hub_content_type="Model"))

    for call in client.describe_hub_content.call_args_list:
        assert call.kwargs["HubName"] == HUB


def test_get_all_binds_the_session_and_region_to_each_resource(client):
    session = boto3.Session(
        aws_access_key_id="AKID", aws_secret_access_key="secret", region_name="eu-west-1"
    )

    content = next(
        iter(
            HubContent.get_all(
                hub_name=HUB, hub_content_type="Model", session=session, region="eu-west-1"
            )
        )
    )

    assert content._session is session
    assert content._region == "eu-west-1"


def test_get_all_versions_binds_the_hub_name_of_the_parent(client):
    parent = HubContent(hub_name=HUB, hub_content_type="Model", hub_content_name="model-a")

    versions = list(parent.get_all_versions())

    assert [version.hub_content_version for version in versions] == ["1.0.0", "2.0.0"]
    assert all(version.hub_name == HUB for version in versions)
    for call in client.describe_hub_content.call_args_list:
        assert call.kwargs["HubName"] == HUB


def test_resource_iterator_extra_init_data_does_not_override_the_summary():
    fake = MagicMock()
    fake.list_hub_contents.return_value = {"HubContentSummaries": [_summary("model-a")]}
    fake.describe_hub_content.side_effect = _describe

    iterator = ResourceIterator(
        client=fake,
        summaries_key="HubContentSummaries",
        summary_name="HubContentInfo",
        resource_cls=HubContent,
        list_method="list_hub_contents",
        list_method_kwargs={"HubName": HUB, "HubContentType": "Model"},
        extra_init_data={"hub_name": HUB, "hub_content_name": "ignored-when-present"},
    )
    with patch.object(Base, "get_sagemaker_client", return_value=fake):
        content = next(iterator)

    assert content.hub_content_name == "model-a"
    assert content.hub_name == HUB
