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
"""``HubContent.search`` runs ``Search(Resource="HubContent")`` with ``Metadata.*`` filters.

Before this change the SDK had no binding for the ``HubContent`` search resource: the typed
``SearchRecord.hub_content`` shape existed since v3.25.0 and nothing read it, so a caller who
wanted the public hub narrowed by license, provider, task, or model size either listed every
model with ``ListHubContents`` or wrote the ``Search`` request, the paging, and the result
parsing by hand.
"""

from __future__ import absolute_import

import json
from unittest.mock import Mock, patch

import pytest

from sagemaker.core.hub_content import (
    MAX_SEARCH_FILTERS,
    HubContentFilter,
    _search_expression,
)
from sagemaker.core.resources import HubContent

HUB = "SageMakerPublicHub"


def _record(name, license_name="apache-2.0", hub_name=HUB):
    return {
        "HubContent": {
            "HubContentName": name,
            "HubName": hub_name,
            "HubContentType": "Model",
            "HubContentVersion": "1.0.0",
            "HubContentArn": f"arn:aws:sagemaker:us-west-2:aws:hub-content/{hub_name}/Model/{name}/1.0.0",
            "HubContentSearchKeywords": [f"@license:{license_name}", "@task:text generation"],
            "HubContentDocument": json.dumps({"License": license_name, "ModelSize": "1b-10b"}),
            "HubContentStatus": "Available",
            "Tags": [{"Key": "owner", "Value": "aws"}],
        }
    }


def _client(pages):
    client = Mock()
    client.search.side_effect = pages
    return client


class TestHubContentFilter:
    def test_metadata_filter_maps_an_attribute_to_its_document_field(self):
        search_filter = HubContentFilter.metadata("license", "apache-2.0")

        assert search_filter.to_request() == {
            "Name": "Metadata.License",
            "Operator": "Equals",
            "Value": "apache-2.0",
        }

    def test_metadata_filter_with_a_sequence_uses_in(self):
        request = HubContentFilter.metadata("model_size", ["<1b", "1b-10b"]).to_request()

        assert request == {"Name": "Metadata.ModelSize", "Operator": "In", "Value": "<1b,1b-10b"}

    def test_metadata_filter_rejects_an_unknown_attribute(self):
        with pytest.raises(ValueError, match="not a hub content metadata attribute"):
            HubContentFilter.metadata("flavor", "sweet")

    def test_keyword_filter_matches_a_search_keyword(self):
        assert HubContentFilter.keyword("@task:summarization").to_request() == {
            "Name": "HubContentSearchKeywords",
            "Operator": "Equals",
            "Value": "@task:summarization",
        }

    def test_exists_takes_no_value_and_other_operators_require_one(self):
        assert HubContentFilter("Metadata.License", "Exists").to_request() == {
            "Name": "Metadata.License",
            "Operator": "Exists",
        }
        with pytest.raises(ValueError, match="takes a value"):
            HubContentFilter("Metadata.License", "Contains").to_request()

    def test_operator_and_value_limits(self):
        with pytest.raises(ValueError, match="not a Search operator"):
            HubContentFilter("Metadata.License", "Like", "x").to_request()
        with pytest.raises(ValueError, match="1 to 1024 characters"):
            HubContentFilter("Metadata.License", "Equals", "x" * 1025).to_request()
        with pytest.raises(ValueError, match="1 to 1024 characters"):
            HubContentFilter("Metadata.License", "Equals", "").to_request()


class TestSearchExpression:
    def test_expression_leads_with_hub_name_and_content_type(self):
        expression = _search_expression(
            HUB, "Model", [HubContentFilter.metadata("provider", "Meta")]
        )

        assert expression == {
            "Operator": "And",
            "Filters": [
                {"Name": "HubName", "Operator": "Equals", "Value": HUB},
                {"Name": "HubContentType", "Operator": "Equals", "Value": "Model"},
                {"Name": "Metadata.Provider", "Operator": "Equals", "Value": "Meta"},
            ],
        }

    def test_expression_enforces_the_filter_cap(self):
        too_many = [HubContentFilter.keyword(f"@task:{i}") for i in range(MAX_SEARCH_FILTERS - 1)]

        with pytest.raises(ValueError, match=f"at most {MAX_SEARCH_FILTERS} filters"):
            _search_expression(HUB, "Model", too_many)


class TestHubContentSearch:
    def test_search_pages_and_yields_bound_hub_content(self):
        client = _client(
            [
                {"Results": [_record("model-a")], "NextToken": "page-2"},
                {"Results": [_record("model-b", "llama3")]},
            ]
        )
        session = Mock(name="session")

        with patch.object(HubContent, "get_sagemaker_client", return_value=client):
            results = list(
                HubContent.search(
                    HUB,
                    filters=[HubContentFilter.metadata("license", "apache-2.0")],
                    session=session,
                    region="us-west-2",
                )
            )

        assert [content.hub_content_name for content in results] == ["model-a", "model-b"]
        assert all(isinstance(content, HubContent) for content in results)
        assert results[0].hub_name == HUB
        assert results[0].license == "apache-2.0"
        assert results[0].model_size == "1b-10b"
        assert results[1].license == "llama3"
        assert results[0]._session is session
        assert results[0]._region == "us-west-2"
        first_request = client.search.call_args_list[0].kwargs
        assert first_request["Resource"] == "HubContent"
        assert first_request["SortBy"] == "HubContentName"
        assert first_request["SearchExpression"]["Filters"][2] == {
            "Name": "Metadata.License",
            "Operator": "Equals",
            "Value": "apache-2.0",
        }
        assert "NextToken" not in first_request
        assert client.search.call_args_list[1].kwargs["NextToken"] == "page-2"

    def test_search_stops_on_a_repeated_token_and_an_empty_result(self):
        client = _client(
            [
                {"Results": [_record("model-a")], "NextToken": "same"},
                {"Results": [], "NextToken": "same"},
            ]
        )

        with patch.object(HubContent, "get_sagemaker_client", return_value=client):
            results = list(HubContent.search(HUB))

        assert [content.hub_content_name for content in results] == ["model-a"]
        assert client.search.call_count == 2

    def test_search_drops_record_members_the_resource_lacks(self):
        client = _client([{"Results": [_record("model-a")]}])

        with patch.object(HubContent, "get_sagemaker_client", return_value=client):
            (content,) = list(HubContent.search(HUB, hub_content_type="Model"))

        assert content.hub_content_status == "Available"
        assert not hasattr(content, "tags")
