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

from unittest.mock import Mock

from sagemaker.core.jumpstart.hub.hub import Hub

HUB_NAME = "mock-hub-name"


def test_list_models_sends_next_token_to_the_following_page():
    pages = {
        ("ModelReference", None): {
            "HubContentSummaries": [{"HubContentName": "reference-a"}],
            "NextToken": "page-2",
        },
        ("ModelReference", "page-2"): {"HubContentSummaries": [{"HubContentName": "reference-b"}]},
        ("Model", None): {
            "HubContentSummaries": [{"HubContentName": "model-a"}],
            "NextToken": "page-2",
        },
        ("Model", "page-2"): {
            "HubContentSummaries": [{"HubContentName": "model-b"}],
            "NextToken": "page-3",
        },
        ("Model", "page-3"): {"HubContentSummaries": [{"HubContentName": "model-c"}]},
    }

    def list_hub_contents(**kwargs):
        assert kwargs["hub_name"] == HUB_NAME
        assert kwargs["max_results"] == 2
        # Each page is served once. A second request for a page raises KeyError, not a loop.
        return pages.pop((kwargs["hub_content_type"], kwargs.get("next_token")))

    sagemaker_session = Mock(name="sagemaker_session", boto_region_name="us-east-1")
    sagemaker_session.list_hub_contents = Mock(side_effect=list_hub_contents)
    hub = Hub(hub_name=HUB_NAME, sagemaker_session=sagemaker_session)

    response = hub.list_models(max_results=2)

    assert [summary["HubContentName"] for summary in response["hub_content_summaries"]] == [
        "reference-a",
        "reference-b",
        "model-a",
        "model-b",
        "model-c",
    ]
    assert pages == {}
