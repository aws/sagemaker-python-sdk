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


def _hub_with_session_client():
    """A Hub backed by a real Session whose SageMaker client is mocked."""
    from sagemaker.core.helper.session_helper import Session

    session = Session.__new__(Session)
    session._region_name = "us-west-2"
    session.sagemaker_client = Mock()
    return Hub(hub_name=HUB_NAME, sagemaker_session=session), session.sagemaker_client


def test_create_calls_create_hub():
    hub, client = _hub_with_session_client()

    hub.create(description="my hub", search_keywords=["a"], tags=[{"Key": "k", "Value": "v"}])

    client.create_hub.assert_called_once_with(
        HubName=HUB_NAME,
        HubDescription="my hub",
        HubDisplayName=HUB_NAME,
        HubSearchKeywords=["a"],
        Tags=[{"Key": "k", "Value": "v"}],
    )


def test_describe_calls_describe_hub():
    hub, client = _hub_with_session_client()

    hub.describe()

    client.describe_hub.assert_called_once_with(HubName=HUB_NAME)


def test_delete_calls_delete_hub():
    hub, client = _hub_with_session_client()

    hub.delete()

    client.delete_hub.assert_called_once_with(HubName=HUB_NAME)


def test_create_model_reference_calls_create_hub_content_reference():
    hub, client = _hub_with_session_client()
    model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/m/1.0.0"

    hub.create_model_reference(model_arn=model_arn, model_name="m", min_version="1.0.0")

    client.create_hub_content_reference.assert_called_once_with(
        HubName=HUB_NAME,
        SageMakerPublicHubContentArn=model_arn,
        HubContentName="m",
        MinVersion="1.0.0",
    )


def test_delete_model_reference_calls_delete_hub_content_reference():
    hub, client = _hub_with_session_client()

    hub.delete_model_reference(model_name="m")

    client.delete_hub_content_reference.assert_called_once_with(
        HubName=HUB_NAME, HubContentType="ModelReference", HubContentName="m"
    )


def test_init_without_session_uses_default_jumpstart_session():
    from unittest.mock import patch

    default_session = Mock(boto_region_name="us-east-2")
    with patch(
        "sagemaker.core.jumpstart.hub.hub.utils.get_default_jumpstart_session_with_user_agent_suffix",
        return_value=default_session,
    ):
        hub = Hub(hub_name=HUB_NAME, sagemaker_session=None)

    assert hub._sagemaker_session is default_session
    assert hub.region == "us-east-2"
