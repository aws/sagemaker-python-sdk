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
from mock import Mock
from sagemaker.jumpstart.curated_hub.curated_hub import CuratedHub

REGION = "us-east-1"
ACCOUNT_ID = "123456789123"
HUB_NAME = "mock-hub-name"


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


# @pytest.fixture()
# def sagemaker_session():
#     boto_mock = Mock(name="boto_session", region_name=REGION)
#     session_mock = Mock(
#         name="sagemaker_session",
#         boto_session=boto_mock,
#         boto_region_name=REGION,
#         config=None,
#         local_mode=False,
#         default_bucket_prefix=None,
#     )
#     session_mock.return_value.sagemkaer_client = Mock(name="sagemaker_client")
#     session_mock.sts_client.get_caller_identity = Mock(return_value={"Account": ACCOUNT_ID})
#     create_hub = {"HubArn": "arn:aws:sagemaker:us-east-1:123456789123:hub/mock-hub-name"}
#     session_mock.sagemaker_client.create_hub = Mock(return_value=create_hub)
#     print(session_mock.sagemaker_client)
#     return session_mock


def test_instantiates(sagemaker_session):
    hub = CuratedHub(hub_name=HUB_NAME, region=REGION, sagemaker_session=sagemaker_session)
    assert hub.hub_name == HUB_NAME
    assert hub.region == "us-east-1"
    assert hub._sagemaker_session == sagemaker_session


def test_instantiates_handles_conflicting_regions(sagemaker_session):
    conflicting_region = "us-east-2"

    with pytest.raises(ValueError):
        CuratedHub(
            hub_name=HUB_NAME, region=conflicting_region, sagemaker_session=sagemaker_session
        )


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
def test_create_with_no_bucket_name(
    sagemaker_session,
    hub_name,
    hub_description,
    hub_bucket_name,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    hub = CuratedHub(hub_name=hub_name, region=REGION, sagemaker_session=sagemaker_session)
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_bucket_name": "sagemaker-hubs-us-east-1-123456789123",
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "tags": tags,
    }
    response = hub.create(
        description=hub_description,
        display_name=hub_display_name,
        bucket_name=hub_bucket_name,
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
def test_create_with_bucket_name(
    sagemaker_session,
    hub_name,
    hub_description,
    hub_bucket_name,
    hub_display_name,
    hub_search_keywords,
    tags,
):
    create_hub = {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
    sagemaker_session.create_hub = Mock(return_value=create_hub)
    hub = CuratedHub(hub_name=hub_name, region=REGION, sagemaker_session=sagemaker_session)
    request = {
        "hub_name": hub_name,
        "hub_description": hub_description,
        "hub_bucket_name": hub_bucket_name,
        "hub_display_name": hub_display_name,
        "hub_search_keywords": hub_search_keywords,
        "tags": tags,
    }
    response = hub.create(
        description=hub_description,
        display_name=hub_display_name,
        bucket_name=hub_bucket_name,
        search_keywords=hub_search_keywords,
        tags=tags,
    )
    sagemaker_session.create_hub.assert_called_with(**request)
    assert response == {"HubArn": f"arn:aws:sagemaker:us-east-1:123456789123:hub/{hub_name}"}
