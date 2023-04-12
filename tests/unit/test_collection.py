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
import json
from mock import Mock

from sagemaker.collection import Collection

REGION = "us-west-2"
COLLECTION_NAME = "test-collection"
QUERY = {
    "ResourceTypeFilters": ["AWS::SageMaker::ModelPackageGroup", "AWS::ResourceGroups::Group"],
    "TagFilters": [
        {"Key": "sagemaker:collection-path:1676120428.4811652", "Values": ["test-collection-k"]}
    ],
}
CREATE_COLLECTION_RESPONSE = {
    "Group": {
        "GroupArn": f"arn:aws:resource-groups:us-west-2:205984106344:group/{COLLECTION_NAME}",
        "Name": COLLECTION_NAME,
    },
    "ResourceQuery": {
        "Type": "TAG_FILTERS_1_0",
        "Query": json.dumps(QUERY),
    },
    "Tags": {"sagemaker:collection-path:root": "true"},
}


@pytest.fixture()
def sagemaker_session():
    boto_mock = Mock(name="boto_session", region_name=REGION)
    session_mock = Mock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name=REGION,
        config=None,
        local_mode=False,
    )

    session_mock.create_group = Mock(
        name="create_collection", return_value=CREATE_COLLECTION_RESPONSE
    )
    session_mock.delete_resource_group = Mock(name="delete_resource_group", return_value=True)
    session_mock.list_group_resources = Mock(name="list_group_resources", return_value={})

    return session_mock


def test_create_collection_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    create_response = collection.create(collection_name=COLLECTION_NAME)
    assert create_response["Name"] is COLLECTION_NAME
    assert create_response["Arn"] is not None


def test_delete_collection_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    delete_response = collection.delete(collections=[COLLECTION_NAME])
    assert len(delete_response["deleted_collections"]) == 1
    assert len(delete_response["delete_collection_failures"]) == 0


def test_delete_collection_failure_when_collection_is_not_empty(sagemaker_session):
    collection = Collection(sagemaker_session)
    sagemaker_session.list_group_resources = Mock(
        name="list_group_resources", return_value={"Resources": [{}]}
    )
    delete_response = collection.delete(collections=[COLLECTION_NAME])
    assert len(delete_response["deleted_collections"]) == 0
    assert len(delete_response["delete_collection_failures"]) == 1
