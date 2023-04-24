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
DESCRIBE_MODEL_PACKAGE_GROUP = {
    "ModelPackageGroupArn": "arn:aws:resource-groups:us-west-2:205984106344:group/group}"
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

GROUP_QUERY_RESPONSE = {
    "GroupQuery": {
        "ResourceQuery": {
            "Query": '{"TagFilters": [{"Key": "key", "Values": ["value"]}]}',
        }
    }
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
        default_bucket_prefix=None,
    )

    session_mock.create_group = Mock(
        name="create_collection", return_value=CREATE_COLLECTION_RESPONSE
    )
    session_mock.delete_resource_group = Mock(name="delete_resource_group", return_value=True)
    session_mock.list_group_resources = Mock(name="list_group_resources", return_value={})
    session_mock.get_resource_group_query = Mock(
        name="get_resource_group_query", return_value=GROUP_QUERY_RESPONSE
    )
    session_mock.sagemaker_client.describe_model_package_group = Mock(
        name="describe_model_package_group", return_value=DESCRIBE_MODEL_PACKAGE_GROUP
    )
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


def test_add_model_groups_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    add_response = collection.add_model_groups(
        collection_name=[COLLECTION_NAME], model_groups=["test-model-group"]
    )
    assert len(add_response["added_groups"]) == 1
    assert len(add_response["failure"]) == 0


def test_remove_model_groups_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    add_response = collection.remove_model_groups(
        collection_name=[COLLECTION_NAME], model_groups=["test-model-group"]
    )
    assert len(add_response["removed_groups"]) == 1
    assert len(add_response["failure"]) == 0


def test_add_and_remove_model_groups_limit(sagemaker_session):
    collection = Collection(sagemaker_session)
    model_groups = []
    for i in range(11):
        model_groups.append(f"test-model-group{i}")
    try:
        collection.add_model_groups(collection_name=[COLLECTION_NAME], model_groups=model_groups)
    except Exception as e:
        assert "Model groups can have a maximum length of 10" in str(e)

    try:
        collection.remove_model_groups(collection_name=[COLLECTION_NAME], model_groups=model_groups)
    except Exception as e:
        assert "Model groups can have a maximum length of 10" in str(e)
