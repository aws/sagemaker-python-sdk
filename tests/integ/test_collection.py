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

from sagemaker.utils import unique_name_from_base
from sagemaker.collection import Collection


def test_create_collection_root_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    collection_name = unique_name_from_base("test-collection")
    collection.create(collection_name)
    collection_filter = [
        {
            "Name": "resource-type",
            "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
        },
    ]
    collection_details = sagemaker_session.list_group_resources(
        group=collection_name, filters=collection_filter
    )
    assert collection_details["ResponseMetadata"]["HTTPStatusCode"] == 200
    delete_response = collection.delete([collection_name])
    assert len(delete_response["deleted_collections"]) == 1
    assert len(delete_response["delete_collection_failures"]) == 0


def test_create_collection_nested_success(sagemaker_session):
    collection = Collection(sagemaker_session)
    collection_name = unique_name_from_base("test-collection")
    child_collection_name = unique_name_from_base("test-collection-2")
    collection.create(collection_name)
    collection.create(collection_name=child_collection_name, parent_collection_name=collection_name)
    collection_filter = [
        {
            "Name": "resource-type",
            "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
        },
    ]
    collection_details = sagemaker_session.list_group_resources(
        group=collection_name, filters=collection_filter
    )
    # has one child i.e child collection
    assert len(collection_details["Resources"]) == 1

    collection_details = sagemaker_session.list_group_resources(
        group=child_collection_name, filters=collection_filter
    )
    collection_details["ResponseMetadata"]["HTTPStatusCode"]
    delete_response = collection.delete([child_collection_name, collection_name])
    assert len(delete_response["deleted_collections"]) == 2
    assert len(delete_response["delete_collection_failures"]) == 0


def test_add_remove_model_groups_in_collection_success(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )
    collection = Collection(sagemaker_session)
    collection_name = unique_name_from_base("test-collection")
    collection.create(collection_name)
    model_groups = []
    model_groups.append(model_group_name)
    add_response = collection.add_model_groups(
        collection_name=collection_name, model_groups=model_groups
    )
    collection_filter = [
        {
            "Name": "resource-type",
            "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
        },
    ]
    collection_details = sagemaker_session.list_group_resources(
        group=collection_name, filters=collection_filter
    )

    assert len(add_response["failure"]) == 0
    assert len(add_response["added_groups"]) == 1
    assert len(collection_details["Resources"]) == 1

    remove_response = collection.remove_model_groups(
        collection_name=collection_name, model_groups=model_groups
    )
    collection_details = sagemaker_session.list_group_resources(
        group=collection_name, filters=collection_filter
    )
    assert len(remove_response["failure"]) == 0
    assert len(remove_response["removed_groups"]) == 1
    assert len(collection_details["Resources"]) == 0

    delete_response = collection.delete([collection_name])
    assert len(delete_response["deleted_collections"]) == 1
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )


def test_move_model_groups_in_collection_success(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )
    collection = Collection(sagemaker_session)
    source_collection_name = unique_name_from_base("test-collection-source")
    destination_collection_name = unique_name_from_base("test-collection-destination")
    collection.create(source_collection_name)
    collection.create(destination_collection_name)
    model_groups = []
    model_groups.append(model_group_name)
    add_response = collection.add_model_groups(
        collection_name=source_collection_name, model_groups=model_groups
    )
    collection_filter = [
        {
            "Name": "resource-type",
            "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
        },
    ]
    collection_details = sagemaker_session.list_group_resources(
        group=source_collection_name, filters=collection_filter
    )

    assert len(add_response["failure"]) == 0
    assert len(add_response["added_groups"]) == 1
    assert len(collection_details["Resources"]) == 1

    move_response = collection.move_model_group(
        source_collection_name=source_collection_name,
        model_group=model_group_name,
        destination_collection_name=destination_collection_name,
    )

    assert move_response["moved_success"] == model_group_name

    collection_details = sagemaker_session.list_group_resources(
        group=destination_collection_name, filters=collection_filter
    )

    assert len(collection_details["Resources"]) == 1

    collection_details = sagemaker_session.list_group_resources(
        group=source_collection_name, filters=collection_filter
    )
    assert len(collection_details["Resources"]) == 0

    remove_response = collection.remove_model_groups(
        collection_name=destination_collection_name, model_groups=model_groups
    )

    assert len(remove_response["failure"]) == 0
    assert len(remove_response["removed_groups"]) == 1

    delete_response = collection.delete([source_collection_name, destination_collection_name])
    assert len(delete_response["deleted_collections"]) == 2
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )


def test_list_collection_success(sagemaker_session):
    model_group_name = unique_name_from_base("test-model-group")
    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_group_name
    )
    collection = Collection(sagemaker_session)
    collection_name = unique_name_from_base("test-collection")
    collection.create(collection_name)
    model_groups = []
    model_groups.append(model_group_name)
    collection.add_model_groups(collection_name=collection_name, model_groups=model_groups)
    child_collection_name = unique_name_from_base("test-collection")
    collection.create(parent_collection_name=collection_name, collection_name=child_collection_name)
    root_collections = collection.list_collection()
    is_collection_found = False
    for root_collection in root_collections:
        if root_collection["Name"] == collection_name:
            is_collection_found = True
    assert is_collection_found

    collection_content = collection.list_collection(collection_name)
    assert len(collection_content) == 2

    collection.remove_model_groups(collection_name=collection_name, model_groups=model_groups)
    collection.delete([child_collection_name, collection_name])
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_group_name
    )
