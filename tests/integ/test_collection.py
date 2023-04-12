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
