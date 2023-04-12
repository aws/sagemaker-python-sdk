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

"""This module contains code related to Amazon SageMaker Collection.

These Classes helps in providing features to maintain and create collections
"""

from __future__ import absolute_import
import json
import time
from typing import List


from botocore.exceptions import ClientError
from sagemaker.session import Session


class Collection(object):
    """Sets up Amazon SageMaker Collection."""

    def __init__(self, sagemaker_session):
        """Initializes a Collection instance.

        The collection provides a logical grouping for model groups

        Args:
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using
                the default AWS configuration chain.
        """
        self.sagemaker_session = sagemaker_session or Session()

    def _check_access_error(self, err: ClientError):
        """To check if the error is related to the access error and to provide the relavant message

        Args:
            err: The client error that needs to be checked
        """
        error_code = err.response["Error"]["Code"]
        if error_code == "AccessDeniedException":
            raise Exception(
                f"{error_code}: This account needs to attach a custom policy "
                "to the user role to gain access to Collections. Refer - "
                "https://docs.aws.amazon.com/sagemaker/latest/dg/modelcollections-permissions.html"
            )

    def create(self, collection_name: str, parent_collection_name: str = None):
        """Creates a collection

        Args:
            collection_name (str): The name of the collection to be created
            parent_collection_name (str): The name of the parent collection.
                To be None if the collection is to be created on the root level
        """

        tag_rule_key = f"sagemaker:collection-path:{time.time()}"
        tags_on_collection = {
            "sagemaker:collection": "true",
            "sagemaker:collection-path:root": "true",
        }
        tag_rule_values = [collection_name]

        if parent_collection_name is not None:
            try:
                group_query = self.sagemaker_session.get_resource_group_query(
                    group=parent_collection_name
                )
            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "NotFoundException":
                    raise ValueError(f"Cannot find collection: {parent_collection_name}")
                self._check_access_error(err=e)
                raise
            if group_query.get("GroupQuery"):
                parent_tag_rule_query = json.loads(
                    group_query["GroupQuery"].get("ResourceQuery", {}).get("Query", "")
                )
                parent_tag_rule = parent_tag_rule_query.get("TagFilters", [])[0]
                if not parent_tag_rule:
                    raise "Invalid parent_collection_name"
                parent_tag_value = parent_tag_rule["Values"][0]
                tags_on_collection = {
                    parent_tag_rule["Key"]: parent_tag_value,
                    "sagemaker:collection": "true",
                }
                tag_rule_values = [f"{parent_tag_value}/{collection_name}"]
        try:
            resource_filters = [
                "AWS::SageMaker::ModelPackageGroup",
                "AWS::ResourceGroups::Group",
            ]

            tag_filters = [
                {
                    "Key": tag_rule_key,
                    "Values": tag_rule_values,
                }
            ]
            resource_query = {
                "Query": json.dumps(
                    {"ResourceTypeFilters": resource_filters, "TagFilters": tag_filters}
                ),
                "Type": "TAG_FILTERS_1_0",
            }
            collection_create_response = self.sagemaker_session.create_group(
                collection_name, resource_query, tags_on_collection
            )
            return {
                "Name": collection_create_response["Group"]["Name"],
                "Arn": collection_create_response["Group"]["GroupArn"],
            }

        except ClientError as e:
            message = e.response["Error"]["Message"]
            error_code = e.response["Error"]["Code"]

            if error_code == "BadRequestException" and "group already exists" in message:
                raise ValueError("Collection with the given name already exists")

            self._check_access_error(err=e)
            raise

    def delete(self, collections: List[str]):
        """Deletes a lits of collection

        Args:
            collections (List[str]): List of collections to be deleted
                Only deletes a collection if it is empty
        """

        if len(collections) > 10:
            raise ValueError("Can delete upto 10 collections at a time")

        delete_collection_failures = []
        deleted_collection = []
        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::ResourceGroups::Group", "AWS::SageMaker::ModelPackageGroup"],
            },
        ]
        for collection in collections:
            try:
                collection_details = self.sagemaker_session.list_group_resources(
                    group=collection, filters=collection_filter
                )
            except ClientError as e:
                self._check_access_error(err=e)
                delete_collection_failures.append(
                    {"collection": collection, "message": e.response["Error"]["Message"]}
                )
                continue
            if collection_details.get("Resources") and len(collection_details["Resources"]) > 0:
                delete_collection_failures.append(
                    {"collection": collection, "message": "Validation error: Collection not empty"}
                )
            else:
                try:
                    self.sagemaker_session.delete_resource_group(group=collection)
                    deleted_collection.append(collection)
                except ClientError as e:
                    self._check_access_error(err=e)
                    delete_collection_failures.append(
                        {"collection": collection, "message": e.response["Error"]["Message"]}
                    )
        return {
            "deleted_collections": deleted_collection,
            "delete_collection_failures": delete_collection_failures,
        }
