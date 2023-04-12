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

    def _add_model_group(self, model_package_group, tag_rule_key, tag_rule_value):
        """To add a model package group to a collection

        Args:
            model_package_group (str): The name of the model package group
            tag_rule_key (str): The tag key of the corresponing collection to be added into
            tag_rule_value (str): The tag value of the corresponing collection to be added into
        """
        model_group_details = self.sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group
        )
        self.sagemaker_session.sagemaker_client.add_tags(
            ResourceArn=model_group_details["ModelPackageGroupArn"],
            Tags=[
                {
                    "Key": tag_rule_key,
                    "Value": tag_rule_value,
                }
            ],
        )

    def _remove_model_group(self, model_package_group, tag_rule_key):
        """To remove a model package group from a collection

        Args:
            model_package_group (str): The name of the model package group
            tag_rule_key (str): The tag key of the corresponing collection to be removed from
        """
        model_group_details = self.sagemaker_session.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=model_package_group
        )
        self.sagemaker_session.sagemaker_client.delete_tags(
            ResourceArn=model_group_details["ModelPackageGroupArn"], TagKeys=[tag_rule_key]
        )

    def create(self, collection_name: str, parent_collection_name: str = None):
        """Creates a collection

        Args:
            collection_name (str): The name of the collection to be created
            parent_collection_name (str): The name of the parent collection.
                To be None if the collection is to be created on the root level
        """

        tag_rule_key = f"sagemaker:collection-path:{int(time.time() * 1000)}"
        tags_on_collection = {
            "sagemaker:collection": "true",
            "sagemaker:collection-path:root": "true",
        }
        tag_rule_values = [collection_name]

        if parent_collection_name is not None:
            parent_tag_rules = self._get_collection_tag_rule(collection_name=parent_collection_name)
            parent_tag_rule_key = parent_tag_rules["tag_rule_key"]
            parent_tag_value = parent_tag_rules["tag_rule_value"]
            tags_on_collection = {
                parent_tag_rule_key: parent_tag_value,
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
        """Deletes a list of collection.

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

        # loops over the list of collection and deletes one at a time.
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

    def _get_collection_tag_rule(self, collection_name: str):
        """Returns the tag rule key and value for a collection"""

        if collection_name is not None:
            try:
                group_query = self.sagemaker_session.get_resource_group_query(group=collection_name)
            except ClientError as e:
                error_code = e.response["Error"]["Code"]

                if error_code == "NotFoundException":
                    raise ValueError(f"Cannot find collection: {collection_name}")
                self._check_access_error(err=e)
                raise
            if group_query.get("GroupQuery"):
                tag_rule_query = json.loads(
                    group_query["GroupQuery"].get("ResourceQuery", {}).get("Query", "")
                )
                tag_rule = tag_rule_query.get("TagFilters", [])[0]
                if not tag_rule:
                    raise "Unsupported parent_collection_name"
                tag_rule_value = tag_rule["Values"][0]
                tag_rule_key = tag_rule["Key"]

            return {
                "tag_rule_key": tag_rule_key,
                "tag_rule_value": tag_rule_value,
            }
        raise ValueError("Collection name is required")

    def add_model_groups(self, collection_name: str, model_groups: List[str]):
        """To add list of model package groups to a collection

        Args:
            collection_name (str): The name of the collection
            model_groups List[str]: Model pckage group names list to be added into the collection
        """
        if len(model_groups) > 10:
            raise Exception("Model groups can have a maximum length of 10")
        tag_rules = self._get_collection_tag_rule(collection_name=collection_name)
        tag_rule_key = tag_rules["tag_rule_key"]
        tag_rule_value = tag_rules["tag_rule_value"]

        add_groups_success = []
        add_groups_failure = []
        if tag_rule_key is not None and tag_rule_value is not None:
            for model_group in model_groups:
                try:
                    self._add_model_group(
                        model_package_group=model_group,
                        tag_rule_key=tag_rule_key,
                        tag_rule_value=tag_rule_value,
                    )
                    add_groups_success.append(model_group)
                except ClientError as e:
                    self._check_access_error(err=e)
                    message = e.response["Error"]["Message"]
                    add_groups_failure.append(
                        {
                            "model_group": model_group,
                            "failure_reason": message,
                        }
                    )
        return {
            "added_groups": add_groups_success,
            "failure": add_groups_failure,
        }

    def remove_model_groups(self, collection_name: str, model_groups: List[str]):
        """To remove list of model package groups from a collection

        Args:
            collection_name (str): The name of the collection
            model_groups List[str]: Model package group names list to be removed
        """

        if len(model_groups) > 10:
            raise Exception("Model groups can have a maximum length of 10")
        tag_rules = self._get_collection_tag_rule(collection_name=collection_name)

        tag_rule_key = tag_rules["tag_rule_key"]
        tag_rule_value = tag_rules["tag_rule_value"]

        remove_groups_success = []
        remove_groups_failure = []
        if tag_rule_key is not None and tag_rule_value is not None:
            for model_group in model_groups:
                try:
                    self._remove_model_group(
                        model_package_group=model_group,
                        tag_rule_key=tag_rule_key,
                    )
                    remove_groups_success.append(model_group)
                except ClientError as e:
                    self._check_access_error(err=e)
                    message = e.response["Error"]["Message"]
                    remove_groups_failure.append(
                        {
                            "model_group": model_group,
                            "failure_reason": message,
                        }
                    )
        return {
            "removed_groups": remove_groups_success,
            "failure": remove_groups_failure,
        }

    def move_model_group(
        self, source_collection_name: str, model_group: str, destination_collection_name: str
    ):
        """To move a model package group from one collection to another

        Args:
            source_collection_name (str): Collection name of the source
            model_group (str): Model package group names which is to be moved
            destination_collection_name (str): Collection name of the destination
        """
        remove_details = self.remove_model_groups(
            collection_name=source_collection_name, model_groups=[model_group]
        )
        if len(remove_details["failure"]) == 1:
            raise Exception(remove_details["failure"][0]["failure"])

        added_details = self.add_model_groups(
            collection_name=destination_collection_name, model_groups=[model_group]
        )

        if len(added_details["failure"]) == 1:
            # adding the model group back to the source collection in case of an add failure
            self.add_model_groups(
                collection_name=source_collection_name, model_groups=[model_group]
            )
            raise Exception(added_details["failure"][0]["failure"])

        return {
            "moved_success": model_group,
        }

    def _convert_tag_collection_response(self, tag_collections: List[str]):
        """Converts collection response from tag api to collection list response

        Args:
            tag_collections List[dict]: Collections list response from tag api
        """
        collection_details = []
        for collection in tag_collections:
            collection_arn = collection["ResourceARN"]
            collection_name = collection_arn.split("group/")[1]
            collection_details.append(
                {
                    "Name": collection_name,
                    "Arn": collection_arn,
                    "Type": "Collection",
                }
            )
        return collection_details

    def _convert_group_resource_response(
        self, group_resource_details: List[dict], is_model_group: bool = False
    ):
        """Converts collection response from resource group api to collection list response

        Args:
            group_resource_details (List[dict]): Collections list response from resource group api
            is_model_group (bool): If the reponse is of collection or model group type
        """
        collection_details = []
        if group_resource_details["Resources"]:
            for resource_group in group_resource_details["Resources"]:
                collection_arn = resource_group["Identifier"]["ResourceArn"]
                collection_name = collection_arn.split("group/")[1]
                collection_details.append(
                    {
                        "Name": collection_name,
                        "Arn": collection_arn,
                        "Type": resource_group["Identifier"]["ResourceType"]
                        if is_model_group
                        else "Collection",
                    }
                )
        return collection_details

    def _get_full_list_resource(self, collection_name, collection_filter):
        """Iterating to the full resource group list and returns appended paginated response

        Args:
            collection_name (str): Name of the collection to get the details
            collection_filter (dict): Filter details to be passed to get the resource list

        """
        list_group_response = self.sagemaker_session.list_group_resources(
            group=collection_name, filters=collection_filter
        )
        next_token = list_group_response.get("NextToken")
        while next_token is not None:

            paginated_group_response = self.sagemaker_session.list_group_resources(
                group=collection_name,
                filters=collection_filter,
                next_token=next_token,
            )
            list_group_response["Resources"] = (
                list_group_response["Resources"] + paginated_group_response["Resources"]
            )
            list_group_response["ResourceIdentifiers"] = (
                list_group_response["ResourceIdentifiers"]
                + paginated_group_response["ResourceIdentifiers"]
            )
            next_token = paginated_group_response.get("NextToken")

        return list_group_response

    def list_collection(self, collection_name: str = None):
        """To all list the collections and content of the collections

        In case there is no collection_name, it lists all the collections on the root level

        Args:
            collection_name (str): The name of the collection to list the contents of
        """
        collection_content = []
        if collection_name is None:
            tag_filters = [
                {
                    "Key": "sagemaker:collection-path:root",
                    "Values": ["true"],
                },
            ]
            resource_type_filters = ["resource-groups:group"]
            tag_collections = self.sagemaker_session.get_tagging_resources(
                tag_filters=tag_filters, resource_type_filters=resource_type_filters
            )

            return self._convert_tag_collection_response(tag_collections)

        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::ResourceGroups::Group"],
            },
        ]
        list_group_response = self._get_full_list_resource(
            collection_name=collection_name, collection_filter=collection_filter
        )
        collection_content = self._convert_group_resource_response(list_group_response)

        collection_filter = [
            {
                "Name": "resource-type",
                "Values": ["AWS::SageMaker::ModelPackageGroup"],
            },
        ]
        list_group_response = self._get_full_list_resource(
            collection_name=collection_name, collection_filter=collection_filter
        )

        collection_content = collection_content + self._convert_group_resource_response(
            list_group_response, True
        )

        return collection_content
