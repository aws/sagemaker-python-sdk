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
"""This module contains code to create and manage SageMaker ``Artifact``."""
from __future__ import absolute_import

from sagemaker.apiutils import _base_types
from sagemaker.lineage import _api_types


class Association(_base_types.Record):
    """An Amazon SageMaker artifact, which is part of a SageMaker lineage.

    Examples:
        .. code-block:: python

            from sagemaker.lineage import association

            my_association = association.Association.create(
                source_arn=artifact_arn,
                destination_arn=trial_component_arn,
                association_type='ContributedTo')

            for assoctn in association.Association.list():
                print(assoctn)

            my_association.delete()

    Attributes:
        source_arn (str): The ARN of the source entity.
        source_type (str): The type of the source entity.
        destination_arn (str): The ARN of the destination entity.
        destination_type (str): The type of the destination entity.
        association_type (str): the type of the association.
    """

    source_arn = None
    destination_arn = None

    _boto_create_method = "add_association"
    _boto_delete_method = "delete_association"

    _custom_boto_types = {}

    _boto_delete_members = [
        "source_arn",
        "destination_arn",
    ]

    def delete(self):
        """Delete this Association from SageMaker."""
        self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:
            tag (obj): Key value pair to set tag.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.source_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.source_arn, tags=tags)

    @classmethod
    def create(
        cls,
        source_arn,
        destination_arn,
        association_type=None,
        sagemaker_session=None,
    ):
        """Add an association and return an ``Association`` object representing it.

        Args:
            source_arn (str): The ARN of the source.
            destination_arn (str): The ARN of the destination.
            association_type (str): The type of the association. ContributedTo, AssociatedWith,
                DerivedFrom, or Produced.
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            association: A SageMaker ``Association`` object.
        """
        return super(Association, cls)._construct(
            cls._boto_create_method,
            source_arn=source_arn,
            destination_arn=destination_arn,
            association_type=association_type,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_arn=None,
        destination_arn=None,
        source_type=None,
        destination_type=None,
        association_type=None,
        created_after=None,
        created_before=None,
        sort_by=None,
        sort_order=None,
        max_results=None,
        next_token=None,
        sagemaker_session=None,
    ):
        """Return a list of context summaries.

        Args:
            source_arn (str): The ARN of the source entity.
            destination_arn (str): The ARN of the destination entity.
            source_type (str): The type of the source entity.
            destination_type (str): The type of the destination entity.
            association_type (str): The type of the association.
            created_after (datetime.datetime, optional): Return contexts created after this
                instant.
            created_before (datetime.datetime, optional): Return contexts created before this
                instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of contexts to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[AssociationSummary]: An iterator
                over ``AssociationSummary`` objects.
        """
        return super(Association, cls)._list(
            "list_associations",
            _api_types.AssociationSummary.from_boto,
            "AssociationSummaries",
            source_arn=source_arn,
            destination_arn=destination_arn,
            source_type=source_type,
            destination_type=destination_type,
            association_type=association_type,
            created_after=created_after,
            created_before=created_before,
            sort_by=sort_by,
            sort_order=sort_order,
            max_results=max_results,
            next_token=next_token,
            sagemaker_session=sagemaker_session,
        )
