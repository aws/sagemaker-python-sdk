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
"""This module contains code to create and manage SageMaker ``Actions``."""
from __future__ import absolute_import

from sagemaker.apiutils import _base_types
from sagemaker.lineage import _api_types, _utils


class Action(_base_types.Record):
    """An Amazon SageMaker action, which is part of a SageMaker lineage.

    Examples:
        .. code-block:: python

            from sagemaker.lineage import action

            my_action = action.Action.create(
                action_name='MyAction',
                action_type='EndpointDeployment',
                source_uri='s3://...')

            my_action.properties["added"] = "property"
            my_action.save()

            for actn in action.Action.list():
                print(actn)

            my_action.delete()

    Attributes:
        action_arn (str): The ARN of the action.
        action_name (str): The name of the action.
        action_type (str): The type of the action.
        description (str): A description of the action.
        status (str): The status of the action.
        source (obj): The source of the action with a URI and type.
        properties (dict): Dictionary of properties.
        tags (List[dict[str, str]]): A list of tags to associate with the action.
        creation_time (datetime): When the action was created.
        created_by (obj): Contextual info on which account created the action.
        last_modified_time (datetime): When the action was last modified.
        last_modified_by (obj): Contextual info on which account created the action.
    """

    action_arn = None
    action_name = None
    action_type = None
    description = None
    status = None
    source = None
    properties = None
    properties_to_remove = None
    tags = None
    creation_time = None
    created_by = None
    last_modified_time = None
    last_modified_by = None

    _boto_create_method = "create_action"
    _boto_load_method = "describe_action"
    _boto_update_method = "update_action"
    _boto_delete_method = "delete_action"

    _boto_update_members = [
        "action_name",
        "description",
        "status",
        "properties",
        "properties_to_remove",
    ]

    _boto_delete_members = ["action_name"]

    _custom_boto_types = {"source": (_api_types.ActionSource, False)}

    def save(self):
        """Save the state of this Action to SageMaker.

        Returns:
            Action: A SageMaker ``Action``object.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self, disassociate=False):
        """Delete the action.

        Args:
            disassociate (bool): When set to true, disassociate incoming and outgoing association.

        """
        if disassociate:
            _utils._disassociate(
                source_arn=self.action_arn, sagemaker_session=self.sagemaker_session
            )
            _utils._disassociate(
                destination_arn=self.action_arn, sagemaker_session=self.sagemaker_session
            )

        self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, action_name, sagemaker_session=None):
        """Load an existing action and return an ``Action`` object representing it.

        Args:
            action_name (str): Name of the action
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Action: A SageMaker ``Action`` object
        """
        result = cls._construct(
            cls._boto_load_method,
            action_name=action_name,
            sagemaker_session=sagemaker_session,
        )
        return result

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.action_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.action_arn, tags=tags)

    @classmethod
    def create(
        cls,
        action_name=None,
        source_uri=None,
        source_type=None,
        action_type=None,
        description=None,
        status=None,
        properties=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Create an action and return an ``Action`` object representing it.

        Args:
            action_name (str): Name of the action
            source_uri (str): Source URI of the action
            source_type (str): Source type of the action
            action_type (str): The type of the action
            description (str): Description of the action
            status (str): Status of the action.
            properties (dict): key/value properties
            tags (dict): AWS tags for the action
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Action: A SageMaker ``Action`` object.
        """
        return super(Action, cls)._construct(
            cls._boto_create_method,
            action_name=action_name,
            source=_api_types.ContextSource(source_uri=source_uri, source_type=source_type),
            action_type=action_type,
            description=description,
            status=status,
            properties=properties,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_uri=None,
        action_type=None,
        created_after=None,
        created_before=None,
        sort_by=None,
        sort_order=None,
        sagemaker_session=None,
        max_results=None,
        next_token=None,
    ):
        """Return a list of action summaries.

        Args:
            source_uri (str, optional): A source URI.
            action_type (str, optional): An action type.
            created_before (datetime.datetime, optional): Return actions created before this
                instant.
            created_after (datetime.datetime, optional): Return actions created after this instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore', 'CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of actions to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[ActionSummary]: An iterator
                over ``ActionSummary`` objects.
        """
        return super(Action, cls)._list(
            "list_actions",
            _api_types.ActionSummary.from_boto,
            "ActionSummaries",
            source_uri=source_uri,
            action_type=action_type,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            sagemaker_session=sagemaker_session,
            max_results=max_results,
            next_token=next_token,
        )
