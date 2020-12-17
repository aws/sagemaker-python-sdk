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

import logging
import math

from sagemaker.apiutils import _base_types, _utils
from sagemaker.lineage import _api_types
from sagemaker.lineage._utils import get_module, _disassociate
from sagemaker.lineage.association import Association

LOGGER = logging.getLogger("sagemaker")


class Artifact(_base_types.Record):
    """An Amazon SageMaker artifact, which is part of a SageMaker lineage.

    Examples:
        .. code-block:: python

            from sagemaker.lineage import artifact

            my_artifact = artifact.Artifact.create(
                artifact_name='MyArtifact',
                artifact_type='S3File',
                source_uri='s3://...')

            my_artifact.properties["added"] = "property"
            my_artifact.save()

            for artfct in artifact.Artifact.list():
                print(artfct)

            my_artifact.delete()

    Attributes:
        artifact_arn (str): The ARN of the artifact.
        artifact_name (str): The name of the artifact.
        artifact_type (str): The type of the artifact.
        source (obj): The source of the artifact with a URI and types.
        properties (dict): Dictionary of properties.
        tags (List[dict[str, str]]): A list of tags to associate with the artifact.
        creation_time (datetime): When the artifact was created.
        created_by (obj): Contextual info on which account created the artifact.
    """

    artifact_arn = None
    artifact_name = None
    artifact_type = None
    source = None
    properties = None
    tags = None
    creation_time = None
    created_by = None
    last_modified_time = None
    last_modified_by = None

    _boto_create_method = "create_artifact"
    _boto_load_method = "describe_artifact"
    _boto_update_method = "update_artifact"
    _boto_delete_method = "delete_artifact"

    _boto_update_members = ["artifact_arn", "artifact_name", "properties", "properties_to_remove"]

    _boto_delete_members = ["artifact_arn"]

    _custom_boto_types = {"source": (_api_types.ArtifactSource, False)}

    def save(self):
        """Save the state of this Artifact to SageMaker.

        Note that this method must be run from a SageMaker context such as Studio or a training job
        due to restrictions on the CreateArtifact API.

        Returns:
            Artifact: A SageMaker `Artifact` object.
        """
        return self._invoke_api(self._boto_update_method, self._boto_update_members)

    def delete(self, disassociate=False):
        """Delete the artifact object.

        Args:
            disassociate (bool): When set to true, disassociate incoming and outgoing association.
        """
        if disassociate:
            _disassociate(source_arn=self.artifact_arn, sagemaker_session=self.sagemaker_session)
            _disassociate(
                destination_arn=self.artifact_arn, sagemaker_session=self.sagemaker_session
            )
        self._invoke_api(self._boto_delete_method, self._boto_delete_members)

    @classmethod
    def load(cls, artifact_arn, sagemaker_session=None):
        """Load an existing artifact and return an ``Artifact`` object representing it.

        Args:
            artifact_arn (str): ARN of the artifact
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Artifact: A SageMaker ``Artifact`` object
        """
        artifact = cls._construct(
            cls._boto_load_method,
            artifact_arn=artifact_arn,
            sagemaker_session=sagemaker_session,
        )
        return artifact

    def downstream_trials(self, sagemaker_session=None):
        """Retrieve all trial runs which that use this artifact.

        Args:
            sagemaker_session (obj): Sagemaker Sesssion to use. If not provided a default session
                will be created.

        Returns:
            [Trial]: A list of SageMaker `Trial` objects.
        """
        # don't specify destination type because for Trial Components it could be one of
        # SageMaker[TrainingJob|ProcessingJob|TransformJob|ExperimentTrialComponent]
        outgoing_associations = Association.list(
            source_arn=self.artifact_arn, sagemaker_session=sagemaker_session
        )
        trial_component_arns = list(map(lambda x: x.destination_arn, outgoing_associations))

        if not trial_component_arns:
            # no outgoing associations for this artifact
            return []

        get_module("smexperiments")
        from smexperiments import trial_component, search_expression

        max_search_by_arn = 60
        num_search_batches = math.ceil(len(trial_component_arns) % max_search_by_arn)
        trial_components = []

        sagemaker_session = sagemaker_session or _utils.default_session()
        sagemaker_client = sagemaker_session.sagemaker_client

        for i in range(num_search_batches):
            start = i * max_search_by_arn
            end = start + max_search_by_arn
            arn_batch = trial_component_arns[start:end]
            se = self._get_search_expression(arn_batch, search_expression)
            search_result = trial_component.TrialComponent.search(
                search_expression=se, sagemaker_boto_client=sagemaker_client
            )

            trial_components = trial_components + list(search_result)

        trials = set()

        for tc in list(trial_components):
            for parent in tc.parents:
                trials.add(parent["TrialName"])

        return list(trials)

    def _get_search_expression(self, arns, search_expression):
        """Convert a set of arns to a search expression.

        Args:
            arns (list): Trial Component arns to search for.
            search_expression (obj): smexperiments.search_expression

        Returns:
            search_expression (obj): Arns converted to a Trial Component search expression.
        """
        max_arn_per_filter = 3
        num_filters = math.ceil(len(arns) / max_arn_per_filter)
        filters = []

        for i in range(num_filters):
            start = i * max_arn_per_filter
            end = i + max_arn_per_filter
            batch_arns = arns[start:end]
            search_filter = search_expression.Filter(
                name="TrialComponentArn",
                operator=search_expression.Operator.EQUALS,
                value=",".join(batch_arns),
            )

            filters.append(search_filter)

        search_expression = search_expression.SearchExpression(
            filters=filters,
            boolean_operator=search_expression.BooleanOperator.OR,
        )
        return search_expression

    def set_tag(self, tag=None):
        """Add a tag to the object.

        Args:
            tag (obj): Key value pair to set tag.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.artifact_arn, tags=[tag])

    def set_tags(self, tags=None):
        """Add tags to the object.

        Args:
            tags ([{key:value}]): list of key value pairs.

        Returns:
            list({str:str}): a list of key value pairs
        """
        return self._set_tags(resource_arn=self.artifact_arn, tags=tags)

    @classmethod
    def create(
        cls,
        artifact_name=None,
        source_uri=None,
        source_types=None,
        artifact_type=None,
        properties=None,
        tags=None,
        sagemaker_session=None,
    ):
        """Create an artifact and return an ``Artifact`` object representing it.

        Args:
            artifact_name (str, optional): Name of the artifact
            source_uri (str, optional): Source URI of the artifact
            source_types (list, optional): Source types
            artifact_type (str, optional): Type of the artifact
            properties (dict, optional): key/value properties
            tags (dict, optional): AWS tags for the artifact
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            Artifact: A SageMaker ``Artifact`` object.
        """
        return super(Artifact, cls)._construct(
            cls._boto_create_method,
            artifact_name=artifact_name,
            source=_api_types.ContextSource(source_uri=source_uri, source_types=source_types),
            artifact_type=artifact_type,
            properties=properties,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

    @classmethod
    def list(
        cls,
        source_uri=None,
        artifact_type=None,
        created_before=None,
        created_after=None,
        sort_by=None,
        sort_order=None,
        max_results=None,
        next_token=None,
        sagemaker_session=None,
    ):
        """Return a list of artifact summaries.

        Args:
            source_uri (str, optional): A source URI.
            artifact_type (str, optional): An artifact type.
            created_before (datetime.datetime, optional): Return artifacts created before this
                instant.
            created_after (datetime.datetime, optional): Return artifacts created after this
                instant.
            sort_by (str, optional): Which property to sort results by.
                One of 'SourceArn', 'CreatedBefore','CreatedAfter'
            sort_order (str, optional): One of 'Ascending', or 'Descending'.
            max_results (int, optional): maximum number of artifacts to retrieve
            next_token (str, optional): token for next page of results
            sagemaker_session (sagemaker.session.Session): Session object which
                manages interactions with Amazon SageMaker APIs and any other
                AWS services needed. If not specified, one is created using the
                default AWS configuration chain.

        Returns:
            collections.Iterator[ArtifactSummary]: An iterator
                over ``ArtifactSummary`` objects.
        """
        return super(Artifact, cls)._list(
            "list_artifacts",
            _api_types.ArtifactSummary.from_boto,
            "ArtifactSummaries",
            source_uri=source_uri,
            artifact_type=artifact_type,
            created_before=created_before,
            created_after=created_after,
            sort_by=sort_by,
            sort_order=sort_order,
            max_results=max_results,
            next_token=next_token,
            sagemaker_session=sagemaker_session,
        )


class ModelArtifact(Artifact):
    """A SageMaker lineage artifact representing a model.

    Common model specific lineage traversals to discover how the model is connected
    to otherentities.
    """

    def endpoints(self):
        """Given a model artifact, get all associated endpoint context.

        Returns:
            [AssociationSummary]: A list of associations repesenting the endpoints using the model.
        """
        endpoint_development_actions = Association.list(
            source_arn=self.artifact_arn,
            destination_type="Action",
            sagemaker_session=self.sagemaker_session,
        )

        endpoint_context_list = [
            endpoint_context_associations
            for endpoint_development_action in endpoint_development_actions
            for endpoint_context_associations in Association.list(
                source_arn=endpoint_development_action.destination_arn,
                destination_type="Context",
                sagemaker_session=self.sagemaker_session,
            )
        ]
        return endpoint_context_list


class DatasetArtifact(Artifact):
    """A SageMaker Lineage artifact representing a dataset.

    Encapsulates common dataset specific lineage traversals to discover how the dataset is
    connect to related entities.
    """

    def trained_models(self):
        """Given a dataset artifact, get associated trained models.

        Returns:
            list(Association): List of Contexts representing model artifacts.
        """
        trial_components = Association.list(
            source_arn=self.artifact_arn, sagemaker_session=self.sagemaker_session
        )
        result = []
        for trial_component in trial_components:
            if "experiment-trial-component" in trial_component.destination_arn:
                models = Association.list(
                    source_arn=trial_component.destination_arn,
                    destination_type="Context",
                    sagemaker_session=self.sagemaker_session,
                )
                result.extend(models)

        return result
