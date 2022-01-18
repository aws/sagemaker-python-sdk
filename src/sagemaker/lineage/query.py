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
"""This module contains code to query SageMaker lineage."""
from __future__ import absolute_import

from datetime import datetime
from enum import Enum
from typing import Optional, Union, List, Dict

from sagemaker.lineage._utils import get_resource_name_from_arn


class LineageEntityEnum(Enum):
    """Enum of lineage entities for use in a query filter."""

    TRIAL = "Trial"
    ACTION = "Action"
    ARTIFACT = "Artifact"
    CONTEXT = "Context"
    TRIAL_COMPONENT = "TrialComponent"


class LineageSourceEnum(Enum):
    """Enum of lineage types for use in a query filter."""

    CHECKPOINT = "Checkpoint"
    DATASET = "DataSet"
    ENDPOINT = "Endpoint"
    IMAGE = "Image"
    MODEL = "Model"
    MODEL_DATA = "ModelData"
    MODEL_DEPLOYMENT = "ModelDeployment"
    MODEL_GROUP = "ModelGroup"
    MODEL_REPLACE = "ModelReplaced"
    TENSORBOARD = "TensorBoard"
    TRAINING_JOB = "TrainingJob"
    APPROVAL = "Approval"
    PROCESSING_JOB = "ProcessingJob"
    TRANSFORM_JOB = "TransformJob"


class LineageQueryDirectionEnum(Enum):
    """Enum of query filter directions."""

    BOTH = "Both"
    ASCENDANTS = "Ascendants"
    DESCENDANTS = "Descendants"


class Edge:
    """A connecting edge for a lineage graph."""

    def __init__(
        self,
        source_arn: str,
        destination_arn: str,
        association_type: str,
    ):
        """Initialize ``Edge`` instance."""
        self.source_arn = source_arn
        self.destination_arn = destination_arn
        self.association_type = association_type

    def __hash__(self):
        """Define hash function for ``Edge``."""
        return hash(
            (
                "source_arn",
                self.source_arn,
                "destination_arn",
                self.destination_arn,
                "association_type",
                self.association_type,
            )
        )

    def __eq__(self, other):
        """Define equal function for ``Edge``."""
        return (
            self.association_type == other.association_type
            and self.source_arn == other.source_arn
            and self.destination_arn == other.destination_arn
        )


class Vertex:
    """A vertex for a lineage graph."""

    def __init__(
        self,
        arn: str,
        lineage_entity: str,
        lineage_source: str,
        sagemaker_session,
    ):
        """Initialize ``Vertex`` instance."""
        self.arn = arn
        self.lineage_entity = lineage_entity
        self.lineage_source = lineage_source
        self._session = sagemaker_session

    def __hash__(self):
        """Define hash function for ``Vertex``."""
        return hash(
            (
                "arn",
                self.arn,
                "lineage_entity",
                self.lineage_entity,
                "lineage_source",
                self.lineage_source,
            )
        )

    def __eq__(self, other):
        """Define equal function for ``Vertex``."""
        return (
            self.arn == other.arn
            and self.lineage_entity == other.lineage_entity
            and self.lineage_source == other.lineage_source
        )

    def to_lineage_object(self):
        """Convert the ``Vertex`` object to its corresponding lineage object.

        Returns:
            A ``Vertex`` object to its corresponding ``Artifact``,``Action``, ``Context``
            or ``TrialComponent`` object.
        """
        from sagemaker.lineage.context import Context, EndpointContext
        from sagemaker.lineage.action import Action
        from sagemaker.lineage.lineage_trial_component import LineageTrialComponent

        if self.lineage_entity == LineageEntityEnum.CONTEXT.value:
            resource_name = get_resource_name_from_arn(self.arn)
            if self.lineage_source == LineageSourceEnum.ENDPOINT.value:
                return EndpointContext.load(
                    context_name=resource_name, sagemaker_session=self._session
                )
            return Context.load(context_name=resource_name, sagemaker_session=self._session)

        if self.lineage_entity == LineageEntityEnum.ARTIFACT.value:
            return self._artifact_to_lineage_object()

        if self.lineage_entity == LineageEntityEnum.ACTION.value:
            return Action.load(action_name=self.arn.split("/")[1], sagemaker_session=self._session)

        if self.lineage_entity == LineageEntityEnum.TRIAL_COMPONENT.value:
            trial_component_name = get_resource_name_from_arn(self.arn)
            return LineageTrialComponent.load(
                trial_component_name=trial_component_name, sagemaker_session=self._session
            )
        raise ValueError("Vertex cannot be converted to a lineage object.")

    def _artifact_to_lineage_object(self):
        """Convert the ``Vertex`` object to its corresponding ``Artifact``."""
        from sagemaker.lineage.artifact import Artifact, ModelArtifact, ImageArtifact
        from sagemaker.lineage.artifact import DatasetArtifact

        if self.lineage_source == LineageSourceEnum.MODEL.value:
            return ModelArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        if self.lineage_source == LineageSourceEnum.DATASET.value:
            return DatasetArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        if self.lineage_source == LineageSourceEnum.IMAGE.value:
            return ImageArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
        return Artifact.load(artifact_arn=self.arn, sagemaker_session=self._session)


class LineageQueryResult(object):
    """A wrapper around the results of a lineage query."""

    def __init__(
        self,
        edges: List[Edge] = None,
        vertices: List[Vertex] = None,
    ):
        """Init for LineageQueryResult.

        Args:
            edges (List[Edge]): The edges of the query result.
            vertices (List[Vertex]): The vertices of the query result.
        """
        self.edges = []
        self.vertices = []

        if edges is not None:
            self.edges = edges

        if vertices is not None:
            self.vertices = vertices


class LineageFilter(object):
    """A filter used in a lineage query."""

    def __init__(
        self,
        entities: Optional[List[Union[LineageEntityEnum, str]]] = None,
        sources: Optional[List[Union[LineageSourceEnum, str]]] = None,
        created_before: Optional[datetime] = None,
        created_after: Optional[datetime] = None,
        modified_before: Optional[datetime] = None,
        modified_after: Optional[datetime] = None,
        properties: Optional[Dict[str, str]] = None,
    ):
        """Initialize ``LineageFilter`` instance."""
        self.entities = entities
        self.sources = sources
        self.created_before = created_before
        self.created_after = created_after
        self.modified_before = modified_before
        self.modified_after = modified_after
        self.properties = properties

    def _to_request_dict(self):
        """Convert the lineage filter to its API representation."""
        filter_request = {}
        if self.sources:
            filter_request["Types"] = list(
                map(lambda x: x.value if isinstance(x, LineageSourceEnum) else x, self.sources)
            )
        if self.entities:
            filter_request["LineageTypes"] = list(
                map(lambda x: x.value if isinstance(x, LineageEntityEnum) else x, self.entities)
            )
        if self.created_before:
            filter_request["CreatedBefore"] = self.created_before
        if self.created_after:
            filter_request["CreatedAfter"] = self.created_after
        if self.modified_before:
            filter_request["ModifiedBefore"] = self.modified_before
        if self.modified_after:
            filter_request["ModifiedAfter"] = self.modified_after
        if self.properties:
            filter_request["Properties"] = self.properties
        return filter_request


class LineageQuery(object):
    """Creates an object used for performing lineage queries."""

    def __init__(self, sagemaker_session):
        """Initialize ``LineageQuery`` instance."""
        self._session = sagemaker_session

    def _get_edge(self, edge):
        """Convert lineage query API response to an Edge."""
        return Edge(
            source_arn=edge["SourceArn"],
            destination_arn=edge["DestinationArn"],
            association_type=edge["AssociationType"] if "AssociationType" in edge else None,
        )

    def _get_vertex(self, vertex):
        """Convert lineage query API response to a Vertex."""
        vertex_type = None
        if "Type" in vertex:
            vertex_type = vertex["Type"]
        return Vertex(
            arn=vertex["Arn"],
            lineage_source=vertex_type,
            lineage_entity=vertex["LineageType"],
            sagemaker_session=self._session,
        )

    def _convert_api_response(self, response) -> LineageQueryResult:
        """Convert the lineage query API response to its Python representation."""
        converted = LineageQueryResult()
        converted.edges = [self._get_edge(edge) for edge in response["Edges"]]
        converted.vertices = [self._get_vertex(vertex) for vertex in response["Vertices"]]

        edge_set = set()
        for edge in converted.edges:
            if edge in edge_set:
                converted.edges.remove(edge)
            edge_set.add(edge)

        vertex_set = set()
        for vertex in converted.vertices:
            if vertex in vertex_set:
                converted.vertices.remove(vertex)
            vertex_set.add(vertex)

        return converted

    def _collapse_cross_account_artifacts(self, query_response):
        """Collapse the duplicate vertices and edges for cross-account."""
        for edge in query_response.edges:
            if (
                "artifact" in edge.source_arn
                and "artifact" in edge.destination_arn
                and edge.source_arn.split("/")[1] == edge.destination_arn.split("/")[1]
                and edge.source_arn != edge.destination_arn
            ):
                edge_source_arn = edge.source_arn
                edge_destination_arn = edge.destination_arn
                self._update_cross_account_edge(
                    edges=query_response.edges,
                    arn=edge_source_arn,
                    duplicate_arn=edge_destination_arn,
                )
                self._update_cross_account_vertex(
                    query_response=query_response, duplicate_arn=edge_destination_arn
                )

        # remove the duplicate edges from cross account
        new_edge = [e for e in query_response.edges if not e.source_arn == e.destination_arn]
        query_response.edges = new_edge

        return query_response

    def _update_cross_account_edge(self, edges, arn, duplicate_arn):
        """Replace the duplicate arn with arn in edges list."""
        for idx, e in enumerate(edges):
            if e.destination_arn == duplicate_arn:
                edges[idx].destination_arn = arn
            elif e.source_arn == duplicate_arn:
                edges[idx].source_arn = arn

    def _update_cross_account_vertex(self, query_response, duplicate_arn):
        """Remove the vertex with duplicate arn in the vertices list."""
        query_response.vertices = [v for v in query_response.vertices if not v.arn == duplicate_arn]

    def query(
        self,
        start_arns: List[str],
        direction: LineageQueryDirectionEnum = LineageQueryDirectionEnum.BOTH,
        include_edges: bool = True,
        query_filter: LineageFilter = None,
        max_depth: int = 10,
    ) -> LineageQueryResult:
        """Perform a lineage query.

        Args:
            start_arns (List[str]): A list of ARNs that will be used as the starting point
                for the query.
            direction (LineageQueryDirectionEnum, optional): The direction of the query.
            include_edges (bool, optional): If true, return edges in addition to vertices.
            query_filter (LineageQueryFilter, optional): The query filter.

        Returns:
            LineageQueryResult: The lineage query result.
        """
        query_response = self._session.sagemaker_client.query_lineage(
            StartArns=start_arns,
            Direction=direction.value,
            IncludeEdges=include_edges,
            Filters=query_filter._to_request_dict() if query_filter else {},
            MaxDepth=max_depth,
        )
        query_response = self._convert_api_response(query_response)
        query_response = self._collapse_cross_account_artifacts(query_response)

        return query_response
