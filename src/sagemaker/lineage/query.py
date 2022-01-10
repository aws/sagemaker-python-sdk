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

    def to_lineage_object(self):
        """Convert the ``Vertex`` object to its corresponding ``Artifact`` or ``Context`` object."""
        from sagemaker.lineage.artifact import Artifact, ModelArtifact
        from sagemaker.lineage.context import Context, EndpointContext
        from sagemaker.lineage.artifact import DatasetArtifact

        if self.lineage_entity == LineageEntityEnum.CONTEXT.value:
            resource_name = get_resource_name_from_arn(self.arn)
            if self.lineage_source == LineageSourceEnum.ENDPOINT.value:
                return EndpointContext.load(
                    context_name=resource_name, sagemaker_session=self._session
                )
            return Context.load(context_name=resource_name, sagemaker_session=self._session)

        if self.lineage_entity == LineageEntityEnum.ARTIFACT.value:
            if self.lineage_source == LineageSourceEnum.MODEL.value:
                return ModelArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
            if self.lineage_source == LineageSourceEnum.DATASET.value:
                return DatasetArtifact.load(artifact_arn=self.arn, sagemaker_session=self._session)
            return Artifact.load(artifact_arn=self.arn, sagemaker_session=self._session)

        raise ValueError("Vertex cannot be converted to a lineage object.")


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
        if self.entities:
            filter_request["Types"] = list(
                map(lambda x: x.value if isinstance(x, LineageSourceEnum) else x, self.sources)
            )
        if self.sources:
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
        return Vertex(
            arn=vertex["Arn"],
            lineage_source=vertex["Type"],
            lineage_entity=vertex["LineageType"],
            sagemaker_session=self._session,
        )

    def _convert_api_response(self, response) -> LineageQueryResult:
        """Convert the lineage query API response to its Python representation."""
        converted = LineageQueryResult()
        converted.edges = [self._get_edge(edge) for edge in response["Edges"]]
        converted.vertices = [self._get_vertex(vertex) for vertex in response["Vertices"]]

        return converted

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

        return self._convert_api_response(query_response)
