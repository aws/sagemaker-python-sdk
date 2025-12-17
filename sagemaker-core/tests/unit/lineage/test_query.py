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

"""Unit tests for sagemaker.core.lineage.query module"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from sagemaker.core.lineage.query import (
    LineageEntityEnum,
    LineageSourceEnum,
    LineageQueryDirectionEnum,
    Edge,
    Vertex,
    LineageQueryResult,
    LineageFilter,
    LineageQuery,
    PyvisVisualizer,
)


class TestLineageEntityEnum:
    """Test cases for LineageEntityEnum"""

    def test_enum_values(self):
        """Test enum values"""
        assert LineageEntityEnum.TRIAL.value == "Trial"
        assert LineageEntityEnum.ACTION.value == "Action"
        assert LineageEntityEnum.ARTIFACT.value == "Artifact"
        assert LineageEntityEnum.CONTEXT.value == "Context"
        assert LineageEntityEnum.TRIAL_COMPONENT.value == "TrialComponent"


class TestLineageSourceEnum:
    """Test cases for LineageSourceEnum"""

    def test_enum_values(self):
        """Test enum values"""
        assert LineageSourceEnum.CHECKPOINT.value == "Checkpoint"
        assert LineageSourceEnum.DATASET.value == "DataSet"
        assert LineageSourceEnum.ENDPOINT.value == "Endpoint"
        assert LineageSourceEnum.MODEL.value == "Model"
        assert LineageSourceEnum.TRAINING_JOB.value == "TrainingJob"


class TestLineageQueryDirectionEnum:
    """Test cases for LineageQueryDirectionEnum"""

    def test_enum_values(self):
        """Test enum values"""
        assert LineageQueryDirectionEnum.BOTH.value == "Both"
        assert LineageQueryDirectionEnum.ASCENDANTS.value == "Ascendants"
        assert LineageQueryDirectionEnum.DESCENDANTS.value == "Descendants"


class TestEdge:
    """Test cases for Edge class"""

    def test_edge_creation(self):
        """Test edge creation"""
        edge = Edge(
            source_arn="arn:aws:sagemaker:us-west-2:123456789:artifact/source",
            destination_arn="arn:aws:sagemaker:us-west-2:123456789:artifact/dest",
            association_type="ContributedTo",
        )

        assert edge.source_arn == "arn:aws:sagemaker:us-west-2:123456789:artifact/source"
        assert edge.destination_arn == "arn:aws:sagemaker:us-west-2:123456789:artifact/dest"
        assert edge.association_type == "ContributedTo"

    def test_edge_equality(self):
        """Test edge equality"""
        edge1 = Edge("source1", "dest1", "type1")
        edge2 = Edge("source1", "dest1", "type1")
        edge3 = Edge("source2", "dest1", "type1")

        assert edge1 == edge2
        assert edge1 != edge3

    def test_edge_hash(self):
        """Test edge hashing"""
        edge1 = Edge("source1", "dest1", "type1")
        edge2 = Edge("source1", "dest1", "type1")

        assert hash(edge1) == hash(edge2)

        edge_set = {edge1, edge2}
        assert len(edge_set) == 1

    def test_edge_str(self):
        """Test edge string representation"""
        edge = Edge("source", "dest", "type")
        str_repr = str(edge)

        assert "source_arn" in str_repr
        assert "destination_arn" in str_repr
        assert "association_type" in str_repr


class TestVertex:
    """Test cases for Vertex class"""

    def test_vertex_creation(self):
        """Test vertex creation"""
        mock_session = Mock()
        vertex = Vertex(
            arn="arn:aws:sagemaker:us-west-2:123456789:artifact/test",
            lineage_entity="Artifact",
            lineage_source="Model",
            sagemaker_session=mock_session,
        )

        assert vertex.arn == "arn:aws:sagemaker:us-west-2:123456789:artifact/test"
        assert vertex.lineage_entity == "Artifact"
        assert vertex.lineage_source == "Model"

    def test_vertex_equality(self):
        """Test vertex equality"""
        mock_session = Mock()
        vertex1 = Vertex("arn1", "Artifact", "Model", mock_session)
        vertex2 = Vertex("arn1", "Artifact", "Model", mock_session)
        vertex3 = Vertex("arn2", "Artifact", "Model", mock_session)

        assert vertex1 == vertex2
        assert vertex1 != vertex3

    def test_vertex_hash(self):
        """Test vertex hashing"""
        mock_session = Mock()
        vertex1 = Vertex("arn1", "Artifact", "Model", mock_session)
        vertex2 = Vertex("arn1", "Artifact", "Model", mock_session)

        assert hash(vertex1) == hash(vertex2)

        vertex_set = {vertex1, vertex2}
        assert len(vertex_set) == 1

    @patch("sagemaker.core.lineage.context.EndpointContext")
    def test_to_lineage_object_context(self, mock_endpoint_context_class):
        """Test converting vertex to Context"""
        mock_session = Mock()
        mock_context = Mock()
        mock_endpoint_context_class.load.return_value = mock_context

        vertex = Vertex(
            "arn:aws:sagemaker:us-west-2:123456789:context/test-context",
            "Context",
            "Endpoint",
            mock_session,
        )

        result = vertex.to_lineage_object()
        # Should call EndpointContext.load for Endpoint source
        assert result is not None

    @patch("sagemaker.core.lineage.action.Action")
    def test_to_lineage_object_action(self, mock_action_class):
        """Test converting vertex to Action"""
        mock_session = Mock()
        mock_action = Mock()
        mock_action_class.load.return_value = mock_action

        vertex = Vertex(
            "arn:aws:sagemaker:us-west-2:123456789:action/test-action",
            "Action",
            "TrainingJob",
            mock_session,
        )

        result = vertex.to_lineage_object()
        assert result == mock_action

    def test_to_lineage_object_invalid(self):
        """Test converting invalid vertex"""
        mock_session = Mock()
        vertex = Vertex("arn", "InvalidType", "Source", mock_session)

        with pytest.raises(ValueError, match="cannot be converted"):
            vertex.to_lineage_object()


class TestLineageQueryResult:
    """Test cases for LineageQueryResult"""

    def test_empty_result(self):
        """Test empty query result"""
        result = LineageQueryResult()

        assert result.edges == []
        assert result.vertices == []
        assert result.startarn == []

    def test_result_with_data(self):
        """Test query result with data"""
        mock_session = Mock()
        edges = [Edge("source", "dest", "type")]
        vertices = [Vertex("arn", "Artifact", "Model", mock_session)]
        startarn = ["arn:start"]

        result = LineageQueryResult(edges, vertices, startarn)

        assert len(result.edges) == 1
        assert len(result.vertices) == 1
        assert len(result.startarn) == 1

    def test_convert_edges_to_tuples(self):
        """Test converting edges to tuples"""
        edges = [
            Edge("source1", "dest1", "type1"),
            Edge("source2", "dest2", "type2"),
        ]
        result = LineageQueryResult(edges=edges)

        tuples = result._covert_edges_to_tuples()
        assert len(tuples) == 2
        assert tuples[0] == ("source1", "dest1", "type1")

    def test_convert_vertices_to_tuples(self):
        """Test converting vertices to tuples"""
        mock_session = Mock()
        vertices = [
            Vertex("arn1", "Artifact", "Model", mock_session),
            Vertex("arn2", "Context", "Endpoint", mock_session),
        ]
        result = LineageQueryResult(vertices=vertices, startarn=["arn1"])

        tuples = result._covert_vertices_to_tuples()
        assert len(tuples) == 2
        assert tuples[0][0] == "arn1"
        assert tuples[0][3] is True  # is_start_arn
        assert tuples[1][3] is False  # not start_arn

    def test_get_visualization_elements(self):
        """Test getting visualization elements"""
        mock_session = Mock()
        edges = [Edge("source", "dest", "type")]
        vertices = [Vertex("arn", "Artifact", "Model", mock_session)]

        result = LineageQueryResult(edges, vertices)
        elements = result._get_visualization_elements()

        assert "nodes" in elements
        assert "edges" in elements
        assert len(elements["nodes"]) == 1
        assert len(elements["edges"]) == 1


class TestLineageFilter:
    """Test cases for LineageFilter"""

    def test_empty_filter(self):
        """Test empty filter"""
        filter_obj = LineageFilter()
        request_dict = filter_obj._to_request_dict()

        assert request_dict == {}

    def test_filter_with_entities(self):
        """Test filter with entities"""
        filter_obj = LineageFilter(entities=[LineageEntityEnum.ARTIFACT, LineageEntityEnum.ACTION])
        request_dict = filter_obj._to_request_dict()

        assert "LineageTypes" in request_dict
        assert len(request_dict["LineageTypes"]) == 2
        assert "Artifact" in request_dict["LineageTypes"]

    def test_filter_with_sources(self):
        """Test filter with sources"""
        filter_obj = LineageFilter(sources=[LineageSourceEnum.MODEL, LineageSourceEnum.DATASET])
        request_dict = filter_obj._to_request_dict()

        assert "Types" in request_dict
        assert len(request_dict["Types"]) == 2

    def test_filter_with_dates(self):
        """Test filter with date ranges"""
        created_before = datetime(2023, 1, 1)
        created_after = datetime(2022, 1, 1)

        filter_obj = LineageFilter(
            created_before=created_before,
            created_after=created_after,
        )
        request_dict = filter_obj._to_request_dict()

        assert "CreatedBefore" in request_dict
        assert "CreatedAfter" in request_dict

    def test_filter_with_properties(self):
        """Test filter with properties"""
        filter_obj = LineageFilter(properties={"key1": "value1", "key2": "value2"})
        request_dict = filter_obj._to_request_dict()

        assert "Properties" in request_dict
        assert request_dict["Properties"]["key1"] == "value1"

    def test_filter_with_string_entities(self):
        """Test filter with string entities"""
        filter_obj = LineageFilter(entities=["Artifact", "Action"])
        request_dict = filter_obj._to_request_dict()

        assert "LineageTypes" in request_dict
        assert "Artifact" in request_dict["LineageTypes"]


class TestLineageQuery:
    """Test cases for LineageQuery"""

    def test_query_creation(self):
        """Test query creation"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        assert query._session == mock_session

    def test_get_edge(self):
        """Test converting API edge to Edge object"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        api_edge = {
            "SourceArn": "source_arn",
            "DestinationArn": "dest_arn",
            "AssociationType": "ContributedTo",
        }

        edge = query._get_edge(api_edge)
        assert edge.source_arn == "source_arn"
        assert edge.destination_arn == "dest_arn"
        assert edge.association_type == "ContributedTo"

    def test_get_edge_without_association_type(self):
        """Test converting API edge without association type"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        api_edge = {
            "SourceArn": "source_arn",
            "DestinationArn": "dest_arn",
        }

        edge = query._get_edge(api_edge)
        assert edge.association_type is None

    def test_get_vertex(self):
        """Test converting API vertex to Vertex object"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        api_vertex = {
            "Arn": "test_arn",
            "Type": "Model",
            "LineageType": "Artifact",
        }

        vertex = query._get_vertex(api_vertex)
        assert vertex.arn == "test_arn"
        assert vertex.lineage_source == "Model"
        assert vertex.lineage_entity == "Artifact"

    def test_convert_api_response(self):
        """Test converting full API response"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        api_response = {
            "Edges": [
                {
                    "SourceArn": "source1",
                    "DestinationArn": "dest1",
                    "AssociationType": "ContributedTo",
                }
            ],
            "Vertices": [
                {
                    "Arn": "arn1",
                    "Type": "Model",
                    "LineageType": "Artifact",
                }
            ],
        }

        result = LineageQueryResult()
        converted = query._convert_api_response(api_response, result)

        assert len(converted.edges) == 1
        assert len(converted.vertices) == 1

    def test_convert_api_response_removes_duplicates(self):
        """Test that duplicate edges and vertices are removed"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        api_response = {
            "Edges": [
                {"SourceArn": "s1", "DestinationArn": "d1", "AssociationType": "type1"},
                {"SourceArn": "s1", "DestinationArn": "d1", "AssociationType": "type1"},
            ],
            "Vertices": [
                {"Arn": "arn1", "Type": "Model", "LineageType": "Artifact"},
                {"Arn": "arn1", "Type": "Model", "LineageType": "Artifact"},
            ],
        }

        result = LineageQueryResult()
        converted = query._convert_api_response(api_response, result)

        assert len(converted.edges) == 1
        assert len(converted.vertices) == 1

    def test_query_execution(self):
        """Test query execution"""
        mock_session = Mock()
        mock_session.sagemaker_client.query_lineage.return_value = {
            "Edges": [],
            "Vertices": [],
        }

        query = LineageQuery(mock_session)
        result = query.query(
            start_arns=["arn:start"],
            direction=LineageQueryDirectionEnum.BOTH,
        )

        assert isinstance(result, LineageQueryResult)
        mock_session.sagemaker_client.query_lineage.assert_called_once()

    def test_query_with_filter(self):
        """Test query with filter"""
        mock_session = Mock()
        mock_session.sagemaker_client.query_lineage.return_value = {
            "Edges": [],
            "Vertices": [],
        }

        query = LineageQuery(mock_session)
        filter_obj = LineageFilter(entities=[LineageEntityEnum.ARTIFACT])

        result = query.query(
            start_arns=["arn:start"],
            query_filter=filter_obj,
        )

        call_args = mock_session.sagemaker_client.query_lineage.call_args
        assert "Filters" in call_args[1]

    def test_collapse_cross_account_artifacts(self):
        """Test collapsing cross-account artifacts"""
        mock_session = Mock()
        query = LineageQuery(mock_session)

        # Create test data with cross-account artifacts
        edges = [
            Edge(
                "arn:aws:sagemaker:us-west-2:111:artifact/test-artifact",
                "arn:aws:sagemaker:us-west-2:222:artifact/test-artifact",
                "ContributedTo",
            )
        ]
        vertices = [
            Vertex(
                "arn:aws:sagemaker:us-west-2:111:artifact/test-artifact",
                "Artifact",
                "Model",
                mock_session,
            ),
            Vertex(
                "arn:aws:sagemaker:us-west-2:222:artifact/test-artifact",
                "Artifact",
                "Model",
                mock_session,
            ),
        ]

        query_response = LineageQueryResult(edges=edges, vertices=vertices)
        result = query._collapse_cross_account_artifacts(query_response)

        # Should collapse duplicate artifacts
        assert len(result.vertices) < len(vertices)


class TestPyvisVisualizer:
    """Test cases for PyvisVisualizer"""

    @patch("sagemaker.core.lineage.query.PyvisVisualizer._import_visual_modules")
    def test_visualizer_creation(self, mock_import):
        """Test visualizer creation"""
        mock_network = Mock()
        mock_options = Mock()
        mock_iframe = Mock()
        mock_bs = Mock()
        mock_import.return_value = (mock_network, mock_options, mock_iframe, mock_bs)

        graph_styles = {
            "Artifact": {
                "name": "Artifact",
                "style": {"background-color": "#146eb4"},
                "isShape": "False",
            }
        }

        visualizer = PyvisVisualizer(graph_styles)
        assert visualizer.graph_styles == graph_styles

    @patch("sagemaker.core.lineage.query.PyvisVisualizer._import_visual_modules")
    def test_visualizer_with_custom_options(self, mock_import):
        """Test visualizer with custom pyvis options"""
        mock_network = Mock()
        mock_options = Mock()
        mock_iframe = Mock()
        mock_bs = Mock()
        mock_import.return_value = (mock_network, mock_options, mock_iframe, mock_bs)

        graph_styles = {}
        custom_options = {"physics": {"enabled": True}}

        visualizer = PyvisVisualizer(graph_styles, custom_options)
        assert "physics" in visualizer._pyvis_options

    @patch("sagemaker.core.lineage.query.PyvisVisualizer._import_visual_modules")
    def test_node_color(self, mock_import):
        """Test getting node color"""
        mock_network = Mock()
        mock_options = Mock()
        mock_iframe = Mock()
        mock_bs = Mock()
        mock_import.return_value = (mock_network, mock_options, mock_iframe, mock_bs)

        graph_styles = {
            "Artifact": {
                "name": "Artifact",
                "style": {"background-color": "#146eb4"},
                "isShape": "False",
            }
        }

        visualizer = PyvisVisualizer(graph_styles)
        color = visualizer._node_color("Artifact")
        assert color == "#146eb4"
