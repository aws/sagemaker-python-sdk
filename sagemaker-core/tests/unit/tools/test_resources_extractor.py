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
"""Unit tests for sagemaker.core.tools.resources_extractor module."""
from __future__ import absolute_import

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from sagemaker.core.tools.resources_extractor import ResourcesExtractor


class TestResourcesExtractorInit:
    """Test ResourcesExtractor initialization."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_init_basic(self, mock_additional, mock_ops, mock_shapes):
        """Test basic initialization."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {"CreateResource": {}}
        mock_additional.return_value = {}
        
        with patch.object(ResourcesExtractor, "_extract_resources_plan"):
            extractor = ResourcesExtractor()
        
        assert extractor.operations is not None
        assert extractor.shapes is not None

    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_init_with_custom_data(self, mock_additional):
        """Test initialization with custom data."""
        custom_shapes = {"Shape1": {}}
        custom_ops = {"Op1": {}}
        mock_additional.return_value = {}
        
        with patch.object(ResourcesExtractor, "_extract_resources_plan"):
            extractor = ResourcesExtractor(
                combined_shapes=custom_shapes,
                combined_operations=custom_ops
            )
        
        assert extractor.shapes == custom_shapes
        assert extractor.operations == custom_ops


class TestFilterAdditionalOperations:
    """Test _filter_additional_operations method."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_filter_additional_operations(self, mock_additional, mock_ops, mock_shapes):
        """Test filtering additional operations."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {"DescribeClusterNode": {}}
        mock_additional.return_value = {
            "Cluster": {
                "DescribeClusterNode": {
                    "method_name": "describe_node",
                    "return_type": "NodeInfo"
                }
            }
        }
        
        extractor = ResourcesExtractor()
        
        assert "Cluster" in extractor.resources
        assert "Cluster" in extractor.resource_methods


class TestFilterActionsForResources:
    """Test _filter_actions_for_resources method."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_filter_actions_for_resources_basic(self, mock_additional, mock_ops, mock_shapes):
        """Test filtering actions for resources."""
        mock_shapes.return_value = {
            "CreateModelInput": {"members": {}},
            "DescribeModelOutput": {"members": {}}
        }
        mock_ops.return_value = {
            "CreateModel": {"input": {"shape": "CreateModelInput"}},
            "DescribeModel": {"output": {"shape": "DescribeModelOutput"}},
            "DeleteModel": {},
            "ListModels": {}
        }
        mock_additional.return_value = {}
        
        extractor = ResourcesExtractor()
        
        assert "Model" in extractor.resource_actions
        assert len(extractor.resource_actions["Model"]) > 0


class TestExtractResourcesPlan:
    """Test _extract_resources_plan method."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_extract_resources_plan_creates_resources(self, mock_additional, mock_ops, mock_shapes):
        """Test that extract resources plan creates resources."""
        mock_shapes.return_value = {
            "CreateEndpointInput": {"members": {}},
            "DescribeEndpointOutput": {
                "members": {
                    "EndpointName": {"shape": "String"},
                    "EndpointStatus": {"shape": "EndpointStatus"}
                }
            },
            "EndpointStatus": {
                "type": "string",
                "enum": ["Creating", "InService", "Failed"]
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {
            "CreateEndpoint": {"input": {"shape": "CreateEndpointInput"}},
            "DescribeEndpoint": {
                "output": {"shape": "DescribeEndpointOutput"}
            }
        }
        mock_additional.return_value = {}
        
        extractor = ResourcesExtractor()
        
        assert "Endpoint" in extractor.resources


class TestGetStatusChainAndStates:
    """Test get_status_chain_and_states method."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_get_status_chain_and_states_basic(self, mock_additional, mock_ops, mock_shapes):
        """Test getting status chain and states."""
        mock_shapes.return_value = {
            "DescribeEndpointOutput": {
                "members": {
                    "EndpointName": {"shape": "String"},
                    "EndpointStatus": {"shape": "EndpointStatus"}
                }
            },
            "EndpointStatus": {
                "type": "string",
                "enum": ["Creating", "InService", "Failed"]
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {
            "DescribeEndpoint": {
                "output": {"shape": "DescribeEndpointOutput"}
            }
        }
        mock_additional.return_value = {}
        
        extractor = ResourcesExtractor()
        status_chain, states = extractor.get_status_chain_and_states("Endpoint")
        
        assert len(status_chain) > 0
        assert len(states) > 0

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_get_status_chain_and_states_nested(self, mock_additional, mock_ops, mock_shapes):
        """Test getting nested status chain."""
        mock_ops.return_value = {
            "DescribeResource": {
                "output": {"shape": "DescribeResourceOutput"}
            }
        }
        mock_shapes.return_value = {
            "DescribeResourceOutput": {
                "members": {
                    "Resource": {"shape": "ResourceInfo"}
                }
            },
            "ResourceInfo": {
                "members": {
                    "Status": {"shape": "ResourceStatus"}
                }
            },
            "ResourceStatus": {
                "type": "string",
                "enum": ["Active", "Inactive"]
            }
        }
        mock_additional.return_value = {}
        
        extractor = ResourcesExtractor()
        status_chain, states = extractor.get_status_chain_and_states("Resource")
        
        assert len(status_chain) > 0


class TestGetResourceMethods:
    """Test get_resource_methods method."""

    @patch("sagemaker.core.tools.resources_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.resources_extractor.load_combined_operations_data")
    @patch("sagemaker.core.tools.resources_extractor.load_additional_operations_data")
    def test_get_resource_methods(self, mock_additional, mock_ops, mock_shapes):
        """Test getting resource methods."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {"DescribeClusterNode": {}}
        mock_additional.return_value = {
            "Cluster": {
                "DescribeClusterNode": {
                    "method_name": "describe_node",
                    "return_type": "NodeInfo"
                }
            }
        }
        
        extractor = ResourcesExtractor()
        result = extractor.get_resource_methods()
        
        assert isinstance(result, dict)
        assert "Cluster" in result
