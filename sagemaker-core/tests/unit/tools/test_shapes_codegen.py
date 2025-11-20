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
"""Unit tests for sagemaker.core.tools.shapes_codegen module."""
from __future__ import absolute_import

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from sagemaker.core.tools.shapes_codegen import ShapesCodeGen


class TestShapesCodeGenInit:
    """Test ShapesCodeGen initialization."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_init_basic(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test basic initialization."""
        mock_shapes.return_value = {"Shape1": {"type": "structure", "members": {}}}
        mock_ops.return_value = {"Operation1": {}}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()

        assert codegen.combined_shapes is not None
        assert codegen.combined_operations is not None
        assert codegen.shapes_extractor is not None


class TestBuildGraph:
    """Test build_graph method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_build_graph_simple(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test building graph with simple shapes."""
        mock_shapes.return_value = {
            "Shape1": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                }
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        graph = codegen.build_graph()

        assert "Shape1" in graph
        assert "String" in graph

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_build_graph_with_list(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test building graph with list type."""
        mock_shapes.return_value = {
            "Shape1": {
                "type": "structure",
                "members": {
                    "Items": {"shape": "ItemList"}
                }
            },
            "ItemList": {
                "type": "list",
                "member": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        graph = codegen.build_graph()

        assert "Shape1" in graph
        assert "String" in graph["Shape1"]

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_build_graph_with_map(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test building graph with map type."""
        mock_shapes.return_value = {
            "Shape1": {
                "type": "structure",
                "members": {
                    "Tags": {"shape": "TagMap"}
                }
            },
            "TagMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        graph = codegen.build_graph()

        assert "Shape1" in graph


class TestTopologicalSort:
    """Test topological_sort method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_topological_sort_basic(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test topological sort with simple dependency."""
        mock_shapes.return_value = {
            "Shape1": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                }
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen.topological_sort()

        assert isinstance(result, list)
        assert len(result) > 0


class TestGenerateDataClassForShape:
    """Test generate_data_class_for_shape method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_data_class_basic(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test generating data class for basic shape."""
        mock_shapes.return_value = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                },
                "documentation": "Test shape documentation"
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_extractor_instance.generate_data_shape_string_body.return_value = "field1: str"
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen.generate_data_class_for_shape("TestShape")

        assert "TestShape" in result
        assert "class" in result


class TestGenerateDocString:
    """Test _generate_doc_string_for_shape method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_doc_string_basic(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test generating docstring for shape."""
        mock_shapes.return_value = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {
                        "shape": "String",
                        "documentation": "Field documentation"
                    }
                },
                "documentation": "Shape documentation"
            },
            "String": {"type": "string"}
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen._generate_doc_string_for_shape("TestShape")

        assert "TestShape" in result
        assert "Attributes" in result
        assert "field1" in result


class TestGenerateImports:
    """Test generate_imports method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_imports(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test generating imports."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen.generate_imports()

        assert "import datetime" in result
        assert "from pydantic import BaseModel" in result
        assert "from typing import" in result


class TestGenerateLicense:
    """Test generate_license method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_license(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test generating license."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen.generate_license()

        assert "Copyright" in result
        assert "Amazon" in result


class TestGenerateBaseClass:
    """Test generate_base_class method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_base_class(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test generating base class."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen.generate_base_class()

        assert "class Base" in result
        assert "BaseModel" in result


class TestFilterInputOutputShapes:
    """Test _filter_input_output_shapes method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_filter_input_output_shapes_input_shape(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test filtering input shapes."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {
            "CreateResource": {
                "input": {"shape": "CreateResourceRequest"},
                "output": {"shape": "CreateResourceResponse"}
            }
        }
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen._filter_input_output_shapes("CreateResourceRequest")

        assert result is False

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_filter_input_output_shapes_other_shape(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test filtering other shapes."""
        mock_shapes.return_value = {}
        mock_ops.return_value = {
            "CreateResource": {
                "input": {"shape": "CreateResourceRequest"}
            }
        }
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        result = codegen._filter_input_output_shapes("SomeOtherShape")

        assert result is True


class TestGenerateShapes:
    """Test generate_shapes method."""

    @patch("sagemaker.core.tools.shapes_codegen.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_codegen.load_combined_operations_data")
    @patch("sagemaker.core.tools.shapes_codegen.ShapesExtractor")
    @patch("sagemaker.core.tools.shapes_codegen.ResourcesExtractor")
    def test_generate_shapes_creates_file(self, mock_resources_extractor, mock_shapes_extractor, mock_ops, mock_shapes):
        """Test that generate_shapes creates output file."""
        mock_shapes.return_value = {
            "TestShape": {
                "type": "structure",
                "members": {},
                "documentation": "Test"
            }
        }
        mock_ops.return_value = {}
        mock_extractor_instance = Mock()
        mock_extractor_instance.get_shapes_dag.return_value = {}
        mock_extractor_instance.generate_data_shape_string_body.return_value = "pass"
        mock_shapes_extractor.return_value = mock_extractor_instance
        
        mock_resources_instance = Mock()
        mock_resources_instance.get_resource_plan.return_value = Mock()
        mock_resources_instance.get_resource_methods.return_value = {}
        mock_resources_extractor.return_value = mock_resources_instance

        codegen = ShapesCodeGen()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "test_shapes.py")
            codegen.generate_shapes(output_folder=tmpdir, file_name="test_shapes.py")
            
            assert os.path.exists(output_file)
            
            with open(output_file, "r") as f:
                content = f.read()
                assert "Copyright" in content
                assert "import" in content
