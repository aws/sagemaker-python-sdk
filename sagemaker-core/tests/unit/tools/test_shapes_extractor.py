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
"""Unit tests for sagemaker.core.tools.shapes_extractor module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock

from sagemaker.core.tools.shapes_extractor import ShapesExtractor


class TestShapesExtractorInit:
    """Test ShapesExtractor initialization."""

    @patch("sagemaker.core.tools.shapes_extractor.load_combined_shapes_data")
    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_init_with_default_shapes(self, mock_reformat, mock_load):
        """Test initialization with default shapes."""
        mock_load.return_value = {
            "Shape1": {"type": "structure", "members": {}}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor()
        
        assert extractor.combined_shapes is not None
        assert extractor.shape_dag is not None

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_init_with_custom_shapes(self, mock_reformat):
        """Test initialization with custom shapes."""
        custom_shapes = {
            "CustomShape": {"type": "structure", "members": {}}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=custom_shapes)
        
        assert extractor.combined_shapes == custom_shapes


class TestGetShapesDag:
    """Test get_shapes_dag method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_get_shapes_dag_structure(self, mock_reformat):
        """Test DAG generation for structure type."""
        shapes = {
            "TestStruct": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                }
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        dag = extractor.get_shapes_dag()
        
        assert "TestStruct" in dag
        assert dag["TestStruct"]["type"] == "structure"
        assert len(dag["TestStruct"]["members"]) == 1

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_get_shapes_dag_list(self, mock_reformat):
        """Test DAG generation for list type."""
        shapes = {
            "StringList": {
                "type": "list",
                "member": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        dag = extractor.get_shapes_dag()
        
        assert "StringList" in dag
        assert dag["StringList"]["type"] == "list"
        assert dag["StringList"]["member_shape"] == "String"

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_get_shapes_dag_map(self, mock_reformat):
        """Test DAG generation for map type."""
        shapes = {
            "TagMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        dag = extractor.get_shapes_dag()
        
        assert "TagMap" in dag
        assert dag["TagMap"]["type"] == "map"
        assert dag["TagMap"]["key_shape"] == "String"
        assert dag["TagMap"]["value_shape"] == "String"


class TestEvaluateListType:
    """Test _evaluate_list_type method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_list_type_basic(self, mock_reformat):
        """Test evaluating basic list type."""
        shapes = {
            "StringList": {
                "type": "list",
                "member": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_list_type(shapes["StringList"])
        
        assert "List[StrPipeVar]" in result

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_list_type_nested(self, mock_reformat):
        """Test evaluating nested list type."""
        shapes = {
            "NestedList": {
                "type": "list",
                "member": {"shape": "StringList"}
            },
            "StringList": {
                "type": "list",
                "member": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_list_type(shapes["NestedList"])
        
        assert "List[List[" in result

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_list_type_structure(self, mock_reformat):
        """Test evaluating list of structures."""
        shapes = {
            "StructList": {
                "type": "list",
                "member": {"shape": "MyStruct"}
            },
            "MyStruct": {
                "type": "structure",
                "members": {}
            }
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_list_type(shapes["StructList"])
        
        assert "List[MyStruct]" in result


class TestEvaluateMapType:
    """Test _evaluate_map_type method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_map_type_basic(self, mock_reformat):
        """Test evaluating basic map type."""
        shapes = {
            "StringMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "String"}
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_map_type(shapes["StringMap"])
        
        assert "Dict[StrPipeVar, StrPipeVar]" in result

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_map_type_structure_value(self, mock_reformat):
        """Test evaluating map with structure value."""
        shapes = {
            "StructMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "MyStruct"}
            },
            "String": {"type": "string"},
            "MyStruct": {
                "type": "structure",
                "members": {}
            }
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_map_type(shapes["StructMap"])
        
        assert "Dict[StrPipeVar, MyStruct]" in result

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_evaluate_map_type_list_value(self, mock_reformat):
        """Test evaluating map with list value."""
        shapes = {
            "ListMap": {
                "type": "map",
                "key": {"shape": "String"},
                "value": {"shape": "StringList"}
            },
            "String": {"type": "string"},
            "StringList": {
                "type": "list",
                "member": {"shape": "String"}
            }
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor._evaluate_map_type(shapes["ListMap"])
        
        assert "Dict[StrPipeVar, List[" in result


class TestGenerateShapeMembers:
    """Test generate_shape_members method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_generate_shape_members_basic(self, mock_reformat):
        """Test generating shape members."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"},
                    "Field2": {"shape": "Integer"}
                },
                "required": ["Field1"]
            },
            "String": {"type": "string"},
            "Integer": {"type": "integer"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.generate_shape_members("TestShape")
        
        assert "field1" in result
        assert "field2" in result
        assert "Optional" in result["field2"]

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_generate_shape_members_with_override(self, mock_reformat):
        """Test generating shape members with required override."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"},
                    "Field2": {"shape": "String"}
                },
                "required": []
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.generate_shape_members("TestShape", required_override=("Field1",))
        
        assert "field1" in result
        assert "Optional" not in result["field1"]


class TestGenerateDataShapeStringBody:
    """Test generate_data_shape_string_body method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_generate_data_shape_string_body_basic(self, mock_reformat):
        """Test generating data shape string body."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                },
                "required": ["Field1"]
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.generate_data_shape_string_body("TestShape", None)
        
        assert "field1" in result
        assert "StrPipeVar" in result


class TestFetchShapeMembersAndDocStrings:
    """Test fetch_shape_members_and_doc_strings method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_fetch_shape_members_and_doc_strings(self, mock_reformat):
        """Test fetching shape members with documentation."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {
                        "shape": "String",
                        "documentation": "Field 1 documentation"
                    }
                },
                "required": ["Field1"]
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.fetch_shape_members_and_doc_strings("TestShape")
        
        assert "Field1" in result
        assert result["Field1"] == "Field 1 documentation"


class TestGetRequiredMembers:
    """Test get_required_members method."""

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_get_required_members_basic(self, mock_reformat):
        """Test getting required members."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"},
                    "Field2": {"shape": "String"}
                },
                "required": ["Field1"]
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.get_required_members("TestShape")
        
        assert "field1" in result
        assert "field2" not in result

    @patch("sagemaker.core.tools.shapes_extractor.reformat_file_with_black")
    def test_get_required_members_none(self, mock_reformat):
        """Test getting required members when none exist."""
        shapes = {
            "TestShape": {
                "type": "structure",
                "members": {
                    "Field1": {"shape": "String"}
                }
            },
            "String": {"type": "string"}
        }
        
        with patch("builtins.open", create=True):
            extractor = ShapesExtractor(combined_shapes=shapes)
        
        result = extractor.get_required_members("TestShape")
        
        assert len(result) == 0
