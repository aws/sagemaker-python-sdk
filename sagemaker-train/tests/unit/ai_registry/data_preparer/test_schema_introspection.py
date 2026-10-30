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
"""Tests for DatasetJobConfig.schema / print_schema and the schema renderer."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer._schema_renderer import render_schema_tree
from sagemaker.ai_registry.data_preparer._schema_resolver import (
    _MODEL_CACHE,
    _SCHEMA_CACHE,
    _SchemaResolver,
)
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer


def _mock_session():
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    return s


# A comprehensive synthetic schema exercising all features
SYNTHETIC_SCHEMA = {
    "properties": {
        "DatasetName": {
            "type": "string",
            "description": "Name of the dataset.",
            "minLength": 1,
            "maxLength": 63,
            "pattern": "^[a-zA-Z][a-zA-Z0-9-]*$",
        },
        "NumSamples": {
            "type": "integer",
            "description": "Number of samples to generate.",
            "default": 100,
            "minimum": 1,
            "maximum": 10000,
        },
        "Mode": {
            "type": "string",
            "enum": ["sync", "async"],
            "default": "sync",
            "description": "Run mode for the job.",
        },
        "OutputConfig": {
            "type": "object",
            "description": "Output configuration.",
            "properties": {
                "OutputS3Uri": {
                    "type": "string",
                    "description": "S3 URI for output.",
                    "format": "uri",
                    "examples": ["s3://my-bucket/out/"],
                },
                "Compression": {
                    "type": "string",
                    "enum": ["gzip", "none"],
                    "default": "none",
                },
            },
            "required": ["OutputS3Uri"],
        },
        "Tags": {
            "type": "array",
            "description": "Tags for the job.",
            "items": {
                "type": "object",
                "properties": {
                    "Key": {"type": "string"},
                    "Value": {"type": "string"},
                },
                "required": ["Key", "Value"],
            },
            "minItems": 0,
            "maxItems": 50,
        },
        "Metadata": {
            "type": "object",
            "description": "Arbitrary metadata.",
            "additionalProperties": True,
        },
        "TypedMetadata": {
            "type": "object",
            "description": "Typed metadata dict.",
            "additionalProperties": {"type": "integer"},
        },
        "Strategy": {
            "description": "Generation strategy.",
            "oneOf": [
                {"type": "string", "description": "Simple strategy name."},
                {
                    "type": "object",
                    "description": "Complex strategy config.",
                    "properties": {
                        "Name": {"type": "string"},
                        "Params": {"type": "object", "additionalProperties": True},
                    },
                    "required": ["Name"],
                },
            ],
        },
        "OptionalField": {
            "description": "An optional union field.",
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ],
        },
        "MultiUnion": {
            "description": "Multi-branch union.",
            "anyOf": [
                {"type": "string"},
                {"type": "integer"},
                {"type": "boolean"},
            ],
        },
        "ListOfStrings": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of strings.",
        },
    },
    "required": ["DatasetName", "OutputConfig"],
}


def _setup_mock(schema):
    """Helper to set up mock client for schema resolution."""
    _SCHEMA_CACHE.clear()
    _MODEL_CACHE.clear()
    mock_client = MagicMock()
    mock_client.list_job_schema_versions.return_value = {
        "JobConfigSchemas": [
            {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(schema)},
        ]
    }
    return mock_client


class TestSchemaMethodMove:
    """Verify schema/print_schema moved from DataPreparer to DatasetJobConfig."""

    def test_data_preparer_no_longer_has_schema(self):
        # Pydantic BaseModel has a .schema() classmethod (deprecated v2 compat),
        # so we check that DataPreparer's schema is NOT our custom one (no category param).
        import inspect

        if hasattr(DataPreparer, "schema"):
            sig = inspect.signature(DataPreparer.schema)
            assert "category" not in sig.parameters

    def test_data_preparer_no_longer_has_print_schema(self):
        assert not hasattr(DataPreparer, "print_schema")

    def test_dataset_job_config_has_schema(self):
        assert hasattr(DatasetJobConfig, "schema")
        assert callable(DatasetJobConfig.schema)

    def test_dataset_job_config_has_print_schema(self):
        assert hasattr(DatasetJobConfig, "print_schema")
        assert callable(DatasetJobConfig.print_schema)


class TestSchemaIsPure:
    """schema() returns dict and does NOT print."""

    def test_schema_returns_dict(self):
        mock_client = _setup_mock(SYNTHETIC_SCHEMA)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert isinstance(result, dict)
        assert "properties" in result

    def test_schema_does_not_print(self, capsys):
        mock_client = _setup_mock(SYNTHETIC_SCHEMA)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            DatasetJobConfig.schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        captured = capsys.readouterr()
        assert captured.out == ""


class TestPrintSchemaReturnsNone:
    """print_schema() prints to stdout and returns None."""

    def test_print_schema_returns_none(self):
        mock_client = _setup_mock(SYNTHETIC_SCHEMA)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.print_schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result is None

    def test_print_schema_writes_to_stdout(self, capsys):
        mock_client = _setup_mock(SYNTHETIC_SCHEMA)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            DatasetJobConfig.print_schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        captured = capsys.readouterr()
        assert len(captured.out) > 0
        assert "dataset_name" in captured.out


class TestComprehensiveRendering:
    """Verify every schema feature is surfaced in render output."""

    @pytest.fixture(autouse=True)
    def rendered(self):
        self.output = render_schema_tree(SYNTHETIC_SCHEMA, title="TestSchema")

    def test_title_shown(self):
        assert "TestSchema" in self.output

    def test_required_field_marked(self):
        assert "dataset_name" in self.output
        assert "[required]" in self.output

    def test_optional_field_marked(self):
        # NumSamples is not required
        assert "num_samples" in self.output
        assert "(optional)" in self.output

    def test_description_shown(self):
        assert "Name of the dataset." in self.output
        assert "Number of samples to generate." in self.output

    def test_default_value_shown(self):
        assert "= 100" in self.output
        assert "= 'sync'" in self.output

    def test_enum_values_shown(self):
        assert "'sync'" in self.output
        assert "'async'" in self.output

    def test_constraints_min_max(self):
        assert "minimum=1" in self.output
        assert "maximum=10000" in self.output

    def test_constraints_string(self):
        assert "minLength=1" in self.output
        assert "maxLength=63" in self.output
        assert "pattern=" in self.output

    def test_format_constraint(self):
        assert "format=uri" in self.output

    def test_array_constraints(self):
        assert "minItems=0" in self.output
        assert "maxItems=50" in self.output

    def test_examples_shown(self):
        assert "s3://my-bucket/out/" in self.output

    def test_nested_object_recursed(self):
        assert "output_s3_uri" in self.output
        assert "compression" in self.output

    def test_array_items_recursed(self):
        # Tags array has object items with Key/Value
        assert "key" in self.output.lower()
        assert "value" in self.output.lower()

    def test_additional_properties_true(self):
        assert "additional properties" in self.output

    def test_additional_properties_typed(self):
        # TypedMetadata has additionalProperties: {type: integer}
        assert "int" in self.output

    def test_one_of_branches_shown(self):
        assert "oneOf" in self.output
        assert "branch" in self.output

    def test_any_of_optional_rendered(self):
        # OptionalField is anyOf [string, null] -> Optional[str]
        assert "optional_field" in self.output

    def test_multi_union_branches(self):
        # MultiUnion is anyOf [string, integer, boolean]
        assert "multi_union" in self.output
        # Should show the union type
        assert "str" in self.output
        assert "int" in self.output
        assert "bool" in self.output

    def test_list_type_shown(self):
        assert "List[" in self.output

    def test_dict_type_shown(self):
        assert "Dict[str," in self.output

    def test_alias_shown(self):
        # PascalCase alias should be shown for snake_case fields
        assert "DatasetName" in self.output
        assert "NumSamples" in self.output

    def test_nested_required_propagated(self):
        # OutputS3Uri is required within OutputConfig
        # Check that the output contains both the field and required marker near it
        lines = self.output.split("\n")
        for line in lines:
            if "output_s3_uri" in line:
                assert "[required]" in line
                break
        else:
            pytest.fail("output_s3_uri not found in output")


class TestRefResolution:
    """Test $ref resolution in rendering."""

    def test_ref_resolved_and_rendered(self):
        schema = {
            "properties": {
                "Config": {"$ref": "#/$defs/MyConfig"},
            },
            "required": ["Config"],
            "$defs": {
                "MyConfig": {
                    "type": "object",
                    "properties": {
                        "Enabled": {
                            "type": "boolean",
                            "description": "Whether enabled.",
                            "default": True,
                        },
                    },
                    "required": ["Enabled"],
                },
            },
        }
        output = render_schema_tree(schema)
        assert "config" in output
        assert "enabled" in output
        assert "Whether enabled." in output
        assert "= True" in output


class TestEdgeCases:
    """Edge cases for the renderer."""

    def test_empty_schema(self):
        output = render_schema_tree({"properties": {}, "required": []})
        assert "DatasetJobConfig" in output

    def test_type_array_with_null(self):
        """JSON schema type: ["string", "null"] should render as Optional[str]."""
        schema = {
            "properties": {
                "MaybeStr": {"type": ["string", "null"], "description": "Nullable string."},
            },
            "required": [],
        }
        output = render_schema_tree(schema)
        assert "Optional[str]" in output

    def test_const_value(self):
        schema = {
            "properties": {
                "Version": {"const": "1.0", "description": "Fixed version."},
            },
            "required": ["Version"],
        }
        output = render_schema_tree(schema)
        assert "'1.0'" in output


class TestDeepNesting:
    """Verify 3+ level nesting renders without truncation."""

    def test_three_level_nesting(self):
        schema = {
            "properties": {
                "Level1": {
                    "type": "object",
                    "properties": {
                        "Level2": {
                            "type": "object",
                            "properties": {
                                "Level3": {
                                    "type": "object",
                                    "properties": {
                                        "DeepField": {
                                            "type": "string",
                                            "description": "Deeply nested.",
                                        }
                                    },
                                    "required": ["DeepField"],
                                }
                            },
                            "required": ["Level3"],
                        }
                    },
                    "required": ["Level2"],
                }
            },
            "required": ["Level1"],
        }
        output = render_schema_tree(schema)
        assert "level1" in output
        assert "level2" in output
        assert "level3" in output
        assert "deep_field" in output
        assert "Deeply nested." in output

    def test_four_level_nesting_via_refs(self):
        schema = {
            "properties": {
                "Root": {"$ref": "#/$defs/A"},
            },
            "required": ["Root"],
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {"Child": {"$ref": "#/$defs/B"}},
                    "required": ["Child"],
                },
                "B": {
                    "type": "object",
                    "properties": {"Grandchild": {"$ref": "#/$defs/C"}},
                    "required": ["Grandchild"],
                },
                "C": {
                    "type": "object",
                    "properties": {"Leaf": {"type": "integer", "description": "4 levels deep."}},
                    "required": ["Leaf"],
                },
            },
        }
        output = render_schema_tree(schema)
        assert "root" in output
        assert "child" in output
        assert "grandchild" in output
        assert "leaf" in output
        assert "4 levels deep." in output


class TestCircularRefProtection:
    """Verify circular $ref does not cause infinite recursion (M1 fix)."""

    def test_circular_ref_terminates(self):
        schema = {
            "properties": {
                "Node": {"$ref": "#/$defs/TreeNode"},
            },
            "required": ["Node"],
            "$defs": {
                "TreeNode": {
                    "type": "object",
                    "properties": {
                        "Value": {"type": "string"},
                        "Children": {
                            "type": "array",
                            "items": {"$ref": "#/$defs/TreeNode"},
                        },
                    },
                    "required": ["Value"],
                },
            },
        }
        # Should not raise RecursionError
        output = render_schema_tree(schema)
        assert "node" in output
        assert "value" in output
        assert "<circular ref>" in output

    def test_direct_self_ref(self):
        schema = {
            "properties": {
                "Self": {"$ref": "#/$defs/Recursive"},
            },
            "required": ["Self"],
            "$defs": {
                "Recursive": {
                    "type": "object",
                    "properties": {
                        "Name": {"type": "string"},
                        "Next": {"$ref": "#/$defs/Recursive"},
                    },
                    "required": ["Name"],
                },
            },
        }
        output = render_schema_tree(schema)
        assert "name" in output
        assert "<circular ref>" in output


class TestRequiredFields:
    """Tests for DatasetJobConfig.required_fields classmethod."""

    @pytest.fixture(autouse=True)
    def _clear_caches(self):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()

    def test_happy_path_sdg(self):
        schema = {
            "properties": {
                "DatasetName": {"type": "string"},
                "GenerationConfig": {"type": "object"},
            },
            "required": ["DatasetName", "GenerationConfig"],
        }
        mock_client = _setup_mock(schema)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result == ["dataset_name", "generation_config"]

    def test_empty_required(self):
        schema = {"properties": {"Foo": {"type": "string"}}, "required": []}
        mock_client = _setup_mock(schema)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result == []

    def test_no_required_key_in_schema(self):
        schema = {"properties": {"Foo": {"type": "string"}}}
        mock_client = _setup_mock(schema)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result == []

    def test_version_pinning_forwarded(self):
        schema = {"properties": {"X": {"type": "string"}}, "required": ["X"]}
        mock_client = MagicMock()
        mock_client.describe_job_schema_version.return_value = {
            "JobConfigSchemaVersion": "1.2.3",
            "JobConfigSchema": json.dumps(schema),
        }
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                schema_version="1.2.3",
                sagemaker_session=_mock_session(),
            )
        mock_client.describe_job_schema_version.assert_called_once()
        call_kwargs = mock_client.describe_job_schema_version.call_args[1]
        assert call_kwargs["JobConfigSchemaVersion"] == "1.2.3"

    def test_schema_not_found_propagates(self):
        from sagemaker.ai_registry.data_preparer.exceptions import (
            DataPreparerSchemaNotFoundError,
        )

        with patch.object(
            _SchemaResolver,
            "get_schema",
            side_effect=DataPreparerSchemaNotFoundError(
                JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", []
            ),
        ):
            with pytest.raises(DataPreparerSchemaNotFoundError):
                DatasetJobConfig.required_fields(
                    JobCategory.SYNTHETIC_DATA_GENERATION,
                    sagemaker_session=_mock_session(),
                )

    def test_snake_case_conversion_complex_names(self):
        schema = {
            "properties": {
                "DatasetName": {"type": "string"},
                "HubName": {"type": "string"},
                "S3Uri": {"type": "string"},
            },
            "required": ["DatasetName", "HubName", "S3Uri"],
        }
        mock_client = _setup_mock(schema)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result == ["dataset_name", "hub_name", "s3_uri"]

    def test_ordering_preserved_non_alphabetical(self):
        """Ordering must match schema declaration, not alphabetical sort."""
        schema = {
            "properties": {
                "Zebra": {"type": "string"},
                "Alpha": {"type": "string"},
                "Middle": {"type": "string"},
            },
            "required": ["Zebra", "Alpha", "Middle"],
        }
        mock_client = _setup_mock(schema)
        with patch.object(_SchemaResolver, "_client", return_value=mock_client):
            result = DatasetJobConfig.required_fields(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )
        assert result == ["zebra", "alpha", "middle"]
