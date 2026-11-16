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
"""Unit tests for sagemaker.ai_registry.data_preparer._schema_resolver."""

from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from sagemaker.ai_registry.data_preparer._schema_resolver import (
    _MODEL_CACHE,
    _SCHEMA_CACHE,
    _SchemaResolver,
    _semver_key,
)
from sagemaker.ai_registry.data_preparer.constants import JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerSchemaNotFoundError,
)


@pytest.fixture(autouse=True)
def clear_caches():
    """Clear module-level caches before each test."""
    _SCHEMA_CACHE.clear()
    _MODEL_CACHE.clear()
    yield
    _SCHEMA_CACHE.clear()
    _MODEL_CACHE.clear()


SIMPLE_SCHEMA = {
    "properties": {
        "NumSamples": {"type": "integer"},
        "SpecS3Uri": {"type": "string"},
    },
    "required": ["NumSamples"],
}

NESTED_SCHEMA = {
    "properties": {
        "OutputDataConfig": {
            "type": "object",
            "title": "OutputDataConfig",
            "properties": {
                "OutputS3Uri": {"type": "string"},
                "HubName": {"type": "string"},
            },
            "required": ["OutputS3Uri"],
        },
        "NumSamples": {"type": "integer"},
    },
    "required": ["NumSamples"],
}

ARRAY_SCHEMA = {
    "properties": {
        "Tags": {
            "type": "array",
            "items": {"type": "string"},
        },
        "ModelConfigs": {
            "type": "array",
            "items": {
                "type": "object",
                "title": "ModelConfig",
                "properties": {
                    "ModelId": {"type": "string"},
                    "TokensPerMinute": {"type": "integer"},
                },
                "required": ["ModelId"],
            },
        },
    },
    "required": [],
}

REF_SCHEMA = {
    "properties": {
        "SampleConfig": {"$ref": "#/$defs/LlmModelConfig"},
        "VerifyConfig": {"$ref": "#/$defs/LlmModelConfig"},
    },
    "required": [],
    "$defs": {
        "LlmModelConfig": {
            "properties": {
                "ModelId": {"type": "string"},
                "TargetTokensPerMinute": {"type": "integer"},
            },
            "required": ["ModelId"],
        }
    },
}


class TestSemverKey:
    def test_basic_ordering(self):
        versions = ["1.0.0", "1.1.0", "2.0.0", "0.9.0"]
        assert max(versions, key=_semver_key) == "2.0.0"

    def test_non_lexicographic(self):
        versions = ["1.9.0", "1.10.0", "1.2.0"]
        assert max(versions, key=_semver_key) == "1.10.0"

    def test_single_version(self):
        assert max(["1.0.0"], key=_semver_key) == "1.0.0"

    def test_prerelease_sorts_lower_than_release(self):
        versions = ["1.0.0-rc1", "1.0.0"]
        assert max(versions, key=_semver_key) == "1.0.0"

    def test_prerelease_sorts_lower_than_next_release(self):
        versions = ["1.5.0-beta1", "1.5.0"]
        assert max(versions, key=_semver_key) == "1.5.0"

    def test_two_part_version(self):
        versions = ["1.0", "0.9"]
        assert max(versions, key=_semver_key) == "1.0"


class TestListVersions:
    @patch.object(_SchemaResolver, "_client")
    def test_returns_version_strings(self, mock_client):
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": "{}"},
                {"JobConfigSchemaVersion": "1.1.0", "JobConfigSchema": "{}"},
            ]
        }
        mock_client.return_value = fake
        versions = _SchemaResolver.list_versions(JobCategory.SYNTHETIC_DATA_GENERATION)
        assert versions == ["1.0.0", "1.1.0"]
        fake.list_job_schema_versions.assert_called_once_with(JobCategory="SyntheticDataGeneration")


class TestGetSchema:
    @patch.object(_SchemaResolver, "_client")
    def test_latest_version_selection(self, mock_client):
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "1.10.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "1.9.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
            ]
        }
        mock_client.return_value = fake

        version, schema = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert version == "1.10.0"
        assert schema == SIMPLE_SCHEMA

    @patch.object(_SchemaResolver, "_client")
    def test_explicit_version(self, mock_client):
        fake = MagicMock()
        fake.describe_job_schema_version.return_value = {
            "JobCategory": "SyntheticDataGeneration",
            "JobConfigSchemaVersion": "2.0.0",
            "JobConfigSchema": json.dumps(SIMPLE_SCHEMA),
        }
        mock_client.return_value = fake

        version, schema = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, "2.0.0")
        assert version == "2.0.0"
        assert schema == SIMPLE_SCHEMA
        fake.describe_job_schema_version.assert_called_once_with(
            JobCategory="SyntheticDataGeneration",
            JobConfigSchemaVersion="2.0.0",
        )

    @patch.object(_SchemaResolver, "_client")
    def test_cache_hit_avoids_second_call(self, mock_client):
        fake = MagicMock()
        fake.describe_job_schema_version.return_value = {
            "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)
        }
        mock_client.return_value = fake

        _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0")
        _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0")

        assert fake.describe_job_schema_version.call_count == 1

    @patch.object(_SchemaResolver, "_client")
    def test_empty_versions_raises_not_found(self, mock_client):
        """Pre-GA behavior: hardcoded fallback is gone. Empty list must raise."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {"JobConfigSchemas": []}
        mock_client.return_value = fake

        with pytest.raises(DataPreparerSchemaNotFoundError) as exc_info:
            _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert exc_info.value.schema_version == "latest"
        assert exc_info.value.available_versions == []

    @patch.object(_SchemaResolver, "_client")
    def test_resource_not_found_raises(self, mock_client):
        fake = MagicMock()

        class ResourceNotFoundException(Exception):
            pass

        fake.describe_job_schema_version.side_effect = ResourceNotFoundException("not found")
        # After the describe fails, list_versions is called to populate available_versions
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": "{}"},
            ]
        }
        mock_client.return_value = fake

        with pytest.raises(DataPreparerSchemaNotFoundError) as exc_info:
            _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, "99.0.0")

        assert exc_info.value.schema_version == "99.0.0"
        assert exc_info.value.available_versions == ["1.0.0"]

    @patch.object(_SchemaResolver, "_client")
    def test_latest_falls_back_to_describe_when_schema_not_inline(self, mock_client):
        """Service may return versions without schema body in list response."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "1.0.0"},  # no JobConfigSchema
            ]
        }
        fake.describe_job_schema_version.return_value = {
            "JobCategory": "SyntheticDataGeneration",
            "JobConfigSchemaVersion": "1.0.0",
            "JobConfigSchema": json.dumps(SIMPLE_SCHEMA),
        }
        mock_client.return_value = fake

        version, schema = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert version == "1.0.0"
        assert schema == SIMPLE_SCHEMA
        fake.describe_job_schema_version.assert_called_once_with(
            JobCategory="SyntheticDataGeneration",
            JobConfigSchemaVersion="1.0.0",
        )


class TestBuildDynamicModel:
    def test_simple_schema_produces_model(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        assert issubclass(model, BaseModel)
        # Required field
        instance = model(num_samples=10)
        assert instance.num_samples == 10
        assert instance.spec_s3_uri is None

    def test_pascal_case_alias_input(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        instance = model(NumSamples=42, SpecS3Uri="s3://bucket/key")
        assert instance.num_samples == 42
        assert instance.spec_s3_uri == "s3://bucket/key"

    def test_serialization_uses_pascal_case(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        instance = model(num_samples=5)
        dumped = instance.model_dump(by_alias=True, exclude_none=True)
        assert "NumSamples" in dumped
        assert "num_samples" not in dumped

    def test_nested_object_produces_nested_model(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "2.0.0", NESTED_SCHEMA
        )
        instance = model(
            num_samples=10,
            output_data_config={"OutputS3Uri": "s3://bucket/out"},
        )
        assert instance.output_data_config.output_s3_uri == "s3://bucket/out"

    def test_array_of_primitives(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "3.0.0", ARRAY_SCHEMA
        )
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]

    def test_array_of_objects(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "3.0.0", ARRAY_SCHEMA
        )
        instance = model(model_configs=[{"ModelId": "claude-v3", "TokensPerMinute": 1000}])
        assert instance.model_configs[0].model_id == "claude-v3"

    def test_ref_produces_same_class(self):
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "4.0.0", REF_SCHEMA
        )
        instance = model(
            sample_config={"ModelId": "m1"},
            verify_config={"ModelId": "m2"},
        )
        # Both fields should use the same class
        assert type(instance.sample_config) is type(instance.verify_config)

    def test_model_cache_hit(self):
        model1 = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        model2 = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        assert model1 is model2

    def test_different_categories_different_models(self):
        model1 = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "1.0.0", SIMPLE_SCHEMA
        )
        model2 = _SchemaResolver.build_dynamic_model(
            JobCategory.DATA_QUALITY_EVALUATION, "1.0.0", SIMPLE_SCHEMA
        )
        assert model1 is not model2

    def test_oneof_maps_to_any(self):
        schema_with_union = {
            "properties": {
                "InputSource": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "object", "properties": {"Uri": {"type": "string"}}},
                    ]
                },
            },
            "required": [],
        }
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "5.0.0", schema_with_union
        )
        # Should accept both string and dict
        instance = model(input_source="s3://bucket")
        assert instance.input_source == "s3://bucket"
        instance2 = model(input_source={"Uri": "s3://bucket"})
        assert instance2.input_source == {"Uri": "s3://bucket"}

    def test_untyped_property_accepts_any_value(self):
        # DataQualityEvaluation 3.0.0 leaves Config.View untyped so the service, not
        # the schema, explains a malformed View. Modeled as str, the SDK rejected
        # every list View before the request was sent.
        schema = {
            "properties": {
                "Operations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "title": "Operation",
                        "properties": {
                            "Operator": {"type": "string"},
                            "Config": {
                                "type": "object",
                                "title": "OperationConfig",
                                "additionalProperties": True,
                                "properties": {
                                    "View": {
                                        "items": {
                                            "properties": {
                                                "Role": {"type": "string"},
                                                "Column": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                        },
                        "required": ["Operator"],
                    },
                },
            },
            "required": ["Operations"],
        }
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.DATA_QUALITY_EVALUATION, "3.0.0", schema
        )
        view = [{"Role": "Text", "Column": "prompt"}]
        instance = model(
            operations=[
                {"Operator": "fuzzy_dedup", "Config": {"View": view, "JaccardThreshold": 0.85}}
            ]
        )
        dumped = instance.model_dump(by_alias=True, exclude_none=True)
        assert dumped["Operations"][0]["Config"] == {"View": view, "JaccardThreshold": 0.85}


# Schema with optional field that has a declared default (like HubName: "AIRegistry")
SCHEMA_WITH_OPTIONAL_DEFAULT = {
    "properties": {
        "NumSamples": {"type": "integer"},
        "HubName": {"type": "string", "default": "AIRegistry"},
        "OutputS3Uri": {"type": "string"},
    },
    "required": ["NumSamples"],
}

# Schema with required field that has a declared default
SCHEMA_WITH_REQUIRED_DEFAULT = {
    "properties": {
        "NumSamples": {"type": "integer", "default": 100},
        "SpecS3Uri": {"type": "string"},
    },
    "required": ["NumSamples"],
}


class TestOptionalFieldDefaults:
    """Optional fields should NOT use schema defaults — service resolves them."""

    def test_optional_field_with_schema_default_is_none_when_unset(self):
        """HubName has default='AIRegistry' in schema but should be None if not provided."""
        _MODEL_CACHE.clear()
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "opt_default_1", SCHEMA_WITH_OPTIONAL_DEFAULT
        )
        instance = model(NumSamples=10)
        assert instance.hub_name is None

    def test_optional_field_excluded_from_serialization_when_unset(self):
        """Unset optional field with schema default should not appear in payload."""
        _MODEL_CACHE.clear()
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "opt_default_2", SCHEMA_WITH_OPTIONAL_DEFAULT
        )
        instance = model(NumSamples=10)
        dumped = instance.model_dump(by_alias=True, exclude_none=True)
        assert "HubName" not in dumped
        assert "NumSamples" in dumped

    def test_optional_field_included_when_explicitly_set(self):
        """If user explicitly sets HubName, it should appear in payload."""
        _MODEL_CACHE.clear()
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "opt_default_3", SCHEMA_WITH_OPTIONAL_DEFAULT
        )
        instance = model(NumSamples=10, HubName="my-custom-hub")
        dumped = instance.model_dump(by_alias=True, exclude_none=True)
        assert dumped["HubName"] == "my-custom-hub"

    def test_required_field_with_schema_default_still_uses_default(self):
        """Required fields should still use schema defaults (needed for valid payloads)."""
        _MODEL_CACHE.clear()
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "req_default_1", SCHEMA_WITH_REQUIRED_DEFAULT
        )
        # Can instantiate without providing NumSamples — default kicks in
        instance = model()
        assert instance.num_samples == 100


class TestMajorVersionCeiling:
    """Tests for the major-version ceiling applied in the latest-version path.

    SyntheticDataGeneration's ceiling is 2, so the version above it here is 3.x.
    """

    @patch.object(_SchemaResolver, "_client")
    def test_ceiling_filters_out_above_ceiling(self, mock_client):
        """v3.0.0 should be excluded when the ceiling is 2."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "2.5.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "3.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
            ]
        }
        mock_client.return_value = fake

        version, _ = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert version == "2.5.0"

    @patch.object(_SchemaResolver, "_client")
    def test_ceiling_picks_highest_within_major(self, mock_client):
        """When multiple v2.x versions exist, the highest is selected."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "2.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "2.9.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "2.2.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "3.1.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
            ]
        }
        mock_client.return_value = fake

        version, _ = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert version == "2.9.0"

    @patch.object(_SchemaResolver, "_client")
    def test_sdg_picks_v2_over_v1(self, mock_client):
        """An unpinned SDG request resolves to v2, which is where auto-labeling lives."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "2.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
            ]
        }
        mock_client.return_value = fake

        version, _ = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)
        assert version == "2.0.0"

    @patch.object(_SchemaResolver, "_client")
    def test_ceiling_fallback_when_no_versions_within_ceiling(self, mock_client, caplog):
        """When all available versions exceed the ceiling, fall back to latest with warning."""
        import logging

        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": "3.0.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
                {"JobConfigSchemaVersion": "3.1.0", "JobConfigSchema": json.dumps(SIMPLE_SCHEMA)},
            ]
        }
        mock_client.return_value = fake

        with caplog.at_level(logging.WARNING):
            version, _ = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)

        assert version == "3.1.0"
        assert "Falling back to latest available" in caplog.text

    @patch.object(_SchemaResolver, "_client")
    def test_explicit_version_bypasses_ceiling(self, mock_client):
        """Pinned schema_version= skips ceiling entirely."""
        fake = MagicMock()
        fake.describe_job_schema_version.return_value = {
            "JobConfigSchema": json.dumps(SIMPLE_SCHEMA),
        }
        mock_client.return_value = fake

        version, schema = _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, "2.0.0")
        assert version == "2.0.0"
        assert schema == SIMPLE_SCHEMA
        fake.describe_job_schema_version.assert_called_once_with(
            JobCategory="SyntheticDataGeneration",
            JobConfigSchemaVersion="2.0.0",
        )


# Exact set of DataQualityEvaluation schema versions the SDK must handle
# gracefully when it appears in ``ListJobSchemaVersions`` responses. The
# service renamed the preview "0.2.0" v2 schema to "2.0.0" and now publishes
# both "2.0.0" and "1.0.0"; "2.0.0" is the intended default.
_WIRE_DQ_VERSIONS = ["2.0.0", "1.0.0"]


class TestDqWireContract:
    """Pins how the SDK resolves DQ ``latest`` when v2.0.0 is present alongside
    v1.0.0. If the service adds or removes DQ schema versions, update
    ``_WIRE_DQ_VERSIONS`` (and any similar checks maintained outside this
    repo) at the same time so mismatches surface as a test failure.

    Regression guard for F2: before the DataQualityEvaluation ceiling was
    bumped to 2, the resolver filtered "2.0.0" out (major 2 > ceiling 1) and
    silently pinned "1.0.0", so every completed v2 job was read with the v1
    shape and failed. These tests would have caught that — the old fixtures
    pinned the obsolete "0.2.0" (major 0, under the ceiling) and never
    exercised the real wire versions."""

    @patch.object(_SchemaResolver, "_client")
    def test_sdk_picks_v2_as_latest_for_dqe(self, mock_client):
        """When both v2.0.0 and v1.0.0 are advertised, the SDK's latest-version
        resolver must pick 2.0.0 — the DataQualityEvaluation ceiling is 2, so
        v2 is the default a naive ``pick-latest-ListJobSchemaVersions`` caller
        gets. This is the F2 fix: a Completed v2 job resolves to the v2 shape
        and returns its scorecard instead of throwing."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {
            "JobConfigSchemas": [
                {"JobConfigSchemaVersion": v, "JobConfigSchema": "{}"} for v in _WIRE_DQ_VERSIONS
            ],
        }
        mock_client.return_value = fake

        version, _schema = _SchemaResolver.get_schema(JobCategory.DATA_QUALITY_EVALUATION, None)

        assert version == "2.0.0", (
            f"Expected the SDK to pick '2.0.0' as latest given advertised versions "
            f"{_WIRE_DQ_VERSIONS}, but it picked {version!r}. The DataQualityEvaluation "
            f"ceiling must be 2 so v2 is the default; if it regressed to 1 the resolver "
            f"falls back to the v1 shape and every completed v2 job fails (F2)."
        )

    @patch.object(_SchemaResolver, "_client")
    def test_sdk_still_reaches_v1_when_pinned_explicitly(self, mock_client):
        """Callers that opt in by pinning '1.0.0' must still get the v1 schema —
        the ceiling gates only the 'latest' path, not explicit pins."""
        fake = MagicMock()
        fake.describe_job_schema_version.return_value = {
            "JobCategory": "DataQualityEvaluation",
            "JobConfigSchemaVersion": "1.0.0",
            "JobConfigSchema": "{}",
        }
        mock_client.return_value = fake

        version, _schema = _SchemaResolver.get_schema(JobCategory.DATA_QUALITY_EVALUATION, "1.0.0")

        assert version == "1.0.0"
        fake.describe_job_schema_version.assert_called_once_with(
            JobCategory="DataQualityEvaluation",
            JobConfigSchemaVersion="1.0.0",
        )


class TestGetSchemaTransientA2:
    """A2 (P0): a raising list_job_schema_versions call must produce transient=True.

    Network / auth / service failures hit the 'except Exception as e' block in the
    latest-version path. The resulting DataPreparerSchemaNotFoundError must carry
    transient=True so callers can distinguish "network hiccup → retry" from
    "service has no schema yet → wait for a publish".
    """

    @patch.object(_SchemaResolver, "_client")
    def test_list_raises_produces_transient_error(self, mock_client):
        """FAILS on original code (no transient flag); passes after A2 fix."""
        fake = MagicMock()
        fake.list_job_schema_versions.side_effect = ConnectionError("network blip")
        mock_client.return_value = fake

        with pytest.raises(DataPreparerSchemaNotFoundError) as exc_info:
            _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)

        assert exc_info.value.transient is True, (
            "A raising list_job_schema_versions must produce transient=True "
            "so callers know it's a network/auth failure, not 'no schema published'"
        )
        assert exc_info.value.__cause__ is not None

    @patch.object(_SchemaResolver, "_client")
    def test_empty_list_response_is_not_transient(self, mock_client):
        """Successful call returning empty list is NOT transient — the service is up."""
        fake = MagicMock()
        fake.list_job_schema_versions.return_value = {"JobConfigSchemas": []}
        mock_client.return_value = fake

        with pytest.raises(DataPreparerSchemaNotFoundError) as exc_info:
            _SchemaResolver.get_schema(JobCategory.SYNTHETIC_DATA_GENERATION, None)

        assert exc_info.value.transient is False
