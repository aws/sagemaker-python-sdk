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
"""Property-based tests for _SchemaResolver (P2, P3, P12).

Requires hypothesis. Skipped if not installed.
"""
from __future__ import absolute_import

import json

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings, assume, HealthCheck  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from sagemaker.ai_registry.data_preparer._schema_resolver import (  # noqa: E402
    _MODEL_CACHE,
    _SCHEMA_CACHE,
    _SchemaResolver,
    _major,
    _semver_key,
)
from sagemaker.ai_registry.data_preparer.constants import JobCategory, SCHEMA_VERSION_CEILING  # noqa: E402


# -----------------------------------------------------------------------------
# Hypothesis strategies
# -----------------------------------------------------------------------------

# Keys used for schema-property names — PascalCase-ish identifiers that survive
# snake_case aliasing. We intentionally avoid underscores/digits at boundaries
# to keep alias round-tripping deterministic.
_PASCAL_IDENT = st.from_regex(r"[A-Z][a-z]{2,6}([A-Z][a-z]{2,6}){0,2}", fullmatch=True)


def _primitive_schema():
    return st.sampled_from(
        [
            {"type": "string"},
            {"type": "integer"},
            {"type": "number"},
            {"type": "boolean"},
        ]
    )


def _primitive_value_for(schema):
    t = schema.get("type")
    if t == "string":
        return st.text(alphabet="abcdefghij", min_size=0, max_size=8)
    if t == "integer":
        return st.integers(min_value=-1000, max_value=1000)
    if t == "number":
        return st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False)
    if t == "boolean":
        return st.booleans()
    return st.just(None)


@st.composite
def _flat_object_schema(draw, min_props=1, max_props=4):
    """Generate a flat (non-nested) object schema with primitive properties."""
    n = draw(st.integers(min_value=min_props, max_value=max_props))
    names = draw(
        st.lists(_PASCAL_IDENT, min_size=n, max_size=n, unique=True)
    )
    props = {name: draw(_primitive_schema()) for name in names}
    required_count = draw(st.integers(min_value=0, max_value=len(names)))
    required = names[:required_count]
    return {"properties": props, "required": required}


@st.composite
def _nested_object_schema(draw):
    """Generate a schema with at least one nested object and maybe an array."""
    inner = draw(_flat_object_schema(min_props=1, max_props=3))
    top = draw(_flat_object_schema(min_props=1, max_props=3))
    nested_key = draw(_PASCAL_IDENT)
    assume(nested_key not in top["properties"])
    top["properties"][nested_key] = {
        "type": "object",
        "title": nested_key,
        "properties": inner["properties"],
        "required": inner["required"],
    }
    # Sometimes also add an array-of-objects field
    if draw(st.booleans()):
        array_key = draw(_PASCAL_IDENT)
        assume(array_key not in top["properties"])
        top["properties"][array_key] = {
            "type": "array",
            "items": {
                "type": "object",
                "title": array_key + "Item",
                "properties": {
                    "Id": {"type": "string"},
                    "Value": {"type": "integer"},
                },
                "required": ["Id"],
            },
        }
    return top


@st.composite
def _payload_for(draw, schema):
    """Generate a valid payload (PascalCase keys) for a flat schema."""
    payload = {}
    for name, subschema in schema["properties"].items():
        if subschema.get("type") == "object":
            inner = draw(_payload_for(subschema))
            payload[name] = inner
        elif subschema.get("type") == "array":
            items_schema = subschema.get("items", {})
            if items_schema.get("type") == "object":
                count = draw(st.integers(min_value=0, max_value=2))
                payload[name] = [draw(_payload_for(items_schema)) for _ in range(count)]
            else:
                count = draw(st.integers(min_value=0, max_value=3))
                payload[name] = [draw(_primitive_value_for(items_schema)) for _ in range(count)]
        else:
            # Always generate primitives (even if optional, simpler & non-None round-trip)
            payload[name] = draw(_primitive_value_for(subschema))
    # Remove optional keys sometimes
    required = set(schema.get("required", []))
    for name in list(payload.keys()):
        if name not in required and draw(st.booleans()):
            del payload[name]
    return payload


# -----------------------------------------------------------------------------
# P3: Default schema version selection
# -----------------------------------------------------------------------------

# Strategy: arbitrary semver strings including non-lexicographic pairs
_SEMVER_PART = st.integers(min_value=0, max_value=50)
_SEMVER_STR = st.builds(
    lambda a, b, c: f"{a}.{b}.{c}", _SEMVER_PART, _SEMVER_PART, _SEMVER_PART
)


class TestPropertyDefaultSchemaVersion:
    """P3: When version=None, get_schema picks max(versions) under semver ordering.

    Validates Requirements 3.1, 3.11.
    """

    @given(versions=st.lists(_SEMVER_STR, min_size=1, max_size=15, unique=True))
    @settings(max_examples=100, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_max_semver_chosen(self, versions):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()

        ceiling = SCHEMA_VERSION_CEILING.get(JobCategory.SYNTHETIC_DATA_GENERATION)
        within = [v for v in versions if ceiling is None or _major(v) <= ceiling]
        expected = max(within if within else versions, key=_semver_key)

        with patch.object(_SchemaResolver, "_client") as mock_client:
            fake = MagicMock()
            fake.list_job_schema_versions.return_value = {
                "JobConfigSchemas": [
                    {
                        "JobConfigSchemaVersion": v,
                        "JobConfigSchema": json.dumps({"properties": {}, "required": []}),
                    }
                    for v in versions
                ]
            }
            mock_client.return_value = fake

            resolved_version, _ = _SchemaResolver.get_schema(
                JobCategory.SYNTHETIC_DATA_GENERATION, None
            )

        assert resolved_version == expected

    def test_non_lexicographic_case(self):
        """Regression case cited in tasks.md: 1.10.0 > 1.9.0."""
        versions = ["1.9.0", "1.10.0", "1.2.0"]
        assert max(versions, key=_semver_key) == "1.10.0"


# -----------------------------------------------------------------------------
# P2: Schema round-trip
# -----------------------------------------------------------------------------


class TestPropertySchemaRoundTrip:
    """P2: Build dynamic model from a schema, instantiate with a valid payload,
    dump as JSON (by_alias=True), re-parse under the same model — the re-parsed
    payload must equal the original modulo ordering and optional-None fields.

    Validates Requirements 3.7, 3.12.
    """

    @given(schema=_flat_object_schema(min_props=1, max_props=4))
    @settings(max_examples=50, deadline=5000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_flat_round_trip(self, schema):
        _MODEL_CACHE.clear()
        # Build a payload deterministically: all required keys with a fixed value per type
        payload = {}
        for name, subschema in schema["properties"].items():
            t = subschema["type"]
            payload[name] = {"string": "x", "integer": 42, "number": 1.5, "boolean": True}[t]

        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "test", schema
        )
        instance = model(**payload)
        # Dump with PascalCase keys (wire format)
        dumped = instance.model_dump(by_alias=True, exclude_none=True)
        # Keys that were set should round-trip verbatim
        for k, v in payload.items():
            assert dumped[k] == v

        # Re-parsing the JSON must validate and produce equivalent payload
        as_json = instance.model_dump_json(by_alias=True, exclude_none=True)
        reparsed = model.model_validate_json(as_json)
        reparsed_dump = reparsed.model_dump(by_alias=True, exclude_none=True)
        assert reparsed_dump == dumped

    @given(schema=_nested_object_schema())
    @settings(max_examples=30, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nested_round_trip(self, schema):
        _MODEL_CACHE.clear()
        # Build a deterministic payload: required-only, primitive defaults by type
        def _make(schm):
            out = {}
            for name, subschema in schm["properties"].items():
                t = subschema.get("type")
                if t == "object":
                    out[name] = _make(subschema)
                elif t == "array":
                    items = subschema.get("items", {})
                    if items.get("type") == "object":
                        out[name] = [_make(items)]
                    else:
                        out[name] = []
                else:
                    out[name] = {
                        "string": "x",
                        "integer": 42,
                        "number": 1.5,
                        "boolean": True,
                    }[t]
            return out

        payload = _make(schema)
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "nested_test", schema
        )
        instance = model(**payload)
        dumped_json = instance.model_dump_json(by_alias=True, exclude_none=True)
        reparsed = model.model_validate_json(dumped_json)
        # Round-trip must produce the identical serialization
        assert reparsed.model_dump_json(by_alias=True, exclude_none=True) == dumped_json


# -----------------------------------------------------------------------------
# P12: Nested-model structure mirrors the schema tree
# -----------------------------------------------------------------------------


class TestPropertyNestedModelStructure:
    """P12: For nested objects/arrays/refs, the dynamic model mirrors the schema:
    - object property → nested BaseModel subclass
    - array → List[item_model] (recursively)
    - $ref → memoized nested model (identity at every reference site)

    Validates Requirements 3.7, 3.8.
    """

    @given(schema=_nested_object_schema())
    @settings(max_examples=30, deadline=8000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_nested_objects_become_basemodel_subclasses(self, schema):
        _MODEL_CACHE.clear()
        from pydantic import BaseModel

        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "p12_test", schema
        )

        for name, subschema in schema["properties"].items():
            from sagemaker.ai_registry.data_preparer._schema_resolver import _pascal_to_snake

            snake = _pascal_to_snake(name)
            field = model.model_fields[snake]
            ann = field.annotation
            # Unwrap Optional[...]
            origin = getattr(ann, "__origin__", None)
            args = getattr(ann, "__args__", ())
            if origin is not None and type(None) in args:
                ann = next(a for a in args if a is not type(None))  # noqa: E721

            if subschema.get("type") == "object":
                assert isinstance(ann, type) and issubclass(ann, BaseModel), (
                    f"Field {snake} should be a BaseModel subclass, got {ann}"
                )
            elif subschema.get("type") == "array":
                origin2 = getattr(ann, "__origin__", None)
                assert origin2 is list, f"Field {snake} should be List[...], got {ann}"
                inner_items = subschema.get("items", {})
                if inner_items.get("type") == "object":
                    item_ann = ann.__args__[0]
                    assert isinstance(item_ann, type) and issubclass(item_ann, BaseModel)

    def test_ref_identity(self):
        """Every $ref to the same $def resolves to the same Python class."""
        _MODEL_CACHE.clear()
        schema = {
            "properties": {
                "A": {"$ref": "#/$defs/Inner"},
                "B": {"$ref": "#/$defs/Inner"},
                "C": {
                    "type": "object",
                    "title": "Outer",
                    "properties": {"Nested": {"$ref": "#/$defs/Inner"}},
                    "required": [],
                },
            },
            "required": [],
            "$defs": {
                "Inner": {
                    "properties": {
                        "X": {"type": "string"},
                    },
                    "required": [],
                }
            },
        }
        model = _SchemaResolver.build_dynamic_model(
            JobCategory.SYNTHETIC_DATA_GENERATION, "ref_identity_test", schema
        )

        def _unwrap(ann):
            origin = getattr(ann, "__origin__", None)
            args = getattr(ann, "__args__", ())
            if origin is not None and type(None) in args:
                return next(a for a in args if a is not type(None))  # noqa: E721
            return ann

        type_a = _unwrap(model.model_fields["a"].annotation)
        type_b = _unwrap(model.model_fields["b"].annotation)
        outer_type = _unwrap(model.model_fields["c"].annotation)
        type_nested = _unwrap(outer_type.model_fields["nested"].annotation)
        assert type_a is type_b
        assert type_a is type_nested
