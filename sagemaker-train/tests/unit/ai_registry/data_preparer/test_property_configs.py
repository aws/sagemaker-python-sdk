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
"""Property-based tests for DatasetJobConfig (P6, P13).

Validates:
- P6: Cross-category schema isolation (Requirement 3.13).
- P13: Dict input == typed-nested-model input (Requirements 13.1, 13.2, 13.3, 13.4).
"""

from __future__ import absolute_import

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from sagemaker.ai_registry.data_preparer._schema_resolver import (  # noqa: E402
    _MODEL_CACHE,
    _SCHEMA_CACHE,
    _SchemaResolver,
)
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig  # noqa: E402
from sagemaker.ai_registry.data_preparer.constants import JobCategory  # noqa: E402
from sagemaker.ai_registry.data_preparer.exceptions import (  # noqa: E402
    DataPreparerSchemaValidationError,
)

# -----------------------------------------------------------------------------
# Helpers: schema pair generators that guarantee a non-equivalent required-field
# set across the two categories.
# -----------------------------------------------------------------------------

_PASCAL_IDENT = st.from_regex(r"[A-Z][a-z]{2,6}([A-Z][a-z]{2,6}){0,2}", fullmatch=True)


@st.composite
def _disjoint_required_schema_pair(draw):
    """Generate two schemas whose `required` field sets do not overlap.

    A payload valid under schema_c1 will therefore NOT be valid under schema_c2
    (missing required fields after renaming).
    """
    a_names = draw(st.lists(_PASCAL_IDENT, min_size=1, max_size=3, unique=True))
    b_names = draw(
        st.lists(_PASCAL_IDENT, min_size=1, max_size=3, unique=True).filter(
            lambda ns: not any(n in a_names for n in ns)
        )
    )
    schema_a = {
        "properties": {n: {"type": "string"} for n in a_names},
        "required": a_names,
    }
    schema_b = {
        "properties": {n: {"type": "string"} for n in b_names},
        "required": b_names,
    }
    return schema_a, schema_b, a_names, b_names


# -----------------------------------------------------------------------------
# P6: Schema isolation across categories
# -----------------------------------------------------------------------------


class TestPropertySchemaIsolation:
    """P6: A payload valid under one category's schema does not silently validate
    under another category's schema when required fields differ.

    Validates Requirement 3.13.
    """

    @given(pair=_disjoint_required_schema_pair())
    @settings(
        max_examples=40,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_valid_under_a_fails_under_b(self, pair):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()
        schema_a, schema_b, a_names, _ = pair
        # Payload valid under schema_a (all required present)
        payload_kwargs = {n: "v" for n in a_names}

        mock_session = MagicMock()

        # First, validate under schema_a — must succeed
        with (
            patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver") as mock_resolver,
            patch(
                "sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults"
            ) as mock_defaults,
        ):
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.resolve_into_payload.side_effect = lambda user_payload, *a, **k: dict(
                user_payload
            )

            # Build real dynamic models so validation actually runs
            model_a = _SchemaResolver.build_dynamic_model(
                JobCategory.SYNTHETIC_DATA_GENERATION, "a_" + "_".join(a_names), schema_a
            )
            mock_resolver.get_schema.return_value = ("1.0.0", schema_a)
            mock_resolver.build_dynamic_model.return_value = model_a

            cfg_a = DatasetJobConfig(
                job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
                **payload_kwargs,
            )
            cfg_a._resolve_and_validate(sagemaker_session=mock_session)
            assert cfg_a._serialized_document is not None

        # Now validate the same payload under schema_b — must fail
        with (
            patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver") as mock_resolver,
            patch(
                "sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults"
            ) as mock_defaults,
        ):
            mock_defaults.get_sagemaker_session.return_value = mock_session
            mock_defaults.resolve_into_payload.side_effect = lambda user_payload, *a, **k: dict(
                user_payload
            )

            model_b = _SchemaResolver.build_dynamic_model(
                JobCategory.DATA_QUALITY_EVALUATION, "b_" + "_".join(a_names), schema_b
            )
            mock_resolver.get_schema.return_value = ("1.0.0", schema_b)
            mock_resolver.build_dynamic_model.return_value = model_b

            cfg_b = DatasetJobConfig(
                job_category=JobCategory.DATA_QUALITY_EVALUATION,
                **payload_kwargs,
            )
            with pytest.raises(DataPreparerSchemaValidationError):
                cfg_b._resolve_and_validate(sagemaker_session=mock_session)


# -----------------------------------------------------------------------------
# P13: Dict ≡ typed equivalence on input
# -----------------------------------------------------------------------------


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


def _make_nested_model():
    _MODEL_CACHE.clear()
    return _SchemaResolver.build_dynamic_model(
        JobCategory.SYNTHETIC_DATA_GENERATION, "p13_test", NESTED_SCHEMA
    )


class TestPropertyDictTypedEquivalence:
    """P13: Constructing a DatasetJobConfig with a nested value as a dict vs as
    a typed-model instance produces identical serialization. Both snake_case and
    PascalCase input shapes round-trip to PascalCase on the wire.

    Validates Requirements 13.1, 13.2, 13.3, 13.4.

    Note: these tests intentionally disable the HubName auto-injection branch
    in DatasetJobConfig._resolve_and_validate by using
    JobCategory.DATA_QUALITY_EVALUATION (the HubName branch is SDG-only per
    current configs.py). This keeps the dict-vs-typed comparison symmetric.
    """

    @given(
        output_uri=st.text(alphabet="abcdefghij0123456789/", min_size=5, max_size=30).map(
            lambda s: f"s3://bucket/{s}"
        ),
        num_samples=st.integers(min_value=1, max_value=10000),
    )
    @settings(
        max_examples=50,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_dict_and_typed_nested_produce_same_wire(self, output_uri, num_samples):
        _SCHEMA_CACHE.clear()
        model = _make_nested_model()
        nested_cls = _extract_nested_model(model, "output_data_config")

        mock_session = MagicMock()
        kwargs_common = dict(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            num_samples=num_samples,
        )

        # Dict form (PascalCase keys)
        with _patched_resolver(model, NESTED_SCHEMA, mock_session):
            cfg_dict = DatasetJobConfig(
                output_data_config={"OutputS3Uri": output_uri},
                **kwargs_common,
            )
            cfg_dict._resolve_and_validate(sagemaker_session=mock_session)
            dict_payload = cfg_dict._validated_payload

        # Typed-model form
        with _patched_resolver(model, NESTED_SCHEMA, mock_session):
            nested_instance = nested_cls(OutputS3Uri=output_uri)
            cfg_typed = DatasetJobConfig(
                output_data_config=nested_instance,
                **kwargs_common,
            )
            cfg_typed._resolve_and_validate(sagemaker_session=mock_session)
            typed_payload = cfg_typed._validated_payload

        assert dict_payload == typed_payload

    @given(
        output_uri=st.text(alphabet="abcdefghij0123456789/", min_size=5, max_size=30).map(
            lambda s: f"s3://bucket/{s}"
        ),
        num_samples=st.integers(min_value=1, max_value=10000),
    )
    @settings(
        max_examples=50,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_snake_case_and_pascal_case_equivalent(self, output_uri, num_samples):
        _SCHEMA_CACHE.clear()
        model = _make_nested_model()
        mock_session = MagicMock()
        common = dict(
            job_category=JobCategory.DATA_QUALITY_EVALUATION,
            num_samples=num_samples,
        )

        with _patched_resolver(model, NESTED_SCHEMA, mock_session):
            cfg_snake = DatasetJobConfig(
                output_data_config={"output_s3_uri": output_uri},
                **common,
            )
            cfg_snake._resolve_and_validate(sagemaker_session=mock_session)

        with _patched_resolver(model, NESTED_SCHEMA, mock_session):
            cfg_pascal = DatasetJobConfig(
                output_data_config={"OutputS3Uri": output_uri},
                **common,
            )
            cfg_pascal._resolve_and_validate(sagemaker_session=mock_session)

        # Both forms must produce the same serialized wire document
        assert cfg_snake._serialized_document == cfg_pascal._serialized_document
        # And both must emit PascalCase on the wire
        assert "OutputS3Uri" in cfg_snake._serialized_document
        assert "output_s3_uri" not in cfg_snake._serialized_document


def _extract_nested_model(root_model, field_snake):
    field = root_model.model_fields[field_snake]
    ann = field.annotation
    args = getattr(ann, "__args__", ())
    if type(None) in args:
        return next(a for a in args if a is not type(None))  # noqa: E721
    return ann


class _patched_resolver:
    """Context manager that mocks _SchemaResolver and DataPreparerDefaults in configs."""

    def __init__(self, model, schema, session):
        self.model = model
        self.schema = schema
        self.session = session

    def __enter__(self):
        self._p_resolver = patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
        self._p_defaults = patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
        mock_resolver = self._p_resolver.__enter__()
        mock_defaults = self._p_defaults.__enter__()
        mock_resolver.get_schema.return_value = ("1.0.0", self.schema)
        mock_resolver.build_dynamic_model.return_value = self.model
        mock_defaults.get_sagemaker_session.return_value = self.session
        # Pass-through: the intelligent-defaults walk is orthogonal to the
        # dict-vs-typed equivalence property being tested here.
        mock_defaults.resolve_into_payload.side_effect = lambda user_payload, *a, **k: dict(
            user_payload
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        self._p_defaults.__exit__(exc_type, exc, tb)
        self._p_resolver.__exit__(exc_type, exc, tb)
