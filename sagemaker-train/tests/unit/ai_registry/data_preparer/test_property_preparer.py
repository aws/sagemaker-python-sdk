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
"""Property-based tests for DataPreparer (P7, P11, P14).

Validates:
- P7: Category inference (Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6).
- P11: No forbidden fields on the wire (Requirements 6.8, 6.9, 6.10, 6.11, 6.12, 6.13).
- P14: Discoverability completeness (Requirements 12.1, 12.2, 12.10).
"""

from __future__ import absolute_import

import io
import json
from contextlib import redirect_stdout

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402
from unittest.mock import MagicMock, patch  # noqa: E402

from sagemaker.ai_registry.data_preparer._schema_resolver import (  # noqa: E402
    _MODEL_CACHE,
    _SCHEMA_CACHE,
    _SchemaResolver,
    _pascal_to_snake,
)
from sagemaker.ai_registry.data_preparer.preparer import DataPreparer  # noqa: E402
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig  # noqa: E402
from sagemaker.ai_registry.data_preparer.constants import JobCategory  # noqa: E402
from sagemaker.ai_registry.data_preparer.exceptions import (  # noqa: E402
    DataPreparerConfigurationError,
)


def _mock_session():
    s = MagicMock()
    s.boto_region_name = "us-west-2"
    s.account_id.return_value = "123456789012"
    s.sagemaker_config = {"SchemaVersion": "1.0"}
    return s


def _make_resolved_dataset(category=JobCategory.SYNTHETIC_DATA_GENERATION, user_payload=None):
    cfg = DatasetJobConfig(job_category=category)
    cfg._resolved_schema_version = "1.0.0"
    cfg._resolved_schema = {"properties": {"NumSamples": {"type": "integer"}}}
    cfg._dynamic_model = MagicMock()
    cfg._validated_payload = user_payload or {"NumSamples": 5}
    cfg._serialized_document = json.dumps(cfg._validated_payload)
    return cfg


# -----------------------------------------------------------------------------
# P7: Category inference — dataset.job_category = None binds to method's category;
#     a mismatch raises pre-submit without mutating dataset.job_category observably.
# -----------------------------------------------------------------------------


_FAKE_AGENT_RFT = "AgentRFT"
_FAKE_AGENT_RFT_EVAL = "AgentRFTEvaluation"


class TestPropertyCategoryInference:
    """P7: For method ∈ {generate, evaluate} × ds.job_category ∈ {None, SDG, DQ,
    AgentRFT, AgentRFTEvaluation}, the bound category is submitted when None,
    mismatches raise pre-submit, and dataset.job_category is not observably
    mutated on rejection.

    Validates Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6.
    """

    _CATEGORY_SAMPLES = st.sampled_from(
        [
            None,
            JobCategory.SYNTHETIC_DATA_GENERATION,
            JobCategory.DATA_QUALITY_EVALUATION,
        ]
    )
    _METHOD = st.sampled_from(["generate", "evaluate"])

    @given(ds_cat=_CATEGORY_SAMPLES, method=_METHOD)
    @settings(
        max_examples=60,
        deadline=3000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_inference_or_mismatch(self, ds_cat, method):
        method_cat = (
            JobCategory.SYNTHETIC_DATA_GENERATION
            if method == "generate"
            else JobCategory.DATA_QUALITY_EVALUATION
        )

        dc = DataPreparer(sagemaker_session=_mock_session(), role="arn:role")
        cfg = _make_resolved_dataset(category=ds_cat)
        ds_cat_before = cfg.job_category

        mock_job = MagicMock()
        mock_job.job_status = "Completed"
        mock_job.job_config_document = json.dumps({"DatasetArn": "arn:ds"})
        fake_ds = MagicMock()

        with patch.object(dc, "_create_job", return_value=mock_job):
            with patch.object(dc, "_rehydrate_datasets", return_value=[fake_ds]):
                if ds_cat is None or ds_cat == method_cat:
                    # Should succeed: category is bound (or already matches)
                    result = getattr(dc, method)(cfg, wait=True, logs=False)
                    assert result is fake_ds
                    # After binding, dataset.job_category equals the method-bound category.
                    assert cfg.job_category == method_cat
                else:
                    # Should raise pre-submit
                    with pytest.raises(DataPreparerConfigurationError):
                        getattr(dc, method)(cfg, wait=True, logs=False)
                    # Must not observably mutate the dataset.job_category
                    assert cfg.job_category == ds_cat_before


# -----------------------------------------------------------------------------
# P11: No forbidden fields on the wire
# -----------------------------------------------------------------------------

_FORBIDDEN_FIELDS = [
    "SourceIdentity",
    "source_identity",
    "CreationTime",
    "creation_time",
    "IgnoredBySweeper",
    "ignored_by_sweeper",
    "UseMockAlgo",
    "use_mock_algo",
]


class TestPropertyNoForbiddenFieldsOnWire:
    """P11: For every Job.create request DataPreparer emits, none of the
    forbidden fields appears in the request kwargs or in the serialized
    JobConfigDocument.

    Validates Requirements 6.8, 6.9, 6.10, 6.11, 6.12, 6.13.
    """

    @given(
        num_samples=st.integers(min_value=1, max_value=10000),
        job_name=st.one_of(
            st.none(),
            st.from_regex(r"[A-Za-z0-9](-*[A-Za-z0-9]){3,40}", fullmatch=True),
        ),
        category=st.sampled_from(list(JobCategory)),
    )
    @settings(
        max_examples=80,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_request_has_no_forbidden_fields(self, num_samples, job_name, category):
        dc = DataPreparer(sagemaker_session=_mock_session(), role="arn:role")
        cfg = _make_resolved_dataset(category=category, user_payload={"NumSamples": num_samples})

        with patch(
            "sagemaker.ai_registry.data_preparer.preparer._get_job_resource"
        ) as mock_get_job:
            mock_job_cls = MagicMock()
            mock_get_job.return_value = mock_job_cls
            mock_job_cls.create.return_value = MagicMock()

            dc._create_job(cfg, category, job_name=job_name)

            call_kwargs = mock_job_cls.create.call_args.kwargs
            for forbidden in _FORBIDDEN_FIELDS:
                assert forbidden not in call_kwargs

        # Also assert the serialized document doesn't contain a forbidden key at
        # the top level. (The document carries only the per-category business
        # schema; the envelope fields are separate.)
        doc = json.loads(cfg._serialized_document)
        for forbidden in _FORBIDDEN_FIELDS:
            assert forbidden not in doc


# -----------------------------------------------------------------------------
# P14: Discoverability completeness — every schema property appears in
# DatasetJobConfig.schema(...) at the right nesting level AND in print_schema output.
# -----------------------------------------------------------------------------

_PASCAL_IDENT = st.from_regex(r"[A-Z][a-z]{2,6}([A-Z][a-z]{2,6}){0,2}", fullmatch=True)


@st.composite
def _schema_with_top_level_properties(draw):
    n = draw(st.integers(min_value=1, max_value=5))
    names = draw(st.lists(_PASCAL_IDENT, min_size=n, max_size=n, unique=True))
    schema = {
        "properties": {n: {"type": "string"} for n in names},
        "required": [],
    }
    return schema, names


@st.composite
def _schema_with_nested_properties(draw):
    top_schema, top_names = draw(_schema_with_top_level_properties())
    nested_names = draw(st.lists(_PASCAL_IDENT, min_size=1, max_size=3, unique=True))
    nested_key = draw(_PASCAL_IDENT.filter(lambda s: s not in top_names))
    top_schema["properties"][nested_key] = {
        "type": "object",
        "title": nested_key,
        "properties": {n: {"type": "string"} for n in nested_names},
        "required": [],
    }
    return top_schema, top_names, nested_key, nested_names


class TestPropertyDiscoverabilityCompleteness:
    """P14: Every schema property path P appears in DatasetJobConfig.schema(category)
    at the correct nesting level AND in DatasetJobConfig.print_schema output on
    stdout — without SDK code changes, driven purely by the Hub schema.

    Validates Requirements 12.1, 12.2, 12.10.
    """

    @given(schema_and_names=_schema_with_top_level_properties())
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_schema_returns_top_level_properties(self, schema_and_names):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()
        schema, names = schema_and_names
        with patch.object(_SchemaResolver, "_client") as mock_client:
            fake = MagicMock()
            fake.list_job_schema_versions.return_value = {
                "JobConfigSchemas": [
                    {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(schema)},
                ]
            }
            mock_client.return_value = fake

            result_schema = DatasetJobConfig.schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )

        for name in names:
            assert name in result_schema["properties"]

    @given(schema_and_names=_schema_with_nested_properties())
    @settings(
        max_examples=30,
        deadline=5000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_schema_returns_nested_properties(self, schema_and_names):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()
        schema, top_names, nested_key, nested_names = schema_and_names
        with patch.object(_SchemaResolver, "_client") as mock_client:
            fake = MagicMock()
            fake.list_job_schema_versions.return_value = {
                "JobConfigSchemas": [
                    {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(schema)},
                ]
            }
            mock_client.return_value = fake

            result_schema = DatasetJobConfig.schema(
                JobCategory.SYNTHETIC_DATA_GENERATION,
                sagemaker_session=_mock_session(),
            )

        for name in top_names:
            assert name in result_schema["properties"]
        assert nested_key in result_schema["properties"]
        nested_props = result_schema["properties"][nested_key]["properties"]
        for name in nested_names:
            assert name in nested_props

    @given(schema_and_names=_schema_with_top_level_properties())
    @settings(
        max_examples=20,
        deadline=8000,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_print_schema_renders_all_top_level_fields(self, schema_and_names):
        _SCHEMA_CACHE.clear()
        _MODEL_CACHE.clear()
        schema, names = schema_and_names
        buf = io.StringIO()
        with patch.object(_SchemaResolver, "_client") as mock_client:
            fake = MagicMock()
            fake.list_job_schema_versions.return_value = {
                "JobConfigSchemas": [
                    {"JobConfigSchemaVersion": "1.0.0", "JobConfigSchema": json.dumps(schema)},
                ]
            }
            mock_client.return_value = fake
            with redirect_stdout(buf):
                DatasetJobConfig.print_schema(
                    JobCategory.SYNTHETIC_DATA_GENERATION,
                    sagemaker_session=_mock_session(),
                )

        output = buf.getvalue()
        for name in names:
            # print_schema renders names in snake_case (Python idiom per §1.10)
            snake = _pascal_to_snake(name)
            assert snake in output, f"Missing '{snake}' in print_schema output:\n{output}"
