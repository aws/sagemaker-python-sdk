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
"""Unit tests for _case_conversion helpers and DatasetJobConfig integration."""
from __future__ import absolute_import

import json
from unittest.mock import MagicMock, patch

import pytest

from sagemaker.ai_registry.data_preparer._case_conversion import (
    _normalize_keys_to_pascal,
    _snake_to_pascal,
)
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.constants import JobCategory


class TestSnakeToPascal:
    """Edge cases for _snake_to_pascal."""

    def test_empty_string(self):
        assert _snake_to_pascal("") == ""

    def test_single_word(self):
        assert _snake_to_pascal("foo") == "Foo"

    def test_two_words(self):
        assert _snake_to_pascal("foo_bar") == "FooBar"

    def test_numeric_segment(self):
        assert _snake_to_pascal("s3_uri") == "S3Uri"

    def test_output_s3_uri(self):
        assert _snake_to_pascal("output_s3_uri") == "OutputS3Uri"

    def test_num_samples(self):
        assert _snake_to_pascal("num_samples") == "NumSamples"

    def test_spec_s3_uri(self):
        assert _snake_to_pascal("spec_s3_uri") == "SpecS3Uri"

    def test_kms_key_id(self):
        assert _snake_to_pascal("kms_key_id") == "KmsKeyId"

    def test_idempotent_pascal_no_underscore(self):
        assert _snake_to_pascal("FooBar") == "FooBar"

    def test_idempotent_s3uri(self):
        assert _snake_to_pascal("S3Uri") == "S3Uri"

    def test_idempotent_spec_s3uri(self):
        assert _snake_to_pascal("SpecS3Uri") == "SpecS3Uri"

    def test_mixed_case_with_underscore(self):
        # Has underscore → treated as snake-ish
        assert _snake_to_pascal("Foo_Bar") == "FooBar"

    def test_all_upper_no_underscore(self):
        # Has uppercase, no underscore → already PascalCase
        assert _snake_to_pascal("ABC") == "ABC"

    def test_all_lower_no_underscore(self):
        assert _snake_to_pascal("foo") == "Foo"

    def test_non_string_int(self):
        assert _snake_to_pascal(42) == 42

    def test_non_string_none(self):
        assert _snake_to_pascal(None) is None

    def test_leading_underscore(self):
        # Leading underscore: split produces empty first part, skipped
        assert _snake_to_pascal("_foo_bar") == "FooBar"

    def test_trailing_underscore(self):
        assert _snake_to_pascal("foo_bar_") == "FooBar"

    def test_double_underscore(self):
        assert _snake_to_pascal("foo__bar") == "FooBar"

    def test_single_char_segments(self):
        assert _snake_to_pascal("a_b_c") == "ABC"


class TestNormalizeKeysToPascal:
    """Edge cases for _normalize_keys_to_pascal."""

    def test_flat_dict(self):
        assert _normalize_keys_to_pascal({"foo_bar": 1, "baz": 2}) == {"FooBar": 1, "Baz": 2}

    def test_nested_dict(self):
        inp = {"output_data_config": {"s3_uri": "s3://bucket", "kms_key_id": "key123"}}
        expected = {"OutputDataConfig": {"S3Uri": "s3://bucket", "KmsKeyId": "key123"}}
        assert _normalize_keys_to_pascal(inp) == expected

    def test_list_of_dicts(self):
        inp = {"items": [{"foo_bar": 1}, {"baz_qux": 2}]}
        expected = {"Items": [{"FooBar": 1}, {"BazQux": 2}]}
        assert _normalize_keys_to_pascal(inp) == expected

    def test_tuple_preserved(self):
        inp = {"items": ({"foo_bar": 1},)}
        result = _normalize_keys_to_pascal(inp)
        assert isinstance(result["Items"], tuple)
        assert result["Items"][0] == {"FooBar": 1}

    def test_non_dict_values_unchanged(self):
        inp = {"num_samples": 500, "name": "test", "flag": True}
        result = _normalize_keys_to_pascal(inp)
        assert result == {"NumSamples": 500, "Name": "test", "Flag": True}

    def test_non_string_keys(self):
        inp = {42: "value", None: "other"}
        result = _normalize_keys_to_pascal(inp)
        assert result == {42: "value", None: "other"}

    def test_idempotent_on_pascal(self):
        inp = {"NumSamples": 500, "OutputDataConfig": {"S3Uri": "s3://b"}}
        assert _normalize_keys_to_pascal(inp) == inp

    def test_mixed_casing(self):
        inp = {"NumSamples": 500, "spec_s3_uri": "s3://spec"}
        expected = {"NumSamples": 500, "SpecS3Uri": "s3://spec"}
        assert _normalize_keys_to_pascal(inp) == expected

    def test_does_not_mutate_input(self):
        inp = {"foo_bar": {"baz_qux": 1}}
        original_inner = inp["foo_bar"]
        _normalize_keys_to_pascal(inp)
        assert inp == {"foo_bar": {"baz_qux": 1}}
        assert inp["foo_bar"] is original_inner

    def test_empty_dict(self):
        assert _normalize_keys_to_pascal({}) == {}

    def test_non_dict_input(self):
        assert _normalize_keys_to_pascal("hello") == "hello"
        assert _normalize_keys_to_pascal(42) == 42
        assert _normalize_keys_to_pascal(None) is None

    def test_operation_config_is_kept_as_written(self):
        settings = {"View": [{"Role": "Text", "Column": "answer"}], "languages": ["en"],
                    "min_score": 0.5, "topic_keywords": {"math": ["sum"]}}
        inp = {"evaluation_config": {"operations": [
            {"operator": "language_detection", "config": settings},
            {"operator": "default_metrics"},
        ]}}
        assert _normalize_keys_to_pascal(inp) == {"EvaluationConfig": {"Operations": [
            {"Operator": "language_detection", "Config": settings},
            {"Operator": "default_metrics"},
        ]}}

    def test_transformation_config_keys_are_kept_as_written(self):
        inp = {"Operations": [
            {"Operator": "Split", "Config": {"Ratios": {"train": 0.8, "test": 0.2}}},
            {"Operator": "PiiRedaction", "Config": {"ActionMap": {"CREDIT_DEBIT_NUMBER": "MASK"}}},
        ]}
        assert _normalize_keys_to_pascal(inp) == inp


SIMPLE_SCHEMA = {
    "properties": {
        "NumSamples": {"type": "integer"},
        "SpecS3Uri": {"type": "string"},
        "OutputDataConfig": {
            "type": "object",
            "properties": {
                "S3Uri": {"type": "string"},
                "KmsKeyId": {"type": "string"},
            },
        },
    },
    "required": ["NumSamples"],
}


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.boto_region_name = "us-west-2"
    session.account_id.return_value = "123456789012"
    return session


@pytest.fixture
def mock_config_mgr():
    mgr = MagicMock()
    mgr.resolve_value_from_config.return_value = None
    return mgr


def _build_test_model():
    from pydantic import Field

    from sagemaker.ai_registry.data_preparer._schema_resolver import _DynamicBase, create_model

    nested_model = create_model(
        "OutputDataConfig",
        __base__=_DynamicBase,
        s3_uri=(str | None, Field(default=None, alias="S3Uri")),
        kms_key_id=(str | None, Field(default=None, alias="KmsKeyId")),
    )
    return create_model(
        "TestModel",
        __base__=_DynamicBase,
        num_samples=(int, Field(..., alias="NumSamples")),
        spec_s3_uri=(str | None, Field(default=None, alias="SpecS3Uri")),
        output_data_config=(nested_model | None, Field(default=None, alias="OutputDataConfig")),
    )


class TestDatasetJobConfigSnakeCaseIntegration:
    """End-to-end: snake_case input → PascalCase JSON output."""

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_snake_case_input_produces_pascal_json(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {
            "NumSamples": 500,
            "SpecS3Uri": "s3://bucket/spec.txt",
        }
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)
        mock_resolver.build_dynamic_model.return_value = _build_test_model()

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            num_samples=500,
            spec_s3_uri="s3://bucket/spec.txt",
        )
        cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        doc = json.loads(cfg.to_job_config_document())
        assert "NumSamples" in doc
        assert "SpecS3Uri" in doc
        # No snake_case keys
        for key in doc:
            assert "_" not in key, f"snake_case key leaked: {key}"

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_pascal_case_input_backward_compat(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {
            "NumSamples": 100,
            "SpecS3Uri": "s3://bucket/spec.txt",
        }
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)
        mock_resolver.build_dynamic_model.return_value = _build_test_model()

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            NumSamples=100,
            SpecS3Uri="s3://bucket/spec.txt",
        )
        cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        doc = json.loads(cfg.to_job_config_document())
        assert doc["NumSamples"] == 100
        assert doc["SpecS3Uri"] == "s3://bucket/spec.txt"

    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_mixed_case_input_normalized(
        self, mock_defaults, mock_resolver, mock_session, mock_config_mgr
    ):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {
            "NumSamples": 200,
            "SpecS3Uri": "s3://bucket/spec.txt",
        }
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)
        mock_resolver.build_dynamic_model.return_value = _build_test_model()

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            NumSamples=200,
            spec_s3_uri="s3://bucket/spec.txt",
        )
        cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        doc = json.loads(cfg.to_job_config_document())
        assert doc["NumSamples"] == 200
        assert doc["SpecS3Uri"] == "s3://bucket/spec.txt"

    @patch("sagemaker.ai_registry.data_preparer.configs.AIRHub")
    @patch("sagemaker.ai_registry.data_preparer.configs._SchemaResolver")
    @patch("sagemaker.ai_registry.data_preparer.configs.DataPreparerDefaults")
    def test_nested_dict_snake_case_normalized(
        self, mock_defaults, mock_resolver, mock_air_hub, mock_session, mock_config_mgr
    ):
        mock_defaults.get_sagemaker_session.return_value = mock_session
        mock_defaults.resolve_into_payload.return_value = {
            "NumSamples": 50,
            "OutputDataConfig": {"S3Uri": "s3://out", "KmsKeyId": "key-123"},
        }
        mock_resolver.get_schema.return_value = ("1.0.0", SIMPLE_SCHEMA)
        mock_resolver.build_dynamic_model.return_value = _build_test_model()
        # Root cause: configs._resolve_and_validate calls AIRHub.get_hub_name() when
        # OutputDataConfig is a dict without HubName and job_category is SDG.
        # AIRHub.get_hub_name() calls sts.get_caller_identity() which needs real
        # credentials. Mock it here so the test stays hermetic.
        mock_air_hub.get_hub_name.return_value = "amazon-sagemaker-hub-123456789012"

        cfg = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            num_samples=50,
            output_data_config={"s3_uri": "s3://out", "kms_key_id": "key-123"},
        )
        cfg._resolve_and_validate(sagemaker_session=mock_session, config_mgr=mock_config_mgr)

        doc = json.loads(cfg.to_job_config_document())
        assert doc["NumSamples"] == 50
        odc = doc["OutputDataConfig"]
        assert odc["S3Uri"] == "s3://out"
        assert odc["KmsKeyId"] == "key-123"
        # No snake_case keys in nested
        for key in odc:
            assert "_" not in key, f"snake_case key leaked in nested: {key}"
