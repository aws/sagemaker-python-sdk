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
"""Tests for SageMaker Evaluation Module Constants."""
from __future__ import absolute_import

import pytest
from enum import Enum

from sagemaker.train.evaluate.constants import (
    EvalType,
    _PIPELINE_NAME_PREFIX,
    _get_pipeline_name,
    _get_pipeline_name_prefix,
    _get_eval_type_display_name,
    _TAG_EVAL_TYPE_PREFIX,
    _TAG_EVALUATION,
    _TAG_SAGEMAKER_MODEL_EVALUATION,
    _get_eval_type_tag_key,
)


class TestEvalType:
    """Test cases for EvalType enum."""

    def test_eval_type_is_enum(self):
        """Test that EvalType is an Enum."""
        assert issubclass(EvalType, Enum)

    def test_eval_type_benchmark_value(self):
        """Test BENCHMARK enum value."""
        assert EvalType.BENCHMARK.value == "benchmark"

    def test_eval_type_custom_scorer_value(self):
        """Test CUSTOM_SCORER enum value."""
        assert EvalType.CUSTOM_SCORER.value == "customscorer"

    def test_eval_type_llm_as_judge_value(self):
        """Test LLM_AS_JUDGE enum value."""
        assert EvalType.LLM_AS_JUDGE.value == "llmasjudge"

    def test_eval_type_has_all_expected_members(self):
        """Test that EvalType has exactly three expected members."""
        expected_members = {"BENCHMARK", "CUSTOM_SCORER", "LLM_AS_JUDGE"}
        actual_members = {member.name for member in EvalType}
        assert actual_members == expected_members

    def test_eval_type_member_count(self):
        """Test that EvalType has exactly three members."""
        assert len(EvalType) == 3

    def test_eval_type_members_accessible(self):
        """Test that all enum members are accessible."""
        assert hasattr(EvalType, "BENCHMARK")
        assert hasattr(EvalType, "CUSTOM_SCORER")
        assert hasattr(EvalType, "LLM_AS_JUDGE")

    def test_eval_type_equality(self):
        """Test enum member equality."""
        assert EvalType.BENCHMARK == EvalType.BENCHMARK
        assert EvalType.BENCHMARK != EvalType.CUSTOM_SCORER
        assert EvalType.CUSTOM_SCORER != EvalType.LLM_AS_JUDGE

    def test_eval_type_can_iterate(self):
        """Test that we can iterate over EvalType members."""
        members = list(EvalType)
        assert len(members) == 3
        assert EvalType.BENCHMARK in members
        assert EvalType.CUSTOM_SCORER in members
        assert EvalType.LLM_AS_JUDGE in members

    def test_eval_type_value_uniqueness(self):
        """Test that all enum values are unique."""
        values = [member.value for member in EvalType]
        assert len(values) == len(set(values))

    def test_eval_type_access_by_value(self):
        """Test accessing enum members by value."""
        assert EvalType("benchmark") == EvalType.BENCHMARK
        assert EvalType("customscorer") == EvalType.CUSTOM_SCORER
        assert EvalType("llmasjudge") == EvalType.LLM_AS_JUDGE

    def test_eval_type_invalid_value_raises_error(self):
        """Test that accessing with invalid value raises ValueError."""
        with pytest.raises(ValueError):
            EvalType("invalid_type")


class TestConstants:
    """Test cases for module constants."""

    def test_pipeline_name_prefix_value(self):
        """Test _PIPELINE_NAME_PREFIX constant value."""
        assert _PIPELINE_NAME_PREFIX == "SagemakerEvaluation"

    def test_pipeline_name_prefix_is_string(self):
        """Test _PIPELINE_NAME_PREFIX is a string."""
        assert isinstance(_PIPELINE_NAME_PREFIX, str)

    def test_tag_eval_type_prefix_value(self):
        """Test _TAG_EVAL_TYPE_PREFIX constant value."""
        assert _TAG_EVAL_TYPE_PREFIX == "sagemaker-pysdk"

    def test_tag_eval_type_prefix_is_string(self):
        """Test _TAG_EVAL_TYPE_PREFIX is a string."""
        assert isinstance(_TAG_EVAL_TYPE_PREFIX, str)

    def test_tag_evaluation_value(self):
        """Test _TAG_EVALUATION constant value."""
        assert _TAG_EVALUATION == "sagemaker-pysdk-evaluation"

    def test_tag_evaluation_is_string(self):
        """Test _TAG_EVALUATION is a string."""
        assert isinstance(_TAG_EVALUATION, str)

    def test_tag_sagemaker_model_evaluation_value(self):
        """Test _TAG_SAGEMAKER_MODEL_EVALUATION constant value."""
        assert _TAG_SAGEMAKER_MODEL_EVALUATION == "SagemakerModelEvaluation"

    def test_tag_sagemaker_model_evaluation_is_string(self):
        """Test _TAG_SAGEMAKER_MODEL_EVALUATION is a string."""
        assert isinstance(_TAG_SAGEMAKER_MODEL_EVALUATION, str)


class TestGetEvalTypeDisplayName:
    """Test cases for _get_eval_type_display_name function."""

    @pytest.mark.parametrize(
        "eval_type,expected_name",
        [
            (EvalType.BENCHMARK, "BenchmarkEvaluation"),
            (EvalType.CUSTOM_SCORER, "CustomScorerEvaluation"),
            (EvalType.LLM_AS_JUDGE, "LLMAJEvaluation"),
        ],
        ids=["benchmark", "custom_scorer", "llm_as_judge"],
    )
    def test_get_eval_type_display_name_valid_types(self, eval_type, expected_name):
        """Test _get_eval_type_display_name with valid evaluation types."""
        result = _get_eval_type_display_name(eval_type)
        assert result == expected_name

    def test_get_eval_type_display_name_returns_string(self):
        """Test that _get_eval_type_display_name returns a string."""
        result = _get_eval_type_display_name(EvalType.BENCHMARK)
        assert isinstance(result, str)

    def test_get_eval_type_display_name_no_spaces(self):
        """Test that display names don't contain spaces."""
        for eval_type in EvalType:
            result = _get_eval_type_display_name(eval_type)
            assert " " not in result

    def test_get_eval_type_display_name_starts_with_capital(self):
        """Test that display names start with capital letter."""
        for eval_type in EvalType:
            result = _get_eval_type_display_name(eval_type)
            assert result[0].isupper()


class TestGetPipelineName:
    """Test cases for _get_pipeline_name function."""

    def test_get_pipeline_name_generates_uuid(self):
        """Test _get_pipeline_name generates unique names with UUID."""
        name1 = _get_pipeline_name(EvalType.BENCHMARK)
        name2 = _get_pipeline_name(EvalType.BENCHMARK)
        # Names should be different due to UUID
        assert name1 != name2

    def test_get_pipeline_name_returns_string(self):
        """Test that _get_pipeline_name returns a string."""
        result = _get_pipeline_name(EvalType.BENCHMARK)
        assert isinstance(result, str)

    def test_get_pipeline_name_format(self):
        """Test that pipeline name follows expected format."""
        for eval_type in EvalType:
            result = _get_pipeline_name(eval_type)
            assert result.startswith(_PIPELINE_NAME_PREFIX)
            assert "-" in result
            # Should contain display name
            display_name = _get_eval_type_display_name(eval_type)
            assert display_name in result

    def test_get_pipeline_name_with_provided_uuid(self):
        """Test _get_pipeline_name with provided UUID."""
        test_uuid = "test-uuid-12345"
        result = _get_pipeline_name(EvalType.BENCHMARK, unique_id=test_uuid)
        assert test_uuid in result
        # Should be consistent with same UUID
        result2 = _get_pipeline_name(EvalType.BENCHMARK, unique_id=test_uuid)
        assert result == result2

    def test_get_pipeline_name_different_types_different_names(self):
        """Test that different eval types produce different pipeline names."""
        test_uuid = "same-uuid"
        benchmark_name = _get_pipeline_name(EvalType.BENCHMARK, unique_id=test_uuid)
        custom_scorer_name = _get_pipeline_name(EvalType.CUSTOM_SCORER, unique_id=test_uuid)
        llm_judge_name = _get_pipeline_name(EvalType.LLM_AS_JUDGE, unique_id=test_uuid)

        assert benchmark_name != custom_scorer_name
        assert benchmark_name != llm_judge_name
        assert custom_scorer_name != llm_judge_name

    def test_get_pipeline_name_no_spaces(self):
        """Test that pipeline names don't contain spaces."""
        for eval_type in EvalType:
            result = _get_pipeline_name(eval_type)
            assert " " not in result

    def test_get_pipeline_name_uses_display_name(self):
        """Test that pipeline name uses display name (not enum value)."""
        result = _get_pipeline_name(EvalType.BENCHMARK)
        assert "BenchmarkEvaluation" in result
        assert "benchmark" not in result.replace("Benchmark", "")


class TestGetPipelineNamePrefix:
    """Test cases for _get_pipeline_name_prefix function."""

    @pytest.mark.parametrize(
        "eval_type,expected_prefix",
        [
            (EvalType.BENCHMARK, "SagemakerEvaluation-BenchmarkEvaluation"),
            (EvalType.CUSTOM_SCORER, "SagemakerEvaluation-CustomScorerEvaluation"),
            (EvalType.LLM_AS_JUDGE, "SagemakerEvaluation-LLMAJEvaluation"),
        ],
        ids=["benchmark", "custom_scorer", "llm_as_judge"],
    )
    def test_get_pipeline_name_prefix_valid_types(self, eval_type, expected_prefix):
        """Test _get_pipeline_name_prefix with valid evaluation types."""
        result = _get_pipeline_name_prefix(eval_type)
        assert result == expected_prefix

    def test_get_pipeline_name_prefix_returns_string(self):
        """Test that _get_pipeline_name_prefix returns a string."""
        result = _get_pipeline_name_prefix(EvalType.BENCHMARK)
        assert isinstance(result, str)

    def test_get_pipeline_name_prefix_no_trailing_hyphen(self):
        """Test that prefix doesn't end with hyphen (AWS validation requirement)."""
        for eval_type in EvalType:
            result = _get_pipeline_name_prefix(eval_type)
            assert not result.endswith('-')

    def test_get_pipeline_name_prefix_format(self):
        """Test that prefix follows expected format."""
        for eval_type in EvalType:
            result = _get_pipeline_name_prefix(eval_type)
            assert result.startswith(_PIPELINE_NAME_PREFIX)
            display_name = _get_eval_type_display_name(eval_type)
            assert display_name in result

    def test_get_pipeline_name_starts_with_prefix(self):
        """Test that generated pipeline names start with the prefix."""
        for eval_type in EvalType:
            prefix = _get_pipeline_name_prefix(eval_type)
            pipeline_name = _get_pipeline_name(eval_type, unique_id="test")
            assert pipeline_name.startswith(prefix)


class TestGetEvalTypeTagKey:
    """Test cases for _get_eval_type_tag_key function."""

    @pytest.mark.parametrize(
        "eval_type,expected_tag",
        [
            (EvalType.BENCHMARK, "sagemaker-pysdk-benchmark"),
            (EvalType.CUSTOM_SCORER, "sagemaker-pysdk-customscorer"),
            (EvalType.LLM_AS_JUDGE, "sagemaker-pysdk-llmasjudge"),
        ],
        ids=["benchmark", "custom_scorer", "llm_as_judge"],
    )
    def test_get_eval_type_tag_key_valid_types(self, eval_type, expected_tag):
        """Test _get_eval_type_tag_key with valid evaluation types."""
        result = _get_eval_type_tag_key(eval_type)
        assert result == expected_tag

    def test_get_eval_type_tag_key_returns_string(self):
        """Test that _get_eval_type_tag_key returns a string."""
        result = _get_eval_type_tag_key(EvalType.BENCHMARK)
        assert isinstance(result, str)

    def test_get_eval_type_tag_key_format(self):
        """Test that tag key follows expected format."""
        for eval_type in EvalType:
            result = _get_eval_type_tag_key(eval_type)
            assert result.startswith(_TAG_EVAL_TYPE_PREFIX)
            assert result.endswith(eval_type.value)
            assert "-" in result

    def test_get_eval_type_tag_key_consistency(self):
        """Test that _get_eval_type_tag_key returns consistent results."""
        result1 = _get_eval_type_tag_key(EvalType.BENCHMARK)
        result2 = _get_eval_type_tag_key(EvalType.BENCHMARK)
        assert result1 == result2

    def test_get_eval_type_tag_key_different_types_different_keys(self):
        """Test that different eval types produce different tag keys."""
        benchmark_tag = _get_eval_type_tag_key(EvalType.BENCHMARK)
        custom_scorer_tag = _get_eval_type_tag_key(EvalType.CUSTOM_SCORER)
        llm_judge_tag = _get_eval_type_tag_key(EvalType.LLM_AS_JUDGE)

        assert benchmark_tag != custom_scorer_tag
        assert benchmark_tag != llm_judge_tag
        assert custom_scorer_tag != llm_judge_tag

    def test_get_eval_type_tag_key_no_spaces(self):
        """Test that tag keys don't contain spaces."""
        for eval_type in EvalType:
            result = _get_eval_type_tag_key(eval_type)
            assert " " not in result

    def test_get_eval_type_tag_key_uses_enum_value(self):
        """Test that tag key uses enum value, not name."""
        # Enum value is lowercase, name is uppercase
        result = _get_eval_type_tag_key(EvalType.BENCHMARK)
        assert "benchmark" in result
        assert "BENCHMARK" not in result

    def test_get_eval_type_tag_key_all_lowercase(self):
        """Test that tag keys are all lowercase."""
        for eval_type in EvalType:
            result = _get_eval_type_tag_key(eval_type)
            assert result == result.lower()


class TestIntegration:
    """Integration tests for constants module."""

    def test_pipeline_prefix_and_name_consistency(self):
        """Test that pipeline names start with their prefixes."""
        for eval_type in EvalType:
            prefix = _get_pipeline_name_prefix(eval_type)
            pipeline_name = _get_pipeline_name(eval_type, unique_id="test")
            assert pipeline_name.startswith(prefix)

    def test_all_eval_types_have_valid_pipeline_names(self):
        """Test that all eval types produce valid pipeline names."""
        for eval_type in EvalType:
            pipeline_name = _get_pipeline_name(eval_type)
            assert pipeline_name
            assert len(pipeline_name) > 0
            assert isinstance(pipeline_name, str)

    def test_all_eval_types_have_valid_tag_keys(self):
        """Test that all eval types produce valid tag keys."""
        for eval_type in EvalType:
            tag_key = _get_eval_type_tag_key(eval_type)
            assert tag_key
            assert len(tag_key) > 0
            assert isinstance(tag_key, str)

    def test_all_eval_types_have_valid_prefixes(self):
        """Test that all eval types produce valid pipeline name prefixes."""
        for eval_type in EvalType:
            prefix = _get_pipeline_name_prefix(eval_type)
            assert prefix
            assert len(prefix) > 0
            assert isinstance(prefix, str)
            # Prefix should not end with hyphen
            assert not prefix.endswith('-')

    def test_constants_module_exports(self):
        """Test that all expected exports are available."""
        from sagemaker.train.evaluate import constants

        # Test that main items are accessible (internal functions)
        assert hasattr(constants, "EvalType")
        assert hasattr(constants, "_PIPELINE_NAME_PREFIX")
        assert hasattr(constants, "_get_pipeline_name")
        assert hasattr(constants, "_get_pipeline_name_prefix")
        assert hasattr(constants, "_get_eval_type_display_name")
        assert hasattr(constants, "_TAG_EVAL_TYPE_PREFIX")
        assert hasattr(constants, "_TAG_EVALUATION")
        assert hasattr(constants, "_TAG_SAGEMAKER_MODEL_EVALUATION")
        assert hasattr(constants, "_get_eval_type_tag_key")
