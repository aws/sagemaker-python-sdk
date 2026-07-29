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

"""Tests for dataset validation utilities."""

import pytest
import tempfile
import json
import os

from sagemaker.ai_registry.dataset_validation import (
    load_jsonl,
    validate_sft,
    validate_dpo,
    validate_rlvr,
    detect_dataset_type,
    normalize_rlvr_row,
    validate_dataset
)


class TestLoadJsonl:
    def test_load_valid_jsonl(self):
        """Test loading valid JSONL file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"key": "value1"}\n')
            f.write('{"key": "value2"}\n')
            temp_path = f.name
        
        try:
            result = load_jsonl(temp_path)
            assert len(result) == 2
            assert result[0]["key"] == "value1"
            assert result[1]["key"] == "value2"
        finally:
            os.unlink(temp_path)

    def test_load_jsonl_with_empty_lines(self):
        """Test loading JSONL with empty lines."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"key": "value1"}\n')
            f.write('\n')
            f.write('{"key": "value2"}\n')
            temp_path = f.name
        
        try:
            result = load_jsonl(temp_path)
            assert len(result) == 2
        finally:
            os.unlink(temp_path)

    def test_load_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"key": "value1"}\n')
            f.write('invalid json\n')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="JSON decode error"):
                load_jsonl(temp_path)
        finally:
            os.unlink(temp_path)


class TestValidateSft:
    def test_validate_sft_input_output(self):
        """Test SFT validation with input/output format."""
        rows = [
            {"input": "test input", "output": "test output"},
            {"input": "another input", "output": "another output"}
        ]
        validate_sft(rows)

    def test_validate_sft_prompt_completion(self):
        """Test SFT validation with prompt/completion format."""
        rows = [
            {"prompt": "test prompt", "completion": "test completion"},
            {"prompt": "another prompt", "completion": "another completion"}
        ]
        validate_sft(rows)

    def test_validate_sft_invalid_type(self):
        """Test SFT validation fails with non-string values."""
        rows = [{"input": 123, "output": "test"}]
        with pytest.raises(ValueError, match="input/output must be strings"):
            validate_sft(rows)

    def test_validate_sft_missing_fields(self):
        """Test SFT validation fails with missing fields."""
        rows = [{"input": "test"}]
        with pytest.raises(ValueError, match="missing SFT fields"):
            validate_sft(rows)


class TestValidateDpo:
    def test_validate_dpo_valid(self):
        """Test DPO validation with valid data."""
        rows = [
            {
                "prompt": "test prompt",
                "chosen": "chosen response",
                "rejected": "rejected response"
            }
        ]
        validate_dpo(rows)

    def test_validate_dpo_missing_field(self):
        """Test DPO validation fails with missing field."""
        rows = [{"prompt": "test", "chosen": "chosen"}]
        with pytest.raises(ValueError, match="missing prompt|chosen|rejected"):
            validate_dpo(rows)

    def test_validate_dpo_invalid_type(self):
        """Test DPO validation fails with non-string values."""
        rows = [{"prompt": "test", "chosen": 123, "rejected": "rejected"}]
        with pytest.raises(ValueError, match="must be string"):
            validate_dpo(rows)


class TestValidateRlvr:
    def test_validate_rlvr_valid(self):
        """Test RLVR validation with valid data."""
        rows = [
            {
                "prompt": "test prompt",
                "samples": [
                    {"completion": "completion1", "score": 0.9},
                    {"completion": "completion2", "score": 0.7}
                ]
            }
        ]
        validate_rlvr(rows)

    def test_validate_rlvr_invalid_prompt(self):
        """Test RLVR validation fails with non-string prompt."""
        rows = [{"prompt": 123, "samples": []}]
        with pytest.raises(ValueError, match="prompt must be string"):
            validate_rlvr(rows)

    def test_validate_rlvr_missing_samples(self):
        """Test RLVR validation fails without samples."""
        rows = [{"prompt": "test"}]
        with pytest.raises(ValueError, match="samples must be list"):
            validate_rlvr(rows)

    def test_validate_rlvr_invalid_completion(self):
        """Test RLVR validation fails with non-string completion."""
        rows = [
            {
                "prompt": "test",
                "samples": [{"completion": 123, "score": 0.9}]
            }
        ]
        with pytest.raises(ValueError, match="completion must be string"):
            validate_rlvr(rows)

    def test_validate_rlvr_invalid_score(self):
        """Test RLVR validation fails with non-numeric score."""
        rows = [
            {
                "prompt": "test",
                "samples": [{"completion": "test", "score": "invalid"}]
            }
        ]
        with pytest.raises(ValueError, match="score must be number"):
            validate_rlvr(rows)


class TestDetectDatasetType:
    def test_detect_rlvr(self):
        """Test detecting RLVR format."""
        record = {
            "prompt": "test",
            "samples": [{"completion": "test", "score": 0.9}]
        }
        assert detect_dataset_type(record) == "rlvr"

    def test_detect_dpo(self):
        """Test detecting DPO format."""
        record = {
            "prompt": "test",
            "chosen": "chosen",
            "rejected": "rejected"
        }
        assert detect_dataset_type(record) == "dpo"

    def test_detect_sft_input_output(self):
        """Test detecting SFT format with input/output."""
        record = {"input": "test", "output": "test"}
        assert detect_dataset_type(record) == "sft"

    def test_detect_sft_prompt_completion(self):
        """Test detecting SFT format with prompt/completion."""
        record = {"prompt": "test", "completion": "test"}
        assert detect_dataset_type(record) == "sft"

    def test_detect_unknown(self):
        """Test detecting unknown format."""
        record = {"unknown": "field"}
        assert detect_dataset_type(record) is None


class TestNormalizeRlvrRow:
    def test_normalize_string_prompt(self):
        """Test normalizing RLVR row with string prompt."""
        record = {
            "prompt": "test prompt",
            "extra_info": {"answer": "test answer"}
        }
        result = normalize_rlvr_row(record)
        
        assert result["prompt"] == "test prompt"
        assert result["samples"][0]["completion"] == "test answer"
        assert result["samples"][0]["score"] == 1.0

    def test_normalize_list_prompt(self):
        """Test normalizing RLVR row with list prompt."""
        record = {
            "prompt": [
                {"content": "line1"},
                {"content": "line2"}
            ],
            "reward_model": {"ground_truth": "answer"}
        }
        result = normalize_rlvr_row(record)
        
        assert result["prompt"] == "line1\nline2"
        assert result["samples"][0]["completion"] == "answer"

    def test_normalize_no_completion(self):
        """Test normalizing RLVR row without completion."""
        record = {"prompt": "test"}
        result = normalize_rlvr_row(record)
        
        assert result["prompt"] == "test"
        assert result["samples"][0]["completion"] == ""
        assert result["samples"][0]["score"] == 0.0


class TestValidateDataset:
    def test_validate_dataset_sft(self):
        """Test validating SFT dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input": "test1", "output": "output1"}\n')
            f.write('{"input": "test2", "output": "output2"}\n')
            temp_path = f.name
        
        try:
            validate_dataset(temp_path, "sft")
        finally:
            os.unlink(temp_path)

    def test_validate_dataset_dpo(self):
        """Test validating DPO dataset."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"prompt": "p1", "chosen": "c1", "rejected": "r1"}\n')
            temp_path = f.name
        
        try:
            validate_dataset(temp_path, "dpo")
        finally:
            os.unlink(temp_path)

    def test_validate_dataset_auto_detect(self):
        """Test auto-detecting dataset type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input": "test", "output": "output"}\n')
            temp_path = f.name
        
        try:
            validate_dataset(temp_path, "auto")
        finally:
            os.unlink(temp_path)

    def test_validate_dataset_invalid_technique(self):
        """Test validation fails with invalid technique."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"input": "test", "output": "output"}\n')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="technique must be one of"):
                validate_dataset(temp_path, "invalid")
        finally:
            os.unlink(temp_path)

    def test_validate_dataset_auto_detect_failure(self):
        """Test auto-detect fails with unknown format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"unknown": "field"}\n')
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Cannot auto-detect"):
                validate_dataset(temp_path, "auto")
        finally:
            os.unlink(temp_path)
