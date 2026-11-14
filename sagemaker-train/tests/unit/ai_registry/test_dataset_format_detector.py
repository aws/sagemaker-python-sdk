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
"""Focused unit tests for DatasetFormatDetector.

Each test asserts a specific boolean or value; nothing asserts only ``is not None``.
Tests are grouped by the method they exercise. Where a method is also reachable
through ``validate_dataset``, the dispatch path is exercised in the
TestValidateDatasetDispatch class so that extension-dispatch coverage is
distinct from method-implementation coverage.
"""

import json

from sagemaker.ai_registry.dataset_format_detector import DatasetFormatDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(tmp_path, name: str, record: dict) -> str:
    """Write a single-record JSONL file and return its path as a string."""
    p = tmp_path / name
    p.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# _load_schema
# ---------------------------------------------------------------------------


class TestLoadSchema:
    """_load_schema returns the parsed JSON schema for a known format and {} otherwise."""

    def test_known_format_returns_non_empty_dict(self):
        schema = DatasetFormatDetector._load_schema("genqa")
        assert isinstance(schema, dict)
        assert schema != {}

    def test_known_format_contains_required_field(self):
        schema = DatasetFormatDetector._load_schema("genqa")
        assert schema.get("required") == ["query"]

    def test_converse_schema_loads_and_requires_messages(self):
        schema = DatasetFormatDetector._load_schema("converse")
        assert "messages" in schema.get("required", [])

    def test_unknown_format_returns_empty_dict(self):
        result = DatasetFormatDetector._load_schema("does_not_exist_at_all")
        assert result == {}

    def test_all_seven_schemas_load(self):
        formats = [
            "dpo",
            "converse",
            "hf_preference",
            "hf_prompt_completion",
            "verl",
            "openai_chat",
            "genqa",
        ]
        for fmt in formats:
            schema = DatasetFormatDetector._load_schema(fmt)
            assert schema != {}, f"schema for {fmt!r} should not be empty"


# ---------------------------------------------------------------------------
# _validate_parquet (direct method)
# ---------------------------------------------------------------------------


class TestValidateParquetMethod:
    """_validate_parquet checks the PAR1 magic bytes at offset 0."""

    def test_par1_magic_returns_true(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"PAR1" + b"\x00" * 64)
        assert DatasetFormatDetector._validate_parquet(str(p)) is True

    def test_wrong_magic_returns_false(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"FAKE" + b"\x00" * 64)
        assert DatasetFormatDetector._validate_parquet(str(p)) is False

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.parquet"
        p.write_bytes(b"")
        assert DatasetFormatDetector._validate_parquet(str(p)) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert DatasetFormatDetector._validate_parquet(str(tmp_path / "gone.parquet")) is False


# ---------------------------------------------------------------------------
# _validate_csv (direct method)
# ---------------------------------------------------------------------------


class TestValidateCsvMethod:
    """_validate_csv accepts a file when a non-empty header row is present."""

    def test_csv_with_header_returns_true(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("prompt,completion\nhello,world\n", encoding="utf-8")
        assert DatasetFormatDetector._validate_csv(str(p)) is True

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert DatasetFormatDetector._validate_csv(str(p)) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert DatasetFormatDetector._validate_csv(str(tmp_path / "gone.csv")) is False

    def test_single_column_header_returns_true(self, tmp_path):
        p = tmp_path / "one_col.csv"
        p.write_text("text\nhello\n", encoding="utf-8")
        assert DatasetFormatDetector._validate_csv(str(p)) is True


# ---------------------------------------------------------------------------
# _validate_json (direct method)
# ---------------------------------------------------------------------------


class TestValidateJsonMethod:
    """_validate_json accepts only a non-empty JSON array of objects."""

    def test_non_empty_array_returns_true(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"prompt": "hi", "completion": "there"}]), encoding="utf-8")
        assert DatasetFormatDetector._validate_json(str(p)) is True

    def test_empty_array_returns_false(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]", encoding="utf-8")
        assert DatasetFormatDetector._validate_json(str(p)) is False

    def test_json_dict_not_array_returns_false(self, tmp_path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"prompt": "hi"}), encoding="utf-8")
        assert DatasetFormatDetector._validate_json(str(p)) is False

    def test_malformed_json_returns_false(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        assert DatasetFormatDetector._validate_json(str(p)) is False

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert DatasetFormatDetector._validate_json(str(tmp_path / "gone.json")) is False


# ---------------------------------------------------------------------------
# _is_rft_format (direct method)
# ---------------------------------------------------------------------------


class TestIsRftFormat:
    """_is_rft_format requires a non-empty messages list; each message needs
    string role and string content."""

    # --- positive branch ---

    def test_valid_messages_with_string_content_returns_true(self):
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ]
        }
        assert DatasetFormatDetector._is_rft_format(data) is True

    def test_single_message_with_string_content_returns_true(self):
        data = {"messages": [{"role": "user", "content": "Hello"}]}
        assert DatasetFormatDetector._is_rft_format(data) is True

    # --- not a dict ---

    def test_list_input_returns_false(self):
        assert DatasetFormatDetector._is_rft_format([{"messages": []}]) is False

    def test_string_input_returns_false(self):
        assert DatasetFormatDetector._is_rft_format("not a dict") is False

    def test_none_input_returns_false(self):
        assert DatasetFormatDetector._is_rft_format(None) is False

    # --- missing or malformed messages key ---

    def test_missing_messages_key_returns_false(self):
        assert DatasetFormatDetector._is_rft_format({"role": "user"}) is False

    def test_messages_not_a_list_returns_false(self):
        assert DatasetFormatDetector._is_rft_format({"messages": "not a list"}) is False

    def test_messages_empty_list_returns_false(self):
        assert DatasetFormatDetector._is_rft_format({"messages": []}) is False

    # --- malformed message items ---

    def test_message_not_a_dict_returns_false(self):
        data = {"messages": ["not a dict"]}
        assert DatasetFormatDetector._is_rft_format(data) is False

    def test_message_missing_role_returns_false(self):
        data = {"messages": [{"content": "Hello"}]}
        assert DatasetFormatDetector._is_rft_format(data) is False

    def test_message_missing_content_returns_false(self):
        data = {"messages": [{"role": "user"}]}
        assert DatasetFormatDetector._is_rft_format(data) is False

    def test_message_role_not_string_returns_false(self):
        data = {"messages": [{"role": 42, "content": "Hello"}]}
        assert DatasetFormatDetector._is_rft_format(data) is False

    def test_message_content_not_string_returns_false(self):
        # content as a list (like the Converse format) must not pass RFT check
        data = {"messages": [{"role": "user", "content": [{"text": "Hello"}]}]}
        assert DatasetFormatDetector._is_rft_format(data) is False

    def test_second_message_malformed_returns_false(self):
        # First message valid, second missing content
        data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant"},
            ]
        }
        assert DatasetFormatDetector._is_rft_format(data) is False


# ---------------------------------------------------------------------------
# validate_dataset — extension dispatch
# ---------------------------------------------------------------------------


class TestValidateDatasetDispatch:
    """validate_dataset routes to the right sub-validator based on file extension."""

    def test_parquet_extension_valid_magic_returns_true(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"PAR1" + b"\x00" * 64)
        assert DatasetFormatDetector.validate_dataset(str(p)) is True

    def test_parquet_extension_wrong_magic_returns_false(self, tmp_path):
        p = tmp_path / "data.parquet"
        p.write_bytes(b"XLSX" + b"\x00" * 64)
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_parquet_extension_nonexistent_returns_false(self, tmp_path):
        assert DatasetFormatDetector.validate_dataset(str(tmp_path / "gone.parquet")) is False

    def test_csv_extension_with_header_returns_true(self, tmp_path):
        p = tmp_path / "data.csv"
        p.write_text("prompt,completion\nhello,world\n", encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is True

    def test_csv_extension_empty_returns_false(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_csv_extension_nonexistent_returns_false(self, tmp_path):
        assert DatasetFormatDetector.validate_dataset(str(tmp_path / "gone.csv")) is False

    def test_json_extension_array_returns_true(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"prompt": "hi", "completion": "there"}]), encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is True

    def test_json_extension_dict_returns_false(self, tmp_path):
        p = tmp_path / "dict.json"
        p.write_text(json.dumps({"prompt": "hi"}), encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_json_extension_nonexistent_returns_false(self, tmp_path):
        assert DatasetFormatDetector.validate_dataset(str(tmp_path / "gone.json")) is False


# ---------------------------------------------------------------------------
# validate_dataset — JSONL schema-based validation (happy path, one per format)
# ---------------------------------------------------------------------------


class TestValidateDatasetJsonlFormats:
    """Each of the seven named JSONL schemas validates a conforming record."""

    def test_converse_format_validates(self, tmp_path):
        """Minimal Converse record: messages array with user + assistant turns;
        content is an array of content blocks (text items)."""
        record = {
            "messages": [
                {"role": "user", "content": [{"text": "Hello"}]},
                {"role": "assistant", "content": [{"text": "Hi there"}]},
            ]
        }
        path = _write_jsonl(tmp_path, "converse.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_dpo_format_validates(self, tmp_path):
        """DPO record: one user message followed by an assistant candidates block
        with preferred and non-preferred labels."""
        record = {
            "messages": [
                {"role": "user", "content": [{"text": "What is 2+2?"}]},
                {
                    "role": "assistant",
                    "candidates": [
                        {"content": [{"text": "4"}], "preferenceLabel": "preferred"},
                        {"content": [{"text": "5"}], "preferenceLabel": "non-preferred"},
                    ],
                },
            ]
        }
        path = _write_jsonl(tmp_path, "dpo.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_hf_preference_format_validates(self, tmp_path):
        """HF Preference record: chosen and rejected response strings."""
        record = {"chosen": "Response A is correct", "rejected": "Response B is wrong"}
        path = _write_jsonl(tmp_path, "hf_pref.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_hf_prompt_completion_format_validates(self, tmp_path):
        """HF Prompt-Completion record: prompt and completion strings."""
        record = {"prompt": "What is the capital of France?", "completion": "Paris"}
        path = _write_jsonl(tmp_path, "hf_pc.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_verl_format_validates(self, tmp_path):
        """VERL record: a prompt string (the only required field)."""
        record = {"prompt": "Solve: what is 1 + 1?"}
        path = _write_jsonl(tmp_path, "verl.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_openai_chat_format_validates(self, tmp_path):
        """OpenAI chat record: messages with string content (not array content)
        to distinguish from Converse/DPO, using non-enum role to avoid matching
        those schemas first."""
        record = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        path = _write_jsonl(tmp_path, "openai.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_genqa_format_validates(self, tmp_path):
        """GenQA record: only the query field is required."""
        record = {"query": "What is the capital of France?", "response": "Paris"}
        path = _write_jsonl(tmp_path, "genqa.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True


# ---------------------------------------------------------------------------
# validate_dataset — JSONL schema violation cases (should return False)
# ---------------------------------------------------------------------------


class TestValidateDatasetJsonlSchemaViolations:
    """Records that violate every known schema return False."""

    def test_missing_all_required_fields_returns_false(self, tmp_path):
        """An empty object has no required fields for any schema."""
        path = _write_jsonl(tmp_path, "empty_obj.jsonl", {})
        assert DatasetFormatDetector.validate_dataset(path) is False

    def test_genqa_missing_required_query_returns_false(self, tmp_path):
        """A record with only 'response' fails genqa (query required) and all others."""
        path = _write_jsonl(tmp_path, "genqa_bad.jsonl", {"response": "Paris"})
        assert DatasetFormatDetector.validate_dataset(path) is False

    def test_genqa_wrong_query_type_returns_false(self, tmp_path):
        """genqa requires query to be a string; an integer type fails validation."""
        path = _write_jsonl(tmp_path, "genqa_type.jsonl", {"query": 123})
        assert DatasetFormatDetector.validate_dataset(path) is False

    def test_hf_preference_missing_rejected_returns_false(self, tmp_path):
        """hf_preference requires both chosen and rejected; omitting rejected fails."""
        path = _write_jsonl(tmp_path, "hf_pref_bad.jsonl", {"chosen": "A"})
        assert DatasetFormatDetector.validate_dataset(path) is False

    def test_hf_prompt_completion_missing_completion_matches_verl_instead(self, tmp_path):
        """A record with only 'prompt' fails hf_prompt_completion (missing
        completion) but VERL requires only 'prompt', so the detector returns
        True — the record is legitimately valid under VERL."""
        path = _write_jsonl(tmp_path, "hf_pc_verl.jsonl", {"prompt": "question"})
        assert DatasetFormatDetector.validate_dataset(path) is True


# ---------------------------------------------------------------------------
# validate_dataset — JSONL failure modes
# ---------------------------------------------------------------------------


class TestValidateDatasetJsonlFailures:
    """Error and edge-case handling for the JSONL validation path."""

    def test_nonexistent_file_returns_false(self, tmp_path):
        assert DatasetFormatDetector.validate_dataset(str(tmp_path / "missing.jsonl")) is False

    def test_empty_file_returns_false(self, tmp_path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_only_blank_lines_returns_false(self, tmp_path):
        p = tmp_path / "blanks.jsonl"
        p.write_text("\n\n\n", encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_malformed_json_line_returns_false(self, tmp_path):
        p = tmp_path / "bad.jsonl"
        p.write_text("{not valid json at all\n", encoding="utf-8")
        assert DatasetFormatDetector.validate_dataset(str(p)) is False

    def test_rft_fallback_with_nonstandard_role_returns_true(self, tmp_path):
        """A record whose messages fail every schema role enum but whose content
        is a plain string is accepted via the _is_rft_format fallback."""
        record = {
            "messages": [
                {"role": "tool", "content": "Tool output"},
                {"role": "user", "content": "Thanks"},
            ]
        }
        path = _write_jsonl(tmp_path, "rft.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is True

    def test_no_schema_match_and_no_rft_returns_false(self, tmp_path):
        """A record that fails all schemas and _is_rft_format hits the break path."""
        record = {"unknown_key": "some value with no known required fields"}
        path = _write_jsonl(tmp_path, "nomatch.jsonl", record)
        assert DatasetFormatDetector.validate_dataset(path) is False
