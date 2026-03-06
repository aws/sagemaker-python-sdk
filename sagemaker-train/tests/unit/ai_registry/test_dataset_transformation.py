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

import json
import os
import tempfile
import pytest
from unittest.mock import patch, Mock

from sagemaker.ai_registry.dataset_transformation import (
    DatasetFormat,
    DatasetTransformation,
    convert_to_genqa,
    convert_from_genqa,
    _extract_text_from_messages,
    _extract_all_from_messages,
    _convert_openai_to_genqa,
    _convert_converse_to_genqa,
    _convert_hf_to_genqa,
    _convert_hf_preference_to_genqa,
    _convert_verl_to_genqa,
)


class TestExtractTextFromMessages:
    def test_string_input(self):
        assert _extract_text_from_messages("hello") == "hello"

    def test_list_with_target_role(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "question"},
        ]
        assert _extract_text_from_messages(messages, target_role="user") == "question"

    def test_last_occurrence_wins(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "user", "content": "second"},
        ]
        assert _extract_text_from_messages(messages, target_role="user") == "second"

    def test_no_match_returns_empty(self):
        messages = [{"role": "user", "content": "hi"}]
        assert _extract_text_from_messages(messages, target_role="assistant") == ""

    def test_non_dict_items_skipped(self):
        messages = ["not a dict", {"role": "user", "content": "valid"}]
        assert _extract_text_from_messages(messages, target_role="user") == "valid"

    def test_non_string_non_list_returns_empty(self):
        assert _extract_text_from_messages(12345) == ""
        assert _extract_text_from_messages(None) == ""


class TestExtractAllFromMessages:
    def test_string_input(self):
        system, query, response = _extract_all_from_messages("hello")
        assert system == ""
        assert query == "hello"
        assert response == ""

    def test_list_input(self):
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
        system, query, response = _extract_all_from_messages(messages)
        assert system == "sys prompt"
        assert query == "question"
        assert response == "answer"

    def test_last_occurrence_wins(self):
        messages = [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "resp1"},
            {"role": "user", "content": "second"},
            {"role": "assistant", "content": "resp2"},
        ]
        system, query, response = _extract_all_from_messages(messages)
        assert query == "second"
        assert response == "resp2"

    def test_non_string_non_list(self):
        system, query, response = _extract_all_from_messages(None)
        assert system == ""
        assert query == ""
        assert response == ""


class TestConvertOpenaiToGenqa:
    def test_basic_conversion(self):
        data = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
        result = _convert_openai_to_genqa(data)
        assert result["query"] == "What is 2+2?"
        assert result["response"] == "4"
        assert result["system"] == "You are helpful."
        assert result["metadata"] == {}

    def test_no_assistant_last_clears_response(self):
        data = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
                {"role": "user", "content": "follow up"},
            ]
        }
        result = _convert_openai_to_genqa(data)
        assert result["query"] == "follow up"
        assert result["response"] == ""

    def test_empty_messages(self):
        result = _convert_openai_to_genqa({"messages": []})
        assert result["query"] == ""
        assert result["response"] == ""

    def test_no_system(self):
        data = {
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ]
        }
        result = _convert_openai_to_genqa(data)
        assert result["system"] == ""


class TestConvertConverseToGenqa:
    def test_basic_conversion(self):
        data = {
            "system": [{"text": "Be helpful."}],
            "messages": [
                {"role": "user", "content": [{"text": "What is AI?"}]},
                {"role": "assistant", "content": [{"text": "Artificial Intelligence."}]},
            ],
        }
        result = _convert_converse_to_genqa(data)
        assert result["query"] == "What is AI?"
        assert result["response"] == "Artificial Intelligence."
        assert result["system"] == "Be helpful."

    def test_multiple_text_blocks_concatenated(self):
        data = {
            "messages": [
                {"role": "user", "content": [{"text": "part1"}, {"text": "part2"}]},
                {"role": "assistant", "content": [{"text": "answer"}]},
            ]
        }
        result = _convert_converse_to_genqa(data)
        assert result["query"] == "part1 part2"

    def test_no_system(self):
        data = {
            "messages": [
                {"role": "user", "content": [{"text": "q"}]},
                {"role": "assistant", "content": [{"text": "a"}]},
            ]
        }
        result = _convert_converse_to_genqa(data)
        assert result["system"] == ""

    def test_non_text_blocks_skipped(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"image": {"format": "png"}}, {"text": "describe this"}],
                },
                {"role": "assistant", "content": [{"text": "a cat"}]},
            ]
        }
        result = _convert_converse_to_genqa(data)
        assert result["query"] == "describe this"

    def test_last_message_not_assistant_clears_response(self):
        data = {
            "messages": [
                {"role": "user", "content": [{"text": "q1"}]},
                {"role": "assistant", "content": [{"text": "a1"}]},
                {"role": "user", "content": [{"text": "q2"}]},
            ]
        }
        result = _convert_converse_to_genqa(data)
        assert result["response"] == ""


class TestConvertHfToGenqa:
    def test_string_format(self):
        data = {"prompt": "What is Python?", "completion": "A programming language."}
        result = _convert_hf_to_genqa(data)
        assert result["query"] == "What is Python?"
        assert result["response"] == "A programming language."

    def test_conversational_format(self):
        data = {
            "prompt": [{"role": "user", "content": "question"}],
            "completion": [{"role": "assistant", "content": "answer"}],
        }
        result = _convert_hf_to_genqa(data)
        assert result["query"] == "question"
        assert result["response"] == "answer"

    def test_empty_prompt(self):
        data = {"prompt": "", "completion": "answer"}
        result = _convert_hf_to_genqa(data)
        assert result["query"] == ""


class TestConvertHfPreferenceToGenqa:
    def test_basic_conversion(self):
        data = {
            "input": "What is ML?",
            "chosen": "Machine Learning is...",
            "rejected": "I don't know.",
        }
        result = _convert_hf_preference_to_genqa(data)
        assert result["query"] == "What is ML?"
        assert result["response"] == "Machine Learning is..."
        assert result["metadata"]["chosen"] == "Machine Learning is..."
        assert result["metadata"]["rejected"] == "I don't know."

    def test_prompt_field_fallback(self):
        data = {
            "prompt": "What is DL?",
            "chosen": "Deep Learning.",
            "rejected": "No idea.",
        }
        result = _convert_hf_preference_to_genqa(data)
        assert result["query"] == "What is DL?"

    def test_conversational_format(self):
        data = {
            "input": [{"role": "user", "content": "q"}],
            "chosen": [{"role": "assistant", "content": "good"}],
            "rejected": [{"role": "assistant", "content": "bad"}],
        }
        result = _convert_hf_preference_to_genqa(data)
        assert result["query"] == "q"
        assert result["response"] == "good"
        assert result["metadata"]["rejected"] == "bad"

    def test_preserves_optional_fields(self):
        data = {
            "input": "q",
            "chosen": "a",
            "rejected": "b",
            "id": "123",
            "attributes": {"key": "val"},
            "difficulty": "hard",
        }
        result = _convert_hf_preference_to_genqa(data)
        assert result["metadata"]["id"] == "123"
        assert result["metadata"]["attributes"] == {"key": "val"}
        assert result["metadata"]["difficulty"] == "hard"


class TestConvertVerlToGenqa:
    def test_string_prompt(self):
        data = {"prompt": "Solve 2+2"}
        result = _convert_verl_to_genqa(data)
        assert result["query"] == "Solve 2+2"
        assert result["response"] == ""

    def test_messages_prompt(self):
        data = {
            "prompt": [
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 3+3?"},
            ]
        }
        result = _convert_verl_to_genqa(data)
        assert result["query"] == "What is 3+3?"
        assert result["system"] == "You are a math tutor."

    def test_extra_info_answer(self):
        data = {"prompt": "q", "extra_info": {"answer": "42"}}
        result = _convert_verl_to_genqa(data)
        assert result["response"] == "42"

    def test_reward_model_ground_truth_fallback(self):
        data = {"prompt": "q", "reward_model": {"ground_truth": "correct"}}
        result = _convert_verl_to_genqa(data)
        assert result["response"] == "correct"

    def test_preserves_metadata_fields(self):
        data = {
            "prompt": "q",
            "id": "abc",
            "data_source": "gsm8k",
            "ability": "math",
            "difficulty": "easy",
        }
        result = _convert_verl_to_genqa(data)
        assert result["metadata"]["id"] == "abc"
        assert result["metadata"]["data_source"] == "gsm8k"
        assert result["metadata"]["ability"] == "math"
        assert result["metadata"]["difficulty"] == "easy"


class TestConvertToGenqa:
    def test_genqa_passthrough(self):
        data = {"query": "q", "response": "r", "system": "s", "metadata": {}}
        result = convert_to_genqa(data, DatasetFormat.GENQA)
        assert result is data

    def test_openai_dispatch(self):
        data = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }
        result = convert_to_genqa(data, DatasetFormat.OPENAI_CHAT)
        assert result["query"] == "hi"

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            convert_to_genqa({}, DatasetFormat("converse"))


class TestConvertFromGenqa:
    def test_genqa_passthrough(self):
        data = {"query": "q", "response": "r"}
        result = convert_from_genqa(data, DatasetFormat.GENQA)
        assert result is data

    def test_unimplemented_raises(self):
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            convert_from_genqa({"query": "q"}, DatasetFormat.OPENAI_CHAT)


class TestDatasetTransformationDetectFormat:
    @patch("sagemaker.ai_registry.dataset_transformation.DatasetFormatDetector")
    def test_detect_known_format(self, mock_detector_cls):
        mock_detector_cls.detect_format.return_value = "openai_chat"
        result = DatasetTransformation.detect_format("/path/to/file.jsonl")
        assert result == DatasetFormat.OPENAI_CHAT

    @patch("sagemaker.ai_registry.dataset_transformation.DatasetFormatDetector")
    def test_detect_none_raises(self, mock_detector_cls):
        mock_detector_cls.detect_format.return_value = None
        with pytest.raises(ValueError, match="Unable to detect dataset format"):
            DatasetTransformation.detect_format("/path/to/file.jsonl")

    @patch("sagemaker.ai_registry.dataset_transformation.DatasetFormatDetector")
    def test_detect_unsupported_format_raises(self, mock_detector_cls):
        mock_detector_cls.detect_format.return_value = "rft"
        with pytest.raises(ValueError, match="not a supported transformation format"):
            DatasetTransformation.detect_format("/path/to/file.jsonl")


class TestDatasetTransformationTransformFile:
    def test_transform_file_to_genqa(self):
        records = [
            {
                "messages": [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            },
        ]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            src_path = f.name

        try:
            out_path = DatasetTransformation.transform_file(
                file_path=src_path,
                source_format=DatasetFormat.OPENAI_CHAT,
                target_format=DatasetFormat.GENQA,
            )
            try:
                with open(out_path, "r") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                assert len(lines) == 2
                assert lines[0]["query"] == "q1"
                assert lines[1]["query"] == "q2"
            finally:
                os.unlink(out_path)
        finally:
            os.unlink(src_path)

    def test_transform_file_skips_empty_lines(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"messages": [{"role": "user", "content": "q"}]}) + "\n")
            f.write("\n")
            f.write("   \n")
            src_path = f.name

        try:
            out_path = DatasetTransformation.transform_file(
                file_path=src_path,
                source_format=DatasetFormat.OPENAI_CHAT,
                target_format=DatasetFormat.GENQA,
            )
            try:
                with open(out_path, "r") as f:
                    lines = [json.loads(line) for line in f if line.strip()]
                assert len(lines) == 1
            finally:
                os.unlink(out_path)
        finally:
            os.unlink(src_path)
