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
"""``detect_dataset_format`` names the format of a local or S3 dataset sample.

Before this change ``sagemaker.ai_registry`` exported nothing and
``DatasetFormatDetector.validate_dataset`` returned only ``True`` or ``False`` for a local
path: it stopped at the first line, named no format, read no ``s3://`` source, and did not
know the Nova RLVR, PPO, or CPT shapes. Two SageMaker agent skill scripts each carried their
own detector with 9 and 11 format names.
"""

from __future__ import absolute_import

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry import (
    FormatDetectionOptions,
    FormatDetectionResult,
    detect_dataset_format,
)
from sagemaker.ai_registry.dataset_format_detector import (
    FORMAT_MATCHERS,
    SKILL_FORMAT_ALIASES,
    DatasetFormatDetector,
)

DATA_DIR = Path(__file__).parent / "data" / "format_detection"
READ_BYTES = "sagemaker.ai_registry.dataset_format_detector.S3Downloader.read_bytes"

EXPECTED_FORMATS = {
    "valid_nova_dpo.jsonl": "dpo",
    "valid_gpt_oss_sft.jsonl": "openai_chat",
    "valid_open_weights_sft.jsonl": "hf_prompt_completion",
    "valid_sagemaker_eval.jsonl": "genqa",
    "valid_verl.jsonl": "verl",
    "valid_nova_rlvr.jsonl": "nova_rlvr",
}

NOVA_SFT_LINE = json.dumps(
    {
        "messages": [
            {"role": "user", "content": [{"text": "What is the capital of France?"}]},
            {"role": "assistant", "content": [{"text": "Paris."}]},
        ]
    }
)
GPT_OSS_LINE = json.dumps(
    {
        "messages": [
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "A field of AI."},
        ]
    }
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


@pytest.mark.parametrize("fixture_name, expected", sorted(EXPECTED_FORMATS.items()))
def test_local_fixture_resolves_to_its_format(fixture_name, expected):
    result = detect_dataset_format(str(DATA_DIR / fixture_name))

    assert result.format == expected
    assert result.confidence == 1.0
    assert result.lines_sampled == len((DATA_DIR / fixture_name).read_text().splitlines())
    assert result.line_errors == []
    assert result.candidates[expected] == 1.0


@pytest.mark.parametrize("fixture_name, expected", sorted(EXPECTED_FORMATS.items()))
def test_strict_mode_accepts_every_fixture(fixture_name, expected):
    result = detect_dataset_format(
        str(DATA_DIR / fixture_name), FormatDetectionOptions(strict=True)
    )

    assert result.format == expected


def test_s3_source_reads_one_byte_range_through_s3_downloader():
    payload = (NOVA_SFT_LINE + "\n") * 3 + NOVA_SFT_LINE[:20]
    session = Mock(name="sagemaker_session")
    with patch(READ_BYTES, return_value=payload.encode("utf-8")) as read_bytes:
        result = detect_dataset_format(
            "s3://bucket/train/data.jsonl",
            FormatDetectionOptions(sample_size_bytes=len(payload), sagemaker_session=session),
        )

    read_bytes.assert_called_once_with(
        "s3://bucket/train/data.jsonl", session, byte_range=(0, len(payload) - 1)
    )
    assert result.format == "converse"
    assert result.lines_sampled == 3
    assert result.line_errors == []


def test_s3_sample_drops_the_last_partial_line_only_when_the_sample_is_full():
    payload = (GPT_OSS_LINE + "\n") * 2 + GPT_OSS_LINE
    with patch(READ_BYTES, return_value=payload.encode("utf-8")):
        full_sample = detect_dataset_format(
            "s3://bucket/data.jsonl", FormatDetectionOptions(sample_size_bytes=len(payload))
        )
        short_file = detect_dataset_format(
            "s3://bucket/data.jsonl", FormatDetectionOptions(sample_size_bytes=len(payload) + 1)
        )

    assert full_sample.lines_sampled == 2
    assert short_file.lines_sampled == 3
    assert short_file.format == "openai_chat"


def test_s3_zero_byte_object_returns_no_format():
    error = ClientError({"Error": {"Code": "InvalidRange", "Message": "x"}}, "GetObject")
    with patch(READ_BYTES, side_effect=error):
        result = detect_dataset_format("s3://bucket/empty.jsonl")

    assert result == FormatDetectionResult(
        format=None, confidence=0.0, lines_sampled=0, line_errors=[], candidates={}
    )


def test_other_s3_errors_propagate():
    error = ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
    with patch(READ_BYTES, side_effect=error):
        with pytest.raises(ClientError):
            detect_dataset_format("s3://bucket/missing.jsonl")


def test_strict_mode_reports_the_first_mismatching_line(tmp_path):
    path = _write(tmp_path, "mixed.jsonl", "\n".join([NOVA_SFT_LINE, NOVA_SFT_LINE, GPT_OSS_LINE]))

    relaxed = detect_dataset_format(path)
    strict = detect_dataset_format(path, FormatDetectionOptions(strict=True))

    assert relaxed.format == "converse"
    assert relaxed.confidence == pytest.approx(2 / 3)
    # The dpo schema also accepts a plain converse line. The tie resolves to converse by order.
    assert relaxed.candidates == {
        "converse": pytest.approx(2 / 3),
        "dpo": pytest.approx(2 / 3),
        "openai_chat": pytest.approx(1 / 3),
    }
    assert strict.format is None
    assert strict.confidence == 0.0
    assert strict.lines_sampled == 3
    assert len(strict.line_errors) == 1
    assert strict.line_errors[0].startswith("3: $.messages[")
    assert strict.line_errors[0].endswith("is not of type 'array'")


def test_relaxed_mode_rejects_a_candidate_at_half_or_below(tmp_path):
    path = _write(tmp_path, "half.jsonl", "\n".join([NOVA_SFT_LINE, GPT_OSS_LINE]))

    result = detect_dataset_format(path)

    assert result.format is None
    assert result.candidates == {"converse": 0.5, "dpo": 0.5, "openai_chat": 0.5}
    assert len(result.line_errors) == 1
    assert result.line_errors[0].startswith("2: $.messages[")
    assert result.line_errors[0].endswith("is not of type 'array'")


def test_invalid_json_lines_count_as_errors_and_lower_confidence(tmp_path):
    path = _write(tmp_path, "broken.jsonl", "\n".join([GPT_OSS_LINE, "{not json", GPT_OSS_LINE]))

    result = detect_dataset_format(path)

    assert result.format == "openai_chat"
    assert result.confidence == pytest.approx(2 / 3)
    assert result.lines_sampled == 3
    assert len(result.line_errors) == 1
    assert result.line_errors[0].startswith("2: invalid JSON: ")


def test_empty_file_returns_no_format(tmp_path):
    path = _write(tmp_path, "empty.jsonl", "")

    result = detect_dataset_format(path)

    assert result == FormatDetectionResult(
        format=None, confidence=0.0, lines_sampled=0, line_errors=[], candidates={}
    )


def test_unknown_shape_returns_no_format_and_no_candidates(tmp_path):
    path = _write(tmp_path, "unknown.jsonl", json.dumps({"foo": 1}) + "\n")

    result = detect_dataset_format(path)

    assert result.format is None
    assert result.candidates == {}
    assert result.line_errors == []


def test_local_sample_size_limits_the_lines_read(tmp_path):
    path = _write(tmp_path, "many.jsonl", (GPT_OSS_LINE + "\n") * 10)

    result = detect_dataset_format(
        path, FormatDetectionOptions(sample_size_bytes=len(GPT_OSS_LINE) * 3 + 2)
    )

    assert result.lines_sampled == 2
    assert result.format == "openai_chat"


def test_nova_cpt_text_only_lines(tmp_path):
    path = _write(tmp_path, "cpt.jsonl", json.dumps({"text": "a paragraph"}) + "\n")
    mixed_path = _write(
        tmp_path, "cpt_mixed.jsonl", json.dumps({"text": "a", "prompt": "b"}) + "\n"
    )

    assert detect_dataset_format(path).format == "nova_cpt"
    assert "nova_cpt" not in detect_dataset_format(mixed_path).candidates


def test_nova_ppo_lines(tmp_path):
    record = {"turns": ["User: hi", "Bot: hello"], "turns_to_mask": [0], "reward_category": "math"}
    path = _write(tmp_path, "ppo.jsonl", json.dumps(record) + "\n")
    bad_record = dict(record, reward_category="unknown")
    bad_path = _write(tmp_path, "ppo_bad.jsonl", json.dumps(bad_record) + "\n")

    assert detect_dataset_format(path).format == "nova_ppo"
    assert "nova_ppo" not in detect_dataset_format(bad_path).candidates


def test_parquet_csv_and_json_extensions(tmp_path):
    parquet_path = tmp_path / "data.parquet"
    parquet_path.write_bytes(b"PAR1" + b"\x00" * 16)
    csv_path = _write(tmp_path, "data.csv", "prompt,completion\na,b\n")
    json_path = _write(tmp_path, "data.json", json.dumps([{"prompt": "a", "completion": "b"}]))
    bad_parquet_path = tmp_path / "bad.parquet"
    bad_parquet_path.write_bytes(b"notparquet")

    assert detect_dataset_format(str(parquet_path)).format == "parquet"
    assert detect_dataset_format(csv_path).format == "csv"
    assert detect_dataset_format(json_path).format == "json"
    bad = detect_dataset_format(str(bad_parquet_path))
    assert bad.format is None
    assert bad.line_errors == ["1: missing the Parquet magic bytes PAR1"]


def test_extension_of_an_s3_key_selects_the_whole_file_matcher():
    with patch(READ_BYTES, return_value=b"PAR1" + b"\x00" * 16):
        result = detect_dataset_format("s3://bucket/dir/part-0.parquet")

    assert result.format == "parquet"
    assert result.candidates == {"parquet": 1.0}


def test_sample_size_below_one_is_rejected(tmp_path):
    path = _write(tmp_path, "x.jsonl", GPT_OSS_LINE)

    with pytest.raises(ValueError):
        detect_dataset_format(path, FormatDetectionOptions(sample_size_bytes=0))


def test_matchers_cover_the_sdk_schemas_and_the_skill_aliases():
    schema_names = {path.stem for path in DatasetFormatDetector.SCHEMA_DIR.glob("*.json")}

    assert schema_names <= set(FORMAT_MATCHERS)
    assert set(SKILL_FORMAT_ALIASES.values()) <= set(FORMAT_MATCHERS)
    assert {"nova_rlvr", "nova_ppo", "nova_cpt"} <= set(FORMAT_MATCHERS) - schema_names
