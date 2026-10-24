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
"""Utilities for detecting the format of a customization dataset.

``DatasetFormatDetector`` answers whether a local file matches one known format.
``detect_dataset_format`` samples the first bytes of a local file or an ``s3://`` object and
reports the best matching format, its confidence, the other candidates, and the lines that
failed.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple
from urllib.parse import urlparse

from botocore.exceptions import ClientError
from jsonschema.exceptions import best_match
from jsonschema.validators import validator_for

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.s3.client import S3Downloader, is_s3_url

DEFAULT_SAMPLE_SIZE_BYTES = 1_048_576
RELAXED_MIN_RATIO = 0.5

# Names the two agent-skill detectors use for shapes the SDK schemas already describe.
# Lower-case names: SageMakerAIAgentSkills dataset-evaluation/scripts/format_detector.py.
# Capitalized names: the verse dataset-quality/scripts/format_detector.py, aligned with the
# Hub DatasetFileFormat enum. The value is the key in ``FORMAT_MATCHERS``.
SKILL_FORMAT_ALIASES: Dict[str, str] = {
    "nova_sft": "converse",
    "Bedrock_Converse": "converse",
    "nova_dpo": "dpo",
    "nova_rlvr": "nova_rlvr",
    "Nova_RLVR": "nova_rlvr",
    "Nova_PPO": "nova_ppo",
    "Nova_CPT": "nova_cpt",
    "gpt_oss_sft": "openai_chat",
    "OpenAI_Messages": "openai_chat",
    "gpt_oss_dpo": "hf_preference",
    "open_weights_dpo": "hf_preference",
    "HF_Preference": "hf_preference",
    "open_weights_sft": "hf_prompt_completion",
    "open_weights_sft_conv": "hf_prompt_completion",
    "HF_Prompt_Completion": "hf_prompt_completion",
    "sagemaker_eval": "genqa",
    "SageMaker_Eval": "genqa",
    "verl_legacy": "verl",
    "Verl": "verl",
}

NOVA_PPO_REWARD_CATEGORIES = frozenset({"default", "math", "coding", "if", "rag", "rai"})

# A bare ``text`` column is a weak signal, so a field of another format rules Nova CPT out.
# The list matches the verse skill and the Data Quality service.
NON_CPT_FIELDS = frozenset(
    {
        "prompt",
        "completion",
        "messages",
        "query",
        "chosen",
        "rejected",
        "label",
        "label_text",
        "labels",
        "intent",
        "category",
        "answer",
    }
)

# ``None`` means the record matches. A string names the first mismatch.
MismatchReason = Optional[str]
Matcher = Callable[[Any], MismatchReason]
LineError = Tuple[int, str]


@dataclass(frozen=True)
class FormatDetectionResult:
    """Outcome of ``detect_dataset_format``.

    Attributes:
        format: The detected format name, ``None`` when nothing matched.
        confidence: Matched lines divided by sampled lines for ``format``, 0.0 to 1.0.
        lines_sampled: Non-empty complete lines in the sample.
        line_errors: ``"<line number>: <reason>"`` for each line that failed to parse or
            failed ``format``. Without a detected format the reasons refer to the best candidate.
        candidates: Every format that matched at least one line, with its ratio.
    """

    format: Optional[str]
    confidence: float
    lines_sampled: int
    line_errors: List[str]
    candidates: Dict[str, float]


@dataclass
class FormatDetectionOptions:
    """Options for ``detect_dataset_format``.

    Attributes:
        sample_size_bytes: Bytes read from the start of the source. Defaults to 1 MiB.
        strict: When ``True`` every sampled line must match the format. When ``False`` the
            best candidate above a 0.5 ratio wins.
        sagemaker_session: Session for ``s3://`` sources. ``None`` creates a default session.
    """

    sample_size_bytes: int = DEFAULT_SAMPLE_SIZE_BYTES
    strict: bool = False
    sagemaker_session: Optional[Session] = None


class _Sample(NamedTuple):
    data: bytes
    complete: bool


class DatasetFormatDetector:
    """Utility class for detecting dataset formats."""

    # Schema directory
    SCHEMA_DIR = Path(__file__).parent / "schemas"

    @staticmethod
    def _load_schema(format_name: str) -> Dict[str, Any]:
        """Load JSON schema for a format."""
        schema_path = DatasetFormatDetector.SCHEMA_DIR / f"{format_name}.json"
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def validate_dataset(file_path: str) -> bool:
        """Validate if the dataset adheres to any known format.

        Args:
            file_path: Path to the JSONL, Parquet, JSON, or CSV file

        Returns:
            True if dataset is valid according to any known format, False otherwise
        """
        # Parquet files are binary — validate by reading the header
        if file_path.endswith(".parquet"):
            return DatasetFormatDetector._validate_parquet(file_path)

        # CSV files — validate by checking header row exists
        if file_path.endswith(".csv"):
            return DatasetFormatDetector._validate_csv(file_path)

        # JSON array files
        if file_path.endswith(".json"):
            return DatasetFormatDetector._validate_json(file_path)

        import jsonschema

        # Schema-based formats (JSONL)
        schema_formats = [
            "dpo",
            "converse",
            "hf_preference",
            "hf_prompt_completion",
            "verl",
            "openai_chat",
            "genqa",
        ]

        try:
            with open(file_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)

                        # Try schema validation first
                        for format_name in schema_formats:
                            schema = DatasetFormatDetector._load_schema(format_name)
                            if schema:
                                try:
                                    jsonschema.validate(instance=data, schema=schema)
                                    return True
                                except jsonschema.exceptions.ValidationError:
                                    continue

                        # Check for RFT-style format (messages + additional fields)
                        if DatasetFormatDetector._is_rft_format(data):
                            return True
                        break
            return False
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False

    @staticmethod
    def _validate_parquet(file_path: str) -> bool:
        """Validate that a file is a valid Parquet file by checking its magic bytes."""
        try:
            with open(file_path, "rb") as f:
                magic = f.read(4)
                return magic == b"PAR1"
        except (FileNotFoundError, IOError):
            return False

    @staticmethod
    def _validate_csv(file_path: str) -> bool:
        """Validate that a file is a valid CSV with a header row."""
        import csv

        try:
            with open(file_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                return header is not None and len(header) > 0
        except (FileNotFoundError, IOError):
            return False

    @staticmethod
    def _validate_json(file_path: str) -> bool:
        """Validate that a file is a valid JSON array of objects."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return isinstance(data, list) and len(data) > 0
        except (json.JSONDecodeError, FileNotFoundError, IOError):
            return False

    @staticmethod
    def _is_rft_format(data: Dict[str, Any]) -> bool:
        """Check if data matches RFT format pattern."""
        if not isinstance(data, dict) or "messages" not in data:
            return False

        messages = data["messages"]
        if not isinstance(messages, list) or not messages:
            return False

        # Check message structure
        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                return False

        return True


def _json_path(error) -> str:
    parts = [f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path]
    return "$" + "".join(parts)


def _schema_matcher(format_name: str) -> Matcher:
    """Return a matcher that validates a record against the SDK JSON schema of ``format_name``."""
    schema = DatasetFormatDetector._load_schema(format_name)
    if not schema:
        raise ValueError(
            f"No JSON schema named {format_name!r} in {DatasetFormatDetector.SCHEMA_DIR}"
        )
    validator = validator_for(schema)(schema)

    def match(record: Any) -> MismatchReason:
        error = best_match(validator.iter_errors(record))
        if error is None:
            return None
        return f"{_json_path(error)}: {error.message}"

    return match


_match_openai_chat = _schema_matcher("openai_chat")


def _not_an_object(record: Any) -> str:
    return f"$: {type(record).__name__} is not an object"


def _match_nova_rlvr(record: Any) -> MismatchReason:
    """Nova RLVR: OpenAI-style messages plus a ``reference_answer`` object."""
    if not isinstance(record, dict):
        return _not_an_object(record)
    if not isinstance(record.get("reference_answer"), dict):
        return "$.reference_answer: missing or not an object"
    return _match_openai_chat(record)


def _is_nova_ppo_turn(turn: Any) -> bool:
    return isinstance(turn, str) and turn.startswith(("User:", "Bot:"))


def _match_nova_ppo(record: Any) -> MismatchReason:
    """Nova PPO: ``turns``, ``turns_to_mask``, and a known ``reward_category``."""
    if not isinstance(record, dict):
        return _not_an_object(record)
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns or not all(map(_is_nova_ppo_turn, turns)):
        return "$.turns: expected a non-empty list of strings that start with 'User:' or 'Bot:'"
    turns_to_mask = record.get("turns_to_mask")
    if not isinstance(turns_to_mask, list) or not all(isinstance(i, int) for i in turns_to_mask):
        return "$.turns_to_mask: expected a list of integers"
    if record.get("reward_category") not in NOVA_PPO_REWARD_CATEGORIES:
        return f"$.reward_category: expected one of {sorted(NOVA_PPO_REWARD_CATEGORIES)}"
    return None


def _match_nova_cpt(record: Any) -> MismatchReason:
    """Nova CPT: a ``text`` string and no field of another format."""
    if not isinstance(record, dict):
        return _not_an_object(record)
    if not isinstance(record.get("text"), str):
        return "$.text: missing or not a string"
    other_fields = sorted(NON_CPT_FIELDS.intersection(record))
    if other_fields:
        return f"$: fields {other_fields} belong to another format"
    return None


# Priority order: the most specific shape first. A tie in match ratio resolves to the earlier
# entry, so an SFT converse line (which the dpo schema also accepts) reports ``converse`` and a
# prompt/completion line (which the verl schema also accepts) reports ``hf_prompt_completion``.
FORMAT_MATCHERS: Dict[str, Matcher] = {
    "nova_rlvr": _match_nova_rlvr,
    "nova_ppo": _match_nova_ppo,
    "converse": _schema_matcher("converse"),
    "dpo": _schema_matcher("dpo"),
    "openai_chat": _match_openai_chat,
    "hf_preference": _schema_matcher("hf_preference"),
    "genqa": _schema_matcher("genqa"),
    "hf_prompt_completion": _schema_matcher("hf_prompt_completion"),
    "nova_cpt": _match_nova_cpt,
    "verl": _schema_matcher("verl"),
}


def _match_parquet(sample: _Sample) -> MismatchReason:
    if sample.data[:4] == b"PAR1":
        return None
    return "missing the Parquet magic bytes PAR1"


def _match_csv(sample: _Sample) -> MismatchReason:
    first_line = sample.data.split(b"\n", 1)[0].decode("utf-8", errors="replace")
    header = next(csv.reader([first_line]), None)
    if header:
        return None
    return "missing a CSV header row"


def _match_json(sample: _Sample) -> MismatchReason:
    text = sample.data.decode("utf-8", errors="replace")
    if not sample.complete:
        return None if text.lstrip().startswith("[") else "expected a JSON array"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as error:
        return f"invalid JSON: {error}"
    if isinstance(parsed, list) and parsed:
        return None
    return "expected a non-empty JSON array"


# Formats that a file extension names. The matcher reads the sampled bytes, not JSONL lines.
EXTENSION_MATCHERS: Dict[str, Callable[[_Sample], MismatchReason]] = {
    ".parquet": _match_parquet,
    ".csv": _match_csv,
    ".json": _match_json,
}


def _read_s3_sample(source: str, options: FormatDetectionOptions) -> bytes:
    byte_range = (0, options.sample_size_bytes - 1)
    try:
        return S3Downloader.read_bytes(source, options.sagemaker_session, byte_range=byte_range)
    except ClientError as error:
        # GetObject with a Range header fails with 416 InvalidRange on a zero-byte object.
        if error.response.get("Error", {}).get("Code") == "InvalidRange":
            return b""
        raise


def _read_sample(source: str, options: FormatDetectionOptions) -> _Sample:
    if options.sample_size_bytes < 1:
        raise ValueError(f"sample_size_bytes must be at least 1, got {options.sample_size_bytes}.")
    if is_s3_url(source):
        data = _read_s3_sample(source, options)
    else:
        with open(source, "rb") as handle:
            data = handle.read(options.sample_size_bytes)
    return _Sample(data=data, complete=len(data) < options.sample_size_bytes)


def _complete_lines(sample: _Sample) -> List[Tuple[int, str]]:
    """Return ``(line number, text)`` for each non-empty complete line of the sample."""
    data = sample.data
    if not sample.complete:
        # Trim at the last newline byte before decoding. A byte range can cut a multibyte
        # character, and 0x0A is never a UTF-8 continuation byte.
        data = data[: data.rfind(b"\n") + 1]
    text = data.decode("utf-8", errors="replace")
    return [(number, line) for number, line in enumerate(text.split("\n"), start=1) if line.strip()]


def _parse_lines(lines: List[Tuple[int, str]]) -> Tuple[List[LineError], List[Tuple[int, Any]]]:
    parse_errors: List[LineError] = []
    records: List[Tuple[int, Any]] = []
    for number, line in lines:
        try:
            records.append((number, json.loads(line)))
        except json.JSONDecodeError as error:
            parse_errors.append((number, f"invalid JSON: {error}"))
    return parse_errors, records


def _mismatches(matcher: Matcher, records: List[Tuple[int, Any]]) -> List[LineError]:
    mismatches: List[LineError] = []
    for number, record in records:
        reason = matcher(record)
        if reason is not None:
            mismatches.append((number, reason))
    return mismatches


def _select_format(best_name: Optional[str], ratio: float, strict: bool) -> Optional[str]:
    if best_name is None:
        return None
    if strict:
        return best_name if ratio >= 1.0 else None
    return best_name if ratio > RELAXED_MIN_RATIO else None


def _extension_result(format_name: str, reason: MismatchReason) -> FormatDetectionResult:
    if reason is None:
        return FormatDetectionResult(
            format=format_name,
            confidence=1.0,
            lines_sampled=0,
            line_errors=[],
            candidates={format_name: 1.0},
        )
    return FormatDetectionResult(
        format=None, confidence=0.0, lines_sampled=0, line_errors=[f"1: {reason}"], candidates={}
    )


def _detect_jsonl(lines: List[Tuple[int, str]], strict: bool) -> FormatDetectionResult:
    if not lines:
        return FormatDetectionResult(
            format=None, confidence=0.0, lines_sampled=0, line_errors=[], candidates={}
        )
    parse_errors, records = _parse_lines(lines)
    mismatches_by_format = {
        name: _mismatches(matcher, records) for name, matcher in FORMAT_MATCHERS.items()
    }
    candidates = {
        name: (len(records) - len(mismatches)) / len(lines)
        for name, mismatches in mismatches_by_format.items()
        if len(mismatches) < len(records)
    }
    best_name = max(candidates, key=candidates.get, default=None)
    detected = _select_format(best_name, candidates.get(best_name, 0.0), strict)
    reported = detected or best_name
    line_errors = list(parse_errors)
    if reported is not None:
        line_errors.extend(mismatches_by_format[reported])
    return FormatDetectionResult(
        format=detected,
        confidence=candidates[detected] if detected else 0.0,
        lines_sampled=len(lines),
        line_errors=[f"{number}: {reason}" for number, reason in sorted(line_errors)],
        candidates=candidates,
    )


def detect_dataset_format(
    source: str, options: Optional[FormatDetectionOptions] = None
) -> FormatDetectionResult:
    """Detect the format of a dataset from the first bytes of a local file or an S3 object.

    An ``s3://`` source is read with one ranged ``GetObject`` through
    ``S3Downloader.read_bytes``. A ``.parquet``, ``.csv``, or ``.json`` extension selects a
    whole-file matcher. Every other source is JSONL: each sampled line is matched against
    ``FORMAT_MATCHERS``, the SDK schema names plus the Nova shapes the agent skills detect.

    Args:
        source: A local path or an ``s3://bucket/key`` URI.
        options: Sample size, strict or relaxed selection, and the session for S3 reads.

    Returns:
        FormatDetectionResult: The detected format, its confidence, the sampled line count,
            the lines that failed, and every candidate with its match ratio.
    """
    options = options or FormatDetectionOptions()
    sample = _read_sample(source, options)
    path = urlparse(source).path if is_s3_url(source) else source
    extension = os.path.splitext(path)[1].lower()
    extension_matcher = EXTENSION_MATCHERS.get(extension)
    if extension_matcher is not None:
        return _extension_result(extension.lstrip("."), extension_matcher(sample))
    return _detect_jsonl(_complete_lines(sample), options.strict)
