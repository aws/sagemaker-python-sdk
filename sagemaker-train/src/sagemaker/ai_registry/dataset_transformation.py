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

"""Hub-and-spoke dataset format transformation using genqa as the canonical intermediate."""
from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class DatasetFormat(str, Enum):
    """Supported dataset formats for transformation."""

    GENQA = "genqa"
    CONVERSE = "converse"
    DPO = "dpo"
    HF_PREFERENCE = "hf_preference"
    HF_PROMPT_COMPLETION = "hf_prompt_completion"
    OPENAI_CHAT = "openai_chat"
    VERL = "verl"


# ------------------------------------------------------------------ #
#  Private helpers
# ------------------------------------------------------------------ #


def _extract_text_from_messages(messages: Any, target_role: str = "user") -> str:
    """Extract text content from messages array or return string as-is.

    Uses the LAST occurrence of the target role to match training behavior.

    Args:
        messages: Either a string or array of message objects.
        target_role: The role to extract (e.g., 'user', 'assistant', 'system').

    Returns:
        Extracted text content (last occurrence of target_role).
    """
    if isinstance(messages, str):
        return messages
    elif isinstance(messages, list):
        last_content = ""
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == target_role:
                    last_content = content
        return last_content
    return ""


def _extract_all_from_messages(messages: Any) -> Tuple[str, str, str]:
    """Extract system, query, and response from messages array or string.

    Uses the LAST occurrence of each role to match training behavior.

    Args:
        messages: Either a string or array of message objects.

    Returns:
        Tuple of (system, query, response) - last occurrence of each role.
    """
    system = ""
    query = ""
    response = ""

    if isinstance(messages, str):
        query = messages
    elif isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system = content
                elif role == "user":
                    query = content
                elif role == "assistant":
                    response = content

    return system, query, response


# ------------------------------------------------------------------ #
#  Inbound converters  (X → genqa)
# ------------------------------------------------------------------ #


def _convert_openai_to_genqa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert OpenAI Chat format to GenQA."""
    messages = data.get("messages", [])
    system = ""
    query = ""
    response = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            system = content
        elif role == "user":
            query = content
        elif role == "assistant":
            response = content

    # Only use response if last message is from assistant
    if messages and isinstance(messages, list):
        last_message = messages[-1]
        if isinstance(last_message, dict):
            if last_message.get("role") != "assistant":
                response = ""

    if not query:
        logger.warning(
            "OpenAI Chat conversion resulted in empty query. "
            "Check that messages contain a 'user' role."
        )

    return {
        "query": query,
        "response": response,
        "system": system,
        "metadata": {},
    }


def _convert_converse_to_genqa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert AWS Converse format to GenQA.

    Converse uses structured content arrays rather than plain strings:
    - system: [{"text": "..."}]
    - messages: [{"role": "user", "content": [{"text": "..."}]}, ...]
    """
    # Extract system prompt from array of {text: "..."} objects
    system_blocks = data.get("system", [])
    system = " ".join(block.get("text", "") for block in system_blocks if isinstance(block, dict))

    query = ""
    response = ""

    for msg in data.get("messages", []):
        role = msg.get("role", "")
        content_blocks = msg.get("content", [])
        # Concatenate all text blocks within a single message
        text = " ".join(
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and "text" in block
        )
        if role == "user":
            query = text
        elif role == "assistant":
            response = text

    # Only keep response if last message is from assistant
    messages = data.get("messages", [])
    if messages and isinstance(messages, list):
        last_message = messages[-1]
        if isinstance(last_message, dict) and last_message.get("role") != "assistant":
            response = ""

    if not query:
        logger.warning(
            "Converse conversion resulted in empty query. "
            "Check that messages contain a 'user' role with text content."
        )

    return {
        "query": query,
        "response": response,
        "system": system,
        "metadata": {},
    }


def _convert_hf_to_genqa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HuggingFace Prompt-Completion format to GenQA.

    Supports both standard format (string) and conversational format (messages array):
    - Standard: {"prompt": "text", "completion": "text"}
    - Conversational: {"prompt": [{"role": "user", "content": "..."}],
                       "completion": [{"role": "assistant", "content": "..."}]}
    """
    prompt_field = data.get("prompt", "")
    completion_field = data.get("completion", "")

    system, query, _ = _extract_all_from_messages(prompt_field)

    response = _extract_text_from_messages(completion_field, target_role="assistant")
    if not response and isinstance(completion_field, str):
        response = completion_field

    if not query:
        logger.warning(
            "HuggingFace Prompt-Completion conversion resulted in empty query. "
            "Check that data contains 'prompt' field with text or 'user' role message."
        )

    return {
        "query": query,
        "response": response,
        "system": system,
        "metadata": {},
    }


def _convert_hf_preference_to_genqa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert HuggingFace Preference format (DPO) to GenQA.

    For evaluation purposes, we use the 'chosen' response as the reference.
    The 'input' or 'prompt' field is used as the query.

    Supports both standard format (string) and conversational format (messages array).
    """
    prompt_field = data.get("input", data.get("prompt", ""))
    system, query, _ = _extract_all_from_messages(prompt_field)

    if not query:
        logger.warning(
            "HuggingFace Preference conversion resulted in empty query. "
            "Check that data contains 'input' or 'prompt' field with text or 'user' role message."
        )

    # Extract chosen response
    chosen_field = data.get("chosen", "")
    response = _extract_text_from_messages(chosen_field, target_role="assistant")
    if not response and isinstance(chosen_field, str):
        response = chosen_field

    # Extract rejected response for metadata
    rejected_field = data.get("rejected", "")
    rejected_text = _extract_text_from_messages(rejected_field, target_role="assistant")
    if not rejected_text and isinstance(rejected_field, str):
        rejected_text = rejected_field

    metadata: Dict[str, Any] = {
        "chosen": response,
        "rejected": rejected_text,
    }
    if "id" in data:
        metadata["id"] = data["id"]
    if "attributes" in data:
        metadata["attributes"] = data["attributes"]
    if "difficulty" in data:
        metadata["difficulty"] = data["difficulty"]

    return {
        "query": query,
        "response": response,
        "system": system,
        "metadata": metadata,
    }


def _convert_verl_to_genqa(data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert VERL format to GenQA.

    VERL format is primarily used for RL training. The prompt can be either
    a string (legacy format) or an array of conversation messages.
    """
    prompt = data.get("prompt", "")
    query = ""
    system = ""

    if isinstance(prompt, str):
        query = prompt
    elif isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    query = content
                elif role == "system":
                    system = content

    if not query:
        logger.warning(
            "VERL conversion resulted in empty query. "
            "Check that 'prompt' field contains a string or array with 'user' role message."
        )

    # VERL typically doesn't have a direct response; check extra_info then reward_model
    response = ""
    extra_info = data.get("extra_info", {})
    if isinstance(extra_info, dict) and "answer" in extra_info:
        response = extra_info["answer"]
    else:
        reward_model = data.get("reward_model", {})
        if isinstance(reward_model, dict) and "ground_truth" in reward_model:
            response = reward_model["ground_truth"]

    metadata: Dict[str, Any] = {}
    for key in (
        "id",
        "data_source",
        "ability",
        "reward_model",
        "extra_info",
        "attributes",
        "difficulty",
    ):
        if key in data:
            metadata[key] = data[key]

    return {
        "query": query,
        "response": response,
        "system": system,
        "metadata": metadata,
    }


# ------------------------------------------------------------------ #
#  Dispatch: convert any supported format → genqa
# ------------------------------------------------------------------ #

_INBOUND_CONVERTERS = {
    DatasetFormat.CONVERSE: _convert_converse_to_genqa,
    DatasetFormat.OPENAI_CHAT: _convert_openai_to_genqa,
    DatasetFormat.HF_PROMPT_COMPLETION: _convert_hf_to_genqa,
    DatasetFormat.HF_PREFERENCE: _convert_hf_preference_to_genqa,
    DatasetFormat.VERL: _convert_verl_to_genqa,
}


def convert_to_genqa(data: Dict[str, Any], source_format: DatasetFormat) -> Dict[str, Any]:
    """Convert a single record to genqa format.

    Args:
        data: Input record dictionary.
        source_format: The format of the input record.

    Returns:
        Record in genqa format.

    Raises:
        ValueError: If the source format is unsupported.
    """
    if source_format == DatasetFormat.GENQA:
        return data

    converter = _INBOUND_CONVERTERS.get(source_format)
    if converter is None:
        raise ValueError(
            f"Unsupported dataset format '{source_format}'. "
            f"Supported inbound formats: {list(_INBOUND_CONVERTERS.keys())}"
        )

    logger.debug("Converting from %s to genqa", source_format.value)
    return converter(data)


# ------------------------------------------------------------------ #
#  Outbound converters  (genqa → X) — stubs
# ------------------------------------------------------------------ #

_OUTBOUND_CONVERTERS: Dict[DatasetFormat, Any] = {}


def convert_from_genqa(data: Dict[str, Any], target_format: DatasetFormat) -> Dict[str, Any]:
    """Convert a genqa record to the target format.

    Args:
        data: Input record in genqa format.
        target_format: The desired output format.

    Returns:
        Record in the target format.

    Raises:
        ValueError: If the target format is unsupported.
        NotImplementedError: If the outbound converter is not yet implemented.
    """
    if target_format == DatasetFormat.GENQA:
        return data

    converter = _OUTBOUND_CONVERTERS.get(target_format)
    if converter is None:
        raise NotImplementedError(
            f"Outbound conversion genqa → {target_format.value} is not yet implemented."
        )

    logger.debug("Converting from genqa to %s", target_format.value)
    return converter(data)


class DatasetTransformation:
    """Hub-and-spoke transformer using genqa as the canonical intermediate format.

    All conversions route through genqa to achieve 2n complexity instead of n².
    To convert format A → format B: A → genqa → B.

    Accepts and returns Spark DataFrames (loaded from JSONL). Uses mapInPandas
    for efficient Arrow-based batch processing — each partition is handed to
    Python as a pandas DataFrame, avoiding per-row ser/de overhead.
    """

    @classmethod
    def _genqa_schema(cls) -> "StructType":
        """Return the Spark schema for genqa format."""
        from pyspark.sql.types import MapType, StringType, StructField, StructType

        return StructType(
            [
                StructField("query", StringType(), True),
                StructField("response", StringType(), True),
                StructField("system", StringType(), True),
                StructField("metadata", MapType(StringType(), StringType()), True),
            ]
        )

    @classmethod
    def transform(
        cls,
        df: "pyspark.sql.DataFrame",
        source_format: DatasetFormat,
        target_format: DatasetFormat,
    ) -> "pyspark.sql.DataFrame":
        """Transform a Spark DataFrame from one format to another via genqa.

        Uses mapInPandas for Arrow-based batch processing. Each partition is
        converted as a pandas DataFrame — zero-copy transfer, no per-row
        serialization overhead.

        Args:
            df: Spark DataFrame where each row is a record in the source format.
            source_format: The format of the input records.
            target_format: The desired output format.

        Returns:
            Spark DataFrame with each row converted to the target format.

        Raises:
            ValueError: If source or target format is unsupported.
        """
        import pandas as pd

        src_fmt = source_format
        tgt_fmt = target_format
        genqa_schema = cls._genqa_schema()

        def _to_genqa_batches(batch_iter):
            for pdf in batch_iter:
                records = pdf.to_dict(orient="records")
                converted = [convert_to_genqa(r, src_fmt) for r in records]
                yield pd.DataFrame(converted)

        genqa_df = df.mapInPandas(_to_genqa_batches, schema=genqa_schema)

        if tgt_fmt == DatasetFormat.GENQA:
            return genqa_df

        def _from_genqa_batches(batch_iter):
            for pdf in batch_iter:
                records = pdf.to_dict(orient="records")
                converted = [convert_from_genqa(r, tgt_fmt) for r in records]
                yield pd.DataFrame(converted)

        # Target schema must be provided by the outbound converter;
        # for now infer from a dry-run of the first batch.
        # TODO: Add explicit schema registry per target format.
        return genqa_df.mapInPandas(_from_genqa_batches, schema=genqa_df.schema)

    @classmethod
    def detect_format(cls, file_path: str) -> DatasetFormat:
        """Auto-detect the format of a JSONL dataset file.

        Delegates to DatasetFormatDetector for schema-based detection,
        ensuring a single source of truth for format identification.

        Args:
            file_path: Path to the JSONL file.

        Returns:
            The detected DatasetFormat.

        Raises:
            ValueError: If the format cannot be detected.
        """
        from sagemaker.ai_registry.dataset_format_detector import DatasetFormatDetector

        detected = DatasetFormatDetector.detect_format(file_path)
        if detected is None:
            raise ValueError(
                f"Unable to detect dataset format for '{file_path}'. "
                f"Supported formats: {[f.value for f in DatasetFormat]}"
            )

        try:
            return DatasetFormat(detected)
        except ValueError:
            raise ValueError(
                f"Detected format '{detected}' is not a supported transformation format. "
                f"Supported formats: {[f.value for f in DatasetFormat]}"
            )

    @classmethod
    def transform_file(
        cls,
        file_path: str,
        source_format: DatasetFormat,
        target_format: DatasetFormat,
    ) -> str:
        """Transform a JSONL file from one format to another via genqa.

        Reads the file line-by-line, converts each record, and writes the
        result to a new temporary JSONL file.

        Args:
            file_path: Path to the source JSONL file.
            source_format: The format of the input records.
            target_format: The desired output format.

        Returns:
            Path to the transformed JSONL file.
        """
        import json
        import tempfile

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        genqa_records = [convert_to_genqa(r, source_format) for r in records]

        if target_format == DatasetFormat.GENQA:
            converted = genqa_records
        else:
            converted = [convert_from_genqa(r, target_format) for r in genqa_records]

        out = tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w", encoding="utf-8")
        with out:
            for record in converted:
                out.write(json.dumps(record) + "\n")

        return out.name
