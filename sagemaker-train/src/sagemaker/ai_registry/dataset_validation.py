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
"""Dataset validation utilities for AI Registry."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

# -------------- IO ---------------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries.
    
    Args:
        path: Path to JSONL file
        
    Returns:
        List of parsed JSON objects
        
    Raises:
        ValueError: If JSON parsing fails
    """
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception as e:
                raise ValueError(f"JSON decode error line {lineno}: {e}") from e
    return out


# -------------- SFT --------------
def _normalize_sft(record: Dict[str, Any]) -> None:
    """Normalize and validate SFT record format.
    
    Args:
        record: Dictionary containing SFT data
        
    Raises:
        ValueError: If record format is invalid
    """
    if "input" in record and "output" in record:
        if not isinstance(record["input"], str) or not isinstance(record["output"], str):
            raise ValueError("input/output must be strings")
        return
    if "prompt" in record and "completion" in record:
        if not isinstance(record["prompt"], str) or not isinstance(record["completion"], str):
            raise ValueError("prompt/completion must be strings")
        return
    raise ValueError("missing SFT fields: need input/output or prompt/completion")


@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.validate_sft")
def validate_sft(rows: Iterable[Dict[str, Any]]) -> None:
    """Validate SFT dataset format.
    
    Args:
        rows: Iterable of SFT records
        
    Raises:
        ValueError: If any record is invalid
    """
    for i, record in enumerate(rows):
        try:
            _normalize_sft(record)
        except Exception as e:
            raise ValueError(f"SFT row {i}: {e}") from e


# -------------- DPO --------------
@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.validate_dpo")
def validate_dpo(rows: Iterable[Dict[str, Any]]) -> None:
    """Validate DPO dataset format.
    
    Args:
        rows: Iterable of DPO records
        
    Raises:
        ValueError: If any record is invalid
    """
    for i, record in enumerate(rows):
        if not all(k in record for k in ("prompt", "chosen", "rejected")):
            raise ValueError(f"DPO row {i}: missing prompt|chosen|rejected")
        for k in ("prompt", "chosen", "rejected"):
            if not isinstance(record[k], str):
                raise ValueError(f"DPO row {i}: {k} must be string")


# -------------- RLVR --------------
@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.validate_rlvr")
def validate_rlvr(rows: Iterable[Dict[str, Any]]) -> None:
    """Validate RLVR dataset format.
    
    Args:
        rows: Iterable of RLVR records
        
    Raises:
        ValueError: If any record is invalid
    """
    for i, record in enumerate(rows):
        if not isinstance(record.get("prompt"), str):
            raise ValueError(f"RLVR row {i}: prompt must be string")
        if "samples" not in record or not isinstance(record["samples"], list):
            raise ValueError(f"RLVR row {i}: samples must be list")
        for j, sample in enumerate(record["samples"]):
            if not isinstance(sample.get("completion"), str):
                raise ValueError(f"RLVR row {i} sample {j}: completion must be string")
            if not isinstance(sample.get("score"), (int, float)):
                raise ValueError(f"RLVR row {i} sample {j}: score must be number")

@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.normalize_rlvr_row")
def normalize_rlvr_row(record: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a row into the standard RLVR format.
    
    Converts formats like GSM8K example into the standard RLVR format:
    - prompt -> string (join list of {'content'} entries)
    - samples -> list of one sample with completion and score
    
    Args:
        record: Input record to normalize
        
    Returns:
        Normalized RLVR record
    """
    # flatten prompt list to string
    prompt_data = record.get("prompt")
    if isinstance(prompt_data, list):
        prompt_text = "\n".join([
            item.get("content", "") for item in prompt_data 
            if isinstance(item, dict) and "content" in item
        ])
    elif isinstance(prompt_data, str):
        prompt_text = prompt_data
    else:
        prompt_text = ""

    # extract completion from extra_info.answer or reward_model.ground_truth
    completion = ""
    if "extra_info" in record and "answer" in record["extra_info"]:
        completion = record["extra_info"]["answer"]
    elif "reward_model" in record and "ground_truth" in record["reward_model"]:
        completion = str(record["reward_model"]["ground_truth"])

    # simple scoring heuristic
    score = 1.0 if completion else 0.0

    return {
        "prompt": prompt_text,
        "samples": [
            {"completion": completion, "score": score}
        ]
    }


# -------------- auto detect --------------
@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.detect_dataset_type")
def detect_dataset_type(record: Dict[str, Any]) -> Optional[str]:
    """Auto-detect dataset type from record format.
    
    Args:
        record: Sample record to analyze
        
    Returns:
        Detected type ('rlvr', 'dpo', 'sft') or None if unknown
    """
    if "samples" in record and isinstance(record["samples"], list) and isinstance(record.get("prompt"), str):
        return "rlvr"
    if all(k in record for k in ("prompt", "chosen", "rejected")):
        return "dpo"
    if ("input" in record and "output" in record) or ("prompt" in record and "completion" in record):
        return "sft"
    return None


@_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="dataset_validation.validate_dataset")
def validate_dataset(path: str, technique: str) -> None:
    """Validate dataset file against specified technique format.
    
    Args:
        path: Path to JSONL dataset file
        technique: Validation technique ('sft', 'dpo', 'rlvr', 'auto')
        
    Raises:
        ValueError: If dataset format is invalid or technique is unsupported
    """
    rows = load_jsonl(path)
    
    if not rows:
        raise ValueError(f"Dataset file is empty: {path}")

    # auto detect if requested
    if technique == "auto":
        detected_type = detect_dataset_type(rows[0])
        if detected_type is None:
            raise ValueError(f"Cannot auto-detect dataset type for file: {path}")
        technique = detected_type

    technique = technique.lower().strip()

    if technique == "sft":
        validate_sft(rows)
    elif technique == "dpo":
        validate_dpo(rows)
    elif technique == "rlvr":
        rows_normalized = [normalize_rlvr_row(record) for record in rows]
        validate_rlvr(rows_normalized)
    else:
        raise ValueError("technique must be one of: sft | dpo | rlvr | auto")
