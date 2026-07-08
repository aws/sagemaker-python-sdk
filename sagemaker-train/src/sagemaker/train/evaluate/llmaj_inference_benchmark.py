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
"""
This module generates the InspectAI benchmark Python file and supporting
configuration that runs inside the InspectAI container to produce inference
responses for LLM-as-Judge evaluation. It also handles dataset format
conversion from LLMAJ format to InspectAI format.
"""

import json
import logging

_logger = logging.getLogger(__name__)


INFERENCE_BENCHMARK_TEMPLATE = '''\
"""Auto-generated inference-only benchmark for LLM-as-Judge Phase 1.
"""
import json
import os
from typing import Dict, List, Optional
from urllib.parse import urlparse

import boto3
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import generate, TaskState


def _upload_to_s3(content: str, s3_uri: str, region: Optional[str] = None):
    """Upload string content to S3."""
    parsed = urlparse(s3_uri)
    kwargs = {"region_name": region} if region else {}
    boto3.client("s3", **kwargs).put_object(
        Bucket=parsed.netloc, Key=parsed.path.lstrip("/"),
        Body=content.encode("utf-8")
    )


@scorer(metrics=[])
def inference_collector(output_s3_uri: str, region: Optional[str] = None) -> Scorer:
    """Collect model responses and write to S3 incrementally."""
    collected: List[Dict[str, str]] = []

    async def score(state: TaskState, target: Target) -> Score:
        response_text = state.output.completion if state.output else ""
        collected.append({
            "prompt": str([{"role": "user", "content": state.input_text}]),
            "modelResponses": [
                {"response": json.dumps([response_text]), "modelIdentifier": "model"}
            ],
        })
        content = "\\n".join(json.dumps(e) for e in collected) + "\\n"
        _upload_to_s3(content, output_s3_uri, region)
        return Score(value=1.0, answer=response_text)

    return score


@task
def inference_only(
    dataset: str = "dataset.jsonl",
    output_s3_uri: str = "",
    region: Optional[str] = None,
) -> Task:
    """Generate inference responses and write to S3 for LLMAJ Phase 2."""
    if not output_s3_uri:
        raise ValueError("output_s3_uri is required via task_args")
    return Task(
        dataset=json_dataset(dataset),
        solver=[generate()],
        scorer=inference_collector(output_s3_uri=output_s3_uri, region=region),
    )
'''

PYPROJECT_TOML_TEMPLATE = """[project]
name = "llmaj-inference-benchmark"
version = "0.1.0"
description = "Auto-generated benchmark for LLM-as-Judge inference generation"
requires-python = ">=3.10"
"""


def generate_benchmark_files() -> dict[str, str]:
    """Generate the benchmark directory file contents.

    :returns: Dict mapping filename to file content string.
    :rtype: dict[str, str]
    """
    return {
        "inference_only.py": INFERENCE_BENCHMARK_TEMPLATE,
        "pyproject.toml": PYPROJECT_TOML_TEMPLATE,
    }


def convert_dataset_to_inspectai_format(dataset_content: str) -> str:
    """Convert LLMAJ dataset format to InspectAI format.

    Transform each JSONL line from ``{"prompt": "..."}`` or ``{"query": "..."}``
    to ``{"input": "...", "target": ""}`` as expected by InspectAI's
    ``json_dataset`` loader.

    :param dataset_content: Raw JSONL content from the customer's dataset.
        Each non-empty line must be a JSON object containing either a
        ``"prompt"`` or ``"query"`` field.
    :type dataset_content: str
    :returns: Converted JSONL string in InspectAI format with one
        ``{"input": ..., "target": ""}`` object per line.
    :rtype: str
    :raises ValueError: If a line contains neither ``"prompt"`` nor
        ``"query"`` field.
    """
    converted_lines: list[str] = []
    for line_number, line in enumerate(dataset_content.split("\n"), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if "prompt" in record:
            prompt_text = record["prompt"]
        elif "query" in record:
            prompt_text = record["query"]
        else:
            raise ValueError(
                f"Line {line_number} has neither 'prompt' nor 'query' field: "
                f"{line.strip()!r}"
            )
        converted_lines.append(json.dumps({"input": prompt_text, "target": ""}))
    return "\n".join(converted_lines) + "\n"
