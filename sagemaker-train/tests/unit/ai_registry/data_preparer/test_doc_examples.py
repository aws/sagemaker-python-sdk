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
"""Guard: DataPreparer docs and docstrings use the nested job-config shape.

The published v2 schemas take ``generation_config={"DatasetSpecification":
{"SpecS3Uri": ...}, "NumSamples": ...}`` and ``input_data_config=[{"DatasetArn":
...}]``. Older flat keyword names are rejected by schema validation ("GenerationConfig:
Field required" / "InputDataConfig: Field required"), so an example using them fails
for anyone who copies it.
"""
from __future__ import absolute_import

import re
from pathlib import Path
from typing import List

import pytest

# Flat keyword names from the pre-v2 examples, written as call arguments.
_RETIRED_KWARGS = re.compile(r"\b(num_requested_samples|num_samples|spec_s3_uri)\s*=")


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "sagemaker-train").is_dir():
            return parent
    raise RuntimeError(f"Could not locate repo root from {__file__}.")


def _files() -> List[Path]:
    root = _repo_root()
    files = sorted((root / "docs" / "data_preparation").glob("*.rst"))
    files.append(root / "docs" / "training" / "data_preparer.rst")
    files.extend(sorted((root / "sagemaker-train" / "src" / "sagemaker" / "ai_registry").rglob("*.py")))
    return [f for f in files if f.is_file()]


def _hits(text: str) -> List[str]:
    return [f"line {n}: {line.strip()}" for n, line in enumerate(text.splitlines(), 1)
            if _RETIRED_KWARGS.search(line)]


class TestScanner:
    @pytest.mark.parametrize("snippet", [
        'DatasetJobConfig(num_requested_samples=500, spec_s3_uri="s3://b/s.txt")',
        "DatasetJobConfig(num_samples = 500)",
    ])
    def test_flat_kwargs_flagged(self, snippet):
        assert _hits(snippet)

    @pytest.mark.parametrize("snippet", [
        'generation_config={"DatasetSpecification": {"SpecS3Uri": "s3://b"}, "NumSamples": 5}',
        "self.num_samples_seen = 3",
        "``DatasetSpecification.SpecS3Uri`` points to the spec file",
    ])
    def test_nested_shape_not_flagged(self, snippet):
        assert not _hits(snippet)


def test_docs_and_docstrings_use_nested_shape():
    files = _files()
    assert files, "no files found to scan"
    violations = [f"{f.relative_to(_repo_root())} {hit}"
                  for f in files for hit in _hits(f.read_text(encoding="utf-8"))]
    assert not violations, (
        "Retired flat DatasetJobConfig keywords found; use the nested v2 shape:\n"
        + "\n".join(violations)
    )
