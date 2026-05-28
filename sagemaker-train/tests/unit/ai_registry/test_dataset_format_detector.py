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

from sagemaker.ai_registry.dataset_format_detector import DatasetFormatDetector


class TestDetectFormat:
    def _write_jsonl(self, records):
        """Write records to a temp JSONL file and return the path."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for r in records:
            f.write(json.dumps(r) + "\n")
        f.close()
        return f.name

    def test_detect_openai_chat(self):
        path = self._write_jsonl(
            [
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "hello"},
                    ]
                }
            ]
        )
        try:
            result = DatasetFormatDetector.detect_format(path)
            # openai_chat or converse could match — both have messages
            assert result in ("openai_chat", "converse", "dpo")
        finally:
            os.unlink(path)

    def test_detect_genqa(self):
        path = self._write_jsonl([{"query": "What is AI?"}])
        try:
            result = DatasetFormatDetector.detect_format(path)
            assert result == "genqa"
        finally:
            os.unlink(path)

    def test_detect_hf_prompt_completion(self):
        path = self._write_jsonl([{"prompt": "q", "completion": "a"}])
        try:
            result = DatasetFormatDetector.detect_format(path)
            assert result is not None
        finally:
            os.unlink(path)

    def test_detect_returns_none_for_unknown(self):
        path = self._write_jsonl([{"random_key": "random_value"}])
        try:
            result = DatasetFormatDetector.detect_format(path)
            assert result is None
        finally:
            os.unlink(path)

    def test_detect_returns_none_for_empty_file(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.close()
        try:
            result = DatasetFormatDetector.detect_format(f.name)
            assert result is None
        finally:
            os.unlink(f.name)

    def test_detect_returns_none_for_invalid_json(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write("not valid json\n")
        f.close()
        try:
            result = DatasetFormatDetector.detect_format(f.name)
            assert result is None
        finally:
            os.unlink(f.name)

    def test_detect_returns_none_for_missing_file(self):
        result = DatasetFormatDetector.detect_format("/nonexistent/path.jsonl")
        assert result is None

    def test_detect_rft_format(self):
        path = self._write_jsonl(
            [{"messages": [{"role": "user", "content": "hi"}], "extra_field": "value"}]
        )
        try:
            result = DatasetFormatDetector.detect_format(path)
            # Could match converse/dpo/openai_chat schema or fall through to rft
            assert result is not None
        finally:
            os.unlink(path)


class TestValidateDatasetDelegatesToDetect:
    def test_validate_returns_true_when_detected(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write(json.dumps({"query": "test"}) + "\n")
        f.close()
        try:
            assert DatasetFormatDetector.validate_dataset(f.name) is True
        finally:
            os.unlink(f.name)

    def test_validate_returns_false_when_not_detected(self):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write(json.dumps({"unknown": "format"}) + "\n")
        f.close()
        try:
            assert DatasetFormatDetector.validate_dataset(f.name) is False
        finally:
            os.unlink(f.name)
