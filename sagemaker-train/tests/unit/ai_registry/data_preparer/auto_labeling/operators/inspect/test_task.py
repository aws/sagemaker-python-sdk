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
"""Unit tests for operators.auto_labeling.detect_task: the task-type detector."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import task as dt


class TestGoldSignals:
    @pytest.mark.parametrize(
        "field, task",
        [
            ("gt_entities", "ner"),
            ("gt_summary", "summarization"),
            ("gt_answers", "qa"),
            ("gt_label", "classification"),
        ],
    )
    def test_gold_field_decides_at_adopt(self, field, task):
        res = dt.detect("", [{"text": "t"}, {"text": "t", field: "x"}])
        assert (res.task, res.confidence, res.decision) == (task, 0.95, "adopt")
        assert res.signals == {"gt_field": field}

    def test_precedence_is_by_specificity_not_row_order(self):
        rows = [{"gt_label": "a"}, {"gt_entities": []}]
        assert dt.detect("", rows).task == "ner"

    def test_gt_score_disambiguation(self):
        assert dt.detect("", [{"source": "s", "candidate": "c", "gt_score": 3}]).task == "llm_judge"
        ord_rows = [{"gt_score": v} for v in (1, 2, 3, 4, 5)]
        res = dt.detect("", ord_rows)
        assert (res.task, res.confidence, res.decision) == ("ordinal", 0.85, "adopt")
        res = dt.detect("", [{"gt_score": 12.5}, {"gt_score": 70.1}])
        assert (res.task, res.confidence, res.decision) == ("llm_judge", 0.7, "confirm")
        assert dt.detect("", [{"gt_score": True}]).task == "llm_judge"  # bool is not a score


class TestInputAndSpecSignals:
    def test_input_signatures(self):
        res = dt.detect("", [{"question": "q", "context": "c"}])
        assert res.task == "qa" and res.signals["input_fields"] == ["question", "context"]
        assert dt.detect("", [{"source": "s", "candidate": "c"}]).task == "llm_judge"

    @pytest.mark.parametrize(
        "block, task, conf",
        [
            ('{"reasoning": "...", "entities": []}', "ner", 0.7),
            ("SUMMARY: ...", "summarization", 0.7),
            ("ANSWER: ...", "qa", 0.6),
            ('{"reasoning": "...", "judge_score": 3}', "llm_judge", 0.6),
            ("SCORE: 3", "ordinal", 0.5),
            ("LABEL: a", "classification", 0.6),
        ],
    )
    def test_output_schema_markers(self, block, task, conf):
        res = dt.detect(f"# Spec\n## Output Schema\n{block}\n## Next\nSUMMARY: not here", [])
        assert (res.task, res.confidence) == (task, conf)

    def test_headings_and_prose_do_not_set_the_task(self):
        assert dt.detect("## Summary:\nprose\nthe ANSWER: is no", []).task == "classification"
        assert dt.detect("## Label Set\n| a |", []).signals == {"has_label_set": True}
        res = dt.detect(None, None)
        assert (res.task, res.confidence, res.decision) == ("classification", 0.3, "ask")

    def test_reply_lines_without_a_schema_heading(self):
        spec = 'An output schema:\n{"reasoning": "x", "summary": "y"}\nnot {"reasoning": "x", "label": "z"}'
        assert dt.detect(spec, []).task == "summarization"

    def test_image_field_is_reported(self):
        res = dt.detect("", [{"text": "t", "image_path": "a.png", "gt_label": "x"}])
        assert res.signals == {"image_field": "image_path", "gt_field": "gt_label"}
        assert set(res.to_json()) == {"task", "confidence", "reason", "signals", "decision"}
