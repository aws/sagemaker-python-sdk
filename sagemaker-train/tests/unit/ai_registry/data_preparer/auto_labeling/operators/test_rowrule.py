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
"""Unit tests for operators.auto_labeling.correctness: the job's per-task row rule."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rowrule as c


class TestLabelText:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (None, None),
            (True, "true"),
            (False, "false"),
            (1, "1"),
            (1.0, "1"),
            (2.5, "2.5"),
            (" a ", "a"),
            ([1], [1]),
            ({"k": 1}, {"k": 1}),
        ],
    )
    def test_cases(self, value, expected):
        assert c.label_text(value) == expected

    def test_matches_compares_as_text(self):
        assert c.label_matches("1", 1) and c.label_matches(" a", "a ")
        assert not c.label_matches(None, "a") and not c.label_matches("a", None)
        assert not c.label_matches("A", "a")


class TestNer:
    def test_span_set_rules(self):
        assert c.ner_span_set(None) is None and c.ner_span_set("[]") is None
        assert c.ner_span_set([{"type": " PER ", "text": " Anna ", "confidence": 0.3}, 3]) == {
            ("PER", "Anna")
        }
        assert (
            c.task_correct(
                "ner", [{"type": "PER", "text": "Anna"}], [{"type": "PER", "text": "anna"}]
            )
            == 0
        )
        assert c.task_correct("ner", [], []) == 1
        assert c.task_correct("ner", None, []) == 0


class TestQa:
    def test_squad_normalize(self):
        assert c.squad_normalize("The quick, brown Fox!") == "quick brown fox"
        assert c.squad_normalize(None) == ""
        assert c.squad_normalize(42) == "42"

    def test_parse_qa_match(self):
        assert c.parse_qa_match(None) == "exact"
        assert c.parse_qa_match(" Contains ") == "contains"
        assert c.parse_qa_match("F1 >= 0.50") == "f1>=0.5"
        for bad in ("f1>=0", "f1>=1.5", "fuzzy"):
            with pytest.raises(DataPreparerOperatorError, match="--qa-match must be"):
                c.parse_qa_match(bad)

    def test_rules(self):
        assert c.qa_correct("paris.", ["Paris", "the city of Paris"]) == 1
        assert c.qa_correct(None, "x") == 0
        assert c.qa_correct("", []) == 1  # unanswerable
        assert c.qa_correct("", ["x"]) == 0
        assert c.qa_correct("Smith", "Bob Smith", "contains") == 1
        assert c.qa_correct("Smith", "Bob Smith") == 0
        assert c.qa_correct("Bob Smith", "Smith", "f1>=0.6") == 1
        assert c.qa_correct("Bob Smith", "Smith", "f1>=0.7") == 0
        assert c.qa_correct("", ["", "x"], "contains") == 1


class TestNumericTasks:
    def test_ordinal_rounds_to_tier(self):
        assert c.task_correct("ordinal", 2.4, 2) == 1
        assert c.task_correct("ordinal", "3", 3.0) == 1
        assert c.task_correct("ordinal", 2.6, 2) == 0
        assert c.task_correct("ordinal", "high", "high") == 1  # non-numeric: text compare
        assert c.task_correct("ordinal", None, 2) == 0

    def test_llm_judge_tolerance(self):
        assert c.task_correct("llm_judge", 3.5, 3) == 1
        assert c.task_correct("llm_judge", 3.6, 3) == 0
        assert c.task_correct("llm_judge", 8, 3, tolerance=5.0) == 1
        assert c.job_tolerance(None) == 0.5
        assert c.job_tolerance((1, 5)) == 0.5
        assert c.job_tolerance((0, 100)) == 5.0

    def test_as_number(self):
        assert c.as_number(True) is None and c.as_number("x") is None and c.as_number("2") == 2.0


class TestTables:
    def test_task_fields_and_no_promise(self):
        assert c.TASK_FIELDS["ner"] == ("entities", "gt_entities")
        assert c.GT_FIELD_FOR_PRED["judge_score"] == "gt_score"
        assert c.no_promise_reason("ordinal").startswith("no accuracy promise for ordinal")
        assert c.no_promise_reason("classification") is None
        assert c.task_correct(None, "1", 1) == 1 and c.task_correct("summarization", "a", "b") == 0
