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
"""Unit tests for operators.auto_labeling.scoring: the local scorer over every task."""

from __future__ import absolute_import

import json

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.score import scorers as scoring


def _rows(task, pairs, pred_field=None, gt_field=None):
    pf = pred_field or scoring.PRED_FIELDS[task]
    gf = gt_field or scoring.GOLD_FIELDS[task]
    preds = [{"id": str(i), pf: p} for i, (p, _) in enumerate(pairs)]
    gts = [{"id": str(i), gf: g} for i, (_, g) in enumerate(pairs)]
    return preds, gts


class TestAlign:
    def test_by_id_when_both_sides_carry_it(self):
        preds = [{"id": "b", "label": 1}, {"id": "a", "label": 2}, {"id": "z", "label": 3}]
        gts = [{"id": "a", "gt_label": "A"}, {"id": "b", "gt_label": "B"}]
        assert scoring.align(preds, gts, "id", "label", "gt_label") == [(1, "B"), (2, "A")]

    def test_by_order_otherwise(self):
        preds = [{"label": 1}, {"label": 2}]
        gts = [{"id": "a", "gt_label": "A"}]
        assert scoring.align(preds, gts, "id", "label", "gt_label") == [(1, "A")]
        assert scoring.align(preds, gts, None, "label", "gt_label") == [(1, "A")]


class TestClassification:
    def test_macro_f1_accuracy_breakdown_confusion(self):
        preds, gts = _rows(
            "classification", [("a", "a"), ("b", "a"), ("b", "b"), (None, "b"), ("c", "b")]
        )
        r = scoring.score("classification", preds, gts)
        assert r.primary_name == "macro_f1"
        assert r.primary == round((2 / 3 + 0.4) / 2, 6)
        assert r.accuracy == 0.4 and r.counts == {"correct": 2, "total": 5}
        assert r.breakdown == {"a": round(2 / 3, 6), "b": 0.4}
        assert r.confusion_pairs[0]["n"] == 1 and len(r.confusion_pairs) == 3
        js = r.to_json()
        assert "events" not in js and "selection" not in js and js["n_no_ground_truth"] == 0
        # score.json's bytes: the command line's key order
        assert list(js) == [
            "primary",
            "primary_name",
            "accuracy",
            "breakdown",
            "confusion_pairs",
            "counts",
            "n_no_ground_truth",
        ]

    def test_rows_without_gold_are_counted_not_scored(self):
        preds, gts = _rows("classification", [("a", "a"), ("a", None)])
        r = scoring.score("classification", preds, gts)
        assert r.counts["total"] == 1 and r.n_no_ground_truth == 1 and r.primary == 1.0


class TestNer:
    def test_span_multiset_and_row_rule(self):
        gold = [{"type": "PER", "text": "Anna"}, {"type": "PER", "text": "Anna"}]
        pred_ok = [
            {"type": "PER", "text": " Anna ", "confidence": 0.9},
            {"type": "PER", "text": "Anna"},
        ]
        pred_half = [{"type": "PER", "text": "Anna"}, {"type": "ORG", "text": "Zed"}]
        preds, gts = _rows("ner", [(pred_ok, gold), (pred_half, gold), ("garbage", gold)])
        r = scoring.score("ner", preds, gts)
        assert r.primary_name == "micro_f1"
        assert r.counts == {"total_tp": 3, "total_fp": 1, "total_fn": 3, "total": 3, "correct": 1}
        assert r.accuracy == round(1 / 3, 6) and r.breakdown["ORG"] == 0.0
        assert scoring.span_multiset('[{"type":"PER","text":"a"}]') == {("PER", "a"): 1}
        assert scoring.span_multiset("nope") == {}


class TestQa:
    def test_token_f1_exact_match_and_correct(self):
        pairs = [
            ("Paris", ["Paris", "city of Paris"]),
            ("the Bob", "Bob Smith"),
            (None, "x"),
            ("", []),
        ]
        preds, gts = _rows("qa", pairs)
        r = scoring.score("qa", preds, gts)
        assert r.primary_name == "token_f1"
        assert r.counts == {"total": 4, "missing": 1, "correct": 2}
        assert r.exact_match == 0.5 and r.accuracy == 0.5
        assert r.primary == round((1 + 2 / 3 + 0 + 1) / 4, 6)


class TestNumeric:
    def test_ordinal_primary_is_the_row_rule_and_agreement_reported(self):
        preds, gts = _rows("ordinal", [(1, 1), (2.4, 2), (5, 3), (None, 4), ("x", 5)])
        r = scoring.score("ordinal", preds, gts)
        assert r.primary_name == "accuracy" and r.primary == 0.4  # 2 of 5 gold rows
        assert r.counts == {"total": 5, "scored": 3, "missing": 2, "correct": 2}
        assert r.selection == scoring.SELECTION_NOTE and r.correct_tolerance is None
        assert r.mae == round((0 + 0.4 + 2) / 3, 6) and r.adjacent_accuracy == round(2 / 3, 6)
        assert -1.0 <= r.quadratic_weighted_kappa <= 1.0

    def test_llm_judge_tolerance_in_result(self):
        preds, gts = _rows("llm_judge", [(3.4, 3), (3.6, 3)])
        assert scoring.score("llm_judge", preds, gts).primary == 0.5
        r = scoring.score("llm_judge", preds, gts, tolerance=1.0)
        assert r.primary == 1.0 and r.correct_tolerance == 1.0
        assert r.to_json()["correct_tolerance"] == 1.0

    def test_under_two_points_zeroes_agreement(self):
        preds, gts = _rows("ordinal", [(2, 2)])
        r = scoring.score("ordinal", preds, gts)
        assert r.primary == 1.0 and r.quadratic_weighted_kappa == 0.0 and r.spearman == 0.0

    def test_word_gold_needs_label_map(self):
        preds, gts = _rows("ordinal", [(1, "low"), (3, "high")])
        with pytest.raises(DataPreparerOperatorError, match="non-numeric and have no rank mapping"):
            scoring.score("ordinal", preds, gts)
        lm = scoring.label_map_from_names(["Low", "Mid", "High"])
        assert lm == {"low": 1.0, "mid": 2.0, "high": 3.0}
        assert scoring.score("ordinal", preds, gts, label_map=lm).primary == 1.0

    def test_label_map_keys_are_case_insensitive(self):
        preds, gts = _rows("ordinal", [(1, "Very Negative"), (2, " positive ")])
        cased = scoring.score("ordinal", preds, gts, label_map={"Very Negative": 1, "Positive": 2})
        assert cased.primary == 1.0
        assert (
            cased.to_json()
            == scoring.score(
                "ordinal", preds, gts, label_map={"very negative": 1.0, "positive": 2.0}
            ).to_json()
        )

    def test_parse_label_map(self):
        assert scoring.parse_label_map('{"A ": 1, "b": 2.5}') == {"a": 1.0, "b": 2.5}
        for bad in ("nope", "[]", "{}", '{"a": true}', '{"a": "1"}'):
            with pytest.raises(DataPreparerOperatorError, match="--label-map must be"):
                scoring.parse_label_map(bad)


class TestRefusals:
    def test_nothing_aligned(self):
        with pytest.raises(
            DataPreparerOperatorError, match="no prediction/ground-truth rows aligned"
        ):
            scoring.score("classification", [], [{"id": "a", "gt_label": "x"}])

    def test_no_gold_in_field(self):
        preds, gts = _rows("classification", [("a", "a")], gt_field="label_gold")
        with pytest.raises(DataPreparerOperatorError, match="pass --gt-field"):
            scoring.score("classification", preds, gts)
        assert scoring.score("classification", preds, gts, gt_field="label_gold").primary == 1.0


@pytest.mark.parametrize(
    "task, pairs, keys",
    [
        (
            "qa",
            [("x", ["x"]), ("y", ["z"])],
            "primary primary_name exact_match accuracy counts n_no_ground_truth",
        ),
        (
            "ner",
            [([{"type": "ORG", "text": "Acme"}], [{"type": "ORG", "text": "Acme"}])],
            "primary primary_name accuracy precision recall breakdown counts n_no_ground_truth",
        ),
        (
            "llm_judge",
            [(1, 1), (3, 2), (5, 5)],
            "primary primary_name counts n_no_ground_truth selection correct_tolerance "
            "quadratic_weighted_kappa spearman pearson mae adjacent_accuracy",
        ),
    ],
)
def test_score_keys_in_the_command_lines_order(task, pairs, keys):
    preds, gts = _rows(task, pairs)
    assert list(scoring.score(task, preds, gts).to_json()) == keys.split()


@pytest.mark.parametrize(
    "labels, keys",
    [
        ([0, 1, 2], ["0", "1", "2"]),
        ([True, False], ["false", "true"]),
        (["a", "b"], ["a", "b"]),
    ],
)
def test_breakdown_keeps_the_labels_as_they_are(labels, keys):
    """Int and bool class labels score, and ``json.dumps`` writes them as the skill did."""
    pairs = [(lbl, lbl) for lbl in labels] + [(labels[0], labels[-1])]
    preds, gts = _rows("classification", pairs)
    result = scoring.score("classification", preds, gts)
    assert sorted(result.breakdown) == sorted(labels)
    assert all(type(k) is type(lbl) for k, lbl in zip(sorted(result.breakdown), sorted(labels)))
    assert list(json.loads(json.dumps(result.to_json()))["breakdown"]) == keys
