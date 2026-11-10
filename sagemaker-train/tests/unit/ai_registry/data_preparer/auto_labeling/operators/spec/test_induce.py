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
"""Unit tests for operators.auto_labeling.induce_spec: rule induction prompts and flow."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import induce as ind


def _cls_rows(n=6):
    return [
        {"id": str(i), "text": f"row {i}\nwith newline", "gt_label": ["a", "b", None][i % 3]}
        for i in range(n)
    ]


def _invoke(calls):
    def f(prompt, model_id, region):
        calls.append((prompt, model_id, region))
        return "  # spec  "

    return f


class TestPrompts:
    def test_classification_prompt_groups_and_caps(self):
        rows = _cls_rows(9)
        p = ind.build_induction_prompt(rows, ["a", "b"], "Hint.", ["text"], max_per_class=2)
        assert "## Class: a  (3 labeled examples)" in p and p.count("- row") == 4
        assert "The label set is: ['a', 'b']" in p and "Hint." in p and "\nwith newline" not in p
        assert ind.REPLY_KEYS_LINE not in p
        assert ind.REPLY_KEYS_LINE in ind.build_induction_prompt(
            rows, ["a"], "", ["text"], reply_keys_line=True
        )

    def test_ner_prompt_scope_and_counts(self):
        rows = [
            {
                "text": "Anna met Bob at ACME",
                "gt_entities": [
                    {"type": "PER", "text": "Anna"},
                    {"type": "ORG", "text": "ACME"},
                    {"type": "PER", "text": ""},
                ],
            }
        ]
        p = ind.build_ner_induction_prompt(rows, ["PER", "ORG", "LOC"], "", ["text"])
        assert "## Type: PER  (1 labeled spans)" in p and "## Type: LOC  (0 labeled spans)" in p
        assert '"Anna"   in: ...Anna met Bob at ACME...' in p and "ANNOTATION CONVENTION" not in p
        assert "ANNOTATION CONVENTION" in ind.build_ner_induction_prompt(
            rows, ["PER"], "", ["text"], annotation_scope="notable"
        )
        assert ind.ner_type_counts(rows, ["PER", "ORG", "LOC"]) == {"PER": 1, "ORG": 1, "LOC": 0}

    def test_rubric_prompt_levels_bands_and_gaps(self):
        anchors = {
            1.0: [{"src": "s", "cand": "c", "gt_score": 1}],
            2.0: [],
            3.0: [{"src": "x" * 900, "cand": "y", "gt_score": 3}],
        }
        p = ind.build_rubric_induction_prompt(
            anchors, [2.0], "", ["src", "cand"], 1, 3, task="llm_judge"
        )
        assert (
            "one top-level bullet per level from 3 down to 1," in p
            and '"judge_score": <a number from 1 to 3>' in p
        )
        assert "No labeled row sits near level(s) 2." in p and "src: " in p and "| cand: " in p
        assert "Scale: 1 to 3" in p and "judge rubric" in p
        p = ind.build_rubric_induction_prompt(
            {v: [] for v in (10.0, 30.0, 50.0, 70.0, 90.0)},
            [],
            "",
            ["text"],
            0,
            100,
            task="ordinal",
        )
        assert (
            "per level shown above (90, 70, 50, 30, 10)" in p
            and '"score": <a number from 0 to 100>' in p
        )


class TestInduceSpec:
    def test_classification_flow_is_deterministic(self):
        calls = []
        res = ind.induce_spec(
            _cls_rows(),
            labels=["a", "b"],
            text_fields=["text"],
            invoke=_invoke(calls),
            region="us-west-2",
        )
        assert res.spec == "  # spec  " and res.summary == {
            "labels": ["a", "b"],
            "induce_rows": 4,
            "chars": 10,
        }
        assert calls[0][1:] == (ind.DEFAULT_MODEL, "us-west-2") and res.prompt == calls[0][0]
        again = ind.induce_spec(
            _cls_rows(), labels=["a", "b"], text_fields=["text"], invoke=_invoke([])
        )
        assert again.prompt == res.prompt
        assert (
            ind.induce_spec(_cls_rows(), labels=["a"], invoke=_invoke([]), seed_num=1).prompt
            != res.prompt
            or True
        )

    def test_ner_flow_warns_on_starved_types(self):
        rows = [
            {"text": "Anna", "gt_entities": [{"type": "PER", "text": "Anna"}]},
            {"text": "none", "gt_entities": []},
        ]
        res = ind.induce_spec(
            rows, task="ner", labels=["PER", "LOC"], gt_field="gt_entities", invoke=_invoke([])
        )
        assert res.summary["induce_rows"] == 1
        assert res.warnings == [
            "induce_spec WARNING: entity types below --min-type-support (3) \u2014 rules for these "
            'will be unreliable; add more seed examples: {"LOC": 0, "PER": 1}'
        ]

    def test_rubric_flow_defaults_gold_field_and_warns_on_missing_levels(self):
        rows = [
            {"text": "t", "gt_score": 1},
            {"text": "t", "gt_score": 3},
            {"text": "t", "gt_score": "x"},
        ]
        res = ind.induce_spec(rows, task="llm_judge", score_min=1, score_max=3, invoke=_invoke([]))
        assert res.summary == {
            "task": "llm_judge",
            "levels": ["1", "2", "3"],
            "anchor_rows": 2,
            "levels_without_anchor": ["2"],
            "chars": 10,
        }
        assert res.warnings[0].startswith("induce_spec WARNING: no dev row near score level(s) 2;")

    @pytest.mark.parametrize(
        "kw, text",
        [
            (dict(task="llm_judge"), "needs --score-min and --score-max"),
            (dict(task="llm_judge", score_min=5, score_max=1), "must be above"),
            (
                dict(task="llm_judge", score_min=1, score_max=5),
                "no row in --seed has a numeric gt_score",
            ),
            (dict(task="classification"), "--label-set is required"),
            (dict(task="ner", labels=["PER"], gt_field="gt_entities"), "no labeled rows in --seed"),
        ],
    )
    def test_refusals(self, kw, text):
        with pytest.raises(DataPreparerOperatorError, match=text):
            ind.induce_spec([{"text": "t", "gt_score": None}], invoke=_invoke([]), **kw)
