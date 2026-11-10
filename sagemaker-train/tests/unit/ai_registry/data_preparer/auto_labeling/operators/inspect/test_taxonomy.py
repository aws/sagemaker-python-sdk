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
"""Unit tests for operators.auto_labeling.taxonomy_gate: the pre-spend taxonomy check."""

from __future__ import absolute_import

import sys

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import taxonomy as tg

TABLE = """# Spec
### Label Set
| Label | Definition |
|---|---|
| **billing** | money owed on an account |
| refund | money given back to the customer |
| other | everything else goes here |
| a \\| b | an escaped pipe |
### Next
| x | y |
|---|---|
| nope | nope |
"""


class TestParseLabelSet:
    def test_table_under_the_heading_first(self):
        assert tg.parse_label_set(TABLE) == [
            ("billing", "money owed on an account"),
            ("refund", "money given back to the customer"),
            ("other", "everything else goes here"),
            ("a | b", "an escaped pipe"),
        ]

    def test_bullet_and_rules_fallbacks_skip_vs_lines(self):
        spec = "### Label Set\n- **billing:** money\n- **a vs b:** disambiguation\n"
        assert tg.parse_label_set(spec) == [("billing", "money")]
        spec = "### Labelling Rules\n- **refund** given back\n## Notes\n- **Warning:** stray\n"
        assert tg.parse_label_set(spec) == [("refund", "given back")]

    def test_whole_spec_table_last_and_dedup(self):
        spec = "## Metrics\n| m | v |\n|---|---|\n| Acc | 0.9 |\n| acc | again |\n"
        assert tg.parse_label_set(spec) == [("Acc", "0.9")]
        assert tg.parse_label_set(None) == []


class TestChecks:
    def test_absence_classes(self):
        ls = [
            ("none", ""),
            ("other", "x"),
            ("non-fiction", "books"),
            ("no_label", ""),
            ("not applicable", ""),
            ("fees", "charges not covered by any insurance plan"),
            ("rest", "catch-all for the rest"),
        ]
        assert tg.find_absence_classes(ls) == [
            "none",
            "other",
            "no_label",
            "not applicable",
            "rest",
        ]

    def test_seed_coverage(self):
        ls = [("billing", ""), ("Refund", ""), ("other", "")]
        rows = (
            [{"gt_label": "BILLING"}] * 12
            + [{"gt_label": " refund "}] * 2
            + [{"gt_label": "typo"}, {"gt_label": ["x"]}, "junk", {}]
        )
        counts, under, missing, imbalanced, unknown = tg.seed_coverage(ls, rows, "gt_label")
        assert dict(counts) == {"billing": 12, "Refund": 2}
        assert under == ["Refund"] and missing == ["other"] and imbalanced is True
        assert unknown == ["['x']", "typo"]

    def test_overlap_is_lexical(self, monkeypatch):
        ls = [
            ("joy", "a feeling of great happiness and delight"),
            ("happiness", "a feeling of great happiness and delight"),
            ("weather", "the atmospheric conditions of rain wind and sun"),
        ]
        # an installed sentence-transformers is never imported: the check is lexical
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        pairs, method = tg.overlap_pairs(ls)
        assert method == "lexical"
        flagged = {frozenset(p[:2]) for p in pairs}
        assert frozenset({"joy", "happiness"}) in flagged
        assert frozenset({"joy", "weather"}) not in flagged
        assert tg.overlap_pairs(ls[:1]) == ([], "none")
        assert not hasattr(tg, "EMBED_MODEL") and not hasattr(tg, "SEM_SIM_THRESHOLD")

    def test_clear_and_review(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)
        spec = "### Label Set\n| L | D |\n|---|---|\n| billing | money owed |\n| login | cannot sign in |\n"
        rows = [{"gt_label": "billing"}] * 10 + [{"gt_label": "login"}] * 10
        rep = tg.gate(spec, rows)
        assert (rep.verdict, rep.decision, rep.n_classes) == ("clear", "proceed", 2)
        assert (
            rep.coverage["per_class"] == {"billing": 10, "login": 10}
            and rep.separability["method"] == "lexical"
        )
        rep = tg.gate(spec, rows[:12])
        assert rep.decision == "review-before-spend"
        assert [i["type"] for i in rep.issues] == ["class_undersampled", "severe_imbalance"]
        assert rep.issues[0]["classes"] == ["login"]

    def test_unparsed_label_set_is_review_for_classification_and_skip_otherwise(self):
        rep = tg.gate("# nothing", None)
        assert (
            rep.decision == "review-before-spend" and rep.issues[0]["type"] == "label_set_unparsed"
        )
        rep = tg.gate("# nothing", None, task="summarization")
        assert (rep.verdict, rep.decision) == (
            "skipped",
            "skip",
        ) and "no label set" in rep.skipped_reason
        assert rep.to_json()["separability"] == {
            "checked": False,
            "skipped_reason": "summarization: no label set expected",
        }

    def test_ordinal_skips_overlap_and_absence_is_a_warning(self):
        spec = "### Label Set\n| L | D |\n|---|---|\n| low | bad |\n| high | good |\n| other | everything else |\n"
        rep = tg.gate(spec, None, task="ordinal")
        assert rep.separability == {
            "checked": False,
            "skipped_reason": "ordinal: adjacent tiers are neighbors, not confusions",
        }
        assert rep.absence_classes == ["other"] and rep.issues[0]["severity"] == "warn"
        assert rep.coverage == {"checked": False}
