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
"""Unit tests for operators.auto_labeling.demos: the fixed few-shot draw."""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import demos


def _ent(*types):
    return [{"type": t, "text": "x"} for t in types]


class TestFixedDemos:
    def test_k_per_class_sorted_and_only_labeled_rows(self):
        seed = [
            {"id": "1", "gt_label": "b"},
            {"id": "2", "gt_label": "a"},
            {"id": "3", "gt_label": "b"},
            {"id": "4", "gt_label": "b"},
            {"id": "5"},
            {"id": "6", "gt_label": None},
        ]
        assert [r["id"] for r in demos.fixed_demos(seed, 2)] == ["2", "1", "3"]
        assert demos.fixed_demos(seed, 2) == demos.fixed_demos(seed, 2)

    def test_ner_covers_each_type_rarest_first_and_skips_empty_rows(self):
        seed = [
            {"id": "1", "gt_entities": _ent("PER")},
            {"id": "2", "gt_entities": []},
            {"id": "3", "gt_entities": _ent("PER", "LOC")},
            {"id": "4", "gt_entities": _ent("PER")},
            {"id": "5", "gt_entities": ["junk"]},
        ]
        chosen = demos.fixed_demos_ner(seed, 1)
        assert [r["id"] for r in chosen] == ["3"]  # LOC is rarest; row 3 covers PER too
        chosen = demos.fixed_demos_ner(seed, 2)
        assert [r["id"] for r in chosen] == ["3", "1"]


class TestSelectDemos:
    def test_modes_and_summary(self):
        seed = [
            {"id": "1", "gt_label": "a"},
            {"id": "2", "gt_label": "a"},
            {"id": "3", "gt_label": "b"},
        ]
        res = demos.select_demos(seed, k=1)
        assert [r["id"] for r in res.demos] == ["1", "3"]
        assert res.summary == {
            "mode": "fixed",
            "task": "classification",
            "k_per_class": 1,
            "demos": 2,
        }
        assert demos.select_demos(seed, mode="none").summary == {
            "mode": "none",
            "task": "classification",
            "demos": 0,
        }
        ner = demos.select_demos([{"gt_entities": _ent("PER")}], task="ner", k=3)
        assert ner.summary == {"mode": "fixed", "task": "ner", "k_per_type": 3, "demos": 1}
