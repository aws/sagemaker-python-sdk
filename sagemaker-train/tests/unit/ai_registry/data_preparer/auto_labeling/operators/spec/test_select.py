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
"""Unit tests for operators.auto_labeling.spec_select: the gold gate."""

from __future__ import absolute_import

import importlib
import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# the module (the subpackage attribute 'select' is the operator function)
ss = importlib.import_module(
    "sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.select"
)


def _doc(initial_val, *iter_vals, breakdowns=None):
    d = {
        "initial": {"spec": "work/labeling_spec_iter_0.md", "val": initial_val},
        "iters": [
            {"spec": f"work/labeling_spec_iter_{i + 1}.md", "val": v}
            for i, v in enumerate(iter_vals)
        ],
    }
    if breakdowns:
        d["initial"]["breakdown"] = breakdowns[0]
        for it, bd in zip(d["iters"], breakdowns[1:]):
            it["breakdown"] = bd
    return d


class TestGoldGate:
    def test_ships_best_iteration_not_last(self):
        d = ss.select(_doc(0.885, 0.921, 0.880), min_gain=0.005)
        assert d.ship == "refined" and d.spec.endswith("iter_1.md") and d.gain == 0.036
        assert d.shipped_spec_file == "labeling_spec_iter_1.md" and d.selection_mode == "gold-gate"
        assert d.to_json()["initial_val"] == 0.885

    def test_do_no_harm_floor_is_inclusive(self):
        assert ss.select(_doc(0.90, 0.92), min_gain=0.02).ship == "refined"
        d = ss.select(_doc(0.90, 0.915), min_gain=0.02)
        assert d.ship == "initial" and d.best_refined_val == 0.915 and "do no harm" in d.reason

    def test_regression_and_no_iterations(self):
        assert ss.select(_doc(0.90, 0.80)).ship == "initial"
        d = ss.select(_doc(0.90))
        assert d.ship == "initial" and d.reason == "no refinement iterations ran"
        assert d.to_json() == {
            "ship": "initial",
            "spec": "work/labeling_spec_iter_0.md",
            "val": 0.9,
            "min_gain": 0.02,
            "reason": "no refinement iterations ran",
            "selection_mode": "gold-gate",
            "shipped_spec_file": "labeling_spec_iter_0.md",
        }

    def test_null_val_is_refused_not_skipped(self):
        with pytest.raises(DataPreparerOperatorError, match="must be a number, got NoneType"):
            ss.select(_doc(0.90, None, 0.95))

    def test_null_spec_keeps_keys(self):
        js = ss.select({"initial": {"spec": None, "val": 0.5}, "iters": []}).to_json()
        assert js["spec"] is None and js["shipped_spec_file"] is None
        assert list(js) == ["ship", "spec", "val", "min_gain", "reason"] + [
            "shipped_spec_file",
            "selection_mode",
        ]

    @pytest.mark.parametrize(
        "doc, gain, tol, keys",
        [
            (
                _doc(0.885, 0.921),
                0.0,
                0.5,
                "ship spec val initial_val gain min_gain reason per_type_tol "
                "shipped_spec_file selection_mode",
            ),
            (
                _doc(0.90, 0.915),
                0.02,
                None,
                "ship spec val best_refined_val gain min_gain reason shipped_spec_file "
                "selection_mode",
            ),
            (
                _doc(0.80, 0.90, breakdowns=[{"A": 0.9}, {"A": 0.1}]),
                0.0,
                0.1,
                "ship spec val min_gain per_type_tol reason per_type_rejected "
                "shipped_spec_file selection_mode",
            ),
        ],
    )
    def test_decision_keys_in_the_command_lines_order(self, doc, gain, tol, keys):
        # decision.json's bytes: the command line's key order (no refined pick puts the
        # tolerance before the reason)
        assert list(ss.select(doc, min_gain=gain, per_type_tol=tol).to_json()) == keys.split()


class TestPerTypeFloor:
    def test_rejects_minority_regression_despite_aggregate_gain(self):
        bds = [{"PER": 0.9, "ORG": 0.6}, {"PER": 0.98, "ORG": 0.4}]
        d = ss.select(_doc(0.80, 0.90, breakdowns=bds), min_gain=0.0, per_type_tol=0.1)
        assert d.ship == "initial" and d.per_type_tol == 0.1
        assert d.per_type_rejected == [
            {
                "spec": "work/labeling_spec_iter_1.md",
                "val": 0.9,
                "regressed_types": [
                    {"type": "ORG", "initial": 0.6, "candidate": 0.4, "delta": -0.2}
                ],
            }
        ]
        assert "regressed a passing type" in d.reason

    def test_allows_clean_gain_and_noop_without_breakdowns(self):
        bds = [{"PER": 0.9, "ORG": 0.6}, {"PER": 0.95, "ORG": 0.62}]
        assert (
            ss.select(_doc(0.80, 0.90, breakdowns=bds), min_gain=0.0, per_type_tol=0.1).ship
            == "refined"
        )
        assert ss.select(_doc(0.80, 0.90), min_gain=0.0, per_type_tol=0.1).ship == "refined"
        assert ss.regressed_types(
            {"breakdown": {"A": 0.0, "B": 0.5}}, {"breakdown": {"B": 0.1}}, 0.1
        ) == [{"type": "B", "initial": 0.5, "candidate": 0.1, "delta": -0.4}]


class TestValidation:
    def test_iterations_alias_warns(self):
        d = ss.select(
            {"initial": {"spec": "a", "val": 0.5}, "iterations": [{"spec": "b", "val": 0.9}]}
        )
        assert d.ship == "refined" and d.warnings == [ss.ITERS_ALIAS_WARNING]

    @pytest.mark.parametrize(
        "data, text",
        [
            ([], "must be a JSON object"),
            ({"initial": {"spec": "a", "val": 0.5}}, "missing required key"),
            (
                {"initial": {"spec": "a", "val": 0.5}, "iter": []},
                "Got keys: \\['initial', 'iter'\\]",
            ),
            ({"initial": "x", "iters": []}, "scores\\['initial'\\] must carry"),
            ({"initial": {"spec": "a", "val": True}, "iters": []}, "must be a number, got bool"),
            ({"initial": {"spec": "a", "val": 0.5}, "iters": "x"}, "must be a list"),
            (
                {"initial": {"spec": "a", "val": 0.5}, "iters": [{"spec": "b"}]},
                "iters'\\]\\[0\\] must be a dict",
            ),
            (
                {"initial": {"spec": "a", "val": 0.5, "breakdown": {"A": "x"}}, "iters": []},
                "breakdown'\\]\\['A'\\] must be a number",
            ),
        ],
    )
    def test_refusals(self, data, text):
        with pytest.raises(DataPreparerOperatorError, match=text):
            ss.select(data)

    def test_iter_spec_name(self):
        assert ss.iter_spec_name(0) == "labeling_spec_iter_0.md"
        assert ss.iter_spec_name(2, "sentiment") == "sentiment_labeling_spec_iter_2.md"


class TestNumbersAsPassed:
    def test_int_scores_stay_ints(self):
        """A val of ``1`` is written as ``1``, not ``1.0``, like the candidates file."""
        d = ss.select(_doc(0, 1), min_gain=0.0)
        doc = d.to_json()
        assert doc["val"] == 1 and isinstance(doc["val"], int)
        assert doc["initial_val"] == 0 and isinstance(doc["initial_val"], int)
        assert doc["gain"] == 1 and isinstance(doc["gain"], int)
        assert isinstance(ss.select(_doc(0.5, 0.9)).to_json()["val"], float)
