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
"""Unit tests for operators.common: metrics, bounds and selection primitives."""

from __future__ import absolute_import

import math
from collections import Counter

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import _bounds as bounds
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.score import _metrics as metrics
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import _floor as selection


class TestMacroF1Parts:
    def test_hand_computed(self):
        pairs = [("a", "a"), ("b", "a"), ("b", "b"), (None, "b"), ("c", "b")]
        got = metrics.macro_f1_parts(pairs)
        # a: tp1 fp0 fn1 -> 2/3 ; b: tp1 fp1 fn2 -> 2/5 ; c has no gold support
        assert got["per_class"] == {"a": pytest.approx(2 / 3), "b": pytest.approx(0.4)}
        assert got["macro_f1"] == pytest.approx((2 / 3 + 0.4) / 2)
        assert got["correct"] == 2 and got["total"] == 5
        assert got["accuracy"] == 0.4
        assert got["confusion"] == Counter({("a", "b"): 1, ("b", None): 1, ("b", "c"): 1})

    def test_empty(self):
        got = metrics.macro_f1_parts([])
        assert got["macro_f1"] == 0.0 and got["accuracy"] == 0.0 and got["total"] == 0


class TestMultisetMicroF1:
    def test_counts_duplicates_as_multiset(self):
        gold = Counter({("PER", "Anna"): 2, ("ORG", "ACME"): 1})
        pred = Counter({("PER", "Anna"): 1, ("ORG", "Zed"): 1})
        got = metrics.multiset_micro_f1([(pred, gold)])
        assert (got["tp"], got["fp"], got["fn"]) == (1, 1, 2)
        assert got["precision"] == 0.5 and got["recall"] == pytest.approx(1 / 3)
        assert got["per_type"]["PER"] == pytest.approx(2 / 3) and got["per_type"]["ORG"] == 0.0

    def test_empty(self):
        got = metrics.multiset_micro_f1([])
        assert got["f1"] == 0.0 and got["per_type"] == {}


class TestCorrelations:
    def test_token_f1(self):
        assert metrics.token_f1([], []) == 1.0
        assert metrics.token_f1(["a"], []) == 0.0
        assert metrics.token_f1(["a", "b"], ["b", "c"]) == 0.5
        assert metrics.token_f1(["a"], ["b"]) == 0.0

    def test_average_ranks_with_ties(self):
        assert metrics.average_ranks([3, 1, 3, 2]) == [3.5, 1.0, 3.5, 2.0]

    def test_pearson_and_spearman(self):
        assert metrics.pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
        assert metrics.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
        assert metrics.pearson([1], [1]) == 0.0
        assert metrics.pearson([1, 1], [1, 2]) == 0.0
        assert metrics.spearman([1, 2, 10], [1, 3, 4]) == pytest.approx(1.0)

    def test_quadratic_weighted_kappa(self):
        assert metrics.quadratic_weighted_kappa([1, 2, 3], [1, 2, 3]) == 1.0
        assert metrics.quadratic_weighted_kappa([2, 2], [2, 2]) == 1.0
        assert metrics.quadratic_weighted_kappa([1, 3], [3, 1]) == pytest.approx(-1.0)
        k = metrics.quadratic_weighted_kappa([1, 2, 3, 4], [1, 2, 4, 3])
        assert 0 < k < 1


class TestBounds:
    def test_sample_sizes(self):
        assert bounds.calibration_sample_size(0.85) == 19
        assert bounds.calibration_sample_size(0.90) == 29
        assert bounds.calibration_sample_size(0.95) == 59
        assert bounds.calibration_sample_size(0.95, alpha=0.01) == 90

    def test_zero_error_bound_roundtrip(self):
        n = bounds.calibration_sample_size(0.9)
        assert bounds.zero_error_lower_bound(n) >= 0.9 > bounds.zero_error_lower_bound(n - 1)
        assert bounds.zero_error_lower_bound(0) == 0.0
        assert bounds.best_zero_error_target(29) == math.floor(0.05 ** (1 / 29) * 1000) / 1000

    @pytest.mark.parametrize(
        "target, alpha, text",
        [
            (0.5, 0.05, "at least 0.85"),
            (1.0, 0.05, "in \\(0, 1\\)"),
            (True, 0.05, "in \\(0, 1\\)"),
            (0.9, 1.0, "alpha must be"),
        ],
    )
    def test_refusals(self, target, alpha, text):
        with pytest.raises(DataPreparerOperatorError, match=text) as e:
            bounds.calibration_sample_size(target, alpha)
        assert isinstance(e.value, ValueError)


class TestSelectWithFloor:
    def test_best_wins_and_floor_is_inclusive(self):
        cands = [{"spec": "a", "val": 0.90}, {"spec": "b", "val": 0.92}, {"spec": "c", "val": None}]
        got = selection.select_with_floor(0.90, cands, 0.02)
        assert got["best"]["spec"] == "b" and got["cleared"] is True
        assert got["gain"] == pytest.approx(0.02)
        assert selection.select_with_floor(0.90, cands, 0.03)["cleared"] is False

    def test_skip_drops_before_ranking(self):
        cands = [{"spec": "a", "val": 0.95}, {"spec": "b", "val": 0.92}]
        got = selection.select_with_floor(
            0.9, cands, 0.0, skip=lambda c: "bad" if c["spec"] == "a" else None
        )
        assert got["best"]["spec"] == "b"
        assert got["skipped"] == [({"spec": "a", "val": 0.95}, "bad")]

    def test_no_candidates_or_no_baseline(self):
        assert selection.select_with_floor(0.9, [], 0.0) == {
            "best": None,
            "gain": None,
            "cleared": False,
            "skipped": [],
        }
        got = selection.select_with_floor(None, [{"val": 0.5}], 0.0)
        assert got["best"] == {"val": 0.5} and got["gain"] is None and got["cleared"] is False


class TestHeldoutReviewFlag:
    def test_flag_gap_and_tolerance(self):
        assert selection.heldout_review_flag(0.9, None)["flag"] is None
        assert selection.heldout_review_flag(0.9, 0.80)["flag"] is False  # gap == tol
        assert selection.heldout_review_flag(0.9, 0.79)["flag"] is True
        assert selection.heldout_review_flag(0.9, 0.85, tol=0.04)["gap"] == 0.05
        assert selection.heldout_review_flag(None, 0.5)["gap"] == -0.5
