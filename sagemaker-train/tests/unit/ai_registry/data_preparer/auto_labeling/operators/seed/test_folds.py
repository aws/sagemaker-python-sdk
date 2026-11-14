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
"""Unit tests for operators.auto_labeling.kfold: stratified folds and the rotation manifest."""

from __future__ import absolute_import

import hashlib

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import folds as kfold
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import rows_bytes


def _rows(n, labels=("a", "b", "c")):
    return [{"id": str(i), "text": f"t{i}", "gt_label": labels[i % len(labels)]} for i in range(n)]


class TestStratifiedFolds:
    def test_disjoint_complete_and_stratified(self):
        rows = _rows(24)
        folds = kfold.stratified_folds(rows, "gt_label", 4, 0)
        flat = sorted(i for f in folds for i in f)
        assert flat == list(range(24)) and len(flat) == len(set(flat))
        for f in folds:
            assert {rows[i]["gt_label"] for i in f} == {"a", "b", "c"}

    def test_deterministic_given_seed(self):
        rows = _rows(30)
        assert kfold.stratified_folds(rows, "gt_label", 3, 7) == kfold.stratified_folds(
            rows, "gt_label", 3, 7
        )
        assert kfold.stratified_folds(rows, "gt_label", 3, 7) != kfold.stratified_folds(
            rows, "gt_label", 3, 8
        )

    def test_ner_spreads_the_rarest_type(self):
        rows = [{"id": str(i), "gt_entities": [{"type": "PER", "text": "x"}]} for i in range(8)]
        rows += [
            {
                "id": f"r{i}",
                "gt_entities": [{"type": "PER", "text": "x"}, {"type": "LOC", "text": "y"}],
            }
            for i in range(4)
        ]
        rows.append({"id": "none", "gt_entities": ["junk"]})
        folds = kfold.stratified_folds(rows, "gt_entities", 4, 0, task="ner")
        assert all(
            any(
                "LOC" in {e["type"] for e in rows[i]["gt_entities"] if isinstance(e, dict)}
                for i in f
            )
            for f in folds
        )
        assert sum(len(f) for f in folds) == 13


class TestBuildKfold:
    def test_files_manifest_and_hashes(self):
        rows = _rows(12)
        res = kfold.build_kfold(rows, out_dir="folds", k=3, seed=1)
        assert set(res.files) == {
            "fold_0.jsonl",
            "fold_1.jsonl",
            "fold_2.jsonl",
            "iter0_dev.jsonl",
            "iter1_dev.jsonl",
            "iter2_dev.jsonl",
        }
        assert res.fold_sizes == [6, 3, 3] and res.n_rows == 12  # round-robin within each class
        m = res.manifest
        assert m["folds"] == ["folds/fold_0.jsonl", "folds/fold_1.jsonl", "folds/fold_2.jsonl"]
        assert m["rotation"][1] == {
            "iter": 1,
            "val": "folds/fold_1.jsonl",
            "dev": "folds/iter1_dev.jsonl",
        }
        assert (
            m["sha256"]["fold_0.jsonl"]
            == hashlib.sha256(rows_bytes(res.files["fold_0.jsonl"])).hexdigest()
        )
        ids_dev1 = {r["id"] for r in res.files["iter1_dev.jsonl"]}
        assert ids_dev1 == {r["id"] for r in res.files["fold_0.jsonl"]} | {
            r["id"] for r in res.files["fold_2.jsonl"]
        }
        assert m["usage"] == kfold.USAGE_NOTE and res.warnings == []

    def test_exclude_drops_by_content_and_warns(self):
        rows = _rows(9)
        held = [dict(rows[0], id="other"), {"id": "x", "text": "stranger", "gt_label": "a"}]
        res = kfold.build_kfold(rows, out_dir="f", k=2, exclude=held)
        assert res.n_rows == 8 and res.dropped == 1
        assert res.warnings == [
            "kfold_rotate: dropped 1 row(s) present in --exclude so the reserved test "
            "partition does not leak into the dev/val folds."
        ]
        assert kfold.content_key({"id": "1", "a": 1}) == kfold.content_key({"id": "2", "a": 1})

    def test_refusals(self):
        with pytest.raises(DataPreparerOperatorError, match="--k must be >= 2"):
            kfold.build_kfold(_rows(4), out_dir="f", k=1)
        with pytest.raises(DataPreparerOperatorError, match="with a frozen split drop --exclude"):
            kfold.build_kfold(_rows(4), out_dir="f", exclude=[], frozen_split=True)
