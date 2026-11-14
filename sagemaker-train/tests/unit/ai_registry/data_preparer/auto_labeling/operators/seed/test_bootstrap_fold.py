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
"""Unit tests for operators.seed.bootstrap's fold: the labels people returned, into the seed.

Ported from the auto-labeling skill's bootstrap_seed hil/check tests. No AWS call.
"""

from __future__ import absolute_import

import json
import os
import stat

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import bootstrap as B
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels import (
    SeedAppendRefused,
)


def _spot_sample(n, answer="pos", reason="calibration"):
    """``n`` spot-check rows, each carrying the model's answer ``answer``."""
    return [
        {"id": f"c{i}", "text": f"t{i}", "label": answer, "_review_reason": reason}
        for i in range(n)
    ]


def _returns(n, wrong):
    """People's labels for ``n`` spot-check rows: the first ``wrong`` disagree."""
    return [{"id": f"c{i}", "gt_label": "neg" if i < wrong else "pos"} for i in range(n)]


class TestSpotCheckVerdict:
    def test_empty_without_rows(self):
        assert B.spot_check_verdict(0, 0, 0) == {}

    def test_calibrated_withdraws_at_seven(self):
        ok = B.spot_check_verdict(29, 6, 0, calibrated=True)
        assert (ok["spot_check_failed"], ok["certificate_withdrawn"]) == (False, False)
        bad = B.spot_check_verdict(29, 6, 1, calibrated=True)
        assert (bad["spot_check_failed"], bad["certificate_withdrawn"]) == (True, True)
        assert bad["spot_check_errors"] == "7 of 29"

    def test_bar_and_unknown_have_no_certificate(self):
        for basis in (B.BASIS_BAR, B.BASIS_UNKNOWN):
            out = B.spot_check_verdict(29, 7, 0, basis=basis)
            assert out["spot_check_failed"] and "certificate_withdrawn" not in out
            assert out["spot_check"]["basis"] == basis

    def test_agreement_and_no_answer(self):
        sc = B.spot_check_verdict(10, 2, 1, conflicts=1, no_model_answer=2)["spot_check"]
        assert (sc["confirmed"], sc["agreement_rate"], sc["complete"]) == (
            4,
            round(4 / 6, 4),
            False,
        )
        sc = B.spot_check_verdict(3, 0, 0, no_model_answer=3)["spot_check"]
        assert sc["agreement_rate"] is None


class TestSpotCheckMessage:
    @pytest.mark.parametrize(
        "basis,errors,no_answer,expected",
        [
            (B.BASIS_CALIBRATED, 7, 0, "CERTIFICATE WITHDRAWN"),
            (B.BASIS_CALIBRATED, 0, 1, "INCOMPLETE: the certificate stands only"),
            (B.BASIS_CALIBRATED, 0, 0, "the certificate stands."),
            (B.BASIS_UNKNOWN, 7, 0, "SPOT-CHECK FAILED (7 of 29 wrong); basis unknown"),
            (B.BASIS_UNKNOWN, 0, 1, "INCOMPLETE; basis unknown"),
            (B.BASIS_UNKNOWN, 0, 0, "pass the spot-check; basis unknown"),
            (B.BASIS_BAR, 7, 0, "SPOT-CHECK FAILED THE BAR"),
            (B.BASIS_BAR, 0, 1, "INCOMPLETE at the 0.85 bar."),
            (B.BASIS_BAR, 0, 0, "pass the spot-check at the 0.85 bar."),
        ],
    )
    def test_by_basis(self, basis, errors, no_answer, expected):
        report = B.spot_check_verdict(29, errors, 0, basis=basis, no_model_answer=no_answer)
        msg = B.spot_check_message(report)
        assert expected in msg and msg.startswith("spot-check errors") and msg.endswith("\n")
        assert ("carry no model answer" in msg) == bool(no_answer)

    def test_conflict_note(self):
        rets = _returns(29, 6) + [{"id": "c28", "gt_label": "neg"}]
        _, r = B.fold_human_labels(_spot_sample(29), rets)
        assert (r["spot_check"]["conflicts"], r["spot_check"]["errors"]) == (1, 7)
        assert r["conflicting_ids"] == ["c28"]
        assert "1 row(s) had conflicting reviewer answers" in B.spot_check_message(r)


class TestFoldHumanLabels:
    def test_cold_start_sample(self):
        sample = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
        labels = [
            {"id": "1", "gt_label": "x", "annotator": "ann1"},
            {"id": 2, "verdict": "unsure"},
        ]
        rows, rep = B.fold_human_labels(sample, labels, source_export="e.jsonl")
        assert [(r["id"], r["gt_label"], r["provenance"]) for r in rows] == [
            (1, "x", "human_verified")
        ]
        assert rows[0]["_bootstrap"] == "hil" and rows[0]["source_export"] == "e.jsonl"
        assert rows[0]["annotator"] == "ann1"
        assert (rep["kept"], rep["sample_unresolved"], rep["sample_unlabeled"]) == (1, 1, 1)
        assert "spot_check_errors" not in rep

    def test_multi_annotation_and_export_id_field(self):
        sample = [{"id": "r1", "text": "a"}]
        labels = [
            {"record_id": "r1", "label": "x", "annotator": "a"},
            {"record_id": "r1", "label": "x", "annotator": "b"},
        ]
        rows, rep = B.fold_human_labels(sample, labels, export_id_field="record_id")
        assert rows[0]["n_annotations"] == 2 and rows[0]["annotators"] == ["a", "b"]
        assert rep["multi_annotation_agreed"] == 1

    def test_model_outputs_count_each_reason(self):
        sample = [
            {"id": "a", "text": "t", "_review_reason": "low_confidence"},
            {"id": "b", "text": "t", "_review_reason": "low_confidence"},
            {"id": "c", "text": "t"},
            {"id": "d", "text": "t"},
        ]
        side = {
            "a": {"pred_field": "label", "model_output": {"label": "x", "confidence": 0.4}},
            "b": {"pred_field": "label", "model_output": {"label": "y"}},
            "c": {"pred_field": "label", "model_output": {}},
            "d": {"pred_field": "weird", "model_output": {"weird": {"k": 1}}},
        }
        labels = [{"id": k, "gt_label": "x"} for k in "abcd"]
        rows, rep = B.fold_human_labels(sample, labels, model_outputs=side)
        assert rows[0]["confidence"] == 0.4
        assert rep["model_output_joined"] == 4
        outcome = rep["review_outcome"]
        assert outcome["low_confidence"] == {"confirmed": 1, "corrected": 1}
        assert sum(outcome["routed"].values()) == 2

    def test_calibrated_basis_from_the_side_file(self):
        sample = [{"id": f"c{i}", "text": "t", "split": "spot_check"} for i in range(29)]
        side = {
            f"c{i}": {
                "pred_field": "label",
                "model_output": {"label": "pos", "calib_accepted": True},
            }
            for i in range(29)
        }
        _, rep = B.fold_human_labels(sample, _returns(29, 7), model_outputs=side)
        assert rep["spot_check"]["basis"] == B.BASIS_CALIBRATED
        assert rep["certificate_withdrawn"] is True

    @pytest.mark.parametrize(
        "status,basis", [("Certified", B.BASIS_CALIBRATED), ("Labeled", B.BASIS_BAR)]
    )
    def test_job_status_decides_a_split_row(self, status, basis):
        sample = [
            {"id": f"c{i}", "text": "t", "label": "pos", "split": "spot_check"} for i in range(29)
        ]
        _, rep = B.fold_human_labels(sample, _returns(29, 0), job_status=status)
        assert rep["spot_check"]["basis"] == basis

    def test_split_rows_without_a_routing_tag_are_unknown(self):
        sample = [
            {"id": f"c{i}", "text": "t", "label": "pos", "split": "spot_check"} for i in range(29)
        ]
        _, rep = B.fold_human_labels(sample, _returns(29, 0))
        assert rep["spot_check"]["basis"] == B.BASIS_UNKNOWN

    def test_missing_and_no_answer_rows(self):
        sample = _spot_sample(3) + [{"id": "n", "text": "t", "_review_reason": "calibration"}]
        labels = [{"id": "c0", "gt_label": "pos"}, {"id": "n", "gt_label": "pos"}]
        _, rep = B.fold_human_labels(sample, labels)
        sc = rep["spot_check"]
        assert (sc["not_returned"], sc["no_model_answer"], sc["confirmed"]) == (2, 1, 1)

    def test_known_spot_check_counts_as_returned(self):
        known = {"rows": 5, "corrected": 1, "no_model_answer": 1}
        _, rep = B.fold_human_labels(_spot_sample(2), _returns(2, 0), known_spot_check=known)
        assert rep["spot_check_known"] == 5 and rep["spot_check"]["rows"] == 7
        assert rep["spot_check"]["corrected"] == 1

    def test_spot_check_off_gives_no_verdict(self):
        _, rep = B.fold_human_labels(_spot_sample(3), [], spot_check_reviewed=False)
        assert rep["spot_check_skipped"]["rows"] == 3 and "spot_check_errors" not in rep
        _, rep = B.fold_human_labels([{"id": 1}], [], spot_check_reviewed=False)
        assert "spot_check_skipped" not in rep

    def test_empty_label_and_non_dict_rows(self):
        rows, rep = B.fold_human_labels(["junk", {"id": 1}], [{"id": 1, "label": ""}])
        assert rows == [] and rep.get("empty_labels") == 1


class TestAppendToSeed:
    def test_creates_appends_and_repeats(self, tmp_path):
        seed = str(tmp_path / "seed.jsonl")
        rows = [{"id": 1, "gt_label": "x", "_bootstrap": "hil"}]
        assert B.append_to_seed(seed, rows) == (1, [])
        assert B.append_to_seed(seed, rows) == (0, ["1"])
        more = [{"id": 2, "gt_label": "y", "_bootstrap": "hil"}]
        assert B.append_to_seed(seed, more) == (1, [])
        with open(seed, encoding="utf-8") as f:
            assert [json.loads(line)["id"] for line in f] == [1, 2]

    def test_keeps_a_seed_without_a_final_newline_and_its_mode(self, tmp_path):
        seed = tmp_path / "seed.jsonl"
        seed.write_text('{"id": 0, "gt_label": "z"}', encoding="utf-8")
        os.chmod(seed, 0o640)
        B.append_to_seed(str(seed), [{"id": 1, "gt_label": "x", "_bootstrap": "hil"}])
        assert seed.read_text(encoding="utf-8").splitlines()[0] == '{"id": 0, "gt_label": "z"}'
        assert stat.S_IMODE(os.stat(seed).st_mode) == 0o640

    def test_refuses_a_relabel_or_a_clash(self, tmp_path):
        seed = str(tmp_path / "seed.jsonl")
        B.append_to_seed(seed, [{"id": 1, "gt_label": "x", "_bootstrap": "hil"}])
        before = open(seed, encoding="utf-8").read()
        with pytest.raises(SeedAppendRefused, match="with another label"):
            B.append_to_seed(seed, [{"id": 1, "gt_label": "y", "_bootstrap": "hil"}])
        assert open(seed, encoding="utf-8").read() == before

    def test_nothing_to_append(self, tmp_path):
        seed = str(tmp_path / "seed.jsonl")
        assert B.append_to_seed(seed, []) == (0, [])
        assert not os.path.exists(seed)


class TestCheckFold:
    def test_folded_and_merged(self):
        export = [{"id": 1, "gt_label": "x"}, {"id": 2, "verdict": "skip"}]
        seed = [{"id": 1, "gt_label": "x", "_bootstrap": "hil"}]
        rep = B.check_fold(export, seed, labels_rows=[{"id": 1, "label": "x"}])
        assert rep["ok"] and (rep["folded"], rep["merged"]) == (1, 1)

    def test_missing_and_not_merged(self):
        export = [{"id": 1, "gt_label": "x"}]
        rep = B.check_fold(export, [], labels_rows=[{"id": 1, "label": "y"}])
        assert not rep["ok"] and rep["missing_from_seed"] == ["1"] and rep["not_merged"] == ["1"]

    def test_human_provenance_counts_as_folded(self):
        rep = B.check_fold([{"id": 1, "gt_label": "x"}], [{"id": 1, "provenance": "spot_checked"}])
        assert rep["ok"]

    def test_conflicts_and_no_label_column(self):
        rep = B.check_fold([{"id": 1, "gt_label": "x"}, {"id": 1, "gt_label": "y"}], [])
        assert not rep["ok"] and rep["conflicting_ids"] == ["1"]
        rep = B.check_fold([{"id": 1, "category": "x"}], [])
        assert not rep["ok"] and rep["no_label_ids"] == ["1"]

    def test_every_sample_row_needs_one_label(self):
        sample = [{"id": i} for i in (1, 2, 3, 4)]
        export = [
            {"id": 1, "gt_label": "x"},
            {"id": 2, "verdict": "unsure"},
            {"id": 3, "gt_label": "a"},
            {"id": 3, "gt_label": "b"},
        ]
        seed = [{"id": 1, "_bootstrap": "hil"}]
        rep = B.check_fold(export, seed, sample_rows=sample)
        assert (rep["sample_missing"], rep["sample_unsure"], rep["sample_conflicting"]) == (
            ["4"],
            ["2"],
            ["3"],
        )
        assert rep["sample_rows"] == 4 and not rep["ok"]
