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
"""Unit tests for operators.auto_labeling.labeling_round: the pieces of one labeling round."""

from __future__ import absolute_import

import hashlib
import json
import os

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import _round_parts as L

# ---- run id and input ---------------------------------------------------------------------


def test_check_run_id():
    assert L.check_run_id("run-1") == "run-1" and L.check_run_id("run_1") == "run-1"
    assert L.check_run_id("x" * 40) == "x" * 40
    for bad, match in (
        ("-a", "must be letters, digits and hyphens"),
        ("a.b", "no dots"),
        ("", "must be"),
        (None, "must be"),
        ("a b", "spaces"),
        ("x" * 41, "is 41 characters; the DataSet names"),
    ):
        with pytest.raises(DataPreparerOperatorError, match=match):
            L.check_run_id(bad)


def test_input_kind(tmp_path):
    assert L.input_kind("arn:aws:x") == ("arn", "arn:aws:x") and L.input_kind("s3://b/k") == (
        "s3",
        "s3://b/k",
    )
    assert L.input_kind("corpus.jsonl") == ("local", "corpus.jsonl")
    ref = tmp_path / "corpus.ref.json"
    ref.write_text(json.dumps({"source_s3_uri": "s3://b/c.jsonl"}))
    assert L.input_kind(str(ref)) == ("s3", "s3://b/c.jsonl")
    ref.write_text("{}")
    with pytest.raises(
        DataPreparerOperatorError, match="has no 'source_s3_uri'; re-run the hub_ops resolve action"
    ):
        L.input_kind(str(ref))
    ref.write_text("{nope")
    with pytest.raises(
        DataPreparerOperatorError, match="cannot read corpus.ref.json pointer .*JSONDecodeError"
    ):
        L.input_kind(str(ref))
    with pytest.raises(DataPreparerOperatorError, match="FileNotFoundError"):
        L.input_kind(str(tmp_path / "missing.ref.json"))
    with pytest.raises(DataPreparerOperatorError, match="--input is required"):
        L.input_kind("")


def test_read_local_rows_and_task_sample(tmp_path):
    p = tmp_path / "rows.jsonl"
    p.write_text('{"id": 1}\n\n{"id": 2, "text": "t"}\n')
    assert L.read_local_rows(str(p), "x") == [{"id": 1}, {"id": 2, "text": "t"}]
    assert L.read_task_sample(str(p), limit=1) == [{"id": 1}]
    p.write_text('{"id": 1}\nnot json\n[1]\n')
    with pytest.raises(
        DataPreparerOperatorError, match="line 2 is not JSON: .* The labeling job reads JSON Lines"
    ):
        L.read_local_rows(str(p), "x")
    assert L.read_task_sample(str(p)) == [{"id": 1}, [1]]  # tolerant of a bad line
    p.write_text('{"id": 1}\n[1]\n')
    with pytest.raises(DataPreparerOperatorError, match="line 2 is not a JSON object"):
        L.read_local_rows(str(p), "x")
    assert L.read_task_sample(str(tmp_path / "missing")) == [] and L.read_task_sample(None) == []


def test_seed_evidence_and_no_seed_message(tmp_path):
    out, spec = str(tmp_path / "work" / "out.jsonl"), str(tmp_path / "spec.md")
    assert L.seed_evidence(out, spec, eval_path="e.jsonl") == "--eval"
    assert L.seed_evidence(out, spec, demos_path="d.jsonl", demos_per_class=2) == "--demos"
    assert L.seed_evidence(out, spec, demos_path="d.jsonl", demos_per_class=0) is None
    assert L.seed_evidence(out, spec) is None
    (tmp_path / "splits").mkdir()
    (tmp_path / "splits" / "manifest.json").write_text("{}")
    assert L.seed_evidence(out, spec) == str(tmp_path / "splits" / "manifest.json")
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "decision.json").write_text("{}")
    assert L.seed_evidence(out, spec) == str(
        tmp_path / "work" / "decision.json"
    )  # next to --out first
    msg = L.no_seed_message()
    assert msg.startswith("no labeled seed: no --eval, no --demos, and no decision.json")
    assert "19 rows for 0.85, 29 rows for 0.90, 59 rows for 0.95" in msg and msg.endswith(
        "--confirm-no-seed."
    )


# ---- gold -----------------------------------------------------------------------------------


def test_labels_missing_from_spec():
    spec = "Use billing for bills. Card arrival covers cards. refund_requests are refunds."
    assert L.labels_missing_from_spec(
        spec,
        [
            "billing",
            "Billing",
            "card_arrival",
            "card-arrival",
            "refund",
            "refund_requests",
            "bill",
            "",
            " ",
        ],
    ) == ["refund", "bill"]
    assert (
        L.labels_missing_from_spec("", ["a"]) == ["a"] and L.labels_missing_from_spec("a", []) == []
    )
    assert L.fold_label("Card-Arrival ") == "cardarrival" and L.fold_label(3) == "3"


def test_check_gold_labels():
    gold = [
        {"id": i, "gt_label": lab}
        for i, lab in enumerate(["billing", "refund", "Card Arrival", "CARD_ARRIVAL"])
    ]
    assert (
        L.check_gold_labels(
            gold, "gt_label", ["billing", "refund", "Card Arrival", "CARD_ARRIVAL"], "", "e"
        )
        is None
    )
    assert L.check_gold_labels([], "gt_label", ["x"], "", "e") is None
    assert (
        L.check_gold_labels(gold, "gt_label", None, "no label set here", "e") is None
    )  # unreadable set: skipped
    # against the spec's Label Set: a warning only
    spec = "### Label Set\n\n| Label | Definition |\n|---|---|\n| billing | a |\n| refund | b |\n| card_arrival | c |\n"
    w = L.check_gold_labels(gold, "gt_label", None, spec, "e")
    assert w == (
        "2 of the 4 --eval gold rows have a 'gt_label' that is not in the label set ('Card Arrival' x1 (the label "
        "set spells it 'card_arrival'), 'CARD_ARRIVAL' x1 (the label set spells it 'card_arrival')). The model "
        "answers only with the label set's spelling and the job compares labels exactly, so each such row "
        "scores wrong."
    )
    # against an explicit set: refused above 5 percent
    with pytest.raises(
        DataPreparerOperatorError,
        match=r"^--eval 'e.jsonl': 2 of the 4 --eval gold rows .* "
        r"Respell the gold to the label set .* nothing was uploaded\.$",
    ):
        L.check_gold_labels(gold, "gt_label", ["billing", "refund", "card_arrival"], "", "e.jsonl")
    many = [{"id": i, "gt_label": "billing"} for i in range(40)] + [{"id": 99, "gt_label": "x"}]
    assert L.check_gold_labels(many, "gt_label", ["billing"], "", "e").startswith(
        "1 of the 41 --eval gold rows"
    )
    assert L.check_gold_labels([{"gt_label": None}], "gt_label", ["a"], "", "e") is None


def test_row_ids():
    assert (
        L.row_id({"id": "a"}, 3) == "a"
        and L.row_id({}, 3) == "3"
        and L.row_id({"id": None}, 0) == "0"
    )
    assert (
        L.row_id({"id": 300.0}, 0) == "300.0" and L.row_id({"id": 300.0}, 0, int_ids=True) == "300"
    )
    assert (
        L.row_id({"id": 300.5}, 0, int_ids=True) == "300.5"
        and L.row_id({"rid": 7}, 0, "rid") == "7"
    )
    assert L.float_ids_to_join([{"id": 1.0}, {"id": 1.5}, {"id": 2}, {}]) == 1
    assert L.corpus_ids_are_ints({"id": 3}) and L.corpus_ids_are_ints({"id": "12"})
    assert not L.corpus_ids_are_ints({"id": True}) and not L.corpus_ids_are_ints({"id": "a"})
    assert not L.corpus_ids_are_ints("junk") and not L.corpus_ids_are_ints({})
    assert L.check_ids([{"id": "a"}, {}, {"id": 2}]) == ["a", "1", "2"]
    with pytest.raises(
        DataPreparerOperatorError, match=r"2 duplicate id\(s\) \(e.g. \['a', '2'\]\); the job joins"
    ):
        L.check_ids([{"id": "a"}, {"id": "a"}, {}, {"id": "2"}])


def test_split_gold():
    rows = [
        {
            "id": "r1",
            "text": "t1",
            "gt_label": "a",
            "gt_other": 1,
            "provenance": "human_verified",
            "_review_reason": "x",
            "source": "Reuters",
            "_bootstrap": None,
            "note": "n",
        },
        {"text": "t2", "gt_label": None},
        {"id": 300.0, "text": "t3", "gt_label": "b"},
    ]
    inputs, gold, unlabeled = L.split_gold(rows, "gt_label", text_fields=("text", "gt_label"))
    assert inputs == [
        {"id": "r1", "text": "t1", "note": "n"},
        {"text": "t2", "id": "1"},
        {"id": "300.0", "text": "t3"},
    ]
    assert gold == [
        {
            "id": "r1",
            "gt_label": "a",
            "text": "t1",
            "provenance": "human_verified",
            "source": "Reuters",
        },
        {"id": "300.0", "gt_label": "b", "text": "t3"},
    ]
    assert unlabeled == ["1"]
    inputs, gold, _ = L.split_gold(rows, "gt_label", id_prefix="seed-", int_ids=True)
    assert [i["id"] for i in inputs] == ["r1", "seed-1", "300"] and gold[1]["id"] == "300"
    assert (
        "text" not in gold[0] and "_bootstrap" not in gold[0]
    )  # a null provenance field is not copied
    assert L.split_gold([], "gt_label") == ([], [], [])


def test_is_kfold_file(tmp_path):
    fold = tmp_path / "dev_fold.jsonl"
    fold.write_text("x")
    assert L.is_kfold_file(str(fold)) is False
    (tmp_path / "manifest.json").write_text(
        json.dumps({"sha256": {"dev_fold.jsonl": hashlib.sha256(b"x").hexdigest()}})
    )
    assert L.is_kfold_file(str(fold)) is True
    (tmp_path / "manifest.json").write_text("[]")
    assert L.is_kfold_file(str(fold)) is False


# ---- demos ------------------------------------------------------------------------------------


def test_select_demo_rows():
    rows = [
        {"id": i, "text": f"t{i}", "gt_label": lab}
        for i, lab in enumerate(["a", "b", "a", None, "b"])
    ]
    demos = L.select_demo_rows(rows, "classification", 1, "gt_label", "label")
    assert sorted(d["gt_label"] for d in demos) == ["a", "b"] and all("text" in d for d in demos)
    assert all(d["gt_label"] == d["gt_label"] and "gt_label" in d for d in demos)
    other = L.select_demo_rows(
        [dict(r, category=r["gt_label"]) for r in rows], "classification", 1, "category", "label"
    )
    assert all(d["gt_label"] == d["category"] for d in other)  # the fallback gold field is set
    ner = [
        {"id": i, "text": "x", "gt_entities": [{"text": "x", "type": t}]}
        for i, t in enumerate(["PER", "ORG", "PER"])
    ]
    picked = L.select_demo_rows(ner, "ner", 1, "gt_entities", "entities")
    assert len(picked) == 2 and all(d["gt_entities"] == d["gt_entities"] for d in picked)
    first = L.select_demo_rows(rows, "qa", 2, "gt_label", "answer")
    assert [d["id"] for d in first] == [0, 1] and first[0]["gt_answer"] == "a"
    assert L.select_demo_rows(rows, "qa", 0, "gt_label", "answer") == []
    assert L.text_key({"text": "a", "q": 1}, ("text", "q", "missing")) == ('"a"', "1", "null")


# ---- the job's rows -----------------------------------------------------------------------


def test_labels_from_job():
    rows = [
        {
            "id": 1,
            "label": "a",
            "confidence": 0.9,
            "ballots": [{"model": "m", "label": "a"}, "junk"],
        },
        {"id": 2, "label": None, "prediction_error": "boom", "confidence": True},
        {"id": 3, "label": None, "_failed_reason": "kept"},
        {"id": 4, "label": "b", "_labeled_by": "people", "confidence": 0.5, "ballots": []},
        {
            "id": 5,
            "label": "c",
            "confidence_score": 0.1,
            "confidence": 0.2,
            "jury_votes": ["kept"],
            "ballots": [{}],
        },
    ]
    out = L.labels_from_job(rows, "classification")
    assert out[0] == {
        "id": 1,
        "label": "a",
        "confidence": 0.9,
        "confidence_score": 0.9,
        "ballots": [{"model": "m", "label": "a"}, "junk"],
        "jury_votes": [{"model": "m", "label": "a"}, "junk"],
    }
    assert out[1] == {
        "id": 2,
        "label": None,
        "prediction_error": "boom",
        "confidence": True,
        "_failed": True,
        "_failed_reason": "boom",
    }
    assert out[2] == {"id": 3, "label": None, "_failed_reason": "kept", "_failed": True}
    assert out[3] == rows[3]  # a person's row is left alone
    assert out[4] == {**rows[4]}
    assert rows[0] == {
        "id": 1,
        "label": "a",
        "confidence": 0.9,
        "ballots": [{"model": "m", "label": "a"}, "junk"],
    }
    ner = L.labels_from_job(
        [{"id": 1, "entities": [], "ballots": [{"model": "m", "entities": []}]}], "ner"
    )
    assert ner[0]["jury_votes"] == [{"model": "m", "entities": [], "label": []}]
    assert L.jury_vote("junk", "label") == "junk" and L.jury_vote(
        {"label": "a", "score": 1}, "score"
    ) == {"label": "a", "score": 1}


def test_ballots_and_juror_shortfall():
    assert L.ballot_cast({"model": "m", "label": "a"}) and not L.ballot_cast(
        {"model": "m", "label": None}
    )
    assert not L.ballot_cast({"model": "m", "stated_confidence": 0.5}) and not L.ballot_cast("junk")
    assert L.ballot_cast({"model": "m", "entities": []}) and not L.ballot_cast(
        {"model": "m", "entities": None}
    )
    assert not L.ballot_cast({"model": "m", "entities": "x", "label": "a"})
    assert L.degraded_jurors({"m": 1}, 0) == {} and L.degraded_jurors({"m": 1, "n": 0}, 10) == {
        "m": {"missing": 1, "rows": 10, "missing_rate": 0.1}
    }
    assert L.degraded_jurors({"m": 1}, 100) == {}  # 1 percent is within the rate
    rows = [
        {"ballots": [{"model": "m1", "label": "a"}, {"model": "m2", "label": None}]},
        {"ballots": [{"model": "m1", "label": "a"}, {"model": "m3", "label": "b"}, None]},
        {"ballots": [{"model": "m1", "label": "a"}], "_labeled_by": "people"},
        {"label": "a"},
    ]
    assert L.juror_shortfall(rows, ["m2"]) == {
        "rows": 2,
        "ballot_unit": "juror",
        "missing_by_juror": {"m2": 2, "m1": 0, "m3": 1},
        "degraded_jurors": {
            "m2": {"missing": 2, "rows": 2, "missing_rate": 1.0},
            "m3": {"missing": 1, "rows": 2, "missing_rate": 0.5},
        },
    }
    assert L.juror_shortfall(rows[2:], ["m1"]) is None and L.juror_shortfall([], None) is None


def test_review_model_outputs(tmp_path):
    side = tmp_path / "review.model_output.jsonl"
    side.write_text(
        '{"id": 1, "model_output": {"label": "a"}}\n\n{"id": 2, "model_output": "x"}\n[1]\n{"id": 3}\n'
    )
    assert L.review_model_outputs(str(side)) == [{"label": "a"}]


def test_held_out_report_from_job():
    assert L.held_out_report_from_job({}, {}, "classification") is None
    assert L.held_out_report_from_job({"HeldOutRows": 10}, None, "classification") is None
    summary = {
        "HeldOutRows": 10,
        "HeldOutAccuracy": 0.9,
        "HeldOutLowerBound95": 0.7,
        "Status": "Certified",
        "StatusReason": None,
        "AcceptedHeldOutAccuracy": 0.95,
        "AcceptedLowerBound95": 0.8,
        "AcceptedHeldOutRows": 8,
        "AcceptThreshold": 0.6,
        "SpotCheckRows": 19,
        "SpotCheckWithdrawAt": 7,
    }
    report = L.held_out_report_from_job(
        summary,
        {"macro_f1": 0.8, "per_class_f1": {"a": 1.0}, "confusion_pairs": [["a", "b"]]},
        "classification",
    )
    assert report == {
        "primary": 0.9,
        "primary_name": "accuracy",
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "held_out_accuracy": 0.9,
        "lower_bound_95": 0.7,
        "breakdown": {"a": 1.0},
        "confusion_pairs": [["a", "b"]],
        "counts": {"correct": 9, "total": 10},
        "accepted": {"accuracy": 0.95, "lower_bound_95": 0.8, "rows": 8},
        "status": "Certified",
        "status_reason": None,
        "accept_threshold": 0.6,
        "spot_check_rows": 19,
        "spot_check_withdraw_at": 7,
        "task": "classification",
        "job": summary,
    }
    ner = L.held_out_report_from_job(summary, {"macro_f1": 0.8, "per_type_f1": {"PER": 0.5}}, "ner")
    assert (
        ner["macro_f1"] is None
        and ner["breakdown"] == {"PER": 0.5}
        and ner["confusion_pairs"] == []
    )
    graded = L.held_out_report_from_job(
        {"GoodRate": 0.75, "GradedRows": 8, "Status": "Done", "GraderModelIds": ["g"]},
        None,
        "summarization",
    )
    assert graded["primary"] == 0.75 and graded["primary_name"] == "good_rate"
    assert graded["counts"] == {"good": 6, "total": 8} and graded["graded"]["GraderModelIds"] == [
        "g"
    ]
    assert graded["graded"]["GradeFailedRows"] is None and graded["task"] == "summarization"
    assert L.held_out_report_from_job({"GradedRows": 8}, None, "summarization") is None


# ---- the record -----------------------------------------------------------------------------


def test_s3_paths_and_manifest():
    paths = L.s3_paths(
        "s3://b/spec.md",
        "s3://b/in",
        None,
        "s3://b/d",
        "s3://b/out/",
        "s3://b/i.json",
        "out.jsonl",
        {"Labels": "s3://b/l", "Report": "s3://b/r", "LabelingSummary": None},
    )
    assert [p["name"] for p in paths] == [
        "Labeling spec",
        "Input data",
        "Demos",
        "Job output",
        "Labeling index",
        "Job Labels",
        "Labeled dataset",
    ]
    results = {
        "summary": {
            "Status": "Certified",
            "RowsLabeled": 5,
            "GoodRate": None,
            "Unknown": 1,
            "Optimizer": {},
        },
        "index_uri": "s3://b/i.json",
        "labeled_dataset_arn": "arn:d",
        "review_dataset_arn": "arn:r",
        "artifact_uris": {"Failures": "s3://b/f"},
    }
    m = L.build_job_manifest(
        task="classification",
        n_rows=5,
        stage="corpus",
        region="us-west-2",
        job={"job_arn": "arn:j", "job_name": "j", "output_s3_uri": "s3://b/out/"},
        results=results,
        request={"models": ["m1", "m2"], "routing": {"x": 1}},
        labels_path="out.jsonl",
        spec_uri="s3://b/spec.md",
        input_ref="s3://b/in",
        eval_uri="s3://b/e",
        demos_uri=None,
        scratch=False,
    )
    (card,) = m["jobs"]
    assert card["type"] == "Labeling" and card["arn"] == "arn:j" and card["model"] == "m1"
    assert (
        card["description"]
        == "SDG labeling job (corpus) over 5 rows for task 'classification': the jury, routed to review"
    )
    assert (
        card["dataset_arn"] == "arn:d"
        and card["review_dataset_arn"] == "arn:r"
        and card["tokens"] is None
    )
    assert [p["name"] for p in card["s3_paths"]] == [
        "Labeling spec",
        "Input data",
        "Evaluation data",
        "Job output",
        "Labeling index",
        "Job Failures",
        "Labeled dataset",
    ]
    assert m["report_s3_paths"] == card["s3_paths"] == m["run_report"]["s3_paths"]
    assert (
        m["run_report"]["description"]
        == "Auto-labeling (corpus): 5 rows labeled for task 'classification' by one SDG labeling job"
    )
    assert m["job_summary"] == {
        "Status": "Certified",
        "RowsLabeled": 5,
        "GoodRate": None,
        "Optimizer": {},
    }
    assert (m["region"], m["task"], m["rows"], m["stage"], m["scratch"]) == (
        "us-west-2",
        "classification",
        5,
        "corpus",
        False,
    )
    scratch = L.build_job_manifest(
        task="ner",
        n_rows=1,
        stage="dev",
        region="r",
        job={},
        results=None,
        request={},
        labels_path="o",
        spec_uri="s",
        input_ref="i",
        eval_uri=None,
        demos_uri=None,
        scratch=True,
    )
    assert (
        scratch["jobs"][0]["type"] == "Labeling (scratch)" and scratch["jobs"][0]["model"] is None
    )
    assert (
        scratch["jobs"][0]["description"].endswith("for task 'ner': one model")
        and scratch["job_summary"] == {}
    )


def test_request_sha256(tmp_path):
    f = tmp_path / "corpus.jsonl"
    f.write_text('{"id": 1}\n')
    a = L.request_sha256({"a": 1}, spec_text="s")
    assert a == L.request_sha256({"a": 1}, spec_text="s") and len(a) == 64
    assert a != L.request_sha256({"a": 1}, spec_text="s2") and a != L.request_sha256(
        {"a": 2}, spec_text="s"
    )
    assert a != L.request_sha256({"a": 1}, spec_text="s", demos_text="d")
    by_file = L.request_sha256({"a": 1}, spec_text="s", input_file=str(f))
    assert by_file == L.request_sha256({"a": 1}, spec_text="s", input_file=str(f))
    f.write_text('{"id": 1}\n{"id": 2}\n')
    assert by_file != L.request_sha256({"a": 1}, spec_text="s", input_file=str(f))
    ident = L.file_identity(str(f))
    assert ident["path"] == str(f) and ident["size"] == os.path.getsize(f) and "mtime_ns" in ident
    assert L.sha256_text(None) is None and L.sha256_text("x") == hashlib.sha256(b"x").hexdigest()
