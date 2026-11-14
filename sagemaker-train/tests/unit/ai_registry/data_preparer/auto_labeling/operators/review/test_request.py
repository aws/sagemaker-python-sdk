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
"""Unit tests for operators.auto_labeling.label_review: a label review's checks, request and read-back."""

from __future__ import absolute_import

import csv
import io
import json

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import decisions as D
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import request as R

MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def _rows(prefix, n, **extra):
    return [
        {
            "id": f"{prefix}{i}",
            "text": f"{prefix} ticket {i}",
            "gt_label": "a" if i % 2 else "b",
            **extra,
        }
        for i in range(n)
    ]


def _state(dev, val, test, label_field="gt_label"):
    return {
        "id_field": "id",
        "label_field": label_field,
        "splits": {
            s: {"ids": [r["id"] for r in rows]}
            for s, rows in (("dev", dev), ("val", val), ("test", test))
        },
    }


def _args(**kw):
    base = dict(
        s3_prefix="s3://b/p/",
        preds=[],
        scope="seed",
        judge_model_id=None,
        estimate_only=False,
        max_judge_rows=None,
        judge_scope=None,
        label_field="gt_label",
        fields=["text"],
    )
    base.update(kw)
    return base


# ---- client checks ---------------------------------------------------------------------------


def test_arguments_pass():
    R.check_review_arguments(**_args())
    R.check_review_arguments(
        **_args(judge_model_id=MODEL, estimate_only=True, max_judge_rows=5000, judge_scope="all")
    )


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"s3_prefix": "nope"}, "set --s3-prefix"),
        ({"s3_prefix": None}, "set --s3-prefix"),
        ({"preds": ["a"] * 5}, "--preds takes at most 4 local files"),
        ({"scope": "golden", "preds": ["a"]}, "--scope golden sends only test rows"),
        ({"estimate_only": True}, "need --judge-model-id"),
        ({"max_judge_rows": 3}, "need --judge-model-id"),
        ({"judge_scope": "all"}, "need --judge-model-id"),
        ({"judge_model_id": MODEL, "max_judge_rows": 0}, "--max-judge-rows must be 1 to 5000"),
        ({"judge_model_id": MODEL, "max_judge_rows": 5001}, "--max-judge-rows must be 1 to 5000"),
        ({"label_field": "text"}, "--label-field 'text' is a text field, `id` or `split`"),
        ({"label_field": "id"}, "is a text field"),
        ({"label_field": "split"}, "is a text field"),
    ],
)
def test_arguments_refuse_before_any_upload(kw, match):
    with pytest.raises(DataPreparerOperatorError, match=match) as info:
        R.check_review_arguments(**_args(**kw))
    assert str(info.value).endswith(". Nothing was uploaded.")


def test_file_size_checks():
    R.check_file_size("p.jsonl", 10, "x", 100, "--preds")
    for path, size, text, match in (
        ("s3://b/x", 10, "x", "pass a local file"),
        ("missing", None, "", "pass a local file"),
        ("e.jsonl", 0, "  \n", "'e.jsonl' is empty"),
        (
            "big.jsonl",
            101,
            "x",
            "'big.jsonl' is 101 bytes; the review job takes a non-empty file of at most 100 bytes",
        ),
    ):
        with pytest.raises(DataPreparerOperatorError, match=match) as info:
            R.check_file_size(path, size, text, 100, "--preds")
        assert str(info.value).startswith("--preds ") and "Nothing was uploaded" in str(info.value)


# ---- the request -----------------------------------------------------------------------------


def test_review_config_sends_only_what_is_given():
    assert R.review_config(label_field="gt_label", scope="seed", pred_uris=[], rules_uri=None) == {
        "LabelField": "gt_label",
        "Scope": "seed",
    }
    assert R.review_config(
        label_field="gt_label",
        scope="golden",
        pred_uris=["s3://b/p0"],
        rules_uri="s3://b/r",
        judge_model_id=MODEL,
        judge_scope="candidates",
        max_judge_rows=10,
        estimate_only=True,
    ) == {
        "LabelField": "gt_label",
        "Scope": "golden",
        "PredictionsS3Uris": ["s3://b/p0"],
        "RulesS3Uri": "s3://b/r",
        "JudgeModels": [{"ModelId": MODEL}],
        "JudgeScope": "candidates",
        "MaxJudgeRows": 10,
        "EstimateOnly": True,
    }
    # judge settings without a judge are not sent
    assert "JudgeScope" not in R.review_config(
        label_field="l",
        scope="seed",
        pred_uris=[],
        rules_uri=None,
        judge_scope="all",
        max_judge_rows=3,
    )


def test_seed_text_keeps_only_what_the_job_reads():
    rows = [
        {"id": "d0", "text": "t", "gt_label": "a", "note": "private", "split": "dev"},
        {"id": "d1", "text": "u"},
    ]
    assert R.seed_text(rows, "gt_label", ["text"]) == (
        '{"id": "d0", "split": "dev", "gt_label": "a", "text": "t"}\n{"id": "d1", "text": "u"}\n'
    )
    assert (
        R.seed_text([{"id": "x", "text": "é"}], "gt_label", ["text"])
        == '{"id": "x", "text": "é"}\n'
    )


def test_digests():
    content, sha = R.content_digest("spec", "seed", [b"p0", b"p1"], b"rules")
    assert content == {
        "spec": _digest.sha256_hex("spec"),
        "seed": _digest.sha256_hex("seed"),
        "preds": [_digest.sha256_hex(b"p0"), _digest.sha256_hex(b"p1")],
        "rules": _digest.sha256_hex(b"rules"),
    }
    assert sha == _digest.sha256_hex(json.dumps(content, sort_keys=True)) and len(sha) == 64
    assert R.content_digest("spec", "seed", [])[0]["rules"] is None
    assert R.request_digest({"a": 1}, sha) == _digest.sha256_hex(
        json.dumps({"request": {"a": 1}, "content": sha}, sort_keys=True)
    )
    assert R.request_digest({"a": 1}, sha) != R.request_digest({"a": 2}, sha)
    assert _digest.sha256_hex("x") == _digest.sha256_hex(b"x")


def test_check_resume():
    R.check_resume(None, "x", run_id="r1", mode="checks", record="rec")
    R.check_resume({"request_sha256": "x"}, "x", run_id="r1", mode="checks", record="rec")
    with pytest.raises(DataPreparerOperatorError) as info:
        R.check_resume(
            {"request_sha256": "y"}, "x", run_id="r1", mode="judge", record="/p/r1-seed-judge.json"
        )
    assert str(info.value) == (
        "--run-id 'r1' already names a judge review job submitted with a different request, spec, seed, "
        "predictions or rules (/p/r1-seed-judge.json); rerun with --resubmit or a new --run-id. Nothing was "
        "uploaded."
    )


# ---- the rows to send ------------------------------------------------------------------------


def _review(rows, **kw):
    base = dict(scope="seed", label_field="gt_label", fields=["text"], seed_name="seed.jsonl")
    base.update(kw)
    return R.review_rows(rows, **base)


def test_seed_scope_without_a_split_leaves_split_test_rows_out():
    rows = _rows("d", 4, split="dev") + _rows("v", 2, split="val") + _rows("t", 3, split=" TEST ")
    keep, test, step, allowed = _review(rows)
    assert [r["id"] for r in keep] == [r["id"] for r in rows[:6]]
    assert test == {"t0", "t1", "t2"} and (step, allowed) == ("label-review", {"dev", "val"})
    assert keep[0]["split"] == "dev"  # the row's own column stays without a frozen split


def test_seed_scope_with_a_frozen_split_names_each_rows_split():
    dev, val, test = _rows("d", 4), _rows("v", 2), _rows("t", 3)
    st = _state(dev, val, test)
    rows = [dict(r, split="bogus") for r in dev + val] + [
        {"id": "x", "text": "stray", "gt_label": "a"}
    ]
    gold = {r["id"]: r["gt_label"] for r in dev + val}
    keep, test_ids, step, allowed = _review(rows, state=st, split_gold=gold)
    assert test_ids == set() and len(keep) == 7
    assert {r["id"]: r.get("split") for r in keep} == {
        **{r["id"]: "dev" for r in dev},
        **{r["id"]: "val" for r in val},
        "x": None,
    }
    assert "split" not in keep[-1]  # a row in no split carries no split


def test_golden_scope_sends_the_frozen_test_rows_only():
    dev, val, test = _rows("d", 4), _rows("v", 2), _rows("t", 3)
    st = _state(dev, val, test)
    keep, test_ids, step, allowed = _review(
        dev + test, scope="golden", state=st, split_gold={r["id"]: r["gt_label"] for r in test}
    )
    assert [r["id"] for r in keep] == ["t0", "t1", "t2"] and all(r["split"] == "test" for r in keep)
    assert test_ids == {"t0", "t1", "t2"} and (step, allowed) == ("golden-review", {"test"})
    with pytest.raises(DataPreparerOperatorError, match="freeze the split first") as info:
        _review(test, scope="golden")
    assert "Nothing was uploaded" not in str(info.value)


@pytest.mark.parametrize(
    "rows,kw,match",
    [
        ([{"text": "t", "gt_label": "a"}], {}, "every row needs a unique, non-empty `id`"),
        ([{"id": "", "text": "t", "gt_label": "a"}], {}, "unique, non-empty `id`"),
        (_rows("d", 2) + _rows("d", 1), {}, "unique, non-empty `id`"),
        (
            _rows("d", 2),
            {"label_field": "label", "state": _state([], [], [])},
            "--label-field 'label' is not the frozen split's label field 'gt_label'; pass --label-field gt_label",
        ),
        ([dict(r, split="test") for r in _rows("d", 3)], {}, "has no dev or val row to review"),
        (
            _rows("d", 2),
            {"scope": "golden", "state": _state(_rows("d", 2), [], [])},
            "has no test row to review",
        ),
        (
            [{"id": f"d{i}", "text": f"t{i}"} for i in range(3)],
            {},
            "no row has the gold field 'gt_label'",
        ),
    ],
)
def test_rows_refusals(rows, kw, match):
    with pytest.raises(DataPreparerOperatorError, match=match) as info:
        _review(rows, **kw)
    assert str(info.value).endswith("Nothing was uploaded.")


def test_stale_gold_and_row_limits(monkeypatch):
    dev = _rows("d", 4)
    st = _state(dev, [], [])
    gold = {r["id"]: r["gt_label"] for r in dev}
    rows = [dict(r, gt_label="b") if r["id"] in ("d1", "d3") else r for r in dev]
    with pytest.raises(
        DataPreparerOperatorError,
        match=r"2 row\(s\) have another gold label than the frozen split "
        r"\(e.g. \['d1', 'd3'\]\)",
    ):
        _review(rows, state=st, split_gold=gold)
    monkeypatch.setattr(R, "MAX_ROWS", 3)
    with pytest.raises(
        DataPreparerOperatorError, match="4 rows to review; the job takes at most 3"
    ):
        _review(dev)
    monkeypatch.setattr(R, "MAX_ROWS", 50000)
    monkeypatch.setattr(R, "MAX_ROWS_PER_TEXT", 2)
    same = [
        {"id": f"d{i}", "text": "Same  TEXT" if i else "same text", "gt_label": "a"}
        for i in range(3)
    ]
    with pytest.raises(
        DataPreparerOperatorError, match="3 rows share one text; the job takes at most 2 per text"
    ):
        _review(same)
    # rows with no text are not one text
    empty = [{"id": f"d{i}", "text": "", "gt_label": "a"} for i in range(3)]
    assert len(_review(empty)[0]) == 3


def test_test_predictions():
    assert R.test_predictions([{"id": "t1"}, {"id": "d0"}, {"id": 2}], {"t1", "2", "t9"}) == [
        "2",
        "t1",
    ]
    assert R.test_predictions([], {"t1"}) == []


# ---- approvals and entries -------------------------------------------------------------------


def _analysis(status, **f):
    return {"kind": "analysis", "step": "label-review", **f, "status": status}


def test_unused_approval():
    settings = dict(judge_model_id=MODEL, judge_scope="candidates", max_judge_rows=None)
    est = _analysis("estimate", id="e1", content_sha256="sha", **settings)
    yes = {
        "kind": "approval",
        "step": "label-review",
        "id": "a1",
        "status": "approved",
        "estimate": "e1",
    }
    entries = [est, yes]
    assert R.unused_approval(entries, "label-review", "sha", **settings) == "a1"
    assert R.unused_approval(entries, "label-review", "other", **settings) is None
    assert (
        R.unused_approval(entries, "label-review", "sha", **dict(settings, judge_scope="all"))
        is None
    )
    assert R.unused_approval(entries, "golden-review", "sha", **settings) is None
    assert (
        R.unused_approval([est, dict(yes, status="pending")], "label-review", "sha", **settings)
        is None
    )
    # used from the submitting entry on; a refused submission gives it back
    assert (
        R.unused_approval(
            entries + [_analysis("submitting", approval="a1")], "label-review", "sha", **settings
        )
        is None
    )
    assert (
        R.unused_approval(
            entries + [_analysis("refused", approval="a1")], "label-review", "sha", **settings
        )
        == "a1"
    )
    assert (
        R.unused_approval(
            entries + [_analysis("submitting", approval="a1"), _analysis("refused", approval="a1")],
            "label-review",
            "sha",
            **settings,
        )
        == "a1"
    )
    # the newest unused approval wins
    yes2 = dict(yes, id="a2")
    assert R.unused_approval([est, yes, yes2], "label-review", "sha", **settings) == "a2"
    assert (
        R.unused_approval(
            [est, yes, yes2, _analysis("submitting", approval="a2")],
            "label-review",
            "sha",
            **settings,
        )
        == "a1"
    )


def test_messages_and_steps():
    assert R.step_for("seed") == "label-review" and R.step_for("golden") == "golden-review"
    assert R.judge_needs_approval_message("golden").startswith(
        "the judge runs only after its --estimate-only job for --scope golden,"
    )
    assert R.judge_needs_approval_message("seed").endswith(" Nothing was uploaded.")
    assert R.rerun_hint("judge").startswith(
        "Rerun with --resubmit or a new --run-id: that starts a second, billed"
    )
    assert R.rerun_hint("checks") == "Rerun with --resubmit or a new --run-id."
    assert R.job_ended_message(
        {"job_name": "j", "job_status": "Failed", "failure_reason": "boom"}
    ) == ("review job j ended Failed: boom.")
    assert R.job_ended_message({"job_name": "j", "job_status": "Stopped"}) == (
        "review job j ended Stopped: no failure reason returned."
    )
    assert R.job_ended_message({"job_name": "j", "job_status": "Completed"}) is None
    assert R.output_files("seed") == [
        "contradictions.jsonl",
        "contradictions.csv",
        "flags.jsonl",
        "summary.json",
        "report.md",
    ]
    assert R.output_files("golden") == ["flags.jsonl", "summary.json", "routed.jsonl"]


def test_events():
    assert R.job_event("label-review", 1, "submitted", "/p/rec.json", job_name="j") == {
        "kind": "analysis",
        "step": "label-review",
        "iteration": 1,
        "status": "submitted",
        "record": "/p/rec.json",
        "job_name": "j",
    }
    assert R.job_event("golden-review", 0, "submitting", "r", approval="a1")["approval"] == "a1"
    assert "approval" not in R.job_event("golden-review", 0, "failed", "r", reason="x")
    ev = R.review_event(
        "label-review",
        2,
        estimate_only=True,
        job_name="j",
        run_id="r1",
        judge_model_id=MODEL,
        judge_scope="candidates",
        max_judge_rows=None,
        content_sha="sha",
        inputs={"spec": "s"},
        estimate={"rows": 2},
        summary_path="/p/audit/summary.json",
    )
    assert ev == {
        "kind": "analysis",
        "step": "label-review",
        "iteration": 2,
        "status": "estimate",
        "job_name": "j",
        "run_id": "r1",
        "judge_model_id": MODEL,
        "judge_scope": "candidates",
        "max_judge_rows": None,
        "content_sha256": "sha",
        "inputs": {"spec": "s"},
        "estimate": {"rows": 2},
        "summary": "/p/audit/summary.json",
    }
    assert (
        R.review_event(
            "label-review",
            2,
            estimate_only=False,
            job_name="j",
            run_id="r1",
            judge_model_id=None,
            judge_scope="candidates",
            max_judge_rows=None,
            content_sha="sha",
            inputs={},
            estimate=None,
            summary_path="s",
        )["status"]
        == "flagged"
    )


# ---- the read-back ---------------------------------------------------------------------------


def _listing_s3(keys):
    """An S3 client whose list_objects_v2 paginator serves the keys under each prefix."""

    class _S3:
        def __init__(self):
            self.prefixes = []

        def get_paginator(self, op):
            assert op == "list_objects_v2"
            outer = self

            class _P:
                def paginate(self, Bucket, Prefix):  # noqa: N803
                    assert Bucket == "b"
                    outer.prefixes.append(Prefix)
                    yield {"Contents": [{"Key": k} for k in keys if k.startswith(Prefix)]}

            return _P()

    return _S3()


def test_review_folder(monkeypatch):
    job = "autolabel-review-7"
    new = f"p/out/synthetic-data-generation/{job}/2026-11-09T18Z"
    old = f"p/out/{job}-2026-10-01T00Z"
    sibling_new = f"p/out/synthetic-data-generation/{job}0/2026-11-09T19Z"
    sibling_old = f"p/out/{job}0-2026-10-01T01Z"

    # the job folder (new layout), listed ahead of a sibling job's
    s3 = _listing_s3(
        [
            f"{sibling_new}/review/summary.json",
            f"{new}/labeling/spec.md",
            f"{new}/review/summary.json",
        ]
    )
    monkeypatch.setattr(R._s3, "s3_client", lambda region: s3)
    assert R.review_folder("s3://b/p/out/", job, "us-west-2") == f"s3://b/{new}/review/"
    assert s3.prefixes == [f"p/out/synthetic-data-generation/{job}/"]

    # a job that started before the move: the earlier layout, never the sibling's
    s3 = _listing_s3([f"{sibling_old}/review/summary.json", f"{old}/review/summary.json"])
    monkeypatch.setattr(R._s3, "s3_client", lambda region: s3)
    assert R.review_folder("s3://b/p/out", job, "us-west-2") == f"s3://b/{old}/review/"
    assert s3.prefixes == [f"p/out/synthetic-data-generation/{job}/", f"p/out/{job}-"]

    # only a sibling job's folder in each layout: not found
    s3 = _listing_s3([f"{sibling_new}/review/summary.json", f"{sibling_old}/review/summary.json"])
    monkeypatch.setattr(R._s3, "s3_client", lambda region: s3)
    with pytest.raises(FileNotFoundError, match=f"synthetic-data-generation/{job}/ or "):
        R.review_folder("s3://b/p/out/", job, "us-west-2")


def test_check_summary():
    ok = {"review_contract_version": 1, "items": 1, "judge": {"status": "off"}}
    R.check_summary(ok, scope="seed", judge_model_id=None, estimate_only=False)
    R.check_summary(
        dict(ok, judge={"status": "ran", "model_id": MODEL}),
        scope="seed",
        judge_model_id=MODEL,
        estimate_only=False,
    )
    R.check_summary(
        dict(ok, judge={"status": "estimate", "model_id": MODEL, "estimate": {"rows": 2}}),
        scope="seed",
        judge_model_id=MODEL,
        estimate_only=True,
    )
    R.check_summary(
        dict(ok, routed_rows=2),
        scope="golden",
        judge_model_id=None,
        estimate_only=False,
        routed_rows=2,
    )
    for summary, kw, match in (
        (
            dict(ok, review_contract_version=2),
            {},
            "wrote contract version 2; this skill reads version 1",
        ),
        (
            ok,
            {"judge_model_id": MODEL},
            f"judge status 'off' for model None is not what --judge-model-id {MODEL} "
            "asked for \\('ran'\\)",
        ),
        (
            dict(ok, judge={"status": "ran", "model_id": "other"}),
            {"judge_model_id": MODEL},
            "for model 'other'",
        ),
        (
            dict(ok, judge={"status": "estimate", "model_id": MODEL}),
            {"judge_model_id": MODEL, "estimate_only": True},
            "asked for \\('estimate'\\)",
        ),
        (
            dict(ok, routed_rows=2),
            {"scope": "golden", "routed_rows": 1},
            "summary says 2 routed rows but routed.jsonl has 1; no person should see these rows",
        ),
        (
            ok,
            {"scope": "golden", "routed_rows": 0},
            "summary says None routed rows but routed.jsonl has 0",
        ),
    ):
        args = dict(scope="seed", judge_model_id=None, estimate_only=False)
        args.update(kw)
        with pytest.raises(DataPreparerOperatorError, match=match):
            R.check_summary(summary, **args)


ITEM = {
    "item_id": "keyword_rules:refund-1a2b3c4d",
    "check": "keyword_rules",
    "title": "=refund means b",
    "member_ids": ["d0", "d1"],
    "proposed_changes": {"d0": "b", "d1": "b"},
}


def _csv_rows(text):
    return list(csv.DictReader(io.StringIO(text)))


def test_decision_csv_fills_from_the_log_and_keeps_rows_of_the_same_item_version():
    items = [ITEM, dict(ITEM, item_id="other", title="t2")]
    ledger = {
        "other": {
            "decision": "neither",
            "if_neither": "b",
            "sop_wording": None,
            "overrides": {"d1": "keep"},
        }
    }
    rows, prev, undecided, apart = R.decision_csv(items, ledger)
    assert rows == [
        {
            "decision": "",
            "if_neither": "",
            "sop_wording": "",
            "overrides": "",
            "item_id": ITEM["item_id"],
            "title": "=refund means b",
            "item_sha256": D.item_sha(ITEM),
        },
        {
            "decision": "neither",
            "if_neither": "b",
            "sop_wording": "",
            "overrides": "d1=keep",
            "item_id": "other",
            "title": "t2",
            "item_sha256": D.item_sha(items[1]),
        },
    ]
    assert (prev, undecided, apart) == (None, 1, 0)
    text = R.csv_text(rows)
    assert text.startswith(
        "item_id,title,decision,if_neither,sop_wording,overrides,item_sha256\r\n"
    )
    assert "'=refund means b" in text  # escaped against a formula
    # the user fills the first row; the second item changes; a stray filled row is kept apart
    current = _csv_rows(text.replace(",,,,,", ",guidelines_correct,,,,", 1))
    current.append(
        {
            "item_id": "gone",
            "title": "x",
            "decision": "neither",
            "if_neither": "b",
            "sop_wording": "",
            "overrides": "",
            "item_sha256": "dead",
        }
    )
    changed = dict(items[1], member_ids=["d0", "d1", "v0"])
    rows2, prev2, undecided2, apart2 = R.decision_csv(
        [ITEM, changed], ledger, current, [{"item_id": "older"}]
    )
    assert rows2[0]["decision"] == "guidelines_correct" and rows2[0]["title"] == "=refund means b"
    # the changed item: its version differs from the CSV's, so the log's decision applies only if its sha matches
    assert rows2[1] == {
        "decision": "neither",
        "if_neither": "b",
        "sop_wording": "",
        "overrides": "d1=keep",
        "item_id": "other",
        "title": "t2",
        "item_sha256": D.item_sha(changed),
    }
    # the filled row of the changed item and the stray row both go apart
    assert prev2 == [{"item_id": "older"}, current[1], current[2]] and (undecided2, apart2) == (
        0,
        2,
    )
    # a blank row for the same item version is kept blank even when the log has a decision
    blank = _csv_rows(R.csv_text(R.decision_csv(items, {})[0]))
    rows3, _, undecided3, _ = R.decision_csv(items, ledger, blank)
    assert rows3[1]["decision"] == "" and undecided3 == 2


def test_review_result():
    summary = {
        "items": 2,
        "judge": {"status": "estimate", "estimate": {"rows": 2}},
        "warnings": ["w"],
        "routed_per_check": {"k": 1},
    }
    assert R.review_result(
        job_name="j",
        scope="seed",
        rows_sent=6,
        test_rows_left_out=3,
        summary=summary,
        out_dir="/a",
        decisions={"items": 2, "undecided_items": 1, "decisions_kept_apart": 0},
    ) == {
        "job_name": "j",
        "scope": "seed",
        "rows_sent": 6,
        "test_rows_left_out": 3,
        "items": 2,
        "judge_status": "estimate",
        "estimate": {"rows": 2},
        "warnings": ["w"],
        "out_dir": "/a",
        "undecided_items": 1,
        "decisions_kept_apart": 0,
    }
    golden = R.review_result(
        job_name="j",
        scope="golden",
        rows_sent=3,
        test_rows_left_out=0,
        summary={},
        out_dir="/g",
        routed_rows=1,
    )
    assert (
        golden["routed_rows"] == 1 and golden["routed_per_check"] == {} and golden["warnings"] == []
    )
    assert golden["items"] is None and golden["judge_status"] is None
