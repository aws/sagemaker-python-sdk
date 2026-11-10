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
"""A label review through the labeling job: the checks, the request and the read-back.

A review job is a review-only ``CreateJob`` (``Config.Review``) over the labeled seed rows:
it labels nothing and writes its ``review/`` files (the contradiction list, flags, a summary,
for the golden scope the routed rows). This module holds what the submitting client decides
without touching S3 or the project:

* the client checks on the arguments before anything is read or uploaded
  (:func:`check_review_arguments`) and on the rows to send (:func:`review_rows`);
* the request's ``Config.Review`` (:func:`review_config`), the seed text the job reads
  (:func:`seed_text`), the content and request digests a re-attach compares
  (:func:`content_digest`, :func:`request_digest`);
* the approval a judged job needs (:func:`unused_approval`) and the run-log entries that
  record a run (:func:`job_event`, :func:`review_event`);
* the checks on the job's ``summary.json`` (:func:`check_summary`), the ``decisions.csv``
  rows the seed scope leaves for the user (:func:`decision_csv`) and the result the client
  reports (:func:`review_result`).

:func:`review_folder` is the one S3 call: the job's own ``review/`` folder under its output
prefix. Refusals are :class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`
with the message the client prints. Uploading, submitting, the job record and the split
guard stay with the caller.
"""

from __future__ import absolute_import

import collections
import csv
import io
import json
from typing import Any, Dict, List, Optional

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.decisions import (
    DECISION_COLS,
    RUN_LOG_STEP,
    csv_cell,
    item_sha,
    parse_overrides,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import S3_URI_RE

REVIEW_SUBDIR = "review"
GOLDEN_STEP = "golden-review"
# The job's review/ files per scope (contradictions.csv carries the item cells the user reads).
OUTPUT_FILES = {
    "seed": (
        "contradictions.jsonl",
        "contradictions.csv",
        "flags.jsonl",
        "summary.json",
        "report.md",
    ),
    "golden": ("flags.jsonl", "summary.json", "routed.jsonl"),
}
REVIEW_CONTRACT_VERSION = 1
MAX_PREDICTIONS, MAX_PREDICTIONS_BYTES = 4, 16 * 1024 * 1024
MAX_RULES_BYTES, MAX_JUDGE_ROWS = 256 * 1024, 5000
MAX_ROWS, MAX_ROWS_PER_TEXT = 50000, 500  # the job fails above either
CSV_FIELDS = ["item_id", "title", *DECISION_COLS, "item_sha256"]
NOTHING_UPLOADED = " Nothing was uploaded."


def _refuse(msg):
    raise DataPreparerOperatorError(f"{msg}.{NOTHING_UPLOADED}")


def step_for(scope):
    """The run-log step of a scope: ``label-review`` for the seed, ``golden-review`` for golden."""
    return RUN_LOG_STEP if scope == "seed" else GOLDEN_STEP


def check_review_arguments(  # pylint: disable=too-many-arguments
    *,
    s3_prefix,
    preds,
    scope,
    judge_model_id,
    estimate_only,
    max_judge_rows,
    judge_scope,
    label_field,
    fields,
):
    """The client checks before anything is read or uploaded.

    Raises:
        DataPreparerOperatorError: The first failing check, ending "Nothing was uploaded.".
    """
    for refused, msg in (
        (
            not S3_URI_RE.match(s3_prefix or ""),
            "set --s3-prefix (or AUTOLABEL_S3_PREFIX) to a writable s3://bucket/prefix",
        ),
        (len(preds) > MAX_PREDICTIONS, f"--preds takes at most {MAX_PREDICTIONS} local files"),
        (
            scope == "golden" and preds,
            "--preds are dev or val predictions; --scope golden sends only test rows",
        ),
        (
            not judge_model_id and (estimate_only or max_judge_rows is not None or judge_scope),
            "--estimate-only, --max-judge-rows and --judge-scope need --judge-model-id",
        ),
        (
            max_judge_rows is not None and not 1 <= max_judge_rows <= MAX_JUDGE_ROWS,
            f"--max-judge-rows must be 1 to {MAX_JUDGE_ROWS}",
        ),
        (
            label_field in {"id", "split", *fields},
            f"--label-field {label_field!r} is a text field, `id` or `split`; the job refuses it",
        ),
    ):
        if refused:
            _refuse(msg)


def check_file_size(path, size, text, cap, what):
    """A local, non-empty file within its byte cap, else a refusal before any upload.

    Args:
        path: The path as given.
        size: Its size in bytes, or ``None`` when it is not a local file.
        text: Its text (checked for blankness).
        cap: The byte cap.
        what: The flag, for the message.
    """
    if path.startswith("s3://") or size is None:
        _refuse(f"{what} {path!r}: pass a local file (its content is hashed)")
    if size > cap or not text.strip():
        _refuse(
            f"{what} {path!r} is {f'{size:,} bytes' if size > cap else 'empty'}; the review job "
            f"takes a non-empty file of at most {cap:,} bytes"
        )


def review_config(  # pylint: disable=too-many-arguments
    *,
    label_field,
    scope,
    pred_uris,
    rules_uri,
    judge_model_id=None,
    judge_scope=None,
    max_judge_rows=None,
    estimate_only=False,
):
    """``Config.Review`` as the request sends it; a key is sent only when its flag is given."""
    judge = judge_model_id
    review = {
        "LabelField": label_field,
        "Scope": scope,
        "PredictionsS3Uris": pred_uris,
        "RulesS3Uri": rules_uri,
        "JudgeModels": judge and [{"ModelId": judge}],
        "JudgeScope": judge and judge_scope,
        "MaxJudgeRows": judge and max_judge_rows,
        "EstimateOnly": estimate_only,
    }
    return {k: v for k, v in review.items() if v not in (None, [], False, "")}


def review_rows(  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches
    rows,
    *,
    scope,
    label_field,
    fields,
    seed_name,
    state=None,
    split_gold=None,
    splits=("dev", "val", "test"),
):
    """The rows the job reviews, and the test ids left out.

    Seed scope: no test row (a frozen test row in the seed is a stop, which the caller's
    split guard raises; without a frozen split, rows whose ``split`` is test are left out).
    Golden scope: the frozen test rows only. A row whose gold differs from the frozen
    split's stops the run, as do the job's row limits. With a frozen split, each row's
    ``split`` is the one the state names (none for a row in no split), never the row's own.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import review_rows

        rows_sent, left_out, step, allowed = review_rows(
            seed_rows, scope="seed", label_field="gt_label", fields=["text"], seed_name="seed.jsonl",
            state=load_state("proj/splits"),
        )

    Args:
        rows: The seed rows.
        scope: ``seed`` or ``golden``.
        label_field: The gold field.
        fields: The text fields.
        seed_name: The seed path, for the messages.
        state: The frozen split's state, or ``None``.
        split_gold: ``{id: gold}`` from the split files the scope reads, with a state.
        splits: The split names, in the state's order.

    Returns:
        ``(rows to send, test ids, step, allowed splits)``.

    Raises:
        DataPreparerOperatorError: Duplicate or missing ids, another label field than the
            split's, golden without a split, no row to review, stale gold, or a row limit.
    """
    ids = [r.get("id") for r in rows]
    if any(i in (None, "") for i in ids) or len({str(i) for i in ids}) != len(ids):
        _refuse(f"--seed {seed_name!r}: every row needs a unique, non-empty `id`")
    st = state
    if st and label_field != st["label_field"]:
        _refuse(
            f"--label-field {label_field!r} is not the frozen split's label field "
            f"{st['label_field']!r}; pass --label-field {st['label_field']}"
        )
    if scope == "golden":
        if not st:
            raise DataPreparerOperatorError(
                "--scope golden reviews the frozen test rows: freeze the split first "
                "(split_state.py freeze)."
            )
        test = set(st["splits"]["test"]["ids"])
        keep, step, allowed = [r for r in rows if str(r["id"]) in test], GOLDEN_STEP, {"test"}
    else:
        test = (
            set()
            if st
            else {str(r["id"]) for r in rows if str(r.get("split")).strip().casefold() == "test"}
        )
        keep, step, allowed = (
            [r for r in rows if str(r["id"]) not in test],
            RUN_LOG_STEP,
            {"dev", "val"},
        )
    if not keep:
        _refuse(
            f"--seed {seed_name!r} has no {'test' if scope == 'golden' else 'dev or val'} row to review"
        )
    gold = dict(split_gold or {}) if st else {}
    stale = [
        str(r["id"])
        for r in keep
        if str(r["id"]) in gold and r.get(label_field) != gold[str(r["id"])]
    ]
    same_text = max(
        [
            n
            for k, n in collections.Counter(
                tuple(" ".join(str(r.get(f) or "").split()).casefold() for f in fields)
                for r in keep
            ).items()
            if any(k)
        ]
        or [0]
    )
    if st:
        owner = {i: s for s in splits for i in st["splits"][s]["ids"]}
        keep = [
            {k: v for k, v in dict(r, split=owner.get(str(r["id"]))).items() if k != "split" or v}
            for r in keep
        ]
    for refused, msg in (
        (
            stale,
            f"{len(stale)} row(s) have another gold label than the frozen split (e.g. "
            f"{stale[:3]}); pass the split's rows as they are now (references/label-review.md)",
        ),
        (len(keep) > MAX_ROWS, f"{len(keep):,} rows to review; the job takes at most {MAX_ROWS:,}"),
        (
            same_text > MAX_ROWS_PER_TEXT,
            f"{same_text} rows share one text; the job takes at most {MAX_ROWS_PER_TEXT} per text",
        ),
        (
            all(r.get(label_field) in (None, "") for r in keep),
            f"no row has the gold field {label_field!r}",
        ),
    ):
        if refused:
            _refuse(f"--seed {seed_name!r}: {msg}")
    return keep, test, step, allowed


def test_predictions(pred_rows, test_ids):
    """The test ids a prediction file covers, sorted: a seed-scope review must see none."""
    return sorted(set(test_ids) & {str(r.get("id")) for r in pred_rows})


def seed_text(rows, label_field, fields):
    """The seed as the job reads it: ``id``, ``split``, the gold and the text fields only."""
    keep = ("id", "split", label_field, *fields)
    return "".join(
        json.dumps({k: r[k] for k in keep if k in r}, ensure_ascii=False) + "\n" for r in rows
    )


def content_digest(spec_text, seed, preds_bytes, rules_bytes=None):
    """The inputs' digests and their combined SHA-256.

    Returns:
        ``(content, sha)``: ``content`` is ``{"spec", "seed", "preds": [...], "rules"}``.
    """
    content = {
        "spec": _digest.sha256_hex(spec_text),
        "seed": _digest.sha256_hex(seed),
        "preds": [_digest.sha256_hex(b) for b in preds_bytes],
        "rules": _digest.sha256_hex(rules_bytes) if rules_bytes is not None else None,
    }
    return content, _digest.sha256_json(content, ascii=True)


def request_digest(document, content_sha):
    """The SHA-256 a re-attach compares: the request document and the content digest."""
    return _digest.sha256_json({"request": document, "content": content_sha}, ascii=True)


def check_resume(resume, request_sha, *, run_id, mode, record):
    """Refuse a re-attach whose request, spec, seed, predictions or rules changed."""
    if resume and resume.get("request_sha256") != request_sha:
        _refuse(
            f"--run-id {run_id!r} already names a {mode} review job submitted with a different "
            f"request, spec, seed, predictions or rules ({record}); rerun with --resubmit or a new "
            "--run-id"
        )


def unused_approval(entries, step, content_sha, *, judge_model_id, judge_scope, max_judge_rows):
    """The id of an approved, unused approval naming this scope's estimate for these inputs.

    An approval names an ``estimate`` entry (``analysis``, status ``estimate``) with the same
    content digest and judge settings. It is used from the ``submitting`` entry of the judged
    job it opened on, unless that job was refused.

    Args:
        entries: The project's run-log entries, oldest first.
        step: The scope's step.
        content_sha: :func:`content_digest`'s sha.
        judge_model_id: The judge model.
        judge_scope: ``candidates`` or ``all``.
        max_judge_rows: The row cap.

    Returns:
        The approval entry's id, or ``None``.
    """
    analyses = [e for e in entries if e.get("kind") == "analysis" and e.get("step") == step]
    last = {e.get("approval"): e.get("status") for e in analyses}  # used from `submitting` on
    used = {a for a, s in last.items() if s != "refused"}
    estimates = {
        e["id"]
        for e in analyses
        if e.get("status") == "estimate"
        and e.get("content_sha256") == content_sha
        and (e.get("judge_model_id"), e.get("judge_scope"), e.get("max_judge_rows"))
        == (judge_model_id, judge_scope, max_judge_rows)
    }
    approvals = [e for e in entries if e.get("kind") == "approval" and e.get("step") == step]
    for a in reversed(approvals):
        if a.get("status") == "approved" and a.get("estimate") in estimates and a["id"] not in used:
            return a["id"]
    return None


def judge_needs_approval_message(scope):
    """The refusal when a judged job has no approved estimate to run on."""
    return (
        f"the judge runs only after its --estimate-only job for --scope {scope}, this "
        "judge model and settings, and this spec, seed, preds and rules, and an unused approval "
        f"entry naming that estimate (references/label-review.md, Step 1).{NOTHING_UPLOADED}"
    )


def rerun_hint(mode):
    """What a rerun after a failure costs and needs."""
    return (
        "Rerun with --resubmit or a new --run-id: that starts a second, billed judged job and needs "
        "a new approval (references/label-review.md, Step 1)."
        if mode == "judge"
        else "Rerun with --resubmit or a new --run-id."
    )


def job_event(step, iteration, status, record, approval=None, **fields):
    """A run-log ``analysis`` entry of the review job's submission or failure."""
    return run_log_event(
        "analysis",
        step,
        iteration,
        status,
        record=str(record),
        **({"approval": approval} if approval else {}),
        **fields,
    )


def review_event(  # pylint: disable=too-many-arguments
    step,
    iteration,
    *,
    estimate_only,
    job_name,
    run_id,
    judge_model_id,
    judge_scope,
    max_judge_rows,
    content_sha,
    inputs,
    estimate,
    summary_path,
):
    """The run-log ``analysis`` entry that records a finished review (``estimate`` or ``flagged``)."""
    return run_log_event(
        "analysis",
        step,
        iteration,
        "estimate" if estimate_only else "flagged",
        job_name=job_name,
        run_id=run_id,
        judge_model_id=judge_model_id,
        judge_scope=judge_scope,
        max_judge_rows=max_judge_rows,
        content_sha256=content_sha,
        inputs=inputs,
        estimate=estimate,
        summary=summary_path,
    )


def review_folder(output_uri, job_name, region):
    """The job's own ``review/`` folder, under its job subfolder of ``OutputS3Uri``.

    Raises:
        FileNotFoundError: No ``review/summary.json`` under the job's folder.
    """
    bucket, key = _s3.split_s3_uri(output_uri)
    for page in (
        _s3.s3_client(region)
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix=key + job_name)
    ):
        for obj in page.get("Contents") or []:
            if obj["Key"].endswith(f"/{REVIEW_SUBDIR}/summary.json"):
                return f"s3://{bucket}/{obj['Key'][:-len('summary.json')]}"
    raise FileNotFoundError(f"no {REVIEW_SUBDIR}/summary.json under {output_uri}{job_name}")


def job_ended_message(job):
    """The failure message of a job that ended Failed or Stopped, or ``None``."""
    if job.get("job_status") in ("Failed", "Stopped"):
        return (
            f"review job {job['job_name']} ended {job['job_status']}: "
            f"{job.get('failure_reason') or 'no failure reason returned'}."
        )
    return None


def check_summary(summary, *, scope, judge_model_id, estimate_only, routed_rows=None):
    """The checks on the job's ``summary.json`` before its files are kept.

    Args:
        summary: The parsed ``summary.json``.
        scope: ``seed`` or ``golden``.
        judge_model_id: The judge the run asked for, or ``None``.
        estimate_only: Whether an estimate was asked for.
        routed_rows: For the golden scope, the row count of ``routed.jsonl``.

    Raises:
        DataPreparerOperatorError: Another contract version, a judge status other than the one
            asked for, or a routed count that does not match.
    """
    judge = summary.get("judge") or {}
    if summary.get("review_contract_version") != REVIEW_CONTRACT_VERSION:
        raise DataPreparerOperatorError(
            f"the review job wrote contract version {summary.get('review_contract_version')!r}; this "
            f"skill reads version {REVIEW_CONTRACT_VERSION}."
        )
    want = "estimate" if estimate_only else "ran"
    if judge_model_id and (
        judge.get("status") != want
        or judge.get("model_id") != judge_model_id
        or (estimate_only and not judge.get("estimate"))
    ):
        raise DataPreparerOperatorError(
            f"the job's summary.json judge status {judge.get('status')!r} for model "
            f"{judge.get('model_id')!r} is not what --judge-model-id {judge_model_id} asked for "
            f"({want!r})."
        )
    if scope == "golden" and summary.get("routed_rows") != routed_rows:
        raise DataPreparerOperatorError(
            f"the review job's summary says {summary.get('routed_rows')!r} routed rows but "
            f"routed.jsonl has {routed_rows}; no person should see these rows."
        )


def decision_csv(items, ledger, current_rows=(), prev_rows=()):
    """The ``decisions.csv`` rows for the contradiction items, and the rows kept apart.

    A current row for the same item version (``item_sha256``) is kept as it is, blank or
    not; else the run log's decision for an unchanged item fills the row. Any other filled
    current row is appended to ``decisions.prev.csv``.

    Args:
        items: The job's ``contradictions.jsonl`` rows.
        ledger: :func:`~.label_decisions.logged_decisions` for ``items``.
        current_rows: The existing ``decisions.csv`` rows (``csv.DictReader`` dicts).
        prev_rows: The existing ``decisions.prev.csv`` rows.

    Returns:
        ``(rows, prev, undecided, apart)``: the new CSV rows, the new ``decisions.prev.csv``
        rows (``None`` when it need not be written), the undecided count and the count kept
        apart.
    """
    now = {it["item_id"]: item_sha(it) for it in items}
    present = {
        r["item_id"]: r
        for r in current_rows
        if r.get("item_sha256") and now.get(r["item_id"]) == r["item_sha256"]
    }
    apart = [
        r
        for r in current_rows
        if (r.get("decision") or "").strip() and r.get("item_id") not in present
    ]
    rows = []
    for it in items:
        d = present.get(it["item_id"]) or ledger.get(it["item_id"]) or {}
        rows.append(
            dict(
                {c: d.get(c) or "" for c in DECISION_COLS},
                item_id=it["item_id"],
                title=it.get("title", ""),
                item_sha256=now[it["item_id"]],
                overrides=";".join(
                    f"{k}={v}" for k, v in parse_overrides(d.get("overrides")).items()
                ),
            )
        )
    prev = list(prev_rows) + apart if apart else None
    return rows, prev, sum(1 for r in rows if not r["decision"]), len(apart)


def csv_text(rows):
    """``decisions.csv`` text for :func:`decision_csv`'s rows, cells escaped against formulas."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows({k: csv_cell(v) for k, v in r.items()} for r in rows)
    return buf.getvalue()


def review_result(  # pylint: disable=too-many-arguments
    *,
    job_name,
    scope,
    rows_sent,
    test_rows_left_out,
    summary,
    out_dir,
    routed_rows=None,
    decisions: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """What the client reports after a review: the job, the counts and the summary's figures.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import request

        result = request.review_result(job_name=job["job_name"], scope="seed", rows_sent=120,
                                       test_rows_left_out=30, summary=summary_doc, out_dir="proj/review")
        result["items"], result["judge_status"], result["warnings"]

    Args:
        job_name: The job.
        scope: ``seed`` or ``golden``.
        rows_sent: The rows the job reviewed.
        test_rows_left_out: The seed's test rows left out (0 for golden).
        summary: The job's ``summary.json``.
        out_dir: Where the files landed.
        routed_rows: For golden, the row count of ``routed.jsonl``.
        decisions: For the seed scope, ``{"items", "undecided_items", "decisions_kept_apart"}``
            from :func:`decision_csv`.
    """
    judge = summary.get("judge") or {}
    out: Dict[str, Any] = {
        "job_name": job_name,
        "scope": scope,
        "rows_sent": rows_sent,
        "test_rows_left_out": test_rows_left_out,
        "items": summary.get("items"),
        "judge_status": judge.get("status"),
        "estimate": judge.get("estimate"),
        "warnings": summary.get("warnings") or [],
        "out_dir": out_dir,
    }
    if scope == "golden":
        out.update(routed_rows=routed_rows, routed_per_check=summary.get("routed_per_check") or {})
    else:
        out.update(decisions or {})
    return out


def output_files(scope) -> List[str]:
    """The job's ``review/`` files for a scope."""
    return list(OUTPUT_FILES[scope])
