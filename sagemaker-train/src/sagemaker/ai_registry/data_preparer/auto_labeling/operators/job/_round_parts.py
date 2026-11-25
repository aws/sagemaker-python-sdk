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
"""The pieces of one labeling round, around the job: inputs, gold, demos, the job's rows, the record.

A round labels a split (dev, val) or the corpus with one labeling job. Before the job: the
run id (:func:`check_run_id`), the input's form (:func:`input_kind`), the gold rows split
from their inputs (:func:`split_gold`), the gold's labels against the label set
(:func:`check_gold_labels`), the demos (:func:`select_demo_rows`) and the re-attach digest
(:func:`request_sha256`). After the job: its labels in the skill's shape
(:func:`labels_from_job`), the jurors' shortfall (:func:`juror_shortfall`), the held-out
report (:func:`held_out_report_from_job`) and the run's record for the run log
(:func:`build_job_manifest`). Refusals are
:class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`.
"""

from __future__ import absolute_import

import json
import os
import re
from pathlib import Path

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import (
    JSONL_HINT,
    report_client_read,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import label_text
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.demos import (
    fixed_demos,
    fixed_demos_ner,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import SEED_METADATA_FIELDS
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
    PREDICTION_FIELDS,
    SPOT_CHECK_ROWS,
)

# The run id goes into every scratch DataSet name (`<dataset>-<stage>-<run id>`), so it
# is bounded: 40 characters.
RUN_ID_MAX_CHARS = 40
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9](-*[A-Za-z0-9])*$")

# Refine-loop artifacts that mark a run as seeded (the no-seed safeguard).
SEED_ARTIFACTS = (
    "decision.json",
    os.path.join("work", "decision.json"),
    os.path.join("splits", "manifest.json"),
    "held_out_report.json",
)

# A classification seed whose labels are spelled unlike the Label Set scores every such
# row wrong. Above this share of the gold rows the run is refused before anything is
# spent; below it, a warning names the values.
MAX_UNKNOWN_GOLD_SHARE = 0.05

# The provenance fields the job reads to tell a person's gold from the model's: they go
# onto the gold rows as given.
GOLD_PROVENANCE_FIELDS = ("provenance", "_provenance", "_bootstrap", "source")
# The id a corpus run gives a gold row that has none: a prefixed id is never a corpus
# position, so the job labels the row to score it.
GOLD_ID_PREFIX = "seed-"

# A juror missing on more than this share of the voted rows is degraded.
DEGRADED_JUROR_RATE = 0.05

# What the corpus run downloads: never `Labels` (the labeled corpus stays behind the
# API) nor `ReviewRows`.
CORPUS_ARTIFACTS = ("LabelingSummary", "HeldOutReport", "ReviewModelOutput", "Report")

# The labeling_summary.json fields a summarization job's graders write.
GRADER_SUMMARY_KEYS = (
    "GoodRate",
    "GradedRows",
    "GradeSampleRows",
    "GradeIncompleteRows",
    "GradeFailedRows",
    "GraderModelIds",
    "GraderFailedCalls",
    "GradeSourceTruncatedRows",
    "GradeSummaryTruncatedRows",
)
# The labeling_summary.json fields the opt-in options write, each with its stdout
# summary key; the job writes each only when the request carries its option.
OPTION_SUMMARY_KEYS = {
    "ConfidenceMethod": "confidence_method",
    "StatedConfidenceMissing": "stated_confidence_missing",
    "VerbalizedConfidenceMissing": "verbalized_confidence_missing",
    "ConfidenceEstimator": "confidence_estimator",
    "ConfidenceEstimatorUsage": "confidence_estimator_usage",
    "ConfidenceRecalibration": "confidence_recalibration",
    "ConfidenceFloorBypass": "confidence_floor_bypass",
    "Optimizer": "optimizer",
}
# The option flags; the run-log invocation records each only when it is given.
OPTION_INVOCATION_KEYS = (
    "confidence_method",
    "golden_review",
    "confidence_estimator",
    "judge_models",
    "confidence_recalibration",
    "recalibration_fit",
    "optimizer",
    "confidence_floor_bypass",
)
_JOB_SUMMARY_KEYS = (
    "Status",
    "StatusReason",
    "Type",
    "RowsLabeled",
    "RowsAccepted",
    "RowsRouted",
    "RowsHumanLabeled",
    "RowsFailed",
    "RowsMissing",
    "HeldOutAccuracy",
    "HeldOutLowerBound95",
    "HeldOutRows",
    "AcceptedHeldOutAccuracy",
    "AcceptedLowerBound95",
    "AcceptedHeldOutRows",
    "Target",
    "AcceptThreshold",
    "MinAgreement",
    "SpotCheckRows",
    "SpotCheckWithdrawAt",
    "SpecSha256",
    "ProgramSha256",
    "JurorModelIds",
    "ReviewDatasetName",
    "FailureRows",
    *GRADER_SUMMARY_KEYS,
    *OPTION_SUMMARY_KEYS,
)


# ---- the run id and the input --------------------------------------------------------------


def check_run_id(run_id):
    """The run id as a Hub-name fragment: letters, digits and single hyphens, at most 40.

    Underscores become hyphens. Call it before a run id goes into a DataSet name.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import check_run_id

        check_run_id("iter_2")   # "iter-2"

    Args:
        run_id: The run id to check.

    Returns:
        The run id as it goes into a DataSet name.

    Raises:
        DataPreparerOperatorError: Other characters, or too long.
    """
    slug = (run_id or "").replace("_", "-")
    if not _RUN_ID_RE.fullmatch(slug):
        raise DataPreparerOperatorError(
            f"--run-id {run_id!r} must be letters, digits and hyphens (no dots, spaces, "
            "colons, leading or trailing hyphen): it goes into the scratch DataSet names"
        )
    if len(slug) > RUN_ID_MAX_CHARS:
        raise DataPreparerOperatorError(
            f"--run-id {run_id!r} is {len(slug)} characters; the DataSet names it goes into "
            f"take at most {RUN_ID_MAX_CHARS}. Use a shorter run id."
        )
    return slug


def input_kind(ref):
    """The form of an input reference.

    ``("arn", arn)`` for a DataSet ARN, ``("s3", uri)`` for an s3:// URI or a corpus.ref.json
    pointer, ``("local", path)`` otherwise.

    Raises:
        DataPreparerOperatorError: No reference, or an unreadable or incomplete pointer.
    """
    if not ref:
        raise DataPreparerOperatorError("--input is required")
    if ref.startswith("arn:"):
        return "arn", ref
    if ref.startswith("s3://"):
        return "s3", ref
    if ref.endswith(".ref.json"):
        try:
            with open(ref) as fh:
                doc = json.load(fh)
        except (OSError, ValueError) as e:
            raise DataPreparerOperatorError(
                f"cannot read corpus.ref.json pointer {os.path.abspath(ref)!r}: {type(e).__name__}: {e}"
            ) from None
        uri = doc.get("source_s3_uri")
        if not uri:
            raise DataPreparerOperatorError(
                f"corpus.ref.json pointer {os.path.abspath(ref)!r} has no "
                "'source_s3_uri'; re-run the hub_ops resolve action (DOWNLOAD = False)."
            )
        return "s3", uri
    return "local", ref


def read_local_rows(path, primitive):
    """Every non-blank JSON object of a local JSONL file (a client read, reported as ``primitive``).

    Raises:
        DataPreparerOperatorError: A malformed line, named.
    """
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError as e:
                raise DataPreparerOperatorError(
                    f"{path} line {n} is not JSON: {e}. {JSONL_HINT}"
                ) from None
            if not isinstance(row, dict):
                raise DataPreparerOperatorError(f"{path} line {n} is not a JSON object")
            rows.append(row)
    report_client_read(primitive, len(rows))
    return rows


def read_task_sample(input_path, limit=200):
    """Up to ``limit`` rows of a local JSONL for task detection only; a missing file or a bad line never fails it."""
    rows = []
    if not input_path:
        return rows
    try:
        with open(input_path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
                if len(rows) >= limit:
                    break
    except OSError:
        return rows
    if rows:
        report_client_read("detect_task_type", len(rows))
    return rows


def seed_evidence(out_path, spec_path, *, eval_path=None, demos_path=None, demos_per_class=None):
    """Why this run counts as seeded, or ``None`` when it shows no labeled seed.

    An eval file (gold rides along), demos in use, or a refine-loop artifact next to the
    output or the spec, or one level up.
    """
    if eval_path:
        return "--eval"
    if demos_path and (demos_per_class or 0) > 0:
        return "--demos"
    dirs = []
    for p in (Path(out_path).resolve().parent, Path(spec_path).resolve().parent):
        dirs += [p, p.parent]
    for d in dict.fromkeys(dirs):
        for rel in SEED_ARTIFACTS:
            if (d / rel).is_file():
                return str(d / rel)
    return None


def no_seed_message():
    """The refusal of a run with no labeled seed and no opt-out."""
    floor = ", ".join(f"{SPOT_CHECK_ROWS[t]} rows for {t:.2f}" for t in sorted(SPOT_CHECK_ROWS))
    return (
        "no labeled seed: no --eval, no --demos, and no decision.json, "
        "splits/manifest.json or held_out_report.json near --out or --spec. Labels from "
        "this run would carry no accuracy promise. Build a seed first "
        "(references/cold-start.md): people label a random sample of the corpus, then "
        f"the normal loop runs from it. With zero errors, a spot-check needs {floor} to "
        "show that accuracy. Only if the customer declines and says yes to labels with "
        "no accuracy promise (cold-start.md, Opt-out), pass --confirm-no-seed."
    )


# ---- gold ------------------------------------------------------------------------------------


def labels_missing_from_spec(spec_text, label_set):
    """Labels from the label set the spec text never mentions.

    Whole word, case-insensitive; ``_`` and ``-`` in a label also match a space.
    """
    missing = []
    for lab in label_set:
        s = str(lab).strip()
        if not s:
            continue
        variants = {s, s.replace("_", " "), s.replace("-", " ")}
        if not any(
            re.search(
                r"(?<![A-Za-z0-9_])" + re.escape(v) + r"(?![A-Za-z0-9_])", spec_text, re.IGNORECASE
            )
            for v in variants
        ):
            missing.append(s)
    return missing


def fold_label(v):
    """A label folded for a spelling comparison: alphanumerics only, case folded."""
    return "".join(ch for ch in str(v).casefold() if ch.isalnum())


def check_gold_labels(gold_rows, label_field, label_set, spec_text, eref):
    """Classification gold labels against the label set.

    Against an explicit label set the run is refused above ``MAX_UNKNOWN_GOLD_SHARE``;
    against the spec's Label Set, which is parsed best-effort, it only warns. A label set
    that cannot be read skips the check.

    Returns:
        The warning text (without its ``WARNING:`` prefix), or ``None``.

    Raises:
        DataPreparerOperatorError: Too many unknown labels against an explicit label set.
    """
    names = [str(x) for x in label_set] if label_set else []
    explicit = bool(names)
    if not names:
        try:
            from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.taxonomy import (
                parse_label_set,
            )

            names = [n for n, _ in parse_label_set(spec_text)]
        except Exception:  # noqa: BLE001 - the job reads the spec itself; this is a pre-check
            names = []
    if not names or not gold_rows:
        return None
    allowed = {label_text(n) for n in names}
    folded = {fold_label(n): n for n in names}
    unknown = {}
    for g in gold_rows:
        v = label_text(g.get(label_field))
        if v is not None and v not in allowed:
            unknown[v] = unknown.get(v, 0) + 1
    if not unknown:
        return None
    n_bad = sum(unknown.values())
    shown = ", ".join(
        f"{v!r} x{n}"
        + (
            f" (the label set spells it {folded[fold_label(v)]!r})"
            if fold_label(v) in folded
            else ""
        )
        for v, n in sorted(unknown.items(), key=lambda kv: -kv[1])[:5]
    )
    msg = (
        f"{n_bad} of the {len(gold_rows)} --eval gold rows have a {label_field!r} that is not "
        f"in the label set ({shown}). The model answers only with the label set's spelling "
        "and the job compares labels exactly, so each such row scores wrong"
    )
    if explicit and n_bad > MAX_UNKNOWN_GOLD_SHARE * len(gold_rows):
        raise DataPreparerOperatorError(
            f"--eval {eref!r}: {msg}. Respell the gold to the label set (or fix "
            "--label-set / the spec's Label Set) and re-run; nothing was uploaded."
        )
    return f"{msg}."


def row_id(row, index, id_field="id", int_ids=False):
    """A row's id as text: its ``id_field``, else its position (the container's rule).

    With ``int_ids`` (the corpus's ids are integers), an integral float id such as 300.0 is
    written 300, so it joins the corpus row 300: the job compares ids as text.
    """
    v = row.get(id_field)
    if v is None:
        return str(index)
    if int_ids and isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


def float_ids_to_join(rows, id_field="id"):
    """How many rows carry an integral float id (300.0) that :func:`row_id` rewrites with ``int_ids``."""
    return sum(1 for r in rows if isinstance(r.get(id_field), float) and r[id_field].is_integer())


def corpus_ids_are_ints(first_row, id_field="id"):
    """True when the corpus's first row has an integer id (an int, or a string of digits)."""
    v = first_row.get(id_field) if isinstance(first_row, dict) else None
    return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, str) and v.isdigit())


def split_gold(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows, label_field, id_field="id", text_fields=(), id_prefix=None, int_ids=False
):
    """``(input_rows, gold_rows, unlabeled_ids)`` for a labeled split.

    The input rows go without the gold (the label field and every ``gt_*`` field) and
    without the seed's metadata, so a delivered labels file fed back as the input is
    labeled afresh. The gold rows are ``{id, <label_field>}`` plus the provenance fields
    the row has and each of ``text_fields`` the row has (the job labels a gold row whose id
    is not a corpus row from those fields, to score it). A row with a null gold is left out
    of the gold with its id reported; a missing id is the row's position, or ``id_prefix``
    plus its position when given, on the input rows and the gold rows alike.
    """
    inputs, gold, unlabeled = [], [], []
    for i, r in enumerate(rows):
        rid = (
            row_id(r, i, id_field, int_ids)
            if id_prefix is None or r.get(id_field) is not None
            else f"{id_prefix}{i}"
        )
        inp = {
            k: v
            for k, v in r.items()
            if k != label_field and not str(k).startswith("gt_") and k not in SEED_METADATA_FIELDS
        }
        inp[id_field] = rid
        inputs.append(inp)
        if r.get(label_field) is None:
            unlabeled.append(rid)
            continue
        g = {id_field: rid, label_field: r[label_field]}
        g.update({f: r[f] for f in text_fields if f in r and f != label_field})
        g.update(
            {
                k: r[k]
                for k in GOLD_PROVENANCE_FIELDS
                if r.get(k) is not None and k != label_field and k not in text_fields
            }
        )
        gold.append(g)
    return inputs, gold, unlabeled


def is_kfold_file(path):
    """True when the ``manifest.json`` beside ``path`` (a k-fold rotation's) lists the file with its current sha256."""
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(path)), "manifest.json")) as fh:
            listed = (json.load(fh).get("sha256") or {}).get(os.path.basename(path))
    except (OSError, ValueError, AttributeError):
        return False
    return listed == _digest.sha256_file(path)


def check_ids(rows, id_field="id"):
    """Refuse duplicate ids among ``rows``; a missing or null id is the row's position.

    Returns:
        The id list.

    Raises:
        DataPreparerOperatorError: Duplicate ids.
    """
    ids, seen, dup = [], set(), []
    for i, r in enumerate(rows):
        rid = row_id(r, i, id_field)
        if rid in seen:
            dup.append(rid)
        seen.add(rid)
        ids.append(rid)
    if dup:
        raise DataPreparerOperatorError(
            f"{len(dup)} duplicate id(s) (e.g. {dup[:3]}); the job joins gold "
            "and review rows by id, so every row needs its own"
        )
    return ids


# ---- demos -----------------------------------------------------------------------------------


def select_demo_rows(seed_rows, task, k, label_field, pred_field):
    """``k`` labeled examples per class from ``seed_rows``.

    Classification by ``label_field``, ner by entity type, otherwise the first ``k`` rows;
    each carries every input field and its label under ``label_field`` and under
    ``gt_<prediction field>`` (the program's fallback when no EvaluationData names a
    LabelField).
    """
    rows = [r for r in seed_rows if r.get(label_field) is not None]
    if task == "ner":
        demos = fixed_demos_ner(rows, k, entities_field=label_field)
    elif task == "classification":
        picked = fixed_demos([dict(r, gt_label=r[label_field]) for r in rows], k)
        demos = [
            {kk: v for kk, v in d.items() if kk != "gt_label" or label_field == "gt_label"}
            for d in picked
        ]
    else:
        demos = rows[:k]
    out = []
    for d in demos:
        d = dict(d)
        d.setdefault(f"gt_{pred_field}", d[label_field])
        out.append(d)
    return out


def text_key(row, text_fields):
    """A row's input text as the job compares a demo to an eval row: the JSON of each text field, in order."""
    return tuple(json.dumps(row.get(f), sort_keys=True) for f in text_fields)


# ---- the job's rows ----------------------------------------------------------------------------


def labels_from_job(rows, task):
    """The job's labels.jsonl rows in the skill's shape, everything the container wrote kept.

    ``confidence_score`` beside ``confidence`` on a model-labeled row, ``jury_votes`` from
    the job's ``ballots``, and ``_failed`` / ``_failed_reason`` on a row whose prediction is
    null.
    """
    pred_field = PREDICTION_FIELDS[task]
    out = []
    for r in rows:
        r = dict(r)
        if r.get("_labeled_by") != "people":
            if (
                "confidence_score" not in r
                and isinstance(r.get("confidence"), (int, float))
                and not isinstance(r.get("confidence"), bool)
            ):
                r["confidence_score"] = r["confidence"]
            if "jury_votes" not in r and isinstance(r.get("ballots"), list):
                r["jury_votes"] = [jury_vote(b, pred_field) for b in r["ballots"]]
            if r.get(pred_field) is None:
                r["_failed"] = True
                if r.get("prediction_error") and "_failed_reason" not in r:
                    r["_failed_reason"] = r["prediction_error"]
        out.append(r)
    return out


def jury_vote(ballot, pred_field):
    """One job ballot as a jury vote: its value (the task's prediction field) under ``label``, with the other keys."""
    if not isinstance(ballot, dict):
        return ballot
    vote = dict(ballot)
    if "label" not in vote and pred_field in vote:
        vote["label"] = vote[pred_field]
    return vote


def ballot_cast(ballot):
    """Whether a juror's ballot carries a usable answer.

    A null value (an off-list label, an unparseable reply) and a ner entity list that did
    not parse as a list are missing votes; ``stated_confidence`` is not an answer.
    """
    if not isinstance(ballot, dict):
        return False
    values = {k: v for k, v in ballot.items() if k not in ("model", "stated_confidence")}
    if "entities" in values:
        return isinstance(values["entities"], list)
    return any(v is not None for v in values.values())


def degraded_jurors(missing_by_juror, n_rows):
    """``{juror: {missing, rows, missing_rate}}`` for every juror missing on too many rows.

    Too many is more than ``DEGRADED_JUROR_RATE`` of ``n_rows``.
    """
    if not n_rows:
        return {}
    out = {}
    for m, c in (missing_by_juror or {}).items():
        rate = c / n_rows
        if rate > DEGRADED_JUROR_RATE:
            out[m] = {"missing": c, "rows": n_rows, "missing_rate": round(rate, 4)}
    return out


def juror_shortfall(rows, juror_model_ids):
    """Per-juror missing ballots over the model-labeled rows that carry ``ballots``.

    Returns:
        ``{"rows", "ballot_unit", "missing_by_juror", "degraded_jurors"}``, or ``None`` when
        no row was voted.
    """
    voted = [
        r for r in rows if isinstance(r.get("ballots"), list) and r.get("_labeled_by") != "people"
    ]
    if not voted:
        return None
    jurors = list(juror_model_ids or [])
    for r in voted:
        for b in r["ballots"]:
            m = (b or {}).get("model") if isinstance(b, dict) else None
            if m and m not in jurors:
                jurors.append(m)
    missing = {m: 0 for m in jurors}
    for r in voted:
        answered = {b.get("model") for b in r["ballots"] if ballot_cast(b)}
        for m in jurors:
            if m not in answered:
                missing[m] += 1
    n = len(voted)
    return {
        "rows": n,
        "ballot_unit": "juror",
        "missing_by_juror": missing,
        "degraded_jurors": degraded_jurors(missing, n),
    }


def review_model_outputs(path):
    """The model's answer of each review row from the job's side file (``review.model_output.jsonl``)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                out = rec.get("model_output") if isinstance(rec, dict) else None
                if isinstance(out, dict):
                    rows.append(out)
    return rows


def held_out_report_from_job(summary, job_report, task):
    """The skill's ``held_out_report.json`` shape from the job's summary and held-out report.

    ``primary`` / ``primary_name`` (HeldOutAccuracy, the one number the refine loop steers
    on), ``accuracy``, ``macro_f1`` (classification, when the job reported it; recorded,
    never selected on), ``lower_bound_95``, ``breakdown``, ``confusion_pairs``, ``counts``,
    the certificate fields and the whole job summary under ``job``. A summarization job
    has no gold: its graders' ``GoodRate`` on ``GradedRows`` is the primary. ``None`` when
    the job scored nothing.
    """
    summary = summary or {}
    job_report = job_report or {}
    n = summary.get("HeldOutRows")
    acc = summary.get("HeldOutAccuracy")
    if task == "summarization" and summary.get("GoodRate") is not None:
        graded = summary.get("GradedRows") or 0
        good = summary["GoodRate"]
        return {
            "primary": good,
            "primary_name": "good_rate",
            "good_rate": good,
            "counts": {"good": round(good * graded), "total": graded},
            "graded": {k: summary.get(k) for k in GRADER_SUMMARY_KEYS},
            "status": summary.get("Status"),
            "status_reason": summary.get("StatusReason"),
            "task": task,
            "job": summary,
        }
    if n is None or acc is None:
        return None
    macro = job_report.get("macro_f1") if task == "classification" else None
    return {
        "primary": acc,
        "primary_name": "accuracy",
        "accuracy": acc,
        "macro_f1": macro,
        "held_out_accuracy": acc,
        "lower_bound_95": summary.get("HeldOutLowerBound95"),
        "breakdown": job_report.get("per_class_f1") or job_report.get("per_type_f1") or {},
        "confusion_pairs": job_report.get("confusion_pairs") or [],
        "counts": {"correct": round(acc * n), "total": n},
        "accepted": {
            "accuracy": summary.get("AcceptedHeldOutAccuracy"),
            "lower_bound_95": summary.get("AcceptedLowerBound95"),
            "rows": summary.get("AcceptedHeldOutRows"),
        },
        "status": summary.get("Status"),
        "status_reason": summary.get("StatusReason"),
        "accept_threshold": summary.get("AcceptThreshold"),
        "spot_check_rows": summary.get("SpotCheckRows"),
        "spot_check_withdraw_at": summary.get("SpotCheckWithdrawAt"),
        "task": task,
        "job": summary,
    }


# ---- the record ----------------------------------------------------------------------------------


def s3_paths(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    spec_uri, input_ref, eval_uri, demos_uri, output_uri, index_uri, labels_path, artifacts
):
    """Named S3/local paths for the run report."""
    items = [{"name": "Labeling spec", "path": spec_uri}, {"name": "Input data", "path": input_ref}]
    if eval_uri:
        items.append({"name": "Evaluation data", "path": eval_uri})
    if demos_uri:
        items.append({"name": "Demos", "path": demos_uri})
    items.append({"name": "Job output", "path": output_uri})
    if index_uri:
        items.append({"name": "Labeling index", "path": index_uri})
    for name in ("Labels", "ReviewRows", "Failures", "LabelingSummary"):
        if artifacts.get(name):
            items.append({"name": f"Job {name}", "path": artifacts[name]})
    items.append({"name": "Labeled dataset", "path": labels_path})
    return items


def build_job_manifest(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    task,
    n_rows,
    stage,
    region,
    job,
    results,
    request,
    labels_path,
    spec_uri,
    input_ref,
    eval_uri,
    demos_uri,
    scratch,
):
    """The run's record for the run log (:func:`~.run_records.labeling_run_record`).

    One ``job`` card (the labeling job: ARN, region, models, S3 paths, the labeled DataSet
    ARN), the planning run-report contract (``report_s3_paths``, ``run_report``) and the
    job's summary numbers.
    """
    summary = (results or {}).get("summary") or {}
    paths = s3_paths(
        spec_uri,
        input_ref,
        eval_uri,
        demos_uri,
        job.get("output_s3_uri"),
        (results or {}).get("index_uri"),
        labels_path,
        (results or {}).get("artifact_uris") or {},
    )
    models = request.get("models") or []
    card = {
        "type": "Labeling (scratch)" if scratch else "Labeling",
        "description": (
            f"SDG labeling job ({stage}) over {n_rows} rows for task {task!r}: "
            + ("the jury" if len(models) > 1 else "one model")
            + (", routed to review" if request.get("routing") else "")
        ),
        "arn": job.get("job_arn"),
        "region": region,
        "model": models[0] if models else None,
        "models": models,
        "dataset_arn": (results or {}).get("labeled_dataset_arn"),
        "review_dataset_arn": (results or {}).get("review_dataset_arn"),
        "job_name": job.get("job_name"),
        "tokens": None,
        "cost_usd": None,
        "s3_paths": paths,
    }
    run_report = {
        "type": "Labeling",
        "description": (
            f"Auto-labeling ({stage}): {n_rows} rows labeled for task {task!r} by one SDG labeling job"
        ),
        "arn": card["arn"],
        "region": region,
        "s3_paths": paths,
    }
    return {
        "region": region,
        "task": task,
        "rows": n_rows,
        "stage": stage,
        "scratch": scratch,
        "jobs": [card],
        "report_s3_paths": paths,
        "run_report": run_report,
        "job_summary": {k: summary.get(k) for k in _JOB_SUMMARY_KEYS if k in summary},
    }


def sha256_text(text):
    """The SHA-256 of a text, or ``None`` for ``None``."""
    return None if text is None else _digest.sha256_hex(text)


def file_identity(path):
    """A local corpus file's identity for the re-attach check: path, size and modification time, never its bytes."""
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def request_sha256(  # pylint: disable=too-many-arguments
    document, *, spec_text, demos_text=None, gold_text=None, input_text=None, input_file=None
):
    """What a re-attach compares: the request document and the content of every file this run uploads.

    A local corpus file counts by its path, size and modification time, never its bytes;
    an s3:// or DataSet input is not uploaded, so only its URI or ARN counts.
    """
    content = {
        "spec": sha256_text(spec_text),
        "demos": sha256_text(demos_text),
        "evaluation_data": sha256_text(gold_text),
        "input": file_identity(input_file) if input_file is not None else sha256_text(input_text),
    }
    return _digest.sha256_json({"request": document, "content": content}, ascii=True)
