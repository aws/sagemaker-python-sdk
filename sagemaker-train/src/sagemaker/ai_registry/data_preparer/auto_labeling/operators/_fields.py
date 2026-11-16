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
"""The row and field contract between the labeling job, a human review and a seed.

Every name the job writes on a labeled row, every field a review row hides from a
reviewer, the routing tag, the side file that keeps the model's answers, and the
provenance a seed row carries about who labeled it. Readers across the loop (the
fold, the report, the input inspection) share these tables so a field is reserved,
hidden or restored by one rule.
"""

from __future__ import absolute_import

from typing import Any, Dict, Iterable, List, Optional

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest

# The record golden_review.py apply puts on every row of a reviewed copy of the corpus
# run's --eval (a person resolved it, or `resolved_by: null`); no score reads that copy as gold.
GOLDEN_REVIEW_FIELD = "_golden_review"
# Fields a seed row carries about who labeled it and how it was routed, never input.
SEED_METADATA_FIELDS = frozenset(
    {
        "_bootstrap",
        "_bootstrap_confidence",
        "_provenance",
        "_round",
        "_spec_version",
        "_annotator",
        "_route_reason",
        "_review_reason",
        "_model_output",
        "_labeled_by",
        "provenance",
        "round",
        "spec_version",
        "cal_version",
        "route_reason",
        "verdict",
        "annotator",
        "annotators",
        "n_annotations",
        "source_export",
        "split",
        "source",
        GOLDEN_REVIEW_FIELD,
    }
)


# --- review-file contract (labeling job -> human-in-the-loop -> bootstrap_seed.py)
# One routing tag everywhere: `_review_reason`. Older files carry `routed_reason`
# (an older calibration's output), `route_reason` or `_route_reason`; readers accept every
# one of them, writers write `_review_reason` only.
REVIEW_REASON_FIELD = "_review_reason"
LEGACY_REVIEW_REASON_FIELDS = ("routed_reason", "route_reason", "_route_reason")

# A blind review row keeps the model's answer out of the reviewer's view: the fields
# below move into MODEL_OUTPUT_FIELD, which the reviewer UI must not show (like every
# `_`-prefixed field), and bootstrap_seed.py hil puts them back when it folds the export.
MODEL_OUTPUT_FIELD = "_model_output"
# The row fields the labeling job writes only with Confidence.Method stated: a model's
# stated confidence and the jury's vote share.
STATED_CONFIDENCE_FIELDS = ("stated_confidence", "agreement_confidence")
# The row field a Confidence.Method verbalized job adds: true on a labeled row whose
# confidence_score was missing or not a number (its confidence is then 0). Read as
# model output on verbalized runs only, so a corpus column of that name passes through
# every other run as before.
VERBALIZED_CONFIDENCE_FIELDS = ("confidence_missing",)
# The row field a Confidence.Recalibration job adds: the model's raw confidence beside the
# mapped confidence_score (the container's RECALIBRATION_FIELDS). Reserved only with that
# option, so a corpus column of that name passes through every other run.
RECALIBRATION_FIELDS = ("raw_confidence_score",)
# Names the job reserves only with an option, and the flag that reserves each (Step 0 warns).
OPTION_RESERVED_FIELDS = {
    **{f: "--confidence-method verbalized" for f in VERBALIZED_CONFIDENCE_FIELDS},
    **{f: "--confidence-recalibration" for f in RECALIBRATION_FIELDS},
}
# Above this share of ballots (jury) or rows (one model) with no stated confidence, the
# submitter warns: a missing stated confidence counts 0 in the row score.
STATED_CONFIDENCE_MISSING_RATE = 0.05


def verbalized_confidence_missing(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How many labeled rows of a verbalized-confidence job stated no confidence.

    Reads ``VerbalizedConfidenceMissing`` over ``RowsLabeled`` (one model, so the
    unit is rows).

    Args:
        summary: The job's ``labeling_summary.json`` document.

    Returns:
        ``{"missing", "of", "unit", "rate", "high"}``, ``high`` when the rate is over
        ``STATED_CONFIDENCE_MISSING_RATE``; ``None`` when the summary has no count or
        no row count.
    """
    missing = summary.get("VerbalizedConfidenceMissing")
    rows = summary.get("RowsLabeled")
    if not isinstance(missing, int) or not isinstance(rows, int) or rows <= 0:
        return None
    rate = missing / rows
    return {
        "missing": missing,
        "of": rows,
        "unit": "rows",
        "rate": round(rate, 4),
        "high": rate > STATED_CONFIDENCE_MISSING_RATE,
    }


def stated_confidence_missing(summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How many ballots (or rows, for one model) of a stated-confidence job carried none.

    Reads ``StatedConfidenceMissing`` over ``RowsLabeled`` times the jury size
    (``JurorModelIds``), so the unit is ``ballots`` for a jury and ``rows`` otherwise.

    Args:
        summary: The job's ``labeling_summary.json`` document.

    Returns:
        ``{"missing", "of", "unit", "rate", "high"}``, ``high`` when the rate is over
        ``STATED_CONFIDENCE_MISSING_RATE``; ``None`` when the summary has no count or
        no row count.
    """
    missing = summary.get("StatedConfidenceMissing")
    rows = summary.get("RowsLabeled")
    if not isinstance(missing, int) or not isinstance(rows, int) or rows <= 0:
        return None
    jurors = len(summary.get("JurorModelIds") or []) or 1
    unit = "ballots" if jurors > 1 else "rows"
    of = rows * jurors
    rate = missing / of
    return {
        "missing": missing,
        "of": of,
        "unit": unit,
        "rate": round(rate, 4),
        "high": rate > STATED_CONFIDENCE_MISSING_RATE,
    }


BLIND_FIELDS = (
    "label",
    "reasoning",
    "confidence",
    "confidence_score",
    "min_span_confidence",
    "jury_votes",
    "jury_span_votes",
    "n_responders",
    "n_jurors",
    "juror_errors",
    "judge_confidence",
    "samples",
    "self_rated_confidence",
    "self_rated_confidence_n",
    "self_rated_by_juror",
    "self_rated_input_truncated",
    "self_rated_skipped",
    "p_correct",
    "calib_accepted",
    "_failed_reason",
    "calib_certificate",
    "calib_group",
    "low_agreement",
    "promise",
    "promise_confidence",
    "promise_interval",
    "promise_shortlist",
    "stacked_confidence",
    *STATED_CONFIDENCE_FIELDS,
)

# Per-run outputs of the labeling job (and of the earlier client-side engine and
# calibration, which older files still carry). Two labeling runs of the same rows
# differ in these (even at temperature 0), so they do not identify the row: the seed
# merge's input-field check (bootstrap_seed._seed_input_fields) leaves them out.
MODEL_OUTPUT_FIELDS = frozenset(
    {
        "reasoning",
        "confidence",
        "confidence_score",
        "min_span_confidence",
        "jury_votes",
        "jury_span_votes",
        "n_responders",
        "n_jurors",
        "juror_errors",
        "judge_confidence",
        "samples",
        "self_rated_confidence",
        "self_rated_confidence_n",
        "self_rated_by_juror",
        "self_rated_input_truncated",
        "self_rated_skipped",
        "_input_index",
        "_failed",
        "_failed_reason",
        "_extra",
        "_text_missing",
        "_text_degenerate",
        "p_correct",
        "calib_accepted",
        "calib_certificate",
        "calib_group",
        "routed",
        "routed_reason",
        "low_agreement",
        "missing_agreement",
        "audit_sample",
        # the labeling job's vote / resample fields and its routing (the container's
        # AGGREGATE_FIELDS and FINALIZER_FIELDS): `accepted: false` or a split ballot on a
        # review row tells the reviewer what the model thought.
        "votes",
        "ballots",
        "panel_size",
        "n_failed",
        "failed_jurors",
        "n_out_of_scale",
        "n_types_dropped",
        "prediction_error",
        "accepted",
        *STATED_CONFIDENCE_FIELDS,
    }
)

# The one set of model-output fields a routed review row hides (with the prediction
# field): both lists above. The job moves them to the side file, and
# bootstrap_seed.py's merge reads a seed row carrying any of them as a reviewed row.
BLIND_FIELD_SET = frozenset(BLIND_FIELDS) | MODEL_OUTPUT_FIELDS


# The fields the labeling job writes on a labeled row beside the prediction (the
# container's reserved_row_fields: its AGGREGATE_FIELDS, FINALIZER_FIELDS and
# PROVENANCE_INPUT_FIELDS, plus `reasoning` and `prediction_error`). An input column
# with one of these names is replaced by the job's value.
JOB_ROW_FIELDS = frozenset(
    {
        "reasoning",
        "prediction_error",
        "confidence",
        "votes",
        "n_responders",
        "panel_size",
        "ballots",
        "samples",
        "min_span_confidence",
        "n_failed",
        "failed_jurors",
        "n_out_of_scale",
        "n_types_dropped",
        "accepted",
        "routed",
        "source",
        "provenance",
        "confidence_score",
        "_labeled_by",
        "_model_output",
        "_provenance",
        "_bootstrap",
    }
)

# The one set a review row never shows a reviewer (with the task's prediction field):
# every model-output field, every field the job writes, and the review row's own
# routing tags (`split` and `_review_reason` stay on the row for the fold, but a
# visible `spot_check` tells the reviewer the model was confident). hil_handoff.py
# states it in the hand-off contract; the job moves the model's fields of it to the
# side file.
REVIEW_HIDDEN_FIELDS = (
    BLIND_FIELD_SET | JOB_ROW_FIELDS | {MODEL_OUTPUT_FIELD, "split", REVIEW_REASON_FIELD}
)


def review_hidden_fields(pred_field: Optional[str] = None) -> List[str]:
    """``REVIEW_HIDDEN_FIELDS`` plus the task's prediction field, sorted."""
    return sorted(REVIEW_HIDDEN_FIELDS | ({pred_field} if pred_field else set()))


# Names the container's TextFields check still refuses (its SKILL_BLIND_FIELDS) that no
# skill script writes any more: `_image_missing` left the skill's lists with image input.
CONTAINER_BLIND_FIELDS = frozenset({"_image_missing"})


def reserved_text_field_conflicts(
    text_fields: Iterable[str], pred_field: Optional[str] = None
) -> List[str]:
    """The text fields the job refuses as input columns, sorted.

    A name the job writes on a labeled row (``id``, ``JOB_ROW_FIELDS``, the task's
    prediction field), ``split``, or a field a review row hides from people
    (``BLIND_FIELD_SET``). Such a column would be overwritten in the labeled output,
    or reach the model but not the reviewer. The job's own ``TextFields`` check
    refuses the same names at ``CreateJob``; checking here reports them before spend.

    Args:
        text_fields: The input columns the spec labels from.
        pred_field: The task's prediction field.

    Returns:
        The conflicting names, sorted; empty when there is none.
    """
    reserved = (
        BLIND_FIELD_SET
        | JOB_ROW_FIELDS
        | CONTAINER_BLIND_FIELDS
        | {"id", "split", MODEL_OUTPUT_FIELD}
        | ({pred_field} if pred_field else set())
    )
    return sorted({f for f in text_fields if f in reserved})


def reserved_field_reason(name: str, pred_field: Optional[str] = None) -> Optional[str]:
    """Why the job reserves ``name`` as an input column, as one short clause.

    The order matches the job's own check: a name it writes on each labeled row is
    overwritten there; a model-output name is hidden from reviewers.

    Args:
        name: The input column.
        pred_field: The task's prediction field.

    Returns:
        The reason, or ``None`` when the job does not reserve ``name``.
    """
    if name == "id":
        return "the job keys rows on it"
    if name == "split":
        return "the run's split tag"
    if name == pred_field or name in JOB_ROW_FIELDS or name == MODEL_OUTPUT_FIELD:
        return "the labeling job writes it on each labeled row, replacing the input value"
    if name in BLIND_FIELD_SET or name in CONTAINER_BLIND_FIELDS:
        return "a model-output field, hidden from reviewers"
    return None


def review_reason_of(row: Dict[str, Any]) -> Any:
    """The routing tag on a row (``_review_reason``, else a legacy name), or ``None``."""
    for key in (REVIEW_REASON_FIELD,) + LEGACY_REVIEW_REASON_FIELDS:
        v = row.get(key)
        if v is not None:
            return v
    return None


def blind_row(row: Dict[str, Any], pred_field: Optional[str] = None) -> Dict[str, Any]:
    """A copy of ``row`` with the model's answer hidden for a reviewer.

    The prediction field and every ``BLIND_FIELDS`` key present move under
    ``MODEL_OUTPUT_FIELD``, so a reviewer labels the row without seeing, and copying,
    what the model said. :func:`unblind_row` is the inverse.

    Args:
        row: A labeled row.
        pred_field: The task's prediction field.

    Returns:
        The blinded copy; ``row`` is not changed.
    """
    out = dict(row)
    hidden = dict(out.pop(MODEL_OUTPUT_FIELD, None) or {})
    for key in ((pred_field,) if pred_field else ()) + BLIND_FIELDS:
        if key in out:
            hidden[key] = out.pop(key)
    out[MODEL_OUTPUT_FIELD] = hidden
    return out


def unblind_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Inverse of :func:`blind_row`: the hidden model fields restored onto a copy.

    A field the row carries again with a non-null value is kept over the hidden one.
    """
    out = dict(row)
    hidden = out.pop(MODEL_OUTPUT_FIELD, None)
    if isinstance(hidden, dict):
        for key, val in hidden.items():
            if out.get(key) is None:
                out[key] = val
    return out


# The side file next to a review file holds the model's answers the review rows do not
# show, one line per row: {"id", "pred_field", "model_output": {...}}. Only the fold
# reads it (bootstrap_seed.py hil), to tell a confirmed label from a corrected one.
MODEL_OUTPUT_SUFFIX = ".model_output.jsonl"


def fold_report_path(seed_path: str) -> str:
    """The fold report's path beside the seed it wrote (``x.jsonl`` -> ``x.fold.json``)."""
    stem = seed_path[: -len(".jsonl")] if seed_path.endswith(".jsonl") else seed_path
    return stem + ".fold.json"


def job_summary_sha256(summary: Any) -> Optional[str]:
    """A labeling job's identity: the SHA-256 of its summary as JSON.

    Key order and whitespace are ignored. The summary names no job ARN, and a
    re-run's differs, so a fold report is shown only beside the job it folded.

    Args:
        summary: The ``labeling_summary.json`` document.

    Returns:
        The hex digest, or ``None`` when ``summary`` is not a dict.
    """
    if not isinstance(summary, dict):
        return None
    return _digest.sha256_json(summary)


def model_output_path(review_path: str) -> str:
    """The side file beside ``review_path`` (``review.jsonl`` -> ``review.model_output.jsonl``)."""
    stem = review_path[: -len(".jsonl")] if review_path.endswith(".jsonl") else review_path
    return stem + MODEL_OUTPUT_SUFFIX


def model_output_candidates(review_path: str) -> List[str]:
    """The side files a fold looks for next to ``review_path``, in order.

    :func:`model_output_path`'s name first; then, for a job's ``<stem>_rows.jsonl``
    review file, its ``<stem>_model_output.jsonl`` (``review_rows.jsonl`` ->
    ``review_model_output.jsonl``).
    """
    out = [model_output_path(review_path)]
    stem = review_path[: -len(".jsonl")] if review_path.endswith(".jsonl") else review_path
    if stem.endswith("_rows"):
        out.append(stem[: -len("_rows")] + "_model_output.jsonl")
    return out


def join_model_output(row: Dict[str, Any], answer: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """``row`` with a side-file record's model output put back.

    A field the row carries again with a non-null value is kept.
    """
    out = dict(row)
    for key, val in (answer or {}).items():
        if out.get(key) is None:
            out[key] = val
    return out


# Rows people labeled. bootstrap_seed.py merge marks each row it gives a person's label
# LABELED_BY_FIELD: LABELED_BY_PEOPLE, beside the seed row's provenance. Only merge
# writes the marker and a labeling run drops it from its input rows, so a model label
# over a merged row is never taken for a person's; `provenance` alone is not enough,
# since a labeling run copies every input field. A fold drops marked rows from the
# review rows, and report.py counts them as labeled by people.
LABELED_BY_FIELD = "_labeled_by"
LABELED_BY_PEOPLE = "people"
PEOPLE_LABELED_PROVENANCE = ("human_verified", "spot_checked")


def labeled_by_people(row: Any) -> bool:
    """True when a labeled row carries a person's label.

    The seed merge marked it (``LABELED_BY_FIELD``) and its ``provenance`` is
    ``human_verified`` or ``spot_checked``.
    """
    return (
        isinstance(row, dict)
        and row.get(LABELED_BY_FIELD) == LABELED_BY_PEOPLE
        and row.get("provenance") in PEOPLE_LABELED_PROVENANCE
    )


# The routing tag merge writes on a reviewed row that was routed without one.
REVIEWED_UNTAGGED = "routed"
# Routing tags of rows drawn at random from the accepted band (the job's
# "calibration", an older calibration's "audit", and the legacy "spot_check").
SPOT_CHECK_REASONS = ("calibration", "audit", "spot_check")
