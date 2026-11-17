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
"""The labeling seed and the people's labels on the labeled rows: append, merge, overlay.

The seed is the people's labels, never the model's. This module puts those labels where
the pipeline reads them:

* :func:`check_seed_append` decides whether a review fold's rows may be appended to the
  seed (the same fold run again is a no-op; a different review of the same rows is
  refused);
* :func:`merge_seed_labels` puts the seed's human labels on the labeled corpus rows they
  label, moving the model's answer aside, and :func:`drop_people_rows_from_review` takes
  the rows people labeled already out of a job's review set, with
  :func:`score_known_spot_check` scoring the spot-check rows among them;
* :func:`build_overlay` turns the seed into the patch a dataset-transformation job applies
  to the labeled DataSet, :func:`overlay_seed_rows` reads each seed file under its own
  gold field, :func:`overlay_spec` is the job's instruction and :func:`overlay_handoff`
  the hand-off record;
* :func:`review_verdict` is the one rule that says whether a person's label confirms the
  model's.

Everything takes rows and returns rows and reports; the caller reads and writes the files.
"""

from __future__ import absolute_import

import logging
import re
from collections import defaultdict

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import (
    DEFAULT_CORRECT_TOLERANCE,
    GT_FIELD_FOR_PRED,
    TASK_FIELDS,
    task_correct,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    BLIND_FIELD_SET,
    LABELED_BY_FIELD,
    LABELED_BY_PEOPLE,
    LEGACY_REVIEW_REASON_FIELDS,
    MODEL_OUTPUT_FIELD,
    MODEL_OUTPUT_FIELDS,
    REVIEW_REASON_FIELD,
    REVIEWED_UNTAGGED,
    SEED_METADATA_FIELDS,
    blind_row,
    join_model_output,
    labeled_by_people,
    review_reason_of,
    unblind_row,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export import (
    AUTO_APPROVED,
    BOOTSTRAP_TO_PROVENANCE,
    GA_SPLIT_FIELD,
    GA_SPOT_CHECK_SPLIT,
    HUMAN_PROVENANCE,
    HUMAN_VERIFIED,
    PROVENANCE_VALUES,
    id_key,
    same_label,
)

_log = logging.getLogger(__name__)

# The model's answer field on a spot-check row when the prediction field is not given.
PRED_FIELDS = ("label", "score", "judge_score", "entities", "answer")
NO_MODEL_ANSWER = "no_model_answer"
NOT_SCORED = "not_scored"
# Prediction fields a review fold already warned about (no task rule scores them): warn
# once per field, not once per row.
_WARNED_UNSCORED_FIELDS = set()

# The fields an overlay row replaces on the labeled DataSet row with the same id, besides
# the prediction field; the model's answer moves under MODEL_OUTPUT_FIELD.
OVERLAY_FIELDS = ("provenance", LABELED_BY_FIELD, REVIEW_REASON_FIELD)
# The model-output fields the overlay moves into MODEL_OUTPUT_FIELD with the prediction,
# as the labeling job's finalizer does on a row people labeled, plus the confidence_score
# copy it writes.
OVERLAY_MODEL_FIELDS = (
    "reasoning",
    "prediction_error",
    "confidence",
    "confidence_score",
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
)
# The key overlay_seed_rows copies each seed row's gold value under, whatever the file
# names it, so one build_overlay pass reads every seed file.
OVERLAY_GOLD_KEY = "__overlay_gold__"

# A seed row's `source` the labeling job writes for a model's label: `jury`, `model`,
# `resample:<model id>`, or a Bedrock model id or inference profile. A model id is the
# container's form, dotted and ending in `:<version>`, or a Bedrock provider's id with no
# version, which the container's form misses.
_VERSIONED_MODEL_ID = re.compile(r"^[a-z0-9-]+(\.[A-Za-z0-9-]+)+:\d+$")
_BEDROCK_PROVIDERS = (
    "ai21",
    "amazon",
    "anthropic",
    "cohere",
    "deepseek",
    "google",
    "meta",
    "minimax",
    "mistral",
    "moonshot",
    "moonshotai",
    "nvidia",
    "openai",
    "qwen",
    "stability",
    "twelvelabs",
    "writer",
)
_UNVERSIONED_MODEL_ID = re.compile(
    r"^(?:[a-z]{2,6}(?:-[a-z]+)?\.)?(?:"
    + "|".join(_BEDROCK_PROVIDERS)
    + r")\.(?=[A-Za-z0-9.-]*\d)[A-Za-z0-9.-]+$"
)


class SeedAppendRefused(DataPreparerOperatorError):
    """Some, but not all, of a fold's ids are in the seed already, or one is there with another label."""


# ---- the verdict rule ---------------------------------------------------------------------


def review_verdict(record, gold, qa_match=None, tolerance=None):
    """Whether a person's label ``gold`` confirms the model's proposed label in a side-file record.

    The rule is the job's row rule (:func:`~.correctness.task_correct`) for the task whose
    prediction field the record names: a classification label compared as text (a
    person's 1 confirms the model's "1"); for qa ``qa_match`` and for llm_judge
    ``tolerance``, at the job's defaults (exact match, 0.5) when ``None``.

    Args:
        record: ``{"pred_field", "model_output"}`` from the job's ``review.model_output.jsonl``.
        gold: The person's label.
        qa_match: The qa match rule.
        tolerance: The llm_judge tolerance.

    Returns:
        ``True`` (confirms), ``False`` (corrects), ``NO_MODEL_ANSWER`` when the model gave no
        answer for the row, or ``None`` when the record names no prediction a task rule
        scores (logged once per field).
    """
    pred_field = record.get("pred_field")
    answer = record.get("model_output") or {}
    if not pred_field or pred_field not in answer:
        return None
    task = next((t for t, (p, _g) in TASK_FIELDS.items() if p == pred_field), None)
    if task is None:
        if pred_field not in _WARNED_UNSCORED_FIELDS:
            _WARNED_UNSCORED_FIELDS.add(pred_field)
            _log.warning(
                "review rows name a prediction field %r that no task rule scores (not in "
                "TASK_FIELDS); they are counted as not_scored, not confirmed or corrected. "
                "Pass a scored task's prediction field if these should be scored.",
                pred_field,
            )
        return None
    if answer[pred_field] is None:
        return NO_MODEL_ANSWER
    if tolerance is None:
        tolerance = DEFAULT_CORRECT_TOLERANCE
    return bool(
        task_correct(task, answer[pred_field], gold, tolerance=tolerance, qa_match=qa_match)
    )


def spot_check_confirms(answered, pred_field, gold, qa_match=None, tolerance=None):
    """Whether a person's label confirms the model's answer on an unblinded spot-check row.

    The answer is ``pred_field``, else the first ``PRED_FIELDS`` key the row carries.

    Returns:
        ``True``, ``False``, or ``None`` when there is nothing to compare (no model answer,
        or one no task rule scores).
    """
    field = pred_field or next((k for k in PRED_FIELDS if k in answered), None)
    v = review_verdict(
        {"pred_field": field, "model_output": answered},
        gold,
        qa_match=qa_match,
        tolerance=tolerance,
    )
    return None if v is None or v == NO_MODEL_ANSWER else v


# ---- append ---------------------------------------------------------------------------------


def check_seed_append(seed_rows, rows, seed_name, id_field="id", gt_field="gt_label"):
    """Whether the fold's ``rows`` may be appended to the seed.

    When every row's id is already in the seed as a hil row with the same ``gt_field``
    label (the same fold run again), nothing is to be written. When only some are, or one
    is there with another label (a later review of the same rows), or an id matches a seed
    row that is not a hil row, the append is refused. Ids join as strings.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import labels

        appended, already = labels.check_seed_append(seed_rows, fold_rows, "proj/seed.jsonl")
        if appended:
            write_jsonl("proj/seed.jsonl", seed_rows + fold_rows)

    Args:
        seed_rows: The seed's rows as they are.
        rows: The fold's rows.
        seed_name: The seed's path, for the message.
        id_field: The id field.
        gt_field: The gold field.

    Returns:
        ``(appended, already_in_seed_ids)``: ``appended`` is ``len(rows)`` when the caller
        should append them, 0 otherwise.

    Raises:
        SeedAppendRefused: A different review of the same rows, or an id clash.
    """
    hil_labels, other_ids = defaultdict(list), set()
    for r in seed_rows:
        if isinstance(r, dict):
            if r.get("_bootstrap") == "hil":
                hil_labels[id_key(r.get(id_field))].append(r.get(gt_field))
            else:
                other_ids.add(id_key(r.get(id_field)))
    new_ids = [id_key(r.get(id_field)) for r in rows]
    present = sorted({k for k in new_ids if k in hil_labels or k in other_ids}, key=str)
    relabeled = sorted(
        {
            id_key(r.get(id_field))
            for r in rows
            if not all(
                same_label(old, r.get(gt_field))
                for old in hil_labels.get(id_key(r.get(id_field)), ())
            )
        },
        key=str,
    )
    if (
        present
        and len(present) == len(set(new_ids))
        and not (set(present) & other_ids)
        and not relabeled
    ):
        return 0, present
    if present:
        raise SeedAppendRefused(
            f"{len(present)} of the {len(set(new_ids))} folded id(s) are already in "
            f"{seed_name} ({', '.join(map(str, present[:10]))}"
            f"{', ...' if len(present) > 10 else ''}"
            + (
                f"; {len(relabeled)} with another label there: "
                f"{', '.join(map(str, relabeled[:10]))}{', ...' if len(relabeled) > 10 else ''}"
                if relabeled
                else ""
            )
            + "); a different review of the same "
            "rows or an id clash with the seed. Nothing was appended: fold only the new "
            "rows, or remove the old ones from the seed first"
        )
    if not rows:
        return 0, []
    return len(rows), []


# ---- merge ----------------------------------------------------------------------------------


def seed_input_fields(seed_row, id_field, gt_field, pred_field):
    """The input fields a seed row carries.

    Every key but the id, the gold, the prediction and the seed and model fields.
    """
    skip = {id_field, gt_field, pred_field, "confidence_score", LABELED_BY_FIELD}
    return [
        k
        for k in seed_row
        if k not in skip and k not in SEED_METADATA_FIELDS and k not in MODEL_OUTPUT_FIELDS
    ]


def seed_review_tag(seed_row):
    """The routing tag a merged row takes from its seed row.

    The seed row's own (``_review_reason`` or a legacy name); ``REVIEWED_UNTAGGED`` when the
    seed row was routed without one (it carries the model's answer); ``None`` for a row of
    the cold-start sample.
    """
    tag = review_reason_of(seed_row)
    if tag is not None:
        return tag
    if isinstance(seed_row.get(MODEL_OUTPUT_FIELD), dict) or any(
        k in seed_row for k in BLIND_FIELD_SET
    ):
        return REVIEWED_UNTAGGED
    return None


def merge_seed_labels(  # pylint: disable=too-many-locals,too-many-branches
    labels_rows, seed_rows, id_field="id", gt_field="gt_label", pred_field="label"
):
    """Put the people's labels from a seed on the labeled corpus rows they label.

    For every seed row with a human provenance (or the legacy ``_bootstrap: "hil"`` marker
    alone) and a gold value, the labeled row with the same id (ids compare as strings)
    gets that gold as its ``pred_field``, that provenance, the labeled-by-people marker and
    the seed row's routing tag. The model's answer, with ``_failed`` and ``_failed_reason``,
    moves under the model-output field first; a row with no model answer gets none. A row
    merged before keeps the model's answer it stored then. Rows the seed does not name are
    returned unchanged, and an ``auto_approved`` seed row is not merged.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import labels

        rows, report = labels.merge_seed_labels(labeled_rows, seed_rows, pred_field="label")
        report["label_changed"], report["missing_from_labels"]

    Returns:
        ``(rows, report)``: ``label_changed`` counts the merged rows whose person's label
        differs from the model's, ``missing_from_labels`` lists the seed ids no labeled row
        has, and ``content_mismatch`` the seed ids whose labeled row differs from the seed
        row on an input field.
    """
    gold, prov, seed_by_id = {}, {}, {}
    for s in seed_rows:
        if not isinstance(s, dict):
            continue
        p = s.get("provenance")
        if p is None and s.get("_bootstrap") == "hil":
            p = HUMAN_VERIFIED
        key = id_key(s.get(id_field))
        if p not in HUMAN_PROVENANCE or key is None or s.get(gt_field) is None:
            continue
        gold[key] = s[gt_field]
        prov[key] = p
        seed_by_id[key] = s
    out, seen, mismatch = [], set(), []
    merged = changed = already = 0
    for r in labels_rows:
        key = id_key(r.get(id_field)) if isinstance(r, dict) else None
        if key not in gold:
            out.append(r)
            continue
        seen.add(key)
        s = seed_by_id[key]
        if any(r.get(k) != s[k] for k in seed_input_fields(s, id_field, gt_field, pred_field)):
            mismatch.append(key)
        if labeled_by_people(r) or (
            r.get("provenance") in HUMAN_PROVENANCE and isinstance(r.get(MODEL_OUTPUT_FIELD), dict)
        ):
            row = dict(r)
            already += 1
        else:
            row = blind_row(r, pred_field)
            for k in ("_failed", "_failed_reason"):
                if k in row:
                    row[MODEL_OUTPUT_FIELD][k] = row.pop(k)
            if not row[MODEL_OUTPUT_FIELD]:
                del row[MODEL_OUTPUT_FIELD]
        row[pred_field] = gold[key]
        row["provenance"] = prov[key]
        row[LABELED_BY_FIELD] = LABELED_BY_PEOPLE
        for k in (REVIEW_REASON_FIELD,) + LEGACY_REVIEW_REASON_FIELDS:
            row.pop(k, None)
        tag = seed_review_tag(s)
        if tag is not None:
            row[REVIEW_REASON_FIELD] = tag
        merged += 1
        changed += int(
            not same_label((row.get(MODEL_OUTPUT_FIELD) or {}).get(pred_field), gold[key])
        )
        out.append(row)
    report = {
        "mode": "merge",
        "labels_rows": len(labels_rows),
        "seed_human_rows": len(gold),
        "merged": merged,
        "label_changed": changed,
        "already_merged": already,
        "missing_from_labels": sorted(k for k in gold if k not in seen),
        "content_mismatch": sorted(mismatch),
    }
    return out, report


def drop_people_rows_from_review(review_rows, model_output_rows, people_ids, id_field="id"):
    """The review set without the rows people have labeled already.

    The labeling job routes a corpus row it knows nothing about, so a cold-start dev or val
    row the model was unsure about lands in its review set, and people would be asked to
    label it twice.

    Returns:
        ``(review_rows, model_output_rows, dropped_ids)``; the side file is keyed by ``id``.
    """
    people = {id_key(k) for k in people_ids}
    kept, dropped = [], []
    for r in review_rows:
        key = id_key(r.get(id_field)) if isinstance(r, dict) else None
        (dropped if key in people else kept).append(r)
    dropped_ids = {id_key(r.get(id_field)) for r in dropped}
    side = [
        m
        for m in model_output_rows
        if not (isinstance(m, dict) and id_key(m.get("id")) in dropped_ids)
    ]
    return kept, side, sorted(k for k in dropped_ids if k is not None)


def known_spot_check_path(review_path):
    """Where the verdicts of the spot-check rows people had labeled already go: ``review.known_spot_check.json``."""
    stem = review_path[: -len(".jsonl")] if review_path.endswith(".jsonl") else review_path
    return stem + ".known_spot_check.json"


def score_known_spot_check(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    dropped_rows,
    side_rows,
    labeled_rows,
    id_field="id",
    pred_field="label",
    qa_match=None,
    tolerance=None,
):
    """Score the spot-check rows dropped from a review because people labeled them already.

    Scored with that known label against the model's answer, as a returned spot-check row
    is. Without this the draw would shrink below its 19/29/59 rows while the withdraw count
    stays at 7.

    Returns:
        ``{"rows", "corrected", "no_model_answer", "ids"}``, or ``None`` when none was a
        spot-check row.
    """
    spot = [
        r
        for r in dropped_rows
        if isinstance(r, dict) and r.get(GA_SPLIT_FIELD) == GA_SPOT_CHECK_SPLIT
    ]
    if not spot:
        return None
    side = {id_key(m.get("id")): m for m in side_rows if isinstance(m, dict)}
    gold = {
        id_key(r.get(id_field)): r.get(pred_field)
        for r in labeled_rows
        if isinstance(r, dict) and labeled_by_people(r)
    }
    corrected = no_answer = 0
    ids = []
    for r in spot:
        key = id_key(r.get(id_field))
        ids.append(key)
        record = side.get(key) or {}
        answered = join_model_output(unblind_row(r), record.get("model_output"))
        ok = spot_check_confirms(
            answered,
            record.get("pred_field") or pred_field,
            gold.get(key),
            qa_match=qa_match,
            tolerance=tolerance,
        )
        if ok is None:
            no_answer += 1
        elif not ok:
            corrected += 1
    return {
        "rows": len(spot),
        "corrected": corrected,
        "no_model_answer": no_answer,
        "ids": sorted(ids),
    }


# ---- overlay --------------------------------------------------------------------------------


def model_source(source):
    """True when a seed row's ``source`` names a model (the labeling job's own values or a model id)."""
    if not isinstance(source, str):
        return False
    return (
        source in ("jury", "model")
        or source.startswith("resample:")
        or bool(_VERSIONED_MODEL_ID.match(source))
        or bool(_UNVERSIONED_MODEL_ID.match(source))
    )


def overlay_provenance(s):  # pylint: disable=too-many-return-statements
    """The provenance an overlay takes a seed row's gold under, or ``None`` to leave it out.

    The container's rule for human gold: the tag is ``provenance``, else ``_provenance``,
    else the legacy ``_bootstrap`` marker (hil = human_verified, self = auto_approved). Two
    tags that disagree, an unknown tag or marker, or a marker that contradicts the tag
    leave the row out, and so does ``auto_approved``. A person's tag wins over a model
    ``source``. A seed row with no tag at all is a person's label too, unless its
    ``source`` names a model.
    """
    a, b = s.get("provenance"), s.get("_provenance")
    if a is not None and b is not None and a != b:
        return None
    tag = a if a is not None else b
    raw = s.get("_bootstrap")
    marker = None
    if raw is not None:
        marker = BOOTSTRAP_TO_PROVENANCE.get(
            raw.strip().casefold() if isinstance(raw, str) else None, "unknown"
        )
        if marker == "unknown":
            return None
    if tag is not None:
        if tag not in PROVENANCE_VALUES:
            return None
        if marker is not None and (marker == AUTO_APPROVED) != (tag == AUTO_APPROVED):
            return None
        return tag
    if marker is not None:
        return marker
    if model_source(s.get("source")):
        return None
    return HUMAN_VERIFIED


def untagged(s):
    """True for a seed row with no provenance tag and no ``_bootstrap`` marker."""
    return all(s.get(k) is None for k in ("provenance", "_provenance", "_bootstrap"))


def overlay_seed_rows(seed_files, deliverable, gt_field=None, pred_field="label"):
    """The overlay's seed rows, each file read under its own gold field.

    The gold field of a file is the first of ``gt_field``, the task's default gold field
    and the corpus run's gold column (``label_field`` in the deliverable record) that any
    row of the file carries. Each row gets that value under ``OVERLAY_GOLD_KEY``.

    Args:
        seed_files: ``[(path, rows)]``.
        deliverable: The corpus run's ``labeled_dataset.json`` document.
        gt_field: The gold field the caller named, if any.
        pred_field: The prediction field.

    Returns:
        ``(rows, {path: gold field or None})``.

    Raises:
        DataPreparerOperatorError: A file has rows but none of those fields.
    """
    candidates = []
    for f in (
        gt_field,
        GT_FIELD_FOR_PRED.get(pred_field, "gt_label"),
        deliverable.get("label_field") if isinstance(deliverable, dict) else None,
    ):
        if f and f not in candidates:
            candidates.append(f)
    rows, fields = [], {}
    for path, file_rows in seed_files:
        dict_rows = [r for r in file_rows if isinstance(r, dict)]
        field = next((f for f in candidates if any(r.get(f) is not None for r in dict_rows)), None)
        if field is None:
            if dict_rows:
                cols = sorted({k for r in dict_rows for k in r})[:12]
                raise DataPreparerOperatorError(
                    f"--seed {path}: no row has a gold value under {' / '.join(candidates)} "
                    f"(its columns: {', '.join(cols)}), so none of its people's labels would "
                    "reach the overlay. Pass --gt-field <its label column>"
                )
            fields[path] = None
            continue
        fields[path] = field
        rows.extend(dict(r, **{OVERLAY_GOLD_KEY: r.get(field)}) for r in dict_rows)
    return rows, fields


def seed_rows_for_overlay(seed_rows, deliverable):
    """The seed rows the overlay may patch onto the labeled DataSet, and how many were left out.

    When the corpus run's record says the job labeled eval rows that are not corpus rows
    (``eval_rows_labeled_for_scoring`` above 0), the seed's ids are not all this corpus's
    ids for the same rows, so only the review fold's rows (they carry ``source_export``)
    are kept. A record without the count keeps every row.
    """
    outside = (
        deliverable.get("eval_rows_labeled_for_scoring") if isinstance(deliverable, dict) else None
    )
    if not isinstance(outside, int) or isinstance(outside, bool) or outside <= 0:
        return list(seed_rows), 0
    kept = [s for s in seed_rows if isinstance(s, dict) and s.get("source_export") is not None]
    return kept, len(seed_rows) - len(kept)


def build_overlay(seed_rows, id_field="id", gt_field="gt_label", pred_field="label"):
    """The people's labels as overlay rows for the corpus run's labeled DataSet.

    One row per seed id with a human provenance (:func:`overlay_provenance`) and a gold
    value, carrying the id, the person's label under ``pred_field``, the provenance, the
    labeled-by-people marker and the seed row's routing tag. No input field is copied: the
    overlay is a patch on rows the DataSet already holds. A later seed row for an id wins.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import labels

        overlay_rows, report = labels.build_overlay(seed_rows, pred_field="label")
        write_jsonl("proj/work/overlay.jsonl", overlay_rows)

    Returns:
        ``(rows, report)``.
    """
    by_id, untagged_rows = {}, 0
    for s in seed_rows:
        if not isinstance(s, dict):
            continue
        p = overlay_provenance(s)
        key = id_key(s.get(id_field))
        if p not in HUMAN_PROVENANCE or key is None or s.get(gt_field) is None:
            continue
        if untagged(s):
            untagged_rows += 1
        row = {
            id_field: s.get(id_field),
            pred_field: s[gt_field],
            "provenance": p,
            LABELED_BY_FIELD: LABELED_BY_PEOPLE,
        }
        tag = seed_review_tag(s)
        if tag is not None:
            row[REVIEW_REASON_FIELD] = tag
        by_id.pop(key, None)
        by_id[key] = row
    rows = list(by_id.values())
    tags = defaultdict(int)
    for r in rows:
        tags[r.get(REVIEW_REASON_FIELD) or "sample"] += 1
    return rows, {
        "mode": "overlay",
        "overlay_rows": len(rows),
        "by_source": dict(tags),
        "untagged_as_human": untagged_rows,
    }


def overlay_spec(pred_field="label", id_field="id"):
    """The overlay CodeGen job's instruction, as it ran on the service.

    DT samples each input before the full run, so an overlay id with no labeled row is not
    an error, and the code returns ``overlay_matched`` for the hand-off's ``Check``.
    """
    fields = ", ".join(repr(f) for f in (pred_field, *OVERLAY_FIELDS))
    moved = ", ".join(repr(f) for f in (pred_field, *OVERLAY_MODEL_FIELDS))
    return (
        f"Two inputs. input_0 is the labeled dataset (JSONL, every row has {id_field!r}). "
        f"input_1 is the overlay: rows with {id_field!r} and {fields} (a row may lack "
        f"{REVIEW_REASON_FIELD!r}). Output every input_0 row in its original order with "
        f"every field kept, except as follows. For each input_0 row whose {id_field!r} "
        f"matches an overlay row: if the row has no {MODEL_OUTPUT_FIELD!r} field, remove "
        f"from it each of {moved} that it has and put them, under the same names, in a "
        f"new dict field {MODEL_OUTPUT_FIELD!r}; a row that already has "
        f"{MODEL_OUTPUT_FIELD!r} keeps it unchanged. Then set {fields} to the overlay "
        f"row's values (leave a field the overlay row lacks as it is), set 'reasoning' to "
        f"the empty string and 'source' to 'human'. Rows with no match are unchanged. Do "
        f"not raise on an overlay id with no input_0 row (the inputs may be samples). "
        f"Return the stats dict with total_records and transformed_records both = the "
        f"number of output rows, and overlay_matched = the number of input_0 rows that "
        f"matched an overlay row. Write JSONL."
    )


def overlay_handoff(
    deliverable, overlay_path, pred_field="label", id_field="id", overlay_rows=None
):
    """The dataset-transformation hand-off for an overlay, in the five steps the full loop runs.

    ``Register`` the overlay rows, ``DtView`` of the labeled DataSet, one CodeGen
    ``Transform`` over [the view, the overlay], ``Check`` (rows in = rows out, and the full
    run's ``overlay_matched`` from the job's CloudWatch log) and ``Promote`` to the labeled
    DataSet's next version.

    Raises:
        DataPreparerOperatorError: The deliverable record names no ``dataset_arn``.
    """
    arn = deliverable.get("dataset_arn")
    if not arn:
        raise DataPreparerOperatorError(
            "the deliverable record names no dataset_arn (the corpus run's labeled DataSet)"
        )
    name = deliverable.get("dataset_name") or "labeled"
    count = overlay_rows if overlay_rows is not None else "the overlay's row count"
    return {
        "Operation": "overlay_by_id",
        "Inputs": {
            "LabeledDatasetArn": arn,
            "OverlayRows": overlay_path,
            "OverlayRowCount": overlay_rows,
            "OverlayDatasetArn": "<the dataset_arn the hub_ops register action prints for the overlay>",
            "Order": ["<the dataset_arn the hub_ops dt-view action prints>", "OverlayDatasetArn"],
        },
        "KeyField": id_field,
        "ReplaceFields": [pred_field, *OVERLAY_FIELDS],
        "ModelAnswerTo": MODEL_OUTPUT_FIELD,
        "Register": (
            f"code_templates/hub_ops.py with ACTION = 'register', LABELS = {overlay_path!r}, "
            f"NAME = '{name}-corrections', S3_PREFIX = '<s3://...>'"
        ),
        "DtView": f"code_templates/hub_ops.py with ACTION = 'dt-view', ARN = {arn!r}, PROJECT_DIR = '<proj>'",
        "Transform": (
            "one dataset-transformation CodeGen job with AutoExecute true over "
            "Inputs.Order, Specification as its instruction; its output, in "
            "OutputDatasetArns, is a version of the view"
        ),
        "Specification": overlay_spec(pred_field, id_field),
        "Check": (
            "show the user the code the job ran (TransformationCodeS3Uri on the "
            "output), confirm in its report.json (TransformationReportS3Uri) that "
            "the first input's record count equals the output's, and confirm that "
            f"the full run's overlay_matched equals {count}. The code's stats are "
            "in the job's CloudWatch log, not report.json: the last \"Code execution "
            'complete" event of `aws logs filter-log-events --log-group-name '
            "/aws/sagemaker/Job/DataTransformation/ --log-stream-name-prefix "
            "<job name>/ --filter-pattern '\"Code execution complete\"'` (stats."
            "overlay_matched; the sample run's event comes first with a lower count, "
            "ignore it). If overlay_matched is lower, that many corrections are not "
            "in the output: tell the user before promoting it"
        ),
        "Promote": (
            "code_templates/hub_ops.py with ACTION = 'promote', FROM_ARN = <the OutputDatasetArns "
            f"entry>, TO_ARN = {arn!r}, PROJECT_DIR = '<proj>'"
        ),
        "LaterExport": (
            f"after an earlier promote, the labeled DataSet's latest version is "
            f"the one promote added, not {arn}: pass that version's ARN as "
            f"DtView's ARN and Promote's TO_ARN (promote refuses a version "
            f"that is not the latest)"
        ),
        "Output": {
            "DatasetName": name,
            "As": "the labeled DataSet's next version, the dataset_arn the promote action prints",
        },
        "DtJobs": 1,
    }


def fill_missing_ids(rows, id_field="id"):
    """Give every row with no ``id``, or a null one, ``str(i)`` (its position), in place.

    Only ``id`` is filled, as the labeling job fills it.

    Returns:
        How many were filled.
    """
    if id_field != "id":
        return 0
    filled = 0
    for i, r in enumerate(rows):
        if isinstance(r, dict) and r.get("id") is None:
            r["id"] = str(i)
            filled += 1
    return filled
