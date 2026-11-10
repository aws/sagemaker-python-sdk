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
"""Reading a human review's export: the people's labels, one decided label per row.

A review tool returns its rows either flat (an id and a label column per row) or as
Label Studio tasks (the record under ``data``, the reviewers' answers under
``annotations``). :func:`flatten_export` turns both into flat rows; :func:`decided_by_id`
folds them into one decided label per id, naming the reviewers, the rows nobody decided
and the ids whose reviewers disagree (a disagreement is never resolved by file order).
The label comparisons (:func:`same_label`, :func:`same_answer`) hold for list and dict
labels in any order, so NER spans marked in another order agree.

The provenance values a seed row carries (``human_verified``, ``spot_checked``,
``auto_approved``) and :func:`provenance_for_review`, which gives a reviewed row its
provenance from its routing tag, live here too: the labeling job scores only a person's
row as gold.
"""

from __future__ import absolute_import

from collections import defaultdict

from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import ner_span_set
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    SPOT_CHECK_REASONS,
    review_reason_of,
)

# Seed-row provenance values. Only AUTO_APPROVED is model-derived; the labeling job never
# scores it as gold, so it backs no certificate.
HUMAN_VERIFIED = "human_verified"
SPOT_CHECKED = "spot_checked"
AUTO_APPROVED = "auto_approved"
PROVENANCE_VALUES = (HUMAN_VERIFIED, SPOT_CHECKED, AUTO_APPROVED)
HUMAN_PROVENANCE = (HUMAN_VERIFIED, SPOT_CHECKED)
# Legacy `_bootstrap` mode -> provenance (self promotes the model label to gold).
BOOTSTRAP_TO_PROVENANCE = {"self": AUTO_APPROVED, "hil": HUMAN_VERIFIED}
UNRESOLVED_VERDICTS = ("unsure", "skip", "skipped")
VERDICT_FIELDS = ("verdict", "status")
ANNOTATOR_FIELDS = (
    "annotator",
    "annotator_id",
    "reviewer",
    "reviewer_id",
    "completed_by_profile",
    "completed_by",
)
# The labeling job's review rows mark a spot-check row with split: spot_check instead of
# _review_reason.
GA_SPLIT_FIELD = "split"
GA_SPOT_CHECK_SPLIT = "spot_check"

# A Label Studio export keeps each task's record under `data` and the reviewers' answers
# under `annotations`: one `result` list per annotation, one item per answered control
# of the label config (named by `from_name`), and the reviewer beside it.
LS_SPAN_TYPES = ("labels", "hypertextlabels", "paragraphlabels")
LS_VALUE_KEYS = ("choices", "taxonomy", "text", "rating", "number", "datetime")
# The gold field an ner fold names (--gt-field gt_entities): its export label column is
# a span control even when no annotation marked a span.
NER_GT_FIELD = "gt_entities"
# A blank export label is no label, except for a qa gold field.
EMPTY_ANSWER_GOLD_FIELDS = ("gt_answers",)


def provenance_for_review(row):
    """The provenance of one reviewed row, from its routing tag.

    A random accepted-band draw is ``spot_checked``, every other reviewed row
    ``human_verified``. A job's review row marks the draw with ``split: spot_check``.
    """
    if (
        review_reason_of(row) in SPOT_CHECK_REASONS
        or row.get(GA_SPLIT_FIELD) == GA_SPOT_CHECK_SPLIT
    ):
        return SPOT_CHECKED
    return HUMAN_VERIFIED


def is_unresolved(h):
    """True when a returned row says the reviewer could not decide (unsure / skip / skipped)."""
    return any(
        str(h.get(k) or "").strip().casefold() in UNRESOLVED_VERDICTS for k in VERDICT_FIELDS
    )


def annotator_of(h):
    """The reviewer id a returned row names, or ``None``."""
    for k in ANNOTATOR_FIELDS:
        v = h.get(k)
        if v is not None and v != "":
            return v
    return None


def is_label_studio_task(h):
    """True for a Label Studio export row: the record under ``data``, the answers under ``annotations``."""
    return (
        isinstance(h, dict)
        and isinstance(h.get("data"), dict)
        and isinstance(h.get("annotations"), list)
    )


def _ls_reviewer(ann):
    """The reviewer one Label Studio annotation names.

    The profile or email the export resolved (``completed_by_profile``, ``annotator``),
    else Label Studio's user.
    """
    for k in ("completed_by_profile", "annotator"):
        if ann.get(k) not in (None, ""):
            return ann[k]
    cb = ann.get("completed_by")
    if isinstance(cb, dict):
        return cb.get("email") or cb.get("username") or cb.get("id")
    return cb


def _ls_spans(item):
    """The entities one span item marks: one ``{text, type, start, end}`` per label it carries."""
    v = item.get("value") or {}
    return [
        {"text": v.get("text"), "type": lab, "start": v.get("start"), "end": v.get("end")}
        for lab in (v.get("labels") or [])
    ]


def _ls_value(item):
    """The answer in one result item: its choice, text, rating or number (one alone, several as a list)."""
    v = item.get("value") or {}
    for k in LS_VALUE_KEYS:
        if k in v:
            x = v[k]
            return x[0] if isinstance(x, list) and len(x) == 1 else x
    return None


def span_fields(gt_field, label_field):
    """The export label columns to read as span controls: the label field on an ner fold, none otherwise."""
    return (label_field,) if gt_field == NER_GT_FIELD else ()


def flatten_export(rows, id_field="id", span_fields=()):  # pylint: disable=redefined-outer-name
    """The export rows as flat rows.

    A flat row (an id and a label column) is kept as it is. A Label Studio task gives one
    row per annotation: ``id_field`` (and ``id`` / ``record_id``) from its ``data``, each
    answered control's value under the control's name, ``annotator`` and ``status: skipped``
    for a skipped annotation (``was_cancelled``). A span control the export uses gets ``[]``
    on an annotation that marked no span, since "no entities" is an answer. When no
    annotation marked a span at all, the ``span_fields`` controls no annotation answered
    otherwise are the span controls.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import review_export

        flat = review_export.flatten_export(export_rows, id_field="id")
        decided = [r for r in flat if not review_export.is_unresolved(r)]

    Args:
        rows: The export rows.
        id_field: The record id field.
        span_fields: The label columns that are span controls (:func:`span_fields`).

    Returns:
        The flat rows.
    """
    rows = list(rows)
    span_controls = sorted(
        {
            item["from_name"]
            for h in rows
            if is_label_studio_task(h)
            for ann in h["annotations"]
            if isinstance(ann, dict)
            for item in (ann.get("result") or [])
            if isinstance(item, dict)
            and item.get("from_name")
            and item.get("type") in LS_SPAN_TYPES
        }
    )
    if not span_controls and span_fields:
        answered = {
            item["from_name"]
            for h in rows
            if is_label_studio_task(h)
            for ann in h["annotations"]
            if isinstance(ann, dict)
            for item in (ann.get("result") or [])
            if isinstance(item, dict) and item.get("from_name")
        }
        span_controls = sorted(set(span_fields) - answered)
    out = []
    for h in rows:
        if not is_label_studio_task(h):
            out.append(h)
            continue
        data = h["data"]
        base = {k: data[k] for k in dict.fromkeys((id_field, "id", "record_id")) if k in data}
        for ann in h["annotations"]:
            if not isinstance(ann, dict):
                continue
            row = dict(base)
            reviewer = _ls_reviewer(ann)
            if reviewer not in (None, ""):
                row["annotator"] = reviewer
            if ann.get("was_cancelled"):
                row["status"] = "skipped"
                out.append(row)
                continue
            for name in span_controls:
                row[name] = []
            for item in ann.get("result") or []:
                if not isinstance(item, dict) or not item.get("from_name"):
                    continue
                name = item["from_name"]
                if item.get("type") in LS_SPAN_TYPES:
                    row[name].extend(_ls_spans(item))
                elif name not in row:
                    row[name] = _ls_value(item)
            out.append(row)
    return out


def id_key(v):
    """An id as the rows are joined: its string, or ``None``."""
    return None if v is None else str(v)


def class_key(v):
    """A hashable key for a label: the label, or the JSON form of a list or dict label."""
    try:
        hash(v)
        return v
    except TypeError:
        return _digest.canonical_json(v)


def returned_label(h, gt_field="gt_label", label_field="label"):
    """The label a returned row gives.

    ``gt_field`` as it is (a falsy gold such as 0 or "" is kept), else ``label_field``,
    else ``None``. A blank ``label_field`` value is no label, except for a qa gold field.
    """
    lab = h.get(gt_field)
    if lab is not None:
        return lab
    lab = h.get(label_field)
    if isinstance(lab, str) and not lab.strip() and gt_field not in EMPTY_ANSWER_GOLD_FIELDS:
        return None
    return lab


def same_label(a, b):
    """Label equality that also holds for list and dict labels in any list order; scalars as they are."""
    if a == b:
        return True
    try:
        return _digest.canonical_json(_digest.canonical(a)) == _digest.canonical_json(
            _digest.canonical(b)
        )
    except (TypeError, ValueError):
        return False


def is_span_list(v):
    """True for an NER label: a non-empty list of span dicts (``text`` / ``type``)."""
    return (
        isinstance(v, list)
        and bool(v)
        and all(isinstance(e, dict) and ("text" in e or "type" in e) for e in v)
    )


def same_answer(a, b):
    """Whether two people gave the same answer: NER spans as sets, any other label :func:`same_label`."""
    if is_span_list(a) or is_span_list(b):
        if isinstance(a, list) and isinstance(b, list):
            return ner_span_set(a) == ner_span_set(b)
    return same_label(a, b)


def decided_by_id(rows, id_field, gt_field, label_field):  # pylint: disable=too-many-locals
    """The decided label per id of the flat export rows.

    The ``gt_field`` value, else the ``label_field`` one, of every row that names an id
    and is not unsure / skipped. Several decided rows for one id (one per annotation of a
    task reviewed more than once) fold once when their labels agree and go into
    ``conflicts`` when they do not: gold is one label per row, so a disagreement is never
    resolved by file order.

    Args:
        rows: :func:`flatten_export`'s rows.
        id_field: The id field.
        gt_field: The gold field.
        label_field: The export's label column.

    Returns:
        ``(labels, annotators, counts, unresolved, conflicts, no_label, empty)``:
        ``annotators`` lists each id's distinct named reviewers in export order, ``counts``
        its decided rows, ``unresolved`` the ids with no decided row, ``no_label`` the ids
        whose rows have neither label column, and ``empty`` the number of rows whose label
        is blank.
    """
    labels, annotators, counts = {}, defaultdict(list), defaultdict(int)
    unresolved, conflicts, no_label = set(), set(), set()
    empty = 0
    for h in rows:
        if not isinstance(h, dict):
            continue
        hid = id_key(h.get(id_field))
        if hid is None:
            continue
        if is_unresolved(h):
            unresolved.add(hid)
            continue
        lab = returned_label(h, gt_field, label_field)
        if lab is None:
            if isinstance(h.get(label_field), str):
                empty += 1
            elif gt_field not in h and label_field not in h:
                no_label.add(hid)
            continue
        if hid in labels and not same_answer(labels[hid], lab):
            conflicts.add(hid)
        labels.setdefault(hid, lab)
        counts[hid] += 1
        who = annotator_of(h)
        if who is not None and who not in annotators[hid]:
            annotators[hid].append(who)
    for hid in conflicts:
        del labels[hid]
    return (labels, annotators, counts, unresolved - set(labels), conflicts, no_label, empty)


def conflict_hint(conflicts):
    """The message for ids whose reviewers disagree (the first ten named)."""
    shown = ", ".join(sorted(conflicts)[:10]) + (", ..." if len(conflicts) > 10 else "")
    return (
        f"{len(conflicts)} id(s) carry disagreeing labels from more than one "
        f"annotation ({shown}); they were left out of the seed. Adjudicate them in "
        "Label Studio (keep one agreed annotation per task) and re-export, or set "
        "the project's maximum_annotations to 1 for single-pass review"
    )
