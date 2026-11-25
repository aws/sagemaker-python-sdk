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
"""Golden-row cleaning before the corpus run: only a person changes a golden row.

The golden rows are the certificate's rows (the frozen test split). A review job flags
the ones whose labels look wrong; :func:`route` turns the flagged rows into a blind review
file and a manifest; :func:`apply` folds the person's export into a reviewed copy of the
golden rows (a row the person relabels leaves the certificate's rows, a confirmed row
stays, every other row is kept byte for byte); :func:`check_reviewed` (the corpus run's
``--golden-review``) derives everything again from the inputs and refuses any other
change.

The functions read the files they are given and return what to write (the review rows,
the reviewed copy's text, the manifest); the caller writes them. A read of the frozen
test split goes through the split guard, whose test-read entry comes back in ``events``.
Every refusal is a :class:`GoldenReviewError`.
"""

from __future__ import absolute_import

import collections
import json
import os
from typing import Any, Dict, List, Optional

from pydantic import Field

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    BLIND_FIELD_SET,
    GOLDEN_REVIEW_FIELD,
    REVIEW_REASON_FIELD,
    SEED_METADATA_FIELDS,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export import (
    AUTO_APPROVED,
    HUMAN_VERIFIED,
    annotator_of,
    decided_by_id,
    flatten_export,
    id_key,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import (
    find_splits_dir,
    guard_read,
    nearest_project,
)

MANIFEST_FILE = "golden_review.json"
REVIEW_FILE = "review.jsonl"
REVIEWED_FILE = "golden_reviewed.jsonl"
MANIFEST_VERSION = 1
RUN_LOG_STEP = "golden-review"
REVIEW_REASON = "golden flag"  # the routing tag on each review row
MARK_FIELD = GOLDEN_REVIEW_FIELD  # on every row of a reviewed copy; the job never receives it
# Export statuses that say a person finished the row (with no reviewer named).
PERSON_STATUSES = ("completed",)
# What apply derives from the export, and check_reviewed derives again.
RESOLVED_KEYS = ("resolutions", "unresolved", "conflicting", "invalid_labels", "excluded")


class GoldenReviewError(DataPreparerOperatorError):
    """The reviewed golden rows break the rule that only a person changes a label."""


def _str(v):
    return None if v is None else str(v)


def read_lines(path):
    """``(raw lines, parsed rows)`` of a JSONL file, blank lines skipped.

    Raises:
        GoldenReviewError: A line that is not a JSON object.
    """
    raw, rows = [], []
    with open(path, encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                assert isinstance(row, dict)
            except (ValueError, AssertionError):
                raise GoldenReviewError(f"{path}:{n} is not a JSON object") from None
            raw.append(line if line.endswith("\n") else line + "\n")
            rows.append(row)
    return raw, rows


def read_eval(path, step, project_dir=None, id_field="id"):
    """``(raw lines, rows, events)`` of the golden rows.

    Under a frozen split the file must be its ``test.jsonl``, read as a logged test read
    (the split guard's entry is in ``events``).
    """
    sdir = find_splits_dir(path, project_dir or nearest_project(path))
    if sdir and os.path.abspath(path) != os.path.join(sdir, "test.jsonl"):
        raise GoldenReviewError(
            f"{path} is not the frozen test split {os.path.join(sdir, 'test.jsonl')}"
        )
    raw, rows = read_lines(path)
    events: List[Dict[str, Any]] = []
    if sdir:
        events = guard_read(
            path, step, {"test"}, splits_dir=sdir, rows=rows, id_field=id_field
        ).events
    return raw, rows, events


def _ids(rows, id_field, path):
    ids = []
    for n, r in enumerate(rows, 1):
        rid = _str(r.get(id_field))
        if rid is None:
            raise GoldenReviewError(
                f"{path}: row {n} has no {id_field!r}; every golden row needs its id"
            )
        ids.append(rid)
    if len(set(ids)) != len(ids):
        raise GoldenReviewError(f"{path}: duplicate ids")
    return ids


def read_routed(routed_rows, eval_ids):
    """``{id: checks}`` from the job's ``routed.jsonl``; every id must be an eval row, once."""
    out = {}
    wanted = set(eval_ids)
    for n, r in enumerate(routed_rows, 1):
        rid = _str(r.get("id"))
        checks = r.get("checks")
        if rid not in wanted or rid in out or not isinstance(checks, list) or not checks:
            raise GoldenReviewError(
                f"routed.jsonl row {n} ({rid!r}) is not a new --eval row with "
                "its checks; the job reviewed other rows than this --eval"
            )
        out[rid] = sorted(str(c) for c in checks)
    return out


def review_row(row, id_field, label_field, text_fields):
    """A blind review row: the id and the input fields only."""
    out = {id_field: row.get(id_field)}
    for f in text_fields:
        if (
            f == label_field
            or f.startswith("gt_")
            or f in SEED_METADATA_FIELDS
            or f in BLIND_FIELD_SET
        ):
            raise GoldenReviewError(
                f"--text-fields names {f!r}, which is gold, metadata or a "
                "model answer; a reviewer must not see it"
            )
        if f in row:
            out[f] = row[f]
    out[REVIEW_REASON_FIELD] = REVIEW_REASON
    return out


def observed_label_set(rows, label_field):
    """The eval rows' own string labels, sorted; ``None`` unless every label is a string."""
    got = [r.get(label_field) for r in rows if r.get(label_field) is not None]
    return sorted(set(got)) if got and all(isinstance(x, str) for x in got) else None


def _observed_or_refuse(rows, label_field):
    got = observed_label_set(rows, label_field)
    if got is None:
        raise GoldenReviewError(
            "the golden labels are not strings: pass --label-set with the spec's Label Set"
        )
    return got


def valid_label(label, label_set):
    """A person's label may become gold: a non-blank string in the label set."""
    return (
        isinstance(label, str) and bool(label.strip()) and (label_set is None or label in label_set)
    )


def _routed_rows(rows, ids, flagged, label_field):
    return {
        i: {"original": r.get(label_field), "checks": flagged[i]}
        for i, r in zip(ids, rows)
        if i in flagged
    }


def manifest_text(manifest):
    """The manifest as it is written: sorted keys, one-space indent, a trailing newline."""
    return json.dumps(manifest, indent=1, ensure_ascii=False, sort_keys=True) + "\n"


class RouteResult(OperatorResult):
    """What :func:`route` produces: the review rows and the manifest, for the caller to write."""

    manifest: Dict[str, Any] = Field(default_factory=dict)
    review_rows: List[Dict[str, Any]] = Field(default_factory=list)
    review_text: str = ""


def route(  # pylint: disable=too-many-arguments,too-many-locals,too-many-positional-arguments
    eval_path,
    routed_path,
    out_dir,
    text_fields,
    label_field="gt_label",
    id_field="id",
    label_set=None,
    project_dir=None,
):
    """The flagged golden rows as a blind review file, and the manifest.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import golden

        routed = golden.route("proj/splits/test.jsonl", "proj/review/routed.jsonl", "proj/golden",
                              text_fields=["text"], project_dir="proj")
        open("proj/golden/review.jsonl", "w").write(routed.review_text)
        open("proj/golden/manifest.json", "w").write(golden.manifest_text(routed.manifest))

    Args:
        eval_path: The corpus run's eval file (the golden rows).
        routed_path: The review job's ``routed.jsonl``.
        out_dir: Where the caller writes ``review.jsonl`` and the manifest (recorded in it).
        text_fields: The input fields a reviewer sees.
        label_field: The gold field.
        id_field: The id field.
        label_set: The labels a return may set; the eval rows' own when ``None``.
        project_dir: The project, for the split guard.

    Returns:
        A :class:`RouteResult`; the manifest names ``out_dir/review.jsonl`` as ``review``.

    Raises:
        GoldenReviewError: A bad eval or routed file, a text field a reviewer must not see,
            or non-string labels without a label set.
    """
    _, rows, events = read_eval(eval_path, RUN_LOG_STEP, project_dir, id_field)
    ids = _ids(rows, id_field, eval_path)
    _, routed_rows = read_lines(routed_path)
    flagged = read_routed(routed_rows, ids)
    by_id = dict(zip(ids, rows))
    review_path = os.path.join(out_dir, REVIEW_FILE)
    routed = [i for i in ids if i in flagged]
    review_rows = [review_row(by_id[i], id_field, label_field, text_fields) for i in routed]
    per_check = collections.Counter(c for checks in flagged.values() for c in checks)
    manifest = {
        "version": MANIFEST_VERSION,
        "eval": os.path.abspath(eval_path),
        "eval_sha256": _digest.sha256_file(eval_path),
        "id_field": id_field,
        "label_field": label_field,
        "routed_file": os.path.abspath(routed_path),
        "routed_sha256": _digest.sha256_file(routed_path),
        "golden_rows": len(ids),
        "routed": _routed_rows(rows, ids, flagged, label_field),
        "routed_per_check": dict(sorted(per_check.items())),
        "review": os.path.abspath(review_path),
        "label_set": list(label_set) if label_set else _observed_or_refuse(rows, label_field),
        "label_set_given": bool(label_set),
    }
    return RouteResult(
        manifest=manifest,
        review_rows=review_rows,
        review_text="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in review_rows),
        events=events,
    )


def read_manifest(path):
    """The manifest at ``path``, or a refusal when it is not one of this module's."""
    with open(path, encoding="utf-8") as fh:
        m = json.load(fh)
    if not isinstance(m, dict) or m.get("version") != MANIFEST_VERSION:
        raise GoldenReviewError(
            f"{path} is not a golden_review.py manifest (version {MANIFEST_VERSION})"
        )
    return m


def _same(a, b):
    return _digest.canonical_json(a) == _digest.canonical_json(b)


def resolve(m, rows, export_path):  # pylint: disable=too-many-locals
    """The resolutions of the person's export against the eval rows (``RESOLVED_KEYS``).

    Raises:
        GoldenReviewError: A model's label in the export, a returned row nobody finished,
            an export id that was not routed, or a relabel with no reviewer named.
    """
    id_field, label_field = m["id_field"], m["label_field"]
    ids = _ids(rows, id_field, m["eval"])
    _, export = read_lines(export_path)
    eid = m["export_id_field"]
    flat = [h for h in flatten_export(export, eid) if isinstance(h, dict)]
    if any(str(h.get("provenance") or "") == AUTO_APPROVED for h in flat):
        raise GoldenReviewError(
            f"{export_path} has a row marked {AUTO_APPROVED}: a model's label is not a person's resolution"
        )
    labels, annotators, _, _, conflicts, _, _ = decided_by_id(
        flat, eid, label_field, m["export_label_field"]
    )
    by_person = {
        id_key(h.get(eid))
        for h in flat
        if annotator_of(h) is not None
        or str(h.get("status") or "").strip().casefold() in PERSON_STATUSES
    }
    nobody = sorted(set(labels) - by_person)
    if nobody:
        raise GoldenReviewError(
            f"{len(nobody)} returned row(s) name no reviewer and no completed "
            f"status (e.g. {nobody[:3]}): only a person's return resolves a "
            "golden row"
        )
    routed = m["routed"]
    extra = sorted(set(labels) - set(routed))
    if extra:
        raise GoldenReviewError(
            f"{len(extra)} export id(s) were not routed (e.g. {extra[:3]}); "
            "only the flagged golden rows go to the person"
        )
    label_set = m.get("label_set", observed_label_set(rows, label_field))
    invalid = sorted(rid for rid, lab in labels.items() if not valid_label(lab, label_set))
    resolutions, excluded = {}, []
    for rid in ids:
        if rid in labels and rid not in invalid:
            changed = not _same(labels[rid], routed[rid]["original"])
            if changed and not annotators.get(rid):
                raise GoldenReviewError(
                    f"golden row {rid!r} is relabeled with no reviewer named; a relabel "
                    "(it leaves the golden rows) needs a named reviewer"
                )
            resolutions[rid] = {
                "label": labels[rid],
                "changed": changed,
                "annotators": annotators.get(rid) or [],
            }
            if changed:
                excluded.append(rid)
    return {
        "resolutions": resolutions,
        "unresolved": sorted(set(routed) - set(resolutions)),
        "conflicting": sorted(set(conflicts) & set(routed)),
        "invalid_labels": invalid,
        "excluded": excluded,
    }


class ApplyResult(OperatorResult):
    """What :func:`apply` produces: the reviewed copy's text and the manifest, for the caller to write.

    ``reviewed_path`` is where the copy goes (``golden_reviewed.jsonl`` beside the
    manifest); the manifest's ``reviewed_sha256`` is the text's, so the copy written as is
    passes :func:`check_reviewed`.
    """

    manifest: Dict[str, Any] = Field(default_factory=dict)
    reviewed_text: str = ""
    reviewed_path: str = ""


def apply(  # pylint: disable=too-many-locals
    manifest_path, export_path, export_id_field=None, export_label_field="label", project_dir=None
):
    """Fold the person's export into the reviewed copy of the golden rows.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import golden

        applied = golden.apply("proj/golden/manifest.json", "proj/golden/export.jsonl", project_dir="proj")
        open(applied.reviewed_path, "w").write(applied.reviewed_text)
        open("proj/golden/manifest.json", "w").write(golden.manifest_text(applied.manifest))

    Args:
        manifest_path: :func:`route`'s manifest, as written.
        export_path: The person's returned labels.
        export_id_field: The export's id field; the manifest's ``id_field`` when ``None``.
        export_label_field: The export's label column.
        project_dir: The project, for the split guard.

    Returns:
        An :class:`ApplyResult`. The caller writes ``reviewed_text`` to ``reviewed_path``
        and the manifest to ``manifest_path``, then runs :func:`check_reviewed`.

    Raises:
        GoldenReviewError: The eval file changed since route, the copy would overwrite the
            eval file, or :func:`resolve` refuses.
    """
    m = read_manifest(manifest_path)
    eval_path = m["eval"]
    if _digest.sha256_file(eval_path) != m["eval_sha256"]:
        raise GoldenReviewError(f"{eval_path} changed since route; run route again")
    raw, rows, events = read_eval(eval_path, RUN_LOG_STEP, project_dir, m["id_field"])
    ids = _ids(rows, m["id_field"], eval_path)
    m.update(
        export_id_field=export_id_field or m["id_field"], export_label_field=export_label_field
    )
    res = resolve(m, rows, export_path)
    routed, out_lines = m["routed"], []
    for _line, row, rid in zip(raw, rows, ids):
        if rid in res["excluded"]:
            continue  # a gold-edited row never certifies: it leaves the golden rows
        if rid in res["resolutions"]:
            new = dict(row, provenance=HUMAN_VERIFIED)
            new[MARK_FIELD] = {
                "original": routed[rid]["original"],
                "resolved_by": "person",
                "annotators": res["resolutions"][rid]["annotators"],
            }
            out_lines.append(json.dumps(new, ensure_ascii=False) + "\n")
        else:  # unrouted or unresolved: kept as it was, marked too, so any reviewed copy is
            # recognized as one (even with no confirmed row); the job never receives the mark
            out_lines.append(
                json.dumps(dict(row, **{MARK_FIELD: {"resolved_by": None}}), ensure_ascii=False)
                + "\n"
            )
    out_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), REVIEWED_FILE)
    if os.path.abspath(out_path) == os.path.abspath(eval_path):
        raise GoldenReviewError("the reviewed copy would overwrite --eval")
    text = "".join(out_lines)
    m.update(
        res,
        export=os.path.abspath(export_path),
        export_sha256=_digest.sha256_file(export_path),
        reviewed=os.path.abspath(out_path),
        reviewed_sha256=_digest.sha256_hex(text),
    )
    return ApplyResult(manifest=m, reviewed_text=text, reviewed_path=out_path, events=events)


class CheckResult(OperatorResult):
    """What :func:`check_reviewed` found in a reviewed copy that passed."""

    golden_rows: int = 0
    changed: int = 0
    confirmed: int = 0
    unresolved: int = 0


def check_reviewed(  # pylint: disable=too-many-locals,too-many-branches
    manifest_path, reviewed_path, step="certificate", project_dir=None
):
    """Refuse a reviewed copy unless only a person decided it.

    The inputs, the manifest and every row are checked: the eval file, the routed file and
    the export are the ones route and apply read; the manifest's routed rows, label set and
    resolutions are derived again; the copy holds every golden row but the relabeled ones,
    with a person's provenance and label on each resolved row and no other change.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import golden

        check = golden.check_reviewed(
            "proj/golden/manifest.json", "proj/golden/test.reviewed.jsonl", project_dir="proj"
        )
        check.changed, check.confirmed, check.unresolved

    Args:
        manifest_path: The manifest apply wrote.
        reviewed_path: The reviewed copy.
        step: The reading step (the test read is logged under it).
        project_dir: The project, for the split guard.

    Returns:
        A :class:`CheckResult`: ``golden_rows``, ``changed``, ``confirmed``, ``unresolved``.

    Raises:
        GoldenReviewError: Any change a person did not make. A refusal after the eval file
            was read carries that read's run-log entries in ``events``, for the caller to
            append (the read happened).
    """
    m = read_manifest(manifest_path)
    if "reviewed_sha256" not in m:
        raise GoldenReviewError(f"{manifest_path} has no person's resolutions yet; run apply")
    if _digest.sha256_file(m["eval"]) != m["eval_sha256"]:
        raise GoldenReviewError(f"{m['eval']} changed after the review; run route again")
    if _digest.sha256_file(reviewed_path) != m["reviewed_sha256"]:
        raise GoldenReviewError(
            f"{reviewed_path} is not the file golden_review.py apply wrote (hash differs)"
        )
    if (
        not os.path.isfile(m.get("export") or "")
        or _digest.sha256_file(m["export"]) != m["export_sha256"]
    ):
        raise GoldenReviewError(
            f"the person's export {m.get('export')!r} is missing or changed since apply; run apply again"
        )
    id_field, label_field = m["id_field"], m["label_field"]
    _, src, events = read_eval(m["eval"], step, project_dir, id_field)
    try:
        return _check_read_rows(m, src, reviewed_path, id_field, label_field, events)
    except (OSError, KeyError, ValueError) as e:
        e.events = events  # pylint: disable=attribute-defined-outside-init
        raise


def _check_read_rows(  # pylint: disable=too-many-locals,too-many-arguments,too-many-positional-arguments
    m, src, reviewed_path, id_field, label_field, events
):
    """:func:`check_reviewed`'s checks of the rows it read from the eval file."""
    src_ids = _ids(src, id_field, m["eval"])
    if (
        not os.path.isfile(m.get("routed_file") or "")
        or _digest.sha256_file(m["routed_file"]) != m["routed_sha256"]
    ):
        raise GoldenReviewError("the review job's routed.jsonl is missing or changed since route")
    routed = _routed_rows(
        src, src_ids, read_routed(read_lines(m["routed_file"])[1], src_ids), label_field
    )
    if not _same(routed, m["routed"]) or (
        not m.get("label_set_given") and m.get("label_set") != observed_label_set(src, label_field)
    ):
        raise GoldenReviewError(
            "the manifest's routed rows or label set are not the ones route wrote (was it edited?); run route again"
        )
    again = resolve(m, src, m["export"])
    if any(not _same(again[k], m.get(k)) for k in RESOLVED_KEYS):
        raise GoldenReviewError(
            "the manifest's resolutions are not the ones the person's export gives (was it edited?); run apply again"
        )
    _, got = read_lines(reviewed_path)
    got_ids = _ids(got, id_field, reviewed_path)
    res = m.get("resolutions") or {}
    excluded = set(m.get("excluded") or [])  # resolve() derived it again: only relabeled rows
    want_ids = [rid for rid in src_ids if rid not in excluded]
    if want_ids != got_ids:
        missing = sorted(set(want_ids) - set(got_ids))
        raise GoldenReviewError(
            f"the reviewed golden rows are not the --eval rows ({len(missing)} missing, e.g. "
            f"{missing[:3]}): a golden row is never dropped or added; only a row a person "
            "relabeled leaves"
        )
    by_src = dict(zip(src_ids, src))
    changed, confirmed = len(excluded), 0
    for b, rid in zip(got, got_ids):
        a = by_src[rid]
        # The job uploads each golden row's id, gold and provenance, and skips a row whose
        # provenance says a model labeled it: a provenance change is a drop.
        want_prov = HUMAN_VERIFIED if rid in res else a.get("provenance")
        if b.get("provenance") != want_prov:
            raise GoldenReviewError(
                f"golden row {rid!r} has provenance {b.get('provenance')!r}, not "
                f"{want_prov!r}: only a person's return changes a golden row"
            )
        if not _same(a.get(label_field), b.get(label_field)):
            raise GoldenReviewError(
                f"golden row {rid!r} has a label no person's return set; a row a person "
                "relabels leaves the golden rows, and no other label changes"
            )
        if rid in res:
            confirmed += 1
    return CheckResult(
        golden_rows=len(src_ids),
        changed=changed,
        confirmed=confirmed,
        unresolved=len(m.get("unresolved") or []),
        events=events,
    )


def has_mark(rows):
    """True when any row carries the reviewed-copy record."""
    return any(isinstance(r, dict) and MARK_FIELD in r for r in rows)


def route_summary(manifest, manifest_path, run_log_entry: Optional[str] = None):
    """What the route command reports."""
    return {
        "golden_rows": manifest["golden_rows"],
        "routed": len(manifest["routed"]),
        "routed_per_check": manifest["routed_per_check"],
        "review": manifest["review"],
        "manifest": manifest_path,
        "run_log_entry": run_log_entry,
    }


def apply_summary(manifest, run_log_entry: Optional[str] = None):
    """What the apply command reports."""
    res = manifest["resolutions"]
    return {
        "golden_rows": manifest["golden_rows"],
        "routed": len(manifest["routed"]),
        "changed": sum(1 for x in res.values() if x["changed"]),
        "confirmed": sum(1 for x in res.values() if not x["changed"]),
        **{k: len(manifest[k]) for k in ("unresolved", "conflicting", "invalid_labels")},
        "excluded_from_golden_rows": len(manifest["excluded"]),
        "reviewed": manifest["reviewed"],
        "run_log_entry": run_log_entry,
    }
