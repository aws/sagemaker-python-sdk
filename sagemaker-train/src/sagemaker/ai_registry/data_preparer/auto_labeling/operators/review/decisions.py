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
"""The user's decisions on a label review's contradiction list, applied to the next spec and seed.

A label review (:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.review.request`)
flags items: groups of seed rows whose labels contradict each other or the guidelines. The
user decides each item in ``decisions.csv``:

* ``guidelines_correct``: relabel the item's conflicting members (its ``proposed_changes``);
* ``label_correct``: keep the labels; add ``sop_wording`` (required) to the spec;
* ``neither``: ``if_neither`` is the right label for every member (wording goes in
  ``sop_wording``).

Per-row overrides (``r12=a;r13=keep``) win; two items that set one row differently cancel.
:func:`apply_decisions` is the pure core: new seed rows, the new spec text, a changelog and
the ledger of decisions to record. Around it: the decisions the run log already records
(:func:`logged_decisions`), the merge of those with the CSV (:func:`merge_decisions`), the
test rows a decision would touch (:func:`touched_test_rows`), the label set a decision may
set (:func:`default_label_set`) and the run-log entries that record the applied decisions
(:func:`decision_events`). Writing the files, the split relabel and its undo stay with the
caller.
"""

from __future__ import absolute_import

import csv
import datetime
import io
import re
from typing import Any, Dict, List, Optional

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event

DECISIONS = ("guidelines_correct", "label_correct", "neither")
DECISION_COLS = ("decision", "if_neither", "sop_wording", "overrides")

CLARIFY_HEADER = "## Clarifications from label review"
RUN_LOG_STEP = "label-review"  # every label-review run-log entry
FORMULA_LEAD = ("=", "+", "-", "@", "\t", "\r")  # a spreadsheet runs a cell starting with one
_NUMBER = re.compile(r"[+-]?\d+(\.\d*)?")
# The fields of a recorded decision that make it the same decision.
DECISION_KEYS = ("decision", "if_neither", "sop_wording", "overrides", "item_sha256")
_ITER_SPEC = re.compile(r"^(.*_iter_)(\d+)(\.[^.]+)$")


class OverrideOutsideItem(DataPreparerOperatorError):
    """An override names a row that is not a member of its item."""


def csv_cell(v):
    """A ``decisions.csv`` cell a spreadsheet shows as text.

    A leading ``'`` goes before a formula start; a plain number such as ``-1`` stays as it is.
    """
    s = "" if v is None else str(v)
    return "'" + s if s[:1] in FORMULA_LEAD and not _NUMBER.fullmatch(s) else s


def _plain(v):
    """The cell as the user meant it: the ``'`` that :func:`csv_cell` adds is dropped."""
    v = v or ""
    return v[1:] if v[:1] == "'" and v[1:2] in FORMULA_LEAD else v


def canon(v):
    """A classification label, whitespace collapsed (a one-item list is its item)."""
    if isinstance(v, (list, tuple)):
        v = v[0] if len(v) == 1 else "|".join(map(str, v))
    return " ".join(str(v).split()) or None if v is not None else None


def item_sha(it):
    """What a decision was made on: the item's members and proposed changes.

    A logged decision applies again only to an item with the same sha.
    """
    key = {
        "members": sorted(str(m) for m in it.get("member_ids") or []),
        "changes": {str(k): v for k, v in (it.get("proposed_changes") or {}).items()},
    }
    return _digest.sha256_json(key)


def review_inputs(spec_sha256, seed_sha256, split_sha256=None):
    """What a review read, and what applying its decisions needs unchanged.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import review_inputs

        inputs = review_inputs(spec_sha256=sha_spec, seed_sha256=sha_seed, split_sha256=sha_split)

    Args:
        spec_sha256: The spec file's SHA-256.
        seed_sha256: The seed file's SHA-256.
        split_sha256: The frozen split state's SHA-256, ``None`` without a frozen split.

    Returns:
        ``{"spec", "seed", "split"}``.
    """
    return {"spec": spec_sha256, "seed": seed_sha256, "split": split_sha256}


def logged_decisions(entries, items=None):
    """The decisions the project's run log records, by item id; the latest wins.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import logged_decisions

        known = logged_decisions(run_log_entries, items)  # {item_id: decision}, the latest wins

    Args:
        entries: The project's run-log entries, oldest first (only ``decision`` entries of
            step ``label-review`` with an ``item_id`` count).
        items: With the current items, only the decisions whose item still has the decided
            members and changes.

    Returns:
        A dict keyed by item id; each value holds ``item_id``, ``check``, ``if_neither``,
        ``sop_wording``, ``decided_by``, ``item_sha256``, ``decision`` and ``overrides``.
    """
    keys = ("item_id", "check", "if_neither", "sop_wording", "decided_by", "item_sha256")
    led = {
        e["item_id"]: dict(
            {k: e.get(k) for k in keys},
            decision=e.get("status"),
            overrides=e.get("overrides") or {},
        )
        for e in entries
        if e.get("kind") == "decision" and e.get("step") == RUN_LOG_STEP and e.get("item_id")
    }
    if items is not None:
        now = {it["item_id"]: item_sha(it) for it in items}
        led = {k: v for k, v in led.items() if v.get("item_sha256") == now.get(k)}
    return led


def parse_overrides(s):
    """``{row id: value}`` from an overrides cell (``r12=a;r13=keep``) or a dict."""
    if isinstance(s, dict):
        return {str(k): v for k, v in s.items()}
    pairs = (part.split("=", 1) for part in re.split(r"[;\n]", s or "") if "=" in part)
    return {k.strip(): v.strip() for k, v in pairs if k.strip()}


def decisions_from_csv(text):
    """The decisions a ``decisions.csv`` holds, by item id.

    Args:
        text: The CSV text.

    Returns:
        ``{item_id: {decision, if_neither, sop_wording, overrides, item_sha256}}``; a blank
        decision is ``None`` (an explicit "no decision").
    """
    out = {}
    for r in csv.DictReader(io.StringIO(text, newline="")):
        r = {k: _plain(v).strip() for k, v in r.items() if k}
        out[r.get("item_id") or ""] = {
            "decision": (r.get("decision") or "").lower().replace(" ", "_") or None,
            "if_neither": r.get("if_neither") or None,
            "sop_wording": r.get("sop_wording") or None,
            "overrides": parse_overrides(r.get("overrides") or ""),
            "item_sha256": r.get("item_sha256") or None,
        }
    out.pop("", None)
    return out


def read_decisions_csv(path):
    """:func:`decisions_from_csv` over the file at ``path``."""
    with open(path, newline="", encoding="utf-8") as f:
        return decisions_from_csv(f.read())


def spec_iteration(spec_path):
    """The iteration N of a ``labeling_spec_iter_N.md`` spec, or 0 for any other name."""
    import os

    m = _ITER_SPEC.match(os.path.basename(spec_path or ""))
    return int(m.group(2)) if m else 0


def next_spec_name(spec_path):
    """``labeling_spec_iter_{N+1}.md`` (keeping a prefix) for iteration N, else iteration 1."""
    import os

    m = _ITER_SPEC.match(os.path.basename(spec_path))
    return f"{m.group(1)}{int(m.group(2)) + 1}{m.group(3)}" if m else "labeling_spec_iter_1.md"


def merge_decisions(known, csv_decisions, items, source="decisions.csv"):
    """The decisions to apply: the run log's for unchanged items, overridden by the CSV's.

    A blank CSV row is an explicit "no decision" and clears a logged one.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import merge_decisions

        decided = merge_decisions(known, csv_decisions, items)  # the CSV overrides the run log's

    Args:
        known: :func:`logged_decisions` for the current items.
        csv_decisions: :func:`decisions_from_csv`, or ``None`` without a CSV.
        items: The current contradiction items.
        source: The CSV's name, for the refusal.

    Returns:
        ``{item_id: decision}`` with a decision set.

    Raises:
        DataPreparerOperatorError: A CSV decision was made on another version of its item.
    """
    decisions = dict(known)
    shas = {it["item_id"]: item_sha(it) for it in items}
    for k, v in (csv_decisions or {}).items():
        if v.get("decision") and v.get("item_sha256") != shas.get(k):
            raise DataPreparerOperatorError(
                f"{source}: the decision for {k!r} was made on another version of the item; rerun "
                "review_labels.py and decide again. Nothing was written."
            )
        decisions[k] = v
    return {k: v for k, v in decisions.items() if v.get("decision")}


def touched_test_rows(items, decisions, test_ids):
    """The frozen test rows the decided items would change, sorted; a test row never changes here.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import touched_test_rows

        if touched_test_rows(items, decided, test_ids(project_dir="proj")):
            raise SystemExit("a decision would change a frozen test row")
    """
    return sorted(
        {str(m) for it in items if it["item_id"] in decisions for m in it["member_ids"]}
        & set(test_ids)
    )


def default_label_set(spec_text, rows, label_field):
    """The label set a decision may set: the spec's Label Set, else the rows' own labels, sorted.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import default_label_set

        label_set = default_label_set(spec_text, seed_rows, "gt_label")
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.taxonomy import (
        parse_label_set,
    )

    return [name for name, _ in parse_label_set(spec_text)] or sorted(
        {str(r[label_field]) for r in rows if r.get(label_field) not in (None, "")}
    )


def apply_decisions(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    items,
    decisions,
    seed_rows,
    spec_text,
    *,
    id_field="id",
    label_field="gt_label",
    label_set=(),
    version=1,
    decided_by=None,
    today=None,
):
    """Apply the decisions: new seed rows, new spec text, a changelog and the ledger.

    An item with any invalid part (a label outside ``label_set``, a missing sentence) is
    listed in the changelog's ``invalid_decisions`` and neither applied nor recorded. Two
    items that set one row differently cancel: the row is listed under ``needs_a_decision``.
    A sentence the spec already carries (marked with its item id) is not repeated.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import decisions

        applied = decisions.apply_decisions(items, decided, seed_rows, spec_text,
                                            label_set=["TP", "FP"], version=2, decided_by="alice")
        new_seed_rows, new_spec_text, changelog, ledger = applied

    Args:
        items: The contradiction items (``item_id``, ``check``, ``member_ids``,
            ``proposed_changes``, ``recheck_ids``).
        decisions: ``{item_id: {decision, if_neither, sop_wording, overrides}}``.
        seed_rows: The seed rows.
        spec_text: The spec.
        id_field: The rows' id field.
        label_field: The rows' label field.
        label_set: The labels a decision may set.
        version: The iteration the outputs belong to.
        decided_by: Who decided, recorded on each ledger entry.
        today: The changelog's date (ISO); today's when ``None``.

    Returns:
        ``(new_seed_rows, new_spec_text, changelog, ledger)``.

    Raises:
        OverrideOutsideItem: An applied decision's override names a row outside the item's
            members (every such id is named); nothing is applied, so a typo never relabels
            another row.
    """
    labels = {canon(x) for x in label_set} - {None}
    by_id = {str(r.get(id_field)): r for r in seed_rows}
    sets: Dict[str, Dict[str, Any]] = {}  # row id -> {item_id: new label}
    sentences, rechecks, ledger, invalid = [], [], [], []
    decided = [
        (it, decisions[it["item_id"]])
        for it in items
        if (decisions.get(it["item_id"]) or {}).get("decision")
    ]
    outside = [
        f"{it['item_id']}: {r}"
        for it, d in decided
        for r, v in parse_overrides(d.get("overrides")).items()
        if r not in {str(m) for m in it["member_ids"]} and str(v or "").strip()
    ]
    if outside:
        raise OverrideOutsideItem(
            "override ids are not members of their item (nothing was applied): "
            + "; ".join(outside)
        )
    for it, d in decided:
        dec, iid, bad = d["decision"], it["item_id"], []
        members = [str(m) for m in it["member_ids"]]
        changes, said, again = {}, [], []
        if dec not in DECISIONS:
            bad.append(f"{iid}: decision {dec!r} not in {DECISIONS}")
        elif dec == "label_correct" and not d.get("sop_wording"):
            bad.append(f"{iid}: label_correct needs the sentence for the spec in sop_wording")
        elif dec == "guidelines_correct":
            # Only the item's own rows: a proposed change outside member_ids is ignored.
            for r, v in (it.get("proposed_changes") or {}).items():
                if str(r) in members:
                    changes[str(r)] = canon(v)
                    if canon(v) not in labels:
                        bad.append(f"{iid}: proposed label {v!r} for {r} is not in the label set")
        elif dec == "label_correct":
            said, again = [d["sop_wording"]], [str(r) for r in it.get("recheck_ids") or []]
        else:  # neither
            lab = d.get("if_neither")
            c = canon(re.sub(r"(?i)^\s*label\s*[:=]\s*", "", lab)) if lab else None
            if lab and c not in labels:
                bad.append(f"{iid}: if_neither {lab!r} is not a label of the label set")
            changes = {
                r: c
                for r in members
                if lab and r in by_id and canon(by_id[r].get(label_field)) != c
            }
            said = [d["sop_wording"]] if d.get("sop_wording") else []
        ov = {}
        for r, v in parse_overrides(d.get("overrides") or {}).items():
            v = v.strip() if isinstance(v, str) else v
            if v in (None, ""):
                continue
            if isinstance(v, str) and v.lower() == "keep":
                ov[r] = "keep"
                changes.pop(r, None)
            elif canon(v) in labels:
                ov[r] = changes[r] = canon(v)
            else:
                bad.append(f"{iid}: override {r}={v!r} is not keep or a label of the label set")
        if dec == "neither" and not (
            str(d.get("if_neither") or "").strip() or str(d.get("sop_wording") or "").strip() or ov
        ):
            bad.append(
                f"{iid}: neither needs the right label in if_neither, a sentence in sop_wording "
                "or an override"
            )
        if bad:  # nothing of an invalid item is applied or recorded
            invalid.extend(bad)
            continue
        ledger.append(
            {
                "item_id": iid,
                "check": it["check"],
                "decision": dec,
                "if_neither": d.get("if_neither"),
                "sop_wording": d.get("sop_wording"),
                "overrides": ov,
                "decided_by": decided_by,
                "item_sha256": item_sha(it),
            }
        )
        sentences.extend((iid, s) for s in said)
        rechecks.extend((iid, r) for r in again)
        for r, v in changes.items():
            sets.setdefault(r, {})[iid] = v
    relabel = {r: next(iter(m.values())) for r, m in sets.items() if len(set(m.values())) == 1}
    needs = [{"id": r, "items": m} for r, m in sets.items() if r not in relabel]
    new_rows, changed = [], []
    for row in seed_rows:
        rid = str(row.get(id_field))
        if rid in relabel and relabel[rid] != canon(row.get(label_field)):
            changed.append(
                {
                    "id": rid,
                    "from": row.get(label_field),
                    "to": relabel[rid],
                    "items": sorted(sets[rid]),
                }
            )
            row = dict(row, **{label_field: relabel[rid]})
        new_rows.append(row)
    new_spec = spec_text
    # A decision the run log already records is applied again on the next review; its
    # sentence is in the spec already (marked with the item id), so it is not repeated.
    sentences = [(iid, s) for iid, s in sentences if f"<!-- {iid} -->" not in spec_text]
    if sentences:
        block = "\n".join(f"- {s}  <!-- {iid} -->" for iid, s in sentences)
        if CLARIFY_HEADER in new_spec:
            new_spec = new_spec.rstrip() + "\n" + block + "\n"
        else:
            new_spec = (
                new_spec.rstrip() + f"\n\n{CLARIFY_HEADER} (iteration {version})\n\n{block}\n"
            )
    changelog = {
        "iteration": version,
        "date": today or datetime.date.today().isoformat(),
        "decisions_applied": len(ledger),
        "labels_changed": changed,
        "sentences_added": sentences,
        "recheck": [{"item_id": a, "id": b} for a, b in rechecks],
        "needs_a_decision": needs,
        "invalid_decisions": invalid,
    }
    return new_rows, new_spec, changelog, ledger


def decision_events(ledger, known, iteration):
    """The run-log ``decision`` entries for the applied decisions that are new or changed.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import decision_events

        run_log.extend(decision_events(ledger, known, iteration=2))

    Args:
        ledger: :func:`apply_decisions`'s ledger.
        known: :func:`logged_decisions` before this apply.
        iteration: The spec's iteration (the entries' ``iteration``).

    Returns:
        The entries (status is the decision; empty fields left out), in ledger order.
    """
    events: List[Dict[str, Any]] = []
    for r in ledger:
        prev = known.get(r["item_id"])
        if prev and all(prev.get(k) == r.get(k) for k in DECISION_KEYS):
            continue
        fields = {
            k: r[k]
            for k in (
                "item_id",
                "check",
                "if_neither",
                "sop_wording",
                "overrides",
                "decided_by",
                "item_sha256",
            )
            if r.get(k) not in (None, {}, "")
        }
        events.append(run_log_event("decision", RUN_LOG_STEP, iteration, r["decision"], **fields))
    return events


def dry_run_summary(
    version, ledger, changelog, would_write, *, split_files=(), split_copy: Optional[str] = None
):
    """What an apply would change, as the dry run reports it.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import dry_run_summary

        print(json.dumps(dry_run_summary(2, ledger, changelog, would_write=["seed.jsonl", "spec.md"]), indent=2))
    """
    import collections

    return {
        "dry_run": True,
        "iteration": version,
        "decisions": dict(collections.Counter(r["decision"] for r in ledger)),
        "labels_changed": changelog["labels_changed"],
        "sentences_added": len(changelog["sentences_added"]),
        "needs_a_decision": len(changelog["needs_a_decision"]),
        "invalid_decisions": changelog["invalid_decisions"],
        "would_write": list(would_write),
        "split_files_overwritten": list(split_files),
        "split_copy": split_copy,
    }
