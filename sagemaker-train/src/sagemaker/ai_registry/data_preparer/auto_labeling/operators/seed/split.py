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
"""Freeze a labeled seed's dev / val / test split once, keep it frozen, and guard every read.

A refinement loop splits the labeled seed once and never again. Each split has one job:

* ``dev``: spec refinement, candidate mining, demonstrations.
* ``val``: candidate selection, the held-out gate, a calibration fit.
* ``test``: the corpus run's cutoff and certificate, and the reported metrics. Never
  a demonstration, never an example for a model, never used to choose.

:func:`freeze` records each split's row ids and file hash in a state document
(``split_state.json`` beside the split files). It needs the label field and a text on
every row and no empty split, and refuses splits that share an id or a near-duplicate
text (``move_near_duplicates`` moves the val or test member of each such pair into dev
first). :func:`rehash` accepts edited dev and val files (a person's label fixes, new
dev rows) and rewrites the state; it refuses a changed ``test.jsonl``, a frozen row in
another split, a new row that near-duplicates a test row, and an empty split. Test
rows are immutable once frozen.

Every step that reads a split file, or a file built from one, calls :func:`guard_read`:
the split files must match the state, the splits must share no id or near-duplicate
text, and the file must hold only rows of the splits the step may read. A test row may
be read only by the corpus run's certificate, the reported metrics and the golden-row
review (:data:`TEST_STEPS`), and every test read is a run-log event.

Nothing here writes a file. ``freeze``, ``rehash`` and ``relabel`` return the rows and
the state document for the caller to write, with the hashes computed over exactly the
bytes :func:`rows_bytes` serializes, so a caller that writes those bytes gets a state
that verifies. Reads (split files, the state) take the directory as a parameter.
"""

from __future__ import absolute_import

import datetime
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import report_client_read

STATE_FILE = "split_state.json"
STATE_VERSION = 1
SPLITS = ("dev", "val", "test")
# The steps that may read a test row. Anything else that reads one stops the run.
TEST_STEPS = frozenset({"certificate", "report", "golden-review"})
# Two rows in different splits whose texts' word sets overlap at least this much
# (Jaccard) are near-duplicates. Texts with fewer words are compared exactly.
NEAR_DUP_JACCARD = 0.9
NEAR_DUP_MIN_TOKENS = 5
NEAR_COPY_HINT = "a row and its near-copy must share a split"

_WORD = re.compile(r"\w+", re.UNICODE)
_VERIFIED: Dict[Tuple[str, Tuple[str, ...]], bool] = {}

Rows = List[Dict[str, Any]]
RowsBySplit = Dict[str, Rows]


class SplitGuardError(DataPreparerOperatorError):
    """A split guard refusal.

    A split was re-split, changed outside this module, overlaps another split, or a
    step read rows it may not read.
    """


class GuardRead(OperatorResult):
    """What :func:`guard_read` found.

    ``touched`` maps each split the file holds rows of to the row count, with
    ``outside`` for ids in no split; ``None`` when the project has no frozen split
    (``guarded`` is then ``False``). ``events`` carries the test-read entry when the
    step read test rows.
    """

    guarded: bool
    touched: Optional[Dict[str, int]] = None


class FreezeResult(OperatorResult):
    """What :func:`freeze` produced, for the caller to write.

    ``state`` is the state document to write to ``state_path``. ``rows`` holds the
    rows of every split, in the order and serialization (:func:`rows_bytes`) the state's
    hashes assume; when ``moved_near_duplicates`` is non-empty the caller must write the
    three split files from it before the state (nothing moved: the files on disk are
    already what was hashed). ``events`` carries the ``frozen`` artifact entry.
    """

    state: Dict[str, Any]
    state_path: str
    rows: RowsBySplit
    row_counts: Dict[str, int]
    moved_near_duplicates: List[Dict[str, str]]


class RehashResult(OperatorResult):
    """What :func:`rehash` produced.

    The new ``state`` to write to ``state_path`` and a ``summary`` (row counts, reason,
    and the added / removed / gold-changed counts per split), with the ``rehashed``
    artifact entry in ``events``.
    """

    state: Dict[str, Any]
    state_path: str
    summary: Dict[str, Any]


class RelabelResult(RehashResult):
    """What :func:`relabel` produced.

    The relabeled ``rows`` of ``dev`` and ``val`` to write (as :func:`rows_bytes`)
    before the state, and everything :class:`RehashResult` carries.
    """

    rows: RowsBySplit


# --- text keys ---------------------------------------------------------------------


def row_text(row: Dict[str, Any], text_fields: Iterable[str]) -> str:
    """The row's input text: the text fields joined, lower-cased, whitespace collapsed."""
    parts = []
    for f in text_fields:
        v = row.get(f)
        if v is None:
            continue
        parts.append(v if isinstance(v, str) else _digest.canonical_json(v))
    return " ".join(" ".join(parts).lower().split())


def _words(text: str) -> frozenset:
    return frozenset(_WORD.findall(text))


def _jaccard(a: frozenset, b: frozenset) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 1.0


def _prefix_len(n: int, threshold: float = NEAR_DUP_JACCARD) -> int:
    """Prefix filtering's prefix length for a set of ``n`` words.

    Two sets with Jaccard at least ``threshold`` share a word within both prefixes when
    each set's words are taken in one global order. The small epsilon keeps float noise
    (0.9 * 10) from shortening it.
    """
    return n - math.ceil(threshold * n - 1e-9) + 1


def near_duplicate_pairs(
    texts_by_split: Dict[str, Dict[str, str]],
) -> List[Tuple[str, str, str, str]]:
    """Pairs of rows in different splits with the same or a near-duplicate text.

    Two texts are near-duplicates when both have at least ``NEAR_DUP_MIN_TOKENS`` words
    and their word-set Jaccard is at least ``NEAR_DUP_JACCARD``. Exact: prefix filtering
    never misses a pair at the threshold, and each candidate is confirmed in full.

    Args:
        texts_by_split: split -> {id: text}.

    Returns:
        Sorted ``(split_a, id_a, split_b, id_b)`` tuples with ``split_a`` before
        ``split_b`` in :data:`SPLITS` order.
    """
    order = {s: i for i, s in enumerate(SPLITS)}
    rows = [
        (s, rid, t)
        for s in sorted(texts_by_split, key=lambda x: order.get(x, 99))
        for rid, t in texts_by_split[s].items()
    ]
    pairs = set()
    by_text: Dict[str, list] = {}
    for s, rid, t in rows:
        by_text.setdefault(t, []).append((s, rid))
    for members in by_text.values():
        for i, (sa, ia) in enumerate(members):
            for sb, ib in members[i + 1 :]:
                if sa != sb:
                    pairs.add(_ordered(sa, ia, sb, ib, order))
    words = [(s, rid, _words(t)) for s, rid, t in rows]
    df: Dict[str, int] = {}
    for _s, _r, w in words:
        for tok in w:
            df[tok] = df.get(tok, 0) + 1
    prefixes = {}
    index: Dict[str, list] = {}
    for k, (_s, _r, w) in enumerate(words):
        if len(w) < NEAR_DUP_MIN_TOKENS:
            continue
        ranked = sorted(w, key=lambda tok: (df[tok], tok))
        prefixes[k] = ranked[: _prefix_len(len(w))]
        for tok in prefixes[k]:
            index.setdefault(tok, []).append(k)
    for k, (sa, ia, wa) in enumerate(words):
        if k not in prefixes:
            continue
        cands = set()
        for tok in prefixes[k]:
            cands.update(j for j in index[tok] if j > k)
        for j in cands:
            sb, ib, wb = words[j]
            if sa == sb or len(wb) < NEAR_DUP_MIN_TOKENS:
                continue
            small, big = sorted((len(wa), len(wb)))
            if small < NEAR_DUP_JACCARD * big - 1e-9:
                continue
            if _jaccard(wa, wb) >= NEAR_DUP_JACCARD:
                pairs.add(_ordered(sa, ia, sb, ib, order))
    return sorted(pairs)


def _ordered(sa, ia, sb, ib, order):
    if (order.get(sa, 99), ia) <= (order.get(sb, 99), ib):
        return (sa, ia, sb, ib)
    return (sb, ib, sa, ia)


# --- files (read only) and bytes ---------------------------------------------------------


def rows_bytes(rows: Rows) -> bytes:
    """The bytes a split file holds for ``rows``.

    One compact JSON object per line, unicode preserved. The hashes in a state document
    are over these bytes, so a caller writing a split file writes exactly this.
    """
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def sha256_rows(rows: Rows) -> str:
    """The SHA-256 of :func:`rows_bytes`."""
    return _digest.sha256_hex(rows_bytes(rows))


def read_rows(path: str) -> Rows:
    """The non-blank JSON Lines records of a split file."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gold_hash(value: Any) -> str:
    """A short hash of a gold value, as the state records it per row."""
    return _digest.sha256_json(value)[:16]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def state_path(splits_dir: str) -> str:
    """Where the state document lives for ``splits_dir``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import state_path

        state_path("proj/splits")  # 'proj/splits/split_state.json'
    """
    return os.path.join(splits_dir, STATE_FILE)


def load_state(splits_dir: str) -> Optional[Dict[str, Any]]:
    """The state document of ``splits_dir``, or ``None`` when the split is not frozen.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import load_state

        state = load_state("proj/splits")
        frozen = state is not None

    Raises:
        SplitGuardError: The file is not a state of :data:`STATE_VERSION`.
    """
    p = state_path(splits_dir)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as fh:
        st = json.load(fh)
    if st.get("version") != STATE_VERSION:
        raise SplitGuardError(f"{p} is not a split state file (version {STATE_VERSION})")
    return st


def split_rows(splits_dir: str) -> RowsBySplit:
    """The rows of every split file in ``splits_dir``.

    Raises:
        SplitGuardError: A split file is missing.
    """
    out = {}
    for s in SPLITS:
        p = os.path.join(splits_dir, f"{s}.jsonl")
        if not os.path.isfile(p):
            raise SplitGuardError(
                f"{p} is missing: Step 1 writes dev, val and test there (after an API "
                "Split, download each split DataSet to that path first)"
            )
        out[s] = read_rows(p)
    return out


def _ids(rows: Rows, id_field: str, where: str) -> List[str]:
    ids = []
    for r in rows:
        v = r.get(id_field) if isinstance(r, dict) else None
        if v is None:
            raise SplitGuardError(f"a row of {where} has no {id_field!r}: the split is keyed by id")
        ids.append(str(v))
    return ids


def _check_disjoint(rows_by_split, id_field, text_fields, hint=NEAR_COPY_HINT):
    """Raise when two splits share an id or a near-duplicate text; ``hint`` says what to do."""
    owner = {}
    for s in SPLITS:
        for rid in _ids(rows_by_split[s], id_field, f"{s}.jsonl"):
            if rid in owner:
                raise SplitGuardError(f"id {rid!r} is in both {owner[rid]} and {s}")
            owner[rid] = s
    texts = {
        s: {str(r[id_field]): row_text(r, text_fields) for r in rows_by_split[s]} for s in SPLITS
    }
    pairs = near_duplicate_pairs(texts)
    if pairs:
        sa, ia, sb, ib = pairs[0]
        raise SplitGuardError(
            f"{len(pairs)} near-duplicate pair(s) across splits (e.g. {sa} {ia!r} and {sb} "
            f"{ib!r}); {hint}"
        )
    return owner


def _check_rows(rows, label_field, text_fields):
    """Every split has rows, and every row a label and a non-empty text field."""
    for s in SPLITS:
        if not rows[s]:
            raise SplitGuardError(f"{s}.jsonl is empty; every split needs rows")
        bad = [
            r.get("id")
            for r in rows[s]
            if r.get(label_field) in (None, "") or not row_text(r, text_fields).strip()
        ]
        if bad:
            raise SplitGuardError(
                f"{len(bad)} {s} row(s) (e.g. {bad[:3]}) have no {label_field!r} or no text in "
                f"{', '.join(text_fields)}"
            )


def _split_entry(sha256, s, rows, id_field, label_field):
    return {
        "file": f"{s}.jsonl",
        "sha256": sha256,
        "ids": [str(r[id_field]) for r in rows],
        "gold": {str(r[id_field]): gold_hash(r.get(label_field)) for r in rows},
    }


def _split_event(splits_dir, status, **counts):
    """A freeze or rehash as an ``artifact`` entry of the project's run log, naming the state."""
    return run_log_event(
        "artifact", "split", 0, status, file=os.path.abspath(state_path(splits_dir)), **counts
    )


# --- freeze / verify -----------------------------------------------------------------


def freeze(
    splits_dir: str,
    text_fields: Sequence[str],
    *,
    id_field: str = "id",
    label_field: str = "gt_label",
    move_near_duplicates: bool = False,
) -> FreezeResult:
    """Record a seed split so no step can change or misuse it.

    Use it once, right after the seed is split into ``dev.jsonl``, ``val.jsonl`` and
    ``test.jsonl`` in ``splits_dir`` and before anything reads them. Never call it a
    second time: a frozen split is never re-split (``rehash`` accepts dev and val
    edits). Every later read goes through :func:`guard_read`.

    Args:
        splits_dir: The folder holding the three split files.
        text_fields: The input fields the spec labels from; the near-duplicate check
            compares their text.
        id_field: Must be ``id``: the labeling job keys rows on it.
        label_field: The seed's gold field; every row needs a value.
        move_near_duplicates: Move the val or test member of each cross-split
            near-duplicate pair (both of a val-test pair) into dev before freezing,
            repeated until none is left.

    Returns:
        A :class:`FreezeResult`. Write ``rows`` to the split files first when
        ``moved_near_duplicates`` is non-empty, then ``state`` to ``state_path``, then
        append ``events``.

    Raises:
        SplitGuardError: Already frozen, ``id_field`` is not ``id``, no text fields, a
            split file missing or empty, a row with no label or no text, or two splits
            sharing an id or a near-duplicate text (with ``move_near_duplicates`` off).
            Nothing is returned, so nothing changes.

    Example::

        result = freeze("proj/splits", ["subject", "body"], label_field="gt_label")
        result.row_counts
        # {'dev': 120, 'val': 60, 'test': 60}
    """
    if id_field != "id":
        raise SplitGuardError(
            "the labeling job keys rows on `id`; copy the id column to `id` in the seed"
        )
    if load_state(splits_dir) is not None:
        raise SplitGuardError(
            f"{state_path(splits_dir)} exists: the split is frozen; never "
            "re-split (`rehash` accepts dev and val edits)"
        )
    if not text_fields:
        raise SplitGuardError("--text-fields names at least one input field")
    rows = split_rows(splits_dir)
    report_client_read("split_state freeze (the seed splits)", sum(len(v) for v in rows.values()))
    _check_rows(rows, label_field, text_fields)
    moved: List[Dict[str, str]] = []
    while move_near_duplicates:
        # Every val or test member of a cross-split pair moves to dev (a val-test pair: both),
        # repeated until no pair is left, since a moved row can be near another one.
        texts = {
            s: {str(r.get(id_field)): row_text(r, text_fields) for r in rows[s]} for s in SPLITS
        }
        to_dev = {
            (s, i)
            for sa, ia, sb, ib in near_duplicate_pairs(texts)
            for s, i in ((sa, ia), (sb, ib))
            if s != "dev"
        }
        if not to_dev:
            break
        for s in ("val", "test"):
            keep, gone = [], []
            for r in rows[s]:
                (gone if (s, str(r.get(id_field))) in to_dev else keep).append(r)
            rows[s] = keep
            rows["dev"].extend(gone)
            moved += [{"id": str(r.get(id_field)), "from": s} for r in gone]
    _check_rows(rows, label_field, text_fields)
    _check_disjoint(
        rows,
        id_field,
        text_fields,
        hint=(NEAR_COPY_HINT + ": --move-near-duplicates moves the val or test member into dev"),
    )
    st: Dict[str, Any] = {
        "version": STATE_VERSION,
        "frozen_at": _now(),
        "id_field": id_field,
        "label_field": label_field,
        "text_fields": list(text_fields),
        "splits": {},
        "history": [],
    }
    for s in SPLITS:
        # Moved rows will be written from `rows`, so hash those bytes; otherwise the file
        # on disk stays as it is and its own bytes are the ones a later verify hashes.
        sha = (
            sha256_rows(rows[s])
            if moved
            else _digest.sha256_file(os.path.join(splits_dir, f"{s}.jsonl"))
        )
        st["splits"][s] = _split_entry(sha, s, rows[s], id_field, label_field)
    counts = {s: len(rows[s]) for s in SPLITS}
    st["history"].append(
        {
            "at": st["frozen_at"],
            "event": "freeze",
            "rows": counts,
            "moved_near_duplicates": len(moved),
        }
    )
    return FreezeResult(
        state=st,
        state_path=state_path(splits_dir),
        rows=rows,
        row_counts=counts,
        moved_near_duplicates=moved,
        events=[_split_event(splits_dir, "frozen", rows=counts, moved=len(moved))],
    )


def verify(splits_dir: str, st: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Check that the split files match the state and share no id or near-duplicate.

    Memoized per process on the files' hashes, so a loop that guards many reads pays
    the disjointness check once per file state.

    Args:
        splits_dir: The folder holding the split files and the state.
        st: The state, when already loaded.

    Returns:
        The state document.

    Raises:
        SplitGuardError: No state (run ``freeze``), a split file missing, a file changed
            since the freeze (test: restore it; dev or val: restore it or ``rehash``),
            or two splits sharing an id or a near-duplicate.
    """
    st = st or load_state(splits_dir)
    if st is None:
        raise SplitGuardError(
            f"no {STATE_FILE} in {splits_dir}: freeze the split (code_templates/split_freeze.py, ACTION freeze)"
        )
    shas = []
    for s in SPLITS:
        p = os.path.join(splits_dir, st["splits"][s]["file"])
        if not os.path.isfile(p):
            raise SplitGuardError(f"{p} is missing")
        got = _digest.sha256_file(p)
        if got != st["splits"][s]["sha256"]:
            raise SplitGuardError(
                f"{p} changed since the split was frozen: "
                + (
                    "test is immutable; restore its frozen bytes"
                    if s == "test"
                    else "restore its frozen bytes, or after a person's edits run "
                    "`code_templates/split_freeze.py` with ACTION = 'rehash' and REASON"
                )
            )
        shas.append(got)
    key = (os.path.abspath(splits_dir), tuple(shas))
    if key not in _VERIFIED:
        rows = split_rows(splits_dir)
        _check_disjoint(rows, st["id_field"], st["text_fields"])
        _VERIFIED[key] = True
    return st


def find_splits_dir(path: Optional[str] = None, project_dir: Optional[str] = None) -> Optional[str]:
    """The splits folder that guards ``path``.

    The file's own folder when it holds a state, else ``<project_dir>/splits`` when that
    holds one, else ``None``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import find_splits_dir

        splits_dir = find_splits_dir("proj/work/fold_0.jsonl", project_dir="proj")  # None: no frozen split
    """
    cands = []
    if path:
        cands.append(os.path.dirname(os.path.abspath(path)))
    if project_dir:
        cands.append(os.path.join(os.path.abspath(project_dir), "splits"))
    for d in cands:
        if os.path.isfile(state_path(d)):
            return d
    return None


def test_read_event(project_dir: str, path: str, step: str, rows: int) -> Dict[str, Any]:
    """A test read as a run-log entry (kind ``analysis``, status ``test-read``).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import test_read_event

        run_log.append(test_read_event("proj", "proj/splits/test.jsonl", step="report", rows=50))
    """
    del project_dir  # the caller appends the entry to that project's log
    return run_log_event("analysis", step, 0, "test-read", file=os.path.abspath(path), rows=rows)


def guard_read(
    path: str,
    step: str,
    allowed: Iterable[str],
    *,
    project_dir: Optional[str] = None,
    rows: Optional[Rows] = None,
    id_field: Optional[str] = None,
    splits_dir: Optional[str] = None,
    record_test_read: bool = True,
) -> GuardRead:
    """Check a read of ``path`` by ``step`` against the frozen split.

    Call it before any step reads a split file or a file built from one. ``allowed``
    names the splits the step may read; a step that is not one of :data:`TEST_STEPS`
    may never name ``test``. A step that may not read test, and the certificate (test
    only), also refuse rows with no id and rows whose text is a row's of a split they
    may not read; the certificate refuses any row that is not a frozen test row.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split

        guard = split.guard_read("proj/splits/dev.jsonl", "demos", {"dev"}, project_dir="proj")
        for event in guard.events:  # a test read is a run-log entry
            run_log.append(event)

    Args:
        path: The file being read.
        step: The step name, recorded on a test read.
        allowed: The splits the step may read.
        project_dir: The project, to find the splits folder when ``path`` is not in it.
        rows: The file's rows, when the caller already read them.
        id_field: The id field; the state's when ``None``.
        splits_dir: The splits folder, when the caller resolved it already.
        record_test_read: Put the test-read entry in ``events`` when test rows were
            read; ``False`` when the caller records it itself.

    Returns:
        A :class:`GuardRead`: ``guarded`` is ``False`` with ``touched`` ``None`` when
        the project has no frozen split; else ``touched`` counts the rows per split
        (``outside`` for ids in no split).

    Raises:
        SplitGuardError: The step may not read test, the files do not verify, the file
            holds rows (or near-copies of rows) of a split the step may not read, rows
            without ids under a frozen split, or non-test rows for the certificate.
    """
    allowed = set(allowed)
    if "test" in allowed and step not in TEST_STEPS:
        raise SplitGuardError(f"step {step!r} may not read the test split")
    splits_dir = splits_dir or find_splits_dir(path, project_dir)
    if splits_dir is None:
        return GuardRead(guarded=False, touched=None)
    st = verify(splits_dir)
    idf = id_field or st["id_field"]
    if rows is None:
        rows = read_rows(path)
    owner = {rid: s for s in SPLITS for rid in st["splits"][s]["ids"]}
    touched: Dict[str, int] = {}
    for r in rows:
        rid = r.get(idf) if isinstance(r, dict) else None
        s = owner.get(str(rid)) if rid is not None else None
        touched[s or "outside"] = touched.get(s or "outside", 0) + 1
    if allowed == {"test"} and touched.get("outside"):
        raise SplitGuardError(
            f"{step}: {path} holds {touched['outside']} row(s) that are not frozen test rows "
            f"(no {idf!r}, or an id in no split); this step reads test rows only"
        )
    if "test" not in allowed or allowed == {"test"}:
        if touched.get("outside") and any(
            not isinstance(r, dict) or r.get(idf) is None for r in rows
        ):
            raise SplitGuardError(
                f"{step}: {path} has rows with no {idf!r}; under a frozen split every row needs its id"
            )
        # A row of a split this step may not read, or a near-copy of one, under another id.
        for s, n in _near_copies(
            rows,
            idf,
            st["text_fields"],
            owner=owner,
            rows_by_split=split_rows(splits_dir),
            splits=[x for x in SPLITS if x not in allowed],
        ).items():
            touched[s] = touched.get(s, 0) + n
    bad = {s: n for s, n in touched.items() if s in SPLITS and s not in allowed}
    if bad:
        detail = ", ".join(f"{n} {s} row(s)" for s, n in sorted(bad.items()))
        raise SplitGuardError(
            f"{step}: {path} holds {detail}; this step may read "
            f"{', '.join(sorted(allowed)) or 'no split'} only (references/full-loop.md, Step 1)"
        )
    events = []
    if touched.get("test") and record_test_read:
        events.append(
            test_read_event(
                project_dir or os.path.dirname(os.path.abspath(splits_dir)),
                path,
                step,
                touched["test"],
            )
        )
    return GuardRead(guarded=True, touched=touched, events=events)


def _near_copies(rows, idf, text_fields, *, owner, rows_by_split, splits):
    """How many of ``rows`` copy a row of each split in ``splits``.

    ``{split: n}``: rows other than that split's own whose text is the same as, or a
    near-copy of, one of its rows' texts.
    """
    texts = {s: {str(r[idf]): row_text(r, text_fields) for r in rows_by_split[s]} for s in splits}
    read = {str(i): row_text(r, text_fields) for i, r in enumerate(rows) if isinstance(r, dict)}
    hits: Dict[str, Set[str]] = {}
    for sa, ia, sb, ib in near_duplicate_pairs({"read": read, **texts}):
        if "read" in (sa, sb) and sa != sb:
            s, i = (sb, ia) if sa == "read" else (sa, ib)
            if owner.get(str(rows[int(i)].get(idf))) != s:
                hits.setdefault(s, set()).add(i)
    return {s: len(v) for s, v in hits.items()}


def nearest_project(path: str) -> Optional[str]:
    """The nearest folder above ``path`` whose ``splits/`` is frozen, or ``None``.

    A file built from the splits (in ``work/``, say) is guarded too.
    """
    d = os.path.dirname(os.path.abspath(path))
    while d != os.path.dirname(d):
        if os.path.isfile(state_path(os.path.join(d, "splits"))):
            return d
        d = os.path.dirname(d)
    return None


def test_ids(
    project_dir: Optional[str] = None, splits_dir: Optional[str] = None
) -> Optional[Set[str]]:
    """The frozen test ids, or ``None`` when the project has no frozen split.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import test_ids

        held_out = test_ids(project_dir="proj") or set()
        rows = [r for r in rows if r["id"] not in held_out]
    """
    d = splits_dir or find_splits_dir(None, project_dir)
    if d is None:
        return None
    st = load_state(d)
    return set(st["splits"]["test"]["ids"]) if st else None


# Not tests: the names mirror the command line's. pytest skips a function whose
# ``__test__`` is false, so a ``from ... import *`` into a test module collects neither.
test_read_event.__test__ = False
test_ids.__test__ = False

# --- rehash -------------------------------------------------------------------------


def _rehash(splits_dir, st, rows, reason, *, allow_val_removal, sha_of):
    """The shared body of :func:`rehash` and :func:`relabel`.

    ``rows`` are the split rows as they will be on disk and ``sha_of(split)`` their
    hashes.
    """
    idf, lf, tf = st["id_field"], st["label_field"], st["text_fields"]
    _check_rows(rows, lf, tf)
    owner = {rid: s for s in SPLITS for rid in st["splits"][s]["ids"]}
    for s in ("dev", "val"):
        ids = [str(r.get(idf)) for r in rows[s]]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise SplitGuardError(
                f"duplicate id in {s}: {dup[:3]}; a fixed row is edited in place, not appended"
            )
        for i in ids:
            if owner.get(i) not in (None, s):
                raise SplitGuardError(
                    f"id {i!r} is a {owner[i]} row appended to {s}; a new row needs a new id"
                )
    moved = [
        str(r.get(idf))
        for s in ("dev", "val")
        for r in rows[s]
        if owner.get(str(r.get(idf)), "dev") != s
    ]
    if moved:
        raise SplitGuardError(
            f"{len(moved)} row(s) (e.g. {moved[:3]}) changed split or are new in val; a frozen "
            "row keeps its split and new rows go to dev"
        )
    # An id or a near-duplicate shared with test (or across dev and val).
    _check_disjoint(
        rows,
        idf,
        tf,
        hint="test rows are immutable, so drop the new or edited dev or val "
        "row that copies another split's row, then rehash again",
    )
    st = json.loads(json.dumps(st))  # the caller's document is not changed
    change = {}
    for s in ("dev", "val"):
        old = st["splits"][s]
        new = _split_entry(sha_of(s), s, rows[s], idf, lf)
        change[s] = {
            "removed": sorted(set(old["ids"]) - set(new["ids"])),
            "added": sorted(set(new["ids"]) - set(old["ids"])),
            "gold_changed": sorted(
                i for i in new["ids"] if i in old["gold"] and old["gold"][i] != new["gold"][i]
            ),
        }
        st["splits"][s] = new
    if change["val"]["removed"] and not allow_val_removal:
        raise SplitGuardError(
            f"{len(change['val']['removed'])} val row(s) removed (e.g. {change['val']['removed'][:3]}); "
            "pass --allow-val-removal if the user meant it"
        )
    counts = {s: len(rows[s]) for s in SPLITS}
    st["history"].append(
        {"at": _now(), "event": "rehash", "reason": reason, "rows": counts, **change}
    )
    _VERIFIED.clear()
    summary = {
        "rows": counts,
        "reason": reason,
        **{f"{s}_{k}": len(v) for s, c in change.items() for k, v in c.items()},
    }
    return st, summary, [_split_event(splits_dir, "rehashed", reason=reason, rows=counts)]


def rehash(splits_dir: str, reason: str, allow_val_removal: bool = False) -> RehashResult:
    """Accept edited dev and val files into the frozen split.

    Use it after a person fixed labels in ``dev.jsonl`` or ``val.jsonl`` or added rows to
    dev, so :func:`guard_read` stops refusing the changed files. Test never changes: a
    test label fix or a new test sample needs a new project and freeze.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split

        res = split.rehash("proj/splits", reason="fixed three dev labels")
        open(res.state_path, "w").write(json.dumps(res.state, indent=2))
        run_log.extend(res.events)

    Args:
        splits_dir: The folder holding the split files and the state.
        reason: Why dev or val changed; recorded in the state's history and the run log.
        allow_val_removal: Accept val rows that were removed (the default refuses, since
            a smaller val changes what selection can resolve).

    Returns:
        A :class:`RehashResult`; write ``state`` to ``state_path``, then append
        ``events``. ``summary`` is the count report.

    Raises:
        SplitGuardError: No state, no reason, ``test.jsonl`` changed, a duplicate id, a
            frozen row in another split, a new row in val, a near-duplicate of another
            split's row, an empty split, or val rows removed without
            ``allow_val_removal``.
    """
    st = load_state(splits_dir)
    if st is None:
        raise SplitGuardError(
            f"no {STATE_FILE} in {splits_dir}: freeze the split (code_templates/split_freeze.py, ACTION freeze)"
        )
    if not (reason or "").strip():
        raise SplitGuardError("--reason names why dev or val changed")
    if (
        _digest.sha256_file(os.path.join(splits_dir, "test.jsonl"))
        != st["splits"]["test"]["sha256"]
    ):
        raise SplitGuardError(
            "test.jsonl changed: test rows are immutable once frozen; restore it (a test "
            "label fix or a new test sample needs a new project and freeze)"
        )
    rows = split_rows(splits_dir)
    new_st, summary, events = _rehash(
        splits_dir,
        st,
        rows,
        reason,
        allow_val_removal=allow_val_removal,
        sha_of=lambda s: _digest.sha256_file(os.path.join(splits_dir, f"{s}.jsonl")),
    )
    return RehashResult(
        state=new_st, state_path=state_path(splits_dir), summary=summary, events=events
    )


def relabel(splits_dir: str, labels: Dict[str, Any], reason: str) -> RelabelResult:
    """Apply ``{id: label}`` to the dev and val rows and accept the result as a rehash.

    Use it to land a person's label decisions on a frozen split in one step. The
    current files must verify first (the split is as frozen). Nothing is written here:
    a refused relabel returns nothing and the files are untouched; an accepted one
    returns the rows and the state for the caller to write.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split

        res = split.relabel("proj/splits", {"r12": "TP", "r40": "FP"}, reason="review decisions")
        for name in ("dev", "val"):
            open(f"proj/splits/{name}.jsonl", "wb").write(split.rows_bytes(res.rows[name]))

    Args:
        splits_dir: The folder holding the split files and the state.
        labels: Row id -> new gold value, applied to dev and val rows with that id.
        reason: Why, for the rehash history.

    Returns:
        A :class:`RelabelResult`; write ``rows["dev"]`` and ``rows["val"]`` as
        :func:`rows_bytes`, then ``state``, then append ``events``.

    Raises:
        SplitGuardError: The split does not verify, or any :func:`rehash` refusal.
    """
    st = verify(splits_dir)
    if not (reason or "").strip():
        raise SplitGuardError("--reason names why dev or val changed")
    rows = split_rows(splits_dir)
    for s in ("dev", "val"):
        for r in rows[s]:
            if str(r.get(st["id_field"])) in labels:
                r[st["label_field"]] = labels[str(r[st["id_field"]])]
    new_st, summary, events = _rehash(
        splits_dir, st, rows, reason, allow_val_removal=False, sha_of=lambda s: sha256_rows(rows[s])
    )
    return RelabelResult(
        state=new_st,
        state_path=state_path(splits_dir),
        summary=summary,
        events=events,
        rows={s: rows[s] for s in ("dev", "val")},
    )


# ---------------------------------------------------------------------------
# The project-level step: one call that makes the seed trustworthy. It owns the
# project's files: <project>/splits/{dev,val,test}.jsonl and split_state.json, the demos
# preview and the folds under <project>/work, and the run log.
# ---------------------------------------------------------------------------

SEED_FILE = "seed.jsonl"
COLD_START_SAMPLE_FILE = os.path.join("work", "cold_start_sample.jsonl")
DEFAULT_DEV_FRAC = 0.5
DEFAULT_VAL_FRAC = 0.25  # test gets the rest
SPLIT_SEED = 42


class SeedReport(OperatorResult):
    """What :func:`prepare_labeling_seed` did.

    ``status`` is ``ready`` (the split is frozen, demos and folds written) or
    ``needs_labels`` (a cold start: ``sample`` names the rows for people to label, as
    ``label_field``, and save as the project's ``seed.jsonl``). ``splits_dir``, ``demos``
    and ``folds_dir`` are the files written; ``row_counts`` the rows per split.
    """

    status: str
    project_dir: str
    seed: Optional[str] = None
    splits_dir: Optional[str] = None
    row_counts: Dict[str, int] = {}
    frozen_now: bool = False
    split_strategy: Optional[str] = None
    demos: Optional[str] = None
    demos_per_class: int = 0
    folds_dir: Optional[str] = None
    fold_sizes: List[int] = []
    sample: Optional[str] = None
    sample_rows: int = 0
    label_field: Optional[str] = None


def _label_split(rows, gt_field, dev_frac, val_frac, seed):
    """Stratified three-way split by label value: round-robin from each label's bucket.

    The label strategy of the dataset-transformation skill's ``cluster_split.py`` (its
    fallback when no embedder is installed): every bucket of three or more rows gives at
    least one test row, every bucket of two a val row, so the held-out gate is never empty
    on a small class.
    """
    import random

    buckets: Dict[Any, List[int]] = {}
    for i, row in enumerate(rows):
        key = row.get(gt_field) if isinstance(row, dict) else None
        buckets.setdefault(_digest.canonical_json(key), []).append(i)
    rng = random.Random(seed)
    dev: List[int] = []
    val: List[int] = []
    test: List[int] = []
    for _key in sorted(buckets):
        ids = list(buckets[_key])
        rng.shuffle(ids)
        n = len(ids)
        if n >= 3:
            d_cut = max(1, min(int(round(n * dev_frac)), n - 2))
            v_cut = max(d_cut + 1, min(d_cut + int(round(n * val_frac)), n - 1))
        elif n == 2:
            d_cut, v_cut = 1, 2
        else:
            d_cut, v_cut = 1, 1
        dev.extend(ids[:d_cut])
        val.extend(ids[d_cut:v_cut])
        test.extend(ids[v_cut:])
    return {"dev": sorted(dev), "val": sorted(val), "test": sorted(test)}, len(buckets)


def _write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def prepare_labeling_seed(  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    project_dir: str,
    data,
    *,
    seed: Optional[str] = None,
    demos_per_class: int = 2,
    k: int = 4,
    target: float = 0.9,
    move_near_duplicates: bool = False,
) -> SeedReport:
    """Make the seed trustworthy: split it once, freeze the split, pick demos, build folds.

    With a seed (``data.seed``, or ``seed``, or ``<project_dir>/seed.jsonl``): when the
    project has no ``splits/`` yet, the seed is split dev / val / test, stratified by
    label (:func:`_label_split`); when the split is not frozen yet it is frozen
    (:func:`freeze`), after which no step can change or misuse it. Then the dev split is
    read under the guard (:func:`guard_read`), the few-shot demos a round will send are
    previewed (:func:`~..demos.select_demos`) and the rotation folds are built
    (:func:`~..folds.build_kfold`). A second call finds the frozen split and only
    refreshes the demos and folds.

    Without a seed (a cold start): the corpus head is read on the client, a sample sized
    to certify ``target`` is drawn (:func:`~..cold_start.cold_start_sample_size`,
    :func:`~..cold_start.draw_sample`) and written to ``work/cold_start_sample.jsonl``;
    the result says ``needs_labels``. People label it as ``data.label_field``, save it as
    ``<project_dir>/seed.jsonl``, and call this again.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import inspect_labeling_data, prepare_labeling_seed

        data = inspect_labeling_data(spec="tickets/labeling_spec.md", seed="tickets/seed.jsonl", region="us-west-2")
        seed = prepare_labeling_seed("tickets", data)           # splits/ frozen, work/demos.jsonl, work/folds/
        seed.status, seed.row_counts, seed.demos

    Args:
        project_dir: The project folder; its ``splits/``, ``work/`` and run log are written.
        data: :func:`~..inspect.inputs.inspect_labeling_data`'s report (task, fields, label field).
        seed: The labeled seed, when not ``data.seed`` nor ``<project_dir>/seed.jsonl``.
        demos_per_class: Demos per class the rounds will draw from dev (0 = zero-shot).
        k: Rotation folds.
        target: Cold start: the accuracy the sample must be able to certify.
        move_near_duplicates: Freeze: move the val or test member of a near-duplicate
            pair into dev instead of refusing.

    Returns:
        A :class:`SeedReport`.

    Raises:
        SplitGuardError: The split files changed after the freeze, or a near-duplicate
            pair without ``move_near_duplicates``.
        DataPreparerOperatorError: No seed and no corpus, or a seed row without the label.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows, _run_log
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import cold_start
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import demos as demos_op
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import folds as folds_op

    os.makedirs(project_dir, exist_ok=True)
    splits_dir = os.path.join(project_dir, "splits")
    seed_path = seed or data.seed
    if seed_path is None and os.path.isfile(os.path.join(project_dir, SEED_FILE)):
        seed_path = os.path.join(project_dir, SEED_FILE)
    label_field = data.label_field
    warnings: List[str] = []

    # --- cold start: no seed, no split -> the sample for people to label ---
    if seed_path is None and not os.path.isfile(state_path(splits_dir)):
        if not data.corpus:
            raise DataPreparerOperatorError(
                "no seed and no corpus: pass seed=... (labeled rows) or a corpus to sample from"
            )
        corpus_rows = _rows.read_input_rows(
            data.corpus, region=data.region, primitive="prepare_labeling_seed"
        )
        n = cold_start.cold_start_sample_size(target, len(corpus_rows), data.task)
        sample_rows, report = cold_start.draw_sample(
            corpus_rows,
            n,
            SPLIT_SEED,
            text_fields=tuple(data.text_fields),
            corpus_total=len(corpus_rows),
        )
        sample_path = os.path.join(project_dir, COLD_START_SAMPLE_FILE)
        _write_jsonl(sample_path, sample_rows)
        note = cold_start.certificate_warning(len(sample_rows), target, data.task)
        if note:
            warnings.append(note)
        _run_log.append_entry(
            project_dir,
            run_log_event(
                "artifact",
                "cold-start",
                0,
                "sampled",
                sample=sample_path,
                rows=len(sample_rows),
                target=target,
                label_field=label_field,
                **{k: v for k, v in report.items() if k != "rows"},
            ),
        )
        warnings.append(
            f"cold start: label the {len(sample_rows)} rows of {sample_path} as {label_field!r}, "
            f"save them as {os.path.join(project_dir, SEED_FILE)}, and call prepare_labeling_seed again"
        )
        return SeedReport(
            status="needs_labels",
            project_dir=project_dir,
            sample=sample_path,
            sample_rows=len(sample_rows),
            label_field=label_field,
            warnings=warnings,
        )

    # --- split once ---
    strategy = None
    if not all(os.path.isfile(os.path.join(splits_dir, f"{s}.jsonl")) for s in SPLITS):
        if seed_path is None:
            raise DataPreparerOperatorError(
                f"{splits_dir} is incomplete and there is no seed to split"
            )
        seed_rows = _rows.read_input_rows(
            seed_path, region=data.region, primitive="prepare_labeling_seed"
        )
        missing = [
            i
            for i, r in enumerate(seed_rows)
            if not isinstance(r, dict) or r.get(label_field) is None
        ]
        if missing:
            raise DataPreparerOperatorError(
                f"{len(missing)} seed row(s) have no {label_field!r} (first at line {missing[0] + 1}); "
                "a seed row needs its gold label"
            )
        for i, row in enumerate(seed_rows):
            row.setdefault("id", str(i))
        parts, n_buckets = _label_split(
            seed_rows, label_field, DEFAULT_DEV_FRAC, DEFAULT_VAL_FRAC, SPLIT_SEED
        )
        for name, idxs in parts.items():
            _write_jsonl(os.path.join(splits_dir, f"{name}.jsonl"), [seed_rows[i] for i in idxs])
        strategy = "label"
        manifest = {
            "strategy": strategy,
            "n_rows": len(seed_rows),
            "n_buckets": n_buckets,
            "seed": SPLIT_SEED,
            "dev_frac": DEFAULT_DEV_FRAC,
            "val_frac": DEFAULT_VAL_FRAC,
            "counts": {name: len(idxs) for name, idxs in parts.items()},
            "source": seed_path,
            "gt_field": label_field,
            "text_fields": list(data.text_fields),
        }
        with open(os.path.join(splits_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            fh.write(json.dumps(manifest, indent=2) + "\n")
        _run_log.append_entry(
            project_dir,
            run_log_event("artifact", "split", 0, "written", splits_dir=splits_dir, **manifest),
        )

    # --- freeze once ---
    frozen_now = False
    if not os.path.isfile(state_path(splits_dir)):
        frozen = freeze(
            splits_dir,
            data.text_fields,
            label_field=label_field,
            move_near_duplicates=move_near_duplicates,
        )
        with open(frozen.state_path, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(frozen.state, indent=2) + "\n")
        for name, rows in frozen.rows.items():  # moved near-duplicates land in their new split
            _write_jsonl(os.path.join(splits_dir, f"{name}.jsonl"), rows)
        _run_log.append_entries(project_dir, frozen.events)
        warnings.extend(frozen.warnings)
        frozen_now = True
    else:
        verify(splits_dir)  # the files are as frozen

    # --- the dev split, read under the guard ---
    dev_path = os.path.join(splits_dir, "dev.jsonl")
    dev_rows = _rows.read_jsonl(dev_path, add_id=True)
    guard = guard_read(dev_path, "demos", {"dev"}, project_dir=project_dir, rows=dev_rows)
    _run_log.append_entries(project_dir, guard.events)
    report_client_read("prepare_labeling_seed", len(dev_rows))
    row_counts = {
        s: sum(
            1 for _ in open(os.path.join(splits_dir, f"{s}.jsonl"), encoding="utf-8") if _.strip()
        )
        for s in SPLITS
    }

    # --- demos preview and folds ---
    work = os.path.join(project_dir, "work")
    demos_path = None
    if demos_per_class > 0:
        picked = demos_op.select_demos(
            dev_rows, mode="fixed", k=demos_per_class, task=data.task, gt_field=label_field
        )
        demos_path = os.path.join(work, "demos.jsonl")
        _write_jsonl(demos_path, picked.demos)
        warnings.extend(picked.warnings)
    folds_dir = os.path.join(work, "folds")
    folds = folds_op.build_kfold(
        dev_rows,
        out_dir=folds_dir,
        gt_field=label_field,
        k=min(k, max(2, len(dev_rows) // 2)),
        task=data.task if data.task in ("classification", "ner") else "classification",
        frozen_split=True,
    )
    os.makedirs(folds_dir, exist_ok=True)
    for name, fold_rows in folds.files.items():
        _write_jsonl(os.path.join(folds_dir, name), fold_rows)
    with open(os.path.join(folds_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(folds.manifest, indent=2) + "\n")
    warnings.extend(folds.warnings)
    return SeedReport(
        status="ready",
        project_dir=project_dir,
        seed=seed_path,
        splits_dir=splits_dir,
        row_counts=row_counts,
        frozen_now=frozen_now,
        split_strategy=strategy,
        demos=demos_path,
        demos_per_class=demos_per_class,
        folds_dir=folds_dir,
        fold_sizes=list(folds.fold_sizes),
        label_field=label_field,
        warnings=warnings,
    )
