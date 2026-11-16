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
"""The frozen seed split, kept in the project folder.

:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split` decides;
this module writes: the split files and ``split_state.json`` under ``<proj>/splits/``, and
the freeze, rehash, relabel and test-read entries in the project's run log. Freeze the
Step 1 split once, accept dev and val edits with :func:`rehash_split` or :func:`relabel_split`, and
guard every read of a split file with :func:`guard_split_read`.
"""

from __future__ import absolute_import

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows as _client_rows
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._digest import sha256_file
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import append_entry
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as _split
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import (
    SPLITS,
    SplitGuardError,
    find_splits_dir,
    nearest_project,
    state_path,
    verify,
)

__all__ = [
    "guard_split_read_or_raise",
    "freeze_split",
    "verify_splits",
    "rehash_split",
    "relabel_split",
    "guard_split_read",
    "log_test_read",
]


# --- files -------------------------------------------------------------------------


def _project_of(splits_dir: str) -> str:
    return os.path.dirname(os.path.abspath(splits_dir))


def _write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _write_splits(splits_dir: str, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    """Write every split's temp file first, then rename them all.

    A failed write changes no split file.
    """
    tmps = []
    try:
        for s in SPLITS:
            path = os.path.join(splits_dir, f"{s}.jsonl")
            with open(path + ".tmp", "w", encoding="utf-8") as fh:
                for r in rows[s]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            tmps.append((path + ".tmp", path))
    except BaseException:
        for tmp, _ in tmps:
            if os.path.exists(tmp):
                os.remove(tmp)
        raise
    for tmp, path in tmps:
        os.replace(tmp, path)


def _write_state(splits_dir: str, st: Dict[str, Any]) -> None:
    p = state_path(splits_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, p)


def _log_events(project_dir: str, events: Iterable[Dict[str, Any]]) -> List[str]:
    return [append_entry(project_dir, dict(ev)) for ev in events]


# --- freeze / guard / rehash ---------------------------------------------------------


def freeze_split(
    splits_dir: str,
    text_fields: List[str],
    *,
    id_field: str = "id",
    label_field: str = "gt_label",
    move_near_duplicates: bool = False,
) -> Dict[str, Any]:
    """Record the Step 1 split in ``split_state.json`` (once).

    Refuses to freeze twice, an id field other than ``id`` (the labeling job keys rows on
    it), a row with no label or no text, and an empty split. With ``move_near_duplicates``
    the val or test member of each cross-split near-duplicate pair is moved into dev first
    and the split files are rewritten.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import freeze_split

        out = freeze_split("proj/splits", ["text"], label_field="gt_label")
        out["rows"]  # {"dev": 54, "val": 38, "test": 62}

    Returns:
        ``{"rows": {split: count}, "moved_near_duplicates": [...], "state": path}``.

    Raises:
        SplitGuardError: The split cannot be frozen as it is.
    """
    _client_rows.print_client_reads()
    res = _split.freeze(
        splits_dir,
        text_fields,
        id_field=id_field,
        label_field=label_field,
        move_near_duplicates=move_near_duplicates,
    )
    if res.moved_near_duplicates:
        _write_splits(splits_dir, res.rows)
    _write_state(splits_dir, res.state)
    _log_events(_project_of(splits_dir), res.events)
    return {
        "rows": res.row_counts,
        "moved_near_duplicates": res.moved_near_duplicates,
        "state": res.state_path,
    }


def verify_splits(splits_dir: str) -> Dict[str, Any]:
    """Check the split files against the frozen state; ``{"ok", "rows", "events"}``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import verify_splits

        verify_splits("proj/splits")["ok"]
    """
    st = verify(splits_dir)
    return {
        "ok": True,
        "rows": {s: len(st["splits"][s]["ids"]) for s in SPLITS},
        "events": len(st["history"]),
    }


def guard_split_read(
    path: str,
    step: str,
    allowed: Iterable[str],
    *,
    project_dir: Optional[str] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    id_field: Optional[str] = None,
    splits_dir: Optional[str] = None,
    log: bool = True,
) -> Optional[Dict[str, Any]]:
    """Check a read of ``path`` by ``step`` against the frozen split.

    ``allowed`` is the set of splits the step may read. A test read is a run-log entry;
    with ``log=False`` the caller writes it (:func:`log_test_read`).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import guard_split_read

        touched = guard_split_read("proj/splits/dev.jsonl", "refine", {"dev"}, project_dir="proj")

    Returns:
        ``{split: rows}`` touched (with ``outside`` for ids in no split), or ``None`` when
        the project has no frozen split.

    Raises:
        SplitGuardError: The read is not allowed.
    """
    splits_dir = splits_dir or find_splits_dir(path, project_dir)
    res = _split.guard_read(
        path,
        step,
        allowed,
        project_dir=project_dir,
        rows=rows,
        id_field=id_field,
        splits_dir=splits_dir,
        record_test_read=log,
    )
    if res.events:  # a project with no frozen split has nothing to log
        _log_events(project_dir or _project_of(splits_dir), res.events)
    return res.touched


def log_test_read(project_dir: str, path: str, step: str, rows: int) -> str:
    """A test read as a run-log entry of the project (kind analysis, status test-read).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import log_test_read

        log_test_read("proj", "proj/splits/test.jsonl", "report", 62)
    """
    return append_entry(project_dir, _split.test_read_event(project_dir, path, step, rows))


def guard_split_read_or_raise(
    path: str, step: str, allowed: Iterable[str], project_dir: Optional[str] = None, log=True
):
    """:func:`guard_split_read` with the project found from ``path`` when none is given.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import guard_split_read_or_raise

        guard_split_read_or_raise("proj/splits/val.jsonl", "select", {"val"})
    """
    return guard_split_read(
        path, step, allowed, project_dir=project_dir or nearest_project(path), log=log
    )


def rehash_split(splits_dir: str, reason: str, allow_val_removal: bool = False) -> Dict[str, Any]:
    """Accept edited dev and val files.

    The state is rewritten (tmp file, then rename), and the history names each split's
    removed, added and gold-changed ids. Refuses any test change, a moved row, a new val
    row, a removed val row (unless ``allow_val_removal``) and a near-copy of another
    split's row.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import rehash_split

        rehash_split("proj/splits", "fixed two dev labels")
    """
    res = _split.rehash(splits_dir, reason, allow_val_removal)
    _write_state(splits_dir, res.state)
    _log_events(_project_of(splits_dir), res.events)
    return res.summary


def relabel_split(splits_dir: str, labels: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Write ``{id: label}`` into ``dev.jsonl`` and ``val.jsonl`` in place and accept them.

    Any failure before the state is rewritten (a failed write) puts both files back; once
    the state is rewritten the relabel is accepted and the files stay.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import relabel_split

        relabel_split("proj/splits", {"d1": "refund"}, "label review")
    """
    res = _split.relabel(splits_dir, labels, reason)
    before = {}
    for s in ("dev", "val"):
        with open(os.path.join(splits_dir, f"{s}.jsonl"), "rb") as fh:
            before[s] = fh.read()
    state_sha = sha256_file(state_path(splits_dir))
    try:
        for s in ("dev", "val"):
            _write_rows(os.path.join(splits_dir, f"{s}.jsonl"), res.rows[s])
        _write_state(splits_dir, res.state)
        _log_events(_project_of(splits_dir), res.events)
        return res.summary
    except BaseException:
        if sha256_file(state_path(splits_dir)) == state_sha:  # not accepted: put the bytes back
            for s, data in before.items():
                path = os.path.join(splits_dir, f"{s}.jsonl")
                with open(path + ".tmp", "wb") as fh:
                    fh.write(data)
                os.replace(path + ".tmp", path)
        raise


_ = SplitGuardError  # re-exported for callers that catch it from here
