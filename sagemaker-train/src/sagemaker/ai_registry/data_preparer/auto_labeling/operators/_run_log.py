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
"""The run-log entry an auto-labeling operator returns in its result's ``events``.

The entry kinds, the reserved fields and the refusal of ``approval`` are the skill
project's run-log format (its ``run_log.py`` writer), which is why this is private to
the labeling primitive and not part of :mod:`~sagemaker.ai_registry.data_preparer._operator_result`.
"""

from __future__ import absolute_import

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# Entry fields the project's run-log writer sets or takes as arguments; an event's
# other fields must not use them.
RUN_LOG_RESERVED = ("kind", "step", "iteration", "created_at", "status", "arn", "id")


def run_log_event(
    kind: str,
    step: str,
    iteration: int,
    status: str,
    arn: Optional[str] = None,
    **fields: Any,
) -> Dict[str, Any]:
    """One run-log entry for a result's ``events`` list.

    Args:
        kind: The entry kind: ``job``, ``artifact``, ``analysis``, ``decision``,
            ``score`` or ``incident``. Never ``approval``: a person records those.
        step: The workflow step that produced it.
        iteration: The refinement iteration it belongs to.
        status: The entry's status word (``launched``, ``completed``, ``read``, ...).
        arn: A job entry's ARN.
        **fields: The fields of the kind.

    Returns:
        The entry dict, ready for the project's run-log writer.

    Raises:
        DataPreparerOperatorError: ``kind`` is ``approval``, or a field uses a
            reserved key.
    """
    if kind == "approval":
        raise DataPreparerOperatorError(
            "approvals come from the user; the agent records them with the run_log.py CLI"
        )
    clash = sorted(set(fields) & set(RUN_LOG_RESERVED))
    if clash:
        raise DataPreparerOperatorError(f"pass {', '.join(clash)} as arguments, not as fields")
    entry: Dict[str, Any] = {"kind": kind, "step": step, "iteration": iteration, "status": status}
    if arn:
        entry["arn"] = arn
    entry.update(fields)
    return entry


# ---------------------------------------------------------------------------
# The project's run log on disk, for the project-level operators (run_labeling_round and the
# others that own a project folder). The step operators return entries; these write them,
# in the format the skill's run_log.py writer uses: one JSON object per line of
# <project>/run-log.jsonl, id = <kind>-<step>-iter<N>-<status>[-<n>], created_at in UTC,
# paths inside the project stored relative to it.
# ---------------------------------------------------------------------------

RUN_LOG_FILE = "run-log.jsonl"
_SLUG = re.compile(r"[^a-z0-9]+")


def _slug(value: Any) -> str:
    return _SLUG.sub("-", str(value).lower()).strip("-") or "x"


def _relativize(value: Any, root: str) -> Any:
    if isinstance(value, dict):
        return {k: _relativize(v, root) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_relativize(v, root) for v in value]
    if not isinstance(value, str) or "://" in value or not value:
        return value
    candidate = os.path.normpath(os.path.abspath(value)) if os.path.isabs(value) else None
    if candidate is None and os.path.exists(value):
        candidate = os.path.abspath(value)
    if candidate and (candidate == root or candidate.startswith(root + os.sep)):
        return os.path.relpath(candidate, root)
    return value


def read_entries(project_dir: str) -> List[Dict[str, Any]]:
    """The project's run-log entries, oldest first (``[]`` when there is no log yet)."""
    path = os.path.join(os.path.abspath(project_dir), RUN_LOG_FILE)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            if isinstance(entry, dict):
                out.append(entry)
    return out


def append_entry(project_dir: str, entry: Dict[str, Any]) -> str:
    """Append one entry (a :func:`run_log_event` dict) to the project's run log; its id.

    The id is ``<kind>-<step>-iter<N>-<status>``, suffixed ``-2``, ``-3``, ... when taken.
    ``created_at`` is set when the entry has none. The file is created with the project
    folder.
    """
    root = os.path.abspath(project_dir)
    for key in ("kind", "step", "iteration", "status"):
        if entry.get(key) in (None, ""):
            raise DataPreparerOperatorError(f"a run-log entry needs {key!r}: {entry!r}")
    entry = {k: _relativize(v, root) for k, v in entry.items()}
    entry.setdefault("created_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ"))
    stem = "-".join(
        (
            _slug(entry["kind"]),
            _slug(entry["step"]),
            f"iter{entry['iteration']}",
            _slug(entry["status"]),
        )
    )
    os.makedirs(root, exist_ok=True)
    taken = {str(e["id"]) for e in read_entries(root) if e.get("id")}
    entry_id, n = stem, 1
    while entry_id in taken:
        n += 1
        entry_id = f"{stem}-{n}"
    line = json.dumps({"id": entry_id, **entry}, default=str)
    path = os.path.join(root, RUN_LOG_FILE)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return entry_id


def append_entries(project_dir: str, entries: Iterable[Dict[str, Any]]) -> List[str]:
    """Append each entry in order; their ids."""
    return [append_entry(project_dir, e) for e in entries]
