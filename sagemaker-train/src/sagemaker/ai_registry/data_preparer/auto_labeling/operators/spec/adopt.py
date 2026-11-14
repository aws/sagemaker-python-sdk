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
"""Adopting the in-job search's pick as the shipped spec, after the user's recorded yes.

A labeling job run with the optimizer leaves a pick (``spec.md``) beside its
``optimizer_result.json`` and ``pick.json``. The user's yes is an ``approval`` run-log entry
(step ``optimizer-adopt``) naming the pick's SHA-256 and the job; a pick whose bytes or job
differ is refused. :func:`adopt` checks that and returns the ``decision.json`` the project
keeps, the relative path the pick is copied to and the ``decision`` entry to record. Step 6
ships it: :func:`check_shippable` re-checks the spec's bytes against the last adopt entry
and the approval it names before the caller copies the spec. No job is submitted and no
model is called; the file copies stay with the caller.
"""

from __future__ import absolute_import

import json
import os
from typing import Any, Dict, Optional

from pydantic import Field

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event

STEP = "optimizer-adopt"
# The job's id for the incumbent spec in optimizer_result.json (`pick_val`).
INCUMBENT_PICK = "Round 0 | Candidate 0"
PICK_FILES = ("optimizer_result.json", "pick.json")


class AdoptError(DataPreparerOperatorError):
    """A refusal, with the reason the command prints."""

    def __init__(self, msg):
        """Prefix ``msg`` with the command's name."""
        super().__init__(f"adopt_job_pick: {msg}")


def adopted_ship(optimizer_result):
    """``initial`` when the search kept the incumbent, ``refined`` otherwise.

    Args:
        optimizer_result: The job's ``optimizer_result.json`` document, or ``None`` when it
            could not be read.
    """
    try:
        picked = optimizer_result.get("pick_val")
    except AttributeError:
        picked = None
    return "initial" if picked == INCUMBENT_PICK else "refined"


def check_pick_folder(pick_name, present):
    """Refuse unless the pick, ``optimizer_result.json`` and ``pick.json`` are all in its folder.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import check_pick_folder

        check_pick_folder("labeling_spec_pick.md", present=set(os.listdir("proj/job/dev")))

    Args:
        pick_name: The pick's path as given, for the message.
        present: The file names present in the pick's folder.
    """
    if not all(f in present for f in (os.path.basename(pick_name), *PICK_FILES)):
        raise AdoptError(f"no pick, optimizer_result.json and pick.json at {pick_name}'s folder")


def approval_for(entries, approval_id, sha, job_name):
    """The approved ``optimizer-adopt`` entry ``approval_id`` naming this pick, or a refusal.

    Args:
        entries: The project's run-log entries.
        approval_id: The approval entry's id.
        sha: The pick's SHA-256.
        job_name: The job named in the pick's ``pick.json``.

    Returns:
        The approval entry.

    Raises:
        AdoptError: No approved entry with that id names the pick's sha and job.
    """
    named = [
        e
        for e in entries
        if e.get("kind") == "approval"
        and e.get("id") == approval_id
        and e.get("status") == "approved"
        and e.get("step") == STEP
        and e.get("optimizer_pick_sha256") == sha
        and e.get("job_name") == job_name
    ]
    if not named:
        raise AdoptError(
            f"no approved {STEP} entry {approval_id!r} names this pick's sha256 {sha[:12]} and job "
            f"{job_name}; the pick changed, or the agent records the user's yes with "
            "`run_log.py write --kind approval` and the run's pick.json values"
        )
    return named[0]


class Adoption(OperatorResult):
    """What adopting a pick leaves: the ``decision.json`` document and where the pick goes.

    ``events`` holds the ``decision`` entry (step ``optimizer-adopt``, status ``adopted``)
    naming the approval, the sha and the job. The caller copies the pick's bytes to
    ``spec`` (under the project) and writes ``decision`` as ``decision.json``.
    """

    decision: Dict[str, Any] = Field(default_factory=dict)
    spec: str = ""
    sha256: str = ""
    job_name: Optional[str] = None


def adopt(entries, *, pick_bytes, pick_json, optimizer_result, approval_id):
    """Adopt the in-job search's pick as the shipped spec, after the user's recorded yes.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import adopt as adopt_op

        adoption = adopt_op.adopt(run_log_entries, pick_bytes=pick, pick_json=pick_doc,
                                  optimizer_result=optimizer_doc, approval_id="a-17")
        open("proj/labeling_spec.md", "wb").write(pick)
        run_log.extend(adoption.events)

    Args:
        entries: The project's run-log entries.
        pick_bytes: The pick's bytes.
        pick_json: The run's ``pick.json`` document (``job_name``).
        optimizer_result: The run's ``optimizer_result.json`` document, or ``None``.
        approval_id: The approval entry's id.

    Returns:
        An :class:`Adoption`.

    Raises:
        AdoptError: The approval does not name this pick.
    """
    sha = _digest.sha256_hex(pick_bytes)
    job_name = (pick_json or {}).get("job_name")
    approval_for(entries, approval_id, sha, job_name)
    # Under work/, where Step 6 and the report look for decision.json's shipped_spec_file.
    rel = os.path.join("work", f"optimizer_pick_{sha[:12]}.md")
    decision = {
        "ship": adopted_ship(optimizer_result),
        "spec": rel,
        "val": None,
        "reason": "the user adopted the in-job search pick (chosen on val)",
        "selection_mode": "in-job-search",
        "shipped_spec_file": os.path.basename(rel),
    }
    event = run_log_event(
        "decision",
        STEP,
        0,
        "adopted",
        file="decision.json",
        spec=rel,
        approval=approval_id,
        pick_sha256=sha,
        job_name=job_name,
    )
    return Adoption(decision=decision, spec=rel, sha256=sha, job_name=job_name, events=[event])


def check_shippable(entries, decision, spec_bytes):
    """Refuse unless the adopted spec's bytes still match the approved SHA-256.

    The last ``adopted`` decision entry must name the bytes' sha, and an approved
    ``optimizer-adopt`` approval must name that entry's approval id and the sha.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import check_shippable

        check_shippable(run_log_entries, decision_doc, spec_bytes=open("proj/labeling_spec.md", "rb").read())

    Args:
        entries: The project's run-log entries.
        decision: The project's ``decision.json`` document.
        spec_bytes: The bytes of the spec ``decision["spec"]`` names.

    Returns:
        The sha.

    Raises:
        AdoptError: The spec is not the approved pick.
    """
    sha = _digest.sha256_hex(spec_bytes)
    adopted = [
        e
        for e in entries
        if e.get("kind") == "decision" and e.get("step") == STEP and e.get("status") == "adopted"
    ]
    approved = {
        (e.get("id"), e.get("optimizer_pick_sha256"))
        for e in entries
        if e.get("kind") == "approval" and e.get("step") == STEP and e.get("status") == "approved"
    }
    last = adopted[-1] if adopted else {}
    if (
        decision.get("selection_mode") != "in-job-search"
        or last.get("pick_sha256") != sha
        or (last.get("approval"), sha) not in approved
    ):
        raise AdoptError(
            f"{decision.get('spec')} (sha256 {sha[:12]}) is not the approved pick; "
            "run adopt-job again after the user's yes"
        )
    return sha


def no_adopted_spec_message(project, error):
    """The refusal when ``decision.json`` or the spec it names cannot be read.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import no_adopted_spec_message

        raise SystemExit(no_adopted_spec_message("proj", error))
    """
    return AdoptError(f"no adopted spec to ship in {project}/decision.json ({error})")


# --- the project layer -------------------------------------------------------------------


def _project_path(project_dir, path):
    """A path as given, else relative to the project.

    Absolute, or relative to the working directory when it exists there.
    """
    if not path:
        return path
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.normpath(os.path.abspath(path))
    return os.path.normpath(os.path.join(os.path.abspath(project_dir), path))


def adopt_job_pick(project_dir, pick_path, approval_id):
    """Ship the in-job search's pick as the project's adopted spec, after the user's yes.

    Checks the pick folder (``pick.json`` and ``optimizer_result.json`` beside the spec),
    runs :func:`adopt` against the project's run log, copies the pick's bytes under
    ``work/`` as the decision's ``spec``, writes ``<proj>/decision.json`` so Step 6 ships
    it, and appends the decision entry that names the approval.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import adopt_job_pick

        out = adopt_job_pick("proj", "proj/work/job/optimize/spec.md", "approval-...-approved")
        out["decision_entry"], out["spec"]

    Args:
        project_dir: The project directory.
        pick_path: ``<out dir>/job/optimize/spec.md``.
        approval_id: The approval entry's id.

    Returns:
        ``{"decision_entry": id, "spec": relative path, "decision": decision}``.

    Raises:
        AdoptError: The pick folder is incomplete, the approval is not recorded or does
            not match the pick, or the pick is not the optimizer's.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import (
        append_entry,
        read_entries,
    )

    root = os.path.abspath(project_dir)
    pick = _project_path(project_dir, pick_path)
    folder = os.path.dirname(pick)
    names = (os.path.basename(pick), *PICK_FILES)
    check_pick_folder(pick_path, {f for f in names if os.path.isfile(os.path.join(folder, f))})
    with open(pick, "rb") as fh:
        data = fh.read()  # the bytes hashed are the bytes shipped
    with open(os.path.join(folder, "pick.json"), encoding="utf-8") as fh:
        pick_json = json.load(fh)
    try:
        with open(os.path.join(folder, "optimizer_result.json"), encoding="utf-8") as fh:
            optimizer_result = json.load(fh)
    except (OSError, ValueError):
        optimizer_result = None
    result = adopt(
        read_entries(project_dir),
        pick_bytes=data,
        pick_json=pick_json,
        optimizer_result=optimizer_result,
        approval_id=approval_id,
    )
    os.makedirs(os.path.join(root, "work"), exist_ok=True)
    with open(os.path.join(root, result.spec), "wb") as fh:
        fh.write(data)
    with open(os.path.join(root, "decision.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(result.decision, indent=2, ensure_ascii=False) + "\n")
    entry = None
    for ev in result.events:
        entry = append_entry(project_dir, dict(ev))
    return {"decision_entry": entry, "spec": result.spec, "decision": result.decision}


def ship_adopted_spec(project_dir, out_path):
    """Copy ``decision.json``'s adopted spec to ``out_path`` once its bytes still match.

    The sha256 is re-checked against the last adopt entry and the approval it names
    (:func:`check_shippable`).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import ship_adopted_spec

        ship_adopted_spec("proj", "proj/labeling_spec.md")["sha256"]

    Args:
        project_dir: The project directory.
        out_path: Where the shipped spec is written (``<proj>/labeling_spec.md``).

    Returns:
        ``{"shipped": out_path, "sha256": sha}``.

    Raises:
        AdoptError: No adopted spec, or its bytes changed since the approval.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import read_entries

    root = os.path.abspath(project_dir)
    try:
        with open(os.path.join(root, "decision.json"), encoding="utf-8") as fh:
            decision = json.load(fh)
        with open(os.path.join(root, decision["spec"]), "rb") as fh:
            data = fh.read()
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise no_adopted_spec_message(project_dir, e) from e
    sha = check_shippable(read_entries(project_dir), decision, data)
    with open(_project_path(project_dir, out_path), "wb") as fh:
        fh.write(data)
    return {"shipped": out_path, "sha256": sha}
