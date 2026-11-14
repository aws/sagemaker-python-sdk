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
"""A labeling run's record in the project's run log: the entries that write it, and reading it back.

A run is one ``job`` entry per labeling job card and one ``artifact`` entry
(``labeling_run: true``) naming them. :func:`labeling_run_record` builds those entries
from the run's manifest without writing anything: the caller appends the job entries
through the project's run-log writer, then the artifact entry with the ids it got back in
``job_entries``. :func:`read_labeling_runs` rebuilds each run in the manifest's shape from
the project's entries, for every reader; :func:`latest_labeling_runs`,
:func:`labeling_run_invocation` and :func:`labeling_run_for_output` pick from those.

The entries hold no value twice. The primary job entry carries the run's ARN and S3
paths, so the artifact entry does not; the labeling spec and labeled dataset items of
``s3_paths``, and the run's ``region`` and ``task``, repeat invocation values, so only the
invocation carries them. The reader puts each back. Invocation paths are made absolute
before they are stored, so they read back the same from any working directory; the
writer stores paths inside the project relative to it, and the reader resolves them
against the project root it is given.
"""

from __future__ import absolute_import

import os
from typing import Any, Dict, List

from pydantic import Field

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event

# The run-log entry's own keys, which are not part of the labeling record.
_ENTRY_KEYS = ("id", "kind", "step", "iteration", "status", "created_at")
# Invocation keys that hold local paths, made absolute before they are stored.
_INVOCATION_PATH_KEYS = (
    "spec",
    "input",
    "out",
    "project_dir",
    "eval",
    "demos",
    "golden_review",
    "recalibration_fit",
)
# Named `s3_paths` items whose path is an invocation value: the entries leave them out
# and the reader puts them back from the invocation.
_S3_PATHS_FROM_INVOCATION = (("Labeling spec", "spec", 0), ("Labeled dataset", "out", None))
# Run fields that repeat an invocation value of the same name.
_RUN_KEYS_FROM_INVOCATION = ("region", "task")


def _is_local_path(val):
    """True for a filesystem path: not an ``s3://`` (or other) URI and not an ARN."""
    return isinstance(val, str) and bool(val) and "://" not in val and not val.startswith("arn:")


def _abs_local(val):
    return os.path.abspath(val) if _is_local_path(val) else val


def _resolve(val, root):
    """A path as the run log stores it (relative to the project, or absolute), under ``root``."""
    if _is_local_path(val) and not os.path.isabs(val):
        return os.path.normpath(os.path.join(root, val))
    return val


def _without_invocation_paths(paths, inv):
    if not isinstance(paths, list):
        return paths
    repeats = {name: inv[key] for name, key, _ in _S3_PATHS_FROM_INVOCATION if inv.get(key)}
    return [
        item
        for item in paths
        if not (
            isinstance(item, dict)
            and item.get("name") in repeats
            and _abs_local(item.get("path")) == repeats[item["name"]]
        )
    ]


def _with_invocation_paths(paths, inv, root):
    if not isinstance(paths, list):
        return paths
    paths = list(paths)
    names = {item.get("name") for item in paths if isinstance(item, dict)}
    for name, key, index in _S3_PATHS_FROM_INVOCATION:
        if name not in names and inv.get(key):
            item = {"name": name, "path": _resolve(inv[key], root)}
            if index is None:
                paths.append(item)
            else:
                paths.insert(index, item)
    return paths


class LabelingRunRecord(OperatorResult):
    """The run-log entries that record one labeling run.

    ``jobs`` are the ``job`` entries, one per job card with an ARN, in card order;
    ``artifact`` is the run's ``artifact`` entry without ``job_entries``. The caller
    appends the jobs through the project's run-log writer, in order, and then
    :meth:`artifact_entry` of the ids it got back (the artifact with ``job_entries``
    where the command line put it). ``events`` is empty: the artifact cannot be appended
    before the job ids exist.
    """

    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    artifact: Dict[str, Any] = Field(default_factory=dict)

    def artifact_entry(self, job_entries: List[str]) -> Dict[str, Any]:
        """``artifact`` with ``job_entries`` right after ``labeling_run``, as the command line wrote it.

        Args:
            job_entries: The run-log ids the ``jobs`` entries got, in order.

        Returns:
            The artifact entry to append.
        """
        entry: Dict[str, Any] = {}
        for key, val in self.artifact.items():
            entry[key] = val
            if key == "labeling_run":
                entry["job_entries"] = list(job_entries)
        return entry


def labeling_run_record(manifest, *, step, iteration, status="labeled", **extra):
    """The run-log entries for a labeling run.

    Each job card is a ``job`` entry with status ``completed`` (the labeled output exists
    only after every job ended). The run's ``artifact`` entry carries the manifest's other
    fields and ``extra``. The first card is the primary job, whose ``arn`` is the run
    report's ``arn`` and whose ``s3_paths`` are the run report's paths, so the artifact
    entry carries neither (nor the labeled output, which is ``invocation.out``). When the
    primary card has no ARN there is no job entry to name, and the artifact keeps them.
    The labeling spec and labeled dataset items of ``s3_paths``, and the run's ``region``
    and ``task``, are the invocation's values, so only the invocation carries them.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import run_log as run_records

        record = run_records.labeling_run_record(manifest, step="auto-label", iteration=0)
        for job in record.jobs:  # the jobs first: the artifact names their ids
            run_log.append(job)
        run_log.append(record.artifact)

    Args:
        manifest: The run's manifest (``jobs`` cards, ``report_s3_paths``, ``run_report``,
            ``job_summary``, ...) plus ``invocation``.
        step: The workflow step.
        iteration: The refinement iteration.
        status: ``labeled`` for a deliverable run, ``scratch`` for a working job (a refine
            round, the ceiling pair) the cleanup lists.
        **extra: Further artifact fields (``arn`` for the DataSet version a scratch job
            registered, ``job_arn`` for the job); ``None`` values are left out.

    Returns:
        A :class:`LabelingRunRecord`.
    """
    inv = dict(manifest.get("invocation") or {})
    for key in _INVOCATION_PATH_KEYS:
        if key in inv:
            inv[key] = _abs_local(inv[key])
    jobs = manifest.get("jobs") or []
    job_entries = []
    for card in jobs:
        fields = {k: v for k, v in card.items() if k != "arn"}
        if "s3_paths" in fields:
            fields["s3_paths"] = _without_invocation_paths(fields["s3_paths"], inv)
        if card.get("arn"):
            job_entries.append(
                run_log_event("job", step, iteration, "completed", arn=card["arn"], **fields)
            )
    run = {k: v for k, v in manifest.items() if k != "jobs"}
    for key, val in extra.items():
        if val is not None:
            run[key] = val
    if inv:
        run["invocation"] = inv
    for key in _RUN_KEYS_FROM_INVOCATION:
        if key in run and key in inv and run[key] == inv[key]:
            del run[key]
    report = dict(run.get("run_report") or {})
    if "region" in report and "region" in inv and report["region"] == inv["region"]:
        del report["region"]
    if jobs and jobs[0].get("arn"):
        # The primary job entry holds these; the reader takes them from it.
        run.pop("report_s3_paths", None)
        report = {k: v for k, v in report.items() if k not in ("arn", "s3_paths")}
    else:
        if "report_s3_paths" in run:
            run["report_s3_paths"] = _without_invocation_paths(run["report_s3_paths"], inv)
        if "s3_paths" in report:
            report["s3_paths"] = _without_invocation_paths(report["s3_paths"], inv)
    if "run_report" in run:
        run["run_report"] = report
    arn = run.pop("arn", None)
    artifact = run_log_event("artifact", step, iteration, status, arn=arn, labeling_run=True, **run)
    return LabelingRunRecord(jobs=job_entries, artifact=artifact)


def read_labeling_runs(entries, root):
    """Every labeling run the project's run log records, oldest first, in the manifest's shape.

    ``jobs`` come from the run's job entries, ``report_s3_paths`` and the run report's
    ``arn`` and ``s3_paths`` from the primary job entry, ``region``, ``task`` and the
    labeling spec and labeled dataset paths from the invocation; each run also carries
    ``id`` and ``output`` (the run's ``invocation.out``).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import read_labeling_runs

        runs = read_labeling_runs(run_log_entries, "proj")
        runs[-1]["jobs"][0]["arn"]

    Args:
        entries: The project's run-log entries, oldest first.
        root: The project directory the entries' relative paths resolve against.

    Returns:
        The runs.
    """
    by_id = {e.get("id"): e for e in entries}
    runs = []
    for e in entries:
        if e.get("kind") != "artifact" or not e.get("labeling_run"):
            continue
        inv = e.get("invocation") or {}
        run = {k: v for k, v in e.items() if k not in _ENTRY_KEYS + ("labeling_run", "job_entries")}
        run["id"] = e["id"]
        run["output"] = inv.get("out")
        for key in _RUN_KEYS_FROM_INVOCATION:
            if key not in run and key in inv:
                run[key] = inv[key]
        jobs = []
        for job_id in e.get("job_entries") or []:
            job = by_id.get(job_id)
            if job:
                job = {k: v for k, v in job.items() if k not in _ENTRY_KEYS}
                if "s3_paths" in job:
                    job["s3_paths"] = _with_invocation_paths(job["s3_paths"], inv, root)
                jobs.append(job)
        run["jobs"] = jobs
        report = dict(run.get("run_report") or {})
        if jobs and "report_s3_paths" not in run:
            primary = jobs[0]
            run["report_s3_paths"] = primary.get("s3_paths")
            report.update(arn=primary.get("arn"), s3_paths=primary.get("s3_paths"))
        else:
            if "report_s3_paths" in run:
                run["report_s3_paths"] = _with_invocation_paths(run["report_s3_paths"], inv, root)
            if "s3_paths" in report:
                report["s3_paths"] = _with_invocation_paths(report["s3_paths"], inv, root)
        if "run_report" in run or report:
            if "region" not in report and "region" in inv:
                report["region"] = inv["region"]
            run["run_report"] = report
        runs.append(run)
    return runs


def latest_labeling_runs(entries, root):
    """The newest labeling run per labeled output, in the order each output was first labeled.

    A re-run that writes the same output replaces the earlier run.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import latest_labeling_runs

        for run in latest_labeling_runs(run_log_entries, "proj"):
            run["invocation"]["out"], run["run_report"]

    Args:
        entries: The project's run-log entries, oldest first.
        root: The project directory.

    Returns:
        The runs (:func:`read_labeling_runs`'s shape).
    """
    latest = {}
    for run in read_labeling_runs(entries, root):
        out = run.get("output")
        key = os.path.normpath(os.path.join(root, out)) if out else run["id"]
        latest[key] = run
    return list(latest.values())


def labeling_run_invocation(entries, root, entry_id=None):
    """The ``invocation`` a labeling run recorded, with its paths resolved against the project.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import labeling_run_invocation

        invocation = labeling_run_invocation(run_log_entries, "proj")  # the newest run's, paths resolved

    Args:
        entries: The project's run-log entries, oldest first.
        root: The project directory.
        entry_id: The run's entry id; the newest run when ``None``.

    Returns:
        The invocation dict.

    Raises:
        DataPreparerOperatorError: No such run, or the run recorded no invocation.
    """
    runs = read_labeling_runs(entries, root)
    if entry_id:
        runs = [r for r in runs if r["id"] == entry_id]
    if not runs:
        raise DataPreparerOperatorError(
            f"run-log.jsonl under {root} has no labeling run"
            + (f" with id {entry_id}" if entry_id else "")
        )
    inv = dict(runs[-1].get("invocation") or {})
    if not inv:
        raise DataPreparerOperatorError(f"labeling run {runs[-1]['id']} recorded no invocation")
    for key in _INVOCATION_PATH_KEYS:
        if key in inv:
            inv[key] = _resolve(inv[key], root)
    return inv


def labeling_run_for_output(entries, root, output_path):
    """The newest labeling run whose labeled output is ``output_path``, or ``None``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import labeling_run_for_output

        run = labeling_run_for_output(run_log_entries, "proj", "proj/work/dev_pred_iter_1.jsonl")

    Args:
        entries: The project's run-log entries, oldest first.
        root: The project directory.
        output_path: The labeled output's path.

    Returns:
        The run (:func:`read_labeling_runs`'s shape), or ``None``.
    """
    target = os.path.abspath(output_path)
    for run in reversed(read_labeling_runs(entries, root)):
        out = run.get("output")
        if out and os.path.abspath(_resolve(out, root)) == target:
            return run
    return None
