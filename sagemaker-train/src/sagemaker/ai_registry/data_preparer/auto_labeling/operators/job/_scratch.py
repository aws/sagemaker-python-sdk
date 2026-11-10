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
"""The scratch DataSet versions an auto-labeling project registers along the way.

:func:`scratch_event` is the run-log entry that records one, :func:`scratch_entries` the
entries that list them, and :func:`scratch_cleanup` the cleanup that deletes each version
(never a name) while keeping another project's versions of a shared view. The cleanup
reads the project's run-log entries, which the caller passes in.
"""

from __future__ import absolute_import

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import Field

from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.dt_handoff import (
    DT_VIEW_SUFFIX,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register import (
    _not_found,
    parse_dataset_arn,
)

_log = logging.getLogger(__name__)


def scratch_event(arn, stage):
    """The run-log entry that records a working DataSet version as scratch.

    A ``dt-view`` entry stands for its whole view: the cleanup takes every version of it.

    Args:
        arn: The DataSet version's ARN.
        stage: ``dt-view`` or ``dt-output``.

    Returns:
        The entry dict for the project's run-log writer (an ``artifact`` with status
        ``scratch``).
    """
    return run_log_event(
        "artifact",
        "auto-label",
        0,
        "scratch",
        arn=arn,
        labeling_run=True,
        invocation={"stage": stage},
    )


def scratch_entries(entries):
    """The project's scratch artifacts, from its run-log entries.

    Args:
        entries: The project's run-log entries, oldest first.

    Returns:
        ``[{"id", "arn", "job_arn", "stage", "run_id", "dataset_name", "created_at"}]`` for
        every ``artifact`` entry with ``labeling_run`` and status ``scratch``, in order.
    """
    out = []
    for e in entries:
        if e.get("kind") == "artifact" and e.get("labeling_run") and e.get("status") == "scratch":
            inv = e.get("invocation") or {}
            out.append(
                {
                    "id": e.get("id"),
                    "arn": e.get("arn"),
                    "job_arn": e.get("job_arn"),
                    "stage": inv.get("stage"),
                    "run_id": inv.get("run_id"),
                    "dataset_name": (e.get("request") or {}).get("dataset_name"),
                    "created_at": e.get("created_at"),
                }
            )
    return out


def _version_key(arn):
    return [int(p) if p.isdigit() else 0 for p in arn.rsplit("/", 1)[-1].split(".")]


def _job_first_input(sm, job_arn):
    """The first ``InputDataConfig`` DatasetArn of the DT job ``job_arn``, or ``None``."""
    resp = sm.describe_job(JobName=job_arn.rsplit("/", 1)[-1], JobCategory="DataTransformation")
    doc = resp.get("JobConfigDocument") or {}
    if isinstance(doc, str):
        doc = json.loads(doc or "{}")
    inputs = doc.get("InputDataConfig") or resp.get("InputDataConfig") or []
    return (inputs[0] or {}).get("DatasetArn") if inputs else None


def _view_versions(arn):
    """The versions of a dt-view entry that belong to the view version ``arn``.

    ``arn`` itself and each DT output whose job took ``arn`` as its first input (through
    ``DescribeJob``), newest first. A view name is shared by every project that takes a
    view of the same DataSet, so another project's view version and its jobs' outputs are
    kept, with the reason. ``([], [])`` when the entry is gone; only an entry named
    ``<name>-dt-input`` is listed.

    Returns:
        ``(ours, kept)``: the ARNs to delete, and ``[{"arn", "reason"}]`` for the rest.
    """
    from botocore.exceptions import ClientError

    hub, name, _v, region = parse_dataset_arn(arn)
    if not name.endswith(DT_VIEW_SUFFIX):
        return [], []
    sm = register._sm_client(region)
    versions, token = [], None
    try:
        while True:
            resp = sm.list_hub_content_versions(
                HubName=hub,
                HubContentType="DataSet",
                HubContentName=name,
                MaxResults=100,
                **({"NextToken": token} if token else {}),
            )
            versions.extend(
                v for v in resp.get("HubContentSummaries") or [] if v.get("HubContentArn")
            )
            token = resp.get("NextToken")
            if not token:
                break
    except ClientError as e:
        if _not_found(e):
            return [], []
        raise
    ours, kept = [arn], []
    for v in versions:
        other = v["HubContentArn"]
        if other == arn:
            continue
        try:
            doc = json.loads(
                sm.describe_hub_content(
                    HubName=hub,
                    HubContentType="DataSet",
                    HubContentName=name,
                    HubContentVersion=other.rsplit("/", 1)[-1],
                ).get("HubContentDocument")
                or "{}"
            )
            job_arn = doc.get("TransformationJobArn")
            if not job_arn:
                kept.append({"arn": other, "reason": "another view version"})
            elif _job_first_input(sm, job_arn) == arn:
                ours.append(other)
            else:
                kept.append({"arn": other, "reason": "the output of a job on another view version"})
        except Exception as ex:  # noqa: BLE001 - keep what cannot be tied to this view
            kept.append({"arn": other, "reason": f"not checked ({type(ex).__name__}: {ex})"})
    return sorted(ours, key=_version_key, reverse=True), kept


class ScratchResult(OperatorResult):
    """What :func:`scratch_cleanup` found and, with ``delete``, did.

    ``events`` carries the run-log entry recording the deletions, when any.
    """

    project: Optional[str] = None
    scratch: List[Dict[str, Any]] = Field(default_factory=list)
    deleted: List[str] = Field(default_factory=list)
    not_found: List[Dict[str, Any]] = Field(default_factory=list)
    kept: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)


def scratch_cleanup(
    entries, *, delete=False, project=None
):  # pylint: disable=too-many-locals,too-many-branches
    """List, and with ``delete`` remove, the scratch DataSet versions a project registered.

    Each version is deleted in its ARN's region, never a name. A ``dt-view`` entry goes
    with its jobs' outputs; another project's view version on the same name is kept. A
    version already gone is listed under ``not_found``, never counted as deleted.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import _scratch

        listed = _scratch.scratch_cleanup(run_log_entries)            # list only
        cleaned = _scratch.scratch_cleanup(run_log_entries, delete=True)
        run_log.extend(cleaned.events)

    Args:
        entries: The project's run-log entries (:func:`scratch_entries` picks the scratch
            ones).
        delete: Delete the versions; list them only when ``False``.
        project: The project's path, echoed in the result.

    Returns:
        A :class:`ScratchResult`; with deletions, one ``events`` entry (an ``artifact``
        with status ``deleted`` and ``scratch_cleanup``) for the caller's run-log writer.
    """
    scratch = scratch_entries(entries)
    result = ScratchResult(project=project, scratch=scratch)
    if not delete:
        return result
    targets: List[str] = []
    seen = set()
    for e in scratch:
        if not e.get("arn"):
            result.errors.append({"id": e["id"], "error": "no DataSet ARN recorded"})
            continue
        try:
            arns, kept = _view_versions(e["arn"]) if e.get("stage") == "dt-view" else ([], [])
        except Exception as ex:  # noqa: BLE001 - report the view, delete the rest
            result.errors.append({"arn": e["arn"], "error": f"{type(ex).__name__}: {ex}"})
            continue
        result.kept.extend(kept)
        for arn in arns or [e["arn"]]:
            if arn not in seen:
                seen.add(arn)
                targets.append(arn)
    # A version one view kept may be this project's own (a second view of the same
    # DataSet, or its outputs): another entry deletes it, so it is not listed as kept.
    kept_seen: set = set()
    result.kept = [
        k
        for k in result.kept
        if k["arn"] not in seen and not (k["arn"] in kept_seen or kept_seen.add(k["arn"]))
    ]
    for arn in targets:
        try:
            # region=None: delete_hub_content_version uses the ARN's own region
            register.delete_hub_content_version(arn)
            result.deleted.append(arn)
        except Exception as ex:  # noqa: BLE001 - report every version, delete the rest
            msg = f"{type(ex).__name__}: {ex}"
            if "ResourceNotFound" in msg or "does not exist" in msg:
                result.not_found.append({"arn": arn, "error": msg})
            else:
                result.errors.append({"arn": arn, "error": msg})
    if result.deleted:
        result.events.append(
            run_log_event(
                "artifact",
                "auto-label",
                0,
                "deleted",
                scratch_cleanup=True,
                deleted=list(result.deleted),
            )
        )
    return result
