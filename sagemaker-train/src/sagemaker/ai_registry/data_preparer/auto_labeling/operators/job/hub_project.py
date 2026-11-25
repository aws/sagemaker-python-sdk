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
"""The Hub doors, as a project uses them.

:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.register` and
:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.dt_handoff` talk to
the Hub; this module writes what a project keeps of it: the ``corpus.ref.json`` pointer and
``head.jsonl`` a resolve leaves in place of a remote corpus, the provenance sidecar, and
the ``scratch`` run-log entries the hand-off cleanup
(:func:`cleanup_scratch`) deletes.
"""

from __future__ import absolute_import

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import (
    append_entry,
    read_entries,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import (
    _scratch,
    dt_handoff,
    register,
)

__all__ = [
    "resolve_input",
    "record_scratch",
    "cleanup_scratch",
    "dt_view_for_project",
    "promote_for_project",
    "register_transform_output_for_project",
    "scratch_entries",
]

_log = logging.getLogger(__name__)


def _write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def resolve_input(
    ref: str, out_path: str, region: Optional[str] = None, *, download: bool = False
) -> Dict[str, Any]:
    """Resolve a local path, an ``s3://`` URI or a DataSet ARN into the project's ``corpus/`` slot.

    A local ``ref`` is copied to ``out_path``. A remote one stays behind the API: the
    pointer ``<out stem>.ref.json`` (:func:`~...job.register.corpus_ref_document`) and
    ``head.jsonl`` (the first rows, the shape) are written beside ``out_path`` instead of
    the body. With ``download`` the object is downloaded to ``out_path``: for a
    client-readable seed or sample, never the corpus (a ``client-read:`` notice is logged).
    Every resolve writes the provenance sidecar ``<out_path>.provenance.json``, which the
    Hub registration reads.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import resolve_input

        out = resolve_input("s3://bucket/corpus.jsonl", "proj/corpus/corpus.jsonl")
        out["ref"]  # 'proj/corpus/corpus.ref.json'

    Args:
        ref: The local path, ``s3://`` URI or DataSet ARN.
        out_path: The local file the pipeline reads (or would read).
        region: The region; the session's (else ``us-west-2``) when ``None``.
        download: Download a remote ``ref`` to ``out_path``.

    Returns:
        For a remote ref without download: ``{"ref", "downloaded": False, "provenance",
        "head_sample", **pointer}``; otherwise ``{"out", "downloaded": True, "provenance",
        **provenance}``.
    """
    region = region or register._default_region()
    prov = register.resolve_dataset_ref(ref, out_path, region, download=download)
    if download and prov.get("resolved_s3_uri"):
        logging.getLogger(_rows.CLIENT_READ_LOGGER).info(
            "client-read: resolve_input downloaded %s to %s (a seed or sample, never the corpus)",
            prov["resolved_s3_uri"],
            out_path,
        )
    sidecar = out_path + ".provenance.json"
    _write_json(sidecar, prov)
    remote = bool(prov.get("resolved_s3_uri"))
    if not download and remote:
        ref_path = str(Path(out_path).with_suffix(".ref.json"))
        ref_doc = register.corpus_ref_document(prov)
        _write_json(ref_path, ref_doc)
        head_written = None
        try:
            head_path = str(Path(ref_path).with_name("head.jsonl"))
            _write_jsonl(head_path, _rows.head_rows(ref_doc["source_s3_uri"], region=region))
            head_written = head_path
        except Exception as e:  # noqa: BLE001 - the head sample is best-effort
            _log.warning(
                "wrote the corpus.ref.json pointer but could not write head.jsonl (%s: %s); "
                "the labeling run will create it.",
                type(e).__name__,
                e,
            )
        return {
            "ref": ref_path,
            "downloaded": False,
            "provenance": sidecar,
            "head_sample": head_written,
            **ref_doc,
        }
    return {"out": out_path, "downloaded": True, "provenance": sidecar, **prov}


def record_scratch(project_dir: Optional[str], arn: Optional[str], stage: str) -> Optional[str]:
    """Record a working DataSet version as a ``scratch`` run-log entry.

    A DT view, or a DT output under it, so :func:`cleanup_scratch` removes it at the
    hand-off. Nothing is recorded without a project or an ARN.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import record_scratch

        record_scratch("proj", view["dataset_arn"], "dt-view")

    Args:
        project_dir: The project whose run log records it.
        arn: The versioned DataSet ARN.
        stage: ``dt-view`` or ``dt-output``.

    Returns:
        The entry id, or ``None`` when nothing was recorded.
    """
    if not project_dir or not arn:
        return None
    return append_entry(project_dir, _scratch.scratch_event(arn, stage))


def cleanup_scratch(project_dir: str, *, delete: bool = False) -> Dict[str, Any]:
    """List, and with ``delete`` delete, the scratch DataSet versions the project recorded.

    Each version (never a name) is deleted in its ARN's region; the catalog entry goes,
    the S3 files stay. The deletions are run-log entries.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import cleanup_scratch

        cleanup_scratch("proj")["scratch"]            # list
        cleanup_scratch("proj", delete=True)["deleted"]  # after the customer's yes

    Args:
        project_dir: The project directory (its ``run-log.jsonl``).
        delete: Delete each version; otherwise only list them.

    Returns:
        :meth:`~...job._scratch.ScratchResult.to_json`.
    """
    result = _scratch.scratch_cleanup(read_entries(project_dir), delete=delete, project=project_dir)
    for ev in result.events:
        append_entry(project_dir, dict(ev))
    return result.to_json()


def dt_view_for_project(
    project_dir: Optional[str],
    *,
    arn: Optional[str] = None,
    s3_uri: Optional[str] = None,
    name: Optional[str] = None,
    hub_name_override: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """The first input of a dataset-transformation job, recorded as scratch.

    :func:`~...job.dt_handoff.dt_view`, then :func:`record_scratch` of the view.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import dt_view_for_project

        view = dt_view_for_project("proj", arn=labeled["dataset_arn"])

    Args:
        project_dir: The project whose run log records the view.
        arn: The DataSet the DT job would take.
        s3_uri: Instead of ``arn``: an ``s3://`` object that is not a DataSet; needs ``name``.
        name: With ``s3_uri``, the dataset's name.
        hub_name_override: A hub other than the account's AI Registry hub.
        region: The region.

    Returns:
        :func:`~...job.dt_handoff.dt_view`'s result.
    """
    result = dt_handoff.dt_view(
        arn=arn, s3_uri=s3_uri, name=name, hub_name_override=hub_name_override, region=region
    )
    record_scratch(project_dir, result["dataset_arn"], "dt-view")
    return result


def promote_for_project(
    project_dir: Optional[str], *, from_arn: str, to_arn: str, region: Optional[str] = None
) -> Dict[str, Any]:
    """Promote the overlay's output as the labeled DataSet's next version, recording the output as scratch.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import promote_for_project

        promote_for_project("proj", from_arn=out_arn, to_arn=labeled["dataset_arn"])

    Args:
        project_dir: The project whose run log records the output version.
        from_arn: The overlay job's ``OutputDatasetArns`` entry (a version of the view).
        to_arn: The labeled DataSet version the view was taken of.
        region: The region; the ARN's own when ``None``.

    Returns:
        :func:`~...job.dt_handoff.promote_transform_output`'s result.
    """
    result = dt_handoff.promote_transform_output(from_arn=from_arn, to_arn=to_arn, region=region)
    record_scratch(project_dir, result["transform_output_arn"], "dt-output")
    return result


def register_transform_output_for_project(
    project_dir: Optional[str],
    *,
    from_arn: str,
    name: str,
    source_arn: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Register a DT output that landed on a dt-view under a new name, recording the output as scratch.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import (
            register_transform_output_for_project,
        )

        register_transform_output_for_project("proj", from_arn=out_arn, name="tickets-split-dev")

    Args:
        project_dir: The project whose run log records the output version.
        from_arn: The DataSet ARN the transform job returned.
        name: The new DataSet name (refused if it exists).
        source_arn: The DataSet the job was run on, for the description.
        region: The region.

    Returns:
        :func:`~...job.dt_handoff.register_transform_output`'s result.
    """
    result = dt_handoff.register_transform_output(
        from_arn=from_arn, name=name, source_arn=source_arn, region=region
    )
    if register.parse_dataset_arn(from_arn)[1].endswith(dt_handoff.DT_VIEW_SUFFIX):
        record_scratch(project_dir, from_arn, "dt-output")
    return result


def scratch_entries(project_dir: str) -> List[Dict[str, Any]]:
    """The scratch entries the project's run log records.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import scratch_entries

        [e["arn"] for e in scratch_entries("proj")]

    Args:
        project_dir: The project directory (its ``run-log.jsonl``).

    Returns:
        :func:`~...job._scratch.scratch_entries` over the run log.
    """
    return _scratch.scratch_entries(read_entries(project_dir))
