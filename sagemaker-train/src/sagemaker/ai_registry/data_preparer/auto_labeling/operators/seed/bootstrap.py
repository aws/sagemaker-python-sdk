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
"""The cold-start seed's files: the sample people label, and their labels on the corpus.

:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.cold_start` and
:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels` decide;
this module reads and writes the files around them:

* :func:`cold_start_sample` checks the target, the task and the row limit and draws the
  sample; :func:`write_sample` reads the Split job's random sample (a local file or an
  ``s3://`` URI) and writes the rows people label (the id and the input fields only);
* :func:`write_merged_labels` puts the people's labels from the seed on a local labeled
  file (``labels.jsonl``): the corpus itself when the sample is the whole corpus;
* :func:`write_overlay` writes the people's labels as a dataset-transformation overlay
  (``overlay.jsonl``) and its hand-off record (``overlay.handoff.json``) for the corpus
  run's labeled DataSet, whose labels stay behind the API.

With a review file, the merge and the overlay drop the rows people labeled already from
the job's review set and its side file, and score the spot-check rows among them
(``review.known_spot_check.json``). :func:`review_rows_from_side_file` and
:func:`review_path_of` are the side-file helpers the human-in-the-loop fold shares.

When the labels come back from people, :func:`fold_human_labels` merges them with the
rows they reviewed into seed rows and scores the spot-check, :func:`append_to_seed`
appends those rows to the seed in one atomic step, :func:`check_fold` proves an export
was folded (and merged onto the labels), and :func:`spot_check_message` states the
verdict. Getting the labels from people is not here: these start from the export. Every
refusal is a :class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`;
:class:`SeedMergeRefused` is the one that means the seed does not name this corpus's rows.
"""

from __future__ import absolute_import

import json
import logging
import os
import tempfile
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    LABELED_BY_FIELD,
    MODEL_OUTPUT_FIELD,
    MODEL_OUTPUT_SUFFIX,
    REVIEW_REASON_FIELD,
    join_model_output,
    labeled_by_people,
    model_output_path,
    review_reason_of,
    unblind_row,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import (
    GT_FIELD_FOR_PRED,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import (
    read_input_rows,
    read_jsonl,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed._bounds import (
    MIN_CALIBRATION_TARGET,
    TARGET_FLOOR_MESSAGE,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.cold_start import (
    certificate_warning,
    cold_start_sample_size,
    draw_sample,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels import (
    NO_MODEL_ANSWER,
    NOT_SCORED,
    OVERLAY_GOLD_KEY,
    build_overlay,
    check_seed_append,
    drop_people_rows_from_review,
    fill_missing_ids,
    known_spot_check_path,
    merge_seed_labels,
    overlay_handoff,
    overlay_seed_rows,
    review_verdict,
    score_known_spot_check,
    seed_rows_for_overlay,
    spot_check_confirms,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export import (
    GA_SPLIT_FIELD,
    GA_SPOT_CHECK_SPLIT,
    HUMAN_PROVENANCE,
    SPOT_CHECKED,
    class_key,
    decided_by_id,
    flatten_export,
    id_key,
    provenance_for_review,
    same_label,
    span_fields,
)

__all__ = [
    "cold_start_sample",
    "write_sample",
    "write_merged_labels",
    "write_overlay",
    "review_rows_from_side_file",
    "review_path_of",
    "fold_human_labels",
    "append_to_seed",
    "check_fold",
    "spot_check_message",
    "SeedMergeRefused",
]

_log = logging.getLogger(__name__)

# The client-read notice's name for the sample read, as the skill's command printed it.
_SAMPLE_PRIMITIVE = "bootstrap_seed sample"

# The job's routing tag per split (the container finalizer's SKILL_REVIEW_REASONS).
_JOB_REVIEW_REASONS = {"low_confidence": "low confidence", "spot_check": "calibration"}

# Keys only a labeling run (or an earlier merge) writes on a row: a failed row
# `_failed`, a jury or judge `confidence_score`, a merge `_model_output` and
# `_labeled_by`. A raw corpus row has none of them.
_RUN_MARKERS = (
    "confidence_score",
    "_failed",
    "_failed_reason",
    MODEL_OUTPUT_FIELD,
    LABELED_BY_FIELD,
)
# The labeling job gives every row it returns `reasoning` (a program output field), and on
# a run with no jury (summarization, ordinal, llm_judge, or qa without jury models) a row
# that did not fail carries nothing else. A raw corpus can carry a `reasoning` column too,
# so it marks a run row only when the prediction field is not given.
_RUN_REASONING_MARKER = "reasoning"


class SeedMergeRefused(DataPreparerOperatorError):
    """The seed's human rows do not all name a row of the labeled file.

    A seed row people labeled has no labeled row with its id, or that row differs from
    the seed row on an input field: the seed was drawn from another corpus, or the corpus
    changed since. Nothing is written.
    """


# --- files -------------------------------------------------------------------------------


def _read_errors() -> Tuple[type, ...]:
    """The errors that mean an input file could not be read.

    OSError and ValueError, plus botocore's ClientError (access denied, no such key) and
    BotoCoreError (no credentials, endpoint unreachable) for an ``s3://`` input, which are
    not OSError subclasses. botocore is only imported when present.
    """
    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        return (OSError, ValueError)
    return (OSError, ValueError, ClientError, BotoCoreError)


def _read_rows(path_or_uri: str, primitive: Optional[str] = None) -> List[Any]:
    """:func:`read_input_rows`, with a read failure as a DataPreparerOperatorError."""
    try:
        if primitive is None:
            return read_input_rows(path_or_uri)
        return read_input_rows(path_or_uri, primitive=primitive)
    except DataPreparerOperatorError:
        raise
    except _read_errors() as e:
        raise DataPreparerOperatorError(str(e)) from e


def _write_jsonl(path: str, rows: Iterable[Any]) -> None:
    """Write rows as JSONL (one compact object per line, unicode preserved)."""
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _write_text_atomic(path: str, text: str) -> None:
    """Write via a temp file beside ``path`` and ``os.replace``.

    A killed or failed write leaves the old file whole. The file keeps its old permission
    bits (or the umask's, for a new file).
    """
    if os.path.exists(path):
        mode = os.stat(path).st_mode & 0o7777
    else:  # reading the umask sets it, so only a new file does
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except BaseException:
        try:  # a failed cleanup never hides the error that stopped the write
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_jsonl_atomic(path: str, rows: Iterable[Any]) -> None:
    """Write rows as JSONL with :func:`_write_text_atomic`."""
    _write_text_atomic(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


# --- checks ------------------------------------------------------------------------------


def _task_names() -> set:
    """The task values the labeling job accepts (``job.request.LABELING_TASKS``)."""
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
        LABELING_TASKS,
    )

    return set(LABELING_TASKS)


def _check_target(v: Any) -> float:
    """An accuracy target: a number strictly between 0 and 1, and at least 0.85.

    ``MIN_CALIBRATION_TARGET`` (0.85) is the lowest target a certificate and its
    spot-check are defined for.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise DataPreparerOperatorError(f"--target must be a number, got {v!r}")
    if not 0.0 < x < 1.0:
        raise DataPreparerOperatorError(f"--target must be strictly between 0 and 1, got {v}")
    if x < MIN_CALIBRATION_TARGET:
        raise DataPreparerOperatorError(f"{TARGET_FLOOR_MESSAGE} (got {v})")
    return x


def _check_task(task: Optional[str]) -> None:
    if task is not None and task not in _task_names():
        raise DataPreparerOperatorError(
            f"--task {task!r} is not a labeling task; one of {', '.join(sorted(_task_names()))}"
        )


def _text_fields(text_fields: Union[str, Sequence[str]]) -> List[str]:
    """The input fields to keep: a comma-separated string or a list, blanks dropped."""
    parts = text_fields.split(",") if isinstance(text_fields, str) else list(text_fields)
    fields = [f.strip() for f in parts if f.strip()]
    if not fields:
        raise DataPreparerOperatorError("--text-fields names no field")
    return fields


def _require_row_limit(n_rows: int) -> None:
    """Refuse rows above the labeling job's row limit (``job.request.MAX_LABELING_ROWS``).

    The container enforces the limit at load, so people do not label a sample for a
    corpus run that would be refused.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
        MAX_LABELING_ROWS,
    )

    if n_rows > MAX_LABELING_ROWS:
        raise DataPreparerOperatorError(
            f"the file has {n_rows:,} rows; a labeling job labels at most "
            f"{MAX_LABELING_ROWS:,}. Split the corpus into files of at most that many rows "
            "and run each as its own project, or sample it down, before drawing the "
            "cold-start sample"
        )


def _require_pred_field(
    rows: List[Any], pred_field: str, gt_field: str, pred_field_given: bool = False
) -> None:
    """Refuse a labeled file none of whose rows has ``pred_field`` but whose run rows carry another task's.

    A summarization run merged with the defaults would leave the model's summary in place
    under a person's provenance. Only a labeling run's rows (a ``_RUN_MARKERS`` key, or
    ``_RUN_REASONING_MARKER`` unless ``pred_field_given``) hold a model prediction: on the
    raw corpus (the whole-corpus merge) a ``summary`` or ``answer`` column is an input
    field, so it is not checked.
    """
    if any(isinstance(r, dict) and pred_field in r for r in rows):
        return
    markers = _RUN_MARKERS + (() if pred_field_given else (_RUN_REASONING_MARKER,))
    run_rows = [r for r in rows if isinstance(r, dict) and any(k in r for k in markers)]
    others = sorted({k for r in run_rows for k in ("label", *GT_FIELD_FOR_PRED) if k in r})
    if others:
        raise DataPreparerOperatorError(
            f"no row of --labels has --pred-field {pred_field!r}; its labeling-run "
            f"rows carry {', '.join(others)}. Pass the task's --pred-field and --gt-field "
            f"(full-loop Step 7b lists them; --gt-field is now {gt_field!r})"
            + (
                ""
                if pred_field_given
                else f"; if --labels is the raw corpus, pass --pred-field {pred_field!r}"
            )
        )


# --- the review set ----------------------------------------------------------------------


def review_rows_from_side_file(side_rows: Iterable[Any], id_field: str = "id") -> List[Dict]:
    """The review rows for the fold from the job's side file alone.

    The side file holds ``{id, pred_field, model_output}`` per review row. Each review row
    gets its id, its ``split`` (``routed: true`` in the model output is ``low_confidence``,
    false is ``spot_check``) and its routing tag, so ``split`` never comes from the export.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import review_rows_from_side_file

        rows = review_rows_from_side_file(read_jsonl("proj/review.model_output.jsonl"))
        # [{"id": "r7", "split": "spot_check", "_review_reason": "calibration"}, ...]

    Args:
        side_rows: The side file's records.
        id_field: The id field the review rows carry.

    Returns:
        One ``{id_field, "split", "_review_reason"}`` row per record, in file order.

    Raises:
        DataPreparerOperatorError: A record has no id, a repeated id, or no boolean
            ``routed`` in its model output.
    """
    rows, seen = [], set()
    for rec in side_rows:
        if not isinstance(rec, dict):
            continue
        key = id_key(rec.get("id"))
        if key is None or key in seen:
            raise DataPreparerOperatorError(
                f"a side-file record has a null or repeated id ({rec.get('id')!r})"
            )
        seen.add(key)
        answer = rec.get("model_output") if isinstance(rec.get("model_output"), dict) else {}
        if not isinstance(answer.get("routed"), bool):
            raise DataPreparerOperatorError(
                f"the record for id {rec.get('id')!r} has no `routed` in its "
                "model_output, so its split cannot be told; pass --sample "
                "<review.jsonl> instead"
            )
        split = "low_confidence" if answer["routed"] else "spot_check"
        rows.append(
            {
                id_field: rec.get("id"),
                "split": split,
                REVIEW_REASON_FIELD: _JOB_REVIEW_REASONS[split],
            }
        )
    return rows


def review_path_of(side_path: str) -> str:
    """The review file a side file goes with (``review.model_output.jsonl`` to ``review.jsonl``).

    That is where the known spot-check verdicts sit
    (``review.known_spot_check.json``). Any other path is returned as it is.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import review_path_of

        review_path_of("proj/review.model_output.jsonl")  # 'proj/review.jsonl'

    Args:
        side_path: The side file's path.

    Returns:
        The review file's path.
    """
    return (
        side_path[: -len(MODEL_OUTPUT_SUFFIX)] + ".jsonl"
        if side_path.endswith(MODEL_OUTPUT_SUFFIX)
        else side_path
    )


def _drop_and_score_review(
    review_path: str, people_rows: List[Any], id_field: str, pred_field: str
) -> Dict[str, Any]:
    """Drop the rows people labeled already from a review file, and score its spot-check rows.

    ``people_rows`` that carry the labeled-by-people marker leave the review file and its
    side file, and the spot-check rows among them are scored with the person's label
    (``review.known_spot_check.json``). A corpus run downloads the side file only:
    ``review_path`` may name it (``review.model_output.jsonl``), or a ``review.jsonl`` that
    is not there beside it. Then the review rows' ids and split come from the side file
    (:func:`review_rows_from_side_file`) and only the side file is rewritten.

    Returns:
        The report fields.

    Raises:
        DataPreparerOperatorError: The review file cannot be read.
    """
    side_only = review_path.endswith(MODEL_OUTPUT_SUFFIX) or (
        not os.path.exists(review_path) and os.path.exists(model_output_path(review_path))
    )
    if review_path.endswith(MODEL_OUTPUT_SUFFIX):
        review_path = review_path_of(review_path)
    side_path = model_output_path(review_path)
    try:
        side_rows = read_jsonl(side_path) if os.path.exists(side_path) else []
        review_rows = (
            review_rows_from_side_file(side_rows, id_field)
            if side_only
            else read_jsonl(review_path)
        )
    except (OSError, ValueError) as e:
        raise DataPreparerOperatorError(f"cannot read --review: {e}") from e
    people_ids = [r.get(id_field) for r in people_rows if labeled_by_people(r)]
    all_review, all_side = review_rows, side_rows
    review_rows, side_rows, dropped = drop_people_rows_from_review(
        review_rows, side_rows, people_ids, id_field=id_field
    )
    dropped_set = set(dropped)
    known = score_known_spot_check(
        [r for r in all_review if isinstance(r, dict) and id_key(r.get(id_field)) in dropped_set],
        all_side,
        people_rows,
        id_field=id_field,
        pred_field=pred_field,
    )
    report = {}
    if known:
        with open(known_spot_check_path(review_path), "w", encoding="utf-8") as fh:
            json.dump(known, fh, indent=2)
            fh.write("\n")
        report["spot_check_known"] = {k: known[k] for k in ("rows", "corrected")}
    report["review_rows_dropped"] = len(dropped)
    report["review_rows"] = len(review_rows)
    if not side_only:
        _write_jsonl_atomic(review_path, review_rows)
    if os.path.exists(side_path):
        _write_jsonl_atomic(side_path, side_rows)
    if dropped:
        _log.warning(
            "%d review row(s) people labeled already dropped from --review and its side file.",
            len(dropped),
        )
    return report


# --- sample ------------------------------------------------------------------------------


def cold_start_sample(
    rows: List[Any],
    text_fields: Union[str, Sequence[str]],
    *,
    target: float = 0.90,
    task: Optional[str] = None,
    seed: Any = "0",
    id_field: str = "id",
    corpus_rows: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """The cold-start sample of the Split job's random rows, with its report.

    200 rows, 400 when ``target`` is 0.95 or higher (200 at any target for the tasks that
    ship without an accuracy promise: ordinal, llm_judge, summarization), or every row of
    a smaller input, drawn by :func:`draw_sample`. Each sample row keeps only ``id_field``
    and the ``text_fields``. The report adds ``target``, ``task`` when given, and
    ``warning`` when the target is above what the sample can certify even with no errors
    (logged too); a sample that covers the corpus (``corpus_rows``) gets no warning, since
    people label every row and nothing needs a certificate.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import cold_start_sample

        sample, report = cold_start_sample(rows, "text", target=0.95, task="classification")
        report["sample_rows"], report.get("warning")

    Args:
        rows: The Split job's random sample of the corpus, never the corpus itself.
        text_fields: The input fields to keep: a list, or a comma-separated string.
        target: The accuracy target, which sets the size.
        task: The detected task.
        seed: The draw's seed; the same rows and seed draw the same sample.
        id_field: The id field.
        corpus_rows: The rows in the corpus the Split job sampled from, when known.

    Returns:
        ``(sample_rows, report)``.

    Raises:
        DataPreparerOperatorError: No text field, a target outside (0, 1) or below 0.85,
            a task the labeling job does not accept, more rows than a labeling job takes,
            or a refusal of :func:`draw_sample`.
    """
    fields = _text_fields(text_fields)
    target = _check_target(target)
    _check_task(task)
    _require_row_limit(len(rows))
    sample, report = draw_sample(
        rows,
        cold_start_sample_size(target, len(rows), task),
        seed,
        id_field=id_field,
        text_fields=fields,
        corpus_total=corpus_rows,
    )
    report["target"] = target
    if task:
        report["task"] = task
    # A corpus sampled whole is labeled by people throughout: no model label ships and
    # there is no certificate for the warning to be about.
    warning = None if report["whole_corpus"] else certificate_warning(len(sample), target, task)
    if warning:
        report["warning"] = warning
        _log.warning("sample: %s", warning)
    return sample, report


def write_sample(
    input_path: str,
    out_path: str,
    text_fields: Union[str, Sequence[str]],
    *,
    target: float = 0.90,
    task: Optional[str] = None,
    seed: Any = "0",
    id_field: str = "id",
    corpus_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Write the cold-start sample people label (``sample.jsonl``).

    Reads the Split job's random sample of the corpus (a local JSONL or an ``s3://`` URI;
    a ``client-read:`` notice) and writes :func:`cold_start_sample`'s rows to
    ``out_path``. Nothing is written on a refusal.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import write_sample

        report = write_sample("proj/sample_split.jsonl", "proj/sample.jsonl", ["text"], target=0.9)

    Args:
        input_path: The Split job's random sample (local path, ``s3://`` URI or
            ``*.ref.json``).
        out_path: Where to write the sample.
        text_fields: The input fields to keep: a list, or a comma-separated string.
        target: The accuracy target, which sets the size.
        task: The detected task.
        seed: The draw's seed.
        id_field: The id field.
        corpus_rows: The rows in the corpus the Split job sampled from, when known.

    Returns:
        The report :func:`cold_start_sample` returns.

    Raises:
        DataPreparerOperatorError: The input cannot be read (a missing file, a line that
            is not JSON, an S3 error, over the client-side read limit), or a refusal of
            :func:`cold_start_sample`.
    """
    _text_fields(text_fields)
    _check_target(target)
    _check_task(task)
    rows = _read_rows(input_path, primitive=_SAMPLE_PRIMITIVE)
    sample, report = cold_start_sample(
        rows,
        text_fields,
        target=target,
        task=task,
        seed=seed,
        id_field=id_field,
        corpus_rows=corpus_rows,
    )
    _write_jsonl(out_path, sample)
    return report


# --- merge -------------------------------------------------------------------------------


def write_merged_labels(
    labels_path: str,
    seed_path: str,
    out_path: str,
    *,
    id_field: str = "id",
    gt_field: str = "gt_label",
    pred_field: Optional[str] = None,
    review_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Put the people's labels from the seed on a local labeled file, and write it (``labels.jsonl``).

    ``labels_path`` is a locally labeled file, or the corpus itself when the sample is the
    whole corpus; never the corpus run's output, which stays behind the API (use
    :func:`write_overlay`). An id-less row gets its position as its id, as the labeling
    job gives it. :func:`merge_seed_labels` moves each matched row's model answer under
    ``_model_output``, sets ``pred_field`` to the person's label and copies the seed
    row's provenance and routing tag. With ``review_path`` the rows people labeled are
    dropped from the review file and its side file. The out file is written atomically,
    last; nothing is written on a refusal.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import write_merged_labels

        report = write_merged_labels("proj/corpus.jsonl", "proj/seed.jsonl", "proj/labels.jsonl")
        report["merged"], report["label_changed"]

    Args:
        labels_path: The labeled file (local path, ``s3://`` URI or ``*.ref.json``).
        seed_path: The seed whose rows people labeled.
        out_path: Where to write the merged rows (``labels_path`` is fine).
        id_field: The id field.
        gt_field: The seed's gold field.
        pred_field: The labeled file's label field (default ``label``; ``score`` for
            ordinal, ``entities`` for ner). Name it on a whole-corpus merge onto a raw
            corpus with a ``reasoning`` column.
        review_path: The corpus job's ``review.jsonl`` (or its side file).

    Returns:
        :func:`merge_seed_labels`'s report, with ``ids_filled`` when ids were filled and
        the review fields with ``review_path``.

    Raises:
        SeedMergeRefused: A seed row people labeled has no labeled row with its id, or
            differs from it on an input field.
        DataPreparerOperatorError: The labeled file or the review file cannot be read,
            no labeled row has ``pred_field`` while its run rows carry another task's, or
            the seed has no human row with a gold value.
        OSError: The seed cannot be read.
    """
    labels = _read_rows(labels_path)
    filled = fill_missing_ids(labels, id_field)
    pred_field_given = pred_field is not None
    if not pred_field_given:
        pred_field = "label"
    _require_pred_field(labels, pred_field, gt_field, pred_field_given)
    rows, report = merge_seed_labels(
        labels,
        read_jsonl(seed_path),
        id_field=id_field,
        gt_field=gt_field,
        pred_field=pred_field,
    )
    if filled:
        report["ids_filled"] = filled
    if not report["seed_human_rows"]:
        raise DataPreparerOperatorError(
            "no seed row has a human provenance (human_verified or "
            f"spot_checked, or _bootstrap hil) and a {gt_field!r} value, so "
            "there is no label to merge"
        )
    missing = report["missing_from_labels"]
    if missing:
        shown = ", ".join(repr(k) for k in missing[:5])
        raise SeedMergeRefused(
            f"{len(missing)} seed row(s) people labeled have "
            f"no row in --labels with their --id-field {id_field!r} ({shown}). "
            "The cold-start sample is drawn from this corpus, so every one should "
            "be there: check --id-field and that --labels is this corpus's run"
        )
    bad = report["content_mismatch"]
    if bad:
        shown = ", ".join(repr(k) for k in bad[:5])
        raise SeedMergeRefused(
            f"{len(bad)} seed row(s) differ on an input field "
            f"from the --labels row with the same id ({shown}), so the ids no longer "
            "name the rows people labeled: the corpus changed since the sample was "
            "drawn, or --labels is another corpus's run"
        )
    if review_path:
        report.update(_drop_and_score_review(review_path, rows, id_field, pred_field))
    _write_jsonl_atomic(out_path, rows)
    return report


# --- overlay -----------------------------------------------------------------------------


def _handoff_path(out_path: str) -> str:
    stem = out_path[: -len(".jsonl")] if out_path.endswith(".jsonl") else out_path
    return stem + ".handoff.json"


def write_overlay(
    seed_paths: Union[str, Sequence[str]],
    dataset_path: str,
    out_path: str,
    *,
    id_field: str = "id",
    gt_field: Optional[str] = None,
    pred_field: str = "label",
    review_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Write the people's labels as a dataset-transformation overlay (``overlay.jsonl``).

    The corpus run's labels stay behind the API, so the people's labels reach them as an
    overlay by id on a view of the labeled DataSet. Each seed file is read under its own
    gold field (:func:`overlay_seed_rows`: ``gt_field`` when the file has it, else the
    task's default, else the corpus run's gold column), and a later seed's row for an id
    wins. When the corpus run labeled eval rows that are not corpus rows, only the review
    fold's rows are overlaid (a warning says how many seed rows are left out). Writes the
    overlay rows (:func:`build_overlay`) to ``out_path`` and the hand-off record beside it
    (``overlay.handoff.json``). With ``review_path`` the rows people labeled are dropped
    from the review file and its side file, and the spot-check rows among them scored.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import write_overlay

        report = write_overlay(["proj/seed.jsonl", "proj/review_seed.jsonl"],
                               "proj/labeled_dataset.json", "proj/overlay.jsonl")
        report["overlay_rows"], report["handoff"]

    Args:
        seed_paths: The seeds whose human rows go in the overlay, in order.
        dataset_path: The corpus run's deliverable record (``labeled_dataset.json``).
        out_path: Where to write the overlay rows.
        id_field: The id field.
        gt_field: The seeds' gold field, when named.
        pred_field: The labeled DataSet's prediction field (``score`` for ordinal,
            ``entities`` for ner).
        review_path: The corpus job's ``review.jsonl`` (or its side file).

    Returns:
        :func:`build_overlay`'s report, with ``gold_fields`` (per seed file),
        ``seed_rows_left_out`` when rows were left out, the review fields with
        ``review_path``, and ``handoff`` (the hand-off record's path).

    Raises:
        DataPreparerOperatorError: A seed, the deliverable record or the review file
            cannot be read, a seed file has rows but no gold value, no seed row has a
            human provenance and a gold value, or the record names no ``dataset_arn``.
    """
    if isinstance(seed_paths, str):
        seed_paths = [seed_paths]
    seed_files = []
    try:
        for path in seed_paths:
            seed_files.append((path, read_jsonl(path)))
        with open(dataset_path, encoding="utf-8") as fh:
            deliverable = json.load(fh)
    except (OSError, ValueError) as e:
        raise DataPreparerOperatorError(str(e)) from e
    seed_rows, gold_fields = overlay_seed_rows(seed_files, deliverable, gt_field, pred_field)
    seed_rows, left_out = seed_rows_for_overlay(seed_rows, deliverable)
    if left_out:
        _log.warning(
            "the corpus run labeled %s --eval rows that are not corpus rows (an id the "
            "corpus does not have, or a corpus id with another row's text), so the seed's "
            "ids are not all this corpus's ids for the same rows. %d seed row(s) are left "
            "out of the overlay (it patches the labeled DataSet by id, and could put a "
            "person's label on an unrelated row); only the review fold's rows are "
            "overlaid. Those seed rows ship the model's labels.",
            deliverable["eval_rows_labeled_for_scoring"],
            left_out,
        )
    rows, report = build_overlay(
        seed_rows, id_field=id_field, gt_field=OVERLAY_GOLD_KEY, pred_field=pred_field
    )
    report["gold_fields"] = gold_fields
    if left_out:
        report["seed_rows_left_out"] = left_out
    if not rows:
        raise DataPreparerOperatorError(
            "no seed row has a human provenance (human_verified or "
            f"spot_checked, or _bootstrap hil) and a gold value ({gold_fields}), so "
            "there is no label to overlay"
            + (
                " (the seed's own rows were left out, above; pass the review fold)"
                if left_out
                else ""
            )
        )
    try:
        handoff = overlay_handoff(
            deliverable, out_path, pred_field, id_field, overlay_rows=len(rows)
        )
    except DataPreparerOperatorError as e:
        raise DataPreparerOperatorError(f"{dataset_path}: {e}") from e
    if review_path:
        report.update(_drop_and_score_review(review_path, rows, id_field, pred_field))
    _write_jsonl_atomic(out_path, rows)
    handoff_path = _handoff_path(out_path)
    with open(handoff_path, "w", encoding="utf-8") as fh:
        json.dump(handoff, fh, indent=2)
        fh.write("\n")
    report["handoff"] = handoff_path
    return report


# --- the fold of the labels people returned ------------------------------------------

# The spot-check withdraws the accepted-band certificate at this many errors or more,
# not at one: the same count for the 29-row (0.90) and 59-row (0.95) draws. A
# spot-check row that never comes back, or comes back unsure, is one error.
SPOT_CHECK_WITHDRAW_AT = 7
# The spot-check's basis (spot_check.basis): the accepted band was certified by the
# labeling job (its side file carries calib_accepted), or routed on the bar.
BASIS_CALIBRATED = "calibrated"
BASIS_BAR = "bar"
# A labeling job's review rows that carry no calib_accepted and no routing tag
# (_review_reason): nothing on the rows says how the run was routed or whether it was
# certified, so the fold does not guess. A row with a routing tag was routed on the bar.
BASIS_UNKNOWN = "unknown"
CALIB_ACCEPT_FIELD = "calib_accepted"


# --- the spot-check verdict ------------------------------------------------------------


def spot_check_verdict(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    rows: int,
    corrected: int,
    not_returned: int,
    withdraw_at: int = SPOT_CHECK_WITHDRAW_AT,
    calibrated: bool = False,
    conflicts: int = 0,
    no_model_answer: int = 0,
    basis: Optional[str] = None,
) -> Dict[str, Any]:
    """The report keys of the spot-check verdict.

    Errors are the corrected rows, plus the rows not returned (unsure, skipped or
    missing), plus ``conflicts`` (rows whose reviewers disagreed: not verified correct,
    so they count against the model, but reported apart). The spot-check fails at
    ``withdraw_at`` errors or more. On a ``calibrated`` run (basis ``calibrated``) a
    failure also withdraws the certificate (``certificate_withdrawn``); on a run routed on
    the bar, or one whose basis is unknown, that key is left out. ``no_model_answer``
    counts returned rows with no model answer to compare (a blind row whose answers were
    never joined back): neither errors nor confirmations, and while any remain the verdict
    is incomplete (``spot_check["complete"]`` false).

    Args:
        rows: The spot-check rows.
        corrected: Returned rows whose label differs from the model's.
        not_returned: Rows that did not come back, or came back unsure or skipped.
        withdraw_at: The error count at which the spot-check fails.
        calibrated: Whether the run was certified (when ``basis`` is not given).
        conflicts: Rows whose reviewers disagreed.
        no_model_answer: Returned rows with no model answer to compare.
        basis: ``calibrated``, ``bar`` or ``unknown``; default from ``calibrated``.

    Returns:
        The keys to add to the fold's report; empty when there were no spot-check rows.
    """
    if not rows:
        return {}
    basis = basis or (BASIS_CALIBRATED if calibrated else BASIS_BAR)
    errors = corrected + not_returned + conflicts
    failed = errors >= withdraw_at
    out: Dict[str, Any] = {"spot_check_failed": failed}
    if basis == BASIS_CALIBRATED:
        out["certificate_withdrawn"] = failed
    out["spot_check_errors"] = f"{errors} of {rows}"
    # The agreement rate: the rows that came back with a model answer and confirmed it,
    # over the rows that came back with one to compare.
    confirmed = max(0, rows - corrected - not_returned - conflicts - no_model_answer)
    compared = confirmed + corrected
    out["spot_check"] = {
        "rows": rows,
        "errors": errors,
        "corrected": corrected,
        "confirmed": confirmed,
        "agreement_rate": (round(confirmed / compared, 4) if compared > 0 else None),
        "not_returned": not_returned,
        "conflicts": conflicts,
        "no_model_answer": no_model_answer,
        "complete": failed or not no_model_answer,
        "withdraw_at": withdraw_at,
        "basis": basis,
    }
    return out


def spot_check_message(report: Mapping[str, Any]) -> str:
    """The one-line verdict for a fold report's spot-check, by its basis.

    A calibrated run withdraws or keeps the certificate; a run routed on the bar has
    none, so a failure says the auto-accepted rows do not meet the 0.85 bar.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators import seed

        rows, report = seed.fold_human_labels(review_rows, returned_labels)
        if "spot_check_errors" in report:
            print(seed.spot_check_message(report))

    Args:
        report: A :func:`fold_human_labels` report with a spot-check verdict.

    Returns:
        The message, ending in a newline.
    """
    sc = report["spot_check"]
    return (
        _spot_check_text(report, sc).rstrip("\n") + _conflict_note(sc) + _no_answer_note(sc) + "\n"
    )


def _no_answer_note(sc: Mapping[str, Any]) -> str:
    n = sc.get("no_model_answer") or 0
    return (
        f" {n} spot-check row(s) carry no model answer to compare (blind rows): "
        "join the model's answers (the side file) and fold again; the verdict does not "
        "count them."
        if n
        else ""
    )


def _conflict_note(sc: Mapping[str, Any]) -> str:
    n = sc.get("conflicts") or 0
    return (
        f" {n} row(s) had conflicting reviewer answers; resolve them and re-run the check."
        if n
        else ""
    )


def _spot_check_text(  # pylint: disable=too-many-return-statements
    report: Mapping[str, Any], sc: Mapping[str, Any]
) -> str:
    n_of_m = f"{sc['errors']} of {sc['rows']} wrong"
    head = f"spot-check errors {report['spot_check_errors']} (fails at {sc['withdraw_at']}): "
    if sc.get("basis") == BASIS_CALIBRATED:
        if report["spot_check_failed"]:
            return (
                head + f"CERTIFICATE WITHDRAWN ({n_of_m}). The labeling job's certificate "
                "no longer holds: never state its accuracy figure. Fold the review into "
                "the seed and run the corpus job again.\n"
            )
        if not sc.get("complete", True):
            return head + "INCOMPLETE: the certificate stands only if the rows below confirm.\n"
        return head + "the certificate stands.\n"
    if sc.get("basis") == BASIS_UNKNOWN:
        why = (
            "basis unknown: the review file has no calib_accepted; pass the job's "
            "labeling_summary.json"
        )
        if report["spot_check_failed"]:
            return (
                head + f"SPOT-CHECK FAILED ({n_of_m}); {why}. If the run was "
                "certified, the certificate is withdrawn; if not, the auto-accepted "
                "rows do not meet the bar.\n"
            )
        if not sc.get("complete", True):
            return head + f"INCOMPLETE; {why}.\n"
        return head + f"the auto-accepted rows pass the spot-check; {why}.\n"
    if report["spot_check_failed"]:
        return (
            head + f"SPOT-CHECK FAILED THE BAR ({n_of_m}): the auto-accepted rows do not "
            "meet the 0.85 bar; review more rows, or fold the review into the seed, "
            "refine the spec and run the job again.\n"
        )
    if not sc.get("complete", True):
        return head + "INCOMPLETE at the 0.85 bar.\n"
    return head + "the auto-accepted rows pass the spot-check at the 0.85 bar.\n"


# --- the fold --------------------------------------------------------------------------


# pylint: disable-next=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
def fold_human_labels(  # noqa: C901
    sample_rows: Iterable[Any],
    human_labels: Iterable[Any],
    *,
    id_field: str = "id",
    gt_field: str = "gt_label",
    label_field: str = "label",
    source_export: Optional[str] = None,
    export_id_field: Optional[str] = None,
    pred_field: Optional[str] = None,
    model_outputs: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    qa_match: Any = None,
    correct_tolerance: Optional[float] = None,
    job_status: Optional[str] = None,
    spot_check_reviewed: bool = True,
    known_spot_check: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[Dict], Dict[str, Any]]:
    """Merge the rows people reviewed with the labels they returned into seed rows.

    An export row joins on ``export_id_field`` (default ``id_field``; ids as strings) and
    gives its label in ``gt_field`` or ``label_field``. Unlabeled rows are skipped and
    unsure or skipped ones counted apart. Each seed row is the reviewed row, unblinded,
    with the label as gold, ``_bootstrap: "hil"``, ``provenance``, ``source_export`` and
    the ``annotator`` when named.

    The spot-check verdict (:func:`spot_check_verdict`) has basis ``calibrated`` when a
    spot-check row or its side-file record carries a boolean ``calib_accepted``; else,
    for a job's review rows, ``job_status`` decides (``Certified`` is calibrated, any
    other the bar); else ``unknown`` for such rows with no routing tag; else the bar.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators import seed

        rows, report = seed.fold_human_labels(
            sample_rows, returned_labels, gt_field="gt_label", source_export="export.jsonl"
        )
        seed.append_to_seed("proj/seed.jsonl", rows)

    Args:
        sample_rows: The rows people reviewed.
        human_labels: The returned export (flat rows or Label Studio tasks).
        id_field: The reviewed rows' id field.
        gt_field: The seed's gold field (also read from the export).
        label_field: The export's label field when ``gt_field`` is absent.
        source_export: The export's name, recorded on each seed row.
        export_id_field: The export's id field when it differs from ``id_field``.
        pred_field: The model's answer field on the spot-check rows.
        model_outputs: Each id's side-file record: its answer goes back on the seed row,
            and ``review_outcome`` counts each reason's confirmed and corrected rows, with
            ``no_model_answer`` and ``not_scored`` apart, adding up to
            ``model_output_joined``.
        qa_match: The QA correctness rule for the counts and the verdict.
        correct_tolerance: The llm_judge tolerance for a confirmed row.
        job_status: The labeling job's ``Status`` (labeling_summary.json).
        spot_check_reviewed: False when the spot-check review was turned off: no
            verdict is given (``spot_check_skipped``).
        known_spot_check: Spot-check rows people had labeled already
            (``review.known_spot_check.json``), counted as returned.

    Returns:
        ``(seed_rows, report)``.
    """
    export_id_field = export_id_field or id_field
    human_labels = flatten_export(human_labels, export_id_field, span_fields(gt_field, label_field))
    label_by_id, annot_by_id, n_by_id, unresolved_ids, conflicts, _, empty = decided_by_id(
        human_labels, export_id_field, gt_field, label_field
    )

    seed: List[Dict] = []
    per_class: Dict[Any, int] = defaultdict(int)
    per_prov: Dict[str, int] = defaultdict(int)
    unlabeled = unresolved = multi = 0
    spot_rows = spot_corrected = spot_missing = spot_conflicts = spot_no_answer = 0
    spot_calibrated = spot_ga = spot_split = False
    joined = 0
    outcome: Dict[str, Dict[str, int]] = {}
    for r in sample_rows:
        if not isinstance(r, dict):
            continue
        key = id_key(r.get(id_field))
        prov = provenance_for_review(r)
        if prov == SPOT_CHECKED:
            spot_rows += 1
            # The model's answer: on the row (a blind row's _model_output), else in the
            # side file's record, whose pred_field names it.
            record = (model_outputs or {}).get(key) or {}
            answered = join_model_output(unblind_row(r), record.get("model_output"))
            # The job's calib_accepted travels with the model's answer (the side file):
            # it marks a certified run, whether or not the row came back.
            if isinstance(answered.get(CALIB_ACCEPT_FIELD), bool):
                spot_calibrated = True
            if r.get(GA_SPLIT_FIELD) == GA_SPOT_CHECK_SPLIT:
                spot_split = True
                if review_reason_of(r) is None:
                    # A job's split row with no routing tag says nothing about how the
                    # run was routed, so the basis is unknown.
                    spot_ga = True
            ok = (
                spot_check_confirms(
                    answered,
                    pred_field or record.get("pred_field"),
                    label_by_id[key],
                    qa_match=qa_match,
                    tolerance=correct_tolerance,
                )
                if key in label_by_id
                else None
            )
            if key in conflicts:
                spot_conflicts += 1  # reviewers disagreed: not verified, not missing
            elif key not in label_by_id:
                spot_missing += 1
            elif ok is None:
                spot_no_answer += 1  # nothing to compare: neither an error nor a pass
            elif not ok:
                spot_corrected += 1
        if key in label_by_id:
            row = unblind_row(r)
            record = (model_outputs or {}).get(key)
            if record is not None:
                row = join_model_output(row, record.get("model_output"))
                joined += 1
                verdict = review_verdict(
                    record, label_by_id[key], qa_match=qa_match, tolerance=correct_tolerance
                )
                # Every joined row is counted once, so the counts add up to
                # model_output_joined: one no task rule scores is NOT_SCORED.
                counts = outcome.setdefault(
                    review_reason_of(r) or "routed", {"confirmed": 0, "corrected": 0}
                )
                if verdict is None or verdict == NO_MODEL_ANSWER:
                    extra = NOT_SCORED if verdict is None else NO_MODEL_ANSWER
                    counts[extra] = counts.get(extra, 0) + 1
                else:
                    counts["confirmed" if verdict else "corrected"] += 1
            row[gt_field] = label_by_id[key]
            row["_bootstrap"] = "hil"
            row["provenance"] = prov
            if source_export is not None:
                row["source_export"] = source_export
            if annot_by_id.get(key):
                row["annotator"] = annot_by_id[key][0]
            if n_by_id[key] > 1:
                row["n_annotations"] = n_by_id[key]
                row["annotators"] = annot_by_id[key]
                multi += 1
            seed.append(row)
            per_class[class_key(label_by_id[key])] += 1
            per_prov[prov] += 1
        elif key in conflicts:
            continue
        elif key in unresolved_ids:
            unresolved += 1
        else:
            unlabeled += 1

    report: Dict[str, Any] = {
        "mode": "hil",
        "kept": len(seed),
        "sample_unlabeled": unlabeled,
        "sample_unresolved": unresolved,
        "per_provenance_kept": dict(sorted(per_prov.items())),
        "per_class_kept": dict(sorted(per_class.items(), key=lambda kv: str(kv[0]))),
    }
    if multi:
        report["multi_annotation_agreed"] = multi
    if conflicts:
        report["conflicting_ids"] = sorted(conflicts)
    if model_outputs is not None:
        report["model_output_joined"] = joined
        report["review_outcome"] = dict(sorted(outcome.items()))
    if empty:
        report["empty_labels"] = empty
    if spot_calibrated:
        basis = BASIS_CALIBRATED
    elif spot_split and job_status is not None:
        # The job's answers carry no calib_accepted: its Status says whether it certified.
        basis = BASIS_CALIBRATED if str(job_status) == "Certified" else BASIS_BAR
    elif spot_ga:
        basis = BASIS_UNKNOWN
    else:
        basis = BASIS_BAR
    if isinstance(known_spot_check, dict) and known_spot_check.get("rows"):
        spot_rows += int(known_spot_check["rows"])
        spot_corrected += int(known_spot_check.get("corrected") or 0)
        spot_no_answer += int(known_spot_check.get("no_model_answer") or 0)
        report["spot_check_known"] = int(known_spot_check["rows"])
    if not spot_check_reviewed:
        if spot_rows:
            report["spot_check_skipped"] = {
                "rows": spot_rows,
                "not_returned": spot_missing,
                "reason": "the spot-check review was turned off (review_high_confidence: "
                "false): no verdict, and the labels carry no accuracy promise",
            }
        return seed, report
    report.update(
        spot_check_verdict(
            spot_rows,
            spot_corrected,
            spot_missing,
            conflicts=spot_conflicts,
            no_model_answer=spot_no_answer,
            basis=basis,
        )
    )
    return seed, report


def append_to_seed(
    seed_path: str,
    rows: Sequence[Mapping[str, Any]],
    id_field: str = "id",
    gt_field: str = "gt_label",
) -> Tuple[int, List[Any]]:
    """Append the folded ``rows`` to the seed at ``seed_path`` in one atomic step.

    The new seed is written to a temporary file beside it and renamed over it, so an
    interrupted run leaves the old seed as it was. The same fold run again appends
    nothing; a fold whose ids are partly in the seed, or there with another label, is
    refused (:func:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels.check_seed_append`).
    A missing seed is created.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators import seed

        appended, already = seed.append_to_seed("proj/seed.jsonl", folded_rows)

    Args:
        seed_path: The seed file.
        rows: The folded rows.
        id_field: The id field.
        gt_field: The gold field.

    Returns:
        ``(appended, already_in_seed_ids)``.

    Raises:
        SeedAppendRefused: A different review of the same rows, or an id clash.
    """
    existing_text = ""
    if os.path.exists(seed_path):
        with open(seed_path, encoding="utf-8") as f:
            existing_text = f.read()
    seed_rows = [json.loads(line) for line in existing_text.splitlines() if line.strip()]
    appended, already = check_seed_append(seed_rows, rows, seed_path, id_field, gt_field)
    if not appended:
        return 0, already
    if existing_text and not existing_text.endswith("\n"):
        existing_text += "\n"
    _write_text_atomic(
        seed_path,
        existing_text + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
    )
    return appended, []


# --- the check -------------------------------------------------------------------------


def check_fold(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    export_rows: Iterable[Any],
    seed_rows: Iterable[Any],
    labels_rows: Optional[Iterable[Any]] = None,
    id_field: str = "id",
    gt_field: str = "gt_label",
    label_field: str = "label",
    pred_field: str = "label",
    export_id_field: Optional[str] = None,
    sample_rows: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """Verify one returned export was folded into the seed (and merged onto the labels).

    A decided export row names an id and a label and is not unsure or skipped. Each must
    appear in ``seed_rows`` as a human row (``_bootstrap: "hil"`` or a human provenance);
    with ``labels_rows``, the labeled rows' ``pred_field`` for that id must equal the
    reviewer's label. When export rows name an id but none has a ``gt_field`` or
    ``label_field`` column, the export's label column has another name: ``no_label_ids``
    lists them. An id whose annotations disagree is not decided (``conflicting_ids``).
    With ``sample_rows`` (the cold-start sample), every sample row must have exactly one
    decided label: ``sample_missing``, ``sample_unsure`` and ``sample_conflicting`` list
    the ones that do not.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators import seed

        report = seed.check_fold(export_rows, seed_rows, labels_rows=overlay_rows)
        assert report["ok"], report

    Args:
        export_rows: The returned export.
        seed_rows: The seed the folded rows were appended to.
        labels_rows: The overlay rows, or a merged local labeled file.
        id_field: The id field.
        gt_field: The gold field.
        label_field: The export's label field when ``gt_field`` is absent.
        pred_field: The label field of ``labels_rows``.
        export_id_field: The export's id field when it differs from ``id_field``.
        sample_rows: The cold-start sample.

    Returns:
        A report whose ``ok`` is True only when nothing is missing.
    """
    export_id_field = export_id_field or id_field
    export_rows = flatten_export(export_rows, export_id_field, span_fields(gt_field, label_field))
    decided, _, _, unresolved, conflicts, no_label, _ = decided_by_id(
        export_rows, export_id_field, gt_field, label_field
    )
    human_seed = {
        id_key(r.get(id_field))
        for r in seed_rows
        if isinstance(r, dict)
        and (r.get("_bootstrap") == "hil" or r.get("provenance") in HUMAN_PROVENANCE)
    }
    missing = sorted(k for k in decided if k not in human_seed)
    report: Dict[str, Any] = {
        "mode": "check",
        "export_decided": len(decided),
        "folded": len(decided) - len(missing),
        "missing_from_seed": missing,
    }
    ok = not missing
    if conflicts:
        report["conflicting_ids"] = sorted(conflicts)
        ok = False
    if no_label and not decided and not conflicts:
        report["no_label_ids"] = sorted(no_label)
        ok = False
    if sample_rows is not None:
        ids = [id_key(r.get(id_field)) for r in sample_rows if isinstance(r, dict)]
        report["sample_rows"] = len(ids)
        undecided = [k for k in ids if k not in decided and k not in conflicts]
        report["sample_missing"] = sorted(k for k in undecided if k not in unresolved)
        report["sample_unsure"] = sorted(k for k in undecided if k in unresolved)
        report["sample_conflicting"] = sorted(k for k in ids if k in conflicts)
        ok = ok and not (
            report["sample_missing"] or report["sample_unsure"] or report["sample_conflicting"]
        )
    if labels_rows is not None:
        by_id = {id_key(r.get(id_field)): r for r in labels_rows if isinstance(r, dict)}
        not_merged = sorted(
            k
            for k, lab in decided.items()
            if k not in by_id or not same_label(by_id[k].get(pred_field), lab)
        )
        report["merged"] = len(decided) - len(not_merged)
        report["not_merged"] = not_merged
        ok = ok and not not_merged
    report["ok"] = ok
    return report
