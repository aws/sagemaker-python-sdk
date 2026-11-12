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
"""One labeling round end to end: check and stage the inputs, submit the job, read its result.

A round is one SDG labeling job: a refine round (``stage`` ``dev`` or ``val``, one model,
registered under a scratch name), another working job (any other stage word), or the
corpus run (``stage`` ``corpus``, with ``review_dataset_name`` the jury, the cutoff and the
review DataSet). The work splits into three calls so a caller can write what each returns
before the next one runs:

1. :func:`prepare_round` checks the arguments, reads the client-readable inputs (the
   first row of the input, a local split, the gold, the demos), guards the frozen split,
   builds the request document and the re-attach digest. Nothing is uploaded. It returns
   a :class:`RoundPlan` whose ``request_text`` the caller writes to ``request_path`` and
   whose ``events`` (the test reads) the caller appends to the run log.
2. :func:`submit_round` re-attaches to the job a run with the same run id and stage
   already submitted, or checks that a corpus run's DataSet names are free, uploads the
   inputs and submits the job. The job record is handed to ``on_submit`` before
   ``CreateJob``, so a crashed run can re-attach.
3. :func:`read_round` reads the job's result through the labeled DataSet and returns
   every file the round writes (:class:`RoundOutputs`: ``texts``, ``copies``,
   ``removes``), the run's record for the run log and the summary;
   :func:`round_summary` adds the run-log entry id to the summary.

Refusals are :class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`
with the message the command line prints. Notes for the person running the round are in
``warnings``, each line as the command line prints it to stderr (``WARNING: ...``,
``NOTE: ...``); client reads of customer rows are announced on the ``client_read`` logger.
"""

from __future__ import absolute_import

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from pydantic import Field

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import report_client_read
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job._run_context import run_context
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import (
    inputs as inspect_input,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import client as job_client
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import request as backend
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    PEOPLE_LABELED_PROVENANCE,
    RECALIBRATION_FIELDS,
    STATED_CONFIDENCE_FIELDS,
    VERBALIZED_CONFIDENCE_FIELDS,
    stated_confidence_missing,
    verbalized_confidence_missing,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job._round_parts import (
    CORPUS_ARTIFACTS,
    GOLD_ID_PREFIX,
    OPTION_INVOCATION_KEYS,
    OPTION_SUMMARY_KEYS,
    build_job_manifest,
    check_gold_labels,
    check_ids,
    check_run_id,
    corpus_ids_are_ints,
    float_ids_to_join,
    held_out_report_from_job,
    input_kind,
    is_kfold_file,
    juror_shortfall,
    labels_from_job,
    labels_missing_from_spec,
    no_seed_message,
    read_local_rows,
    read_task_sample,
    request_sha256,
    review_model_outputs,
    row_id,
    seed_evidence,
    select_demo_rows,
    split_gold,
    text_key,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log import (
    LabelingRunRecord,
    labeling_run_record,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as split_state

_log = logging.getLogger(__name__)

# The summary keys that follow the run-log entry id (round_summary puts it before them).
_SUMMARY_KEYS_AFTER_RUN_LOG = (
    "unrefined_spec",
    "labels_missing_from_spec",
    "degraded_jurors",
    "jury_check",
    "golden_review",
    "note",
    "deliverable",
)


class RoundRefused(DataPreparerOperatorError):
    """A refusal after the round had already read the frozen test split or noted something.

    ``events`` holds the run-log entries of the test reads made before the refusal (the
    golden-row review check reads the test split before the gold is checked): the log
    records reads, not submissions, so the caller appends them before it reports the
    refusal. ``warnings`` are the notes collected before the refusal, each as the command
    line prints it, for the caller to show before the refusal's message.
    """

    def __init__(
        self,
        message: str,
        events: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
    ):
        """The refusal ``message``, the run-log ``events`` it leaves to append and the ``warnings`` to show."""
        super().__init__(message)
        self.events = list(events or [])
        self.warnings = list(warnings or [])


class RoundPlan(OperatorResult):
    """Everything :func:`prepare_round` checked and built, for :func:`submit_round` and :func:`read_round`.

    The caller creates ``staging_dir`` (and the folder of ``out``), writes
    ``request_text`` to ``request_path`` and appends ``events`` (the test reads; the run
    log records reads, not submissions) before it calls :func:`submit_round`.
    ``warnings`` are the notes to show, each as the command line prints it.

    ``to_json()`` leaves out the fields that carry the customer's text and rows
    (``CUSTOMER_CONTENT_FIELDS``: the input rows, the gold rows and their upload text, the
    demos, the spec and the request text), so a serialized plan does not copy them into a
    log or a file; read them from the attributes.
    """

    CUSTOMER_CONTENT_FIELDS: ClassVar[Tuple[str, ...]] = (
        "input_rows_text",
        "gold_rows",
        "gold_text",
        "demos_text",
        "spec_text",
        "request_text",
    )

    stage: str
    is_corpus: bool
    routing: bool
    task: str
    label_field: str
    pred_field: str
    text_fields: List[str]
    models: List[str]
    jury: List[str]
    dataset_name: str
    run_id: str
    explicit_run_id: bool
    out: str
    project_dir: str
    staging_dir: str
    request_path: str
    job_record_path: str
    document: Dict[str, Any]
    request: Dict[str, Any]
    request_text: str
    request_sha256: str
    spec_text: str
    spec_uri: str
    output_uri: str
    demos_text: Optional[str] = None
    demos_uri: Optional[str] = None
    gold_text: Optional[str] = None
    gold_rows: List[Dict[str, Any]] = Field(default_factory=list)
    eval_uri: Optional[str] = None
    eval_arn: Optional[str] = None
    input_rows_text: Optional[str] = None
    input_file: Optional[str] = None
    input_s3: Optional[str] = None
    input_arn: Optional[str] = None
    n_rows: Optional[int] = None
    n_fit: int = 0
    estimated_rows: Optional[int] = None
    unpassed_fields: List[str] = Field(default_factory=list)
    labels_missing: List[str] = Field(default_factory=list)
    seed_evidence: Optional[str] = None
    golden_check: Optional[Dict[str, int]] = None
    region: Optional[str] = None
    role_arn: Optional[str] = None
    hub_name: Optional[str] = None
    review_dataset_name: Optional[str] = None
    append: bool = False
    submit_max_wait: Optional[int] = None
    option_fields: List[str] = Field(default_factory=list)
    optimizer: Optional[str] = None
    invocation: Dict[str, Any] = Field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        """The plan's JSON-ready dict, without ``CUSTOMER_CONTENT_FIELDS``."""
        out = super().to_json()
        for key in self.CUSTOMER_CONTENT_FIELDS:
            out.pop(key, None)
        return out


# DescribeJob's terminal job statuses; any other status is a job that is still running.
_ENDED_JOB_STATUSES = ("Completed", "Failed", "Stopped")


class SubmittedRound(OperatorResult):
    """The job :func:`submit_round` submitted or re-attached to: ``job`` is ``submit_labeling_job``'s record."""

    job: Dict[str, Any]

    @property
    def running(self) -> bool:
        """Whether the job has not ended yet (only after ``submit_round(wait=False)``)."""
        status = (self.job or {}).get("job_status")
        return status is not None and status not in _ENDED_JOB_STATUSES


class RoundOutputs(OperatorResult):
    """What :func:`read_round` read back, for the caller to write.

    ``texts`` maps each path to the text to write there (UTF-8), ``copies`` each path to
    the downloaded artifact whose bytes go there, and ``removes`` lists the files to
    delete when they exist (a review file an earlier corpus run left). ``record`` is the
    run's run-log record (append its ``jobs``, then its
    :meth:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.run_log.LabelingRunRecord.artifact_entry`
    of their ids), ``None`` with ``run_log_error`` when it could not be
    built. ``summary`` is the round's report without the run-log entry id
    (:func:`round_summary` adds it).
    """

    texts: Dict[str, str] = Field(default_factory=dict)
    copies: Dict[str, str] = Field(default_factory=dict)
    removes: List[str] = Field(default_factory=list)
    record: Optional[LabelingRunRecord] = None
    run_log_error: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)


def default_run_id():
    """A per-run unique id: ``<timestamp>-<short uuid>``, safe in a Hub name."""
    return time.strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def resolve_task(task, spec, sample_path, warnings, log=_log):
    """The labeling task: an explicit ``task`` wins; else detect it from the spec and a gold sample.

    ``sample_path`` is the eval or demos file, whose ``gt_*`` field is the detector's
    strongest signal. The detector's line is appended to ``warnings``.

    Raises:
        DataPreparerOperatorError: A low-confidence pick (decision ``ask``); ``confirm``
            adopts it with a logged notice.
    """
    if task is not None:
        log.info("task=%r (explicit --task; overrides auto-detection)", task)
        return task
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect.task import detect

    spec_text = ""
    if spec:
        try:
            with open(spec, encoding="utf-8", errors="replace") as fh:
                spec_text = fh.read()
        except OSError:
            spec_text = ""
    result = detect(spec_text, read_task_sample(sample_path) if sample_path else [])
    task, conf = result.task, result.confidence
    decision, reason = result.decision, result.reason
    msg = (
        f"[task-detector] task={task} (confidence={conf:.2f}, decision={decision}): "
        f"{reason} -- pass --task to override."
    )
    warnings.append(msg)
    log.info("%s", msg)
    if decision == "ask":
        raise DataPreparerOperatorError(
            f"task detection is low-confidence (decision=ask, task={task!r}, "
            f"conf={conf:.2f}); pass an explicit --task to proceed."
        )
    if decision == "confirm":
        log.warning(
            "task detection is medium-confidence (decision=confirm, conf=%.2f); "
            "adopting %r. Pass --task to override.",
            conf,
            task,
        )
    return task


def optimizer_config(optimizer):
    """The optimizer option parsed (a JSON object), or ``None``.

    Raises:
        DataPreparerOperatorError: Not JSON.
    """
    if optimizer is None:
        return None
    try:
        return json.loads(optimizer)
    except ValueError as e:
        raise DataPreparerOperatorError(
            f"--optimizer is not JSON ({e}). Nothing was uploaded."
        ) from None


def check_option_flags(  # pylint: disable=too-many-arguments
    *,
    stage,
    routing,
    input_data,
    eval_data,
    optimizer,
    recalibration_fit,
    confidence_recalibration,
    target,
):
    """Stop before any read or upload on an option combination the job would mis-run.

    The gold tags of the optimizer and of the recalibration fit are written on a local
    eval file only, and fit rows are left out of the scoring only with Recalibration on a
    corpus run.

    Raises:
        DataPreparerOperatorError: The combination, named.
    """
    fit, recal = recalibration_fit, confidence_recalibration
    if bool(fit) != bool(recal):
        raise DataPreparerOperatorError(
            "--confidence-recalibration and --recalibration-fit go together: the map is "
            "fit on those rows, and without the map they would be scored as gold. "
            "Nothing was uploaded."
        )
    if fit and not (stage == "corpus" and routing and target is not None):
        raise DataPreparerOperatorError(
            "--recalibration-fit runs on the corpus run only (--stage corpus, "
            "--review-dataset-name, --target). Nothing was uploaded."
        )
    try:  # a .ref.json pointer is a file too, but names S3 rows the guard cannot check
        local_input = os.path.isfile(input_data) and input_kind(input_data)[0] == "local"
    except ValueError:
        local_input = False
    if optimizer is not None and not local_input:
        raise DataPreparerOperatorError(
            "--optimizer needs a local --input (the dev and val rows), so the split guard checks it. "
            "Nothing was uploaded."
        )
    if optimizer is not None and stage in ("dev", "val"):
        raise DataPreparerOperatorError(
            f"--optimizer runs on dev and val rows together: pass --stage optimize, not "
            f"--stage {stage} (references/spec-refinement.md). Nothing was uploaded."
        )
    if fit or optimizer is not None:
        try:
            local = bool(eval_data) and input_kind(eval_data)[0] == "local"
        except ValueError:
            local = True  # the eval read below names the error
        if not local:
            raise DataPreparerOperatorError(
                f"--{'optimizer' if optimizer is not None else 'recalibration-fit'} "
                "needs a local --eval: the gold rows are tagged on the client, which a "
                "DataSet ARN or S3 --eval does not allow. Nothing was uploaded."
            )


def split_reads(  # pylint: disable=too-many-arguments
    *, stage, is_corpus, local_input, input_data, eval_data, demos
):
    """``(path, step, splits the step may read)`` for each local file this job reads that a frozen split guards.

    A dev round reads dev, a val round val, another working stage dev and val, the corpus
    run's gold test only (the certificate), demos dev only. The corpus itself is not a
    split file and is never read here.
    """
    # The test read goes last: a refusal before it leaves no test read in the log.
    reads = [(demos, "demos", {"dev"})]
    if is_corpus:
        reads.append((eval_data, "certificate", {"test"}))
    else:
        # A dev or val round reads its own split's rows, or a fold a k-fold rotation wrote
        # (dev and val rows); another working stage may read dev and val, never test.
        reads += [
            (
                p,
                f"labeling-{stage}",
                (
                    {stage}
                    if stage in ("dev", "val") and not (os.path.isfile(p) and is_kfold_file(p))
                    else {"dev", "val"}
                ),
            )
            for p in (input_data if local_input else None, eval_data)
            if p
        ]
    return [(p, step, allowed) for p, step, allowed in reads if p and os.path.isfile(p)]


def fit_rows_in_corpus(input_data, rows, fit_ids):
    """The recalibration-fit ids the job would refuse at load as corpus rows, and how they were found.

    A row people labeled from the corpus (a cold-start sample row, provenance
    ``human_verified`` or ``spot_checked``) is one by construction, so that is checked
    first and the corpus is not read. Otherwise a local input's ids are compared (ids
    only; the run uploads that file anyway). An S3 or DataSet corpus is not read: the job
    checks it.
    """
    people = {
        str(r.get("id"))
        for r in rows
        if r.get("provenance") in PEOPLE_LABELED_PROVENANCE and str(r.get("id")) in fit_ids
    }
    if people:
        return (
            people,
            "people labeled them from the corpus (provenance "
            + "/".join(PEOPLE_LABELED_PROVENANCE)
            + ")",
        )
    if not (input_data and os.path.isfile(input_data) and input_kind(input_data)[0] == "local"):
        return set(), None
    hit, n = set(), 0
    with open(input_data, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rid = json.loads(line).get("id")
            except (ValueError, AttributeError):
                continue
            if rid is not None and str(rid) in fit_ids:
                hit.add(str(rid))
    report_client_read("recalibration-fit corpus id check (ids only)", n)
    return hit, "their ids are in --input"


def option_gold(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    gold_rows,
    label_field,
    text_fields,
    optimizer,
    recalibration_fit,
    project_dir,
    input_data,
    int_ids=False,
):
    """The container options' gold tags.

    The optimizer: each gold row gets ``optimizer_split`` dev or val from the frozen split
    (any other row stops the run); ``gold_rows`` are tagged in place. The recalibration
    fit: that val file's gold rows, marked ``recalibration_fit``, to be added to the gold.

    Returns:
        The rows to add, their ids written as the eval ids are (``int_ids``: 300.0 as 300).

    Raises:
        DataPreparerOperatorError: The run cannot carry the tags.
    """
    if optimizer is not None:
        d = split_state.find_splits_dir(None, project_dir)
        st = split_state.load_state(d) if d else None
        if not st:
            raise DataPreparerOperatorError(
                "--optimizer needs a frozen split (code_templates/split_freeze.py, ACTION freeze): the job takes "
                "dev and val rows only. Nothing was uploaded."
            )
        owner = {i: s for s in ("dev", "val") for i in st["splits"][s]["ids"]}
        outside = [g["id"] for g in gold_rows if str(g["id"]) not in owner]
        if outside:
            raise DataPreparerOperatorError(
                f"--optimizer: {len(outside)} --eval row(s) are not dev or val rows; "
                "the job takes no other row. Nothing was uploaded."
            )
        for g in gold_rows:
            g["optimizer_split"] = owner[str(g["id"])]
    if not recalibration_fit:
        return []
    try:
        rows = read_local_rows(recalibration_fit, "evaluation_data")
        touched = split_state.guard_read(
            recalibration_fit,
            "recalibration-fit",
            {"val"},
            project_dir=project_dir,
            rows=rows,
        ).touched
    except (OSError, ValueError) as e:
        raise DataPreparerOperatorError(
            f"--recalibration-fit: {e}. Nothing was uploaded."
        ) from None
    if touched is None or touched.get("outside") or any(r.get("id") is None for r in rows):
        raise DataPreparerOperatorError(
            "--recalibration-fit takes val rows of a frozen split (split_freeze.py "
            "freeze), each with its id. Nothing was uploaded."
        )
    _, fit, _ = split_gold(rows, label_field, text_fields=text_fields, int_ids=int_ids)
    if {str(g["id"]) for g in fit} & {str(g["id"]) for g in gold_rows}:
        raise DataPreparerOperatorError(
            "--recalibration-fit rows overlap --eval: the fit rows must not be the "
            "certified rows. Nothing was uploaded."
        )
    in_corpus, why = fit_rows_in_corpus(input_data, rows, {str(g["id"]) for g in fit})
    if in_corpus:
        raise DataPreparerOperatorError(
            f"--recalibration-fit: {len(in_corpus)} fit row(s) are corpus rows (e.g. "
            f"{sorted(in_corpus)[:3]}; {why}). The job refuses a fit row that is a corpus row "
            "at load, after the upload, so the map cannot run when the seed rows are corpus "
            "rows, as on a cold start (references/cold-start.md). Nothing was uploaded."
        )
    return [dict(g, recalibration_fit=True) for g in fit]


def guard_split_reads(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    stage,
    is_corpus,
    local_input,
    input_data,
    eval_data,
    demos,
    golden_review,
    project_dir,
    warnings,
):
    """Stop before any upload when a file breaks the frozen split.

    The note for a project whose splits are not frozen is appended to ``warnings``.

    Returns:
        The test reads, ``[(path, step, rows)]``, which the caller records once the client
        checks pass and the request is built.

    Raises:
        DataPreparerOperatorError: The split guard's refusal.
    """
    reads = split_reads(
        stage=stage,
        is_corpus=is_corpus,
        local_input=local_input,
        input_data=input_data,
        eval_data=eval_data,
        demos=demos,
    )
    if split_state.find_splits_dir(None, project_dir) is None and any(
        os.path.basename(p) in ("dev.jsonl", "val.jsonl", "test.jsonl") for p, _s, _a in reads
    ):
        warnings.append(
            "WARNING: the project's splits/ has no split_state.json, so nothing guards the "
            "split: freeze the split (code_templates/split_freeze.py) after Step 1 (references/full-loop.md)."
        )
    test_reads = []
    for path, step, allowed in reads:
        try:
            touched = split_state.guard_read(
                path, step, allowed, project_dir=project_dir, record_test_read=False
            ).touched
        except (OSError, ValueError) as e:
            raise DataPreparerOperatorError(f"split guard: {e}. Nothing was uploaded.") from None
        if touched and touched.get("test"):
            test_reads.append((path, step, touched["test"]))
    test = split_state.test_ids(project_dir=project_dir)
    if is_corpus and test and eval_data and not os.path.isfile(eval_data):
        raise DataPreparerOperatorError(
            "split guard: with a frozen split the corpus run's --eval is the local splits/test.jsonl, so "
            "the certificate can be checked against every test row. Nothing was uploaded."
        )
    if is_corpus and test and eval_data and os.path.isfile(eval_data):
        # The certificate runs on the whole frozen test split, less only the rows a person
        # relabeled in the golden-row review: a test row is never dropped on a flag.
        got = {str(r.get("id")) for r in read_local_rows(eval_data, "evaluation_data")}
        excluded = set()
        if golden_review and os.path.isfile(golden_review):
            try:
                with open(golden_review, encoding="utf-8") as fh:
                    manifest = json.load(fh)
                if not isinstance(manifest, dict) or not isinstance(
                    manifest.get("excluded") or [], list
                ):
                    raise ValueError(
                        "not a golden_review.py manifest (a JSON object with an 'excluded' list)"
                    )
            except (OSError, ValueError) as e:
                raise DataPreparerOperatorError(
                    f"--golden-review {golden_review!r} is not readable: {e}. Nothing was uploaded."
                ) from None
            excluded = {str(i) for i in manifest.get("excluded") or []}
        missing = sorted(test - got - excluded)
        if missing:
            raise DataPreparerOperatorError(
                f"split guard: --eval leaves out {len(missing)} row(s) of the frozen test "
                f"split (e.g. {missing[:3]}): the certificate runs on every test row "
                "except the ones a person relabeled (references/label-review.md, Golden "
                "rows). Nothing was uploaded."
            )
    return test_reads


def check_golden_review(manifest, eval_path, eval_rows):
    """The golden-row review check of a reviewed eval file.

    Returns:
        ``(counts, note, events)``: :func:`~.golden_review.check_reviewed`'s counts, the
        note to show and the test-read entries the check made; ``(None, None, [])``
        without a manifest.

    Raises:
        RoundRefused: The check refused the copy after it read the test split; its
            ``events`` are that read's entries.
        DataPreparerOperatorError: A reviewed copy without its manifest, or the check's
            refusal before any read.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import (
        golden as golden_review,
    )

    if manifest is None:
        if golden_review.has_mark(eval_rows):
            raise DataPreparerOperatorError(
                f"--eval {eval_path!r} is a golden_review.py reviewed copy: pass "
                "--golden-review <its golden_review.json> so the run checks that only "
                "a person changed its labels. Nothing was uploaded."
            )
        return None, None, []
    try:
        checked = golden_review.check_reviewed(manifest, eval_path)
    except (OSError, KeyError, ValueError) as e:
        message = f"--golden-review: {e}. Nothing was uploaded."
        read = list(getattr(e, "events", None) or [])
        if read:
            raise RoundRefused(message, read) from None
        raise DataPreparerOperatorError(message) from None
    counts = checked.to_json()
    note = (
        f"NOTE: golden rows reviewed: {counts['changed']} row(s) a person relabeled left "
        f"the golden rows, {counts['confirmed']} confirmed, {counts['unresolved']} "
        f"unresolved (kept as they were), of {counts['golden_rows']}."
    )
    return counts, note, list(checked.events)


def prepare_round(  # pylint: disable=too-many-arguments,too-many-locals,unused-argument
    *,
    spec: str,
    input_data: str,
    out: str,
    project_dir: str,
    stage: str,
    generation_model_id: str,
    dataset_name: Optional[str] = None,
    jury_models: Optional[str] = None,
    task: Optional[str] = None,
    text_fields: str = "text",
    label_set: Optional[str] = None,
    eval_data: Optional[str] = None,
    golden_review: Optional[str] = None,
    label_field: Optional[str] = None,
    demos: Optional[str] = None,
    demos_per_class: int = 0,
    confidence_method: Optional[str] = None,
    confidence_estimator: Optional[str] = None,
    judge_models: Optional[str] = None,
    confidence_floor_bypass: Optional[bool] = None,
    confidence_recalibration: Optional[str] = None,
    recalibration_fit: Optional[str] = None,
    optimizer: Optional[str] = None,
    target: Optional[float] = None,
    review_dataset_name: Optional[str] = None,
    graders: Optional[str] = None,
    hub_name: Optional[str] = None,
    append: bool = False,
    confirm_no_seed: bool = False,
    s3_prefix: Optional[str] = None,
    region: Optional[str] = None,
    role_arn: Optional[str] = None,
    run_id: Optional[str] = None,
    submit_max_wait: Optional[int] = None,
) -> RoundPlan:
    """Check a round's arguments and inputs and build its request; nothing is uploaded.

    The arguments are the command line's values as given (comma-separated lists, JSON
    text): the run log's invocation records them so, for a replay. The command line's
    parser checks the choices and ranges before this runs, and so must another caller:
    ``task`` in ``LABELING_TASKS``, ``confidence_method`` in ``CONFIDENCE_METHODS``,
    ``confidence_estimator`` in ``CONFIDENCE_ESTIMATORS``, ``confidence_recalibration`` in
    ``RECALIBRATIONS``, ``target`` between ``MIN_TARGET`` and ``MAX_TARGET``,
    ``submit_max_wait`` at least 0 (the names are
    :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request`'s).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import round as labeling_round

        plan = labeling_round.prepare_round(
            spec="proj/labeling_spec.md", input_data="proj/splits/dev.jsonl", out="proj/work/dev_pred.jsonl",
            project_dir="proj", stage="dev", generation_model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            eval_data="proj/splits/dev.jsonl", s3_prefix="s3://bucket/prefix", region="us-west-2",
            role_arn="arn:aws:iam::123456789012:role/SDG",
        )
        run_log.extend(plan.events)

    Args:
        spec: The labeling spec (local Markdown file).
        input_data: The rows to label: a local JSONL, an ``s3://`` URI or
            ``corpus.ref.json`` pointer, or a DataSet ARN.
        out: A refine round: where the job's labels land (.jsonl). The corpus run: where
            the deliverable record lands (a .json file).
        project_dir: The project directory whose run log records this run.
        stage: ``dev`` or ``val`` (a refine round), ``corpus`` (the deliverable run), or
            another word for a working job.
        generation_model_id: The one model a refine round runs (and the first juror).
        dataset_name: The deliverable's DataSet name; ``autolabel-<task>`` by default.
        jury_models: Comma-separated model ids: the corpus run's jury.
        task: The labeling task; detected from the spec and the gold rows when ``None``.
        text_fields: Comma-separated input fields.
        label_set: A JSON list of the classes, checked against the spec text.
        eval_data: The gold rows the job scores against: a local labeled split, an
            ``s3://`` URI or a DataSet ARN.
        golden_review: The golden-row review manifest of a reviewed eval file (corpus run).
        label_field: The gold field of the eval and demos rows; the task's by default.
        demos: A local JSONL of labeled dev rows for few-shot demos.
        demos_per_class: Demos per class drawn from ``demos`` (0: zero-shot).
        confidence_method: ``Confidence.Method``.
        confidence_estimator: ``Confidence.Estimator``.
        judge_models: Comma-separated judge model ids (``Confidence.Models``).
        confidence_floor_bypass: ``Confidence.FloorBypass``.
        confidence_recalibration: ``Confidence.Recalibration``.
        recalibration_fit: The val split file whose gold rows fit the recalibration map.
        optimizer: ``Config.Optimizer`` as JSON text.
        target: ``Scoring.Target`` (corpus run).
        review_dataset_name: ``Routing.ReviewDatasetName``: makes this the corpus run.
        graders: Comma-separated grader model ids (summarization).
        hub_name: ``OutputDataConfig.HubName``.
        append: The corpus run may add a version to an existing DataSet name.
        confirm_no_seed: The customer said yes to labels with no accuracy promise.
        s3_prefix: The writable S3 base for the job's inputs and output.
        region: The region; ``None`` resolves to the ambient boto3 default region, else
            ``us-west-2`` (the command line passes ``AWS_REGION``, else ``us-west-2``).
            The resolved region is the one the plan, the invocation, the job and the
            DataSet name check use.
        role_arn: The job's IAM role.
        run_id: Names this run; a fresh id when ``None``. The same id and stage
            re-attach to the job already submitted.
        submit_max_wait: Seconds ``CreateJob`` may back off on limits.

    Returns:
        A :class:`RoundPlan`. The caller creates the folder of ``plan.out`` and
        ``plan.staging_dir``, may prune old run scopes there
        (:meth:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.job._run_context.RunContext.prune_old_scopes`),
        writes ``plan.request_text`` to ``plan.request_path`` and appends ``plan.events``
        to the run log, then calls :func:`submit_round`. (The command line made those
        folders and pruned before it drew the demos, so a refusal from there on left them;
        here a refused round creates and prunes nothing.)

    Raises:
        RoundRefused: A refusal after notes were collected or the test split was read;
            the caller shows its ``warnings``, appends its ``events``, then reports it.
        DataPreparerOperatorError: Any other refusal.
    """
    kwargs = dict(locals())  # the first statement: the arguments, and nothing else
    kwargs["region"] = _s3.resolve_region(region)
    events: List[Dict[str, Any]] = []
    warnings: List[str] = []
    try:
        return _prepare(events, warnings, **kwargs)
    except RoundRefused as e:
        raise RoundRefused(str(e), events + e.events, warnings) from e
    except DataPreparerOperatorError as e:
        if events or warnings:
            raise RoundRefused(str(e), events, warnings) from e
        raise


def _prepare(  # noqa: C901 pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    events,
    warnings,
    *,
    spec,
    input_data,
    out,
    project_dir,
    stage,
    generation_model_id,
    dataset_name,
    jury_models,
    task,
    text_fields,
    label_set,
    eval_data,
    golden_review,
    label_field,
    demos,
    demos_per_class,
    confidence_method,
    confidence_estimator,
    judge_models,
    confidence_floor_bypass,
    confidence_recalibration,
    recalibration_fit,
    optimizer,
    target,
    review_dataset_name,
    graders,
    hub_name,
    append,
    confirm_no_seed,
    s3_prefix,
    region,
    role_arn,
    run_id,
    submit_max_wait,
):
    """The body of :func:`prepare_round`.

    ``events`` collects the run-log entries of the reads so far, ``warnings`` the notes.
    """
    if not s3_prefix:
        raise DataPreparerOperatorError(
            "set --s3-prefix (or AUTOLABEL_S3_PREFIX) to a writable S3 base, "
            "e.g. s3://my-bucket/autolabel/PROJECT/."
        )
    if not backend.S3_URI_RE.match(s3_prefix):
        raise DataPreparerOperatorError(
            f"--s3-prefix must be an s3://bucket/prefix URI, got {s3_prefix!r}"
        )
    raw_stage = stage
    stage = re.sub(r"[^A-Za-z0-9]+", "-", stage).strip("-").lower()
    if not stage:
        raise DataPreparerOperatorError(
            "--stage must contain a letter or digit (dev, val, corpus, ...)"
        )
    is_corpus = stage == "corpus"
    routing = review_dataset_name is not None
    if routing and not is_corpus:
        raise DataPreparerOperatorError(
            "--review-dataset-name sets Routing, which is the corpus run: pass --stage corpus"
        )
    if target is not None and not routing:
        raise DataPreparerOperatorError(
            "--target (Scoring.Target) certifies the corpus run; it needs "
            "--review-dataset-name (Routing). A refine round sends no target."
        )
    if golden_review is not None and not (is_corpus and eval_data):
        raise DataPreparerOperatorError(
            "--golden-review checks the corpus run's reviewed golden rows: pass it with "
            "--stage corpus and --eval <golden_reviewed.jsonl>"
        )
    check_option_flags(
        stage=stage,
        routing=routing,
        input_data=input_data,
        eval_data=eval_data,
        optimizer=optimizer,
        recalibration_fit=recalibration_fit,
        confidence_recalibration=confidence_recalibration,
        target=target,
    )

    # Model ids are checked by the service at CreateJob against its allow-list (an
    # unlisted id is refused synchronously, naming the allowed ids); the refusal reaches
    # the user through CreateJobRefused before any job starts. The seed-sized uploads that
    # precede it take seconds. The graders, jury and judge lists are parsed here only for
    # the request.
    grader_ids = [g.strip() for g in (graders or "").split(",") if g.strip()]
    jury = [m.strip() for m in (jury_models or "").split(",") if m.strip()]
    judges = [m.strip() for m in (judge_models or "").split(",") if m.strip()]
    opt = optimizer_config(optimizer)
    if is_corpus and Path(out).suffix.lower() == ".jsonl":
        raise DataPreparerOperatorError(
            f"--out {out!r}: the corpus run downloads no labels (the labeled corpus "
            "stays behind the API); --out names its deliverable record, a .json file "
            "such as <proj>/labeled_dataset.json (references/full-loop.md, Step 7)."
        )

    try:
        kind, input_ref = input_kind(input_data)
    except ValueError as e:
        raise DataPreparerOperatorError(str(e)) from None
    raw_text_fields = text_fields
    text_fields = [s.strip() for s in (text_fields or "").split(",") if s.strip()]
    if not text_fields:
        raise DataPreparerOperatorError("--text-fields names at least one input field")
    option_fields = {}  # row fields the job writes only with an option, and that option
    if confidence_method == backend.VERBALIZED:
        option_fields.update(
            dict.fromkeys(VERBALIZED_CONFIDENCE_FIELDS, "Confidence.Method verbalized")
        )
    if confidence_recalibration:
        option_fields.update(dict.fromkeys(RECALIBRATION_FIELDS, "Confidence.Recalibration"))
    clash = sorted(set(text_fields) & set(option_fields))
    if clash:
        raise DataPreparerOperatorError(
            f"--text-fields {', '.join(clash)}: a job with "
            f"{' and '.join(sorted({option_fields[f] for f in clash}))} writes that row "
            "field itself; rename the input column"
        )

    # The first row of the input: field names and a size estimate (the only corpus read).
    try:
        first_row, first_bytes, size_bytes, _resolved_input = inspect_input.read_first_row(
            input_ref, region
        )
    except (OSError, ValueError) as e:
        raise DataPreparerOperatorError(
            f"cannot read the first row of --input {input_ref!r}: {e}"
        ) from None
    missing_fields = [f for f in text_fields if f not in first_row]
    if missing_fields:
        raise DataPreparerOperatorError(
            f"--text-fields {', '.join(missing_fields)} not on the first row of --input "
            f"(fields: {', '.join(first_row)}); a row missing a text field fails the job "
            "before any model call"
        )
    estimated_rows = inspect_input.estimate_rows(size_bytes, first_bytes)
    if estimated_rows and estimated_rows > backend.MAX_LABELING_ROWS:
        warnings.append(
            f"WARNING: --input is about {estimated_rows:,} rows by size; the job labels at "
            f"most {backend.MAX_LABELING_ROWS:,} and refuses a larger file at load "
            "(overview.md, Limits). The count is an estimate from the file size and the "
            "first row's length, so it is off when rows vary in length; the job counts the "
            "rows exactly at load, before any model call."
        )

    # The task, from the spec and the gold rows (eval, else demos) when not given.
    sample_path = None
    for cand in (eval_data, demos):
        if cand and os.path.isfile(cand):
            sample_path = cand
            break
    task = resolve_task(task, spec, sample_path, warnings)
    label_field_given = label_field
    label_field = label_field or backend.GOLD_FIELDS[task]
    pred_field = backend.PREDICTION_FIELDS[task]

    # The frozen split (references/full-loop.md, Step 1): every local split file this job
    # reads holds only the splits its stage may read; a test row only as the corpus run's
    # gold, where the job picks the cutoff and certifies.
    test_reads = guard_split_reads(
        stage=stage,
        is_corpus=is_corpus,
        local_input=kind == "local",
        input_data=input_data,
        eval_data=eval_data,
        demos=demos,
        golden_review=golden_review,
        project_dir=project_dir,
        warnings=warnings,
    )

    # Text columns the labeler would not see (a page title, a source, a question): the
    # customer decides whether they belong in the text fields.
    unpassed_fields = inspect_input.unpassed_text_fields(
        first_row, text_fields, exclude={label_field}
    )
    if unpassed_fields:
        warnings.append(
            f"WARNING: --input has text column(s) {', '.join(unpassed_fields)} that "
            "--text-fields does not pass to the labeler. If the label can depend on them "
            "(a title, a source, a question, the surrounding context), ask the customer "
            "whether to add them to --text-fields and to the spec's Input Data "
            "(references/full-loop.md, Step 0)."
        )

    # The panel: the jury of a corpus run on a vote task, else the one model.
    verbalized = confidence_method == backend.VERBALIZED or bool(confidence_estimator)
    if routing and task in backend.VOTE_TASKS and not verbalized:
        models = jury or [generation_model_id]
        if generation_model_id not in models:
            models = [generation_model_id] + models
        if target is None:
            note = backend.bar_vote_note(len(models))
            if note:
                warnings.append(f"WARNING: {note}")
    else:
        models = [generation_model_id]
        if routing and jury and jury != [generation_model_id] and verbalized:
            option = (
                "Confidence.Method verbalized"
                if confidence_method == backend.VERBALIZED
                else f"Confidence.Estimator {confidence_estimator}"
            )
            warnings.append(
                f"NOTE: {option} labels with one model; --jury-models "
                f"is ignored ({generation_model_id} labels)."
            )
        elif routing and jury and jury != [generation_model_id]:
            warnings.append(
                f"NOTE: labeling.{task} runs one model; --jury-models is ignored "
                f"({generation_model_id} labels)."
            )

    # The spec: size-checked here, uploaded by submit_round.
    try:
        spec_text = Path(spec).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise DataPreparerOperatorError(f"cannot read --spec {spec!r}: {e}") from None
    label_names = json.loads(label_set) if label_set else []
    labels_missing = labels_missing_from_spec(spec_text, label_names) if label_names else []
    if labels_missing:
        warnings.append(
            f"WARNING: label(s) {', '.join(labels_missing)} from --label-set are never "
            "mentioned in the spec, so the model must guess what they mean. Draft a one-line "
            "definition per label and have the user confirm it before labeling "
            "(references/spec-authoring.md)."
        )
    if task in ("llm_judge", "ordinal"):
        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.rubric import (
            rubric_warning,
        )

        rw = rubric_warning(task, spec_text, None, None)
        if rw:
            warnings.append(f"WARNING: {rw}")
        if routing and backend.spec_score_scale(spec_text) is None:
            # The job sizes the resample band from the stated scale and fails at load
            # without one; refuse here, before anything is uploaded.
            raise DataPreparerOperatorError(
                f"labeling.{task} sizes its resample agreement band from the score "
                "scale, but the spec states none. Add a line such as 'Scale: 1 to 5' "
                "under the spec's Output Schema (references/task-types.md, Numeric "
                "scoring)."
            )

    # No seed: the labels would carry no promise; only the customer's yes allows it.
    evidence = seed_evidence(
        out, spec, eval_path=eval_data, demos_path=demos, demos_per_class=demos_per_class
    )
    if evidence is None:
        if not confirm_no_seed:
            raise DataPreparerOperatorError(no_seed_message())
        warnings.append(
            "WARNING: no labeled seed. The customer said yes to labels with no accuracy "
            "promise (--confirm-no-seed), so the run goes on; never state an accuracy "
            "figure for these labels (references/cold-start.md, Opt-out)."
        )

    # Run scope: staging dir + S3 sub-prefix per run id, per stage.
    explicit_run_id = bool(run_id)
    run_id = run_id or default_run_id()
    try:
        run_slug = check_run_id(run_id)
    except ValueError as e:
        raise DataPreparerOperatorError(str(e)) from None
    rc = run_context(run_id)
    out_path = Path(out)
    work = rc.staging_dir(out_path.parent) / stage
    s3_run = rc.s3_prefix(s3_prefix).rstrip("/") + f"/{stage}/"

    # Dataset names.
    base_name = dataset_name or f"autolabel-{task.replace('_', '-')}"
    try:
        if is_corpus:
            labeled_name = backend.validate_dataset_name(base_name, "--dataset-name")
        else:
            labeled_name = backend.scratch_dataset_name(base_name, stage, run_slug)
    except backend.LabelingRequestError as e:
        raise DataPreparerOperatorError(str(e)) from None

    # Demos (the dev split only): drawn here, uploaded as demos.jsonl, and left out of
    # this job's gold (the job fails when a demo is an eval row). They stay in the input:
    # a corpus row that is also a demo is still labeled, so the job's labels keep every row.
    demo_ids = set()
    demo_texts = set()
    demos_uri = None
    demos_text = None
    demo_rows = []
    if demos and demos_per_class > 0:
        try:
            seed_rows = read_local_rows(demos, "demos")
            demo_rows = select_demo_rows(seed_rows, task, demos_per_class, label_field, pred_field)
        except (OSError, ValueError) as e:
            raise DataPreparerOperatorError(
                f"cannot build demos from --demos {demos!r}: {e}"
            ) from None
        if not demo_rows:
            raise DataPreparerOperatorError(
                f"--demos {demos!r} has no row labeled under {label_field!r}; a few-shot "
                "job must not run zero-shot. Pass --label-field or drop --demos."
            )
        if len(demo_rows) > backend.MAX_DEMOS:
            raise DataPreparerOperatorError(
                f"{len(demo_rows)} demos; the job takes at most {backend.MAX_DEMOS}. Lower "
                "--demos-per-class."
            )
    if demo_rows:
        # The job matches a demo to an eval row by the demo's own id (no position):
        # a demo without an id matches by text only.
        demo_ids = {str(r["id"]) for r in demo_rows if r.get("id") is not None}
        demo_texts = {text_key(r, text_fields) for r in demo_rows}
        demos_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in demo_rows)
        demos_uri = s3_run + "labeling/demos.jsonl"

    # Gold (EvaluationData): a local split is uploaded as id, gold and the text fields.
    eval_uri = eval_arn = None
    gold_rows = []
    eval_ids = set()
    no_text = set()
    golden_check = None
    n_fit = 0  # recalibration-fit rows in the gold: never corpus rows, never scored
    if eval_data:
        ek, eref = input_kind(eval_data)
        if ek in ("arn", "s3") and golden_review is not None:
            raise DataPreparerOperatorError(
                "--golden-review needs a local --eval: the reviewed golden rows are "
                "checked on the client before upload. Nothing was uploaded."
            )
        if ek in ("arn", "s3"):
            if ek == "arn":
                eval_arn = eref
            else:
                eval_uri = eref
            # The job reads remote gold as it is: the checks below (gold that is a demo,
            # fewer than 50 gold rows, rows with no text field or no id, and the scored
            # count after the run) need the rows on the client.
            if eref == input_ref:
                # One file with labeled and unlabeled rows: the gold is the corpus itself,
                # so it is never downloaded to check it.
                warnings.append(
                    "NOTE: --eval is the --input itself (one file with labeled and unlabeled "
                    "rows). The job scores the rows that carry a gold value against their "
                    "own labels, and those rows ship with the people's labels; it labels "
                    "the rest; its summary then warns that the unlabeled rows had no gold "
                    "value and were not scored, which is expected. This gold is the corpus: "
                    "it is not checked here and is never downloaded to check it. Each row "
                    "needs an id the file itself carries, so a gold row joins its own corpus "
                    "row."
                )
            else:
                warnings.append(
                    f"NOTE: --eval is {'a DataSet' if ek == 'arn' else 'an s3:// object'}, so "
                    "the gold is not checked here (demos among the gold rows, fewer than "
                    f"{backend.MIN_EVAL_ROWS} gold rows, rows with no text field or no id, "
                    "labels outside the label set). "
                    "The gold is seed-sized: to check it, download it with the hub_ops resolve action "
                    "(REF = <it>, OUT = <proj>/work/gold.jsonl, DOWNLOAD = True) and pass "
                    "that file."
                )
        else:
            try:
                eval_rows = read_local_rows(eref, "evaluation_data")
                check_ids(eval_rows)
            except (OSError, ValueError) as e:
                raise DataPreparerOperatorError(f"cannot read --eval {eref!r}: {e}") from None
            golden_check, golden_note, golden_events = check_golden_review(
                golden_review, eref, eval_rows
            )
            events.extend(golden_events)
            if golden_note:
                warnings.append(golden_note)
            gold_prefix = GOLD_ID_PREFIX if is_corpus else None
            int_ids = corpus_ids_are_ints(first_row)
            n_float = float_ids_to_join(eval_rows) if int_ids else 0
            if n_float:
                try:
                    check_ids(
                        [dict(r, id=row_id(r, i, int_ids=True)) for i, r in enumerate(eval_rows)]
                    )
                except ValueError as e:
                    raise DataPreparerOperatorError(
                        f"cannot read --eval {eref!r}: {e} (after writing ids such as 300.0 "
                        "as 300)"
                    ) from None
            _, gold_rows, unlabeled = split_gold(
                eval_rows,
                label_field,
                text_fields=text_fields,
                id_prefix=gold_prefix,
                int_ids=int_ids,
            )
            if n_float:
                warnings.append(
                    f"NOTE: {n_float} --eval row(s) have an id written as a whole-number float "
                    "(such as 300.0, as a pandas export writes it); they are sent as integers "
                    "(300), the way --input's ids are written, so they join their input rows: "
                    "the job compares ids as text."
                )
            no_id = sum(1 for r in eval_rows if r.get("id") is None)
            if gold_prefix and no_id:
                warnings.append(
                    f"NOTE: {no_id} --eval row(s) have no id; they are sent as "
                    f"'{GOLD_ID_PREFIX}<position>', so the job never takes them for corpus "
                    "rows (a corpus row with no id is its position) and labels them to "
                    "score them. If these rows are also --input rows (one file with labeled "
                    "and unlabeled rows), write ids into that file first and pass it again: "
                    "otherwise they are labeled twice and ship the model's labels, not the "
                    "people's (references/full-loop.md, One file with labeled and unlabeled "
                    "rows)."
                )
            if task == "classification":
                gold_warning = check_gold_labels(
                    gold_rows, label_field, label_names, spec_text, eref
                )
                if gold_warning:
                    warnings.append(f"WARNING: {gold_warning}")
            if unlabeled:
                warnings.append(
                    f"NOTE: {len(unlabeled)} --eval row(s) have no {label_field!r} and are not "
                    "gold."
                )
            # A demo must not be an eval row by id or by text: the job fails at load
            # when one is (its gold would be in every prompt).
            twin_ids = {
                (
                    f"{gold_prefix}{i}"
                    if gold_prefix and r.get("id") is None
                    else str(row_id(r, i, int_ids=int_ids))
                )
                for i, r in enumerate(eval_rows)
                if text_key(r, text_fields) in demo_texts
            }
            before = len(gold_rows)
            gold_rows = [
                g for g in gold_rows if g["id"] not in demo_ids and str(g["id"]) not in twin_ids
            ]
            if before != len(gold_rows):
                warnings.append(
                    f"NOTE: {before - len(gold_rows)} gold row(s) left out of the job's gold: "
                    "they are demos, or have a demo's text."
                )
            if not gold_rows:
                raise DataPreparerOperatorError(
                    f"--eval {eref!r} has no gold under {label_field!r} (after leaving out "
                    "the demo rows); pass --label-field or a labeled split"
                )
            eval_ids = {g["id"] for g in gold_rows}
            # A gold row the job finds no input row for is labeled from its text fields to
            # be scored; one without every text field cannot be (it is scored only when it
            # is an input row).
            no_text = {g["id"] for g in gold_rows if any(g.get(f) is None for f in text_fields)}
            option_rows = option_gold(
                gold_rows=gold_rows,
                label_field=label_field,
                text_fields=text_fields,
                optimizer=opt,
                recalibration_fit=recalibration_fit,
                project_dir=project_dir,
                input_data=input_data,
                int_ids=int_ids,
            )
            n_fit = len(option_rows)
            if (
                is_corpus
                and estimated_rows
                and estimated_rows + len(gold_rows) + n_fit > backend.MAX_LABELING_ROWS
            ):
                # The fit rows are never corpus rows (the job refuses one that is), so the
                # job labels every one of them on top of the corpus.
                room = max(backend.MAX_LABELING_ROWS - estimated_rows - n_fit, 0)
                fit_note = (
                    (
                        f", and --recalibration-fit adds {n_fit:,} rows the job labels "
                        "too (never corpus rows)"
                    )
                    if n_fit
                    else ""
                )
                warnings.append(
                    f"WARNING: --input is about {estimated_rows:,} rows by size and --eval has "
                    f"{len(gold_rows):,} gold rows{fit_note}. The job also labels the gold rows "
                    "that are not corpus rows, and refuses the run at load when they and the "
                    f"corpus pass {backend.MAX_LABELING_ROWS:,} rows: at most about {room:,} "
                    "of these gold rows can be outside the corpus (overview.md, Limits)."
                )
            if len(gold_rows) < backend.MIN_EVAL_ROWS and routing:
                warnings.append(
                    f"WARNING: {len(gold_rows)} gold rows; under {backend.MIN_EVAL_ROWS} the job "
                    "certifies nothing (NotCertified, insufficient_eval_rows) and routes every "
                    "row to review."
                )
            if (target or 0) >= backend.WARN_TARGET and len(gold_rows) < backend.WARN_EVAL_ROWS:
                warnings.append(
                    f"WARNING: a target of {target} with {len(gold_rows)} gold rows (under "
                    f"{backend.WARN_EVAL_ROWS}) gives a wide bound."
                )
            if option_rows and task == "classification":
                fit_warning = check_gold_labels(
                    option_rows, label_field, label_names, spec_text, recalibration_fit
                )
                if fit_warning:
                    warnings.append(f"WARNING: {fit_warning}")
            gold_rows += option_rows
            eval_uri = s3_run + "eval/gold.jsonl"
    if is_corpus and routing and (eval_uri or eval_arn):
        # The job scores a gold row against the corpus row with its id when the row's text
        # fields are that row's (or it has none to compare), and labels any other gold row
        # from its text fields, to score it; the client cannot read the corpus to tell
        # which is which, so say both before the run.
        if no_text:
            warnings.append(
                f"WARNING: {len(no_text)} --eval gold row(s) lack a text field "
                f"({', '.join(text_fields)}); the job scores them only if they are corpus "
                "rows. Seed rows labeled outside this corpus must carry every text field."
            )
        warnings.append(
            "NOTE: the job scores each --eval row against the corpus row with the same "
            "id when the gold row has that row's text fields. A gold row whose id is not a "
            "corpus id (a seed labeled outside this corpus), or whose id is a corpus id but "
            "whose text is another row's, "
            "is labeled by the job too, from its text fields, so it can be scored: "
            "it costs its share of the jury's calls and is not in the labeled DataSet, the "
            "review set or the row counts. The job fails when these rows and the corpus pass "
            f"{backend.MAX_LABELING_ROWS:,} rows. Confirm with the user that the seed's ids "
            "are this corpus's ids for the same rows, or ids the corpus does not use: "
            "a gold row with another row's id and no text field to compare is scored against that row, "
            "and ships its gold there."
        )

    # The input: a local split is stripped of gold, then uploaded.
    input_s3 = input_arn = None
    n_rows = None
    input_rows_text = None
    input_file = None
    if kind == "arn":
        input_arn = input_ref
    elif kind == "s3":
        input_s3 = input_ref
    elif is_corpus:
        # The corpus is not client-readable: upload the file as it is. The job takes each
        # row's text fields only, drops input columns named like its own output fields,
        # and refuses a missing text field or a shared id at load.
        input_file = input_ref
        input_s3 = s3_run + "input/" + Path(input_ref).name
    else:
        try:
            rows = read_local_rows(input_ref, "input")
            check_ids(rows)
        except (OSError, ValueError) as e:
            raise DataPreparerOperatorError(f"cannot read --input {input_ref!r}: {e}") from None
        inputs, _, _ = split_gold(rows, label_field, int_ids=corpus_ids_are_ints(first_row))
        if not inputs:
            raise DataPreparerOperatorError(f"--input {input_ref!r} has no rows to label")
        reserved = sorted({k for r in inputs for k in STATED_CONFIDENCE_FIELDS if k in r})
        if reserved:
            # The skill treats both names as model output on every run (the blind
            # fields), whatever the confidence method.
            warnings.append(
                f"WARNING: --input rows have a column named {', '.join(reserved)}. The "
                "skill reads that name as model output on every run: review rows hide "
                "it and a seed merge drops it from the input fields and tags the row as "
                "reviewed. Rename the column before the seed is built."
            )
        outside = eval_ids - {r["id"] for r in inputs}
        if outside:
            unscorable = outside & no_text
            warnings.append(
                f"NOTE: {len(outside)} --eval gold row(s) are not --input rows; the job "
                "labels them from their text fields to score them, and they are not in "
                "this round's labels."
                + (
                    f" {len(unscorable)} of them lack a text field "
                    f"({', '.join(text_fields)}) and cannot be scored."
                    if unscorable
                    else ""
                )
            )
        n_rows = len(inputs)
        input_rows_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in inputs)
        input_s3 = s3_run + "input/" + Path(input_ref).name
    if n_rows is None:
        n_rows = estimated_rows

    spec_uri = s3_run + "labeling/spec.md"
    output_uri = s3_run + "output/"
    try:
        document = backend.build_labeling_request(
            task=task,
            text_fields=text_fields,
            # By name in a dict: the docs guard reads the bare keyword as the retired flat
            # DatasetJobConfig argument.
            **{"spec_s3_uri": spec_uri},
            model_ids=models,
            dataset_name=labeled_name,
            output_s3_uri=output_uri,
            input_s3_uri=input_s3,
            input_dataset_arn=input_arn,
            demos_s3_uri=demos_uri,
            eval_s3_uri=eval_uri,
            eval_dataset_arn=eval_arn,
            label_field=label_field if (eval_uri or eval_arn) else None,
            target=target,
            review_dataset_name=review_dataset_name,
            graders=grader_ids or None,
            hub_name=hub_name,
            confidence_method=confidence_method,
            confidence_estimator=confidence_estimator,
            judge_models=judges,
            confidence_recalibration=confidence_recalibration,
            confidence_floor_bypass=bool(confidence_floor_bypass),
            optimizer=opt,
        )
    except backend.LabelingRequestError as e:
        raise DataPreparerOperatorError(
            f"labeling request refused before anything was uploaded: {e}"
        ) from None
    request = backend.request_summary(document)

    # The client checks passed and the request is built: the test rows this run read go to
    # the run log. A run the resume or dataset-exists check stops, and a re-attach, read
    # them too, so each is logged (the log records reads, not submissions).
    for path, step, n in test_reads:
        events.append(split_state.test_read_event(project_dir, path, step, n))
    # A re-run with the same run id and stage re-attaches to its job (job.json) when the
    # request and its files are unchanged; submit_round checks that before any upload.
    gold_text = (
        "".join(json.dumps(g, ensure_ascii=False) + "\n" for g in gold_rows) if gold_rows else None
    )
    req_sha = request_sha256(
        document,
        spec_text=spec_text,
        demos_text=demos_text if demos_uri else None,
        gold_text=gold_text,
        input_text=input_rows_text,
        input_file=input_file,
    )
    # The option fields this request sends: a CreateJob refusal naming one is that option's.
    sent = [
        k
        for k, given in (("Confidence", confidence_method), ("Optimizer", optimizer))
        if given is not None
    ]
    invocation = {
        "spec": spec,
        "input": input_data,
        "out": str(out_path),
        "project_dir": os.path.abspath(project_dir),
        "stage": raw_stage,
        "dataset_name": dataset_name,
        "generation_model_id": generation_model_id,
        "jury_models": jury_models,
        "task": task,
        "text_fields": raw_text_fields,
        "label_set": label_set,
        "eval": eval_data,
        "label_field": label_field_given,
        "demos": demos,
        "demos_per_class": demos_per_class,
        "target": target,
        "review_dataset_name": review_dataset_name,
        "graders": graders,
        "hub_name": hub_name,
        "confirm_no_seed": confirm_no_seed,
        "append": append,
        "s3_prefix": s3_prefix,
        "region": region,
        "role_arn": role_arn,
        "run_id": run_id,
        "submit_max_wait": submit_max_wait,
    }
    options = {
        "confidence_method": confidence_method,
        "golden_review": golden_review,
        "confidence_estimator": confidence_estimator,
        "judge_models": judge_models,
        "confidence_recalibration": confidence_recalibration,
        "recalibration_fit": recalibration_fit,
        "optimizer": optimizer,
        "confidence_floor_bypass": confidence_floor_bypass,
    }
    for key in OPTION_INVOCATION_KEYS:
        if options[key] is not None:
            invocation[key] = options[key]
    return RoundPlan(
        stage=stage,
        is_corpus=is_corpus,
        routing=routing,
        task=task,
        label_field=label_field,
        pred_field=pred_field,
        text_fields=text_fields,
        models=models,
        jury=jury,
        dataset_name=labeled_name,
        run_id=run_id,
        explicit_run_id=explicit_run_id,
        out=str(out_path),
        project_dir=project_dir,
        staging_dir=str(work),
        request_path=str(work / "request.json"),
        job_record_path=str(work / "job.json"),
        document=document,
        request=request,
        request_text=json.dumps(document, indent=2) + "\n",
        request_sha256=req_sha,
        spec_text=spec_text,
        spec_uri=spec_uri,
        output_uri=output_uri,
        demos_text=demos_text,
        demos_uri=demos_uri,
        gold_text=gold_text,
        gold_rows=gold_rows,
        eval_uri=eval_uri,
        eval_arn=eval_arn,
        input_rows_text=input_rows_text,
        input_file=input_file,
        input_s3=input_s3,
        input_arn=input_arn,
        n_rows=n_rows,
        n_fit=n_fit,
        estimated_rows=estimated_rows,
        unpassed_fields=unpassed_fields,
        labels_missing=labels_missing,
        seed_evidence=evidence,
        golden_check=golden_check,
        region=region,
        role_arn=role_arn,
        hub_name=hub_name,
        review_dataset_name=review_dataset_name,
        append=append,
        submit_max_wait=submit_max_wait,
        option_fields=sent,
        optimizer=optimizer,
        invocation=invocation,
        events=list(events),
        warnings=list(warnings),
    )


def submit_round(  # pylint: disable=too-many-branches
    plan: RoundPlan,
    *,
    on_submit: Optional[Callable[[str], None]] = None,
    on_refused: Optional[Callable[[], None]] = None,
    endpoint: Optional[str] = None,
    wait: bool = True,
) -> SubmittedRound:
    """Re-attach to the round's job, or upload its inputs and submit it.

    A run with an explicit run id whose ``plan.job_record_path`` holds a job record with
    the same request digest re-attaches to that job; a different digest is refused. A
    corpus run that is not a re-attach checks first that its DataSet names are free
    (unless ``plan.append``). A job that is submitted (not re-attached) first has its spec,
    demos, gold and input uploaded; then it is waited on.

    With ``wait=False`` it returns once the job is submitted, or once a re-attach finds the
    job still running; :attr:`SubmittedRound.running` is then ``True``, and the caller ends
    its turn and later calls it again under the same run id (``plan.explicit_run_id``) to
    collect the job before :func:`read_round`. A re-attach reads the job's status from
    ``DescribeJob``, never from the job record.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import round as labeling_round

        submitted = labeling_round.submit_round(
            plan,
            on_submit=lambda text: open(plan.job_record_path, "w").write(text),
            on_refused=lambda: os.remove(plan.job_record_path),
        )
        submitted.job["job_name"], submitted.job["status"]

    Args:
        plan: :func:`prepare_round`'s plan; its ``events`` already appended.
        on_submit: Called with the job record's text before ``CreateJob``; the caller
            writes it to ``plan.job_record_path``.
        on_refused: Called when ``CreateJob`` made no job; the caller deletes the record
            ``on_submit`` wrote, so a rerun under the same run id is not refused as a
            changed request.
        endpoint: The SageMaker endpoint override (``AWS_ENDPOINT_URL_SAGEMAKER``), named
            in a refusal.
        wait: ``True`` (the default) waits until the job ends; ``False`` returns while it
            runs (fire-and-forget).

    Returns:
        A :class:`SubmittedRound` with the job record.

    Raises:
        DataPreparerOperatorError: A changed request under the run id, a taken DataSet
            name, an upload or ``CreateJob`` refusal, a job that could not be waited on,
            or one that ended Failed or Stopped.
    """
    stage = plan.stage
    job_record_path = Path(plan.job_record_path)
    resume = None
    if plan.explicit_run_id and job_record_path.exists():
        try:
            resume = json.loads(job_record_path.read_text())
        except ValueError:
            resume = None
        if resume and resume.get("request_sha256") != plan.request_sha256:
            raise DataPreparerOperatorError(
                f"--run-id {plan.run_id!r} --stage {stage!r} already names a job submitted with a "
                f"different request or different spec, demos, gold or input content "
                f"({job_record_path}); use another --run-id or --stage, or delete that "
                "record to resubmit."
            )
    if plan.is_corpus and not resume and not plan.append:
        # Hub import adds a version to a name that exists, so a taken dataset name or
        # review dataset name stops the run. A re-attach is this run's own job, so it is
        # not checked.
        try:
            taken = register.existing_datasets(
                [plan.dataset_name, plan.review_dataset_name], plan.region, plan.hub_name
            )
        except Exception as e:  # noqa: BLE001 - cannot tell: do not risk a silent append
            raise DataPreparerOperatorError(
                f"could not check whether --dataset-name / --review-dataset-name already "
                f"exist ({type(e).__name__}: {e}); nothing was uploaded or submitted."
            ) from None
        if taken:
            shown = "; ".join(
                f"{t['name']} (version {t['version']} in hub {t['hub']})" for t in taken
            )
            raise DataPreparerOperatorError(
                f"already registered: {shown}. The corpus job would add these rows as the "
                "next version of that DataSet. Ask the user: pick another --dataset-name / "
                "--review-dataset-name, or pass --append to add a version on purpose. "
                "Nothing was uploaded or submitted."
            )

    def _upload():
        # Only a new submission uploads: a re-attach's request digest already matched,
        # so the same spec, demos, gold and input are at these URIs.
        try:
            job_client.upload_text(
                plan.spec_text,
                plan.spec_uri,
                plan.region,
                field="DatasetSpecification.SpecS3Uri (the spec)",
                max_bytes=backend.MAX_SPEC_BYTES,
            )
            if plan.demos_uri:
                job_client.upload_text(
                    plan.demos_text,
                    plan.demos_uri,
                    plan.region,
                    field="Demos (demos.jsonl)",
                    max_bytes=backend.MAX_DEMOS_BYTES,
                )
            if plan.gold_text is not None:
                job_client.upload_text(
                    plan.gold_text, plan.eval_uri, plan.region, field="EvaluationData"
                )
            if plan.input_rows_text is not None:
                job_client.upload_text(
                    plan.input_rows_text, plan.input_s3, plan.region, field="InputDataS3Uri"
                )
            if plan.input_file is not None:
                _s3.upload_file(plan.input_file, plan.input_s3, plan.region)
        except backend.LabelingRequestError as e:
            raise DataPreparerOperatorError(str(e)) from None

    def _on_submit(rec):
        rec = dict(
            rec,
            request_sha256=plan.request_sha256,
            stage=stage,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        if on_submit is not None:
            on_submit(json.dumps(rec, indent=2) + "\n")

    try:
        job = job_client.submit_labeling_job(
            plan.document,
            region=plan.region,
            role_arn=plan.role_arn,
            base_job_name=f"autolabel-{stage}",
            submit_max_wait_s=plan.submit_max_wait,
            on_submit=_on_submit,
            resume_job=resume,
            endpoint=endpoint,
            option_fields=plan.option_fields,
            wait=wait,
            before_submit=_upload,
        )
    except (job_client.CreateJobRefused, job_client.LabelingBackendUnavailable) as e:
        # CreateJob made no job: the record on_submit wrote goes, so a rerun under the same
        # run id (without the option) is not refused as a changed request.
        if on_refused is not None:
            on_refused()
        if isinstance(e, job_client.OptionRefused):
            raise DataPreparerOperatorError(
                f"CreateJob refused the {stage} request; the service says: {e}\nNothing was "
                "labeled. Rerun without the option it names (references/troubleshooting.md, "
                "Option refused by the service)."
            ) from None
        raise DataPreparerOperatorError(str(e)) from None
    except RuntimeError as e:
        # describe/wait failures (JobWaitError: not resubmitting, so no duplicate job) and an
        # SDK wait timeout (re-attach with the same run id)
        raise DataPreparerOperatorError(
            f"SDG labeling job ({stage}) did not complete: {e}"
        ) from None
    if job.get("job_status") in ("Failed", "Stopped"):
        raise DataPreparerOperatorError(
            f"SDG labeling job {job['job_name']} ended {job['job_status']}: "
            f"{job.get('failure_reason') or 'no failure reason returned'}. Read the job's "
            "CloudWatch log for the container's message; the run spent nothing beyond the "
            "job. Re-run with the same --run-id and --stage after fixing the input."
        )
    return SubmittedRound(job=job)


def read_round(  # noqa: C901 pylint: disable=too-many-locals,too-many-branches,too-many-statements
    plan: RoundPlan,
    job: Dict[str, Any],
    *,
    run_log_step: str = "auto-label",
    run_log_iteration: int = 0,
) -> RoundOutputs:
    """Read the round's result back through the labeled DataSet and return what the round writes.

    The job's artifacts are downloaded under ``<folder of plan.out>/job/<stage>``; that is
    the only write. A refine round reads its labels (the seed's dev / val rows) and
    returns them for ``plan.out``; the corpus run never downloads the labels: its
    deliverable record names the labeled DataSet. Beside ``plan.out``: the summary, the
    failures, the held-out report and, on the corpus run, the review rows' side file and
    the job's report.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import round as labeling_round

        outputs = labeling_round.read_round(plan, submitted.job)
        for path, text in outputs.texts.items():
            open(path, "w").write(text)
        run_log.extend(outputs.record.jobs + [outputs.record.artifact])

    Args:
        plan: :func:`prepare_round`'s plan.
        job: :func:`submit_round`'s job record.
        run_log_step: The run log's step for the run's record.
        run_log_iteration: The run log's iteration for the run's record.

    Returns:
        A :class:`RoundOutputs`. The caller writes ``texts`` and ``copies``, deletes
        ``removes``, appends ``record``'s entries, and reports
        ``round_summary(outputs, run_log_entry=<the artifact entry's id>)``.

    Raises:
        DataPreparerOperatorError: The job is still running, or the result could not be
            read, or names no labels.
    """
    if job.get("job_status") is not None and job["job_status"] not in _ENDED_JOB_STATUSES:
        raise DataPreparerOperatorError(
            f"SDG labeling job {job.get('job_name')} is still {job.get('job_status')}; "
            "nothing to read yet. Call submit_round again under the same run id to "
            "re-attach (it submits nothing new), then read_round once the job has ended."
        )
    warnings: List[str] = []
    texts: Dict[str, str] = {}
    copies: Dict[str, str] = {}
    removes: List[str] = []
    stage, is_corpus, routing = plan.stage, plan.is_corpus, plan.routing
    task, pred_field, models = plan.task, plan.pred_field, plan.models
    out = Path(plan.out)
    artifacts_dir = out.parent / "job" / stage
    try:
        results = job_client.read_job_results(
            region=plan.region,
            dataset_arn=job.get("dataset_arn"),
            dest_dir=artifacts_dir,
            names=CORPUS_ARTIFACTS if is_corpus else None,
            review_dataset_arn=job.get("review_dataset_arn"),
        )
    except (FileNotFoundError, ValueError) as e:
        raise DataPreparerOperatorError(
            f"SDG job {job['job_name']} completed but its labeling result could not be "
            f"read: {e}"
        ) from None
    files = results["files"]
    summary = results["summary"] or {}
    uris = results.get("artifact_uris") or {}
    if not uris.get("Labels"):
        raise DataPreparerOperatorError(
            f"the job's index {results['index_uri']} names no labels file"
        )
    side_rows = []
    if files.get("ReviewModelOutput"):
        side_rows = review_model_outputs(files["ReviewModelOutput"])
    if is_corpus:
        # The labeled corpus stays behind the API: the deliverable is the labeled DataSet
        # (and the labels' S3 URI), never a local copy of its rows.
        labels = None
        n = (summary.get("RowsLabeled") or 0) + (summary.get("RowsMissing") or 0)
        # A row with no usable answer is routed to review, so the side file counts them;
        # with review off nothing local shows them.
        failed = sum(1 for r in side_rows if r.get(pred_field) is None) if routing else None
    else:
        with open(files["Labels"], encoding="utf-8") as fh:
            label_rows = [json.loads(line) for line in fh if line.strip()]
        labels = labels_from_job(label_rows, task)
        # A refine round labels the seed's dev / val rows: client-readable.
        report_client_read(
            f"labeling_round (the {stage} round's labels, seed rows)", len(label_rows)
        )
        texts[str(out)] = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels)
        n = len(labels)
        failed = sum(1 for r in labels if r.get("_failed"))
    scored = summary.get("HeldOutRows")
    n_gold = sum(
        1 for g in plan.gold_rows if not g.get("recalibration_fit")
    )  # fit rows are never scored
    if is_corpus and n_gold and isinstance(scored, int) and scored < n_gold:
        warnings.append(
            f"WARNING: the job scored {scored} of the {n_gold} --eval gold rows; the "
            f"other {n_gold - scored} were not scored (no text field to label them "
            "by, a labeling call that returned nothing, or a gold value the job reads as "
            "model-labeled). The job's warnings in labeling_summary.json name the reason."
        )
    if labels is not None:
        shortfall = juror_shortfall(labels, summary.get("JurorModelIds"))
    else:
        # The corpus run sees only the review rows' ballots (the side file): a juror
        # missing mostly on accepted rows is not caught. The job's summary does not
        # count missing ballots per juror.
        shortfall = juror_shortfall(side_rows, summary.get("JurorModelIds"))
        if shortfall:
            shortfall["scope"] = "review rows only"
    jury_check = None
    if is_corpus:
        # Say what the juror check covered, even when it flags nothing: a clean run
        # must not read as every row checked.
        accepted = summary.get("RowsAccepted")
        if not routing:
            if len(plan.jury) > 1:
                jury_check = (
                    "no juror check: without --review-dataset-name the job runs "
                    "one model and routes nothing, so there are no ballots to count"
                )
        elif len(models) > 1 and shortfall:
            jury_check = (
                f"juror check covered the {shortfall['rows']} review rows only; "
                + (
                    f"the {accepted} auto-accepted rows were not checked "
                    if isinstance(accepted, int)
                    else "the auto-accepted rows were not checked "
                )
                + "(their ballots stay behind the API)"
            )
        elif len(models) > 1:
            jury_check = (
                "no juror check: the job routed no voted rows to review, so there "
                "are no ballots on the client to count"
            )
        if jury_check:
            warnings.append(f"NOTE: {jury_check}.")
    degraded = (shortfall or {}).get("degraded_jurors") or {}
    for m, d in degraded.items():
        warnings.append(
            f"WARNING: JURY DEGRADED: juror {m} cast no usable ballot on "
            f"{d['missing']}/{d['rows']} {'review ' if 'review' in shortfall.get('scope', '') else ''}"
            f"rows ({100 * d['missing_rate']:.0f}%); those rows "
            "were voted by the other jurors only. The job does not say why a reply was "
            "dropped; a model refusal is the usual cause. Consider another juror."
        )
    verbalized_missing = verbalized_confidence_missing(summary)
    if verbalized_missing and verbalized_missing["high"]:
        warnings.append(
            f"WARNING: VERBALIZED CONFIDENCE MISSING on {verbalized_missing['missing']}/"
            f"{verbalized_missing['of']} rows ({100 * verbalized_missing['rate']:.0f}%). A "
            "missing confidence_score counts 0, so with the 0.75 floor on (the default) "
            "those rows never auto-accept (with --confidence-recalibration the floor "
            "reads the mapped value). Check "
            "that the spec does not tell the model to leave out the confidence, or swap "
            "the model that drops it."
        )
    stated_missing = stated_confidence_missing(summary)
    if stated_missing and stated_missing["high"]:
        warnings.append(
            f"WARNING: STATED CONFIDENCE MISSING on {stated_missing['missing']}/"
            f"{stated_missing['of']} {stated_missing['unit']} "
            f"({100 * stated_missing['rate']:.0f}%). A missing stated confidence counts 0 "
            "in the row score, so fewer rows auto-accept than the spec deserves. Check "
            "that the spec does not tell the models to leave out the confidence, or "
            "swap the model that drops it."
        )

    # Beside the output: the summary, the failures (the refiner's error view), the held-out
    # report in the skill's shape, and on the corpus run the review rows' side file the
    # fold reads (review.model_output.jsonl) and the job's report.html.
    stem = out.with_suffix("")
    written = {} if is_corpus else {"labels": str(out)}
    if files.get("LabelingSummary"):
        p = Path(f"{stem}.summary.json")
        texts[str(p)] = json.dumps(summary, indent=2) + "\n"
        written["summary"] = str(p)
    if files.get("Failures"):
        p = Path(f"{stem}.failures.jsonl")
        copies[str(p)] = files["Failures"]
        written["failures"] = str(p)
    job_report = {}
    if files.get("HeldOutReport"):
        with open(files["HeldOutReport"], encoding="utf-8") as fh:
            job_report = json.load(fh)
    hor = held_out_report_from_job(summary, job_report, task)
    outside = job_report.get("eval_rows_labeled_for_scoring") if is_corpus else None
    if isinstance(outside, int) and outside and plan.n_fit:
        # The job's count can take in the fit rows: they are never corpus rows either.
        warnings.append(
            f"NOTE: the job labeled {outside} gold rows that are not corpus rows: up to "
            f"{plan.n_fit} of them are the --recalibration-fit rows (labeled to fit the map, "
            "never scored), the rest --eval gold rows labeled to score them. None are in "
            "the labeled DataSet or the review set."
        )
    elif isinstance(outside, int) and outside:
        warnings.append(
            f"NOTE: {outside} --eval gold rows are not corpus rows; the job labeled them to "
            "score them, and they are not in the labeled DataSet or the review set."
        )
    if hor is not None:
        p = Path(f"{stem}.held_out_report.json")
        texts[str(p)] = json.dumps(hor, indent=2) + "\n"
        written["held_out_report"] = str(p)
    if is_corpus:
        texts[str(out.parent / "labeling_summary.json")] = json.dumps(summary, indent=2) + "\n"
        written["labeling_summary"] = str(out.parent / "labeling_summary.json")
        if hor is not None:
            texts[str(out.parent / "held_out_report.json")] = json.dumps(hor, indent=2) + "\n"
            written["held_out_report"] = str(out.parent / "held_out_report.json")
        if files.get("ReviewModelOutput"):
            side = out.parent / "review.model_output.jsonl"
            copies[str(side)] = files["ReviewModelOutput"]
            written["review_model_output"] = str(side)
            # The side file is seed-sized (one line per review row); the read is reported
            # like the seed reads.
            report_client_read("labeling_round (the review rows' model answers)", len(side_rows))
            # A review.jsonl an earlier run left is not this job's review set, and an
            # overlay with it would score that job's spot-check rows in this verdict.
            from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.labels import (
                known_spot_check_path,
            )

            stale = out.parent / "review.jsonl"
            removes.append(str(Path(known_spot_check_path(str(stale)))))
            removes.append(str(stale))
        if files.get("Report"):
            # The job's own report (built from its summary, no rows).
            page = out.parent / "job_report.html"
            copies[str(page)] = files["Report"]
            written["job_report"] = str(page)
        deliverable = {
            "dataset_name": plan.dataset_name,
            "dataset_arn": results.get("labeled_dataset_arn"),
            "review_dataset_name": plan.review_dataset_name,
            "review_dataset_arn": results.get("review_dataset_arn"),
            "labels_s3_uri": uris.get("Labels"),
            "review_rows_s3_uri": uris.get("ReviewRows"),
            "report_s3_uri": uris.get("Report"),
            "index_uri": results.get("index_uri"),
            "job_name": job.get("job_name"),
            "job_arn": job.get("job_arn"),
            "region": plan.region,
            "task": task,
            "rows": n,
            "status": summary.get("Status"),
            "status_reason": summary.get("StatusReason"),
            "spec_sha256": summary.get("SpecSha256"),
            "run_id": plan.run_id,
            # The eval rows the job found were not corpus rows (another id, or a corpus
            # id with another row's text) and labeled only to score them: the overlay
            # leaves the seed's own rows out when this is above 0.
            "eval_rows_labeled_for_scoring": outside if isinstance(outside, int) else None,
            # The seed's gold column: the overlay reads a seed file's people's labels from
            # it when the file has no gold column of its own.
            "label_field": plan.label_field,
            "note": (
                "The labeled rows stay behind the API: downstream steps take dataset_arn. "
                "Corrections are applied by a dataset-transformation overlay by id, "
                "whose output becomes this DataSet's next version "
                "(references/full-loop.md, Step 7b)."
            ),
        }
        texts[str(out)] = json.dumps(deliverable, indent=2) + "\n"
        written["deliverable"] = str(out)

    # The run log: one job entry, one artifact entry (status scratch for a working job).
    scratch = not is_corpus
    record = None
    run_log_error = None
    try:
        manifest = build_job_manifest(
            task=task,
            n_rows=n,
            stage=stage,
            region=plan.region,
            job=job,
            results=results,
            request=plan.request,
            labels_path=(
                (results.get("labeled_dataset_arn") or uris.get("Labels"))
                if is_corpus
                else str(out)
            ),
            spec_uri=plan.spec_uri,
            input_ref=plan.input_s3 or plan.input_arn,
            eval_uri=plan.eval_uri or plan.eval_arn,
            demos_uri=plan.demos_uri,
            scratch=scratch,
        )
        manifest["invocation"] = dict(plan.invocation)
        manifest["seed_evidence"] = plan.seed_evidence
        manifest["unrefined_spec"] = plan.seed_evidence is None
        if plan.labels_missing:
            manifest["labels_missing_from_spec"] = plan.labels_missing
        if shortfall:
            manifest["confidence_shortfall"] = shortfall
        if jury_check:
            manifest["jury_check"] = jury_check
        manifest["dataset_arn"] = results.get("labeled_dataset_arn")
        manifest["review_dataset_arn"] = results.get("review_dataset_arn")
        manifest["job_arn"] = job.get("job_arn")
        manifest["index_uri"] = results.get("index_uri")
        manifest["request"] = plan.request
        record = labeling_run_record(
            manifest,
            step=run_log_step,
            iteration=run_log_iteration,
            status="scratch" if scratch else "labeled",
            arn=results.get("labeled_dataset_arn"),
            job_arn=job.get("job_arn"),
        )
    except Exception as e:  # noqa: BLE001 - the record is a report aid, not the deliverable
        run_log_error = f"{type(e).__name__}: {e}"

    out_summary = {
        "stage": stage,
        "scratch": scratch,
        "rows": n,
        "failed_rows": failed,
        "output": str(out),
        "files": written,
        "job_name": job.get("job_name"),
        "job_arn": job.get("job_arn"),
        "job_status": job.get("job_status"),
        "reattached": job.get("reattached"),
        "dataset_name": plan.dataset_name,
        "dataset_arn": results.get("labeled_dataset_arn"),
        "review_dataset_arn": results.get("review_dataset_arn"),
        "review_rows_uri": (results.get("artifact_uris") or {}).get("ReviewRows"),
        "labels_s3_uri": uris.get("Labels"),
        "index_uri": results.get("index_uri"),
        "status": summary.get("Status"),
        "status_reason": summary.get("StatusReason"),
        "held_out_accuracy": summary.get("HeldOutAccuracy"),
        "held_out_lower_bound_95": summary.get("HeldOutLowerBound95"),
        "held_out_rows": summary.get("HeldOutRows"),
        "accepted_lower_bound_95": summary.get("AcceptedLowerBound95"),
        "accept_threshold": summary.get("AcceptThreshold"),
        "rows_accepted": summary.get("RowsAccepted"),
        "rows_routed": summary.get("RowsRouted"),
        "rows_human_labeled": summary.get("RowsHumanLabeled"),
        "rows_missing": summary.get("RowsMissing"),
        "spot_check_rows": summary.get("SpotCheckRows"),
        "spot_check_withdraw_at": summary.get("SpotCheckWithdrawAt"),
        "failure_rows": summary.get("FailureRows"),
        "good_rate": summary.get("GoodRate"),
        "graded_rows": summary.get("GradedRows"),
        "macro_f1": job_report.get("macro_f1"),
        "spec_sha256": summary.get("SpecSha256"),
        "juror_model_ids": summary.get("JurorModelIds"),
        "warnings": summary.get("Warnings") or [],
        "estimated_rows": plan.estimated_rows,
        "run_id": plan.run_id,
        "eval_rows_labeled_for_scoring": outside if isinstance(outside, int) else None,
    }
    if plan.unpassed_fields:
        out_summary["unpassed_text_fields"] = plan.unpassed_fields
    for key, out_key in OPTION_SUMMARY_KEYS.items():
        if key in summary:
            out_summary[out_key] = summary[key]
    if plan.optimizer is not None and files.get("Spec"):
        # What the user approves: the pick's bytes and the job that wrote it (adopt_pick).
        pick = {
            "path": os.path.abspath(files["Spec"]),
            "job_name": job.get("job_name"),
            "sha256": _digest.sha256_file(files["Spec"]),
        }
        texts[str(Path(files["Spec"]).with_name("pick.json"))] = json.dumps(pick, indent=2) + "\n"
        out_summary["optimizer_pick"] = pick
    if stated_missing:
        out_summary["stated_confidence_missing_rate"] = stated_missing["rate"]
    if verbalized_missing:
        out_summary["verbalized_confidence_missing_rate"] = verbalized_missing["rate"]
    if plan.seed_evidence is None:
        out_summary["unrefined_spec"] = True
    if plan.labels_missing:
        out_summary["labels_missing_from_spec"] = plan.labels_missing
    if degraded:
        out_summary["degraded_jurors"] = degraded
    if jury_check:
        out_summary["jury_check"] = jury_check
    if plan.golden_check:
        out_summary["golden_review"] = plan.golden_check
    if failed:
        out_summary["note"] = (
            f"{failed} row(s) have a null prediction (the labeling call "
            "returned no usable answer); flagged _failed and routed to review "
            "by the job."
        )
    if is_corpus:
        out_summary["deliverable"] = (
            f"the labeled DataSet {results.get('labeled_dataset_arn')} (labels at "
            f"{uris.get('Labels')}); the labeled rows were not downloaded"
        )
    if summary.get("Status") == "Partial":
        warnings.append(
            f"WARNING: the job is Partial: {summary.get('RowsMissing')} row(s) have no labeled "
            "output and were routed to review; the certificate scored only the rows present."
        )
    for w in summary.get("Warnings") or []:
        warnings.append(f"WARNING (job): {w}")
    return RoundOutputs(
        texts=texts,
        copies=copies,
        removes=removes,
        record=record,
        run_log_error=run_log_error,
        summary=out_summary,
        warnings=warnings,
    )


def round_summary(
    outputs: RoundOutputs,
    run_log_entry: Optional[str] = None,
    run_log_error: Optional[str] = None,
) -> Dict[str, Any]:
    """The round's report, with the run-log entry id (or the error that kept the record out).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import round_summary

        summary = round_summary(outputs, run_log_entry=entry_id)
        print(json.dumps(summary, indent=2))

    Args:
        outputs: :func:`read_round`'s outputs.
        run_log_entry: The id of the run's artifact entry, once the caller appended it.
        run_log_error: Why the caller could not append the record (``<type>: <message>``).

    Returns:
        The summary dict, in the command line's key order.
    """
    error = outputs.run_log_error or run_log_error
    out: Dict[str, Any] = {}
    placed = False
    for key, value in outputs.summary.items():
        if not placed and key in _SUMMARY_KEYS_AFTER_RUN_LOG:
            placed = True
            _put_run_log(out, run_log_entry, error)
        out[key] = value
    if not placed:
        _put_run_log(out, run_log_entry, error)
    return out


def _put_run_log(out, run_log_entry, run_log_error):
    if run_log_entry:
        out["run_log_entry"] = run_log_entry
    if run_log_error:
        out["run_log_error"] = run_log_error


# ---------------------------------------------------------------------------
# The project-level steps of full-loop.md, as functions over the operators above. They
# own the project folder: the round's files, the job record and the run log are written
# here, exactly as the skill's labeling_round template writes them (Cells 2-4).
#
#   run_labeling_round        Step 2, one job: dev, val (or optimize) on a frozen split
#   tune_labeling_spec        Steps 2-5 as an iterator: dev + val per spec version, the scores,
#                      the refine-or-stop routing; the person edits the spec between calls
#   label_corpus       Step 7: the corpus run, certified on the test split
#
# select_labeling_spec (Steps 5-6, which spec ships) is spec/select.py's.
# ---------------------------------------------------------------------------

DEFAULT_LABELING_MODEL = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
KEEP_RUN_SCOPES = 10
# refinement-routing.md: the gate by task (the val accuracy that accepts a spec) and the
# val-size floor under which no refinement is attempted.
DEFAULT_GATE = {"classification": 0.85}
VAL_ROWS_PER_CLASS_FLOOR = 5
CONFUSED_VAL_ROWS_TO_REFINE = 5
# full-loop.md Step 6: a test score this far below the selecting val score is a review flag.
HELDOUT_REVIEW_TOL = 0.10


class RoundResult(OperatorResult):
    """What one :func:`run_labeling_round` left: the files and the job's figures.

    ``out`` is the predictions (or, on the corpus run, the deliverable record);
    ``held_out_report`` and ``failures`` are the files beside it; ``summary`` is
    :func:`round_summary`'s dict (``held_out_accuracy``, ``macro_f1``,
    ``dataset_arn``, ...); ``accuracy`` is the job's ``HeldOutAccuracy``.
    """

    stage: str
    iteration: int
    spec: str
    out: str
    run_id: str
    job_name: Optional[str] = None
    accuracy: Optional[float] = None
    lower_bound_95: Optional[float] = None
    macro_f1: Optional[float] = None
    breakdown: Dict[str, Any] = Field(default_factory=dict)
    confusion_pairs: List[Any] = Field(default_factory=list)
    held_out_report: Optional[str] = None
    failures: Optional[str] = None
    dataset_arn: Optional[str] = None
    review_dataset_arn: Optional[str] = None
    summary: Dict[str, Any] = Field(default_factory=dict)
    run_log_entry: Optional[str] = None


def _project_run_log(project_dir):
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log

    return _run_log


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def run_labeling_round(  # pylint: disable=too-many-locals,too-many-arguments,too-many-statements
    project_dir: str,
    data,
    *,
    spec: str,
    stage: str,
    s3_prefix: str,
    role_arn: str,
    input_data: Optional[str] = None,
    eval_data: Optional[str] = None,
    out: Optional[str] = None,
    iteration: int = 0,
    run_id: Optional[str] = None,
    generation_model_id: str = DEFAULT_LABELING_MODEL,
    demos_per_class: int = 2,
    dataset_name: Optional[str] = None,
    submit_max_wait: Optional[int] = None,
    **options: Any,
) -> RoundResult:
    """One labeling job on a split, as the skill's labeling round runs it (full-loop.md Step 2).

    ``stage`` ``dev`` or ``val`` labels that split against its own gold
    (``--input <split> --eval <split>``), with few-shot demos drawn from dev
    (``--demos splits/dev.jsonl --demos-per-class k``) and the output at
    ``work/<stage>_pred_iter_<N>.jsonl``; ``optimize`` and ``corpus`` take their input
    and gold from the caller. The round is :func:`prepare_round`, :func:`submit_round`
    and :func:`read_round` with the project's files written around them: the request,
    the job record before ``CreateJob`` (so a timed-out wait re-attaches under the same
    run id), the outputs, and the run-log record.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import run_labeling_round

        dev = run_labeling_round("tickets", data, spec="tickets/labeling_spec.md", stage="dev",
                          s3_prefix="s3://bucket/autolabel", role_arn=role_arn)
        dev.accuracy, dev.macro_f1, dev.failures   # the job's HeldOutAccuracy on dev; the error view

    Args:
        project_dir: The project (its ``splits/``, ``work/`` and run log).
        data: :func:`~..inspect.inputs.inspect_labeling_data`'s report.
        spec: The labeling spec file.
        stage: ``dev``, ``val``, ``optimize`` or ``corpus``.
        s3_prefix: The run's S3 prefix.
        role_arn: The job's role.
        input_data: The rows to label; the split for ``dev`` / ``val``.
        eval_data: The gold; the split for ``dev`` / ``val``.
        out: The output; ``work/<stage>_pred_iter_<N>.jsonl`` for ``dev`` / ``val``.
        iteration: The refinement iteration (``iter_0`` is the initial spec).
        run_id: Names the run; the same id re-attaches. When ``None``, ``<stage>-iter<N>-<hash>``
            of the spec, model, demos, dataset name and inputs, so an unchanged round
            re-attaches and a changed one is a new run.
        generation_model_id: The labeling model.
        demos_per_class: Demos per class drawn from dev (0 = zero-shot).
        dataset_name: The scratch DataSet base name; the corpus run's deliverable name.
        submit_max_wait: Seconds ``CreateJob`` may back off on throttling.
        **options: :func:`prepare_round`'s other keyword arguments (``jury_models``,
            ``target``, ``optimizer``, ``review_dataset_name``, ``confirm_no_seed``, ...).

    Returns:
        A :class:`RoundResult`.

    Raises:
        DataPreparerOperatorError: Any refusal of the three operators; a refusal after a
            split read (:class:`RoundRefused`) has its read logged first.
    """
    run_log = _project_run_log(project_dir)
    splits_dir = os.path.join(project_dir, "splits")
    if stage in ("dev", "val"):
        split_path = os.path.join(splits_dir, f"{stage}.jsonl")
        input_data = input_data or split_path
        eval_data = eval_data or split_path
        out = out or os.path.join(project_dir, "work", f"{stage}_pred_iter_{iteration}.jsonl")
    if input_data is None or out is None:
        raise DataPreparerOperatorError(f"stage {stage!r} needs input_data and out")
    demos = os.path.join(splits_dir, "dev.jsonl") if demos_per_class > 0 else None
    if (
        run_id is None
    ):  # the same spec, model, demos and dataset name re-attach; a change is a new run
        key = _digest.sha256_json(
            [
                _digest.sha256_file(spec),
                generation_model_id,
                demos_per_class,
                dataset_name,
                input_data,
                eval_data,
            ]
        )
        run_id = f"{stage}-iter{iteration}-{key[:8]}"
    try:
        plan = prepare_round(
            spec=spec,
            input_data=input_data,
            out=out,
            project_dir=project_dir,
            stage=stage,
            generation_model_id=generation_model_id,
            dataset_name=dataset_name,
            task=data.task,
            text_fields=",".join(data.text_fields),
            label_set=json.dumps(data.labels) if data.labels else None,
            eval_data=eval_data,
            label_field=data.label_field,
            demos=demos,
            demos_per_class=demos_per_class if demos else 0,
            s3_prefix=s3_prefix,
            region=data.region,
            role_arn=role_arn,
            run_id=run_id,
            submit_max_wait=submit_max_wait,
            **options,
        )
    except RoundRefused as e:  # a refusal after a split was read: the read is logged
        run_log.append_entries(project_dir, e.events)
        raise
    Path(plan.out).parent.mkdir(parents=True, exist_ok=True)
    Path(plan.staging_dir).mkdir(parents=True, exist_ok=True)
    run_context(plan.run_id).prune_old_scopes(Path(plan.out).parent, KEEP_RUN_SCOPES)
    Path(plan.request_path).write_text(plan.request_text, encoding="utf-8")
    run_log.append_entries(project_dir, plan.events)
    record = Path(plan.job_record_path)
    submitted = submit_round(
        plan,
        on_submit=lambda text: record.write_text(text, encoding="utf-8"),
        on_refused=lambda: record.unlink(missing_ok=True),
        endpoint=os.environ.get("AWS_ENDPOINT_URL_SAGEMAKER"),
    )
    outputs = read_round(plan, submitted.job, run_log_iteration=iteration)
    for path, text in outputs.texts.items():
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
    for dest, src in outputs.copies.items():
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(Path(src).read_bytes())
    for path in outputs.removes:
        Path(path).unlink(missing_ok=True)
    entry = error = None
    if outputs.record is not None:
        try:
            job_ids = run_log.append_entries(project_dir, outputs.record.jobs)
            entry = run_log.append_entry(project_dir, outputs.record.artifact_entry(job_ids))
        except (
            Exception
        ) as e:  # noqa: BLE001 - the labels are written; the record's failure is reported
            error = f"{type(e).__name__}: {e}"
    summary = round_summary(outputs, run_log_entry=entry, run_log_error=error)
    stem = str(Path(plan.out).with_suffix(""))
    report_path = stem + ".held_out_report.json"
    if plan.is_corpus:
        report_path = os.path.join(project_dir, "held_out_report.json")
    report = _read_json(report_path) if os.path.isfile(report_path) else {}
    failures_path = stem + ".failures.jsonl"
    return RoundResult(
        stage=stage,
        iteration=iteration,
        spec=spec,
        out=plan.out,
        run_id=plan.run_id,
        job_name=summary.get("job_name"),
        accuracy=summary.get("held_out_accuracy", report.get("held_out_accuracy")),
        lower_bound_95=summary.get("held_out_lower_bound_95", report.get("lower_bound_95")),
        macro_f1=report.get("macro_f1", summary.get("macro_f1")),
        breakdown=report.get("breakdown") or {},
        confusion_pairs=report.get("confusion_pairs") or [],
        held_out_report=report_path if os.path.isfile(report_path) else None,
        failures=failures_path if os.path.isfile(failures_path) else None,
        dataset_arn=summary.get("dataset_arn"),
        review_dataset_arn=summary.get("review_dataset_arn"),
        summary=summary,
        run_log_entry=entry,
        warnings=list(plan.warnings) + list(outputs.warnings),
    )


class RoundFindings(OperatorResult):
    """One refinement round's findings (full-loop.md Steps 2-4), for the person refining the spec.

    ``dev`` and ``val`` are the two rounds; ``val_accuracy`` is the steering metric (the
    val job's ``HeldOutAccuracy``); ``failures`` are the dev rows the job got wrong
    (``gold``, ``predicted``, the text), the refiner's error view; ``decision`` is the
    routing: ``accept``, ``refine``, ``surface`` (too little signal) or ``stop`` (no
    improvement), with ``decision_reason``.
    """

    iteration: int
    spec: str
    spec_sha256: str
    dev: RoundResult
    val: RoundResult
    val_accuracy: Optional[float] = None
    val_macro_f1: Optional[float] = None
    val_gain: Optional[float] = None
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    confusion_pairs: List[Any] = Field(default_factory=list)
    classes_below_gate: List[str] = Field(default_factory=list)
    decision: str
    decision_reason: str
    gate: float

    def summary(self) -> str:
        """One line: the scores, the error view's size and the routing."""
        val = "n/a" if self.val_accuracy is None else f"{self.val_accuracy:.3f}"
        gain = "" if self.val_gain is None else f" ({self.val_gain:+.3f})"
        f1 = "" if self.val_macro_f1 is None else f", macro_f1 {self.val_macro_f1:.3f}"
        dev = "n/a" if self.dev.accuracy is None else f"{self.dev.accuracy:.3f}"
        pairs = ", ".join(
            f"{a}->{b} x{n}" for a, b, n in (_pair(p) for p in self.confusion_pairs[:3])
        )
        return (
            f"iter_{self.iteration}: val {val}{gain}{f1} · dev {dev} · {len(self.failures)} failures"
            + (f" · {pairs}" if pairs else "")
            + f" · {self.decision.upper()}: {self.decision_reason}"
        )


def _pair(pair):
    """A confusion pair as ``(gold, predicted, count)``; the job writes ``{"gt", "pred", "n"}``."""
    if isinstance(pair, dict):
        return pair.get("gt"), pair.get("pred"), pair.get("n", 0)
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        return pair[0], pair[1], pair[2] if len(pair) > 2 else 0
    return pair, None, 0


def _val_rows_per_class(val_path, label_field):
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows

    counts: Dict[Any, int] = {}
    for row in _rows.read_jsonl(val_path):
        if isinstance(row, dict):
            key = row.get(label_field)
            counts[key] = counts.get(key, 0) + 1
    return counts


def route_refinement(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    val_accuracy, breakdown, confusion_pairs, val_counts, gate, previous_val_accuracy, iteration
):
    """refinement-routing.md's decision for one round: ``(decision, reason)``.

    The val-size floor first (fewer than ``VAL_ROWS_PER_CLASS_FLOOR`` rows per class on
    average: never refine); then accept when val is at the gate and no class is below
    it; refine when a class is below the gate and the confused classes hold at least
    ``CONFUSED_VAL_ROWS_TO_REFINE`` val rows between them; surface it when they do not;
    stop when a refined spec did not improve val.
    """
    n_classes = max(1, len(val_counts))
    total = sum(val_counts.values())
    if total / n_classes < VAL_ROWS_PER_CLASS_FLOOR:
        return "surface", (
            f"val has {total} rows for {n_classes} classes (< {VAL_ROWS_PER_CLASS_FLOOR} per class): "
            "too noisy to rank specs; ship the initial spec, or label more val rows"
        )
    if val_accuracy is None:
        return "surface", "the val job reported no HeldOutAccuracy"
    below = sorted(
        c for c, v in (breakdown or {}).items() if isinstance(v, (int, float)) and v < gate
    )
    if val_accuracy >= gate and not below:
        return "accept", f"val {val_accuracy:.3f} >= gate {gate} and no class below it"
    if (
        iteration > 0
        and previous_val_accuracy is not None
        and val_accuracy <= previous_val_accuracy
    ):
        return "stop", (
            f"val did not improve ({previous_val_accuracy:.3f} -> {val_accuracy:.3f}); "
            "ship the best-seen spec"
        )
    confused = set()
    for pair in confusion_pairs or []:
        gold, pred, _ = _pair(pair)
        confused.update({str(gold), str(pred)})
    confused.update(below)
    confused_rows = sum(n for c, n in val_counts.items() if str(c) in confused)
    what = ", ".join(below) if below else f"val {val_accuracy:.3f} < gate {gate}"
    if confused_rows >= CONFUSED_VAL_ROWS_TO_REFINE:
        return "refine", f"{what} below gate; {confused_rows} val rows in the confused classes"
    return "surface", (
        f"{what} below gate but only {confused_rows} val rows in the confused classes "
        f"(< {CONFUSED_VAL_ROWS_TO_REFINE}): too little signal; label more rows or merge classes"
    )


class RefinementRounds:
    """The refinement loop of full-loop.md (Steps 2-5), one ``next()`` per spec version.

    Each ``next()`` snapshots the spec file as it is now to
    ``work/labeling_spec_iter_<N>.md``, labels dev and val with it
    (:func:`run_labeling_round` twice, a fresh run id each), reads the two jobs' scores and
    returns :class:`RoundFindings` with the routing of refinement-routing.md. Between
    calls the person edits the spec from ``failures`` (at most three edits, the
    skill's rule). Iteration ends when the routing is not ``refine`` or
    ``max_iterations`` have run; then :func:`~..spec.select.select_labeling_spec` picks the
    shipped spec and :func:`label_corpus` runs it.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import tune_labeling_spec

        rounds = tune_labeling_spec("tickets", data, spec="tickets/labeling_spec.md",
                                   s3_prefix="s3://bucket/autolabel", role_arn=role_arn)
        for findings in rounds:          # or: findings = next(rounds)
            print(findings.summary())
            if findings.decision == "refine":
                input("edit tickets/labeling_spec.md from findings.failures, then press Enter")
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        project_dir: str,
        data,
        *,
        spec: str,
        s3_prefix: str,
        role_arn: str,
        gate: Optional[float] = None,
        max_iterations: int = 4,
        demos_per_class: int = 2,
        generation_model_id: str = DEFAULT_LABELING_MODEL,
        dataset_name: Optional[str] = None,
        **options: Any,
    ):
        self.project_dir = project_dir
        self.data = data
        self.spec = spec
        self.s3_prefix = s3_prefix
        self.role_arn = role_arn
        self.gate = DEFAULT_GATE.get(data.task, 0.85) if gate is None else gate
        self.max_iterations = max_iterations
        self.demos_per_class = demos_per_class
        self.generation_model_id = generation_model_id
        self.dataset_name = dataset_name
        self.options = options
        self.rounds: List[RoundFindings] = []
        self._done = False

    def __iter__(self):
        """The loop iterates itself."""
        return self

    def __next__(self) -> RoundFindings:
        """The next round, until the routing is not ``refine`` or the cap is reached."""
        if self._done or len(self.rounds) > self.max_iterations:
            raise StopIteration
        findings = self.run_round()
        if findings.decision != "refine" or len(self.rounds) > self.max_iterations:
            self._done = True
        return findings

    @property
    def iteration(self) -> int:
        """The next round's iteration number (``iter_0`` is the initial spec)."""
        return len(self.rounds)

    def run_round(self) -> RoundFindings:
        """Run the next round regardless of the routing (dev + val on the spec as it is now)."""
        from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows, _run_log
        from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import (
            run_log_event,
        )

        n = self.iteration
        snapshot = os.path.join(self.project_dir, "work", f"labeling_spec_iter_{n}.md")
        text = Path(self.spec).read_text(encoding="utf-8")
        sha = _digest.sha256_hex(text)
        warnings: List[str] = []
        if self.rounds and sha == self.rounds[-1].spec_sha256:
            warnings.append(
                f"the spec is unchanged since iter_{n - 1}; this round repeats it (edit {self.spec} first)"
            )
        if os.path.abspath(snapshot) != os.path.abspath(self.spec):
            Path(snapshot).parent.mkdir(parents=True, exist_ok=True)
            Path(snapshot).write_text(text, encoding="utf-8")
        common = dict(
            spec=snapshot,
            s3_prefix=self.s3_prefix,
            role_arn=self.role_arn,
            iteration=n,
            generation_model_id=self.generation_model_id,
            demos_per_class=self.demos_per_class,
            dataset_name=self.dataset_name,
            **self.options,
        )
        dev = run_labeling_round(self.project_dir, self.data, stage="dev", **common)
        val = run_labeling_round(self.project_dir, self.data, stage="val", **common)
        failures = [
            r
            for r in (_rows.read_jsonl(dev.failures) if dev.failures else [])
            if isinstance(r, dict) and not r.get("correct")
        ]
        previous = self.rounds[-1].val_accuracy if self.rounds else None
        val_counts = _val_rows_per_class(
            os.path.join(self.project_dir, "splits", "val.jsonl"), self.data.label_field
        )
        decision, reason = route_refinement(
            val.accuracy, val.breakdown, val.confusion_pairs, val_counts, self.gate, previous, n
        )
        below = sorted(
            c
            for c, v in (val.breakdown or {}).items()
            if isinstance(v, (int, float)) and v < self.gate
        )
        findings = RoundFindings(
            iteration=n,
            spec=snapshot,
            spec_sha256=sha,
            dev=dev,
            val=val,
            val_accuracy=val.accuracy,
            val_macro_f1=val.macro_f1,
            val_gain=None if previous is None or val.accuracy is None else val.accuracy - previous,
            failures=failures,
            confusion_pairs=list(val.confusion_pairs),
            classes_below_gate=below,
            decision=decision,
            decision_reason=reason,
            gate=self.gate,
            warnings=warnings + dev.warnings + val.warnings,
        )
        _run_log.append_entry(
            self.project_dir,
            run_log_event(
                "analysis",
                "refine",
                n,
                decision,
                spec=snapshot,
                spec_sha256=sha,
                val_accuracy=val.accuracy,
                val_macro_f1=val.macro_f1,
                dev_accuracy=dev.accuracy,
                failures=len(failures),
                gate=self.gate,
                reason=reason,
            ),
        )
        self.rounds.append(findings)
        return findings

    def scores(self) -> Dict[str, Any]:
        """The rounds so far as the ``scores.json`` document ``select_labeling_spec`` selects from."""
        rows = [
            {
                "spec": r.spec,
                "val": r.val_accuracy,
                "val_macro_f1": r.val_macro_f1,
                "breakdown": r.val.breakdown or None,
            }
            for r in self.rounds
        ]
        rows = [{k: v for k, v in row.items() if v is not None} for row in rows]
        if not rows:
            raise DataPreparerOperatorError("no round has run yet")
        return {"initial": rows[0], "iters": rows[1:]}


def tune_labeling_spec(project_dir: str, data, **kwargs: Any) -> RefinementRounds:
    """Tune the spec over labeling rounds (full-loop.md Steps 2-5): see :class:`RefinementRounds`.

    Each ``next()`` labels dev and val with the spec as it is, reports the scores and the
    dev failures, and says whether another round is worth it; the edit itself is the
    person's, between calls. Writing a first spec from examples is
    :func:`~..spec.induce.draft_labeling_spec`.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import tune_labeling_spec

        rounds = tune_labeling_spec("tickets", data, spec="tickets/labeling_spec.md",
                                   s3_prefix="s3://bucket/autolabel", role_arn=role_arn, gate=0.85)
        round_0 = next(rounds)      # dev + val on the spec as it is; the failures to edit from
        round_1 = next(rounds)      # after the edit; StopIteration once the routing says accept or stop
    """
    return RefinementRounds(project_dir, data, **kwargs)


def label_corpus(  # pylint: disable=too-many-arguments
    project_dir: str,
    data,
    *,
    corpus: str,
    dataset_name: str,
    s3_prefix: str,
    role_arn: str,
    spec: Optional[str] = None,
    target: Optional[float] = None,
    review_dataset_name: Optional[str] = None,
    jury_models: Optional[str] = None,
    demos_per_class: int = 2,
    run_id: Optional[str] = None,
    confirm_no_seed: bool = False,
    selected_val_accuracy: Optional[float] = None,
    **options: Any,
) -> RoundResult:
    """Label the corpus with the shipped spec, certified on the frozen test split (Step 7).

    ``--input <corpus> --eval splits/test.jsonl --target --dataset-name --out
    labeled_dataset.json``: the job labels every row, scores the test rows, picks the
    confidence cutoff that certifies the target and registers the labeled DataSet. The
    labels stay behind the API; ``labeled_dataset.json`` is the deliverable record. Test
    is scored here once, never used to pick a spec; when it comes in more than
    ``HELDOUT_REVIEW_TOL`` below the val score that selected the spec, the result carries
    the review flag as a warning (Step 6).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import label_corpus

        result = label_corpus("tickets", data, corpus="s3://bucket/tickets.jsonl",
                              dataset_name="tickets-labeled", s3_prefix="s3://bucket/autolabel",
                              role_arn=role_arn, target=0.85)
        result.dataset_arn, result.accuracy, result.held_out_report

    Args:
        project_dir: The project.
        data: :func:`~..inspect.inputs.inspect_labeling_data`'s report.
        corpus: The rows to label: a local path, an ``s3://`` URI or a DataSet ARN.
        dataset_name: The labeled DataSet's name.
        s3_prefix: The run's S3 prefix.
        role_arn: The job's role.
        spec: The spec to ship; ``<project_dir>/labeling_spec.md`` (what
            :func:`~..spec.select.select_labeling_spec` wrote) when ``None``.
        target: The accuracy target the job certifies.
        review_dataset_name: The review DataSet's name (the low-confidence tail).
        jury_models: Comma-separated juror model ids.
        demos_per_class: The same demos as the refinement rounds.
        run_id: Names the run; the same id re-attaches.
        confirm_no_seed: The cold-start opt-out: label with no accuracy promise.
        selected_val_accuracy: The val score that selected the spec, for the review flag.
        **options: :func:`prepare_round`'s other keyword arguments.

    Returns:
        A :class:`RoundResult`; ``dataset_arn`` is the deliverable.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec._floor import (
        heldout_review_flag,
    )

    spec = spec or os.path.join(project_dir, "labeling_spec.md")
    test_path = os.path.join(project_dir, "splits", "test.jsonl")
    eval_data = test_path if os.path.isfile(test_path) else None
    if not os.path.isfile(os.path.join(project_dir, "splits", "dev.jsonl")):
        demos_per_class = 0  # no seed: nothing to draw demos from
    if target is not None and review_dataset_name is None:
        review_dataset_name = (
            f"{dataset_name}-review"  # Step 7: the target routes the tail to review
        )
    result = run_labeling_round(
        project_dir,
        data,
        spec=spec,
        stage="corpus",
        s3_prefix=s3_prefix,
        role_arn=role_arn,
        input_data=corpus,
        eval_data=eval_data,
        out=os.path.join(project_dir, "labeled_dataset.json"),
        run_id=run_id,
        demos_per_class=demos_per_class,
        dataset_name=dataset_name,
        target=target,
        review_dataset_name=review_dataset_name,
        jury_models=jury_models,
        confirm_no_seed=confirm_no_seed,
        **options,
    )
    if selected_val_accuracy is None:
        decision = _read_json(os.path.join(project_dir, "work", "decision.json"))
        selected_val_accuracy = decision.get("val")
    flag = heldout_review_flag(selected_val_accuracy, result.accuracy, tol=HELDOUT_REVIEW_TOL)
    if flag.get("flag"):
        result.warnings.append(
            f"review flag: test {result.accuracy:.3f} is {flag['gap']:.3f} below the val score "
            f"{selected_val_accuracy:.3f} that selected the spec; the refinement may have over-fit val. "
            "Report it; do not swap specs on test."
        )
    return result
