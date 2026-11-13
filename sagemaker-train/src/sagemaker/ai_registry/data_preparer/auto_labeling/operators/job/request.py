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
"""The labeling job request: the task contract, the client-side checks, and the typed config.

A SyntheticDataGeneration job runs in labeling mode when its request carries a
``labeling.<task>`` ``DatasetOptions.Type``; ``DatasetOptions.Config`` carries the
task's options and the service supplies the labeling program. One ``CreateJob`` then
labels, votes, scores, calibrates, routes and registers; the client reads the result
back through the Hub entry.

:func:`build_labeling_request` builds that request document from plain arguments and
checks, before anything is uploaded or spent, the rules a client can check (the ones the
service also refuses at ``CreateJob``: text fields, the input channel, the panel size
for the task and routing, evaluation data, the target range). It is pure.
:func:`to_dataset_job_config` turns the document into a
:class:`~sagemaker.ai_registry.data_preparer.configs.DatasetJobConfig` carrying the typed
``*LabelingOptions`` shapes, so the request the SDK sends is the typed one.

The contract tables (``PREDICTION_FIELDS``, ``GOLD_FIELDS``, the vote and resample tasks,
the fixed service limits) and :func:`spec_score_scale` (how the job reads a spec's score
scale) live here too.
"""

from __future__ import absolute_import

import math
import re
from typing import Any, Dict

from sagemaker.ai_registry.data_preparer.auto_labeling import shapes as S
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    reserved_field_reason,
    reserved_text_field_conflicts,
)


class LabelingRequestError(DataPreparerOperatorError):
    """A labeling request the service would reject, caught before anything is spent.

    The message names the request field.
    """


# The CreateJob schema version that carries DatasetOptions. A current SDK resolves
# SyntheticDataGeneration to 2.0.0 on its own (its ceiling is 2); the pin is kept so
# older SDKs (ceiling 1) still send v2, not a v1 request that rejects every labeling
# field.
LABELING_SCHEMA_VERSION = "2.0.0"
TYPE_PREFIX = "labeling."
# task -> the field the service-owned program writes its prediction to
PREDICTION_FIELDS = {
    "classification": "label",
    "ner": "entities",
    "qa": "answer",
    "ordinal": "score",
    "llm_judge": "judge_score",
    "summarization": "summary",
}
LABELING_TASKS = tuple(PREDICTION_FIELDS)
# task -> the gold field the skill's seeds carry (input-output-contract.md, Inputs)
GOLD_FIELDS = {
    "classification": "gt_label",
    "ner": "gt_entities",
    "qa": "gt_answers",
    "ordinal": "gt_score",
    "llm_judge": "gt_score",
    "summarization": "gt_summary",
}
# Confidence mechanism per task (design section 4): a vote across the jury, or one
# model resampled; summarization has neither and is graded by Scoring.Graders.
VOTE_TASKS = frozenset({"classification", "ner", "qa"})
RESAMPLE_TASKS = frozenset({"ordinal", "llm_judge"})
CERTIFIABLE_TASKS = VOTE_TASKS
NO_PROMISE_TASKS = ("ordinal", "llm_judge", "summarization")

# The labeling program's score-scale reader (the scale a resample band and the
# llm_judge tolerance are sized from): a range under a `Scale` heading, a "scale of X
# to Y" / "Scale: X to Y" / "X to Y scale" phrase, or "from X to Y" / "between X and
# Y" in a sentence about a score or rating.
_SCALE_NUMBER = r"(-?\d+(?:\.\d+)?)"
_SCALE_RANGE = _SCALE_NUMBER + r"\s*(?:to|-|\u2013|\u2014)\s*" + _SCALE_NUMBER
_SCALE_KEYWORD = r"(?i)\b(?:score|scores|scored|rate|rated|rating|grade)\b[^.\n]{0,80}?"


def spec_score_scale(spec_text):
    """The ``[low, high]`` score scale a spec states, read the way the labeling job reads it.

    A range under a ``Scale`` heading, a "scale of X to Y" / "Scale: X to Y" / "X to Y
    scale" phrase, or "from X to Y" / "between X and Y" in a sentence about a score or
    rating. ``None`` when the spec states none; a corpus run of ordinal / llm_judge with
    none fails at load.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import spec_score_scale

        scale = spec_score_scale(spec_text)  # [1, 5], read the way the labeling job reads it
    """
    spec = spec_text or ""
    block = re.search(
        r"(?im)^#+\s*(?:score\s+|rating\s+)?scale\b(.*?)(?=^#+\s|\Z)", spec, re.DOTALL
    )
    patterns = [
        (block.group(1) if block else "", _SCALE_RANGE),
        (spec, r"(?i)\bscale\s*:?\s*(?:of\s+|from\s+)?" + _SCALE_RANGE),
        (spec, r"(?i)" + _SCALE_RANGE + r"\s+(?:point\s+)?scale\b"),
        (spec, _SCALE_KEYWORD + r"\bfrom\s+" + _SCALE_NUMBER + r"\s+to\s+" + _SCALE_NUMBER),
        (spec, _SCALE_KEYWORD + r"\bbetween\s+" + _SCALE_NUMBER + r"\s+and\s+" + _SCALE_NUMBER),
    ]
    for text, pattern in patterns:
        m = re.search(pattern, text)
        if m and float(m.group(1)) < float(m.group(2)):
            return [float(m.group(1)), float(m.group(2))]
    return None


# Fixed service rules (the control plane's labeling validation and the labeling
# program's configuration). Not knobs: a request outside them is rejected.
MIN_TARGET = 0.85  # the spot-check's 7-error withdraw rule is defined for 0.85/0.90/0.95
MAX_TARGET = 0.99
MIN_JURORS = 2  # a vote needs a panel
MAX_JURORS = 5  # ModelConfig.SampleGenerationModelConfigs maxItems
MAX_LABELING_ROWS = 50_000  # enforced by the container at load; stated to the user
MAX_SPEC_BYTES = 64 * 1024
MAX_DEMOS = 100
MAX_DEMOS_BYTES = 64 * 1024
MIN_EVAL_ROWS = 50  # rule 7: fewer eval rows cannot certify
WARN_EVAL_ROWS = 100  # rule 7: under 100 at a target >= 0.95 the bound is wide
WARN_TARGET = 0.95
MIN_AGREEMENT = 0.75  # finalizer constant, reported as MinAgreement
BAR_CUTOFF = 0.85  # the cutoff of a job with no gold or no target
SPOT_CHECK_WITHDRAW_AT = 7
# spot-check size at the three targets the withdraw rule is defined for: the smallest
# zero-error Clopper-Pearson draw (the container's finalizer.spot_check_size). Only the
# no-seed message prints these; the job reports the size it drew as SpotCheckRows.
SPOT_CHECK_ROWS = {0.85: 19, 0.90: 29, 0.95: 59}

# Opt-in options. Each is sent only when its argument is given; absent, the job runs
# as it does without them. The tasks that take each option are one set apiece.
CONFIDENCE_METHODS = ("agreement", "stated", "verbalized")
# Confidence.Method verbalized (opt-in, off by default): one model states a 0 to 1
# confidence_score with its label and the job routes on it (the 0.75 floor stays on unless
# Confidence.FloorBypass). A corpus
# run (Routing) labels with exactly one model: there is no jury and no vote.
VERBALIZED = "verbalized"
# The container's opt-in options, passed through (the service checks the combinations at submit):
# Confidence.Estimator judge / panel with Confidence.Models, Confidence.Recalibration platt, and
# Config.Optimizer (an object, Mode protegi). Each is sent only when its argument is given.
CONFIDENCE_ESTIMATORS = ("judge", "panel")
RECALIBRATIONS = ("platt",)

DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9](-*[A-Za-z0-9])*$")
DATASET_NAME_MAX = 63
# InputDatasetArn / EvaluationData.DatasetArn: a DataSet ARN. The control plane
# requires the version, so the version group is not optional here.
DATASET_ARN_RE = re.compile(
    r"^arn:[a-z0-9-.]{1,63}:sagemaker:[a-zA-Z0-9]+(?:-[a-zA-Z0-9]+)+:[0-9]{12}:hub-content/"
    r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}/DataSet/[a-zA-Z0-9](-*[a-zA-Z0-9]){0,63}"
    r"/[0-9]{1,4}(\.[0-9]{1,4}){0,2}$"
)
S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")
# A model id under a cross-region inference profile (us., eu., apac., global., jp., ...) is the
# same model as the bare id: the container's duplicate-grader check canonicalizes this
# way, so the skill does too.
_INFERENCE_PROFILE_PREFIX = re.compile(r"^(us|eu|apac|global|jp|au|ca|sa|us-gov)\.")

# The job folder's layout (container contract) and the index keys the finalizer writes.
OUTPUT_SUBDIR = "output"
REVIEW_SUBDIR = "review"
LABELING_SUBDIR = "labeling"
INDEX_FILENAME = "index.json"
SUMMARY_FILENAME = "labeling_summary.json"
LABELS_FILENAME = "labels.jsonl"
FAILURES_FILENAME = "failures.jsonl"
HELD_OUT_FILENAME = "held_out_report.json"
REVIEW_FILENAME = "review_rows.jsonl"
REVIEW_MODEL_OUTPUT_FILENAME = "review_rows.model_output.jsonl"
SPEC_FILENAME = "spec.md"
REPORT_HTML_FILENAME = "report.html"
# index.json "Artifacts" key -> the local file name the skill keeps it under
INDEX_ARTIFACTS = {
    "LabelingSummary": SUMMARY_FILENAME,
    "Labels": LABELS_FILENAME,
    "ReviewRows": REVIEW_FILENAME,
    "ReviewModelOutput": REVIEW_MODEL_OUTPUT_FILENAME,
    "Failures": FAILURES_FILENAME,
    "HeldOutReport": HELD_OUT_FILENAME,
    "Report": REPORT_HTML_FILENAME,
    "Spec": SPEC_FILENAME,
    "OptimizerResult": "optimizer_result.json",  # only a Config.Optimizer job lists it
}


def labeling_type(task):
    """`labeling.<task>` for a task of LABELING_TASKS, else LabelingRequestError."""
    if task not in PREDICTION_FIELDS:
        raise LabelingRequestError(
            f"task {task!r} is not a labeling task; one of {', '.join(LABELING_TASKS)}"
        )
    return TYPE_PREFIX + task


def task_of_type(type_value):
    """The task of a `labeling.<task>` value, or None."""
    if isinstance(type_value, str) and type_value.startswith(TYPE_PREFIX):
        task = type_value[len(TYPE_PREFIX) :]
        return task if task in PREDICTION_FIELDS else None
    return None


def bar_vote_note(n_jurors, cutoff=BAR_CUTOFF):
    """What the fixed bar a no-target run accepts at means on an ``n_jurors`` panel.

    The job's vote confidence is the agreeing jurors over the panel size, so a row clears
    the bar only with ``ceil(cutoff * n)`` votes: 3 of 3, but 4 of 4 and 5 of 5. Above 3
    jurors that is every juror, and the bar tightens as the panel grows.

    Returns:
        The note, or ``None`` when the bar is not stricter than a person would expect.
    """
    if not isinstance(n_jurors, int) or n_jurors <= 3:
        return None
    need = math.ceil(cutoff * n_jurors - 1e-9)
    if need < n_jurors:
        return None
    return (
        f"with no --target the job accepts a row at the fixed {cutoff} bar, which on "
        f"{n_jurors} jurors means all {n_jurors} agree ({need} of {n_jurors}; "
        f"{n_jurors - 1} of {n_jurors} is {round((n_jurors - 1) / n_jurors, 4)}). "
        "Every juror agreeing is rarer on a bigger panel, so more rows go to review "
        "than on 3 jurors. Use 3 jurors, or give the run gold and a --target so the "
        "job fits the cutoff to the panel."
    )


def canonical_model_id(model_id):
    """A model id without its cross-region inference-profile prefix."""
    return _INFERENCE_PROFILE_PREFIX.sub("", model_id or "")


def validate_dataset_name(name, field="DatasetName"):
    """``name`` as a Hub DataSet name, else :class:`LabelingRequestError` naming ``field``.

    1 to 63 letters, digits and hyphens, starting and ending with a letter or digit.
    """
    if not isinstance(name, str) or not name:
        raise LabelingRequestError(f"{field} is required")
    if len(name) > DATASET_NAME_MAX or not DATASET_NAME_RE.fullmatch(name):
        raise LabelingRequestError(
            f"{field} {name!r} must be 1 to {DATASET_NAME_MAX} letters, digits and hyphens, "
            "starting and ending with a letter or digit (no underscores, dots or spaces)"
        )
    return name


def scratch_dataset_name(dataset_name, stage, run_id):
    """The run-unique name a working job registers under: ``<dataset_name>-<stage>-<run id>``.

    A reused name appends a version to an existing entry instead of failing, so every
    round gets its own. The stage and run id are kept whole and the dataset name is
    trimmed to fit ``DATASET_NAME_MAX``.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import scratch_dataset_name

        scratch_dataset_name("tickets", "dev", "r01")  # 'tickets-dev-r01'

    Raises:
        LabelingRequestError: No stage, a run id with no letter or digit, or no room left
            for the name.
    """
    stage_slug = re.sub(r"[^A-Za-z0-9]+", "-", stage or "").strip("-").lower()
    run_slug = re.sub(r"[^A-Za-z0-9]+", "-", run_id or "").strip("-")
    if not stage_slug:
        raise LabelingRequestError("--stage is required for a scratch job (e.g. dev, val)")
    if not run_slug:
        raise LabelingRequestError("--run-id must contain a letter or digit")
    suffix = f"-{stage_slug}-{run_slug}"
    room = DATASET_NAME_MAX - len(suffix)
    if room < 1:
        raise LabelingRequestError(
            f"--stage {stage!r} and --run-id {run_id!r} leave no room for the dataset "
            f"name in a {DATASET_NAME_MAX}-character Hub name; use a shorter run id"
        )
    base = re.sub(r"[^A-Za-z0-9-]+", "-", dataset_name or "autolabel")[:room].strip("-")
    return validate_dataset_name(
        (base or "autolabel")[:room].rstrip("-") + suffix, "the scratch DatasetName"
    )


def _check_s3_uri(value, field):
    if not isinstance(value, str) or not S3_URI_RE.match(value):
        raise LabelingRequestError(f"{field} must be an s3://bucket/key URI, got {value!r}")
    return value


def _check_dataset_arn(value, field):
    if not isinstance(value, str) or not DATASET_ARN_RE.match(value):
        raise LabelingRequestError(
            f"{field} must be a versioned Hub DataSet ARN "
            "(arn:aws:sagemaker:<region>:<account>:hub-content/<hub>/DataSet/<name>/<version>), "
            f"got {value!r}"
        )
    return value


def _model_config(model_id):
    # ModelId only: CreateJob refuses TargetTokensPerMinute on every model config.
    return {"ModelId": model_id}


def _graders_config(graders):
    """``Scoring.Graders`` in the request shape, passed through.

    Model ids or ``{ModelId, ReasoningEffort?}`` dicts. The service checks the count,
    the task, the reasoning effort and that no grader is the labeling model, and refuses
    at ``CreateJob`` before any job starts.
    """
    return [{"ModelId": g} if isinstance(g, str) else dict(g or {}) for g in graders] or None


def build_labeling_request(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    *,
    task,
    text_fields,
    spec_s3_uri,
    model_ids,
    dataset_name,
    output_s3_uri,
    input_s3_uri=None,
    input_dataset_arn=None,
    demos_s3_uri=None,
    eval_s3_uri=None,
    eval_dataset_arn=None,
    label_field=None,
    target=None,
    review_dataset_name=None,
    graders=None,
    hub_name=None,
    confidence_method=None,
    confidence_estimator=None,
    judge_models=None,
    confidence_recalibration=None,
    optimizer=None,
    confidence_floor_bypass=False,
):
    """The labeling ``CreateJob`` document, checked before anything is uploaded or spent.

    Use it to build every labeling request, a refine round or the corpus run, from plain
    values; hand the result to :func:`to_dataset_job_config` to submit it. The checks here
    are the rules a client can check (the service refuses the same things at ``CreateJob``,
    and the option combinations are the service's alone), so a wrong request fails here
    with the field named instead of after an upload. Pure: no I/O.

    Args:
        task: ``classification``, ``ner``, ``qa``, ``ordinal``, ``llm_judge`` or
            ``summarization``.
        text_fields: The input fields to label from, in order; no name the output or the
            review file reserves.
        spec_s3_uri: The labeling spec's ``s3://`` URI (at most 64 KiB).
        model_ids: The panel: the jury of a corpus run (``review_dataset_name`` set; 2 to
            5 models on a vote task, exactly 1 on ordinal / llm_judge), or the one model
            of a refine round.
        dataset_name: The labeled DataSet's name.
        output_s3_uri: Where the job writes.
        input_s3_uri: The rows to label, as an S3 URI (one of this and
            ``input_dataset_arn``).
        input_dataset_arn: The rows to label, as a versioned DataSet ARN.
        demos_s3_uri: Labeled examples (at most 100 rows, 64 KiB).
        eval_s3_uri: Gold rows, as an S3 URI (one of this and ``eval_dataset_arn``).
        eval_dataset_arn: Gold rows, as a DataSet ARN.
        label_field: The gold field; required with evaluation data, never a text field.
        target: ``Scoring.Target`` in ``[MIN_TARGET, MAX_TARGET]``; not on a task with no
            accuracy promise.
        review_dataset_name: Sets ``Routing``: the corpus run that routes low-confidence
            rows to a review DataSet of this name (which must differ from
            ``dataset_name``).
        graders: Summarization only: grader model ids or ``{ModelId, ReasoningEffort?}``.
        hub_name: ``OutputDataConfig.HubName``.
        confidence_method: ``agreement``, ``stated`` or ``verbalized`` (opt-in).
        confidence_estimator: ``judge`` or ``panel`` (needs ``confidence_method``).
        judge_models: The estimator's models.
        confidence_recalibration: ``platt``.
        optimizer: The in-job search (``{"Mode": "protegi", ...}``), refine rounds only.
        confidence_floor_bypass: Route on the stated confidence alone.

    Returns:
        ``{"DatasetName", "GenerationConfig", "OutputDataConfig"}`` in the schema 2.0.0
        shape.

    Raises:
        LabelingRequestError: Any check fails; the message names the request field.

    Example::

        request = {
            "task": "classification",
            "text_fields": ["text"],
            "spec_s3_uri": "s3://b/spec.md",
            "model_ids": ["us.anthropic.claude-sonnet-4-5-20250929-v1:0"],
            "dataset_name": "tickets-dev-r1",
            "output_s3_uri": "s3://b/out/",
            "input_s3_uri": "s3://b/dev.jsonl",
            "eval_s3_uri": "s3://b/dev.jsonl",
            "label_field": "gt_label",
        }
        doc = build_labeling_request(**request)
        to_dataset_job_config(doc).dataset_options.text_fields
        # ['text']
    """
    type_value = labeling_type(task)
    validate_dataset_name(dataset_name)
    # TextFields: non-empty, non-blank, distinct, and no name the output or the review
    # file reserves (the container rejects those; reviewers would never see such a column).
    if not isinstance(text_fields, (list, tuple)) or not text_fields:
        raise LabelingRequestError("TextFields is required: the input field(s) to label from")
    fields = []
    for f in text_fields:
        if not isinstance(f, str) or not f.strip():
            raise LabelingRequestError("TextFields entries must be non-blank field names")
        if f in fields:
            raise LabelingRequestError(f"TextFields repeats {f!r}")
        fields.append(f)
    clash = reserved_text_field_conflicts(fields, PREDICTION_FIELDS[task])
    if clash:
        why = "; ".join(f"{f}: {reserved_field_reason(f, PREDICTION_FIELDS[task])}" for f in clash)
        raise LabelingRequestError(
            f"TextFields {', '.join(clash)} are reserved column names ({why}). Copy each "
            "column to a new name (for example `source` to `evidence`) in the corpus and "
            "the seed, and pass the new name"
        )
    # One input channel.
    if bool(input_s3_uri) == bool(input_dataset_arn):
        raise LabelingRequestError(
            "exactly one of InputDataS3Uri (--input s3://...) and InputDatasetArn "
            "(--input arn:...) names the rows to label"
        )
    if input_s3_uri:
        _check_s3_uri(input_s3_uri, "InputDataS3Uri")
    else:
        _check_dataset_arn(input_dataset_arn, "InputDatasetArn")
    _check_s3_uri(spec_s3_uri, "DatasetSpecification.SpecS3Uri")
    _check_s3_uri(output_s3_uri, "OutputDataConfig.S3OutputPath")
    # The panel.
    if not isinstance(model_ids, (list, tuple)) or not model_ids:
        raise LabelingRequestError(
            "ModelConfig.SampleGenerationModelConfigs needs at least one model"
        )
    models = [m for m in model_ids if isinstance(m, str) and m.strip()]
    if len(models) != len(model_ids):
        raise LabelingRequestError("a model id is blank")
    if len(models) > MAX_JURORS:
        raise LabelingRequestError(
            f"ModelConfig.SampleGenerationModelConfigs takes at most {MAX_JURORS} models, got {len(models)}"
        )
    if len(set(models)) != len(models):
        raise LabelingRequestError("ModelConfig.SampleGenerationModelConfigs repeats a model id")
    routing = review_dataset_name is not None
    if routing:
        validate_dataset_name(review_dataset_name, "Routing.ReviewDatasetName")
        # Compared case-insensitively, as the control plane's labeling validation does
        # (JobRequestValidator: equalsIgnoreCase).
        if review_dataset_name.casefold() == dataset_name.casefold():
            raise LabelingRequestError(
                "Routing.ReviewDatasetName must differ from DatasetName (ignoring case): the "
                "job registers two DataSets, so it needs two names"
            )
        if task in VOTE_TASKS and (confidence_method == VERBALIZED or confidence_estimator):
            if len(models) != 1:
                raise LabelingRequestError(
                    f"Confidence.Method verbalized and a judge or panel Estimator label a "
                    f"corpus run (Routing) with one model, no jury; got {len(models)} "
                    f"(--jury-models)"
                )
        elif task in VOTE_TASKS and len(models) < MIN_JURORS:
            raise LabelingRequestError(
                f"a corpus run (Routing) on labeling.{task} votes a jury of at least "
                f"{MIN_JURORS} models; got {len(models)} (--jury-models)"
            )
        if task in RESAMPLE_TASKS and len(models) != 1:
            raise LabelingRequestError(
                f"a corpus run on labeling.{task} resamples one model (its confidence is "
                f"the resample agreement); got {len(models)} models"
            )
    elif len(models) != 1:
        raise LabelingRequestError(
            "a job without Routing (a refine round) runs one model; the service would use "
            f"only the first of the {len(models)} given. Pass a single --generation-model-id"
        )
    if task == "summarization" and len(models) != 1:
        raise LabelingRequestError(
            f"labeling.summarization runs one model (no vote); got {len(models)}"
        )
    # Evaluation data.
    has_eval = bool(eval_s3_uri) or bool(eval_dataset_arn)
    if eval_s3_uri and eval_dataset_arn:
        raise LabelingRequestError("EvaluationData takes S3Uri or DatasetArn, not both")
    if has_eval and task == "summarization":
        raise LabelingRequestError(
            "labeling.summarization takes no EvaluationData: Scoring.Graders grade each "
            "summary against its own source text"
        )
    if has_eval:
        if not isinstance(label_field, str) or not label_field.strip():
            raise LabelingRequestError("EvaluationData.LabelField is required with evaluation data")
        if label_field in fields:
            raise LabelingRequestError(
                f"EvaluationData.LabelField {label_field!r} is also a TextFields entry; "
                "gold must not be an input the labeler reads"
            )
        if eval_s3_uri:
            _check_s3_uri(eval_s3_uri, "EvaluationData.S3Uri")
        else:
            _check_dataset_arn(eval_dataset_arn, "EvaluationData.DatasetArn")
    # Scoring.
    if target is not None:
        if isinstance(target, bool) or not isinstance(target, (int, float)):
            raise LabelingRequestError("Scoring.Target must be a number")
        if not MIN_TARGET - 1e-9 <= float(target) <= MAX_TARGET + 1e-9:
            raise LabelingRequestError(
                f"Scoring.Target must be between {MIN_TARGET} and {MAX_TARGET}, got {target}; "
                "a lower target means labels with no accuracy promise (send no --target)"
            )
        if task in NO_PROMISE_TASKS:
            raise LabelingRequestError(
                f"labeling.{task} ships without an accuracy promise, so it takes no "
                "Scoring.Target; its cutoff is the fixed 0.85 bar"
            )
    grader_cfgs = _graders_config(list(graders or []))
    if demos_s3_uri:
        _check_s3_uri(demos_s3_uri, "Demos.S3Uri")

    config = {"TextFields": fields}
    if demos_s3_uri:
        config["Demos"] = {"S3Uri": demos_s3_uri}
    if has_eval:
        ev = {"LabelField": label_field}
        if eval_s3_uri:
            ev["S3Uri"] = eval_s3_uri
        else:
            ev["DatasetArn"] = eval_dataset_arn
        config["EvaluationData"] = ev
    scoring = {}
    if target is not None:
        scoring["Target"] = float(target)
    if grader_cfgs:
        scoring["Graders"] = grader_cfgs
    if scoring:
        config["Scoring"] = scoring
    if (
        confidence_estimator or judge_models or confidence_recalibration or confidence_floor_bypass
    ) and confidence_method is None:
        raise LabelingRequestError(
            "Confidence.Estimator, Models, Recalibration and FloorBypass need "
            "Confidence.Method (--confidence-method)"
        )
    if optimizer is not None and (not isinstance(optimizer, dict) or routing):
        raise LabelingRequestError(
            "Config.Optimizer is an object (Mode protegi) on a refine " "round, never with Routing"
        )
    if confidence_method is not None:
        config["Confidence"] = {"Method": confidence_method}
        for key, value in (
            ("Estimator", confidence_estimator),
            ("Models", [{"ModelId": m} for m in judge_models or []] or None),
            ("Recalibration", confidence_recalibration),
            ("FloorBypass", confidence_floor_bypass or None),
        ):
            if value:
                config["Confidence"][key] = value
    if optimizer is not None:
        config["Optimizer"] = dict(optimizer)
    if routing:
        config["Routing"] = {"ReviewDatasetName": review_dataset_name}
    generation_config = {
        "DatasetSpecification": {"SpecS3Uri": spec_s3_uri},
        "DatasetOptions": {"Type": type_value, "Config": config},
        "ModelConfig": {"SampleGenerationModelConfigs": [_model_config(m) for m in models]},
    }
    if input_s3_uri:
        generation_config["InputDataS3Uri"] = input_s3_uri
    else:
        generation_config["InputDatasetArn"] = input_dataset_arn
    output_data_config = {"S3OutputPath": output_s3_uri}
    if hub_name:
        output_data_config["HubName"] = hub_name
    return {
        "DatasetName": dataset_name,
        "GenerationConfig": generation_config,
        "OutputDataConfig": output_data_config,
    }


def to_job_config_kwargs(document):
    """The DatasetJobConfig keyword arguments for a request document."""
    return {
        "dataset_name": document["DatasetName"],
        "generation_config": document["GenerationConfig"],
        "output_data_config": document["OutputDataConfig"],
        "schema_version": LABELING_SCHEMA_VERSION,
    }


def request_summary(document):
    """The one-line facts of a request, for an approval summary and the run log.

    Task, panel, routing, target, whether gold rides along; an opt-in option is listed
    only when the request carries it.
    """
    gc = document["GenerationConfig"]
    cfg = gc["DatasetOptions"]["Config"]
    summary = {
        "task": task_of_type(gc["DatasetOptions"]["Type"]),
        "dataset_name": document["DatasetName"],
        "text_fields": list(cfg["TextFields"]),
        "models": [m["ModelId"] for m in gc["ModelConfig"]["SampleGenerationModelConfigs"]],
        "routing": (cfg.get("Routing") or {}).get("ReviewDatasetName"),
        "target": (cfg.get("Scoring") or {}).get("Target"),
        "graders": [g["ModelId"] for g in (cfg.get("Scoring") or {}).get("Graders") or []],
        "evaluation_data": cfg.get("EvaluationData"),
        "demos": (cfg.get("Demos") or {}).get("S3Uri"),
        "input": gc.get("InputDataS3Uri") or gc.get("InputDatasetArn"),
        "spec": gc["DatasetSpecification"]["SpecS3Uri"],
        "output": document["OutputDataConfig"]["S3OutputPath"],
    }
    if "Confidence" in cfg:
        summary["confidence_method"] = cfg["Confidence"]["Method"]
        for key, name in (
            ("Estimator", "confidence_estimator"),
            ("Models", "judge_models"),
            ("Recalibration", "confidence_recalibration"),
            ("FloorBypass", "confidence_floor_bypass"),
        ):
            if key in cfg["Confidence"]:
                summary[name] = cfg["Confidence"][key]
    if "Optimizer" in cfg:
        summary["optimizer"] = cfg["Optimizer"]
    return summary


_OPTIONS_CLASSES = {
    "classification": S.ClassificationLabelingOptions,
    "ner": S.NerLabelingOptions,
    "qa": S.QaLabelingOptions,
    "ordinal": S.OrdinalLabelingOptions,
    "llm_judge": S.LlmJudgeLabelingOptions,
    "summarization": S.SummarizationLabelingOptions,
}


def _judge(value: Any) -> S.JudgeModel:
    if isinstance(value, dict):
        return S.JudgeModel(
            model_id=value["ModelId"], reasoning_effort=value.get("ReasoningEffort")
        )
    return S.JudgeModel(model_id=value)


def labeling_options_from_document(document: Dict[str, Any]) -> S.LabelingOptions:
    """The typed ``*LabelingOptions`` behind a request document's ``DatasetOptions``.

    Args:
        document: :func:`build_labeling_request`'s return.

    Returns:
        The options object for the document's task; its ``to_dataset_options()`` equals
        the document's ``GenerationConfig.DatasetOptions``.
    """
    options = document["GenerationConfig"]["DatasetOptions"]
    task = task_of_type(options["Type"])
    cfg = options["Config"]
    kwargs: Dict[str, Any] = {"text_fields": list(cfg["TextFields"])}
    if cfg.get("Demos"):
        kwargs["demos"] = S.Demos(s3_uri=cfg["Demos"]["S3Uri"])
    if cfg.get("Routing"):
        kwargs["routing"] = S.Routing(review_dataset_name=cfg["Routing"]["ReviewDatasetName"])
    scoring = cfg.get("Scoring") or {}
    if task == "summarization":
        kwargs["graders"] = [
            S.Grader(
                model_id=g["ModelId"],
                target_tokens_per_minute=g.get("TargetTokensPerMinute"),
                reasoning_effort=g.get("ReasoningEffort"),
            )
            for g in scoring.get("Graders") or []
        ]
    else:
        ev = cfg.get("EvaluationData")
        if ev:
            kwargs["evaluation_data"] = S.EvaluationData(
                label_field=ev["LabelField"],
                s3_uri=ev.get("S3Uri"),
                dataset_arn=ev.get("DatasetArn"),
            )
        if scoring.get("Target") is not None:
            kwargs["scoring"] = S.Scoring(target=float(scoring["Target"]))
    conf = cfg.get("Confidence")
    if conf:
        kwargs["confidence"] = S.Confidence(
            method=conf["Method"],
            estimator=conf.get("Estimator"),
            models=[_judge(m) for m in conf["Models"]] if conf.get("Models") else None,
            recalibration=conf.get("Recalibration"),
            floor_bypass=conf.get("FloorBypass"),
        )
    opt = cfg.get("Optimizer")
    if opt:
        kwargs["optimizer"] = S.Optimizer(
            mode=opt["Mode"],
            teacher_model=_judge(opt["TeacherModel"]) if opt.get("TeacherModel") else None,
            seed=opt.get("Seed"),
            parser=opt.get("Parser"),
            elicitation=opt.get("Elicitation"),
        )
    rev = cfg.get("Review")
    if rev:
        kwargs["review"] = S.Review(
            label_field=rev.get("LabelField"),
            scope=rev.get("Scope"),
            predictions_s3_uris=rev.get("PredictionsS3Uris"),
            rules_s3_uri=rev.get("RulesS3Uri"),
            judge_models=(
                [_judge(m) for m in rev["JudgeModels"]] if rev.get("JudgeModels") else None
            ),
            judge_scope=rev.get("JudgeScope"),
            max_judge_rows=rev.get("MaxJudgeRows"),
            estimate_only=rev.get("EstimateOnly"),
        )
    unmapped = sorted(set(cfg) - _MAPPED_CONFIG_KEYS)
    if unmapped:
        # A key this does not map would be dropped from the job silently: refuse it.
        raise LabelingRequestError(
            f"DatasetOptions.Config carries {', '.join(unmapped)}, which the typed "
            "options do not take; nothing was submitted"
        )
    return _OPTIONS_CLASSES[task](**kwargs)


# Every Config key labeling_options_from_document carries into the typed options.
_MAPPED_CONFIG_KEYS = frozenset(
    {
        "TextFields",
        "Demos",
        "Routing",
        "Scoring",
        "EvaluationData",
        "Confidence",
        "Optimizer",
        "Review",
    }
)


def to_dataset_job_config(document: Dict[str, Any], sagemaker_session=None) -> DatasetJobConfig:
    """The :class:`DatasetJobConfig` for a request document, with typed ``dataset_options``.

    The document's ``GenerationConfig.DatasetOptions`` becomes the typed options object
    (:func:`labeling_options_from_document`); everything else passes through as the raw
    request, pinned to ``LABELING_SCHEMA_VERSION``.

    Args:
        document: :func:`build_labeling_request`'s return.
        sagemaker_session: The region-bound session the job runs under.

    Returns:
        The config to hand to ``DataPreparer.generate``.
    """
    gc = {k: v for k, v in document["GenerationConfig"].items() if k != "DatasetOptions"}
    kwargs = {
        "dataset_name": document["DatasetName"],
        "generation_config": gc,
        "output_data_config": document["OutputDataConfig"],
        "schema_version": LABELING_SCHEMA_VERSION,
        "dataset_options": labeling_options_from_document(document),
    }
    if sagemaker_session is not None:
        kwargs["sagemaker_session"] = sagemaker_session
    return DatasetJobConfig(**kwargs)
