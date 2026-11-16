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
"""Deterministic task-type detection from the spec and a sample of the labeled seed.

Auto-labeling supports several task types (classification, ner, summarization, qa,
llm_judge, ordinal). :func:`detect` reads the one-page spec plus a small sample of
rows and picks the task from structural evidence, so nobody has to guess. Run it
first; its ``task`` is the default for the run. Pass the *labeled seed*, not the
corpus: the seed's ground-truth (``gt_*``) column is the strongest signal, and the
seed carries every input column the corpus has plus that gold column. The corpus is
unlabeled, so it never fires the best clue; only a cold start with no seed yet passes
the corpus head rows.

The result is a suggestion with a confidence, the signals used and a ``decision``
tier: ``adopt`` (confidence at least 0.85), ``confirm`` (0.5 to 0.85: adopt but say
so) or ``ask`` (under 0.5). The person confirms the task at the run's gate either way:
this is infer, then confirm, not a silent pick.

Once chosen, the task must be the one the labeling run and the scorer use. Do not
force a non-default shape (summarization, ordinal, llm_judge, ner, qa) into
classification: summarization emits free text and ordinal / llm_judge emit a number,
so a label vote and macro-F1 are meaningless for them.

Signals, highest-precision first:

1. A ground-truth field in the sample rows: ``gt_entities`` -> ner, ``gt_summary`` ->
   summarization, ``gt_answers`` -> qa, ``gt_score`` -> ordinal or llm_judge,
   ``gt_label`` -> classification.
2. ``gt_score`` disambiguation: ``source`` + ``candidate`` input fields -> llm_judge; a
   small bounded integer scale -> ordinal; else llm_judge.
3. Input-field signature with no gold: ``question`` + ``context`` -> qa; ``source`` +
   ``candidate`` -> llm_judge.
4. The spec's Output Schema: the job's JSON reply key (``entities``, ``summary``,
   ``answer``, ``judge_score``, ``score``, ``label``) or an older line marker
   (``ENTITIES:``, ``SUMMARY:``, ...). ``score`` is ordinal or judge: confirm. A
   ``Label Set`` table -> classification.
5. An image path field is reported, so the caller can say only the text fields are
   labeled.
"""

from __future__ import absolute_import

import re
from typing import Any, Dict, Iterable, Optional, Sequence

from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult


class TaskDetection(OperatorResult):
    """What :func:`detect` concluded.

    ``task`` is the suggested labeling task, ``confidence`` in ``[0, 1]``, ``reason``
    one line of why, ``signals`` the evidence used (``gt_field``, ``input_fields``,
    ``output_marker``, ``has_label_set``, ``image_field``) and ``decision`` the tier:
    ``adopt``, ``confirm`` or ``ask``.
    """

    task: str
    confidence: float
    reason: str
    signals: Dict[str, Any]
    decision: str


_GT_FIELD_TASK = {
    "gt_entities": "ner",
    "gt_summary": "summarization",
    "gt_answers": "qa",
    "gt_label": "classification",
}

_IMAGE_KEYS = ("image_path", "image", "image_url", "img_path")


def _first_present(rows: Sequence[Dict[str, Any]], keys: Iterable[str]) -> Optional[str]:
    """The first key, in ``keys`` order (not row order), that any row carries non-None."""
    for k in keys:
        for r in rows:
            if isinstance(r, dict) and r.get(k) is not None:
                return k
    return None


# gt-field precedence, most-specific first (gt_label is the generic fallback, last).
# Used so a mixed/messy dataset picks the gt field by precision, not by row order.
_GT_PRECEDENCE = ("gt_entities", "gt_summary", "gt_answers", "gt_score", "gt_label")


def _looks_ordinal(rows: Sequence[Dict[str, Any]], gt: str = "gt_score") -> bool:
    """Whether ``gt_score`` values look like a small bounded integer rating scale.

    A few distinct integers within a span of at most 10 read as ordinal; anything else
    reads as a continuous judge score.
    """
    nums = [
        r.get(gt)
        for r in rows
        if isinstance(r, dict)
        and isinstance(r.get(gt), (int, float))
        and not isinstance(r.get(gt), bool)
    ]
    if not nums:
        return False
    all_int = all(float(v).is_integer() for v in nums)
    distinct = len({int(v) for v in nums})
    return all_int and (max(nums) - min(nums) <= 10) and distinct <= 11


def _output_schema_block(spec_text: Optional[str]) -> str:
    """The text under an ``## Output Schema`` heading, or the whole spec without one.

    The output markers are matched there, so a ``## Summary:`` heading or a prose
    ``ANSWER:`` elsewhere does not set the task.
    """
    m = re.search(r"(?im)^#+\s*output\s*schema\b(.*?)(?=^#+\s|\Z)", spec_text or "", re.DOTALL)
    return m.group(1) if m else (spec_text or "")


# The job's reply key per task (engine/sdg_backend.PREDICTION_FIELDS); score is
# ordinal or llm_judge, decided later.
_REPLY_KEY_TASKS = {
    "entities": "ner",
    "summary": "summarization",
    "answer": "qa",
    "judge_score": "llm_judge",
    "score": "score",
    "label": "classification",
}


def _spec_signals(spec_text: Optional[str]) -> Dict[str, Any]:
    """The spec's signals.

    The Label Set heading (whole spec) and the output markers
    (:func:`_output_schema_block`).
    """
    s = spec_text or ""
    sig = {}
    has_label_set = bool(re.search(r"#+\s*label\s*set", s, re.IGNORECASE))
    if has_label_set:
        sig["has_label_set"] = True
    block = _output_schema_block(s)
    schema = block.upper()
    for marker, task in (
        ("ENTITIES", "ner"),
        ("SUMMARY", "summarization"),
        ("ANSWER", "qa"),
        ("SCORE", "score"),
        ("LABEL", "classification"),
    ):
        # Match a marker only as an output-contract LINE (`MARKER:` at line start),
        # never a `## Summary:` heading or a prose mention of `ANSWER:` - both would
        # otherwise mis-fire even in the whole-spec fallback.
        if re.search(r"(?m)^\s*" + marker + r"\s*:", schema):
            sig["output_marker"] = task
            return sig
    # The job's JSON reply format (`{"reasoning": ..., "entities": [...]}`). Inside an
    # Output Schema section the task is the first known key after the first
    # `"reasoning":`, so prose before the reply and a second example after it do not
    # change it. Without that heading (the NER prompt asks for "An output schema:" as
    # a numbered item) only lines that start with `{"reasoning":` are read.
    if not re.search(r"(?im)^#+\s*output\s*schema\b", s):
        block = "\n".join(ln for ln in s.splitlines() if re.match(r'\s*\{\s*"reasoning"\s*:', ln))
    reply = re.search(r'"reasoning"\s*:', block)
    if reply:
        for km in re.finditer(r'"(\w+)"\s*:', block[reply.end() :]):
            if km.group(1) in _REPLY_KEY_TASKS:
                sig["output_marker"] = _REPLY_KEY_TASKS[km.group(1)]
                break
    return sig


def _decide(conf: float) -> str:
    """The decision tier for a confidence.

    ``adopt`` (at least 0.85), ``confirm`` (0.5 to 0.85: adopt but call it out) or
    ``ask`` (under 0.5). The person confirms the task at the run's gate either way.
    """
    if conf >= 0.85:
        return "adopt"
    if conf >= 0.5:
        return "confirm"
    return "ask"


def _result(task, conf, reason, signals) -> TaskDetection:
    """Assemble the detector output, attaching the deterministic decision tier."""
    return TaskDetection(
        task=task, confidence=conf, reason=reason, signals=signals, decision=_decide(conf)
    )


def detect(  # pylint: disable=too-many-return-statements
    spec_text: Optional[str], sample_rows: Optional[Iterable[Any]]
) -> TaskDetection:
    """Suggest the labeling task from the spec and a sample of the labeled seed.

    Use it before anything else in a labeling run. Pass the labeled seed's rows (a
    couple of hundred is plenty); pass corpus head rows only on a cold start with no
    seed. Do not treat the answer as final when ``decision`` is ``confirm`` or ``ask``:
    say what was inferred and let the person correct it at the gate.

    Args:
        spec_text: The labeling spec's Markdown; ``None`` or empty when there is none yet.
        sample_rows: Sample rows; non-dict entries are ignored.

    Returns:
        A :class:`TaskDetection`; ``to_json()`` is the detector's output document.

    Example::

        detect("## Label Set ...", [{"text": "t", "gt_label": "billing"}]).task
        # 'classification'
    """
    rows = [r for r in (sample_rows or []) if isinstance(r, dict)]
    signals = {}

    img = _first_present(rows, _IMAGE_KEYS)
    if img:
        signals["image_field"] = img

    # 1 + 2. ground-truth field in the sample (strongest signal). Precision-ordered
    # (gt_score before the generic gt_label), row-order-independent.
    gt = _first_present(rows, _GT_PRECEDENCE)
    if gt:
        signals["gt_field"] = gt
        if gt == "gt_score":
            has_sc = _first_present(rows, ["source"]) and _first_present(rows, ["candidate"])
            if has_sc:
                task, conf, why = "llm_judge", 0.9, "gt_score with source+candidate fields"
            elif _looks_ordinal(rows):
                task, conf, why = "ordinal", 0.85, "gt_score on a small bounded integer scale"
            else:
                task, conf, why = "llm_judge", 0.7, "gt_score present (non-ordinal range)"
            return _result(task, conf, f"{why} -> {task}", signals)
        task = _GT_FIELD_TASK[gt]
        return _result(task, 0.95, f"ground-truth field {gt} -> {task}", signals)

    # 3. input-field signature (no gt field in the sample)
    if _first_present(rows, ["question"]) and _first_present(rows, ["context"]):
        signals["input_fields"] = ["question", "context"]
        return _result("qa", 0.75, "question+context input fields -> extractive QA", signals)
    if _first_present(rows, ["source"]) and _first_present(rows, ["candidate"]):
        signals["input_fields"] = ["source", "candidate"]
        return _result("llm_judge", 0.7, "source+candidate input fields -> LLM-as-a-judge", signals)

    # 4. spec Output Schema markers
    ss = _spec_signals(spec_text)
    signals.update(ss)
    marker = ss.get("output_marker")
    if marker == "ner":
        return _result("ner", 0.7, "spec output schema emits ENTITIES", signals)
    if marker == "summarization":
        return _result("summarization", 0.7, "spec output schema emits SUMMARY", signals)
    if marker == "qa":
        return _result("qa", 0.6, "spec output schema emits ANSWER", signals)
    if marker == "llm_judge":
        return _result("llm_judge", 0.6, "spec output schema emits judge_score", signals)
    if marker == "score":
        return _result(
            "ordinal",
            0.5,
            "spec output schema emits SCORE; ordinal vs llm_judge is "
            "ambiguous from the spec alone - confirm",
            signals,
        )
    if marker == "classification" or ss.get("has_label_set"):
        return _result(
            "classification", 0.6, "spec has a Label Set / emits LABEL -> classification", signals
        )

    return _result(
        "classification",
        0.3,
        "no strong signal; defaulting to classification - confirm with the user",
        signals,
    )
