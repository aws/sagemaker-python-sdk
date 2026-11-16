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
"""The per-task row-correctness rule the labeling job applies, ported for the client.

A labeling job scores its own gold rows and reports ``HeldOutAccuracy``. Several
client-side steps need the same row rule (folding a human review into a seed,
re-scoring a split the job did not see, the spot-check verdict), and they must agree
with the job's certificate, so the rule lives here once:

* ``classification``: prediction and gold are equal as text (:func:`label_text`), so
  the prediction ``"1"`` matches a JSON-number gold ``1``.
* ``ner``: the ``(type, text)`` span sets are equal, trimmed and case-sensitive.
* ``qa``: the SQuAD-normalized answer matches any acceptable gold answer, or a
  relaxed rule chosen with ``qa_match``.
* ``ordinal``: the prediction rounds to the gold tier.
* ``llm_judge``: ``|prediction - gold| <= tolerance``; the job's tolerance is 5% of
  the spec's score scale and never under 0.5.

A missing prediction is never correct under any rule.
"""

from __future__ import absolute_import

import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# Task types that ship without an accuracy promise. The labeling job certifies none of
# them (StatusReason task_not_certifiable) and the report shows no certified bound.
NO_PROMISE_TASKS = ("ordinal", "llm_judge", "summarization")

# task -> (the prediction field the job writes, the gold field the seeds carry).
TASK_FIELDS: Dict[str, Tuple[str, str]] = {
    "classification": ("label", "gt_label"),
    "ner": ("entities", "gt_entities"),
    "qa": ("answer", "gt_answers"),
    "llm_judge": ("judge_score", "gt_score"),
    "ordinal": ("score", "gt_score"),
}
# A prediction field other than `label` -> the gold field a fold or overlay names for it.
GT_FIELD_FOR_PRED = {
    "score": "gt_score",
    "judge_score": "gt_score",
    "entities": "gt_entities",
    "answer": "gt_answers",
    "summary": "gt_summary",
}
# llm_judge: a score within this distance of the (often fractional) gold counts as
# correct; 0.5 = "rounds to the same tier" on an integer scale. The job widens it to 5%
# of a wider stated scale (its held_out_report.json records `correct_tolerance`).
DEFAULT_CORRECT_TOLERANCE = 0.5

QA_MATCH_DEFAULT = "exact"
_SQUAD_ARTICLES = re.compile(r"\b(a|an|the)\b")
_SQUAD_PUNCT = str.maketrans("", "", string.punctuation)


def no_promise_reason(task: Optional[str]) -> Optional[str]:
    """Why ``task`` gets no certificate, or ``None`` when it can get one."""
    if task in NO_PROMISE_TASKS:
        return f"no accuracy promise for {task} (overview.md Limits)"
    return None


def ner_span_set(spans: Any) -> Optional[Set[Tuple[str, str]]]:
    """The ``(type, text)`` span set of a NER entity list, or ``None``.

    Type and text are trimmed; otherwise the match is exact and case-sensitive
    (``Anna`` != ``anna``), the rule behind the job's held-out NER metric. Non-dict
    entries are skipped; order and extra keys (a voted span's ``confidence``) never
    matter. ``None`` when ``spans`` is not a list (a missing prediction, or a JSON
    string, which no producer writes): such a row is never correct.
    """
    if not isinstance(spans, list):
        return None
    return {
        (str(e.get("type", "")).strip(), str(e.get("text", "")).strip())
        for e in spans
        if isinstance(e, dict)
    }


def label_text(v: Any) -> Any:
    """A classification label as the job compares it.

    ``None`` stays ``None``, a bool is ``"true"``/``"false"``, an int or integral float
    is its digits, another float is ``str()``, a string is stripped (case kept). A list
    or dict is returned unchanged.
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else str(v)
    if isinstance(v, str):
        return v.strip()
    return v


def label_matches(pred: Any, gold: Any) -> bool:
    """Classification correctness: ``pred`` and ``gold`` are equal as :func:`label_text`.

    A missing prediction or gold is never correct.
    """
    p, g = label_text(pred), label_text(gold)
    return p is not None and g is not None and p == g


def squad_normalize(s: Any) -> str:
    """The SQuAD ``normalize_answer`` rule.

    Lowercase, strip punctuation and articles, collapse whitespace. ``None``
    normalizes to ``""``.
    """
    if s is None:
        return ""
    s = str(s).lower().translate(_SQUAD_PUNCT)
    return " ".join(_SQUAD_ARTICLES.sub(" ", s).split())


def parse_qa_match(spec: Any) -> str:
    """Normalize a QA match rule: ``"exact"``, ``"contains"`` or ``"f1>=X"``.

    Args:
        spec: The rule as given (``None`` means the default, ``exact``). Whitespace
            and case are ignored.

    Returns:
        The normalized rule; ``f1>=X`` keeps ``X`` with ``0 < X <= 1``.

    Raises:
        DataPreparerOperatorError: The rule is none of the three.
    """
    if spec is None:
        return QA_MATCH_DEFAULT
    s = str(spec).strip().lower().replace(" ", "")
    if s in ("exact", "contains"):
        return s
    if s.startswith("f1>="):
        try:
            x = float(s[4:])
        except ValueError:
            x = None
        if x is not None and 0.0 < x <= 1.0:
            return f"f1>={x:g}"
    raise DataPreparerOperatorError(
        f"--qa-match must be exact, contains or f1>=X with 0 < X <= 1, got {spec!r}"
    )


def _qa_tokens(s: Any) -> List[str]:
    return squad_normalize(s).split()


def _qa_f1(p: List[str], g: List[str]) -> float:
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def _qa_run_in(small: List[str], big: List[str]) -> bool:
    n = len(small)
    return 0 < n <= len(big) and any(big[i : i + n] == small for i in range(len(big) - n + 1))


def qa_correct(pred: Any, gold: Any, qa_match: Optional[str] = None) -> int:
    """1/0 QA correctness under ``qa_match``.

    ``exact`` (the job's rule) is the SQuAD-normalized match against any gold answer.
    The relaxed rules change only answered rows: an empty answer is right iff the
    question is unanswerable under every rule. A ``None`` prediction is a failed row
    (never correct), not a deliberate ``""``.

    Args:
        pred: The predicted answer.
        gold: One gold answer or a list of acceptable ones.
        qa_match: ``exact``, ``contains`` or ``f1>=X``; ``None`` is ``exact``.

    Returns:
        1 when correct, else 0.
    """
    m = parse_qa_match(qa_match)
    if pred is None:
        return 0
    golds = (gold or [""]) if isinstance(gold, list) else [gold]
    if m == "exact":
        return 1 if squad_normalize(pred) in {squad_normalize(g) for g in golds} else 0
    p = _qa_tokens(pred)
    answers = [t for t in (_qa_tokens(g) for g in golds) if t]
    if not answers:
        return 0 if p else 1
    if not p:
        # an empty answer is right iff the question may be unanswerable: the same
        # rule as exact match, where an empty gold among the answers accepts it
        return 1 if any(not _qa_tokens(g) for g in golds) else 0
    if m == "contains":
        return 1 if any(_qa_run_in(g, p) or _qa_run_in(p, g) for g in answers) else 0
    thr = float(m[4:])
    return 1 if max(_qa_f1(p, g) for g in answers) >= thr - 1e-12 else 0


def as_number(v: Any) -> Optional[float]:
    """``v`` as a float, or ``None`` for a bool, ``None`` or a non-numeric value."""
    if isinstance(v, bool) or v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def task_correct(
    task: Optional[str],
    pred: Any,
    gold: Any,
    tolerance: float = DEFAULT_CORRECT_TOLERANCE,
    qa_match: Optional[str] = None,
) -> int:
    """1/0 correctness of ``pred`` against ``gold`` under ``task``'s rule.

    This is the rule the labeling job's finalizer applies, so counts built from it
    (a fold's confirmed and corrected rows, a spot-check verdict) agree with the job's
    certificate. Use it row by row; use
    :func:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers.score`
    for a whole file with the task's aggregate metrics.

    Args:
        task: ``classification``, ``ner``, ``qa``, ``ordinal`` or ``llm_judge``. Any
            other value (and ``None``) compares labels as text.
        pred: The prediction; ``None`` is never correct.
        gold: The gold value, not ``None``.
        tolerance: ``llm_judge`` only: the largest ``|pred - gold|`` that counts.
        qa_match: ``qa`` only: see :func:`qa_correct`.

    Returns:
        1 when correct, else 0.
    """
    if task == "ner":
        p = ner_span_set(pred)
        return 1 if p is not None and p == ner_span_set(gold) else 0
    if task == "qa":
        return qa_correct(pred, gold, qa_match)
    if task in ("ordinal", "llm_judge"):
        p, g = as_number(pred), as_number(gold)
        if p is None or g is None:
            return 1 if pred is not None and str(pred).strip() == str(gold).strip() else 0
        if task == "ordinal":
            return 1 if round(p) == round(g) else 0
        return 1 if abs(p - g) <= tolerance + 1e-9 else 0
    return 1 if label_matches(pred, gold) else 0


def job_tolerance(scale: Optional[Tuple[float, float]]) -> float:
    """The ``llm_judge`` tolerance the job uses for a ``[low, high]`` score scale.

    5% of the span, never under the 0.5 of an integer scale (0.5 on 1-5, 5.0 on 0-100).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import job_tolerance

        tolerance = job_tolerance((1, 5))  # the llm_judge tolerance the job applies on a 1-5 scale
    """
    if not scale:
        return DEFAULT_CORRECT_TOLERANCE
    return max(DEFAULT_CORRECT_TOLERANCE, 0.05 * (float(scale[1]) - float(scale[0])))
