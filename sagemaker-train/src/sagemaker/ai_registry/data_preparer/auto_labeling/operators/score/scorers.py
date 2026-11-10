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
"""Score a seed-sized predictions file against gold, locally, with every metric.

The labeling job scores its own gold rows and reports ``HeldOutAccuracy``. This
operator re-scores rows the job did not score: a test-split re-score, a file labeled
outside a job, an A/B arm. It is standard-library only, deterministic, and makes no
model call, so the number it gives is one a reviewer can recompute.

Use :func:`score` when you hold predictions and gold for the same rows and want the
task's aggregate metrics. Do not use it on the corpus run's output: that run's labels
stay behind the API and its metrics are in the job's summary. Score only a seed split
(at most :data:`MAX_ROWS` rows).

``primary`` is the job's row rule for the task (the number the refinement loop steers
on): macro-F1 for classification, entity micro-F1 for NER, token-F1 for QA, and for
ordinal and llm_judge the job's accuracy. For the numeric tasks the agreement metrics
(quadratic weighted kappa, Spearman, Pearson, MAE, adjacent accuracy) are reported
beside it and never selected on; ``selection`` in the result says so.
"""

from __future__ import absolute_import

import json
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.score import _metrics as metrics
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import (
    DEFAULT_CORRECT_TOLERANCE,
    squad_normalize,
    task_correct,
)

# A seed-sized file; the corpus never comes to the client.
MAX_ROWS = 5000
TASKS = ("classification", "ner", "qa", "ordinal", "llm_judge")
GOLD_FIELDS = {
    "classification": "gt_label",
    "ner": "gt_entities",
    "qa": "gt_answers",
    "ordinal": "gt_score",
    "llm_judge": "gt_score",
}
PRED_FIELDS = {
    "classification": "label",
    "ner": "entities",
    "qa": "answer",
    "ordinal": "score",
    "llm_judge": "judge_score",
}
SELECTION_NOTE = (
    "steer on `primary` (the job's row rule, the same as HeldOutAccuracy); "
    "the agreement metrics are reported, never selected on"
)

Pair = Tuple[Any, Any]


class ScoreResult(OperatorResult):
    """What :func:`score` returns: the task's primary metric and everything beside it.

    ``primary`` is the number to steer on and ``primary_name`` says which metric it
    is. ``counts["correct"]`` and ``counts["total"]`` are present for every task, so a
    caller can hand them to a bound. The other fields are set for the tasks that
    define them. The fields are declared in the order the command line's ``score.json``
    lists them for every task (``to_json`` keeps it). ``breakdown`` is keyed by the
    class label as it appears in the data (a string, an int or a bool), so a numeric
    label set scores without a conversion and ``json.dumps`` writes the same keys
    the skill's ``score.json`` has.
    """

    primary: float
    primary_name: str
    exact_match: Optional[float] = None
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    breakdown: Optional[Dict[Union[str, int, bool], float]] = None
    confusion_pairs: Optional[List[Dict[str, Any]]] = None
    counts: Dict[str, int]
    n_no_ground_truth: int
    selection: Optional[str] = None
    correct_tolerance: Optional[float] = None
    quadratic_weighted_kappa: Optional[float] = None
    spearman: Optional[float] = None
    pearson: Optional[float] = None
    mae: Optional[float] = None
    adjacent_accuracy: Optional[float] = None

    def to_json(self) -> Dict[str, Any]:
        """The ``score.json`` document.

        ``breakdown`` keeps its keys as the labels are (``json.dumps`` writes an int
        label as ``"1"`` and a bool as ``"true"``, as the skill's file has them);
        pydantic's JSON mode would write the bool as ``"True"``.
        """
        out = super().to_json()
        if self.breakdown is not None:
            out["breakdown"] = dict(self.breakdown)
        return out


def align(
    preds: Sequence[Dict[str, Any]],
    gts: Sequence[Dict[str, Any]],
    id_field: Optional[str],
    pred_field: str,
    gt_field: str,
) -> List[Pair]:
    """``(prediction, gold)`` pairs aligned by id when both sides carry it, else by order.

    Args:
        preds: Prediction rows.
        gts: Gold rows.
        id_field: The id field; alignment by id needs it on every row of both sides.
        pred_field: The prediction field.
        gt_field: The gold field.

    Returns:
        One pair per prediction row that found its gold (by id) or per position (by
        order, up to the shorter file).
    """
    have_ids = id_field and all(id_field in r for r in preds) and all(id_field in r for r in gts)
    if have_ids:
        by_id = {r[id_field]: r.get(gt_field) for r in gts}
        return [
            (p.get(pred_field), by_id.get(p.get(id_field)))
            for p in preds
            if p.get(id_field) in by_id
        ]
    n = min(len(preds), len(gts))
    return [(preds[i].get(pred_field), gts[i].get(gt_field)) for i in range(n)]


def _r(x: float) -> float:
    return round(x, 6)


def score_classification(pairs: Sequence[Pair]) -> ScoreResult:
    """Macro-F1 (primary), accuracy, per-class F1 and the confusion pairs."""
    scored = [(p, g) for p, g in pairs if g is not None]
    m = metrics.macro_f1_parts(scored)
    return ScoreResult(
        primary=_r(m["macro_f1"]),
        primary_name="macro_f1",
        accuracy=_r(m["accuracy"]),
        breakdown={k: _r(v) for k, v in sorted(m["per_class"].items(), key=lambda kv: str(kv[0]))},
        confusion_pairs=[
            {"gt": g, "pred": p, "n": n} for (g, p), n in m["confusion"].most_common()
        ],
        counts={"correct": m["correct"], "total": m["total"]},
        n_no_ground_truth=len(pairs) - m["total"],
    )


def span_multiset(entities: Any) -> Counter:
    """``(type, text)`` span occurrences of an entity list, trimmed, case-sensitive.

    A multiset, so a document that mentions the same span twice contributes two. A
    JSON string is parsed; malformed entries score as misses, not crashes.
    """
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except ValueError:
            entities = []
    return Counter(
        (e["type"].strip(), e["text"].strip())
        for e in (entities or [])
        if isinstance(e, dict)
        and isinstance(e.get("text"), str)
        and e["text"].strip()
        and isinstance(e.get("type"), str)
        and e["type"].strip()
    )


def score_ner(pairs: Sequence[Pair]) -> ScoreResult:
    """Entity-level micro-F1 on ``(type, text)`` spans (primary) and per-type F1.

    The job's row rule (span sets equal) is the accuracy.
    """
    scored = [(p, g) for p, g in pairs if g is not None]
    m = metrics.multiset_micro_f1((span_multiset(p), span_multiset(g)) for p, g in scored)
    rows_correct = sum(task_correct("ner", p, g) for p, g in scored)
    return ScoreResult(
        primary=_r(m["f1"]),
        primary_name="micro_f1",
        accuracy=_r(rows_correct / len(scored) if scored else 0.0),
        precision=_r(m["precision"]),
        recall=_r(m["recall"]),
        breakdown={k: _r(v) for k, v in sorted(m["per_type"].items())},
        counts={
            "total_tp": m["tp"],
            "total_fp": m["fp"],
            "total_fn": m["fn"],
            "total": len(scored),
            "correct": rows_correct,
        },
        n_no_ground_truth=len(pairs) - len(scored),
    )


def score_qa(pairs: Sequence[Pair]) -> ScoreResult:
    """Token-F1 as SQuAD defines it (primary) and exact match over the gold answers."""
    scored = [(p, g) for p, g in pairs if g is not None]

    def golds(g):
        return (g or [""]) if isinstance(g, list) else [g]

    def tf1(p, g):
        return metrics.token_f1(squad_normalize(p).split(), squad_normalize(g).split())

    f1s = [max(tf1(p, g) for g in golds(gold)) if p is not None else 0.0 for p, gold in scored]
    ems = [
        (
            max(float(squad_normalize(p) == squad_normalize(g)) for g in golds(gold))
            if p is not None
            else 0.0
        )
        for p, gold in scored
    ]
    n = len(scored)
    correct = sum(task_correct("qa", p, g) for p, g in scored)
    return ScoreResult(
        primary=_r(sum(f1s) / n if n else 0.0),
        primary_name="token_f1",
        exact_match=_r(sum(ems) / n if n else 0.0),
        accuracy=_r(correct / n if n else 0.0),
        counts={"total": n, "missing": sum(1 for p, _ in scored if p is None), "correct": correct},
        n_no_ground_truth=len(pairs) - n,
    )


def _as_rank(v: Any, label_map: Optional[Dict[str, float]]):
    if v is None:
        return None, None
    try:
        return float(v), None
    except (TypeError, ValueError):
        pass
    if label_map:
        r = label_map.get(str(v).strip().lower())
        if r is not None:
            return r, None
    return None, str(v)


def score_numeric(
    pairs: Sequence[Pair],
    task: str,
    label_map: Optional[Dict[str, float]] = None,
    tolerance: float = DEFAULT_CORRECT_TOLERANCE,
) -> ScoreResult:
    """Ordinal / llm_judge: the job's row rule as ``primary``, agreement metrics beside.

    The row rule is the exact tier for ordinal and ``|pred - gold| <= tolerance`` for
    llm_judge. A missing or unparseable prediction is wrong, as the job scores it, so
    accuracy is over every gold-carrying row; the agreement metrics need two numbers
    per row and use the mapped rows only.

    Args:
        pairs: ``(prediction, gold)`` pairs.
        task: ``ordinal`` or ``llm_judge``.
        label_map: Label -> rank for gold given as words; keys are matched case- and
            whitespace-insensitively.
        tolerance: The llm_judge tolerance.

    Returns:
        The result, with ``correct_tolerance`` set for llm_judge.

    Raises:
        DataPreparerOperatorError: A gold label is non-numeric and ``label_map`` has no
            rank for it.
    """
    scored = [(p, g) for p, g in pairs if g is not None]
    # The map is case- and whitespace-insensitive however the caller wrote its keys.
    label_map = {str(k).strip().lower(): v for k, v in (label_map or {}).items()} or None
    pts, unmapped, missing = [], set(), 0
    for p, g in scored:
        pr, _ = _as_rank(p, label_map)
        gr, bad = _as_rank(g, label_map)
        if bad is not None:
            unmapped.add(bad)
        if pr is None or gr is None:
            missing += 1
            continue
        pts.append((pr, gr))
    if unmapped:
        raise DataPreparerOperatorError(
            "ordinal ground-truth labels are non-numeric and have no rank mapping: "
            + ", ".join(repr(x) for x in sorted(unmapped))
            + ". Pass --label-map (JSON label -> rank) or --spec so the ranks come from the "
            "Label Set order."
        )
    correct = sum(task_correct(task, p, g, tolerance=tolerance) for p, g in pts)
    n = len(pts)
    out = ScoreResult(
        primary=_r(correct / len(scored) if scored else 0.0),
        primary_name="accuracy",
        counts={"total": len(scored), "scored": n, "missing": missing, "correct": correct},
        n_no_ground_truth=len(pairs) - len(scored),
        selection=SELECTION_NOTE,
    )
    if task == "llm_judge":
        out.correct_tolerance = tolerance
    if n < 2:
        out.quadratic_weighted_kappa = out.spearman = out.pearson = out.mae = 0.0
        out.adjacent_accuracy = 0.0
        return out
    pr, gd = [x[0] for x in pts], [x[1] for x in pts]
    lv_p, lv_g = [int(round(x)) for x in pr], [int(round(x)) for x in gd]
    out.quadratic_weighted_kappa = _r(metrics.quadratic_weighted_kappa(lv_p, lv_g))
    out.spearman = _r(metrics.spearman(pr, gd))
    out.pearson = _r(metrics.pearson(pr, gd))
    out.mae = _r(sum(abs(a - b) for a, b in pts) / n)
    out.adjacent_accuracy = _r(sum(1 for a, b in zip(lv_p, lv_g) if abs(a - b) <= 1) / n)
    return out


def label_map_from_names(names: Sequence[str]) -> Optional[Dict[str, float]]:
    """Lower-cased label -> 1-based rank from a spec's Label Set order.

    ``None`` for no names.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import label_map_from_names, parse_label_set

        label_map = label_map_from_names([name for name, _ in parse_label_set(spec_text)])
    """
    return {n.strip().lower(): float(i + 1) for i, n in enumerate(names)} or None


def parse_label_map(text: str) -> Dict[str, float]:
    """A ``--label-map`` JSON object (label -> numeric rank) as lower-cased label -> float.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import parse_label_map

        label_map = parse_label_map('{"very negative": 1, "negative": 2, "neutral": 3}')

    Raises:
        DataPreparerOperatorError: Not JSON, or not a non-empty object of label ->
            number.
    """
    try:
        m = json.loads(text)
    except json.JSONDecodeError as e:
        raise DataPreparerOperatorError(
            f'--label-map must be JSON like \'{{"very negative": 1, "negative": 2}}\' ({e})'
        ) from e
    if (
        not isinstance(m, dict)
        or not m
        or not all(
            isinstance(k, str) and isinstance(v, (int, float)) and not isinstance(v, bool)
            for k, v in m.items()
        )
    ):
        raise DataPreparerOperatorError(
            "--label-map must be a non-empty JSON object of label -> numeric rank"
        )
    return {k.strip().lower(): float(v) for k, v in m.items()}


def score(
    task: str,
    pred_rows: Sequence[Dict[str, Any]],
    gt_rows: Sequence[Dict[str, Any]],
    *,
    id_field: Optional[str] = "id",
    pred_field: Optional[str] = None,
    gt_field: Optional[str] = None,
    label_map: Optional[Dict[str, float]] = None,
    tolerance: float = DEFAULT_CORRECT_TOLERANCE,
) -> ScoreResult:
    """Score ``pred_rows`` against ``gt_rows`` under ``task``'s rule.

    Use it on a seed split the job did not score. Rows align by ``id_field`` when both
    files carry it on every row, else by order. The gold rows must be a seed split,
    never the corpus and never a golden-review copy; a command-line caller enforces the
    row cap and the split guard before calling.

    Args:
        task: ``classification``, ``ner``, ``qa``, ``ordinal`` or ``llm_judge``.
        pred_rows: The prediction rows.
        gt_rows: The gold rows.
        id_field: The id field both files carry; ``None`` aligns by order.
        pred_field: The prediction field; the task's default when ``None``
            (``label``, ``entities``, ``answer``, ``score``, ``judge_score``).
        gt_field: The gold field; the task's default when ``None`` (``gt_label``,
            ``gt_entities``, ``gt_answers``, ``gt_score``).
        label_map: Ordinal / llm_judge: lower-cased label -> rank for gold given as
            words (see :func:`parse_label_map`, :func:`label_map_from_names`).
        tolerance: llm_judge: the largest ``|pred - gold|`` that counts as correct.
            The job uses 5% of the spec's scale, at least 0.5
            (:func:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule.job_tolerance`).

    Returns:
        The :class:`ScoreResult`; ``to_json()`` is the ``score.json`` document.

    Raises:
        DataPreparerOperatorError: No rows aligned, no aligned row carries gold in
            ``gt_field``, or an ordinal gold label has no rank.

    Example::

        result = score("classification", preds, gold)
        result.primary, result.primary_name, result.counts
        # (0.9125, 'macro_f1', {'correct': 46, 'total': 50})
    """
    pairs = align(
        pred_rows, gt_rows, id_field, pred_field or PRED_FIELDS[task], gt_field or GOLD_FIELDS[task]
    )
    if not pairs:
        raise DataPreparerOperatorError(
            "no prediction/ground-truth rows aligned: check that --pred and --gt "
            "are the right non-empty files and either share an --id-field or line "
            "up by order"
        )
    if all(g is None for _, g in pairs):
        raise DataPreparerOperatorError(
            f"0 of {len(pairs)} aligned rows carry gold in field "
            f"{gt_field or GOLD_FIELDS[task]!r}; pass --gt-field"
        )
    if task == "classification":
        return score_classification(pairs)
    if task == "ner":
        return score_ner(pairs)
    if task == "qa":
        return score_qa(pairs)
    return score_numeric(pairs, task, label_map, tolerance)
