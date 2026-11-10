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
"""Metric primitives with no labeling vocabulary in them.

These are the sums behind the task scorers in
:mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.score.scorers`: macro-F1 over
classes, micro-F1 over span multisets, SQuAD token-F1, quadratic weighted kappa,
Spearman and Pearson correlation, and average ranks with ties. Each takes plain
Python values and returns a float or a dict of floats; none reads a file or knows a
field name. Everything is standard-library and deterministic, so the number a caller
gets here is the number a reviewer can recompute by hand.
"""

from __future__ import absolute_import

import math
from collections import Counter, defaultdict
from collections.abc import Hashable
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def macro_f1_parts(pairs: Iterable[Tuple[Any, Any]]) -> Dict[str, Any]:
    """Per-class F1 and its macro average over ``(prediction, gold)`` pairs.

    Classes are averaged over gold support only (scikit-learn's rule): a class that
    appears only in predictions does not drag the average down. A ``None`` prediction
    counts as a miss for its gold class and as no false positive for any class.

    Args:
        pairs: ``(prediction, gold)`` pairs; gold must not be ``None``.

    Returns:
        ``{"macro_f1", "accuracy", "per_class", "correct", "total", "confusion"}``:
        ``per_class`` maps each gold class to its F1, ``confusion`` is a
        :class:`collections.Counter` of ``(gold, prediction)`` for the wrong rows.
    """
    scored = list(pairs)
    total = len(scored)
    correct = sum(1 for p, g in scored if p == g)
    counts: Dict[Any, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "total": 0})
    for p, g in scored:
        counts[g]["total"] += 1
        if p == g:
            counts[g]["tp"] += 1
        else:
            counts[g]["fn"] += 1
            if p is not None:
                counts[p]["fp"] += 1
    per_class = {}
    for lbl, c in counts.items():
        if lbl is None or c["total"] == 0:
            continue
        denom = 2 * c["tp"] + c["fp"] + c["fn"]
        per_class[lbl] = (2 * c["tp"] / denom) if denom else 0.0
    macro = sum(per_class.values()) / len(per_class) if per_class else 0.0
    return {
        "macro_f1": macro,
        "accuracy": correct / total if total else 0.0,
        "per_class": per_class,
        "correct": correct,
        "total": total,
        "confusion": Counter((g, p) for p, g in scored if p != g),
    }


def multiset_micro_f1(
    pairs: Iterable[Tuple[Counter, Counter]], type_of=lambda item: item[0]
) -> Dict[str, Any]:
    """Micro precision, recall and F1 over multisets, with a per-type F1.

    Each pair is ``(predicted, gold)`` as :class:`collections.Counter` objects whose
    keys are the items (for NER, ``(type, text)`` spans). A key that occurs twice in
    the gold contributes two matches. ``type_of`` maps an item to the type the
    per-type breakdown groups by.

    Args:
        pairs: ``(predicted_counter, gold_counter)`` pairs.
        type_of: Item -> type key. Defaults to the item's first element.

    Returns:
        ``{"precision", "recall", "f1", "per_type", "tp", "fp", "fn"}`` with
        ``per_type`` mapping each type to its F1.
    """
    tp: Counter = Counter()
    fp: Counter = Counter()
    fn: Counter = Counter()
    for ps, gs in pairs:
        for item, n in (ps & gs).items():
            tp[type_of(item)] += n
        for item, n in (ps - gs).items():
            fp[type_of(item)] += n
        for item, n in (gs - ps).items():
            fn[type_of(item)] += n
    t, p, n = sum(tp.values()), sum(fp.values()), sum(fn.values())
    prec = t / (t + p) if t + p else 0.0
    rec = t / (t + n) if t + n else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    per_type = {}
    for k in set(tp) | set(fp) | set(fn):
        pp = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        rr = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        per_type[k] = 2 * pp * rr / (pp + rr) if pp + rr else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "per_type": per_type,
        "tp": t,
        "fp": p,
        "fn": n,
    }


def token_f1(pred_tokens: Sequence[Hashable], gold_tokens: Sequence[Hashable]) -> float:
    """The SQuAD token-F1 of two token lists: 1.0 when both are empty, 0.0 when one is."""
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if not same:
        return 0.0
    pr, rc = same / len(pred_tokens), same / len(gold_tokens)
    return 2 * pr * rc / (pr + rc)


def average_ranks(xs: Sequence[float]) -> List[float]:
    """1-based ranks of ``xs`` with ties given the average of their positions."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks, i = [0.0] * len(xs), 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation of two equal-length sequences.

    0.0 under two points, or when either sequence has no variance.
    """
    n = len(a)
    if n < 2:
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db) if da and db else 0.0


def spearman(a: Sequence[float], b: Sequence[float]) -> float:
    """Spearman rank correlation: :func:`pearson` over :func:`average_ranks`."""
    return pearson(average_ranks(a), average_ranks(b))


def quadratic_weighted_kappa(pred: Sequence[int], gold: Sequence[int]) -> float:
    """Cohen's kappa with quadratic weights over the integer levels both raters used.

    Args:
        pred: Integer levels one rater gave.
        gold: Integer levels the other gave, same length.

    Returns:
        The kappa; 1.0 when only one level occurs, or when agreement is perfect and
        chance agreement is total.
    """
    lo, hi = min(*pred, *gold), max(*pred, *gold)
    k = hi - lo + 1
    if k <= 1:
        return 1.0
    n = len(pred)
    obs = [[0] * k for _ in range(k)]
    for p, g in zip(pred, gold):
        obs[p - lo][g - lo] += 1
    row = [sum(obs[i]) for i in range(k)]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / float((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * (row[i] * col[j] / n)
    if den == 0.0:
        return 1.0 if num == 0.0 else 0.0
    return 1.0 - num / den
