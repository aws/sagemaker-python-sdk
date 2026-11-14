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
"""Choosing between candidates by a score, with a do-no-harm floor.

Two primitives with no labeling vocabulary in them. :func:`select_with_floor`
answers "is the best candidate better than the baseline by enough?", with an
optional predicate that disqualifies a candidate before it can win.
:func:`heldout_review_flag` answers "is the held-out number far enough below the
selecting score to suspect over-fitting?" without changing any choice.
"""

from __future__ import absolute_import

from typing import Any, Callable, Dict, Iterable, Optional


def select_with_floor(
    base_val: Optional[float],
    candidates: Iterable[Any],
    min_gain: float,
    *,
    val: Callable[[Any], Optional[float]] = lambda c: c["val"],
    skip: Optional[Callable[[Any], Any]] = None,
) -> Dict[str, Any]:
    """Pick the best candidate that beats ``base_val`` by at least ``min_gain``.

    Candidates are ranked by ``val``; any that ``skip`` disqualifies are dropped first.

    Use it when several alternatives were measured on the same held-out data and one
    must be chosen mechanically: the best score wins, never the last one tried, and
    only if it beats the baseline by a margin the measurement can resolve. A
    candidate whose ``val`` is ``None`` (a scoring step that failed) stays in the
    input but never ranks.

    Args:
        base_val: The baseline's score; ``None`` when there is none, in which case
            nothing can clear the floor.
        candidates: The alternatives, in any order.
        min_gain: The least ``best - base_val`` that counts as an improvement.
        val: Candidate -> score. Defaults to ``c["val"]``.
        skip: Candidate -> a truthy reason to drop it, or falsy to keep it.

    Returns:
        ``{"best", "gain", "cleared", "skipped"}``: the best surviving candidate (or
        ``None``), its gain over ``base_val`` (``None`` without a baseline or a best),
        whether ``gain >= min_gain``, and ``[(candidate, reason), ...]`` for the
        candidates ``skip`` dropped, in rank order.

    Example::

        select_with_floor(0.885, [{"val": 0.921}, {"val": 0.880}], 0.02)
        # {"best": {"val": 0.921}, "gain": 0.036, "cleared": True, "skipped": []}
    """
    skipped = []
    best = None
    ranked = [c for c in candidates if val(c) is not None]
    for c in sorted(ranked, key=val, reverse=True):
        if skip is not None:
            reason = skip(c)
            if reason:
                skipped.append((c, reason))
                continue
        best = c
        break
    if best is None:
        return {"best": None, "gain": None, "cleared": False, "skipped": skipped}
    if base_val is None:
        return {"best": best, "gain": None, "cleared": False, "skipped": skipped}
    gain = val(best) - base_val
    return {"best": best, "gain": gain, "cleared": gain >= min_gain, "skipped": skipped}


def heldout_review_flag(
    selected_val: Optional[float], test_score: Optional[float], tol: float = 0.10
) -> Dict[str, Any]:
    """An advisory over-fit flag.

    Raised when the held-out score is more than ``tol`` below the score that selected
    the candidate.

    It never changes which candidate ships; it tells the caller to look.

    Args:
        selected_val: The score the choice was made on.
        test_score: The score on data the choice never saw, or ``None`` when there
            is none.
        tol: The gap that raises the flag.

    Returns:
        ``{"flag", "val", "test", "gap", "tol"}`` with ``gap = selected_val -
        test_score`` rounded to 4 places; ``flag``, ``gap`` and ``test`` are ``None``
        without a test score.
    """
    if test_score is None:
        return {"flag": None, "val": selected_val, "test": None, "gap": None, "tol": tol}
    sv = selected_val or 0.0
    ts = test_score or 0.0
    gap = round(sv - ts, 6)  # round off float noise so a gap == tol is not a flag
    return {
        "flag": gap > tol,
        "val": selected_val,
        "test": test_score,
        "gap": round(gap, 4),
        "tol": tol,
    }
