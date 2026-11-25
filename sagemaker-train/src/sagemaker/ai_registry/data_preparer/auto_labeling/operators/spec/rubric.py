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
"""Anchored-rubric helpers for the numeric judge tasks (llm_judge, and optionally ordinal).

An anchored rubric is the default spec structure for ``llm_judge``: a ``### Score
Levels`` section with one entry per score level, each carrying one or two example
rationales drawn from the dev split of the labeled seed. The anchors bring the judge's
ranking and its score scale closer to the experts'.

Two pure functions, no network: :func:`select_rubric_anchors` picks the dev rows that
anchor each level (spec induction uses it), and :func:`rubric_gaps` checks a spec's
``### Score Levels`` section so a run can warn, at no cost and before any job, when an
llm_judge spec has no anchored levels.

Leakage contract: anchors come from the dev split only. They are quoted into the spec,
so a val or test row used as an anchor would be scored against its own gold. The caller
passes only dev rows; nothing here reads val or test.
"""

from __future__ import absolute_import

import math
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# A scale with at most this many integer steps gets one level per integer (1-5, 0-10).
# A wider scale (0-100) is cut into WIDE_SCALE_BANDS equal bands, one level per band.
MAX_INTEGER_LEVELS = 11
WIDE_SCALE_BANDS = 5
SCORE_LEVELS_HEADING = re.compile(r"(?im)^#{2,4}\s*score\s+levels\b[^\n]*$")


def scale_levels(score_min: Any, score_max: Any) -> Tuple[List[float], bool]:
    """The level values a rubric anchors, low to high, and whether they are band midpoints.

    One predicate for both answers, so :func:`score_levels` and :func:`is_banded` cannot
    disagree. 1-5 -> ``[1, 2, 3, 4, 5]``; 0-10 -> 0..10 (one level per integer, up to
    ``MAX_INTEGER_LEVELS``); 0-100 -> the midpoints of ``WIDE_SCALE_BANDS`` equal bands
    (10, 30, 50, 70, 90), banded.

    Args:
        score_min: The scale's low end.
        score_max: The scale's high end.

    Returns:
        ``(levels, banded)``.

    Raises:
        DataPreparerOperatorError: No scale, or ``score_max`` not above ``score_min``.
    """
    if score_min is None or score_max is None:
        raise DataPreparerOperatorError(
            "an anchored rubric needs the score scale (--score-min and --score-max)"
        )
    lo, hi = float(score_min), float(score_max)
    if hi <= lo:
        raise DataPreparerOperatorError(
            f"--score-max ({score_max}) must be above --score-min ({score_min})"
        )
    first, last = math.ceil(lo), math.floor(hi)
    if last - first + 1 <= MAX_INTEGER_LEVELS and first == lo and last == hi:
        return [float(v) for v in range(first, last + 1)], False
    width = (hi - lo) / WIDE_SCALE_BANDS
    return [lo + width * (i + 0.5) for i in range(WIDE_SCALE_BANDS)], True


def is_banded(score_min: Any, score_max: Any) -> bool:
    """True when the levels are band midpoints (a wide or fractional scale).

    Band midpoints can be integers (0-100 gives 10, 30, ...), so callers ask this
    instead of testing the values. Same validation as :func:`scale_levels`.
    """
    return scale_levels(score_min, score_max)[1]


def score_levels(score_min: Any, score_max: Any) -> List[float]:
    """The level values a rubric anchors, low to high (scale_levels)."""
    return scale_levels(score_min, score_max)[0]


def _num(v):
    """A gold score as a float, or None (bool is not a score)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _nearest_level(levels: List[float], g: float) -> float:
    """The level nearest to ``g``; ties go to the lower level."""
    return min(levels, key=lambda x: (abs(x - g), x))


def select_rubric_anchors(
    rows: Optional[Sequence[Dict[str, Any]]],
    score_min: Any,
    score_max: Any,
    per_level: int = 2,
    gt_key: str = "gt_score",
) -> Tuple[Dict[float, List[Dict[str, Any]]], List[float]]:
    """Pick up to ``per_level`` dev rows per score level, closest to the level first.

    Each row goes to its nearest level (ties go to the lower level). Within a level,
    rows whose gold sits closest to the level value come first, then file order, so the
    same dev split always yields the same anchors. Pass dev rows only: anchors are quoted
    into the spec.

    Args:
        rows: The dev rows; a row without a numeric ``gt_key`` is skipped.
        score_min: The scale's low end.
        score_max: The scale's high end.
        per_level: How many anchors per level, at least 1.
        gt_key: The gold score field.

    Returns:
        ``(anchors, missing)``: ``anchors`` maps each level to its rows, ``missing``
        lists the levels with no dev row at all (a level with no anchor is defined from
        the scale alone, which the caller should say).

    Raises:
        DataPreparerOperatorError: ``per_level`` under 1, or a bad scale.
    """
    if per_level < 1:
        raise DataPreparerOperatorError("per_level must be >= 1")
    levels = score_levels(score_min, score_max)
    by_level = {lv: [] for lv in levels}
    for pos, r in enumerate(rows or []):
        g = _num(r.get(gt_key)) if isinstance(r, dict) else None
        if g is None:
            continue
        lv = _nearest_level(levels, g)
        by_level[lv].append((abs(lv - g), pos, r))
    anchors, missing = {}, []
    for lv in levels:
        picked = [r for _d, _p, r in sorted(by_level[lv], key=lambda t: (t[0], t[1]))][:per_level]
        anchors[lv] = picked
        if not picked:
            missing.append(lv)
    return anchors, missing


def format_level(v: float) -> str:
    """1.0 -> "1", 2.5 -> "2.5": how a level is written in the spec."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def _score_levels_section(spec_text):
    """The text under the ``Score Levels`` heading, or ``None`` without one.

    Up to the next heading of the same or a higher level.
    """
    s = spec_text or ""
    m = SCORE_LEVELS_HEADING.search(s)
    if not m:
        return None
    depth = len(m.group(0)) - len(m.group(0).lstrip("#"))
    rest = s[m.end() :]
    nxt = re.search(r"(?m)^#{1,%d}\s" % depth, rest)
    return rest[: nxt.start()] if nxt else rest


def rubric_gaps(spec_text: Optional[str], score_min: Any, score_max: Any) -> Dict[str, Any]:
    """What an anchored rubric is missing.

    On a wide (banded) scale only the section and its example count are checked, since
    band labels vary.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec import rubric

        gaps = rubric.rubric_gaps(spec_text, score_min=1, score_max=5)
        gaps["has_section"], gaps["levels_missing"], gaps["levels_without_example"]

    Args:
        spec_text: The spec's Markdown.
        score_min: The scale's low end, or ``None``.
        score_max: The scale's high end, or ``None``.

    Returns:
        ``{"has_section", "levels_missing", "levels_without_example", "n_examples"}``:
        whether the spec has a ``### Score Levels`` section, the levels with no
        ``- **<level> ...`` entry (integer scales only), the levels whose entry has no
        ``Example:`` line under it, and the total ``Example:`` lines. Empty lists with
        ``has_section`` true means every level is defined and anchored.
    """
    section = _score_levels_section(spec_text)
    out = {
        "has_section": section is not None,
        "levels_missing": [],
        "levels_without_example": [],
        "n_examples": 0,
    }
    if section is None:
        return out
    out["n_examples"] = len(re.findall(r"(?im)^\s*[-*]\s*example\s*:", section))
    levels, banded = (
        scale_levels(score_min, score_max) if None not in (score_min, score_max) else ([], False)
    )
    if not levels or banded:
        return out
    # Level entries: a top-level bullet opening with a bold level number ("- **5 -
    # Excellent:** ..." or "- **Score 5:** ..."), matched case-insensitively.
    entry = re.compile(r"(?im)^[-*]\s*\*\*\s*(?:score\s*)?(-?\d+(?:\.\d+)?)\b")
    hits = [(float(m.group(1)), m.start()) for m in entry.finditer(section)]
    bounds = hits + [(None, len(section))]
    found = {}
    for (lv, start), (_n, end) in zip(hits, bounds[1:]):
        body = section[start:end]
        found[lv] = found.get(lv, False) or bool(re.search(r"(?im)^\s+[-*]\s*example\s*:", body))
    for lv in levels:
        if lv not in found:
            out["levels_missing"].append(lv)
        elif not found[lv]:
            out["levels_without_example"].append(lv)
    return out


def rubric_warning(
    task: Optional[str], spec_text: Optional[str], score_min: Any, score_max: Any
) -> Optional[str]:
    """A one-paragraph warning when an llm_judge spec is not an anchored rubric.

    Only llm_judge warns: the anchored rubric is its default spec structure. An ordinal
    spec may use one, but one without it is normal.

    Args:
        task: The labeling task.
        spec_text: The spec's Markdown.
        score_min: The scale's low end, or ``None``.
        score_max: The scale's high end, or ``None``.

    Returns:
        The warning, or ``None`` when the spec is anchored or the task is not llm_judge.
    """
    if task != "llm_judge":
        return None
    try:
        gaps = rubric_gaps(spec_text, score_min, score_max)
    except DataPreparerOperatorError:
        gaps = rubric_gaps(spec_text, None, None)
    if not gaps["has_section"]:
        problem = "the spec has no `### Score Levels` section"
    elif gaps["levels_missing"] or gaps["levels_without_example"]:
        parts = []
        if gaps["levels_missing"]:
            parts.append(
                "no entry for level(s) "
                + ", ".join(format_level(v) for v in gaps["levels_missing"])
            )
        if gaps["levels_without_example"]:
            parts.append(
                "no example under level(s) "
                + ", ".join(format_level(v) for v in gaps["levels_without_example"])
            )
        problem = "the `### Score Levels` section has " + " and ".join(parts)
    elif gaps["n_examples"] < 2:
        problem = "the `### Score Levels` section has fewer than 2 examples"
    else:
        return None
    return (
        f"llm_judge spec is not an anchored rubric: {problem}. The default judge spec "
        "gives every score level a definition plus 1 or 2 example rationales from the "
        "dev split (references/labelling-spec-template.md, Score Levels). "
        "Induce one with `induce_spec.py --task llm_judge` (references/spec-induction.md)."
    )
