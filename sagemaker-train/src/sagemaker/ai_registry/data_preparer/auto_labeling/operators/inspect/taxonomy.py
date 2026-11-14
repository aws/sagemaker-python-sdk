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
"""An upfront taxonomy gate for classification, run before the first run that spends.

It catches an un-labelable label set while fixing it is still free.

No amount of spec refinement can carve a stable rule for two classes that genuinely
overlap, a class the seed barely covers, or a class defined only by absence (``none``,
``other``). Fixing the taxonomy is cheaper than any labeling run, so :func:`gate` runs
before the confirm gate and surfaces the problems there. It never blocks on its own:
the person owns the taxonomy.

Checks, all upfront and free:

1. **Absence-defined classes**: a class named or defined by absence has no positive
   signal, so expect lower recall and more review routing. Flagged, not blocked.
2. **Seed coverage** (with a seed): classes with zero or under ``MIN_PER_CLASS`` rows
   cannot yield a refiner rule; severe imbalance (``min / max`` under
   ``IMBALANCE_RATIO``) skews it; a seed label that matches no class is a typo, a
   casing difference or a retired class.
3. **Overlap**: class pairs whose ``name: definition`` token sets overlap (Jaccard)
   over ``LEX_SIM_THRESHOLD`` are flagged as likely-confusable merge candidates. It
   compares shared surface tokens, not meaning, so it can miss synonyms with no shared
   word ("joy" vs "elation"). Reported as ``separability.method`` ``lexical``.

The overlap check is skipped for ``ordinal``: adjacent tiers are neighbors, not
confusions. A task with no label set (summarization, qa, ner, llm_judge, a scale-only
ordinal spec) is not sent to review for lacking one: with no Label Set parsed and none
required, the gate returns ``skip``. Only ``classification`` treats a missing or
unparseable Label Set as a review-before-spend problem.
"""

from __future__ import absolute_import

import re
from collections import Counter
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult


class TaxonomyReport(OperatorResult):
    """What :func:`gate` found.

    ``verdict`` is ``clear``, ``review`` or ``skipped``; ``decision`` is ``proceed``,
    ``review-before-spend`` or ``skip``. ``issues`` lists ``{type, severity, classes,
    detail, suggestion}`` findings; ``coverage`` and ``separability`` say what was
    checked and how (``separability.method`` is ``lexical``).
    """

    verdict: str
    decision: str
    task: str
    n_classes: int
    absence_classes: List[str]
    coverage: Dict[str, Any]
    separability: Dict[str, Any]
    issues: List[Dict[str, Any]]
    skipped_reason: Optional[str] = None


# Thresholds mirror references/taxonomy-guidance.md.
MIN_PER_CLASS = 10  # < ~10 rows can't build a rule after the val split
IMBALANCE_RATIO = 0.25  # min(count) / max(count) below this is severe
LEX_SIM_THRESHOLD = 0.6  # Jaccard token-overlap threshold
# Tasks for which a missing Label Set is itself a review-worthy problem. Only
# classification requires a fixed label set to gate; every other task type
# (summarization / qa / ner / llm_judge, and scale-only ordinal specs) has no
# Label Set to check, so the gate skips rather than forcing a pointless review.
_CLASSIFICATION_TASKS = {"classification"}

_ABSENCE_NAMES = {
    "none",
    "other",
    "others",
    "na",
    "n/a",
    "unknown",
    "neither",
    "nan",
    "null",
    "no_label",
    "nolabel",
    "misc",
    "miscellaneous",
}
# Specific catch-all / absence phrases only. Generic fragments ("otherwise",
# "no clear", "cannot be", ...) were dropped: they occur in legitimate positive
# definitions (e.g. "charges that cannot be reversed", "delivery with no clear
# ETA") and would falsely flag real classes as absence-defined.
_ABSENCE_DEF_MARKERS = (
    "none of the",
    "none of these",
    "none apply",
    "absence of",
    "everything else",
    "anything else",
    "catch-all",
    "catch all",
    "does not fit any",
    "no other category",
    "no matching category",
    "not covered by any category",
    "not covered by any other",
)
# Only the catch-all phrasings ("not covered by any category" / "... any other"):
# "charges not covered by any insurance plan" is a positive class. Likewise only
# "no/not/non <catch-all noun>", since "non-fiction" or "no-show" are real classes.
_ABSENCE_NAME_RE = re.compile(
    r"^(no|not|non)[-_ ](label|labels|match|matches|category|categories|categ|"
    r"class|classes|tag|tags|value|values|answer|answers|entity|entities|fit|"
    r"applicable|other|others|response|responses|intent|intents)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def _strip_md(text):
    """A parsed cell without surrounding ``*`` and backticks; inner text is untouched.

    A bolded table label must match the seed's plain label.
    """
    return (text or "").strip().strip("*`").strip()


def parse_label_set(spec_text):
    """Extract ``[(name, definition)]`` from the spec's Label Set.

    Primary: the markdown table under a ``### Label Set`` heading
    (``| Label | Definition |``). Fallbacks, in order, so a terse or lightly
    formatted spec still yields class names: the first pipe-table anywhere in the
    spec, then the ``### Labelling Rules`` bullets (``- **name:** ...``).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import parse_label_set

        labels = [name for name, definition in parse_label_set(spec_text)]
    """
    s = spec_text or ""
    block = _label_set_block(s)
    # Order: Label Set table, Label Set bullets, Labelling Rules bullets, and the
    # whole-spec table last, so an unrelated config or metrics table is a last resort.
    rows = (
        _parse_pipe_table(block)
        or _parse_rule_bullets(block)
        or _parse_rule_bullets(_rules_block(s))
        or _parse_pipe_table(s)
    )
    # De-dup on name, keep first definition seen; drop blanks.
    seen, out = set(), []
    for name, defn in rows:
        name = _strip_md(name)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            out.append((name, _strip_md(defn)))
    return out


def _label_set_block(spec_text):
    """Text under a ``Label Set`` heading up to the next heading (empty if none)."""
    m = re.search(r"(?im)^#+\s*label\s*set\b(.*?)(?=^#+\s|\Z)", spec_text, re.DOTALL)
    return m.group(1) if m else ""


def _rules_block(spec_text):
    """Text under a ``Label(l)ing Rules`` heading up to the next heading.

    The bullet fallback is scoped to this section so a stray ``- **Note:** ...`` /
    ``- **Warning:** ...`` bullet elsewhere in the spec is not read as a class.
    """
    m = re.search(r"(?im)^#+\s*label(?:l?ing)?\s*rules\b(.*?)(?=^#+\s|\Z)", spec_text, re.DOTALL)
    return m.group(1) if m else ""


def _is_table_separator(line):
    """True for a markdown table separator row like ``|---|:--:|``."""
    core = line.replace("|", "").strip()
    return bool(core) and set(core) <= {"-", ":", " "}


def _parse_pipe_table(text):
    """Body rows ``[(col0, col1)]`` of the first pipe table with a ``|---|`` separator.

    The header (through the separator) is dropped. One table only, so an unrelated
    second table is never read as classes.
    """
    runs, cur = [], []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if s.startswith("|"):
            cur.append(s)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    for run in runs:
        sep_idx = next((i for i, line in enumerate(run) if _is_table_separator(line)), None)
        if sep_idx is None:
            continue
        rows = []
        for line in run[sep_idx + 1 :]:
            if _is_table_separator(line):
                continue
            # Unescape ``\|`` first: an escaped pipe in a definition is not a column.
            fields = line.strip("|").replace("\\|", "\x00").split("|")
            cells = [c.strip().replace("\x00", "|") for c in fields]
            if len(cells) >= 2:
                rows.append((cells[0], cells[1]))
        if rows:
            return rows  # first real (separator-bearing) table only
    return []


def _parse_rule_bullets(text):
    """Fallback: ``- **name:** definition`` bullets (e.g. a Labelling Rules list).

    Only bullets whose bolded key has no ``vs`` are taken as class rows, so a
    ``- **a vs b:**`` disambiguation line is not mistaken for a class.
    """
    rows = []
    for m in re.finditer(r"(?m)^\s*[-*]\s*\*\*(.+?):?\*\*\s*:?\s*(.*)$", text or ""):
        name, defn = m.group(1).strip(), m.group(2).strip()
        if " vs " not in name.lower():
            rows.append((name, defn))
    return rows


def _tokens(text):
    return set(_WORD_RE.findall((text or "").lower()))


def find_absence_classes(label_set):
    """Names whose class is defined by absence (weak positive signal)."""
    out = []
    for name, defn in label_set:
        nl = name.strip().lower()
        d = defn.lower()
        if (
            nl in _ABSENCE_NAMES
            or _ABSENCE_NAME_RE.match(name.strip())
            or any(mark in d for mark in _ABSENCE_DEF_MARKERS)
        ):
            out.append(name)
    return out


def _norm_label(x):
    """Canonical form for matching a seed label to a spec class name."""
    return str(x).strip().casefold()


def seed_coverage(label_set, labeled_rows, gt_field):
    """Seed coverage per class, labels matched case- and whitespace-insensitively.

    Args:
        label_set: ``[(name, definition)]`` from :func:`parse_label_set`.
        labeled_rows: The seed rows.
        gt_field: The seed's gold field.

    Returns:
        ``(counts, under, missing, imbalanced, unknown)``: a Counter keyed by the spec's
        class name, classes with 1 to ``MIN_PER_CLASS - 1`` rows, classes absent from
        the seed, whether ``min / max`` is below ``IMBALANCE_RATIO``, and the sorted
        seed labels that match no class.
    """
    names = [n for n, _ in label_set]
    norm_to_name = {_norm_label(n): n for n in names}

    # A non-hashable gt_label (a list or object) goes to `unknown`, never raises.
    def _hashable(v):
        return v if isinstance(v, (str, int, float, bool)) else str(v)

    raw = Counter(
        _hashable(r.get(gt_field))
        for r in labeled_rows
        if isinstance(r, dict) and r.get(gt_field) is not None
    )
    counts, unknown = Counter(), Counter()
    for val, c in raw.items():
        canon = norm_to_name.get(_norm_label(val))
        if canon is not None:
            counts[canon] += c
        else:
            unknown[val] += c
    present = [counts.get(n, 0) for n in names]
    missing = [n for n in names if counts.get(n, 0) == 0]
    under = [n for n in names if 0 < counts.get(n, 0) < MIN_PER_CLASS]
    nonzero = [c for c in present if c > 0]
    imbalanced = bool(nonzero) and (min(nonzero) / max(nonzero) < IMBALANCE_RATIO)
    # key=str: a seed may mix int and str labels.
    return counts, under, missing, imbalanced, sorted(unknown, key=str)


def overlap_pairs(label_set):
    """Likely-confusable class pairs by Jaccard token overlap of ``name: definition``.

    Args:
        label_set: ``[(name, definition)]``.

    Returns:
        ``(pairs, method)``: ``pairs`` is ``[(a, b, score)]`` sorted by score, each over
        ``LEX_SIM_THRESHOLD``; ``method`` is ``lexical`` (``none`` with fewer than two
        classes). It catches reworded near-duplicates but misses synonyms with no shared
        word.
    """
    if len(label_set) < 2:
        return [], "none"
    toks = [_tokens(f"{n}: {d}") for n, d in label_set]
    names = [n for n, _ in label_set]
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        inter, union = toks[i] & toks[j], toks[i] | toks[j]
        score = len(inter) / len(union) if union else 0.0
        if score >= LEX_SIM_THRESHOLD:
            pairs.append((names[i], names[j], round(score, 4)))
    return sorted(pairs, key=lambda p: -p[2]), "lexical"


def gate(  # pylint: disable=too-many-locals,too-many-branches
    spec_text: Optional[str],
    labeled_rows: Optional[Sequence[Any]] = None,
    gt_field: str = "gt_label",
    task: str = "classification",
) -> TaxonomyReport:
    """Check a labeling spec's taxonomy, and its seed coverage, before any run spends.

    Run it on the spec once the task is known and before the person confirms the run.
    Show every issue at that gate: the person merges or restructures classes or brings
    more examples. Do not run it to decide anything on its own, and do not read a
    ``lexical`` separability result as proof that classes are separable.

    Args:
        spec_text: The labeling spec's Markdown.
        labeled_rows: The labeled seed's rows, when there is a seed.
        gt_field: The seed's gold field.
        task: The labeling task; ``ordinal`` skips the overlap check and a task with no
            label set skips the gate.

    Returns:
        A :class:`TaxonomyReport`; ``to_json()`` is the gate's report document.

    Example::

        report = gate(spec_text, seed_rows)
        report.decision, [i["type"] for i in report.issues]
        # ('review-before-spend', ['class_undersampled'])
    """
    label_set = parse_label_set(spec_text)
    # A task without a fixed label set has nothing to check: with no parseable Label
    # Set it proceeds (one that has a Label Set is still checked).
    if not label_set and task not in _CLASSIFICATION_TASKS:
        return TaxonomyReport(
            verdict="skipped",
            decision="skip",
            task=task,
            n_classes=0,
            absence_classes=[],
            coverage={"checked": False},
            separability={"checked": False, "skipped_reason": f"{task}: no label set expected"},
            issues=[],
            skipped_reason=(
                f"non-classification task ({task}) has no label set "
                "to gate; proceeding without a taxonomy pre-check"
            ),
        )
    issues = []
    if not label_set:
        # Could not read a taxonomy at all. The gate must not greenlight spend
        # on a spec it could not parse - surface it as review, don't proceed.
        issues.append(
            {
                "type": "label_set_unparsed",
                "severity": "review",
                "classes": [],
                "detail": "could not parse a Label Set from the spec (no '### Label Set' "
                "table and no rule bullets)",
                "suggestion": "add a '### Label Set' section with a | Label | Definition | "
                "table so the gate can check the taxonomy",
            }
        )
    absence = find_absence_classes(label_set)
    if absence:
        issues.append(
            {
                "type": "absence_defined_class",
                "severity": "warn",
                "classes": absence,
                "detail": "class(es) defined by absence have no positive signal",
                "suggestion": "expect lower recall; consider a positive definition or "
                "accept more review routing for these",
            }
        )

    # Coverage needs a parsed label set; without one every seed label would read as
    # unknown.
    coverage = {"checked": bool(labeled_rows) and bool(label_set)}
    if labeled_rows and label_set:
        counts, under, missing, imbalanced, unknown = seed_coverage(
            label_set, labeled_rows, gt_field
        )
        coverage.update(
            per_class=dict(counts), min_per_class=MIN_PER_CLASS, unknown_seed_labels=unknown
        )
        if unknown:
            issues.append(
                {
                    "type": "unknown_seed_labels",
                    "severity": "review",
                    "classes": unknown,
                    "detail": "seed label(s) map to no class in the spec (typo, casing, "
                    "or a retired class)",
                    "suggestion": "fix the seed label or add the class to the spec; a "
                    "case/space-only mismatch is matched automatically",
                }
            )
        if missing:
            issues.append(
                {
                    "type": "class_missing_from_seed",
                    "severity": "review",
                    "classes": missing,
                    "detail": f"class(es) have 0 rows in the seed (need >= {MIN_PER_CLASS})",
                    "suggestion": "add labeled examples for these or drop the class",
                }
            )
        if under:
            issues.append(
                {
                    "type": "class_undersampled",
                    "severity": "review",
                    "classes": under,
                    "detail": f"class(es) have < {MIN_PER_CLASS} rows; the refiner can't "
                    "build a rule from so few",
                    "suggestion": "add examples or merge into a coarser class",
                }
            )
        if imbalanced:
            issues.append(
                {
                    "type": "severe_imbalance",
                    "severity": "warn",
                    "classes": [],
                    "detail": f"min/max class count < {IMBALANCE_RATIO}",
                    "suggestion": "rebalance the seed or expect the rare classes to suffer",
                }
            )

    separability = {"checked": task != "ordinal" and len(label_set) >= 2}
    if separability["checked"]:
        pairs, method = overlap_pairs(label_set)
        separability.update(method=method, flagged_pairs=pairs)
        if pairs:
            issues.append(
                {
                    "type": "overlapping_classes",
                    "severity": "review",
                    "classes": sorted({n for p in pairs for n in p[:2]}),
                    "detail": f"{len(pairs)} class pair(s) look confusable "
                    f"(similarity via {method})",
                    "suggestion": "merge the overlapping classes or add a disambiguation "
                    "rule; see references/taxonomy-guidance.md",
                }
            )
    elif task == "ordinal":
        separability["skipped_reason"] = "ordinal: adjacent tiers are neighbors, not confusions"

    verdict = "review" if issues else "clear"
    decision = "review-before-spend" if issues else "proceed"
    return TaxonomyReport(
        verdict=verdict,
        decision=decision,
        task=task,
        n_classes=len(label_set),
        absence_classes=absence,
        coverage=coverage,
        separability=separability,
        issues=issues,
    )
