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
"""Which labeling spec ships: best-iteration rollback with a val do-no-harm floor.

A refinement loop produces several candidate specs, each scored on the val split (the
val job's ``HeldOutAccuracy``). Two rules decide which ships, and they are mechanical
so the choice is reproducible and a reviewer can trust it:

1. **Best-iteration rollback.** Ship the spec with the highest val score across all
   iterations, never "the last one tried".
2. **Val do-no-harm floor.** If the best refined spec does not beat the *initial*
   spec by at least ``min_gain``, ship the initial spec.

Selection reads val only. The reserved test split is never an input here; it is
scored once, after this pick, as the reported held-out number. Keeping test out of
selection is what makes that number honest.

Optional per-type floor: when ``initial`` and each iteration carry a ``breakdown``
(per-class or per-type F1), ``per_type_tol`` rejects a candidate that lifts the
aggregate while regressing any type that was passing initially by more than the
tolerance, so the loop cannot buy micro-F1 by sacrificing a minority type. Without
breakdowns it is a no-op.
"""

from __future__ import absolute_import

import os
from typing import Any, Dict, List, Optional, Tuple, Union

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec._floor import (
    select_with_floor,
)

# The exact JSON shape select() expects, shown in the missing-key error.
EXPECTED_SHAPE = (
    '{"initial": {"spec": "work/labeling_spec_iter_0.md", "val": 0.885},\n'
    ' "iters":   [{"spec": "work/labeling_spec_iter_1.md", "val": 0.921},\n'
    '             {"spec": "work/labeling_spec_iter_2.md", "val": 0.880}]}'
)
# Finer gains are within the noise a small val split can resolve.
DEFAULT_MIN_GAIN = 0.02
ITERS_ALIAS_WARNING = (
    "select_best_spec: warning: top-level 'iterations' is a deprecated "
    "alias for 'iters'; rename the key to 'iters'."
)


class SpecDecision(OperatorResult):
    """Which spec ships and why; ``to_json()`` is the ``decision.json`` document.

    The fields are declared in the order the command line's ``decision.json`` lists them.
    The val scores are written as the caller passed them (an int stays an int), so
    ``decision.json`` reads back exactly as the candidates file was written.
    """

    ship: str
    spec: Optional[str]
    val: Optional[Union[int, float]]
    initial_val: Optional[Union[int, float]] = None
    best_refined_val: Optional[Union[int, float]] = None
    gain: Optional[Union[int, float]] = None
    min_gain: float
    reason: str
    per_type_tol: Optional[Union[int, float]] = None
    per_type_rejected: Optional[List[Dict[str, Any]]] = None
    shipped_spec_file: Optional[str] = None
    selection_mode: str = "gold-gate"

    def to_json(self) -> Dict[str, Any]:
        """The ``decision.json`` document.

        ``spec`` and ``shipped_spec_file`` are always present, even when null, so a
        caller reads them without a key check. When no refined spec was picked (no
        ``gain``), ``per_type_tol`` comes before ``reason``, as the command line wrote it.
        """
        out = super().to_json()
        out.setdefault("spec", self.spec)
        out.setdefault("shipped_spec_file", self.shipped_spec_file)
        order = list(type(self).model_fields)
        if self.gain is None:
            order.remove("per_type_tol")
            order.insert(order.index("reason"), "per_type_tol")
        return {k: out[k] for k in order if k in out}


def iter_spec_name(n: int, label_field: Optional[str] = None) -> str:
    """The spec file name for iteration ``n`` (``iter_0`` is the initial spec).

    Prefixed with the label field when a run has several label columns
    (``<field>_labeling_spec_iter_<n>.md``).
    """
    stem = f"{label_field}_labeling_spec" if label_field else "labeling_spec"
    return f"{stem}_iter_{n}.md"


def regressed_types(
    initial: Dict[str, Any], cand: Dict[str, Any], tol: float
) -> List[Dict[str, Any]]:
    """Types passing in ``initial`` (F1 > 0) that ``cand`` drops by more than ``tol``.

    Returns:
        ``[{"type", "initial", "candidate", "delta"}, ...]`` sorted by ``delta``;
        empty when either side has no breakdown.
    """
    ib = initial.get("breakdown") or {}
    cb = cand.get("breakdown") or {}
    if not ib or not cb:
        return []
    out = []
    for t, iv in ib.items():
        if iv <= 0:
            continue
        cv = cb.get(t, 0.0)
        if iv - cv > tol:
            out.append(
                {
                    "type": t,
                    "initial": round(iv, 4),
                    "candidate": round(cv, 4),
                    "delta": round(cv - iv, 4),
                }
            )
    return sorted(out, key=lambda x: x["delta"])


def _require_number(value: Any, where: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataPreparerOperatorError(f"{where} must be a number, got {type(value).__name__}.")


def _require_breakdown_numbers(bd: Any, where: str) -> None:
    if isinstance(bd, dict):
        for t, v in bd.items():
            _require_number(v, f"{where}['breakdown'][{t!r}]")


def validate_scores(data: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Check a scores document and return it normalized.

    ``initial`` and ``iters`` are required (a run with no iteration passes
    ``"iters": []``), and every ``val`` and breakdown value must be a real number.
    ``iterations`` is read as a deprecated alias for ``iters`` when ``iters`` is absent.

    Args:
        data: The parsed scores JSON.

    Returns:
        ``(data, warnings)``: a shallow copy with the alias applied when it applied,
        else the input (never mutated), and the warnings to show.

    Raises:
        DataPreparerOperatorError: The shape would be misread; the message shows the
            expected shape.
    """
    if not isinstance(data, dict):
        raise DataPreparerOperatorError(f"scores must be a JSON object, got {type(data).__name__}")
    notes: List[str] = []
    if "iters" not in data and "iterations" in data:
        notes.append(ITERS_ALIAS_WARNING)
        data = {**data, "iters": data["iterations"]}
    missing = [k for k in ("initial", "iters") if k not in data]
    if missing:
        raise DataPreparerOperatorError(
            f"scores is missing required key(s) {missing}. Expected this shape "
            f"(top-level 'iters' may be an empty list for a no-iteration run):\n"
            f"{EXPECTED_SHAPE}\n"
            f"Got keys: {sorted(data)}. (A misnamed 'iters' key would otherwise "
            "silently ship the initial spec. 'iterations' is accepted as a "
            "deprecated alias for 'iters'.)"
        )
    if not isinstance(data["initial"], dict) or "val" not in data["initial"]:
        raise DataPreparerOperatorError("scores['initial'] must carry a numeric held-out 'val'.")
    _require_number(data["initial"]["val"], "scores['initial']['val']")
    _require_breakdown_numbers(data["initial"].get("breakdown"), "scores['initial']")
    if not isinstance(data["iters"], list):
        raise DataPreparerOperatorError("scores['iters'] must be a list (possibly empty).")
    for i, it in enumerate(data["iters"]):
        if not isinstance(it, dict) or "val" not in it:
            raise DataPreparerOperatorError(
                f"scores['iters'][{i}] must be a dict carrying a numeric 'val'; "
                f"got {type(it).__name__}. (pick_with_floor sorts iters by val.)"
            )
        _require_number(it["val"], f"scores['iters'][{i}]['val']")
        _require_breakdown_numbers(it.get("breakdown"), f"scores['iters'][{i}]")
    return data, notes


def _decision(initial, res, min_gain, per_type_tol) -> SpecDecision:
    best = res["best"]
    rejected = [
        {"spec": c["spec"], "val": c["val"], "regressed_types": reason}
        for c, reason in res["skipped"]
    ]
    rejected = rejected or None
    if best is None:
        if rejected:
            reason = (
                "every refined spec that beat initial regressed a "
                "passing type beyond --per-type-tol; shipping initial"
            )
        else:
            reason = "no refinement iterations ran"
        spec = initial["spec"]
        return SpecDecision(
            ship="initial",
            spec=spec,
            val=initial["val"],
            min_gain=min_gain,
            reason=reason,
            per_type_tol=per_type_tol,
            per_type_rejected=rejected,
            shipped_spec_file=os.path.basename(spec) if spec else None,
        )
    gain = res["gain"]
    if res["cleared"]:
        out = SpecDecision(
            ship="refined",
            spec=best["spec"],
            val=best["val"],
            initial_val=initial["val"],
            gain=round(gain, 4),
            min_gain=min_gain,
            reason=(
                f"best refined spec beat initial by {gain:.4f} "
                f"(>= min_gain {min_gain}) with no per-type regression "
                f"beyond tolerance"
            ),
        )
    else:
        out = SpecDecision(
            ship="initial",
            spec=initial["spec"],
            val=initial["val"],
            best_refined_val=best["val"],
            gain=round(gain, 4),
            min_gain=min_gain,
            reason=(
                f"val do-no-harm floor: best refined gain {gain:.4f} "
                f"< min_gain {min_gain}; shipping initial (do no harm)"
            ),
        )
    out.per_type_tol = per_type_tol
    out.per_type_rejected = rejected
    out.shipped_spec_file = os.path.basename(out.spec) if out.spec else None
    return out


def select(
    data: Any, min_gain: float = DEFAULT_MIN_GAIN, per_type_tol: Optional[float] = None
) -> SpecDecision:
    """Pick which spec ships: the gold gate (val + the do-no-harm floor).

    Use it once the refinement loop has a val score for the initial spec and every
    candidate. Do not feed it test scores: selection on test would make the reported
    held-out number optimistic. Its ``spec`` is what the corpus run uses.

    Args:
        data: ``{"initial": {"spec", "val", "breakdown"?}, "iters": [{"spec", "val",
            "breakdown"?}, ...]}``. ``val`` is the val job's ``HeldOutAccuracy``;
            ``breakdown`` is that job's per-class or per-type F1, for the per-type floor.
            Every ``val`` must be a number: a candidate whose scoring failed is left
            out of the document, not passed as ``null``.
        min_gain: The least val improvement that prefers a refined spec over the
            initial one. The 0.02 default is the noise a small val split can resolve;
            a finer gain is treated as no improvement.
        per_type_tol: When set and breakdowns are present, reject a refined spec that
            regresses any initially-passing type by more than this, even if aggregate
            val improved.

    Returns:
        A :class:`SpecDecision`: ``ship`` is ``refined`` or ``initial``, ``spec`` the
        file to use, ``reason`` why, with ``gain``, ``per_type_rejected`` and
        ``shipped_spec_file`` as applicable. ``warnings`` carries the deprecated-alias
        note when ``iterations`` was read for ``iters``.

    Raises:
        DataPreparerOperatorError: The scores document is malformed; the message shows
            the expected shape. A misnamed key is refused rather than silently shipping
            the initial spec.

    Example::

        decision = select({"initial": {"spec": "iter_0.md", "val": 0.885},
                           "iters": [{"spec": "iter_1.md", "val": 0.921}]})
        decision.ship, decision.spec, decision.gain
        # ('refined', 'iter_1.md', 0.036)
    """
    data, notes = validate_scores(data)
    initial = data["initial"]
    iters = data.get("iters", [])

    def _skip(cand):
        if per_type_tol is None:
            return None
        return regressed_types(initial, cand, per_type_tol) or None

    res = select_with_floor(initial["val"], iters, min_gain, skip=_skip)
    decision = _decision(initial, res, min_gain, per_type_tol)
    decision.warnings = notes
    return decision


# ---------------------------------------------------------------------------
# The project-level step (full-loop.md Steps 5-6): which spec ships, from the refinement
# rounds' val scores, and the one file the customer reviews.
# ---------------------------------------------------------------------------


def select_labeling_spec(
    project_dir: str,
    rounds=None,
    *,
    scores: Optional[Dict[str, Any]] = None,
    min_gain: float = DEFAULT_MIN_GAIN,
    per_type_tol: Optional[float] = None,
    out: Optional[str] = None,
) -> SpecDecision:
    """Pick the spec that ships from the refinement rounds, and write it as the project's spec.

    Builds ``<project_dir>/scores.json`` in the documented shape (``initial`` is
    ``iter_0``, each refined candidate one entry of ``iters``; ``val`` is the val job's
    ``HeldOutAccuracy``, the steering metric, ``val_macro_f1`` recorded beside it), runs
    :func:`select` (best-iter rollback and the val do-no-harm floor; test is never
    consulted), writes ``work/decision.json``, and copies the shipped iteration's file to
    ``<project_dir>/labeling_spec.md``, the one spec the customer reviews and
    :func:`~..job.round.label_corpus` runs. The ``work/labeling_spec_iter_*.md`` files stay
    as the record of each round.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import select_labeling_spec

        decision = select_labeling_spec("tickets", rounds)        # the RefinementRounds object
        decision.ship, decision.spec, decision.reason      # 'refined', 'tickets/work/labeling_spec_iter_1.md', ...

    Args:
        project_dir: The project.
        rounds: :func:`~..job.round.tune_labeling_spec`'s object (its ``scores()``), or a
            list of ``RoundFindings``.
        scores: The ``scores.json`` document instead of ``rounds``.
        min_gain: The do-no-harm floor: the val gain a refined spec needs to ship.
        per_type_tol: With per-class breakdowns, reject a refined spec that regresses a
            passing class by more than this.
        out: Where the shipped spec is copied; ``<project_dir>/labeling_spec.md`` when ``None``.

    Returns:
        The :class:`SpecDecision`; ``spec`` is the shipped iteration's file.

    Raises:
        DataPreparerOperatorError: No rounds and no scores, or a scores document of the
            wrong shape.
    """
    import json
    import shutil

    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators._run_log import run_log_event

    if scores is None:
        if rounds is None:
            raise DataPreparerOperatorError("pass the refinement rounds or a scores document")
        if hasattr(rounds, "scores"):
            scores = rounds.scores()
        else:
            rows = [
                {
                    k: v
                    for k, v in {
                        "spec": r.spec,
                        "val": r.val_accuracy,
                        "val_macro_f1": r.val_macro_f1,
                        "breakdown": r.val.breakdown or None,
                    }.items()
                    if v is not None
                }
                for r in rounds
            ]
            if not rows:
                raise DataPreparerOperatorError("no round has run yet")
            scores = {"initial": rows[0], "iters": rows[1:]}
    scores_path = os.path.join(project_dir, "scores.json")
    with open(scores_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(scores, indent=2) + "\n")
    decision = select(scores, min_gain=min_gain, per_type_tol=per_type_tol)
    decision_path = os.path.join(project_dir, "work", "decision.json")
    os.makedirs(os.path.dirname(decision_path), exist_ok=True)
    with open(decision_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(decision.to_json(), indent=2) + "\n")
    shipped = out or os.path.join(project_dir, "labeling_spec.md")
    if decision.spec and os.path.isfile(decision.spec):
        if os.path.abspath(decision.spec) != os.path.abspath(shipped):
            shutil.copyfile(decision.spec, shipped)
    _run_log.append_entry(
        project_dir,
        run_log_event(
            "decision",
            "select",
            len(scores.get("iters") or []),
            decision.ship,
            file=decision_path,
            spec=decision.spec,
            shipped=shipped,
            val=decision.val,
            gain=decision.gain,
            min_gain=decision.min_gain,
            reason=decision.reason,
        ),
    )
    return decision
