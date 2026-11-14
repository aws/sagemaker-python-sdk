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
"""Few-shot demonstrations drawn from the labeled seed: the same K per class.

Demonstrations are the single biggest accuracy lever (zero-shot to few-shot was about
+7 points of macro-F1 on a 151-class intent task in our ablations). They are drawn
only from the dev split the customer labeled, never from val or test, so this is fair
few-shot rather than leakage. The labeling job takes one demo set for the whole run
(the request's ``Demos``, at most 100 rows), so the set is fixed: K rows per class, or
K rows per entity type for NER, the same for every input. One or two per class
captures about 95% of the gain.
"""

from __future__ import absolute_import

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set

from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult

Rows = List[Dict[str, Any]]


class DemosResult(OperatorResult):
    """What :func:`select_demos` chose.

    The ``demos`` rows and a ``summary`` (``mode``, ``task``, the K and the count) a
    caller prints.
    """

    demos: Rows
    summary: Dict[str, Any]


def fixed_demos(seed: Sequence[Dict[str, Any]], k: int, label_field: str = "gt_label") -> Rows:
    """The first ``k`` labeled rows of each class, classes in sorted order.

    Args:
        seed: The dev rows.
        k: Demos per class.
        label_field: The gold field; rows without one are skipped.

    Returns:
        The chosen rows, grouped by class.
    """
    by: Dict[Any, Rows] = defaultdict(list)
    for r in seed:
        if r.get(label_field) is not None:
            by[r[label_field]].append(r)
    out: Rows = []
    for lbl in sorted(by):
        out.extend(by[lbl][:k])
    return out


def _entity_types(row: Dict[str, Any], entities_field: str) -> Set[str]:
    return {
        e["type"] for e in (row.get(entities_field) or []) if isinstance(e, dict) and e.get("type")
    }


def fixed_demos_ner(
    seed: Sequence[Dict[str, Any]], k: int, entities_field: str = "gt_entities"
) -> Rows:
    """Rows covering every entity type at least ``k`` times, rarest type first.

    Greedy: a row counts for every type it contains, so the set stays small. Rows with
    no entities are never demos.

    Args:
        seed: The dev rows.
        k: Rows per entity type.
        entities_field: The gold entity-list field.

    Returns:
        The chosen rows, in the order picked.
    """
    labeled = [r for r in seed if _entity_types(r, entities_field)]
    support: Dict[str, int] = defaultdict(int)
    for r in labeled:
        for t in _entity_types(r, entities_field):
            support[t] += 1
    chosen: Rows = []
    chosen_ids: Set[int] = set()
    covered: Dict[str, int] = defaultdict(int)
    for typ in sorted(support, key=lambda t: (support[t], t)):
        for i, r in enumerate(labeled):
            if covered[typ] >= k:
                break
            if i in chosen_ids or typ not in _entity_types(r, entities_field):
                continue
            chosen.append(r)
            chosen_ids.add(i)
            for t in _entity_types(r, entities_field):
                covered[t] += 1
    return chosen


def select_demos(
    seed: Sequence[Dict[str, Any]],
    *,
    mode: str = "fixed",
    k: int = 2,
    task: str = "classification",
    gt_field: str = "gt_entities",
) -> DemosResult:
    """Draw the fixed demonstration set a labeling run will send with every input.

    Use it on the dev split only (the caller guards the read). ``mode="none"`` is
    zero-shot and the weakest choice; use it only when there is no labeled seed.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import demos

        picked = demos.select_demos(dev_rows, mode="fixed", k=2, task="classification")
        picked.demos, picked.summary

    Args:
        seed: The dev rows.
        mode: ``fixed`` (K per class) or ``none``.
        k: Demos per class, or rows per entity type for NER.
        task: ``classification`` or ``ner``.
        gt_field: NER only: the entity-list field on each row.

    Returns:
        A :class:`DemosResult` with the rows and a summary.
    """
    is_ner = task == "ner"
    if mode == "none":
        return DemosResult(demos=[], summary={"mode": "none", "task": task, "demos": 0})
    demos = fixed_demos_ner(seed, k, gt_field) if is_ner else fixed_demos(seed, k)
    key = "k_per_type" if is_ner else "k_per_class"
    return DemosResult(
        demos=demos, summary={"mode": "fixed", "task": task, key: k, "demos": len(demos)}
    )
