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
"""K stratified folds of a labeled seed, for rotating the val split across iterations.

A single fixed val split is noisy: with many classes (151 intents and a few val rows
each, say) the val score ranks candidate specs unreliably, and a refiner can pick a
spec that wins on this val draw but not in general. Rotating which fold serves as val
across refinement iterations averages that noise out and stops the refiner from
memorizing one split. On iteration ``i`` the loop uses fold ``i % K`` as val and the
other folds, concatenated, as dev.

Deterministic, standard-library only, seeded. :func:`build_kfold` returns every file's
rows and a manifest whose hashes are over exactly those rows' bytes, so a caller that
writes them gets files a frozen-split guard accepts as listed folds.
"""

from __future__ import absolute_import

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split import sha256_rows

Rows = List[Dict[str, Any]]

USAGE_NOTE = (
    "On refine iteration i, use rotation[i % k]: dev for the error "
    "signal, val for the gate. Track best val macro-F1 across iters."
)


class KFoldResult(OperatorResult):
    """What :func:`build_kfold` produced, for the caller to write under ``out_dir``.

    ``files`` maps each file name (``fold_<i>.jsonl``, ``iter<i>_dev.jsonl``) to its
    rows; ``manifest`` is the ``manifest.json`` document (``k``, ``seed``,
    ``gt_field``, ``n_rows``, ``folds`` and ``rotation`` as paths under ``out_dir``,
    ``sha256`` per file over the rows' bytes, ``usage``). ``fold_sizes`` is the row
    count per fold and ``n_rows`` the rows folded after any exclusion.
    """

    files: Dict[str, Rows]
    manifest: Dict[str, Any]
    fold_sizes: List[int]
    n_rows: int
    dropped: int = 0


def stratified_folds(
    rows: Sequence[Dict[str, Any]], gt_field: str, k: int, seed: int, task: str = "classification"
) -> List[List[int]]:
    """Assign each row to one of ``k`` folds, round-robin within each stratum.

    The stratum is the ``gt_field`` label for classification, or the row's globally
    rarest entity type for NER, so a rare type is spread across folds rather than left
    in one.

    Args:
        rows: The labeled rows.
        gt_field: The gold field (a label, or an entity list for NER).
        k: The number of folds.
        seed: The shuffle seed; the same seed gives the same folds.
        task: ``classification`` or ``ner``.

    Returns:
        ``k`` lists of row indexes into ``rows``.
    """
    rng = random.Random(seed)
    if task == "ner":
        freq: Dict[str, int] = defaultdict(int)
        for r in rows:
            for e in r.get(gt_field) or []:
                if isinstance(e, dict) and e.get("type"):
                    freq[e["type"]] += 1

        def stratum(r):
            types = [
                e["type"] for e in (r.get(gt_field) or []) if isinstance(e, dict) and e.get("type")
            ]
            return "NONE" if not types else min(types, key=lambda t: (freq[t], t))

    else:

        def stratum(r):
            return r.get(gt_field)

    by_label: Dict[Any, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_label[stratum(r)].append(i)
    folds: List[List[int]] = [[] for _ in range(k)]
    for _label, idxs in by_label.items():
        ids = list(idxs)
        rng.shuffle(ids)
        for pos, idx in enumerate(ids):
            folds[pos % k].append(idx)
    return folds


def content_key(row: Dict[str, Any]) -> str:
    """A row's content without its ``id``, as JSON with sorted keys.

    A reader fills a missing id with the line index, so one row can carry different
    ids in two files; matching on content finds it anyway.
    """
    return _digest.canonical_json({k: v for k, v in row.items() if k != "id"})


def build_kfold(
    rows: Sequence[Dict[str, Any]],
    *,
    out_dir: str,
    gt_field: str = "gt_label",
    k: int = 4,
    seed: int = 0,
    task: str = "classification",
    exclude: Optional[Sequence[Dict[str, Any]]] = None,
    frozen_split: bool = False,
) -> KFoldResult:
    """Fold a labeled seed into ``k`` stratified folds plus one dev file per rotation.

    Use it once at the start of a refinement loop over a seed whose val split would be
    too small to rank candidates on its own. Under a frozen split (see
    :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.split`) pass
    only dev and val rows and no ``exclude``: the split guard already keeps test rows
    out, and the manifest's hashes let a dev or val round accept a listed fold.

    Args:
        rows: The labeled rows to fold (already guard-checked by the caller).
        out_dir: Where the caller will write the files; the manifest names paths under it.
        gt_field: The gold field (``gt_label``; an entity-list field for NER).
        k: The number of folds, at least 2.
        seed: The shuffle seed.
        task: ``classification`` or ``ner`` (NER stratifies by the rarest entity type).
        exclude: A seed with no frozen split only: the reserved test rows, dropped from
            the folds by content (ignoring ``id``).
        frozen_split: Whether the project has a frozen split; ``exclude`` is then refused.

    Returns:
        A :class:`KFoldResult`; write each entry of ``files`` as JSON Lines and
        ``manifest`` as ``manifest.json`` under ``out_dir``. ``warnings`` notes the rows
        ``exclude`` dropped.

    Raises:
        DataPreparerOperatorError: ``k`` under 2, or ``exclude`` with a frozen split.

    Example::

        result = build_kfold(rows, out_dir="folds", k=4)
        result.fold_sizes, result.manifest["rotation"][1]["val"]
        # ([30, 30, 30, 30], 'folds/fold_1.jsonl')
    """
    rows = list(rows)
    notes: List[str] = []
    dropped = 0
    if exclude is not None and frozen_split:
        raise DataPreparerOperatorError(
            "kfold_rotate: with a frozen split drop --exclude; the split guard already "
            "refuses test rows in --labeled"
        )
    if exclude is not None:
        held = {content_key(r) for r in exclude}
        kept = [r for r in rows if content_key(r) not in held]
        dropped = len(rows) - len(kept)
        if dropped:
            notes.append(
                f"kfold_rotate: dropped {dropped} row(s) present in --exclude "
                "so the reserved test partition does not leak into the dev/val folds."
            )
        rows = kept
    if k < 2:
        raise DataPreparerOperatorError("--k must be >= 2")
    folds = stratified_folds(rows, gt_field, k, seed, task=task)

    files: Dict[str, Rows] = {}
    fold_paths = []
    for f_i, idxs in enumerate(folds):
        name = f"fold_{f_i}.jsonl"
        files[name] = [rows[i] for i in sorted(idxs)]
        fold_paths.append(os.path.join(out_dir, name))
    iters = []
    for v in range(k):
        name = f"iter{v}_dev.jsonl"
        files[name] = [rows[i] for f_i, idxs in enumerate(folds) if f_i != v for i in sorted(idxs)]
        iters.append({"iter": v, "val": fold_paths[v], "dev": os.path.join(out_dir, name)})
    manifest = {
        "k": k,
        "seed": seed,
        "gt_field": gt_field,
        "n_rows": len(rows),
        "folds": fold_paths,
        "rotation": iters,
        "sha256": {name: sha256_rows(rs) for name, rs in files.items()},
        "usage": USAGE_NOTE,
    }
    return KFoldResult(
        files=files,
        manifest=manifest,
        fold_sizes=[len(f) for f in folds],
        n_rows=len(rows),
        dropped=dropped,
        warnings=notes,
    )
