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
"""The cold-start sample: the corpus rows people label first, when there is no seed yet.

:func:`cold_start_sample_size` says how many rows (more for a high accuracy target),
:func:`certificate_warning` whether that many rows can certify the target at all, and
:func:`draw_sample` draws them uniformly without replacement, keeping only the id and
the text fields, with a report of what was drawn. Every refusal is a
:class:`~sagemaker.ai_registry.data_preparer.exceptions.DataPreparerOperatorError`.
"""

from __future__ import absolute_import

import random

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed._bounds import (
    best_zero_error_target,
    calibration_sample_size,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import NO_PROMISE_TASKS
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.review_export import id_key

# People label this many random corpus rows, more when the accuracy target is high.
COLD_START_SAMPLE_ROWS = 200
COLD_START_SAMPLE_ROWS_HIGH_TARGET = 400
COLD_START_HIGH_TARGET = 0.95
# The share of the sample that backs the certificate: val and test of the split (dev never
# calibrates) when the spec ships unchanged, and test alone, about a quarter, once
# refinement has chosen the spec on dev and val.
CERTIFICATE_SHARE = 0.5
REFINED_CERTIFICATE_SHARE = 0.25


def cold_start_sample_size(target, n_rows, task=None):
    """Rows in the cold-start sample.

    ``COLD_START_SAMPLE_ROWS``, or ``COLD_START_SAMPLE_ROWS_HIGH_TARGET`` when the accuracy
    target is ``COLD_START_HIGH_TARGET`` or higher and the task is not one of the
    no-promise tasks, and never more than the corpus has.
    """
    want = (
        COLD_START_SAMPLE_ROWS_HIGH_TARGET
        if target >= COLD_START_HIGH_TARGET and task not in NO_PROMISE_TASKS
        else COLD_START_SAMPLE_ROWS
    )
    return max(0, min(want, n_rows))


def certificate_warning(n_sample, target, task=None, alpha=0.05):
    """A warning when ``target`` is above what an ``n_sample``-row sample can certify, else ``None``.

    ``None`` too for the no-promise tasks, which get no certificate. The check uses the
    stricter share, test alone (the rows independent of a refined spec), and the warning
    states both requirements: val and test for a spec that ships unchanged, and test alone
    for a refined one.
    """
    if task in NO_PROMISE_TASKS:
        return None
    need = calibration_sample_size(target, alpha)
    eval_rows = int(n_sample * CERTIFICATE_SHARE)
    test_rows = int(n_sample * REFINED_CERTIFICATE_SHARE)
    if need <= test_rows:
        return None
    unchanged = (
        "an unchanged spec can reach it only if every val and test row is right"
        if need <= eval_rows
        else "even an unchanged spec cannot reach "
        f"it (about {best_zero_error_target(eval_rows, alpha):g} at most)"
    )
    return (
        f"a {target:g} target needs at least {need} rows, every one right, for a "
        f"certificate. This {n_sample}-row sample puts about {eval_rows} rows in "
        f"val and test, which certify a spec that ships unchanged, and about "
        f"{test_rows} in test alone, the only rows independent of a refined spec: "
        f"{unchanged}, and a refined spec cannot (about "
        f"{best_zero_error_target(test_rows, alpha):g} at most)."
    )


def draw_sample(  # pylint: disable=too-many-locals,too-many-branches,too-many-positional-arguments
    corpus_rows, n, seed, id_field="id", text_fields=("text",), corpus_total=None
):
    """Draw the cold-start sample.

    ``n`` input rows drawn uniformly without replacement (the first ``n`` positions of a
    ``random.Random(seed)`` shuffle, kept in file order, so a smaller sample is part of a
    larger one with the same seed). Each sample row keeps only ``id_field`` and its
    ``text_fields``. A row with no id gets ``str(i)``, the job's rule, which is its corpus
    position only when the input is the whole corpus (``corpus_total`` equals the row
    count), so an id-less row is otherwise refused.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import cold_start

        n = cold_start.cold_start_sample_size(0.9, len(rows))  # the target accuracy sizes the sample
        sample, report = cold_start.draw_sample(rows, n, seed=0, text_fields=("text",), corpus_total=len(rows))

    Args:
        corpus_rows: The input rows.
        n: How many to draw.
        seed: The shuffle seed.
        id_field: The id field.
        text_fields: The fields to keep.
        corpus_total: The corpus's row count when known; the report's ``whole_corpus``
            is true only when the sample covers it.

    Returns:
        ``(sample_rows, report)``.

    Raises:
        DataPreparerOperatorError: No rows, a row that is not an object, an id-less row
            when the input is not the whole corpus, a missing non-default id field, two rows
            sharing an id, no row with any text field, or a sampled row with no text.
    """
    if not corpus_rows:
        raise DataPreparerOperatorError("the input has no rows")
    is_corpus = corpus_total is not None and corpus_total == len(corpus_rows)
    ids, seen, dups = [], set(), set()
    for i, r in enumerate(corpus_rows):
        if not isinstance(r, dict):
            raise DataPreparerOperatorError(f"input row {i} is not a JSON object")
        if r.get(id_field) is not None:
            rid = r[id_field]
        elif id_field == "id":
            # The labeling job treats a null id as missing and uses the row's position in
            # the corpus; only the whole corpus gives that position here.
            if not is_corpus:
                raise DataPreparerOperatorError(
                    f"input row {i} has no id; the labeling job gives an id-less row its "
                    "position in the corpus, which a sample of a larger corpus does not "
                    "carry, so its label could not join back. Give every corpus row an "
                    "id first, or pass --corpus-rows when the input is the whole corpus"
                )
            rid = str(i)
        else:
            raise DataPreparerOperatorError(
                f"input row {i} has no {id_field!r}; the labeling job fills only `id` "
                f"(its position), so with --id-field {id_field!r} every row needs its own"
            )
        key = id_key(rid)
        if key in seen:
            dups.add(key)
        seen.add(key)
        ids.append(rid)
    if dups:
        shown = ", ".join(repr(d) for d in sorted(dups)[:5])
        raise DataPreparerOperatorError(
            f"{len(dups)} id(s) name more than one input row ({shown}); "
            f"the labels join back by --id-field {id_field!r}, so every "
            "row needs its own id"
        )
    fields = list(text_fields)
    if not any(f in r for r in corpus_rows for f in fields):
        raise DataPreparerOperatorError(
            f"no input row has any of --text-fields {fields}; the first "
            f"row has {sorted(corpus_rows[0])}"
        )
    order = list(range(len(corpus_rows)))
    random.Random(seed).shuffle(order)
    sample, filled, no_text = [], 0, []
    for i in sorted(order[: max(0, n)]):
        r = corpus_rows[i]
        row = {id_field: ids[i]}
        if r.get(id_field) is None:
            filled += 1
        for f in fields:
            if f in r:
                row[f] = r[f]
        # Content as the labeling run reads it: present, non-null and non-blank.
        if not any(r.get(f) is not None and str(r[f]).strip() for f in fields):
            no_text.append(ids[i])
        sample.append(row)
    if no_text:
        shown = ", ".join(repr(x) for x in no_text[:5])
        raise DataPreparerOperatorError(
            f"{len(no_text)} sampled row(s) have no text in any of "
            f"--text-fields {fields} ({shown}); people would get nothing "
            "to label: fill or drop those input rows, or name the fields "
            "they carry"
        )
    total = corpus_total if corpus_total is not None else len(corpus_rows)
    report = {
        "mode": "sample",
        "input_rows": len(corpus_rows),
        "corpus_rows": total,
        "sample_rows": len(sample),
        "whole_corpus": corpus_total is not None and len(sample) >= corpus_total,
        "seed": seed,
        "id_field": id_field,
        "text_fields": fields,
        "ids_filled": filled,
    }
    return sample, report
