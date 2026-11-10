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
"""A first-row read of each input file, for the field names and the label field.

Field names, proposed label and text fields, the task shape, and a row-count estimate
from the object size.

The corpus stays behind the API. :func:`inspect` reads one row of each input (one
ranged ``GetObject`` of the first few KB for an ``s3://`` object or a DataSet ARN, the
first line of a local file) and takes the object size from S3 metadata or the local
file size. Nothing else of the corpus is read on the client; the labeling job counts
the rows it loads and enforces its row cap. The seed is small and client-readable, but
this step reads one row of it too: the field names and the label field are all it
needs. Every read is announced as a client read of one row.

Use it before a labeling run to confirm the inputs are what the person thinks they
are: which columns are text, which one holds the gold, what shape that gold has, and
whether a column name is one the job reserves (``source``, for instance), which the job
would refuse in ``TextFields``.
"""

from __future__ import absolute_import

import json
import os
import re
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pydantic import Field

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

from sagemaker.ai_registry.data_preparer._operator_result import OperatorResult
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import (
    JSONL_HINT,
    report_client_read,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._fields import (
    OPTION_RESERVED_FIELDS,
    REVIEW_HIDDEN_FIELDS,
    reserved_field_reason,
    reserved_text_field_conflicts,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows, _s3

# The review fold's annotator columns, never text columns.
ANNOTATOR_FIELDS = (
    "annotator",
    "annotator_id",
    "reviewer",
    "reviewer_id",
    "completed_by_profile",
    "completed_by",
)
# One ranged read is enough for a first row: a 64 KB window covers rows up to that size.
FIRST_ROW_WINDOW_BYTES = 64 * 1024
# A string label this long is a summary, not a class (summarization), unless the
# guidelines name it.
LONG_STRING_CHARS = 200
_GT_PREFIX = "gt_"
_ID_FIELDS = ("id", "record_id", "unique_id")
# Field names that are never text fields.
_NON_TEXT_FIELDS = frozenset(
    {"id", "record_id", "unique_id", "label", "labels", "gt_label", "split", "source", "provenance"}
)

ArnResolver = Callable[[str, Optional[str]], str]


class InspectReport(OperatorResult):
    """What :func:`inspect` found.

    ``corpus`` and ``seed`` (each when given) carry ``uri``, ``fields``,
    ``size_bytes``, ``first_row_bytes``, ``estimated_rows``, ``proposed_text_fields``,
    ``has_id``, ``reserved_text_fields`` and, when ``text_fields`` was passed,
    ``unpassed_text_fields``; the seed adds ``proposed_label_field`` and
    ``task_shape``. ``proposed_text_fields`` is the one list to label from, and
    ``warnings`` the findings to show the person (part of the report, so
    ``to_json()`` keeps them).
    """

    corpus: Optional[Dict[str, Any]] = None
    seed: Optional[Dict[str, Any]] = None
    proposed_text_fields: Optional[List[str]] = None

    def to_json(self) -> Dict[str, Any]:
        """The report document, ``warnings`` included."""
        out = self.model_dump(exclude={"events"}, exclude_none=True, mode="json")
        return {"warnings": out.pop("warnings"), **out}


def _default_arn_resolver(arn: str, region: Optional[str]) -> str:
    """A DataSet ARN -> its single ``.jsonl`` object URI, through the Hub resolver."""
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register

    return register.resolve_dataset_ref(arn, out_path=None, region=region, download=False)[
        "resolved_s3_uri"
    ]


def _s3_folder_object(uri, region):
    """A prefix (folder) -> the single data object under it; an object URI as is."""
    if not uri.endswith("/"):
        return uri
    import boto3

    bucket, key = _s3.split_s3_uri(uri)
    hits = []
    for page in (
        boto3.client("s3", region_name=region)
        .get_paginator("list_objects_v2")
        .paginate(Bucket=bucket, Prefix=key)
    ):
        hits.extend(o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/"))
    if len(hits) != 1:
        raise DataPreparerOperatorError(
            f"expected exactly one object under {uri} (found {len(hits)})"
        )
    return f"s3://{bucket}/{hits[0]}"


def read_first_row(
    path_or_uri: str,
    region: Optional[str] = None,
    window: int = FIRST_ROW_WINDOW_BYTES,
    resolve_arn: Optional[ArnResolver] = None,
) -> Tuple[Dict[str, Any], int, int, str]:
    """The first row of a JSON Lines input, its size, the object size and the resolved URI.

    Reads the first ``window`` bytes only (one ranged ``GetObject`` for S3); a ``.gz``
    object is decompressed from that chunk, and its first-row size is scaled by the
    chunk's compression ratio so ``size / first_row`` still estimates rows.

    Args:
        path_or_uri: A local path, an ``s3://`` URI (object or single-object folder), a
            DataSet ARN, or a ``corpus.ref.json`` pointer a resolve left in place of a
            remote corpus (read as its ``source_s3_uri``).
        region: The S3 / Hub region.
        window: How many bytes to read.
        resolve_arn: ``(arn, region) -> s3 uri`` for a DataSet ARN; the Hub resolver by
            default.

    Returns:
        ``(first_row, first_row_bytes, size_bytes, resolved_uri)``.

    Raises:
        DataPreparerOperatorError: No complete first row fits the window, the first line
            is blank or not a JSON object, a folder holds other than one object, or a
            ``.gz`` object is not a gzip stream.
    """
    resolved = path_or_uri
    if path_or_uri.endswith(".ref.json"):
        pointed = _rows.s3_uri_for_input(path_or_uri)
        if not pointed:
            raise DataPreparerOperatorError(
                f"{path_or_uri} is not a readable corpus.ref.json pointer (no 'source_s3_uri')"
            )
        resolved = pointed
    if resolved.startswith("arn:"):
        resolved = (resolve_arn or _default_arn_resolver)(resolved, region)
    if resolved.startswith("s3://"):
        import boto3

        resolved = _s3_folder_object(resolved, region)
        bucket, key = _s3.split_s3_uri(resolved)
        cli = boto3.client("s3", region_name=region)
        head = cli.head_object(Bucket=bucket, Key=key)
        size = int(head.get("ContentLength") or 0)
        data = cli.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{window - 1}")["Body"].read()
    else:
        size = os.path.getsize(resolved)
        with open(resolved, "rb") as fh:
            data = fh.read(window)
    ratio = 1.0
    if resolved.endswith(".gz"):
        # Decompress the first chunk only: a streaming decompressor yields the leading
        # bytes of a truncated member, which is all a first row needs.
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        compressed = len(data)
        try:
            data = d.decompress(data)
        except zlib.error as e:
            raise DataPreparerOperatorError(f"{resolved} is not a gzip stream: {e}") from None
        # The object's size is compressed bytes: report the first row in compressed
        # bytes too (the chunk's own ratio), so size / first row estimates rows.
        if data and compressed:
            ratio = len(data) / compressed
    nl = data.find(b"\n")
    if nl == -1:
        if len(data) >= window and not resolved.endswith(".gz"):
            raise DataPreparerOperatorError(
                f"the first row of {resolved} is longer than {window:,} bytes"
            )
        line = data
    else:
        line = data[:nl]
    text = line.decode("utf-8", errors="replace").strip()
    if not text:
        raise DataPreparerOperatorError(
            f"{resolved} starts with a blank line; a JSONL file has one object per line"
        )
    try:
        row = json.loads(text)
    except ValueError as e:
        raise DataPreparerOperatorError(
            f"the first line of {resolved} is not JSON: {e}. {JSONL_HINT}"
        ) from None
    if not isinstance(row, dict):
        raise DataPreparerOperatorError(f"the first line of {resolved} is not a JSON object")
    report_client_read("inspect_input.read_first_row", 1)
    return row, max(1, round((len(line) + 1) / ratio)), size, resolved


def estimate_rows(size_bytes, first_row_bytes):
    """``size / first-row size``, rounded; ``None`` when either is unknown.

    A first estimate the labeling job's summary (``RowsLabeled``) makes exact.
    """
    if not size_bytes or not first_row_bytes:
        return None
    return max(1, round(size_bytes / first_row_bytes))


def label_set_from_guidelines(text):
    """The label names a guidelines or spec file lists under its Label Set heading.

    ``- name: definition`` / ``- **name**`` bullets, or a JSON list of strings,
    lowercased; empty when none is found.
    """
    if not text:
        return set()
    m = re.search(r"#+\s*Label Set\s*\n(.*?)(?=\n#+\s|\Z)", text, re.IGNORECASE | re.DOTALL)
    block = m.group(1) if m else text
    labels = set()
    for line in block.splitlines():
        b = re.match(r"\s*[-*]\s+\**`?([^`*:]+?)`?\**\s*(?::|$|\s-\s)", line)
        if b:
            labels.add(b.group(1).strip().lower())
    j = re.search(r"\[\s*\"[^\]]*\"\s*\]", block)
    if j:
        try:
            labels.update(str(x).lower() for x in json.loads(j.group(0)))
        except ValueError:
            pass
    return {label for label in labels if label}


def task_shape(  # pylint: disable=too-many-return-statements
    value: Any, long_string_chars: int = LONG_STRING_CHARS
) -> Optional[str]:
    """The task shape a label value implies.

    Classification for a short string or bool, ner for a list of ``{type, text}``, qa
    for a list of strings or an ``{answer}`` object, ordinal for a number, summarization
    for a long string; ``None`` when nothing fits.
    """
    if isinstance(value, bool):
        return "classification"
    if isinstance(value, (int, float)):
        return "ordinal"
    if isinstance(value, str):
        return "summarization" if len(value) > long_string_chars else "classification"
    if isinstance(value, list):
        if not value:
            return "ner"  # an empty span list; a qa gold is rarely empty
        if all(isinstance(x, dict) and {"type", "text"} <= set(x) for x in value):
            return "ner"
        if all(isinstance(x, str) for x in value):
            return "qa"
    if isinstance(value, dict) and ("answer" in value or "text" in value):
        return "qa"
    return None


def propose_label_field(seed_row, corpus_fields, guideline_labels=()):
    """The seed field that holds the gold.

    The field whose string value is in the guidelines' label set, else the first
    ``gt_*`` field, else the last field the corpus lacks whose value has a task shape (a
    seed's label is usually its last column), else ``None``.
    """
    if not seed_row:
        return None
    labels = {label.lower() for label in guideline_labels}
    for k, v in seed_row.items():
        if labels and isinstance(v, str) and v.strip().lower() in labels:
            return k
    for k in seed_row:
        if k.startswith(_GT_PREFIX):
            return k
    candidates = [
        k
        for k, v in seed_row.items()
        if k not in corpus_fields and k not in _ID_FIELDS and task_shape(v) is not None
    ]
    return candidates[-1] if candidates else None


def propose_text_fields(row, exclude=()):
    """The string fields of a row that read as text, in the row's order.

    Not an id, a label or a marker.
    """
    out = []
    for k, v in row.items():
        if k in exclude or k in _NON_TEXT_FIELDS or k.startswith((_GT_PREFIX, "_")):
            continue
        if isinstance(v, str) and v.strip():
            out.append(k)
    return out


def unpassed_text_fields(row, text_fields, exclude=()):
    """The text columns of ``row`` that ``text_fields`` leaves out, in the row's order.

    Columns the labeler never sees. Model-output and job-written names, and the review
    fold's annotator and ``source_export`` columns, are not text columns.
    """
    skip = set(exclude) | set(REVIEW_HIDDEN_FIELDS) | set(ANNOTATOR_FIELDS) | {"source_export"}
    return [f for f in propose_text_fields(row, exclude=skip) if f not in set(text_fields)]


def reserved_text_columns(row, exclude=()):
    """The non-empty text columns of ``row`` whose names the labeling job reserves.

    In the row's order, leaving out ids, gold (``gt_*``, ``label``, ``labels``), ``_``
    markers and ``exclude`` (the seed's label field). The job refuses such a name in
    ``TextFields``, so the labeler cannot read the column under that name.
    """
    out = []
    for k, v in row.items():
        if (
            k in exclude
            or k in _ID_FIELDS
            or k in ("split", "label", "labels")
            or k.startswith((_GT_PREFIX, "_"))
            or not (isinstance(v, str) and v.strip())
        ):
            continue
        if reserved_text_field_conflicts([k]):
            out.append(k)
    return out


def _reserved_warning(where, fields, passed=False):
    why = "; ".join(f"`{f}`: {reserved_field_reason(f)}" for f in fields)
    lead = (
        f"--text-fields names reserved column(s) {', '.join(fields)}, which the labeling "
        "job refuses in TextFields"
        if passed
        else f"the {where}'s first row has text column(s) {', '.join(fields)} whose names the "
        "labeling job reserves, so the labeler cannot read them under those names"
    )
    return (
        f"{lead} ({why}). If the label can depend on one, ask the customer to copy it "
        "to a new name (for example `source` to `evidence`) in the corpus and the "
        "seed, then pass the new name"
    )


def _unpassed_warning(where, fields):
    return (
        f"the {where}'s first row has text column(s) {', '.join(fields)} that "
        "--text-fields does not pass to the labeler; if the label can depend on them "
        "(a title, a source, a question, the surrounding context), ask the customer "
        "whether to add them to --text-fields and the spec's Input Data"
    )


def inspect(  # pylint: disable=too-many-positional-arguments
    corpus: Optional[str] = None,
    seed: Optional[str] = None,
    guidelines_text: Optional[str] = None,
    region: Optional[str] = None,
    text_fields: Optional[Sequence[str]] = None,
    resolve_arn: Optional[ArnResolver] = None,
) -> InspectReport:
    """The input inventory for the corpus and the seed, from one row of each.

    Use it at the start of a labeling run, before any spec is written, to learn the
    field names and propose the text fields and the label field. Pass the unlabeled
    corpus and the labeled seed; either may be left out. With ``text_fields`` (the
    fields the spec passes to the labeler) each file also lists its
    ``unpassed_text_fields`` and the report warns about them.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import inputs

        report = inputs.inspect(corpus="s3://bucket/corpus.jsonl", seed="seed.jsonl", region="us-west-2")
        report.proposed_text_fields, report.seed["proposed_label_field"], report.warnings

    Args:
        corpus: The corpus: a local path, an ``s3://`` URI or a DataSet ARN.
        seed: The labeled seed, same forms.
        guidelines_text: The guidelines or spec Markdown; its Label Set picks the label
            field.
        region: The S3 / Hub region.
        text_fields: The fields the spec passes to the labeler, when known.
        resolve_arn: ``(arn, region) -> s3 uri`` for DataSet ARNs; the Hub resolver by
            default.

    Returns:
        An :class:`InspectReport`; ``to_json()`` is the report document.

    Raises:
        DataPreparerOperatorError: A file cannot be read as JSON Lines (see
            :func:`read_first_row`).
    """
    passed_fields = list(text_fields or [])
    result: Dict[str, Any] = {"warnings": []}
    corpus_fields = []
    if corpus:
        row, rb, size, uri = read_first_row(corpus, region, resolve_arn=resolve_arn)
        corpus_fields = list(row)
        result["corpus"] = {
            "uri": uri,
            "fields": corpus_fields,
            "size_bytes": size,
            "first_row_bytes": rb,
            "estimated_rows": estimate_rows(size, rb),
            "proposed_text_fields": propose_text_fields(row),
            "has_id": "id" in row,
        }
        reserved = reserved_text_columns(row)
        result["corpus"]["reserved_text_fields"] = reserved
        if reserved:
            result["warnings"].append(_reserved_warning("corpus", reserved))
        if passed_fields:
            unpassed = unpassed_text_fields(row, passed_fields)
            result["corpus"]["unpassed_text_fields"] = unpassed
            if unpassed:
                result["warnings"].append(_unpassed_warning("corpus", unpassed))
        if not result["corpus"]["has_id"]:
            result["warnings"].append(
                "the corpus's first row has no `id` field (the job keys rows on `id` only, "
                "not record_id or unique_id); the job uses each row's position as "
                'its id ("0", "1", ...), so a seed cannot be joined to its rows by id. A '
                "seed labeled outside the corpus still certifies (the job labels its rows to "
                'score them) when its ids are not positions: a seed row with id "7" is '
                "scored against the corpus's eighth row"
            )
    if seed:
        row, rb, size, uri = read_first_row(seed, region, resolve_arn=resolve_arn)
        labels = label_set_from_guidelines(guidelines_text or "")
        label_field = propose_label_field(row, corpus_fields, labels)
        shape = task_shape(row.get(label_field)) if label_field else None
        text_fields = propose_text_fields(row, exclude={label_field} if label_field else ())
        result["seed"] = {
            "uri": uri,
            "fields": list(row),
            "size_bytes": size,
            "first_row_bytes": rb,
            "estimated_rows": estimate_rows(size, rb),
            "proposed_label_field": label_field,
            "task_shape": shape,
            "proposed_text_fields": text_fields,
            "has_id": "id" in row,
        }
        reserved = reserved_text_columns(row, exclude={label_field} if label_field else ())
        result["seed"]["reserved_text_fields"] = reserved
        if reserved and not (
            corpus and set(reserved) <= set(result["corpus"]["reserved_text_fields"])
        ):
            result["warnings"].append(_reserved_warning("seed", reserved))
        if passed_fields:
            unpassed = unpassed_text_fields(
                row, passed_fields, exclude={label_field} if label_field else ()
            )
            result["seed"]["unpassed_text_fields"] = unpassed
            if unpassed and not (
                corpus and set(unpassed) <= set(result["corpus"].get("unpassed_text_fields") or [])
            ):
                result["warnings"].append(_unpassed_warning("seed", unpassed))
        if label_field is None:
            result["warnings"].append("no label field found on the seed's first row: name it")
        if corpus_fields:
            missing = [f for f in corpus_fields if f not in row and f not in _ID_FIELDS]
            if missing:
                result["warnings"].append(
                    f"corpus field(s) {', '.join(missing)} are not on the seed's first row"
                )
            extra = [f for f in text_fields if f not in corpus_fields]
            if extra:
                result["warnings"].append(
                    f"seed text field(s) {', '.join(extra)} are not on the corpus's first row; "
                    "a row missing a TextFields entry fails the job before any model call"
                )
        if not result["seed"]["has_id"]:
            result["warnings"].append(
                "the seed's first row has no `id` field (the job " "scores gold by `id` only)"
            )
    passed_reserved = reserved_text_field_conflicts(passed_fields)
    if passed_reserved:
        result["warnings"].append(_reserved_warning(None, passed_reserved, passed=True))
    option_reserved = [f for f in passed_fields if f in OPTION_RESERVED_FIELDS]
    if option_reserved:
        result["warnings"].append(
            "--text-fields "
            + ", ".join(
                f"{f} (the job writes it with {OPTION_RESERVED_FIELDS[f]})" for f in option_reserved
            )
            + ": offer that option only after the column is copied to a new name"
        )
    if "corpus" in result and "seed" in result:
        result["proposed_text_fields"] = [
            f for f in result["seed"]["proposed_text_fields"] if f in corpus_fields
        ] or result["corpus"]["proposed_text_fields"]
    elif "corpus" in result:
        result["proposed_text_fields"] = result["corpus"]["proposed_text_fields"]
    elif "seed" in result:
        result["proposed_text_fields"] = result["seed"]["proposed_text_fields"]
    return InspectReport(**result)


# ---------------------------------------------------------------------------
# The project-level step: one call that answers "what is in my data, and can this be
# labeled?" and carries the answer (fields, task, label set, region) through the rest of
# the flow, so no later call asks for them again.
# ---------------------------------------------------------------------------

DETECT_SAMPLE_ROWS = 50


class DataReport(OperatorResult):
    """What :func:`inspect_labeling_data` found; the ``data`` every later project-level call takes.

    ``task``, ``text_fields``, ``label_field`` and ``labels`` are what the labeling runs
    use; ``spec``, ``corpus``, ``seed`` and ``region`` are where they came from.
    ``inspect``, ``detection`` and ``gate`` are the three step reports as JSON;
    ``gate["decision"]`` is ``go`` or ``review-before-spend``.
    """

    task: str
    text_fields: List[str]
    label_field: str
    labels: List[str] = Field(default_factory=list)
    score_scale: Optional[List[float]] = None
    spec: str
    corpus: Optional[str] = None
    seed: Optional[str] = None
    region: Optional[str] = None
    inspect: Dict[str, Any] = Field(default_factory=dict)
    detection: Dict[str, Any] = Field(default_factory=dict)
    gate: Optional[Dict[str, Any]] = None


def inspect_labeling_data(  # pylint: disable=too-many-locals
    *,
    spec: str,
    corpus: Optional[str] = None,
    seed: Optional[str] = None,
    region: Optional[str] = None,
    text_fields: Optional[Sequence[str]] = None,
    task: Optional[str] = None,
) -> DataReport:
    """What is in my data, and can this be labeled? One call before anything is spent.

    Reads one row of the corpus and of the seed (:func:`inspect`), the first
    ``DETECT_SAMPLE_ROWS`` seed rows for the task (:func:`~..task.detect`), the spec's
    Label Set (:func:`~..taxonomy.parse_label_set`) and, with a seed, the seed's coverage
    of it (:func:`~..taxonomy.gate`). The seed is the only file read on the client; the
    corpus stays where it is.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import inspect_labeling_data

        data = inspect_labeling_data(spec="tickets/labeling_spec.md", corpus="s3://bucket/tickets.jsonl",
                            seed="tickets/seed.jsonl", region="us-west-2")
        data.task, data.text_fields, data.label_field, data.labels, data.gate["decision"]

    Args:
        spec: The labeling spec (Markdown file).
        corpus: The rows to label: a local path, an ``s3://`` URI or a DataSet ARN.
        seed: The labeled seed, same forms; ``None`` on a cold start.
        region: The S3 / Hub / model region.
        text_fields: The input fields, when the person already knows them.
        task: The task, when the person already knows it (else detected).

    Returns:
        A :class:`DataReport`; its ``warnings`` gather the three steps' findings.

    Raises:
        DataPreparerOperatorError: No spec, or a spec with no Label Set for a task that
            needs one.
    """
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import task as task_op
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import taxonomy
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
        spec_score_scale,
    )

    try:
        with open(spec, encoding="utf-8") as fh:
            spec_text = fh.read()
    except OSError as e:
        raise DataPreparerOperatorError(f"cannot read the spec {spec!r}: {e}") from e
    if corpus is None and seed is None:
        raise DataPreparerOperatorError("pass a corpus, a seed, or both")

    report = inspect(
        corpus=corpus, seed=seed, guidelines_text=spec_text, region=region, text_fields=text_fields
    )
    warnings = list(report.warnings)
    fields = list(text_fields or report.proposed_text_fields or [])
    if not fields:
        raise DataPreparerOperatorError(
            "no text field found; pass text_fields=[...] (the input fields the spec labels from)"
        )
    sample: List[Any] = []
    if seed is not None:
        sample = _rows.read_input_rows(seed, region=region, primitive="inspect_labeling_data")
    detection = task_op.detect(spec_text, sample[:DETECT_SAMPLE_ROWS])
    resolved_task = task or detection.task
    warnings.extend(detection.warnings)
    labels = [name for name, _ in taxonomy.parse_label_set(spec_text)] or sorted(
        label_set_from_guidelines(spec_text)
    )
    label_field = (report.seed or {}).get("proposed_label_field") or _default_gold_field(
        resolved_task
    )
    gate_report = None
    if sample:
        gate_report = taxonomy.gate(spec_text, sample, gt_field=label_field, task=resolved_task)
        warnings.extend(gate_report.warnings)
    scale = spec_score_scale(spec_text)
    return DataReport(
        task=resolved_task,
        text_fields=fields,
        label_field=label_field,
        labels=labels,
        score_scale=list(scale) if scale else None,
        spec=spec,
        corpus=corpus,
        seed=seed,
        region=region,
        inspect=report.to_json(),
        detection=detection.to_json(),
        gate=gate_report.to_json() if gate_report is not None else None,
        warnings=warnings,
    )


def _default_gold_field(task: str) -> str:
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import GOLD_FIELDS

    return GOLD_FIELDS.get(task, "gt_label")
