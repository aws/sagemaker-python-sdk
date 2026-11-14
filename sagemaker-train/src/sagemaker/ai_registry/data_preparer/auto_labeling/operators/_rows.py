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
"""Reading JSON Lines rows, and the client-read notice.

Shared by every operator that handles customer rows. The important rule is in
:func:`read_input_rows` and :func:`head_rows`: the corpus a labeling job labels stays
behind the API, and the operators read only a seed-sized file (the cold-start sample
or a split of it) or the first row or two of the corpus to see its field names. Each
such read is announced through the ``client_read`` logger (see
:func:`report_client_read`) with the row count, never a row's content, so a caller
can show the person where their data was read.

The ``confidence`` helper exists because several callers once guarded a per-row
confidence with ``isinstance(v, (int, float))``, which is also true for ``bool``, so
a stray ``true`` read as 1.0. It rejects booleans in one place.
"""

from __future__ import absolute_import

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3

# Every client-side read of customer rows logs one record here, at INFO, whose message
# is the full notice. A command-line caller attaches a handler that prints it to stderr.
CLIENT_READ_LOGGER = "sagemaker.ai_registry.data_preparer.auto_labeling.client_read"
_client_read_log = logging.getLogger(CLIENT_READ_LOGGER)

# The corpus stays behind the API: a caller reads a row or two of it to see its shape
# (the field names), never a sample of its content.
HEAD_ROWS = 2

# The most a client-side read takes in: larger data is read in a job, not on the client.
CLIENT_READ_MAX_BYTES = 50 * 1024 * 1024

# What to do with a file that is not JSON Lines (a CSV export, a spreadsheet).
JSONL_HINT = (
    "The labeling job reads JSON Lines (one JSON object per line). Convert a CSV, TSV or "
    "Parquet file first: dataset-transformation's ingest operation, or pandas "
    "(pd.read_csv(path).to_json(out, orient='records', lines=True))."
)


def client_read_message(primitive: str, n_rows: int) -> str:
    """The notice for ``primitive`` reading ``n_rows`` customer rows on the client.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import client_read_message

        client_read_message("seed_folds", 120)
        # 'client-read: seed_folds read 120 row(s) of your data on the client (cloud desktop),
        #  not in the SDG container.'
    """
    return (
        f"client-read: {primitive} read {int(n_rows)} row(s) of your data on the "
        "client (cloud desktop), not in the SDG container."
    )


def report_client_read(primitive: str, n_rows: int) -> None:
    """Announce that ``primitive`` read ``n_rows`` customer rows on the client.

    Logs one INFO record on the ``CLIENT_READ_LOGGER`` logger whose message is
    :func:`client_read_message`. The prefix is ``client-read:``, not ``WARNING:``, so
    it does not trip checks that look for actionable warnings.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import report_client_read

        report_client_read("scoring", len(pred_rows) + len(gold_rows))

    Args:
        primitive: The operator or helper that read the rows.
        n_rows: How many rows it read.
    """
    _client_read_log.info(client_read_message(primitive, n_rows))


class _CurrentStderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is when a record is emitted.

    A handler bound to the stderr object of attach time would keep writing there after
    the caller redirected stderr (a test capturing output, a wrapper), and would fail once
    that object is closed.
    """

    def __init__(self):
        super().__init__(sys.stderr)

    @property
    def stream(self):
        """The current ``sys.stderr``."""
        return sys.stderr

    @stream.setter
    def stream(self, value):  # StreamHandler.__init__ and setStream assign it; ignored
        pass


def print_client_reads(stream=None) -> logging.Handler:
    """Attach a handler that prints each client-read notice, for a command-line caller.

    Prints to ``stream``, or to the current ``sys.stderr`` at the time of each notice when
    ``stream`` is ``None`` (so a redirected stderr sees the notices). Idempotent per stream.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import print_client_reads

        print_client_reads()  # a command-line caller: every client-read: notice goes to stderr

    Args:
        stream: The text stream to print to.

    Returns:
        The handler attached (already attached, when one for that stream existed).
    """
    for h in _client_read_log.handlers:
        if stream is None and isinstance(h, _CurrentStderrHandler):
            return h
        if stream is not None and isinstance(h, logging.StreamHandler) and h.stream is stream:
            return h
    handler = _CurrentStderrHandler() if stream is None else logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _client_read_log.addHandler(handler)
    _client_read_log.setLevel(logging.INFO)
    _client_read_log.propagate = False
    return handler


def read_jsonl(path: str, add_id: bool = False) -> List[Any]:
    """Read a JSON Lines file into a list, skipping blank lines.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import read_jsonl

        rows = read_jsonl("proj/splits/dev.jsonl", add_id=True)  # a missing id is the line index

    Args:
        path: The file to read.
        add_id: Fill a missing ``id`` on each object row with the row's line index, as
            a string, for splitters that align rows by id.

    Returns:
        One parsed value per non-blank line.
    """
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if add_id and isinstance(row, dict):
                row.setdefault("id", str(i))
            rows.append(row)
    return rows


def confidence(
    row: Dict[str, Any], field: str = "confidence_score", fallback: Optional[str] = "confidence"
) -> Optional[float]:
    """A row's confidence as a float, or ``None``.

    Reads ``field``, then ``fallback`` unless it is ``None``. A ``bool`` is not a score
    and reads as ``None``.

    Args:
        row: The labeled row.
        field: The confidence field the job writes.
        fallback: An older field name to try next.

    Returns:
        The confidence, or ``None`` when neither field holds a number.
    """
    for key in (field, fallback):
        if key is None:
            continue
        v = row.get(key)
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
    return None


def head_rows(
    uri: str, max_rows: int = HEAD_ROWS, max_bytes: int = 65_536, region: Optional[str] = None
) -> List[Any]:
    """The first ``max_rows`` JSON Lines records of an ``s3://`` object.

    One ranged ``GetObject`` of ``max_bytes``.

    This is the only corpus data an operator brings to the client in ``s3://`` mode (to
    read the field names), and it is announced as a client read. It never raises: a
    garbled or range-split line is skipped, and a short file, an empty object or a read
    error returns the rows read so far (possibly none).

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import head_rows

        first = head_rows("s3://bucket/corpus.jsonl", max_rows=5, region="us-west-2")  # one ranged GetObject

    Args:
        uri: The ``s3://bucket/key`` of the corpus.
        max_rows: How many records to return at most.
        max_bytes: How many bytes to fetch.
        region: Binds the S3 client to the corpus's region.

    Returns:
        The parsed records, oldest first.
    """
    rows: List[Any] = []
    if not _s3.is_s3_uri(uri):
        return rows
    try:
        bucket, key = _s3.split_s3_uri(uri)
        client = _s3.s3_client(region)
        resp = client.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{max_bytes - 1}")
        data = resp["Body"].read()
    except Exception:  # noqa: BLE001 - a peek must never crash the run; degrade to []
        return rows
    if not data:
        return rows
    # Trim to the last newline byte only when the window was full: the last line may
    # then be cut, even mid-character (0x0A is never a UTF-8 continuation byte). A
    # smaller object is whole, and its last record may have no trailing newline.
    if len(data) >= max_bytes:
        last_nl = data.rfind(b"\n")
        if last_nl != -1:
            data = data[: last_nl + 1]
    text = data.decode("utf-8", errors="replace")
    for line in text.split("\n"):
        if len(rows) >= max_rows:
            break
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError, ValueError):
            continue  # garbled / range-split line: skip, never raise
    if rows:
        report_client_read("head_rows", len(rows))
    return rows


def s3_uri_for_input(path_or_uri: Optional[str]) -> Optional[str]:
    """The ``s3://`` URI a corpus input points at.

    Args:
        path_or_uri: An ``s3://`` URI, a ``*.ref.json`` pointer file, or a local path.

    Returns:
        The URI as given, the pointer's ``source_s3_uri``, or ``None`` for a local path
        (and for an unreadable pointer).
    """
    if not path_or_uri:
        return None
    if path_or_uri.startswith("s3://"):
        return path_or_uri
    if path_or_uri.endswith(".ref.json"):
        try:
            with open(path_or_uri) as f:
                doc = json.load(f)
        except (OSError, ValueError):
            return None
        return doc.get("source_s3_uri")
    return None


def _check_client_read_size(size: Optional[int], where: str) -> None:
    if size is not None and size > CLIENT_READ_MAX_BYTES:
        raise DataPreparerOperatorError(
            f"{where} is {size:,} bytes, over the {CLIENT_READ_MAX_BYTES:,}-byte "
            "client-side read limit; draw a smaller sample in a job first"
        )


def read_input_rows(
    path_or_uri: str, region: Optional[str] = None, primitive: str = "read_input_rows"
) -> List[Any]:
    """Every non-blank JSON Lines record of a client-readable file.

    For the cold-start sample, or a whole corpus no larger than it; never the unlabeled
    corpus or a corpus run's labels. An ``s3://`` object is streamed and nothing is
    written to disk. The read is announced as a client read under ``primitive``. A file
    over :data:`CLIENT_READ_MAX_BYTES` is refused before any row is read: larger data is
    read in a job, not on the client.

    Example::

        from sagemaker.ai_registry.data_preparer.auto_labeling import read_input_rows

        rows = read_input_rows("proj/splits/dev.jsonl", primitive="seed_folds")  # prints a client-read: notice

    Args:
        path_or_uri: A local path, an ``s3://`` URI or a ``*.ref.json`` pointer.
        region: Binds the S3 client to the object's region.
        primitive: The name the client-read notice carries.

    Returns:
        The parsed records in file order.

    Raises:
        DataPreparerOperatorError: The file is over the client-side read limit.
        json.JSONDecodeError: A line is not JSON.
        OSError: The local file cannot be read.
    """
    uri = s3_uri_for_input(path_or_uri)
    if uri is None:
        _check_client_read_size(os.path.getsize(path_or_uri), path_or_uri)
        with open(path_or_uri) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        report_client_read(primitive, len(rows))
        return rows
    # Remote: one streamed pass, nothing written to disk.
    import boto3

    bucket, _, key = uri[len("s3://") :].partition("/")
    client = boto3.client("s3", region_name=region) if region else boto3.client("s3")
    obj = client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"]
    try:
        _check_client_read_size(obj.get("ContentLength"), uri)
    except DataPreparerOperatorError:
        body.close()
        raise
    rows = []
    pending = b""
    try:
        for chunk in body.iter_chunks(chunk_size=1 << 20):
            if not chunk:
                continue
            pending += chunk
            nl = pending.find(b"\n")
            while nl != -1:
                line = pending[:nl].decode("utf-8").strip()
                if line:
                    rows.append(json.loads(line))
                pending = pending[nl + 1 :]
                nl = pending.find(b"\n")
    finally:
        try:
            body.close()
        except Exception:  # noqa: BLE001
            pass
    last = pending.decode("utf-8").strip()
    if last:
        rows.append(json.loads(last))
    report_client_read(primitive, len(rows))
    return rows
