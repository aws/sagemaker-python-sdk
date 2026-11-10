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
"""Unit tests for operators.common.rows: JSONL reading, S3 peeks, the client-read notice."""

from __future__ import absolute_import

import io
import json
import logging
import re
import sys

import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows


class _FakeS3Head:
    """get_object(Range=bytes=0-(N-1)) returns the first N bytes; optionally raises."""

    def __init__(self, data, err=None):
        self.data = data
        self.err = err
        self.range = None

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803 - boto3 signature
        self.range = Range
        if self.err:
            raise self.err
        n = int(re.match(r"bytes=0-(\d+)", Range).group(1)) + 1
        return {"Body": io.BytesIO(self.data[:n])}


class _FakeS3Full:
    """A full get_object whose Body streams the object through iter_chunks."""

    def __init__(self, data, content_length=None):
        self.data = data
        self.gets = 0
        self.closed = 0
        self.content_length = len(data) if content_length is None else content_length

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        self.gets += 1
        data, fake = self.data, self

        class _Body:
            def iter_chunks(self, chunk_size=None):
                step = chunk_size or 4
                for i in range(0, len(data), step):
                    yield data[i : i + step]

            def close(self):
                fake.closed += 1

        return {"Body": _Body(), "ContentLength": self.content_length}


@pytest.fixture
def client_reads(monkeypatch):
    """Capture client-read notices as the logger emits them."""
    records = []
    logger = logging.getLogger(_rows.CLIENT_READ_LOGGER)

    class _H(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    h = _H()
    logger.addHandler(h)
    logger.setLevel(logging.INFO)
    yield records
    logger.removeHandler(h)


class TestReadJsonl:
    def test_roundtrip_and_skip_blank(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text('{"a": 1}\n\n   \n{"b": "é"}\n')
        assert _rows.read_jsonl(str(p)) == [{"a": 1}, {"b": "é"}]

    def test_add_id_fills_missing_id_with_line_index(self, tmp_path):
        p = tmp_path / "a.jsonl"
        p.write_text('{"a": 1}\n\n{"id": "x"}\n{"c": 3}\n')
        got = _rows.read_jsonl(str(p), add_id=True)
        assert [r["id"] for r in got] == ["0", "x", "3"]


class TestClientRead:
    def test_message_names_primitive_and_count(self):
        msg = _rows.client_read_message("read_input_rows", 12)
        assert msg.startswith("client-read: read_input_rows read 12 row(s)")
        assert "WARNING" not in msg

    def test_report_logs_on_the_dedicated_logger(self, client_reads):
        _rows.report_client_read("x", 3)
        assert client_reads == [_rows.client_read_message("x", 3)]

    def test_print_client_reads_goes_to_the_stream_once(self):
        stream = io.StringIO()
        h1 = _rows.print_client_reads(stream)
        h2 = _rows.print_client_reads(stream)
        try:
            assert h1 is h2
            _rows.report_client_read("p", 2)
            assert stream.getvalue() == _rows.client_read_message("p", 2) + "\n"
        finally:
            logging.getLogger(_rows.CLIENT_READ_LOGGER).removeHandler(h1)

    def test_the_default_handler_follows_a_redirected_stderr(self, monkeypatch):
        h1 = _rows.print_client_reads()
        try:
            assert _rows.print_client_reads() is h1
            first, second = io.StringIO(), io.StringIO()
            monkeypatch.setattr(sys, "stderr", first)
            _rows.report_client_read("p", 1)
            first.close()
            monkeypatch.setattr(sys, "stderr", second)
            _rows.report_client_read("q", 2)  # no "Logging error": the closed stream is not written
            assert second.getvalue() == _rows.client_read_message("q", 2) + "\n"
        finally:
            logging.getLogger(_rows.CLIENT_READ_LOGGER).removeHandler(h1)


class TestConfidence:
    def test_numeric_and_zero(self):
        assert _rows.confidence({"confidence_score": 0.7}) == 0.7
        assert _rows.confidence({"confidence_score": 0}) == 0.0

    def test_bool_rejected(self):
        assert _rows.confidence({"confidence_score": True}) is None

    def test_fallback_and_absent(self):
        assert _rows.confidence({"confidence": 0.4}) == 0.4
        assert _rows.confidence({}) is None
        assert _rows.confidence({"confidence": 0.4}, fallback=None) is None


class TestHeadRows:
    def _run(self, monkeypatch, blob, err=None, **kw):
        fake = _FakeS3Head(blob, err)
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        return _rows.head_rows("s3://b/k.jsonl", **kw), fake

    def test_plain_and_range_header(self, monkeypatch):
        rows, fake = self._run(monkeypatch, b'{"text":"a"}\n{"text":"b"}\n', max_bytes=1_048_576)
        assert rows == [{"text": "a"}, {"text": "b"}]
        assert fake.range == "bytes=0-1048575"

    def test_default_is_two_rows_and_announced(self, monkeypatch, client_reads):
        blob = "".join(json.dumps({"i": i}) + "\n" for i in range(50)).encode()
        rows, _ = self._run(monkeypatch, blob)
        assert _rows.HEAD_ROWS == 2
        assert rows == [{"i": 0}, {"i": 1}]
        assert client_reads == [_rows.client_read_message("head_rows", 2)]

    def test_truncated_window_trims_partial_multibyte_line(self, monkeypatch):
        good = (json.dumps({"text": "café 東京"}) + "\n").encode()
        tail = json.dumps({"text": "x"}).encode()
        rows, _ = self._run(monkeypatch, good + tail, max_bytes=len(good) + 3)
        assert rows == [{"text": "café 東京"}]

    def test_whole_object_keeps_last_row_without_newline(self, monkeypatch):
        blob = (json.dumps({"a": 1}) + "\n" + json.dumps({"b": 2})).encode()
        rows, _ = self._run(monkeypatch, blob, max_bytes=1_048_576)
        assert rows == [{"a": 1}, {"b": 2}]

    def test_garbled_blank_and_non_utf8_lines_skipped(self, monkeypatch):
        blob = b'{"text":"ok"}\nNOT JSON\n\n{"text":"\xff\xfe"}\n{"text":"after"}\n'
        rows, _ = self._run(monkeypatch, blob, max_rows=5)
        assert {"text": "ok"} in rows and {"text": "after"} in rows

    def test_errors_and_non_s3_never_raise(self, monkeypatch):
        err = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        assert self._run(monkeypatch, b'{"a":1}\n', err=err)[0] == []
        assert self._run(monkeypatch, b"")[0] == []
        assert _rows.head_rows("/local/path.jsonl") == []
        assert _rows.head_rows("") == []
        assert _rows.head_rows("s3://bucket-only") == []

    def test_region_threaded_to_client(self, monkeypatch):
        seen = {}

        def _spy(service, **kw):
            seen.update(service=service, region_name=kw.get("region_name"))
            return _FakeS3Head(b'{"text":"a"}\n')

        monkeypatch.setattr(boto3, "client", _spy)
        assert _rows.head_rows("s3://b/k.jsonl", region="eu-west-1") == [{"text": "a"}]
        assert seen == {"service": "s3", "region_name": "eu-west-1"}


class TestReadInputRows:
    def test_local_reads_and_announces(self, tmp_path, client_reads):
        p = tmp_path / "s.jsonl"
        p.write_text('{"a":1}\n\n{"a":2}\n')
        assert _rows.read_input_rows(str(p), primitive="split") == [{"a": 1}, {"a": 2}]
        assert client_reads == [_rows.client_read_message("split", 2)]

    def test_s3_streams_all_rows_in_one_get(self, monkeypatch, client_reads):
        blob = b'{"a":1}\n\n{"a":2}\n{"a":3}'  # blank line, no trailing newline
        fake = _FakeS3Full(blob)
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        assert _rows.read_input_rows("s3://b/k.jsonl") == [{"a": 1}, {"a": 2}, {"a": 3}]
        assert fake.gets == 1
        assert client_reads[-1] == _rows.client_read_message("read_input_rows", 3)

    def test_ref_json_resolves_then_streams(self, tmp_path, monkeypatch):
        ref = tmp_path / "corpus.ref.json"
        ref.write_text(json.dumps({"source_s3_uri": "s3://b/c.jsonl"}))
        seen = {}

        def _spy(service, **kw):
            seen["region"] = kw.get("region_name")
            return _FakeS3Full(b'{"a":1}\n')

        monkeypatch.setattr(boto3, "client", _spy)
        assert _rows.read_input_rows(str(ref), region="us-east-2") == [{"a": 1}]
        assert seen["region"] == "us-east-2"

    def test_a_local_file_over_the_client_read_cap_is_refused_unread(
        self, tmp_path, monkeypatch, client_reads
    ):
        p = tmp_path / "big.jsonl"
        p.write_text('{"a":1}\n' * 4)
        monkeypatch.setattr(_rows, "CLIENT_READ_MAX_BYTES", 10)
        with pytest.raises(
            DataPreparerOperatorError, match="over the 10-byte client-side read limit"
        ):
            _rows.read_input_rows(str(p))
        assert client_reads == []

    def test_an_s3_object_over_the_cap_is_refused_from_its_size_and_the_body_closed(
        self, monkeypatch, client_reads
    ):
        fake = _FakeS3Full(b'{"a":1}\n', content_length=_rows.CLIENT_READ_MAX_BYTES + 1)
        monkeypatch.setattr(boto3, "client", lambda *a, **k: fake)
        with pytest.raises(DataPreparerOperatorError, match="client-side read limit"):
            _rows.read_input_rows("s3://b/k.jsonl")
        assert fake.closed == 1 and client_reads == []


class TestS3UriForInput:
    def test_cases(self, tmp_path):
        assert _rows.s3_uri_for_input("s3://b/k") == "s3://b/k"
        assert _rows.s3_uri_for_input("/x.jsonl") is None
        assert _rows.s3_uri_for_input("") is None
        ref = tmp_path / "c.ref.json"
        ref.write_text('{"source_s3_uri": "s3://b/c"}')
        assert _rows.s3_uri_for_input(str(ref)) == "s3://b/c"
        assert _rows.s3_uri_for_input(str(tmp_path / "missing.ref.json")) is None
        ref.write_text("not json")
        assert _rows.s3_uri_for_input(str(ref)) is None

    def test_split_s3_uri(self):
        assert _rows._s3.split_s3_uri("s3://b/k/x") == ("b", "k/x")
        assert not _rows._s3.is_s3_uri("s3://b")
        assert not _rows._s3.is_s3_uri("http://b/k")
