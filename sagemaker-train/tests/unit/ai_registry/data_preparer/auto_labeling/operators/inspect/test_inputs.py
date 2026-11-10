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
"""Unit tests for operators.auto_labeling.inspect_input: the first-row inventory."""

from __future__ import absolute_import

import gzip
import io
import json
import re

import boto3
import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import inputs as ii


def _jsonl(path, rows, prefix=""):
    path.write_text(prefix + "".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


class _S3:
    def __init__(self, objects):
        self.objects = objects

    def head_object(self, Bucket, Key):  # noqa: N803
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        n = int(re.match(r"bytes=0-(\d+)", Range).group(1)) + 1
        return {"Body": io.BytesIO(self.objects[Key][:n])}

    def get_paginator(self, name):
        objs = self.objects

        class _P:
            def paginate(self, Bucket, Prefix):  # noqa: N803
                yield {"Contents": [{"Key": k} for k in objs if k.startswith(Prefix)]}

        return _P()


class TestReadFirstRow:
    def test_local_and_estimate(self, tmp_path):
        p = _jsonl(tmp_path / "c.jsonl", [{"id": "1", "text": "a"}] * 10)
        row, rb, size, uri = ii.read_first_row(p)
        assert row == {"id": "1", "text": "a"} and uri == p
        assert rb == len(json.dumps(row)) + 1 and size == rb * 10
        assert ii.estimate_rows(size, rb) == 10 and ii.estimate_rows(0, rb) is None

    def test_gzip_scales_the_first_row_by_the_chunk_ratio(self, tmp_path):
        rows = [{"id": str(i), "text": "hello " * 20} for i in range(50)]
        raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
        p = tmp_path / "c.jsonl.gz"
        p.write_bytes(gzip.compress(raw))
        row, rb, size, _ = ii.read_first_row(str(p))
        assert row == rows[0] and size == p.stat().st_size
        assert 25 <= ii.estimate_rows(size, rb) <= 100  # scaled, so in the right order of magnitude

    def test_refusals(self, tmp_path):
        with pytest.raises(DataPreparerOperatorError, match="starts with a blank line"):
            ii.read_first_row(_jsonl(tmp_path / "b.jsonl", [{"a": 1}], prefix="\n"))
        with pytest.raises(DataPreparerOperatorError, match="is not JSON.*JSON Lines"):
            ii.read_first_row(_jsonl(tmp_path / "n.jsonl", [], prefix="nope\n"))
        with pytest.raises(DataPreparerOperatorError, match="is not a JSON object"):
            ii.read_first_row(_jsonl(tmp_path / "l.jsonl", [[1, 2]]))
        long = tmp_path / "long.jsonl"
        long.write_text('{"text": "' + "x" * 70000)
        with pytest.raises(DataPreparerOperatorError, match="longer than 65,536 bytes"):
            ii.read_first_row(str(long))

    def test_s3_object_folder_and_arn(self, monkeypatch):
        objects = {
            "d/c.jsonl": b'{"id": "1", "text": "t"}\n{"id": "2"}\n',
            "f/": b"",
            "f/one.jsonl": b'{"text": "a"}\n',
            "two/a.jsonl": b"{}\n",
            "two/b.jsonl": b"{}\n",
        }
        monkeypatch.setattr(boto3, "client", lambda *a, **k: _S3(objects))
        row, rb, size, uri = ii.read_first_row("s3://b/d/c.jsonl")
        assert (
            row == {"id": "1", "text": "t"}
            and size == len(objects["d/c.jsonl"])
            and uri == "s3://b/d/c.jsonl"
        )
        assert ii.read_first_row("s3://b/f/")[3] == "s3://b/f/one.jsonl"
        with pytest.raises(DataPreparerOperatorError, match="expected exactly one object"):
            ii.read_first_row("s3://b/two/")
        arn = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/h/DataSet/n/1.0.0"
        assert (
            ii.read_first_row(arn, resolve_arn=lambda a, r: "s3://b/d/c.jsonl")[3]
            == "s3://b/d/c.jsonl"
        )


class TestProposals:
    def test_task_shape(self):
        assert (
            ii.task_shape("short") == "classification"
            and ii.task_shape("x" * 201) == "summarization"
        )
        assert ii.task_shape(True) == "classification" and ii.task_shape(3.5) == "ordinal"
        assert ii.task_shape([]) == "ner" and ii.task_shape([{"type": "a", "text": "b"}]) == "ner"
        assert ii.task_shape(["a"]) == "qa" and ii.task_shape({"answer": "a"}) == "qa"
        assert ii.task_shape({"k": 1}) is None and ii.task_shape(None) is None

    def test_label_set_from_guidelines(self):
        text = (
            "## Label Set\n- billing: money\n- **refund**\n- `other` - the rest\n## Next\n- nope\n"
        )
        assert ii.label_set_from_guidelines(text) == {"billing", "refund", "other"}
        assert ii.label_set_from_guidelines('["Short", "x"]') == {"short", "x"}
        assert ii.label_set_from_guidelines("") == set()

    def test_label_field_precedence(self):
        row = {"id": "1", "text": "t", "gt_score": 3, "category": "billing"}
        assert ii.propose_label_field(row, ["id", "text"], {"billing"}) == "category"
        assert ii.propose_label_field(row, ["id", "text"]) == "gt_score"
        assert (
            ii.propose_label_field(
                {"id": "1", "text": "t", "extra": "a", "cat": "b"}, ["id", "text"]
            )
            == "cat"
        )
        assert ii.propose_label_field({"id": "1", "text": "t"}, ["id", "text"]) is None

    def test_text_fields_reserved_and_unpassed(self):
        row = {
            "id": "1",
            "text": "t",
            "title": "T",
            "source": "s",
            "label": "a",
            "_m": "x",
            "reasoning": "r",
            "n": 3,
            "annotator": "p",
        }
        assert ii.propose_text_fields(row) == ["text", "title", "reasoning", "annotator"]
        assert ii.reserved_text_columns(row) == ["source", "reasoning"]
        assert ii.unpassed_text_fields(row, ["text"]) == ["title"]


class TestInspect:
    def test_corpus_and_seed_report(self, tmp_path):
        corpus = _jsonl(
            tmp_path / "c.jsonl", [{"id": "1", "text": "hello", "title": "T", "source": "s"}] * 4
        )
        seed = _jsonl(
            tmp_path / "s.jsonl",
            [{"id": "1", "text": "hello", "title": "T", "source": "s", "gt_label": "billing"}],
        )
        rep = ii.inspect(corpus, seed, "## Label Set\n- billing\n", text_fields=["text"])
        assert (
            rep.corpus["fields"] == ["id", "text", "title", "source"]
            and rep.corpus["estimated_rows"] == 4
        )
        assert rep.corpus["reserved_text_fields"] == ["source"] and rep.corpus[
            "unpassed_text_fields"
        ] == ["title"]
        assert (
            rep.seed["proposed_label_field"] == "gt_label"
            and rep.seed["task_shape"] == "classification"
        )
        assert rep.proposed_text_fields == ["text", "title"]
        js = rep.to_json()
        assert (
            list(js)[0] == "warnings" and len(js["warnings"]) == 2
        )  # corpus reserved + unpassed; seed repeats neither
        assert "source" in js["warnings"][0] and "title" in js["warnings"][1]

    def test_warnings_for_missing_ids_fields_and_passed_reserved(self, tmp_path):
        corpus = _jsonl(tmp_path / "c.jsonl", [{"text": "hello", "extra": "e"}])
        seed = _jsonl(tmp_path / "s.jsonl", [{"text": "hello", "note": "n", "gt_label": "x"}])
        rep = ii.inspect(corpus, seed, None, text_fields=["source", "confidence_missing"])
        w = "\n".join(rep.warnings)
        assert (
            "corpus's first row has no `id` field" in w
            and "seed's first row has no `id` field" in w
        )
        assert "corpus field(s) extra are not on the seed's first row" in w
        assert "seed text field(s) note are not on the corpus's first row" in w
        assert "--text-fields names reserved column(s) source" in w
        assert "confidence_missing (the job writes it with --confidence-method verbalized)" in w

    def test_seed_only_and_no_label_field(self, tmp_path):
        # with no corpus, the last shaped field is taken as the label (a seed's label is usually last)
        seed = _jsonl(tmp_path / "s.jsonl", [{"id": "1", "text": "hello"}])
        rep = ii.inspect(None, seed)
        assert rep.corpus is None and rep.seed["proposed_label_field"] == "text"
        assert rep.proposed_text_fields == []
        seed = _jsonl(tmp_path / "s2.jsonl", [{"id": "1", "meta": {"k": 1}}])
        rep = ii.inspect(None, seed)
        assert rep.seed["proposed_label_field"] is None
        assert rep.warnings == ["no label field found on the seed's first row: name it"]


def test_one_s3_splitter():
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3

    assert ii._s3 is _s3
    assert _s3.split_s3_uri("s3://b/k/x.jsonl") == ("b", "k/x.jsonl")


def test_read_first_row_follows_a_corpus_ref_pointer(tmp_path, monkeypatch):
    # The pointer a resolve leaves in place of a remote corpus names its s3:// object; the
    # first-row read follows it (one ranged read of that object), so Step 0 can take the
    # pointer or the URI alike.
    seen = {}

    class _Body:
        def read(self):
            return b'{"id": 1, "text": "hello"}\n{"id": 2, "text": "x"}\n'

    class _S3:
        def head_object(self, Bucket, Key):
            seen["head"] = (Bucket, Key)
            return {"ContentLength": 52}

        def get_object(self, Bucket, Key, Range):
            seen["get"] = (Bucket, Key, Range)
            return {"Body": _Body()}

    monkeypatch.setattr(boto3, "client", lambda *a, **k: _S3())
    monkeypatch.setattr(ii, "_s3_folder_object", lambda uri, region: uri)
    ref = tmp_path / "corpus.ref.json"
    ref.write_text(json.dumps({"source_s3_uri": "s3://b/k/corpus.jsonl", "etag": "e"}, indent=2))
    row, n, size, resolved = ii.read_first_row(str(ref), region="us-west-2")
    assert row == {"id": 1, "text": "hello"} and size == 52 and resolved == "s3://b/k/corpus.jsonl"
    assert seen["head"] == ("b", "k/corpus.jsonl")
    bad = tmp_path / "bad.ref.json"
    bad.write_text("{}")
    with pytest.raises(DataPreparerOperatorError, match="source_s3_uri"):
        ii.read_first_row(str(bad))
