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
"""Unit tests for operators.seed.bootstrap: the cold-start sample, merge and overlay files.

Ported from the auto-labeling skill's bootstrap_seed tests (sample, merge, overlay); the
command line's runs are calls of the writers, and an exit 2 is a
DataPreparerOperatorError. No AWS, model or network call (S3 is stubbed).
"""

from __future__ import absolute_import

import json
import logging
import os
import random

import pytest
from unittest import mock

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _fields
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rowrule import NO_PROMISE_TASKS
from sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows import CLIENT_READ_MAX_BYTES
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job.request import (
    MAX_LABELING_ROWS,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import bootstrap as B
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed._bounds import (
    calibration_sample_size,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed.cold_start import (
    certificate_warning,
    cold_start_sample_size,
    draw_sample,
)

_LOGGER = B.__name__


def _corpus(n):
    """Corpus rows that also carry what a labeled file holds: none of it may reach the sample."""
    return [
        {
            "id": f"r{i:04d}",
            "text": f"text {i}",
            "label": "a",
            "confidence_score": 1.0,
            "reasoning": "because",
            "jury_votes": [{"model": "m", "label": "a"}],
            "gt_label": "b",
            "meta": {"k": i},
        }
        for i in range(n)
    ]


def _write(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.fixture
def corpus500(tmp_path):
    return _write(tmp_path / "c.jsonl", _corpus(500))


# --- sample: the size rule ---------------------------------------------------------------


def test_size_rule_and_its_edges():
    cases = [
        (0.90, 20000, 200),
        (0.70, 20000, 200),
        (0.9499, 20000, 200),
        (0.95, 20000, 400),
        (0.99, 20000, 400),
        (0.90, 201, 200),
        (0.90, 200, 200),
        (0.90, 199, 199),
        (0.90, 1, 1),
        (0.95, 401, 400),
        (0.95, 400, 400),
        (0.95, 399, 399),
    ]
    for target, n_rows, want in cases:
        assert cold_start_sample_size(target, n_rows) == want, (target, n_rows)


def test_target_sets_the_size(tmp_path, corpus500):
    for target, want in ((0.90, 200), (0.9499, 200), (0.95, 400), (0.99, 400)):
        out = str(tmp_path / f"s{target}.jsonl")
        rep = B.write_sample(corpus500, out, "text", target=target)
        assert rep["sample_rows"] == want
        assert len(_read(out)) == want
    out = str(tmp_path / "default.jsonl")
    rep = B.write_sample(corpus500, out, "text")
    assert (rep["target"], len(_read(out))) == (0.90, 200)


def test_tasks_without_a_promise_keep_200_rows(tmp_path, corpus500):
    assert set(NO_PROMISE_TASKS) == {"ordinal", "llm_judge", "summarization"}
    for task in NO_PROMISE_TASKS:
        assert cold_start_sample_size(0.99, 20000, task) == 200, task
    for task in ("classification", "ner", "qa", None):
        assert cold_start_sample_size(0.95, 20000, task) == 400, task
    out = str(tmp_path / "s.jsonl")
    rep = B.write_sample(corpus500, out, "text", target=0.95, task="ordinal")
    assert (rep["sample_rows"], rep["task"]) == (200, "ordinal")
    with pytest.raises(DataPreparerOperatorError):  # not a task the labeler accepts
        B.write_sample(corpus500, out, "text", task="vision")


def test_warning_when_the_target_is_above_what_the_sample_can_certify(tmp_path, corpus500, caplog):
    for n, target in ((400, 0.95), (200, 0.90), (200, 0.94)):
        assert calibration_sample_size(target) <= n // 4
        assert certificate_warning(n, target) is None, (n, target)
    for n, target in ((400, 0.99), (400, 0.98), (200, 0.98), (50, 0.90)):
        w = certificate_warning(n, target)
        assert str(calibration_sample_size(target)) in w, (n, target)
    assert certificate_warning(400, 0.99, task="ordinal") is None
    out = str(tmp_path / "s.jsonl")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = B.write_sample(corpus500, out, "text", target=0.99)
    assert "299" in rep["warning"]
    assert any(rep["warning"] in r.getMessage() for r in caplog.records)
    assert len(_read(out)) == 400


def test_a_whole_corpus_sample_gives_no_certificate_warning(tmp_path, caplog):
    assert certificate_warning(150, 0.99) is not None
    out = str(tmp_path / "s.jsonl")
    c = _write(tmp_path / "small.jsonl", _corpus(150))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = B.write_sample(c, out, "text", target=0.99, corpus_rows=150)
    assert rep["whole_corpus"] is True
    assert rep["corpus_rows"] == 150
    assert "warning" not in rep
    assert not [r for r in caplog.records if r.name == _LOGGER]
    assert len(_read(out)) == 150
    # the same 150-row input with no corpus_rows is a sample of a larger corpus: it warns
    rep = B.write_sample(c, out, "text", target=0.99)
    assert rep["whole_corpus"] is False
    assert rep["corpus_rows"] == 150  # best known: the input's rows
    assert "299" in rep["warning"]
    # so does a 150-row sample of a 10,000-row corpus
    rep = B.write_sample(c, out, "text", target=0.99, corpus_rows=10000)
    assert (rep["whole_corpus"], rep["corpus_rows"]) == (False, 10000)
    assert "299" in rep["warning"]
    # a sample from a larger input still warns
    c = _write(tmp_path / "big.jsonl", _corpus(500))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = B.write_sample(c, out, "text", target=0.99)
    assert rep["whole_corpus"] is False
    assert "299" in rep["warning"]
    assert [r for r in caplog.records if r.name == _LOGGER]


def test_a_target_a_refined_spec_cannot_reach_reports_both_requirements():
    assert calibration_sample_size(0.98) == 149
    w = certificate_warning(400, 0.98)
    assert w is not None
    for part in (
        "at least 149 rows",
        "about 200 rows in val and test",
        "about 100 in test alone",
        "an unchanged spec can reach it only if every val and test row is right",
        "a refined spec cannot",
    ):
        assert part in w
    w = certificate_warning(200, 0.98)
    assert "even an unchanged spec cannot reach it" in w


@pytest.mark.parametrize("bad", ["0", "1", "1.5", "-0.1", "nan", "high", 0, 1.0])
def test_rejects_a_target_outside_0_1(tmp_path, bad):
    c = _write(tmp_path / "c.jsonl", _corpus(10))
    out = str(tmp_path / "s.jsonl")
    with pytest.raises(DataPreparerOperatorError):
        B.write_sample(c, out, "text", target=bad)
    assert not os.path.exists(out)


def test_refuses_a_target_below_085(tmp_path):
    c = _write(tmp_path / "c.jsonl", _corpus(10))
    out = str(tmp_path / "s.jsonl")
    with pytest.raises(
        DataPreparerOperatorError, match="certification needs a target of at least 0.85"
    ):
        B.write_sample(c, out, "text", target=0.84)
    assert not os.path.exists(out)
    assert B.write_sample(c, out, "text", target="0.85")["target"] == 0.85


# --- sample: the draw --------------------------------------------------------------------


def test_draw_is_the_first_n_of_a_seeded_shuffle_in_corpus_order():
    rows = _corpus(1000)
    seed = "exp-coldstart-v1|demo|draw3"
    sample, _ = draw_sample(rows, 200, seed)
    perm = list(range(len(rows)))
    random.Random(seed).shuffle(perm)
    assert [r["id"] for r in sample] == [rows[p]["id"] for p in sorted(perm[:200])]


def test_same_seed_same_sample_other_seed_other_sample():
    rows = _corpus(1000)
    a, _ = draw_sample(rows, 200, "s1")
    b, _ = draw_sample(rows, 200, "s1")
    c, _ = draw_sample(rows, 200, "s2")
    assert a == b
    assert [r["id"] for r in a] != [r["id"] for r in c]


def test_the_200_row_sample_is_part_of_the_400_row_one():
    rows = _corpus(5000)
    for k in range(5):
        seed = f"draw{k}"
        small = {r["id"] for r in draw_sample(rows, 200, seed)[0]}
        big = {r["id"] for r in draw_sample(rows, 400, seed)[0]}
        assert len(small) == 200
        assert small < big, seed


def test_input_smaller_than_the_sample_is_kept_whole_in_order():
    rows = _corpus(150)
    for target in (0.90, 0.95):
        n = cold_start_sample_size(target, len(rows))
        sample, rep = draw_sample(rows, n, "0")
        assert [r["id"] for r in sample] == [r["id"] for r in rows]
        assert rep["input_rows"] == 150
        assert rep["whole_corpus"] is False  # whole_corpus needs the corpus's row count
        assert draw_sample(rows, n, "0", corpus_total=150)[1]["whole_corpus"] is True
        assert draw_sample(rows, n, "0", corpus_total=151)[1]["whole_corpus"] is False
    _, rep = draw_sample(_corpus(300), 200, "0", corpus_total=300)
    assert rep["whole_corpus"] is False


def test_every_row_is_equally_likely():
    rows = _corpus(40)
    hits = [0] * len(rows)
    for k in range(2000):
        for r in draw_sample(rows, 10, f"u{k}")[0]:
            hits[int(r["id"][1:])] += 1
    # 2000 draws of 10 of 40: each row is drawn 500 times on average
    assert all(400 < h < 600 for h in hits), hits


# --- sample: the fields ------------------------------------------------------------------


def test_only_the_id_and_the_input_fields_reach_the_sample():
    rows = _corpus(30)
    rows[3]["_model_output"] = {"label": "a", "confidence_score": 0.9}
    sample, _ = draw_sample(rows, 30, "0", text_fields=["text"])
    assert len(sample) == 30
    for r in sample:
        assert list(r) == ["id", "text"]


def test_fields_keep_their_order_and_a_missing_one_is_left_out():
    rows = [
        {"id": "q1", "context": "c1", "question": "q1?", "answer": "x"},
        {"id": "q2", "question": "q2?", "gt_answers": ["y"]},
    ]
    sample, _ = draw_sample(rows, 2, "0", text_fields=["question", "context"])
    assert sample == [
        {"id": "q1", "question": "q1?", "context": "c1"},
        {"id": "q2", "question": "q2?"},
    ]


def test_no_row_with_any_input_field_is_refused():
    with pytest.raises(DataPreparerOperatorError, match="body"):
        draw_sample(_corpus(5), 5, "0", text_fields=["body"])


@pytest.mark.parametrize("bad", [{}, {"text": None}, {"text": "  "}])
def test_a_sampled_row_with_no_input_text_is_refused(bad):
    rows = _corpus(5)
    rows[2] = dict({"id": "r2"}, **bad)
    with pytest.raises(DataPreparerOperatorError, match="'r2'"):
        draw_sample(rows, 5, "0", text_fields=["text"])


def test_a_zero_is_input_text_and_a_missing_one_writes_nothing(tmp_path):
    rows = _corpus(5)
    rows[2]["text"] = 0  # 0 renders into the prompt: content
    assert len(draw_sample(rows, 5, "0", text_fields=["text"])[0]) == 5
    rows = _corpus(5)
    del rows[2]["text"]
    c = _write(tmp_path / "c.jsonl", rows)
    out = str(tmp_path / "s.jsonl")
    with pytest.raises(DataPreparerOperatorError):
        B.write_sample(c, out, "text")
    assert not os.path.exists(out)


def test_text_fields(tmp_path):
    rows = [{"id": "q1", "question": "q?", "context": "c", "label": "z"}]
    c = _write(tmp_path / "c.jsonl", rows)
    out = str(tmp_path / "s.jsonl")
    rep = B.write_sample(c, out, " question , context ,")
    assert rep["text_fields"] == ["question", "context"]
    assert _read(out) == [{"id": "q1", "question": "q?", "context": "c"}]
    assert B.write_sample(c, out, ["question"])["text_fields"] == ["question"]
    os.remove(out)
    for bad in (",", "body", []):
        with pytest.raises(DataPreparerOperatorError):
            B.write_sample(c, out, bad)
        assert not os.path.exists(out), bad


# --- sample: the ids ---------------------------------------------------------------------


def test_two_rows_with_one_id_are_refused():
    rows = _corpus(10)
    rows[7]["id"] = rows[2]["id"]
    with pytest.raises(DataPreparerOperatorError, match="r0002"):
        draw_sample(rows, 5, "0")


def test_ids_compare_as_strings_as_the_fold_joins_them():
    with pytest.raises(DataPreparerOperatorError):
        draw_sample([{"id": 1, "text": "a"}, {"id": "1", "text": "b"}], 2, "0")


def test_duplicate_ids_write_nothing(tmp_path):
    rows = _corpus(10)
    rows[9]["id"] = rows[0]["id"]
    c = _write(tmp_path / "c.jsonl", rows)
    out = str(tmp_path / "s.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="r0000"):
        B.write_sample(c, out, "text")
    assert not os.path.exists(out)


def test_a_row_without_an_id_gets_its_corpus_position():
    rows = [{"text": f"t{i}"} for i in range(5)]
    rows[2]["id"] = "keep"
    sample, rep = draw_sample(rows, 5, "0", corpus_total=5)
    assert [r["id"] for r in sample] == ["0", "1", "keep", "3", "4"]
    assert rep["ids_filled"] == 4


def test_a_null_id_gets_its_corpus_position():
    rows = [{"id": f"r{i}", "text": f"t{i}"} for i in range(5)]
    rows[3]["id"] = None
    sample, rep = draw_sample(rows, 5, "0", corpus_total=5)
    assert [r["id"] for r in sample] == ["r0", "r1", "r2", "3", "r4"]
    assert rep["ids_filled"] == 1


@pytest.mark.parametrize("total", [None, 500])
def test_an_id_less_row_in_a_sample_of_a_larger_corpus_is_refused(total):
    rows = [{"id": f"r{i}", "text": f"t{i}"} for i in range(5)]
    rows[3] = {"text": "t3"}
    with pytest.raises(DataPreparerOperatorError) as ei:
        draw_sample(rows, 5, "0", corpus_total=total)
    assert "input row 3 has no id" in str(ei.value)
    assert "--corpus-rows" in str(ei.value)


def test_a_missing_non_default_id_field_is_refused():
    rows = [{"key": "k0", "text": "a"}, {"text": "b"}]
    with pytest.raises(DataPreparerOperatorError, match="input row 1"):
        draw_sample(rows, 2, "0", id_field="key")
    sample, _ = draw_sample(
        [{"key": "k0", "text": "a"}, {"key": "k1", "text": "b"}], 2, "0", id_field="key"
    )
    assert sample == [{"key": "k0", "text": "a"}, {"key": "k1", "text": "b"}]


def test_position_counts_rows_not_blank_lines(tmp_path):
    c = str(tmp_path / "c.jsonl")
    with open(c, "w", encoding="utf-8") as f:
        f.write('{"text": "a"}\n\n{"text": "b"}\n   \n{"text": "c"}\n')
    out = str(tmp_path / "s.jsonl")
    rep = B.write_sample(c, out, "text", corpus_rows=3)
    assert [r["id"] for r in _read(out)] == ["0", "1", "2"]
    assert rep["corpus_rows"] == 3


def test_a_filled_id_that_another_row_already_has_is_refused():
    with pytest.raises(DataPreparerOperatorError):
        draw_sample([{"text": "a"}, {"id": "0", "text": "b"}], 2, "0", corpus_total=2)


# --- sample: the writer ------------------------------------------------------------------


def test_writes_the_sample_and_reports_it(tmp_path):
    rows = _corpus(500)
    c = _write(tmp_path / "c.jsonl", rows)
    out = str(tmp_path / "s.jsonl")
    rep = B.write_sample(c, out, "text", target=0.95, seed="abc")
    assert rep == {
        "mode": "sample",
        "input_rows": 500,
        "corpus_rows": 500,
        "sample_rows": 400,
        "whole_corpus": False,
        "seed": "abc",
        "id_field": "id",
        "text_fields": ["text"],
        "ids_filled": 0,
        "target": 0.95,
    }
    assert _read(out) == draw_sample(rows, 400, "abc")[0]


def test_cold_start_sample_is_what_write_sample_writes(tmp_path, corpus500):
    out = str(tmp_path / "s.jsonl")
    rep = B.write_sample(corpus500, out, "text", target=0.95, task="qa", seed="z")
    sample, pure = B.cold_start_sample(_read(corpus500), "text", target=0.95, task="qa", seed="z")
    assert (_read(out), rep) == (sample, pure)


def test_same_corpus_and_seed_give_the_same_bytes(tmp_path):
    c = _write(tmp_path / "c.jsonl", _corpus(800))
    outs = []
    for k in range(2):
        out = str(tmp_path / f"s{k}.jsonl")
        B.write_sample(c, out, "text")
        with open(out, "rb") as f:
            outs.append(f.read())
    assert outs[0] == outs[1]


def test_an_input_above_the_row_limit_is_refused(tmp_path):
    assert MAX_LABELING_ROWS == 50_000
    c = str(tmp_path / "c.jsonl")
    with open(c, "w", encoding="utf-8") as f:
        for i in range(MAX_LABELING_ROWS + 1):
            f.write(json.dumps({"text": str(i)}) + "\n")
    out = str(tmp_path / "s.jsonl")
    with pytest.raises(DataPreparerOperatorError) as ei:
        B.write_sample(c, out, "text")
    assert f"{MAX_LABELING_ROWS + 1:,} rows" in str(ei.value)
    assert f"labels at most {MAX_LABELING_ROWS:,}" in str(ei.value)
    assert not os.path.exists(out)
    with open(c, "w", encoding="utf-8") as f:  # at the limit: drawn
        for i in range(MAX_LABELING_ROWS):
            f.write(json.dumps({"id": f"r{i}", "text": str(i)}) + "\n")
    B.write_sample(c, out, "text")
    assert len(_read(out)) == 200


def test_an_unreadable_input_is_refused(tmp_path):
    out = str(tmp_path / "s.jsonl")
    empty = _write(tmp_path / "empty.jsonl", [])
    not_obj = _write(tmp_path / "list.jsonl", [["a", "b"]])
    broken = str(tmp_path / "broken.jsonl")
    with open(broken, "w", encoding="utf-8") as f:
        f.write('{"id": "1", "text": "a"}\n{"id": \n')
    for c in (empty, not_obj, broken, str(tmp_path / "missing.jsonl")):
        with pytest.raises(DataPreparerOperatorError):
            B.write_sample(c, out, "text")
        assert not os.path.exists(out), c


# --- sample and merge: an s3:// input ----------------------------------------------------


class _FakeS3:
    """Stub boto3 S3 client: get_object streams ``data`` (no network)."""

    def __init__(self, data):
        self.data, self.calls = data, []

    def get_object(self, Bucket, Key, Range=None):
        self.calls.append((Bucket, Key))
        data = self.data

        class _Body:
            def iter_chunks(self, chunk_size=None):
                for i in range(0, len(data), 7):
                    yield data[i : i + 7]

            def close(self):
                pass

        return {"Body": _Body()}


def test_s3_uri_draws_what_a_local_file_draws(tmp_path):
    rows = _corpus(300)
    blob = ("".join(json.dumps(r) + "\n" for r in rows)).encode()
    local = _write(tmp_path / "c.jsonl", rows)
    outs = {}
    for name, src in (("local", local), ("s3", "s3://bucket/corpus.jsonl")):
        fake = _FakeS3(blob)
        out = str(tmp_path / f"{name}.jsonl")
        with mock.patch("boto3.client", lambda *a, **k: fake):
            rep = B.write_sample(src, out, "text", seed="s")
        assert rep["input_rows"] == 300
        assert fake.calls == ([] if name == "local" else [("bucket", "corpus.jsonl")])
        with open(out, "rb") as f:
            outs[name] = f.read()
    assert outs["s3"] == outs["local"]


def test_an_s3_input_over_the_client_read_limit_is_refused_unread(tmp_path):
    read = []

    class _Big:
        def get_object(self, **_k):
            class _Body:
                def iter_chunks(self, chunk_size=None):
                    read.append(True)
                    yield b""

                def close(self):
                    pass

            return {"Body": _Body(), "ContentLength": CLIENT_READ_MAX_BYTES + 1}

    out = str(tmp_path / "s.jsonl")
    with mock.patch("boto3.client", lambda *a, **k: _Big()):
        with pytest.raises(DataPreparerOperatorError, match="client-side read limit"):
            B.write_sample("s3://bucket/c.jsonl", out, "text")
    assert read == []
    assert not os.path.exists(out)


def test_a_local_input_over_the_client_read_limit_is_refused(tmp_path):
    src = _write(tmp_path / "c.jsonl", _corpus(3))
    out = str(tmp_path / "s.jsonl")
    with mock.patch(
        "sagemaker.ai_registry.data_preparer.auto_labeling.operators._rows.CLIENT_READ_MAX_BYTES",
        10,
    ):
        with pytest.raises(DataPreparerOperatorError, match="client-side read limit"):
            B.write_sample(src, out, "text")
    assert not os.path.exists(out)


def test_an_unreadable_s3_input_writes_nothing(tmp_path):
    class _Denied:
        def get_object(self, **_k):
            raise OSError("access denied")

    out = str(tmp_path / "s.jsonl")
    with mock.patch("boto3.client", lambda *a, **k: _Denied()):
        with pytest.raises(DataPreparerOperatorError, match="access denied"):
            B.write_sample("s3://bucket/c.jsonl", out, "text")
    assert not os.path.exists(out)


def test_an_s3_access_denied_writes_nothing(tmp_path):
    from botocore.exceptions import ClientError, NoCredentialsError

    denied = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject"
    )
    for err in (denied, NoCredentialsError()):

        class _Fails:
            def get_object(self, _err=err, **_k):
                raise _err

        out = str(tmp_path / f"s-{type(err).__name__}.jsonl")
        with mock.patch("boto3.client", lambda *a, **k: _Fails()):
            with pytest.raises(DataPreparerOperatorError):
                B.write_sample("s3://bucket/c.jsonl", out, "text")
        assert not os.path.exists(out)
    assert "AccessDenied" in str(denied)


def test_merge_of_unreadable_s3_labels_writes_nothing(tmp_path):
    from botocore.exceptions import ClientError

    denied = ClientError({"Error": {"Code": "NoSuchKey", "Message": "missing"}}, "GetObject")

    class _Fails:
        def get_object(self, **_k):
            raise denied

    seed = _write(tmp_path / "seed.jsonl", [])
    out = str(tmp_path / "labels.jsonl")
    with mock.patch("boto3.client", lambda *a, **k: _Fails()):
        with pytest.raises(DataPreparerOperatorError):
            B.write_merged_labels("s3://bucket/l.jsonl", seed, out)
    assert not os.path.exists(out)


# --- merge -------------------------------------------------------------------------------


def test_the_fold_keys_are_seed_metadata():
    # n_annotations and annotators are written by the fold on a seed row only, so the
    # merge's input comparison never reads them.
    assert {"n_annotations", "annotators"} <= _fields.SEED_METADATA_FIELDS


@pytest.fixture
def merge_files(tmp_path):
    labels = _write(
        tmp_path / "corpus.jsonl",
        [
            {"id": "a", "text": "x", "label": "pos", "confidence_score": 0.9},
            {"id": "b", "text": "y", "label": "neg", "confidence_score": 0.8},
        ],
    )
    seed = _write(
        tmp_path / "seed.jsonl",
        [{"id": "a", "text": "x", "gt_label": "neg", "provenance": "human_verified"}],
    )
    return tmp_path, labels, seed


def test_merge_puts_the_people_s_label_on_the_row(merge_files):
    d, labels, seed = merge_files
    out = str(d / "labels.jsonl")
    rep = B.write_merged_labels(labels, seed, out)
    assert (rep["mode"], rep["merged"], rep["label_changed"]) == ("merge", 1, 1)
    rows = {r["id"]: r for r in _read(out)}
    assert rows["a"]["label"] == "neg"
    assert rows["a"]["provenance"] == "human_verified"
    assert _fields.labeled_by_people(rows["a"])
    assert rows["a"]["_model_output"]["label"] == "pos"
    assert rows["b"]["label"] == "neg" and "_model_output" not in rows["b"]


def test_merge_fills_missing_ids(tmp_path):
    labels = _write(tmp_path / "c.jsonl", [{"text": "x"}, {"text": "y"}])
    seed = _write(
        tmp_path / "seed.jsonl",
        [{"id": "1", "text": "y", "gt_label": "neg", "provenance": "human_verified"}],
    )
    out = str(tmp_path / "labels.jsonl")
    rep = B.write_merged_labels(labels, seed, out)
    assert rep["ids_filled"] == 2
    assert [r["id"] for r in _read(out)] == ["0", "1"]


@pytest.mark.parametrize(
    "seed_row,match",
    [
        ({"id": "zz", "text": "x"}, "have no row in --labels"),
        ({"id": "a", "text": "changed"}, "differ on an input field"),
    ],
)
def test_merge_refuses_a_seed_that_does_not_name_the_corpus(merge_files, seed_row, match):
    d, labels, _ = merge_files
    seed = _write(d / "seed2.jsonl", [dict(seed_row, gt_label="neg", provenance="human_verified")])
    out = str(d / "labels.jsonl")
    with pytest.raises(B.SeedMergeRefused, match=match):
        B.write_merged_labels(labels, seed, out)
    assert not os.path.exists(out)


def test_merge_refuses_a_seed_with_no_human_row(merge_files):
    d, labels, _ = merge_files
    seed = _write(
        d / "seed2.jsonl", [{"id": "a", "gt_label": "neg", "provenance": "auto_approved"}]
    )
    out = str(d / "labels.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="no label to merge") as ei:
        B.write_merged_labels(labels, seed, out)
    assert not isinstance(ei.value, B.SeedMergeRefused)
    assert not os.path.exists(out)


def test_merge_refuses_run_rows_with_another_task_s_prediction(tmp_path):
    labels = _write(tmp_path / "c.jsonl", [{"id": "a", "text": "x", "score": 3, "reasoning": "r"}])
    seed = _write(
        tmp_path / "seed.jsonl",
        [{"id": "a", "text": "x", "gt_label": 4, "provenance": "human_verified"}],
    )
    out = str(tmp_path / "labels.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="pass --pred-field 'label'"):
        B.write_merged_labels(labels, seed, out)
    assert not os.path.exists(out)
    # named, the reasoning column of a raw corpus is not a run marker
    raw = _write(tmp_path / "raw.jsonl", [{"id": "a", "text": "x", "reasoning": "r"}])
    B.write_merged_labels(raw, seed, out, pred_field="score", gt_field="gt_label")
    assert _read(out)[0]["score"] == 4


def _review_files(d):
    review = _write(
        d / "review.jsonl",
        [
            {"id": "a", "text": "x", "split": "spot_check", "_review_reason": "calibration"},
            {"id": "b", "text": "y", "split": "low_confidence"},
        ],
    )
    side = _write(
        d / "review.model_output.jsonl",
        [
            {"id": "a", "pred_field": "label", "model_output": {"label": "pos", "routed": False}},
            {"id": "b", "pred_field": "label", "model_output": {"label": "neg", "routed": True}},
        ],
    )
    return review, side


def test_merge_drops_the_people_s_rows_from_the_review_and_scores_the_spot_check(
    merge_files, caplog
):
    d, labels, seed = merge_files
    review, side = _review_files(d)
    out = str(d / "labels.jsonl")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = B.write_merged_labels(labels, seed, out, review_path=review)
    assert rep["review_rows_dropped"] == 1 and rep["review_rows"] == 1
    assert rep["spot_check_known"] == {"rows": 1, "corrected": 1}
    assert [r["id"] for r in _read(review)] == ["b"]
    assert [r["id"] for r in _read(side)] == ["b"]
    with open(str(d / "review.known_spot_check.json"), encoding="utf-8") as fh:
        assert json.load(fh)["ids"] == ["a"]
    assert any("1 review row(s)" in r.getMessage() for r in caplog.records)


def test_a_side_file_alone_is_the_review(merge_files):
    d, labels, seed = merge_files
    review, side = _review_files(d)
    os.remove(review)
    out = str(d / "labels.jsonl")
    rep = B.write_merged_labels(labels, seed, out, review_path=side)
    assert (rep["review_rows_dropped"], rep["review_rows"]) == (1, 1)
    assert rep["spot_check_known"]["rows"] == 1
    assert [r["id"] for r in _read(side)] == ["b"]
    assert not os.path.exists(review)  # only the side file is rewritten


def test_an_unreadable_review_writes_nothing(merge_files):
    d, labels, seed = merge_files
    side = _write(d / "review.model_output.jsonl", [{"id": "a", "model_output": {}}])
    out = str(d / "labels.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="cannot read --review"):
        B.write_merged_labels(labels, seed, out, review_path=side)
    assert not os.path.exists(out)


# --- overlay -----------------------------------------------------------------------------


@pytest.fixture
def overlay_files(tmp_path):
    seed = _write(
        tmp_path / "seed.jsonl",
        [
            {"id": "a", "text": "x", "gt_label": "neg", "provenance": "human_verified"},
            {"id": "b", "text": "y", "gt_label": "pos", "provenance": "auto_approved"},
        ],
    )
    review_seed = _write(
        tmp_path / "review_seed.jsonl",
        [
            {
                "id": "a",
                "text": "x",
                "gt_label": "pos",
                "_bootstrap": "hil",
                "provenance": "spot_checked",
                "source_export": "hil.jsonl",
            }
        ],
    )
    dataset = str(tmp_path / "labeled_dataset.json")
    with open(dataset, "w", encoding="utf-8") as fh:
        json.dump({"dataset_arn": "arn:aws:sagemaker:x:1:dataset/d", "dataset_name": "d"}, fh)
    return tmp_path, seed, review_seed, dataset


def test_overlay_writes_the_rows_and_the_handoff(overlay_files):
    d, seed, review_seed, dataset = overlay_files
    out = str(d / "overlay.jsonl")
    rep = B.write_overlay([seed, review_seed], dataset, out)
    rows = _read(out)
    assert rows == [
        {"id": "a", "label": "pos", "provenance": "spot_checked", "_labeled_by": "people"}
    ]  # a later seed's row for an id wins; auto_approved is never overlaid
    assert rep["overlay_rows"] == 1
    assert rep["gold_fields"] == {seed: "gt_label", review_seed: "gt_label"}
    assert rep["handoff"] == str(d / "overlay.handoff.json")
    with open(rep["handoff"], encoding="utf-8") as fh:
        handoff = json.load(fh)
    assert handoff["Inputs"]["LabeledDatasetArn"] == "arn:aws:sagemaker:x:1:dataset/d"
    assert handoff["Inputs"]["OverlayRowCount"] == 1


def test_overlay_takes_one_seed_path(overlay_files):
    d, seed, _, dataset = overlay_files
    out = str(d / "o.jsonl")
    assert B.write_overlay(seed, dataset, out)["overlay_rows"] == 1
    assert os.path.exists(str(d / "o.handoff.json"))


def test_overlay_with_no_human_row_writes_nothing(overlay_files):
    d, _, _, dataset = overlay_files
    seed = _write(d / "s.jsonl", [{"id": "b", "gt_label": "pos", "provenance": "auto_approved"}])
    out = str(d / "overlay.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="no label to overlay"):
        B.write_overlay([seed], dataset, out)
    assert not os.path.exists(out)


def test_overlay_refuses_a_record_with_no_dataset_arn(overlay_files):
    d, seed, _, _ = overlay_files
    dataset = str(d / "bad.json")
    with open(dataset, "w", encoding="utf-8") as fh:
        json.dump({}, fh)
    out = str(d / "overlay.jsonl")
    with pytest.raises(DataPreparerOperatorError, match=r"bad\.json: .*dataset_arn"):
        B.write_overlay([seed], dataset, out)
    assert not os.path.exists(out)


def test_overlay_refuses_an_unreadable_record_or_a_seed_with_no_gold(overlay_files):
    d, seed, _, _ = overlay_files
    out = str(d / "overlay.jsonl")
    with pytest.raises(DataPreparerOperatorError):
        B.write_overlay([seed], str(d / "missing.json"), out)
    dataset = str(d / "ld.json")
    with open(dataset, "w", encoding="utf-8") as fh:
        json.dump({"dataset_arn": "arn:x"}, fh)
    nogold = _write(d / "nogold.jsonl", [{"id": "a", "verdict": "x"}])
    with pytest.raises(DataPreparerOperatorError, match="no row has a gold value"):
        B.write_overlay([nogold], dataset, out)
    assert not os.path.exists(out)


def test_overlay_leaves_the_seed_out_when_the_run_labeled_eval_rows(overlay_files, caplog):
    d, seed, review_seed, _ = overlay_files
    dataset = str(d / "ld.json")
    with open(dataset, "w", encoding="utf-8") as fh:
        json.dump({"dataset_arn": "arn:x", "eval_rows_labeled_for_scoring": 4}, fh)
    out = str(d / "overlay.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="pass the review fold"):
        B.write_overlay([seed], dataset, out)
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        rep = B.write_overlay([seed, review_seed], dataset, out)
    assert rep["seed_rows_left_out"] == 2 and rep["overlay_rows"] == 1
    assert any("2 seed row(s) are left out" in r.getMessage() for r in caplog.records)


def test_overlay_drops_the_people_s_rows_from_the_review(overlay_files):
    d, seed, _, dataset = overlay_files
    review, side = _review_files(d)
    out = str(d / "overlay.jsonl")
    rep = B.write_overlay([seed], dataset, out, review_path=review)
    assert rep["review_rows_dropped"] == 1
    assert rep["spot_check_known"] == {"rows": 1, "corrected": 1}
    assert [r["id"] for r in _read(review)] == ["b"]
    assert [r["id"] for r in _read(side)] == ["b"]


# --- the side-file helpers ---------------------------------------------------------------


def test_review_rows_from_side_file():
    side = [
        {"id": "a", "model_output": {"routed": True}},
        "junk",
        {"id": 7, "model_output": {"routed": False}},
    ]
    assert B.review_rows_from_side_file(side, id_field="rid") == [
        {"rid": "a", "split": "low_confidence", "_review_reason": "low confidence"},
        {"rid": 7, "split": "spot_check", "_review_reason": "calibration"},
    ]


@pytest.mark.parametrize(
    "side,match",
    [
        ([{"model_output": {"routed": True}}], "null or repeated id"),
        (
            [{"id": "1", "model_output": {"routed": True}}, {"id": 1, "model_output": {}}],
            "null or repeated id",
        ),
        ([{"id": "a", "model_output": {"routed": "yes"}}], "has no `routed`"),
        ([{"id": "a"}], "has no `routed`"),
    ],
)
def test_review_rows_from_side_file_refusals(side, match):
    with pytest.raises(DataPreparerOperatorError, match=match):
        B.review_rows_from_side_file(side)


def test_review_path_of():
    assert B.review_path_of("p/review.model_output.jsonl") == "p/review.jsonl"
    assert B.review_path_of("p/review.jsonl") == "p/review.jsonl"
