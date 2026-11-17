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
"""Unit tests for operators.auto_labeling.cold_start: the cold-start sample."""

from __future__ import absolute_import

import random

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import cold_start as C


def test_cold_start_sample_size():
    assert C.cold_start_sample_size(0.9, 1000) == 200
    assert C.cold_start_sample_size(0.95, 1000) == 400
    assert C.cold_start_sample_size(0.99, 1000, "classification") == 400
    assert (
        C.cold_start_sample_size(0.95, 1000, "ordinal") == 200
    )  # no certificate, no bigger sample
    assert C.cold_start_sample_size(0.95, 50) == 50 and C.cold_start_sample_size(0.9, 0) == 0
    assert C.cold_start_sample_size(0.9, -3) == 0


def test_certificate_warning():
    assert C.certificate_warning(200, 0.9) is None  # 29 rows needed, 50 in test alone
    assert C.certificate_warning(200, 0.95, "summarization") is None
    w = C.certificate_warning(200, 0.95)  # 59 needed; 100 in val+test, 50 in test
    assert w.startswith(
        "a 0.95 target needs at least 59 rows, every one right, for a certificate. This 200-row sample"
    )
    assert "about 100 rows in val and test" in w and "about 50 in test alone" in w
    assert "an unchanged spec can reach it only if every val and test row is right" in w
    assert w.endswith("and a refined spec cannot (about 0.941 at most).")
    w = C.certificate_warning(40, 0.95)  # 20 in val+test: even an unchanged spec cannot
    assert "even an unchanged spec cannot reach it (about 0.86 at most)" in w
    assert "(about 0.741 at most)." in w
    with pytest.raises(ValueError, match="at least 0.85"):
        C.certificate_warning(200, 0.5)


ROWS = [
    {"id": f"c{i}", "text": f"text {i}", "title": f"t{i}", "extra": "private"} for i in range(10)
]


def test_draw_sample_is_a_prefix_in_file_order_and_keeps_only_the_fields():
    small, rep = C.draw_sample(ROWS, 3, 7)
    large, _ = C.draw_sample(ROWS, 6, 7)
    assert [r["id"] for r in small] == [
        r["id"] for r in large if r["id"] in {s["id"] for s in small}
    ]
    assert [ROWS.index(next(x for x in ROWS if x["id"] == r["id"])) for r in large] == sorted(
        ROWS.index(next(x for x in ROWS if x["id"] == r["id"])) for r in large
    )
    order = list(range(10))
    random.Random(7).shuffle(order)
    assert {r["id"] for r in small} == {f"c{i}" for i in order[:3]}
    assert all(set(r) == {"id", "text"} for r in small)
    assert rep == {
        "mode": "sample",
        "input_rows": 10,
        "corpus_rows": 10,
        "sample_rows": 3,
        "whole_corpus": False,
        "seed": 7,
        "id_field": "id",
        "text_fields": ["text"],
        "ids_filled": 0,
    }
    both, _ = C.draw_sample(ROWS, 2, 1, text_fields=("text", "title"))
    assert all(set(r) == {"id", "text", "title"} for r in both)
    assert C.draw_sample(ROWS, 0, 1)[0] == [] and C.draw_sample(ROWS, -1, 1)[0] == []
    whole, rep = C.draw_sample(ROWS, 20, 1, corpus_total=10)
    assert len(whole) == 10 and rep["whole_corpus"] is True and rep["corpus_rows"] == 10
    assert C.draw_sample(ROWS, 10, 1, corpus_total=50)[1] == {
        **rep,
        "whole_corpus": False,
        "corpus_rows": 50,
    }


def test_draw_sample_fills_ids_only_for_the_whole_corpus():
    rows = [{"text": "a"}, {"id": "x", "text": "b"}, {"text": "c"}]
    sample, rep = C.draw_sample(rows, 3, 1, corpus_total=3)
    assert [r["id"] for r in sample] == ["0", "x", "2"] and rep["ids_filled"] == 2
    with pytest.raises(
        DataPreparerOperatorError,
        match="input row 0 has no id; the labeling job gives an id-less row",
    ):
        C.draw_sample(rows, 3, 1)
    with pytest.raises(
        DataPreparerOperatorError,
        match="input row 0 has no 'rid'; the labeling job fills only `id`",
    ):
        C.draw_sample(rows, 3, 1, id_field="rid", corpus_total=3)
    rid = [{"rid": 1, "text": "a"}, {"rid": 2, "text": "b"}]
    assert C.draw_sample(rid, 2, 1, id_field="rid")[0] == [
        {"rid": 1, "text": "a"},
        {"rid": 2, "text": "b"},
    ]


@pytest.mark.parametrize(
    "rows,kw,match",
    [
        ([], {}, "the input has no rows"),
        ([{"id": "a", "text": "t"}, "junk"], {}, "input row 1 is not a JSON object"),
        (
            [
                {"id": "a", "text": "t"},
                {"id": "a", "text": "u"},
                {"id": 1, "text": "v"},
                {"id": "1", "text": "w"},
            ],
            {},
            r"2 id\(s\) name more than one input row \('1', 'a'\); the labels join back by --id-field 'id'",
        ),
        (
            [{"id": "a", "body": "t"}],
            {},
            r"no input row has any of --text-fields \['text'\]; the first row has \['body', 'id'\]",
        ),
        (
            [
                {"id": "a", "text": "t"},
                {"id": "b", "text": "  "},
                {"id": "c", "text": None},
                {"id": "d"},
            ],
            {},
            r"3 sampled row\(s\) have no text in any of --text-fields \['text'\] \('b', 'c', 'd'\); "
            r"people would get nothing",
        ),
    ],
)
def test_draw_sample_refusals(rows, kw, match):
    with pytest.raises(DataPreparerOperatorError, match=match):
        C.draw_sample(rows, 10, 1, **kw)


def test_draw_sample_text_in_any_field_counts():
    rows = [{"id": "a", "text": "", "title": "ok"}, {"id": "b", "text": "fine"}]
    sample, _ = C.draw_sample(rows, 2, 1, text_fields=("text", "title"))
    assert sample == [{"id": "a", "text": "", "title": "ok"}, {"id": "b", "text": "fine"}]
