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
"""Unit tests for operators.auto_labeling.review_export: a human review's export, one label per row."""

from __future__ import absolute_import


from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import review_export as E

SPAN_A = [
    {"text": "Acme", "type": "ORG", "start": 0, "end": 4},
    {"text": "Bob", "type": "PER", "start": 5, "end": 8},
]
SPAN_B = [
    {"text": "Bob", "type": "PER", "start": 5, "end": 8},
    {"text": "Acme", "type": "ORG", "start": 0, "end": 4},
]


def _task(rid, *anns, data=None):
    return {"data": {"id": rid, "text": "t", **(data or {})}, "annotations": list(anns)}


def _ann(result, who=None, cancelled=False, key="completed_by"):
    ann = {"result": result}
    if who is not None:
        ann[key] = who
    if cancelled:
        ann["was_cancelled"] = True
    return ann


def _choice(name, *choices):
    return {"from_name": name, "type": "choices", "value": {"choices": list(choices)}}


def _span(name, text, start, end, *labels):
    return {
        "from_name": name,
        "type": "labels",
        "value": {"start": start, "end": end, "text": text, "labels": list(labels)},
    }


# ---- provenance and row readers --------------------------------------------------------------


def test_provenance_for_review():
    assert E.provenance_for_review({"_review_reason": "calibration"}) == "spot_checked"
    assert E.provenance_for_review({"_review_reason": "audit"}) == "spot_checked"
    assert E.provenance_for_review({"split": "spot_check"}) == "spot_checked"
    assert E.provenance_for_review({"_review_reason": "low_confidence"}) == "human_verified"
    assert E.provenance_for_review({}) == "human_verified"
    assert E.PROVENANCE_VALUES == ("human_verified", "spot_checked", "auto_approved")
    assert E.HUMAN_PROVENANCE == ("human_verified", "spot_checked")
    assert E.BOOTSTRAP_TO_PROVENANCE == {"self": "auto_approved", "hil": "human_verified"}


def test_is_unresolved_and_annotator_of():
    for v in ("unsure", " Skip ", "SKIPPED"):
        assert E.is_unresolved({"verdict": v}) and E.is_unresolved({"status": v})
    assert not E.is_unresolved({"status": "completed"}) and not E.is_unresolved({})
    assert E.annotator_of({"annotator": "a", "reviewer": "r"}) == "a"
    assert E.annotator_of({"reviewer_id": 7}) == 7
    assert E.annotator_of({"completed_by": {"id": 3}}) == {"id": 3}
    assert E.annotator_of({"annotator": "", "completed_by_profile": None}) is None
    assert E.annotator_of({}) is None


def test_returned_label():
    assert E.returned_label({"gt_label": 0, "label": "b"}) == 0
    assert E.returned_label({"gt_label": "", "label": "b"}) == ""
    assert E.returned_label({"label": "b"}) == "b"
    assert E.returned_label({"label": "  "}) is None
    assert E.returned_label({"label": "  "}, gt_field="gt_answers") == "  "
    assert E.returned_label({}) is None
    assert E.returned_label({"gt_entities": []}, "gt_entities", "label") == []


# ---- Label Studio tasks ------------------------------------------------------------------------


def test_is_label_studio_task():
    assert E.is_label_studio_task(_task("r1"))
    assert not E.is_label_studio_task({"id": "r1", "label": "a"})
    assert not E.is_label_studio_task(
        {"data": [], "annotations": []}
    ) and not E.is_label_studio_task("x")


def test_flatten_export_keeps_flat_rows_and_splits_tasks_per_annotation():
    flat = {"id": "f1", "label": "a"}
    task = _task(
        "r1",
        _ann(
            [
                _choice("label", "a"),
                _choice("tags", "x", "y"),
                {"from_name": "rating", "type": "rating", "value": {"rating": 4}},
                {"from_name": "notes", "type": "textarea", "value": {"text": ["n"]}},
                "junk",
                {"type": "choices", "value": {}},
            ],
            who={"email": "bob@x", "id": 7},
        ),
        _ann([_choice("label", "b")], who="alice", key="annotator"),
        _ann([], cancelled=True, who={"id": 9}),
        _ann([], who=3),
        "junk",
        data={"record_id": "rec1", "rid": "x"},
    )
    out = E.flatten_export([flat, task], "rid")
    assert out[0] is flat
    assert out[1] == {
        "rid": "x",
        "id": "r1",
        "record_id": "rec1",
        "annotator": "bob@x",
        "label": "a",
        "tags": ["x", "y"],
        "rating": 4,
        "notes": "n",
    }
    assert out[2] == {
        "rid": "x",
        "id": "r1",
        "record_id": "rec1",
        "annotator": "alice",
        "label": "b",
    }
    assert out[3] == {
        "rid": "x",
        "id": "r1",
        "record_id": "rec1",
        "annotator": 9,
        "status": "skipped",
    }
    assert out[4] == {"rid": "x", "id": "r1", "record_id": "rec1", "annotator": 3}
    assert len(out) == 5
    # a first value wins; the reviewer's username is used when there is no email
    dup = _task("r2", _ann([_choice("label", "a"), _choice("label", "b")], who={"username": "u"}))
    assert E.flatten_export([dup])[0] == {"id": "r2", "annotator": "u", "label": "a"}
    assert E.flatten_export([_task("r3", _ann([_choice("label")]))])[0] == {"id": "r3", "label": []}


def test_flatten_export_span_controls():
    tasks = [
        _task(
            "r1",
            _ann(
                [
                    _span("gt_entities", "Acme", 0, 4, "ORG", "LOC"),
                    _span("gt_entities", "Bob", 5, 8, "PER"),
                ]
            ),
        ),
        _task(
            "r2", _ann([_choice("label", "a")])
        ),  # no span marked: the span control is [] (an answer)
    ]
    out = E.flatten_export(tasks)
    assert out[0]["gt_entities"] == [
        {"text": "Acme", "type": "ORG", "start": 0, "end": 4},
        {"text": "Acme", "type": "LOC", "start": 0, "end": 4},
        {"text": "Bob", "type": "PER", "start": 5, "end": 8},
    ]
    assert out[1] == {"id": "r2", "label": "a", "gt_entities": []}
    # no annotation marked a span: the named span fields no annotation answered otherwise are span controls
    none = [_task("r1", _ann([_choice("label", "a")])), _task("r2", _ann([]))]
    assert (
        E.span_fields("gt_entities", "label") == ("label",)
        and E.span_fields("gt_label", "label") == ()
    )
    assert E.flatten_export(none, span_fields=("entities",)) == [
        {"id": "r1", "entities": [], "label": "a"},
        {"id": "r2", "entities": []},
    ]
    assert (
        "label" not in E.flatten_export(none, span_fields=("label",))[1]
    )  # answered as a choice elsewhere


# ---- labels ------------------------------------------------------------------------------------


def test_label_comparisons():
    assert E.same_label("a", "a") and not E.same_label("a", "A")
    assert E.same_label(SPAN_A, SPAN_B) and E.same_label({"k": [2, 1]}, {"k": [1, 2]})
    assert E.same_label(["a", "b"], ["b", "a"]) and not E.same_label(["a"], ["a", "a"])
    assert not E.same_label(1, "1") and not E.same_label(object(), 1)
    assert _digest.canonical({"k": [{"b": 2}, {"a": 1}]}) == {"k": [{"a": 1}, {"b": 2}]}
    assert _digest.canonical((2, 1)) == [1, 2] and _digest.canonical("x") == "x"
    assert E.is_span_list(SPAN_A) and not E.is_span_list([]) and not E.is_span_list(["a"])
    assert E.is_span_list([{"type": "PER"}]) and not E.is_span_list([{"x": 1}])
    # spans compare as (text, type) sets: offsets do not matter, a scalar against a span list does not
    assert E.same_answer(SPAN_A, [dict(s, start=99) for s in SPAN_B])
    assert not E.same_answer(SPAN_A, SPAN_A[:1]) and not E.same_answer(SPAN_A, "Acme")
    assert E.same_answer("a", "a") and not E.same_answer("a", "b")
    assert (
        E.class_key("a") == "a"
        and E.class_key(["a"]) == '["a"]'
        and E.class_key({"b": 1, "a": 2}) == '{"a": 2, "b": 1}'
    )
    assert E.id_key(None) is None and E.id_key(3) == "3"


def test_decided_by_id():
    rows = [
        {"id": "r1", "label": "a", "annotator": "alice"},
        {"id": "r1", "label": "a", "annotator": "bob"},  # agrees: folded once, both named
        {"id": "r1", "label": "a", "annotator": "alice"},  # a reviewer once
        {"id": "r2", "label": "a"},
        {"id": "r2", "label": "b", "annotator": "carl"},  # disagrees: dropped
        {"id": "r3", "status": "unsure", "label": "a"},
        {"id": "r4", "label": "  "},  # blank: no label
        {"id": "r5", "text": "nothing"},  # neither column
        {"id": "r6", "gt_label": 0, "label": "z"},  # a falsy gold wins
        {"id": "r7", "gt_entities": SPAN_A, "annotator": "a"},
        {"id": "r7", "gt_entities": SPAN_B, "annotator": "b"},  # the same spans in another order
        {"id": 8, "label": "c"},
        {"text": "no id"},
        "junk",
        {"id": "r3", "label": "b"},  # r3 decided after all: not unresolved
    ]
    labels, annotators, counts, unresolved, conflicts, no_label, empty = E.decided_by_id(
        rows, "id", "gt_label", "label"
    )
    assert labels == {"r1": "a", "r3": "b", "r6": 0, "8": "c"}
    assert dict(annotators) == {"r1": ["alice", "bob"], "r2": ["carl"]}
    assert dict(counts) == {"r1": 3, "r2": 2, "r6": 1, "8": 1, "r3": 1}
    # r7 carries only gt_entities: neither label column of this fold
    assert unresolved == set() and conflicts == {"r2"} and no_label == {"r5", "r7"} and empty == 1
    labels, _, _, unresolved, conflicts, _, _ = E.decided_by_id(rows, "id", "gt_entities", "label")
    assert labels["r7"] == SPAN_A and "r7" not in conflicts
    assert "r2" in conflicts
    assert E.decided_by_id([{"id": "r3", "verdict": "skip"}], "id", "gt_label", "label")[3] == {
        "r3"
    }


def test_conflict_hint():
    hint = E.conflict_hint({"r2", "r1"})
    assert hint.startswith(
        "2 id(s) carry disagreeing labels from more than one annotation (r1, r2); "
    )
    assert "maximum_annotations to 1" in hint
    many = E.conflict_hint({f"r{i:02d}" for i in range(12)})
    assert "r09, ..." in many and "r10" not in many
