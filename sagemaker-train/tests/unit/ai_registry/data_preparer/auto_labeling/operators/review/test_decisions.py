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
"""Unit tests for operators.auto_labeling.label_decisions: the decisions on a contradiction list."""

from __future__ import absolute_import

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import decisions as D

LABELS = ["billing", "refund", "other"]
ROWS = [
    {"id": "r1", "text": "a", "gt_label": "billing"},
    {"id": "r2", "text": "b", "gt_label": "refund"},
    {"id": "r3", "text": "c", "gt_label": "billing"},
    {"id": "r4", "text": "d", "gt_label": "other"},
]
ITEMS = [
    {
        "item_id": "it1",
        "check": "contradiction",
        "member_ids": ["r1", "r2"],
        "proposed_changes": {"r1": "refund", "r9": "other"},
        "recheck_ids": ["r3"],
    },
    {"item_id": "it2", "check": "guideline", "member_ids": ["r3", "r4"], "proposed_changes": {}},
]
SPEC = "# Spec\n\nRules.\n"


def _apply(decisions, spec=SPEC, rows=ROWS, **kw):
    kw.setdefault("label_set", LABELS)
    kw.setdefault("today", "2026-01-01")
    return D.apply_decisions(ITEMS, decisions, rows, spec, **kw)


# ---- cells and labels ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "v,cell",
    [
        ("=1+1", "'=1+1"),
        ("-1", "-1"),
        ("-x", "'-x"),
        ("+2.5", "+2.5"),
        ("@a", "'@a"),
        ("\tx", "'\tx"),
        ("plain", "plain"),
        (None, ""),
        (3, "3"),
    ],
)
def test_csv_cell_escapes_formula_starts_and_plain_undoes_it(v, cell):
    assert D.csv_cell(v) == cell
    assert D._plain(cell) == ("" if v is None else str(v))
    assert D._plain("'quoted") == "'quoted"  # a ' before a non-formula is the user's own


@pytest.mark.parametrize(
    "v,want",
    [
        ("a", "a"),
        ("  a   b ", "a b"),
        (["a"], "a"),
        (["a", "b"], "a|b"),
        (None, None),
        ("", None),
        (3, "3"),
    ],
)
def test_canon(v, want):
    assert D.canon(v) == want


def test_item_sha_is_over_members_and_changes_only():
    a = D.item_sha(ITEMS[0])
    assert a == D.item_sha({**ITEMS[0], "title": "x", "member_ids": ["r2", "r1"]})
    assert a != D.item_sha({**ITEMS[0], "proposed_changes": {"r1": "other"}})
    assert a != D.item_sha({**ITEMS[0], "member_ids": ["r1"]})
    assert len(a) == 64 and D.item_sha({}) == D.item_sha({"member_ids": [], "proposed_changes": {}})


def test_parse_overrides():
    assert D.parse_overrides("r1=a;r2=keep") == {"r1": "a", "r2": "keep"}
    assert D.parse_overrides("r1 = a\n r2=b=c ; junk ; =x") == {"r1": "a", "r2": "b=c"}
    assert (
        D.parse_overrides({"r1": 1}) == {"r1": 1}
        and D.parse_overrides("") == D.parse_overrides(None) == {}
    )


def test_review_inputs():
    assert D.review_inputs("s", "e") == {"spec": "s", "seed": "e", "split": None}
    assert D.review_inputs("s", "e", "p")["split"] == "p"


def test_spec_names():
    assert D.spec_iteration("work/labeling_spec_iter_3.md") == 3
    assert D.spec_iteration("labeling_spec.md") == 0 and D.spec_iteration(None) == 0
    assert D.next_spec_name("work/labeling_spec_iter_3.md") == "labeling_spec_iter_4.md"
    assert D.next_spec_name("pre_iter_12.txt") == "pre_iter_13.txt"
    assert D.next_spec_name("labeling_spec.md") == "labeling_spec_iter_1.md"


# ---- CSV and the run log ------------------------------------------------------------------


CSV = (
    "item_id,title,decision,if_neither,sop_wording,overrides,item_sha256\n"
    "it1,'=t,Guidelines Correct,,,r1=refund;r2=keep,abc\n"
    "it2,t,,,,,def\n"
    ",x,neither,,,,\n"
    "it3,t,neither,'=label: other,'-Say so.,,ghi\n"
)


def test_decisions_from_csv(tmp_path):
    got = D.decisions_from_csv(CSV)
    assert got == {
        "it1": {
            "decision": "guidelines_correct",
            "if_neither": None,
            "sop_wording": None,
            "overrides": {"r1": "refund", "r2": "keep"},
            "item_sha256": "abc",
        },
        "it2": {
            "decision": None,
            "if_neither": None,
            "sop_wording": None,
            "overrides": {},
            "item_sha256": "def",
        },
        "it3": {
            "decision": "neither",
            "if_neither": "=label: other",
            "sop_wording": "-Say so.",
            "overrides": {},
            "item_sha256": "ghi",
        },
    }
    p = tmp_path / "decisions.csv"
    p.write_text(CSV, encoding="utf-8")
    assert D.read_decisions_csv(str(p)) == got


def _entry(item_id, status, sha, **fields):
    return {
        "kind": "decision",
        "step": D.RUN_LOG_STEP,
        "iteration": 0,
        "status": status,
        "item_id": item_id,
        "check": "c",
        "item_sha256": sha,
        **fields,
    }


def test_logged_decisions_latest_wins_and_filters_by_item_version():
    entries = [
        _entry("it1", "neither", "old", if_neither="x"),
        _entry(
            "it1", "label_correct", D.item_sha(ITEMS[0]), sop_wording="s", overrides={"r1": "keep"}
        ),
        _entry("it2", "guidelines_correct", "stale"),
        {"kind": "decision", "step": "other", "status": "neither", "item_id": "it9"},
        {"kind": "analysis", "step": D.RUN_LOG_STEP, "status": "flagged"},
        {"kind": "decision", "step": D.RUN_LOG_STEP, "status": "neither"},  # no item_id
    ]
    assert D.logged_decisions(entries) == {
        "it1": {
            "item_id": "it1",
            "check": "c",
            "if_neither": None,
            "sop_wording": "s",
            "decided_by": None,
            "item_sha256": D.item_sha(ITEMS[0]),
            "decision": "label_correct",
            "overrides": {"r1": "keep"},
        },
        "it2": {
            "item_id": "it2",
            "check": "c",
            "if_neither": None,
            "sop_wording": None,
            "decided_by": None,
            "item_sha256": "stale",
            "decision": "guidelines_correct",
            "overrides": {},
        },
    }
    assert list(D.logged_decisions(entries, ITEMS)) == ["it1"]
    assert D.logged_decisions([], ITEMS) == {}


def test_merge_decisions():
    known = D.logged_decisions(
        [_entry("it1", "label_correct", D.item_sha(ITEMS[0]), sop_wording="s")], ITEMS
    )
    csv_d = {
        "it2": {"decision": "neither", "if_neither": "other", "item_sha256": D.item_sha(ITEMS[1])}
    }
    merged = D.merge_decisions(known, csv_d, ITEMS)
    assert set(merged) == {"it1", "it2"} and merged["it1"]["decision"] == "label_correct"
    # a blank CSV row clears a logged decision
    assert D.merge_decisions(known, {"it1": {"decision": None, "item_sha256": "x"}}, ITEMS) == {}
    assert D.merge_decisions(known, None, ITEMS) == known
    with pytest.raises(
        DataPreparerOperatorError, match="'it2' was made on another version"
    ) as info:
        D.merge_decisions(
            known, {"it2": {"decision": "neither", "item_sha256": "old"}}, ITEMS, source="d.csv"
        )
    assert str(info.value).startswith("d.csv:") and "Nothing was written" in str(info.value)


def test_touched_test_rows_and_default_label_set():
    assert D.touched_test_rows(ITEMS, {"it1": {}}, {"r2", "r4"}) == ["r2"]
    assert D.touched_test_rows(ITEMS, {}, {"r2"}) == []
    spec = "## Label Set\n\n| Label | Definition |\n|---|---|\n| refund | money back |\n| billing | a bill |\n"
    assert D.default_label_set(spec, ROWS, "gt_label") == ["refund", "billing"]
    assert D.default_label_set(
        "# no set", ROWS + [{"gt_label": None}, {"gt_label": ""}], "gt_label"
    ) == ["billing", "other", "refund"]


# ---- apply_decisions ------------------------------------------------------------------------


def test_guidelines_correct_relabels_members_only():
    rows, spec, cl, led = _apply({"it1": {"decision": "guidelines_correct"}}, decided_by="v")
    assert rows[0]["gt_label"] == "refund" and rows[1:] == ROWS[1:]
    assert spec == SPEC
    assert cl == {
        "iteration": 1,
        "date": "2026-01-01",
        "decisions_applied": 1,
        "labels_changed": [{"id": "r1", "from": "billing", "to": "refund", "items": ["it1"]}],
        "sentences_added": [],
        "recheck": [],
        "needs_a_decision": [],
        "invalid_decisions": [],
    }
    assert led == [
        {
            "item_id": "it1",
            "check": "contradiction",
            "decision": "guidelines_correct",
            "if_neither": None,
            "sop_wording": None,
            "overrides": {},
            "decided_by": "v",
            "item_sha256": D.item_sha(ITEMS[0]),
        }
    ]


def test_label_correct_adds_the_sentence_once_and_rechecks():
    d = {"it1": {"decision": "label_correct", "sop_wording": "Say refund when money moves."}}
    rows, spec, cl, led = _apply(d, version=2)
    assert rows == ROWS
    assert spec == SPEC.rstrip() + (
        "\n\n## Clarifications from label review (iteration 2)\n\n"
        "- Say refund when money moves.  <!-- it1 -->\n"
    )
    assert cl["sentences_added"] == [("it1", "Say refund when money moves.")]
    assert cl["recheck"] == [{"item_id": "it1", "id": "r3"}]
    # a second sentence joins the existing block; one already in the spec is not repeated
    rows, spec2, cl2, _ = _apply(
        {**d, "it2": {"decision": "label_correct", "sop_wording": "Other."}}, spec=spec
    )
    assert spec2 == spec.rstrip() + "\n- Other.  <!-- it2 -->\n"
    assert cl2["sentences_added"] == [("it2", "Other.")]
    with_header = SPEC + "\n## Clarifications from label review\n"
    _, spec3, _, _ = _apply(d, spec=with_header)
    assert spec3 == with_header.rstrip() + "\n- Say refund when money moves.  <!-- it1 -->\n"


def test_label_correct_needs_its_sentence():
    rows, spec, cl, led = _apply({"it1": {"decision": "label_correct"}})
    assert led == [] and rows == ROWS and spec == SPEC
    assert cl["invalid_decisions"] == [
        "it1: label_correct needs the sentence for the spec in sop_wording"
    ]


def test_neither_sets_the_label_on_every_member():
    rows, _, cl, led = _apply({"it2": {"decision": "neither", "if_neither": "Label: refund"}})
    assert [r["gt_label"] for r in rows] == ["billing", "refund", "refund", "refund"]
    assert sorted(c["id"] for c in cl["labels_changed"]) == ["r3", "r4"]
    assert led[0]["if_neither"] == "Label: refund"
    # wording only is a valid neither; an empty one is not
    _, spec, cl, led = _apply({"it2": {"decision": "neither", "sop_wording": "Hmm."}})
    assert "- Hmm.  <!-- it2 -->" in spec and cl["labels_changed"] == [] and len(led) == 1
    _, _, cl, led = _apply({"it2": {"decision": "neither"}})
    assert led == [] and cl["invalid_decisions"] == [
        "it2: neither needs the right label in if_neither, a sentence in sop_wording or an override"
    ]


def test_labels_outside_the_label_set_are_invalid_and_not_recorded():
    _, _, cl, led = _apply({"it2": {"decision": "neither", "if_neither": "bogus"}})
    assert led == [] and cl["invalid_decisions"] == [
        "it2: if_neither 'bogus' is not a label of the label set"
    ]
    _, _, cl, _ = _apply({"it1": {"decision": "guidelines_correct"}}, label_set=["billing"])
    assert cl["invalid_decisions"] == [
        "it1: proposed label 'refund' for r1 is not in the label set"
    ]
    _, _, cl, _ = _apply({"it1": {"decision": "bogus"}})
    assert cl["invalid_decisions"] == [f"it1: decision 'bogus' not in {D.DECISIONS}"]
    _, _, cl, _ = _apply({"it2": {"decision": "neither", "overrides": "r3=nope"}})
    # an invalid override is not an override, so the neither has nothing left either
    assert cl["invalid_decisions"] == [
        "it2: override r3='nope' is not keep or a label of the label set",
        "it2: neither needs the right label in if_neither, a sentence in sop_wording or an override",
    ]


def test_overrides_win_keep_cancels_and_outside_rows_refuse():
    rows, _, cl, led = _apply(
        {"it1": {"decision": "guidelines_correct", "overrides": "r1=keep;r2=other"}}
    )
    assert [r["gt_label"] for r in rows] == ["billing", "other", "billing", "other"]
    assert led[0]["overrides"] == {"r1": "keep", "r2": "other"}
    # an empty override value is ignored; KEEP is keep
    _, _, _, led = _apply(
        {"it1": {"decision": "guidelines_correct", "overrides": {"r1": "KEEP", "r2": " "}}}
    )
    assert led[0]["overrides"] == {"r1": "keep"}
    with pytest.raises(
        D.OverrideOutsideItem, match=r"nothing was applied\): it1: r9; it2: r1"
    ) as info:
        _apply(
            {
                "it1": {"decision": "guidelines_correct", "overrides": "r9=other"},
                "it2": {"decision": "neither", "if_neither": "other", "overrides": "r1=billing"},
            }
        )
    assert isinstance(info.value, DataPreparerOperatorError)
    # an outside override with no value is not an override
    rows, _, _, _ = _apply({"it1": {"decision": "guidelines_correct", "overrides": "r9="}})
    assert rows[0]["gt_label"] == "refund"


def test_two_items_setting_one_row_differently_cancel():
    items = ITEMS + [
        {"item_id": "it3", "check": "c", "member_ids": ["r1"], "proposed_changes": {"r1": "other"}}
    ]
    rows, _, cl, led = D.apply_decisions(
        items,
        {"it1": {"decision": "guidelines_correct"}, "it3": {"decision": "guidelines_correct"}},
        ROWS,
        SPEC,
        label_set=LABELS,
        today="2026-01-01",
    )
    assert rows == ROWS and cl["labels_changed"] == []
    assert cl["needs_a_decision"] == [{"id": "r1", "items": {"it1": "refund", "it3": "other"}}]
    assert len(led) == 2
    # the same label from two items is one relabel naming both
    items[2]["proposed_changes"] = {"r1": "refund"}
    rows, _, cl, _ = D.apply_decisions(
        items,
        {"it1": {"decision": "guidelines_correct"}, "it3": {"decision": "guidelines_correct"}},
        ROWS,
        SPEC,
        label_set=LABELS,
    )
    assert cl["labels_changed"] == [
        {"id": "r1", "from": "billing", "to": "refund", "items": ["it1", "it3"]}
    ]


def test_undecided_items_and_other_id_fields():
    rows, spec, cl, led = _apply({"it1": {"decision": None}, "it2": {}})
    assert (rows, spec, led) == (ROWS, SPEC, []) and cl["decisions_applied"] == 0
    rows2 = [dict(r, rid=r["id"]) for r in ROWS]
    rows, _, cl, _ = D.apply_decisions(
        ITEMS,
        {"it1": {"decision": "guidelines_correct"}},
        rows2,
        SPEC,
        id_field="rid",
        label_set=LABELS,
    )
    assert rows[0]["gt_label"] == "refund"
    assert cl["date"] and len(cl["date"]) == 10  # today's date when none is given


# ---- entries and the dry run ---------------------------------------------------------------


def test_decision_events_skip_unchanged_decisions():
    ledger = [
        {
            "item_id": "it1",
            "check": "c",
            "decision": "label_correct",
            "if_neither": None,
            "sop_wording": "s",
            "overrides": {},
            "decided_by": None,
            "item_sha256": "a",
        },
        {
            "item_id": "it2",
            "check": "c",
            "decision": "neither",
            "if_neither": "x",
            "sop_wording": None,
            "overrides": {"r3": "keep"},
            "decided_by": "v",
            "item_sha256": "b",
        },
    ]
    known = {
        "it1": {
            "decision": "label_correct",
            "if_neither": None,
            "sop_wording": "s",
            "overrides": {},
            "item_sha256": "a",
        },
        "it2": {
            "decision": "neither",
            "if_neither": "y",
            "sop_wording": None,
            "overrides": {},
            "item_sha256": "b",
        },
    }
    assert D.decision_events(ledger, known, 3) == [
        {
            "kind": "decision",
            "step": "label-review",
            "iteration": 3,
            "status": "neither",
            "item_id": "it2",
            "check": "c",
            "if_neither": "x",
            "overrides": {"r3": "keep"},
            "decided_by": "v",
            "item_sha256": "b",
        }
    ]
    assert len(D.decision_events(ledger, {}, 0)) == 2


def test_dry_run_summary():
    _, _, cl, led = _apply({"it1": {"decision": "guidelines_correct"}})
    assert D.dry_run_summary(
        2, led, cl, ["s.md", "seed.jsonl"], split_files=["dev.jsonl"], split_copy="/c"
    ) == {
        "dry_run": True,
        "iteration": 2,
        "decisions": {"guidelines_correct": 1},
        "labels_changed": cl["labels_changed"],
        "sentences_added": 0,
        "needs_a_decision": 0,
        "invalid_decisions": [],
        "would_write": ["s.md", "seed.jsonl"],
        "split_files_overwritten": ["dev.jsonl"],
        "split_copy": "/c",
    }
    assert D.dry_run_summary(1, [], cl, [])["split_copy"] is None
