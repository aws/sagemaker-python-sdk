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
"""Unit tests for operators.auto_labeling.split_state: freeze, guard, rehash, relabel."""

from __future__ import absolute_import

import json
import os
import random

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as ss

LONG = "the quick brown fox jumps over the lazy dog near the river bank today"


def _row(rid, text, label="a"):
    return {"id": rid, "text": text, "gt_label": label}


def _write(path, rows):
    with open(path, "wb") as fh:
        fh.write(ss.rows_bytes(rows))


def _land(res, splits):
    """What a command-line caller does with a result: write rows (if any), then the state."""
    rows = getattr(res, "rows", None)
    if isinstance(res, ss.FreezeResult) and res.moved_near_duplicates:
        for s in ss.SPLITS:
            _write(os.path.join(splits, f"{s}.jsonl"), rows[s])
    elif isinstance(res, ss.RelabelResult):
        for s in ("dev", "val"):
            _write(os.path.join(splits, f"{s}.jsonl"), rows[s])
    with open(res.state_path, "w", encoding="utf-8") as fh:
        json.dump(res.state, fh, indent=1, ensure_ascii=False)
        fh.write("\n")


@pytest.fixture(autouse=True)
def _fresh_memo():
    ss._VERIFIED.clear()
    yield
    ss._VERIFIED.clear()


@pytest.fixture
def project(tmp_path):
    splits = tmp_path / "splits"
    splits.mkdir()
    rows = {
        "dev": [_row("d1", "billing question one"), _row("d2", "refund please now", "b")],
        "val": [_row("v1", "login trouble today"), _row("v2", "card declined again", "b")],
        "test": [_row("t1", "where is my parcel"), _row("t2", "close my account", "b")],
    }
    for s, rs in rows.items():
        _write(str(splits / f"{s}.jsonl"), rs)
    return str(tmp_path), str(splits), rows


def _freeze(splits, **kw):
    res = ss.freeze(splits, ["text"], **kw)
    _land(res, splits)
    return res


class TestFreeze:
    def test_records_ids_hashes_gold_and_an_event(self, project):
        root, splits, rows = project
        res = _freeze(splits)
        assert res.row_counts == {"dev": 2, "val": 2, "test": 2}
        st = ss.load_state(splits)
        assert st["splits"]["test"]["ids"] == ["t1", "t2"]
        assert st["splits"]["dev"]["sha256"] == _digest.sha256_file(
            os.path.join(splits, "dev.jsonl")
        )
        assert st["splits"]["val"]["gold"]["v2"] == ss.gold_hash("b")
        assert st["history"][0]["event"] == "freeze"
        assert res.events == [
            {
                "kind": "artifact",
                "step": "split",
                "iteration": 0,
                "status": "frozen",
                "file": os.path.join(splits, "split_state.json"),
                "rows": {"dev": 2, "val": 2, "test": 2},
                "moved": 0,
            }
        ]
        assert ss.verify(splits)["version"] == ss.STATE_VERSION

    def test_second_freeze_shared_id_and_bad_id_field_refused(self, project):
        root, splits, rows = project
        _freeze(splits)
        with pytest.raises(ss.SplitGuardError, match="the split is frozen; never"):
            ss.freeze(splits, ["text"])
        os.remove(ss.state_path(splits))
        _write(os.path.join(splits, "val.jsonl"), [_row("d1", "something else entirely")])
        with pytest.raises(ss.SplitGuardError, match="id 'd1' is in both dev and val"):
            ss.freeze(splits, ["text"])
        with pytest.raises(ss.SplitGuardError, match="keys rows on `id`"):
            ss.freeze(splits, ["text"], id_field="row_id")
        with pytest.raises(ss.SplitGuardError, match="names at least one input field"):
            ss.freeze(splits, [])

    def test_near_duplicate_refused_or_moved_to_dev(self, project):
        root, splits, rows = project
        _write(
            os.path.join(splits, "test.jsonl"), [_row("t1", LONG), _row("t2", "close my account")]
        )
        _write(
            os.path.join(splits, "val.jsonl"),
            [_row("v1", LONG + " please"), _row("v2", "card declined")],
        )
        with pytest.raises(
            ss.SplitGuardError, match="near-duplicate pair.*moves the val or test member into dev"
        ):
            ss.freeze(splits, ["text"])
        assert ss.load_state(splits) is None  # a refused freeze writes no state
        res = _freeze(splits, move_near_duplicates=True)
        assert res.moved_near_duplicates == [
            {"id": "v1", "from": "val"},
            {"id": "t1", "from": "test"},
        ]
        assert res.row_counts == {"dev": 4, "val": 1, "test": 1}
        assert res.events[0]["moved"] == 2
        # the state's hashes are over the rows the caller wrote
        assert ss.verify(splits)["splits"]["dev"]["ids"] == ["d1", "d2", "v1", "t1"]

    def test_freeze_needs_labels_texts_and_rows(self, project):
        root, splits, rows = project
        _write(os.path.join(splits, "dev.jsonl"), [_row("d1", "x", None)])
        with pytest.raises(ss.SplitGuardError, match="have no 'gt_label' or no text"):
            ss.freeze(splits, ["text"])
        _write(os.path.join(splits, "dev.jsonl"), [])
        with pytest.raises(ss.SplitGuardError, match="dev.jsonl is empty"):
            ss.freeze(splits, ["text"])
        os.remove(os.path.join(splits, "dev.jsonl"))
        with pytest.raises(ss.SplitGuardError, match="dev.jsonl is missing"):
            ss.freeze(splits, ["text"])

    def test_a_typo_in_text_fields_cannot_empty_test(self, project):
        root, splits, rows = project
        with pytest.raises(ss.SplitGuardError, match="no text in txt"):
            ss.freeze(splits, ["txt"], move_near_duplicates=True)


class TestNearDuplicates:
    def test_exact_and_word_overlap(self):
        texts = {
            "dev": {"a": LONG},
            "val": {"b": LONG + " please", "c": "short one"},
            "test": {"d": "short one"},
        }
        assert ss.near_duplicate_pairs(texts) == [
            ("dev", "a", "val", "b"),
            ("val", "c", "test", "d"),
        ]
        assert (
            ss.near_duplicate_pairs({"dev": {"a": "one two"}, "val": {"b": "one two three"}}) == []
        )

    def test_matches_an_all_pairs_check(self):
        rng = random.Random(3)
        words = [f"w{i}" for i in range(12)]
        for _ in range(100):
            texts = {
                s: {
                    f"{s}{i}": " ".join(rng.choice(words) for _ in range(rng.randint(1, 9)))
                    for i in range(6)
                }
                for s in ss.SPLITS
            }
            flat = [(s, i, t) for s in ss.SPLITS for i, t in texts[s].items()]
            expect = set()
            for x, (sa, ia, ta) in enumerate(flat):
                for sb, ib, tb in flat[x + 1 :]:
                    if sa == sb:
                        continue
                    wa, wb = frozenset(ta.split()), frozenset(tb.split())
                    near = ta == tb or (
                        len(wa) >= ss.NEAR_DUP_MIN_TOKENS
                        and len(wb) >= ss.NEAR_DUP_MIN_TOKENS
                        and len(wa & wb) / len(wa | wb) >= ss.NEAR_DUP_JACCARD
                    )
                    if near:
                        expect.add(
                            ss._ordered(sa, ia, sb, ib, {s: i for i, s in enumerate(ss.SPLITS)})
                        )
            assert set(ss.near_duplicate_pairs(texts)) == expect

    def test_row_text(self):
        assert (
            ss.row_text(
                {"text": "A  B", "meta": {"k": [1]}, "none": None}, ["text", "meta", "none"]
            )
            == 'a b {"k": [1]}'
        )


class TestGuardRead:
    def test_a_step_reads_only_its_splits(self, project):
        root, splits, rows = project
        _freeze(splits)
        dev = os.path.join(splits, "dev.jsonl")
        res = ss.guard_read(dev, "induce", {"dev"}, project_dir=root)
        assert res.guarded and res.touched == {"dev": 2} and res.events == []
        with pytest.raises(
            ss.SplitGuardError, match="holds 2 dev row\\(s\\); this step may read val only"
        ):
            ss.guard_read(dev, "recalibration-fit", {"val"}, project_dir=root)

    def test_only_test_steps_read_test_and_the_read_is_an_event(self, project):
        root, splits, rows = project
        _freeze(splits)
        test = os.path.join(splits, "test.jsonl")
        with pytest.raises(ss.SplitGuardError, match="step 'demos' may not read the test split"):
            ss.guard_read(test, "demos", {"test"}, project_dir=root)
        res = ss.guard_read(test, "certificate", {"test"}, project_dir=root)
        assert res.touched == {"test": 2}
        assert res.events == [
            {
                "kind": "analysis",
                "step": "certificate",
                "iteration": 0,
                "status": "test-read",
                "file": test,
                "rows": 2,
            }
        ]
        assert (
            ss.guard_read(
                test, "report", {"dev", "val", "test"}, project_dir=root, record_test_read=False
            ).events
            == []
        )

    def test_certificate_refuses_rows_that_are_not_test_rows(self, project, tmp_path):
        root, splits, rows = project
        _freeze(splits)
        work = str(tmp_path / "work.jsonl")
        _write(work, rows["test"] + [_row("zzz", "a stranger")])
        with pytest.raises(ss.SplitGuardError, match="1 row\\(s\\) that are not frozen test rows"):
            ss.guard_read(work, "certificate", {"test"}, project_dir=root)
        _write(work, rows["test"][:1])
        assert ss.guard_read(work, "certificate", {"test"}, project_dir=root).touched == {"test": 1}

    def test_near_copies_of_a_forbidden_split_and_rows_without_ids(self, project, tmp_path):
        root, splits, rows = project
        _write(
            os.path.join(splits, "test.jsonl"), [_row("t1", LONG), _row("t2", "close my account")]
        )
        _freeze(splits)
        work = str(tmp_path / "work.jsonl")
        _write(work, [_row("x1", LONG + " please")])
        with pytest.raises(ss.SplitGuardError, match="holds 1 test row\\(s\\)"):
            ss.guard_read(work, "induce", {"dev"}, project_dir=root)
        _write(work, [{"text": "no id", "gt_label": "a"}])
        with pytest.raises(ss.SplitGuardError, match="has rows with no 'id'"):
            ss.guard_read(work, "induce", {"dev"}, project_dir=root)

    def test_an_edited_split_stops_every_step_and_no_state_means_unguarded(self, project):
        root, splits, rows = project
        dev = os.path.join(splits, "dev.jsonl")
        res = ss.guard_read(dev, "induce", {"dev"}, project_dir=root)
        assert res.guarded is False and res.touched is None
        _freeze(splits)
        _write(dev, rows["dev"] + [_row("d3", "new row")])
        with pytest.raises(
            ss.SplitGuardError, match="dev.jsonl changed since the split was frozen.*rehash"
        ):
            ss.guard_read(dev, "induce", {"dev"}, project_dir=root)
        _write(dev, rows["dev"])  # dev restored; now test is edited
        _write(os.path.join(splits, "test.jsonl"), rows["test"][:1])
        with pytest.raises(ss.SplitGuardError, match="test is immutable"):
            ss.verify(splits)

    def test_rows_passed_in_and_helpers(self, project):
        root, splits, rows = project
        _freeze(splits)
        res = ss.guard_read(
            "anything.jsonl",
            "kfold",
            {"dev", "val"},
            project_dir=root,
            rows=rows["dev"] + rows["val"],
        )
        assert res.touched == {"dev": 2, "val": 2}
        assert ss.test_ids(project_dir=root) == {"t1", "t2"}
        assert ss.test_ids(project_dir=os.path.join(root, "nope")) is None
        assert ss.find_splits_dir(os.path.join(splits, "dev.jsonl")) == splits
        assert ss.nearest_project(os.path.join(root, "work", "x.jsonl")) == root
        assert ss.nearest_project("/") is None


class TestRehash:
    def test_label_fixes_and_new_dev_rows_are_accepted(self, project):
        root, splits, rows = project
        _freeze(splits)
        rows["dev"][0]["gt_label"] = "b"
        rows["dev"].append(_row("d9", "a brand new row"))
        _write(os.path.join(splits, "dev.jsonl"), rows["dev"])
        res = ss.rehash(splits, "label fixes")
        _land(res, splits)
        assert res.summary == {
            "rows": {"dev": 3, "val": 2, "test": 2},
            "reason": "label fixes",
            "dev_removed": 0,
            "dev_added": 1,
            "dev_gold_changed": 1,
            "val_removed": 0,
            "val_added": 0,
            "val_gold_changed": 0,
        }
        hist = ss.verify(splits)["history"][-1]
        assert (
            hist["event"] == "rehash"
            and hist["dev"]["gold_changed"] == ["d1"]
            and hist["dev"]["added"] == ["d9"]
        )
        assert res.events[0]["status"] == "rehashed" and res.events[0]["reason"] == "label fixes"

    def test_test_is_immutable_and_frozen_rows_keep_their_split(self, project):
        root, splits, rows = project
        _freeze(splits)
        _write(os.path.join(splits, "test.jsonl"), rows["test"][:1])
        with pytest.raises(ss.SplitGuardError, match="test.jsonl changed: test rows are immutable"):
            ss.rehash(splits, "x")
        _write(os.path.join(splits, "test.jsonl"), rows["test"])
        _write(os.path.join(splits, "dev.jsonl"), rows["dev"] + [rows["val"][0]])
        with pytest.raises(ss.SplitGuardError, match="id 'v1' is a val row appended to dev"):
            ss.rehash(splits, "x")
        _write(os.path.join(splits, "dev.jsonl"), rows["dev"] + [dict(rows["dev"][0])])
        with pytest.raises(ss.SplitGuardError, match="duplicate id in dev"):
            ss.rehash(splits, "x")
        _write(os.path.join(splits, "dev.jsonl"), rows["dev"])
        _write(os.path.join(splits, "val.jsonl"), rows["val"] + [_row("v9", "new in val")])
        with pytest.raises(ss.SplitGuardError, match="new in val; a frozen row keeps its split"):
            ss.rehash(splits, "x")

    def test_val_removal_needs_its_flag_and_a_reason_is_required(self, project):
        root, splits, rows = project
        _freeze(splits)
        with pytest.raises(ss.SplitGuardError, match="--reason names why"):
            ss.rehash(splits, "  ")
        _write(os.path.join(splits, "val.jsonl"), rows["val"][:1])
        with pytest.raises(ss.SplitGuardError, match="1 val row\\(s\\) removed"):
            ss.rehash(splits, "trim")
        res = ss.rehash(splits, "trim", allow_val_removal=True)
        assert res.summary["val_removed"] == 1
        _land(res, splits)
        assert ss.verify(splits)["splits"]["val"]["ids"] == ["v1"]

    def test_a_new_dev_row_that_copies_test_is_refused(self, project):
        root, splits, rows = project
        _write(
            os.path.join(splits, "test.jsonl"), [_row("t1", LONG), _row("t2", "close my account")]
        )
        _freeze(splits)
        _write(os.path.join(splits, "dev.jsonl"), rows["dev"] + [_row("d9", LONG + " please")])
        with pytest.raises(
            ss.SplitGuardError, match="test rows are immutable, so drop the new or edited"
        ):
            ss.rehash(splits, "x")

    def test_no_state_is_refused(self, project):
        root, splits, rows = project
        with pytest.raises(ss.SplitGuardError, match="freeze the split"):
            ss.rehash(splits, "x")


class TestRelabel:
    def test_relabel_returns_rows_and_state_that_verify(self, project):
        root, splits, rows = project
        _freeze(splits)
        res = ss.relabel(splits, {"d1": "b", "v2": "a", "t1": "zzz"}, "review")
        assert res.rows["dev"][0]["gt_label"] == "b" and res.rows["val"][1]["gt_label"] == "a"
        assert "test" not in res.rows  # test is never relabeled
        assert res.summary["dev_gold_changed"] == 1 and res.summary["val_gold_changed"] == 1
        # nothing was written yet: the old files still verify against the old state
        assert ss.verify(splits)["history"][-1]["event"] == "freeze"
        _land(res, splits)
        ss._VERIFIED.clear()
        assert ss.verify(splits)["splits"]["dev"]["gold"]["d1"] == ss.gold_hash("b")
        assert ss.read_rows(os.path.join(splits, "test.jsonl"))[0]["gt_label"] == "a"

    def test_a_refused_relabel_changes_nothing(self, project):
        root, splits, rows = project
        _freeze(splits)
        before = _digest.sha256_file(os.path.join(splits, "dev.jsonl"))
        with pytest.raises(ss.SplitGuardError, match="--reason"):
            ss.relabel(splits, {"d1": "b"}, "")
        with pytest.raises(ss.SplitGuardError, match="have no 'gt_label'"):
            ss.relabel(splits, {"d1": None}, "x")
        assert _digest.sha256_file(os.path.join(splits, "dev.jsonl")) == before
        assert ss.verify(splits)["history"][-1]["event"] == "freeze"


class TestBytes:
    def test_rows_bytes_roundtrip_and_hash(self, tmp_path):
        rows = [{"id": "1", "text": "é"}, {"id": "2", "x": [1, {"y": None}]}]
        p = str(tmp_path / "r.jsonl")
        _write(p, rows)
        assert ss.read_rows(p) == rows
        assert ss.sha256_rows(rows) == _digest.sha256_file(p)
        assert ss.rows_bytes([]) == b""


def test_command_line_named_helpers_are_not_collected():
    # test_read_event / test_ids keep the command line's names; pytest must not collect them
    assert ss.test_read_event.__test__ is False and ss.test_ids.__test__ is False
