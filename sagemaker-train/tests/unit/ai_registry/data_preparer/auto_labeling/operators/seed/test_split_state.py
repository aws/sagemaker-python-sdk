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
"""seed.split_state: the project layer over seed.split writes the files and the run log."""

from __future__ import absolute_import

import json
import os

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as ss
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split_state as st


def _row(rid, text, label="a"):
    return {"id": rid, "text": text, "gt_label": label}


def _write(path, rows):
    with open(path, "wb") as fh:
        fh.write(ss.rows_bytes(rows))


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


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


def test_freeze_writes_the_state_and_logs_it(project):
    proj, splits, _rows = project
    out = st.freeze_split(splits, ["text"])
    assert out["rows"] == {"dev": 2, "val": 2, "test": 2}
    assert out["moved_near_duplicates"] == []
    assert os.path.isfile(ss.state_path(splits))
    assert st.verify_splits(splits)["ok"] is True
    kinds = [(e["kind"], e["status"]) for e in _run_log.read_entries(proj)]
    assert kinds == [("artifact", "frozen")]
    with pytest.raises(ss.SplitGuardError):
        st.freeze_split(splits, ["text"])  # once


def test_freeze_moves_near_duplicates_and_rewrites_the_files(project):
    proj, splits, rows = project
    long = "the quick brown fox jumps over the lazy dog near the river bank today"
    rows["dev"].append(_row("d3", long))
    rows["val"].append(_row("v3", long + " again"))
    for s in ("dev", "val"):
        _write(os.path.join(splits, f"{s}.jsonl"), rows[s])
    out = st.freeze_split(splits, ["text"], move_near_duplicates=True)
    assert len(out["moved_near_duplicates"]) == 1
    assert [r["id"] for r in _read(os.path.join(splits, "val.jsonl"))] == ["v1", "v2"]
    assert "d3" in {r["id"] for r in _read(os.path.join(splits, "dev.jsonl"))}
    assert {e["status"] for e in _run_log.read_entries(proj)} == {"frozen"}


def test_rehash_accepts_a_dev_edit_and_refuses_a_test_edit(project):
    proj, splits, rows = project
    st.freeze_split(splits, ["text"])
    rows["dev"][0]["gt_label"] = "b"
    _write(os.path.join(splits, "dev.jsonl"), rows["dev"])
    summary = st.rehash_split(splits, "fixed d1")
    assert summary["dev_gold_changed"] == 1
    assert [e["status"] for e in _run_log.read_entries(proj)] == ["frozen", "rehashed"]
    rows["test"][0]["gt_label"] = "b"
    _write(os.path.join(splits, "test.jsonl"), rows["test"])
    with pytest.raises(ss.SplitGuardError):
        st.rehash_split(splits, "oops")


def test_relabel_writes_dev_and_val_and_accepts_them(project):
    proj, splits, rows = project
    st.freeze_split(splits, ["text"])
    summary = st.relabel_split(splits, {"d1": "b", "v1": "b"}, "review")
    assert summary["dev_gold_changed"] == 1
    assert summary["val_gold_changed"] == 1
    assert _read(os.path.join(splits, "dev.jsonl"))[0]["gt_label"] == "b"
    assert _read(os.path.join(splits, "val.jsonl"))[0]["gt_label"] == "b"
    assert st.verify_splits(splits)["events"] == 2
    assert [e["status"] for e in _run_log.read_entries(proj)][-1] == "rehashed"
    assert rows  # unused beyond setup


def test_guard_read_logs_a_test_read_and_refuses_the_wrong_step(project):
    proj, splits, _rows = project
    st.freeze_split(splits, ["text"])
    test_file = os.path.join(splits, "test.jsonl")
    with pytest.raises(ss.SplitGuardError):
        st.guard_split_read(test_file, "refine", {"dev", "val"}, project_dir=proj)
    touched = st.guard_split_read(test_file, "report", {"dev", "val", "test"}, project_dir=proj)
    assert touched["test"] == 2
    entries = _run_log.read_entries(proj)
    assert entries[-1]["kind"] == "analysis" and entries[-1]["status"] == "test-read"
    # log=False: the caller writes it
    n = len(entries)
    st.guard_split_read(test_file, "report", {"test"}, project_dir=proj, log=False)
    assert len(_run_log.read_entries(proj)) == n
    st.log_test_read(proj, test_file, "report", 2)
    assert len(_run_log.read_entries(proj)) == n + 1


def test_guard_or_raise_finds_the_project_from_the_path(project):
    proj, splits, _rows = project
    st.freeze_split(splits, ["text"])
    assert (
        st.guard_split_read_or_raise(os.path.join(splits, "dev.jsonl"), "refine", {"dev"})["dev"]
        == 2
    )
    assert os.path.isfile(os.path.join(proj, "run-log.jsonl"))


def test_guard_read_without_a_frozen_split_is_not_guarded(tmp_path):
    f = tmp_path / "rows.jsonl"
    _write(str(f), [_row("x", "no split here")])
    assert st.guard_split_read(str(f), "refine", {"dev"}) is None
    assert not (tmp_path / "run-log.jsonl").exists()
