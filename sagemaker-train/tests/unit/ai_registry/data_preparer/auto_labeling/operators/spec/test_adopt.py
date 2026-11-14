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
"""Unit tests for operators.auto_labeling.adopt_pick: adopting and shipping the in-job search's pick."""

from __future__ import absolute_import

import importlib
import hashlib
import json

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _digest
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

# the module (the subpackage attribute 'adopt' is the operator function)
A = importlib.import_module(
    "sagemaker.ai_registry.data_preparer.auto_labeling.operators.spec.adopt"
)

PICK = b"# the pick\n"
SHA = hashlib.sha256(PICK).hexdigest()
JOB = "autolabel-opt-1"


def _approval(aid="a1", sha=SHA, job=JOB, status="approved", step=A.STEP):
    return {
        "kind": "approval",
        "id": aid,
        "step": step,
        "iteration": 0,
        "status": status,
        "optimizer_pick_sha256": sha,
        "job_name": job,
    }


def test_adopted_ship():
    assert A.adopted_ship({"pick_val": "Round 0 | Candidate 0"}) == "initial"
    assert A.adopted_ship({"pick_val": "Round 1 | Candidate 0"}) == "refined"
    assert A.adopted_ship({}) == "refined" and A.adopted_ship(None) == "refined"
    assert A.adopted_ship("junk") == "refined"


def test_check_pick_folder():
    A.check_pick_folder(
        "/o/job/optimize/spec.md", {"spec.md", "optimizer_result.json", "pick.json", "x"}
    )
    with pytest.raises(A.AdoptError) as info:
        A.check_pick_folder("out/spec.md", {"spec.md", "pick.json"})
    assert str(info.value) == (
        "adopt_job_pick: no pick, optimizer_result.json and pick.json at out/spec.md's folder"
    )
    assert isinstance(info.value, DataPreparerOperatorError)


def test_approval_for():
    entries = [
        _approval(),
        _approval("a2", sha="other"),
        _approval("a3", status="pending"),
        _approval("a4", job="other"),
        _approval("a5", step="other"),
        {
            "kind": "decision",
            "id": "a6",
            "step": A.STEP,
            "status": "approved",
            "optimizer_pick_sha256": SHA,
            "job_name": JOB,
        },
    ]
    assert A.approval_for(entries, "a1", SHA, JOB)["id"] == "a1"
    for aid in ("a2", "a3", "a4", "a5", "a6", "nope"):
        with pytest.raises(
            A.AdoptError,
            match=f"no approved optimizer-adopt entry '{aid}' names this pick's "
            f"sha256 {SHA[:12]} and job {JOB}; the pick changed",
        ):
            A.approval_for(entries, aid, SHA, JOB)


def test_adopt():
    res = A.adopt(
        [_approval()],
        pick_bytes=PICK,
        pick_json={"job_name": JOB},
        optimizer_result={"pick_val": "Round 0 | Candidate 0"},
        approval_id="a1",
    )
    rel = f"work/optimizer_pick_{SHA[:12]}.md"
    assert res.decision == {
        "ship": "initial",
        "spec": rel,
        "val": None,
        "reason": "the user adopted the in-job search pick (chosen on val)",
        "selection_mode": "in-job-search",
        "shipped_spec_file": f"optimizer_pick_{SHA[:12]}.md",
    }
    assert (res.spec, res.sha256, res.job_name) == (rel, SHA, JOB)
    assert res.events == [
        {
            "kind": "decision",
            "step": "optimizer-adopt",
            "iteration": 0,
            "status": "adopted",
            "file": "decision.json",
            "spec": rel,
            "approval": "a1",
            "pick_sha256": SHA,
            "job_name": JOB,
        }
    ]
    assert res.to_json() == {"decision": res.decision, "spec": rel, "sha256": SHA, "job_name": JOB}
    refined = A.adopt(
        [_approval()],
        pick_bytes=PICK,
        pick_json={"job_name": JOB},
        optimizer_result=None,
        approval_id="a1",
    )
    assert refined.decision["ship"] == "refined"
    with pytest.raises(A.AdoptError, match="the pick changed"):
        A.adopt(
            [_approval()],
            pick_bytes=PICK + b"x",
            pick_json={"job_name": JOB},
            optimizer_result=None,
            approval_id="a1",
        )
    with pytest.raises(A.AdoptError, match="and job None;"):
        A.adopt(
            [_approval()], pick_bytes=PICK, pick_json=None, optimizer_result=None, approval_id="a1"
        )


def test_check_shippable():
    res = A.adopt(
        [_approval()],
        pick_bytes=PICK,
        pick_json={"job_name": JOB},
        optimizer_result=None,
        approval_id="a1",
    )
    adopted = dict(res.events[0], id="d1")
    entries = [_approval(), adopted]
    assert A.check_shippable(entries, res.decision, PICK) == SHA
    cases = [
        (entries, res.decision, PICK + b"x"),  # edited after the yes
        (
            entries,
            {**res.decision, "selection_mode": "manual"},
            PICK,
        ),  # not an in-job-search decision
        ([_approval()], res.decision, PICK),  # no adopt entry
        ([adopted], res.decision, PICK),  # the approval is gone
        ([_approval("a9"), adopted], res.decision, PICK),  # another approval, not the one named
        (
            [_approval(), adopted, dict(adopted, pick_sha256="later")],
            res.decision,
            PICK,
        ),  # a later adopt
    ]
    for ents, decision, data in cases:
        sha = hashlib.sha256(data).hexdigest()
        with pytest.raises(A.AdoptError) as info:
            A.check_shippable(ents, decision, data)
        assert str(info.value) == (
            f"adopt_job_pick: {decision.get('spec')} (sha256 {sha[:12]}) is not the approved pick; "
            "run adopt-job again after the user's yes"
        )


def test_no_adopted_spec_message():
    err = A.no_adopted_spec_message("/p", KeyError("spec"))
    assert isinstance(err, A.AdoptError)
    assert str(err) == "adopt_job_pick: no adopted spec to ship in /p/decision.json ('spec')"
    assert _digest.sha256_hex(PICK) == SHA


# --- the project layer -------------------------------------------------------------------


def _pick_folder(tmp_path, pick=PICK, with_result=True):
    folder = tmp_path / "work" / "job" / "optimize"
    folder.mkdir(parents=True)
    (folder / "spec.md").write_bytes(pick)
    (folder / "pick.json").write_text(json.dumps({"job_name": JOB}))
    if with_result:
        (folder / "optimizer_result.json").write_text(
            json.dumps({"pick_val": "Round 0 | Candidate 0"})
        )
    return str(folder / "spec.md")


def test_adopt_job_pick_writes_the_spec_the_decision_and_the_entry(tmp_path):
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log

    proj = str(tmp_path)
    aid = _run_log.append_entry(proj, {k: v for k, v in _approval().items() if k != "id"})
    pick = _pick_folder(tmp_path)
    out = A.adopt_job_pick(proj, pick, aid)
    rel = f"work/optimizer_pick_{SHA[:12]}.md"
    assert out["spec"] == rel
    assert (tmp_path / rel).read_bytes() == PICK
    decision = json.load(open(tmp_path / "decision.json"))
    assert decision == out["decision"] and decision["ship"] == "initial"
    entries = _run_log.read_entries(proj)
    assert entries[-1]["id"] == out["decision_entry"]
    assert (entries[-1]["kind"], entries[-1]["status"], entries[-1]["approval"]) == (
        "decision",
        "adopted",
        aid,
    )
    # ship re-checks the bytes and copies them
    shipped = A.ship_adopted_spec(proj, "labeling_spec.md")
    assert shipped["sha256"] == SHA
    assert (tmp_path / "labeling_spec.md").read_bytes() == PICK


def test_adopt_job_pick_refuses_an_incomplete_folder_and_a_missing_approval(tmp_path):
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log

    proj = str(tmp_path)
    pick = _pick_folder(tmp_path, with_result=False)
    with pytest.raises(A.AdoptError, match="no pick, optimizer_result.json and pick.json"):
        A.adopt_job_pick(proj, pick, "a1")
    (tmp_path / "work" / "job" / "optimize" / "optimizer_result.json").write_text("{}")
    with pytest.raises(A.AdoptError):
        A.adopt_job_pick(proj, pick, "a1")  # no approval entry in the run log
    assert not (tmp_path / "decision.json").exists()
    assert _run_log.read_entries(proj) == []


def test_ship_adopted_spec_refuses_changed_bytes_and_no_decision(tmp_path):
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _run_log

    proj = str(tmp_path)
    with pytest.raises(A.AdoptError, match="no adopted spec to ship"):
        A.ship_adopted_spec(proj, "out.md")
    aid = _run_log.append_entry(proj, {k: v for k, v in _approval().items() if k != "id"})
    out = A.adopt_job_pick(proj, _pick_folder(tmp_path), aid)
    (tmp_path / out["spec"]).write_bytes(b"edited\n")
    with pytest.raises(A.AdoptError):
        A.ship_adopted_spec(proj, "out.md")
    assert not (tmp_path / "out.md").exists()
