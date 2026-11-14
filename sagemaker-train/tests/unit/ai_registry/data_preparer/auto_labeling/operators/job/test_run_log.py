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
"""Unit tests for operators.auto_labeling.run_records: a labeling run's entries and their readers.

A tiny in-memory run-log writer stands in for the project's: it assigns ids and stores
project paths relative to the root, as the real one does.
"""

from __future__ import absolute_import

import os

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import run_log as R

JOB = "arn:aws:sagemaker:us-west-2:123456789012:job/SyntheticDataGeneration/j"
DS = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/H/DataSet/d/1.0.0"


class _Log:
    """The caller's run-log writer: ids, created_at, and project paths stored relative."""

    def __init__(self, root):
        self.root, self.entries = root, []

    def _rel(self, v):
        if isinstance(v, dict):
            return {k: self._rel(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._rel(x) for x in v]
        if isinstance(v, str) and os.path.isabs(v) and v.startswith(self.root + os.sep):
            return os.path.relpath(v, self.root)
        return v

    def write(self, entry, **more):
        e = {
            "id": f"e{len(self.entries) + 1}",
            **self._rel({**entry, **more}),
            "created_at": "2026-01-01T00:00:00Z",
        }
        self.entries.append(e)
        return e["id"]

    def record(self, rec):
        ids = [self.write(j) for j in rec.jobs]
        return self.write(rec.artifact_entry(ids))


def _paths(root):
    return [
        {"name": "Labeling spec", "path": os.path.join(root, "labeling_spec.md")},
        {"name": "Input", "path": "s3://b/in.jsonl"},
        {"name": "Labeled dataset", "path": os.path.join(root, "work", "labeled.jsonl")},
        {"name": "Index", "path": "s3://b/index.json"},
    ]


def _rebuilt(root):
    """The paths as a reader rebuilds them: the spec first, the labeled dataset last."""
    p = _paths(root)
    return [p[0], p[1], p[3], p[2]]


def _manifest(root, *, arn=JOB + "1", with_inv=True):
    paths = _paths(root)
    card = {
        "type": "Labeling",
        "description": "job",
        "arn": arn,
        "region": "us-west-2",
        "models": ["m"],
        "s3_paths": list(paths),
        "job_name": "j1",
    }
    m = {
        "region": "us-west-2",
        "task": "classification",
        "rows": 10,
        "stage": "corpus",
        "scratch": False,
        "jobs": [card],
        "report_s3_paths": list(paths),
        "run_report": {
            "type": "Labeling",
            "arn": arn,
            "region": "us-west-2",
            "s3_paths": list(paths),
        },
        "job_summary": {"Status": "Certified"},
        "request": {"dataset_name": "d"},
    }
    if with_inv:
        m["invocation"] = {
            "spec": "labeling_spec.md",
            "input": "s3://b/in.jsonl",
            "out": os.path.join(root, "work", "labeled.jsonl"),
            "project_dir": root,
            "region": "us-west-2",
            "task": "classification",
            "run_id": "r1",
        }
    return m


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return str(tmp_path)


def test_record_splits_the_run_into_job_and_artifact_entries(root):
    rec = R.labeling_run_record(
        _manifest(root), step="auto-label", iteration=2, arn=DS, job_arn=JOB + "1", nothing=None
    )
    assert rec.events == [] and rec.warnings == []
    (job,) = rec.jobs
    assert job["kind"] == "job" and job["status"] == "completed" and job["arn"] == JOB + "1"
    assert (job["step"], job["iteration"]) == ("auto-label", 2)
    # the spec and labeled-dataset items repeat the invocation: left out of the entry
    assert [p["name"] for p in job["s3_paths"]] == ["Input", "Index"]
    a = rec.artifact
    assert a["kind"] == "artifact" and a["status"] == "labeled" and a["labeling_run"] is True
    assert a["arn"] == DS and a["job_arn"] == JOB + "1" and "nothing" not in a
    assert "job_entries" not in a and "jobs" not in a
    # the primary job entry carries the paths and ARN: the artifact carries neither
    assert "report_s3_paths" not in a and a["run_report"] == {"type": "Labeling"}
    # region and task repeat the invocation, whose paths are absolute
    assert "region" not in a and "task" not in a
    assert a["invocation"]["spec"] == os.path.join(root, "labeling_spec.md")
    assert a["invocation"]["input"] == "s3://b/in.jsonl"
    assert rec.to_json() == {"jobs": rec.jobs, "artifact": a}
    # the entry to append: job_entries right after labeling_run, the artifact untouched
    entry = rec.artifact_entry(["e1"])
    keys = list(entry)
    assert keys[keys.index("labeling_run") + 1] == "job_entries" and entry["job_entries"] == ["e1"]
    assert {k: v for k, v in entry.items() if k != "job_entries"} == a and "job_entries" not in a


def test_record_without_a_job_arn_keeps_the_report_on_the_artifact(root):
    rec = R.labeling_run_record(_manifest(root, arn=None), step="s", iteration=0, status="scratch")
    assert rec.jobs == []
    a = rec.artifact
    assert a["status"] == "scratch"
    assert [p["name"] for p in a["report_s3_paths"]] == ["Input", "Index"]
    assert a["run_report"]["arn"] is None
    assert [p["name"] for p in a["run_report"]["s3_paths"]] == ["Input", "Index"]
    assert "region" not in a["run_report"]


def test_record_keeps_values_that_differ_from_the_invocation(root):
    m = _manifest(root)
    m["invocation"]["region"] = "us-east-1"
    m["invocation"]["task"] = "ner"
    rec = R.labeling_run_record(m, step="s", iteration=0)
    a = rec.artifact
    assert a["region"] == "us-west-2" and a["task"] == "classification"
    assert a["run_report"] == {"type": "Labeling", "region": "us-west-2"}


def test_record_without_an_invocation(root):
    rec = R.labeling_run_record(_manifest(root, with_inv=False), step="s", iteration=0)
    a = rec.artifact
    assert "invocation" not in a and a["region"] == "us-west-2"
    assert [p["name"] for p in rec.jobs[0]["s3_paths"]] == [
        "Labeling spec",
        "Input",
        "Labeled dataset",
        "Index",
    ]


def test_record_refuses_a_reserved_field(root):
    m = _manifest(root)
    m["id"] = "mine"
    with pytest.raises(DataPreparerOperatorError, match="pass id as arguments"):
        R.labeling_run_record(m, step="s", iteration=0)


def test_read_rebuilds_the_manifest_shape(root):
    log = _Log(root)
    m = _manifest(root)
    log.record(R.labeling_run_record(m, step="s", iteration=0, arn=DS, job_arn=JOB + "1"))
    (run,) = R.read_labeling_runs(log.entries, root)
    # `output` is the invocation's `out` as stored (relative to the project)
    assert run["id"] == "e2" and run["output"] == os.path.join("work", "labeled.jsonl")
    assert run["region"] == "us-west-2" and run["task"] == "classification"
    assert run["arn"] == DS and run["job_arn"] == JOB + "1"
    (job,) = run["jobs"]
    assert job["arn"] == JOB + "1" and "id" not in job and "kind" not in job
    # the spec and labeled dataset come back from the invocation: first and last
    assert job["s3_paths"] == _rebuilt(root)
    assert run["report_s3_paths"] == _rebuilt(root)
    assert run["run_report"] == {
        "type": "Labeling",
        "arn": JOB + "1",
        "region": "us-west-2",
        "s3_paths": _rebuilt(root),
    }
    assert run["job_summary"] == {"Status": "Certified"} and run["request"] == {"dataset_name": "d"}
    for key in ("labeling_run", "job_entries", "created_at", "kind", "step", "status"):
        assert key not in run


def test_read_without_a_job_entry_or_report(root):
    log = _Log(root)
    log.record(R.labeling_run_record(_manifest(root, arn=None), step="s", iteration=0))
    m = _manifest(root)
    m.pop("run_report")
    m.pop("report_s3_paths")
    log.record(R.labeling_run_record(m, step="s", iteration=1))
    log.write({"kind": "decision", "step": "s", "iteration": 0, "status": "x"})
    log.write({"kind": "artifact", "step": "s", "iteration": 0, "status": "x"})  # not a run
    first, second = R.read_labeling_runs(log.entries, root)
    assert first["jobs"] == [] and first["report_s3_paths"] == _rebuilt(root)
    assert first["run_report"]["arn"] is None and first["run_report"]["region"] == "us-west-2"
    assert second["jobs"][0]["arn"] == JOB + "1"
    assert second["report_s3_paths"] == second["run_report"]["s3_paths"] == _rebuilt(root)
    assert second["run_report"]["arn"] == JOB + "1"


def test_read_skips_a_job_entry_that_is_gone(root):
    log = _Log(root)
    log.record(R.labeling_run_record(_manifest(root), step="s", iteration=0))
    log.entries = [e for e in log.entries if e["kind"] != "job"]
    (run,) = R.read_labeling_runs(log.entries, root)
    assert run["jobs"] == [] and "report_s3_paths" not in run
    assert run["run_report"] == {"type": "Labeling", "region": "us-west-2"}


def test_latest_runs_per_output(root):
    log = _Log(root)
    m = _manifest(root)
    log.record(R.labeling_run_record(m, step="s", iteration=0))
    other = _manifest(root)
    other["invocation"]["out"] = os.path.join(root, "work", "other.jsonl")
    log.record(R.labeling_run_record(other, step="s", iteration=1))
    log.record(R.labeling_run_record(m, step="s", iteration=2))  # re-labels the first output
    no_out = _manifest(root, with_inv=False)
    log.record(R.labeling_run_record(no_out, step="s", iteration=3))
    runs = R.latest_labeling_runs(log.entries, root)
    assert [(r["id"], r["output"]) for r in runs] == [
        ("e6", os.path.join("work", "labeled.jsonl")),
        ("e4", os.path.join("work", "other.jsonl")),
        ("e8", None),
    ]


def test_invocation_resolves_paths_against_the_root(root):
    log = _Log(root)
    log.record(R.labeling_run_record(_manifest(root), step="s", iteration=0))
    inv = R.labeling_run_invocation(log.entries, root)
    assert inv["spec"] == os.path.join(root, "labeling_spec.md")
    assert inv["out"] == os.path.join(root, "work", "labeled.jsonl")
    assert inv["input"] == "s3://b/in.jsonl" and inv["run_id"] == "r1"
    assert R.labeling_run_invocation(log.entries, root, "e2") == inv
    with pytest.raises(DataPreparerOperatorError, match="no labeling run with id nope"):
        R.labeling_run_invocation(log.entries, root, "nope")
    with pytest.raises(DataPreparerOperatorError, match="has no labeling run$"):
        R.labeling_run_invocation([], root)
    log.record(R.labeling_run_record(_manifest(root, with_inv=False), step="s", iteration=1))
    with pytest.raises(DataPreparerOperatorError, match="e4 recorded no invocation"):
        R.labeling_run_invocation(log.entries, root)


def test_run_for_output(root):
    log = _Log(root)
    log.record(R.labeling_run_record(_manifest(root), step="s", iteration=0))
    log.record(R.labeling_run_record(_manifest(root), step="s", iteration=1))
    out = os.path.join(root, "work", "labeled.jsonl")
    assert R.labeling_run_for_output(log.entries, root, out)["id"] == "e4"
    assert (
        R.labeling_run_for_output(log.entries, root, os.path.join("work", "labeled.jsonl"))["id"]
        == "e4"
    )
    assert R.labeling_run_for_output(log.entries, root, os.path.join(root, "other.jsonl")) is None
    assert R.labeling_run_for_output([], root, out) is None


def test_path_helpers():
    assert (
        R._is_local_path("a/b") and not R._is_local_path("s3://x") and not R._is_local_path("arn:x")
    )
    assert not R._is_local_path("") and not R._is_local_path(None)
    assert R._abs_local("s3://x") == "s3://x" and os.path.isabs(R._abs_local("a"))
    assert R._resolve("a/b", "/r") == "/r/a/b" and R._resolve("/a", "/r") == "/a"
    assert R._without_invocation_paths("not-a-list", {}) == "not-a-list"
    assert R._with_invocation_paths(None, {"spec": "/s"}, "/r") is None
    paths = [{"name": "Input", "path": "s3://x"}, "junk"]
    assert R._with_invocation_paths(paths, {"spec": "spec.md", "out": "/o"}, "/r") == [
        {"name": "Labeling spec", "path": "/r/spec.md"},
        {"name": "Input", "path": "s3://x"},
        "junk",
        {"name": "Labeled dataset", "path": "/o"},
    ]
