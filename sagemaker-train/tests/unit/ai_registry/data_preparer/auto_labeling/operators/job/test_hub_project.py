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
"""job.hub_project: the Hub doors as a project uses them (files and run-log entries)."""

from __future__ import absolute_import

import json
import os

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _rows, _run_log
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import (
    dt_handoff,
    hub_project,
    register,
)
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError

from .test_register import DT_OUT, LABELED, _DtSm, _FakeS3
from .test_register import stub as _stub_fixture

stub = _stub_fixture  # the register tests' Hub stub, as this module's fixture

VIEW = DT_OUT.rsplit("/", 1)[0] + "/0.0.1"


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


# --- resolve -----------------------------------------------------------------------------


def test_resolve_input_copies_a_local_file_and_writes_the_sidecar(tmp_path):
    src = tmp_path / "seed.jsonl"
    _write_jsonl(str(src), [{"id": 1, "text": "a"}])
    out = tmp_path / "corpus" / "corpus.jsonl"
    out.parent.mkdir()
    res = hub_project.resolve_input(str(src), str(out), "us-west-2")
    assert res["downloaded"] is True and res["out"] == str(out)
    assert os.path.isfile(res["provenance"]) and res["provenance"] == str(out) + ".provenance.json"
    assert json.load(open(res["provenance"])) == {}  # a local ref has no provenance
    assert open(out).read() == open(src).read()


def test_resolve_input_leaves_a_remote_corpus_behind_the_api(tmp_path, monkeypatch):
    prov = {
        "source": "s3://b/k/corpus.jsonl",
        "resolved_s3_uri": "s3://b/k/corpus.jsonl",
        "etag": "e1",
        "size_bytes": 1234,
    }
    monkeypatch.setattr(register, "resolve_dataset_ref", lambda ref, out, region, download: prov)
    monkeypatch.setattr(_rows, "head_rows", lambda uri, region=None: [{"id": 1, "text": "a"}])
    out = tmp_path / "corpus.jsonl"
    res = hub_project.resolve_input("s3://b/k/corpus.jsonl", str(out), "us-west-2")
    assert res["downloaded"] is False
    assert res["ref"] == str(tmp_path / "corpus.ref.json")
    assert res["head_sample"] == str(tmp_path / "head.jsonl")
    assert not out.exists()  # the body never lands on the client
    assert json.load(open(res["ref"]))["source_s3_uri"] == "s3://b/k/corpus.jsonl"
    assert open(res["head_sample"]).read().strip() == '{"id": 1, "text": "a"}'
    assert os.path.isfile(res["provenance"])


def test_resolve_input_download_is_a_client_read(tmp_path, monkeypatch, caplog):
    prov = {"source": "s3://b/k/seed.jsonl", "resolved_s3_uri": "s3://b/k/seed.jsonl"}
    monkeypatch.setattr(register, "resolve_dataset_ref", lambda ref, out, region, download: prov)
    out = tmp_path / "seed.jsonl"
    with caplog.at_level("INFO", logger=_rows.CLIENT_READ_LOGGER):
        res = hub_project.resolve_input("s3://b/k/seed.jsonl", str(out), None, download=True)
    assert res["downloaded"] is True
    assert any("client-read: resolve_input downloaded" in r.message for r in caplog.records)


def test_resolve_input_head_sample_is_best_effort(tmp_path, monkeypatch, caplog):
    prov = {"source": "s3://b/k/c.jsonl", "resolved_s3_uri": "s3://b/k/c.jsonl"}
    monkeypatch.setattr(register, "resolve_dataset_ref", lambda ref, out, region, download: prov)

    def boom(uri, region=None):
        raise OSError("no")

    monkeypatch.setattr(_rows, "head_rows", boom)
    res = hub_project.resolve_input("s3://b/k/c.jsonl", str(tmp_path / "c.jsonl"), "us-west-2")
    assert res["head_sample"] is None and os.path.isfile(res["ref"])
    assert any("could not write head.jsonl" in r.message for r in caplog.records)


# --- scratch -----------------------------------------------------------------------------


def test_record_scratch_writes_the_entry_and_skips_without_project_or_arn(tmp_path):
    proj = str(tmp_path)
    assert hub_project.record_scratch(None, VIEW, "dt-view") is None
    assert hub_project.record_scratch(proj, None, "dt-view") is None
    assert _run_log.read_entries(proj) == []
    eid = hub_project.record_scratch(proj, VIEW, "dt-view")
    (entry,) = _run_log.read_entries(proj)
    assert entry["id"] == eid and entry["status"] == "scratch" and entry["arn"] == VIEW
    assert [e["arn"] for e in hub_project.scratch_entries(proj)] == [VIEW]


def test_cleanup_scratch_lists_then_deletes_and_logs(tmp_path, stub, monkeypatch):  # noqa: F811
    proj = str(tmp_path)
    hub_project.record_scratch(proj, DT_OUT, "dt-output")
    stub(_DtSm(), _FakeS3(), fixed_hub=True)
    deleted = []
    monkeypatch.setattr(register, "delete_hub_content_version", deleted.append)
    listed = hub_project.cleanup_scratch(proj)
    assert listed["deleted"] == [] and deleted == []
    assert [s["arn"] for s in listed["scratch"]] == [DT_OUT]
    out = hub_project.cleanup_scratch(proj, delete=True)
    assert out["deleted"] == [DT_OUT] == deleted
    assert out["project"] == proj
    statuses = [e["status"] for e in _run_log.read_entries(proj)]
    assert statuses == ["scratch", "deleted"]
    assert _run_log.read_entries(proj)[-1]["scratch_cleanup"] is True


# --- the DT doors ------------------------------------------------------------------------


def test_dt_view_for_project_records_the_view(tmp_path, stub):  # noqa: F811
    stub(_DtSm(), _FakeS3(listing=["job/output/labels.jsonl"]), fixed_hub=True)
    out = hub_project.dt_view_for_project(str(tmp_path), arn=LABELED)
    assert out["dataset_arn"]
    (entry,) = _run_log.read_entries(str(tmp_path))
    assert (entry["status"], entry["invocation"]["stage"], entry["arn"]) == (
        "scratch",
        "dt-view",
        out["dataset_arn"],
    )


def test_promote_for_project_records_the_output(tmp_path, stub):  # noqa: F811
    stub(_DtSm(), _FakeS3(listing=["job/output/labels.jsonl"]), fixed_hub=True)
    out = hub_project.promote_for_project(str(tmp_path), from_arn=DT_OUT, to_arn=LABELED)
    assert out["transform_output_arn"] == DT_OUT
    (entry,) = _run_log.read_entries(str(tmp_path))
    assert (entry["invocation"]["stage"], entry["arn"]) == ("dt-output", DT_OUT)


def test_register_transform_output_for_project_records_a_view_output_only(tmp_path, monkeypatch):
    calls = []

    def fake(**kw):
        calls.append(kw)
        return {"dataset_arn": "arn:new"}

    monkeypatch.setattr(dt_handoff, "register_transform_output", fake)
    proj = str(tmp_path)
    hub_project.register_transform_output_for_project(proj, from_arn=DT_OUT, name="tickets-split")
    assert calls[0]["name"] == "tickets-split"
    assert [e["arn"] for e in _run_log.read_entries(proj)] == [DT_OUT]
    other = LABELED  # not a dt-view version: nothing recorded
    hub_project.register_transform_output_for_project(proj, from_arn=other, name="x")
    assert len(_run_log.read_entries(proj)) == 1


def test_the_doors_raise_the_operator_error_before_recording(tmp_path, stub):  # noqa: F811
    stub(_DtSm(), _FakeS3(), fixed_hub=True)
    with pytest.raises(DataPreparerOperatorError):
        hub_project.promote_for_project(str(tmp_path), from_arn=LABELED, to_arn=LABELED)
    assert _run_log.read_entries(str(tmp_path)) == []
