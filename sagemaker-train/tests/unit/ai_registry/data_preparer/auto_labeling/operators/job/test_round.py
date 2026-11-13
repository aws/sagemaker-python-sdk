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
"""Unit tests for operators.auto_labeling.run_round: one labeling round, prepared, submitted and read."""

from __future__ import absolute_import

import json
import os
import re
from pathlib import Path

import pytest

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import client as J
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import request as LR
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import round as R
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.seed import split as split_state

MODEL_A = "us.anthropic.claude-sonnet-4-6"
MODEL_B = "qwen.qwen3-32b-v1:0"
MODEL_C = "openai.gpt-oss-120b-1:0"
ROLE = "arn:aws:iam::123456789012:role/r"
SPEC = "# Spec\nLabel `a` for topic A, `b` for topic B.\n## Label Set\n- a\n- b\n"


def _arn(name, version="1.0.0"):
    return (
        f"arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/DataSet/{name}/{version}"
    )


def _jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


class _Proj:
    """A project: a spec, dev / val / test splits (optionally frozen) and a local corpus."""

    def __init__(self, root, freeze=False):
        self.root = str(root)
        self.spec = os.path.join(self.root, "labeling_spec.md")
        Path(self.spec).write_text(SPEC)
        self.splits = os.path.join(self.root, "splits")
        self.dev = _jsonl(
            os.path.join(self.splits, "dev.jsonl"),
            [
                {"id": f"d{i}", "text": f"dev row {i}", "gt_label": "a" if i % 2 else "b"}
                for i in range(8)
            ],
        )
        self.val = _jsonl(
            os.path.join(self.splits, "val.jsonl"),
            [
                {"id": f"v{i}", "text": f"val row {i}", "gt_label": "a" if i % 2 else "b"}
                for i in range(4)
            ],
        )
        self.test = _jsonl(
            os.path.join(self.splits, "test.jsonl"),
            [
                {"id": f"t{i}", "text": f"test row {i}", "gt_label": "a" if i % 2 else "b"}
                for i in range(6)
            ],
        )
        if freeze:
            res = split_state.freeze(self.splits, ["text"])
            Path(res.state_path).write_text(json.dumps(res.state), encoding="utf-8")
        self.corpus = _jsonl(
            os.path.join(self.root, "corpus", "corpus.jsonl"),
            [{"id": f"c{i}", "text": f"corpus row {i}"} for i in range(10)],
        )

    def dev_kw(self, **kw):
        base = dict(
            spec=self.spec,
            input_data=self.dev,
            out=os.path.join(self.root, "work", "dev_pred.jsonl"),
            project_dir=self.root,
            stage="dev",
            generation_model_id=MODEL_A,
            task="classification",
            dataset_name="support",
            eval_data=self.dev,
            s3_prefix="s3://b/p/",
            role_arn=ROLE,
            run_id="run-1",
        )
        base.update(kw)
        return base

    def corpus_kw(self, **kw):
        base = self.dev_kw(
            input_data=self.corpus,
            out=os.path.join(self.root, "labeled_dataset.json"),
            stage="corpus",
            eval_data=self.test,
            jury_models=f"{MODEL_B},{MODEL_C}",
            target=0.9,
            review_dataset_name="support-review",
        )
        base.update(kw)
        return base


@pytest.fixture
def proj(tmp_path):
    return _Proj(tmp_path)


class _Fake:
    """The job_client seams: records uploads and submissions, writes a job folder on read."""

    def __init__(self, summary=None, job_status="Completed", submit_error=None, existing=()):
        self.summary = summary
        self.job_status = job_status
        self.submit_error = submit_error
        self.existing = set(existing)
        self.uploads, self.files, self.submits, self.reads, self.checks = {}, [], [], [], []

    def install(self, monkeypatch):
        monkeypatch.setattr(J, "upload_text", self.upload_text)
        monkeypatch.setattr(_s3, "upload_file", self.upload_file)
        monkeypatch.setattr(J, "submit_labeling_job", self.submit)
        monkeypatch.setattr(J, "read_job_results", self.read)
        monkeypatch.setattr(register, "existing_datasets", self.existing_datasets)
        return self

    def upload_text(self, text, s3_uri, region, *, field, max_bytes=None):
        if max_bytes is not None and len(text.encode()) > max_bytes:
            raise LR.LabelingRequestError(f"{field} is too big")
        self.uploads[field.split(" ")[0]] = (s3_uri, text)
        return s3_uri

    def upload_file(self, path, s3_uri, region):
        self.files.append((path, s3_uri))
        return s3_uri

    def existing_datasets(self, names, region, hub_name=None):
        self.checks.append(list(names))
        return [
            {"name": n, "hub": "AIRegistry", "version": "1.0.0", "arn": _arn(n)}
            for n in names
            if n in self.existing
        ]

    def submit(self, document, **kw):
        self.submits.append(dict(kw, document=document))
        resume = kw.get("resume_job")
        name = resume["job_name"] if resume else f"{kw['base_job_name']}-{len(self.submits)}"
        if not resume and kw.get("before_submit"):
            kw["before_submit"]()
        if not resume and kw.get("on_submit"):
            kw["on_submit"](
                {
                    "job_name": name,
                    "output_s3_uri": document["OutputDataConfig"].get("S3OutputPath")
                    or document["OutputDataConfig"].get("OutputS3Uri"),
                    "dataset_name": document["DatasetName"],
                }
            )
        if self.submit_error is not None:
            raise self.submit_error
        routing = document["GenerationConfig"]["DatasetOptions"]["Config"].get("Routing")
        return {
            "job_name": name,
            "job_arn": f"arn:aws:sagemaker:us-west-2:123456789012:job/{name}",
            "job_status": self.job_status,
            "dataset_arn": _arn(document["DatasetName"]),
            "review_dataset_arn": _arn(routing["ReviewDatasetName"]) if routing else None,
            "output_s3_uri": document["OutputDataConfig"].get("S3OutputPath")
            or document["OutputDataConfig"].get("OutputS3Uri"),
            "reattached": bool(resume),
            "failure_reason": "boom" if self.job_status != "Completed" else None,
        }

    def read(self, *, region, dataset_arn, dest_dir, names=None, review_dataset_arn=None):
        self.reads.append(dict(dataset_arn=dataset_arn, names=names))
        doc = self.submits[-1]["document"]
        cfg = doc["GenerationConfig"]["DatasetOptions"]["Config"]
        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)
        inputs = [
            json.loads(x) for x in self.uploads.get("InputDataS3Uri", ("", ""))[1].splitlines()
        ]
        labels = [dict(r, label="a") for r in inputs]
        summary = self.summary or {
            "Status": "Certified",
            "RowsLabeled": len(labels) or 10,
            "RowsAccepted": 8,
            "RowsMissing": 0,
            "HeldOutAccuracy": 0.9,
            "HeldOutLowerBound95": 0.8,
            "HeldOutRows": 6,
            "JurorModelIds": [
                m["ModelId"]
                for m in doc["GenerationConfig"]["ModelConfig"]["SampleGenerationModelConfigs"]
            ],
            "Warnings": [],
        }
        artifacts = {
            "Labels": ("labels.jsonl", "".join(json.dumps(r) + "\n" for r in labels)),
            "LabelingSummary": ("labeling_summary.json", json.dumps(summary)),
            "Failures": ("failures.jsonl", '{"id": "d1"}\n'),
            "HeldOutReport": ("held_out_report.json", json.dumps({"macro_f1": 0.8})),
        }
        if "Routing" in cfg:
            side = {"id": "c0", "pred_field": "label", "model_output": {"label": None}}
            artifacts["ReviewModelOutput"] = (
                "review_rows.model_output.jsonl",
                json.dumps(side) + "\n",
            )
        if "Optimizer" in cfg:
            artifacts["Spec"] = ("spec.md", "picked spec\n")
        files, uris = {}, {}
        for name, (fname, text) in artifacts.items():
            uris[name] = f"s3://b/out/{fname}"
            if names and name not in names:
                continue
            (dest / fname).write_text(text)
            files[name] = str(dest / fname)
        return {
            "index_uri": "s3://b/out/labeling/index.json",
            "files": files,
            "summary": summary,
            "labeled_dataset_arn": dataset_arn,
            "review_dataset_arn": review_dataset_arn,
            "artifact_uris": uris,
        }


def _shim(plan):
    """What a command-line caller does with a plan before submit_round."""
    Path(plan.staging_dir).mkdir(parents=True, exist_ok=True)
    Path(plan.request_path).write_text(plan.request_text)
    record = Path(plan.job_record_path)
    return {"on_submit": record.write_text, "on_refused": lambda: record.unlink(missing_ok=True)}


def _round(kw, fake, **read_kw):
    plan = R.prepare_round(**kw)
    sub = R.submit_round(plan, **_shim(plan))
    return plan, sub, R.read_round(plan, sub.job, **read_kw)


# ---- a refine round -----------------------------------------------------------------------------


def test_dev_round_plans_submits_and_reads(proj, monkeypatch):
    fake = _Fake().install(monkeypatch)
    plan, sub, outs = _round(proj.dev_kw(demos=proj.dev, demos_per_class=1), fake)
    # the plan: nothing uploaded yet, the request built, the paths scoped to the run id
    assert plan.dataset_name == "support-dev-run-1" and plan.models == [MODEL_A]
    assert plan.document["DatasetName"] == "support-dev-run-1"
    assert plan.request_path.endswith(os.path.join("_sdg_work", "run-1", "dev", "request.json"))
    assert json.loads(plan.request_text) == plan.document and plan.events == []
    assert plan.invocation["stage"] == "dev" and plan.invocation["run_id"] == "run-1"
    assert plan.invocation["project_dir"] == os.path.abspath(proj.root)
    # the uploads: the input without the gold, the gold without the demos
    assert set(fake.uploads) == {
        "DatasetSpecification.SpecS3Uri",
        "Demos",
        "EvaluationData",
        "InputDataS3Uri",
    }
    inputs = [json.loads(x) for x in fake.uploads["InputDataS3Uri"][1].splitlines()]
    assert all(set(r) == {"id", "text"} for r in inputs) and len(inputs) == 8
    gold = [json.loads(x) for x in fake.uploads["EvaluationData"][1].splitlines()]
    demo_ids = {json.loads(x)["id"] for x in fake.uploads["Demos"][1].splitlines()}
    assert len(gold) == 6 and not demo_ids & {g["id"] for g in gold}
    assert (
        fake.submits[0]["base_job_name"] == "autolabel-dev" and fake.submits[0]["role_arn"] == ROLE
    )
    # the job record went through on_submit, with the digest and the stage
    rec = json.loads(Path(plan.job_record_path).read_text())
    assert rec["request_sha256"] == plan.request_sha256 and rec["stage"] == "dev"
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", rec["submitted_at"])
    # what the round writes
    out = os.path.join(proj.root, "work", "dev_pred.jsonl")
    assert list(outs.texts) == [
        out,
        os.path.join(proj.root, "work", "dev_pred.summary.json"),
        os.path.join(proj.root, "work", "dev_pred.held_out_report.json"),
    ]
    assert len(outs.texts[out].splitlines()) == 8
    hor = json.loads(outs.texts[os.path.join(proj.root, "work", "dev_pred.held_out_report.json")])
    assert hor["primary_name"] == "accuracy" and hor["macro_f1"] == 0.8
    assert list(outs.copies) == [os.path.join(proj.root, "work", "dev_pred.failures.jsonl")]
    assert outs.removes == [] and outs.run_log_error is None
    # the record: one job entry, one scratch artifact
    (job,) = outs.record.jobs
    assert (
        job["kind"] == "job" and job["status"] == "completed" and job["arn"] == sub.job["job_arn"]
    )
    art = outs.record.artifact
    assert (art["kind"], art["status"], art["step"]) == ("artifact", "scratch", "auto-label")
    assert art["arn"] == _arn("support-dev-run-1") and art["job_arn"] == sub.job["job_arn"]
    s = R.round_summary(outs, run_log_entry="artifact-1")
    assert list(s)[-1] == "run_log_entry" and s["rows"] == 8 and s["scratch"] is True
    assert s["files"]["labels"] == out and s["macro_f1"] == 0.8


def test_reattach_reads_the_job_already_submitted(proj, monkeypatch):
    fake = _Fake().install(monkeypatch)
    _, first, _ = _round(proj.dev_kw(), fake)
    _, second, _ = _round(proj.dev_kw(), fake)
    assert fake.submits[1]["resume_job"]["job_name"] == first.job["job_name"]
    assert second.job["reattached"] is True
    # a different request under the same run id and stage is refused before any upload
    Path(proj.spec).write_text(SPEC + "\nA new rule.\n")
    plan = R.prepare_round(**proj.dev_kw())
    fake.uploads.clear()
    with pytest.raises(DataPreparerOperatorError, match="different spec, demos, gold or input"):
        R.submit_round(plan, **_shim(plan))
    assert fake.uploads == {}
    # an auto run id never re-attaches
    monkeypatch.setattr(R, "default_run_id", lambda: "20250101T000000-abcdef12")
    plan = R.prepare_round(**proj.dev_kw(run_id=None))
    assert plan.run_id == "20250101T000000-abcdef12" and plan.explicit_run_id is False


def test_fire_and_forget_submit_then_collect(proj, monkeypatch):
    # wait=False returns while the job runs; read_round refuses it; a later submit_round
    # under the same run id re-attaches (nothing re-uploaded or resubmitted) and reads.
    fake = _Fake(job_status="InProgress").install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw())
    sub = R.submit_round(plan, wait=False, **_shim(plan))
    assert sub.running and fake.submits[0]["wait"] is False
    assert json.loads(Path(plan.job_record_path).read_text())["job_name"] == sub.job["job_name"]
    with pytest.raises(DataPreparerOperatorError, match="still InProgress; nothing to read yet"):
        R.read_round(plan, sub.job)
    uploads = dict(fake.uploads)
    fake.uploads.clear()
    again = R.submit_round(R.prepare_round(**proj.dev_kw()), wait=False, **_shim(plan))
    assert again.running and again.job["reattached"] and fake.uploads == {}
    assert fake.submits[1]["resume_job"]["job_name"] == sub.job["job_name"]
    fake.job_status = "Completed"
    plan = R.prepare_round(**proj.dev_kw())
    done = R.submit_round(plan, wait=False, **_shim(plan))
    assert not done.running and fake.uploads == {} and len(fake.submits) == 3
    assert set(uploads) >= {"DatasetSpecification.SpecS3Uri", "InputDataS3Uri"}
    fake.uploads.update(uploads)  # the fake's read builds the labels from the first upload
    outs = R.read_round(plan, done.job)
    assert len(outs.texts[os.path.join(proj.root, "work", "dev_pred.jsonl")].splitlines()) == 8


def test_submitted_round_running():
    for status, running in (
        ("InProgress", True),
        ("Stopping", True),
        ("Completed", False),
        ("Failed", False),
        ("Stopped", False),
        (None, False),
    ):
        assert R.SubmittedRound(job={"job_status": status}).running is running


# ---- the corpus run ----------------------------------------------------------------------------


def test_corpus_run_writes_the_deliverable_and_the_side_files(proj, monkeypatch):
    fake = _Fake().install(monkeypatch)
    Path(proj.root, "review.jsonl").write_text("{}\n")
    plan, sub, outs = _round(proj.corpus_kw(), fake, run_log_step="corpus", run_log_iteration=2)
    assert plan.models == [MODEL_A, MODEL_B, MODEL_C] and plan.dataset_name == "support"
    assert fake.files == [(proj.corpus, "s3://b/p/run-run-1/corpus/input/corpus.jsonl")]
    assert fake.checks == [["support", "support-review"]]
    assert fake.reads[0]["names"] == R.CORPUS_ARTIFACTS
    # the gold is the test split, prefixed ids only for rows with none; a corpus run sends a target
    assert plan.document["GenerationConfig"]["DatasetOptions"]["Config"]["Scoring"] == {
        "Target": 0.9
    }
    root = proj.root
    deliverable = json.loads(outs.texts[os.path.join(root, "labeled_dataset.json")])
    assert list(deliverable)[:4] == [
        "dataset_name",
        "dataset_arn",
        "review_dataset_name",
        "review_dataset_arn",
    ]
    assert deliverable["label_field"] == "gt_label" and deliverable["rows"] == 10
    assert os.path.join(root, "labeling_summary.json") in outs.texts
    assert os.path.join(root, "held_out_report.json") in outs.texts
    assert list(outs.copies) == [os.path.join(root, "review.model_output.jsonl")]
    assert outs.removes[-1] == os.path.join(root, "review.jsonl")
    assert any(
        w.startswith("NOTE: no juror check: the job routed no voted rows") for w in outs.warnings
    )
    assert outs.record.artifact["status"] == "labeled"
    assert (outs.record.artifact["step"], outs.record.artifact["iteration"]) == ("corpus", 2)
    s = R.round_summary(outs, run_log_entry="artifact-2")
    keys = list(s)
    assert keys[-3:] == ["jury_check", "note", "deliverable"]
    assert keys.index("run_log_entry") == keys.index("jury_check") - 1
    assert s["failed_rows"] == 1 and "null prediction" in s["note"]


def test_corpus_run_dataset_names(proj, monkeypatch):
    fake = _Fake(existing={"support"}).install(monkeypatch)
    plan = R.prepare_round(**proj.corpus_kw())
    with pytest.raises(
        DataPreparerOperatorError, match=r"already registered: support \(version 1.0.0"
    ):
        R.submit_round(plan, **_shim(plan))
    assert fake.uploads == {} and fake.submits == []
    plan = R.prepare_round(**proj.corpus_kw(append=True))
    assert R.submit_round(plan, **_shim(plan)).job["job_status"] == "Completed"

    def boom(names, region, hub_name=None):
        raise RuntimeError("denied")

    monkeypatch.setattr(register, "existing_datasets", boom)
    plan = R.prepare_round(**proj.corpus_kw(run_id="run-2"))
    with pytest.raises(
        DataPreparerOperatorError, match=r"could not check .*\(RuntimeError: denied\)"
    ):
        R.submit_round(plan, **_shim(plan))


def test_no_region_resolves_before_the_dataset_name_check(proj, monkeypatch):
    # region=None resolves to the ambient default region at prepare_round; it never reaches
    # existing_datasets or the hub name (AiRegistry-None-<account> names no hub).
    import boto3

    real_existing = register.existing_datasets
    fake = _Fake().install(monkeypatch)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-west-1")
    monkeypatch.delenv("AWS_REGION", raising=False)
    seen = []

    class _SM:
        def describe_hub_content(self, **kw):
            raise RuntimeError("ResourceNotFound")

    class _STS:
        def get_caller_identity(self):
            return {"Account": "123456789012"}

    def hub_name(account_id, region):
        seen.append(("hub", region))
        return "H"

    def existing(names, region, hub_name=None):
        seen.append(("check", region))
        return real_existing(names, region, hub_name)

    monkeypatch.setattr(register, "_sm_client", lambda region: _SM())
    monkeypatch.setattr(boto3, "client", lambda svc, region_name=None: _STS())
    monkeypatch.setattr(register, "ai_registry_hub_name", hub_name)
    monkeypatch.setattr(register, "existing_datasets", existing)
    plan = R.prepare_round(**proj.corpus_kw(region=None))
    assert plan.region == "eu-west-1" and plan.invocation["region"] == "eu-west-1"
    R.submit_round(plan, **_shim(plan))
    assert seen == [("check", "eu-west-1"), ("hub", "eu-west-1")]
    assert fake.submits[0]["region"] == "eu-west-1"
    monkeypatch.delenv("AWS_DEFAULT_REGION")
    monkeypatch.setattr(boto3.session.Session, "region_name", None)
    assert R.prepare_round(**proj.corpus_kw(region=None)).region == "us-west-2"
    assert _s3.resolve_region("ap-south-1") == "ap-south-1"


def test_plan_to_json_leaves_out_the_customer_content(proj, monkeypatch):
    _Fake().install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw(demos=proj.dev, demos_per_class=1))
    js = plan.to_json()
    assert set(R.RoundPlan.CUSTOMER_CONTENT_FIELDS).isdisjoint(js)
    # no row, demo or spec text survives anywhere else in the dict
    text = json.dumps(js)
    assert "dev row" not in text and "topic A" not in text
    assert js["run_id"] == "run-1" and js["request_sha256"] == plan.request_sha256
    # the attributes still carry them for submit_round
    assert plan.spec_text == SPEC and plan.gold_rows and plan.gold_text and plan.demos_text
    assert plan.input_rows_text and plan.request_text


def test_frozen_split_test_reads_are_events(tmp_path, monkeypatch):
    proj = _Proj(tmp_path, freeze=True)
    _Fake().install(monkeypatch)
    plan = R.prepare_round(**proj.corpus_kw())
    (ev,) = plan.events
    assert (ev["kind"], ev["step"], ev["status"], ev["rows"]) == (
        "analysis",
        "certificate",
        "test-read",
        6,
    )
    assert ev["file"] == os.path.abspath(proj.test)
    part = _jsonl(
        tmp_path / "work" / "part.jsonl",
        [json.loads(x) for x in Path(proj.test).read_text().splitlines()[:3]],
    )
    with pytest.raises(DataPreparerOperatorError, match="--eval leaves out 3 row"):
        R.prepare_round(**proj.corpus_kw(eval_data=part))
    bad = _jsonl(
        tmp_path / "work" / "bad.jsonl", [{"id": "t0", "text": "test row 0", "gt_label": "b"}]
    )
    with pytest.raises(
        DataPreparerOperatorError, match=r"^split guard: labeling-dev: .* holds 1 test row"
    ):
        R.prepare_round(**proj.dev_kw(input_data=bad, eval_data=bad))


def test_unfrozen_split_is_noted(proj, monkeypatch):
    plan = R.prepare_round(**proj.dev_kw())
    assert plan.warnings[0].startswith("WARNING: the project's splits/ has no split_state.json")


def test_refusal_after_the_golden_check_carries_its_reads(proj, monkeypatch):
    event = {
        "kind": "analysis",
        "step": "certificate",
        "iteration": 0,
        "status": "test-read",
        "rows": 6,
    }
    monkeypatch.setattr(
        R, "check_golden_review", lambda m, p, rows: ({"golden_rows": 6}, "NOTE: x", [event])
    )
    with pytest.raises(R.RoundRefused) as info:
        R.prepare_round(**proj.corpus_kw(label_set='["x", "y"]'))
    assert info.value.events == [event] and "Respell the gold" in str(info.value)
    assert "NOTE: x" in info.value.warnings
    assert isinstance(info.value, DataPreparerOperatorError)


def _reviewed_copy(proj):
    """A golden-row review of the test split: route, a person relabels t1, apply."""
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.review import golden as G

    routed = _jsonl(
        Path(proj.root, "audit", "routed.jsonl"), [{"id": "t1", "checks": ["keyword_rules"]}]
    )
    out_dir = os.path.join(proj.root, "golden_review")
    res = G.route(proj.test, routed, out_dir, ["text"], project_dir=proj.root)
    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, G.MANIFEST_FILE)
    Path(manifest).write_text(G.manifest_text(res.manifest))
    export = _jsonl(
        Path(proj.root, "export.jsonl"),
        [{"record_id": "t1", "corrected_completion": "b", "annotator": "alice"}],
    )
    res = G.apply(
        manifest,
        export,
        export_id_field="record_id",
        export_label_field="corrected_completion",
        project_dir=proj.root,
    )
    Path(res.reviewed_path).write_text(res.reviewed_text)
    Path(manifest).write_text(G.manifest_text(res.manifest))
    return manifest, res.reviewed_path, routed


def test_golden_review_check(tmp_path):
    proj = _Proj(tmp_path, freeze=True)
    manifest, reviewed, routed = _reviewed_copy(proj)
    plan = R.prepare_round(**proj.corpus_kw(golden_review=manifest, eval_data=reviewed))
    assert plan.golden_check == {"golden_rows": 6, "changed": 1, "confirmed": 0, "unresolved": 0}
    assert any(w.startswith("NOTE: golden rows reviewed: 1 row(s)") for w in plan.warnings)
    # the check's read of test.jsonl, then the guard's read of the reviewed copy
    assert [(e["step"], e["file"]) for e in plan.events] == [
        ("certificate", os.path.abspath(proj.test)),
        ("certificate", os.path.abspath(reviewed)),
    ]
    # a check refused after it read the test split hands that read back
    Path(routed).write_text(Path(routed).read_text().replace("keyword_rules", "kw"))
    with pytest.raises(R.RoundRefused, match="routed.jsonl is missing or changed") as info:
        R.prepare_round(**proj.corpus_kw(golden_review=manifest, eval_data=reviewed))
    assert [e["file"] for e in info.value.events] == [os.path.abspath(proj.test)]
    # a reviewed copy without its manifest is refused (in a project with no frozen split, where
    # the split guard's every-test-row check does not come first)
    os.makedirs(tmp_path / "other")
    other = _Proj(tmp_path / "other")
    copy = _jsonl(
        tmp_path / "other" / "copy.jsonl",
        [json.loads(x) for x in Path(reviewed).read_text().splitlines()],
    )
    with pytest.raises(DataPreparerOperatorError, match="is a golden_review.py reviewed copy"):
        R.prepare_round(**other.corpus_kw(eval_data=copy))


def test_notes_before_a_refusal_ride_on_it(proj):
    with pytest.raises(R.RoundRefused) as info:
        R.prepare_round(**proj.dev_kw(eval_data=None, label_set='["a", "b", "zzz"]'))
    assert str(info.value).startswith("no labeled seed") and info.value.events == []
    assert info.value.warnings[0].startswith("WARNING: the project's splits/ has no split_state")
    assert info.value.warnings[-1].startswith("WARNING: label(s) zzz from --label-set")
    with pytest.raises(DataPreparerOperatorError) as plain:
        R.prepare_round(**proj.dev_kw(s3_prefix=None))
    assert type(plain.value) is DataPreparerOperatorError  # nothing noted yet


# ---- refusals before anything is uploaded --------------------------------------------------------


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(s3_prefix=None), "^set --s3-prefix"),
        (dict(s3_prefix="b/p"), "must be an s3://bucket/prefix URI"),
        (dict(stage="!!"), "--stage must contain a letter or digit"),
        (dict(review_dataset_name="r"), "which is the corpus run: pass --stage corpus"),
        (dict(target=0.9), r"--target \(Scoring.Target\) certifies the corpus run"),
        (dict(golden_review="g.json"), "--golden-review checks the corpus run"),
        (dict(text_fields=" , "), "--text-fields names at least one input field"),
        (dict(text_fields="text,title"), "--text-fields title not on the first row"),
        (dict(run_id="a.b"), "must be letters, digits and hyphens"),
        (dict(optimizer="{nope", stage="optimize"), r"--optimizer is not JSON"),
        (dict(recalibration_fit="v.jsonl"), "go together"),
        (dict(spec="/nope/spec.md"), "cannot read --spec"),
        (
            dict(confidence_method="verbalized", text_fields="text,confidence_missing"),
            "writes that row field",
        ),
        (dict(eval_data=None, demos=None), "^no labeled seed"),
        (dict(input_data="x.ref.json"), "cannot read corpus.ref.json pointer"),
        (
            dict(task="summarization", graders=MODEL_B, label_field="gt_label"),
            "labeling request refused before anything",
        ),
    ],
)
def test_prepare_refusals(proj, change, match):
    with pytest.raises(DataPreparerOperatorError, match=match):
        R.prepare_round(**proj.dev_kw(**change))


def test_corpus_refusals(proj):
    with pytest.raises(DataPreparerOperatorError, match="the corpus run downloads no labels"):
        R.prepare_round(**proj.corpus_kw(out=os.path.join(proj.root, "labels.jsonl")))
    with pytest.raises(DataPreparerOperatorError, match="--dataset-name 'bad_name' must be"):
        R.prepare_round(**proj.corpus_kw(dataset_name="bad_name"))
    with pytest.raises(DataPreparerOperatorError, match="--golden-review needs a local --eval"):
        R.prepare_round(**proj.corpus_kw(golden_review="g.json", eval_data="s3://b/g.jsonl"))


def test_no_seed_confirmed_and_remote_gold(proj):
    plan = R.prepare_round(**proj.dev_kw(eval_data=None, confirm_no_seed=True))
    assert plan.seed_evidence is None and any(
        "WARNING: no labeled seed" in w for w in plan.warnings
    )
    plan = R.prepare_round(**proj.dev_kw(eval_data="s3://b/gold.jsonl"))
    assert plan.eval_uri == "s3://b/gold.jsonl" and plan.gold_text is None
    assert any(w.startswith("NOTE: --eval is an s3:// object") for w in plan.warnings)
    plan = R.prepare_round(**proj.dev_kw(eval_data=_arn("gold")))
    assert plan.eval_arn == _arn("gold")


def test_one_mixed_file_is_never_offered_for_download(proj, monkeypatch):
    # --eval naming the --input object itself: the gold is the corpus, so the note says the
    # job picks the rows with a gold value and never suggests downloading it.
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import (
        inputs as inspect_input,
    )

    uri = "s3://b/mixed.jsonl"
    monkeypatch.setattr(
        inspect_input,
        "read_first_row",
        lambda ref, region=None: ({"id": 0, "text": "t", "gt_label": "a"}, 40, 4000, ref),
    )
    plan = R.prepare_round(**proj.corpus_kw(input_data=uri, eval_data=uri, label_field="gt_label"))
    assert plan.eval_uri == uri and plan.input_s3 == uri and plan.gold_text is None
    (note,) = [w for w in plan.warnings if w.startswith("NOTE: --eval is")]
    assert note.startswith("NOTE: --eval is the --input itself")
    assert "DOWNLOAD" not in note and "never downloaded" in note
    plan = R.prepare_round(
        **proj.corpus_kw(input_data=uri, eval_data="s3://b/gold.jsonl", label_field="gt_label")
    )
    assert any("DOWNLOAD = True" in w for w in plan.warnings)


# ---- the options -------------------------------------------------------------------------------


def test_optimizer_tags_the_gold(tmp_path, monkeypatch):
    proj = _Proj(tmp_path, freeze=True)
    both = _jsonl(
        tmp_path / "work" / "devval.jsonl",
        [json.loads(x) for f in (proj.dev, proj.val) for x in Path(f).read_text().splitlines()],
    )
    fake = _Fake().install(monkeypatch)
    plan, _, outs = _round(
        proj.dev_kw(
            stage="optimize", input_data=both, eval_data=both, optimizer='{"Mode": "protegi"}'
        ),
        fake,
    )
    assert {g["optimizer_split"] for g in plan.gold_rows} == {"dev", "val"}
    assert (
        plan.option_fields == ["Optimizer"]
        and plan.invocation["optimizer"] == '{"Mode": "protegi"}'
    )
    pick = R.round_summary(outs)["optimizer_pick"]
    assert pick["path"].endswith("spec.md") and len(pick["sha256"]) == 64
    assert any(k.endswith("pick.json") for k in outs.texts)
    with pytest.raises(DataPreparerOperatorError, match="pass --stage optimize, not --stage dev"):
        R.prepare_round(**proj.dev_kw(input_data=both, eval_data=both, optimizer="{}"))


def test_recalibration_fit_rows_ride_in_the_gold(tmp_path, monkeypatch):
    proj = _Proj(tmp_path, freeze=True)
    kw = proj.corpus_kw(
        confidence_method="verbalized", confidence_recalibration="platt", recalibration_fit=proj.val
    )
    plan = R.prepare_round(**kw)
    fit = [g for g in plan.gold_rows if g.get("recalibration_fit")]
    assert sorted(g["id"] for g in fit) == ["v0", "v1", "v2", "v3"] and plan.n_fit == 4
    assert plan.models == [MODEL_A]  # verbalized labels with one model
    assert any("--jury-models is ignored" in w for w in plan.warnings)
    # a fit row that is a corpus row stops the run
    _jsonl(proj.corpus, [{"id": "v2", "text": "val row 2"}])
    with pytest.raises(
        DataPreparerOperatorError, match="1 fit row.* are corpus rows .*ids are in --input"
    ):
        R.prepare_round(**kw)
    with pytest.raises(DataPreparerOperatorError, match="runs on the corpus run only"):
        R.prepare_round(**dict(kw, target=None))


def test_fit_rows_in_corpus_people_labeled(tmp_path):
    rows = [{"id": "v1", "provenance": "human_verified"}, {"id": "v2"}]
    hit, why = R.fit_rows_in_corpus("s3://b/c.jsonl", rows, {"v1", "v2"})
    assert hit == {"v1"} and "provenance human_verified/spot_checked" in why
    assert R.fit_rows_in_corpus("s3://b/c.jsonl", [{"id": "v2"}], {"v2"}) == (set(), None)


@pytest.mark.parametrize(
    "kw, match",
    [
        (dict(optimizer="{}", input_data="s3://b/x"), "--optimizer needs a local --input"),
        (
            dict(
                stage="corpus",
                routing=True,
                target=0.9,
                recalibration_fit="f",
                confidence_recalibration="platt",
                eval_data="s3://b/g",
            ),
            "--recalibration-fit needs a local --eval",
        ),
    ],
)
def test_check_option_flags(kw, match):
    args = dict(
        stage="optimize",
        routing=False,
        input_data="x",
        eval_data=None,
        optimizer=None,
        recalibration_fit=None,
        confidence_recalibration=None,
        target=None,
    )
    args.update(kw)
    with pytest.raises(DataPreparerOperatorError, match=match):
        R.check_option_flags(**args)


def test_split_reads(proj):
    reads = R.split_reads(
        stage="dev",
        is_corpus=False,
        local_input=True,
        input_data=proj.dev,
        eval_data=proj.dev,
        demos=proj.dev,
    )
    assert reads == [
        (proj.dev, "demos", {"dev"}),
        (proj.dev, "labeling-dev", {"dev"}),
        (proj.dev, "labeling-dev", {"dev"}),
    ]
    reads = R.split_reads(
        stage="corpus",
        is_corpus=True,
        local_input=True,
        input_data=proj.corpus,
        eval_data=proj.test,
        demos=None,
    )
    assert reads == [(proj.test, "certificate", {"test"})]
    reads = R.split_reads(
        stage="ab-val",
        is_corpus=False,
        local_input=False,
        input_data="s3://x",
        eval_data=proj.val,
        demos="missing.jsonl",
    )
    assert reads == [(proj.val, "labeling-ab-val", {"dev", "val"})]


def test_resolve_task(proj, monkeypatch):
    warnings = []
    assert R.resolve_task("ner", proj.spec, None, warnings) == "ner" and warnings == []
    assert R.resolve_task(None, proj.spec, proj.dev, warnings) == "classification"
    assert warnings[0].startswith("[task-detector] task=classification")
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators.inspect import (
        task as detect_task,
    )

    monkeypatch.setattr(
        detect_task,
        "detect",
        lambda spec, rows: detect_task.TaskDetection(
            task="qa", confidence=0.3, reason="r", signals={}, decision="ask"
        ),
    )
    with pytest.raises(
        DataPreparerOperatorError, match=r"low-confidence \(decision=ask, task='qa'"
    ):
        R.resolve_task(None, None, None, [])


def test_optimizer_config_and_default_run_id():
    assert R.optimizer_config(None) is None and R.optimizer_config('{"Mode": "x"}') == {"Mode": "x"}
    assert re.fullmatch(r"\d{8}T\d{6}-[0-9a-f]{8}", R.default_run_id())


# ---- submission outcomes ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error, match, dropped",
    [
        (
            J.OptionRefused("Optimizer bad"),
            "^CreateJob refused the dev request; the service says: Optimizer bad\n",
            True,
        ),
        (J.CreateJobRefused("nope"), "^nope$", True),
        (J.LabelingBackendUnavailable("HALT"), "^HALT$", True),
        (
            J.JobWaitError("could not describe"),
            r"^SDG labeling job \(dev\) did not complete: could not",
            False,
        ),
        (RuntimeError("timeout"), r"^SDG labeling job \(dev\) did not complete: timeout", False),
    ],
)
def test_submit_refusals(proj, monkeypatch, error, match, dropped):
    _Fake(submit_error=error).install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw(confidence_method="stated"))
    with pytest.raises(DataPreparerOperatorError, match=match):
        R.submit_round(plan, **_shim(plan))
    assert Path(plan.job_record_path).exists() is not dropped


def test_submit_other_errors_propagate(proj, monkeypatch):
    _Fake(submit_error=ValueError("AccessDenied")).install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw())
    with pytest.raises(ValueError, match="AccessDenied"):
        R.submit_round(plan)


def test_failed_job_and_oversized_spec(proj, monkeypatch):
    _Fake(job_status="Failed").install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw())
    with pytest.raises(
        DataPreparerOperatorError, match=r"ended Failed: boom\. Read the job's CloudWatch"
    ):
        R.submit_round(plan, **_shim(plan))
    Path(proj.spec).write_text(SPEC + "x" * (LR.MAX_SPEC_BYTES + 1))
    plan = R.prepare_round(**proj.dev_kw(run_id="run-2"))
    with pytest.raises(
        DataPreparerOperatorError, match="DatasetSpecification.SpecS3Uri .* is too big"
    ):
        R.submit_round(plan, **_shim(plan))


# ---- reading the result -------------------------------------------------------------------------


def test_read_errors(proj, monkeypatch):
    fake = _Fake().install(monkeypatch)
    plan = R.prepare_round(**proj.dev_kw())
    job = R.submit_round(plan, **_shim(plan)).job

    def unreadable(**kw):
        raise FileNotFoundError("no index")

    monkeypatch.setattr(J, "read_job_results", unreadable)
    with pytest.raises(
        DataPreparerOperatorError,
        match="completed but its labeling result could not be read: no index",
    ):
        R.read_round(plan, job)

    def no_labels(**kw):
        res = fake.read(**kw)
        res["artifact_uris"].pop("Labels")
        return res

    monkeypatch.setattr(J, "read_job_results", no_labels)
    with pytest.raises(DataPreparerOperatorError, match="names no labels file"):
        R.read_round(plan, job)


def test_record_failure_is_reported_not_raised(proj, monkeypatch):
    fake = _Fake().install(monkeypatch)

    def broken(**kw):
        raise KeyError("x")

    monkeypatch.setattr(R, "build_job_manifest", broken)
    _, _, outs = _round(proj.dev_kw(), fake)
    assert outs.record is None and outs.run_log_error == "KeyError: 'x'"
    assert R.round_summary(outs)["run_log_error"] == "KeyError: 'x'"


def test_job_summary_warnings(proj, monkeypatch):
    summary = {
        "Status": "Partial",
        "RowsMissing": 2,
        "RowsLabeled": 8,
        "Warnings": ["w1"],
        "ConfidenceMethod": "stated",
        "StatedConfidenceMissing": 4,
        "JurorModelIds": [MODEL_A],
    }
    fake = _Fake(summary=summary).install(monkeypatch)
    _, _, outs = _round(proj.dev_kw(confidence_method="stated"), fake)
    assert outs.warnings[-2].startswith("WARNING: the job is Partial: 2 row(s)")
    assert outs.warnings[-1] == "WARNING (job): w1"
    assert any("STATED CONFIDENCE MISSING on 4/8 rows" in w for w in outs.warnings)
    s = R.round_summary(outs)
    assert s["confidence_method"] == "stated" and s["stated_confidence_missing_rate"] == 0.5


def test_round_summary_places_the_run_log_entry():
    outs = R.RoundOutputs(summary={"stage": "corpus", "unrefined_spec": True, "deliverable": "d"})
    assert list(R.round_summary(outs, "e1", "E")) == [
        "stage",
        "run_log_entry",
        "run_log_error",
        "unrefined_spec",
        "deliverable",
    ]
    assert list(R.round_summary(R.RoundOutputs(summary={"stage": "dev"}), "e1")) == [
        "stage",
        "run_log_entry",
    ]
    assert R.round_summary(R.RoundOutputs(summary={"stage": "dev"})) == {"stage": "dev"}
