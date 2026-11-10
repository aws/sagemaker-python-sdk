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
"""Unit tests for register, dt_handoff and _scratch: resolve in, register out, the DT doors, scratch.

No AWS: the module-level client factories are stubbed. Covers ARN parsing, the three
resolve doors, prefix-to-object resolution, the append guard and consent, the per-run S3
prefix and v4 document shape, the Hub-failure degradation contract, the DT view / output /
promote / import-s3 doors, and the scratch listing and cleanup with a shared view.
"""

from __future__ import absolute_import

import json
import logging

import pytest
from botocore.exceptions import ClientError

from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerOperatorError
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import _scratch
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import dt_handoff
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register

ACCT = "123456789012"
ARN = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/MYHUB/DataSet/support-tickets/1.0.0"
ARN_NOVER = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/MYHUB/DataSet/support-tickets"
LABELED = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/DataSet/tickets/1.0.0"
DT_OUT = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/DataSet/tickets-dt-input/0.0.2"
JOB = f"arn:aws:sagemaker:us-west-2:{ACCT}:job/DataTransformation/"
LABELED_DOC = {
    "DatasetFileFormat": "Custom",
    "DatasetS3Uri": "s3://b/job/output/",
    "DatasetS3Bucket": "b",
    "DatasetS3Prefix": "job/output",
    "ValidationReportS3Uri": "s3://b/job/labeling/report.html",
    "NumValidSamples": 380,
    "JobMetadata": [
        {"JobType": "SyntheticDataGeneration", "MetadataS3Uri": "s3://b/job/labeling/index.json"}
    ],
    "ValidationManifest": {"JobStatus": "SKIPPED"},
    "DatasetType": "SyntheticDataGeneration",
}
DT_DOC = {
    "DatasetFileFormat": "Custom",
    "DatasetS3Uri": "s3://b/dt/out/dataset.jsonl",
    "TransformationReportS3Uri": "s3://b/dt/out/report.json",
    "TransformationJobArn": "arn:dt",
    "ValidationManifest": {"JobStatus": "SKIPPED"},
}


def _nf(op):
    return ClientError({"Error": {"Code": "ResourceNotFound"}}, op)


def _v(n):
    return DT_OUT.replace("/0.0.2", f"/0.0.{n}")


class _FakeSm:
    """Canned describe / import responses and a call log (the register and resolve doors)."""

    def __init__(self, document=None, description=None, exists=True, hub_exists=True):
        self.document = document or {}
        self.description = description
        self.exists = exists
        self.hub_exists = hub_exists
        self.calls = []

    def describe_hub_content(self, **kw):
        self.calls.append(("describe_hub_content", kw))
        if not self.exists:
            raise _nf("DescribeHubContent")
        resp = {
            "HubContentDocument": json.dumps(self.document),
            "HubContentVersion": kw.get("HubContentVersion", "2.0.0"),
        }
        if self.description is not None:
            resp["HubContentDescription"] = self.description
        return resp

    def describe_hub(self, **kw):
        self.calls.append(("describe_hub", kw))
        if not self.hub_exists:
            raise _nf("DescribeHub")
        return {}

    def create_hub(self, **kw):
        self.calls.append(("create_hub", kw))
        self.hub_exists = True
        return {}

    def import_hub_content(self, **kw):
        self.calls.append(("import_hub_content", kw))
        return {"HubContentArn": f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/DataSet/x/1"}

    def imports(self):
        return [kw for c, kw in self.calls if c == "import_hub_content"]


class _FakeS3:
    def __init__(self, listing=None, pages=None):
        self.listing = listing or []
        self.pages = pages
        self.uploads, self.downloads, self.heads = [], [], []

    def get_paginator(self, op):
        assert op == "list_objects_v2"
        pages = self.pages or [self.listing]

        class _Paginator:
            def paginate(_self, Bucket, Prefix):  # noqa: N803
                for page in pages:
                    yield {"Contents": [{"Key": k} for k in page if k.startswith(Prefix)]}

        return _Paginator()

    def download_file(self, bucket, key, local):
        self.downloads.append((bucket, key, local))
        with open(local, "w") as f:
            f.write('{"id": 1, "text": "hi"}\n')

    def head_object(self, Bucket, Key):  # noqa: N803
        self.heads.append((Bucket, Key))
        return {"ETag": '"etag-abc"', "ContentLength": 2048}

    def upload_file(self, local, bucket, key):
        self.uploads.append((local, bucket, key))


class _DtSm:
    """The labeled DataSet as the labeling job registers it, a DT output on a view, names."""

    def __init__(
        self,
        latest="1.0.0",
        existing=("tickets", "tickets-dt-input"),
        view_keywords=None,
        labeled_doc=None,
    ):
        self.latest, self.existing, self.view_keywords = latest, set(existing), view_keywords
        self.labeled_doc = labeled_doc or LABELED_DOC
        self.calls = []

    def describe_hub_content(self, **kw):
        self.calls.append(("describe_hub_content", kw))
        name = kw["HubContentName"]
        if name not in self.existing:
            raise _nf("DescribeHubContent")
        if name.endswith("-dt-input"):
            return {
                "HubContentArn": DT_OUT,
                "HubContentVersion": "0.0.2",
                "DocumentSchemaVersion": "4.0.0",
                "HubContentDocument": json.dumps(DT_DOC),
            }
        version = kw.get("HubContentVersion") or self.latest
        return {
            "HubContentArn": LABELED.replace("/1.0.0", f"/{version}"),
            "HubContentVersion": version,
            "DocumentSchemaVersion": "5.0.0",
            "HubContentDescription": "Auto-labeled dataset from SDG job j",
            "HubContentSearchKeywords": [
                "auto-labeled",
                "sagemaker:hubContent-validationManifest-jobStatus:SKIPPED",
            ],
            "HubContentDocument": json.dumps(self.labeled_doc),
        }

    def list_hub_content_versions(self, **kw):
        self.calls.append(("list_hub_content_versions", kw))
        if self.view_keywords is None:
            raise _nf("ListHubContentVersions")
        return {
            "HubContentSummaries": [
                {"HubContentSearchKeywords": self.view_keywords},
                {"HubContentSearchKeywords": ["sagemaker:x"]},
            ]
        }

    def import_hub_content(self, **kw):
        self.calls.append(("import_hub_content", kw))
        return {
            "HubContentArn": f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/"
            f"DataSet/{kw['HubContentName']}/0.0.9"
        }

    def imports(self):
        return [kw for c, kw in self.calls if c == "import_hub_content"]


class _ViewSm(_DtSm):
    """A view entry shared by two projects: per version, the job that wrote it and its input."""

    def __init__(self, versions, jobs):
        super().__init__()
        self.versions, self.jobs = versions, jobs

    def list_hub_content_versions(self, **kw):
        self.calls.append(("list_hub_content_versions", kw))
        return {"HubContentSummaries": [{"HubContentArn": a} for a in self.versions]}

    def describe_hub_content(self, **kw):
        self.calls.append(("describe_hub_content", kw))
        arn = next(a for a in self.versions if a.endswith("/" + kw["HubContentVersion"]))
        job = self.versions[arn]
        return {
            "HubContentArn": arn,
            "HubContentDocument": json.dumps(
                {"DatasetS3Uri": "s3://b/x.jsonl", **({"TransformationJobArn": job} if job else {})}
            ),
        }

    def describe_job(self, **kw):
        self.calls.append(("describe_job", kw))
        assert kw["JobCategory"] == "DataTransformation"
        return {
            "JobConfigDocument": json.dumps(
                {
                    "InputDataConfig": [
                        {"DatasetArn": self.jobs[kw["JobName"]]},
                        {"DatasetArn": "arn:overlay"},
                    ]
                }
            )
        }


@pytest.fixture
def stub(monkeypatch):
    def _stub(sm=None, s3=None, caller=None, fixed_hub=False):
        sm = sm or _FakeSm()
        s3 = s3 or _FakeS3()
        monkeypatch.setattr(register, "_sm_client", lambda region: sm)
        monkeypatch.setattr(_s3, "s3_client", lambda region: s3)
        monkeypatch.setattr(register, "_sts_account", lambda region: ACCT)
        monkeypatch.setattr(
            register,
            "_sts_caller_arn",
            lambda region: caller or f"arn:aws:sts::{ACCT}:assumed-role/GoodRole/session",
        )
        if fixed_hub:
            monkeypatch.setattr(register, "_resolve_or_create_hub", lambda c, a, r, o=None: "HUB")
        return sm, s3

    return _stub


@pytest.fixture
def labels(tmp_path):
    p = tmp_path / "labels.jsonl"
    p.write_text('{"id": 1, "label": "billing"}\n')
    return str(p)


# ---- ARN parsing and resolve -------------------------------------------------------------


def test_parse_dataset_arn():
    assert register.parse_dataset_arn(ARN) == ("MYHUB", "support-tickets", "1.0.0", "us-west-2")
    assert register.parse_dataset_arn(ARN_NOVER) == ("MYHUB", "support-tickets", None, "us-west-2")
    with pytest.raises(DataPreparerOperatorError, match="expected DataSet"):
        register.parse_dataset_arn(ARN.replace("/DataSet/", "/Model/"))
    with pytest.raises(DataPreparerOperatorError, match="not a hub-content ARN"):
        register.parse_dataset_arn("s3://bucket/key.jsonl")


def test_resolve_local_copies_through_without_aws(tmp_path):
    src, out = tmp_path / "in.jsonl", tmp_path / "corpus.jsonl"
    src.write_text('{"id": 1}\n')
    assert register.resolve_dataset_ref(str(src), str(out), "us-west-2") == {}
    assert out.read_text() == '{"id": 1}\n'
    assert register.resolve_dataset_ref(str(src), str(out), "us-west-2", download=False) == {}


def test_resolve_s3_stops_at_the_uri_by_default(stub, tmp_path):
    _, s3 = stub()
    out = tmp_path / "corpus.jsonl"
    prov = register.resolve_dataset_ref("s3://b/data/corpus.jsonl", str(out), "us-west-2")
    assert s3.downloads == [] and not out.exists()
    assert prov == {"resolved_s3_uri": "s3://b/data/corpus.jsonl", "etag": "etag-abc", "size": 2048}
    assert s3.heads == [("b", "data/corpus.jsonl")]
    assert register.resolve_dataset_ref.__defaults__ == (False,)
    # a remote reference without download needs no out_path at all
    assert (
        register.resolve_dataset_ref("s3://b/data/corpus.jsonl", None, "us-west-2")["size"] == 2048
    )


def test_resolve_s3_downloads_on_request(stub, tmp_path):
    _, s3 = stub()
    out = tmp_path / "seed.jsonl"
    prov = register.resolve_dataset_ref(
        "s3://b/data/seed.jsonl", str(out), "us-west-2", download=True
    )
    assert s3.downloads[0][:2] == ("b", "data/seed.jsonl")
    assert prov == {"resolved_s3_uri": "s3://b/data/seed.jsonl"}
    assert out.exists()


def test_resolve_arn_pinned_and_unpinned(stub, tmp_path):
    sm, s3 = stub(
        _FakeSm(document={"DatasetS3Uri": "s3://b/reg/data.jsonl", "DatasetFileFormat": "Custom"})
    )
    prov = register.resolve_dataset_ref(ARN, str(tmp_path / "c.jsonl"), "us-west-2", download=True)
    assert sm.calls[0][1]["HubContentVersion"] == "1.0.0"
    assert prov == {
        "source_arn": ARN,
        "source_name": "support-tickets",
        "resolved_version": "1.0.0",
        "dataset_file_format": "Custom",
        "resolved_s3_uri": "s3://b/reg/data.jsonl",
    }
    assert s3.downloads
    sm, s3 = stub(
        _FakeSm(document={"DatasetS3Bucket": "b", "DatasetS3Prefix": "byod/x"}),
        _FakeS3(listing=["byod/x/part-0.jsonl"]),
    )
    prov = register.resolve_dataset_ref(ARN_NOVER, None, "us-west-2")
    assert "HubContentVersion" not in sm.calls[0][1]
    assert prov["resolved_s3_uri"] == "s3://b/byod/x/part-0.jsonl"
    assert prov["etag"] == "etag-abc"
    assert s3.downloads == []


def test_resolve_arn_uses_the_arns_own_region(stub, monkeypatch, tmp_path):
    seen = []
    sm, s3 = stub(_FakeSm(document={"DatasetS3Uri": "s3://b/k/data.jsonl"}))
    monkeypatch.setattr(register, "_sm_client", lambda region: seen.append(("sm", region)) or sm)
    monkeypatch.setattr(_s3, "s3_client", lambda region: seen.append(("s3", region)) or s3)
    east = ARN.replace(":us-west-2:", ":us-east-1:")
    register.resolve_dataset_ref(east, str(tmp_path / "c.jsonl"), "us-west-2", download=True)
    assert seen and all(r == "us-east-1" for _, r in seen)


def test_resolve_document_without_location_refuses(stub):
    stub(_FakeSm(document={"NumValidSamples": 3}))
    with pytest.raises(DataPreparerOperatorError, match="neither DatasetS3Uri"):
        register.resolve_dataset_ref(ARN, None, "us-west-2")


def test_resolve_prefix_needs_exactly_one_jsonl(stub):
    stub(s3=_FakeS3(listing=["p/a.jsonl", "p/b.jsonl"]))
    with pytest.raises(DataPreparerOperatorError, match="exactly one .jsonl"):
        _s3.resolve_object("s3://b/p", "us-west-2")
    stub(s3=_FakeS3(pages=[["p/noise0.bin", "p/noise1.bin"], ["p/data.jsonl", "p/more.bin"]]))
    assert _s3.resolve_object("s3://b/p/", "us-west-2") == "s3://b/p/data.jsonl"
    # a bucket root (a document with DatasetS3Bucket and no prefix) lists the bucket
    stub(s3=_FakeS3(pages=[["data.jsonl", "noise.bin"]]))
    assert _s3.resolve_object("s3://b/", "us-west-2") == "s3://b/data.jsonl"
    assert _s3.resolve_object("s3://b", "us-west-2") == "s3://b/data.jsonl"
    with pytest.raises(DataPreparerOperatorError, match="not an s3:// URI"):
        _s3.resolve_object("http://b/p/", "us-west-2")


def test_head_object_is_best_effort(stub):
    class _Boom(_FakeS3):
        def head_object(self, **kw):
            raise RuntimeError("no HeadObject permission")

    stub(s3=_Boom())
    assert register.resolve_dataset_ref("s3://b/d.jsonl", None, "us-west-2") == {
        "resolved_s3_uri": "s3://b/d.jsonl"
    }


def test_corpus_ref_document():
    prov = {
        "resolved_s3_uri": "s3://b/d.jsonl",
        "etag": "e",
        "size": 5,
        "source_arn": ARN,
        "resolved_version": "1.0.0",
    }
    assert register.corpus_ref_document(prov, resolved_at="2026-01-01T00:00:00Z") == {
        "source_s3_uri": "s3://b/d.jsonl",
        "etag": "e",
        "size": 5,
        "hub_arn": ARN,
        "hub_version": "1.0.0",
        "resolved_at": "2026-01-01T00:00:00Z",
    }
    doc = register.corpus_ref_document({"resolved_s3_uri": "s3://b/d.jsonl"})
    assert doc["hub_arn"] is None and doc["resolved_at"].endswith("Z")


# ---- register ----------------------------------------------------------------------------


def _desc(source=ARN, labels=("billing", "refund")):
    return register._build_description(
        source, register._label_set_digest(list(labels)) if labels else None, "OLD"
    )


def test_register_document_shape_and_per_run_prefix(stub, labels, tmp_path):
    sm, s3 = stub(_FakeSm(exists=False))
    rep = tmp_path / "held_out_report.json"
    rep.write_text("{}")
    res = register.register_labeled_output(
        labels_path=labels,
        name="support-tickets",
        s3_prefix="s3://b/autolabel",
        region="us-west-2",
        report_path=str(rep),
        provenance={"source_arn": ARN, "source_name": "Support-Tickets"},
        label_set=["refund", "billing"],
        run_id="RUN1",
    )
    assert res["hub_registered"]
    assert res["labels_s3_uri"] == "s3://b/autolabel/labeled/support-tickets/RUN1/labels.jsonl"
    assert res["report_s3_uri"] == res["report_json_s3_uri"]
    assert res["report_html_s3_uri"] is None
    (imp,) = sm.imports()
    assert imp["HubContentName"] == "support-tickets-labeled"
    assert imp["DocumentSchemaVersion"] == "4.0.0"
    assert "HubContentVersion" not in imp
    doc = json.loads(imp["HubContentDocument"])
    assert sorted(doc) == [
        "DatasetFileFormat",
        "DatasetS3Bucket",
        "DatasetS3Prefix",
        "DatasetS3Uri",
        "ValidationReportS3Uri",
    ]
    assert doc["DatasetFileFormat"] == "Custom"
    assert doc["DatasetS3Prefix"] == "autolabel/labeled/support-tickets/RUN1"
    digest = register._label_set_digest(["refund", "billing"])
    desc = imp["HubContentDescription"]
    assert desc.startswith("auto-labeled") and f"source={ARN}" in desc
    assert f"labels-sha256={digest}" in desc and "run-id=RUN1" in desc
    assert imp["HubContentSearchKeywords"] == [
        "auto-labeled",
        f"labels-sha256:{digest}",
        "source-name:support-tickets",
    ]
    assert len(s3.uploads) == 2


def test_register_keywords_without_a_source_name(stub, labels):
    sm, _ = stub(_FakeSm(exists=False))
    register.register_labeled_output(
        labels_path=labels,
        name="x",
        s3_prefix="s3://b/p",
        region="us-west-2",
        provenance={"source_arn": ARN},
        run_id="R",
        extra_doc={"NumValidSamples": 3},
    )
    (imp,) = sm.imports()
    assert imp["HubContentSearchKeywords"] == ["auto-labeled", "source:hub-dataset"]
    assert json.loads(imp["HubContentDocument"])["NumValidSamples"] == 3


def test_register_html_report_gets_the_pointer(stub, labels, tmp_path):
    sm, s3 = stub(_FakeSm(exists=False))
    rep, html = tmp_path / "held_out_report.json", tmp_path / "report.html"
    rep.write_text("{}")
    html.write_text("<html/>")
    res = register.register_labeled_output(
        labels_path=labels,
        name="x",
        s3_prefix="s3://b/p",
        region="us-west-2",
        report_path=str(rep),
        report_html_path=str(html),
        run_id="RUN1",
    )
    assert res["report_s3_uri"] == "s3://b/p/labeled/x/RUN1/report.html"
    assert res["report_json_s3_uri"] == "s3://b/p/labeled/x/RUN1/held_out_report.json"
    doc = json.loads(sm.imports()[0]["HubContentDocument"])
    assert doc["ValidationReportS3Uri"] == "s3://b/p/labeled/x/RUN1/report.html"
    assert len(s3.uploads) == 3


def test_register_generates_a_run_id(stub, labels):
    stub(_FakeSm(exists=False))
    res = register.register_labeled_output(
        labels_path=labels, name="x", s3_prefix="s3://b/p", region="us-west-2"
    )
    assert res["run_id"] in res["labels_s3_uri"] and len(res["run_id"]) == 22


@pytest.mark.parametrize(
    "description,kw,match",
    [
        (_desc(source="arn:other"), {"provenance": {"source_arn": ARN}}, "different source"),
        (
            _desc(labels=("cats", "dogs")),
            {"label_set": ["refund", "billing"]},
            "different label set",
        ),
        ("Synthetic dataset generated by SDG job: x", {}, "not registered by this skill"),
        (
            _desc(),
            {"provenance": {"source_arn": ARN}, "label_set": ["refund", "billing"]},
            "--append",
        ),
    ],
)
def test_append_guard_and_consent_refuse_before_any_upload(stub, labels, description, kw, match):
    sm, s3 = stub(_FakeSm(description=description))
    with pytest.raises(DataPreparerOperatorError, match=match) as info:
        register.register_labeled_output(
            labels_path=labels,
            name="support-tickets",
            s3_prefix="s3://b/p",
            region="us-west-2",
            run_id="RUN1",
            **kw,
        )
    assert "nothing was uploaded" in str(info.value)
    assert s3.uploads == [] and not sm.imports()
    if match == "--append":
        assert "2.0.0" in str(info.value)


@pytest.mark.parametrize("source", [ARN, ARN.replace("/1.0.0", "/2.0.0"), ARN_NOVER])
def test_append_with_consent_is_version_insensitive_on_the_source(stub, labels, source):
    stub(_FakeSm(description=_desc()))
    res = register.register_labeled_output(
        labels_path=labels,
        name="support-tickets",
        s3_prefix="s3://b/p",
        region="us-west-2",
        allow_append=True,
        provenance={"source_arn": source},
        label_set=["refund", "billing"],
        run_id="RUN1",
    )
    assert res["hub_registered"]


def test_append_refuses_another_dataset_even_with_consent(stub, labels):
    _, s3 = stub(_FakeSm(description=_desc()))
    with pytest.raises(DataPreparerOperatorError, match="different source"):
        register.register_labeled_output(
            labels_path=labels,
            name="support-tickets",
            s3_prefix="s3://b/p",
            region="us-west-2",
            allow_append=True,
            provenance={"source_arn": ARN.replace("support-tickets", "other-corpus")},
            label_set=["refund", "billing"],
            run_id="RUN3",
        )
    assert s3.uploads == []


def test_bad_hub_override_fails_loud_before_any_upload(stub, labels):
    sm, s3 = stub(_FakeSm(hub_exists=False))
    with pytest.raises(DataPreparerOperatorError, match="NopeHub"):
        register.register_labeled_output(
            labels_path=labels,
            name="support-tickets",
            s3_prefix="s3://b/p",
            region="us-west-2",
            hub_name_override="NopeHub",
            run_id="RUN1",
        )
    assert s3.uploads == [] and "create_hub" not in [c for c, _ in sm.calls]


def test_canonical_hub_created_when_missing(stub, labels):
    sm, _ = stub(_FakeSm(hub_exists=False))
    name = register._resolve_or_create_hub(sm, ACCT, "us-west-2", None)
    assert name == register._canonical_ai_registry_hub_name(ACCT, "us-west-2")
    create = [kw for c, kw in sm.calls if c == "create_hub"]
    assert create == [
        {
            "HubName": name,
            "HubDescription": f"AiRegistry-us-west-2-{ACCT}",
            "HubDisplayName": f"AiRegistry-us-west-2-{ACCT}",
            "HubSearchKeywords": ["AiRegistry"],
        }
    ]
    assert register._resolve_or_create_hub(sm, ACCT, "us-west-2", "Mine") == "Mine"


def test_invalid_name_fails_before_any_call(stub, labels):
    sm, s3 = stub()
    with pytest.raises(DataPreparerOperatorError, match="--name work-level") as info:
        register.register_labeled_output(
            labels_path=labels, name="work_level", s3_prefix="s3://b/p", region="us-west-2"
        )
    assert "work_level-labeled" in str(info.value)
    assert s3.uploads == [] and sm.calls == []
    assert register._validate_content_name("work-level") == "work-level-labeled"
    with pytest.raises(DataPreparerOperatorError):
        register._validate_content_name("bad name!")


def test_hub_unreachable_at_the_precheck_skips_registration(stub, labels, caplog):
    class _BoomSm(_FakeSm):
        def describe_hub(self, **kw):
            raise RuntimeError("hub api down")

    sm, s3 = stub(_BoomSm())
    with caplog.at_level(logging.WARNING, logger=register.__name__):
        res = register.register_labeled_output(
            labels_path=labels, name="x", s3_prefix="s3://b/p", region="us-west-2", run_id="RUN1"
        )
    assert res["hub_registered"] is False
    assert "pre-check" in res["hub_error"] and "RuntimeError: hub api down" in res["hub_error"]
    assert len(s3.uploads) == 1
    assert "import_hub_content" not in [c for c, _ in sm.calls]
    assert "labeled data is in S3" in caplog.text


def test_hub_import_failure_degrades_after_the_guard_passes(stub, labels, caplog):
    class _ImportBoomSm(_FakeSm):
        def import_hub_content(self, **kw):
            self.calls.append(("import_hub_content", kw))
            raise RuntimeError("import failed")

    sm, s3 = stub(_ImportBoomSm(exists=False))
    with caplog.at_level(logging.WARNING, logger=register.__name__):
        res = register.register_labeled_output(
            labels_path=labels, name="x", s3_prefix="s3://b/p", region="us-west-2", run_id="RUN1"
        )
    assert res["hub_registered"] is False
    assert res["hub_error"] == "RuntimeError: import failed"
    assert len(s3.uploads) == 1 and sm.imports()
    assert "Hub registration failed" in caplog.text


def test_admin_caller_logs_the_manifest_note(stub, labels, caplog):
    stub(_FakeSm(exists=False), caller=f"arn:aws:sts::{ACCT}:assumed-role/Admin/me")
    with caplog.at_level(logging.WARNING, logger=register.__name__):
        res = register.register_labeled_output(
            labels_path=labels, name="x", s3_prefix="s3://b/p", region="us-west-2", run_id="RUN1"
        )
    assert res["hub_registered"]
    assert "AmazonSageMakerBedrockExecutionRole" in caplog.text


def test_label_set_digest_is_order_insensitive():
    assert register._label_set_digest(["b", "a"]) == register._label_set_digest(["a", "b"])
    assert register._label_set_digest(["a"]) != register._label_set_digest(["a", "b"])
    assert len(register._label_set_digest(["a"])) == 12


def test_default_region(monkeypatch):
    import boto3.session

    class _Sess:
        region_name = "us-east-1"

    monkeypatch.setattr(boto3.session, "Session", lambda: _Sess())
    assert register._default_region() == "us-east-1"
    _Sess.region_name = None
    assert register._default_region() == "us-west-2"


# ---- DT doors --------------------------------------------------------------------------


@pytest.fixture
def dt(stub):
    def _dt(sm=None, keys=("job/output/labels.jsonl",)):
        return stub(sm or _DtSm(), _FakeS3(listing=list(keys)), fixed_hub=True)[0]

    return _dt


def test_promote_keeps_the_jobs_document_and_takes_the_transformation_fields(dt):
    sm = dt()
    out = dt_handoff.promote_transform_output(from_arn=DT_OUT, to_arn=LABELED)
    (imp,) = sm.imports()
    assert (imp["HubName"], imp["HubContentName"]) == ("HUB", "tickets")
    assert "HubContentVersion" not in imp
    assert imp["DocumentSchemaVersion"] == "5.0.0"
    doc = json.loads(imp["HubContentDocument"])
    assert doc["DatasetS3Uri"] == "s3://b/dt/out/dataset.jsonl"
    assert (doc["DatasetS3Bucket"], doc["DatasetS3Prefix"]) == ("b", "dt/out")
    assert doc["TransformationJobArn"] == "arn:dt"
    assert doc["TransformationReportS3Uri"] == "s3://b/dt/out/report.json"
    assert doc["JobMetadata"] == LABELED_DOC["JobMetadata"]
    assert doc["ValidationReportS3Uri"] == LABELED_DOC["ValidationReportS3Uri"]
    assert "ValidationManifest" not in doc
    # the job's own; a customer import carrying SyntheticDataGeneration is refused
    assert "DatasetType" not in doc
    assert imp["HubContentSearchKeywords"] == ["auto-labeled"]
    assert imp["HubContentDescription"] == "Auto-labeled dataset from SDG job j"
    assert out == {
        "dataset_arn": f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/DataSet/tickets/0.0.9",
        "source_arn": LABELED,
        "transform_output_arn": DT_OUT,
        "dataset_s3_uri": "s3://b/dt/out/dataset.jsonl",
    }
    assert {c for c, _ in sm.calls} == {"describe_hub_content", "import_hub_content"}


def test_a_later_promote_replaces_the_transformation_fields(dt):
    doc = dict(
        LABELED_DOC,
        DatasetS3Uri="s3://b/dt/old/dataset.jsonl",
        TransformationJobArn="arn:old",
        TransformationOldField="x",
    )
    sm = dt(_DtSm(labeled_doc=doc))
    dt_handoff.promote_transform_output(from_arn=DT_OUT, to_arn=LABELED)
    new = json.loads(sm.imports()[0]["HubContentDocument"])
    assert new["TransformationJobArn"] == "arn:dt" and "TransformationOldField" not in new
    assert new["DatasetS3Uri"] == "s3://b/dt/out/dataset.jsonl"


def test_promote_refuses_when_someone_added_a_version(dt):
    sm = dt(_DtSm(latest="3.0.0"))
    with pytest.raises(DataPreparerOperatorError, match="at version 3.0.0"):
        dt_handoff.promote_transform_output(from_arn=DT_OUT, to_arn=LABELED)
    assert not sm.imports()


def test_promote_refuses_an_output_that_is_not_on_the_view(dt):
    sm = dt()
    with pytest.raises(DataPreparerOperatorError, match="not a version of 'tickets-dt-input'"):
        dt_handoff.promote_transform_output(from_arn=LABELED, to_arn=LABELED)
    assert not sm.calls


def test_dt_view_points_at_the_jsonl_under_the_job_folder(dt):
    sm = dt()
    out = dt_handoff.dt_view(arn=LABELED)
    (imp,) = sm.imports()
    assert imp["HubContentName"] == "tickets-dt-input"
    assert imp["HubContentSearchKeywords"] == [dt_handoff.DT_VIEW_KEYWORD]
    assert imp["HubContentDescription"] == f"dataset-transformation input for {LABELED}"
    assert json.loads(imp["HubContentDocument"]) == {
        "DatasetFileFormat": "Custom",
        "DatasetS3Uri": "s3://b/job/output/labels.jsonl",
        "DatasetS3Bucket": "b",
        "DatasetS3Prefix": "job/output",
    }
    assert (
        out["source_arn"] == LABELED and out["dataset_s3_uri"] == "s3://b/job/output/labels.jsonl"
    )


def test_dt_view_name_fits_the_hub_limit(dt):
    sm = dt(_DtSm(existing=("t" * 60,)))
    dt_handoff.dt_view(arn=LABELED.replace("/tickets/", "/" + "t" * 60 + "/"))
    (imp,) = sm.imports()
    assert len(imp["HubContentName"]) <= 63 and imp["HubContentName"].endswith("-dt-input")
    assert dt_handoff._view_name("a" * 70) == "a" * 54 + "-dt-input"


def test_dt_view_reuses_its_own_view_and_refuses_someone_elses(dt):
    sm = dt(_DtSm(view_keywords=[dt_handoff.DT_VIEW_KEYWORD]))
    dt_handoff.dt_view(arn=LABELED)
    assert len(sm.imports()) == 1
    sm = dt(_DtSm(view_keywords=["customer"]))
    with pytest.raises(DataPreparerOperatorError, match="not registered by this skill"):
        dt_handoff.dt_view(arn=LABELED)
    assert not sm.imports()


def test_dt_view_of_an_s3_object(dt):
    sm = dt(keys=[])
    with pytest.raises(DataPreparerOperatorError, match="needs --name"):
        dt_handoff.dt_view(s3_uri="s3://b/corpus/corpus.jsonl")
    out = dt_handoff.dt_view(
        s3_uri="s3://b/corpus/corpus.jsonl", name="tickets", region="us-west-2"
    )
    (imp,) = sm.imports()
    assert (imp["HubName"], imp["HubContentName"]) == ("HUB", "tickets-dt-input")
    assert out["dataset_s3_uri"] == "s3://b/corpus/corpus.jsonl"
    with pytest.raises(DataPreparerOperatorError, match="exactly one of"):
        dt_handoff.dt_view(arn=LABELED, s3_uri="s3://b/x.jsonl")
    with pytest.raises(DataPreparerOperatorError, match="exactly one of"):
        dt_handoff.dt_view()


def test_dt_view_refuses_a_compressed_object(dt):
    dt(keys=["job/output/labels.jsonl.gz"])
    with pytest.raises(DataPreparerOperatorError, match="exactly one .jsonl"):
        dt_handoff.dt_view(arn=LABELED)


def test_register_transform_output_under_a_new_name(dt):
    sm = dt()
    out = dt_handoff.register_transform_output(
        from_arn=DT_OUT, name="tickets-merged", source_arn=LABELED
    )
    (imp,) = sm.imports()
    assert imp["HubContentName"] == "tickets-merged"
    assert imp["HubContentSearchKeywords"] == ["auto-labeled", "derived"]
    assert imp["HubContentDescription"] == f"derived by arn:dt from {LABELED}"
    doc = json.loads(imp["HubContentDocument"])
    assert doc["DatasetS3Uri"] == "s3://b/dt/out/dataset.jsonl"
    assert doc["TransformationJobArn"] == "arn:dt" and "ValidationManifest" not in doc
    assert out == {
        "dataset_arn": f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/HUB/DataSet/tickets-merged/0.0.9",
        "dataset_name": "tickets-merged",
        "hub_name": "HUB",
        "registered_from": DT_OUT,
        "dataset_s3_uri": "s3://b/dt/out/dataset.jsonl",
    }


def test_register_transform_output_refuses_an_existing_or_bad_name(dt):
    sm = dt()
    with pytest.raises(DataPreparerOperatorError, match="already exists"):
        dt_handoff.register_transform_output(from_arn=DT_OUT, name="tickets")
    with pytest.raises(DataPreparerOperatorError, match="not a valid Hub content name"):
        dt_handoff.register_transform_output(from_arn=DT_OUT, name="bad_name")
    assert not sm.imports()


def test_register_s3_object_without_reading_and_refuses_a_taken_name(dt):
    sm = dt(keys=["corpus/part.jsonl"])
    out = dt_handoff.register_s3_object(
        s3_uri="s3://b/corpus/", name="tickets-corpus", region="us-west-2"
    )
    with pytest.raises(DataPreparerOperatorError, match="already exists"):
        dt_handoff.register_s3_object(s3_uri="s3://b/corpus/", name="tickets", region="us-west-2")
    (imp,) = sm.imports()
    assert json.loads(imp["HubContentDocument"]) == {
        "DatasetFileFormat": "Custom",
        "DatasetS3Uri": "s3://b/corpus/part.jsonl",
        "DatasetS3Bucket": "b",
        "DatasetS3Prefix": "corpus",
    }
    assert imp["HubContentSearchKeywords"] == ["auto-labeling-input"]
    assert out["dataset_name"] == "tickets-corpus" and out["hub_name"] == "HUB"
    assert not hasattr(_FakeS3, "get_object")


# ---- scratch ---------------------------------------------------------------------------


def _entries(*recorded, with_ids=True):
    entries = []
    for i, (arn, stage) in enumerate(recorded):
        e = _scratch.scratch_event(arn, stage)
        if with_ids:
            e.update(id=f"e{i}", created_at=f"2026-01-0{i + 1}T00:00:00Z")
        entries.append(e)
    return entries


def test_scratch_event_and_entries():
    ev = _scratch.scratch_event(DT_OUT, "dt-view")
    assert ev == {
        "kind": "artifact",
        "step": "auto-label",
        "iteration": 0,
        "status": "scratch",
        "arn": DT_OUT,
        "labeling_run": True,
        "invocation": {"stage": "dt-view"},
    }
    entries = _entries((DT_OUT, "dt-view")) + [
        {"kind": "job", "status": "scratch", "labeling_run": True, "arn": "x"},
        {"kind": "artifact", "status": "deleted", "labeling_run": True},
        {"kind": "artifact", "status": "scratch", "arn": "not-a-labeling-run"},
        {
            "kind": "artifact",
            "status": "scratch",
            "labeling_run": True,
            "arn": "a",
            "job_arn": "j",
            "invocation": {"stage": "refine", "run_id": "r1"},
            "request": {"dataset_name": "d"},
        },
    ]
    assert _scratch.scratch_entries(entries) == [
        {
            "id": "e0",
            "arn": DT_OUT,
            "job_arn": None,
            "stage": "dt-view",
            "run_id": None,
            "dataset_name": None,
            "created_at": "2026-01-01T00:00:00Z",
        },
        {
            "id": None,
            "arn": "a",
            "job_arn": "j",
            "stage": "refine",
            "run_id": "r1",
            "dataset_name": "d",
            "created_at": None,
        },
    ]


def test_scratch_cleanup_lists_without_delete(dt, monkeypatch):
    dt()
    deleted = []
    monkeypatch.setattr(register, "delete_hub_content_version", deleted.append)
    res = _scratch.scratch_cleanup(_entries((DT_OUT, "dt-output")), project="/p")
    assert res.to_json() == {
        "project": "/p",
        "scratch": _scratch.scratch_entries(_entries((DT_OUT, "dt-output"))),
        "deleted": [],
        "not_found": [],
        "kept": [],
        "errors": [],
    }
    assert deleted == [] and res.events == []


def _cleanup(stub, monkeypatch, sm, *recorded, deleter=None):
    stub(sm, _FakeS3(), fixed_hub=True)
    deleted = []
    monkeypatch.setattr(register, "delete_hub_content_version", deleter or deleted.append)
    res = _scratch.scratch_cleanup(_entries(*recorded), delete=True, project="/p")
    return deleted, res


def test_cleanup_deletes_the_view_and_its_jobs_outputs_not_another_projects(stub, monkeypatch):
    # 0.0.1 our view, 0.0.2 another project's view, 0.0.3 our job's output,
    # 0.0.4 the other project's job's output, 0.0.10 ours again (version order)
    sm = _ViewSm(
        {
            _v(1): None,
            _v(2): None,
            _v(3): JOB + "ours",
            _v(4): JOB + "theirs",
            _v(10): JOB + "ours2",
        },
        {"ours": _v(1), "theirs": _v(2), "ours2": _v(1)},
    )
    deleted, res = _cleanup(stub, monkeypatch, sm, (_v(1), "dt-view"))
    assert deleted == [_v(10), _v(3), _v(1)] == res.deleted
    assert sorted(k["arn"] for k in res.kept) == [_v(2), _v(4)]
    reasons = {k["arn"]: k["reason"] for k in res.kept}
    assert reasons[_v(2)] == "another view version"
    assert reasons[_v(4)] == "the output of a job on another view version"
    assert res.events == [
        {
            "kind": "artifact",
            "step": "auto-label",
            "iteration": 0,
            "status": "deleted",
            "scratch_cleanup": True,
            "deleted": [_v(10), _v(3), _v(1)],
        }
    ]
    assert "events" not in res.to_json()


def test_cleanup_a_second_view_of_the_same_dataset_is_not_listed_as_kept(stub, monkeypatch):
    sm = _ViewSm(
        {_v(1): None, _v(2): None, _v(3): JOB + "a", _v(4): JOB + "b"}, {"a": _v(1), "b": _v(2)}
    )
    deleted, res = _cleanup(stub, monkeypatch, sm, (_v(1), "dt-view"), (_v(2), "dt-view"))
    assert sorted(deleted) == sorted([_v(1), _v(2), _v(3), _v(4)])
    assert res.kept == []


def test_cleanup_keeps_an_output_that_cannot_be_checked(stub, monkeypatch):
    sm = _ViewSm({_v(1): None, _v(3): "arn:aws:sagemaker:us-west-2:1:job/DataTransformation/x"}, {})
    deleted, res = _cleanup(stub, monkeypatch, sm, (_v(1), "dt-view"))
    assert deleted == [_v(1)]
    assert res.kept[0]["reason"].startswith("not checked (KeyError")


def test_cleanup_a_view_already_gone_falls_back_to_its_recorded_version(stub, monkeypatch):
    sm = _DtSm()
    deleted, res = _cleanup(stub, monkeypatch, sm, (DT_OUT, "dt-view"))
    assert deleted == [DT_OUT]
    assert [c for c, _ in sm.calls] == ["list_hub_content_versions"]
    # a non-view entry is deleted as recorded, without listing
    sm = _DtSm()
    deleted, res = _cleanup(stub, monkeypatch, sm, (LABELED, "dt-output"))
    assert deleted == [LABELED] and sm.calls == []


def test_cleanup_reports_not_found_errors_and_missing_arns(stub, monkeypatch):
    def deleter(arn):
        if arn == _v(1):
            raise RuntimeError("ResourceNotFound: gone")
        raise RuntimeError("AccessDenied")

    entries = _entries((_v(1), "dt-output"), (_v(2), "dt-output"))
    entries.append({**_scratch.scratch_event(None, "dt-view"), "id": "e9"})
    stub(_DtSm(), _FakeS3(), fixed_hub=True)
    monkeypatch.setattr(register, "delete_hub_content_version", deleter)
    res = _scratch.scratch_cleanup(entries, delete=True)
    assert res.deleted == [] and res.events == []
    assert res.not_found == [{"arn": _v(1), "error": "RuntimeError: ResourceNotFound: gone"}]
    assert res.errors == [
        {"id": "e9", "error": "no DataSet ARN recorded"},
        {"arn": _v(2), "error": "RuntimeError: AccessDenied"},
    ]
    assert res.project is None and "project" not in res.to_json()


def test_cleanup_reports_a_view_listing_failure(stub, monkeypatch):
    class _Boom(_DtSm):
        def list_hub_content_versions(self, **kw):
            raise RuntimeError("throttled")

    deleted, res = _cleanup(stub, monkeypatch, _Boom(), (_v(1), "dt-view"))
    assert deleted == [] and res.errors == [{"arn": _v(1), "error": "RuntimeError: throttled"}]


def test_view_versions_paginates_and_ignores_non_views(stub):
    class _Paged(_ViewSm):
        def list_hub_content_versions(self, **kw):
            self.calls.append(("list_hub_content_versions", kw))
            if "NextToken" not in kw:
                return {"HubContentSummaries": [{"HubContentArn": _v(1)}], "NextToken": "t"}
            return {"HubContentSummaries": [{"HubContentArn": _v(3)}, {}]}

    sm = _Paged({_v(1): None, _v(3): JOB + "a"}, {"a": _v(1)})
    stub(sm, _FakeS3(), fixed_hub=True)
    assert _scratch._view_versions(_v(1)) == ([_v(3), _v(1)], [])
    assert _scratch._view_versions(LABELED) == ([], [])
    assert _scratch._version_key(_v(10)) == [0, 0, 10] and _scratch._version_key("x/1.a") == [1, 0]
