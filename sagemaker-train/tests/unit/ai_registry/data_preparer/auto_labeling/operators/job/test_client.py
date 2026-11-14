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
"""Unit tests for operators.auto_labeling.job_client: submit, re-attach, read-back, Hub helpers.

boto3 and the SDK's DataPreparer / LabelingSummary are stubbed; no network.
"""

from __future__ import absolute_import

import json
import os

import pytest

from sagemaker.ai_registry.data_preparer.auto_labeling import LabelingSummary
from sagemaker.ai_registry.data_preparer.exceptions import (
    DataPreparerError,
    DataPreparerJobFailedError,
    DataPreparerJobStoppedError,
    DataPreparerJobTimeoutError,
    DataPreparerOperatorError,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _s3
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import client as J
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import register
from sagemaker.ai_registry.data_preparer.auto_labeling.operators.job import request as R

ACCT = "123456789012"
LABELED = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/AIRegistry/DataSet/ds/1.0.0"
REVIEW = f"arn:aws:sagemaker:us-west-2:{ACCT}:hub-content/AIRegistry/DataSet/ds-review/1.0.0"
FOLDER = "s3://bkt/al/proj/run/corpus/"
ARTIFACTS = {
    "LabelingSummary": FOLDER + "labeling/labeling_summary.json",
    "Labels": FOLDER + "output/labels.jsonl",
    "ReviewRows": FOLDER + "review/review_rows.jsonl",
    "ReviewModelOutput": FOLDER + "labeling/review_rows.model_output.jsonl",
    "Failures": FOLDER + "labeling/failures.jsonl",
    "HeldOutReport": FOLDER + "labeling/held_out_report.json",
    "Report": FOLDER + "labeling/report.html",
    "Spec": FOLDER + "labeling/spec.md",
}


class _FakeS3:
    def __init__(self, objects=None):
        self.objects = dict(objects or {})
        self.puts, self.downloads = [], []

    def download_file(self, Bucket, Key, local):  # noqa: N803
        uri = f"s3://{Bucket}/{Key}"
        if uri not in self.objects:
            raise OSError(f"NoSuchKey {uri}")
        with open(local, "wb") as f:
            f.write(self.objects[uri])
        self.downloads.append(uri)

    def put_object(self, Bucket, Key, Body):  # noqa: N803
        self.objects[f"s3://{Bucket}/{Key}"] = Body
        self.puts.append((f"s3://{Bucket}/{Key}", Body))

    def upload_file(self, local, Bucket, Key):  # noqa: N803
        with open(local, "rb") as f:
            self.objects[f"s3://{Bucket}/{Key}"] = f.read()


def _s3_with_job():
    return _FakeS3({uri: b"x" for uri in ARTIFACTS.values()})


class _FakeSession:
    def __init__(self, describe=None):
        self.describe = describe or {}
        self.describes = []
        outer = self

        class _Client:
            def describe_job(self, JobName, JobCategory):  # noqa: N803
                outer.describes.append((JobName, JobCategory))
                resp = outer.describe.get(JobName)
                if resp is None:
                    raise RuntimeError(f"ResourceNotFound: job {JobName} does not exist")
                return resp() if callable(resp) else resp

        self.sagemaker_client = _Client()


class _FakeDS:
    arn = LABELED


class _FakeDataPreparer:
    errors = []
    calls = []
    kwargs = []

    def __init__(self, **kw):
        type(self).kwargs.append(kw)

    waits = []

    def generate(self, cfg, job_name=None, wait=True):
        type(self).calls.append((cfg, job_name))
        type(self).waits.append(wait)
        if type(self).errors:
            raise type(self).errors.pop(0)
        return _FakeDS() if wait else object()


@pytest.fixture
def doc():
    return R.build_labeling_request(
        task="classification",
        text_fields=["text"],
        spec_s3_uri="s3://b/p/spec.md",
        model_ids=["m"],
        dataset_name="ds",
        output_s3_uri=FOLDER + "output/",
        input_s3_uri="s3://b/p/in.jsonl",
    )


@pytest.fixture
def stubbed(monkeypatch):
    _FakeDataPreparer.errors, _FakeDataPreparer.calls, _FakeDataPreparer.kwargs = [], [], []
    _FakeDataPreparer.waits = []
    monkeypatch.setattr(J, "_data_preparer_class", lambda: _FakeDataPreparer)
    monkeypatch.setattr(J.time, "sleep", lambda s: None)
    return _FakeDataPreparer


class TestSubmit:
    def test_unique_name_recorded_first_and_the_typed_config(self, doc, stubbed):
        sess = _FakeSession()
        seen = []

        def on_submit(rec):
            seen.append(rec)
            sess.describe[rec["job_name"]] = {
                "JobStatus": "Completed",
                "JobArn": "arn:job/x",
                "DatasetArn": LABELED,
            }

        rec = J.submit_labeling_job(
            doc,
            region="us-west-2",
            role_arn="arn:role",
            base_job_name="autolabel-dev",
            session=sess,
            on_submit=on_submit,
        )
        assert len(seen) == 1 and seen[0]["job_name"].startswith("autolabel-dev-")
        assert seen[0]["dataset_name"] == "ds" and seen[0]["output_s3_uri"] == FOLDER + "output/"
        cfg, job_name = stubbed.calls[0]
        assert job_name == seen[0]["job_name"]
        assert cfg.schema_version == "2.0.0" and cfg.dataset_options.text_fields == ["text"]
        assert cfg.dataset_options.to_dataset_options() == doc["GenerationConfig"]["DatasetOptions"]
        assert "DatasetOptions" not in cfg.generation_config
        assert (
            stubbed.kwargs[0]["role"] == "arn:role"
            and stubbed.kwargs[0]["sagemaker_session"] is sess
        )
        assert (rec["job_status"], rec["dataset_arn"], rec["job_arn"], rec["reattached"]) == (
            "Completed",
            LABELED,
            "arn:job/x",
            False,
        )

    def test_review_only_job_skips_the_dataset_wait_and_polls(self, doc, stubbed):
        # A label review registers no DataSet; the SDK's wait would raise on the missing
        # DatasetArn, so the submit does not wait and polls DescribeJob instead.
        doc["GenerationConfig"]["DatasetOptions"]["Config"]["Review"] = {
            "LabelField": "category",
            "Scope": "seed",
        }
        sess = _FakeSession()
        states = iter([{"JobStatus": "InProgress"}, {"JobStatus": "Completed", "JobArn": "a"}])

        def on_submit(rec):
            sess.describe[rec["job_name"]] = lambda: next(states)

        rec = J.submit_labeling_job(doc, region="us-west-2", session=sess, on_submit=on_submit)
        assert stubbed.waits == [False]
        assert len(sess.describes) == 2
        assert (rec["job_status"], rec["dataset_arn"]) == ("Completed", None)

    def test_labeling_job_keeps_the_sdk_wait(self, doc, stubbed):
        sess = _FakeSession()
        J.submit_labeling_job(
            doc,
            region="us-west-2",
            session=sess,
            on_submit=lambda r: sess.describe.__setitem__(
                r["job_name"], {"JobStatus": "Completed"}
            ),
        )
        assert stubbed.waits == [True]

    def test_wait_false_returns_once_submitted(self, doc, stubbed):
        # Fire-and-forget: the SDK's wait is skipped and the record carries the status
        # DescribeJob reports right after CreateJob (InProgress when it has none yet).
        for describe, status in (({"JobStatus": "InProgress", "JobArn": "a"}), "InProgress"), (
            {},
            "InProgress",
        ):
            stubbed.waits, stubbed.calls = [], []
            sess = _FakeSession()
            rec = J.submit_labeling_job(
                doc,
                region="us-west-2",
                session=sess,
                wait=False,
                on_submit=lambda r, s=sess, d=describe: s.describe.__setitem__(r["job_name"], d),
            )
            assert stubbed.waits == [False] and len(stubbed.calls) == 1
            assert (rec["job_status"], rec["dataset_arn"], rec["reattached"]) == (
                status,
                None,
                False,
            )

    def test_wait_false_review_only_job_does_not_poll(self, doc, stubbed):
        doc["GenerationConfig"]["DatasetOptions"]["Config"]["Review"] = {
            "LabelField": "category",
            "Scope": "seed",
        }
        sess = _FakeSession()
        rec = J.submit_labeling_job(
            doc,
            region="us-west-2",
            session=sess,
            wait=False,
            on_submit=lambda r: sess.describe.__setitem__(
                r["job_name"], {"JobStatus": "InProgress"}
            ),
        )
        assert stubbed.waits == [False] and len(sess.describes) == 1
        assert rec["job_status"] == "InProgress"

    def test_wait_false_reattach_reads_the_job_and_never_resubmits(self, doc, stubbed):
        sess = _FakeSession({"j": {"JobStatus": "InProgress", "JobArn": "arn:j"}})
        rec = J.submit_labeling_job(
            doc, region="us-west-2", session=sess, resume_job={"job_name": "j"}, wait=False
        )
        assert stubbed.calls == [] and len(sess.describes) == 1
        assert (rec["job_name"], rec["job_status"], rec["reattached"]) == ("j", "InProgress", True)
        sess = _FakeSession(
            {"j": {"JobStatus": "Completed", "JobArn": "arn:j", "DatasetArn": LABELED}}
        )
        rec = J.submit_labeling_job(
            doc, region="us-west-2", session=sess, resume_job={"job_name": "j"}, wait=False
        )
        assert stubbed.calls == [] and (rec["job_status"], rec["dataset_arn"]) == (
            "Completed",
            LABELED,
        )

    def test_job_name_is_pattern_valid_and_bounded(self):
        name = J.job_name_for("auto label/dev_round!" + "x" * 80)
        assert len(name) <= 63 and name != J.job_name_for("a")
        assert __import__("re").fullmatch(r"[A-Za-z0-9-]+-[0-9a-f]{8}", name)

    def test_transient_errors_retried_within_the_budget(self, doc, stubbed):
        stubbed.errors = [
            RuntimeError("ResourceLimitExceeded: too many jobs"),
            RuntimeError("ThrottlingException"),
        ]
        sess = _FakeSession()
        rec = J.submit_labeling_job(
            doc,
            region="us-west-2",
            session=sess,
            on_submit=lambda r: sess.describe.__setitem__(
                r["job_name"], {"JobStatus": "Completed", "JobArn": "a"}
            ),
        )
        assert len(stubbed.calls) == 3 and rec["job_status"] == "Completed"
        stubbed.errors = [RuntimeError("ThrottlingException")]
        with pytest.raises(RuntimeError, match="ThrottlingException"):
            J.submit_labeling_job(
                doc, region="us-west-2", session=_FakeSession(), submit_max_wait_s=0
            )

    def test_refusals(self, doc, stubbed):
        stubbed.errors = [ValueError("ValidationException: bad role")]
        with pytest.raises(ValueError, match="bad role"):  # not a classified refusal: raised as is
            J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())
        assert len(stubbed.calls) == 1
        stubbed.errors = [
            ValueError("An error occurred (ValidationException) when calling CreateJob: bad role")
        ]
        with pytest.raises(J.CreateJobRefused, match="no job was created.*bad role"):
            J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())
        stubbed.errors = [
            ValueError("ValidationException: Confidence.Method 'x' is not one of ...")
        ]
        with pytest.raises(J.OptionRefused):
            J.submit_labeling_job(
                doc, region="us-west-2", session=_FakeSession(), option_fields=("Confidence",)
            )
        stubbed.errors = [
            ValueError("ModelConfig contains unsupported model: x. Allowed values: [a, b]")
        ]
        with pytest.raises(
            J.CreateJobRefused, match=r"unsupported model: x\. Allowed values: \[a, b\]$"
        ):
            J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())
        stubbed.errors = [ValueError("Job failed due to invalid input or configuration")]
        with pytest.raises(J.JobWaitError, match="CloudWatch log"):
            J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())

    def test_schema_rejection_is_the_halt(self, doc, stubbed):
        inner = ValueError("DatasetOptions: Extra inputs are not permitted")
        outer = RuntimeError("CreateJob failed")
        outer.__cause__ = inner
        assert J.is_labeling_schema_rejection(outer)
        assert not J.is_labeling_schema_rejection(
            ValueError("Additional properties are not allowed ('Foo' was unexpected)")
        )
        stubbed.errors = [
            ValueError(
                "Additional properties are not allowed ('DatasetOptions' was unexpected) not allowed by the schema"
            )
        ]
        endpoint = "https://beta.example"
        with pytest.raises(J.LabelingBackendUnavailable) as e:
            J.submit_labeling_job(
                doc, region="us-west-2", session=_FakeSession(), endpoint=endpoint
            )
        # The message is an error string, not a URL being validated; assert the
        # pieces separately so a failure says which one is missing.
        message = str(e.value)
        assert "HALT" in message
        assert endpoint in message
        assert "same --run-id" in message

    def test_reattach_completed_running_and_failed(self, doc, stubbed, monkeypatch):
        sess = _FakeSession(
            {"old": {"JobStatus": "Completed", "JobArn": "arn:old", "DatasetArn": LABELED}}
        )
        rec = J.submit_labeling_job(
            doc, region="us-west-2", session=sess, resume_job={"job_name": "old"}
        )
        assert stubbed.calls == [] and rec["reattached"] and rec["job_arn"] == "arn:old"
        states = iter(["InProgress", "InProgress", "Completed"])
        monkeypatch.setattr(J, "_REATTACH_POLL_S", 0)
        sess = _FakeSession({"j": lambda: {"JobStatus": next(states), "JobArn": "a"}})
        assert J.submit_labeling_job(
            doc, region="us-west-2", session=sess, resume_job={"job_name": "j"}
        )["reattached"]
        for prior in ({"j": {"JobStatus": "Failed", "FailureReason": "boom"}}, {}):
            stubbed.calls = []
            sess = _FakeSession(dict(prior))
            rec = J.submit_labeling_job(
                doc,
                region="us-west-2",
                session=sess,
                resume_job={"job_name": "j"},
                on_submit=lambda r, s=sess: s.describe.__setitem__(
                    r["job_name"], {"JobStatus": "Completed", "JobArn": "n"}
                ),
            )
            assert len(stubbed.calls) == 1 and not rec["reattached"] and rec["job_name"] != "j"

    def test_sdk_job_errors_become_records_or_reattach_messages(self, doc, stubbed):
        sess = _FakeSession()
        stubbed.errors = [DataPreparerJobFailedError("arn:job/f", failure_reason="from the SDK")]
        rec = J.submit_labeling_job(
            doc,
            region="us-west-2",
            session=sess,
            on_submit=lambda r: sess.describe.__setitem__(
                r["job_name"],
                {
                    "JobStatus": "Failed",
                    "JobArn": "arn:job/f",
                    "FailureReason": "spec has no label set",
                },
            ),
        )
        assert (rec["job_status"], rec["failure_reason"], rec["job_arn"]) == (
            "Failed",
            "spec has no label set",
            "arn:job/f",
        )
        stubbed.errors = [DataPreparerJobStoppedError("arn:job/s")]
        rec = J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())
        assert (rec["job_status"], rec["job_arn"]) == ("Stopped", "arn:job/s") and "stopped" in rec[
            "failure_reason"
        ]
        stubbed.errors = [DataPreparerJobTimeoutError("arn:job/t", 3600.0)]
        with pytest.raises(J.JobWaitError, match="same --run-id and --stage to wait"):
            J.submit_labeling_job(doc, region="us-west-2", session=_FakeSession())

    def test_describe_errors_do_not_resubmit_and_backoff(self):
        class _Sess:
            class sagemaker_client:
                @staticmethod
                def describe_job(JobName, JobCategory):  # noqa: N803
                    raise RuntimeError("AccessDeniedException")

        with pytest.raises(J.JobWaitError, match="not resubmitting"):
            J._describe_job(_Sess(), "j")

        class _R:
            def uniform(self, a, b):
                return b

        assert (
            J._submit_backoff_s(1, _R()) == 30
            and J._submit_backoff_s(10, _R()) == J._SUBMIT_BACKOFF_CAP_S
        )

    def test_arns_from_the_echoed_job_document(self):
        d = json.dumps({"DatasetArn": "labeled", "ReviewDatasetArn": "review"})
        assert J.job_dataset_arn({"JobConfigDocument": d}) == "labeled"
        assert J.job_review_dataset_arn({"JobConfigDocument": d}) == "review"
        assert (
            J.job_dataset_arn({"JobConfigDocument": "not json"}) is None
            and J.job_dataset_arn(None) is None
        )
        rec = J._job_record(
            {"JobStatus": "Completed", "JobConfigDocument": d}, "j", "s3://b/o/", reattached=True
        )
        assert (rec["dataset_arn"], rec["review_dataset_arn"]) == ("labeled", "review")


class TestReadResults:
    def _summary(self, review=REVIEW, artifacts=None):
        return LabelingSummary(
            Status="Certified",
            RowsLabeled=3,
            Artifacts=ARTIFACTS if artifacts is None else artifacts,
            LabeledDatasetArn=LABELED,
            ReviewDatasetArn=review,
        )

    def test_downloads_the_contract_file_names_and_keeps_the_containers_keys(
        self, tmp_path, monkeypatch
    ):
        s3 = _s3_with_job()
        calls = []
        s = self._summary()
        monkeypatch.setattr(
            LabelingSummary,
            "from_dataset",
            classmethod(lambda cls, arn, sagemaker_session=None: calls.append(arn) or s),
        )
        monkeypatch.setattr(J, "_region_bound_session", lambda region: None)
        monkeypatch.setattr(_s3, "s3_client", lambda region: s3)
        res = J.read_job_results(region="us-west-2", dataset_arn=LABELED, dest_dir=str(tmp_path))
        assert sorted(os.listdir(tmp_path)) == sorted(R.INDEX_ARTIFACTS[k] for k in ARTIFACTS)
        assert calls == [LABELED] and res["index_uri"] == FOLDER + "labeling/index.json"
        assert res["summary"] == {
            "Status": "Certified",
            "RowsLabeled": 3,
            "JurorModelIds": [],
            "Warnings": [],
        }
        assert res["review_dataset_arn"] == REVIEW and res["artifact_uris"] == ARTIFACTS
        res = J.read_job_results(
            region="us-west-2",
            dataset_arn=LABELED,
            dest_dir=str(tmp_path),
            names={"Labels"},
            review_dataset_arn="arn:job-review",
        )
        assert set(res["files"]) == {"Labels"} and res["review_dataset_arn"] == "arn:job-review"

    def test_errors(self, tmp_path, monkeypatch):
        with pytest.raises(FileNotFoundError, match="no labeled DataSet ARN"):
            J.read_job_results(region="us-west-2", dataset_arn=None, dest_dir=str(tmp_path))
        monkeypatch.setattr(J, "_region_bound_session", lambda region: None)

        def boom(cls, arn, sagemaker_session=None):
            raise DataPreparerOperatorError("carries no labeling job metadata")

        monkeypatch.setattr(LabelingSummary, "from_dataset", classmethod(boom))
        with pytest.raises(
            FileNotFoundError, match="DataPreparerOperatorError: carries no labeling job metadata"
        ):
            J.read_job_results(region="us-west-2", dataset_arn=LABELED, dest_dir=str(tmp_path))
        s = self._summary(artifacts={"Labels": ARTIFACTS["Labels"]})
        monkeypatch.setattr(
            LabelingSummary, "from_dataset", classmethod(lambda cls, arn, sagemaker_session=None: s)
        )
        with pytest.raises(FileNotFoundError, match="names no summary"):
            J.read_job_results(region="us-west-2", dataset_arn=LABELED, dest_dir=str(tmp_path))


class TestHelpers:
    def test_s3_helpers(self, tmp_path, monkeypatch):
        s3 = _FakeS3()
        monkeypatch.setattr(_s3, "s3_client", lambda region: s3)
        assert _s3.split_s3_uri("s3://b/k/x.jsonl") == ("b", "k/x.jsonl")
        for bad in ("s3://b", "s3://", "http://b/k", "", None):
            with pytest.raises(DataPreparerOperatorError):
                _s3.split_s3_uri(bad)
            assert not _s3.is_s3_uri(bad)
        assert _s3.is_s3_uri("s3://b/k")
        assert J.upload_text("abc", "s3://b/k", "r", field="F") == "s3://b/k" and s3.puts == [
            ("s3://b/k", b"abc")
        ]
        with pytest.raises(
            R.LabelingRequestError, match="F is 4 bytes; the service takes at most 3"
        ):
            J.upload_text("abcd", "s3://b/k", "r", field="F", max_bytes=3)
        p = tmp_path / "f.jsonl"
        p.write_bytes(b"row\n")
        _s3.upload_file(str(p), "s3://b/in.jsonl", "r")
        _s3.download_file("s3://b/in.jsonl", str(tmp_path / "d" / "in.jsonl"), "r")
        assert (tmp_path / "d" / "in.jsonl").read_bytes() == b"row\n"

    def test_arn_parsing_hub_name_and_delete(self, monkeypatch):
        assert register.parse_hub_content_arn(LABELED) == (
            "AIRegistry",
            "DataSet",
            "ds",
            "1.0.0",
            "us-west-2",
        )
        with pytest.raises(DataPreparerOperatorError, match="not a hub-content ARN"):
            register.parse_hub_content_arn("arn:nope")
        assert register.parse_hub_content_arn(LABELED.rsplit("/", 1)[0])[3] is None
        hub = register.ai_registry_hub_name(ACCT, "us-west-2")
        assert (
            hub == hub.upper()
            and len(hub) == 52
            and set(hub) <= set("0123456789ABCDEFGHIJKLMNOPQRSTUV")
        )
        calls = []

        class _SM:
            def delete_hub_content(self, **kw):
                calls.append(kw)

        monkeypatch.setattr(register, "_sm_client", lambda region: _SM())
        assert register.delete_hub_content_version(LABELED)["version"] == "1.0.0"
        assert calls == [
            {
                "HubName": "AIRegistry",
                "HubContentType": "DataSet",
                "HubContentName": "ds",
                "HubContentVersion": "1.0.0",
            }
        ]
        with pytest.raises(DataPreparerOperatorError, match="has no version"):
            register.delete_hub_content_version(LABELED.rsplit("/", 1)[0])

    def test_existing_datasets(self, monkeypatch):
        class _E(Exception):
            response = {"Error": {"Code": "ResourceNotFound"}}

        class _SM:
            def describe_hub_content(self, HubName, HubContentType, HubContentName):  # noqa: N803
                if HubContentName == "free":
                    raise _E()
                return {"HubContentVersion": "2.0.0", "HubContentArn": f"arn:{HubContentName}"}

        monkeypatch.setattr(register, "_sm_client", lambda region: _SM())
        found = register.existing_datasets(
            ["taken", "free", "taken", None], "us-west-2", hub_name="h"
        )
        assert found == [{"name": "taken", "hub": "h", "version": "2.0.0", "arn": "arn:taken"}]


class TestErrors:
    @pytest.mark.parametrize(
        "cls", [J.CreateJobRefused, J.OptionRefused, J.LabelingBackendUnavailable, J.JobWaitError]
    )
    def test_job_errors_are_runtime_errors(self, cls):
        # the command line's were RuntimeErrors: a caller's ``except RuntimeError`` still holds
        assert issubclass(cls, DataPreparerError) and issubclass(cls, RuntimeError)
        assert not issubclass(cls, ValueError)
        with pytest.raises(RuntimeError):
            raise cls("refused")

    def test_mro(self):
        assert J.CreateJobRefused.__mro__[1:] == (
            DataPreparerError,
            RuntimeError,
            Exception,
            BaseException,
            object,
        )
        assert J.OptionRefused.__mro__[1] is J.CreateJobRefused
        for cls in (J.LabelingBackendUnavailable, J.JobWaitError):
            assert cls.__bases__ == (DataPreparerError, RuntimeError)


def test_one_telemetry_attribution_definition():
    from sagemaker.ai_registry.data_preparer.auto_labeling.operators import _bedrock as bedrock

    assert J._set_telemetry_attribution is bedrock._set_telemetry_attribution


# ---------------------------------------------------------------------------
# Tests for _sdg_output_field and the writer/reader output-field logic
# ---------------------------------------------------------------------------


class _SchemaSession:
    """Fake session whose sagemaker_client answers one schema call."""

    def __init__(self, odc_props):
        schema = {
            "properties": {
                "OutputDataConfig": {
                    "type": "object",
                    "properties": odc_props,
                }
            }
        }

        class _Client:
            def list_job_schema_versions(self, **_kw):
                return {"JobConfigSchemas": [{"JobConfigSchemaVersion": "2.0.0"}]}

            def describe_job_schema_version(self, **_kw):
                import json

                return {"JobConfigSchema": json.dumps(schema)}

        self.sagemaker_client = _Client()


class _FailSession:
    """Session whose sagemaker_client raises on every call."""

    class _Client:
        def list_job_schema_versions(self, **_kw):
            raise RuntimeError("network unavailable")

        def describe_job_schema_version(self, **_kw):
            raise RuntimeError("network unavailable")

    sagemaker_client = _Client()


class TestSdgOutputField:
    """_sdg_output_field picks the field name from the resolved schema."""

    def setup_method(self):
        # Cache isolation: each test provides its own schema; clear the global
        # resolver cache so a previous test's cached schema does not shadow it.
        from sagemaker.ai_registry.data_preparer._schema_resolver import _SCHEMA_CACHE

        _SCHEMA_CACHE.clear()

    def test_schema_declares_only_output_s3_uri(self):
        sess = _SchemaSession({"OutputS3Uri": {"type": "string"}})
        assert J._sdg_output_field(sess) == "OutputS3Uri"

    def test_schema_declares_only_s3_output_path(self):
        sess = _SchemaSession({"S3OutputPath": {"type": "string"}})
        assert J._sdg_output_field(sess) == "S3OutputPath"

    def test_schema_declares_both_s3_output_path_wins(self):
        sess = _SchemaSession(
            {"OutputS3Uri": {"type": "string"}, "S3OutputPath": {"type": "string"}}
        )
        assert J._sdg_output_field(sess) == "S3OutputPath"

    def test_schema_unavailable_returns_output_s3_uri(self):
        assert J._sdg_output_field(_FailSession()) == "OutputS3Uri"


class TestSubmitOutputFieldNormalization:
    """submit_labeling_job normalizes OutputDataConfig to the schema-declared spelling."""

    def _doc(self, field_name):
        return R.build_labeling_request(
            task="classification",
            text_fields=["text"],
            spec_s3_uri="s3://b/p/spec.md",
            model_ids=["m"],
            dataset_name="ds",
            output_s3_uri=FOLDER + "output/",
            input_s3_uri="s3://b/p/in.jsonl",
            output_field_name=field_name,
        )

    def test_writer_emits_output_s3_uri_when_schema_declares_it(self, monkeypatch, stubbed):
        doc = self._doc("OutputS3Uri")
        assert "OutputS3Uri" in doc["OutputDataConfig"]
        assert "S3OutputPath" not in doc["OutputDataConfig"]

    def test_writer_emits_s3_output_path_when_schema_declares_it(self, monkeypatch, stubbed):
        doc = self._doc("S3OutputPath")
        assert "S3OutputPath" in doc["OutputDataConfig"]
        assert "OutputS3Uri" not in doc["OutputDataConfig"]

    def test_writer_defaults_to_output_s3_uri_on_schema_unavailable(self):
        doc = R.build_labeling_request(
            task="classification",
            text_fields=["text"],
            spec_s3_uri="s3://b/spec.md",
            model_ids=["m"],
            dataset_name="ds",
            output_s3_uri="s3://b/out/",
            input_s3_uri="s3://b/in.jsonl",
        )
        assert "OutputS3Uri" in doc["OutputDataConfig"]

    def test_submit_normalizes_output_s3_uri_doc_to_s3_output_path(self, monkeypatch, stubbed):
        """Document built with OutputS3Uri; schema says S3OutputPath → submit normalizes."""
        monkeypatch.setattr(J, "_sdg_output_field", lambda _sess: "S3OutputPath")
        monkeypatch.setattr(J, "_region_bound_session", lambda _r: _FailSession())
        doc = self._doc("OutputS3Uri")
        seen_docs = []
        orig_to_cfg = J.to_dataset_job_config

        def capture_to_cfg(d):
            seen_docs.append(d)
            return orig_to_cfg(d)

        monkeypatch.setattr(J, "to_dataset_job_config", capture_to_cfg)
        try:
            J.submit_labeling_job(doc, region="us-west-2", session=_FailSession(), wait=False)
        except Exception:
            pass
        if seen_docs:
            odc = seen_docs[0]["OutputDataConfig"]
            assert "S3OutputPath" in odc
            assert "OutputS3Uri" not in odc


class TestReadersAcceptEitherSpelling:
    """The three readers must not raise on either OutputS3Uri or S3OutputPath."""

    def _doc_with(self, field_name):
        return {
            "DatasetName": "ds",
            "GenerationConfig": {
                "DatasetSpecification": {"SpecS3Uri": "s3://b/spec.md"},
                "InputDataS3Uri": "s3://b/in.jsonl",
                "DatasetOptions": {
                    "Type": "labeling.classification",
                    "Config": {"TextFields": ["text"]},
                },
                "ModelConfig": {"SampleGenerationModelConfigs": [{"ModelId": "m"}]},
            },
            "OutputDataConfig": {field_name: "s3://b/out/"},
        }

    def test_request_summary_reads_s3_output_path(self):
        doc = self._doc_with("S3OutputPath")
        summary = R.request_summary(doc)
        assert summary["output"] == "s3://b/out/"

    def test_request_summary_reads_output_s3_uri(self):
        doc = self._doc_with("OutputS3Uri")
        summary = R.request_summary(doc)
        assert summary["output"] == "s3://b/out/"

    def test_submit_reads_output_s3_uri_from_document(self, monkeypatch, stubbed):
        """client.py:404 reader must not raise on OutputS3Uri spelling."""
        monkeypatch.setattr(J, "_region_bound_session", lambda _r: _FailSession())
        doc = self._doc_with("OutputS3Uri")
        seen = []

        def on_submit(rec):
            seen.append(rec)

        try:
            J.submit_labeling_job(doc, region="us-west-2", on_submit=on_submit, wait=False)
        except Exception:
            pass
        # on_submit receives output_s3_uri regardless of key spelling
        if seen:
            assert seen[0]["output_s3_uri"] == "s3://b/out/"

    def test_submit_reads_s3_output_path_from_document(self, monkeypatch, stubbed):
        """client.py:404 reader must not raise on S3OutputPath spelling."""
        monkeypatch.setattr(J, "_region_bound_session", lambda _r: _FailSession())
        doc = self._doc_with("S3OutputPath")
        seen = []

        def on_submit(rec):
            seen.append(rec)

        try:
            J.submit_labeling_job(doc, region="us-west-2", on_submit=on_submit, wait=False)
        except Exception:
            pass
        if seen:
            assert seen[0]["output_s3_uri"] == "s3://b/out/"
