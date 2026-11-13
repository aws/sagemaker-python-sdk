# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for write_audit.py"""

import json
import logging
import multiprocessing
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from sagemaker.mlops.feature_store import write_audit as wa
from sagemaker.mlops.feature_store.write_audit import (
    WRITE_AUDIT_ENV_VAR,
    WRITE_AUDIT_LOGGER_NAME,
    WriteAuditConfig,
)
from sagemaker.mlops.feature_store.write_audit import _AttemptResult as AttemptResult
from sagemaker.mlops.feature_store.write_audit import _build_audit_event as build_audit_event
from sagemaker.mlops.feature_store.write_audit import _EntryStatus as EntryStatus
from sagemaker.mlops.feature_store.write_audit import _RecordRef as RecordRef
from sagemaker.mlops.feature_store.write_audit import (
    _request_id_from_exception as request_id_from_exception,
)
from sagemaker.mlops.feature_store.write_audit import _write_audit as write_audit
from sagemaker.mlops.feature_store.write_audit import (
    _write_audit_enabled_from_env as write_audit_enabled_from_env,
)

FG = "test-fg"
CONFIG = WriteAuditConfig(record_identifier_feature_name="id", event_time_feature_name="ts")


def record(record_id, event_time="2026-01-01T00:00:00Z", extra=None):
    values = [
        {"FeatureName": "id", "ValueAsString": record_id},
        {"FeatureName": "ts", "ValueAsString": event_time},
    ]
    return values + (extra or [])


def entry(record_id, feature_group_name=FG):
    return {"FeatureGroupName": feature_group_name, "Record": record(record_id)}


def http_response(status, headers):
    return SimpleNamespace(status_code=status, headers=headers)


def op(name):
    return SimpleNamespace(name=name)


@pytest.fixture
def registry():
    reg = wa._AuditRegistry()
    reg.register(FG, CONFIG)
    return reg


@pytest.fixture
def extractor(registry):
    return wa._RecordRefExtractor(registry)


def make_hooks(registry, clock=None):
    return wa._WriteAuditHooks(registry, clock or iter(("t-sent", "t-received")).__next__)


def simulate_attempt(hooks, operation, status, request_id, parsed=None, attempts=1):
    """Fire the per-attempt events in botocore's order."""
    hooks._on_before_send(request=object())
    response = (http_response(status, {"x-amzn-requestid": request_id}), parsed or {})
    return hooks._on_needs_retry(
        response=response,
        attempts=attempts,
        caught_exception=None,
        operation=op(operation),
        endpoint=None,
        request_dict={},
    )


@pytest.fixture
def audit_records(caplog):
    caplog.set_level(logging.INFO, logger=WRITE_AUDIT_LOGGER_NAME)

    def events():
        return [
            json.loads(r.getMessage()) for r in caplog.records if r.name == WRITE_AUDIT_LOGGER_NAME
        ]

    return events


class TestEnvAndErrors:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_env_var_truthy(self, monkeypatch, value):
        monkeypatch.setenv(WRITE_AUDIT_ENV_VAR, value)
        assert write_audit_enabled_from_env()

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe"])
    def test_env_var_falsy(self, monkeypatch, value):
        monkeypatch.setenv(WRITE_AUDIT_ENV_VAR, value)
        assert not write_audit_enabled_from_env()

    def test_env_var_unset(self, monkeypatch):
        monkeypatch.delenv(WRITE_AUDIT_ENV_VAR, raising=False)
        assert not write_audit_enabled_from_env()

    def test_request_id_from_client_error(self):
        error = Exception("boom")
        error.response = {"ResponseMetadata": {"RequestId": "req-123"}}
        assert request_id_from_exception(error) == "req-123"

    @pytest.mark.parametrize(
        "response",
        [None, "not-a-dict", {}, {"ResponseMetadata": None}, {"ResponseMetadata": "not-a-dict"}],
    )
    def test_request_id_absent(self, response):
        error = Exception("boom")
        error.response = response
        assert request_id_from_exception(error) is None


class TestRegistry:
    def test_register_get_unregister(self):
        reg = wa._AuditRegistry()
        assert reg.get(FG) is None
        token = reg.register(FG, CONFIG)
        assert reg.get(FG) is CONFIG
        reg.unregister(FG, token)
        assert reg.get(FG) is None

    def test_latest_registration_wins_and_earlier_survives(self):
        reg = wa._AuditRegistry()
        other = WriteAuditConfig("other_id", "other_ts")
        first = reg.register(FG, CONFIG)
        second = reg.register(FG, other)
        assert reg.get(FG) is other
        reg.unregister(FG, second)
        assert reg.get(FG) is CONFIG
        reg.unregister(FG, first)
        assert reg.get(FG) is None

    def test_unregister_unknown_is_noop(self):
        reg = wa._AuditRegistry()
        reg.unregister(FG, 42)
        assert reg.get(FG) is None


class TestRecordExtraction:
    def test_extracts_configured_features_only(self, extractor):
        ref = extractor.extract(FG, record("r1", extra=[{"FeatureName": "secret"}]))
        assert ref == RecordRef(FG, "r1", "2026-01-01T00:00:00Z", audited=True)
        assert "secret" not in repr(ref)  # the matching key holds values but is never shown

    def test_unregistered_feature_group_is_not_audited(self, extractor):
        ref = extractor.extract("other-fg", record("r1"))
        assert ref == RecordRef("other-fg", None, None) and not ref.audited

    def test_tolerates_missing_record(self, extractor):
        assert extractor.extract(FG, None) == RecordRef(FG, None, None, audited=True)


class TestAttemptResult:
    def test_reads_request_id_status_and_date_case_insensitively(self):
        headers = {"x-amzn-RequestId": "abc", "Date": "Tue, 10 Nov 2026 16:56:51 GMT"}
        result = AttemptResult.from_needs_retry(
            2, "t0", "t1", (http_response(200, headers), {}), None
        )
        assert (result.request_id, result.http_status, result.server_date, result.attempt) == (
            "abc",
            200,
            "Tue, 10 Nov 2026 16:56:51 GMT",
            2,
        )
        assert result.succeeded

    def test_falls_back_to_parsed_request_id_and_error_code(self):
        parsed = {
            "ResponseMetadata": {"RequestId": "body-id"},
            "Error": {"Code": "ValidationError"},
        }
        result = AttemptResult.from_needs_retry(
            1, "t0", "t1", (http_response(400, {}), parsed), None
        )
        assert (result.request_id, result.error_code, result.succeeded) == (
            "body-id",
            "ValidationError",
            False,
        )

    def test_without_response_records_exception(self):
        result = AttemptResult.from_needs_retry(1, "t0", "t1", None, ConnectionError("reset"))
        assert result.request_id is None and "reset" in result.exception
        assert not result.succeeded


class TestAuditors:
    def attempt(self, status=200):
        return AttemptResult(1, "t0", "t1", request_id="req", http_status=status)

    def test_put_record(self, extractor):
        auditor = wa._PutRecordAuditor(extractor)
        records = auditor.records_in_request({"FeatureGroupName": FG, "Record": record("r1")})
        assert auditor.entry_outcomes(records, {}, self.attempt(200))[0].status is (
            EntryStatus.ACCEPTED
        )
        assert auditor.entry_outcomes(records, {}, self.attempt(500))[0].status is (
            EntryStatus.REQUEST_FAILED
        )

    def test_batch_maps_errors_and_unprocessed(self, extractor):
        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [entry("r1"), entry("r2"), entry("r3")]})
        parsed = {
            "Errors": [{"Entry": entry("r2"), "ErrorCode": "ValidationError", "ErrorMessage": "m"}],
            "UnprocessedEntries": [entry("r3")],
        }
        outcomes = auditor.entry_outcomes(records, parsed, self.attempt())
        assert [o.status for o in outcomes] == [
            EntryStatus.ACCEPTED,
            EntryStatus.ERROR,
            EntryStatus.UNPROCESSED,
        ]
        assert outcomes[1].error_code == "ValidationError"
        assert not hasattr(outcomes[1], "error_message")  # service free text is not logged

    def test_batch_same_id_and_time_matched_by_full_content(self, extractor):
        # Same record ID and event time, different feature values: only index 1 failed.
        def dup(value):
            return {
                "FeatureGroupName": FG,
                "Record": record("dup", extra=[{"FeatureName": "v", "ValueAsString": value}]),
            }

        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [dup("a"), dup("b")]})
        parsed = {"Errors": [{"Entry": dup("b"), "ErrorCode": "E", "ErrorMessage": "m"}]}
        outcomes = auditor.entry_outcomes(records, parsed, self.attempt())
        assert [o.status for o in outcomes] == [EntryStatus.ACCEPTED, EntryStatus.ERROR]

    def test_batch_identical_duplicates_matched_one_to_one(self, extractor):
        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [entry("dup"), entry("dup")]})
        parsed = {"Errors": [{"Entry": entry("dup"), "ErrorCode": "E", "ErrorMessage": "m"}]}
        statuses = sorted(o.status for o in auditor.entry_outcomes(records, parsed, self.attempt()))
        assert statuses == sorted([EntryStatus.ERROR, EntryStatus.ACCEPTED])

    def test_batch_echoed_arn_still_matches(self, extractor):
        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [entry("r1"), entry("r2")]})
        arn = f"arn:aws:sagemaker:us-west-2:111122223333:feature-group/{FG}"
        parsed = {"Errors": [{"Entry": entry("r2", arn), "ErrorCode": "E", "ErrorMessage": "m"}]}
        outcomes = auditor.entry_outcomes(records, parsed, self.attempt())
        assert [o.status for o in outcomes] == [EntryStatus.ACCEPTED, EntryStatus.ERROR]

    def test_batch_unmatched_failure_marks_unconfirmed_entries_unknown(self, extractor, caplog):
        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [entry("r1"), entry("r2")]})
        parsed = {
            "Errors": [{"Entry": entry("r2"), "ErrorCode": "E", "ErrorMessage": "m"}],
            "UnprocessedEntries": [entry("not-in-request")],
        }
        outcomes = auditor.entry_outcomes(records, parsed, self.attempt())
        assert [o.status for o in outcomes] == [EntryStatus.UNKNOWN, EntryStatus.ERROR]
        assert "Could not match 1 reported BatchWriteRecord failure" in caplog.text

    def test_batch_failed_attempt_marks_every_entry(self, extractor):
        auditor = wa._BatchWriteRecordAuditor(extractor)
        records = auditor.records_in_request({"Entries": [entry("r1"), entry("r2")]})
        outcomes = auditor.entry_outcomes(records, {}, self.attempt(503))
        assert {o.status for o in outcomes} == {EntryStatus.REQUEST_FAILED}

    def test_build_audit_event_drops_none_and_never_has_feature_values(self, extractor):
        auditor = wa._PutRecordAuditor(extractor)
        records = auditor.records_in_request(
            {"FeatureGroupName": FG, "Record": record("r1", extra=[{"FeatureName": "secret"}])}
        )
        outcome = auditor.entry_outcomes(records, {}, self.attempt())[0]
        event = build_audit_event("PutRecord", self.attempt(), 0, 1, outcome)
        assert event["record_id"] == "r1" and event["entry_status"] == "accepted"
        assert "exception" not in event and "secret" not in json.dumps(event)


class FakeEvents:
    def __init__(self):
        self.handlers = {}

    def register(self, name, handler, unique_id=None):
        self.handlers.setdefault(name, {})[unique_id] = handler


class TestHooks:
    def test_registers_only_documented_events_idempotently(self, registry):
        hooks, events = make_hooks(registry), FakeEvents()
        hooks.register(events)
        hooks.register(events)
        assert {name.split(".")[0] for name in events.handlers} == {
            "provide-client-params",
            "before-send",
            "needs-retry",
        }
        assert len(events.handlers) == 6
        assert all(len(by_id) == 1 for by_id in events.handlers.values())

    def test_one_event_per_record_per_attempt(self, registry, audit_records):
        hooks = make_hooks(registry)
        hooks._on_client_params(
            params={"Entries": [entry("r1"), entry("r2")]}, model=op("BatchWriteRecord")
        )
        simulate_attempt(hooks, "BatchWriteRecord", 200, "batch-1", {"Errors": []})
        assert [
            (e["record_id"], e["request_id"], e["request_sent_at"]) for e in audit_records()
        ] == [
            ("r1", "batch-1", "t-sent"),
            ("r2", "batch-1", "t-sent"),
        ]

    def test_each_retry_attempt_logged_with_own_send_time(self, registry, audit_records):
        hooks = make_hooks(registry, iter(["s1", "r1", "s2", "r2"]).__next__)
        hooks._on_client_params(
            params={"FeatureGroupName": FG, "Record": record("a")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "PutRecord", 500, "req-a", attempts=1)
        simulate_attempt(hooks, "PutRecord", 200, "req-b", attempts=2)
        assert [(e["attempt"], e["request_id"], e["request_sent_at"]) for e in audit_records()] == [
            (1, "req-a", "s1"),
            (2, "req-b", "s2"),
        ]

    def test_attempt_without_before_send_has_no_send_time(self, registry, audit_records):
        hooks = make_hooks(registry, iter(["s1", "r1", "r2"]).__next__)
        hooks._on_client_params(
            params={"FeatureGroupName": FG, "Record": record("a")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "PutRecord", 500, "req-a")
        hooks._on_needs_retry(
            response=None,
            attempts=2,
            caught_exception=ConnectionError("x"),
            operation=op("PutRecord"),
        )
        assert "request_sent_at" not in audit_records()[1]

    def test_every_handler_returns_none(self, registry):
        # botocore treats non-None as: replace params / short-circuit send / sleep and retry.
        hooks = make_hooks(registry)
        assert (
            hooks._on_client_params(
                params={"FeatureGroupName": FG, "Record": record("a")}, model=op("PutRecord")
            )
            is None
        )
        assert hooks._on_before_send(request=object()) is None
        assert simulate_attempt(hooks, "PutRecord", 200, "req") is None

    def test_handler_errors_are_swallowed(self, registry, caplog):
        hooks = make_hooks(registry)
        assert hooks._on_client_params(params={}, model=op("Unknown")) is None
        assert (
            hooks._on_needs_retry(response=None, attempts=1, caught_exception=None, operation=None)
            is None
        )
        assert "write audit handler failed" in caplog.text

    def test_unaudited_call_does_not_inherit_previous_call(self, registry, audit_records):
        hooks = make_hooks(registry, lambda: "t")
        hooks._on_client_params(
            params={"FeatureGroupName": FG, "Record": record("audited")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "PutRecord", 200, "req-1")
        # Next call on the same thread is for a Feature Group that is not audited.
        hooks._on_client_params(
            params={"FeatureGroupName": "other-fg", "Record": record("x")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "PutRecord", 200, "req-2")
        assert [e["request_id"] for e in audit_records()] == ["req-1"]

    def test_mixed_batch_logs_only_audited_entries_with_true_index(self, registry, audit_records):
        hooks = make_hooks(registry)
        hooks._on_client_params(
            params={"Entries": [entry("x", "other-fg"), entry("r1")]},
            model=op("BatchWriteRecord"),
        )
        simulate_attempt(hooks, "BatchWriteRecord", 200, "batch-1", {"Errors": []})
        assert [(e["record_id"], e["entry_index"], e["batch_size"]) for e in audit_records()] == [
            ("r1", 1, 2)
        ]

    def test_ignores_other_operations(self, registry, audit_records):
        hooks = make_hooks(registry)
        hooks._on_client_params(
            params={"FeatureGroupName": FG, "Record": record("a")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "BatchWriteRecord", 200, "other")
        assert audit_records() == []

    def test_threads_are_kept_apart(self, registry, audit_records):
        hooks = make_hooks(registry, lambda: "t")
        barrier = threading.Barrier(2)

        def call(record_id):
            hooks._on_client_params(
                params={"FeatureGroupName": FG, "Record": record(record_id)}, model=op("PutRecord")
            )
            barrier.wait()  # both calls are in flight before either completes
            simulate_attempt(hooks, "PutRecord", 200, f"req-{record_id}")

        threads = [threading.Thread(target=call, args=(rid,)) for rid in ("a", "b")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted((e["record_id"], e["request_id"]) for e in audit_records()) == [
            ("a", "req-a"),
            ("b", "req-b"),
        ]

    def test_logs_to_documented_logger_at_info(self, caplog):
        caplog.set_level(logging.INFO, logger=WRITE_AUDIT_LOGGER_NAME)
        reg = wa._AuditRegistry()
        reg.register(FG, CONFIG)
        hooks = make_hooks(reg)
        hooks._on_client_params(
            params={"FeatureGroupName": FG, "Record": record("a")}, model=op("PutRecord")
        )
        simulate_attempt(hooks, "PutRecord", 200, "req")
        assert [(r.name, r.levelno) for r in caplog.records] == [
            (WRITE_AUDIT_LOGGER_NAME, logging.INFO)
        ]


class TestWriteAuditContext:
    def test_none_config_is_noop(self):
        with patch.object(wa, "_HOOKS") as hooks:
            with write_audit(FG, None):
                assert wa._REGISTRY.get(FG) is None
        hooks.register.assert_not_called()

    def test_registers_for_the_block_only(self):
        client = MagicMock()
        with patch("sagemaker.core.utils.utils.SageMakerClient") as factory:
            factory.return_value.get_client.return_value = client
            with write_audit(FG, CONFIG, region="eu-west-1"):
                assert wa._REGISTRY.get(FG) is CONFIG
        assert wa._REGISTRY.get(FG) is None
        factory.assert_called_once_with(region_name="eu-west-1")
        factory.return_value.get_client.assert_called_once_with("sagemaker-featurestore-runtime")
        assert client.meta.events.register.call_count == 6

    def test_unregisters_when_block_raises(self):
        with patch("sagemaker.core.utils.utils.SageMakerClient"):
            with pytest.raises(RuntimeError):
                with write_audit(FG, CONFIG):
                    raise RuntimeError("ingestion failed")
        assert wa._REGISTRY.get(FG) is None

    def test_setup_failure_does_not_interrupt_ingestion(self, caplog):
        with patch("sagemaker.core.utils.utils.SageMakerClient", side_effect=RuntimeError("x")):
            with write_audit(FG, CONFIG):
                ran = True
        assert ran and wa._REGISTRY.get(FG) is None
        assert "ingesting without it" in caplog.text


# --------------------------------------------------------------------------- end to end


class _FakeFeatureStore(BaseHTTPRequestHandler):
    """Loopback fake Feature Store runtime: PutRecord (PUT) and BatchWriteRecord (POST).

    Each response's request ID is recorded with the first record ID of its request, so
    tests can verify every logged record is attributed to the request that carried it.
    """

    batch_calls = 0
    first_record_by_request_id = {}
    lock = threading.Lock()

    def _reply(self, status, payload, first_record_id):
        body = json.dumps(payload).encode() if payload is not None else b""
        # The header holds only a server-generated UUID, never request data; the mapping
        # to the request's first record lives server-side for the tests to look up.
        request_id = str(uuid.uuid4())
        with type(self).lock:
            type(self).first_record_by_request_id[request_id] = first_record_id
        self.send_response(status)
        self.send_header("x-amzn-RequestId", request_id)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return json.loads(self.rfile.read(int(self.headers["Content-Length"])))

    @staticmethod
    def _rid(values):
        return next(f["ValueAsString"] for f in values if f["FeatureName"] == "id")

    def do_PUT(self):  # noqa: N802 - http.server naming
        rid = self._rid(self._body()["Record"])
        if rid == "3":
            return self._reply(400, {"__type": "ValidationError", "message": "bad"}, rid)
        return self._reply(200, None, rid)

    def do_POST(self):  # noqa: N802 - http.server naming
        entries = self._body()["Entries"]
        first = self._rid(entries[0]["Record"])
        with type(self).lock:
            type(self).batch_calls += 1
            first_call = type(self).batch_calls == 1
        if first_call:  # retryable failure on the very first batch attempt
            return self._reply(500, {"__type": "InternalFailure", "message": "boom"}, first)
        errors = [
            {"Entry": e, "ErrorCode": "ValidationError", "ErrorMessage": "bad"}
            for e in entries
            if self._rid(e["Record"]) == "3"
        ]
        unprocessed = [e for e in entries if self._rid(e["Record"]) == "4"]
        return self._reply(200, {"Errors": errors, "UnprocessedEntries": unprocessed}, first)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_endpoint(monkeypatch, tmp_path):
    from sagemaker.core.utils.utils import SageMakerClient

    _FakeFeatureStore.batch_calls = 0
    _FakeFeatureStore.first_record_by_request_id = {}
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeFeatureStore)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Isolate from the developer's AWS config: retry settings and profiles change behaviour.
    for name in (
        "AWS_MAX_ATTEMPTS",
        "AWS_RETRY_MODE",
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SESSION_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    # Dummy, non-functional values: requests only reach the loopback fake server.
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIDFAKETEST")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake-test-secret")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv(
        "AWS_ENDPOINT_URL_SAGEMAKER_FEATURESTORE_RUNTIME",
        f"http://127.0.0.1:{server.server_address[1]}",
    )
    SageMakerClient.reset()
    yield
    SageMakerClient.reset()
    server.shutdown()


@pytest.fixture
def audit_file(tmp_path):
    """Collect audit lines in a file so lines from forked worker processes are visible."""
    path = tmp_path / "audit.jsonl"
    handler = logging.FileHandler(path)
    audit_logger = logging.getLogger(WRITE_AUDIT_LOGGER_NAME)
    previous_level = audit_logger.level
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    yield lambda: [json.loads(line) for line in path.read_text().splitlines()]
    audit_logger.removeHandler(handler)
    audit_logger.setLevel(previous_level)
    handler.close()


def run_ingestion(use_batch_write_record, rows=30, max_workers=1):
    from sagemaker.mlops.feature_store.ingestion_manager_pandas import (
        IngestionError,
        IngestionManagerPandas,
    )

    frame = pd.DataFrame(
        {
            "id": [str(i) for i in range(1, rows + 1)],
            "ts": [f"2026-01-01T00:{i % 60:02d}:00Z" for i in range(1, rows + 1)],
            "secret": ["do-not-log"] * rows,
        }
    )
    definitions = {name: {"FeatureType": "String", "CollectionType": None} for name in frame}
    manager = IngestionManagerPandas(
        feature_group_name=FG,
        feature_definitions=definitions,
        region="us-west-2",
        use_batch_write_record=use_batch_write_record,
        max_workers=max_workers,
        write_audit_config=CONFIG,
    )
    with pytest.raises(IngestionError):  # the fake rejects some rows on purpose
        manager.run(frame)
    return manager.failed_rows


class TestEndToEnd:
    def test_put_record(self, fake_endpoint, audit_file):
        failed_rows = run_ingestion(use_batch_write_record=False)
        events = audit_file()
        assert len(events) == 30 and len({e["request_id"] for e in events}) == 30
        rejected = next(e for e in events if e["record_id"] == "3")
        assert (rejected["entry_status"], rejected["http_status"]) == ("request_failed", 400)
        assert failed_rows == [2]
        assert all("do-not-log" not in json.dumps(e) for e in events)

    def test_batch_write_record_with_retry_and_partial_failures(self, fake_endpoint, audit_file):
        failed_rows = run_ingestion(use_batch_write_record=True)
        events = audit_file()
        first_attempt = [e for e in events if e["batch_size"] == 25 and e["attempt"] == 1]
        retry = {e["record_id"]: e for e in events if e["batch_size"] == 25 and e["attempt"] == 2}
        assert len(first_attempt) == 25
        assert {e["entry_status"] for e in first_attempt} == {"request_failed"}
        assert len({e["request_id"] for e in retry.values()}) == 1
        assert retry["3"]["entry_status"] == "error"
        assert retry["4"]["entry_status"] == "unprocessed"
        assert retry["1"]["entry_status"] == "accepted"
        assert len([e for e in events if e["batch_size"] == 5]) == 5
        assert sorted(failed_rows) == [2, 3]

    @pytest.mark.skipif(
        multiprocessing.get_start_method() != "fork",
        reason="IngestionManagerPandas multi-process ingestion currently requires the fork start "
        "method (its Pool initializer is a nested function)",
    )
    @pytest.mark.parametrize("use_batch_write_record", [False, True])
    def test_worker_process_threads_attribute_records_to_their_request(
        self, fake_endpoint, audit_file, use_batch_write_record
    ):
        # max_workers > 1 runs the multiprocessing Pool path; the audit is set up per worker.
        run_ingestion(use_batch_write_record=use_batch_write_record, rows=120, max_workers=6)
        events = audit_file()
        assert {e["record_id"] for e in events} == {str(i) for i in range(1, 121)}
        by_request = {}
        for event in events:
            by_request.setdefault(event["request_id"], []).append(event)
        for request_id, group in by_request.items():
            first = min(group, key=lambda e: e["entry_index"])
            # The fake server runs in this process, so its mapping is visible here even
            # though the writes came from forked worker processes.
            assert _FakeFeatureStore.first_record_by_request_id[request_id] == first["record_id"]
            assert len(group) == first["batch_size"]

    def test_disabled_by_default_logs_nothing(self, fake_endpoint, audit_file):
        from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionManagerPandas

        frame = pd.DataFrame({"id": ["1", "2"], "ts": ["t", "t"]})
        definitions = {name: {"FeatureType": "String"} for name in frame}
        IngestionManagerPandas(
            feature_group_name=FG, feature_definitions=definitions, region="us-west-2"
        ).run(frame)
        assert audit_file() == []
