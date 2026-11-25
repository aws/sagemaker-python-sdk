# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Opt-in, per-record AWS request ID logging for Feature Store writes.

When enabled for an ingestion, every PutRecord and BatchWriteRecord HTTP attempt is
logged as one JSON line per record on the ``sagemaker.mlops.feature_store.write_audit``
logger (INFO level). Each line carries the record identifier, event time, AWS request ID,
HTTP status, per-entry outcome and client-side timestamps, so a record can be traced to
the exact request (and retry attempt) that carried it.

Only the record identifier and event time are logged; no other feature values. Record
identifiers can be sensitive (for example customer IDs), so treat these logs accordingly.

Enable it with ``ingest_dataframe(..., enable_write_audit=True)``, with the
``SAGEMAKER_FEATURE_STORE_WRITE_AUDIT=true`` environment variable, or by passing a
``WriteAuditConfig`` to ``IngestionManagerPandas``.

Limitations:

* Records that fail before any HTTP attempt (for example parameter validation or missing
  credentials) get no audit line; the existing ingestion error log still reports the row.
* Writes still running after an ingestion ``timeout`` are not audited.
* Handlers live on the client sagemaker-core caches per region. If that cache is reset
  during an ingestion, writes made through the new client are not audited.

Implementation notes
--------------------
Only events and keyword arguments documented in the boto3 event system guide are used:

* ``provide-client-params`` (``params``, ``model``): once per API call.
* ``before-send`` (``request``): once per HTTP attempt.
* ``needs-retry`` (``response``, ``attempts``, ``caught_exception``, ``operation``): once per
  HTTP attempt, including the final one.

botocore treats a non-None handler return value as an instruction (replace params,
short-circuit the send, sleep and retry), so every handler returns None, and every
handler swallows its own errors so logging can never break ingestion. Per-call state is
kept in a thread-local: a botocore call runs synchronously on its thread from
``provide-client-params`` through its final ``needs-retry``.
"""

import contextlib
import dataclasses
import datetime
import enum
import functools
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

WRITE_AUDIT_LOGGER_NAME = "sagemaker.mlops.feature_store.write_audit"
WRITE_AUDIT_ENV_VAR = "SAGEMAKER_FEATURE_STORE_WRITE_AUDIT"

_audit_logger = logging.getLogger(WRITE_AUDIT_LOGGER_NAME)

_SERVICE_ID = "sagemaker-featurestore-runtime"
_UNIQUE_ID_PREFIX = "sagemaker-mlops-feature-store-write-audit"
_TRUTHY_VALUES = ("1", "true", "yes", "on")


@dataclass(frozen=True)
class WriteAuditConfig:
    """Settings for write audit logging of one Feature Group.

    Attributes:
        record_identifier_feature_name (str): the Feature Group's RecordIdentifierFeatureName.
        event_time_feature_name (str): the Feature Group's EventTimeFeatureName.
    """

    record_identifier_feature_name: str
    event_time_feature_name: str


def _write_audit_enabled_from_env() -> bool:
    """Return True if ``SAGEMAKER_FEATURE_STORE_WRITE_AUDIT`` is set to a truthy value."""
    return os.environ.get(WRITE_AUDIT_ENV_VAR, "").strip().lower() in _TRUTHY_VALUES


def _request_id_from_exception(error: BaseException) -> Optional[str]:
    """Return the AWS request ID carried by a botocore ``ClientError``, if any."""
    response = getattr(error, "response", None)
    metadata = response.get("ResponseMetadata") if isinstance(response, Mapping) else None
    return metadata.get("RequestId") if isinstance(metadata, Mapping) else None


# --------------------------------------------------------------------------- registry


class _AuditRegistry:
    """Which Feature Groups are currently being audited in this process, and how.

    Entries are added for the duration of an ingestion and removed afterwards, so writes
    outside an audited ingestion are never logged. Concurrent registrations of the same
    Feature Group are tracked separately; the most recent one wins while both are active.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._entries: Dict[str, List[Tuple[int, WriteAuditConfig]]] = {}
        self._next_token = 0

    def register(self, feature_group_name: str, config: WriteAuditConfig) -> int:
        """Start auditing a Feature Group and return a token for ``unregister``."""
        with self._lock:
            self._next_token += 1
            self._entries.setdefault(feature_group_name, []).append((self._next_token, config))
            return self._next_token

    def unregister(self, feature_group_name: str, token: int) -> None:
        """Stop the registration identified by ``token``."""
        with self._lock:
            remaining = [e for e in self._entries.get(feature_group_name, []) if e[0] != token]
            if remaining:
                self._entries[feature_group_name] = remaining
            else:
                self._entries.pop(feature_group_name, None)

    def get(self, feature_group_name: Optional[str]) -> Optional[WriteAuditConfig]:
        """Return the active config for a Feature Group, or None if it is not audited."""
        with self._lock:
            entries = self._entries.get(feature_group_name)
            return entries[-1][1] if entries else None


# --------------------------------------------------------------------------- records


@dataclass(frozen=True)
class _RecordRef:
    """Identity of one record in a write request.

    ``entry_key`` is the record's full content, used only to match BatchWriteRecord
    errors back to entries; it is never logged.
    """

    feature_group_name: Optional[str]
    record_id: Optional[str]
    event_time: Optional[str]
    audited: bool = False
    entry_key: str = field(default="", repr=False, compare=False)


class _RecordRefExtractor:
    """Builds a ``_RecordRef`` from a Feature Group name and a list of FeatureValues."""

    def __init__(self, registry: _AuditRegistry):
        self._registry = registry

    def extract(
        self, feature_group_name: Optional[str], record: Optional[Sequence[Mapping[str, Any]]]
    ) -> _RecordRef:
        """Extract the record identifier and event time named by the active config."""
        entry_key = json.dumps(record or [], sort_keys=True, default=str)
        config = self._registry.get(feature_group_name)
        if config is None:
            return _RecordRef(feature_group_name, None, None, entry_key=entry_key)
        values = {f.get("FeatureName"): f.get("ValueAsString") for f in record or ()}
        return _RecordRef(
            feature_group_name=feature_group_name,
            record_id=values.get(config.record_identifier_feature_name),
            event_time=values.get(config.event_time_feature_name),
            audited=True,
            entry_key=entry_key,
        )

    def extract_entry(self, entry: Mapping[str, Any]) -> _RecordRef:
        """Extract from a BatchWriteRecord entry (``FeatureGroupName`` and ``Record``)."""
        return self.extract(entry.get("FeatureGroupName"), entry.get("Record"))


# --------------------------------------------------------------------------- attempts


def _header(headers: Mapping[str, Any], name: str) -> Optional[str]:
    """Case-insensitive header lookup that does not depend on the mapping type."""
    value = headers.get(name)
    if value is not None:
        return value
    lowered = name.lower()
    return next((v for k, v in headers.items() if str(k).lower() == lowered), None)


@dataclass(frozen=True)
class _AttemptResult:
    """Normalized outcome of a single HTTP attempt."""

    attempt: int
    request_sent_at: Optional[str]
    response_received_at: str
    request_id: Optional[str] = None
    http_status: Optional[int] = None
    server_date: Optional[str] = None
    error_code: Optional[str] = None
    exception: Optional[str] = None

    @property
    def succeeded(self) -> bool:
        """Whether the attempt returned a 2xx status."""
        return self.http_status is not None and 200 <= self.http_status < 300

    @classmethod
    def from_needs_retry(
        cls,
        attempt: int,
        request_sent_at: Optional[str],
        response_received_at: str,
        response: Optional[Sequence[Any]],
        caught_exception: Optional[BaseException],
    ) -> "_AttemptResult":
        """Build from the documented ``needs-retry`` keyword arguments.

        ``response`` is ``(AWSResponse, parsed_dict)``, or None when the attempt raised
        (for example a timeout or connection reset), in which case no request ID exists.
        """
        if not response:
            return cls(
                attempt, request_sent_at, response_received_at, exception=repr(caught_exception)
            )
        http_response, parsed = response[0], response[1] or {}
        headers = getattr(http_response, "headers", None) or {}
        return cls(
            attempt=attempt,
            request_sent_at=request_sent_at,
            response_received_at=response_received_at,
            request_id=_header(headers, "x-amzn-requestid")
            or (parsed.get("ResponseMetadata") or {}).get("RequestId"),
            http_status=getattr(http_response, "status_code", None),
            server_date=_header(headers, "date"),
            error_code=(parsed.get("Error") or {}).get("Code"),
        )


# --------------------------------------------------------------------------- outcomes


class _EntryStatus(str, enum.Enum):
    """Outcome of one record in one attempt."""

    ACCEPTED = "accepted"  # 2xx and not reported as an error or unprocessed entry
    ERROR = "error"  # listed in BatchWriteRecord Errors
    UNPROCESSED = "unprocessed"  # listed in BatchWriteRecord UnprocessedEntries
    REQUEST_FAILED = "request_failed"  # the whole HTTP attempt failed
    UNKNOWN = "unknown"  # 2xx, but reported failures could not all be matched to entries


@dataclass(frozen=True)
class _EntryOutcome:
    """A record together with what happened to it in one attempt."""

    record: _RecordRef
    status: _EntryStatus
    error_code: Optional[str] = None


class _PutRecordAuditor:
    """Records and outcomes for PutRecord (always a single record)."""

    operation_name = "PutRecord"

    def __init__(self, extractor: _RecordRefExtractor):
        self._extractor = extractor

    def records_in_request(self, params: Mapping[str, Any]) -> List[_RecordRef]:
        """Records carried by the request."""
        return [self._extractor.extract(params.get("FeatureGroupName"), params.get("Record"))]

    def entry_outcomes(
        self,
        records: Sequence[_RecordRef],
        parsed_response: Mapping[str, Any],
        attempt: _AttemptResult,
    ) -> List[_EntryOutcome]:
        """Outcome per record for one attempt."""
        del parsed_response  # PutRecord returns no per-record detail
        status = _EntryStatus.ACCEPTED if attempt.succeeded else _EntryStatus.REQUEST_FAILED
        return [_EntryOutcome(record, status) for record in records]


class _BatchWriteRecordAuditor:
    """Records and outcomes for BatchWriteRecord."""

    operation_name = "BatchWriteRecord"

    def __init__(self, extractor: _RecordRefExtractor):
        self._extractor = extractor

    def records_in_request(self, params: Mapping[str, Any]) -> List[_RecordRef]:
        """Records carried by the request, in entry order."""
        return [self._extractor.extract_entry(entry) for entry in params.get("Entries", ())]

    def entry_outcomes(
        self,
        records: Sequence[_RecordRef],
        parsed_response: Mapping[str, Any],
        attempt: _AttemptResult,
    ) -> List[_EntryOutcome]:
        """Outcome per record, using the response's Errors and UnprocessedEntries.

        Reported failures are matched to entries by full record content. If any reported
        failure matches no entry, no entry can be confirmed as accepted, so entries that
        would otherwise be ``accepted`` are reported as ``unknown``.
        """
        if not attempt.succeeded:
            return [_EntryOutcome(record, _EntryStatus.REQUEST_FAILED) for record in records]
        reported = self._reported_failures(parsed_response)
        outcomes = [self._take_reported(reported, record) for record in records]
        unmatched = sum(len(pending) for pending in reported.values())
        if unmatched:
            logger.warning(
                "Could not match %d reported BatchWriteRecord failure(s) to request entries "
                "(request ID: %s); logging unconfirmed entries as 'unknown'",
                unmatched,
                attempt.request_id,
            )
            outcomes = [
                (
                    dataclasses.replace(o, status=_EntryStatus.UNKNOWN)
                    if o.status is _EntryStatus.ACCEPTED
                    else o
                )
                for o in outcomes
            ]
        return outcomes

    def _reported_failures(
        self, parsed_response: Mapping[str, Any]
    ) -> Dict[str, List[Tuple[_EntryStatus, Optional[str]]]]:
        """Reported failures keyed by record content, as lists so duplicates match one-to-one."""
        reported: Dict[str, List[Tuple[_EntryStatus, Optional[str]]]] = {}
        for error in parsed_response.get("Errors", ()):
            key = self._extractor.extract_entry(error.get("Entry") or {}).entry_key
            reported.setdefault(key, []).append((_EntryStatus.ERROR, error.get("ErrorCode")))
        for entry in parsed_response.get("UnprocessedEntries", ()):
            key = self._extractor.extract_entry(entry).entry_key
            reported.setdefault(key, []).append((_EntryStatus.UNPROCESSED, None))
        return reported

    @staticmethod
    def _take_reported(
        reported: Dict[str, List[Tuple[_EntryStatus, Optional[str]]]], record: _RecordRef
    ) -> _EntryOutcome:
        pending = reported.get(record.entry_key)
        if pending:
            status, error_code = pending.pop(0)
            return _EntryOutcome(record, status, error_code)
        return _EntryOutcome(record, _EntryStatus.ACCEPTED)


_Auditor = Union[_PutRecordAuditor, _BatchWriteRecordAuditor]


def _build_audit_event(
    operation_name: str,
    attempt: _AttemptResult,
    entry_index: int,
    batch_size: int,
    outcome: _EntryOutcome,
) -> Dict[str, Any]:
    """Flatten one record's outcome into an operation-independent event (None values dropped)."""
    event: Dict[str, Any] = {
        "operation": operation_name,
        "feature_group_name": outcome.record.feature_group_name,
        "record_id": outcome.record.record_id,
        "event_time": outcome.record.event_time,
        "entry_index": entry_index,
        "batch_size": batch_size,
        "entry_status": outcome.status.value,
        "entry_error_code": outcome.error_code,
        "attempt": attempt.attempt,
        "request_id": attempt.request_id,
        "http_status": attempt.http_status,
        "error_code": attempt.error_code,
        "exception": attempt.exception,
        "request_sent_at": attempt.request_sent_at,
        "response_received_at": attempt.response_received_at,
        "server_date": attempt.server_date,
    }
    return {key: value for key, value in event.items() if value is not None}


# --------------------------------------------------------------------------- botocore adapter


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class _CallState:
    """The audited API call in flight on one thread."""

    auditor: _Auditor
    records: List[_RecordRef]
    request_sent_at: Optional[str] = None

    def take_sent_at(self) -> Optional[str]:
        """Return this attempt's send time and clear it so it cannot leak into the next one."""
        sent_at, self.request_sent_at = self.request_sent_at, None
        return sent_at


class _ThreadCallTracker:
    """The in-flight call per thread (None when the current call is not audited)."""

    def __init__(self):
        self._local = threading.local()

    def start(self, state: Optional[_CallState]) -> None:
        """Begin tracking a new call on this thread, replacing any previous one."""
        self._local.state = state

    def in_flight(self) -> Optional[_CallState]:
        """The call currently tracked on this thread."""
        return getattr(self._local, "state", None)

    def current(self, operation_name: str) -> Optional[_CallState]:
        """The tracked call, if it is for ``operation_name``."""
        state = self.in_flight()
        if state is None or state.auditor.operation_name != operation_name:
            return None
        return state


def _never_raise(handler: Callable) -> Callable:
    """Swallow and report handler failures, and always return None to botocore."""

    @functools.wraps(handler)
    def wrapper(*args, **kwargs):
        # Deliberately returns None: botocore treats any other value as an instruction.
        try:
            handler(*args, **kwargs)
        except Exception:  # pylint: disable=broad-except
            logger.warning("Feature Store write audit handler failed", exc_info=True)

    return wrapper


class _WriteAuditHooks:
    """Connects documented botocore events to the auditors and the audit logger."""

    def __init__(self, registry: _AuditRegistry, clock: Callable[[], str] = _utc_now_iso):
        extractor = _RecordRefExtractor(registry)
        auditors = (_PutRecordAuditor(extractor), _BatchWriteRecordAuditor(extractor))
        self._auditors: Dict[str, _Auditor] = {a.operation_name: a for a in auditors}
        self._clock = clock
        self._calls = _ThreadCallTracker()

    def register(self, events: Any) -> None:
        """Register on a botocore event emitter (``client.meta.events``). Idempotent."""
        handlers = (
            ("provide-client-params", self._on_client_params),
            ("before-send", self._on_before_send),
            ("needs-retry", self._on_needs_retry),
        )
        for operation in self._auditors:
            for event, handler in handlers:
                event_name = f"{event}.{_SERVICE_ID}.{operation}"
                events.register(event_name, handler, unique_id=f"{_UNIQUE_ID_PREFIX}:{event_name}")

    @_never_raise
    def _on_client_params(self, params, model, **_):
        # Always reset first: a call that is not audited must not inherit a previous call.
        self._calls.start(None)
        auditor = self._auditors[model.name]
        records = auditor.records_in_request(params)
        if any(record.audited for record in records):
            self._calls.start(_CallState(auditor, records))

    @_never_raise
    def _on_before_send(self, request, **_):
        del request  # before-send carries no operation name; this thread's call is ours
        state = self._calls.in_flight()
        if state is not None:
            state.request_sent_at = self._clock()

    @_never_raise
    def _on_needs_retry(self, response, attempts, caught_exception, operation, **_):
        state = self._calls.current(operation.name)
        if state is None:
            return
        attempt = _AttemptResult.from_needs_retry(
            attempts, state.take_sent_at(), self._clock(), response, caught_exception
        )
        parsed = (response[1] if response else None) or {}
        outcomes = state.auditor.entry_outcomes(state.records, parsed, attempt)
        for index, outcome in enumerate(outcomes):
            if outcome.record.audited:
                event = _build_audit_event(
                    state.auditor.operation_name, attempt, index, len(outcomes), outcome
                )
                _audit_logger.info(json.dumps(event))


# --------------------------------------------------------------------------- entry point

_REGISTRY = _AuditRegistry()
_HOOKS = _WriteAuditHooks(_REGISTRY)


@contextlib.contextmanager
def _write_audit(
    feature_group_name: str, config: Optional[WriteAuditConfig], region: Optional[str] = None
) -> Iterator[None]:
    """Audit writes to ``feature_group_name`` made in this process while the block runs.

    A no-op when ``config`` is None. Setup failures are logged as warnings and never
    interrupt ingestion.

    Args:
        feature_group_name (str): name (or ARN, exactly as used in the requests) of the
            Feature Group.
        config (WriteAuditConfig): audit settings, or None to disable.
        region (str): the same region the writes use, so the handlers attach to the client
            sagemaker-core caches for that region.
    """
    if config is None:
        yield
        return
    token = None
    try:
        # Imported lazily: sagemaker-core builds boto clients on first use.
        from sagemaker.core.utils.utils import SageMakerClient

        client = SageMakerClient(region_name=region).get_client(_SERVICE_ID)
        _HOOKS.register(client.meta.events)
        token = _REGISTRY.register(feature_group_name, config)
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "Could not enable Feature Store write audit logging; ingesting without it",
            exc_info=True,
        )
    try:
        yield
    finally:
        if token is not None:
            _REGISTRY.unregister(feature_group_name, token)
