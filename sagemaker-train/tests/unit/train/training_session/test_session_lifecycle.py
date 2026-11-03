# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for session lifecycle management (_SessionHandle).

Nothing here touches the network. Each test drops its client inside a helper
frame and forces a collection, so the assertion is about what the finalizer did
after the last reference went away.

The negative tests matter more than the positive ones: a spurious stop kills a
session a training loop is still driving, which is worse than the leak this
feature exists to prevent.
"""

from __future__ import annotations

import gc
import logging
import warnings
import weakref

import pytest

from sagemaker.train.training_session._session_handle import _SessionHandle
from sagemaker.train.training_session.sampling_client import SamplingClient
from sagemaker.train.training_session.training_client import TrainingClient


class FakeServiceClient:
    """Records control-plane lifecycle calls; never performs one."""

    def __init__(
        self,
        *,
        stop_error: Exception | None = None,
        delete_error: Exception | None = None,
        was_active: bool = True,
    ):
        self.stopped: list[str] = []
        self.deleted: list[str] = []
        self._stop_error = stop_error
        self._delete_error = delete_error
        # What _stop_session_if_active reports: False means the session was
        # already stopping or terminal, so the stop was a no-op.
        self._was_active = was_active

    def _get_hub_client(self):
        return None

    def stop_session(self, training_session_name: str) -> None:
        self._stop_session_if_active(training_session_name)

    def _stop_session_if_active(self, training_session_name: str) -> bool:
        self.stopped.append(training_session_name)
        if self._stop_error is not None:
            raise self._stop_error
        return self._was_active

    def delete_session(self, training_session_name: str, **kwargs) -> None:
        self.deleted.append(training_session_name)
        if self._delete_error is not None:
            raise self._delete_error


class FakeTransport:
    """Minimal stand-in; no operation in this module reaches the wire."""

    def execute_sampling_operation(self, **kwargs):  # pragma: no cover
        raise AssertionError("no test here submits an operation")

    def execute_training_operation(self, **kwargs):  # pragma: no cover
        raise AssertionError("no test here submits an operation")


def _collect():
    """Force the finalizers of anything that just became unreachable."""
    gc.collect()


# --- the handle in isolation ------------------------------------------------


def test_handle_stops_session_when_last_reference_drops():
    svc = FakeServiceClient()

    def _scope():
        _SessionHandle("sess-1", svc)

    _scope()
    _collect()
    assert svc.stopped == ["sess-1"]


def test_auto_stop_has_no_opt_out():
    # Deliberate: a session's lifetime is not the caller's problem, so there is
    # no flag or method that leaves one Running. Durable state is a checkpoint.
    handle = _SessionHandle("sess-1", FakeServiceClient())
    assert handle.auto_stop_armed is True
    assert not hasattr(handle, "detach")


def test_handle_without_service_client_cannot_arm():
    # Nothing to issue the stop through, so the handle must not pretend it can.
    handle = _SessionHandle("sess-1", None)
    assert handle.auto_stop_armed is False


def test_disarm_prevents_the_stop():
    svc = FakeServiceClient()

    def _scope():
        handle = _SessionHandle("sess-1", svc)
        handle.disarm()

    _scope()
    _collect()
    assert svc.stopped == []


def test_disarm_is_idempotent():
    svc = FakeServiceClient()
    handle = _SessionHandle("sess-1", svc)
    handle.disarm()
    handle.disarm()
    assert handle.auto_stop_armed is False


def test_the_finalizer_does_not_keep_the_handle_alive():
    # The whole design rests on this: a callback closing over the handle would
    # make it immortal and the cleanup would never run at all.
    svc = FakeServiceClient()
    handle = _SessionHandle("sess-1", svc)
    import weakref

    ref = weakref.ref(handle)
    del handle
    _collect()
    assert ref() is None
    assert svc.stopped == ["sess-1"]


def test_auto_stop_drains_registered_submitters_before_stopping():
    order: list[str] = []

    class RecordingSubmitter:
        def close(self, timeout: float = 5.0) -> None:
            order.append("drain")

    class RecordingService(FakeServiceClient):
        def _stop_session_if_active(self, training_session_name: str) -> bool:
            order.append("stop")
            return super()._stop_session_if_active(training_session_name)

    svc = RecordingService()

    def _scope():
        handle = _SessionHandle("sess-1", svc)
        handle.register_submitter(RecordingSubmitter())

    _scope()
    _collect()
    # Queued operations must reach the service before the stop request.
    assert order == ["drain", "stop"]


def test_stop_failure_during_cleanup_is_swallowed(caplog):
    svc = FakeServiceClient(stop_error=RuntimeError("throttled"))

    def _scope():
        _SessionHandle("sess-1", svc)

    with caplog.at_level("WARNING"):
        _scope()
        _collect()
    # A finalizer must never propagate: the sweeper is the remaining backstop.
    assert "sess-1" in caplog.text


def _not_found_error(code="ResourceNotFound"):
    """The ClientError shape the control plane returns for a missing session."""
    from botocore.exceptions import ClientError

    return ClientError(
        {
            "Error": {"Code": code, "Message": "Training session not found: sess-1"},
            "ResponseMetadata": {"HTTPStatusCode": 400},
        },
        "StopTrainingSession",
    )


@pytest.mark.parametrize("code", ["ResourceNotFound", "ResourceNotFoundException"])
def test_auto_stop_treats_a_deleted_session_as_done(caplog, code):
    """The caller deleted it out of band; the goal state is already reached.

    Warning here would tell someone who cleaned up correctly that the session
    still needs sweeping, which is the opposite of the truth.
    """
    svc = FakeServiceClient(stop_error=_not_found_error(code))

    def _scope():
        _SessionHandle("sess-1", svc)

    with caplog.at_level("DEBUG"):
        _scope()
        _collect()
    assert "no longer exists" in caplog.text
    # Nothing happened, so nothing above DEBUG is emitted.
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG]


def test_auto_stop_still_warns_on_a_real_failure(caplog):
    """Only "it is gone" is absorbed; anything else stays visible."""
    svc = FakeServiceClient(stop_error=_not_found_error(code="AccessDeniedException"))

    def _scope():
        _SessionHandle("sess-1", svc)

    with caplog.at_level("INFO"):
        _scope()
        _collect()
    assert [r for r in caplog.records if r.levelname == "WARNING"]


def test_auto_stop_is_silent_when_the_session_was_already_stopping(caplog):
    """Nothing to report, so report nothing, not even the ResourceWarning.

    Telling a caller who already stopped the session that they forgot to would
    be both noise and untrue.
    """
    svc = FakeServiceClient(was_active=False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level("DEBUG"):

            def _scope():
                _SessionHandle("sess-quiet", svc)

            _scope()
            _collect()

    assert svc.stopped == ["sess-quiet"], "it must still attempt the stop"
    assert not [w for w in caught if w.category is ResourceWarning]
    assert not [r for r in caplog.records if r.levelno > logging.DEBUG]


def test_auto_stop_reports_when_it_really_stopped_something(caplog):
    """The other half of the contract: a real stop is still announced."""
    svc = FakeServiceClient(was_active=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with caplog.at_level("INFO"):

            def _scope():
                _SessionHandle("sess-loud", svc)

            _scope()
            _collect()

    assert [w for w in caught if w.category is ResourceWarning]
    assert "Auto-stopped unreferenced training session" in caplog.text


def test_auto_stop_emits_resource_warning():
    svc = FakeServiceClient()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        def _scope():
            _SessionHandle("sess-noisy", svc)

        _scope()
        _collect()
    messages = [str(w.message) for w in caught if w.category is ResourceWarning]
    assert any("sess-noisy" in m for m in messages)


# --- the invariant: stop, never delete --------------------------------------


def test_auto_stop_never_deletes_the_session():
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-1", FakeTransport(), service_client=svc)
        return tc.create_sampling_client()

    sp = _scope()
    del sp
    _collect()
    assert svc.stopped == ["sess-1"]
    assert svc.deleted == [], "the automatic path must never issue a delete"


def test_explicit_delete_disarms_so_no_stop_follows_it():
    # Otherwise a later GC would stop a session name the caller already removed.
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-1", FakeTransport(), service_client=svc)
        tc.delete()

    _scope()
    _collect()
    assert svc.deleted == ["sess-1"]
    assert svc.stopped == []


def test_failed_delete_leaves_the_automatic_stop_armed():
    """A refused delete must not cost the session its last client-side release.

    ``delete(stop_if_active=False)``, the default, is refused for a session
    that is still running. Detaching is one-way, so disarming before the call
    left a running session with nothing trying to release it: it held its GPUs
    until the inactivity timeout.
    """
    svc = FakeServiceClient(delete_error=RuntimeError("session is not terminal"))

    def _scope():
        tc = TrainingClient("sess-1", FakeTransport(), service_client=svc)
        try:
            tc.delete()
        except RuntimeError as exc:
            # Cleared deliberately: the traceback holds this frame, and with it
            # the client, which would keep the handle reachable past _collect.
            assert "not terminal" in str(exc)
            exc.__traceback__ = None
        else:
            raise AssertionError("delete should have propagated the refusal")

    _scope()
    _collect()
    assert svc.deleted == ["sess-1"]
    assert svc.stopped == ["sess-1"], "the automatic stop is the only release left"


def test_lifecycle_without_a_service_client_names_the_lifecycle_call():
    """The error has to be about stopping, not about listing checkpoints."""
    tc = TrainingClient("sess-1", FakeTransport())
    for call in (tc.stop, tc.delete):
        with pytest.raises(ValueError, match="cannot stop or delete the session"):
            call()


# --- SamplingClient over an implicit (owned) session ------------------------


def test_owned_sampling_session_is_stopped_when_client_drops():
    svc = FakeServiceClient()

    def _scope():
        SamplingClient(
            "sess-implicit",
            FakeTransport(),
            service_client=svc,
        )

    _scope()
    _collect()
    assert svc.stopped == ["sess-implicit"]


def test_explicit_stop_then_drop_stops_exactly_once():
    svc = FakeServiceClient()

    def _scope():
        sp = SamplingClient(
            "sess-implicit",
            FakeTransport(),
            service_client=svc,
        )
        sp.stop()

    _scope()
    _collect()
    assert svc.stopped == ["sess-implicit"]


def test_context_manager_exit_then_drop_stops_exactly_once():
    svc = FakeServiceClient()

    def _scope():
        with SamplingClient(
            "sess-implicit",
            FakeTransport(),
            service_client=svc,
        ):
            pass

    _scope()
    _collect()
    assert svc.stopped == ["sess-implicit"]


# --- shared session: TrainingClient + derived SamplingClient ----------------
#
# These four are the regression tests that keep the automatic stop from becoming an
# outage. Dropping either client must leave the session Running while the other
# is still usable, in EITHER drop order.


def test_dropping_training_client_keeps_session_for_live_sampling_client():
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-shared", FakeTransport(), service_client=svc)
        return tc.create_sampling_client()

    sp = _scope()
    _collect()
    assert svc.stopped == [], "a SamplingClient is still bound to this session"
    # ...and the surviving client stops it when it in turn goes away.
    del sp
    _collect()
    assert svc.stopped == ["sess-shared"]


def test_dropping_sampling_client_keeps_session_for_live_training_client():
    svc = FakeServiceClient()
    tc = TrainingClient("sess-shared", FakeTransport(), service_client=svc)

    # Takes the client as an argument rather than closing over it, so the only
    # reference dropped below is the one this test holds.
    def _scope(client):
        client.create_sampling_client()

    _scope(tc)
    _collect()
    assert svc.stopped == [], "the TrainingClient is still driving this session"
    del tc
    _collect()
    assert svc.stopped == ["sess-shared"]


def test_session_survives_until_the_last_of_several_sampling_clients_drops():
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-shared", FakeTransport(), service_client=svc)
        return tc.create_sampling_client(), tc.create_sampling_client()

    first, second = _scope()
    _collect()
    assert svc.stopped == []
    del first
    _collect()
    assert svc.stopped == [], "one sibling is still bound"
    del second
    _collect()
    assert svc.stopped == ["sess-shared"]


def test_derived_sampling_client_shares_the_training_clients_handle():
    tc = TrainingClient("sess-shared", FakeTransport(), service_client=FakeServiceClient())
    sp = tc.create_sampling_client()
    assert sp._handle is tc._handle


def test_noop_stop_on_derived_client_does_not_disarm_the_shared_session():
    # SamplingClient.stop() is documented as a no-op for a shared session. If it
    # also disarmed the handle, that no-op would silently turn into a leak.
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-shared", FakeTransport(), service_client=svc)
        sp = tc.create_sampling_client()
        sp.stop()
        assert svc.stopped == []
        assert tc._handle.auto_stop_armed is True

    _scope()
    _collect()
    assert svc.stopped == ["sess-shared"]


def test_clients_expose_no_lifecycle_opt_out():
    # The public surface for a session is stop() / delete() and the `with` block.
    # Nothing lets a caller leave one Running; see test_auto_stop_has_no_opt_out.
    tc = TrainingClient("sess-shared", FakeTransport(), service_client=FakeServiceClient())
    sp = tc.create_sampling_client()
    assert not hasattr(tc, "detach")
    assert not hasattr(sp, "detach")


# --- outstanding operations -------------------------------------------------


def test_future_keeps_the_session_alive_after_its_client_is_dropped():
    # Dropping the client while a future is unresolved must not stop the session
    # the operation is running against.
    svc = FakeServiceClient()

    def _scope():
        tc = TrainingClient("sess-1", FakeTransport(), service_client=svc)
        from sagemaker.train.training_session.futures import PendingAPIFuture

        return PendingAPIFuture(submit_future=None, operation_type="training", keepalive=tc._handle)

    future = _scope()
    _collect()
    assert svc.stopped == [], "an operation against this session is outstanding"
    del future
    _collect()
    assert svc.stopped == ["sess-1"]


# --- ServiceClient factory wiring ------------------------------------------


def test_service_client_factory_arms_auto_stop(monkeypatch):
    sp = _sampling_client_from_service_client(monkeypatch)
    assert sp._handle.auto_stop_armed is True


def test_implicit_sampling_client_owns_its_session(monkeypatch):
    # Ownership is derived from "no handle was passed in", not from a flag.
    sp = _sampling_client_from_service_client(monkeypatch)
    assert sp._owns_session is True


def test_first_client_from_a_service_client_is_collectable(monkeypatch):
    """Regression: a first-time import must not pin the client a factory returns.

    CPython retains the frame chain of a first-time import. When
    ``_get_hub_client`` performed its lazy imports on first use — inside
    ``create_sampling_client`` — that pinned the factory's frame and therefore
    the client it returned, so the client was never collected and its session
    stayed Running until interpreter exit. Only the first call in a process
    performs the import, so only the first client was affected.
    ``ServiceClient.__init__`` now warms those imports instead.

    Asserted structurally rather than by simulating a cold import, which cannot
    be undone within a process: the guarantee is that the factory does not
    trigger the import.
    """
    import sys

    from sagemaker.train.training_session import service_client as sc_module

    sp = _sampling_client_from_service_client(monkeypatch)
    ref = weakref.ref(sp)
    del sp
    gc.collect()
    assert ref() is None, "the client a factory returns must be collectable"
    # And the modules whose first-time import caused the pin are already loaded
    # by the time any factory runs.
    assert hasattr(sc_module, "_warm_hub_imports")
    assert "boto3" in sys.modules


def _sampling_client_from_service_client(monkeypatch):
    """Build a real ServiceClient with the control plane and Hub stubbed out."""
    from sagemaker.train.training_session import service_client as sc_module

    class StubControlPlane:
        def __init__(self, **kwargs):
            pass

        def create_training_session(self, **kwargs):
            return {"TrainingSessionArn": "arn:aws:sagemaker:::training-session/s"}

    monkeypatch.setattr(
        "sagemaker.train.training_session._clients._control_plane_client",
        lambda **kwargs: StubControlPlane(),
    )
    monkeypatch.setattr(
        "sagemaker.train.training_session._clients._runtime_client",
        lambda **kwargs: object(),
    )

    svc = sc_module.ServiceClient(
        region="us-west-2",
        role_arn="arn:aws:iam::123456789012:role/R",
        model_package_group_arn="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/g",
        s3_output_path="s3://bucket/out",
    )
    monkeypatch.setattr(svc, "_wait_until_running", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_get_hub_client", lambda: None)

    return svc.create_sampling_client(base_model="arn:aws:sagemaker:::hub-content/m")
