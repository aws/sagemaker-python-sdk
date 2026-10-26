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
"""Refcounted, stop-once ownership of one TrainingSession's lifetime.

A TrainingSession is live compute, but a client object is
just a handle to it. Those two lifetimes are not the same: several clients can
share one session (``TrainingClient.create_sampling_client`` hands out a
``SamplingClient`` bound to the training session), so no single client can be
the one that stops it.

:class:`_SessionHandle` is the session's lifetime, factored out of the clients.
Every client bound to a session holds a strong reference to the *same* handle,
so the handle is what CPython collects last. Its finalizer therefore runs
exactly when the final client referencing the session goes away: no ownership
flags, no parent/child bookkeeping, and no need to transfer responsibility from
a dropped parent to a surviving child.

The automatic teardown stops the session; deleting one stays an explicit call.
See :func:`_auto_stop`.
"""

from __future__ import annotations

import logging
import sys
import threading
import warnings
import weakref
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def _drain_submitters(submitters: List[Any]) -> None:
    """Close every registered background submitter, best effort.

    Draining before the stop request means operations already queued reach the
    service first. ``BackgroundSubmitter.close`` is idempotent, so it does not
    matter whether a client's own finalizer already drained it.
    """
    for submitter in submitters:
        try:
            submitter.close()
        except Exception:  # pragma: no cover - shutdown-time best effort
            logger.debug("submitter drain failed during cleanup", exc_info=True)


#: Error codes meaning the session no longer exists. Absorbed only on the
#: automatic path; see :func:`_session_gone`.
_SESSION_GONE_CODES = frozenset({"ResourceNotFound", "ResourceNotFoundException"})


def _session_gone(error: Exception) -> bool:
    """True when a stop failed because there is no such session any more.

    For a best-effort teardown that is the goal state, not a failure: the caller
    deleted the session out of band and the automatic stop simply arrived after
    them. Duck-typed on ``response`` so this module needs no ``botocore`` import.

    :meth:`ServiceClient.stop_session` deliberately does *not* absorb this:
    an explicit stop against a name that does not exist is a caller mistake and
    must surface.
    """
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return False
    return (response.get("Error") or {}).get("Code") in _SESSION_GONE_CODES


def _warn_not_stopped_explicitly(session_name: str) -> None:
    """Tell the caller a session they still held had to be released for them."""
    try:
        warnings.warn(
            f"training session {session_name!r} was not stopped explicitly; "
            f"stopping it now. Use the client as a context manager, or call "
            f".stop(), to control when the session is released.",
            ResourceWarning,
        )
    except Exception:  # pragma: no cover - warnings machinery may be torn down
        pass


def _auto_stop(
    session_name: str,
    service_client: Any,
    submitters: List[Any],
    explicit_stop_failed: List[bool],
) -> None:
    """Stop ``session_name``: the last client referencing it became unreachable.

    Module-level on purpose. :func:`weakref.finalize` stores its arguments, so a
    callback that closed over the :class:`_SessionHandle` would keep that handle
    permanently alive and the cleanup would never fire at all.

    Stop is the teardown action here, which is what makes an automatic teardown
    safe to do on the caller's behalf:

    * A stopped session is still describable, so its history stays available to
      audit and a run that went wrong can still be inspected.
    * :meth:`ServiceClient.stop_session` is idempotent, so this stop is safe
      whether or not the caller already stopped the session themselves.
    * Stopping already releases the compute the session was using; what remains
      is metadata.

    ``TrainingClient.delete`` / ``ServiceClient.delete_session`` stay
    explicit-call-only and are unreachable from here.

    Stays quiet when there was nothing to stop. Both the ``ResourceWarning`` and
    the log line report an action, so they are emitted only once one has
    happened: a caller who already tore the session down, or whose session the
    service has since removed, should not be told they leaked something.

    ``explicit_stop_failed`` non-empty means the caller did call ``stop()`` and
    the request raised, so this stop retries theirs instead of cleaning up after
    them: the stop still happens, but the ``ResourceWarning`` telling them to
    call ``stop()`` would be wrong and is suppressed.
    """
    # Logged, not acted on: whether a signed HTTPS call still works this late in
    # interpreter teardown is CPython-version and host dependent, so the attempt
    # below is made either way.
    finalizing = sys.is_finalizing()

    # Before the stop either way, so queued operations reach the service first.
    _drain_submitters(submitters)

    try:
        stopped = service_client._stop_session_if_active(session_name)
        if not stopped:
            logger.debug(
                "Unreferenced training session %r was already stopping or "
                "terminal; nothing to stop (interpreter finalizing=%s).",
                session_name,
                finalizing,
            )
            return
        if not explicit_stop_failed:
            _warn_not_stopped_explicitly(session_name)
        logger.info(
            "Auto-stopped unreferenced training session %r (interpreter " "finalizing=%s)",
            session_name,
            finalizing,
        )
    except Exception as e:
        if _session_gone(e):
            logger.debug(
                "Unreferenced training session %r no longer exists; nothing to "
                "stop (interpreter finalizing=%s).",
                session_name,
                finalizing,
            )
            return
        logger.warning(
            "Auto-stop of unreferenced training session %r failed (interpreter "
            "finalizing=%s); it will be stopped automatically after its "
            "inactivity timeout.",
            session_name,
            finalizing,
            exc_info=True,
        )


class _SessionHandle:
    """The lifetime of one TrainingSession, shared by every client bound to it.

    Args:
        session_name: Name of the backing TrainingSession resource.
        service_client: :class:`ServiceClient` used to issue the stop. When
            ``None`` the handle cannot stop anything, so the automatic stop is
            disabled.

    The stop is unconditional whenever a ``service_client`` is available, so a
    caller gets the session released without having to track its lifetime.
    Durability comes from the *checkpoint*, a model package durable on its
    own, so trained weights outlive the process and a new session can be
    created from one at any time.
    """

    def __init__(
        self,
        session_name: str,
        service_client: Any = None,
    ) -> None:
        self.session_name = session_name
        self._service_client = service_client
        self._lock = threading.Lock()
        # Held strongly: the drain must still be possible after the owning
        # clients are gone, and a BackgroundSubmitter's daemon thread would
        # otherwise outlive its client with queued submits never dispatched.
        self._submitters: List[Any] = []
        # A shared list, not a bool attribute, for the reason _auto_stop is
        # module-level: an argument reaching back to this handle would keep it
        # alive and the cleanup would never fire.
        self._explicit_stop_failed: List[bool] = []
        self._finalizer: Optional[weakref.finalize] = None
        if service_client is not None:
            self._finalizer = weakref.finalize(
                self,
                _auto_stop,
                session_name,
                service_client,
                self._submitters,
                self._explicit_stop_failed,
            )

    @property
    def auto_stop_armed(self) -> bool:
        """True while an unreferenced-session stop is still pending."""
        return self._finalizer is not None and self._finalizer.alive

    def register_submitter(self, submitter: Any) -> None:
        """Attach a client's background submitter to this session's teardown."""
        with self._lock:
            self._submitters.append(submitter)

    def note_explicit_stop_failed(self) -> None:
        """Record that the caller's own ``stop()`` was issued and raised.

        The automatic stop stays armed so it can still release the session, but
        it no longer warns that the caller never stopped it: they did, and the
        request failed. Idempotent.
        """
        if not self._explicit_stop_failed:
            self._explicit_stop_failed.append(True)

    def disarm(self) -> None:
        """Cancel the automatic stop: the session is already torn down.

        Called once an explicit ``stop()`` has succeeded, and before ``delete()``,
        which removes the session name the automatic stop would otherwise use.
        Saves a second network call and keeps the ``ResourceWarning`` from
        reaching a caller who *did* clean up.

        Not called when a stop *fails*: detaching is one-way, so disarming there
        would drop the only client-side retry the session has left. Idempotent.
        """
        if self._finalizer is not None:
            self._finalizer.detach()
