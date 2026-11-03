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
"""Background submission machinery for non-blocking client methods.

Base methods stamp submission order at call time, hand the network submit to a
dedicated background event loop, and return a future immediately, so callers
never wait on a submit round trip. Ordered submits (training operations) are
released strictly in call order; unordered submits (sampling operations) run
concurrently. A failed submit raises when the caller resolves the future, not at
call time.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from typing import Any, Awaitable, Callable, Optional

from sagemaker.train.training_session.exceptions import TrainingSessionError

#: Raised when a client is asked for another operation after it has been closed.
#: Worded for the caller: what closes a client is stopping or deleting its
#: session, or leaving its ``with`` block.
_CLOSED_MESSAGE = (
    "This client is closed and accepts no new operations. Stopping or deleting "
    "the session closes its client, as does leaving the client's `with` block. "
    "Create a new client to continue."
)

#: Raised on an operation that had been submitted but had not finished when its
#: client closed. Worded for the caller: the operation's outcome is unknown, and
#: what to do about it depends on the operation.
_ABANDONED_MESSAGE = (
    "This operation was still in flight when its client closed, so its result is "
    "unknown. Keep the client open until the operations you submitted have "
    "resolved."
)


class BackgroundSubmitter:
    """Owns a daemon thread running an asyncio loop for submits."""

    def __init__(self, name: str = "sagemaker-training-submitter") -> None:
        self._name = name
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._next_seq = 1
        self._release_seq = 1
        #: Sequence number -> the future that releases it. Touched only on the
        #: submitter's loop thread, so it needs no lock.
        self._waiters: "dict[int, asyncio.Future[None]]" = {}
        #: Sequences that will never run, recorded by whichever of them was
        #: cancelled before reaching its turn. The holder skips past these when it
        #: advances, so a cancellation cannot let a later submit overtake a
        #: running one. Drained as the turn passes them, so it stays small.
        self._skipped: "set[int]" = set()
        self._closed = False
        #: Futures for submits that have been accepted and not yet finished.
        #: :meth:`close` waits on exactly these, rather than on whatever tasks the
        #: loop happens to hold at the moment it looks.
        self._pending: "set[concurrent.futures.Future[Any]]" = set()

    def submit_ordered(
        self, coro_fn: Callable[[], Awaitable[Any]]
    ) -> "concurrent.futures.Future[Any]":
        """Schedule coro_fn so submits run strictly in call order.

        Returns a concurrent.futures.Future resolving to the coroutine result.

        The sequence slot is taken and scheduled under one lock, which
        :meth:`close` also takes before stopping the loop. A slot can therefore
        never be handed out to a coroutine that the loop stops before running:
        such a slot would never be released, and every later ordered submit
        waits on its turn forever.
        """
        with self._lock:
            loop = self._ensure_loop_locked()
            seq = self._next_seq
            coro = self._run_in_turn(seq, coro_fn)
            try:
                future = asyncio.run_coroutine_threadsafe(coro, loop)
            except RuntimeError:
                # Nothing was scheduled, so the slot must stay unclaimed.
                coro.close()
                raise
            self._next_seq += 1
            self._pending.add(future)
        future.add_done_callback(self._forget)
        return future

    def submit_unordered(
        self, coro_fn: Callable[[], Awaitable[Any]]
    ) -> "concurrent.futures.Future[Any]":
        """Schedule coro_fn with no ordering constraint."""
        with self._lock:
            loop = self._ensure_loop_locked()
            future = asyncio.run_coroutine_threadsafe(self._run_unordered(coro_fn), loop)
            self._pending.add(future)
        future.add_done_callback(self._forget)
        return future

    def close(self, timeout: float = 5.0) -> None:
        """Let accepted submits finish, stop the loop, close it, join the thread.

        Idempotent. A submit after close raises :class:`TrainingSessionError`.

        Marking the submitter closed comes first, so the set of accepted submits
        is final by the time it is read, and each is then waited on directly. The
        wait cannot start before a submit that was already accepted becomes
        visible: if the loop stopped ahead of one, that submit would never run
        and the caller would wait on its result forever.

        A submit still unfinished when the wait runs out is failed rather than
        left alone. Stopping the loop freezes it where it is, so its result would
        never arrive, and a caller waiting on that result with the default
        unbounded timeout would wait for the life of the process.
        """
        with self._lock:
            self._closed = True
            loop = self._loop
            thread = self._thread
            pending = list(self._pending)
        if loop is None:
            # No submit ever ran, so there is no loop and nothing to drain.
            return
        if loop.is_closed():
            # An earlier call already finished the shutdown, and this one must
            # stay quiet: ``call_soon_threadsafe`` on a closed loop raises.
            return

        # Deliberately not gated on ``is_running()``. A loop whose thread has
        # started but has not yet entered ``run_forever`` reports False, and
        # returning on that skips both the drain and the stop, leaving the loop
        # running with submits on it. ``call_soon_threadsafe`` is safe either
        # way: a loop that has not started yet runs the callback when it does.
        concurrent.futures.wait(pending, timeout=timeout)
        for future in pending:
            try:
                future.set_exception(TrainingSessionError(_ABANDONED_MESSAGE))
            except concurrent.futures.InvalidStateError:
                # The loop thread completed this submit between the wait ending
                # and this line. Its real result stands, and letting the error
                # escape here would skip the stop and join below, leaking the
                # thread this is meant to shut down.
                pass
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                # Join ran out with the loop still spinning; closing it now
                # would raise. Leave it open for a later close() to finish.
                return
        # Stopping a loop only parks it: it keeps its readiness poller and its
        # self-pipe pair until it is collected, which is whenever the last
        # reference to this submitter goes. Close so they come back here.
        loop.close()

    def _forget(self, future: "concurrent.futures.Future[Any]") -> None:
        """Drop a finished submit, so the pending set does not grow all session.

        Registered as a done callback outside the lock: an already-finished future
        runs the callback on the registering thread, which would otherwise
        deadlock on the lock that thread still holds.
        """
        with self._lock:
            self._pending.discard(future)

    def _ensure_loop_locked(self) -> asyncio.AbstractEventLoop:
        if self._closed:
            raise TrainingSessionError(_CLOSED_MESSAGE)
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._thread = threading.Thread(
                target=self._loop.run_forever, name=self._name, daemon=True
            )
            self._thread.start()
        return self._loop

    async def _run_in_turn(self, seq: int, coro_fn: Callable[[], Awaitable[Any]]) -> Any:
        """Wait for ``seq``'s turn, run the submit, then release the next one.

        Every path out of here retires ``seq``, cancellation included: the retire
        is in a ``finally`` that only touches state and never awaits. Awaiting
        there would be a second cancellation point, and a sequence that is never
        retired strands every later submit on its turn. Cancelling a task only
        requests cancellation, so this body always reaches its ``try`` and
        therefore always runs its ``finally``, even when the cancel is requested
        before the task is first scheduled.

        Waiters are plain futures rather than a condition variable for the same
        reason: waking the next sequence takes no lock, so it cannot be
        interrupted. The counter, the waiter map and the skipped set are touched
        only on the submitter's single loop thread, so they need no locking.
        """
        loop = asyncio.get_running_loop()
        try:
            if self._release_seq != seq:
                waiter = loop.create_future()
                self._waiters[seq] = waiter
                try:
                    await waiter
                finally:
                    self._waiters.pop(seq, None)
            return await coro_fn()
        finally:
            self._retire(seq)

    def _retire(self, seq: int) -> None:
        """Retire ``seq``, advancing the turn only if ``seq`` was holding it.

        Advancing unconditionally would let a sequence cancelled while queued
        hand the turn on for a slot it never held, waking the next submit while
        the current one is still on the wire. So a sequence that never reached
        its turn is only recorded, and the holder skips past it on its way out.
        """
        if seq != self._release_seq:
            self._skipped.add(seq)
            return
        self._release_seq += 1
        while self._release_seq in self._skipped:
            self._skipped.discard(self._release_seq)
            self._release_seq += 1
        successor = self._waiters.get(self._release_seq)
        if successor is not None and not successor.done():
            successor.set_result(None)

    async def _run_unordered(self, coro_fn: Callable[[], Awaitable[Any]]) -> Any:
        return await coro_fn()


def run_blocking(fn: Callable[[], Any]) -> Callable[[], Awaitable[Any]]:
    """Wrap a blocking callable as a coroutine function for ``submit_*``.

    The callable runs on the loop's default thread pool, so a slow wire call
    never stalls the submitter loop: unordered submits keep overlapping, and
    an ordered submit still holds its turn until the call completes (wire
    order stays equal to call order).
    """

    async def _coro() -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)

    return _coro
