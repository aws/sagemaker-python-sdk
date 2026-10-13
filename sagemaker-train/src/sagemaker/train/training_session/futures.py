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
"""APIFuture and request-status model for async training/sample operations.

A single ``APIFuture`` serves training, sampling and checkpointing operations.
It is generic in the type it resolves to: each method returns an
``APIFuture[T]`` whose ``result()`` is a ``T``, the operation's own response
type. ``APIFuture`` is the one future type callers see; the two subclasses below
are internal, so a method that may return either is still annotated
``APIFuture[T]``.

``timeout`` is the only knob callers pass, and it bounds the whole wait on every
future in this module. A future that resolves in stages
(:class:`PendingAPIFuture` waits for its submit and then polls;
:class:`CombinedFuture` waits for several sub-futures) turns ``timeout`` into a
single deadline at the top of ``result()`` and hands each stage only the time
still left, so ``result(timeout=60)`` returns or raises :class:`TimeoutError`
within 60 seconds regardless of how many stages ran. The default is
``timeout=None`` waits indefinitely; pass an explicit ``timeout`` to bound a
specific wait.

Every future can be waited on either way. ``result()`` blocks the calling
thread. ``await`` (or :meth:`result_async`) suspends a coroutine instead: no
thread is held while the operation runs, only for the individual status
requests. So ``asyncio.gather`` over hundreds of futures keeps them all in
flight.

The polling cadence is the SDK's to pick, so it is set at construction rather
than per wait: ``poll_interval``, ``time_fn``, ``sleep_fn``, ``async_sleep_fn``
and ``rand_fn`` are constructor seams the SDK and its tests use (tests inject a
fake clock and a fixed jitter draw, and run without real waits). Callers receive
futures, never build them.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import random
import threading
import time
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from sagemaker.train.training_session.exceptions import (
    OperationError,
    TrainingSessionError,
    is_retryable,
)

#: The names this module contributes to the public API. The future subclasses
#: below are returned, never constructed by callers, so they are reached through
#: :class:`APIFuture` rather than by name; the defaults and helpers around them
#: are wiring. Naming the surface here keeps the generated reference to what a
#: caller can act on.
__all__ = ["APIFuture", "RequestStatus"]

#: The type an :class:`APIFuture` resolves to.
_T = TypeVar("_T")

#: Default seconds between poll attempts.
DEFAULT_POLL_INTERVAL = 2
#: Default wait bound for ``result()``. ``None`` means wait indefinitely, which
#: is the default: a long-running operation that succeeds should not raise
#: ``TimeoutError`` just because it outran an arbitrary clock. Pass an explicit
#: ``timeout`` (seconds) to bound a specific wait.
DEFAULT_TIMEOUT = None
#: Max consecutive transient failures tolerated on a submit or a poll before
#: giving up: about five minutes at the backoff below. Enough to ride out a
#: traffic spike, while an endpoint that is genuinely unavailable still surfaces
#: instead of spinning forever. To bound a wait by elapsed time instead, pass it
#: an explicit ``timeout``.
DEFAULT_MAX_TRANSIENT_POLL_ERRORS = 150
#: Full-jitter exponential backoff bounds, in seconds. The ceiling doubles per
#: consecutive failure from the base and holds at the max; the delay slept is
#: drawn from below it.
DEFAULT_BACKOFF_BASE_SECONDS = 0.5
DEFAULT_BACKOFF_MAX_SECONDS = 4.0


#: Marks a future that has not resolved yet. A plain ``None`` cannot: it is a
#: value a ``parse_fn`` may legitimately produce, and treating it as "unresolved"
#: would send the future back to the wire on every ``result()`` call.
_UNSET: Any = object()


def _backoff_delay(
    attempt: int,
    *,
    base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    cap: float = DEFAULT_BACKOFF_MAX_SECONDS,
    rand_fn: Callable[[float, float], float] = random.uniform,
) -> float:
    """Seconds to wait before retry ``attempt`` (0-based), with full jitter.

    Backing off eases the request rate a throttled service just refused. Drawing
    the delay from below the ceiling, rather than sleeping the ceiling itself,
    spreads callers throttled together so they do not all return together.
    """
    ceiling = min(cap, base * (2**attempt))
    return rand_fn(0.0, ceiling)


def call_with_transient_retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int = DEFAULT_MAX_TRANSIENT_POLL_ERRORS,
    backoff_base: float = DEFAULT_BACKOFF_BASE_SECONDS,
    backoff_max: float = DEFAULT_BACKOFF_MAX_SECONDS,
    sleep_fn: Callable[[float], None] = time.sleep,
    rand_fn: Callable[[float, float], float] = random.uniform,
) -> Any:
    """Call ``fn``, retrying transient failures up to ``max_attempts``.

    For idempotent runtime calls (e.g. ExecuteTrainingOperation submitted with a
    stable clientToken + sequenceNumber). Errors that
    :func:`~sagemaker.train.training_session.exceptions.is_retryable` rejects
    propagate immediately; the rest wait out a :func:`_backoff_delay` first, so a
    throttled submit is not sent straight back at the rate that throttled it.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as error:
            if not is_retryable(error):
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise
            sleep_fn(
                _backoff_delay(
                    attempt - 1,
                    base=backoff_base,
                    cap=backoff_max,
                    rand_fn=rand_fn,
                )
            )


class RequestStatus(str, Enum):
    """Lifecycle states of an asynchronous training or sampling operation.

    An operation moves from ``Pending`` to ``InProgress`` to a terminal state:
    ``Completed`` (success), ``Failed`` (the operation errored), or ``Skipped``
    (the session failed and abandoned its remaining queued operations).
    """

    PENDING = "Pending"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    SKIPPED = "Skipped"


#: Terminal success state.
_SUCCESS_STATUS = RequestStatus.COMPLETED

#: Terminal failure states: reaching any of these raises OperationError.
#: (``Skipped`` is terminal: it marks operations abandoned after the session
#: itself failed.)
_TERMINAL_FAILURE_STATUSES = frozenset({RequestStatus.FAILED, RequestStatus.SKIPPED})

#: Non-terminal states: keep polling. Named so the three sets can be asserted to
#: partition :class:`RequestStatus` exactly, which is what makes an unrecognized
#: state impossible to poll on forever.
_PENDING_STATUSES = frozenset({RequestStatus.PENDING, RequestStatus.IN_PROGRESS})


def _format_timeout(timeout: Optional[float]) -> str:
    """Render a wait bound for a message, tolerating an unbounded one.

    ``None`` means wait indefinitely, and formatting it as a number raises,
    which would replace whatever error was being reported with a ``TypeError``.
    """
    return "no limit" if timeout is None else f"{timeout:.1f}s"


def _deadline_from(timeout: Optional[float], time_fn: Callable[[], float]) -> Optional[float]:
    """Absolute monotonic deadline for ``timeout``, or ``None`` for unbounded."""
    return None if timeout is None else time_fn() + timeout


def _remaining(deadline: Optional[float], time_fn: Callable[[], float]) -> Optional[float]:
    """Seconds left before ``deadline``; ``None`` when unbounded.

    Clamped at zero so an already-passed deadline hands the next stage a
    non-negative budget it will immediately time out on, not a negative one
    whose meaning varies by API.
    """
    if deadline is None:
        return None
    return max(0.0, deadline - time_fn())


def _coerce_status(raw: Any) -> RequestStatus:
    """Map a raw service status value to a ``RequestStatus``.

    Unknown values are treated as pending (``IN_PROGRESS``) instead of raising,
    so a new server state never crashes a polling client mid-loop.
    """
    try:
        return RequestStatus(raw)
    except ValueError:
        return RequestStatus.IN_PROGRESS


class APIFuture(Generic[_T]):
    """Future for an async runtime operation (training or sample).

    Call :meth:`result` to block until the operation reaches a terminal state.
    The same future type serves both training and sample operations; the caller
    supplies a ``poll_fn`` that fetches the latest response and a ``parse_fn``
    that turns a successful response into the public result object.

    Generic in the result type: ``parse_fn`` fixes it, so ``result()`` on an
    ``APIFuture[T]`` returns a ``T``.

    Args:
        operation_id: Server-assigned id of the operation (for messages).
        poll_fn: Zero-arg callable returning the latest operation response dict.
            Each call should hit ``Get*OperationResult`` for this operation.
        parse_fn: Callable mapping a completed response dict to the result
            object returned by :meth:`result`.
        status_key: Key under which the response carries its status string.
        operation_type: Label used in error messages (e.g. ``"training"``).
        poll_interval: Seconds between poll attempts.
        time_fn: Monotonic clock, injectable for tests. Defaults to
            :func:`time.monotonic`.
        sleep_fn: Sleep function used by :meth:`result`, injectable for tests.
            Defaults to :func:`time.sleep`.
        async_sleep_fn: Awaitable sleep used by :meth:`result_async`, injectable
            for tests. Defaults to :func:`asyncio.sleep`.
        max_transient_poll_errors: Consecutive transient poll errors (throttling
            / 5xx on Get*OperationResult) tolerated before giving up. A
            successful poll resets the counter.
        rand_fn: Jitter draw for the backoff between failed polls, injectable for
            tests. Defaults to :func:`random.uniform`.
    """

    def __init__(
        self,
        operation_id: str,
        poll_fn: Callable[[], Dict[str, Any]],
        parse_fn: Callable[[Dict[str, Any]], _T],
        *,
        status_key: str = "status",
        operation_type: str = "operation",
        poll_interval: float = DEFAULT_POLL_INTERVAL,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], None] = time.sleep,
        async_sleep_fn: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_transient_poll_errors: int = DEFAULT_MAX_TRANSIENT_POLL_ERRORS,
        rand_fn: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self._operation_id = operation_id
        self._poll_fn = poll_fn
        self._parse_fn = parse_fn
        self._status_key = status_key
        self._operation_type = operation_type
        self._poll_interval = poll_interval
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._async_sleep_fn = async_sleep_fn
        self._max_transient_poll_errors = max_transient_poll_errors
        self._rand_fn = rand_fn
        self._result: Any = _UNSET
        #: The terminal failure, once one is known. Memoized beside the
        #: result so a resolved future reports itself resolved either way.
        self._error: Any = _UNSET
        self._lock = threading.Lock()

    @property
    def operation_id(self) -> str:
        """Return the server-assigned operation id."""
        return self._operation_id

    # Shared by both waits --------------------------------------------------
    # result() and result_async() differ only in how they wait: one sleeps the
    # calling thread, the other suspends a coroutine. Everything that decides
    # what the wait *means* lives here, so the two cannot drift apart.

    # Concurrent waiters on one future are allowed to poll independently, but
    # only one may resolve it: `parse_fn` carries the operation's completion
    # side effects, so it must run once. The lock therefore covers classify +
    # memoize, and never a poll: holding it across the wire would make a
    # bounded result(timeout=...) block on another waiter's unbounded poll, and
    # in the async path would occupy the worker threads result_async needs to
    # poll at all.

    def _transient_backoff(self, consecutive_transient: int) -> float:
        """Seconds to wait after ``consecutive_transient`` failed status reads.

        Not ``poll_interval``: a poll that answered keeps the normal cadence,
        while one that failed backs off, since under a spike that cadence is part
        of what is being shed.
        """
        return _backoff_delay(consecutive_transient - 1, rand_fn=self._rand_fn)

    def _classify_poll(self, resp: Dict[str, Any]) -> Tuple[bool, Any]:
        """Read one poll response.

        Returns ``(True, result)`` once the operation has completed and
        ``(False, None)`` while it is still running.

        Raises:
            OperationError: The response reports a terminal failure state.
        """
        status = _coerce_status(resp.get(self._status_key))
        if status == _SUCCESS_STATUS:
            return True, self._parse_fn(resp)
        if status in _TERMINAL_FAILURE_STATUSES:
            raise OperationError(
                self._operation_type,
                resp.get(
                    "errorDetails",
                    f"Operation ended with status: {status.value}",
                ),
                operation_id=self._operation_id,
            )
        return False, None

    def _classify_and_memoize(self, resp: Dict[str, Any]) -> Tuple[bool, Any]:
        """Read one poll response and memoize the outcome, at most once.

        Returns what :meth:`_classify_poll` returns. When two waiters poll
        concurrently and both see a completed response, the first through here
        parses and memoizes; the second is served that same result object and
        never calls ``parse_fn`` again.

        A terminal failure is memoized the same way a success is. Both are the
        operation's final answer, so a later wait replays it rather than asking
        the service again, and the future reports itself resolved either way.
        That covers the failures this SDK judges as well as the ones the service
        reports: a result that does not match the request it was awaited for is
        as final as an operation the service calls failed, and ``parse_fn`` is
        where it is caught.
        """
        with self._lock:
            if self._error is not _UNSET:
                raise self._error
            if self._result is not _UNSET:
                return True, self._result
            try:
                completed, value = self._classify_poll(resp)
            except TrainingSessionError as error:
                self._error = error
                raise
            if completed:
                self._result = value
            return completed, value

    def _timeout_error(self, timeout: Optional[float]) -> TimeoutError:
        """The error both waits raise when ``timeout`` runs out."""
        return TimeoutError(
            f"{self._operation_type} operation {self._operation_id} "
            f"did not complete within {_format_timeout(timeout)}"
        )

    def result(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Block until the operation completes and return its parsed result.

        Args:
            timeout: Max seconds to wait before raising ``TimeoutError``.
                ``None`` waits indefinitely.

        Returns:
            The object produced by ``parse_fn`` from the completed response.

        Raises:
            OperationError: The operation reached a terminal failure state.
            TimeoutError: The operation did not complete within ``timeout``.
            Exception: The error botocore raised for a failed poll, either a
                transient one that recurred past ``max_transient_poll_errors``
                in a row, or one that was never worth retrying.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result

        deadline = _deadline_from(timeout, self._time_fn)
        consecutive_transient = 0
        while deadline is None or self._time_fn() < deadline:
            try:
                resp = self._poll_fn()
            except Exception as error:
                if not is_retryable(error):
                    raise
                # A transient failure reading the status does not mean the
                # operation failed; it is often still running. Keep polling until
                # the deadline, but cap consecutive failures so an endpoint that
                # is genuinely down surfaces instead of spinning.
                consecutive_transient += 1
                if consecutive_transient > self._max_transient_poll_errors:
                    raise
                self._sleep_fn(self._transient_backoff(consecutive_transient))
                continue

            consecutive_transient = 0
            completed, value = self._classify_and_memoize(resp)
            if completed:
                return value
            self._sleep_fn(self._poll_interval)

        raise self._timeout_error(timeout)

    async def result_async(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Await the operation result without occupying a thread while it runs.

        Equivalent to ``await future``, and the way to await with a ``timeout``
        of your own: ``await future`` cannot take arguments, so it uses the
        default.

        Args:
            timeout: Max seconds to wait before raising ``TimeoutError``.
                ``None`` waits indefinitely.

        Raises:
            OperationError: The operation reached a terminal failure state.
            TimeoutError: The operation did not complete within ``timeout``.
            Exception: The error botocore raised for a failed poll, either a
                transient one that recurred past ``max_transient_poll_errors``
                in a row, or one that was never worth retrying.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result

        deadline = _deadline_from(timeout, self._time_fn)
        consecutive_transient = 0
        while deadline is None or self._time_fn() < deadline:
            try:
                # The status request itself blocks, so give it a worker thread
                # for that one request only, not for the whole operation.
                resp = await asyncio.to_thread(self._poll_fn)
            except Exception as error:
                if not is_retryable(error):
                    raise
                consecutive_transient += 1
                if consecutive_transient > self._max_transient_poll_errors:
                    raise
                await self._async_sleep_fn(self._transient_backoff(consecutive_transient))
                continue

            consecutive_transient = 0
            completed, value = self._classify_and_memoize(resp)
            if completed:
                return value
            await self._async_sleep_fn(self._poll_interval)

        raise self._timeout_error(timeout)

    def __await__(self) -> Generator[Any, None, _T]:
        """Make APIFuture awaitable so ``await future`` works in async code.

        Uses the default ``timeout``; call :meth:`result_async` to pass your
        own. Every subclass routes ``await`` through its own
        :meth:`result_async`, so overriding that one method is enough.
        """
        return self.result_async().__await__()

    # The rest of the standard future surface -------------------------------
    # Inherited by both subclasses: each keeps the same ``_result`` slot, so
    # asking whether a future has resolved means the same thing for all three.

    def done(self) -> bool:
        """Whether this future has resolved, without waiting for it.

        ``True`` once the operation's outcome is known, whether that is a result
        or a failure, so a loop that drains futures by checking this terminates
        on a failed operation as well as a successful one. A future whose
        operation has finished on the service but has not been waited on yet
        still reports ``False``: nothing has polled for it.

        A failed *request* is not an outcome and is not recorded here. A call
        that could not be made or answered, credentials that expired mid-run
        being the clearest case, says nothing about the operation, which may
        still be running and may still succeed, so a later wait asks again
        rather than replaying an answer that was never about the operation.
        """
        return self._result is not _UNSET or self._error is not _UNSET

    def exception(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> Optional[BaseException]:
        """Wait for the operation and return the error it failed with.

        Returns ``None`` if the operation succeeded, so a caller can inspect
        several futures without wrapping each in ``try``.

        A failed operation hands back the same error object on every wait, the
        way any settled future does, so identity comparisons hold. Re-raising
        one object repeatedly does lengthen its traceback each time, which is
        worth knowing if a retry loop waits on the same future many times over;
        read ``__cause__`` and the message rather than the traceback depth.

        Args:
            timeout: Max seconds to wait before raising ``TimeoutError``.
                ``None`` waits indefinitely.

        Raises:
            TimeoutError: The operation did not complete within ``timeout``.
        """
        try:
            self.result(timeout=timeout)
        except TimeoutError:
            raise
        except Exception as error:  # noqa: BLE001 - reported, not handled
            return error
        return None

    def cancel(self) -> bool:
        """Always ``False``: a submitted operation cannot be called back.

        Present because the rest of this surface follows the standard future
        protocol, and a caller looping over futures should not have to special
        case this one. The service has the operation by the time a caller holds
        the future, so there is nothing to take back; stop the session to end
        the work it is doing.
        """
        return False

    def cancelled(self) -> bool:
        """Always ``False``: see :meth:`cancel`."""
        return False


class PendingAPIFuture(APIFuture[_T]):
    """An :class:`APIFuture` whose submit round trip is still in flight.

    Returned the moment an operation is queued on the client's background
    submitter, before ``Execute*Operation`` has assigned an operation id.
    Resolving it waits for the submit first (a submit error surfaces here,
    not at the call site), then delegates to the :class:`APIFuture` the
    submit produced. ``await`` works unchanged and holds no thread through
    either stage.

    Args:
        submit_future: ``concurrent.futures.Future`` resolving to the real
            :class:`APIFuture` once the submit round trip completes (or
            raising the submit error).
        operation_type: Label used in error messages (e.g. ``"training"``).
        keepalive: The submitting client's ``_SessionHandle``. Held only to keep
            it reachable: the automatic session stop fires when the last
            reference to the handle goes away, so holding one here guarantees a
            session is never stopped while an operation against it is still
            outstanding. Dropping the client but keeping its futures stays safe.
        time_fn: Monotonic clock used to split ``timeout`` across the two
            stages, injectable for tests. Defaults to :func:`time.monotonic`.
    """

    def __init__(
        self,
        submit_future: Any,
        *,
        operation_type: str = "operation",
        keepalive: Any = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        # No super().__init__: poll_fn/parse_fn/operation_id all arrive with
        # the resolved inner future; this subclass only defers to it.
        self._submit_future = submit_future
        self._operation_type = operation_type
        self._result: Any = _UNSET
        #: The terminal failure, once one is known. Memoized beside the
        #: result so a resolved future reports itself resolved either way.
        self._error: Any = _UNSET
        self._lock = threading.Lock()
        self._time_fn = time_fn
        # Reachability only; never called. The submit closure and the inner
        # future's poll_fn happen to capture their client today, which has the
        # same effect by accident; this makes the guarantee explicit so a later
        # refactor of those closures cannot silently remove it.
        self._keepalive = keepalive

    @property
    def operation_id(self) -> str:
        """Wait for the submit to complete, then return the operation id."""
        return self._resolve_submit().operation_id

    def _operation_id_if_known(self) -> str:
        """The operation id if the submit already landed, else a placeholder.

        For messages built while a wait is timing out: reading
        :attr:`operation_id` there would resume the unbounded wait the caller
        asked to bound.
        """
        if self._submit_future.done():
            try:
                return self._submit_future.result().operation_id
            except Exception:  # noqa: BLE001 - a failed submit has no id to show
                return "<not submitted>"
        return "<not submitted>"

    def _resolve_submit(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> APIFuture[_T]:
        """Block until the background submit yields the real future.

        Normalizes the wait timeout to a built-in :class:`TimeoutError`:
        ``concurrent.futures.TimeoutError`` is only an alias for it from Python
        3.11 on, and callers should not have to know which stage timed out to
        know what to catch.
        """
        try:
            return self._submit_future.result(timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(
                f"{self._operation_type} operation was not submitted within "
                f"{_format_timeout(timeout)}"
            ) from None

    def _memoize(self, value: Any) -> Any:
        """Record ``value`` as this future's result unless one is already set.

        The inner :class:`APIFuture` already parses exactly once, so concurrent
        waiters here arrive with the same object; this only keeps them from
        disagreeing about which one is *the* result.
        """
        with self._lock:
            if self._result is _UNSET:
                self._result = value
            return self._result

    def _memoize_error(self, error: BaseException) -> None:
        """Record a terminal failure so this future reports itself resolved.

        The inner future memoizes its own, but a caller holds *this* one, so it
        has to carry the same answer.
        """
        with self._lock:
            if self._error is _UNSET:
                self._error = error

    def result(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Block until submit + operation complete; return the parsed result.

        ``timeout`` bounds the whole wait: one deadline is taken here and the
        poll loop gets only what the submit wait left over. ``None`` waits
        indefinitely.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result
        deadline = _deadline_from(timeout, self._time_fn)
        inner = self._resolve_submit(timeout=_remaining(deadline, self._time_fn))
        try:
            value = inner.result(timeout=_remaining(deadline, self._time_fn))
        except TimeoutError:
            # The poll loop only saw what the submit wait left over; report the
            # budget the caller actually passed.
            raise self._whole_wait_timeout(inner, timeout) from None
        except TrainingSessionError as error:
            self._memoize_error(error)
            raise
        return self._memoize(value)

    async def result_async(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Await submit + operation without occupying a thread while they run.

        Splits ``timeout`` across the two stages exactly as :meth:`result` does.
        Neither stage blocks the event loop: the submit is awaited on it, and
        the poll loop it hands back is a coroutine.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result
        deadline = _deadline_from(timeout, self._time_fn)
        inner = await self._resolve_submit_async(timeout=_remaining(deadline, self._time_fn))
        try:
            value = await inner.result_async(timeout=_remaining(deadline, self._time_fn))
        except TimeoutError:
            raise self._whole_wait_timeout(inner, timeout) from None
        except TrainingSessionError as error:
            self._memoize_error(error)
            raise
        return self._memoize(value)

    async def _resolve_submit_async(
        self, timeout: Optional[float] = DEFAULT_TIMEOUT
    ) -> APIFuture[_T]:
        """Await the background submit and return the real future.

        Normalizes the wait timeout to a built-in :class:`TimeoutError` for the
        same reason :meth:`_resolve_submit` does: ``asyncio.TimeoutError`` is
        only an alias for it from Python 3.11 on.
        """
        try:
            return await asyncio.wait_for(
                asyncio.shield(asyncio.wrap_future(self._submit_future)), timeout
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"{self._operation_type} operation was not submitted within "
                f"{_format_timeout(timeout)}"
            ) from None

    def _whole_wait_timeout(self, inner: APIFuture[_T], timeout: Optional[float]) -> TimeoutError:
        """Report the budget the caller passed, not the slice a stage got."""
        return TimeoutError(
            f"{self._operation_type} operation {inner.operation_id} "
            f"did not complete within {_format_timeout(timeout)}"
        )


class CombinedFuture(APIFuture[_T]):
    """A future over several sub-futures whose results combine into one.

    Used when a single logical operation is split into multiple runtime ops
    (e.g. a large forward-backward batch chunked to fit the request-size limit).
    :meth:`result` blocks on every sub-future in submission order and folds
    their results through ``combine_fn``, so callers see the same single result
    object they would get from an unchunked call.

    An :class:`APIFuture` subclass so that chunking stays invisible in the public
    signature: a chunked call is annotated ``APIFuture[T]`` exactly as an
    unchunked one is. :attr:`operation_id` and both waits are overridden;
    ``__await__`` is inherited and routes through this :meth:`result_async`.

    The sub-futures were submitted in sequence order, so awaiting them in that
    same order matches how the service processes them; the first terminal
    failure among them propagates from :meth:`result`.

    Args:
        futures: Sub-futures to await, in submission (sequence) order.
        combine_fn: Callable folding the ordered list of sub-results into the
            single result object returned by :meth:`result`.
        time_fn: Monotonic clock used to share ``timeout`` across the
            sub-futures, injectable for tests. Defaults to
            :func:`time.monotonic`.
    """

    def __init__(
        self,
        futures: List[APIFuture[_T]],
        combine_fn: Callable[[List[_T]], _T],
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        # No super().__init__: there is no single operation to poll here, only
        # sub-futures to fold. Same reason PendingAPIFuture skips it.
        if not futures:
            raise ValueError("CombinedFuture requires at least one sub-future")
        self._futures = futures
        self._combine_fn = combine_fn
        self._result: Any = _UNSET
        #: The terminal failure, once one is known. Memoized beside the
        #: result so a resolved future reports itself resolved either way.
        self._error: Any = _UNSET
        self._lock = threading.Lock()
        self._time_fn = time_fn

    @property
    def operation_id(self) -> str:
        """Return the first sub-operation's id (representative, for messages)."""
        return self._futures[0].operation_id

    def _operation_id_if_known(self) -> str:
        """The first sub-operation's id, or a placeholder if it is not resolved.

        Reading :attr:`operation_id` waits for the submit that produced it, with
        no bound. That is fine when a caller asks for it, and wrong inside a
        timeout message: the wait the caller asked to bound would resume in the
        error path.
        """
        try:
            return self._futures[0]._operation_id_if_known()
        except AttributeError:
            return self._futures[0].operation_id

    def result(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Block until every sub-future completes and return the combined result.

        ``timeout`` bounds the whole wait, not each sub-future: one deadline is
        taken here and every sub-future gets only the time still left, so a
        chunked call is bounded the same way an unchunked one is. ``None`` waits
        indefinitely.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result
        deadline = _deadline_from(timeout, self._time_fn)
        try:
            sub_results = [
                f.result(timeout=_remaining(deadline, self._time_fn)) for f in self._futures
            ]
        except TimeoutError:
            raise self._chunked_timeout(timeout) from None
        except TrainingSessionError as error:
            self._memoize_error(error)
            raise
        return self._combine_and_memoize(sub_results)

    async def result_async(self, timeout: Optional[float] = DEFAULT_TIMEOUT) -> _T:
        """Await every sub-future and return the combined result.

        Awaits them in submission order, the same order :meth:`result` waits in,
        so a chunked call reports the same first failure either way. ``timeout``
        bounds the whole wait, not each sub-future.
        """
        if self._error is not _UNSET:
            raise self._error
        if self._result is not _UNSET:
            return self._result
        deadline = _deadline_from(timeout, self._time_fn)
        sub_results = []
        try:
            for f in self._futures:
                sub_results.append(
                    await f.result_async(timeout=_remaining(deadline, self._time_fn))
                )
        except TimeoutError:
            raise self._chunked_timeout(timeout) from None
        except TrainingSessionError as error:
            self._memoize_error(error)
            raise
        return self._combine_and_memoize(sub_results)

    def _memoize_error(self, error: BaseException) -> None:
        """Record the first sub-future failure; see the base class."""
        with self._lock:
            if self._error is _UNSET:
                self._error = error

    def _combine_and_memoize(self, sub_results: List[Any]) -> Any:
        """Fold the sub-results into this future's result, at most once.

        ``combine_fn`` folds already-parsed values and touches no wire, so it is
        cheap to hold the lock across and gets the same once-only guarantee
        ``parse_fn`` has.
        """
        with self._lock:
            if self._result is _UNSET:
                self._result = self._combine_fn(sub_results)
            return self._result

    def _chunked_timeout(self, timeout: Optional[float]) -> TimeoutError:
        """Each sub-future saw only its slice of the budget; report the whole."""
        return TimeoutError(
            f"chunked operation {self._operation_id_if_known()} did not complete "
            f"within {_format_timeout(timeout)} "
            f"({len(self._futures)} sub-operations)"
        )
