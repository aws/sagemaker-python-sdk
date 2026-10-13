# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session.futures.

All timing is injected — no test performs a real sleep or wall-clock wait.
"""

from __future__ import annotations

import concurrent.futures
import threading

import pytest
from botocore.exceptions import ClientError

from sagemaker.train.training_session.exceptions import OperationError, ResultMismatchError
from sagemaker.train.training_session.futures import (
    APIFuture,
    CombinedFuture,
    DEFAULT_BACKOFF_BASE_SECONDS,
    DEFAULT_BACKOFF_MAX_SECONDS,
    DEFAULT_MAX_TRANSIENT_POLL_ERRORS,
    PendingAPIFuture,
    RequestStatus,
    _backoff_delay,
    _coerce_status,
    _PENDING_STATUSES,
    _remaining,
    _SUCCESS_STATUS,
    _TERMINAL_FAILURE_STATUSES,
    call_with_transient_retry,
)


def _max_jitter(low: float, high: float) -> float:
    """A jitter draw pinned to the top of the range, so delays are assertable.

    Real draws are random by design; pinning them makes the *ceiling* schedule
    observable, which is the part the policy fixes.
    """
    return high


class FakeClock:
    """Deterministic monotonic clock; ``sleep`` advances it, never waits."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    async def asleep(self, seconds: float) -> None:
        """The awaitable seam ``result_async`` sleeps through."""
        self.sleep(seconds)


def _make_future(
    clock: FakeClock,
    responses,
    *,
    parse_fn=None,
    status_key="status",
    poll_interval=2,
):
    """Build an APIFuture whose poll_fn yields ``responses`` in order."""
    it = iter(responses)
    return APIFuture(
        operation_id="op-1",
        poll_fn=lambda: next(it),
        parse_fn=parse_fn or (lambda r: r.get("operationResult", "done")),
        status_key=status_key,
        operation_type="training",
        poll_interval=poll_interval,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
    )


def test_the_module_names_only_what_a_caller_can_act_on():
    """The generated reference documents this module by name, so pin the surface.

    Without a declared surface the two future subclasses are published as though
    a caller were meant to build one, alongside the polling defaults and the
    retry helper the clients wire up for themselves.
    """
    import sagemaker.train.training_session.futures as module

    assert module.__all__ == ["APIFuture", "RequestStatus"]
    for internal in ("PendingAPIFuture", "CombinedFuture", "call_with_transient_retry"):
        assert hasattr(module, internal), f"{internal} moved; this test needs updating"
        assert internal not in module.__all__


# --- status model -----------------------------------------------------------


def test_status_partitions_are_complete_and_disjoint():
    all_states = set(RequestStatus)
    assert _SUCCESS_STATUS == RequestStatus.COMPLETED
    assert {_SUCCESS_STATUS} | _TERMINAL_FAILURE_STATUSES | _PENDING_STATUSES == all_states
    assert not (_TERMINAL_FAILURE_STATUSES & _PENDING_STATUSES)
    assert _SUCCESS_STATUS not in _TERMINAL_FAILURE_STATUSES
    assert _SUCCESS_STATUS not in _PENDING_STATUSES


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Completed", RequestStatus.COMPLETED),
        ("Failed", RequestStatus.FAILED),
        ("InProgress", RequestStatus.IN_PROGRESS),
        ("Pending", RequestStatus.PENDING),
        ("Skipped", RequestStatus.SKIPPED),
    ],
)
def test_coerce_status_known(raw, expected):
    assert _coerce_status(raw) == expected


@pytest.mark.parametrize("raw", ["SomethingNew", None, "", "completed"])
def test_coerce_status_unknown_is_pending(raw):
    # Unknown/None/case-mismatch must not raise — treat as still-running.
    assert _coerce_status(raw) == RequestStatus.IN_PROGRESS


# --- result() happy path ----------------------------------------------------


def test_result_returns_on_immediate_completion():
    clock = FakeClock()
    fut = _make_future(clock, [{"status": "Completed", "operationResult": {"loss": 1.0}}])
    assert fut.result() == {"loss": 1.0}
    assert clock.sleeps == []  # completed first poll, never slept


def test_result_polls_through_pending_then_completes():
    clock = FakeClock()
    fut = _make_future(
        clock,
        [
            {"status": "Created"},
            {"status": "Dispatched"},
            {"status": "InProgress"},
            {"status": "Completed", "operationResult": "ok"},
        ],
        poll_interval=5,
    )
    assert fut.result() == "ok"
    assert clock.sleeps == [5, 5, 5]  # slept once per pending poll


def test_result_is_cached_and_does_not_poll_again():
    clock = FakeClock()
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        return {"status": "Completed", "operationResult": "v"}

    fut = APIFuture(
        "op",
        poll,
        lambda r: r["operationResult"],
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    assert fut.result() == "v"
    assert fut.result() == "v"
    assert calls["n"] == 1  # second result() served from cache


def test_parse_fn_side_effects_fire_once_on_success():
    # APIFuture has no on_success callback by design: completion side
    # effects (e.g. caching a checkpoint) are composed into parse_fn by the
    # submitter. parse_fn must run exactly once, and only on success.
    clock = FakeClock()
    calls = {"n": 0}

    def parse_fn(resp):
        calls["n"] += 1
        return resp["operationResult"]

    fut = _make_future(
        clock,
        # Two completed polls available, but the second is never reached:
        # result() caches after the first success.
        [
            {"status": "Completed", "operationResult": "v"},
            {"status": "Completed", "operationResult": "v"},
        ],
        parse_fn=parse_fn,
    )
    assert fut.result() == "v"
    assert fut.result() == "v"  # cached; parse_fn not invoked again
    assert calls["n"] == 1


def test_parse_fn_not_called_on_failure():
    clock = FakeClock()
    calls = {"n": 0}

    def parse_fn(resp):
        calls["n"] += 1
        return resp

    fut = _make_future(
        clock,
        [{"status": "Failed", "errorDetails": "boom"}],
        parse_fn=parse_fn,
    )
    with pytest.raises(OperationError):
        fut.result()
    assert calls["n"] == 0


def test_parse_fn_receives_full_response():
    clock = FakeClock()
    seen = {}
    fut = _make_future(
        clock,
        [{"status": "Completed", "extra": 7}],
        parse_fn=lambda r: seen.setdefault("resp", r),
    )
    fut.result()
    assert seen["resp"]["extra"] == 7


def test_custom_status_key():
    clock = FakeClock()
    fut = _make_future(
        clock,
        [{"requestStatus": "Completed", "operationResult": "x"}],
        status_key="requestStatus",
    )
    assert fut.result() == "x"


# --- failure + timeout paths ------------------------------------------------


@pytest.mark.parametrize("bad", ["Failed", "Skipped"])
def test_result_raises_on_terminal_failure(bad):
    clock = FakeClock()
    fut = _make_future(clock, [{"status": bad, "errorDetails": "boom"}])
    with pytest.raises(OperationError) as ei:
        fut.result()
    assert "boom" in str(ei.value)
    assert ei.value.operation_type == "training"
    # The failed operation must be identifiable from the error alone.
    assert ei.value.operation_id == "op-1"


def test_failure_without_error_details_uses_status_in_message():
    clock = FakeClock()
    fut = _make_future(clock, [{"status": "Failed"}])
    with pytest.raises(OperationError) as ei:
        fut.result()
    assert "Failed" in str(ei.value)


def test_result_times_out_without_real_wait():
    clock = FakeClock()
    # Always pending → loop must exit on the deadline, not hang.
    fut = APIFuture(
        "op-timeout",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=3,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    with pytest.raises(TimeoutError) as ei:
        fut.result(timeout=10)
    assert "op-timeout" in str(ei.value)
    # Advanced only via injected sleeps; bounded by the deadline.
    assert clock.now >= 10
    assert all(s == 3 for s in clock.sleeps)


# --- transient poll-error resilience ----------------------------------------


def _transient(status_code=503):
    """A botocore error the SDK classifies as retryable: what the wire raises."""
    return ClientError(
        {
            "Error": {"Code": "InternalServerException", "Message": "blip"},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        },
        "GetTrainingOperationResult",
    )


def _make_future_seq(
    clock,
    items,
    *,
    max_transient_poll_errors=5,
    poll_interval=2,
    rand_fn=_max_jitter,
):
    """APIFuture whose poll_fn yields dicts or *raises* a retryable error."""
    it = iter(items)

    def poll():
        nxt = next(it)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    return APIFuture(
        operation_id="op-transient",
        poll_fn=poll,
        parse_fn=lambda r: r.get("operationResult", "done"),
        operation_type="training",
        poll_interval=poll_interval,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
        max_transient_poll_errors=max_transient_poll_errors,
        rand_fn=rand_fn,
    )


def test_transient_poll_error_is_retried_then_completes():
    # The op completes server-side; a single transient blip on the status read
    # must NOT abort: a transient on the status read is not an op failure.
    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [
            {"status": "InProgress"},
            _transient(),
            {"status": "Completed", "operationResult": "ok"},
        ],
    )
    assert fut.result(timeout=100) == "ok"


def test_transient_poll_errors_reset_counter_on_success():
    clock = FakeClock()
    seq = []
    for _ in range(4):
        seq.append(_transient())
        seq.append({"status": "InProgress"})
    seq.append({"status": "Completed", "operationResult": "done"})
    fut = _make_future_seq(clock, seq, max_transient_poll_errors=2, poll_interval=1)
    assert fut.result(timeout=1000) == "done"


def test_too_many_consecutive_transient_poll_errors_raises():
    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [_transient() for _ in range(10)],
        max_transient_poll_errors=3,
        poll_interval=1,
    )
    with pytest.raises(ClientError):
        fut.result(timeout=1000)


# --- backoff policy ---------------------------------------------------------


def test_backoff_ceiling_doubles_then_clamps():
    ceilings = [_backoff_delay(n, rand_fn=_max_jitter) for n in range(8)]
    assert ceilings == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    assert ceilings[0] == DEFAULT_BACKOFF_BASE_SECONDS
    assert max(ceilings) == DEFAULT_BACKOFF_MAX_SECONDS


def test_the_default_budget_spans_about_five_minutes():
    """Base, cap and attempt count jointly decide the budget; pin the product.

    Changing any one of them moves it without touching the number that names it.
    """
    average_wait = sum(
        _backoff_delay(n, rand_fn=lambda low, high: high / 2)
        for n in range(DEFAULT_MAX_TRANSIENT_POLL_ERRORS - 1)
    )
    assert 4 * 60 <= average_wait <= 6 * 60, f"budget is now {round(average_wait)}s"


def test_backoff_draws_from_the_whole_range_below_the_ceiling():
    """Full jitter, not equal jitter: the range starts at zero.

    A floor above zero is a wait every caller shares -- the correlation the
    jitter exists to remove.
    """
    seen = []
    _backoff_delay(2, rand_fn=lambda low, high: seen.append((low, high)) or 0.0)
    assert seen == [(0.0, 2.0)]


def test_a_failed_poll_backs_off_while_a_healthy_one_keeps_its_cadence():
    """The two waits are different things and must not share a number.

    A poll that answered keeps the configured cadence; a throttled one cannot,
    since that cadence is what was refused.
    """
    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [
            {"status": "InProgress"},  # healthy → poll_interval
            _transient(),  # 1st consecutive failure → 0.5
            _transient(),  # 2nd → 1.0
            _transient(),  # 3rd → 2.0
            {"status": "Completed", "operationResult": "ok"},
        ],
        poll_interval=7,
    )
    assert fut.result(timeout=1000) == "ok"
    assert clock.sleeps == [7, 0.5, 1.0, 2.0]


def test_backoff_restarts_after_a_poll_gets_through():
    """A recovered endpoint must not stay penalised by the previous burst.

    The consecutive counter already resets on success; the delay schedule has to
    reset with it, not stay at the ceiling it climbed to.
    """
    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [
            _transient(),  # → 0.5
            _transient(),  # → 1.0
            {"status": "InProgress"},  # → poll_interval, counter resets
            _transient(),  # → 0.5 again, not 2.0
            {"status": "Completed", "operationResult": "ok"},
        ],
        poll_interval=3,
    )
    assert fut.result(timeout=1000) == "ok"
    assert clock.sleeps == [0.5, 1.0, 3, 0.5]


# --- call_with_transient_retry (submit-path resilience) ---------------------


def test_call_with_transient_retry_succeeds_after_blip():
    # The submit call itself can blip, not just the status read that follows it.
    clock = FakeClock()
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] == 1:
            raise _transient()
        return {"operationId": "op-9"}

    out = call_with_transient_retry(fn, sleep_fn=clock.sleep, rand_fn=_max_jitter)
    assert out == {"operationId": "op-9"}
    assert calls["n"] == 2
    assert clock.sleeps == [DEFAULT_BACKOFF_BASE_SECONDS]


def test_call_with_transient_retry_gives_up_after_max_attempts():
    clock = FakeClock()

    def fn():
        raise _transient()

    with pytest.raises(ClientError):
        call_with_transient_retry(fn, max_attempts=3, sleep_fn=clock.sleep, rand_fn=_max_jitter)
    # 3 attempts → 2 sleeps between them, and the second waits longer.
    assert clock.sleeps == [0.5, 1.0]


def test_call_with_transient_retry_propagates_non_transient():
    def fn():
        raise OperationError("request", "bad input")

    with pytest.raises(OperationError):
        call_with_transient_retry(fn, sleep_fn=lambda _s: None)


def test_call_with_transient_retry_backs_off_instead_of_hammering():
    """The budget must not be spent at one fixed rate.

    A flat interval reoffers the rate the service just refused, so the delays
    have to grow and then hold at the cap.
    """
    clock = FakeClock()

    def fn():
        raise _transient()

    with pytest.raises(ClientError):
        call_with_transient_retry(fn, max_attempts=8, sleep_fn=clock.sleep, rand_fn=_max_jitter)
    assert clock.sleeps == [0.5, 1.0, 2.0, 4.0, 4.0, 4.0, 4.0]
    assert len(set(clock.sleeps)) > 1, "a flat cadence is the bug being fixed"


def test_call_with_transient_retry_jitters_every_wait():
    """Two callers hitting the same throttle must not wait the same amount.

    Drawn per attempt, not once: one draw reused would still put every caller on
    the same relative schedule.
    """
    draws = []

    def fn():
        raise _transient()

    def recording_rand(low, high):
        draws.append((low, high))
        return low  # exercise the bottom of the range; timing is irrelevant here

    with pytest.raises(ClientError):
        call_with_transient_retry(
            fn, max_attempts=4, sleep_fn=lambda _s: None, rand_fn=recording_rand
        )
    # Every wait is a fresh draw from [0, ceiling), never the ceiling itself.
    assert draws == [(0.0, 0.5), (0.0, 1.0), (0.0, 2.0)]


# --- async surface ----------------------------------------------------------
# result_async() polls as a coroutine: it suspends between polls instead of
# handing the whole blocking wait to a thread. These pin that it reaches every
# outcome result() does, and that awaiting many at once is not serialized by a
# thread pool.


def test_await_and_result_async_inside_event_loop():
    import asyncio

    clock = FakeClock()

    async def go():
        awaited = await _make_future(clock, [{"status": "Completed", "operationResult": "a"}])
        called = await _make_future(
            clock, [{"status": "Completed", "operationResult": "b"}]
        ).result_async()
        return awaited, called

    assert asyncio.run(go()) == ("a", "b")


def test_result_async_polls_through_pending_then_completes():
    """The async wait sleeps through pending polls, on the injected clock."""
    import asyncio

    clock = FakeClock()
    fut = _make_future(
        clock,
        [
            {"status": "Pending"},
            {"status": "InProgress"},
            {"status": "Completed", "operationResult": "ok"},
        ],
        poll_interval=5,
    )
    assert asyncio.run(fut.result_async()) == "ok"
    assert clock.sleeps == [5, 5]


def test_result_async_is_cached_and_does_not_poll_again():
    import asyncio

    clock = FakeClock()
    calls = {"n": 0}

    def poll():
        calls["n"] += 1
        return {"status": "Completed", "operationResult": "v"}

    fut = APIFuture(
        "op",
        poll,
        lambda r: r["operationResult"],
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
    )
    assert asyncio.run(fut.result_async()) == "v"
    assert asyncio.run(fut.result_async()) == "v"
    assert calls["n"] == 1


def test_result_async_shares_the_cache_with_result():
    """One future resolved either way must not poll twice or re-parse."""
    import asyncio

    clock = FakeClock()
    parses = {"n": 0}
    fut = _make_future(
        clock,
        [{"status": "Completed", "operationResult": "v"}],
        parse_fn=lambda r: parses.update(n=parses["n"] + 1) or r["operationResult"],
    )
    assert fut.result() == "v"
    # A second poll would raise StopIteration; a second parse would count twice.
    assert asyncio.run(fut.result_async()) == "v"
    assert parses["n"] == 1


@pytest.mark.parametrize("bad", ["Failed", "Skipped"])
def test_result_async_raises_on_terminal_failure(bad):
    import asyncio

    clock = FakeClock()
    fut = _make_future(clock, [{"status": bad, "errorDetails": "boom"}])
    with pytest.raises(OperationError) as ei:
        asyncio.run(fut.result_async())
    assert "boom" in str(ei.value)


def test_result_async_times_out_without_real_wait():
    import asyncio

    clock = FakeClock()
    fut = APIFuture(
        "op-timeout",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=3,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
    )
    with pytest.raises(TimeoutError) as ei:
        asyncio.run(fut.result_async(timeout=10))
    assert "op-timeout" in str(ei.value)
    assert clock.now >= 10


def test_result_async_retries_transient_then_completes():
    import asyncio

    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [
            {"status": "InProgress"},
            _transient(),
            {"status": "Completed", "operationResult": "ok"},
        ],
    )
    assert asyncio.run(fut.result_async(timeout=100)) == "ok"


def test_result_async_gives_up_after_too_many_consecutive_transients():
    import asyncio

    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [_transient() for _ in range(10)],
        max_transient_poll_errors=3,
        poll_interval=1,
    )
    with pytest.raises(ClientError):
        asyncio.run(fut.result_async(timeout=1000))


def test_result_async_backs_off_on_the_same_schedule_as_result():
    """The two waits must not drift: throttling is not gentler on a coroutine."""
    import asyncio

    clock = FakeClock()
    fut = _make_future_seq(
        clock,
        [
            {"status": "InProgress"},
            _transient(),
            _transient(),
            {"status": "Completed", "operationResult": "ok"},
        ],
        poll_interval=7,
    )
    assert asyncio.run(fut.result_async(timeout=1000)) == "ok"
    assert clock.sleeps == [7, 0.5, 1.0]


def test_await_does_not_hold_a_worker_thread_for_the_whole_operation():
    """The regression this async rewrite exists for.

    The wait used to be a blocking call handed to the event loop's default
    executor, so each pending ``await`` occupied one of its threads until the
    operation finished. Awaiting more operations than that pool has threads
    admitted them in waves, and starved anything else that needed a thread.

    Here every future needs two polls to finish, so with a genuine coroutine
    wait all of them reach their second poll. If the wait were thread-bound,
    at most ``max_workers`` could be in flight at once.
    """
    import asyncio

    n = 64  # comfortably above min(32, cpu_count + 4)
    inflight = 0
    peak = 0

    def make():
        state = {"calls": 0}

        def poll():
            nonlocal inflight, peak
            state["calls"] += 1
            if state["calls"] == 1:
                inflight += 1
                peak = max(peak, inflight)
                return {"status": "InProgress"}
            inflight -= 1
            return {"status": "Completed", "operationResult": "ok"}

        # Real asyncio.sleep(0) so every wait suspends and resumes for real,
        # without spending wall-clock time.
        return APIFuture(
            "op",
            poll,
            lambda r: r["operationResult"],
            poll_interval=0,
            async_sleep_fn=asyncio.sleep,
        )

    async def go():
        return await asyncio.gather(*[f.result_async() for f in [make() for _ in range(n)]])

    assert asyncio.run(go()) == ["ok"] * n
    assert peak == n, f"only {peak}/{n} operations were in flight at once"


def test_awaiting_operations_does_not_starve_asyncio_to_thread():
    """A pending await must not consume a default-executor thread.

    ``asyncio.to_thread`` is how the ``create_*_async`` factories run, and it
    draws on the same pool the old thread-bound wait did. With one pending await
    per worker thread, those factories used to have to wait for an operation to
    finish before they could start.

    Asserted by ordering rather than by elapsed time: the factory must run while
    every operation is still in flight. If a pending await held a thread, the
    factory could only start once one of them had released it by completing.
    """
    import asyncio

    polls_per_op = 30

    def make():
        state = {"calls": 0}

        def poll():
            state["calls"] += 1
            if state["calls"] >= polls_per_op:
                return {"status": "Completed", "operationResult": "ok"}
            return {"status": "InProgress"}

        # Bounded poll count, so the operations finish on their own even when
        # the assertion below fails; a failing test must not hang.
        return APIFuture("op", poll, lambda r: r["operationResult"], poll_interval=0.01)

    async def go():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: None)  # materialize the pool
        workers = loop._default_executor._max_workers

        pending = [
            asyncio.ensure_future(f.result_async()) for f in [make() for _ in range(workers)]
        ]
        await asyncio.sleep(0.02)  # let them all reach their poll loop

        factory_work = await asyncio.to_thread(lambda: "factory ran")
        completed_early = sum(1 for t in pending if t.done())

        await asyncio.gather(*pending)
        return factory_work, completed_early

    factory_work, completed_early = asyncio.run(go())
    assert factory_work == "factory ran"
    assert completed_early == 0, (
        f"{completed_early} operation(s) had to finish before to_thread could "
        f"run; the pending awaits are holding worker threads"
    )


# --- timeout is a bound on the whole wait -----------------------------------
# A future that resolves in stages (submit then poll; several chunks) must not
# spend `timeout` per stage: one deadline is taken up front and each stage gets
# only what is left.


class FakeSubmit:
    """Stands in for the ``concurrent.futures.Future`` a submitter returns.

    ``wait`` seconds of the injected clock are burned before the inner future is
    handed back; a shorter allowance times out the way the real one does.
    """

    def __init__(self, clock: FakeClock, inner, *, wait: float = 0.0) -> None:
        self._clock = clock
        self._inner = inner
        self._wait = wait

    def result(self, timeout=None):
        if timeout is not None and timeout < self._wait:
            self._clock.now += timeout
            raise concurrent.futures.TimeoutError()
        self._clock.now += self._wait
        return self._inner


def test_remaining_is_clamped_at_zero_and_unbounded_stays_none():
    clock = FakeClock()
    assert _remaining(None, clock.time) is None
    assert _remaining(10.0, clock.time) == 10.0
    clock.now = 25.0
    assert _remaining(10.0, clock.time) == 0.0


def test_result_timeout_none_waits_indefinitely():
    clock = FakeClock()
    fut = _make_future(
        clock,
        [{"status": "InProgress"}] * 5 + [{"status": "Completed", "operationResult": "ok"}],
        poll_interval=1000,
    )
    # Would raise TimeoutError on any finite deadline this small.
    assert fut.result(timeout=None) == "ok"
    assert clock.now == 5000


def test_pending_future_timeout_covers_submit_and_poll_together():
    clock = FakeClock()
    inner = APIFuture(
        "op-inner",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=3,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    # 4s of the 10s budget goes to the submit, so the poll loop gets 6s, not a
    # fresh 10s (which would let the whole call run 14s).
    pending = PendingAPIFuture(
        FakeSubmit(clock, inner, wait=4.0),
        operation_type="training",
        time_fn=clock.time,
    )
    with pytest.raises(TimeoutError) as ei:
        pending.result(timeout=10)
    assert clock.now == 10
    # The message reports the budget the caller passed, not the 6s slice the
    # poll loop got after the submit wait.
    assert "within 10.0s" in str(ei.value)
    assert "op-inner" in str(ei.value)


def test_pending_future_submit_timeout_raises_builtin_timeout_error():
    clock = FakeClock()
    pending = PendingAPIFuture(
        FakeSubmit(clock, None, wait=30.0),
        operation_type="training",
        time_fn=clock.time,
    )
    # concurrent.futures.TimeoutError is only an alias for the built-in from
    # Python 3.11 on; callers get the built-in on every supported version.
    with pytest.raises(TimeoutError) as ei:
        pending.result(timeout=5)
    assert "was not submitted within" in str(ei.value)


def test_pending_future_timeout_none_passes_through():
    clock = FakeClock()
    inner = _make_future(clock, [{"status": "Completed", "operationResult": "ok"}])
    pending = PendingAPIFuture(
        FakeSubmit(clock, inner), operation_type="training", time_fn=clock.time
    )
    assert pending.result(timeout=None) == "ok"


def test_combined_future_shares_one_deadline_across_sub_futures():
    clock = FakeClock()
    first = _make_future(
        clock,
        [{"status": "InProgress"}, {"status": "Completed", "operationResult": 1}],
        poll_interval=5,
    )
    second = APIFuture(
        "op-2",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=5,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
    )
    combined = CombinedFuture([first, second], sum, time_fn=clock.time)
    # first burns 5s of the 10s budget, so second gets 5s, not its own 10s.
    with pytest.raises(TimeoutError) as ei:
        combined.result(timeout=10)
    assert clock.now == 10
    assert "within 10.0s" in str(ei.value)
    assert "2 sub-operations" in str(ei.value)


def test_combined_future_returns_combined_result_within_budget():
    clock = FakeClock()
    subs = [_make_future(clock, [{"status": "Completed", "operationResult": n}]) for n in (1, 2, 3)]
    combined = CombinedFuture(subs, sum, time_fn=clock.time)
    assert combined.result(timeout=10) == 6


# --- the same staged-timeout contract, on the async waits --------------------
# Both subclasses now define their own result_async, so the deadline splitting
# has to be pinned there too rather than inherited from the sync path.
# PendingAPIFuture awaits its submit through asyncio.wrap_future, which needs a
# real concurrent.futures.Future; FakeSubmit will not do.


def _submitted(inner):
    """A real, already-resolved submit future yielding ``inner``."""
    cf: concurrent.futures.Future = concurrent.futures.Future()
    cf.set_result(inner)
    return cf


def test_pending_future_result_async_resolves_submit_then_polls():
    import asyncio

    clock = FakeClock()
    inner = _make_future(
        clock,
        [{"status": "InProgress"}, {"status": "Completed", "operationResult": "ok"}],
        poll_interval=5,
    )
    pending = PendingAPIFuture(_submitted(inner), operation_type="training", time_fn=clock.time)
    assert asyncio.run(pending.result_async()) == "ok"
    assert clock.sleeps == [5]


def test_pending_future_result_async_reports_the_whole_budget_on_timeout():
    """The poll stage got what the submit left; the message must not say so."""
    import asyncio

    clock = FakeClock()
    inner = APIFuture(
        "op-inner",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=3,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
    )
    pending = PendingAPIFuture(_submitted(inner), operation_type="training", time_fn=clock.time)
    with pytest.raises(TimeoutError) as ei:
        asyncio.run(pending.result_async(timeout=10))
    assert "within 10.0s" in str(ei.value)
    assert "op-inner" in str(ei.value)


def test_pending_future_result_async_submit_timeout_is_builtin_timeout_error():
    """asyncio.TimeoutError is only the built-in from 3.11 on; normalize it."""
    import asyncio

    clock = FakeClock()
    never = concurrent.futures.Future()  # submit never lands
    pending = PendingAPIFuture(never, operation_type="training", time_fn=clock.time)
    with pytest.raises(TimeoutError) as ei:
        asyncio.run(pending.result_async(timeout=0.05))
    assert "was not submitted within" in str(ei.value)


def test_pending_future_result_async_is_cached():
    import asyncio

    clock = FakeClock()
    inner = _make_future(clock, [{"status": "Completed", "operationResult": "v"}])
    pending = PendingAPIFuture(_submitted(inner), operation_type="training", time_fn=clock.time)
    assert asyncio.run(pending.result_async()) == "v"
    # A second run would poll again and raise StopIteration if not cached.
    assert asyncio.run(pending.result_async()) == "v"


def test_combined_future_result_async_is_cached():
    """Re-awaiting must not re-poll the sub-futures or re-run combine_fn."""
    import asyncio

    clock = FakeClock()
    combines = {"n": 0}

    def combine(results):
        combines["n"] += 1
        return sum(results)

    subs = [_make_future(clock, [{"status": "Completed", "operationResult": n}]) for n in (1, 2, 3)]
    combined = CombinedFuture(subs, combine, time_fn=clock.time)
    assert asyncio.run(combined.result_async()) == 6
    # A second pass would exhaust each sub-future's response iterator.
    assert asyncio.run(combined.result_async()) == 6
    assert combines["n"] == 1


def test_combined_future_result_async_shares_one_deadline():
    import asyncio

    clock = FakeClock()
    first = _make_future(
        clock,
        [{"status": "InProgress"}, {"status": "Completed", "operationResult": 1}],
        poll_interval=5,
    )
    second = APIFuture(
        "op-2",
        poll_fn=lambda: {"status": "InProgress"},
        parse_fn=lambda r: r,
        poll_interval=5,
        time_fn=clock.time,
        sleep_fn=clock.sleep,
        async_sleep_fn=clock.asleep,
    )
    combined = CombinedFuture([first, second], sum, time_fn=clock.time)
    # first burns 5s of the 10s budget, so second gets 5s, not its own 10s.
    with pytest.raises(TimeoutError) as ei:
        asyncio.run(combined.result_async(timeout=10))
    assert clock.now == 10
    assert "within 10.0s" in str(ei.value)
    assert "2 sub-operations" in str(ei.value)


# --- the async surface CombinedFuture inherits -------------------------------
# CombinedFuture is an APIFuture subclass so that chunking stays invisible in
# the public signature, and it no longer defines result_async / __await__ of its
# own. These pin that the inherited pair really does route through its combining
# result(); inheritance is easy to break with a stray override.


def test_combined_future_is_an_api_future():
    """The public annotation is APIFuture[T]; a chunked call must satisfy it."""
    clock = FakeClock()
    subs = [_make_future(clock, [{"status": "Completed", "operationResult": 1}])]
    assert isinstance(CombinedFuture(subs, sum, time_fn=clock.time), APIFuture)


def test_combined_future_await_and_result_async_combine():
    import asyncio

    clock = FakeClock()

    def _combined():
        subs = [
            _make_future(clock, [{"status": "Completed", "operationResult": n}]) for n in (1, 2, 3)
        ]
        return CombinedFuture(subs, sum, time_fn=clock.time)

    async def go():
        return await _combined(), await _combined().result_async()

    # Both inherited paths fold the sub-results, rather than returning one of
    # them or the list.
    assert asyncio.run(go()) == (6, 6)


class TestTimeoutErrorPathIsBounded:
    """Reporting a timeout must not resume the wait the caller bounded."""

    def test_a_chunked_timeout_does_not_wait_on_an_unresolved_submit(self):
        """The message names the operation only if the submit already landed.

        Reading ``operation_id`` waits for the submit with no bound, so building
        the message from it turned a bounded wait into an unbounded one.
        """
        never = concurrent.futures.Future()
        pending = PendingAPIFuture(never, operation_type="forwardBackward")
        combined = CombinedFuture([pending], lambda rs: rs)

        # The defect made this call block forever rather than return something
        # wrong, so the assertion has to be that it returns at all. Building the
        # message on another thread bounds that: an unbounded wait leaves the
        # thread alive, which is what this fails on.
        built: list = []
        builder = threading.Thread(
            target=lambda: built.append(str(combined._chunked_timeout(30.0))),
            daemon=True,
        )
        builder.start()
        builder.join(timeout=5)

        assert not builder.is_alive(), "building the timeout message waited on the submit"
        assert "not submitted" in built[0]

    def test_an_unbounded_timeout_formats_without_raising(self):
        """``None`` means no limit; formatting it as a number would raise."""
        never = concurrent.futures.Future()
        combined = CombinedFuture(
            [PendingAPIFuture(never, operation_type="forwardBackward")], lambda rs: rs
        )

        assert "no limit" in str(combined._chunked_timeout(None))

    def test_a_resolved_submit_still_names_its_operation(self):
        inner = _make_future(FakeClock(), [{"status": "Completed", "operationResult": "v"}])
        landed: concurrent.futures.Future = concurrent.futures.Future()
        landed.set_result(inner)
        combined = CombinedFuture(
            [PendingAPIFuture(landed, operation_type="forwardBackward")], lambda rs: rs
        )

        assert inner.operation_id in str(combined._chunked_timeout(30.0))


# --- the rest of the standard future surface ---------------------------------
# The name promises the protocol callers know from concurrent.futures, so a loop
# over futures should be able to ask these without special casing.


class TestStandardFutureSurface:
    """``done()``, ``exception()`` and ``cancel()`` on all three classes."""

    def test_done_is_false_until_the_result_is_kept(self):
        clock = FakeClock()
        fut = _make_future(
            clock,
            [{"status": "InProgress"}, {"status": "Completed", "operationResult": "v"}],
        )

        assert not fut.done()
        assert fut.result() == "v"
        assert fut.done()

    def test_done_stays_false_for_a_future_nobody_waited_on(self):
        """A finished operation is not a resolved future: nothing polled it."""
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "Completed", "operationResult": "v"}])

        assert not fut.done()

    def test_exception_returns_the_failure_rather_than_raising_it(self):
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "Failed", "errorDetails": "boom"}])

        error = fut.exception()

        assert isinstance(error, OperationError)
        assert "boom" in str(error)

    def test_exception_is_none_on_success(self):
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "Completed", "operationResult": "v"}])

        assert fut.exception() is None

    def test_exception_propagates_a_timeout_rather_than_returning_it(self):
        """A timeout means the question is unanswered, not that the op failed."""
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "InProgress"}] * 10, poll_interval=5)

        with pytest.raises(TimeoutError):
            fut.exception(timeout=10)

    def test_a_none_result_is_kept_rather_than_re_polled(self):
        """``None`` is a value a parse can produce, not a mark of "unresolved".

        Memoizing on truthiness sends the future back to the wire on every look,
        and reports it unresolved forever. No other test in this file has a
        ``parse_fn`` that returns ``None``, so this is the only thing standing
        between the sentinel and a quiet revert.
        """
        clock = FakeClock()
        polls = {"n": 0}

        def poll_fn():
            polls["n"] += 1
            return {"status": "Completed", "operationResult": None}

        fut = APIFuture(
            operation_id="op-1",
            poll_fn=poll_fn,
            parse_fn=lambda resp: resp["operationResult"],
            operation_type="training",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
            async_sleep_fn=clock.asleep,
        )

        assert fut.result() is None
        assert fut.done(), "a None result left the future looking unresolved"
        assert fut.result() is None
        assert polls["n"] == 1, f"the service was polled {polls['n']} times"

    def test_done_becomes_true_for_a_failed_operation(self):
        """A failure resolves the future as surely as a result does.

        Otherwise the loop these methods exist to serve never finishes: a drain
        that removes futures once ``done()`` spins on a failed one forever.
        """
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "Failed", "errorDetails": "boom"}])

        assert not fut.done()
        assert isinstance(fut.exception(), OperationError)
        assert fut.done()

    def test_a_failure_the_sdk_judges_resolves_the_future_too(self):
        """A mismatched result is as terminal as one the service calls failed.

        The check runs while parsing a *completed* response, so this failure
        arrives by a different route than a reported one. Recording only the
        reported kind would leave ``done()`` meaning one thing for an operation
        the service rejected and another for a result that came back wrong, and
        would send every later wait back to the service for an answer that
        cannot change.
        """
        clock = FakeClock()
        polls = {"n": 0}

        def poll_fn():
            polls["n"] += 1
            return {"status": "Completed"}

        def parse_fn(_resp):
            raise ResultMismatchError("computeLogprobs", 5, 3)

        fut = APIFuture(
            poll_fn=poll_fn,
            parse_fn=parse_fn,
            operation_type="computeLogprobs",
            operation_id="op-mismatch",
            status_key="status",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        with pytest.raises(ResultMismatchError):
            fut.result(timeout=5)
        assert fut.done(), "a result the SDK rejected left the future unresolved"

        first = fut.exception(timeout=5)
        second = fut.exception(timeout=5)
        assert first is second
        assert polls["n"] == 1, f"the service was asked again: {polls['n']} polls"

    def test_a_failed_request_is_not_mistaken_for_a_failed_operation(self):
        """Being unable to ask says nothing about the operation, so nothing is kept.

        Credentials expiring mid-run is the case that matters: the operation may
        well be running and may well succeed, so a later wait has to ask again
        rather than replay an answer that was never about the operation.
        """
        clock = FakeClock()
        attempts = {"n": 0}

        def poll_fn():
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise ClientError(
                    {"Error": {"Code": "ExpiredTokenException", "Message": "expired"}},
                    "GetTrainingOperationResult",
                )
            return {"status": "Completed"}

        fut = APIFuture(
            poll_fn=poll_fn,
            parse_fn=lambda resp: "parsed",
            operation_type="forwardBackward",
            operation_id="op-expired",
            status_key="status",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
        )

        with pytest.raises(ClientError):
            fut.result(timeout=5)
        assert not fut.done(), "a failed request was recorded as the operation's outcome"

        assert fut.result(timeout=5) == "parsed"
        assert fut.done()

    def test_a_failure_is_read_once_and_then_replayed(self):
        """The terminal answer is kept, so a second look does not hit the wire."""
        clock = FakeClock()
        polls = {"n": 0}

        def poll_fn():
            polls["n"] += 1
            return {"status": "Failed", "errorDetails": "boom"}

        fut = APIFuture(
            operation_id="op-1",
            poll_fn=poll_fn,
            parse_fn=lambda resp: resp,
            operation_type="training",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
            async_sleep_fn=clock.asleep,
        )

        first = fut.exception()
        second = fut.exception()

        assert first is second, "a second look built a different error"
        assert polls["n"] == 1, f"the service was polled {polls['n']} times"
        with pytest.raises(OperationError) as raised:
            fut.result()
        assert raised.value is first

    def test_a_drain_loop_over_a_failed_future_terminates(self):
        """The canonical shape a caller writes, which the defect made infinite."""
        clock = FakeClock()
        pending = [_make_future(clock, [{"status": "Failed", "errorDetails": "boom"}])]

        for _ in range(5):  # bounded so a regression fails rather than hangs
            for future in list(pending):
                if future.done():
                    pending.remove(future)
                else:
                    future.exception()
            if not pending:
                break

        assert pending == []

    def test_both_subclasses_report_a_failure_as_done(self):
        clock = FakeClock()
        inner = _make_future(clock, [{"status": "Failed", "errorDetails": "boom"}])
        pending = PendingAPIFuture(_submitted(inner), operation_type="training", time_fn=clock.time)
        sub = _make_future(clock, [{"status": "Failed", "errorDetails": "boom"}])
        combined = CombinedFuture([sub], lambda rs: rs, time_fn=clock.time)

        for future in (pending, combined):
            assert not future.done()
            assert isinstance(future.exception(), OperationError)
            assert future.done(), f"{type(future).__name__} did not resolve on failure"

    def test_cancel_reports_that_it_could_not_cancel(self):
        clock = FakeClock()
        fut = _make_future(clock, [{"status": "Completed", "operationResult": "v"}])

        assert fut.cancel() is False
        assert fut.cancelled() is False

    def test_both_subclasses_inherit_the_surface(self):
        clock = FakeClock()
        inner = _make_future(clock, [{"status": "Completed", "operationResult": "v"}])
        pending = PendingAPIFuture(_submitted(inner), operation_type="training", time_fn=clock.time)
        combined = CombinedFuture([inner], lambda rs: rs, time_fn=clock.time)

        for future in (pending, combined):
            assert future.done() is False
            assert future.cancel() is False
            assert future.cancelled() is False

        assert pending.result() == "v"
        assert pending.done() is True
        assert pending.exception() is None


# --- concurrent waiters on one future ----------------------------------------
# Completion side effects are composed into parse_fn (see
# test_parse_fn_side_effects_fire_once_on_success), so two waiters resolving the
# same future must not both run it. These tests race real threads; the injected
# clock keeps them from waiting.


class TestConcurrentResolution:
    """One future, several waiters: resolve once, agree on the result."""

    def test_two_threads_resolving_one_future_parse_once(self):
        clock = FakeClock()
        # Both waiters are released holding a completed response, so both reach
        # the commit with a result to write.
        gate = threading.Barrier(2)
        parses = []

        def poll_fn():
            gate.wait(timeout=5)
            return {"status": "Completed", "operationResult": "v"}

        def parse_fn(resp):
            parses.append(resp)
            return ["v"]  # a fresh object per call, so identity is meaningful

        fut = APIFuture(
            operation_id="op-1",
            poll_fn=poll_fn,
            parse_fn=parse_fn,
            operation_type="training",
            time_fn=clock.time,
            sleep_fn=clock.sleep,
            async_sleep_fn=clock.asleep,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            waiters = [pool.submit(fut.result) for _ in range(2)]
            first, second = (w.result(timeout=5) for w in waiters)

        assert len(parses) == 1, "parse_fn ran per waiter instead of per future"
        assert first is second, "waiters were handed different result objects"
        assert fut.result() is first

    def test_two_coroutines_awaiting_one_future_parse_once(self):
        import asyncio

        clock = FakeClock()
        parses = []

        def parse_fn(resp):
            parses.append(resp)
            return ["v"]

        # Enough completed responses for both coroutines to poll independently.
        fut = _make_future(
            clock,
            [{"status": "Completed", "operationResult": "v"}] * 2,
            parse_fn=parse_fn,
        )

        async def go():
            return await asyncio.gather(fut.result_async(), fut.result_async())

        first, second = asyncio.run(go())

        assert len(parses) == 1
        assert first is second

    def test_combine_fn_runs_once_across_concurrent_waiters(self):
        clock = FakeClock()
        gate = threading.Barrier(2)
        combines = []

        def combine(sub_results):
            combines.append(sub_results)
            return list(sub_results)

        sub = _make_future(clock, [{"status": "Completed", "operationResult": 1}])
        combined = CombinedFuture([sub], combine, time_fn=clock.time)

        # Release both waiters only once each holds its sub-result, so both
        # reach the fold.
        settled = sub.result

        def gated_result(**kwargs):
            gate.wait(timeout=5)
            return settled(**kwargs)

        sub.result = gated_result

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            waiters = [pool.submit(combined.result) for _ in range(2)]
            first, second = (w.result(timeout=5) for w in waiters)

        assert len(combines) == 1, "combine_fn ran per waiter instead of per future"
        assert first is second

    def test_pending_future_waiters_agree_on_one_result(self):
        clock = FakeClock()
        gate = threading.Barrier(2)

        class MemoizesNothing:
            """An inner future returning a fresh object per call.

            A real inner :class:`APIFuture` memoizes, which would hide whether
            ``PendingAPIFuture`` commits its own result once; this leaves it as
            the only thing that can.
            """

            operation_id = "op-1"

            def result(self, timeout=None):
                gate.wait(timeout=5)
                return ["v"]

        pending = PendingAPIFuture(
            _submitted(MemoizesNothing()),
            operation_type="training",
            time_fn=clock.time,
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            waiters = [pool.submit(pending.result) for _ in range(2)]
            first, second = (w.result(timeout=5) for w in waiters)

        assert first is second
        assert pending.result() is first

    def test_the_resolve_lock_is_never_held_across_a_poll(self):
        """The lock guards classify + memoize only, never the wire call.

        Held across a poll it would make a bounded ``result(timeout=...)`` wait
        out another waiter's unbounded poll, and would occupy the worker threads
        ``result_async`` needs to poll on at all.
        """
        clock = FakeClock()
        fut = _make_future(
            clock,
            [{"status": "InProgress"}, {"status": "Completed", "operationResult": "v"}],
        )
        acquirable = []
        wire = fut._poll_fn

        def probing_poll():
            got = fut._lock.acquire(blocking=False)
            acquirable.append(got)
            if got:
                fut._lock.release()
            return wire()

        fut._poll_fn = probing_poll

        assert fut.result() == "v"
        assert acquirable == [True, True]


def test_a_submit_timeout_leaves_the_queued_operation_alone():
    """A timeout waiting on the submit must not cancel the submit.

    The wait used to pass its own cancellation through to the background submit,
    so a timed-out caller silently dropped the operation and every later attempt
    to resolve the same future raised ``CancelledError``.
    """
    import asyncio

    submit = concurrent.futures.Future()
    pending = PendingAPIFuture(submit, operation_type="training")

    async def go():
        with pytest.raises(TimeoutError, match="was not submitted within"):
            await pending.result_async(timeout=0)

    asyncio.run(go())

    assert not submit.cancelled()


def test_the_future_is_still_resolvable_after_a_submit_timeout():
    """The documented contract: the operation keeps running and can be picked up."""
    import asyncio

    clock = FakeClock()
    inner = _make_future(clock, [{"status": "Completed", "operationResult": "ok"}])
    submit = concurrent.futures.Future()
    pending = PendingAPIFuture(submit, operation_type="training", time_fn=clock.time)

    async def timed_out():
        with pytest.raises(TimeoutError):
            await pending.result_async(timeout=0)

    asyncio.run(timed_out())

    submit.set_result(inner)

    assert asyncio.run(pending.result_async(timeout=None)) == "ok"
