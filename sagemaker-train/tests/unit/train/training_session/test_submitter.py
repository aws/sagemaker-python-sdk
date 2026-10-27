# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Unit tests for sagemaker.train.training_session._submitter.

No test performs a network call; waits are sub-millisecond polls with
hard deadlines, so the file stays well under a second per test.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import gc
import os
import threading
import time

import pytest

from sagemaker.train.training_session._submitter import BackgroundSubmitter
from sagemaker.train.training_session.exceptions import TrainingSessionError


def test_ordered_submits_run_in_call_order():
    sub = BackgroundSubmitter()
    order = []
    futures = []

    def make(i):
        async def _op():
            await asyncio.sleep(0.001 if i % 2 == 0 else 0)
            order.append(i)
            return i

        return _op

    for i in range(20):
        futures.append(sub.submit_ordered(make(i)))
    results = [f.result(timeout=5) for f in futures]
    assert results == list(range(20))
    assert order == list(range(20))
    sub.close()


def test_submit_returns_without_waiting_for_completion():
    sub = BackgroundSubmitter()
    release = threading.Event()

    async def _op():
        while not release.is_set():
            await asyncio.sleep(0.001)
        return 42

    start = time.monotonic()
    futures = [sub.submit_ordered(_op) for _ in range(10)]
    elapsed = time.monotonic() - start
    assert elapsed < 0.05
    assert not futures[0].done()
    release.set()
    assert [f.result(timeout=5) for f in futures] == [42] * 10
    sub.close()


def test_submit_error_surfaces_at_result_time_and_gate_releases():
    sub = BackgroundSubmitter()

    async def _boom():
        raise ValueError("submit failed")

    async def _ok():
        return "fine"

    bad = sub.submit_ordered(_boom)
    good = sub.submit_ordered(_ok)
    with pytest.raises(ValueError, match="submit failed"):
        bad.result(timeout=5)
    assert good.result(timeout=5) == "fine"
    sub.close()


def test_unordered_submits_overlap():
    sub = BackgroundSubmitter()
    gate = threading.Event()
    started = []

    async def _op():
        started.append(1)
        while not gate.is_set():
            await asyncio.sleep(0.001)
        return len(started)

    f1 = sub.submit_unordered(_op)
    f2 = sub.submit_unordered(_op)
    deadline = time.monotonic() + 2
    while len(started) < 2 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert len(started) == 2
    gate.set()
    f1.result(timeout=5)
    f2.result(timeout=5)
    sub.close()


def test_close_is_idempotent_and_blocks_new_submits():
    sub = BackgroundSubmitter()

    async def _op():
        return 1

    assert sub.submit_ordered(_op).result(timeout=5) == 1
    sub.close()
    sub.close()
    with pytest.raises(TrainingSessionError, match="closed and accepts no new operations"):
        sub.submit_ordered(_op)


def test_close_before_first_submit_is_a_noop():
    sub = BackgroundSubmitter()
    sub.close()
    with pytest.raises(TrainingSessionError):
        sub.submit_unordered(None)


def test_a_rejected_schedule_claims_no_ordered_slot():
    """A slot handed out but never scheduled would stall every later submit.

    Ordered submits release their turn in sequence, so a slot claimed by a
    coroutine that never runs is never released, and every submit behind it waits
    on that turn forever. The slot is therefore claimed only once the coroutine is
    on the loop.
    """
    sub = BackgroundSubmitter()

    async def _op():
        return 1

    assert sub.submit_ordered(_op).result(timeout=5) == 1
    claimed = sub._next_seq

    # A closed loop is what makes scheduling fail outright.
    sub._loop.call_soon_threadsafe(sub._loop.stop)
    sub._thread.join(timeout=5)
    sub._loop.close()
    with pytest.raises(RuntimeError):
        sub.submit_ordered(_op)

    assert sub._next_seq == claimed


def test_closing_while_submits_are_in_flight_never_hangs():
    """close() must not be able to strand a submit mid-handoff.

    The slot is claimed and scheduled under the lock close() takes, so close()
    cannot land between the two. If it could, the ordered submits behind the
    stranded slot would wait on their turn forever: a hang, not an error.
    """
    for _ in range(20):
        sub = BackgroundSubmitter()

        async def _op():
            return 1

        futures = []
        errors = []

        def _submit():
            for _ in range(5):
                try:
                    futures.append(sub.submit_ordered(_op))
                except TrainingSessionError:
                    # Submitting after close is an error by design.
                    pass
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

        threads = [threading.Thread(target=_submit) for _ in range(4)]
        for t in threads:
            t.start()
        sub.close()
        for t in threads:
            t.join(timeout=5)
            assert not t.is_alive()

        assert errors == []
        for future in futures:
            # Resolving or raising are both fine; blocking is the defect. Only
            # the timeout is tolerated as a failure signal, so a hang fails here
            # instead of being swallowed with everything else.
            try:
                future.result(timeout=5)
            except concurrent.futures.TimeoutError:  # pragma: no cover
                pytest.fail("a submit never resolved after close(); close() hung it")
            except Exception:  # noqa: BLE001 - a failed submit is a resolved one
                pass


def test_a_cancelled_turn_still_releases_the_next_submit():
    """A submit cancelled while waiting its turn must not strand the queue.

    Reachable from the public API: awaiting a future with a timeout cancels the
    submit underneath it. Before, the release only ran for a submit that started,
    so the slot stayed taken and every later ordered submit waited on it forever.
    """
    sub = BackgroundSubmitter(name="cancel-turn")
    started, release_first = threading.Event(), threading.Event()

    async def first():
        started.set()
        await asyncio.get_running_loop().run_in_executor(None, release_first.wait)
        return "first"

    try:
        held = sub.submit_ordered(first)
        assert started.wait(5)

        waiting = sub.submit_ordered(lambda: asyncio.sleep(0, result="waiting"))
        time.sleep(0.2)  # let it reach the turn wait
        assert waiting.cancel()

        release_first.set()
        assert held.result(timeout=5) == "first"

        after = sub.submit_ordered(lambda: asyncio.sleep(0, result="after"))
        assert after.result(timeout=5) == "after"
    finally:
        release_first.set()
        sub.close()


def test_close_fails_a_submit_it_could_not_wait_out():
    """Stopping the loop freezes an unfinished submit, so it has to be failed.

    Left alone, its future never resolves, and a caller waiting on the result
    with the default unbounded timeout waits for the life of the process.
    """
    sub = BackgroundSubmitter(name="strand")
    block = threading.Event()

    async def never_finishes():
        await asyncio.get_running_loop().run_in_executor(None, block.wait)
        return "never"

    try:
        stuck = sub.submit_ordered(never_finishes)
        time.sleep(0.2)
        sub.close(timeout=0.5)

        with pytest.raises(TrainingSessionError, match="still in flight"):
            stuck.result(timeout=5)
    finally:
        block.set()


def test_close_tolerates_a_pending_entry_that_already_settled():
    """Losing the race to fail a submit must not abort the shutdown.

    ``close`` fails whatever is still pending once its wait runs out, and the
    loop thread can complete that same future in between, which makes setting an
    exception on it raise. Letting that escape skips the stop and the join,
    leaking the very thread the shutdown exists to retire.

    The window is a genuine race, so it is set up rather than raced for: a
    finished future placed in the pending set puts ``close`` in exactly the state
    it would reach by losing, without depending on timing to get there.
    """
    sub = BackgroundSubmitter(name="close-settled")
    started = sub.submit_unordered(lambda: asyncio.sleep(0, result="ok"))
    assert started.result(timeout=5) == "ok"

    settled: concurrent.futures.Future = concurrent.futures.Future()
    settled.set_result("already done")
    with sub._lock:
        sub._pending.add(settled)

    sub.close(timeout=0.05)

    assert settled.result(timeout=1) == "already done", "a settled result was replaced"
    assert (
        sub._thread is not None and not sub._thread.is_alive()
    ), "the loop thread outlived close(), so the stop or the join was skipped"


def _await_parked(sub, count, deadline=5.0):
    """Poll until ``count`` submits are parked waiting on their turn."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        if len(sub._waiters) >= count:
            return True
        time.sleep(0.001)
    return False


def test_a_cancelled_turn_does_not_let_a_later_submit_overtake():
    """Cancelling a queued submit must not hand on a turn it never held.

    The holder is still on the wire, so releasing its successor early puts two
    ordered operations on the wire out of call order, which is the one thing
    ordered submission exists to prevent.
    """
    sub = BackgroundSubmitter(name="cancel-overtake")
    order = []
    started, release_first = threading.Event(), threading.Event()

    async def first():
        order.append("1-start")
        started.set()
        await asyncio.get_running_loop().run_in_executor(None, release_first.wait)
        order.append("1-end")
        return "first"

    async def second():
        order.append("2-start")
        return "second"

    try:
        held = sub.submit_ordered(first)
        assert started.wait(5)

        queued_second = sub.submit_ordered(second)
        queued_third = sub.submit_ordered(lambda: asyncio.sleep(0, result="third"))
        assert _await_parked(sub, 2), "the later submits never parked on their turn"

        assert queued_third.cancel()
        # Long enough that a wrongly released successor would have run by now,
        # short enough to keep the test well under a second.
        time.sleep(0.05)
        assert order == ["1-start"], f"a later submit overtook the holder: {order}"

        release_first.set()
        assert held.result(timeout=5) == "first"
        assert queued_second.result(timeout=5) == "second"
        assert order == ["1-start", "1-end", "2-start"]
    finally:
        release_first.set()
        sub.close()


def test_the_turn_skips_a_run_of_cancelled_sequences():
    """Several cancellations in a row must not leave the turn stuck on any of them.

    The holder advances past every sequence that was cancelled while queued, so a
    submit behind a run of them still gets its turn rather than waiting on a
    sequence that will never run.
    """
    sub = BackgroundSubmitter(name="cancel-run")
    started, release_first = threading.Event(), threading.Event()

    async def first():
        started.set()
        await asyncio.get_running_loop().run_in_executor(None, release_first.wait)
        return "first"

    try:
        held = sub.submit_ordered(first)
        assert started.wait(5)

        cancelled = [
            sub.submit_ordered(lambda: asyncio.sleep(0, result="skipped")) for _ in range(3)
        ]
        last = sub.submit_ordered(lambda: asyncio.sleep(0, result="last"))
        assert _await_parked(sub, 4), "the later submits never parked on their turn"

        for future in cancelled:
            assert future.cancel()

        release_first.set()
        assert held.result(timeout=5) == "first"
        assert last.result(timeout=5) == "last"
    finally:
        release_first.set()
        sub.close()


# --- descriptor release ------------------------------------------------------


def _live_fds():
    """Descriptor count for this process; Linux-only, so the caller skips."""
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


def test_close_returns_the_loops_descriptors_rather_than_waiting_for_gc():
    """Stopping a loop only parks it; the descriptors go back when it closes.

    An open loop holds its readiness poller and its self-pipe pair, three
    descriptors, until the loop object is collected. Whoever holds a reference
    therefore decides when they come back, and the session handle holds every
    submitter it was given for the life of the session, so leaving this to
    collection meant a closed submitter kept its descriptors anyway.
    """
    if not os.path.isdir(f"/proc/{os.getpid()}/fd"):
        pytest.skip("descriptor accounting via /proc is Linux-only")

    # Settle cyclic garbage first: an earlier test's parked loop collected
    # mid-measurement would drop the count below the baseline asserted here.
    gc.collect()
    before = _live_fds()
    sub = BackgroundSubmitter(name="fd-release")
    sub.submit_unordered(lambda: asyncio.sleep(0, result="ok")).result(timeout=5)
    assert _live_fds() > before, "the loop did not come up, so nothing is measured"

    sub.close()

    assert sub._loop.is_closed(), "close() left the loop open"
    # Still referenced by ``sub``, so this passes only because close() released it.
    assert _live_fds() == before


def test_close_stays_idempotent_once_it_has_closed_the_loop():
    """A second close() must not raise on the loop the first one closed.

    Callers rely on this: the session teardown drains submitters that may have
    closed themselves, and stop() followed by delete() closes the same one twice.
    """
    sub = BackgroundSubmitter(name="double-close")
    sub.submit_unordered(lambda: asyncio.sleep(0, result="ok")).result(timeout=5)
    sub.close()
    sub.close()
    sub.close()


def test_a_join_that_runs_out_leaves_the_loop_open_for_a_later_close():
    """Closing a still-running loop raises, so the timeout path must not try.

    Losing the descriptors to an exception here would also skip the caller's own
    teardown, and a later close() can still finish the job.
    """
    sub = BackgroundSubmitter(name="join-timeout")
    block = threading.Event()
    entered = threading.Event()

    async def wedge_the_loop():
        # Blocks the loop thread itself, not an executor worker, so run_forever
        # cannot return and the join is guaranteed to run out.
        entered.set()
        block.wait()

    try:
        sub.submit_unordered(wedge_the_loop)
        assert entered.wait(5)
        sub.close(timeout=0.1)

        assert sub._thread is not None and sub._thread.is_alive()
        assert not sub._loop.is_closed(), "closing a running loop would have raised"
    finally:
        block.set()
        sub._thread.join(timeout=5)
        sub.close()
