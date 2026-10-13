# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Locks in the measured cleanup-trigger grid (see session_lifecycle_grid.py).

Each case runs ``session_lifecycle_grid.py`` in a child process, because the two
interesting triggers, interpreter exit and signal death, are not observable
from inside the process they happen in.

The measured results below were identical on CPython 3.9, 3.11, 3.12 and 3.13,
so they are asserted rather than merely recorded. Two of them are the *absence*
of cleanup, and they are the ones worth re-reading if this file ever fails:

* ``sigterm``: Python installs no SIGTERM handler, so the default disposition
  terminates the process outright and no finalizer runs. Anything supervised
  (containers, batch schedulers) must stop its own sessions, or install a
  handler; the SDK deliberately does not hijack a process-wide signal.
* ``os_exit``: skips finalizers and atexit by design. No in-process mechanism
  can cover it; such a session is reclaimed once it passes its inactivity
  timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

_GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "session_lifecycle_grid.py")

# trigger -> (cleanup expected, expected exit code)
CASES = {
    "del_then_collect": (True, 0),
    "falls_out_of_scope": (True, 0),
    "rebound": (True, 0),
    "reference_cycle": (True, 0),
    "process_exit": (True, 0),
    "sys_exit": (True, 3),
    "uncaught_exception": (True, 1),
    "sigint": (True, -2),
    "explicit_disarm": (False, 0),
    "sigterm": (False, -15),
    "os_exit": (False, 0),
}


#: Seconds a grid child gets. Each one starts an interpreter and exits, so this
#: is generous; bounded because the thing under test is interpreter teardown, and
#: a child that wedges there would otherwise hang the run rather than fail it.
_CHILD_TIMEOUT = 60


def _run(trigger: str, tmp_path) -> tuple[bool, int, str]:
    out = tmp_path / f"{trigger}.txt"
    try:
        proc = subprocess.run(
            [sys.executable, _GRID, trigger, str(out)],
            capture_output=True,
            text=True,
            timeout=_CHILD_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"{trigger}: grid child did not exit within {_CHILD_TIMEOUT}s")
    recorded = out.read_text() if out.exists() else ""
    return bool(recorded.strip()), proc.returncode, recorded


@pytest.mark.parametrize("trigger", sorted(CASES))
def test_cleanup_trigger_grid(trigger, tmp_path):
    expected_cleanup, expected_rc = CASES[trigger]
    fired, rc, recorded = _run(trigger, tmp_path)
    assert rc == expected_rc, f"{trigger}: unexpected exit code {rc}"
    assert fired is expected_cleanup, (
        f"{trigger}: cleanup fired={fired}, expected {expected_cleanup} "
        f"(recorded: {recorded!r})"
    )


def test_interpreter_is_not_already_finalizing_when_cleanup_runs(tmp_path):
    """The exit-time path must not be suppressed by a ``sys.is_finalizing`` guard.

    Measured on 3.9-3.13: at the point ``weakref.finalize`` invokes pending
    callbacks during shutdown, ``sys.is_finalizing()`` is still False and a
    normal call is reachable. An implementation that skipped cleanup whenever
    ``is_finalizing()`` were True would therefore be correct but useless: it
    would never skip, while one that skipped on *shutdown* would drop every
    ``python train.py`` session. This test pins the observed behaviour so the
    reasoning behind _auto_stop's unguarded attempt stays checkable.
    """
    fired, rc, recorded = _run("process_exit", tmp_path)
    assert fired and rc == 0
    assert "finalizing=False" in recorded, recorded
