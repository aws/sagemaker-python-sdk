# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
"""Does the automatic-cleanup finalizer fire, for each way a client can go away?

A training client stops its session when it is released, and "released" covers
more shapes than a single test can express. Two of them cannot be settled from
the CPython docs alone, and neither needs a live session:

* At interpreter exit: is a call still reachable that late, and is
  ``sys.is_finalizing()`` already True by then, which would suppress a guarded
  implementation?
* Inside a reference cycle, where refcounting alone cannot reclaim the client.

Run as a child process, one trigger per invocation::

    python session_lifecycle_grid.py <trigger> <outfile>

The service is a stub whose ``stop_session`` appends to ``<outfile>``, so the
parent can read the result after this process is gone, which is the only way to
observe an exit-time finalizer at all. ``test_session_lifecycle_grid.py`` drives
every trigger and asserts the grid.
"""

from __future__ import annotations

import gc
import importlib.util
import os
import signal
import sys

# Loaded straight from its file rather than imported as
# ``sagemaker.train.training_session._session_handle``, which would execute the
# package __init__ and pull in its third-party dependencies. Cleanup semantics
# depend on nothing but the standard library, so this harness stays runnable on
# any bare interpreter, which is what lets the same grid run across CPython
# versions without building an environment for each.
_MODULE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "..",
    "..",
    "src",
    "sagemaker",
    "train",
    "training_session",
    "_session_handle.py",
)
_spec = importlib.util.spec_from_file_location("_session_handle", _MODULE_PATH)
_session_handle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_session_handle)
_SessionHandle = _session_handle._SessionHandle

SESSION = "grid-session"


class FileRecordingService:
    """Appends one line per stop, plus what the interpreter state was.

    Writes and flushes immediately: an exit-time finalizer may be racing the
    teardown of the io layer, so buffering the result could lose it.
    """

    def __init__(self, path: str) -> None:
        self._path = path

    def stop_session(self, training_session_name: str) -> None:
        self._stop_session_if_active(training_session_name)

    def _stop_session_if_active(self, training_session_name: str) -> bool:
        # The automatic path calls this one, so the stub has to offer it too;
        # True means "the session was running and a stop was issued".
        with open(self._path, "a") as fh:
            fh.write(f"{training_session_name} finalizing={sys.is_finalizing()}\n")
            fh.flush()
        return True


def _make(path: str) -> _SessionHandle:
    return _SessionHandle(SESSION, FileRecordingService(path))


# --- triggers ---------------------------------------------------------------
# Each returns normally unless it is meant to kill the process.


def t_del_then_collect(path: str) -> None:
    handle = _make(path)
    del handle
    gc.collect()


def t_falls_out_of_scope(path: str) -> None:
    def _inner() -> None:
        _make(path)

    _inner()
    gc.collect()


def t_rebound(path: str) -> None:
    handle = _make(path)
    handle = None  # noqa: F841 - the point is dropping the last reference
    gc.collect()


def t_reference_cycle(path: str) -> None:
    # Refcounting cannot reclaim this; only the cycle collector can.
    class Holder:
        pass

    holder = Holder()
    holder.handle = _make(path)
    holder.self_ref = holder
    del holder
    gc.collect()


def t_explicit_disarm(path: str) -> None:
    handle = _make(path)
    handle.disarm()
    del handle
    gc.collect()


def t_process_exit(path: str) -> None:
    # Still referenced from this frame at return: only interpreter exit can
    # release it. This is the `python train.py` case, and the reason the
    # exit-time path has to work at all.
    handle = _make(path)
    globals()["_kept_alive"] = handle


def t_sys_exit(path: str) -> None:
    globals()["_kept_alive"] = _make(path)
    sys.exit(3)


def t_uncaught_exception(path: str) -> None:
    globals()["_kept_alive"] = _make(path)
    raise RuntimeError("deliberate: testing cleanup on an uncaught exception")


def t_sigint(path: str) -> None:
    globals()["_kept_alive"] = _make(path)
    # A parent that ignores SIGINT passes that disposition to its children, and
    # some build environments do. Install the interpreter default so the signal
    # raises KeyboardInterrupt here rather than being discarded.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    os.kill(os.getpid(), signal.SIGINT)


def t_sigterm(path: str) -> None:
    globals()["_kept_alive"] = _make(path)
    # Set explicitly for the same reason as the SIGINT case: an inherited
    # disposition would otherwise decide what this measures.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def t_os_exit(path: str) -> None:
    # Documents the gap: os._exit skips finalizers and atexit entirely, so no
    # in-process mechanism can cover it. Such a session is reclaimed once it
    # passes its inactivity timeout.
    globals()["_kept_alive"] = _make(path)
    os._exit(0)


TRIGGERS = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith("t_")}


def main() -> int:
    trigger, path = sys.argv[1], sys.argv[2]
    TRIGGERS[trigger](path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
