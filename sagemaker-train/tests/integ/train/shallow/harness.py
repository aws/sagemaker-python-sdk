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
"""Submit-then-stop harness for shallow training-job integration tests.

Why this exists
---------------
``CreateTrainingJob`` returns a TrainingJobArn only after the request has
cleared every synchronous server-side gate: public-model shape validation,
SigV4, ``sagemaker:CreateTrainingJob`` authorization (including condition
keys), ``iam:PassRole`` on the execution role, the training backend's ~56
synchronous request validators, its role-assuming validators (which make real
S3/ECR/FSx calls as the customer), post-validator business logic (training-plan
capacity, routing, recipe filtering) and finally a conditional write that
rejects duplicate job names.

So "the ARN came back" is a strong assertion: the payload was accepted by the
service exactly as the SDK shaped it, and the caller held the permissions
required to submit it. That is materially more coverage than ``dry_run=True``
(which returns before submitting and so exercises only client-side validation
-- see ``tests/integ/train/test_dry_run_integration.py``), and it costs a
fraction of a full training run because we stop the job immediately instead of
waiting for it to train.

What this deliberately does NOT assert
--------------------------------------
Nothing about training *behaviour*: no model artifacts, no metrics, no
container logs, no convergence. Those require a job to actually run and remain
the job of the existing deep integration tests. These tests answer one
question only -- "would the service accept this request?"

Cost and capacity notes
-----------------------
Stopping is not free and not instantaneous. ``StopTrainingJob`` marks the job
``Stopping`` in the backend and returns; the compute layer reacts
asynchronously. Meanwhile the create call has already handed the job to a state
machine and queued it, so capacity acquisition has begun. In practice a job
stopped within seconds is torn down while still in ``Starting``/``Pending``,
before instances become billable, but that is a timing property rather than a
guarantee.

Two consequences shape this module:

* ``DEFAULT_INSTANCE_TYPE`` is a small CPU instance. Payload validation and
  permission checks are instance-type agnostic, so there is no reason to ask
  for scarce accelerator capacity. Tests that specifically need to prove an
  accelerator-shaped request is accepted say so explicitly.
* We never set ``keep_alive_period_in_seconds``. A warm pool would outlive the
  stop and keep instances provisioned after the test finished.

Teardown runs in a ``finally`` so a failing assertion still stops the job, and
is itself best-effort: a job that already reached a terminal state cannot be
stopped and that is not a failure.
"""

from __future__ import absolute_import

import errno
import inspect
import logging
import os
import random
import tempfile
import time
from contextlib import contextmanager

import pytest
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Concurrency cap (training-job service quotas)
# --------------------------------------------------------------------------
# Two different quotas apply, in two different units, and the cap has to be safe
# for both:
#
#   * serverless (the default recipe-trainer path, no explicit `compute`) is
#     bounded by "Maximum number of concurrent model customization serverless
#     jobs per Region" -- a count of *jobs*, currently 20. Instance-type quotas
#     do not apply to these at all.
#   * serverful (an explicit `TrainingJobCompute`/`Compute`, i.e. the
#     `ModelTrainer` tests, the tuner, and `test_explicit_compute_is_accepted`)
#     is bounded by the per-instance-type quota, e.g. "ml.m5.large for training
#     job usage" -- a count of *instances*.
#
# A slot therefore means "one concurrent job" and a job costs
# `max(1, instance_count)` slots: 1 for a serverless job, and its instance count
# for a serverful one. That is deliberately the stricter of the two readings, so
# one cap keeps the suite inside both quotas without needing to know which kind
# of job a given test produces.
#
# What a slot has to track -- and the trap it is easy to fall into. The service
# counts a job against the concurrency quota from `CreateTrainingJob` until the
# job reaches a *terminal* state, NOT until `StopTrainingJob` returns. Those are
# far apart: measured against the service, `stop()` returns in a few seconds but
# the job does not reach `Stopped` for ~1-3 minutes afterwards while the backend
# tears down the (never-billed) reservation. An earlier version of this cap
# released the slot when `stop()` returned, and it did not bound anything: with
# the cap at 10 and 8 workers, each slot recycled ~20 times inside a single
# job's counted lifetime, so the suite peaked at ~37 concurrent jobs and tripped
# `ResourceLimitExceeded` at a utilization of 21 against the limit of 20. The
# slot must therefore be held until the job is terminal (see
# `_wait_until_terminal`), which is the point of `SHALLOW_MAX_CONCURRENT_JOBS`.
#
# Why a cap rather than batches: capping bounds the *peak* directly and keeps
# bounding it if `-n` is raised or a test starts asking for more instances,
# whereas batches of N only serialize submission. The two are equivalent when
# the slot is held to terminal -- a cap of 10 is exactly "at most 10 jobs
# counted at once" -- but the cap needs no bookkeeping of which test is in which
# batch. `SHALLOW_MAX_CONCURRENT_JOBS=0` disables it for a single-worker
# debugging run.
#
# Cost of holding to terminal: the suite's wall-clock floor becomes roughly
# (#jobs * drain_seconds) / cap rather than tracking the worker count. At ~84
# jobs, a ~75s median drain and cap 10 that is ~8-12 min (versus ~2 min if the
# slot were released early -- but that "fast" run is the one that breaches the
# quota). 10 is under the serverless job quota (20) with room for the deep
# CodeBuild suite, which runs against the same account+region concurrently and
# also submits serverless jobs, to take the rest without the two together
# breaching 20.
DEFAULT_MAX_CONCURRENT_JOBS = 10


def _max_concurrent_jobs():
    """Read the cap at call time so tests can monkeypatch the environment."""
    raw = os.environ.get("SHALLOW_MAX_CONCURRENT_JOBS")
    if raw is None:
        return DEFAULT_MAX_CONCURRENT_JOBS
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring non-integer SHALLOW_MAX_CONCURRENT_JOBS=%r; using %d",
            raw,
            DEFAULT_MAX_CONCURRENT_JOBS,
        )
        return DEFAULT_MAX_CONCURRENT_JOBS


# The slot directory must be shared by every xdist worker, and workers are
# separate processes, so an in-process semaphore would not bound anything. Slots
# are files in a directory keyed to the run: creating one with O_EXCL is atomic
# on POSIX, which is all the mutual exclusion this needs. Keyed on the xdist
# session id (falling back to the parent pid) so two concurrent local runs get
# their own budgets rather than deadlocking against each other -- and so a
# stale directory from a killed run is never mistaken for live slots.
def _slot_dir():
    key = os.environ.get("PYTEST_XDIST_TESTRUNUID") or str(os.getppid())
    return os.path.join(tempfile.gettempdir(), f"sm-shallow-slots-{key}")


# Waiting for a *free* slot is bounded so a leaked slot degrades into a slower
# run rather than a hung one. With the slot now held until the job is terminal
# (~1-3 min), a worker can legitimately queue behind several jobs' drains, so
# this is generous; anything approaching it means slots leaked. The wait logs
# and proceeds instead of failing the test, because the quota is a throttle
# rather than a correctness property.
_SLOT_WAIT_TIMEOUT = 900
_SLOT_POLL_INTERVAL = 0.5


@contextmanager
def job_slots(count=1):
    """Hold ``count`` concurrency slots for the duration of the block.

    Bounds what this suite has in flight at once, across all xdist workers, to
    ``SHALLOW_MAX_CONCURRENT_JOBS`` (default ``DEFAULT_MAX_CONCURRENT_JOBS``).
    A slot is one concurrent job; a serverful job also takes one per additional
    instance, which keeps a single cap valid against both the serverless
    job-count quota and the per-instance-type quota.

    Slots are always released, including when the body raises, so a failing
    assertion cannot strand capacity for the rest of the run.
    """
    cap = _max_concurrent_jobs()
    if cap <= 0 or count <= 0:
        yield
        return

    # A single test asking for more than the cap must not deadlock against
    # itself: clamp, and say so, rather than waiting for slots that can never
    # all be free.
    if count > cap:
        logger.warning(
            "Test requests %d slots but the cap is %d; clamping. "
            "Raise SHALLOW_MAX_CONCURRENT_JOBS if this is intentional.",
            count,
            cap,
        )
        count = cap

    directory = _slot_dir()
    os.makedirs(directory, exist_ok=True)

    held = []
    deadline = time.time() + _SLOT_WAIT_TIMEOUT
    try:
        while len(held) < count:
            for index in range(cap):
                if len(held) == count:
                    break
                path = os.path.join(directory, f"slot-{index}")
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except OSError as e:
                    if e.errno == errno.EEXIST:
                        continue  # taken by another worker
                    raise
                os.close(fd)
                held.append(path)

            if len(held) == count:
                break

            if time.time() > deadline:
                # Proceed rather than fail: the cap is a courtesy to the
                # account's quota, not an assertion about the SDK.
                logger.warning(
                    "Waited %ds for %d job slot(s) and got %d. Proceeding anyway "
                    "(slots may have leaked from a killed run: %s).",
                    _SLOT_WAIT_TIMEOUT,
                    count,
                    len(held),
                    directory,
                )
                break

            time.sleep(_SLOT_POLL_INTERVAL)

        yield
    finally:
        for path in held:
            try:
                os.unlink(path)
            except OSError:  # pragma: no cover - already gone
                pass


# A small CPU instance is sufficient: acceptance of the request does not depend
# on the instance type being an accelerator, and asking for GPU capacity we
# immediately discard is both slower and antisocial in a shared test account.
DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_INSTANCE_COUNT = 1

# Public DLC, present in every commercial region we test in. Using a real image
# matters: the backend's role-assuming validators resolve the training image
# against ECR, so a bogus URI would fail for the wrong reason.
CPU_IMAGE = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.0-cpu-py310"

# Keep the advertised runtime short. It should never be reached (we stop the job
# long before), but if a stop were somehow lost this bounds the damage.
MAX_RUNTIME_IN_SECONDS = 600

# Terminal/near-terminal states that make StopTrainingJob a no-op or an error.
_UNSTOPPABLE_STATUSES = frozenset({"Completed", "Failed", "Stopped", "Stopping"})

# States in which the service no longer counts the job against the concurrency
# quota. A slot is held until the job reaches one of these -- see
# `_wait_until_terminal` and the note on `DEFAULT_MAX_CONCURRENT_JOBS`.
_TERMINAL_STATUSES = frozenset({"Completed", "Failed", "Stopped"})

# How long a slot waits for its job to actually drain before giving up and
# releasing anyway. Measured drains are ~1-3 min; this is a ceiling, not an
# expectation. Releasing early (like the timeout on acquiring a slot) trades a
# possible brief quota overshoot for not hanging the whole suite on one stuck
# job -- the quota is a throttle, not a correctness property.
_DRAIN_WAIT_TIMEOUT = 300
_DRAIN_POLL_INTERVAL = 5


# Name length limits differ per resource, and the service enforces them strictly.
# Verified against AWS: a 34-character tuning job name is rejected with
#   Value '...' at 'hyperParameterTuningJobName' failed to satisfy constraint:
#   Member must have length less than or equal to 32
MAX_TRAINING_JOB_NAME = 63
MAX_TUNING_JOB_NAME = 32


def unique_name(prefix, max_length=MAX_TRAINING_JOB_NAME):
    """Build a collision-free job name that fits the resource's length limit.

    The backend rejects duplicate job names per account with ``ResourceInUse``,
    and these tests run in parallel across many xdist workers, so the name must
    be unique per invocation rather than per test function. Includes randomness
    as well as a timestamp because two xdist workers can enter the same second.

    The uniqueness suffix is preserved and the *prefix* is truncated, so a long
    descriptive prefix degrades readability rather than silently reintroducing
    collisions. Pass ``max_length=MAX_TUNING_JOB_NAME`` for tuning jobs, whose
    limit is roughly half that of training jobs.
    """
    suffix = f"{int(time.time())}-{random.randint(1000, 9999)}"
    # Budget: total, minus the suffix, minus the joining hyphen.
    head = prefix[: max_length - len(suffix) - 1]
    name = f"{head}-{suffix}"
    assert len(name) <= max_length, f"generated name {name!r} exceeds {max_length} chars"
    return name


def stop_quietly(training_job):
    """Stop a submitted job, tolerating races with its own lifecycle.

    Best-effort by design. A job that finished, failed or is already stopping
    cannot be stopped again, and a test must not fail because teardown lost a
    race with the service. Anything genuinely unexpected is logged loudly so it
    stays visible without turning into a spurious test failure.
    """
    if training_job is None:
        return

    name = _first_attr(training_job, _NAME_ATTRS)
    try:
        training_job.stop()
        logger.info("Stopped job %s", name)
    except ClientError as e:
        code = e.response["Error"]["Code"]
        message = e.response["Error"].get("Message", "")
        # ValidationException is what the service returns when the job has
        # already reached a state from which it cannot be stopped.
        if code in ("ValidationException", "ResourceNotFound"):
            logger.info("Job %s no longer stoppable (%s): %s", name, code, message)
            return
        logger.warning("Unexpected error stopping job %s (%s): %s", name, code, message)
    except Exception as e:  # pragma: no cover - defensive teardown
        logger.warning("Unexpected error stopping job %s: %s", name, e)


# Attributes under which the different job resources expose their status. As
# with the ARN, the SDK is not consistent: a TrainingJob uses
# ``training_job_status``, an AgentRFTJob ``job_status`` and a
# HyperParameterTuningJob ``hyper_parameter_tuning_job_status``. Read whichever
# is present.
_STATUS_ATTRS = (
    "training_job_status",
    "job_status",
    "hyper_parameter_tuning_job_status",
)


def _wait_until_terminal(training_job):
    """Block until ``training_job`` leaves the concurrency-quota count.

    The service counts a job against the concurrency quota until it reaches a
    terminal state, not until ``stop()`` returns, so the slot has to be held for
    this whole interval (see the note on ``DEFAULT_MAX_CONCURRENT_JOBS``). This
    is what makes ``SHALLOW_MAX_CONCURRENT_JOBS`` an actual bound on what the
    service sees rather than on how fast slots recycle.

    Best-effort, like ``stop_quietly``: it refreshes and polls the job's status,
    and on timeout or any error it logs and returns so the slot is released
    anyway. A stuck job should slow the suite, not hang it or fail a test that
    already made its assertion. Jobs that expose no readable status (or none of
    the refresh/status plumbing) fall through immediately -- the small quota
    risk there is bounded by the cap itself.
    """
    if training_job is None:
        return

    name = _first_attr(training_job, _NAME_ATTRS)
    refresh = getattr(training_job, "refresh", None)
    deadline = time.time() + _DRAIN_WAIT_TIMEOUT
    while True:
        try:
            if callable(refresh):
                refresh()
            status = _first_attr(training_job, _STATUS_ATTRS)
        except Exception as e:  # pragma: no cover - defensive polling
            logger.info("Could not read status for job %s (%s); releasing slot", name, e)
            return

        if status is None:
            # Nothing to poll on; do not hold a slot forever waiting for a field
            # this job type never exposes.
            logger.info("Job %s exposes no status; releasing slot", name)
            return
        if status in _TERMINAL_STATUSES:
            logger.info("Job %s reached %s; releasing slot", name, status)
            return

        if time.time() > deadline:
            logger.warning(
                "Job %s still %s after %ds; releasing slot anyway "
                "(it may still count against the quota briefly).",
                name,
                status,
                _DRAIN_WAIT_TIMEOUT,
            )
            return

        time.sleep(_DRAIN_POLL_INTERVAL)


# Attributes under which the different job resources expose their ARN and name.
# Not every trainer in this package creates a TrainingJob: MultiTurnRLTrainer
# creates an AgentRFT Job (``job_arn``) and Tuner creates a
# HyperParameterTuningJob, so the harness reads whichever is present rather than
# assuming the TrainingJob shape.
_ARN_ATTRS = (
    "training_job_arn",
    "job_arn",
    "hyper_parameter_tuning_job_arn",
)
_NAME_ATTRS = (
    "training_job_name",
    "job_name",
    "hyper_parameter_tuning_job_name",
)


def _first_attr(obj, attrs):
    """Return the first non-None attribute value from ``attrs``."""
    for attr in attrs:
        value = getattr(obj, attr, None)
        if value is not None:
            return value
    return None


def assert_submitted(job, expected_name=None, resource="training-job"):
    """Assert the service accepted the request and handed back a real ARN.

    This is the single assertion that gives these tests their value, so it checks
    the ARN's shape rather than merely its presence -- a truthy-but-malformed
    value would otherwise pass silently.

    ``resource`` is the expected ARN resource segment. It defaults to
    ``training-job`` because most trainers here create a TrainingJob, but
    MultiTurnRLTrainer creates an AgentRFT ``job`` and Tuner creates a
    ``hyper-parameter-tuning-job``, so those callers pass their own.
    """
    assert job is not None, "train() returned no job; the request was never submitted"

    arn = _first_attr(job, _ARN_ATTRS)
    assert arn, f"job has no ARN: {job!r}"
    assert arn.startswith("arn:"), f"malformed ARN: {arn!r}"
    assert f":{resource}/" in arn, f"ARN is not a {resource} ARN: {arn!r}"

    if expected_name is not None:
        actual = _first_attr(job, _NAME_ATTRS)
        assert (
            actual == expected_name
        ), f"submitted job name {actual!r} does not match requested {expected_name!r}"

    logger.info("Service accepted request; ARN=%s", arn)
    return arn


def _train_kwargs_for(trainer, extra):
    """Build the kwargs for ``trainer.train()``, forcing a non-waiting submit.

    ``wait=False`` is the whole point of this suite: the ARN is returned
    synchronously by ``CreateTrainingJob``, so waiting buys no extra coverage
    and costs a full training run.

    ``logs`` is deliberately conditional. ``ModelTrainer.train`` accepts it, but
    the recipe trainers (``SFTTrainer``, ``DPOTrainer``, ``RLVRTrainer``,
    ``CPTTrainer``, ...) do not -- their signatures are
    ``(training_dataset, validation_dataset, wait, wait_timeout, poll,
    dry_run)``. Passing ``logs`` unconditionally would raise ``TypeError`` for
    the entire recipe-trainer family, so it is introspected rather than assumed.
    """
    kwargs = {"wait": False}
    kwargs.update(extra)

    try:
        parameters = inspect.signature(trainer.train).parameters
    except (TypeError, ValueError):  # pragma: no cover - defensive
        parameters = {}

    # Only silence logs where the trainer understands the option; where it does
    # not, wait=False already prevents log streaming.
    if "logs" in parameters and "logs" not in kwargs:
        kwargs["logs"] = False

    return kwargs


@contextmanager
def submitted(trainer, **train_kwargs):
    """Submit a training job, yield it, and always stop it.

    Usage::

        with submitted(trainer) as job:
            assert_submitted(job)

    Callers must not pass ``wait``: it is forced to ``False`` and a supplied
    value is rejected loudly rather than silently overridden, so a copy-pasted
    ``wait=True`` cannot quietly reintroduce a full training run into the fast
    suite.

    Holds a concurrency slot (see ``job_slots``) until the submitted job reaches
    a terminal state, so the number of jobs the *service* counts against the
    training-job quota across all xdist workers stays inside the cap. Slots are
    taken here rather than in each test so a new test is capped by default
    instead of by remembering to opt in.
    """
    if "wait" in train_kwargs:
        raise TypeError(
            "submitted() controls 'wait'; remove it from the call. "
            "These tests must never wait for a job to run."
        )

    with job_slots(_requested_slots(trainer)):
        training_job = None
        try:
            trainer.train(**_train_kwargs_for(trainer, train_kwargs))
            training_job = _resolve_job(trainer)
            yield training_job
        finally:
            # Stop, then hold the slot until the job is actually terminal. The
            # service counts the job against the concurrency quota until it
            # drains, not until stop() returns, so releasing the slot at stop()
            # would let the next test start while this job still counts -- which
            # is exactly how an earlier version peaked at ~37 jobs against a
            # limit of 20.
            stop_quietly(training_job)
            _wait_until_terminal(training_job)


# Attributes under which trainers stash the job they just submitted. The SDK is
# not consistent here, so the harness checks all of them rather than silently
# yielding None (which would surface as a confusing "train() returned no job"
# failure instead of an attribute-discovery problem):
#   _latest_training_job  -- ModelTrainer and most recipe trainers
#   latest_training_job   -- DPOTrainer (public)
#   _latest_job           -- MultiTurnRLTrainer (AgentRFTJob)
#   latest_tuning_job     -- Tuner (HyperParameterTuningJob)
_JOB_ATTRS = (
    "_latest_training_job",
    "latest_training_job",
    "_latest_job",
    "latest_tuning_job",
)


def _resolve_job(trainer):
    """Return the job resource the trainer just submitted, whatever its type."""
    return _first_attr(trainer, _JOB_ATTRS)


# Where the different trainers keep an explicit compute spec, when they have one.
_COMPUTE_ATTRS = ("compute", "_compute", "compute_config")


def _requested_slots(trainer):
    """Slots the job ``trainer`` is about to submit should consume.

    One slot per concurrent job, plus one per additional instance when the job is
    serverful. See the note on ``DEFAULT_MAX_CONCURRENT_JOBS`` for why the two
    quotas make this the right unit.

    Returns 1 when no explicit compute is set. That is not a fallback but the
    correct answer for the default recipe-trainer path: leaving ``compute=None``
    submits a *serverless* model-customization job, which is bounded by a
    per-Region job count and consumes no instance-type quota at all.

    Falls back to 1 if a compute object exists but exposes no usable count.
    Under-counting is the safe direction to be wrong here: the cap remains a
    useful bound, whereas guessing high would throttle the suite for no reason.
    Tuning jobs are the notable inexact case -- their fan-out is set by the
    tuner's own ``max_parallel_jobs`` rather than a compute block -- and there
    are only two of them, both single-instance.
    """
    for attr in _COMPUTE_ATTRS:
        compute = getattr(trainer, attr, None)
        if compute is None:
            continue
        count = getattr(compute, "instance_count", None)
        if isinstance(count, int) and count > 0:
            return count
    return 1


def assert_rejected(trainer, expected_tokens, **train_kwargs):
    """Assert a request is rejected, and clean up if it is unexpectedly accepted.

    Negative tests are what stop this suite from degenerating into "any ARN is
    fine": without them, a bug that made the SDK send a permissive-but-wrong
    payload would still produce a green suite.

    ``expected_tokens`` is a collection of substrings, any one of which is
    accepted. Matching is deliberately loose because a rejection can legitimately
    surface from three different layers with different wording -- SDK-side
    validation (``ValueError``), the public API model
    (``ValidationException``), or the training backend (``ValidationError``) --
    and pinning exact prose would make these tests fail on harmless message
    changes. It is still specific enough to catch a *wrong* rejection, which is
    the real risk: without it, a test could pass because of an unrelated
    credentials or region error.

    If the request is unexpectedly accepted, the job is stopped before the test
    fails, so a validation regression cannot leak a running job.
    """
    if "wait" in train_kwargs:
        raise TypeError("assert_rejected() controls 'wait'; remove it from the call.")

    # Slot-guarded too: a negative test is expected *not* to consume capacity,
    # but if a validation regression let the request through it would, and that
    # is exactly the case where staying inside the quota matters.
    with job_slots(_requested_slots(trainer)):
        training_job = None
        try:
            with pytest.raises(Exception) as excinfo:
                trainer.train(**_train_kwargs_for(trainer, train_kwargs))
                # Reached only if the service accepted a request we expected it to
                # refuse. Capture the job so the finally-block can stop it, then let
                # pytest.raises report the missing exception.
                training_job = _resolve_job(trainer)
        finally:
            # Normally a no-op (the request was rejected, so no job exists). If a
            # regression let it through, drain it inside the slot for the same
            # reason submitted() does.
            stop_quietly(training_job)
            _wait_until_terminal(training_job)

    message = str(excinfo.value)
    assert any(token in message for token in expected_tokens), (
        f"request was rejected, but not for the expected reason.\n"
        f"  expected one of: {sorted(expected_tokens)}\n"
        f"  actual: {message}"
    )
    return message
