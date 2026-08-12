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

import inspect
import logging
import random
import time
from contextlib import contextmanager

import pytest
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

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
    """
    if "wait" in train_kwargs:
        raise TypeError(
            "submitted() controls 'wait'; remove it from the call. "
            "These tests must never wait for a job to run."
        )

    training_job = None
    try:
        trainer.train(**_train_kwargs_for(trainer, train_kwargs))
        training_job = _resolve_job(trainer)
        yield training_job
    finally:
        stop_quietly(training_job)


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

    training_job = None
    try:
        with pytest.raises(Exception) as excinfo:
            trainer.train(**_train_kwargs_for(trainer, train_kwargs))
            # Reached only if the service accepted a request we expected it to
            # refuse. Capture the job so the finally-block can stop it, then let
            # pytest.raises report the missing exception.
            training_job = _resolve_job(trainer)
    finally:
        stop_quietly(training_job)

    message = str(excinfo.value)
    assert any(token in message for token in expected_tokens), (
        f"request was rejected, but not for the expected reason.\n"
        f"  expected one of: {sorted(expected_tokens)}\n"
        f"  actual: {message}"
    )
    return message
