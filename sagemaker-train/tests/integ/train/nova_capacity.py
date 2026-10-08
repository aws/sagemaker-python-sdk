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
"""Helpers for deep integ tests that fine-tune Nova models on serverless capacity.

Nova serverless fine-tuning jobs (e.g. ``nova-textgeneration-lite-v2`` SFT/RLVR) run
on a shared, service-owned p5 pool that customers draw from as well, and each job
takes 4 p5 instances. When several of these tests run at once (pytest-xdist runs the
deep suite in parallel) they can occupy most of that pool and leave no room for a
customer job. These helpers keep the tests' footprint to one job at a time:

* ``nova_capacity_slot`` serializes Nova serverless jobs across all xdist workers of a
  test run. Hold it from submission until the job is terminal.
* ``wait_for_nova_training_job`` waits for the job with a bounded wait for capacity:
  if the job is still waiting for instances after ``pending_timeout`` it is stopped
  and the test is skipped. It also stops the job whenever the wait ends with the job
  still running (timeout, error, interrupt), so no orphaned job keeps queueing for
  capacity after the test is gone.
"""

from __future__ import absolute_import

import logging
import os
import tempfile
import time
from contextlib import contextmanager

import pytest

from .. import lock

logger = logging.getLogger(__name__)

# The lock file must be shared by every xdist worker of a run. fcntl locks are released
# by the OS when the holding process dies, so a killed run never leaves a stale lock.
NOVA_CAPACITY_LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_nova_capacity_lock")

# In CI, Nova serverless jobs that eventually ran waited a few minutes for instances
# (median 3-6 min, max ~100 min while competing with each other). Once the tests are
# serialized, a job still waiting after an hour means the shared pool is full, so we
# give the capacity back instead of queueing behind customers.
NOVA_PENDING_TIMEOUT_SECONDS = 3600

TERMINAL_STATUSES = ("Completed", "Failed", "Stopped")
_WAITING_FOR_CAPACITY_STATUSES = ("Starting", "Pending")


@contextmanager
def nova_capacity_slot():
    """Allow at most one Nova serverless training job per test run at a time."""
    start = time.time()
    logger.info("Waiting for the Nova capacity slot (%s)", NOVA_CAPACITY_LOCK_PATH)
    with lock.lock(NOVA_CAPACITY_LOCK_PATH):
        logger.info("Acquired the Nova capacity slot after %ds", int(time.time() - start))
        yield


def _stop_quietly(training_job):
    """Best-effort stop of a training job that is not terminal yet."""
    name = getattr(training_job, "training_job_name", None)
    try:
        logger.warning("Stopping non-terminal training job %s", name)
        training_job.stop()
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to stop training job %s: %s", name, e)


def wait_for_nova_training_job(
    training_job,
    max_wait_time,
    poll_interval=30,
    pending_timeout=NOVA_PENDING_TIMEOUT_SECONDS,
):
    """Poll a training job until it is terminal, with a bounded wait for capacity.

    Args:
        training_job: The ``sagemaker.core.resources.TrainingJob`` returned by
            ``train(wait=False)``.
        max_wait_time (int): Maximum total seconds to wait for a terminal status.
        poll_interval (int): Seconds between status checks.
        pending_timeout (int): Seconds after which a job that is still waiting for
            instances (secondary status ``Starting``/``Pending``) is stopped and the
            test is skipped.

    Returns:
        str: The last observed ``training_job_status``. It is terminal unless
        ``max_wait_time`` ran out, in which case the job has been stopped.
    """
    start = time.time()
    status = None
    try:
        while time.time() - start < max_wait_time:
            training_job.refresh()
            status = training_job.training_job_status
            if status in TERMINAL_STATUSES:
                return status
            elapsed = time.time() - start
            if (
                training_job.secondary_status in _WAITING_FOR_CAPACITY_STATUSES
                and elapsed > pending_timeout
            ):
                pytest.skip(
                    f"Nova capacity unavailable: training job "
                    f"{training_job.training_job_name} was still waiting for instances "
                    f"after {int(elapsed)}s; stopped it to release the shared pool."
                )
            time.sleep(poll_interval)
        logger.warning(
            "Training job %s not terminal after %ds (status: %s)",
            training_job.training_job_name,
            max_wait_time,
            status,
        )
        return status
    finally:
        if status not in TERMINAL_STATUSES:
            _stop_quietly(training_job)
