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
"""Helpers for integ tests that submit real training jobs on scarce instances.

When SageMaker has no capacity for the requested instance type, a training job
does not fail: it stays in secondary status ``Pending`` ("Training job waiting
for capacity") indefinitely. ``MaxRuntimeInSeconds`` only counts training time,
so it does not bound that wait, and ``TrainingJob.wait()`` has no timeout. A
test that calls ``train()`` with the default ``wait=True`` therefore blocks
until the CodeBuild build times out, and the queued job is left behind holding
the account's instance quota after the build is gone.

``wait_for_training_job_with_capacity_timeout`` waits with a bounded wait for
capacity instead: if the job is still waiting for instances after
``pending_timeout`` it is stopped and the test is skipped. It also stops the job
whenever the wait ends with the job still running (timeout, error, interrupt),
so no orphaned job keeps queueing for capacity after the test is gone.
"""

from __future__ import absolute_import

import logging
import time

import pytest

logger = logging.getLogger(__name__)

TERMINAL_STATUSES = ("Completed", "Failed", "Stopped")
_WAITING_FOR_CAPACITY_STATUSES = ("Starting", "Pending")

DEFAULT_PENDING_TIMEOUT_SECONDS = 3600


def stop_training_job_quietly(training_job):
    """Best-effort stop of a training job that is not terminal yet."""
    name = getattr(training_job, "training_job_name", None)
    try:
        logger.warning("Stopping non-terminal training job %s", name)
        training_job.stop()
    except Exception as e:  # pylint: disable=broad-except
        logger.warning("Failed to stop training job %s: %s", name, e)


def wait_for_training_job_with_capacity_timeout(
    training_job,
    max_wait_time,
    poll_interval=30,
    pending_timeout=DEFAULT_PENDING_TIMEOUT_SECONDS,
    capacity_label="Training",
):
    """Poll a training job until it is terminal, with a bounded wait for capacity.

    Args:
        training_job: The ``sagemaker.core.resources.TrainingJob`` that was
            submitted without waiting (``train(wait=False)``).
        max_wait_time (int): Maximum total seconds to wait for a terminal status.
        poll_interval (int): Seconds between status checks.
        pending_timeout (int): Seconds after which a job that is still waiting for
            instances (secondary status ``Starting``/``Pending``) is stopped and the
            test is skipped.
        capacity_label (str): Prefix for the skip message, naming the capacity
            that was unavailable (e.g. ``"Nova"`` or ``"ml.g5.2xlarge"``).

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
                    f"{capacity_label} capacity unavailable: training job "
                    f"{training_job.training_job_name} was still waiting for instances "
                    f"after {int(elapsed)}s; stopped it to release the capacity."
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
            stop_training_job_quietly(training_job)
