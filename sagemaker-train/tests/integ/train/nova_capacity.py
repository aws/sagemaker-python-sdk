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
* ``wait_for_nova_training_job`` waits for the job with a bounded wait for capacity
  (see ``tests/integ/capacity.py``): if the job is still waiting for instances after
  ``pending_timeout`` it is stopped and the test is skipped, and the job is stopped
  whenever the wait ends with it still running.
"""

from __future__ import absolute_import

import logging
import os
import tempfile
import time
from contextlib import contextmanager

from .. import lock
from ..capacity import wait_for_training_job_with_capacity_timeout

logger = logging.getLogger(__name__)

# The lock file must be shared by every xdist worker of a run. fcntl locks are released
# by the OS when the holding process dies, so a killed run never leaves a stale lock.
NOVA_CAPACITY_LOCK_PATH = os.path.join(tempfile.gettempdir(), "sagemaker_nova_capacity_lock")

# In CI, Nova serverless jobs that eventually ran waited a few minutes for instances
# (median 3-6 min, max ~100 min while competing with each other). Once the tests are
# serialized, a job still waiting after an hour means the shared pool is full, so we
# give the capacity back instead of queueing behind customers.
NOVA_PENDING_TIMEOUT_SECONDS = 3600


@contextmanager
def nova_capacity_slot():
    """Allow at most one Nova serverless training job per test run at a time."""
    start = time.time()
    logger.info("Waiting for the Nova capacity slot (%s)", NOVA_CAPACITY_LOCK_PATH)
    with lock.lock(NOVA_CAPACITY_LOCK_PATH):
        logger.info("Acquired the Nova capacity slot after %ds", int(time.time() - start))
        yield


def wait_for_nova_training_job(
    training_job,
    max_wait_time,
    poll_interval=30,
    pending_timeout=NOVA_PENDING_TIMEOUT_SECONDS,
):
    """Poll a Nova training job until it is terminal, with a bounded wait for capacity.

    Thin wrapper over ``wait_for_training_job_with_capacity_timeout``; see that
    function for the arguments and return value.
    """
    return wait_for_training_job_with_capacity_timeout(
        training_job,
        max_wait_time=max_wait_time,
        poll_interval=poll_interval,
        pending_timeout=pending_timeout,
        capacity_label="Nova",
    )
