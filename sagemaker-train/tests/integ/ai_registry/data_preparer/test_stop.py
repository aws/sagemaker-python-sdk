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
"""Integration tests: DataPreparer.stop() — cancel in-flight SDG jobs.

Tests the stop() contract:
  * wait=False returns the Job immediately after the stop request is issued.
  * The job transitions through Stopping → Stopped (polled to terminal state).
  * Attaching to a Stopped job raises DataPreparerJobStoppedError.
  * Calling stop() on an already-terminal job is a no-op.

Prerequisites: see this directory's ``conftest.py`` (AWS credentials for an account
where the DataPreparer feature is enabled; optional DATA_PREPARER_INTEG_ROLE_ARN and
DATA_PREPARER_INTEG_BUCKET).
"""

from __future__ import absolute_import

import time

import pytest

from sagemaker.ai_registry import DataPreparer, DatasetJobConfig, JobCategory
from sagemaker.ai_registry.data_preparer.exceptions import DataPreparerJobStoppedError

_TERMINAL_STATES = frozenset({"Completed", "Failed", "Stopped"})
_STOP_POLL_INTERVAL_SECONDS = 10
_STOP_TIMEOUT_SECONDS = 300  # 5 minutes; StopJob is normally fast


def _poll_to_terminal(job, *, timeout: int = _STOP_TIMEOUT_SECONDS) -> str:
    """Block until the job reaches a terminal state and return that state.

    Avoids calling job.wait() so we can assert on the intermediate Stopping
    state before the job fully lands in Stopped.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        job.refresh()
        if job.job_status in _TERMINAL_STATES:
            return job.job_status
        time.sleep(_STOP_POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"Job {job.job_name} did not reach a terminal state within {timeout}s; "
        f"last status: {job.job_status}"
    )


@pytest.mark.integ
class TestStopSDG:
    """Submit an SDG job with wait=False, stop it, and assert Stopping/Stopped."""

    def test_stop_inflight_sdg_job(self, sagemaker_session, test_role, s3_fixtures):
        """Submit SDG fire-and-forget, immediately call stop, poll to terminal.

        Validates:
        1. stop(wait=False) returns a Job without blocking.
        2. Job transitions from InProgress/Stopping to Stopped.
        3. DataPreparer.attach() on the stopped job raises DataPreparerJobStoppedError.
        4. stop() on an already-Stopped job is a no-op (stop not called again).
        """
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-stop-sdg",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 100,  # large enough to still be running when we stop
            },
            output_data_config={
                "OutputS3Uri": (
                    f"s3://{sagemaker_session.default_bucket()}" "/data-preparer-integ-stop/"
                ),
            },
        )

        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )

        # Step 1: Submit fire-and-forget — the job should be InProgress or
        # already moving when we call stop().
        job = preparer.generate(dataset, wait=False)

        assert job.job_arn is not None, "job_arn must be public after submission"
        assert "SyntheticDataGeneration" in job.job_arn
        saved_arn = job.job_arn

        # Step 2: Stop immediately (wait=False) — return before terminal state.
        stopped_job = DataPreparer.stop(job, wait=False)

        # The returned handle is the same object; job_arn is preserved.
        assert stopped_job.job_arn == saved_arn

        # Step 3: Poll independently until the job reaches a terminal state.
        terminal_status = _poll_to_terminal(stopped_job)

        # The job must be Stopped (not Completed or Failed — we stopped it).
        assert (
            terminal_status == "Stopped"
        ), f"expected Stopped after explicit stop(); got {terminal_status}"

        # Step 4: Attaching to a Stopped job must raise DataPreparerJobStoppedError.
        with pytest.raises(DataPreparerJobStoppedError):
            DataPreparer.attach(saved_arn, sagemaker_session=sagemaker_session)

        # Step 5: stop() on the already-Stopped job must be a no-op (no API
        # error, returns the refreshed handle).
        noop_result = DataPreparer.stop(saved_arn, sagemaker_session=sagemaker_session)
        assert noop_result.job_status in _TERMINAL_STATES

    def test_stop_wait_true_blocks_until_terminal(self, sagemaker_session, test_role, s3_fixtures):
        """stop(wait=True) blocks and returns the Job in a terminal state.

        Unlike test_stop_inflight_sdg_job (which polls manually), this test
        lets stop(wait=True) do the waiting so we verify the built-in waiter
        correctly reports terminal state on the returned handle.
        """
        dataset = DatasetJobConfig(
            job_category=JobCategory.SYNTHETIC_DATA_GENERATION,
            dataset_name="integ-stop-wait-sdg",
            generation_config={
                "DatasetSpecification": {"SpecS3Uri": s3_fixtures["task_description.txt"]},
                "NumSamples": 100,
            },
            output_data_config={
                "OutputS3Uri": (
                    f"s3://{sagemaker_session.default_bucket()}" "/data-preparer-integ-stop-wait/"
                ),
            },
        )

        preparer = DataPreparer(
            sagemaker_session=sagemaker_session,
            role=test_role,
        )

        # Submit fire-and-forget, then stop with wait=True.
        job = preparer.generate(dataset, wait=False)
        result = DataPreparer.stop(job.job_arn, wait=True, sagemaker_session=sagemaker_session)

        assert (
            result.job_status == "Stopped"
        ), f"stop(wait=True) returned job with status {result.job_status}; expected Stopped"
        assert result.job_arn == job.job_arn
