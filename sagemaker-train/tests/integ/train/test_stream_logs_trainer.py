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
"""Integration tests for trainer stream_logs()"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import boto3
import pytest

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.resources import TrainingJob
from sagemaker.train.agent_rft_job import AgentRFTJob
from sagemaker.train.sft_trainer import SFTTrainer

logger = logging.getLogger(__name__)

REGION = "us-west-2"
MTRL_JOB_NAME = "mock-oss-test-mtrl-20260729120959"
SERVERFUL_JOB_NAME = "pytorch-training-260729-1927-002-95b83cb6"



@pytest.fixture(scope="module")
def sagemaker_session():
    boto_session = boto3.Session(region_name=REGION)
    return Session(boto_session=boto_session)



class TestMTRLStreamLogs:
    """Verify AgentRFTJob.stream_logs() on a completed MTRL job."""

    def test_stream_logs_exits_on_completed(self, sagemaker_session):
        """AgentRFTJob.stream_logs() exits quickly for a completed job."""
        job = AgentRFTJob.get(MTRL_JOB_NAME, session=sagemaker_session.boto_session)
        assert job.job_status == "Completed"

        start = time.time()
        job.stream_logs(poll=2)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"stream_logs() took {elapsed:.1f}s — should exit quickly for completed job"
        )
        print(f"✓ AgentRFTJob.stream_logs() completed in {elapsed:.1f}s")

    def test_stream_logs_with_start_time(self, sagemaker_session):
        """AgentRFTJob.stream_logs() respects start_time parameter."""
        job = AgentRFTJob.get(MTRL_JOB_NAME, session=sagemaker_session.boto_session)

        # Use a timestamp from when the job was running (extracted from job name)
        
        job_start = datetime(2026, 7, 29, 12, 9, 59, tzinfo=timezone.utc)
        start_time_ms = int(job_start.timestamp() * 1000)

        start = time.time()
        job.stream_logs(poll=2, start_time=start_time_ms)
        elapsed = time.time() - start

        assert elapsed < 30
        print(f"✓ stream_logs(start_time=job_start) completed in {elapsed:.1f}s")



class TestServerfulSMTJStreamLogs:
    """Verify BaseTrainer.stream_logs() on a completed serverful training job."""

    def test_stream_logs_exits_on_completed(self, sagemaker_session):
        """stream_logs() exits quickly for a completed serverful job."""
        tj = TrainingJob.get(
            training_job_name=SERVERFUL_JOB_NAME,
            session=sagemaker_session.boto_session,
        )
        assert tj.training_job_status == "Completed"

        trainer = SFTTrainer.__new__(SFTTrainer)
        trainer._latest_training_job = tj
        trainer.sagemaker_session = sagemaker_session
        trainer.compute = None

        start = time.time()
        trainer.stream_logs(poll=2)
        elapsed = time.time() - start

        assert elapsed < 30, (
            f"stream_logs() took {elapsed:.1f}s — should exit quickly for completed job"
        )
        print(f"✓ Serverful SMTJ stream_logs() completed in {elapsed:.1f}s")



class TestStreamLogsValidation:
    """Verify input validation for stream_logs() parameters."""

    def test_poll_validation(self, sagemaker_session):
        """stream_logs() rejects invalid poll values."""
        job = AgentRFTJob.get(MTRL_JOB_NAME, session=sagemaker_session.boto_session)

        with pytest.raises(ValueError, match="poll must be between"):
            job.stream_logs(poll=0)

        with pytest.raises(ValueError, match="poll must be between"):
            job.stream_logs(poll=400)

    def test_start_time_type_validation(self, sagemaker_session):
        """stream_logs() rejects invalid start_time types."""
        job = AgentRFTJob.get(MTRL_JOB_NAME, session=sagemaker_session.boto_session)

        with pytest.raises(TypeError, match="start_time must be datetime or int"):
            job.stream_logs(start_time="2025-01-01")
