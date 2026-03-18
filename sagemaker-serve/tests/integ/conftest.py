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
"""Shared fixtures for integration tests.

Provides a "get or create" Nova training job that is reused across all
integ test modules in the session.
"""
from __future__ import absolute_import

import json
import logging
import time

import boto3
import pytest

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

NOVA_REGION = "us-east-1"
NOVA_TRAINING_JOB_NAME = "sdk-integ-nova-micro-sft"
NOVA_MODEL_ID = "nova-textgeneration-micro"
NOVA_BUCKET_PREFIX = "sagemaker-us-east-1"


def _get_or_create_nova_training_job():
    """Return a completed Nova training job, creating one if it doesn't exist.

    Uses a fixed job name so the job is created once and reused across runs.
    """
    sm = boto3.client("sagemaker", region_name=NOVA_REGION)

    # ── Check if the job already exists ─────────────────────────────────
    try:
        resp = sm.describe_training_job(TrainingJobName=NOVA_TRAINING_JOB_NAME)
        status = resp["TrainingJobStatus"]
        logger.info("Found existing training job %s (status=%s)", NOVA_TRAINING_JOB_NAME, status)

        if status == "Completed":
            return NOVA_TRAINING_JOB_NAME
        if status == "InProgress":
            logger.info("Training job in progress, waiting for completion...")
            _wait_for_training_job(sm, NOVA_TRAINING_JOB_NAME)
            return NOVA_TRAINING_JOB_NAME
        if status in ("Failed", "Stopped"):
            logger.warning(
                "Training job %s has status %s — will create a new one with timestamp suffix",
                NOVA_TRAINING_JOB_NAME, status,
            )
            # Fall through to create a new one with a unique name
            job_name = f"{NOVA_TRAINING_JOB_NAME}-{int(time.time())}"
        else:
            return NOVA_TRAINING_JOB_NAME
    except sm.exceptions.ClientError as e:
        if "Requested resource not found" in str(e):
            logger.info("Training job %s not found, creating...", NOVA_TRAINING_JOB_NAME)
            job_name = NOVA_TRAINING_JOB_NAME
        else:
            raise

    # ── Upload minimal training data ────────────────────────────────────
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    bucket = f"{NOVA_BUCKET_PREFIX}-{account_id}"
    s3 = boto3.client("s3", region_name=NOVA_REGION)

    try:
        s3.head_bucket(Bucket=bucket)
    except Exception:
        s3.create_bucket(Bucket=bucket)

    train_key = "integ-test-data/nova-sft-train.jsonl"
    train_uri = f"s3://{bucket}/{train_key}"

    # Only upload if not already there
    try:
        s3.head_object(Bucket=bucket, Key=train_key)
    except Exception:
        rows = []
        for i in range(20):
            rows.append(json.dumps({
                "messages": [
                    {"role": "user", "content": f"What is {i+1} + {i+1}?"},
                    {"role": "assistant", "content": f"The answer is {(i+1)*2}."},
                ]
            }))
        s3.put_object(Bucket=bucket, Key=train_key, Body="\n".join(rows).encode())
        logger.info("Uploaded training data to %s", train_uri)

    # ── Launch training job via SFTTrainer ──────────────────────────────
    import os
    original_region = os.environ.get("AWS_DEFAULT_REGION")
    os.environ["AWS_DEFAULT_REGION"] = NOVA_REGION
    try:
        from sagemaker.train.sft_trainer import SFTTrainer

        trainer = SFTTrainer(
            model=NOVA_MODEL_ID,
            training_dataset=train_uri,
            accept_eula=True,
            model_package_group="sdk-integ-nova-models",
        )
        trainer.train(wait=False)
        actual_name = trainer._latest_training_job.training_job_name
        logger.info("Started training job: %s", actual_name)
    finally:
        if original_region:
            os.environ["AWS_DEFAULT_REGION"] = original_region
        else:
            os.environ.pop("AWS_DEFAULT_REGION", None)

    _wait_for_training_job(sm, actual_name)
    return actual_name


def _wait_for_training_job(sm_client, job_name, poll_interval=30, max_wait=7200):
    """Poll until training job completes or fails."""
    elapsed = 0
    while elapsed < max_wait:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        logger.info("Training job %s status: %s (elapsed %ds)", job_name, status, elapsed)
        if status == "Completed":
            return
        if status in ("Failed", "Stopped"):
            reason = resp.get("FailureReason", "unknown")
            raise RuntimeError(
                f"Training job {job_name} ended with status {status}: {reason}"
            )
        time.sleep(poll_interval)
        elapsed += poll_interval
    raise RuntimeError(f"Timed out after {max_wait}s waiting for training job {job_name}")


# ── Session-scoped fixtures ─────────────────────────────────────────────────


@pytest.fixture(scope="session")
def nova_training_job_name():
    """Get or create a completed Nova training job. Reused across all tests."""
    return _get_or_create_nova_training_job()
