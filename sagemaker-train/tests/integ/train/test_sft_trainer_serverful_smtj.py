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
"""Integration test for SFT fine-tuning with serverful Training Jobs (SMTJ).

Based on: v3-examples/model-customization-examples/sft_finetuning_serverful_smtj.ipynb

Prerequisites:
    - Active AWS credentials in us-east-1 with permissions to launch SageMaker
      Training Jobs and access the default SageMaker bucket.

The test is self-sufficient: it uses the account's default SageMaker bucket
(``sagemaker-{region}-{account_id}``, auto-created by ``Session.default_bucket()``).
The sample training data in ``tests/data/train/sft_smtj_sample_data.jsonl`` is
uploaded under the ``sft-smtj-integ/`` prefix if not already present.

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_sft_trainer_serverful_smtj.py -v -s
"""
from __future__ import absolute_import

import logging
import os
import time
import random

import pytest

from sagemaker.train import SFTTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.training.configs import TrainingJobCompute

logger = logging.getLogger(__name__)

# Test configuration
DATA_PREFIX = "sft-smtj-integ"
DATA_S3_KEY = f"{DATA_PREFIX}/sft_smtj_sample_data.jsonl"

# Local sample data file
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "train")
LOCAL_TRAINING_DATA = os.path.join(DATA_DIR, "sft_smtj_sample_data.jsonl")


def _ensure_training_data(s3_client, bucket_name: str) -> str:
    """Upload local training data to S3 if not already present.

    Returns the S3 URI for the training data file.
    """
    s3_uri = f"s3://{bucket_name}/{DATA_S3_KEY}"

    # Check if already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=DATA_S3_KEY)
        logger.info(f"Training data already exists at {s3_uri}")
        return s3_uri
    except s3_client.exceptions.ClientError:
        pass

    logger.info(f"Uploading training data from {LOCAL_TRAINING_DATA} to {s3_uri}")
    s3_client.upload_file(LOCAL_TRAINING_DATA, bucket_name, DATA_S3_KEY)
    logger.info("Training data uploaded successfully.")
    return s3_uri


@pytest.fixture(scope="module")
def training_resources(sagemaker_session_us_east_1):
    """Ensure training data exists in the account's default SageMaker bucket.

    Uses ``Session.default_bucket()`` (creates ``sagemaker-{region}-{account_id}``
    if needed), then uploads the local sample data file under the
    ``sft-smtj-integ/`` prefix if not already present.

    Returns a dict with:
        - training_dataset: S3 URI to the training data file
        - s3_output_path: S3 URI for training output
        - bucket_name: the default bucket name
    """
    bucket_name = sagemaker_session_us_east_1.default_bucket()
    s3_client = sagemaker_session_us_east_1.boto_session.client("s3")
    training_dataset = _ensure_training_data(s3_client, bucket_name)

    return {
        "training_dataset": training_dataset,
        "s3_output_path": f"s3://{bucket_name}/{DATA_PREFIX}/output/",
        "bucket_name": bucket_name,
    }


@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_sft_trainer_serverful_smtj(sagemaker_session_us_east_1, training_resources):
    """Test SFT fine-tuning on serverful compute with recipe overrides.

    Exercises the full notebook flow:
    1. Create SFTTrainer with TrainingJobCompute (dedicated instances)
    2. Apply recipe overrides and verify via get_resolved_recipe()
    3. Set hyperparameters and submit training job
    4. Poll until completion
    """
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    sft_trainer = SFTTrainer(
        model="amazon.nova-micro-v1",
        training_type=TrainingType.LORA,
        training_dataset=training_resources["training_dataset"],
        s3_output_path=training_resources["s3_output_path"],
        compute=TrainingJobCompute(
            instance_type="ml.g6.48xlarge",
            instance_count=1,
        ),
        sagemaker_session=sagemaker_session_us_east_1,
        overrides={"training_config": {"max_epochs": 1}},
        base_job_name=f"sft-smtj-integ-{unique_id}",
    )

    # Verify recipe overrides are resolved correctly
    resolved = sft_trainer.get_resolved_recipe()
    training_config = resolved["training_config"]
    logger.info(f"Resolved training_config: {training_config}")

    # Nova Micro uses trainer.max_epochs for step control
    assert training_config["trainer"]["max_epochs"] == 1, (
        f"Expected max_epochs=1, got: {training_config.get('trainer')}"
    )

    # Submit (non-blocking)
    training_job = sft_trainer.train(wait=False)
    assert training_job is not None
    assert training_job.training_job_name is not None
    logger.info(f"Training job submitted: {training_job.training_job_name}")

    # Poll for completion
    max_wait_time = 10800  # 3 hour timeout (Nova training can be slow)
    poll_interval = 60
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status

        if status in ["Completed", "Failed", "Stopped"]:
            break

        logger.info(
            f"Job {training_job.training_job_name} status: {status} "
            f"({int(time.time() - start_time)}s elapsed)"
        )
        time.sleep(poll_interval)

    assert training_job.training_job_status == "Completed", (
        f"Training job {training_job.training_job_name} ended with status: "
        f"{training_job.training_job_status}"
    )
    logger.info(f"Training job completed successfully: {training_job.training_job_name}")
