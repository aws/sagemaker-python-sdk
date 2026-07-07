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
"""Integration tests for SFT trainer on HyperPod (Nova Micro)"""
from __future__ import absolute_import

import time
import random
import pytest
import boto3
import logging

from sagemaker.train import SFTTrainer
from sagemaker.train.common import TrainingType
from sagemaker.core.training.configs import HyperPodCompute
from sagemaker.core.shapes import StoppingCondition

logger = logging.getLogger(__name__)

S3_BUCKET = "mc-flows-sdk-testing-us-east-1-784379639078"
S3_PREFIX = "input_data"
SFT_TRAIN_S3_KEY = f"{S3_PREFIX}/sft-nova/sft_200_samples.jsonl"
SFT_TRAIN_S3_PATH = f"s3://{S3_BUCKET}/{SFT_TRAIN_S3_KEY}"
S3_OUTPUT_PATH = f"s3://{S3_BUCKET}/output"


@pytest.fixture(scope="module")
def region():
    """Resolve the AWS region for this test module."""
    return "us-east-1"


@pytest.fixture(scope="module")
def s3_client(region):
    """Create an S3 client in the correct region."""
    return boto3.client("s3", region_name=region)


@pytest.fixture(scope="module")
def verified_training_dataset(s3_client):
    """Verify the training dataset exists in S3 and return its path."""
    try:
        bucket_region = s3_client.get_bucket_location(Bucket=S3_BUCKET)[
            "LocationConstraint"
        ] or "us-east-1"
        s3_regional_client = boto3.client("s3", region_name=bucket_region)
        s3_regional_client.head_object(Bucket=S3_BUCKET, Key=SFT_TRAIN_S3_KEY)
    except s3_client.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            pytest.fail(
                f"Training file not found in S3: {SFT_TRAIN_S3_PATH}"
            )
        else:
            raise
    return SFT_TRAIN_S3_PATH


@pytest.fixture(scope="module")
def hyperpod_compute():
    """Create HyperPod compute configuration."""
    return HyperPodCompute(
        cluster_name="pysdk-hp-integ-tests",
        namespace="kubeflow",
        instance_type="ml.g6.12xlarge",
        node_count=1,
    )


@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_sft_trainer_nova_micro_hyperpod_lora(
    verified_training_dataset, hyperpod_compute
):
    """Test SFT training workflow with Nova Micro on HyperPod using LORA."""
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    stopping_condition = StoppingCondition(max_runtime_in_seconds=20800)

    sft_trainer = SFTTrainer(
        model="nova-textgeneration-micro",
        training_type=TrainingType.LORA,
        compute=hyperpod_compute,
        base_job_name=f"hp-micro-sft-integ-{unique_id}",
        training_dataset=verified_training_dataset,
        s3_output_path=S3_OUTPUT_PATH,
        stopping_condition=stopping_condition,
        overrides={"recipes.training_config.optim_config.lr": 5e-6},
    )

    sft_trainer.hyperparameters.max_epochs = 1

    # Submit — SFTTrainer routes to HyperPod based on compute type
    job_name = sft_trainer.train(wait=False)

    logger.info(f"HyperPod SFT job submitted: {job_name}")

    # Poll for job completion by checking for the manifest in S3
    from sagemaker.train.base_trainer import BaseTrainer

    max_wait_time = 21600  # 6 hour timeout (HyperPod jobs can take longer)
    poll_interval = 60  # Check every 60 seconds
    start_time = time.time()
    checkpoint_path = None

    while time.time() - start_time < max_wait_time:
        checkpoint_path = BaseTrainer._resolve_checkpoint_from_manifest(
            job_name=job_name,
            output_s3_path=S3_OUTPUT_PATH,
        )
        if checkpoint_path:
            logger.info(f"Checkpoint resolved: {checkpoint_path}")
            break

        elapsed = int(time.time() - start_time)
        logger.info(f"Waiting for manifest... ({elapsed}s elapsed)")
        time.sleep(poll_interval)

    assert checkpoint_path is not None, (
        f"Job {job_name} did not produce a manifest within {max_wait_time}s"
    )
    logger.info(f"Training complete. Checkpoint: {checkpoint_path}")
