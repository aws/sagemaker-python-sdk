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
"""Integration tests for SFT trainer with data mixing on HyperPod (end-to-end, no mocks).

Prerequisites:
    - Active AWS credentials in us-east-1 with permissions to launch SageMaker
      HyperPod training jobs and access the default SageMaker bucket.
    - A provisioned HyperPod cluster accessible from the test environment.
    - The ``hyperpod`` CLI installed and available on PATH.

The test is self-sufficient: it uses the account's default SageMaker bucket
(``sagemaker-{region}-{account_id}``, auto-created by ``Session.default_bucket()``).
Training data is generated and uploaded under a test-specific prefix.

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_sft_data_mixing_hyperpod.py -v -s
"""
from __future__ import absolute_import

import json
import logging
import time
import random

import pytest
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.training.configs import HyperPodCompute

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test configuration
REGION = "us-east-1"
DATA_PREFIX = "test-sft-data-mixing-hyperpod-integ"
NUM_TRAINING_SAMPLES = 300
HYPERPOD_CLUSTER_NAME = "pysdk-hp-integ-tests"
HYPERPOD_INSTANCE_TYPE = "ml.g6.12xlarge"


def _generate_training_data() -> str:
    """Generate 300 lines of bedrock-conversation format training data."""
    lines = []
    for i in range(NUM_TRAINING_SAMPLES):
        sample = {
            "schemaVersion": "bedrock-conversation-2024",
            "system": [
                {"text": "You are a helpful assistant who answers the question based on the task assigned"}
            ],
            "messages": [
                {"role": "user", "content": [{"text": f"Q{i}"}]},
                {"role": "assistant", "content": [{"text": f"A{i}"}]},
            ],
        }
        lines.append(json.dumps(sample))
    return "\n".join(lines)


def _ensure_training_data(s3_client, bucket_name: str) -> str:
    """Upload generated training data to S3 if not already present.

    Returns the S3 URI for the training data file.
    """
    s3_key = f"{DATA_PREFIX}/sft-data.jsonl"
    s3_uri = f"s3://{bucket_name}/{s3_key}"

    # Check if already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        logger.info(f"Training data already exists at {s3_uri}")
        return s3_uri
    except s3_client.exceptions.ClientError:
        pass

    logger.info(f"Generating and uploading training data to {s3_uri}")
    data = _generate_training_data()
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=data.encode("utf-8"),
    )
    logger.info(f"Uploaded {NUM_TRAINING_SAMPLES} training samples.")
    return s3_uri


@pytest.fixture(scope="module")
def training_resources(sagemaker_session_us_east_1):
    """Ensure training data exists in the account's default SageMaker bucket.

    Uses ``Session.default_bucket()`` (creates ``sagemaker-{region}-{account_id}``
    if needed), then uploads generated training data under the test prefix
    if not already present.

    Returns a dict with:
        - training_dataset: S3 URI to the training data file
        - s3_output_path: S3 URI for training output
        - bucket_name: the default bucket name
    """
    bucket_name = sagemaker_session_us_east_1.default_bucket()
    s3_client = sagemaker_session_us_east_1.boto_session.client("s3", region_name=REGION)
    training_dataset = _ensure_training_data(s3_client, bucket_name)

    return {
        "training_dataset": training_dataset,
        "s3_output_path": f"s3://{bucket_name}/{DATA_PREFIX}/output/",
        "bucket_name": bucket_name,
    }


@pytest.mark.skip(
    reason="Requires the hyperpod CLI and P5 capacity, neither of which is available "
    "in the PySDK CI account (784379639078). Runs only in a Nova-owned account."
)
@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_sft_trainer_nova_micro_data_mixing_hyperpod(sagemaker_session_us_east_1, training_resources):
    """Test SFT trainer with Nova Micro model and data mixing on HyperPod.

    This end-to-end test submits a real HyperPod training job with DataMixingConfig
    for Nova Micro. The SDK resolves the datamix recipe from SageMaker Hub, validates
    categories, and includes the serialized config in the HyperPod override parameters.
    """
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    data_mixing_config = DataMixingConfig(
        customer_data_percent=70.0,
        nova_data_percentages={
            "code": 50.0,
            "chat": 50.0
        },
    )

    compute = HyperPodCompute(
        cluster_name=HYPERPOD_CLUSTER_NAME,
        instance_type=HYPERPOD_INSTANCE_TYPE,
        node_count=1,
    )

    sft_trainer = SFTTrainer(
        model="nova-textgeneration-micro",
        training_type=TrainingType.LORA,
        training_dataset=training_resources["training_dataset"],
        s3_output_path=training_resources["s3_output_path"],
        sagemaker_session=sagemaker_session_us_east_1,
        compute=compute,
        data_mixing_config=data_mixing_config,
        base_job_name=f"sft-hp-datamix-integ-{unique_id}",
    )

    logger.info("Submitting SFT HyperPod training job with data mixing config...")
    try:
        training_job = sft_trainer.train(wait=False)
    except ValueError as e:
        if "Failed to download from S3 Access Point" in str(e):
            pytest.skip(
                "Skipping: account does not have Forge subscription for data mixing recipes. "
                "This is expected for non-subscribed accounts. "
                "See https://docs.aws.amazon.com/sagemaker/latest/dg/nova-forge.html#nova-forge-prereq-access"
            )
        raise

    # _train_hyperpod returns the job name as a string
    assert training_job is not None
    job_name = training_job
    logger.info(f"HyperPod training job submitted: {job_name}")

    # Verify the job exists on the cluster via hyperpod get-job
    import subprocess
    get_job_result = subprocess.run(
        ["hyperpod", "get-job", "--job-name", job_name],
        capture_output=True, text=True,
    )
    assert get_job_result.returncode == 0, (
        f"hyperpod get-job failed for '{job_name}': {get_job_result.stderr}"
    )
    logger.info(f"Verified job '{job_name}' exists on the cluster using hp-cli.")

    # Poll for job completion by checking for the manifest in S3.
    # The manifest is written under {output_s3_path}/{job_name}/manifest.json
    # once training finishes, so its presence confirms end-to-end completion.

    output_s3_path = training_resources["s3_output_path"]
    max_wait_time = 21600  # 6 hour timeout (HyperPod jobs can take longer)
    poll_interval = 60  # Check every 60 seconds
    start_time = time.time()
    checkpoint_path = None

    while time.time() - start_time < max_wait_time:
        checkpoint_path = BaseTrainer._resolve_checkpoint_from_manifest(
            job_name=job_name,
            output_s3_path=output_s3_path,
            sagemaker_session=sagemaker_session_us_east_1,
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
