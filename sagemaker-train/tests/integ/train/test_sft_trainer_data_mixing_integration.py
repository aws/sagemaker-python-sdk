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
"""Integration tests for SFT trainer with data mixing (end-to-end, no mocks).

Prerequisites:
    - Active AWS credentials in us-east-1 with permissions to launch SageMaker
      training jobs and access the default SageMaker bucket.

The test is self-sufficient: it uses the account's default SageMaker bucket
(``sagemaker-{region}-{account_id}``, auto-created by ``Session.default_bucket()``).
Training data is generated and uploaded under a test-specific prefix.

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_sft_trainer_data_mixing_integration.py -v -s
"""
from __future__ import absolute_import

import io
import json
import logging
import time
import random

import pytest
from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.common import TrainingType
from sagemaker.train.data_mixing_config import DataMixingConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test configuration
REGION = "us-east-1"
DATA_PREFIX = "test-sft-trainer-data-mixing-integ"
NUM_TRAINING_SAMPLES = 300


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
def test_sft_trainer_nova_lite2_with_data_mixing(sagemaker_session_us_east_1, training_resources):
    """Test SFT trainer with Nova Lite 2 model and data mixing.

    This end-to-end test submits a real training job with DataMixingConfig
    for Nova Lite 2. The SDK serializes the config into flat hyperparameters
    (customer_data_percent, nova_<category>_percent) automatically.
    """
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    data_mixing_config = DataMixingConfig(
        customer_data_percent=70.0,
        nova_data_percentages={
            "code": 30.0,
            "math": 20.0,
            "planning": 10.0,
            "instruction-following": 10.0,
            "reasoning-instruction-following": 20.0,
            "reasoning-math": 10.0,
        },
    )

    sft_trainer = SFTTrainer(
        model="nova-textgeneration-lite-v2",
        training_type=TrainingType.LORA,
        model_package_group="sdk-test-finetuned-models",
        training_dataset=training_resources["training_dataset"],
        s3_output_path=training_resources["s3_output_path"],
        sagemaker_session=sagemaker_session_us_east_1,
        data_mixing_config=data_mixing_config,
        base_job_name=f"sft-nova-datamix-integ-{unique_id}",
        overrides={"name": f"sft-nova-datamix-integ-{unique_id}"},
    )

    logger.info("Submitting SFT training job with data mixing config...")
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

    assert training_job is not None
    logger.info(f"Training job submitted: {training_job.training_job_name}")

    # Manual wait loop — Nova training can take over an hour
    max_wait_time = 10800  # 3 hour timeout
    poll_interval = 30  # Check every 30 seconds
    start_time = time.time()

    while time.time() - start_time < max_wait_time:
        training_job.refresh()
        status = training_job.training_job_status

        if status in ["Completed", "Failed", "Stopped"]:
            break

        time.sleep(poll_interval)

    # Verify job completed successfully
    assert training_job.training_job_status == "Completed", (
        f"Training job did not complete. Status: {training_job.training_job_status}"
    )
    assert hasattr(training_job, "output_model_package_arn")
    assert training_job.output_model_package_arn is not None
    logger.info("SFT training with data mixing completed successfully.")
