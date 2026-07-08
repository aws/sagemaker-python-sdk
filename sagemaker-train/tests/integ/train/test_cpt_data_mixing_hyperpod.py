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
"""Integration tests for CPTTrainer with data mixing on HyperPod (end-to-end, no mocks).

Prerequisites:
    - Active AWS credentials in us-east-1 with permissions to launch SageMaker
      HyperPod training jobs and access the default SageMaker bucket.
    - A provisioned HyperPod cluster in us-east-1.
    - The ``hyperpod`` CLI installed and on PATH.

The test submits a real CPT job to HyperPod with DataMixingConfig and verifies
the job is accepted by the CLI and exists on the cluster.

Run with:
    export AWS_DEFAULT_REGION=us-east-1
    pytest tests/integ/train/test_cpt_data_mixing_hyperpod.py -v -s
"""
from __future__ import absolute_import

import json
import logging
import os
import subprocess
import time
import random

import pytest
from sagemaker.train.cpt_trainer import CPTTrainer
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.training.configs import HyperPodCompute

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Test configuration
REGION = "us-east-1"
# HyperPod cluster provisioned for the pysdk integ-test account in us-east-1.
# Overridable via env var so a cluster rename does not require a code change.
CLUSTER_NAME = os.environ.get("PYSDK_HYPERPOD_CLUSTER_NAME", "pysdk-hp-integ-tests")
INSTANCE_TYPE = "ml.p5.48xlarge"
NODE_COUNT = 2
MODEL_NAME = "nova-textgeneration-micro"
DATA_PREFIX = "test-cpt-data-mixing-integ"
NUM_TRAINING_SAMPLES = 500


def _generate_cpt_training_data() -> str:
    """Generate unlabeled text corpus for continued pre-training.

    CPT uses raw text data (not conversation format like SFT).
    Each line is a JSON object with a 'text' field.
    """
    lines = []
    domains = [
        "machine learning",
        "natural language processing",
        "computer vision",
        "distributed systems",
        "cloud computing",
    ]
    for i in range(NUM_TRAINING_SAMPLES):
        domain = domains[i % len(domains)]
        sample = {
            "text": (
                f"Document {i}: This is a sample document about {domain}. "
                f"It contains domain-specific knowledge that helps the model learn "
                f"about {domain} concepts, terminology, and patterns. "
                f"The continued pre-training process uses this text to extend "
                f"the foundation model's knowledge base in this area."
            )
        }
        lines.append(json.dumps(sample))
    return "\n".join(lines)


def _ensure_training_data(s3_client, bucket_name: str) -> str:
    """Upload generated CPT training data to S3 if not already present.

    Returns the S3 URI for the training data file.
    """
    s3_key = f"{DATA_PREFIX}/cpt-corpus.jsonl"
    s3_uri = f"s3://{bucket_name}/{s3_key}"

    # Check if already exists
    try:
        s3_client.head_object(Bucket=bucket_name, Key=s3_key)
        logger.info(f"CPT training data already exists at {s3_uri}")
        return s3_uri
    except s3_client.exceptions.ClientError:
        pass

    logger.info(f"Generating and uploading CPT training data to {s3_uri}")
    data = _generate_cpt_training_data()
    s3_client.put_object(
        Bucket=bucket_name,
        Key=s3_key,
        Body=data.encode("utf-8"),
    )
    logger.info(f"Uploaded {NUM_TRAINING_SAMPLES} CPT training samples.")
    return s3_uri


@pytest.fixture(scope="module")
def training_resources(sagemaker_session_us_east_1):
    """Ensure CPT training data exists in the account's default SageMaker bucket."""
    bucket_name = sagemaker_session_us_east_1.default_bucket()
    s3_client = sagemaker_session_us_east_1.boto_session.client("s3", region_name=REGION)
    training_dataset = _ensure_training_data(s3_client, bucket_name)

    return {
        "training_dataset": training_dataset,
        "s3_output_path": f"s3://{bucket_name}/{DATA_PREFIX}/output/",
        "bucket_name": bucket_name,
    }


@pytest.mark.gpu_intensive
@pytest.mark.us_east_1
def test_cpt_trainer_nova_micro_with_data_mixing_hyperpod(
    sagemaker_session_us_east_1, training_resources
):
    """Test CPTTrainer with Nova Micro model and data mixing on HyperPod.

    This end-to-end test submits a real CPT job to HyperPod with DataMixingConfig.
    The SDK downloads the datamix recipe template from Hub, injects data mixing
    values, writes it locally, and passes the generated recipe to the CLI.
    """
    unique_id = f"{int(time.time())}-{random.randint(1000, 9999)}"

    data_mixing_config = DataMixingConfig(
        customer_data_percent=70.0,
        nova_data_percentages={
            "code": 40.0,
            "math": 30.0,
            "en-scientific": 30.0,
        },
    )

    compute = HyperPodCompute(
        cluster_name=CLUSTER_NAME,
        instance_type=INSTANCE_TYPE,
        node_count=NODE_COUNT,
    )

    cpt_trainer = CPTTrainer(
        model=MODEL_NAME,
        compute=compute,
        training_dataset=training_resources["training_dataset"],
        s3_output_path=training_resources["s3_output_path"],
        sagemaker_session=sagemaker_session_us_east_1,
        data_mixing_config=data_mixing_config,
        base_job_name=f"hp-datamix-m1-{unique_id}",
    )

    logger.info("Submitting CPT HyperPod training job with data mixing config...")
    try:
        job_name = cpt_trainer.train(wait=False)
    except ValueError as e:
        if "Failed to download" in str(e) or "Forge subscription" in str(e):
            pytest.skip(
                "Skipping: account does not have Forge subscription for data mixing recipes."
            )
        raise

    # _train_hyperpod returns the job name as a string
    assert job_name is not None
    logger.info(f"HyperPod CPT job submitted: {job_name}")

    # Verify the job exists on the cluster via hyperpod get-job
    get_job_result = subprocess.run(
        ["hyperpod", "get-job", "--job-name", job_name],
        capture_output=True, text=True,
    )
    assert get_job_result.returncode == 0, (
        f"hyperpod get-job failed for '{job_name}': {get_job_result.stderr}"
    )
    logger.info(f"Verified job '{job_name}' exists on the cluster.")
