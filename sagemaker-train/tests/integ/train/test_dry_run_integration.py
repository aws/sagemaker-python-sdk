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
"""Integration tests for dry_run=True on trainers.

These tests validate that dry_run performs real validation against AWS
(IAM role resolution, S3 path existence, hyperparameter constraints)
without consuming compute. No training jobs are submitted.

A small sample dataset is uploaded to the SageMaker default bucket
during test setup and cleaned up afterward.
"""
from __future__ import absolute_import

import json
import time
import random

import pytest

from sagemaker.train.sft_trainer import SFTTrainer
from sagemaker.train.dpo_trainer import DPOTrainer
from sagemaker.train.rlvr_trainer import RLVRTrainer
from sagemaker.train.common import TrainingType
from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator
from sagemaker.core.training.configs import TrainingJobCompute


MODEL_PACKAGE_GROUP = (
    "arn:aws:sagemaker:us-west-2:729646638167:"
    "model-package-group/sdk-test-finetuned-models"
)
MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"
DATASET_KEY = "dry-run-integ-test/sample_train.jsonl"


@pytest.fixture(scope="module")
def valid_dataset(sagemaker_session):
    """Upload a small sample dataset to the default bucket if it doesn't already exist, return its S3 URI."""
    bucket = sagemaker_session.default_bucket()
    s3 = sagemaker_session.boto_session.client("s3")

    # Only upload if the object doesn't already exist
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=DATASET_KEY, MaxKeys=1)
    if resp.get("KeyCount", 0) == 0:
        samples = [
            {"messages": [
                {"role": "user", "content": [{"text": "What is 2+2?"}]},
                {"role": "assistant", "content": [{"text": "4"}]},
            ]},
            {"messages": [
                {"role": "user", "content": [{"text": "Capital of France?"}]},
                {"role": "assistant", "content": [{"text": "Paris"}]},
            ]},
        ]
        body = "\n".join(json.dumps(s) for s in samples)
        s3.put_object(Bucket=bucket, Key=DATASET_KEY, Body=body.encode("utf-8"))

    return f"s3://{bucket}/{DATASET_KEY}"


@pytest.fixture(scope="module")
def nonexistent_dataset_s3(sagemaker_session):
    """Return an S3 URI in the default bucket that does not exist."""
    bucket = sagemaker_session.default_bucket()
    return f"s3://{bucket}/dry-run-integ-test/nonexistent_path_12345.jsonl"


@pytest.fixture(scope="module")
def nonexistent_dataset_arn():
    """Return a DataSet ARN that does not exist."""
    return (
        "arn:aws:sagemaker:us-west-2:123456789012:hub-content/"
        "SageMakerPublicHub/DataSet/nonexistent-dataset-12345/1.0.0"
    )


@pytest.fixture(scope="module")
def nonexistent_dataset_obj():
    """Return a DataSet object with a nonexistent ARN."""
    from sagemaker.ai_registry.dataset import DataSet
    from sagemaker.ai_registry.air_constants import HubContentStatus

    return DataSet(
        name="nonexistent-dataset-12345",
        arn=(
            "arn:aws:sagemaker:us-west-2:123456789012:hub-content/"
            "SageMakerPublicHub/DataSet/nonexistent-dataset-12345/1.0.0"
        ),
        version="1.0.0",
        status=HubContentStatus.AVAILABLE,
    )


class TestDryRunS3PathValidation:
    """Verify dry_run raises when S3 data paths do not exist."""

    def test_sft_fails_on_nonexistent_training_dataset_s3(
        self, sagemaker_session, nonexistent_dataset_s3
    ):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=nonexistent_dataset_s3,
            accept_eula=True,
        )

        with pytest.raises(ValueError, match="does not exist"):
            trainer.train(dry_run=True)

    def test_sft_fails_on_nonexistent_training_dataset_arn(
        self, sagemaker_session, nonexistent_dataset_arn
    ):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=nonexistent_dataset_arn,
            accept_eula=True,
        )

        with pytest.raises(ValueError, match="does not exist"):
            trainer.train(dry_run=True)

    def test_sft_fails_on_nonexistent_training_dataset_obj(
        self, sagemaker_session, nonexistent_dataset_obj
    ):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=nonexistent_dataset_obj,
            accept_eula=True,
        )

        with pytest.raises(ValueError, match="does not exist"):
            trainer.train(dry_run=True)

    def test_sft_fails_on_nonexistent_validation_dataset(
        self, sagemaker_session, valid_dataset, nonexistent_dataset_s3
    ):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            validation_dataset=nonexistent_dataset_s3,
            accept_eula=True,
        )

        with pytest.raises(ValueError, match="does not exist"):
            trainer.train(dry_run=True)


class TestDryRunPassesWithValidInputs:
    """Verify dry_run=True returns None and does not create a job."""

    def test_sft_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None

    def test_dpo_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = DPOTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None

    def test_rlvr_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = RLVRTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None

    def test_no_training_job_created(self, sagemaker_session, valid_dataset):
        """Confirm via the SageMaker API that no job was submitted."""
        unique_id = f"{int(time.time())}-{random.randint(10000, 99999)}"
        base_name = f"dry-run-noop-{unique_id}"

        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            accept_eula=True,
            base_job_name=base_name,
        )

        trainer.train(dry_run=True)

        sm_client = sagemaker_session.sagemaker_client
        response = sm_client.list_training_jobs(
            NameContains=base_name,
            MaxResults=1,
        )
        assert len(response.get("TrainingJobSummaries", [])) == 0


class TestEvaluateDryRun:
    """Verify evaluate(dry_run=True) validates without submitting."""

    def test_benchmark_evaluate_dry_run_returns_none(self, sagemaker_session):
        from sagemaker.train.evaluate import get_benchmarks
        Benchmark = get_benchmarks()

        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            model=MODEL_ID,
            s3_output_path=f"s3://{sagemaker_session.default_bucket()}/dry-run-eval-output/",
        )

        result = evaluator.evaluate(dry_run=True)
        assert result is None


class TestDryRunServerful:
    """Verify dry_run=True works for serverful (TrainingJobCompute) path."""

    def test_sft_serverful_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = SFTTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            compute=TrainingJobCompute(instance_type="ml.g5.2xlarge", instance_count=1),
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None

    def test_dpo_serverful_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = DPOTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            compute=TrainingJobCompute(instance_type="ml.g5.2xlarge", instance_count=1),
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None

    def test_rlvr_serverful_dry_run_returns_none(self, sagemaker_session, valid_dataset):
        trainer = RLVRTrainer(
            model=MODEL_ID,
            training_type=TrainingType.LORA,
            model_package_group=MODEL_PACKAGE_GROUP,
            training_dataset=valid_dataset,
            compute=TrainingJobCompute(instance_type="ml.g5.2xlarge", instance_count=1),
            accept_eula=True,
        )

        result = trainer.train(dry_run=True)
        assert result is None
