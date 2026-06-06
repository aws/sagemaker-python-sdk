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
"""Integration tests for MTRL Evaluator: Attach to existing job → Evaluate → Wait for completion.

Tests the MTRL evaluation workflow by attaching to existing completed training
jobs (discovered via DescribeJob API) and running evaluations in both
fine-tuned and base model modes, waiting for successful completion.

Accounts:
  - PROD (729646638167): Main account
  - PREPROD (391266019386): Staging account
"""
from __future__ import absolute_import

import os
import pytest
import logging

import boto3

from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.evaluate import MultiTurnRLEvaluator

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

_REGION = "us-west-2"

EVAL_TIMEOUT = 14400  # 4 hours


def _get_account_id():
    """Get current AWS account ID via STS."""
    boto_session = boto3.Session(region_name=_REGION)
    return boto_session.client("sts").get_caller_identity()["Account"]

# ============================================================
# Per-account resource configuration
# ============================================================

ACCOUNT_CONFIGS = {
    # PROD — Main account (729646638167)
    "729646638167": {
        "env_name": "PROD",
        "existing_job_name": "openai-reasoning-gpt-oss-20b-mtrl-20260602215955",
        "base_model": "openai-reasoning-gpt-oss-20b",
        "agent_core_arn": "arn:aws:bedrock-agentcore:us-west-2:729646638167:runtime/sagemaker_rft_prod_gsm8k_streaming-Yk6O377mUS",
        "dataset": "s3://sagemaker-rft-729646638167/prompts/gsm8k_small/prompts.parquet",
        "s3_output_path": "s3://sagemaker-us-west-2-729646638167/mtrl-integ/eval-output/",
        "mlflow_resource_arn": "arn:aws:sagemaker:us-west-2:729646638167:mlflow-app/app-TTAUWUNMUHH6",
        "model_package_group": "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/openai-reasoning-gpt-oss-20b-mtrl-mpg",
        "role": "arn:aws:iam::729646638167:role/Admin",
    },
    # PREPROD — Staging account (391266019386)
    "391266019386": {
        "env_name": "PREPROD",
        "existing_job_name": "mtrl-integ-gpt-oss-agentcore-1779143704358",
        "base_model": "openai-reasoning-gpt-oss-20b",
        "agent_core_arn": "arn:aws:bedrock-agentcore:us-west-2:391266019386:runtime/mtrl_integ_gsm8k_streaming-bIz4H5Echk",
        "dataset": "s3://sagemaker-rft-beta-391266019386/prompts/gsm8k_small/prompts.parquet",
        "s3_output_path": "s3://sagemaker-us-west-2-391266019386/mtrl-integ/eval-output/",
        "mlflow_resource_arn": "arn:aws:sagemaker:us-west-2:391266019386:mlflow-app/app-P3FRQFRQTNGI",
        "model_package_group": "arn:aws:sagemaker:us-west-2:391266019386:model-package-group/mtrl-integ-gpt-oss-agentcore",
        "role": "arn:aws:iam::391266019386:role/Admin",
    },
    # BETA — Dev/test account (742774200982)
    "742774200982": {
        "env_name": "BETA",
        "existing_job_name": "openai-reasoning-gpt-oss-20b-mtrl-20260601114439",
        "base_model": "openai-reasoning-gpt-oss-20b",
        "agent_core_arn": "arn:aws:bedrock-agentcore:us-west-2:742774200982:runtime/sagemaker_rft_prod_gsm8k_streaming-UwSB6LEfEq",
        "dataset": "s3://sagemaker-rft-beta-742774200982/prompts/gsm8k_small/prompts.parquet",
        "s3_output_path": "s3://sagemaker-us-west-2-742774200982/mtrl-integ/eval-output/",
        "mlflow_resource_arn": "arn:aws:sagemaker:us-west-2:742774200982:mlflow-app/app-6ZU5TXXH2GUX",
        "model_package_group": "arn:aws:sagemaker:us-west-2:742774200982:model-package-group/openai-reasoning-gpt-oss-20b-mtrl-mpg",
        "role": "arn:aws:iam::742774200982:role/Admin",
    },
}


def _get_config():
    """Get config for the current account."""
    account_id = _get_account_id()
    if account_id not in ACCOUNT_CONFIGS:
        pytest.skip(
            f"Account {account_id} not configured for MTRL integ tests. "
            f"Supported accounts: {list(ACCOUNT_CONFIGS.keys())}"
        )
    return ACCOUNT_CONFIGS[account_id]


@pytest.fixture(scope="module")
def config():
    """Get account-specific test configuration."""
    return _get_config()


@pytest.fixture(scope="module")
def attached_trainer(config):
    """Attach to an existing completed MTRL training job and return a trainer with it set.

    Uses the DescribeJob API (via MultiTurnRLTrainer.attach) to retrieve
    the job and verify it completed successfully with an output model package.
    """
    job = MultiTurnRLTrainer.attach(job_name=config["existing_job_name"])
    logger.info(f"[{config['env_name']}] Attached to job: {job.job_name}")
    logger.info(f"[{config['env_name']}] Status: {job.job_status}")
    logger.info(f"[{config['env_name']}] Output model package: {job.output_model_package_arn}")

    assert job.job_status == "Completed", (
        f"Existing job {config['existing_job_name']} is not Completed "
        f"(status: {job.job_status}). Cannot use for evaluation."
    )
    assert job.output_model_package_arn is not None, (
        f"Existing job {config['existing_job_name']} has no output_model_package_arn."
    )

    trainer = MultiTurnRLTrainer(
        model=config["base_model"],
        agent_env=config["agent_core_arn"],
        training_dataset=config["dataset"],
        output_model_package_group=config["model_package_group"],
        mlflow_app_arn=config["mlflow_resource_arn"],
        s3_output_path=config["s3_output_path"],
        role=config["role"],
        accept_eula=True,
    )
    trainer._latest_job = job
    return trainer


class TestMTRLEvalIntegration:
    """Integration tests for MTRL evaluation: attach → evaluate → wait for success."""

    def test_attach_to_existing_job(self, config):
        """Test that we can attach to an existing completed MTRL job via DescribeJob API."""
        job = MultiTurnRLTrainer.attach(job_name=config["existing_job_name"])

        assert job is not None
        assert job.job_name == config["existing_job_name"]
        assert job.job_status == "Completed"
        assert job.output_model_package_arn is not None

        logger.info(f"[{config['env_name']}] Job name: {job.job_name}")
        logger.info(f"[{config['env_name']}] Job ARN: {job.job_arn}")
        logger.info(f"[{config['env_name']}] Output model package: {job.output_model_package_arn}")

    def test_evaluate_finetuned_model(self, attached_trainer, config):
        """Evaluate a fine-tuned model from attached trainer — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_trainer,
            dataset=config["dataset"],
            s3_output_path=f'{config["s3_output_path"]}finetuned/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            role=config["role"],
            region=_REGION,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started finetuned eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for finetuned eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Finetuned eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Finetuned eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )

    @pytest.mark.skip(reason="Quota limited (1 concurrent eval job) - run manually")
    def test_evaluate_base_model(self, config):
        """Evaluate the base model only — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=config["base_model"],
            dataset=config["dataset"],
            agent_config=config["agent_core_arn"],
            s3_output_path=f'{config["s3_output_path"]}basemodel/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            role=config["role"],
            region=_REGION,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started base model eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for base model eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Base model eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Base model eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )

    @pytest.mark.skip(reason="Comparison template has CreateJob schema validation issue — tracked separately")
    def test_evaluate_comparison(self, attached_trainer, config):
        """Evaluate base + finetuned comparison — submit and wait for completion."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_trainer,
            dataset=config["dataset"],
            s3_output_path=f'{config["s3_output_path"]}comparison/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            role=config["role"],
            region=_REGION,
            evaluate_base_model=True,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"[{config['env_name']}] Started comparison eval: {execution.arn}")

        logger.info(f"[{config['env_name']}] Waiting for comparison eval to complete...")
        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Comparison eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Comparison eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )

    @pytest.mark.skip(reason="Quota limited (1 concurrent eval job) - run manually")
    def test_evaluate_with_hyperparam_override(self, attached_trainer, config):
        """Test that hyperparameter overrides are passed through to the eval job."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_trainer,
            dataset=config["dataset"],
            s3_output_path=f'{config["s3_output_path"]}hyperparam-override/',
            mlflow_resource_arn=config["mlflow_resource_arn"],
            role=config["role"],
            region=_REGION,
        )

        # Override MTRL-specific hyperparams
        evaluator.hyperparameters.sampling_max_tokens = 1024
        evaluator.hyperparameters.eval_group_size = 4

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        logger.info(f"[{config['env_name']}] Started hyperparam override eval: {execution.arn}")

        execution.wait(timeout=EVAL_TIMEOUT)

        status = execution.status.overall_status
        logger.info(f"[{config['env_name']}] Hyperparam override eval completed: {status}")

        assert status == "Succeeded", (
            f"[{config['env_name']}] Hyperparam override eval failed with status: {status}, "
            f"reason: {execution.status.failure_reason}"
        )
