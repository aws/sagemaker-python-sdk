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
"""Integration tests for MultiTurnRLTrainer (Agentic RFT).

These tests run against real SageMaker services in prod us-west-2.
Requires valid AWS credentials with appropriate permissions.
"""
from __future__ import annotations

import os
import time

import boto3
import pytest
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.agent_rft_job import AgentRFTJob

_REGION = "us-west-2"
_ACCOUNT_ID = None  # Resolved lazily in fixtures


def _get_account_id():
    """Resolve account ID lazily."""
    global _ACCOUNT_ID
    if _ACCOUNT_ID is None:
        boto_session = boto3.Session(region_name=_REGION)
        _ACCOUNT_ID = boto_session.client("sts").get_caller_identity()["Account"]
    return _ACCOUNT_ID

AGENT_RUNTIME_ID = "sagemaker_rft_prod_gsm8k_streaming-Yk6O377mUS"
BASE_MODEL = "openai-reasoning-gpt-oss-20b"
EXISTING_JOB_NAME = "openai-reasoning-gpt-oss-20b-mtrl-20260602005937"


@pytest.fixture(scope="module")
def sagemaker_session():
    boto_session = boto3.Session(region_name=_REGION)
    session = Session(boto_session=boto_session)
    yield session


@pytest.fixture(scope="module")
def test_resources():
    """Resolve account-specific resource ARNs lazily."""
    account_id = _get_account_id()
    return {
        "role_arn": f"arn:aws:iam::{account_id}:role/Admin",
        "mlflow_arn": f"arn:aws:sagemaker:{_REGION}:{account_id}:mlflow-app/app-TTAUWUNMUHH6",
        "s3_input_path": f"s3://sagemaker-rft-{account_id}/prompts/gsm8k_small/prompts.parquet",
        "s3_output_path": f"s3://sagemaker-{_REGION}-{account_id}/model-evaluation/mtrl-trainer-integ/",
        "lambda_arn": f"arn:aws:lambda:{_REGION}:{account_id}:function:SageMaker-AgentConnector-Lambda-MTRL-integ-test",
    }


@pytest.mark.skip(reason="GPU resource intensive — run manually")
class TestMultiTurnRLTrainerBedrockAgent:
    """Test MTRL training with Bedrock AgentCore runtime."""

    def test_train_and_wait(self, sagemaker_session, test_resources):
        """Test complete MTRL workflow with Bedrock AgentCore agent."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=AGENT_RUNTIME_ID,
            training_dataset=test_resources["s3_input_path"],
            mlflow_app_arn=test_resources["mlflow_arn"],
            s3_output_path=test_resources["s3_output_path"],
            role=test_resources["role_arn"],
            accept_eula=True,
            sagemaker_session=sagemaker_session,
        )
        trainer.hyperparameters.global_batch_size = 32

        job = trainer.train(wait=False)

        assert job.job_name is not None
        assert job.job_arn is not None

        job.wait(poll=30, timeout=3600)

        assert job.job_status == "Completed"
        assert job.output_model_package_arn is not None
        assert job.s3_output_path is not None

    def test_train_and_stop(self, sagemaker_session, test_resources):
        """Test creating and stopping an MTRL job."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=AGENT_RUNTIME_ID,
            training_dataset=test_resources["s3_input_path"],
            mlflow_app_arn=test_resources["mlflow_arn"],
            role=test_resources["role_arn"],
            accept_eula=True,
            sagemaker_session=sagemaker_session,
        )
        trainer.hyperparameters.global_batch_size = 32

        job = trainer.train(wait=False)
        assert job.job_status in ("Pending", "InProgress")

        # Wait briefly for job to start, then stop
        time.sleep(30)
        job.stop()

        job.refresh()
        assert job.job_status in ("Stopping", "Stopped")


@pytest.mark.skip(reason="GPU resource intensive — run manually")
class TestMultiTurnRLTrainerLambdaAgent:
    """Test MTRL training with Lambda agent."""

    def test_train_with_lambda_arn(self, sagemaker_session, test_resources):
        """Test MTRL workflow using an existing Lambda ARN as agent."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=test_resources["lambda_arn"],
            training_dataset=test_resources["s3_input_path"],
            mlflow_app_arn=test_resources["mlflow_arn"],
            s3_output_path=test_resources["s3_output_path"],
            accept_eula=True,
            role=test_resources["role_arn"],
            sagemaker_session=sagemaker_session,
        )
        trainer.hyperparameters.global_batch_size = 32

        job = trainer.train(wait=False)

        assert job.job_name is not None
        assert job.job_arn is not None

        job.wait(poll=30, timeout=3600)

        assert job.job_status == "Completed"
        assert job.output_model_package_arn is not None


@pytest.mark.skip(reason="GPU resource intensive — run manually")
class TestMultiTurnRLTrainerAttach:
    """Test attaching to existing MTRL jobs."""

    def test_attach_and_get_properties(self, sagemaker_session):
        """Test attaching to a completed job and reading properties."""

        # Attach to the same job by name
        attached_job = MultiTurnRLTrainer.attach(
            EXISTING_JOB_NAME, session=sagemaker_session.boto_session
        )

        assert attached_job.job_name == EXISTING_JOB_NAME
        assert attached_job.job_status == "Completed"
        assert attached_job.output_model_package_arn is not None
        assert attached_job.s3_output_path is not None

    def test_get_all_jobs(self, sagemaker_session):
        """Test listing all MTRL jobs."""
        jobs = list(AgentRFTJob.get_all(
            session=sagemaker_session.boto_session,
            status_equals="Completed",
        ))
        assert len(jobs) > 0
        assert all(j.job_status == "Completed" for j in jobs)


@pytest.mark.skip(reason="GPU resource intensive — run manually")
class TestMultiTurnRLTrainerListModels:
    """Test listing supported models (requires API access)."""

    def test_list_supported_models(self, sagemaker_session):
        """Test that list_supported_models returns models from the hub."""
        result = MultiTurnRLTrainer.list_supported_models(
            session=sagemaker_session.boto_session
        )
        assert isinstance(result, list)
        assert len(result) > 0

    def test_list_bedrock_agentcore_runtimes(self, sagemaker_session):
        """Test listing Bedrock AgentCore runtimes."""
        runtimes = MultiTurnRLTrainer.list_bedrock_agentcore_runtimes(
            session=sagemaker_session.boto_session
        )
        assert isinstance(runtimes, list)
