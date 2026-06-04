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

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("SAGEMAKER_REGION", "us-west-2")
os.environ.setdefault("AWS_REGION", "us-west-2")

import boto3
import pytest
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.agent_rft_job import AgentRFTJob

_REGION = "us-west-2"
_ACCOUNT_ID = boto3.client("sts", region_name=_REGION).get_caller_identity()["Account"]

AGENT_RUNTIME_ID = "sagemaker_rft_prod_gsm8k_streaming-Yk6O377mUS"
ROLE_ARN = f"arn:aws:iam::{_ACCOUNT_ID}:role/Admin"
MLFLOW_ARN = f"arn:aws:sagemaker:{_REGION}:{_ACCOUNT_ID}:mlflow-app/app-O4ZGQYBYHMRH"
S3_INPUT_PATH = f"s3://sagemaker-rft-{_ACCOUNT_ID}/prompts/gsm8k_small/prompts.parquet"
S3_OUTPUT_PATH = f"s3://sagemaker-{_REGION}-{_ACCOUNT_ID}/model-evaluation/mtrl-trainer-integ/"
LAMBDA_ARN = f"arn:aws:lambda:{_REGION}:{_ACCOUNT_ID}:function:SageMaker-AgentConnector-Lambda-MTRL-integ-test"
BASE_MODEL = "openai-reasoning-gpt-oss-20b"
EXISTING_JOB_NAME="openai-reasoning-gpt-oss-20b-mtrl-20260602005937"


@pytest.fixture(scope="module")
def sagemaker_session():
    os.environ.setdefault("AWS_DEFAULT_REGION", _REGION)
    os.environ["SAGEMAKER_MLFLOW_CUSTOM_ENDPOINT"] = f"https://mlflow.sagemaker.{_REGION}.app.aws"
    boto_session = boto3.Session(region_name=_REGION)
    session = Session(boto_session=boto_session)
    yield session


@pytest.mark.skip(reason="GPU resource intensive — run manually")
class TestMultiTurnRLTrainerBedrockAgent:
    """Test MTRL training with Bedrock AgentCore runtime."""

    def test_train_and_wait(self, sagemaker_session):
        """Test complete MTRL workflow with Bedrock AgentCore agent."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=AGENT_RUNTIME_ID,
            training_dataset=S3_INPUT_PATH,
            mlflow_app_arn=MLFLOW_ARN,
            s3_output_path=S3_OUTPUT_PATH,
            role=ROLE_ARN,
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

    def test_train_and_stop(self, sagemaker_session):
        """Test creating and stopping an MTRL job."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=AGENT_RUNTIME_ID,
            training_dataset=S3_INPUT_PATH,
            mlflow_app_arn=MLFLOW_ARN,
            role=ROLE_ARN,
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

    def test_train_with_lambda_arn(self, sagemaker_session):
        """Test MTRL workflow using an existing Lambda ARN as agent."""
        trainer = MultiTurnRLTrainer(
            model=BASE_MODEL,
            agent_env=LAMBDA_ARN,
            training_dataset=S3_INPUT_PATH,
            mlflow_app_arn=MLFLOW_ARN,
            s3_output_path=S3_OUTPUT_PATH,
            accept_eula=True,
            role=ROLE_ARN,
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
