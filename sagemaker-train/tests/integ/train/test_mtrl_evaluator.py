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
"""Integration tests for MultiTurnRLEvaluator.

These tests reuse existing completed MTRLTrainer jobs and feed them into
the MultiTurnRLEvaluator to validate the end-to-end evaluation flow.
"""
from __future__ import absolute_import

import json
import os
import pytest
import logging

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("SAGEMAKER_REGION", "us-west-2")

import boto3
from sagemaker.core.helper.session_helper import Session
from sagemaker.train.evaluate import MultiTurnRLEvaluator
from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Timeout for evaluation pipeline execution (4 hours)
EVALUATION_TIMEOUT_SECONDS = 14400

# Resolve current account ID for account-agnostic paths
_REGION = "us-west-2"
_ACCOUNT_ID = boto3.client("sts", region_name=_REGION).get_caller_identity()["Account"]

# Test configuration — uses current account for all resource paths
TEST_CONFIG = {
    "base_model": "huggingface-vlm-qwen3-6-27b",
    "agent_arn": f"arn:aws:bedrock-agentcore:{_REGION}:{_ACCOUNT_ID}:runtime/sagemaker_rft_prod_gsm8k_streaming-UwSB6LEfEq",
    "dataset": f"s3://sagemaker-{_REGION}-{_ACCOUNT_ID}/mtrl-integ-test/prompts/gsm8k_small/prompts.parquet",
    "s3_output_path": f"s3://sagemaker-{_REGION}-{_ACCOUNT_ID}/model-evaluation/output-artifacts/",
    "mlflow_resource_arn": f"arn:aws:sagemaker:{_REGION}:{_ACCOUNT_ID}:mlflow-app/app-JGGVLM43S4AS",
    "model_package_group": f"arn:aws:sagemaker:{_REGION}:{_ACCOUNT_ID}:model-package-group/mtrl-integ-test-evaluator",
    "role": f"arn:aws:iam::{_ACCOUNT_ID}:role/Admin",
    "region": _REGION,
}


def _ensure_model_package_group_exists(sm_client, group_name):
    """Create the model package group if it doesn't already exist."""
    try:
        sm_client.describe_model_package_group(ModelPackageGroupName=group_name)
    except Exception:
        sm_client.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription="Auto-created for MTRL evaluator integ tests",
        )


def _ensure_model_package_exists(sm_client, group_name, base_model_name):
    """Create a model package in the group if none exists, for test purposes."""
    resp = sm_client.list_model_packages(
        ModelPackageGroupName=group_name,
        MaxResults=1,
    )
    if resp.get("ModelPackageSummaryList"):
        return resp["ModelPackageSummaryList"][0]["ModelPackageArn"]

    # Create a minimal unversioned model package (no InferenceSpecification needed)
    resp = sm_client.create_model_package(
        ModelPackageGroupName=group_name,
        ModelPackageDescription="Test model package for MTRL evaluator integ tests",
        ModelApprovalStatus="Approved",
    )
    return resp["ModelPackageArn"]


@pytest.fixture(scope="module")
def sagemaker_session():
    """Create a SageMaker session with explicit region for CI environments."""
    boto_session = boto3.Session(region_name=TEST_CONFIG["region"])
    return Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def mtrl_trainer(sagemaker_session):
    """Create a lightweight MultiTurnRLTrainer-like object for evaluator tests.

    Instead of going through the full constructor (which validates remote
    resources), we build a minimal object with the attributes the evaluator
    needs. This makes the test account-agnostic — it creates the required
    resources (model package group + model package) on the fly.
    """
    sm_client = sagemaker_session.boto_session.client("sagemaker")
    group_name = "mtrl-integ-test-evaluator"
    _ensure_model_package_group_exists(sm_client, group_name)
    model_package_arn = _ensure_model_package_exists(
        sm_client, group_name, TEST_CONFIG["base_model"]
    )

    trainer = object.__new__(MultiTurnRLTrainer)
    trainer._model_name = TEST_CONFIG["base_model"]
    trainer._model_arn = f"arn:aws:sagemaker:{_REGION}:aws:hub-content/SageMakerPublicHub/Model/{TEST_CONFIG['base_model']}/1.0.0"
    trainer.agent_env = TEST_CONFIG["agent_arn"]
    trainer.bedrock_agentcore_qualifier = "DEFAULT"
    trainer.output_model_package_group = TEST_CONFIG["model_package_group"]
    trainer.sagemaker_session = sagemaker_session

    # Use the real model package ARN from the account
    class _FakeJob:
        job_name = "mtrl-integ-test-fake-job"
        job_status = "Completed"

    _FakeJob.output_model_package_arn = model_package_arn
    trainer._latest_job = _FakeJob()

    logger.info(f"Created test trainer with model: {trainer._model_name}")
    logger.info(f"Output model package ARN: {trainer._latest_job.output_model_package_arn}")

    return trainer


class TestMTRLEvaluatorJobConfigDocument:
    """Tests validating the JobConfigDocument field naming for GA API contract."""

    def test_bedrock_agent_config_fields(self, mtrl_trainer):
        """Verify BedrockAgentCoreConfig uses AgentRuntimeArn and Qualifier."""
        evaluator = MultiTurnRLEvaluator(
            model=mtrl_trainer,
            dataset=TEST_CONFIG["dataset"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}integ-fields-bedrock/',
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
            agent_config=TEST_CONFIG["agent_arn"],
            agent_qualifier="PROD",
        )

        evaluator._resolve_trainer_defaults()
        evaluator._resolve_agent_arn()

        ctx = evaluator._build_template_context(
            aws_context={"region": TEST_CONFIG["region"], "account_id": _ACCOUNT_ID,
                         "role_arn": TEST_CONFIG["role"]},
            artifacts={},
            model_package_group_arn=TEST_CONFIG["model_package_group"],
        )

        doc = json.loads(ctx["job_config_document_ft_str"])
        agent_cfg = doc["AgentConfig"]

        assert "BedrockAgentCoreConfig" in agent_cfg
        assert "AgentRuntimeArn" in agent_cfg["BedrockAgentCoreConfig"]
        assert "Qualifier" in agent_cfg["BedrockAgentCoreConfig"]
        assert agent_cfg["BedrockAgentCoreConfig"]["Qualifier"] == "PROD"
        # Ensure old field names are NOT present
        assert "EndpointConfig" not in agent_cfg
        assert "AgentArn" not in agent_cfg.get("BedrockAgentCoreConfig", {})
        assert "BedrockAgentCoreQualifier" not in agent_cfg.get("BedrockAgentCoreConfig", {})

    def test_lambda_agent_config_fields(self, mtrl_trainer):
        """Verify Lambda agent uses CustomAgentLambdaConfig (not LambdaConfig)."""
        lambda_arn = "arn:aws:lambda:us-east-1:060795915353:function:SageMaker-agent-adapter-gsm8k"
        evaluator = MultiTurnRLEvaluator(
            model=mtrl_trainer,
            dataset=TEST_CONFIG["dataset"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}integ-fields-lambda/',
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
            agent_config=lambda_arn,
        )

        evaluator._resolve_trainer_defaults()
        evaluator._resolve_agent_arn()

        ctx = evaluator._build_template_context(
            aws_context={"region": TEST_CONFIG["region"], "account_id": _ACCOUNT_ID,
                         "role_arn": TEST_CONFIG["role"]},
            artifacts={},
            model_package_group_arn=TEST_CONFIG["model_package_group"],
        )

        doc = json.loads(ctx["job_config_document_ft_str"])
        agent_cfg = doc["AgentConfig"]

        assert "CustomAgentLambdaConfig" in agent_cfg
        assert "LambdaArn" in agent_cfg["CustomAgentLambdaConfig"]
        assert agent_cfg["CustomAgentLambdaConfig"]["LambdaArn"] == lambda_arn
        # Ensure old field name is NOT present
        assert "LambdaConfig" not in agent_cfg

    def test_model_package_config_fields(self, mtrl_trainer):
        """Verify ModelPackageConfig uses InputModelPackageArn only (no OutputModelPackageGroupArn for eval)."""
        evaluator = MultiTurnRLEvaluator(
            model=mtrl_trainer,
            dataset=TEST_CONFIG["dataset"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}integ-fields-mpc/',
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
            agent_config=TEST_CONFIG["agent_arn"],
        )

        evaluator._resolve_trainer_defaults()
        evaluator._resolve_agent_arn()

        ctx = evaluator._build_template_context(
            aws_context={"region": TEST_CONFIG["region"], "account_id": _ACCOUNT_ID,
                         "role_arn": TEST_CONFIG["role"]},
            artifacts={},
            model_package_group_arn=TEST_CONFIG["model_package_group"],
        )

        doc = json.loads(ctx["job_config_document_ft_str"])
        mpc = doc.get("ModelPackageConfig", {})

        assert "InputModelPackageArn" in mpc
        # OutputModelPackageGroupArn is not supported in evaluation job schema
        assert "OutputModelPackageGroupArn" not in mpc
        # Ensure old field names are NOT present
        assert "SourceModelPackageArn" not in mpc
        assert "ModelPackageGroupArn" not in mpc


class TestMTRLEvaluatorIntegration:
    """Integration tests for MultiTurnRLEvaluator construction and resolution.

    Note: Pipeline submission tests (evaluate(), pipeline_reuse) require the
    ``Job`` step type to be enabled in the account. These are tested separately
    in accounts with the feature flag enabled (e.g., 742774200982).
    """

    def test_evaluator_construction_with_trainer(self, mtrl_trainer):
        """Test that MultiTurnRLEvaluator can be constructed from a trainer."""
        evaluator = MultiTurnRLEvaluator(
            model=mtrl_trainer,
            dataset=TEST_CONFIG["dataset"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}integ-construct/',
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
            agent_config=TEST_CONFIG["agent_arn"],
        )

        assert evaluator is not None
        assert evaluator.model is mtrl_trainer
        assert evaluator.dataset == TEST_CONFIG["dataset"]
        assert evaluator.region == TEST_CONFIG["region"]

    def test_evaluator_construction_with_base_model(self):
        """Test that MultiTurnRLEvaluator can be constructed from a base model string."""
        evaluator = MultiTurnRLEvaluator(
            model=TEST_CONFIG["base_model"],
            dataset=TEST_CONFIG["dataset"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}integ-base/',
            agent_config=TEST_CONFIG["agent_arn"],
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
        )

        assert evaluator is not None
        assert evaluator.model == TEST_CONFIG["base_model"]

    def test_get_all_mtrl_evaluations(self):
        """Test listing all MTRL evaluation executions."""
        all_execs = MultiTurnRLEvaluator.get_all(region=TEST_CONFIG["region"])

        if hasattr(all_execs, '__iter__'):
            all_execs = list(all_execs)

        assert all_execs is not None
        logger.info(f"Total MTRL evaluations: {len(all_execs)}")
