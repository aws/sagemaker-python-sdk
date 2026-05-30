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
"""Integration tests for MultiTurnRLTrainer: Attach → Evaluate → Deploy.

Tests the MTRL workflow by attaching to existing completed training jobs
and validating evaluation and deployment via ModelBuilder.
"""
from __future__ import absolute_import

import json
import os
import uuid
import pytest
import logging

import boto3

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("SAGEMAKER_REGION", "us-west-2")
os.environ.setdefault("AWS_REGION", "us-west-2")

from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
from sagemaker.train.evaluate import MultiTurnRLEvaluator
from sagemaker.core.resources import ModelPackage

try:
    from sagemaker.serve import ModelBuilder
except ImportError:
    pytest.skip("sagemaker-serve not installed", allow_module_level=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

_REGION = "us-west-2"
_ACCOUNT_ID = boto3.client("sts", region_name=_REGION).get_caller_identity()["Account"]

TEST_CONFIG = {
    "existing_job_name": "papriwal-gemma-newapi-1778926381662",
    "base_model": "huggingface-reasoning-qwen3-32b",
    "agent_arn": f"arn:aws:bedrock-agentcore:{_REGION}:{_ACCOUNT_ID}:runtime/sagemaker_rft_prod_gsm8k_streaming-UwSB6LEfEq",
    "lambda_agent_arn": f"arn:aws:lambda:{_REGION}:{_ACCOUNT_ID}:function:SageMaker-AgentConnector-1-1777398957761",
    "dataset": f"s3://sagemaker-rft-beta-{_ACCOUNT_ID}/prompts/gsm8k_small/prompts.parquet",
    "s3_output_path": f"s3://sagemaker-{_REGION}-{_ACCOUNT_ID}/model-evaluation/mtrl-trainer-integ/",
    "mlflow_resource_arn": f"arn:aws:sagemaker:{_REGION}:{_ACCOUNT_ID}:mlflow-app/app-JGGVLM43S4AS",
    "model_package_group": f"arn:aws:sagemaker:{_REGION}:{_ACCOUNT_ID}:model-package-group/papriwal-gemma-newapi",
    "role": f"arn:aws:iam::{_ACCOUNT_ID}:role/Admin",
    "region": _REGION,
    "instance_type": "ml.g6e.48xlarge",
    "runtime_endpoint": "https://maeveruntime.loadtest.us-west-2.ml-platform.aws.a2z.com",
}


@pytest.fixture(scope="module")
def attached_job():
    """Attach to an existing completed MTRL training job."""
    job = MultiTurnRLTrainer.attach(job_name=TEST_CONFIG["existing_job_name"])
    logger.info(f"Attached to job: {job.job_name}")
    logger.info(f"Status: {job.job_status}")
    logger.info(f"Output model package: {job.output_model_package_arn}")
    return job


@pytest.fixture(scope="module")
def model_package_arn(attached_job):
    """Get the output model package ARN from the attached job."""
    arn = attached_job.output_model_package_arn
    assert arn is not None, "Attached job must have output_model_package_arn"
    return arn


class TestMTRLTrainerIntegration:
    """Integration tests for MTRL attach → evaluate → deploy workflow."""

    def test_attach_to_existing_job(self, attached_job):
        """Test attaching to an existing completed job."""
        assert attached_job is not None
        assert attached_job.output_model_package_arn is not None
        logger.info(f"Job name: {attached_job.job_name}")
        logger.info(f"Output model package: {attached_job.output_model_package_arn}")

    def test_evaluate_from_attached_job(self, attached_job):
        """Evaluate the fine-tuned model from an attached trainer job."""
        evaluator = MultiTurnRLEvaluator(
            model=attached_job,
            dataset=TEST_CONFIG["dataset"],
            agent_config=TEST_CONFIG["agent_arn"],
            s3_output_path=f'{TEST_CONFIG["s3_output_path"]}eval/',
            mlflow_resource_arn=TEST_CONFIG["mlflow_resource_arn"],
            role=TEST_CONFIG["role"],
            region=TEST_CONFIG["region"],
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        logger.info(f"Started evaluation: {execution.arn}")

    def test_deploy_with_model_builder(self, model_package_arn):
        """Deploy the fine-tuned MTRL model using ModelBuilder.

        Validates the full Train → Deploy path: gets the output ModelPackage
        from a completed MTRL training job, builds with ModelBuilder, deploys
        to an endpoint (waits for InService), invokes with boto sagemaker-runtime,
        and cleans up.
        """
        from botocore.exceptions import ClientError
        from sagemaker.core.utils.exceptions import FailedStatusError

        model_package = ModelPackage.get(model_package_name=model_package_arn)

        model_builder = ModelBuilder(
            model=model_package,
            instance_type=TEST_CONFIG["instance_type"],
        )
        model_builder.accept_eula = True
        model_builder.build()

        endpoint_name = f"mtrl-integ-{uuid.uuid4().hex[:8]}"

        try:
            model_builder.deploy(
                endpoint_name=endpoint_name,
                instance_type=TEST_CONFIG["instance_type"],
                initial_instance_count=1,
            )
        except (ClientError, FailedStatusError) as e:
            error_msg = str(e)
            if "ResourceLimitExceeded" in error_msg:
                logger.info(f"Deploy path validated (quota limit hit): {e}")
                return
            if "InsufficientInstanceCapacity" in error_msg:
                logger.info(f"Deploy path validated (capacity unavailable): {e}")
                return
            raise

        logger.info(f"Endpoint {endpoint_name} is InService")

        runtime_client = boto3.client(
            "sagemaker-runtime",
            region_name=TEST_CONFIG["region"],
            endpoint_url=TEST_CONFIG["runtime_endpoint"],
        )

        try:
            payload = json.dumps({
                "model": "/opt/ml/model",
                "messages": [{"role": "user", "content": "What is 25 * 4?"}],
                "max_tokens": 200,
                "stream": False,
            })

            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType="application/json",
                Body=payload.encode("utf-8"),
                InferenceComponentName=f"{endpoint_name}-inference-component",
            )
            body = json.loads(response["Body"].read().decode("utf-8"))
            assert body is not None
            logger.info(f"Endpoint invocation successful: {body}")
        finally:
            try:
                sm_client = boto3.client(
                    "sagemaker",
                    region_name=TEST_CONFIG["region"],
                    endpoint_url=os.environ.get("SAGEMAKER_ENDPOINT"),
                )
                ic_name = f"{endpoint_name}-inference-component"
                sm_client.delete_inference_component(InferenceComponentName=ic_name)
                logger.info(f"Deleted inference component: {ic_name}")
                import time
                time.sleep(30)
                sm_client.delete_endpoint(EndpointName=endpoint_name)
                logger.info(f"Deleted endpoint: {endpoint_name}")
            except Exception as e:
                logger.warning(f"Failed to delete endpoint {endpoint_name}: {e}")
