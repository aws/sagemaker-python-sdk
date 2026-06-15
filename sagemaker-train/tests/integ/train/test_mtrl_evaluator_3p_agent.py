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
"""Integration tests for MultiTurnRLEvaluator with 3P (Lambda-based) agent.

These tests validate the MTRL evaluation flow using a Lambda function as
the agent environment, which is the pattern for third-party agent
integrations (e.g., LangChain, Strands, custom EKS/Fargate agents
fronted by a Lambda adapter).

The test creates (or reuses) a Lambda forwarder that bridges RFT rollout
requests to an external agent endpoint.
"""
from __future__ import absolute_import

import io
import json
import os
import time
import zipfile
import pytest
import logging

import boto3

from sagemaker.train.evaluate import MultiTurnRLEvaluator
from sagemaker.train.custom_agent_lambda import CustomAgentLambda

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)

# Timeout for evaluation pipeline execution (4 hours)
EVALUATION_TIMEOUT_SECONDS = 14400

_REGION = "us-west-2"

# Lambda configuration
LAMBDA_FUNCTION_NAME = "SageMaker-AgentConnector-Lambda-MTRL-integ-test"
LAMBDA_RUNTIME = "python3.12"
LAMBDA_TIMEOUT = 120
LAMBDA_REGION = _REGION

# Lambda source code — bridges RFT rollout requests to an external agent.
LAMBDA_SOURCE = '''
"""
Lambda Forwarder — bridges RFT rollout requests to LangChain agent on EKS.
"""
import json
import logging
import os
import re
import urllib.error
import urllib.request

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

AGENT_ENDPOINT = os.environ.get("AGENT_ENDPOINT", "")
_SAFE_ID = re.compile(r"^[\\w\\-.]+$")


def _call_agent(prompt, metadata, inference_params):
    """Forward rollout request to the EKS-hosted LangChain agent."""
    payload = json.dumps({
        "prompt": prompt,
        "metadata": metadata,
        "inferenceParams": inference_params,
    }).encode()

    req = urllib.request.Request(
        f"{AGENT_ENDPOINT}/rollout",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _validate(event):
    body = json.loads(event["body"]) if isinstance(event.get("body"), str) else event

    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("'prompt' is required and must be a non-empty string")

    meta = body.get("metadata")
    if not isinstance(meta, dict):
        raise ValueError("'metadata' is required")

    params = body.get("inferenceParams") or {}
    if not isinstance(params, dict):
        raise ValueError("'inferenceParams' must be an object")

    return {
        "prompt": prompt.strip(),
        "metadata": meta,
        "inferenceParams": params,
    }


def _handle_agent_error(exc):
    logger.exception("Agent environment error")
    if isinstance(exc, urllib.error.HTTPError):
        code = exc.code
        if code == 403:
            return _error_response("AccessDenied", "Agent denied access")
        if code == 429:
            return _error_response("Throttling", "Agent rate limit exceeded")
        if 400 <= code < 500:
            return _error_response("ValidationError", str(exc))
        return _error_response("InternalServerError", str(exc))
    return _error_response("InternalServerError", str(exc))


def _error_response(error_type, message):
    return {"errorType": error_type, "errorMessage": message}


def handler(event, context):
    try:
        body = _validate(event)
    except ValueError as exc:
        logger.warning("Validation error: %s", exc)
        return _error_response("ValidationError", str(exc))

    try:
        result = _call_agent(body["prompt"], body["metadata"], body["inferenceParams"])
        logger.info("Rollout %s completed", body["metadata"].get("rolloutId"))
        return None
    except Exception as exc:
        return _handle_agent_error(exc)
'''

# Test configuration for 3P agent evaluation.
def _get_3p_test_config():
    """Build test configuration lazily (only when tests actually run)."""
    boto_session = boto3.Session(region_name=_REGION)
    account_id = boto_session.client("sts").get_caller_identity()["Account"]
    return {
        "base_model": "mock-oss-test",
        "dataset": os.environ.get(
            "MTRL_3P_DATASET",
            f"s3://sagemaker-rft-{account_id}/prompts/gsm8k_small/prompts.parquet",
        ),
        "s3_output_path": os.environ.get(
            "MTRL_3P_S3_OUTPUT",
            f"s3://sagemaker-{_REGION}-{account_id}/model-evaluation/3p-agent-integ/",
        ),
        "mlflow_resource_arn": os.environ.get(
            "MTRL_3P_MLFLOW_ARN",
            f"arn:aws:sagemaker:{_REGION}:{account_id}:mlflow-app/app-TTAUWUNMUHH6",
        ),
        "role": os.environ.get(
            "MTRL_3P_ROLE",
            f"arn:aws:iam::{account_id}:role/Admin",
        ),
        "region": os.environ.get("MTRL_3P_REGION", _REGION),
        "account_id": account_id,
    }


@pytest.fixture(scope="module")
def test_config():
    """Lazily resolve test configuration."""
    return _get_3p_test_config()


def _create_lambda_zip(source_code: str) -> bytes:
    """Package source code into an in-memory zip for Lambda deployment."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("lambda_function.py", source_code)
    return buf.getvalue()


def _ensure_lambda_exists(account_id) -> str:
    """Create the Lambda function if it doesn't exist, return its ARN."""
    from botocore.exceptions import ClientError

    boto_session = boto3.Session(region_name=LAMBDA_REGION)
    client = boto_session.client("lambda")
    lambda_role = f"arn:aws:iam::{account_id}:role/Admin"

    try:
        resp = client.get_function(FunctionName=LAMBDA_FUNCTION_NAME)
        arn = resp["Configuration"]["FunctionArn"]
        logger.info(f"Lambda already exists: {arn}")
        return arn
    except ClientError as e:
        if e.response["Error"]["Code"] == "AccessDeniedException":
            pytest.skip(f"No Lambda permissions: {e}")
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
    except client.exceptions.ResourceNotFoundException:
        pass

    logger.info(f"Creating Lambda function: {LAMBDA_FUNCTION_NAME}")
    zip_bytes = _create_lambda_zip(LAMBDA_SOURCE)

    resp = client.create_function(
        FunctionName=LAMBDA_FUNCTION_NAME,
        Runtime=LAMBDA_RUNTIME,
        Role=lambda_role,
        Handler="lambda_function.handler",
        Code={"ZipFile": zip_bytes},
        Timeout=LAMBDA_TIMEOUT,
        MemorySize=256,
        Environment={
            "Variables": {
                "AGENT_ENDPOINT": "",
                "LOG_LEVEL": "INFO",
            }
        },
    )

    arn = resp["FunctionArn"]
    logger.info(f"Created Lambda: {arn}")

    # Wait for function to be active
    waiter = client.get_waiter("function_active_v2")
    waiter.wait(FunctionName=LAMBDA_FUNCTION_NAME)

    return arn


@pytest.fixture(scope="module")
def lambda_agent_arn(test_config):
    """Ensure the 3P agent Lambda exists and return its ARN."""
    return _ensure_lambda_exists(test_config["account_id"])


@pytest.mark.gpu_intensive
@pytest.mark.serial
class TestMTRLEvaluator3PAgentIntegration:
    """Integration tests for MultiTurnRLEvaluator with Lambda-based 3P agent."""

    def test_evaluate_with_lambda_agent_wait_for_completion(self, lambda_agent_arn, test_config):
        """Test full end-to-end: start evaluation, wait for completion, and verify discoverability.

        This test validates the complete lifecycle including wait() using
        the standard sagemaker-core pipeline execution path, and verifies
        the evaluation is discoverable via get_all().
        """
        evaluator = MultiTurnRLEvaluator(
            model=test_config["base_model"],
            dataset=test_config["dataset"],
            agent_config=lambda_agent_arn,
            s3_output_path=f'{test_config["s3_output_path"]}lambda-e2e/',
            mlflow_resource_arn=test_config["mlflow_resource_arn"],
            role=test_config["role"],
            region=test_config["region"],
            accept_eula=True,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        assert "pipeline" in execution.arn.lower()
        logger.info(f"Started 3P agent base model evaluation: {execution.arn}")

        execution.wait(timeout=EVALUATION_TIMEOUT_SECONDS)
        assert execution.status.overall_status in ("Succeeded", "Failed", "Stopped")
        logger.info(f"Execution completed: {execution.status.overall_status}")

        if execution.status.overall_status == "Failed":
            logger.error(f"Failure reason: {execution.status.failure_reason}")

        # Verify it's discoverable via get_all
        found = False
        for ex in MultiTurnRLEvaluator.get_all(region=test_config["region"]):
            if ex.arn == execution.arn:
                found = True
                break

        assert found, (
            f"Evaluation {execution.arn} not found via get_all(). "
            "Pipeline tagging may not be working correctly."
        )
        logger.info(f"Successfully discovered evaluation via get_all: {execution.arn}")

    def test_evaluate_base_model_with_agent_lambda_object(self, lambda_agent_arn, test_config):
        """Test evaluating using an CustomAgentLambda object as agent_config.

        Validates that the evaluator accepts CustomAgentLambda instances (not
        just raw ARN strings) for the agent_config parameter.
        """
        agent = CustomAgentLambda(lambda_arn=lambda_agent_arn)

        evaluator = MultiTurnRLEvaluator(
            model=test_config["base_model"],
            dataset=test_config["dataset"],
            agent_config=agent,
            s3_output_path=f'{test_config["s3_output_path"]}lambda-object/',
            mlflow_resource_arn=test_config["mlflow_resource_arn"],
            role=test_config["role"],
            region=test_config["region"],
            accept_eula=True,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        logger.info(f"Started CustomAgentLambda object evaluation: {execution.arn}")

        execution.wait(timeout=EVALUATION_TIMEOUT_SECONDS)
        assert execution.status.overall_status == "Succeeded"

    def test_evaluate_with_attached_trainer(self, lambda_agent_arn, test_config):
        """Test evaluating a fine-tuned model by attaching to an existing training job."""
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

        attached_job = MultiTurnRLTrainer.attach(
            "mock-oss-test-mtrl-20260615143910", session=boto3.Session(region_name=_REGION)
        )

        evaluator = MultiTurnRLEvaluator(
            model=attached_job,
            dataset=test_config["dataset"],
            agent_config=lambda_agent_arn,
            s3_output_path=f'{test_config["s3_output_path"]}attached-trainer/',
            mlflow_resource_arn=test_config["mlflow_resource_arn"],
            role=test_config["role"],
            region=test_config["region"],
            accept_eula=True,
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        logger.info(f"Started attached trainer evaluation: {execution.arn}")

        execution.wait(timeout=EVALUATION_TIMEOUT_SECONDS)
        assert execution.status.overall_status == "Succeeded"
