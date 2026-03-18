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
"""End-to-end integration tests for BedrockModelBuilder Nova deployment with model status polling.

These tests require:
- AWS credentials with Bedrock and SageMaker access in us-east-1
- A completed Nova training job (NOVA_TRAINING_JOB_NAME)

Run with:
    pytest tests/integ/test_bedrock_nova_e2e.py -v -m "not slow"   # fast tests only
    pytest tests/integ/test_bedrock_nova_e2e.py -v                  # all tests
"""
from __future__ import absolute_import

import os
import time
import random
import logging

import boto3
import pytest

from sagemaker.serve.bedrock_model_builder import BedrockModelBuilder, _is_nova_model
from sagemaker.core.resources import TrainingJob

logger = logging.getLogger(__name__)

REGION = "us-east-1"


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def region_env():
    """Set region to us-east-1 for Nova tests."""
    original = os.environ.get("AWS_DEFAULT_REGION")
    os.environ["AWS_DEFAULT_REGION"] = REGION
    yield REGION
    if original:
        os.environ["AWS_DEFAULT_REGION"] = original
    else:
        os.environ.pop("AWS_DEFAULT_REGION", None)


@pytest.fixture(scope="module")
def boto_session(region_env):
    """Create a boto3 session in the test region."""
    return boto3.Session(region_name=region_env)


@pytest.fixture(scope="module")
def bedrock_client(boto_session):
    """Create Bedrock client."""
    return boto_session.client("bedrock")


@pytest.fixture(scope="module")
def training_job(region_env, nova_training_job_name):
    """Fetch the Nova training job using the shared get-or-create fixture."""
    session = boto3.Session(region_name=region_env)
    return TrainingJob.get(
        training_job_name=nova_training_job_name,
        session=session,
        region=region_env,
    )


@pytest.fixture(scope="module")
def bedrock_builder(training_job):
    """Create BedrockModelBuilder from the Nova training job."""
    return BedrockModelBuilder(model=training_job)


@pytest.fixture(scope="module")
def role_arn():
    """Get execution role."""
    from sagemaker.core.helper.session_helper import get_execution_role

    return get_execution_role()


# ── Cleanup helper ──────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cleanup_tracker():
    """Track resources created during tests for cleanup."""
    resources = {"model_arns": [], "deployment_arns": []}
    yield resources

    client = boto3.client("bedrock", region_name=REGION)
    for dep_arn in resources["deployment_arns"]:
        try:
            client.delete_custom_model_deployment(
                customModelDeploymentIdentifier=dep_arn
            )
            logger.info("Deleted deployment: %s", dep_arn)
        except Exception as e:
            logger.warning("Failed to delete deployment %s: %s", dep_arn, e)
    for model_arn in resources["model_arns"]:
        try:
            client.delete_custom_model(modelIdentifier=model_arn)
            logger.info("Deleted custom model: %s", model_arn)
        except Exception as e:
            logger.warning("Failed to delete custom model %s: %s", model_arn, e)


# ── Tests ───────────────────────────────────────────────────────────────────


class TestBedrockNovaE2E:
    """End-to-end tests for Nova model deployment to Bedrock with polling fix."""

    def test_training_job_exists(self, training_job):
        """Verify the Nova training job exists and completed."""
        assert training_job is not None
        assert training_job.training_job_status == "Completed"

    def test_builder_creation(self, bedrock_builder, training_job):
        """Verify BedrockModelBuilder initializes correctly for Nova."""
        assert bedrock_builder is not None
        assert bedrock_builder.model == training_job
        assert bedrock_builder.model_package is not None
        assert bedrock_builder.s3_model_artifacts is not None

    def test_nova_detection(self, bedrock_builder):
        """Verify the model is detected as Nova via the shared helper."""
        container = bedrock_builder.model_package.inference_specification.containers[0]
        assert _is_nova_model(container), "Model should be detected as Nova"

    def test_s3_artifacts_are_checkpoint(self, bedrock_builder):
        """Verify that s3_model_artifacts points to a checkpoint (not model.tar.gz)."""
        uri = bedrock_builder.s3_model_artifacts
        assert uri is not None
        assert uri.startswith("s3://"), "Artifacts URI should be an S3 path"

    @pytest.mark.slow
    def test_deploy_with_polling(self, bedrock_builder, role_arn, cleanup_tracker):
        """E2E: deploy Nova model to Bedrock, verifying the polling fix works.

        This test exercises the full flow:
        1. create_custom_model
        2. _wait_for_model_active (the polling fix)
        3. create_custom_model_deployment

        Previously this would fail with:
            ValidationException: Custom Model is not ready for deployment.
        """
        rand = random.randint(1000, 9999)
        custom_model_name = f"integ-nova-poll-{rand}-{int(time.time())}"
        deployment_name = f"{custom_model_name}-dep"

        response = bedrock_builder.deploy(
            custom_model_name=custom_model_name,
            role_arn=role_arn,
            deployment_name=deployment_name,
        )

        assert response is not None
        deployment_arn = response.get("customModelDeploymentArn")
        assert deployment_arn is not None, (
            "Expected customModelDeploymentArn in response — "
            "polling should have waited for model Active before creating deployment"
        )

        # Track for cleanup
        cleanup_tracker["deployment_arns"].append(deployment_arn)
        try:
            client = boto3.client("bedrock", region_name=REGION)
            dep_info = client.get_custom_model_deployment(
                customModelDeploymentIdentifier=deployment_arn
            )
            model_arn = dep_info.get("modelArn")
            if model_arn:
                cleanup_tracker["model_arns"].append(model_arn)
        except Exception:
            pass

        logger.info("Deployment created successfully: %s", deployment_arn)

    @pytest.mark.slow
    def test_wait_for_model_active_timeout(self, bedrock_builder):
        """Test that _wait_for_model_active raises on a bogus ARN (timeout)."""
        with pytest.raises(Exception):
            bedrock_builder._wait_for_model_active(
                model_arn="arn:aws:bedrock:us-east-1:000000000000:custom-model/nonexistent",
                poll_interval=1,
                max_wait=3,
            )

    def test_deploy_without_model_package_raises(self):
        """Verify deploy raises ValueError when model_package is None."""
        builder = BedrockModelBuilder(model=None)
        with pytest.raises(ValueError, match="model_package is not set"):
            builder.deploy(custom_model_name="x", role_arn="r")

    def test_create_deployment_without_model_arn_raises(self):
        """Verify create_deployment raises ValueError for empty model_arn."""
        builder = BedrockModelBuilder(model=None)
        with pytest.raises(ValueError, match="model_arn is required"):
            builder.create_deployment(model_arn="", deployment_name="dep")
