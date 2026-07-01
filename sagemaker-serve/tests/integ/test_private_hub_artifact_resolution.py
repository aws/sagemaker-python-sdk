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
"""Integration test for private hub artifact resolution.

Verifies that ModelBuilder.from_jumpstart_config with hub_name resolves
model artifacts through the private hub content reference, rather than
falling back to the public JumpStart S3 cache.

This test creates its own private hub, adds a ModelReference, runs the
build flow, and tears everything down afterward.
"""
from __future__ import absolute_import

import uuid
import time
import logging

import boto3
import pytest

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# Use a small/fast JumpStart model for the test
TEST_MODEL_ID = "huggingface-llm-gemma-2b"
TEST_MODEL_VERSION = "*"
TEST_INSTANCE_TYPE = "ml.g5.xlarge"
TEST_REGION = "us-east-1"

HUB_NAME_PREFIX = "sdk-integ-test-private-hub"


@pytest.fixture(scope="module")
def private_hub():
    """Create a private hub with a ModelReference, tear down after tests."""
    sm = boto3.client("sagemaker", region_name=TEST_REGION)
    hub_name = f"{HUB_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"

    # --- Setup ---
    logger.info("Creating private hub: %s", hub_name)
    sm.create_hub(
        HubName=hub_name,
        HubDescription="SDK integration test for private hub artifact resolution",
        Tags=[
            {"Key": "Purpose", "Value": "sdk-integ-test"},
            {"Key": "AutoCleanup", "Value": "true"},
        ],
    )

    # Wait for hub to be ready
    for _ in range(30):
        resp = sm.describe_hub(HubName=hub_name)
        if resp["HubStatus"] == "InService":
            break
        time.sleep(2)
    else:
        pytest.fail(f"Hub {hub_name} did not reach InService state")

    # Add a ModelReference to the public JumpStart content
    public_arn = (
        f"arn:aws:sagemaker:{TEST_REGION}:aws:hub-content/"
        f"SageMakerPublicHub/Model/{TEST_MODEL_ID}"
    )
    logger.info("Creating hub content reference to: %s", public_arn)
    sm.create_hub_content_reference(
        HubName=hub_name,
        SageMakerPublicHubContentArn=public_arn,
    )

    # Wait for content reference to be available
    for _ in range(60):
        try:
            contents = sm.list_hub_contents(
                HubName=hub_name,
                HubContentType="ModelReference",
            )
            summaries = contents.get("HubContentSummaries", [])
            if any(
                s["HubContentName"] == TEST_MODEL_ID
                and s.get("HubContentStatus") == "Available"
                for s in summaries
            ):
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        pytest.fail(
            f"ModelReference for {TEST_MODEL_ID} not available in "
            f"hub {hub_name} after 3 minutes"
        )

    yield hub_name

    # --- Teardown ---
    logger.info("Cleaning up hub: %s", hub_name)
    try:
        # Delete hub content references first
        contents = sm.list_hub_contents(
            HubName=hub_name,
            HubContentType="ModelReference",
        )
        for content in contents.get("HubContentSummaries", []):
            try:
                sm.delete_hub_content_reference(
                    HubName=hub_name,
                    HubContentName=content["HubContentName"],
                    HubContentType="ModelReference",
                )
            except Exception as e:
                logger.warning("Failed to delete content ref: %s", e)

        # Delete the hub
        sm.delete_hub(HubName=hub_name)
        logger.info("Hub %s deleted", hub_name)
    except Exception as e:
        logger.warning("Hub cleanup failed: %s", e)


@pytest.fixture(scope="module")
def sagemaker_session():
    """Create a SageMaker session using default credentials."""
    from sagemaker.core.helper.session_helper import Session

    return Session(boto_session=boto3.Session(region_name=TEST_REGION))


@pytest.fixture(scope="module")
def execution_role():
    """Get a SageMaker execution role from the caller's identity."""
    sts = boto3.client("sts", region_name=TEST_REGION)
    identity = sts.get_caller_identity()
    account_id = identity["Account"]
    # Use the standard SageMaker execution role naming convention
    return f"arn:aws:iam::{account_id}:role/Admin"


@pytest.mark.slow_test
def test_from_jumpstart_config_derives_hub_arn(private_hub, sagemaker_session):
    """Verify from_jumpstart_config correctly derives hub_arn from hub_name."""
    js_config = JumpStartConfig(
        model_id=TEST_MODEL_ID,
        model_version=TEST_MODEL_VERSION,
        hub_name=private_hub,
    )

    mb = ModelBuilder.from_jumpstart_config(
        jumpstart_config=js_config,
        compute=Compute(instance_type=TEST_INSTANCE_TYPE),
        sagemaker_session=sagemaker_session,
    )

    assert mb.hub_arn is not None, (
        f"hub_arn is None after from_jumpstart_config with "
        f"hub_name={private_hub}"
    )
    assert private_hub in mb.hub_arn
    logger.info("hub_arn correctly derived: %s", mb.hub_arn)


@pytest.mark.slow_test
def test_build_resolves_artifacts_via_private_hub(
    private_hub, execution_role, sagemaker_session
):
    """Verify build() resolves model data through the private hub."""
    js_config = JumpStartConfig(
        model_id=TEST_MODEL_ID,
        model_version=TEST_MODEL_VERSION,
        hub_name=private_hub,
        accept_eula=True,
    )

    mb = ModelBuilder.from_jumpstart_config(
        jumpstart_config=js_config,
        role_arn=execution_role,
        compute=Compute(instance_type=TEST_INSTANCE_TYPE),
        sagemaker_session=sagemaker_session,
    )

    mb.build()

    # After build, model data URI must NOT reference the public cache
    model_data = getattr(mb, "s3_model_data_url", None)
    assert model_data is not None, "No model_data found after build()"

    model_data_str = str(model_data)
    assert "jumpstart-cache-prod" not in model_data_str, (
        f"Model data still points to public JumpStart cache: "
        f"{model_data_str}. Expected private hub artifact resolution."
    )
    logger.info("Model data resolved to: %s", model_data_str)
