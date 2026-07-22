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
build flow, and tears everything down afterward. Skips gracefully if
the test environment lacks permissions to create hubs.
"""
from __future__ import absolute_import

import os
import uuid
import time
import logging

import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.train.configs import Compute

logger = logging.getLogger(__name__)

# Use a small/fast JumpStart model for the test
TEST_MODEL_ID = "huggingface-llm-gemma-2b"
TEST_MODEL_VERSION = "*"
TEST_INSTANCE_TYPE = "ml.g5.xlarge"
TEST_REGION = os.environ.get("TEST_REGION", "us-east-1")

HUB_NAME_PREFIX = "sdk-integ-test-private-hub"


@pytest.fixture(scope="module")
def private_hub():
    """Create a private hub with a ModelReference, tear down after tests."""
    sm = boto3.client("sagemaker", region_name=TEST_REGION)
    hub_name = f"{HUB_NAME_PREFIX}-{uuid.uuid4().hex[:8]}"

    # --- Setup ---
    logger.info("Creating private hub: %s", hub_name)
    try:
        sm.create_hub(
            HubName=hub_name,
            HubDescription="SDK integration test for private hub artifact resolution",
            Tags=[
                {"Key": "Purpose", "Value": "sdk-integ-test"},
                {"Key": "AutoCleanup", "Value": "true"},
            ],
        )
    except ClientError as e:
        pytest.skip(f"Cannot create hub (likely missing permissions): {e}")

    # Wait for hub to be ready
    for _ in range(30):
        resp = sm.describe_hub(HubName=hub_name)
        if resp["HubStatus"] == "InService":
            break
        time.sleep(2)
    else:
        pytest.skip(f"Hub {hub_name} did not reach InService state")

    # Add a ModelReference to the public JumpStart content
    public_arn = (
        f"arn:aws:sagemaker:{TEST_REGION}:aws:hub-content/"
        f"SageMakerPublicHub/Model/{TEST_MODEL_ID}"
    )
    logger.info("Creating hub content reference to: %s", public_arn)
    try:
        sm.create_hub_content_reference(
            HubName=hub_name,
            SageMakerPublicHubContentArn=public_arn,
        )
    except ClientError as e:
        pytest.skip(f"Cannot create hub content reference: {e}")

    # Wait for content reference to be available
    for _ in range(60):
        try:
            contents = sm.list_hub_contents(
                HubName=hub_name,
                HubContentType="ModelReference",
            )
            summaries = contents.get("HubContentSummaries", [])
            if any(
                s["HubContentName"] == TEST_MODEL_ID and s.get("HubContentStatus") == "Available"
                for s in summaries
            ):
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        pytest.skip(
            f"ModelReference for {TEST_MODEL_ID} not available in "
            f"hub {hub_name} after 3 minutes"
        )

    yield hub_name

    # --- Teardown ---
    logger.info("Cleaning up hub: %s", hub_name)
    try:
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

    assert (
        mb.hub_arn is not None
    ), f"hub_arn is None after from_jumpstart_config with hub_name={private_hub}"
    assert private_hub in mb.hub_arn
    logger.info("hub_arn correctly derived: %s", mb.hub_arn)


@pytest.mark.slow_test
def test_build_resolves_artifacts_via_private_hub(private_hub, execution_role, sagemaker_session):
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


ALIASED_CONTENT_NAME = "sdk-integ-aliased-phi4-mini"
NO_S3_ROLE_PREFIX = "sdk-integ-no-s3-role"

_NO_S3_ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ecr:GetAuthorizationToken",
                "ecr:BatchCheckLayerAvailability",
                "ecr:GetDownloadUrlForLayer",
                "ecr:BatchGetImage",
            ],
            "Resource": "*",
        },
        {
            "Effect": "Allow",
            "Action": [
                "logs:CreateLogGroup",
                "logs:CreateLogStream",
                "logs:PutLogEvents",
            ],
            "Resource": "*",
        },
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:DescribeHub",
                "sagemaker:DescribeHubContent",
                "sagemaker:ListHubContents",
                "sagemaker:ListHubContentVersions",
            ],
            "Resource": "*",
        },
    ],
}


@pytest.fixture(scope="module")
def no_s3_execution_role():
    """Create an execution role with NO S3 permissions whatsoever.

    This encodes the customer-visible contract of private hub brokered
    access: with HubAccessConfig in the CreateModel call, SageMaker
    brokers model data access through the hub content reference, so the
    execution role needs no s3:GetObject on the public JumpStart cache.
    """
    import json

    iam = boto3.client("iam", region_name=TEST_REGION)
    role_name = f"{NO_S3_ROLE_PREFIX}-{uuid.uuid4().hex[:8]}"
    trust = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "sagemaker.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }
    try:
        resp = iam.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust),
            Description="SDK integ test: private hub deploy with zero S3 access",
        )
        iam.put_role_policy(
            RoleName=role_name,
            PolicyName="minimal-no-s3",
            PolicyDocument=json.dumps(_NO_S3_ROLE_POLICY),
        )
    except ClientError as e:
        pytest.skip(f"Cannot create IAM role (likely missing permissions): {e}")

    time.sleep(15)  # IAM propagation
    yield resp["Role"]["Arn"]

    try:
        iam.delete_role_policy(RoleName=role_name, PolicyName="minimal-no-s3")
        iam.delete_role(RoleName=role_name)
    except Exception as e:
        logger.warning("Role cleanup failed: %s", e)


@pytest.fixture(scope="module")
def aliased_model_reference(private_hub):
    """Add a ModelReference whose HubContentName differs from the model_id."""
    sm = boto3.client("sagemaker", region_name=TEST_REGION)
    public_arn = (
        f"arn:aws:sagemaker:{TEST_REGION}:aws:hub-content/"
        f"SageMakerPublicHub/Model/{TEST_MODEL_ID}"
    )
    try:
        sm.create_hub_content_reference(
            HubName=private_hub,
            SageMakerPublicHubContentArn=public_arn,
            HubContentName=ALIASED_CONTENT_NAME,
        )
    except ClientError as e:
        pytest.skip(f"Cannot create aliased hub content reference: {e}")

    for _ in range(60):
        try:
            contents = sm.list_hub_contents(
                HubName=private_hub, HubContentType="ModelReference"
            )
            if any(
                s["HubContentName"] == ALIASED_CONTENT_NAME
                and s.get("HubContentStatus") == "Available"
                for s in contents.get("HubContentSummaries", [])
            ):
                break
        except Exception:
            pass
        time.sleep(3)
    else:
        pytest.skip(f"Aliased reference {ALIASED_CONTENT_NAME} not available")

    yield ALIASED_CONTENT_NAME
    # Teardown handled by the private_hub fixture (deletes all references).


def _deploy_and_assert_hub_access_config(
    hub_name, role_arn, sagemaker_session, hub_content_name=None
):
    """Build + deploy with the given role; assert HubAccessConfig on the model.

    Returns after cleaning up the endpoint, endpoint config, and model.
    """
    suffix = uuid.uuid4().hex[:8]
    endpoint_name = f"sdk-integ-private-hub-{suffix}"
    from sagemaker.train.configs import Compute

    config_kwargs = dict(
        model_id=TEST_MODEL_ID,
        model_version=TEST_MODEL_VERSION,
        hub_name=hub_name,
        accept_eula=True,
    )
    if hub_content_name:
        config_kwargs["hub_content_name"] = hub_content_name

    mb = ModelBuilder.from_jumpstart_config(
        jumpstart_config=JumpStartConfig(**config_kwargs),
        compute=Compute(instance_type=TEST_INSTANCE_TYPE),
        sagemaker_session=sagemaker_session,
        role_arn=role_arn,
    )

    sm = boto3.client("sagemaker", region_name=TEST_REGION)
    try:
        mb.build(sagemaker_session=sagemaker_session)
        # deploy(wait=False): CreateModel is the call under test. It fails
        # synchronously without HubAccessConfig when the role has no S3 access.
        mb.deploy(endpoint_name=endpoint_name, wait=False)

        # Assert the created Model resource carries HubAccessConfig
        endpoint = sm.describe_endpoint(EndpointName=endpoint_name)
        ep_config = sm.describe_endpoint_config(
            EndpointConfigName=endpoint["EndpointConfigName"]
        )
        model_name = ep_config["ProductionVariants"][0]["ModelName"]
        model = sm.describe_model(ModelName=model_name)
        container = model.get("PrimaryContainer") or model["Containers"][0]
        hub_access = (
            container.get("ModelDataSource", {})
            .get("S3DataSource", {})
            .get("HubAccessConfig")
        )
        assert hub_access is not None, (
            "CreateModel succeeded but the model has no "
            "S3DataSource.HubAccessConfig; private hub access was not brokered"
        )
        assert hub_name in hub_access["HubContentArn"]
    finally:
        for op, kwargs in (
            (sm.delete_endpoint, {"EndpointName": endpoint_name}),
            (sm.delete_endpoint_config, {"EndpointConfigName": endpoint_name}),
        ):
            try:
                op(**kwargs)
            except Exception as e:
                logger.warning("Cleanup failed for %s: %s", kwargs, e)


@pytest.mark.slow_test
def test_deploy_with_no_s3_execution_role(
    private_hub, no_s3_execution_role, sagemaker_session
):
    """E2E: deploy from a private hub with an execution role that has ZERO
    S3 permissions. Passes only when the SDK attaches HubAccessConfig to
    the CreateModel call (SageMaker brokers artifact access via the hub).

    Verified against the v3.16.0 baseline: fails at the HubAccessConfig
    assertion (the created Model resource has no
    S3DataSource.HubAccessConfig, so artifact access is not brokered and
    the endpoint cannot serve without public-bucket S3 permissions).
    """
    _deploy_and_assert_hub_access_config(
        hub_name=private_hub,
        role_arn=no_s3_execution_role,
        sagemaker_session=sagemaker_session,
    )


@pytest.mark.slow_test
def test_deploy_with_aliased_hub_content_name(
    private_hub, aliased_model_reference, no_s3_execution_role, sagemaker_session
):
    """E2E: deploy a ModelReference whose HubContentName differs from the
    public model_id, using JumpStartConfig.hub_content_name.

    On v3.16.0 this fails at hub content resolution with ResourceNotFound
    because the SDK only looks up hub content by model_id.
    """
    _deploy_and_assert_hub_access_config(
        hub_name=private_hub,
        role_arn=no_s3_execution_role,
        sagemaker_session=sagemaker_session,
        hub_content_name=aliased_model_reference,
    )
