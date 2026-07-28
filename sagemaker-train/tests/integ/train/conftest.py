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
"""This module contains code to test image builder"""
from __future__ import absolute_import

import pytest

import io
import json
import os
import time
import zipfile

import boto3
from botocore.exceptions import ClientError
from sagemaker.core.helper.session_helper import Session

DEFAULT_REGION = "us-west-2"

# ---------------------------------------------------------------------------
# Reward-function Lambda fixtures
#
# Instead of relying on pre-existing, hardcoded Lambda ARNs, the integration
# tests create (or reuse) the reward-function Lambdas on demand. The Lambda
# code is sourced from the local reward-function files under code/ so the
# remote and local tests exercise the exact same logic.
#
# Naming note: the Nova function name must NOT contain "sagemaker" because
# the SMHP-platform validation test relies on that check failing for Nova.
# ---------------------------------------------------------------------------
LAMBDA_EXECUTION_ROLE_NAME = "pysdk-integ-test-sm-train-reward-lambda-role"

OSS_LAMBDA_FUNCTION_NAME = "pysdk-integ-test-sm-train-oss-reward-fn"
NOVA_LAMBDA_FUNCTION_NAME = "pysdk-integ-test-sm-train-nova-reward-fn"

OSS_LOCAL_REWARD_FN = os.path.join(os.path.dirname(__file__), "code", "oss_reward_fn.py")
NOVA_LOCAL_REWARD_FN = os.path.join(os.path.dirname(__file__), "code", "nova_reward_fn.py")

LAMBDA_RUNTIME = "python3.12"
LAMBDA_ASSUME_ROLE_POLICY = {
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": {"Service": "lambda.amazonaws.com"},
            "Action": "sts:AssumeRole",
        }
    ],
}
LAMBDA_BASIC_EXECUTION_POLICY_ARN = (
    "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
)


def _ensure_lambda_execution_role():
    """Create (or reuse) an IAM role that Lambda can assume.

    Returns the role ARN. The role is left in place after the test run so
    subsequent runs can reuse it.
    """
    iam = boto3.client("iam")
    try:
        return iam.get_role(RoleName=LAMBDA_EXECUTION_ROLE_NAME)["Role"]["Arn"]
    except ClientError as e:
        if e.response["Error"]["Code"] != "NoSuchEntity":
            raise

    role_arn = iam.create_role(
        RoleName=LAMBDA_EXECUTION_ROLE_NAME,
        AssumeRolePolicyDocument=json.dumps(LAMBDA_ASSUME_ROLE_POLICY),
        Description="Auto-created execution role for pysdk sm-train reward-function integ tests",
    )["Role"]["Arn"]
    iam.attach_role_policy(
        RoleName=LAMBDA_EXECUTION_ROLE_NAME,
        PolicyArn=LAMBDA_BASIC_EXECUTION_POLICY_ARN,
    )
    # Give IAM a moment to propagate the new role before Lambda tries to assume it.
    time.sleep(15)
    return role_arn


def _zip_source(source_file, module_name):
    """Zip a single reward-function source file as <module_name>.py."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_file, arcname=f"{module_name}.py")
    return buf.getvalue()


def _ensure_lambda_function(region, function_name, source_file):
    """Create (or reuse) a reward-function Lambda from a local source file.

    Returns the function ARN. Existing functions are reused as-is so the
    Lambdas only need to be created on the first run.
    """
    lambda_client = boto3.client("lambda", region_name=region)

    try:
        return lambda_client.get_function(FunctionName=function_name)["Configuration"][
            "FunctionArn"
        ]
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    role_arn = _ensure_lambda_execution_role()
    module_name = os.path.splitext(os.path.basename(source_file))[0]
    zip_bytes = _zip_source(source_file, module_name)

    # Retry create_function because a freshly-created IAM role may not be
    # assumable by Lambda immediately (eventual consistency).
    last_err = None
    for _ in range(12):
        try:
            function_arn = lambda_client.create_function(
                FunctionName=function_name,
                Runtime=LAMBDA_RUNTIME,
                Role=role_arn,
                Handler=f"{module_name}.lambda_handler",
                Code={"ZipFile": zip_bytes},
                Timeout=60,
                MemorySize=256,
                Description="Auto-created reward function for pysdk sm-train integ tests",
            )["FunctionArn"]
            break
        except ClientError as e:
            # Role not yet propagated / assumable.
            if e.response["Error"]["Code"] == "InvalidParameterValueException":
                last_err = e
                time.sleep(5)
                continue
            raise
    else:
        raise last_err

    lambda_client.get_waiter("function_active_v2").wait(FunctionName=function_name)
    return function_arn


@pytest.fixture(autouse=True, scope="session")
def ensure_default_region():
    """Pin AWS_DEFAULT_REGION for the session so trainers/evaluators built
    without an explicit session can resolve a region under xdist, regardless
    of test execution order. Doesn't clobber an externally provided value."""
    if not os.environ.get("AWS_DEFAULT_REGION"):
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
    yield


@pytest.fixture(autouse=True, scope="session")
def use_private_hub():
    os.environ["SAGEMAKER_HUB_NAME"] = "sdktest"
    yield
    del os.environ["SAGEMAKER_HUB_NAME"]


@pytest.fixture(scope="module")
def sagemaker_session():
    # ensure_default_region (session-scoped, autouse) already guarantees
    # AWS_DEFAULT_REGION is set. Do NOT delete it on teardown: under xdist a
    # worker runs multiple modules, and clobbering the global region here made
    # later tests that build a trainer/evaluator without an explicit session
    # (SDK falls back to Session()) fail with "Must setup local AWS
    # configuration with a region supported by SageMaker".
    region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
    boto_session = boto3.Session(region_name=region)
    return Session(boto_session=boto_session)


@pytest.fixture(scope="module")
def oss_lambda_arn():
    """ARN of the OSS reward-function Lambda, created on demand if missing."""
    region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
    return _ensure_lambda_function(region, OSS_LAMBDA_FUNCTION_NAME, OSS_LOCAL_REWARD_FN)


@pytest.fixture(scope="module")
def nova_lambda_arn():
    """ARN of the Nova reward-function Lambda, created on demand if missing."""
    region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
    return _ensure_lambda_function(region, NOVA_LAMBDA_FUNCTION_NAME, NOVA_LOCAL_REWARD_FN)


NOVA_REGION = "us-east-1"


@pytest.fixture(scope="module")
def sagemaker_session_us_east_1():
    """Create a SageMaker session in us-east-1 for Nova model tests."""
    boto_session = boto3.Session(region_name=NOVA_REGION)
    return Session(boto_session=boto_session)


import time
import logging

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def mlflow_resource_arn():
    """Discover or create an MLflow app for integ tests, clean up if created.

    Looks for an existing MLflow app in Created/Updated state. If none exists,
    creates one and deletes it after the test module finishes.
    """
    region = os.environ.get("AWS_DEFAULT_REGION", DEFAULT_REGION)
    sm_client = boto3.client("sagemaker", region_name=region)
    created_arn = None

    # Try to find an existing ready app
    try:
        paginator = sm_client.get_paginator("list_mlflow_apps")
        for page in paginator.paginate():
            for app in page.get("Summaries", []):
                if app.get("Status") in ("Created", "Updated"):
                    logger.info(f"Using existing MLflow app: {app['Arn']}")
                    yield app["Arn"]
                    return
    except Exception as e:
        logger.warning(f"Failed to list MLflow apps: {e}")

    # No ready app found — create one
    logger.info("No ready MLflow app found. Creating one for integ tests...")
    sts_client = boto3.client("sts", region_name=region)
    account_id = sts_client.get_caller_identity()["Account"]
    app_name = f"integ-test-mlflow-{int(time.time())}"
    artifact_store_uri = f"s3://sagemaker-{region}-{account_id}/mlflow-artifacts"

    # Ensure bucket/prefix exists
    s3_client = boto3.client("s3", region_name=region)
    bucket_name = f"sagemaker-{region}-{account_id}"
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except Exception:
        if region == "us-east-1":
            s3_client.create_bucket(Bucket=bucket_name)
        else:
            s3_client.create_bucket(
                Bucket=bucket_name,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
    try:
        s3_client.put_object(Bucket=bucket_name, Key="mlflow-artifacts/")
    except Exception:
        pass

    # Get execution role
    from sagemaker.train.defaults import TrainDefaults
    boto_session = boto3.Session(region_name=region)
    sagemaker_session = Session(boto_session=boto_session)
    role_arn = TrainDefaults.get_role(role=None, sagemaker_session=sagemaker_session)

    resp = sm_client.create_mlflow_app(
        Name=app_name,
        ArtifactStoreUri=artifact_store_uri,
        RoleArn=role_arn,
        AccountDefaultStatus="DISABLED",
    )
    created_arn = resp["Arn"]
    logger.info(f"Created MLflow app: {created_arn}")

    # Wait for it to become ready
    for _ in range(60):
        desc = sm_client.describe_mlflow_app(Arn=created_arn)
        status = desc.get("Status")
        if status in ("Created", "Updated"):
            break
        if status in ("Failed", "CreateFailed", "DeleteFailed"):
            pytest.skip(f"MLflow app creation failed: {desc.get('FailureReason')}")
        time.sleep(10)

    yield created_arn

    # Cleanup
    logger.info(f"Cleaning up MLflow app: {created_arn}")
    try:
        sm_client.delete_mlflow_app(Arn=created_arn)
    except Exception as e:
        logger.warning(f"Failed to delete MLflow app {created_arn}: {e}")
