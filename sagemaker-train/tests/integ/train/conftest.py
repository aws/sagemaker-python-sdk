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

import os
import boto3
from sagemaker.core.helper.session_helper import Session

DEFAULT_REGION = "us-west-2"


@pytest.fixture(autouse=True, scope="session")
def use_private_hub():
    os.environ["SAGEMAKER_HUB_NAME"] = "sdktest"
    yield
    del os.environ["SAGEMAKER_HUB_NAME"]


@pytest.fixture(scope="module")
def sagemaker_session():
    region = os.environ.get("AWS_DEFAULT_REGION")
    if not region:
        os.environ["AWS_DEFAULT_REGION"] = DEFAULT_REGION
        region_manual_set = True
    else:
        region_manual_set = True

    boto_session = boto3.Session(region_name=os.environ["AWS_DEFAULT_REGION"])
    sagemaker_session = Session(boto_session=boto_session)

    yield sagemaker_session

    if region_manual_set and "AWS_DEFAULT_REGION" in os.environ:
        del os.environ["AWS_DEFAULT_REGION"]


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
