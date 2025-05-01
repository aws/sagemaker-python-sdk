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
from __future__ import absolute_import

import boto3
from botocore.config import Config

from sagemaker import Session, ModelPackage
from sagemaker.utils import unique_name_from_base

CUSTOM_BUCKET_NAME = "this-bucket-should-not-exist"


def test_sagemaker_session_does_not_create_bucket_on_init(
    sagemaker_client_config, sagemaker_runtime_config, boto_session
):
    sagemaker_client_config.setdefault("config", Config(retries=dict(max_attempts=10)))
    sagemaker_client = (
        boto_session.client("sagemaker", **sagemaker_client_config)
        if sagemaker_client_config
        else None
    )
    runtime_client = (
        boto_session.client("sagemaker-runtime", **sagemaker_runtime_config)
        if sagemaker_runtime_config
        else None
    )

    Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=CUSTOM_BUCKET_NAME,
    )

    s3 = boto3.resource("s3", region_name=boto_session.region_name)
    assert s3.Bucket(CUSTOM_BUCKET_NAME).creation_date is None


def test_sagemaker_session_to_return_most_recent_approved_model_package(sagemaker_session):
    model_package_group_name = unique_name_from_base("test-model-package-group")
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is None
    sagemaker_session.sagemaker_client.create_model_package_group(
        ModelPackageGroupName=model_package_group_name
    )
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is None
    source_uri = "dummy source uri"
    model_package = sagemaker_session.sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name, SourceUri=source_uri
    )
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is None
    ModelPackage(
        sagemaker_session=sagemaker_session,
        model_package_arn=model_package["ModelPackageArn"],
    ).update_approval_status(approval_status="Approved")
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is not None
    assert approved_model_package.model_package_arn == model_package.get("ModelPackageArn")
    model_package_2 = sagemaker_session.sagemaker_client.create_model_package(
        ModelPackageGroupName=model_package_group_name, SourceUri=source_uri
    )
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is not None
    assert approved_model_package.model_package_arn == model_package.get("ModelPackageArn")
    ModelPackage(
        sagemaker_session=sagemaker_session,
        model_package_arn=model_package_2["ModelPackageArn"],
    ).update_approval_status(approval_status="Approved")
    approved_model_package = sagemaker_session.get_most_recently_created_approved_model_package(
        model_package_group_name=model_package_group_name
    )
    assert approved_model_package is not None
    assert approved_model_package.model_package_arn == model_package_2.get("ModelPackageArn")

    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package_2["ModelPackageArn"]
    )
    sagemaker_session.sagemaker_client.delete_model_package(
        ModelPackageName=model_package["ModelPackageArn"]
    )
    sagemaker_session.sagemaker_client.delete_model_package_group(
        ModelPackageGroupName=model_package_group_name
    )
