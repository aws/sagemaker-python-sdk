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

import pytest

import time
from boto3 import Session as BotoSession
from sagemaker import get_execution_role
from sagemaker.session import Session
from sagemaker.asset import AssetManager
from sagemaker.model import ModelPackageGroup
from sagemaker.feature_store.feature_group import FeatureGroup
from botocore.exceptions import ClientError
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)
from unittest import SkipTest


@pytest.fixture
def asset_manager(
        sagemaker_session,
        domain_id,
        project_id,
        environment_id,
):
    return AssetManager(
        sagemaker_session=sagemaker_session,
        domain_id=domain_id,
        project_id=project_id,
        environment_id=environment_id,
    )


def test_export_to_inventory_model_package_group(
    asset_manager, 
    sagemaker_session,
    sagemaker_client,
    domain_exec_role_arn,
):
    mpg = ModelPackageGroup(
        name="test-asset-manager-model-package-group",
        description="my test mpg",
        sagemaker_session=sagemaker_session,
    )
    try:
        mpg.create()
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f'ModelPackage {mpg.name} already exists')

    asset = asset_manager.export_to_inventory(mpg)
    asset_manager.publish_to_catalog(asset)

    # clean up
    asset_manager.remove_from_catalog(asset)
    asset_manager.delete_asset(asset)
    mpg.delete()


def test_export_to_inventory_feature_group_with_online_offline(
    asset_manager, sagemaker_session, sagemaker_client, role
):
    fg_name = "test-asset-manager-feature-group"
    feature_group = FeatureGroup(
        name=fg_name,
        feature_definitions=[
            FeatureDefinition(feature_name="feature1", feature_type=FeatureTypeEnum.INTEGRAL),
            FeatureDefinition(feature_name="feature2", feature_type=FeatureTypeEnum.STRING),
            FeatureDefinition(feature_name="feature3", feature_type=FeatureTypeEnum.FRACTIONAL),
        ],
        sagemaker_session=sagemaker_session,
    )
    try:
        feature_group.create(
            s3_uri="s3://" + sagemaker_session.default_bucket() + f"/{fg_name}/offline-store",
            record_identifier_name="feature1",
            event_time_feature_name="feature2",
            role_arn=role,
            enable_online_store=True,
        )
    except Exception as e:
        if "Resource Already Exists" in str(e):
            pass
        else:
            raise e

    feature_group_resp = feature_group.describe()
    while feature_group_resp["FeatureGroupStatus"] == "Creating":
        time.sleep(15)
        feature_group_resp = feature_group.describe()
    if feature_group_resp["FeatureGroupStatus"] != "Created":
        raise ValueError(
            f"FeatureGroupStatus {feature_group_resp['FeatureGroupStatus'] } is invalid."
        )
    asset = asset_manager.export_to_inventory(
        resource_arn=feature_group_resp["FeatureGroupArn"]
    )
    print(asset)


def test_export_to_inventory_feature_group_with_online_only(
    asset_manager, sagemaker_session, sagemaker_client, role
):
    fg_name = "test-asset-manager-feature-group-online-only"
    feature_group = FeatureGroup(
        name=fg_name,
        feature_definitions=[
            FeatureDefinition(feature_name="feature1", feature_type=FeatureTypeEnum.INTEGRAL),
            FeatureDefinition(feature_name="feature2", feature_type=FeatureTypeEnum.STRING),
            FeatureDefinition(feature_name="feature3", feature_type=FeatureTypeEnum.FRACTIONAL),
        ],
        sagemaker_session=sagemaker_session,
    )

    try:
        feature_group.create(
            s3_uri=False,
            record_identifier_name="feature1",
            event_time_feature_name="feature2",
            role_arn=role,
            enable_online_store=True,
        )
    except Exception as e:
        if "Resource Already Exists" in str(e):
            pass
        else:
            raise e

    feature_group_resp = feature_group.describe()
    while feature_group_resp["FeatureGroupStatus"] == "Creating":
        time.sleep(15)
        feature_group_resp = feature_group.describe()

    if feature_group_resp["FeatureGroupStatus"] != "Created":
        raise ValueError(
            f"FeatureGroupStatus {feature_group_resp['FeatureGroupStatus']} is invalid."
        )

    asset = asset_manager.export_to_inventory(
        resource=feature_group,
    )
    print(asset)
