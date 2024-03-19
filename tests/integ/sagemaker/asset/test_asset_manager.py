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
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)


@pytest.fixture
def sagemaker_session():
    return Session(boto_session=BotoSession(region_name="us-east-2"))


@pytest.fixture
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture
def sagemaker_client(sagemaker_session):
    return sagemaker_session.sagemaker_client


@pytest.fixture
def asset_manager(sagemaker_session):
    return AssetManager(
        session=sagemaker_session,
        domain_id="dzd_aqnftwmv8a82w9",
        owning_project_id="bvh4ldxou5mt2h",
        environment_id="ci4w9p3s991fhl",
    )


def test_export_to_inventory_model_package_group(
    asset_manager, sagemaker_session, sagemaker_client
):
    mpg = ModelPackageGroup(
        name="test-asset-manager-model-package-group",
        description="my test mpg",
        sagemaker_session=sagemaker_session,
    )
    mpg.create()

    asset = asset_manager.export_to_inventory(mpg)
    asset_manager.publish_to_catalog(asset)
    # clean up
    asset_manager.remove_from_catalog(asset)
    asset_manager.delete_asset(asset)


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
    feature_group.create(
        s3_uri="s3://" + sagemaker_session.default_bucket() + f"/{fg_name}/offline-store",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn="arn:aws:iam::212136391372:role/service-role/AmazonSageMaker-ExecutionRole-20231208T123997",
        enable_online_store=True,
    )
    feature_group_resp = feature_group.describe()
    while feature_group_resp["FeatureGroupStatus"] == "Creating":
        time.sleep(15)
        feature_group_resp = feature_group.describe()
    if feature_group_resp["FeatureGroupStatus"] != "Created":
        raise ValueError(
            f"FeatureGroupStatus {feature_group_resp['FeatureGroupStatus'] } is invalid."
        )
    asset = asset_manager.export_to_inventory(
        resource_arn="arn:aws:sagemaker:us-east-2:212136391372:feature-group/test-asset-manager-feature-group"
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
            role_arn="arn:aws:iam::212136391372:role/service-role/AmazonSageMaker-ExecutionRole-20231208T123997",
            enable_online_store=True,
        )
    except Exception as e:
        if "Resource Already Exists" in str(e):
            pass

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
