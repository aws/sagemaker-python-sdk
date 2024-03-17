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

import datetime
import json

import pytest
from mock import MagicMock, Mock
from sagemaker.asset import Asset, AssetTypeIdentifier
from sagemaker.asset.asset_builder import (
    AssetBuilder,
    ModelPackageGroupAssetBuilder,
    FeatureGroupAssetBuilder,
)
from sagemaker.feature_store.feature_definition import FeatureTypeEnum, CollectionTypeEnum




@pytest.fixture
def boto_session(sagemaker_client, datazone_client, lf_client):
    boto_mock = Mock(name="boto_session", region_name="us-west-2")

    def get_clients(*args, **kwargs):
        if args[0] == "datazone":
            return datazone_client
        elif args[0] == "sagemaker":
            return sagemaker_client
        elif args[0] == "lakeformation":
            return lf_client
        else:
            raise ValueError(f"Unexpected client type: {args[0]}")

    boto_mock.client = MagicMock(name="boto_client", side_effect=get_clients)
    return boto_mock

@pytest.fixture
def sagemaker_session(boto_session, sagemaker_client):
    sm_session_mock = Mock(name="session", boto_session=boto_session, sagemaker_client=sagemaker_client)
    return sm_session_mock


def test_build_model_package_group_asset(sagemaker_session):
    mpg_arn = "arn:aws:sagemaker:us-east-2:1234567890:model-package-group/my-mpg"
    mpg_name = "my-mpg"
    mpg_asset_builder = ModelPackageGroupAssetBuilder(sagemaker_session)
    now = datetime.datetime.now()

    describe_resp = {
        "ModelPackageGroupName": mpg_name,
        "ModelPackageGroupArn": mpg_arn,
        "ModelPackageGroupStatus": "COMPLETED",
        "ModelPackageGroupDescription": "my good model package group",
        "CreationTime": now,
    }

    sagemaker_session.sagemaker_client.describe_model_package_group.return_value = dict(describe_resp)

    mpg_form = {
        "modelPackageGroupName": describe_resp["ModelPackageGroupName"],
        "modelPackageGroupArn": describe_resp["ModelPackageGroupArn"],
        "creationTime": AssetBuilder.convert_date_time(describe_resp["CreationTime"]),
        "modelPackageGroupStatus": describe_resp["ModelPackageGroupStatus"],
        "modelPackageGroupDescription": describe_resp["ModelPackageGroupDescription"],
    }

    mpg_asset = Asset(
        name=mpg_name,
        type_id=AssetTypeIdentifier.SageMakerModelPackageGroupAssetType,
        external_id=mpg_arn,
        forms_input=[
            {
                "formName": "SageMakerModelPackageGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerModelPackageGroupFormType",
                "content": json.dumps(mpg_form),
            }
        ],
    )
    asset = mpg_asset_builder.build_asset(mpg_arn)
    assert asset == mpg_asset


def test_build_feature_group_asset(
    domain_id,
    project_id,
    environment_id,
    sagemaker_session,
    datazone_client,
    boto_session,
    sagemaker_client,
):
    fg_arn = "arn:aws:sagemaker:us-east-2:1234567890:feature-group/my-fg"
    offline_store_s3_location = "s3://my-bucket/my-offline-store"
    offline_store_asset_id = "offline_store_asset_id"
    fg_name = "my-fg"
    fg_asset_builder = FeatureGroupAssetBuilder(
        sagemaker_session,
        domain_id,
        project_id,
        environment_id,
        datazone_client,
    )
    now = datetime.datetime.now()

    describe_resp = {
        "FeatureGroupName": fg_name,
        "FeatureGroupArn": fg_arn,
        "FeatureGroupStatus": "Created",
        "RecordIdentifierFeatureName": "record_id",
        "EventTimeFeatureName": "event_time",
        "CreationTime": now,
        "LastModifiedTime": now,
        "FeatureDefinitions": [
            {
                "FeatureName": "feature_1",
                "FeatureType": FeatureTypeEnum.INTEGRAL.value,
            },
            {
                "FeatureName": "feature_2",
                "FeatureType": FeatureTypeEnum.FRACTIONAL.value,
            },
            {
                "FeatureName": "feature_3",
                "FeatureType": FeatureTypeEnum.STRING.value,
            },
            {
                "FeatureName": "feature_4",
                "FeatureType": FeatureTypeEnum.STRING.value,
                "CollectionType": CollectionTypeEnum.LIST.value,
            },
        ],
        "OfflineStoreConfig": {
            "DataCatalogConfig": {
                "Catalog": "catalog",
                "Database": "database",
                "TableName": "table",
            },
            "S3StorageConfig": {
                "ResolvedOutputS3Uri": offline_store_s3_location,
            },
        },
        "OfflineStoreStatus": {
            "BlockedReason": "",
            "Status": "Active"
        }
    }

    fg_forms = {
        "featureGroupName": describe_resp["FeatureGroupName"],
        "featureGroupArn": describe_resp["FeatureGroupArn"],
        "recordIdentifierFeatureName": describe_resp["RecordIdentifierFeatureName"],
        "eventTimeFeatureName": describe_resp["EventTimeFeatureName"],
        "creationTime": AssetBuilder.convert_date_time(describe_resp["CreationTime"]),
        "lastModifiedTime": AssetBuilder.convert_date_time(describe_resp["LastModifiedTime"]),
        "offlineStoreStatus": "ACTIVE",
        "featureDefinitions": [
            {
                "featureName": "feature_1",
                "featureType": "INTEGRAL",
            },
            {
                "featureName": "feature_2",
                "featureType": "FRACTIONAL",
            },
            {"featureName": "feature_3", "featureType": "STRING"},
            {
                "featureName": "feature_4",
                "featureType": "STRING",
                "collectionType": "LIST",
            },
        ],
    }

    fg_online_asset = Asset(
        name=fg_name,
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "content": json.dumps(fg_forms),
            },
            {
                "formName": "AssetCommonDetailsForm",
                "typeIdentifier": "amazon.datazone.AssetCommonDetailsFormType",
                "content": json.dumps({
                        "customProperties": {
                            "FeatureGroupOfflineStoreAssetId": offline_store_asset_id,
                        }
                    }
                ),
            },
        ],
    )

    sagemaker_client.describe_feature_group.return_value = describe_resp
    datazone_client.list_data_sources.return_value = {"items": []}
    datazone_client.create_data_source.return_value = {"id": "data-source-id"}
    datazone_client.get_data_source.return_value = {"status": "READY"}
    datazone_client.start_data_source_run.return_value = {"id": "data-source-run-id"}
    datazone_client.get_data_source_run.return_value = {"status": "SUCCESS"}
    datazone_client.list_data_source_run_activities.return_value = {
        "items": [
            {
                "dataAssetId": offline_store_asset_id,
                "dataAssetStatus": "SUCCEEDED_CREATED",
            }
        ]
    }
    glue_form_outputs = [
        {
            "content": json.dumps({"k1": "v1"}),
            "formName": "GlueTableAssetForm",
            "typeName": "amazon.datazone.GlueTableAssetFormType",
        }
    ]
    datazone_client.get_asset.return_value = {
        "id": offline_store_asset_id,
        "name": describe_resp["FeatureGroupName"],
        "typeIdentifier": "amazon.datazone.GlueTableAssetType",
        "externalId": "glue-table-arn",
        "formsOutput": glue_form_outputs,
        "externalIdentifier": "glue-table-arn",
    }
    datazone_client.get_environment.return_value = {
        "provisionedResources": [],
    }

    glue_form_inputs = [
        {
            "content": json.dumps({"k1": "v1"}),
            "formName": "GlueTableAssetForm",
            "typeIdentifier": "amazon.datazone.GlueTableAssetFormType",
        }
    ]

    online_asset, offline_asset = fg_asset_builder.build_asset(fg_arn)
    assert online_asset == fg_online_asset
    assert offline_asset == Asset(
        id=offline_store_asset_id,
        name=describe_resp["FeatureGroupName"],
        type_id=AssetTypeIdentifier.DataZoneGlueAssetType,
        external_id="glue-table-arn",
        forms_input=glue_form_inputs,
    )
