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
import json
import datetime
from mock import MagicMock, Mock, patch
from sagemaker.asset import AssetManager, Asset, AssetTypeIdentifier, ChangeSetAction, Listing
from sagemaker.asset.asset_builder import (
    ModelPackageGroupAssetBuilder,
    FeatureGroupAssetBuilder,
)
from sagemaker.model import ModelPackageGroup
from sagemaker.feature_store.feature_group import FeatureGroup


@pytest.fixture
def domain_id():
    return "dzd_123"


@pytest.fixture
def project_id():
    return "pj_123"


@pytest.fixture
def environment_id():
    return "env_123"


@pytest.fixture
def sagemaker_client():
    return Mock(name="sagemaker_client")


@pytest.fixture
def datazone_client():
    return Mock(name="datazone_client")


@pytest.fixture
def lf_client():
    return Mock(name="lakeformation_client")


@pytest.fixture
def sts_client():
    return Mock(name="sts_client")


@pytest.fixture
def sagemaker_session(sagemaker_client, datazone_client, lf_client, sts_client):
    boto_mock = Mock(name="boto_session", region_name="us-west-2")

    def get_clients(*args, **kwargs):
        if args[0] == "datazone":
            return datazone_client
        elif args[0] == "sagemaker":
            return sagemaker_client
        elif args[0] == "lakeformation":
            return lf_client
        elif args[0] == "sts":
            return sts_client
        else:
            raise ValueError(f"Unexpected client type: {args[0]}")

    boto_mock.client = MagicMock(name="boto_client", side_effect=get_clients)

    session_mock = MagicMock(
        name="sagemaker_session",
        boto_session=boto_mock,
        boto_region_name="us-west-2",
        config=None,
        local_mode=False,
        default_bucket_prefix=None,
    )
    session_mock.boto_session.return_value = boto_mock

    return session_mock


@pytest.fixture
def asset_manager(sagemaker_session, domain_id, project_id, environment_id):
    return AssetManager(
        sagemaker_session=sagemaker_session,
        domain_id=domain_id,
        project_id=project_id,
        environment_id=environment_id,
    )

def test_asset_manager_export_mpg_asset_to_inventory(
    asset_manager: AssetManager,
    datazone_client: Mock,
    sagemaker_session: MagicMock,
    sts_client: Mock,
    environment_id: str
):
    mpg_arn = "arn:aws:sagemaker:us-east-2:1234567890:model-package-group/my-mpg"
    form_content = {
        "modelPackageGroupName": "my-mpg",
        "modelPackageGroupArn": mpg_arn,
        "modelPackageGroupDescription": "my good mpg",
        "modelPackageGroupStatus": "Completed",
        "creationTime": "2022-01-01T00:00:00Z",
    }
    mpg_asset = Asset(
        name="asset-name",
        type_id=AssetTypeIdentifier.SageMakerModelPackageGroupAssetType,
        external_id=mpg_arn,
        forms_input=[
            {
                "formName": "SageMakerModelPackageGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerModelPackageGroupFormType",
                "typeRevision": "1",
                "content": json.dumps(form_content)
            }
        ],
    )

    # setup create asset response
    datazone_client.create_asset.return_value = {"id": "asset-id"}

    # setup list data sources response
    list_ds_response = {
        "items": [
            {
                "dataSourceId": "ds-id",
                "name": f"{environment_id}-default-sagemaker-modelpackagegroup-datasource",
                "createdAt": str(datetime.datetime.now())
            }
        ]
    }
    datazone_client.list_data_sources.return_value = list_ds_response

    get_environment_response = {
        "id": "envId",
        "name": "fooEnvName"
    }
    datazone_client.get_environment.return_value = get_environment_response

    # setup get data source response
    get_ds_response = {
        "id": "ds-id",
        "createdAt": str(datetime.datetime.now()),
        "configuration": {
            "sageMakerRunConfiguration": {
                "dataAccessRole": "ManageAccessRole",
                "trackingAssets": {
                    "SageMakerModelPackageGroupAssetType": []
                }
            }
        }
    }
    datazone_client.get_data_source.return_value = get_ds_response

    get_caller_identity_resp = {
        "Account": "123456789100"
    }
    sts_client.get_caller_identity.return_value = get_caller_identity_resp

    create_data_source_resp = {
        "id": "ds-id"
    }
    datazone_client.create_data_source.return_value = create_data_source_resp

    with patch.object(
        ModelPackageGroupAssetBuilder, "build_asset", return_value=mpg_asset
    ) as mock_method:
        asset_manager.export_to_inventory(
            resource_arn=mpg_arn, description="my description", glossary_terms=["term1", "term2"]
        )
        mock_method.assert_called_once_with(mpg_arn)
        datazone_client.create_asset.assert_called_with(
            domainIdentifier=asset_manager.domain_id,
            name=mpg_asset.name,
            owningProjectIdentifier=asset_manager.project_id,
            typeIdentifier=mpg_asset.type_id.value,
            externalIdentifier=mpg_asset.external_id,
            formsInput=mpg_asset.forms_input,
            description="my description",
            glossaryTerms=["term1", "term2"],
        )
        assert mpg_asset.id == "asset-id"

    mpg_asset_2_form_content = {
        "modelPackageGroupName": "my-mpg-2",
        "modelPackageGroupArn": mpg_arn + "-2",
        "modelPackageGroupStatus": "Completed",
        "modelPackageGroupDescription": "my good mpg",
        "creationTime": "2022-01-01T00:00:00Z",
    }
    mpg_asset_2 = Asset(
        name="asset-name-2",
        type_id=AssetTypeIdentifier.SageMakerModelPackageGroupAssetType,
        external_id=mpg_arn + "-2",
        forms_input=[
            {
                "formName": "SageMakerModelPackageGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerModelPackageGroupFormType",
                "typeRevision": "1",
                "content": json.dumps(mpg_asset_2_form_content)
            }
        ],
    )
    mpg = ModelPackageGroup(
        name="my-mpg-2", description="my good mpg", sagemaker_session=sagemaker_session
    )

    with patch.object(
        ModelPackageGroupAssetBuilder, "build_asset", return_value=mpg_asset_2
    ) as mock_method:
        asset_manager.export_to_inventory(
            resource=mpg, description="my description 2", glossary_terms=["term1", "term2"]
        )
        mock_method.assert_called_once_with(mpg)
        datazone_client.create_asset.assert_called_with(
            domainIdentifier=asset_manager.domain_id,
            name=mpg_asset_2.name,
            owningProjectIdentifier=asset_manager.project_id,
            typeIdentifier=mpg_asset_2.type_id.value,
            externalIdentifier=mpg_asset_2.external_id,
            formsInput=mpg_asset_2.forms_input,
            description="my description 2",
            glossaryTerms=["term1", "term2"],
        )
        assert mpg_asset_2.id == "asset-id"


def test_asset_manager_export_fg_asset_to_inventory(
    asset_manager: AssetManager,
    datazone_client: Mock,
    sagemaker_session: MagicMock,
):
    fg_arn = "arn:aws:sagemaker:us-east-2:1234567890:feature-group/my-fg"
    fg_form_content = {
            "featureGroupName": "my-fg",
            "featureGroupArn": fg_arn,
            "featureGroupDescription": "my good mpg",
            "recordIdentifierFeatureName": "f1",
            "eventTimeFeatureName": "f2",
            "creationTime": "2022-01-01T00:00:00Z",
            "lastModifiedTime": "2022-01-01T00:00:00Z",
        }
    fg_online_asset = Asset(
        name="asset-name",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_form_content)
            }
        ],
    )
    fg_offline_form_content = {
        "glueTableName": "my-glue-table"
    }
    fg_offline_asset = Asset(
        id="offline-asset-id",
        name="asset-name",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_offline_form_content)
            }],
    )

    datazone_client.create_asset.return_value = {"id": "asset-id"}

    with patch.object(
        FeatureGroupAssetBuilder, "build_asset", return_value=(fg_online_asset, fg_offline_asset)
    ) as mock_method:

        with patch.object(
            FeatureGroupAssetBuilder, "append_offline_store_online_store_property"
        ) as mock_append_method:

            asset_manager.export_to_inventory(
                resource_arn=fg_arn, description="my description", glossary_terms=["term1", "term2"]
            )
            mock_method.assert_called_once_with(fg_arn)
            datazone_client.create_asset.assert_called_with(
                domainIdentifier=asset_manager.domain_id,
                name=fg_online_asset.name,
                owningProjectIdentifier=asset_manager.project_id,
                typeIdentifier=fg_online_asset.type_id.value,
                externalIdentifier=fg_online_asset.external_id,
                formsInput=fg_online_asset.forms_input,
                description="my description",
                glossaryTerms=["term1", "term2"],
            )
            mock_append_method.assert_called_once_with(fg_offline_asset, fg_online_asset.id)
            datazone_client.create_asset_revision.assert_called_with(
                domainIdentifier=asset_manager.domain_id,
                identifier=fg_offline_asset.id,
                name=fg_offline_asset.name,
                formsInput=fg_offline_asset.forms_input,
                description="my description",
                glossaryTerms=["term1", "term2"],
            )

    fg_asset_2_content = {
        "featureGroupName": "my-fg-2",
        "featureGroupArn": fg_arn + "-2",
        "featureGroupDescription": "my good mpg",
        "recordIdentifierFeatureName": "f1",
        "eventTimeFeatureName": "f2",
        "creationTime": "2022-01-01T00:00:00Z",
        "lastModifiedTime": "2022-01-01T00:00:00Z",
    }
    fg_asset_2 = Asset(
        name="asset-name-2",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn + "-2",
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_asset_2_content)
            }
        ],
    )

    fg = FeatureGroup(name="my-fg-2", sagemaker_session=sagemaker_session)
    with patch.object(
        FeatureGroupAssetBuilder, "build_asset", return_value=(fg_asset_2, None)
    ) as mock_method:
        with patch.object(
            FeatureGroupAssetBuilder, "append_offline_store_online_store_property"
        ) as mock_append_method:
            asset_manager.export_to_inventory(
                resource=fg, description="my description", glossary_terms=["term1", "term2"]
            )
            mock_method.assert_called_once_with(fg)
            datazone_client.create_asset.assert_called_with(
                domainIdentifier=asset_manager.domain_id,
                name=fg_asset_2.name,
                owningProjectIdentifier=asset_manager.project_id,
                typeIdentifier=fg_asset_2.type_id.value,
                externalIdentifier=fg_asset_2.external_id,
                formsInput=fg_asset_2.forms_input,
                description="my description",
                glossaryTerms=["term1", "term2"],
            )
            mock_append_method.assert_not_called()


def test_export_inventory_existing_asset_create_revision(
        asset_manager: AssetManager,
        datazone_client: Mock,
    ):
    fg_arn = "arn:aws:sagemaker:us-east-2:1234567890:feature-group/my-fg"
    fg_online_asset_form_content =  {
        "featureGroupName": "my-fg",
        "featureGroupArn": fg_arn,
        "featureGroupDescription": "my good mpg",
        "recordIdentifierFeatureName": "f1",
        "eventTimeFeatureName": "f2",
        "creationTime": "2022-01-01T00:00:00Z",
        "lastModifiedTime": "2022-01-01T00:00:00Z",
    }
    fg_online_asset = Asset(
        name="asset-name",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_online_asset_form_content)
            }
        ],
    )
    asset_id = "asset-id"
    datazone_client.create_asset = MagicMock(name='create-asset', side_effect=ValueError("ConflictException"))
    datazone_client.search.return_value = {
        'items': [{
            "assetItem": {"identifier": asset_id}
        }]
    }


    with patch.object(
        FeatureGroupAssetBuilder, "build_asset", return_value=(fg_online_asset, None)
    ) as mock_method:
        asset_manager.export_to_inventory(resource_arn=fg_arn)
        mock_method.assert_called_once_with(fg_arn)
        datazone_client.create_asset.assert_called_with(
            domainIdentifier=asset_manager.domain_id,
            name=fg_online_asset.name,
            owningProjectIdentifier=asset_manager.project_id,
            typeIdentifier=fg_online_asset.type_id.value,
            externalIdentifier=fg_online_asset.external_id,
            formsInput=fg_online_asset.forms_input,
        )
        datazone_client.search.assert_called_once_with(
            domainIdentifier=asset_manager.domain_id,
            owningProjectIdentifier=asset_manager.project_id,
            searchScope="ASSET",
            filters={
                "filter": {
                    "attribute": "SageMakerFeatureGroupForm.featureGroupArn",
                    "value": fg_online_asset.external_id,
                }
            },
        )
        datazone_client.create_asset_revision.assert_called_once_with(
            domainIdentifier=asset_manager.domain_id,
            identifier=asset_id,
            name=fg_online_asset.name,
            formsInput=fg_online_asset.forms_input,
        )


def test_publish_to_catalog(asset_manager, datazone_client):
    fg_arn = "arn:aws:sagemaker:us-east-2:1234567890:feature-group/my-fg"
    fg_online_asset_content = {
        "featureGroupName": "my-fg",
        "featureGroupArn": fg_arn,
        "featureGroupDescription": "my good mpg",
        "recordIdentifierFeatureName": "f1",
        "eventTimeFeatureName": "f2",
        "creationTime": "2022-01-01T00:00:00Z",
        "lastModifiedTime": "2022-01-01T00:00:00Z",
    }
    fg_online_asset = Asset(
        name="asset-name",
        id="asset-id",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_online_asset_content)
            }
        ],
    )

    listing_resp = {"listingId": "123", "listingRevision": "1", "status": "ACTIVE"}
    datazone_client.create_listing_change_set.return_value = listing_resp

    listing: Listing = asset_manager.publish_to_catalog(fg_online_asset)

    datazone_client.create_listing_change_set.assert_called_once_with(
        action="PUBLISH",
        domainIdentifier=asset_manager.domain_id,
        entityIdentifier=fg_online_asset.id,
        entityType="ASSET",
    )
    assert listing.id == listing_resp["listingId"]
    assert listing.revision == listing_resp["listingRevision"]
    assert listing.status == listing_resp["status"]


def test_un_publish_to_catalog(asset_manager, datazone_client):
    fg_arn = "arn:aws:sagemaker:us-east-2:1234567890:feature-group/my-fg"
    fg_online_asset_content = {
        "featureGroupName": "my-fg",
        "featureGroupArn": fg_arn,
        "featureGroupDescription": "my good mpg",
        "recordIdentifierFeatureName": "f1",
        "eventTimeFeatureName": "f2",
        "creationTime": "2022-01-01T00:00:00Z",
        "lastModifiedTime": "2022-01-01T00:00:00Z",
    }
    fg_online_asset = Asset(
        name="asset-name",
        id="asset-id",
        type_id=AssetTypeIdentifier.SageMakerFeatureGroupAssetType,
        external_id=fg_arn,
        forms_input=[
            {
                "formName": "SageMakerFeatureGroupForm",
                "typeIdentifier": "amazon.datazone.SageMakerFeatureGroupFormType",
                "typeRevision": "2",
                "content": json.dumps(fg_online_asset_content)
            }
        ],
    )

    listing_resp = {"listingId": "123", "listingRevision": "1", "status": "INACTIVE"}
    datazone_client.create_listing_change_set.return_value = listing_resp

    listing: Listing = asset_manager.remove_from_catalog(fg_online_asset)

    datazone_client.create_listing_change_set.assert_called_once_with(
        action="UNPUBLISH",
        domainIdentifier=asset_manager.domain_id,
        entityIdentifier=fg_online_asset.id,
        entityType="ASSET",
    )
    assert listing.id == listing_resp["listingId"]
    assert listing.revision == listing_resp["listingRevision"]
    assert listing.status == listing_resp["status"]


def test_delete_asset(asset_manager, datazone_client):
    asset_manager.delete_asset(asset_id="asset-id")
    datazone_client.delete_asset.assert_called_once_with(
        domainIdentifier=asset_manager.domain_id, identifier="asset-id"
    )
