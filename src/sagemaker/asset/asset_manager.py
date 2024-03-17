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
"""This module contains code related to the DataZone SageMaker Asset."""
from __future__ import print_function, absolute_import

import os
import logging
import json
from typing import Optional, Union, List
from sagemaker.session import Session
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.model import ModelPackageGroup
from sagemaker.asset import Asset, Listing, ChangeSetAction, AssetTypeIdentifier
from sagemaker.asset.asset_builder import (
    FeatureGroupAssetBuilder,
    ModelPackageGroupAssetBuilder,
)
import random
logger = logging.getLogger(__name__)


class AssetManager(object):
    """Manages the publishing of ML and data assets to the DataZone inventory and catalog."""

    def __init__(
        self,
        sagemaker_session: Session = None,
        domain_id: Optional[str] = None,
        project_id: Optional[str] = None,
        environment_id: Optional[str] = None,
        override_datazone_endpoint: Optional[str] = None,
    ):
        """Initializes a SageMaker `AssetManager` instance.

        Args:
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            domain_id (str): The id of the DataZone domain
            project_id (str): The id of the DataZone project
            environment_id (str): The id of the DataZone environment
            override_datazone_endpoint (str): The DataZone endpoint to use
        """
        self.__sm_session = sagemaker_session if sagemaker_session else Session()
        self.__boto_session = sagemaker_session.boto_session if sagemaker_session else Session().boto_session

        self.__datazone_client = (
            self.__boto_session.client("datazone")
            if not override_datazone_endpoint
            else self.__boto_session.client("datazone", endpoint_url=override_datazone_endpoint)
        )
        self.__sagemaker_client = self.__sm_session.sagemaker_client

        d_id, p_id, e_id = self.__get_datazone_context()

        self.domain_id = domain_id if domain_id is not None else d_id
        self.project_id = project_id if project_id is not None else p_id
        self.environment_id = environment_id if environment_id is not None else e_id

        if not self.domain_id or not self.project_id or not self.environment_id:
            raise ValueError(
                "Both DataZone domainId, owningProjectId, and environmentId should be given."
            )

        self.__data_source_manager = _DataSourceManager(
            self.__sm_session,
            self.__datazone_client,
            self.domain_id,
            self.project_id,
            self.environment_id)

    def export_to_inventory(
        self,
        resource: Optional[Union[Pipeline, FeatureGroup, ModelPackageGroup]] = None,
        resource_arn: Optional[str] = None,
        description: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ):
        """Export a SageMaker resource to DataZone project inventory.

        Export a SageMaker resource to DataZone project inventory as ML asset. If
        the resource has already been exported to inventory, then an asset revision will
        be created. Note that for feature group, if both online and offline are enabled,
        an asset will be returned for each one of them.

        Args:
            resource:
                The resource instance
            resource_arn:
                The ARN of the resource
            description:
                The description of the asset
            glossary_terms:
                The glossary terms of the asset

        Returns:
            One or more asset instance.
        """
        if (not resource and not resource_arn) or (resource and resource_arn):
            raise ValueError("Either resource instance or resource_arn str needs to be given.")

        if resource_arn:
            resource_type = resource_arn.split("/")[0].split(":")[-1]
            if resource_type == "feature-group":
                return self.__export_feature_group_asset(resource_arn, description, glossary_terms)
            if resource_type == "model-package-group":
                return self.__export_model_package_group_asset(
                    resource_arn, description, glossary_terms
                )
            raise ValueError(f"Unsupported resource type: {resource_type}")

        if isinstance(resource, FeatureGroup):
            return self.__export_feature_group_asset(resource, description, glossary_terms)
        if isinstance(resource, ModelPackageGroup):
            return self.__export_model_package_group_asset(resource, description, glossary_terms)
        raise ValueError(f"Unsupported resource type: {type(resource)}")

    def publish_to_catalog(
        self, asset: Optional[Asset] = None, asset_id: Optional[str] = None
    ) -> Listing:
        """Publish/List the asset to DataZone enterprise data catalog (EDC)

        Once the asset is in the project inventory, you can publish the asset
        to EDC, so members from your enterprise and search and discovery it.

        Args:
            asset: The asset instance
            asset_id: The id of the asset.

        Returns:
            Listing: the listing instance
        """
        if not asset and not asset_id:
            raise ValueError("Asset instance or asset Id must be given.")

        if asset:
            assert asset.id is not None, "Asset id must be given."

        listing_resp = self.__datazone_client.create_listing_change_set(
            action=ChangeSetAction.PUBLISH.value,
            domainIdentifier=self.domain_id,
            entityIdentifier=asset.id,
            entityType="ASSET",
        )

        return Listing(
            id=listing_resp["listingId"],
            revision=listing_resp["listingRevision"],
            status=listing_resp["status"] if "status" in listing_resp else None,
        )

    def remove_from_catalog(
        self, asset: Optional[Asset] = None, asset_id: Optional[str] = None
    ) -> Listing:
        """Unpublish the listing from DataZone enterprise data catalog (EDC).

        Args:
            asset: The asset instance.
            asset_id: The id of the asset.
        Returns:
            Listing: the listing instance
        """
        if not asset and not asset_id:
            raise ValueError("Asset instance or asset Id must be given.")

        if asset:
            assert asset.id is not None, "Asset id must be given."

        un_listing_resp = self.__datazone_client.create_listing_change_set(
            action=ChangeSetAction.UNPUBLISH.value,
            domainIdentifier=self.domain_id,
            entityIdentifier=asset.id if asset else asset_id,
            entityType="ASSET",
        )
        return Listing(
            id=un_listing_resp["listingId"],
            revision=un_listing_resp["listingRevision"],
            status=un_listing_resp["status"] if "status" in un_listing_resp else None,
        )

    def delete_asset(self, asset: Optional[Asset] = None, asset_id: Optional[str] = None):
        """Delete the asset

        Delete the asset from inventory, not that you need to unpublish before delete.

        Args:
            asset: The asset instance.
            asset_id: The id of the asset.
        Returns:
            Listing: the listing instance
        """
        if not asset and not asset_id:
            raise ValueError("Asset instance or asset Id must be given.")

        if asset:
            assert asset.id is not None, "Asset id must be given."

        self.__datazone_client.delete_asset(
            domainIdentifier=self.domain_id,
            identifier=asset.id if asset else asset_id,
        )

    def get_environment_default_data_lake_config(self) -> (Optional[str], Optional[str]):
        """Get default data lake config.

        Args:

        Returns:
            Glue producer output URI, Glue producer DB name.
        """

        get_environment_resp = self.__datazone_client.get_environment(
            domainIdentifier=self.domain_id,
            identifier=self.environment_id,
        )
        glue_producer_output_uri = None
        glue_producer_db_name = None
        for resource in get_environment_resp["provisionedResources"]:
            if resource["name"] == "glueProducerOutputUri":
                glue_producer_output_uri = resource["value"]
            if resource["name"] == "glueProducerDBName":
                glue_producer_db_name = resource["value"]
        return glue_producer_output_uri, glue_producer_db_name

    def __export_feature_group_asset(
        self,
        resource,
        description,
        glossary_terms
    ) -> (Asset, Optional[Asset]):
        """Exports a feature group.

        Args:
            resource: the feature group
            description: the feature group description
            glossary_terms: glossary terms for the feature group

        Returns:
            Asset for the feature group.
        """

        feature_group_asset_builder = FeatureGroupAssetBuilder(
            sagemaker_session=self.__sm_session,
            domain_id=self.domain_id,
            project_id=self.project_id,
            environment_id=self.environment_id,
            datazone_client=self.__datazone_client,
        )
        fg_online_asset, fg_offline_asset = feature_group_asset_builder.build_asset(resource)
        self.__create_asset(fg_online_asset, description, glossary_terms)
        if fg_offline_asset:
            # This is temporarily for beta 2
            feature_group_asset_builder.append_offline_store_online_store_property(
                fg_offline_asset, fg_online_asset.id
            )
            self.__create_asset_revision(fg_offline_asset, description, glossary_terms)
            return fg_online_asset, fg_offline_asset
        return fg_online_asset, None

    def __export_model_package_group_asset(self, resource, description, glossary_terms) -> Asset:
        """Exports a model package group.

        Args:
            resource: the model package group
            glossary_terms: glossary terms for the feature group
            description: the model package group description

        Returns:
            Asset for the model package.
        """
        mpg_builder = ModelPackageGroupAssetBuilder(sagemaker_session=self.__sm_session)
        mpg_asset = mpg_builder.build_asset(resource)
        data_source = self.__data_source_manager.get_mpg_data_source(mpg_asset.external_id)
        self.__data_source_manager.add_ds_reference_form_to_asset(mpg_asset, data_source)
        self.__create_asset(mpg_asset, description, glossary_terms)
        return mpg_asset

    def __get_datazone_context(self) -> (str, str, str):
        """Gets DataZone context information: domain id, project id, and environment id.

        Args:

        Returns:
            domain id, project id, environment id.
        """
        if os.path.exists("/opt/ml/metadata/resource-metadata.json"):  # pylint: disable=E1101
            with open("/opt/ml/metadata/resource-metadata.json") as f:
                metadata = json.load(f)
                resource_arn = metadata["ResourceArn"]
                arn_prefix = resource_arn.split(":")[:5]
                sm_domain_arn = ":".join(arn_prefix) + ":domain/" + metadata["DomainId"]
                list_tags_resp = self.__sagemaker_client.list_tags(
                    ResourceArn=sm_domain_arn, MaxResults=50
                )
                tags = list_tags_resp["Tags"]
                for tag in tags:
                    if tag["Key"] == "AmazonDataZoneDomain":
                        domain_id = tag["Value"]
                    elif tag["Key"] == "AmazonDataZoneProject":
                        project_id = tag["Value"]
                    elif tag["Key"] == "AmazonDataZoneEnvironment":
                        environment_id = tag["Value"]
                logger.info("Reading DataZone domainId: %s", domain_id)
                logger.info("owningProjectId: %s", project_id)
                logger.info("environmentId: %s from context.", environment_id)
                return domain_id, project_id, environment_id
        else:
            return None, None, None

    def __create_asset(
        self,
        asset: Asset,
        description: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ):
        """Creates an asset.

        Args:
            asset: asset information
            description: asset description
            glossary_terms: glossary terms for the asset
        """
        request = {
            "domainIdentifier": self.domain_id,
            "name": asset.name,
            "owningProjectIdentifier": self.project_id,
            "typeIdentifier": asset.type_id.value,
            "externalIdentifier": asset.external_id,
            "formsInput": asset.forms_input,
        }

        if description:
            request["description"] = description
        if glossary_terms:
            request["glossaryTerms"] = glossary_terms

        try:
            create_asset_resp = self.__datazone_client.create_asset(**request)
            asset.id = create_asset_resp["id"]
        except Exception as e:
            if "ConflictException" in str(e):
                logger.warning("Asset already exists, creating an asset revision.")
                asset.id = self.__get_asset_id_by_external_id(asset)
                self.__create_asset_revision(asset, description, glossary_terms)
                return
            raise e

    def __create_asset_revision(
        self,
        asset: Asset,
        description: Optional[str] = None,
        glossary_terms: Optional[List[str]] = None,
    ):
        """Creates an asset revision.

        Args:
            asset: asset information
            description: asset description
            glossary_terms: glossary terms for the asset
        """

        request = {
            "domainIdentifier": self.domain_id,
            "identifier": asset.id,
            "name": asset.name,
            "formsInput": asset.forms_input,
        }

        if description:
            request["description"] = description

        if glossary_terms:
            request["glossaryTerms"] = glossary_terms

        self.__datazone_client.create_asset_revision(**request)

    def __get_asset_id_by_external_id(self, asset: Asset) -> str:
        """Retrieve an asset by external id.

        Args:
            asset: the asset
        """

        def __get_arn_attribute_from_asset_type_id(asset_type_id: AssetTypeIdentifier) -> str:
            if asset_type_id == AssetTypeIdentifier.SageMakerFeatureGroupAssetType:
                return "SageMakerFeatureGroupForm.featureGroupArn"
            if asset_type_id == AssetTypeIdentifier.SageMakerModelPackageGroupAssetType:
                return "SagemakerModelPackageGroupForm.modelPackageGroupArn"
            raise ValueError(f"Unsupported asset type id: {asset_type_id}")

        search_resp = self.__datazone_client.search(
            domainIdentifier=self.domain_id,
            owningProjectIdentifier=self.project_id,
            searchScope="ASSET",
            filters={
                "filter": {
                    "attribute": __get_arn_attribute_from_asset_type_id(asset.type_id),
                    "value": asset.external_id,
                }
            },
        )
        if not search_resp["items"]:
            raise ValueError(f"Asset with external id {asset.external_id} not found.")
        assert (
            len(search_resp["items"]) == 1
        ), "More than one asset found with the same external id."
        return search_resp["items"][0]["assetItem"]["identifier"]


class _DataSourceManager(object):
    """Manages DataZone data sources."""

    DATA_SOURCE_NAME_FORMAT = '{0}-default-sagemaker-{1}-datasource'
    SAGEMAKER_TYPE = 'SAGEMAKER'
    MPG_ASSET_TYPE = 'ModelPackageGroup'
    FG_ASSET_TYPE = 'FeatureGroup'
    MODEL_REGISTRY_ASSET_TYPE = 'SageMakerModelPackageGroupAssetType'
    FEATURE_GROUP_ASSET_TYPE = 'SageMakerFeatureGroupAssetType'
    MANAGE_ACCESS_ROLE_NAME = 'AmazonDataZoneSageMakerManageAccessRole'
    MAX_TRACKED_ASSETS = 500
    DATA_SOURCE_REFERENCE_FORM_NAME = 'DataSourceReferenceForm'
    DATA_SOURCE_REFERENCE_FORM_TYPE = 'amazon.datazone.DataSourceReferenceFormType'

    def __init__(
            self,
            sagemaker_session: Session = None,
            datazone_client: Optional[str] = None,
            domain_id: Optional[str] = None,
            project_id: Optional[str] = None,
            environment_id: Optional[str] = None,
    ):
        """Initializes a SageMaker `AssetManager` instance.

        Args:
            sagemaker_session (sagemaker.session.Session): A SageMaker Session
                object, used for SageMaker interactions (default: None). If not
                specified, one is created using the default AWS configuration
                chain.
            domain_id (str): The id of the DataZone domain
            project_id (str): The id of the DataZone project
            environment_id (str): The id of the DataZone environment
            datazone_client (obj): The datazone client
        """
        self.sagemaker_session = sagemaker_session
        self.datazone_client = datazone_client
        self.domain_id = domain_id
        self.project_id = project_id
        self.environment_id = environment_id

    def get_mpg_data_source(self, resource_arn):
        environment = self.datazone_client.get_environment(
            domainIdentifier=self.domain_id,
            identifier=self.environment_id,
        )
        tracking_list_name = self.DATA_SOURCE_NAME_FORMAT.format(
            environment["name"],
            self.MPG_ASSET_TYPE.lower())

        return self.__get_data_source(resource_arn, self.MPG_ASSET_TYPE, tracking_list_name)

    def add_ds_reference_form_to_asset(self, asset, data_source):
        reference_form_content = {
            "dataSourceIdentifier": {
             "id": data_source["id"]
            },
            "dataSourceType": self.SAGEMAKER_TYPE,
            "filterableDataSourceId": data_source["id"],
        }
        reference_form = {
            "formName": self.DATA_SOURCE_REFERENCE_FORM_NAME,
            "content": json.dumps(reference_form_content),
            "typeIdentifier": self.DATA_SOURCE_REFERENCE_FORM_TYPE
        }
        for form_input in asset.forms_input:
            if "formName" not in form_input:
                raise ValueError("Invalid form")
            # if there is already a datasource form in the asset bail
            if form_input["formName"] == self.DATA_SOURCE_REFERENCE_FORM_NAME:
                return
        asset.forms_input.append(reference_form)


    def __get_data_source(self, resource_arn, asset_type, tracking_list_name):
        #self.datazone_client.delete_data_source(domainIdentifier=self.domain_id, identifier="41gsdxsbz9ifc9")
        #raise ValueError(tracking_list_name)
        data_sources = self.__list_all_data_sources(tracking_list_name)
        data_source = self.__get_existing_data_source(data_sources, resource_arn, asset_type)
        if data_source is None:
            latest_data_source = data_sources[0] if data_sources else None
            data_source = self.__create_data_source(
                latest_data_source,
                resource_arn,
                asset_type,
                tracking_list_name)
        else:
            self.__add_to_tracking_list(data_source, resource_arn, asset_type)

        return data_source


    def __list_all_data_sources(self, tracking_list_name):
        next_token = None
        data_sources = []
        while True:
            list_resp = self.datazone_client.list_data_sources(
                domainIdentifier=self.domain_id,
                projectIdentifier=self.project_id,
                environmentIdentifier=self.environment_id,
                type=self.SAGEMAKER_TYPE)
            data_sources_response = list_resp["items"]
            for data_source in data_sources_response:
                if tracking_list_name in data_source["name"]:
                    data_sources.append(data_source)
            next_token = list_resp["nextToken"] if "nextToken" in list_resp else None
            if next_token is None:
                break

        return sorted(data_sources, key=lambda x: x["createdAt"], reverse=True)

    def __get_existing_data_source(self, data_sources, resource_arn, asset_type):
        non_full_ds = []
        dz_asset_type = self.__get_asset_type_for_resource_type(asset_type)
        for data_source in data_sources:
            id = data_source["dataSourceId"]
            ds = self.datazone_client.get_data_source(
                domainIdentifier=self.domain_id,
                identifier=id)
            if "configuration" in ds and "sageMakerRunConfiguration" in ds["configuration"]:
                sm_config = ds["configuration"]["sageMakerRunConfiguration"]
                if "trackingAssets" in sm_config:
                    assets = sm_config["trackingAssets"][dz_asset_type]
                    if resource_arn in assets:
                        return ds
            else:
                assets = []

            if len(assets) < self.MAX_TRACKED_ASSETS:
                non_full_ds.append(ds)

        # no existing, non-full data sources
        if not non_full_ds:
            return None
        return_val = sorted(non_full_ds, key=lambda x: x["createdAt"], reverse=True)[0]
        return return_val

    def __get_asset_type_for_resource_type(self, resource_type):
        if resource_type.lower() == 'ModelPackageGroup'.lower():
            return self.MODEL_REGISTRY_ASSET_TYPE
        if resource_type.lower() == 'FeatureGroup'.lower():
            return self.FEATURE_GROUP_ASSET_TYPE
        raise ValueError(f'Unrecognized resource type {resource_type}')

    def __add_to_tracking_list(self, data_source, resource_arn, asset_type):
        dz_asset_type = self.__get_asset_type_for_resource_type(asset_type)
        id = data_source["id"]
        assets = []

        if "configuration" in data_source and "sageMakerRunConfiguration" in data_source["configuration"]:
            sm_config = data_source["configuration"]["sageMakerRunConfiguration"]
            assets = sm_config["trackingAssets"][dz_asset_type]
            role = sm_config["dataAccessRole"]
        else:
            role = self.__get_data_access_role(data_source)

        # resource already tracked
        if resource_arn not in assets:
            self.datazone_client.update_data_source(
                domainIdentifier=self.domain_id,
                identifier=id,
                configuration={
                    "sageMakerRunConfiguration": {
                        "trackingAssets": {
                            dz_asset_type: assets + [resource_arn]
                        },
                        "dataAccessRole": role
                    }
                })

    def __create_data_source(self, latest_data_source, resource_arn, asset_type, tracking_list_name):
        dz_asset_type = self.__get_asset_type_for_resource_type(asset_type)
        next_version = 0
        if latest_data_source:
            name = latest_data_source["name"]
            latest_version_text = name.split('-')[-1]
            if latest_version_text.isnumeric():
                # <envid>-default-sagemaker-<asset type>-datasource-<##>
                next_version = int(latest_version_text) + 1
            else:
                # latest_data_source is the default like <envid>-default-sagemaker-<asset type>-datasource
                next_version = 1

        # the latest data source is full, increment version. otherwise default data source doesn't exist yet
        if next_version > 0:
            tracking_list_name += f'-{next_version}'

        random_cron = f'cron({random.randint(0, 59)} {random.randint(0, 23)} * * ? *)'
        data_access_role = self.__get_data_access_role(latest_data_source)
        create_resp = self.datazone_client.create_data_source(
            domainIdentifier=self.domain_id,
            projectIdentifier=self.project_id,
            environmentIdentifier=self.environment_id,
            name=tracking_list_name,
            type=self.SAGEMAKER_TYPE,
            schedule={
                "schedule": random_cron
            },
            publishOnImport=False,
            enableSetting="ENABLED",
            recommendation={
                "enableBusinessNameGeneration": False
            },
            configuration = {
                "sageMakerRunConfiguration": {
                    "dataAccessRole": data_access_role,
                    "trackingAssets": {
                        dz_asset_type: [resource_arn]
                    }
                }
            })
        return create_resp

    def __get_data_access_role(self, latest_data_source):
        if (latest_data_source and
                "configuration" in latest_data_source and
                "sageMakerRunConfiguration" in latest_data_source["configuration"]):
            return latest_data_source["configuration"]["sageMakerRunConfiguration"]["dataAccessRole"]

        sts_client = self.sagemaker_session.boto_session.client('sts')
        identity = sts_client.get_caller_identity()
        account_id = identity['Account']
        return f'arn:aws:iam::{account_id}:role/{self.MANAGE_ACCESS_ROLE_NAME}'