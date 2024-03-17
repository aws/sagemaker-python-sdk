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
"""This module contains code related to the DataZone SageMaker AssetBuilder abstract class."""
from __future__ import print_function, absolute_import

import time
import json
import logging
from typing import Dict, Any
from abc import ABC, abstractmethod  # pylint: disable=E0611
from boto3 import Session, client
import sagemaker
from sagemaker.asset import Asset, AssetTypeIdentifier
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.model import ModelPackageGroup

logger = logging.getLogger(__name__)


class AssetBuilder(ABC):
    """Base class for asset builders."""

    def __init__(self, session: Session):
        """Initializes AssetBuilder

        Args:
            session: the SageMaker Session

        Returns:
            domain id, project id, environment id.
        """
        self.sagemaker_client = session.sagemaker_client
        self.manage_access_role = self.__get_manage_access_role()
        self.asset_common_details_form_name: str = "AssetCommonDetailsForm"
        self.asset_common_details_form_type_id: str = "amazon.datazone.AssetCommonDetailsFormType"

    @abstractmethod
    def build_asset(self, resource):
        """Builds an asset

        Args:
            resource: the resource used to build the asset
        """

    @classmethod
    def convert_date_time(cls, dt):
        """Converts date time

        Args:
            dt: The date time to convert
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def __get_manage_access_role():
        """Gets manage access role

        Returns:
            The arn of the manage access role
        """
        execution_role = sagemaker.get_execution_role()
        account = execution_role.split(":")[4]
        return f"arn:aws:iam::{account}:role/AmazonDataZoneSageMakerManageAccessRole"


class FeatureGroupAssetBuilder(AssetBuilder):
    """Builds a DataZone asset from a feature gorup."""

    def __init__(
        self,
        session: Session,
        domain_id: str,
        project_id: str,
        environment_id: str,
        datazone_client: client,
    ):
        AssetBuilder.__init__(self, session)
        self.__asset_type_id = AssetTypeIdentifier.SageMakerFeatureGroupAssetType
        self.__form_type_id = "amazon.datazone.SageMakerFeatureGroupFormType"
        self.__form_name = "SageMakerFeatureGroupForm"

        self.__lf_client = session.client("lakeformation")
        self.__datazone_client = datazone_client
        self.__domain_id = domain_id
        self.__project_id = project_id
        self.__environment_id = environment_id

    def build_asset(self, resource) -> (Asset, Asset):
        """Builds an asset for a resource.

        Args:
            resource: The resource to build an asset for.

        Returns:
            Online asset, Offline asset
        """

        if isinstance(resource, FeatureGroup):
            resource_name = resource.name
        else:
            resource_name = resource

        describe_resp = self.sagemaker_client.describe_feature_group(FeatureGroupName=resource_name)
        fg_status = describe_resp["FeatureGroupStatus"]
        if fg_status == "Creating":
            raise ValueError(f"Feature group {resource_name} is in {fg_status} state. Please wait.")
        if fg_status in ("CreateFailed", "Deleting", "DeleteFailed"):
            raise ValueError(f"Feature group {resource_name} is in invalid {fg_status} state.")

        # There will be three scenarios, The feature group has
        # 1. only online store enabled
        # 2. only offline store enabled
        # 3. both online and offline stores enabled
        offline_asset = None
        if "OfflineStoreConfig" in describe_resp:
            offline_store_config = describe_resp["OfflineStoreConfig"]
            if (
                "DataCatalogConfig" in offline_store_config
                and offline_store_config["DataCatalogConfig"]
            ):
                # 1. validate the offline store s3 location for lake formation registration
                if self.__validate_for_lake_formation(offline_store_config):
                    # 2. create a datazone glue data source to import the table as data asset
                    data_source_id = self.__create_glue_data_source(
                        feature_group_name=describe_resp["FeatureGroupName"],
                        data_catalog_config=offline_store_config["DataCatalogConfig"],
                    )
                    # 3. start the data source run
                    data_source_run_id = self.__start_data_source_run(data_source_id)
                    # 4. retrieve the glue asset created
                    offline_asset = self.__retrieve_offline_asset(data_source_run_id)

        last_modified_time = describe_resp.get("LastModifiedTime", describe_resp["CreationTime"])
        feature_group_forms = {
            "featureGroupName": describe_resp["FeatureGroupName"],
            "featureGroupArn": describe_resp["FeatureGroupArn"],
            "recordIdentifierFeatureName": describe_resp["RecordIdentifierFeatureName"],
            "eventTimeFeatureName": describe_resp["EventTimeFeatureName"],
            "creationTime": AssetBuilder.convert_date_time(describe_resp["CreationTime"]),
            "lastModifiedTime": AssetBuilder.convert_date_time(last_modified_time),
        }

        if "OfflineStoreStatus" in describe_resp:
            feature_group_forms.update(
                {"offlineStoreStatus": describe_resp["OfflineStoreStatus"]["Status"].upper()}
            )

        # we need to retrieve all the feature definitions
        feature_definitions = describe_resp["FeatureDefinitions"]
        next_token = describe_resp.get("NextToken", None)
        while next_token:
            describe_resp.get("LastModifiedTime", describe_resp["CreationTime"])
            describe_resp = self.sagemaker_client.describe_feature_group(
                FeatureGroupName=resource_name,
                NextToken=next_token,
            )
            feature_definitions += describe_resp["FeatureDefinitions"]
            next_token = describe_resp.get("NextToken", None)

        feature_group_forms.update(
            {"featureDefinitions": [self.__convert_feature_def(f) for f in feature_definitions]}
        )

        form_inputs = [
            {
                "formName": self.__form_name,
                "typeIdentifier": self.__form_type_id,
                "typeRevision": "1",
                "content": json.dumps(feature_group_forms),
            }
        ]

        if offline_asset:
            form_inputs.append(
                {
                    "formName": self.asset_common_details_form_name,
                    "typeIdentifier": self.asset_common_details_form_type_id,
                    "content": json.dumps(
                        {"customProperties": {"FeatureGroupOfflineStoreAssetId": offline_asset.id}}
                    ),
                }
            )

        online_asset = Asset(
            name=describe_resp["FeatureGroupName"],
            type_id=self.__asset_type_id,
            external_id=describe_resp["FeatureGroupArn"],
            forms_input=form_inputs,
        )

        return online_asset, offline_asset

    def __validate_for_lake_formation(self, offline_store_config: Dict):
        """Validates lake formation

        Args:
            offline_store_config: The offline store config.

        Returns:
            True if config is valid.
        """
        s3_location = offline_store_config["S3StorageConfig"]["ResolvedOutputS3Uri"]
        # We need to verify if the ResolvedOutputS3Uri bucket is the same as the
        # default DZ bucket. If not, we will reject to publish offline store. The reason
        # is that in order to make the auto-granting work, the offline store s3 location
        # needs to be registered with LF. ML builder (assuming the execution role) will not
        # have the permission to register. They will need to ask their Admin to do it.
        default_bucket = self.__get_environment_default_bucket()
        if not s3_location.startswith(default_bucket):
            logger.warning(
                "Offline store s3 uri %s is not in the same bucket as the default "
                "DataZone environment bucket. We wont be able to fulfill the access granting"
                "for any subscription. Skip the creation of offline store glue asset.",
                s3_location,
            )
            return False
        return True

    def __get_environment_default_bucket(self) -> str:
        """Get environment default bucket.

        Returns:
            The bucket uri of the glue producer.
        """
        get_environment_resp = self.__datazone_client.get_environment(
            domainIdentifier=self.__domain_id,
            identifier=self.__environment_id,
        )
        for resource in get_environment_resp["provisionedResources"]:
            if resource["name"] == "glueProducerOutputUri":
                return resource["value"]
        return ""

    def __create_glue_data_source(self, feature_group_name: str, data_catalog_config) -> str:
        """Creates a glue data source.

        Returns:
            The id of the created data source.
        """
        offline_store_data_source_name = f"{feature_group_name}-offline-store-data-source"
        # if we already have a corresponding data source exists, we simply trigger its run
        list_data_sources_resp = self.__datazone_client.list_data_sources(
            domainIdentifier=self.__domain_id,
            projectIdentifier=self.__project_id,
            environmentIdentifier=self.__environment_id,
            name=offline_store_data_source_name,
        )

        if len(list_data_sources_resp["items"]) > 0:
            assert (
                len(list_data_sources_resp["items"]) == 1
            ), f"More than offline store data source for {feature_group_name} found"
            return list_data_sources_resp["items"][0]["dataSourceId"]

        # otherwise, we create a new data source
        create_data_source_resp = self.__datazone_client.create_data_source(
            domainIdentifier=self.__domain_id,
            projectIdentifier=self.__project_id,
            environmentIdentifier=self.__environment_id,
            type="GLUE",
            configuration={
                "glueRunConfiguration": {
                    "dataAccessRole": self.manage_access_role,
                    "relationalFilterConfigurations": [
                        {
                            "databaseName": data_catalog_config["Database"],
                            "filterExpressions": [
                                {
                                    "type": "INCLUDE",
                                    "expression": data_catalog_config["TableName"],
                                }
                            ],
                        }
                    ],
                }
            },
            enableSetting="ENABLED",
            publishOnImport=False,
            name=offline_store_data_source_name,
        )
        return create_data_source_resp["id"]

    def __start_data_source_run(self, data_source_id: str) -> str:
        """Start data source run.

        Returns:
            The id of the data source run.
        """
        data_source_status = None
        while not data_source_status or data_source_status == "CREATING":
            get_data_source_resp = self.__datazone_client.get_data_source(
                domainIdentifier=self.__domain_id,
                identifier=data_source_id,
            )
            data_source_status = get_data_source_resp["status"]
            time.sleep(10)

        if data_source_status != "READY":
            raise ValueError(f"Data source {data_source_id} is in {data_source_status} state")

        start_data_source_run_resp = self.__datazone_client.start_data_source_run(
            domainIdentifier=self.__domain_id,
            dataSourceIdentifier=data_source_id,
        )
        return start_data_source_run_resp["id"]

    def __retrieve_offline_asset(self, data_source_run_id: str) -> Asset:
        """Retrieves offline asset.

        Returns:
            The asset.
        """
        status = None
        while not status or status == "RUNNING" or status == "REQUESTED":
            get_data_source_run_resp = self.__datazone_client.get_data_source_run(
                domainIdentifier=self.__domain_id,
                identifier=data_source_run_id,
            )
            status = get_data_source_run_resp["status"]

        if status != "SUCCESS":
            raise ValueError(f"Data source run {data_source_run_id} is in {status} state")

        list_data_source_run_activities_resp = (
            self.__datazone_client.list_data_source_run_activities(
                domainIdentifier=self.__domain_id,
                identifier=data_source_run_id,
            )
        )

        if not list_data_source_run_activities_resp["items"]:
            raise ValueError(f"No data source run activity found for {data_source_run_id}")

        assert (
            len(list_data_source_run_activities_resp["items"]) == 1
        ), f"More than one data source run activity found for {data_source_run_id}"

        item = list_data_source_run_activities_resp["items"][0]

        if item["dataAssetStatus"] == "FAILED":
            raise ValueError(
                f"Data source run activity {item['id']} "
                f"failed with error message: {item['errorMessage']}"
            )

        asset_id = item["dataAssetId"]
        get_asset_resp = self.__datazone_client.get_asset(
            domainIdentifier=self.__domain_id,
            identifier=asset_id,
        )

        return Asset(
            id=asset_id,
            name=get_asset_resp["name"],
            type_id=AssetTypeIdentifier.value_of(
                get_asset_resp["typeIdentifier"]
            ),  # glue table asset type
            external_id=get_asset_resp["externalIdentifier"],
            forms_input=[
                self.__convert_form_output_to_form_input(form_output)
                for form_output in get_asset_resp["formsOutput"]
            ],
        )

    @staticmethod
    def __convert_feature_def(feature_definition) -> Dict[str, Any]:
        """Converts feature definition for asset creation.

        Args:
            feature_definition: the feature definition object

        Returns:
            The feature definition dictionary.
        """

        def convert_feature_type(feature_type: str) -> str:
            if feature_type == "Integral":
                return feature_type.upper()
            if feature_type == "Fractional":
                return feature_type.upper()
            if feature_type == "String":
                return feature_type.upper()
            raise ValueError("Unsupported feature type: {}".format(feature_type))

        feature_def = {
            "featureName": feature_definition["FeatureName"],
            "featureType": convert_feature_type(feature_definition["FeatureType"]),
        }

        if "CollectionType" in feature_definition:
            feature_def.update({"collectionType": feature_definition["CollectionType"].upper()})

        if "CollectionConfig" in feature_definition:
            feature_def.update({"collectionConfig": feature_definition["CollectionConfig"]})

        return feature_def

    @staticmethod
    def __convert_form_output_to_form_input(form_output: Dict[str, Any]) -> Dict[str, Any]:
        """Converts form output to form input.

        Args:
            form_output: the form output

        Returns:
            The form input.
        """
        return {
            "formName": form_output["formName"],
            "content": form_output["content"],
            "typeIdentifier": form_output["typeName"],
            "typeRevision": form_output["typeRevision"],
        }

    def append_offline_store_online_store_property(
        self, offline_asset: Asset, online_store_asset_id: str
    ):
        """Appends offline store to the online store property.

        Args:
            offline_asset: the offline store asset
            online_store_asset_id: the online store asset id
        """
        offline_asset.forms_input.append(
            {
                "formName": self.asset_common_details_form_name,
                "typeIdentifier": self.asset_common_details_form_type_id,
                "content": json.dumps(
                    {
                        "customProperties": {
                            "FeatureGroupOnlineStoreAssetId": online_store_asset_id,
                        }
                    }
                ),
            }
        )


class ModelPackageGroupAssetBuilder(AssetBuilder):
    """Builds asset for model package groups."""

    def __init__(self, session: Session):
        super().__init__(session)
        self.__asset_type_id = AssetTypeIdentifier.SageMakerModelPackageGroupAssetType
        self.__form_type_id = "amazon.datazone.SageMakerModelPackageGroupFormType"
        # TODO: this is a type, we should make "Sagemaker" -> "SageMaker", will address
        # along with beta 2 feedback
        self.__form_name = "SagemakerModelPackageGroupForm"

    def build_asset(self, resource) -> Asset:
        """Builds an asset from a model package group.

        Args:
            resource: The Model Package Group to build an asset.
        """

        if isinstance(resource, ModelPackageGroup):
            resource_name = resource.name
        else:
            resource_name = resource

        describe_resp = self.sagemaker_client.describe_model_package_group(
            ModelPackageGroupName=resource_name
        )

        mpg_forms = {
            "modelPackageGroupName": describe_resp["ModelPackageGroupName"],
            "modelPackageGroupArn": describe_resp["ModelPackageGroupArn"],
            "creationTime": AssetBuilder.convert_date_time(describe_resp["CreationTime"]),
        }

        if "ModelPackageGroupStatus" in describe_resp:
            mpg_forms.update(
                {"modelPackageGroupStatus": describe_resp["ModelPackageGroupStatus"].upper()}
            )

        if "ModelPackageGroupDescription" in describe_resp:
            mpg_forms.update(
                {"modelPackageGroupDescription": describe_resp["ModelPackageGroupDescription"]}
            )

        form_inputs = [
            {
                "formName": self.__form_name,
                "typeIdentifier": self.__form_type_id,
                "typeRevision": "1",
                "content": json.dumps(mpg_forms),
            }
        ]

        return Asset(
            name=describe_resp["ModelPackageGroupName"],
            type_id=self.__asset_type_id,
            external_id=describe_resp["ModelPackageGroupArn"],
            forms_input=form_inputs,
        )
