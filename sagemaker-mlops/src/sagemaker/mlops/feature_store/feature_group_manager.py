# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import json
import logging
from typing import Dict, List, Optional

import botocore.exceptions
from pydantic import model_validator

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.resources import Base
from sagemaker.core.shapes import (
    AddOnlineStoreReplicaAction,
    FeatureDefinition,
    OfflineStoreConfig,
    OnlineStoreConfig,
    OnlineStoreConfigUpdate,
    Tag,
    ThroughputConfig,
    ThroughputConfigUpdate,
)
from sagemaker.core.shapes import Unassigned
from sagemaker.core.helper.pipeline_variable import StrPipeVar
from sagemaker.core.s3.utils import parse_s3_url
from sagemaker.core.common_utils import aws_partition
from boto3 import Session
from sagemaker.mlops.feature_store.feature_utils import _APPROVED_ICEBERG_PROPERTIES


logger = logging.getLogger(__name__)


class IcebergProperties(Base):
    """Configuration for Iceberg table properties in a Feature Group offline store.

    Use this to customize Iceberg table behavior such as compaction settings,
    snapshot retention, and other Iceberg-specific configurations.

    Attributes:
        properties: A dictionary mapping Iceberg property names to their values.
            Common properties include:
            - 'write.target-file-size-bytes': Target size for data files
            - 'commit.manifest.min-count-to-merge': Min manifests before merging
            - 'history.expire.max-snapshot-age-ms': Max age for snapshot expiration
    """

    properties: Optional[Dict[str, str]] = None

    @model_validator(mode="after")
    def validate_property_keys(self):
        if self.properties is None:
            return self
        invalid_keys = set(self.properties.keys()) - _APPROVED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Approved properties are: {_APPROVED_ICEBERG_PROPERTIES}"
            )
        # Check for no duplicate keys
        if len(set(self.properties.keys())) != len(self.properties.keys()):
            raise ValueError(
                f"Invalid duplicate properties. Please only have 1 of each property."
            )
        return self


class FeatureGroupManager(FeatureGroup):

    # Attribute for Iceberg table properties (populated by get() when include_iceberg_properties=True)
    iceberg_properties: Optional[IcebergProperties] = None

    # Inherit parent docstring and append our additions
    if FeatureGroup.__doc__ and __doc__:
        __doc__ = FeatureGroup.__doc__

    def _get_iceberg_properties(
        self,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Fetch the current Glue table definition for the Feature Group's Iceberg offline store.

        Validates that the Feature Group has an Iceberg-formatted offline store,
        retrieves the Glue table, and strips non-TableInput fields.

        Parameters:
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict with keys:
            - 'database_name': The Glue database name
            - 'table_name': The Glue table name
            - 'table_input': The cleaned Glue TableInput dict
            - 'glue_client': The Glue client used for the request

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Glue get_table call fails.
        """
        # Validate offline store is configured
        if self.offline_store_config is None or self.offline_store_config == Unassigned():
            raise ValueError(
                "Cannot update Iceberg properties: offline_store_config is not configured"
            )

        # Validate table format is Iceberg
        if (
            self.offline_store_config.table_format is None
            or str(self.offline_store_config.table_format) != "Iceberg"
        ):
            raise ValueError(
                "Cannot update Iceberg properties: table_format must be 'Iceberg'"
            )

        # Get database and table name from data_catalog_config
        data_catalog_config = self.offline_store_config.data_catalog_config
        if data_catalog_config is None:
            raise ValueError(
                "Cannot update Iceberg properties: data_catalog_config is not available"
            )

        database_name = str(data_catalog_config.database)
        table_name = str(data_catalog_config.table_name)

        # Get Glue client
        if session is None:
            session = Session()

        region_str = str(region) if region else session.region_name
        glue_client = session.client("glue", region_name=region_str)

        try:
            # Get current table definition
            response = glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name,
            )
            table_input = response["Table"]

            # Remove fields that shouldn't be in TableInput
            fields_to_remove = [
                "DatabaseName",
                "CreateTime",
                "UpdateTime",
                "CreatedBy",
                "IsRegisteredWithLakeFormation",
                "CatalogId",
                "VersionId",
                "FederatedTable",
                "IsMultiDialectView",
                "IsMaterializedView",
            ]
            for field in fields_to_remove:
                table_input.pop(field, None)

            return {
                "database_name": database_name,
                "table_name": table_name,
                "table_input": table_input,
                "glue_client": glue_client,
            }

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: "
                f"[{error_code}] {error_message}"
            ) from e

    def _update_iceberg_properties(
        self,
        iceberg_properties: IcebergProperties,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Update Iceberg table properties for the Feature Group's offline store.

        This method updates the Glue table properties for an Iceberg-formatted
        offline store. The Feature Group must have an offline store configured
        with table_format='Iceberg'.

        Parameters:
            iceberg_properties: IcebergProperties object containing the properties to set.
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict containing the update results with keys:
            - 'database': The Glue database name
            - 'table': The Glue table name
            - 'properties_updated': The properties that were updated

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Glue table update fails.
        """
        # Validate iceberg_properties has properties to update
        if iceberg_properties is None or not iceberg_properties.properties:
            raise ValueError(
                "iceberg_properties must contain at least one property to update"
            )

        invalid_keys = set(iceberg_properties.properties.keys()) - _APPROVED_ICEBERG_PROPERTIES
        if invalid_keys:
            raise ValueError(
                f"Invalid iceberg properties: {invalid_keys}. "
                f"Approved properties are: {_APPROVED_ICEBERG_PROPERTIES}"
            )

         # Check for no duplicate keys
        if len(set(iceberg_properties.properties.keys())) != len(iceberg_properties.properties.keys()):
            raise ValueError(
                f"Invalid duplicate properties. Please only have 1 of each property."
            )

        result = self._get_iceberg_properties(session=session, region=region)
        database_name = result["database_name"]
        table_name = result["table_name"]
        table_input = result["table_input"]
        glue_client = result["glue_client"]

        logger.info(
            f"Updating Iceberg properties for {self.feature_group_name} "
            f"(database={database_name}, table={table_name})"
        )

        try:
            # Update parameters with new Iceberg properties
            if "Parameters" not in table_input:
                table_input["Parameters"] = {}

            for key, value in iceberg_properties.properties.items():
                table_input["Parameters"][key] = value

            # Update the table
            glue_client.update_table(
                DatabaseName=database_name,
                TableInput=table_input,
            )

            logger.info(
                f"Successfully updated Iceberg properties for {self.feature_group_name}"
            )

            return {
                "database": database_name,
                "table": table_name,
                "properties_updated": iceberg_properties.properties,
            }

        except botocore.exceptions.ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "Unknown")
            error_message = e.response.get("Error", {}).get("Message", str(e))
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: "
                f"[{error_code}] {error_message}"
            ) from e

    @classmethod
    def get(
        cls,
        *args,
        include_iceberg_properties: bool = False,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Get a FeatureGroup resource with optional Iceberg property retrieval.

        Accepts all parameters from FeatureGroup.get(), plus:

        Parameters:
            include_iceberg_properties: If True, fetches Iceberg table properties
                from Glue and stores them in the iceberg_properties attribute.
                Only applies to Feature Groups with table_format='Iceberg'.

        Returns:
            The FeatureGroup resource.
        """
        session = kwargs.get("session")
        region = kwargs.get("region")

        feature_group = super().get(*args, **kwargs)

        if include_iceberg_properties:
            result = feature_group._get_iceberg_properties(session=session, region=region)
            feature_group.iceberg_properties = IcebergProperties(
                properties=result["table_input"].get("Parameters", {})
            )

        return feature_group

    @classmethod
    def create(
        cls,
        *args,
        iceberg_properties: Optional[IcebergProperties] = None,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Create a FeatureGroup resource with optional Lake Formation governance and Iceberg properties.

        Accepts all parameters from FeatureGroup.create(), plus:

        Parameters:
            lake_formation_config: Optional LakeFormationConfig to configure Lake Formation
                governance. When enabled=True, requires offline_store_config and role_arn.
            iceberg_properties: Optional IcebergProperties to configure Iceberg table
                properties for the offline store. Requires offline_store_config with
                table_format='Iceberg'.

        Returns:
            The FeatureGroup resource.
        """
        offline_store_config = kwargs.get("offline_store_config")
        role_arn = kwargs.get("role_arn")
        session = kwargs.get("session")
        region = kwargs.get("region")

        # Validation for Iceberg properties
        if iceberg_properties is not None and iceberg_properties.properties:
            if offline_store_config is None:
                raise ValueError(
                    "iceberg_properties requires offline_store_config to be configured"
                )
            if (
                offline_store_config.table_format is None
                or str(offline_store_config.table_format) != "Iceberg"
            ):
                raise ValueError(
                    "iceberg_properties requires offline_store_config.table_format to be 'Iceberg'"
                )

        feature_group = super().create(*args, **kwargs)

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            # Wait for feature group to be created before updating Iceberg properties
            feature_group.wait_for_status(target_status="Created")
            feature_group._update_iceberg_properties(
                iceberg_properties=iceberg_properties,
                session=session,
                region=region,
            )

        return feature_group

    def update(
        self,
        *args,
        iceberg_properties: Optional[IcebergProperties] = None,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
        **kwargs,
    ) -> Optional["FeatureGroup"]:
        """
        Update a FeatureGroup resource with optional Iceberg property updates.

        Accepts all parameters from FeatureGroup.update(), plus:

        Parameters:
            iceberg_properties: Optional IcebergProperties to update Iceberg table
                properties for the offline store. Requires offline_store_config with
                table_format='Iceberg'.
            session: Boto3 session for Iceberg property updates.
            region: Region name for Iceberg property updates.

        Returns:
            The FeatureGroup resource.
        """

        offline_store_config = self.offline_store_config

        # Validation for Iceberg properties
        if iceberg_properties is not None and iceberg_properties.properties:
            if offline_store_config is None or offline_store_config == Unassigned():
                raise ValueError(
                    "iceberg_properties requires offline_store_config to be configured"
                )
            if (
                offline_store_config.table_format is None
                or str(offline_store_config.table_format) != "Iceberg"
            ):
                raise ValueError(
                    "iceberg_properties requires offline_store_config.table_format to be 'Iceberg'"
                )

        result = super().update(*args, **kwargs)

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            self._update_iceberg_properties(
                iceberg_properties=iceberg_properties,
                session=session,
                region=region,
            )

        return result
