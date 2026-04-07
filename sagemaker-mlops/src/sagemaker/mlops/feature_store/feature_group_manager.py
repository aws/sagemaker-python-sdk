# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""FeatureGroup with Lake Formation support."""

import json
import logging
from typing import Dict, List, Optional

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
from botocore.exceptions import ClientError
from pyiceberg.catalog import load_catalog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
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

    def _has_lake_formation_config(self) -> bool:
        """Check if this feature group was created with Lake Formation governance.

        Note: Returns False if the object was not hydrated via get()/describe
        (i.e., constructed directly with just a name). In that case, the caller
        falls back to the generic IAM error message, which is still valid advice.
        """
        lf_config = getattr(self, "lake_formation_config", None)
        return lf_config is not None and lf_config != Unassigned()

    def _validate_table_ownership(self, table, database_name: str, table_name: str):
        """Validate that the Iceberg table belongs to this feature group by checking S3 location."""
        table_location = table.metadata.location if table.metadata else None
        s3_config = self.offline_store_config.s3_storage_config
        if s3_config and s3_config.s3_uri:
            expected_prefix = str(s3_config.s3_uri).rstrip("/")
            if table_location and not table_location.startswith(expected_prefix):
                logger.error(
                    f"Table ownership validation failed for feature group "
                    f"'{self.feature_group_name}'. The Glue table "
                    f"'{database_name}.{table_name}' has location '{table_location}' "
                    f"but the feature group's offline store is configured with "
                    f"S3 URI '{expected_prefix}'. This may indicate that the "
                    f"data_catalog_config is pointing to a table that does not belong "
                    f"to this feature group. To fix this, verify that the "
                    f"data_catalog_config.database and data_catalog_config.table_name "
                    f"in your feature group's offline_store_config match the correct "
                    f"Glue table for this feature group."
                )
                raise ValueError(
                    f"Table '{database_name}.{table_name}' location '{table_location}' "
                    f"does not match the feature group's S3 URI '{expected_prefix}'. "
                    f"The table may not belong to feature group '{self.feature_group_name}'."
                )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
    def _get_iceberg_properties(
        self,
        session: Optional[Session] = None,
        region: Optional[StrPipeVar] = None,
    ) -> Dict[str, any]:
        """
        Fetch the current Iceberg catalog table definition for the Feature Group's Iceberg offline store.

        Validates that the Feature Group has an Iceberg-formatted offline store
        and retrieves the table via the Iceberg catalog.

        Parameters:
            session: Optional boto3 session. If not provided, uses default credentials.
            region: Optional AWS region. If not provided, uses default region.

        Returns:
            Dict with keys:
            - 'database_name': The Iceberg catalog database name
            - 'table_name': The Iceberg catalog table name
            - 'table': The pyiceberg Table object
            - 'properties': The table properties dict

        Raises:
            ValueError: If offline_store_config is not configured or table_format is not Iceberg.
            RuntimeError: If the Iceberg catalog table load fails.
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

        if session is None:
            session = Session()
        region_str = str(region) if region else session.region_name
        catalog = load_catalog("glue", **{"type": "glue", "client.region": region_str})

        try:
            table = catalog.load_table(f"{database_name}.{table_name}")
            self._validate_table_ownership(table, database_name, table_name)

            return {
                "database_name": database_name,
                "table_name": table_name,
                "table": table,
                "properties": dict(table.properties),
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                if self._has_lake_formation_config():
                    raise PermissionError(
                        f"Access denied reading Iceberg properties for '{self.feature_group_name}'. "
                        f"This feature group uses Lake Formation governance — ensure you have "
                        f"SELECT and DESCRIBE permissions on the table in Lake Formation, "
                        f"in addition to IAM permissions."
                    ) from e
                raise PermissionError(
                    f"Access denied reading Iceberg properties for '{self.feature_group_name}'. "
                    f"Ensure your role has glue:GetTable permission "
                    f"on the feature group's Glue table."
                ) from e
            raise RuntimeError(
                f"Failed to get Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"Failed to get Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RuntimeError),
        reraise=True,
    )
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
        table = result["table"]
        current_properties = result["properties"]

        self._validate_table_ownership(table, database_name, table_name)

        # Compute before/after diff for audit logging
        changed = {}
        for key, new_value in iceberg_properties.properties.items():
            old_value = current_properties.get(key)
            if old_value != new_value:
                changed[key] = {"old": old_value, "new": new_value}

        logger.info(
            f"Updating Iceberg properties for feature group '{self.feature_group_name}' "
            f"(database={database_name}, table={table_name}). "
            f"Property changes: {changed}"
        )

        try:
            with table.transaction() as txn:
                txn.set_properties(**iceberg_properties.properties)

            logger.info(
                f"Successfully updated Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Properties applied: {changed}"
            )

            return {
                "database": database_name,
                "table": table_name,
                "properties_updated": iceberg_properties.properties,
            }

        except ClientError as e:
            if e.response["Error"]["Code"] == "AccessDeniedException":
                if self._has_lake_formation_config():
                    raise PermissionError(
                        f"Access denied updating Iceberg properties for '{self.feature_group_name}'. "
                        f"This feature group uses Lake Formation governance — ensure you have "
                        f"ALTER permission on the table in Lake Formation, "
                        f"in addition to IAM permissions."
                    ) from e
                raise PermissionError(
                    f"Access denied updating Iceberg properties for '{self.feature_group_name}'. "
                    f"Ensure your role has glue:UpdateTable permission "
                    f"on the feature group's Glue table."
                ) from e
            logger.error(
                f"Failed to update Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Attempted changes: {changed}. Error: {e}"
            )
            raise RuntimeError(
                f"Failed to update Iceberg properties for '{self.feature_group_name}': {e}"
            ) from e
        except Exception as e:
            logger.error(
                f"Failed to update Iceberg properties for feature group "
                f"'{self.feature_group_name}'. Attempted changes: {changed}. Error: {e}"
            )
            raise RuntimeError(
                f"Failed to update Iceberg properties for {self.feature_group_name}: {e}"
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
            all_properties = result["properties"]
            approved_properties = {
                k: v for k, v in all_properties.items()
                if k in _APPROVED_ICEBERG_PROPERTIES
            }
            feature_group.iceberg_properties = IcebergProperties(
                properties=approved_properties
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
            try:
                feature_group._update_iceberg_properties(
                    iceberg_properties=iceberg_properties,
                    session=session,
                    region=region,
                )
            except Exception as e:
                logger.error(
                    f"Feature group '{feature_group.feature_group_name}' was created "
                    f"successfully but failed to update Iceberg properties: {e}."
                    f"Please now run update on the created Feature Group with the"
                    f"Iceberg Properties to avoid recreating your Feature Group again."
                )
                raise

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

        # Only call parent update if there are non-iceberg args to pass
        result = None
        if args or kwargs:
            try:
                result = super().update(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Feature group '{self.feature_group_name}' was not updated successfully: {e}"
                )

        # Update Iceberg properties if requested
        if iceberg_properties is not None and iceberg_properties.properties:
            try:
                self._update_iceberg_properties(
                    iceberg_properties=iceberg_properties,
                    session=session,
                    region=region,
                )
            except Exception as e:
                logger.error(
                    f"Feature group '{self.feature_group_name}' was updated successfully "
                    f"but failed to update Iceberg properties: {e}"
                )
                raise

        return result if result is not None else self
