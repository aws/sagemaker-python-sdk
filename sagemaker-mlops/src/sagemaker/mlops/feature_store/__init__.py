# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""SageMaker Feature Store module.

This is the canonical location for Feature Store functionality in SDK V3.
"""
from __future__ import annotations

import enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


# Enums
class DeletionModeEnum(enum.Enum):
    """Enum for deletion modes."""

    SOFT_DELETE = "SoftDelete"
    HARD_DELETE = "HardDelete"


class ExpirationTimeResponseEnum(enum.Enum):
    """Enum for expiration time response."""

    ENABLED = "Enabled"
    DISABLED = "Disabled"


class FilterOperatorEnum(enum.Enum):
    """Enum for filter operators."""

    EQUALS = "Equals"
    NOT_EQUALS = "NotEquals"
    GREATER_THAN = "GreaterThan"
    GREATER_THAN_OR_EQUAL_TO = "GreaterThanOrEqualTo"
    LESS_THAN = "LessThan"
    LESS_THAN_OR_EQUAL_TO = "LessThanOrEqualTo"
    CONTAINS = "Contains"
    EXISTS = "Exists"
    NOT_EXISTS = "NotExists"
    IN = "In"


class OnlineStoreStorageTypeEnum(enum.Enum):
    """Enum for online store storage types."""

    STANDARD = "Standard"
    IN_MEMORY = "InMemory"


class ResourceEnum(enum.Enum):
    """Enum for resources."""

    FEATURE_GROUP = "FeatureGroup"
    FEATURE_METADATA = "FeatureMetadata"


class SearchOperatorEnum(enum.Enum):
    """Enum for search operators."""

    AND = "And"
    OR = "Or"


class SortOrderEnum(enum.Enum):
    """Enum for sort orders."""

    ASCENDING = "Ascending"
    DESCENDING = "Descending"


class TableFormatEnum(enum.Enum):
    """Enum for table formats."""

    DEFAULT = "Default"
    GLUE = "Glue"
    ICEBERG = "Iceberg"


class TargetStoreEnum(enum.Enum):
    """Enum for target stores."""

    ONLINE_STORE = "OnlineStore"
    OFFLINE_STORE = "OfflineStore"


class ThroughputModeEnum(enum.Enum):
    """Enum for throughput modes."""

    ON_DEMAND = "OnDemand"
    PROVISIONED = "Provisioned"


class FeatureTypeEnum(enum.Enum):
    """Enum for feature types."""

    INTEGRAL = "Integral"
    FRACTIONAL = "Fractional"
    STRING = "String"


class CollectionTypeEnum(enum.Enum):
    """Enum for collection types."""

    LIST = "List"
    SET = "Set"
    VECTOR = "Vector"


class JoinComparatorEnum(enum.Enum):
    """Enum for join comparators."""

    GREATER_THAN = "GreaterThan"
    GREATER_THAN_OR_EQUAL_TO = "GreaterThanOrEqualTo"
    LESS_THAN = "LessThan"
    LESS_THAN_OR_EQUAL_TO = "LessThanOrEqualTo"
    EQUALS = "Equals"


class JoinTypeEnum(enum.Enum):
    """Enum for join types."""

    INNER = "Inner"
    LEFT = "Left"
    RIGHT = "Right"
    FULL = "Full"
    CROSS = "Cross"


# Shapes / Configuration classes
class DataCatalogConfig:
    """Configuration for the Data Catalog."""

    def __init__(
        self,
        table_name: str,
        catalog: str,
        database: str,
    ) -> None:
        self.table_name = table_name
        self.catalog = catalog
        self.database = database


class FeatureParameter:
    """A parameter for a feature."""

    def __init__(self, key: str, value: str) -> None:
        self.key = key
        self.value = value


class FeatureValue:
    """A value for a feature."""

    def __init__(self, feature_name: str, value_as_string: str) -> None:
        self.feature_name = feature_name
        self.value_as_string = value_as_string


class Filter:
    """A filter for searching feature groups."""

    def __init__(
        self,
        name: str,
        operator: Optional[FilterOperatorEnum] = None,
        value: Optional[str] = None,
    ) -> None:
        self.name = name
        self.operator = operator
        self.value = value


class LakeFormationConfig:
    """Configuration for Lake Formation."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled


class IcebergProperties:
    """Properties for Iceberg table format."""

    def __init__(
        self,
        table_name: Optional[str] = None,
        catalog: Optional[str] = None,
        database: Optional[str] = None,
    ) -> None:
        self.table_name = table_name
        self.catalog = catalog
        self.database = database


class S3StorageConfig:
    """Configuration for S3 storage."""

    def __init__(
        self,
        s3_uri: str,
        kms_key_id: Optional[str] = None,
        resolved_output_s3_uri: Optional[str] = None,
    ) -> None:
        self.s3_uri = s3_uri
        self.kms_key_id = kms_key_id
        self.resolved_output_s3_uri = resolved_output_s3_uri


class OfflineStoreConfig:
    """Configuration for the offline store."""

    def __init__(
        self,
        s3_storage_config: S3StorageConfig,
        disable_glue_table_creation: bool = False,
        data_catalog_config: Optional[DataCatalogConfig] = None,
        table_format: Optional[TableFormatEnum] = None,
    ) -> None:
        self.s3_storage_config = s3_storage_config
        self.disable_glue_table_creation = disable_glue_table_creation
        self.data_catalog_config = data_catalog_config
        self.table_format = table_format


class OnlineStoreSecurityConfig:
    """Security configuration for the online store."""

    def __init__(self, kms_key_id: Optional[str] = None) -> None:
        self.kms_key_id = kms_key_id


class OnlineStoreConfig:
    """Configuration for the online store."""

    def __init__(
        self,
        enable_online_store: bool = True,
        security_config: Optional[OnlineStoreSecurityConfig] = None,
        storage_type: Optional[OnlineStoreStorageTypeEnum] = None,
        ttl_duration: Optional["TtlDuration"] = None,
    ) -> None:
        self.enable_online_store = enable_online_store
        self.security_config = security_config
        self.storage_type = storage_type
        self.ttl_duration = ttl_duration


class SearchExpression:
    """A search expression for feature groups."""

    def __init__(
        self,
        filters: Optional[List[Filter]] = None,
        operator: Optional[SearchOperatorEnum] = None,
        nested_expressions: Optional[List["SearchExpression"]] = None,
    ) -> None:
        self.filters = filters or []
        self.operator = operator
        self.nested_expressions = nested_expressions or []


class ThroughputConfig:
    """Configuration for throughput."""

    def __init__(
        self,
        throughput_mode: ThroughputModeEnum = ThroughputModeEnum.ON_DEMAND,
        provisioned_read_capacity_units: Optional[int] = None,
        provisioned_write_capacity_units: Optional[int] = None,
    ) -> None:
        self.throughput_mode = throughput_mode
        self.provisioned_read_capacity_units = provisioned_read_capacity_units
        self.provisioned_write_capacity_units = provisioned_write_capacity_units


class TtlDuration:
    """Time-to-live duration configuration."""

    def __init__(self, unit: str, value: int) -> None:
        self.unit = unit
        self.value = value


# Feature Definitions
class FeatureDefinition:
    """Definition of a feature."""

    def __init__(
        self,
        feature_name: str,
        feature_type: FeatureTypeEnum,
        collection_type: Optional[CollectionTypeEnum] = None,
    ) -> None:
        self.feature_name = feature_name
        self.feature_type = feature_type
        self.collection_type = collection_type


class FractionalFeatureDefinition(FeatureDefinition):
    """Definition for a fractional feature."""

    def __init__(self, feature_name: str) -> None:
        super().__init__(feature_name=feature_name, feature_type=FeatureTypeEnum.FRACTIONAL)


class IntegralFeatureDefinition(FeatureDefinition):
    """Definition for an integral feature."""

    def __init__(self, feature_name: str) -> None:
        super().__init__(feature_name=feature_name, feature_type=FeatureTypeEnum.INTEGRAL)


class StringFeatureDefinition(FeatureDefinition):
    """Definition for a string feature."""

    def __init__(self, feature_name: str) -> None:
        super().__init__(feature_name=feature_name, feature_type=FeatureTypeEnum.STRING)


class ListCollectionType:
    """List collection type."""

    collection_type: CollectionTypeEnum = CollectionTypeEnum.LIST


class SetCollectionType:
    """Set collection type."""

    collection_type: CollectionTypeEnum = CollectionTypeEnum.SET


class VectorCollectionType:
    """Vector collection type."""

    def __init__(self, dimension: int) -> None:
        self.collection_type = CollectionTypeEnum.VECTOR
        self.dimension = dimension


# Resources
class FeatureMetadata:
    """Metadata for a feature in a Feature Group."""

    def __init__(
        self,
        feature_group_name: str,
        feature_name: str,
        parameters: Optional[List[FeatureParameter]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.feature_group_name = feature_group_name
        self.feature_name = feature_name
        self.parameters = parameters or []
        self.description = description


class FeatureGroup:
    """Represents a SageMaker Feature Store Feature Group."""

    def __init__(
        self,
        name: str,
        feature_definitions: Optional[List[FeatureDefinition]] = None,
        record_identifier_name: Optional[str] = None,
        event_time_feature_name: Optional[str] = None,
        online_store_config: Optional[OnlineStoreConfig] = None,
        offline_store_config: Optional[OfflineStoreConfig] = None,
        role_arn: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        self.name = name
        self.feature_definitions = feature_definitions or []
        self.record_identifier_name = record_identifier_name
        self.event_time_feature_name = event_time_feature_name
        self.online_store_config = online_store_config
        self.offline_store_config = offline_store_config
        self.role_arn = role_arn
        self.description = description

    def create(self) -> Dict[str, Any]:
        """Create the feature group in SageMaker."""
        raise NotImplementedError("FeatureGroup.create() is not yet implemented in SDK V3")

    def delete(self) -> None:
        """Delete the feature group."""
        raise NotImplementedError("FeatureGroup.delete() is not yet implemented in SDK V3")

    def describe(self) -> Dict[str, Any]:
        """Describe the feature group."""
        raise NotImplementedError("FeatureGroup.describe() is not yet implemented in SDK V3")

    def put_record(self, record: List[FeatureValue]) -> None:
        """Put a record into the feature group."""
        raise NotImplementedError("FeatureGroup.put_record() is not yet implemented in SDK V3")

    def get_record(self, record_identifier_value: str) -> Dict[str, Any]:
        """Get a record from the feature group."""
        raise NotImplementedError("FeatureGroup.get_record() is not yet implemented in SDK V3")

    def delete_record(
        self,
        record_identifier_value: str,
        event_time: str,
        deletion_mode: DeletionModeEnum = DeletionModeEnum.SOFT_DELETE,
    ) -> None:
        """Delete a record from the feature group."""
        raise NotImplementedError(
            "FeatureGroup.delete_record() is not yet implemented in SDK V3"
        )


class FeatureGroupManager:
    """Manager for SageMaker Feature Store Feature Groups."""

    def __init__(self, session: Optional[Any] = None) -> None:
        self.session = session

    def create_feature_group(self, feature_group: FeatureGroup) -> Dict[str, Any]:
        """Create a feature group."""
        raise NotImplementedError(
            "FeatureGroupManager.create_feature_group() is not yet implemented in SDK V3"
        )

    def list_feature_groups(
        self,
        max_results: Optional[int] = None,
        next_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        """List feature groups."""
        raise NotImplementedError(
            "FeatureGroupManager.list_feature_groups() is not yet implemented in SDK V3"
        )

    def search(
        self,
        search_expression: Optional[SearchExpression] = None,
        sort_order: Optional[SortOrderEnum] = None,
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Search feature groups."""
        raise NotImplementedError(
            "FeatureGroupManager.search() is not yet implemented in SDK V3"
        )


# Utility functions
def as_hive_ddl(
    feature_group_name: str,
    feature_definitions: List[FeatureDefinition],
    s3_uri: str,
    record_identifier_name: str,
    event_time_feature_name: str,
) -> str:
    """Generate Hive DDL for a feature group.

    Args:
        feature_group_name: Name of the feature group.
        feature_definitions: List of feature definitions.
        s3_uri: S3 URI for the offline store.
        record_identifier_name: Name of the record identifier feature.
        event_time_feature_name: Name of the event time feature.

    Returns:
        Hive DDL string.
    """
    raise NotImplementedError("as_hive_ddl() is not yet implemented in SDK V3")


def create_athena_query(
    feature_group_name: str,
    catalog: str,
    database: str,
    table_name: str,
) -> "AthenaQuery":
    """Create an Athena query for a feature group.

    Args:
        feature_group_name: Name of the feature group.
        catalog: Athena catalog name.
        database: Athena database name.
        table_name: Athena table name.

    Returns:
        An AthenaQuery instance.
    """
    raise NotImplementedError("create_athena_query() is not yet implemented in SDK V3")


def get_session_from_role(role_arn: str, region: Optional[str] = None) -> Any:
    """Get a session from a role.

    Args:
        role_arn: ARN of the IAM role.
        region: AWS region name.

    Returns:
        A configured session.
    """
    raise NotImplementedError("get_session_from_role() is not yet implemented in SDK V3")


def ingest_dataframe(
    feature_group: FeatureGroup,
    dataframe: Any,
    max_workers: int = 3,
    max_processes: int = 1,
    wait: bool = True,
    timeout: Optional[float] = None,
) -> None:
    """Ingest a DataFrame into a feature group.

    Args:
        feature_group: The target feature group.
        dataframe: A pandas DataFrame to ingest.
        max_workers: Maximum number of threads per process.
        max_processes: Maximum number of processes.
        wait: Whether to wait for ingestion to complete.
        timeout: Timeout in seconds.
    """
    raise NotImplementedError("ingest_dataframe() is not yet implemented in SDK V3")


def load_feature_definitions_from_dataframe(
    dataframe: Any,
) -> List[FeatureDefinition]:
    """Load feature definitions from a DataFrame.

    Args:
        dataframe: A pandas DataFrame to derive feature definitions from.

    Returns:
        A list of FeatureDefinition objects.
    """
    raise NotImplementedError(
        "load_feature_definitions_from_dataframe() is not yet implemented in SDK V3"
    )


# Additional Classes
class AthenaQuery:
    """Athena query for feature groups."""

    def __init__(
        self,
        catalog: str,
        database: str,
        table_name: str,
        sagemaker_session: Optional[Any] = None,
    ) -> None:
        self.catalog = catalog
        self.database = database
        self.table_name = table_name
        self.sagemaker_session = sagemaker_session

    def run(
        self,
        query_string: str,
        output_location: str,
        workgroup: Optional[str] = None,
    ) -> str:
        """Run an Athena query.

        Args:
            query_string: SQL query string.
            output_location: S3 output location.
            workgroup: Athena workgroup.

        Returns:
            Query execution ID.
        """
        raise NotImplementedError("AthenaQuery.run() is not yet implemented in SDK V3")


class DatasetBuilder:
    """Builder for datasets from feature groups."""

    def __init__(
        self,
        sagemaker_session: Optional[Any] = None,
        base: Optional[Any] = None,
        output_path: Optional[str] = None,
    ) -> None:
        self.sagemaker_session = sagemaker_session
        self.base = base
        self.output_path = output_path

    def with_feature_group(
        self,
        feature_group: FeatureGroup,
        target_feature_name_in_base: Optional[str] = None,
        feature_name_in_target: Optional[str] = None,
        join_type: JoinTypeEnum = JoinTypeEnum.INNER,
    ) -> "DatasetBuilder":
        """Add a feature group to the dataset."""
        raise NotImplementedError(
            "DatasetBuilder.with_feature_group() is not yet implemented in SDK V3"
        )

    def to_dataframe(self) -> Any:
        """Build the dataset and return as a DataFrame."""
        raise NotImplementedError(
            "DatasetBuilder.to_dataframe() is not yet implemented in SDK V3"
        )


class FeatureGroupToBeMerged:
    """Feature group to be merged."""

    def __init__(
        self,
        feature_group: FeatureGroup,
        target_feature_name_in_base: Optional[str] = None,
        feature_name_in_target: Optional[str] = None,
        join_type: JoinTypeEnum = JoinTypeEnum.INNER,
        join_comparator: JoinComparatorEnum = JoinComparatorEnum.EQUALS,
    ) -> None:
        self.feature_group = feature_group
        self.target_feature_name_in_base = target_feature_name_in_base
        self.feature_name_in_target = feature_name_in_target
        self.join_type = join_type
        self.join_comparator = join_comparator


class IngestionError(Exception):
    """Error during ingestion."""

    def __init__(self, message: str, failed_rows: Optional[List[int]] = None) -> None:
        super().__init__(message)
        self.failed_rows = failed_rows or []


class IngestionManagerPandas:
    """Manager for ingesting Pandas DataFrames."""

    def __init__(
        self,
        feature_group: FeatureGroup,
        max_workers: int = 3,
        max_processes: int = 1,
    ) -> None:
        self.feature_group = feature_group
        self.max_workers = max_workers
        self.max_processes = max_processes

    def run(
        self,
        dataframe: Any,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        """Run ingestion of a DataFrame.

        Args:
            dataframe: A pandas DataFrame to ingest.
            wait: Whether to wait for ingestion to complete.
            timeout: Timeout in seconds.
        """
        raise NotImplementedError(
            "IngestionManagerPandas.run() is not yet implemented in SDK V3"
        )


class TableType:
    """Type of table."""

    def __init__(self, table_format: TableFormatEnum = TableFormatEnum.DEFAULT) -> None:
        self.table_format = table_format


__all__: list[str] = [
    # Resources
    "FeatureGroup",
    "FeatureGroupManager",
    "FeatureMetadata",
    # Shapes
    "DataCatalogConfig",
    "FeatureParameter",
    "FeatureValue",
    "Filter",
    "LakeFormationConfig",
    "IcebergProperties",
    "OfflineStoreConfig",
    "OnlineStoreConfig",
    "OnlineStoreSecurityConfig",
    "S3StorageConfig",
    "SearchExpression",
    "ThroughputConfig",
    "TtlDuration",
    # Enums
    "DeletionModeEnum",
    "ExpirationTimeResponseEnum",
    "FilterOperatorEnum",
    "OnlineStoreStorageTypeEnum",
    "ResourceEnum",
    "SearchOperatorEnum",
    "SortOrderEnum",
    "TableFormatEnum",
    "TargetStoreEnum",
    "ThroughputModeEnum",
    # Feature Definitions
    "FeatureDefinition",
    "FeatureTypeEnum",
    "CollectionTypeEnum",
    "FractionalFeatureDefinition",
    "IntegralFeatureDefinition",
    "StringFeatureDefinition",
    "ListCollectionType",
    "SetCollectionType",
    "VectorCollectionType",
    # Utility functions
    "as_hive_ddl",
    "create_athena_query",
    "get_session_from_role",
    "ingest_dataframe",
    "load_feature_definitions_from_dataframe",
    # Classes
    "AthenaQuery",
    "DatasetBuilder",
    "FeatureGroupToBeMerged",
    "IngestionError",
    "IngestionManagerPandas",
    "JoinComparatorEnum",
    "JoinTypeEnum",
    "TableType",
]
