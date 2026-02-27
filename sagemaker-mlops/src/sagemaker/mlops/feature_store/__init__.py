# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""SageMaker FeatureStore V3 - powered by sagemaker-core."""

# Resources from core
from sagemaker.core.resources import FeatureGroup, FeatureMetadata

# Shapes from core (Pydantic - no to_dict() needed)
from sagemaker.core.shapes import (
    DataCatalogConfig,
    FeatureParameter,
    FeatureValue,
    Filter,
    OfflineStoreConfig,
    OnlineStoreConfig,
    OnlineStoreSecurityConfig,
    S3StorageConfig,
    SearchExpression,
    ThroughputConfig,
    TtlDuration,
)

# Enums (local - core uses strings)
from sagemaker.mlops.feature_store.inputs import (
    DeletionModeEnum,
    ExpirationTimeResponseEnum,
    FilterOperatorEnum,
    OnlineStoreStorageTypeEnum,
    ResourceEnum,
    SearchOperatorEnum,
    SortOrderEnum,
    TableFormatEnum,
    TargetStoreEnum,
    ThroughputModeEnum,
)

# Feature Definition helpers (local)
from sagemaker.mlops.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
    CollectionTypeEnum,
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    StringFeatureDefinition,
    ListCollectionType,
    SetCollectionType,
    VectorCollectionType,
)

# Utility functions (local)
from sagemaker.mlops.feature_store.feature_utils import (
    as_hive_ddl,
    create_athena_query,
    get_session_from_role,
    ingest_dataframe,
    load_feature_definitions_from_dataframe,
)

# Classes (local)
from sagemaker.mlops.feature_store.athena_query import AthenaQuery
from sagemaker.mlops.feature_store.dataset_builder import (
    DatasetBuilder,
    FeatureGroupToBeMerged,
    JoinComparatorEnum,
    JoinTypeEnum,
    TableType,
)
from sagemaker.mlops.feature_store.ingestion_manager_pandas import (
    IngestionError,
    IngestionManagerPandas,
)

__all__ = [
    # Resources
    "FeatureGroup",
    "FeatureMetadata",
    # Shapes
    "DataCatalogConfig",
    "FeatureParameter",
    "FeatureValue",
    "Filter",
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
