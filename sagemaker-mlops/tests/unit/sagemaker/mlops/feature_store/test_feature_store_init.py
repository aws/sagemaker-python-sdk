# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests to verify Feature Store module discoverability and imports."""
import pytest


class TestFeatureStoreResourceImports:
    """Test that core resources are importable from feature_store."""

    def test_feature_group_importable(self):
        from sagemaker.mlops.feature_store import FeatureGroup
        assert FeatureGroup is not None

    def test_feature_metadata_importable(self):
        from sagemaker.mlops.feature_store import FeatureMetadata
        assert FeatureMetadata is not None


class TestFeatureStoreShapesImports:
    """Test that core shapes are importable from feature_store."""

    def test_online_store_config_importable(self):
        from sagemaker.mlops.feature_store import OnlineStoreConfig
        assert OnlineStoreConfig is not None

    def test_offline_store_config_importable(self):
        from sagemaker.mlops.feature_store import OfflineStoreConfig
        assert OfflineStoreConfig is not None

    def test_s3_storage_config_importable(self):
        from sagemaker.mlops.feature_store import S3StorageConfig
        assert S3StorageConfig is not None

    def test_ttl_duration_importable(self):
        from sagemaker.mlops.feature_store import TtlDuration
        assert TtlDuration is not None

    def test_feature_value_importable(self):
        from sagemaker.mlops.feature_store import FeatureValue
        assert FeatureValue is not None

    def test_feature_parameter_importable(self):
        from sagemaker.mlops.feature_store import FeatureParameter
        assert FeatureParameter is not None

    def test_filter_importable(self):
        from sagemaker.mlops.feature_store import Filter
        assert Filter is not None

    def test_search_expression_importable(self):
        from sagemaker.mlops.feature_store import SearchExpression
        assert SearchExpression is not None

    def test_throughput_config_importable(self):
        from sagemaker.mlops.feature_store import ThroughputConfig
        assert ThroughputConfig is not None

    def test_data_catalog_config_importable(self):
        from sagemaker.mlops.feature_store import DataCatalogConfig
        assert DataCatalogConfig is not None

    def test_online_store_security_config_importable(self):
        from sagemaker.mlops.feature_store import OnlineStoreSecurityConfig
        assert OnlineStoreSecurityConfig is not None


class TestFeatureStoreEnumImports:
    """Test that all enums are importable from feature_store."""

    def test_target_store_enum_importable(self):
        from sagemaker.mlops.feature_store import TargetStoreEnum
        assert TargetStoreEnum is not None

    def test_deletion_mode_enum_importable(self):
        from sagemaker.mlops.feature_store import DeletionModeEnum
        assert DeletionModeEnum is not None

    def test_throughput_mode_enum_importable(self):
        from sagemaker.mlops.feature_store import ThroughputModeEnum
        assert ThroughputModeEnum is not None

    def test_table_format_enum_importable(self):
        from sagemaker.mlops.feature_store import TableFormatEnum
        assert TableFormatEnum is not None

    def test_online_store_storage_type_enum_importable(self):
        from sagemaker.mlops.feature_store import OnlineStoreStorageTypeEnum
        assert OnlineStoreStorageTypeEnum is not None

    def test_filter_operator_enum_importable(self):
        from sagemaker.mlops.feature_store import FilterOperatorEnum
        assert FilterOperatorEnum is not None

    def test_resource_enum_importable(self):
        from sagemaker.mlops.feature_store import ResourceEnum
        assert ResourceEnum is not None

    def test_search_operator_enum_importable(self):
        from sagemaker.mlops.feature_store import SearchOperatorEnum
        assert SearchOperatorEnum is not None

    def test_sort_order_enum_importable(self):
        from sagemaker.mlops.feature_store import SortOrderEnum
        assert SortOrderEnum is not None

    def test_expiration_time_response_enum_importable(self):
        from sagemaker.mlops.feature_store import ExpirationTimeResponseEnum
        assert ExpirationTimeResponseEnum is not None


class TestFeatureStoreUtilityFunctionImports:
    """Test that utility functions are importable from feature_store."""

    def test_ingest_dataframe_importable(self):
        from sagemaker.mlops.feature_store import ingest_dataframe
        assert callable(ingest_dataframe)

    def test_create_athena_query_importable(self):
        from sagemaker.mlops.feature_store import create_athena_query
        assert callable(create_athena_query)

    def test_as_hive_ddl_importable(self):
        from sagemaker.mlops.feature_store import as_hive_ddl
        assert callable(as_hive_ddl)

    def test_load_feature_definitions_from_dataframe_importable(self):
        from sagemaker.mlops.feature_store import load_feature_definitions_from_dataframe
        assert callable(load_feature_definitions_from_dataframe)

    def test_get_session_from_role_importable(self):
        from sagemaker.mlops.feature_store import get_session_from_role
        assert callable(get_session_from_role)


class TestFeatureStoreClassImports:
    """Test that classes are importable from feature_store."""

    def test_athena_query_importable(self):
        from sagemaker.mlops.feature_store import AthenaQuery
        assert AthenaQuery is not None

    def test_dataset_builder_importable(self):
        from sagemaker.mlops.feature_store import DatasetBuilder
        assert DatasetBuilder is not None

    def test_ingestion_manager_pandas_importable(self):
        from sagemaker.mlops.feature_store import IngestionManagerPandas
        assert IngestionManagerPandas is not None

    def test_ingestion_error_importable(self):
        from sagemaker.mlops.feature_store import IngestionError
        assert IngestionError is not None

    def test_feature_group_to_be_merged_importable(self):
        from sagemaker.mlops.feature_store import FeatureGroupToBeMerged
        assert FeatureGroupToBeMerged is not None

    def test_join_type_enum_importable(self):
        from sagemaker.mlops.feature_store import JoinTypeEnum
        assert JoinTypeEnum is not None

    def test_join_comparator_enum_importable(self):
        from sagemaker.mlops.feature_store import JoinComparatorEnum
        assert JoinComparatorEnum is not None

    def test_table_type_importable(self):
        from sagemaker.mlops.feature_store import TableType
        assert TableType is not None


class TestFeatureDefinitionHelperImports:
    """Test that feature definition helpers are importable from feature_store."""

    def test_feature_definition_importable(self):
        from sagemaker.mlops.feature_store import FeatureDefinition
        assert FeatureDefinition is not None

    def test_feature_type_enum_importable(self):
        from sagemaker.mlops.feature_store import FeatureTypeEnum
        assert FeatureTypeEnum is not None

    def test_collection_type_enum_importable(self):
        from sagemaker.mlops.feature_store import CollectionTypeEnum
        assert CollectionTypeEnum is not None

    def test_fractional_feature_definition_importable(self):
        from sagemaker.mlops.feature_store import FractionalFeatureDefinition
        assert callable(FractionalFeatureDefinition)

    def test_integral_feature_definition_importable(self):
        from sagemaker.mlops.feature_store import IntegralFeatureDefinition
        assert callable(IntegralFeatureDefinition)

    def test_string_feature_definition_importable(self):
        from sagemaker.mlops.feature_store import StringFeatureDefinition
        assert callable(StringFeatureDefinition)

    def test_list_collection_type_importable(self):
        from sagemaker.mlops.feature_store import ListCollectionType
        assert ListCollectionType is not None

    def test_set_collection_type_importable(self):
        from sagemaker.mlops.feature_store import SetCollectionType
        assert SetCollectionType is not None

    def test_vector_collection_type_importable(self):
        from sagemaker.mlops.feature_store import VectorCollectionType
        assert VectorCollectionType is not None


class TestAllExportsMatchAllList:
    """Test that all names in __all__ are actually importable."""

    def test_all_exports_are_importable(self):
        import sagemaker.mlops.feature_store as fs_module
        all_names = fs_module.__all__
        assert len(all_names) > 0, "__all__ should not be empty"
        for name in all_names:
            assert hasattr(fs_module, name), (
                f"{name} is listed in __all__ but not importable from "
                f"sagemaker.mlops.feature_store"
            )

    def test_no_extra_public_names_missing_from_all(self):
        """Verify key public names are in __all__."""
        import sagemaker.mlops.feature_store as fs_module
        all_names = set(fs_module.__all__)
        # Check a representative set of important names
        expected_names = {
            "FeatureGroup",
            "FeatureMetadata",
            "FeatureValue",
            "OnlineStoreConfig",
            "OfflineStoreConfig",
            "ingest_dataframe",
            "create_athena_query",
            "AthenaQuery",
            "DatasetBuilder",
            "IngestionManagerPandas",
            "FeatureDefinition",
        }
        for name in expected_names:
            assert name in all_names, (
                f"{name} should be in __all__ but is missing"
            )


class TestMlopsInitIncludesFeatureStore:
    """Test that feature_store is listed in sagemaker.mlops.__all__."""

    def test_feature_store_in_mlops_all(self):
        import sagemaker.mlops as mlops_module
        assert "feature_store" in mlops_module.__all__, (
            "feature_store should be listed in sagemaker.mlops.__all__ "
            "for discoverability"
        )
