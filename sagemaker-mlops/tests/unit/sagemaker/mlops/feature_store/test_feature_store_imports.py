# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Tests for Feature Store imports and backward compatibility."""
from __future__ import annotations

import enum
import importlib
import warnings

import pytest


class TestFeatureStoreImports:
    """Tests that imports from sagemaker.mlops.feature_store work correctly."""

    def test_import_feature_group(self):
        """Test that FeatureGroup can be imported."""
        from sagemaker.mlops.feature_store import FeatureGroup

        assert FeatureGroup is not None

    def test_import_feature_group_manager(self):
        """Test that FeatureGroupManager can be imported."""
        from sagemaker.mlops.feature_store import FeatureGroupManager

        assert FeatureGroupManager is not None

    def test_import_feature_definition(self):
        """Test that FeatureDefinition can be imported."""
        from sagemaker.mlops.feature_store import FeatureDefinition

        assert FeatureDefinition is not None

    def test_import_enums(self):
        """Test that enum classes can be imported and are proper enums."""
        from sagemaker.mlops.feature_store import (
            DeletionModeEnum,
            FeatureTypeEnum,
            ThroughputModeEnum,
            CollectionTypeEnum,
            FilterOperatorEnum,
            JoinTypeEnum,
            JoinComparatorEnum,
            OnlineStoreStorageTypeEnum,
            ResourceEnum,
            SearchOperatorEnum,
            SortOrderEnum,
            TableFormatEnum,
            TargetStoreEnum,
            ExpirationTimeResponseEnum,
        )

        assert issubclass(DeletionModeEnum, enum.Enum)
        assert issubclass(FeatureTypeEnum, enum.Enum)
        assert issubclass(ThroughputModeEnum, enum.Enum)
        assert issubclass(CollectionTypeEnum, enum.Enum)
        assert issubclass(FilterOperatorEnum, enum.Enum)
        assert issubclass(JoinTypeEnum, enum.Enum)
        assert issubclass(JoinComparatorEnum, enum.Enum)
        assert issubclass(OnlineStoreStorageTypeEnum, enum.Enum)
        assert issubclass(ResourceEnum, enum.Enum)
        assert issubclass(SearchOperatorEnum, enum.Enum)
        assert issubclass(SortOrderEnum, enum.Enum)
        assert issubclass(TableFormatEnum, enum.Enum)
        assert issubclass(TargetStoreEnum, enum.Enum)
        assert issubclass(ExpirationTimeResponseEnum, enum.Enum)

    def test_import_configs(self):
        """Test that configuration classes can be imported."""
        from sagemaker.mlops.feature_store import (
            DataCatalogConfig,
            OfflineStoreConfig,
            OnlineStoreConfig,
            OnlineStoreSecurityConfig,
            S3StorageConfig,
            ThroughputConfig,
            TtlDuration,
        )

        assert DataCatalogConfig is not None
        assert OfflineStoreConfig is not None
        assert OnlineStoreConfig is not None
        assert OnlineStoreSecurityConfig is not None
        assert S3StorageConfig is not None
        assert ThroughputConfig is not None
        assert TtlDuration is not None

    def test_import_utility_functions(self):
        """Test that utility functions can be imported."""
        from sagemaker.mlops.feature_store import (
            as_hive_ddl,
            create_athena_query,
            get_session_from_role,
            ingest_dataframe,
            load_feature_definitions_from_dataframe,
        )

        assert callable(as_hive_ddl)
        assert callable(create_athena_query)
        assert callable(get_session_from_role)
        assert callable(ingest_dataframe)
        assert callable(load_feature_definitions_from_dataframe)

    def test_import_all_public_api(self):
        """Test that __all__ is defined and contains expected entries."""
        import sagemaker.mlops.feature_store as fs_module

        assert hasattr(fs_module, "__all__")
        expected_classes = [
            "FeatureGroup",
            "FeatureGroupManager",
            "FeatureMetadata",
            "FeatureDefinition",
            "AthenaQuery",
            "DatasetBuilder",
            "IngestionError",
            "IngestionManagerPandas",
        ]
        for cls_name in expected_classes:
            assert cls_name in fs_module.__all__, f"{cls_name} missing from __all__"


class TestBackwardCompatibilityShim:
    """Tests that the backward compatibility shim emits DeprecationWarning."""

    def test_import_from_old_path_emits_deprecation_warning(self):
        """Test that importing from sagemaker.feature_store emits DeprecationWarning."""
        # Remove cached module to ensure the warning fires
        import sys

        if "sagemaker.feature_store" in sys.modules:
            del sys.modules["sagemaker.feature_store"]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            import sagemaker.feature_store  # noqa: F401

            assert len(w) >= 1
            deprecation_warnings = [
                x for x in w if issubclass(x.category, DeprecationWarning)
            ]
            assert len(deprecation_warnings) >= 1
            assert "deprecated" in str(deprecation_warnings[0].message).lower()
            assert "sagemaker.mlops.feature_store" in str(
                deprecation_warnings[0].message
            )

    def test_old_path_provides_same_classes(self):
        """Test that the old import path provides the same classes as the new path."""
        import sys

        if "sagemaker.feature_store" in sys.modules:
            del sys.modules["sagemaker.feature_store"]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import sagemaker.feature_store as old_fs

        import sagemaker.mlops.feature_store as new_fs

        assert old_fs.FeatureGroup is new_fs.FeatureGroup
        assert old_fs.FeatureGroupManager is new_fs.FeatureGroupManager
        assert old_fs.FeatureDefinition is new_fs.FeatureDefinition


class TestEnumBehavior:
    """Tests that enum classes behave correctly."""

    def test_feature_type_enum_values(self):
        """Test FeatureTypeEnum has expected values."""
        from sagemaker.mlops.feature_store import FeatureTypeEnum

        assert FeatureTypeEnum.INTEGRAL.value == "Integral"
        assert FeatureTypeEnum.FRACTIONAL.value == "Fractional"
        assert FeatureTypeEnum.STRING.value == "String"

    def test_deletion_mode_enum_values(self):
        """Test DeletionModeEnum has expected values."""
        from sagemaker.mlops.feature_store import DeletionModeEnum

        assert DeletionModeEnum.SOFT_DELETE.value == "SoftDelete"
        assert DeletionModeEnum.HARD_DELETE.value == "HardDelete"

    def test_throughput_mode_enum_values(self):
        """Test ThroughputModeEnum has expected values."""
        from sagemaker.mlops.feature_store import ThroughputModeEnum

        assert ThroughputModeEnum.ON_DEMAND.value == "OnDemand"
        assert ThroughputModeEnum.PROVISIONED.value == "Provisioned"


class TestClassInstantiation:
    """Tests that classes can be instantiated with proper arguments."""

    def test_feature_group_instantiation(self):
        """Test FeatureGroup can be instantiated."""
        from sagemaker.mlops.feature_store import FeatureGroup

        fg = FeatureGroup(name="test-feature-group")
        assert fg.name == "test-feature-group"

    def test_feature_definition_instantiation(self):
        """Test FeatureDefinition can be instantiated."""
        from sagemaker.mlops.feature_store import FeatureDefinition, FeatureTypeEnum

        fd = FeatureDefinition(
            feature_name="my_feature", feature_type=FeatureTypeEnum.STRING
        )
        assert fd.feature_name == "my_feature"
        assert fd.feature_type == FeatureTypeEnum.STRING

    def test_s3_storage_config_instantiation(self):
        """Test S3StorageConfig can be instantiated."""
        from sagemaker.mlops.feature_store import S3StorageConfig

        config = S3StorageConfig(s3_uri="s3://my-bucket/prefix")
        assert config.s3_uri == "s3://my-bucket/prefix"
        assert config.kms_key_id is None

    def test_ingestion_error_is_exception(self):
        """Test IngestionError is a proper exception."""
        from sagemaker.mlops.feature_store import IngestionError

        assert issubclass(IngestionError, Exception)
        error = IngestionError("test error", failed_rows=[0, 1, 2])
        assert str(error) == "test error"
        assert error.failed_rows == [0, 1, 2]
