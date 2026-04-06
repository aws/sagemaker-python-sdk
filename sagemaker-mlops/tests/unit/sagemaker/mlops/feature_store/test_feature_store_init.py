# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for feature_store __init__.py imports and create_dataset function."""
import pytest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd


class TestFeatureStoreImports:
    """Test that all expected symbols are importable from sagemaker.mlops.feature_store."""

    def test_feature_group_importable(self):
        from sagemaker.mlops.feature_store import FeatureGroup
        assert FeatureGroup is not None

    def test_feature_store_importable(self):
        from sagemaker.mlops.feature_store import FeatureStore
        assert FeatureStore is not None

    def test_feature_metadata_importable(self):
        from sagemaker.mlops.feature_store import FeatureMetadata
        assert FeatureMetadata is not None

    def test_create_dataset_importable(self):
        from sagemaker.mlops.feature_store import create_dataset
        assert create_dataset is not None
        assert callable(create_dataset)

    def test_all_shapes_importable(self):
        from sagemaker.mlops.feature_store import (
            OnlineStoreConfig,
            OfflineStoreConfig,
            S3StorageConfig,
            TtlDuration,
            FeatureValue,
            FeatureParameter,
            ThroughputConfig,
            Filter,
            SearchExpression,
            DataCatalogConfig,
            OnlineStoreSecurityConfig,
        )
        assert OnlineStoreConfig is not None
        assert OfflineStoreConfig is not None
        assert S3StorageConfig is not None
        assert TtlDuration is not None
        assert FeatureValue is not None
        assert FeatureParameter is not None
        assert ThroughputConfig is not None
        assert Filter is not None
        assert SearchExpression is not None
        assert DataCatalogConfig is not None
        assert OnlineStoreSecurityConfig is not None

    def test_all_enums_importable(self):
        from sagemaker.mlops.feature_store import (
            TargetStoreEnum,
            DeletionModeEnum,
            OnlineStoreStorageTypeEnum,
            TableFormatEnum,
            ResourceEnum,
            SearchOperatorEnum,
            SortOrderEnum,
            FilterOperatorEnum,
            ExpirationTimeResponseEnum,
            ThroughputModeEnum,
        )
        assert TargetStoreEnum is not None
        assert DeletionModeEnum is not None
        assert OnlineStoreStorageTypeEnum is not None
        assert TableFormatEnum is not None
        assert ResourceEnum is not None
        assert SearchOperatorEnum is not None
        assert SortOrderEnum is not None
        assert FilterOperatorEnum is not None
        assert ExpirationTimeResponseEnum is not None
        assert ThroughputModeEnum is not None

    def test_all_feature_definition_helpers_importable(self):
        from sagemaker.mlops.feature_store import (
            FeatureDefinition,
            FractionalFeatureDefinition,
            IntegralFeatureDefinition,
            StringFeatureDefinition,
            VectorCollectionType,
            ListCollectionType,
            SetCollectionType,
            FeatureTypeEnum,
            CollectionTypeEnum,
        )
        assert FeatureDefinition is not None
        assert FractionalFeatureDefinition is not None
        assert IntegralFeatureDefinition is not None
        assert StringFeatureDefinition is not None
        assert VectorCollectionType is not None
        assert ListCollectionType is not None
        assert SetCollectionType is not None
        assert FeatureTypeEnum is not None
        assert CollectionTypeEnum is not None

    def test_all_utility_functions_importable(self):
        from sagemaker.mlops.feature_store import (
            create_athena_query,
            as_hive_ddl,
            ingest_dataframe,
            load_feature_definitions_from_dataframe,
            create_dataset,
            get_session_from_role,
        )
        assert callable(create_athena_query)
        assert callable(as_hive_ddl)
        assert callable(ingest_dataframe)
        assert callable(load_feature_definitions_from_dataframe)
        assert callable(create_dataset)
        assert callable(get_session_from_role)

    def test_all_classes_importable(self):
        from sagemaker.mlops.feature_store import (
            AthenaQuery,
            DatasetBuilder,
            FeatureGroupToBeMerged,
            IngestionError,
            IngestionManagerPandas,
            JoinComparatorEnum,
            JoinTypeEnum,
            TableType,
        )
        assert AthenaQuery is not None
        assert DatasetBuilder is not None
        assert FeatureGroupToBeMerged is not None
        assert IngestionError is not None
        assert IngestionManagerPandas is not None
        assert JoinComparatorEnum is not None
        assert JoinTypeEnum is not None
        assert TableType is not None

    def test_feature_store_in_all(self):
        import sagemaker.mlops.feature_store as fs_module
        assert "FeatureStore" in fs_module.__all__
        assert "FeatureGroup" in fs_module.__all__
        assert "FeatureMetadata" in fs_module.__all__
        assert "create_dataset" in fs_module.__all__


class TestMlopsInitExposesFeatureStore:
    """Test that feature_store is accessible via sagemaker.mlops."""

    def test_feature_store_submodule_accessible(self):
        import sagemaker.mlops
        assert hasattr(sagemaker.mlops, 'feature_store') or 'feature_store' in sagemaker.mlops.__all__

    def test_feature_store_in_mlops_all(self):
        import sagemaker.mlops
        assert "feature_store" in sagemaker.mlops.__all__

    def test_import_from_mlops_feature_store(self):
        """Verify the full import path works."""
        from sagemaker.mlops.feature_store import FeatureGroup, FeatureStore, create_dataset
        assert FeatureGroup is not None
        assert FeatureStore is not None
        assert create_dataset is not None


class TestCreateDatasetFunction:
    """Test the create_dataset convenience function."""

    @patch("sagemaker.mlops.feature_store.feature_utils.DatasetBuilder")
    def test_create_dataset_with_feature_group_base(self, mock_builder_class):
        """Test create_dataset returns a DatasetBuilder when base is a FeatureGroup."""
        from sagemaker.mlops.feature_store.feature_utils import create_dataset

        mock_builder = MagicMock()
        mock_builder_class.create.return_value = mock_builder

        mock_fg = MagicMock()
        mock_session = MagicMock()

        result = create_dataset(
            base=mock_fg,
            output_path="s3://bucket/output",
            session=mock_session,
        )

        mock_builder_class.create.assert_called_once_with(
            base=mock_fg,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name=None,
            event_time_identifier_feature_name=None,
            included_feature_names=None,
            kms_key_id=None,
        )
        assert result == mock_builder

    @patch("sagemaker.mlops.feature_store.feature_utils.DatasetBuilder")
    def test_create_dataset_with_dataframe_base(self, mock_builder_class):
        """Test create_dataset returns DatasetBuilder when base is a DataFrame."""
        from sagemaker.mlops.feature_store.feature_utils import create_dataset

        mock_builder = MagicMock()
        mock_builder_class.create.return_value = mock_builder

        df = pd.DataFrame({"id": [1, 2], "ts": ["2024-01-01", "2024-01-02"]})
        mock_session = MagicMock()

        result = create_dataset(
            base=df,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name="id",
            event_time_identifier_feature_name="ts",
        )

        mock_builder_class.create.assert_called_once_with(
            base=df,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name="id",
            event_time_identifier_feature_name="ts",
            included_feature_names=None,
            kms_key_id=None,
        )
        assert result == mock_builder

    @patch("sagemaker.mlops.feature_store.feature_utils.DatasetBuilder")
    def test_create_dataset_with_dataframe_missing_required_params_raises(self, mock_builder_class):
        """Test ValueError when base is DataFrame but required params are missing."""
        from sagemaker.mlops.feature_store.feature_utils import create_dataset

        # Make DatasetBuilder.create raise ValueError like the real implementation
        mock_builder_class.create.side_effect = ValueError(
            "record_identifier_feature_name and event_time_identifier_feature_name "
            "are required when base is a DataFrame."
        )

        df = pd.DataFrame({"id": [1, 2], "ts": ["2024-01-01", "2024-01-02"]})
        mock_session = MagicMock()

        with pytest.raises(ValueError, match="record_identifier_feature_name"):
            create_dataset(
                base=df,
                output_path="s3://bucket/output",
                session=mock_session,
            )

    @patch("sagemaker.mlops.feature_store.feature_utils.DatasetBuilder")
    def test_create_dataset_passes_all_params(self, mock_builder_class):
        """Test that all parameters are forwarded to DatasetBuilder.create."""
        from sagemaker.mlops.feature_store.feature_utils import create_dataset

        mock_builder = MagicMock()
        mock_builder_class.create.return_value = mock_builder

        mock_fg = MagicMock()
        mock_session = MagicMock()

        result = create_dataset(
            base=mock_fg,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name="rec_id",
            event_time_identifier_feature_name="event_ts",
            included_feature_names=["feat1", "feat2"],
            kms_key_id="my-kms-key",
        )

        mock_builder_class.create.assert_called_once_with(
            base=mock_fg,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name="rec_id",
            event_time_identifier_feature_name="event_ts",
            included_feature_names=["feat1", "feat2"],
            kms_key_id="my-kms-key",
        )
        assert result == mock_builder
