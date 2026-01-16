# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for ingestion_manager_pandas.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from sagemaker.mlops.feature_store.ingestion_manager_pandas import (
    IngestionManagerPandas,
    IngestionError,
)


class TestIngestionError:
    def test_stores_failed_rows(self):
        error = IngestionError([1, 5, 10], "Some rows failed")
        assert error.failed_rows == [1, 5, 10]
        assert "Some rows failed" in str(error)


class TestIngestionManagerPandas:
    @pytest.fixture
    def feature_definitions(self):
        return {
            "id": {"FeatureName": "id", "FeatureType": "Integral"},
            "value": {"FeatureName": "value", "FeatureType": "Fractional"},
            "name": {"FeatureName": "name", "FeatureType": "String"},
        }

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.1, 2.2, 3.3],
            "name": ["a", "b", "c"],
        })

    @pytest.fixture
    def manager(self, feature_definitions):
        return IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            max_workers=1,
            max_processes=1,
        )

    def test_initialization(self, manager):
        assert manager.feature_group_name == "test-fg"
        assert manager.max_workers == 1
        assert manager.max_processes == 1
        assert manager.failed_rows == []

    def test_failed_rows_property(self, manager):
        manager._failed_indices = [1, 2, 3]
        assert manager.failed_rows == [1, 2, 3]


class TestIngestionManagerHelpers:
    def test_is_feature_collection_type_true(self):
        feature_defs = {
            "tags": {"FeatureName": "tags", "FeatureType": "String", "CollectionType": "List"},
        }
        assert IngestionManagerPandas._is_feature_collection_type("tags", feature_defs) is True

    def test_is_feature_collection_type_false(self):
        feature_defs = {
            "id": {"FeatureName": "id", "FeatureType": "Integral"},
        }
        assert IngestionManagerPandas._is_feature_collection_type("id", feature_defs) is False

    def test_is_feature_collection_type_missing(self):
        feature_defs = {}
        assert IngestionManagerPandas._is_feature_collection_type("unknown", feature_defs) is False

    def test_feature_value_is_not_none_scalar(self):
        assert IngestionManagerPandas._feature_value_is_not_none(5) is True
        assert IngestionManagerPandas._feature_value_is_not_none(None) is False
        assert IngestionManagerPandas._feature_value_is_not_none(np.nan) is False

    def test_feature_value_is_not_none_list(self):
        assert IngestionManagerPandas._feature_value_is_not_none([1, 2, 3]) is True
        assert IngestionManagerPandas._feature_value_is_not_none([]) is True
        assert IngestionManagerPandas._feature_value_is_not_none(None) is False

    def test_convert_to_string_list(self):
        result = IngestionManagerPandas._convert_to_string_list([1, 2, 3])
        assert result == ["1", "2", "3"]

    def test_convert_to_string_list_with_none(self):
        result = IngestionManagerPandas._convert_to_string_list([1, None, 3])
        assert result == ["1", None, "3"]

    def test_convert_to_string_list_raises_for_non_list(self):
        with pytest.raises(ValueError, match="must be an Array"):
            IngestionManagerPandas._convert_to_string_list("not a list")


class TestIngestionManagerRun:
    @pytest.fixture
    def feature_definitions(self):
        return {
            "id": {"FeatureName": "id", "FeatureType": "Integral"},
            "value": {"FeatureName": "value", "FeatureType": "Fractional"},
        }

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.1, 2.2, 3.3],
        })

    @patch.object(IngestionManagerPandas, "_run_single_process_single_thread")
    def test_run_single_thread_mode(self, mock_single, feature_definitions, sample_dataframe):
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            max_workers=1,
            max_processes=1,
        )

        manager.run(sample_dataframe)

        mock_single.assert_called_once()

    @patch.object(IngestionManagerPandas, "_run_multi_process")
    def test_run_multi_process_mode(self, mock_multi, feature_definitions, sample_dataframe):
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            max_workers=2,
            max_processes=2,
        )

        manager.run(sample_dataframe)

        mock_multi.assert_called_once()


class TestIngestionManagerIngestRow:
    @pytest.fixture
    def feature_definitions(self):
        return {
            "id": {"FeatureName": "id", "FeatureType": "Integral"},
            "name": {"FeatureName": "name", "FeatureType": "String"},
        }

    @pytest.fixture
    def collection_feature_definitions(self):
        return {
            "id": {"FeatureName": "id", "FeatureType": "Integral"},
            "tags": {"FeatureName": "tags", "FeatureType": "String", "CollectionType": "List"},
        }

    def test_ingest_row_success(self, feature_definitions):
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_fg = MagicMock()
        failed_rows = []

        for row in df.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=df,
                row=row,
                feature_group=mock_fg,
                feature_definitions=feature_definitions,
                failed_rows=failed_rows,
                target_stores=None,
            )

        mock_fg.put_record.assert_called_once()
        assert len(failed_rows) == 0

    def test_ingest_row_with_collection_type(self, collection_feature_definitions):
        df = pd.DataFrame({
            "id": [1],
            "tags": [["tag1", "tag2"]],
        })
        mock_fg = MagicMock()
        failed_rows = []

        for row in df.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=df,
                row=row,
                feature_group=mock_fg,
                feature_definitions=collection_feature_definitions,
                failed_rows=failed_rows,
                target_stores=None,
            )

        mock_fg.put_record.assert_called_once()
        call_args = mock_fg.put_record.call_args
        record = call_args[1]["record"]
        
        # Find the tags feature value
        tags_value = next(v for v in record if v.feature_name == "tags")
        assert tags_value.value_as_string_list == ["tag1", "tag2"]

    def test_ingest_row_failure_appends_to_failed(self, feature_definitions):
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_fg = MagicMock()
        mock_fg.put_record.side_effect = Exception("API Error")
        failed_rows = []

        for row in df.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=df,
                row=row,
                feature_group=mock_fg,
                feature_definitions=feature_definitions,
                failed_rows=failed_rows,
                target_stores=None,
            )

        assert len(failed_rows) == 1
        assert failed_rows[0] == 0  # Index of failed row

    def test_ingest_row_with_target_stores(self, feature_definitions):
        df = pd.DataFrame({"id": [1], "name": ["test"]})
        mock_fg = MagicMock()
        failed_rows = []

        for row in df.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=df,
                row=row,
                feature_group=mock_fg,
                feature_definitions=feature_definitions,
                failed_rows=failed_rows,
                target_stores=["OnlineStore"],
            )

        call_args = mock_fg.put_record.call_args
        assert call_args[1]["target_stores"] == ["OnlineStore"]

    def test_ingest_row_skips_none_values(self, feature_definitions):
        df = pd.DataFrame({"id": [1], "name": [None]})
        mock_fg = MagicMock()
        failed_rows = []

        for row in df.itertuples():
            IngestionManagerPandas._ingest_row(
                data_frame=df,
                row=row,
                feature_group=mock_fg,
                feature_definitions=feature_definitions,
                failed_rows=failed_rows,
                target_stores=None,
            )

        call_args = mock_fg.put_record.call_args
        record = call_args[1]["record"]
        # Only id should be in record, name is None
        assert len(record) == 1
        assert record[0].feature_name == "id"
