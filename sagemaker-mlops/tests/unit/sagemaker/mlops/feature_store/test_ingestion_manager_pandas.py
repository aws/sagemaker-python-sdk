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


class TestAsyncIngestionValidation:
    """Test async ingestion validation with max_processes=1.
    
    Bug fix: Error message unclear when trying to use async ingestion with 1 process.
    """

    def test_async_with_single_process_single_worker_raises_clear_error(self):
        """Test that wait=False with max_processes=1 and max_workers=1 raises clear error."""
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions={"id": {"FeatureType": "String", "CollectionType": None}},
            max_workers=1,
            max_processes=1,
        )
        
        df = pd.DataFrame({"id": ["1", "2", "3"]})
        
        with pytest.raises(ValueError) as exc_info:
            manager.run(data_frame=df, wait=False)
        
        error_message = str(exc_info.value)
        assert "Async ingestion (wait=False)" in error_message
        assert "max_processes > 1 or max_workers > 1" in error_message
        assert "wait=True" in error_message

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_sync_with_single_process_single_worker_works(self, mock_fg_class):
        """Test that wait=True with max_processes=1 and max_workers=1 works."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions={"id": {"FeatureType": "String", "CollectionType": None}},
            max_workers=1,
            max_processes=1,
        )
        
        df = pd.DataFrame({"id": ["1", "2", "3"]})
        
        # Should not raise validation error
        manager.run(data_frame=df, wait=True)

    @pytest.mark.parametrize("max_workers,max_processes", [
        (2, 1),  # Multiple workers, single process
        (1, 2),  # Single worker, multiple processes
        (2, 2),  # Multiple workers and processes
    ])
    @patch.object(IngestionManagerPandas, '_run_multi_process')
    def test_async_with_parallelism_no_validation_error(self, mock_run, max_workers, max_processes):
        """Test that wait=False works with any parallelism configuration where max_workers > 1 OR max_processes > 1."""
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions={"id": {"FeatureType": "String", "CollectionType": None}},
            max_workers=max_workers,
            max_processes=max_processes,
        )
        
        df = pd.DataFrame({"id": ["1", "2", "3"]})
        
        # Should not raise validation error
        manager.run(data_frame=df, wait=False)
        
        # Verify it called the multi-process method (positive assertion)
        mock_run.assert_called_once()


class TestIngestionManagerRegion:
    """``region`` must reach every FeatureStore runtime call."""

    @pytest.fixture
    def feature_definitions(self):
        return {"id": {"FeatureType": "String", "CollectionType": None}}

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({"id": ["1", "2", "3"]})

    def test_region_defaults_to_none(self, feature_definitions):
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
        )
        assert manager.region is None

    def test_region_stored_on_manager(self, feature_definitions):
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            region="eu-west-1",
        )
        assert manager.region == "eu-west-1"

    def test_ingest_row_passes_region_to_put_record(self, feature_definitions):
        df = pd.DataFrame({"id": ["1"]})
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
                region="eu-west-1",
            )

        assert mock_fg.put_record.call_args[1]["region"] == "eu-west-1"

    def test_ingest_row_region_defaults_to_none(self, feature_definitions):
        df = pd.DataFrame({"id": ["1"]})
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

        assert mock_fg.put_record.call_args[1]["region"] is None

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_single_thread_run_passes_region_to_put_record(
        self, mock_fg_class, feature_definitions, sample_dataframe
    ):
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg

        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            region="eu-west-1",
        )
        manager.run(data_frame=sample_dataframe, wait=True)

        assert mock_fg.put_record.call_count == 3
        for call in mock_fg.put_record.call_args_list:
            assert call[1]["region"] == "eu-west-1"

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_single_batch_passes_region_to_put_record(
        self, mock_fg_class, feature_definitions, sample_dataframe
    ):
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg

        IngestionManagerPandas._ingest_single_batch(
            data_frame=sample_dataframe,
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            start_index=0,
            end_index=3,
            region="eu-west-1",
        )

        assert mock_fg.put_record.call_count == 3
        for call in mock_fg.put_record.call_args_list:
            assert call[1]["region"] == "eu-west-1"

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_passes_region(
        self, mock_fg_class, feature_definitions, sample_dataframe
    ):
        mock_fg = MagicMock()
        mock_fg.batch_write_record.return_value = MagicMock(
            unprocessed_entries=[], errors=[]
        )
        mock_fg_class.return_value = mock_fg

        IngestionManagerPandas._ingest_batch_write(
            data_frame=sample_dataframe,
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            start_index=0,
            end_index=3,
            region="eu-west-1",
        )

        mock_fg.batch_write_record.assert_called_once()
        assert mock_fg.batch_write_record.call_args[1]["region"] == "eu-west-1"

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_run_passes_region(
        self, mock_fg_class, feature_definitions, sample_dataframe
    ):
        mock_fg = MagicMock()
        mock_fg.batch_write_record.return_value = MagicMock(
            unprocessed_entries=[], errors=[]
        )
        mock_fg_class.return_value = mock_fg

        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
            region="eu-west-1",
        )
        manager.run(data_frame=sample_dataframe, wait=True)

        assert mock_fg.batch_write_record.call_args[1]["region"] == "eu-west-1"

    @patch.object(IngestionManagerPandas, "_ingest_single_batch", return_value=[])
    def test_multi_threaded_forwards_region(
        self, mock_ingest, feature_definitions, sample_dataframe
    ):
        IngestionManagerPandas._run_multi_threaded(
            max_workers=2,
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            data_frame=sample_dataframe,
            region="eu-west-1",
        )

        assert mock_ingest.call_count == 2
        for call in mock_ingest.call_args_list:
            assert call[1]["region"] == "eu-west-1"

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.Pool")
    def test_multi_process_args_include_region(
        self, mock_pool_class, feature_definitions, sample_dataframe
    ):
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            max_processes=2,
            region="eu-west-1",
        )
        manager.run(data_frame=sample_dataframe, wait=False)

        starmap_args = mock_pool.starmap_async.call_args[0][1]
        assert len(starmap_args) == 2
        for process_args in starmap_args:
            # region is the last positional argument passed to _run_multi_threaded
            assert process_args[-1] == "eu-west-1"
