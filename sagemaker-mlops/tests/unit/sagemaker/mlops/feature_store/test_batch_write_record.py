# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for BatchWriteRecord and ListRecords wiring."""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from sagemaker.mlops.feature_store.ingestion_manager_pandas import (
    IngestionManagerPandas,
    IngestionError,
    BATCH_WRITE_MAX_ENTRIES,
)


class TestBatchWriteRecordIngestion:
    """Tests for use_batch_write_record=True path."""

    @pytest.fixture
    def feature_definitions(self):
        return {
            "RecordIdentifier": {"FeatureType": "String", "CollectionType": None},
            "EventTime": {"FeatureType": "String", "CollectionType": None},
            "Feature1": {"FeatureType": "String", "CollectionType": None},
        }

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "RecordIdentifier": [f"id-{i}" for i in range(5)],
            "EventTime": ["2026-01-01T00:00:00Z"] * 5,
            "Feature1": [f"value-{i}" for i in range(5)],
        })

    @pytest.fixture
    def large_dataframe(self):
        """DataFrame with 60 rows — should produce 3 BatchWriteRecord calls."""
        return pd.DataFrame({
            "RecordIdentifier": [f"id-{i}" for i in range(60)],
            "EventTime": ["2026-01-01T00:00:00Z"] * 60,
            "Feature1": [f"value-{i}" for i in range(60)],
        })

    def test_batch_write_max_entries_constant(self):
        assert BATCH_WRITE_MAX_ENTRIES == 25

    def test_initialization_with_flag(self, feature_definitions):
        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        assert mgr.use_batch_write_record is True

    def test_initialization_default_flag(self, feature_definitions):
        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
        )
        assert mgr.use_batch_write_record is False

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_single_batch(self, mock_fg_class, feature_definitions, sample_dataframe):
        """5 rows → 1 batch_write_record call."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=sample_dataframe)

        mock_fg.batch_write_record.assert_called_once()
        assert mgr.failed_rows == []

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_multiple_batches(self, mock_fg_class, feature_definitions, large_dataframe):
        """60 rows → 3 batch_write_record calls (25+25+10)."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=large_dataframe)

        assert mock_fg.batch_write_record.call_count == 3
        assert mgr.failed_rows == []

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_exactly_25(self, mock_fg_class, feature_definitions):
        """Exactly 25 rows → 1 call (boundary)."""
        df = pd.DataFrame({
            "RecordIdentifier": [f"id-{i}" for i in range(25)],
            "EventTime": ["2026-01-01T00:00:00Z"] * 25,
            "Feature1": [f"v-{i}" for i in range(25)],
        })
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=df)
        assert mock_fg.batch_write_record.call_count == 1

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_exactly_26(self, mock_fg_class, feature_definitions):
        """Exactly 26 rows → 2 calls (25+1 boundary)."""
        df = pd.DataFrame({
            "RecordIdentifier": [f"id-{i}" for i in range(26)],
            "EventTime": ["2026-01-01T00:00:00Z"] * 26,
            "Feature1": [f"v-{i}" for i in range(26)],
        })
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=df)
        assert mock_fg.batch_write_record.call_count == 2

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_empty_dataframe(self, mock_fg_class, feature_definitions):
        """Empty DataFrame → no batch_write_record calls."""
        df = pd.DataFrame({
            "RecordIdentifier": [],
            "EventTime": [],
            "Feature1": [],
        })
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=df)
        mock_fg.batch_write_record.assert_not_called()
        assert mgr.failed_rows == []

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_partial_failure_maps_to_row(self, mock_fg_class, feature_definitions):
        """Error with matching entry maps to specific row index."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["good-1", None, "good-3"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 3,
            "Feature1": ["v1", "v2", "v3"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Entry for row 1 (None RecordIdentifier → only EventTime + Feature1)
        error_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v2"),
            ],
        )
        mock_error = Mock()
        mock_error.entry = error_entry
        mock_error.error_code = "ValidationError"
        mock_error.error_message = "Missing RecordIdentifier"

        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = [mock_error]
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 1 in exc_info.value.failed_rows
        assert 0 not in exc_info.value.failed_rows
        assert 2 not in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_multiple_errors_in_batch(self, mock_fg_class, feature_definitions):
        """Multiple errors map to correct row indices."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["good-0", None, "good-2", None, "good-4"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 5,
            "Feature1": ["v0", "v1", "v2", "v3", "v4"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Entries for rows 1 and 3 (missing RecordIdentifier)
        error_entry_1 = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v1"),
            ],
        )
        error_entry_3 = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v3"),
            ],
        )

        mock_error_1 = Mock()
        mock_error_1.entry = error_entry_1
        mock_error_1.error_code = "ValidationError"
        mock_error_1.error_message = "Missing RecordIdentifier"

        mock_error_3 = Mock()
        mock_error_3.entry = error_entry_3
        mock_error_3.error_code = "ValidationError"
        mock_error_3.error_message = "Missing RecordIdentifier"

        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = [mock_error_1, mock_error_3]
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 1 in exc_info.value.failed_rows
        assert 3 in exc_info.value.failed_rows
        assert 0 not in exc_info.value.failed_rows
        assert 2 not in exc_info.value.failed_rows
        assert 4 not in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_full_failure(self, mock_fg_class, feature_definitions, sample_dataframe):
        """Exception from batch_write_record marks all rows failed."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_fg.batch_write_record.side_effect = Exception("AccessForbidden")

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=sample_dataframe)

        assert len(exc_info.value.failed_rows) == 5

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_unprocessed_entries(self, mock_fg_class, feature_definitions):
        """Unprocessed entries map back to specific row indices."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", "id-1", "id-2"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 3,
            "Feature1": ["v0", "v1", "v2"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Entry for row 2 is returned as unprocessed
        unprocessed_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="RecordIdentifier", value_as_string="id-2"),
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v2"),
            ],
        )

        mock_response = Mock()
        mock_response.unprocessed_entries = [unprocessed_entry]
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 2 in exc_info.value.failed_rows
        assert 0 not in exc_info.value.failed_rows
        assert 1 not in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_unprocessed_entry_no_match_marks_all(self, mock_fg_class, feature_definitions):
        """If unprocessed entry can't be matched, all rows in batch marked failed."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", "id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 2,
            "Feature1": ["v0", "v1"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Return an entry that doesn't match anything we sent
        unmatched_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="RecordIdentifier", value_as_string="UNKNOWN"),
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
            ],
        )

        mock_response = Mock()
        mock_response.unprocessed_entries = [unmatched_entry]
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        # All rows marked failed since we can't identify which one
        assert 0 in exc_info.value.failed_rows
        assert 1 in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_multiple_unprocessed_entries(self, mock_fg_class, feature_definitions):
        """Multiple unprocessed entries each map to correct row."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", "id-1", "id-2", "id-3", "id-4"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 5,
            "Feature1": ["v0", "v1", "v2", "v3", "v4"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Rows 1 and 3 returned as unprocessed
        unprocessed_1 = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="RecordIdentifier", value_as_string="id-1"),
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v1"),
            ],
        )
        unprocessed_3 = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="RecordIdentifier", value_as_string="id-3"),
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v3"),
            ],
        )

        mock_response = Mock()
        mock_response.unprocessed_entries = [unprocessed_1, unprocessed_3]
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 1 in exc_info.value.failed_rows
        assert 3 in exc_info.value.failed_rows
        assert 0 not in exc_info.value.failed_rows
        assert 2 not in exc_info.value.failed_rows
        assert 4 not in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_both_errors_and_unprocessed(self, mock_fg_class, feature_definitions):
        """Both errors and unprocessed_entries in same response."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", None, "id-2", "id-3", "id-4"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 5,
            "Feature1": ["v0", "v1", "v2", "v3", "v4"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        # Row 1 has error (missing RecordIdentifier)
        error_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v1"),
            ],
        )
        mock_error = Mock()
        mock_error.entry = error_entry
        mock_error.error_code = "ValidationError"
        mock_error.error_message = "Missing RecordIdentifier"

        # Row 4 is unprocessed (throttled)
        unprocessed_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[
                FeatureValue(feature_name="RecordIdentifier", value_as_string="id-4"),
                FeatureValue(feature_name="EventTime", value_as_string="2026-01-01T00:00:00Z"),
                FeatureValue(feature_name="Feature1", value_as_string="v4"),
            ],
        )

        mock_response = Mock()
        mock_response.unprocessed_entries = [unprocessed_entry]
        mock_response.errors = [mock_error]
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        # Row 1 (error) and row 4 (unprocessed) should both be in failed_rows
        assert 1 in exc_info.value.failed_rows
        assert 4 in exc_info.value.failed_rows
        # Rows 0, 2, 3 should NOT be failed
        assert 0 not in exc_info.value.failed_rows
        assert 2 not in exc_info.value.failed_rows
        assert 3 not in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_error_without_entry_marks_all(self, mock_fg_class, feature_definitions):
        """Error object without .entry attribute marks all rows failed."""
        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", "id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 2,
            "Feature1": ["v0", "v1"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        mock_error = Mock()
        mock_error.entry = None  # No entry object

        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = [mock_error]
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 0 in exc_info.value.failed_rows
        assert 1 in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_error_entry_no_match_marks_all(self, mock_fg_class, feature_definitions):
        """Error with entry that doesn't match → marks all rows in batch failed."""
        from sagemaker.core.shapes import BatchWriteRecordEntry, FeatureValue

        df = pd.DataFrame({
            "RecordIdentifier": ["id-0", "id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"] * 2,
            "Feature1": ["v0", "v1"],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg

        unmatched_entry = BatchWriteRecordEntry(
            feature_group_name="test-fg",
            record=[FeatureValue(feature_name="UNKNOWN", value_as_string="x")],
        )
        mock_error = Mock()
        mock_error.entry = unmatched_entry
        mock_error.error_code = "InternalError"
        mock_error.error_message = "Something went wrong"

        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = [mock_error]
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )

        with pytest.raises(IngestionError) as exc_info:
            mgr.run(data_frame=df)

        assert 0 in exc_info.value.failed_rows
        assert 1 in exc_info.value.failed_rows

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_skips_null_values(self, mock_fg_class, feature_definitions):
        """Null/NaN values not included in record."""
        df = pd.DataFrame({
            "RecordIdentifier": ["id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"],
            "Feature1": [None],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=df)

        call_kwargs = mock_fg.batch_write_record.call_args
        entries = call_kwargs.kwargs.get("entries") or call_kwargs[1].get("entries")
        record = entries[0].record
        feature_names = [fv.feature_name for fv in record]
        assert "Feature1" not in feature_names
        assert "RecordIdentifier" in feature_names

    def test_use_batch_write_record_false_uses_put_record(self, feature_definitions):
        """False flag → put_record called, not batch_write_record."""
        df = pd.DataFrame({"RecordIdentifier": ["id-1"], "EventTime": ["2026-01-01T00:00:00Z"], "Feature1": ["v"]})

        with patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup") as mock_fg_class:
            mock_fg = Mock()
            mock_fg_class.return_value = mock_fg

            mgr = IngestionManagerPandas(
                feature_group_name="test-fg",
                feature_definitions=feature_definitions,
                use_batch_write_record=False,
            )
            mgr.run(data_frame=df)

            mock_fg.put_record.assert_called_once()
            mock_fg.batch_write_record.assert_not_called()

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_does_not_pass_none_target_stores(self, mock_fg_class, feature_definitions, sample_dataframe):
        """target_stores=None → entries have Unassigned (not None)."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=sample_dataframe, target_stores=None)

        call_kwargs = mock_fg.batch_write_record.call_args
        entries = call_kwargs.kwargs.get("entries") or call_kwargs[1].get("entries")
        from sagemaker.core.utils.utils import Unassigned
        assert isinstance(entries[0].target_stores, Unassigned)

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_passes_target_stores_when_set(self, mock_fg_class, feature_definitions, sample_dataframe):
        """target_stores=['OnlineStore'] → entries have target_stores set."""
        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_response = Mock()
        mock_response.unprocessed_entries = []
        mock_response.errors = []
        mock_fg.batch_write_record.return_value = mock_response

        mgr = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            use_batch_write_record=True,
        )
        mgr.run(data_frame=sample_dataframe, target_stores=["OnlineStore"])

        call_kwargs = mock_fg.batch_write_record.call_args
        entries = call_kwargs.kwargs.get("entries") or call_kwargs[1].get("entries")
        assert entries[0].target_stores == ["OnlineStore"]

    @patch("sagemaker.mlops.feature_store.ingestion_manager_pandas.CoreFeatureGroup")
    def test_batch_write_sliced_dataframe_preserves_indices(self, mock_fg_class, feature_definitions):
        """Sliced DataFrame (non-zero index) reports correct original indices on failure."""
        df = pd.DataFrame({
            "RecordIdentifier": [f"id-{i}" for i in range(10)],
            "EventTime": ["2026-01-01T00:00:00Z"] * 10,
            "Feature1": [f"v-{i}" for i in range(10)],
        })

        mock_fg = Mock()
        mock_fg_class.return_value = mock_fg
        mock_fg.batch_write_record.side_effect = Exception("Failure")

        # Simulate multi-process split: process gets rows 5-9
        failed = IngestionManagerPandas._ingest_batch_write(
            data_frame=df,
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            start_index=5,
            end_index=10,
            target_stores=None,
        )

        # Should report original DataFrame indices 5,6,7,8,9
        assert failed == [5, 6, 7, 8, 9]


class TestBuildRecord:
    """Tests for _build_record helper — verifying indexing correctness."""

    def test_build_record_single_column(self):
        """1 column boundary case."""
        df = pd.DataFrame({"Col1": ["val1"]})
        defs = {"Col1": {"FeatureType": "String", "CollectionType": None}}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert len(record) == 1
        assert record[0].feature_name == "Col1"
        assert record[0].value_as_string == "val1"

    def test_build_record_five_columns(self):
        """5 columns — verifies first and last column mapped correctly."""
        df = pd.DataFrame({"A": ["a"], "B": ["b"], "C": ["c"], "D": ["d"], "E": ["e"]})
        defs = {c: {"FeatureType": "String", "CollectionType": None} for c in df.columns}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert len(record) == 5
        assert record[0].feature_name == "A"
        assert record[0].value_as_string == "a"
        assert record[4].feature_name == "E"
        assert record[4].value_as_string == "e"

    def test_build_record_multiple_rows_correct_values(self):
        """Each row maps to correct column values (no cross-row contamination)."""
        df = pd.DataFrame({
            "Id": ["id-0", "id-1", "id-2"],
            "Val": ["v0", "v1", "v2"],
        })
        defs = {c: {"FeatureType": "String", "CollectionType": None} for c in df.columns}
        for row in df.itertuples():
            record = IngestionManagerPandas._build_record(df, row, defs)
            result = {fv.feature_name: fv.value_as_string for fv in record}
            idx = row[0]
            assert result["Id"] == f"id-{idx}"
            assert result["Val"] == f"v{idx}"

    def test_build_record_sliced_dataframe(self):
        """Sliced DataFrame (non-zero index) still maps correctly."""
        df = pd.DataFrame({
            "Id": ["id-0", "id-1", "id-2", "id-3", "id-4"],
            "Val": ["v0", "v1", "v2", "v3", "v4"],
        })
        sliced = df[2:4]  # rows at index 2, 3
        defs = {c: {"FeatureType": "String", "CollectionType": None} for c in df.columns}
        for row in sliced.itertuples():
            record = IngestionManagerPandas._build_record(sliced, row, defs)
            result = {fv.feature_name: fv.value_as_string for fv in record}
            idx = row[0]
            assert result["Id"] == f"id-{idx}"
            assert result["Val"] == f"v{idx}"

    def test_build_record_skips_none(self):
        """None values excluded from record."""
        df = pd.DataFrame({
            "RecordIdentifier": ["id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"],
            "Feature1": [None],
        })
        defs = {c: {"FeatureType": "String", "CollectionType": None} for c in df.columns}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert len(record) == 2
        names = {fv.feature_name for fv in record}
        assert "Feature1" not in names

    def test_build_record_skips_nan(self):
        """NaN values excluded from record."""
        df = pd.DataFrame({
            "RecordIdentifier": ["id-1"],
            "EventTime": ["2026-01-01T00:00:00Z"],
            "Feature1": [np.nan],
        })
        defs = {c: {"FeatureType": "String", "CollectionType": None} for c in df.columns}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert len(record) == 2

    def test_build_record_collection_type(self):
        """Collection type feature uses value_as_string_list."""
        df = pd.DataFrame({
            "id": ["id-1"],
            "tags": [["a", "b", "c"]],
        })
        defs = {
            "id": {"FeatureType": "String", "CollectionType": None},
            "tags": {"FeatureType": "String", "CollectionType": "List"},
        }
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert len(record) == 2
        tags_fv = [fv for fv in record if fv.feature_name == "tags"][0]
        assert tags_fv.value_as_string_list == ["a", "b", "c"]

    def test_build_record_integer_value_to_string(self):
        """Integer values converted to string."""
        df = pd.DataFrame({"Num": [42]})
        defs = {"Num": {"FeatureType": "Integral", "CollectionType": None}}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert record[0].value_as_string == "42"

    def test_build_record_float_value_to_string(self):
        """Float values converted to string."""
        df = pd.DataFrame({"Frac": [3.14]})
        defs = {"Frac": {"FeatureType": "Fractional", "CollectionType": None}}
        row = next(df.itertuples())
        record = IngestionManagerPandas._build_record(df, row, defs)
        assert record[0].value_as_string == "3.14"

    def test_build_record_row_index_is_row0(self):
        """row[0] is the DataFrame index, used for error mapping."""
        df = pd.DataFrame({"A": ["x", "y", "z"]})
        rows = list(df.itertuples())
        assert rows[0][0] == 0
        assert rows[1][0] == 1
        assert rows[2][0] == 2


