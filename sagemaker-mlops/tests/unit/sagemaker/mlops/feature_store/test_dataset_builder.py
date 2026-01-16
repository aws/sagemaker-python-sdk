# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for dataset_builder.py"""
import datetime
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from sagemaker.mlops.feature_store import FeatureGroup
from sagemaker.mlops.feature_store.dataset_builder import (
    DatasetBuilder,
    FeatureGroupToBeMerged,
    TableType,
    JoinTypeEnum,
    JoinComparatorEnum,
    construct_feature_group_to_be_merged,
)
from sagemaker.mlops.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)


class TestTableType:
    def test_feature_group_value(self):
        assert TableType.FEATURE_GROUP.value == "FeatureGroup"

    def test_data_frame_value(self):
        assert TableType.DATA_FRAME.value == "DataFrame"


class TestJoinTypeEnum:
    def test_inner_join(self):
        assert JoinTypeEnum.INNER_JOIN.value == "JOIN"

    def test_left_join(self):
        assert JoinTypeEnum.LEFT_JOIN.value == "LEFT JOIN"

    def test_right_join(self):
        assert JoinTypeEnum.RIGHT_JOIN.value == "RIGHT JOIN"

    def test_full_join(self):
        assert JoinTypeEnum.FULL_JOIN.value == "FULL JOIN"

    def test_cross_join(self):
        assert JoinTypeEnum.CROSS_JOIN.value == "CROSS JOIN"


class TestJoinComparatorEnum:
    def test_equals(self):
        assert JoinComparatorEnum.EQUALS.value == "="

    def test_greater_than(self):
        assert JoinComparatorEnum.GREATER_THAN.value == ">"

    def test_less_than(self):
        assert JoinComparatorEnum.LESS_THAN.value == "<"


class TestFeatureGroupToBeMerged:
    def test_initialization(self):
        fg = FeatureGroupToBeMerged(
            features=["id", "value"],
            included_feature_names=["id", "value"],
            projected_feature_names=["id", "value"],
            catalog="AwsDataCatalog",
            database="sagemaker_featurestore",
            table_name="my_table",
            record_identifier_feature_name="id",
            event_time_identifier_feature=FeatureDefinition(
                feature_name="event_time",
                feature_type="String",
            ),
        )

        assert fg.features == ["id", "value"]
        assert fg.catalog == "AwsDataCatalog"
        assert fg.table_name == "my_table"
        assert fg.join_type == JoinTypeEnum.INNER_JOIN
        assert fg.join_comparator == JoinComparatorEnum.EQUALS

    def test_custom_join_settings(self):
        fg = FeatureGroupToBeMerged(
            features=["id"],
            included_feature_names=["id"],
            projected_feature_names=["id"],
            catalog="AwsDataCatalog",
            database="db",
            table_name="table",
            record_identifier_feature_name="id",
            event_time_identifier_feature=FeatureDefinition(
                feature_name="ts",
                feature_type="String",
            ),
            join_type=JoinTypeEnum.LEFT_JOIN,
            join_comparator=JoinComparatorEnum.GREATER_THAN,
        )

        assert fg.join_type == JoinTypeEnum.LEFT_JOIN
        assert fg.join_comparator == JoinComparatorEnum.GREATER_THAN


class TestConstructFeatureGroupToBeMerged:
    @patch("sagemaker.mlops.feature_store.dataset_builder.FeatureGroup")
    def test_constructs_from_feature_group(self, mock_fg_class):
        mock_fg = MagicMock()
        mock_fg.feature_group_name = "test-fg"
        mock_fg.record_identifier_feature_name = "id"
        mock_fg.event_time_feature_name = "event_time"
        mock_fg.feature_definitions = [
            MagicMock(feature_name="id", feature_type="Integral"),
            MagicMock(feature_name="value", feature_type="Fractional"),
            MagicMock(feature_name="event_time", feature_type="String"),
        ]
        mock_fg.offline_store_config.data_catalog_config.catalog = "MyCatalog"
        mock_fg.offline_store_config.data_catalog_config.database = "MyDatabase"
        mock_fg.offline_store_config.data_catalog_config.table_name = "MyTable"
        mock_fg.offline_store_config.data_catalog_config.disable_glue_table_creation = False
        mock_fg_class.get.return_value = mock_fg

        target_fg = MagicMock()
        target_fg.feature_group_name = "test-fg"

        result = construct_feature_group_to_be_merged(
            target_feature_group=target_fg,
            included_feature_names=["id", "value"],
        )

        assert result.table_name == "MyTable"
        assert result.database == "MyDatabase"
        assert result.record_identifier_feature_name == "id"
        assert result.table_type == TableType.FEATURE_GROUP

    @patch("sagemaker.mlops.feature_store.dataset_builder.FeatureGroup")
    def test_raises_when_no_metastore(self, mock_fg_class):
        mock_fg = MagicMock()
        mock_fg.feature_group_name = "test-fg"
        mock_fg.offline_store_config = None
        mock_fg_class.get.return_value = mock_fg

        target_fg = MagicMock()
        target_fg.feature_group_name = "test-fg"

        with pytest.raises(RuntimeError, match="No metastore"):
            construct_feature_group_to_be_merged(target_fg, None)


class TestDatasetBuilder:
    @pytest.fixture
    def mock_session(self):
        return Mock()

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.1, 2.2, 3.3],
            "event_time": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })

    def test_initialization_with_dataframe(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        assert builder._output_path == "s3://bucket/output"
        assert builder._record_identifier_feature_name == "id"

    def test_fluent_api_point_in_time(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        result = builder.point_in_time_accurate_join()

        assert result is builder
        assert builder._point_in_time_accurate_join is True

    def test_fluent_api_include_duplicated(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        result = builder.include_duplicated_records()

        assert result is builder
        assert builder._include_duplicated_records is True

    def test_fluent_api_include_deleted(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        result = builder.include_deleted_records()

        assert result is builder
        assert builder._include_deleted_records is True

    def test_fluent_api_number_of_recent_records(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        result = builder.with_number_of_recent_records_by_record_identifier(5)

        assert result is builder
        assert builder._number_of_recent_records == 5

    def test_fluent_api_number_of_records(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        result = builder.with_number_of_records_from_query_results(100)

        assert result is builder
        assert builder._number_of_records == 100

    def test_fluent_api_as_of(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        timestamp = datetime.datetime(2024, 1, 15, 12, 0, 0)
        result = builder.as_of(timestamp)

        assert result is builder
        assert builder._write_time_ending_timestamp == timestamp

    def test_fluent_api_event_time_range(self, mock_session, sample_dataframe):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        start = datetime.datetime(2024, 1, 1)
        end = datetime.datetime(2024, 1, 31)
        result = builder.with_event_time_range(start, end)

        assert result is builder
        assert builder._event_time_starting_timestamp == start
        assert builder._event_time_ending_timestamp == end

    @patch.object(DatasetBuilder, "_run_query")
    @patch("sagemaker.mlops.feature_store.dataset_builder.construct_feature_group_to_be_merged")
    def test_with_feature_group(self, mock_construct, mock_run, mock_session, sample_dataframe):
        mock_fg_to_merge = MagicMock()
        mock_construct.return_value = mock_fg_to_merge

        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base=sample_dataframe,
            _output_path="s3://bucket/output",
            _record_identifier_feature_name="id",
            _event_time_identifier_feature_name="event_time",
        )

        mock_fg = MagicMock()
        result = builder.with_feature_group(mock_fg, target_feature_name_in_base="id")

        assert result is builder
        assert len(builder._feature_groups_to_be_merged) == 1


class TestDatasetBuilderCreate:
    @pytest.fixture
    def mock_session(self):
        return Mock()

    def test_create_with_feature_group(self, mock_session):
        mock_fg = MagicMock(spec=FeatureGroup)
        builder = DatasetBuilder.create(
            base=mock_fg,
            output_path="s3://bucket/output",
            session=mock_session,
        )
        assert builder._base == mock_fg
        assert builder._output_path == "s3://bucket/output"

    def test_create_with_dataframe(self, mock_session):
        df = pd.DataFrame({"id": [1], "value": [10]})
        builder = DatasetBuilder.create(
            base=df,
            output_path="s3://bucket/output",
            session=mock_session,
            record_identifier_feature_name="id",
            event_time_identifier_feature_name="event_time",
        )
        assert builder._record_identifier_feature_name == "id"

    def test_create_with_dataframe_requires_identifiers(self, mock_session):
        df = pd.DataFrame({"id": [1], "value": [10]})
        with pytest.raises(ValueError, match="record_identifier_feature_name"):
            DatasetBuilder.create(
                base=df,
                output_path="s3://bucket/output",
                session=mock_session,
            )


class TestDatasetBuilderValidation:
    @pytest.fixture
    def mock_session(self):
        return Mock()

    def test_to_csv_raises_for_invalid_base(self, mock_session):
        builder = DatasetBuilder(
            _sagemaker_session=mock_session,
            _base="invalid",  # Not DataFrame or FeatureGroup
            _output_path="s3://bucket/output",
        )

        with pytest.raises(ValueError, match="must be either"):
            builder.to_csv_file()
