# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import datetime

import pandas as pd
import pytest
import os
from mock import Mock, patch

from sagemaker.feature_store.dataset_builder import (
    DatasetBuilder,
    FeatureGroupToBeMerged,
    TableType,
    JoinComparatorEnum,
    JoinTypeEnum,
)
from sagemaker.feature_store.feature_group import FeatureDefinition, FeatureGroup, FeatureTypeEnum


@pytest.fixture
def sagemaker_session_mock():
    mock = Mock()
    mock.sagemaker_config = None
    return mock


@pytest.fixture
def feature_group_mock():
    return Mock()


@pytest.fixture
def read_csv_mock():
    return Mock()


@pytest.fixture
def to_csv_file_mock():
    return Mock()


@pytest.fixture
def remove_mock():
    return Mock()


BASE = FeatureGroupToBeMerged(
    ["target-feature", "other-feature"],
    ["target-feature", "other-feature"],
    ["target-feature", "other-feature"],
    "catalog",
    "database",
    "base-table",
    "target-feature",
    FeatureDefinition("other-feature", FeatureTypeEnum.STRING),
    None,
    TableType.FEATURE_GROUP,
)
FEATURE_GROUP = FeatureGroupToBeMerged(
    ["feature-1", "feature-2"],
    ["feature-1", "feature-2"],
    ["feature-1", "feature-2"],
    "catalog",
    "database",
    "table-name",
    "feature-1",
    FeatureDefinition("feature-2", FeatureTypeEnum.FRACTIONAL),
    "target-feature",
    TableType.FEATURE_GROUP,
)


def test_with_feature_group_throw_runtime_error(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
    )
    sagemaker_session_mock.describe_feature_group.return_value = {"OfflineStoreConfig": {}}
    with pytest.raises(RuntimeError) as error:
        dataset_builder.with_feature_group(
            feature_group, "target-feature", ["feature-1", "feature-2"]
        )
    assert "No metastore is configured with FeatureGroup MyFeatureGroup." in str(error)


def test_with_feature_group(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataframe = pd.DataFrame({"feature-1": [420, 380, 390], "feature-2": [50, 40, 45]})
    feature_group.load_feature_definitions(dataframe)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
        record_identifier_feature_name="target-feature",
    )
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {"DataCatalogConfig": {"TableName": "table", "Database": "database"}},
        "RecordIdentifierFeatureName": "feature-1",
        "EventTimeFeatureName": "feature-2",
        "FeatureDefinitions": [
            {"FeatureName": "feature-1", "FeatureType": "String"},
            {"FeatureName": "feature-2", "FeatureType": "String"},
        ],
    }
    dataset_builder.with_feature_group(feature_group, "target-feature", ["feature-1", "feature-2"])
    assert len(dataset_builder._feature_groups_to_be_merged) == 1
    assert dataset_builder._feature_groups_to_be_merged[0].features == [
        "feature-1",
        "feature-2",
    ]
    assert dataset_builder._feature_groups_to_be_merged[0].included_feature_names == [
        "feature-1",
        "feature-2",
    ]
    assert dataset_builder._feature_groups_to_be_merged[0].database == "database"
    assert dataset_builder._feature_groups_to_be_merged[0].table_name == "table"
    assert (
        dataset_builder._feature_groups_to_be_merged[0].record_identifier_feature_name
        == "feature-1"
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].event_time_identifier_feature.feature_name
        == "feature-2"
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].event_time_identifier_feature.feature_type
        == FeatureTypeEnum.STRING
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].target_feature_name_in_base
        == "target-feature"
    )


def test_with_feature_group_with_additional_params(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataframe = pd.DataFrame({"feature-1": [420, 380, 390], "feature-2": [50, 40, 45]})
    feature_group.load_feature_definitions(dataframe)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
        record_identifier_feature_name="target-feature",
    )
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {"DataCatalogConfig": {"TableName": "table", "Database": "database"}},
        "RecordIdentifierFeatureName": "feature-1",
        "EventTimeFeatureName": "feature-2",
        "FeatureDefinitions": [
            {"FeatureName": "feature-1", "FeatureType": "String"},
            {"FeatureName": "feature-2", "FeatureType": "String"},
        ],
    }
    dataset_builder.with_feature_group(
        feature_group,
        "target-feature",
        ["feature-1", "feature-2"],
        join_comparator=JoinComparatorEnum.LESS_THAN,
        join_type=JoinTypeEnum.LEFT_JOIN,
        feature_name_in_target="feature-2",
    )

    assert len(dataset_builder._feature_groups_to_be_merged) == 1
    assert dataset_builder._feature_groups_to_be_merged[0].features == [
        "feature-1",
        "feature-2",
    ]
    assert dataset_builder._feature_groups_to_be_merged[0].included_feature_names == [
        "feature-1",
        "feature-2",
    ]
    assert dataset_builder._feature_groups_to_be_merged[0].database == "database"
    assert dataset_builder._feature_groups_to_be_merged[0].table_name == "table"
    assert (
        dataset_builder._feature_groups_to_be_merged[0].record_identifier_feature_name
        == "feature-1"
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].event_time_identifier_feature.feature_name
        == "feature-2"
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].event_time_identifier_feature.feature_type
        == FeatureTypeEnum.STRING
    )
    assert dataset_builder._feature_groups_to_be_merged[0].join_type == JoinTypeEnum.LEFT_JOIN
    assert (
        dataset_builder._feature_groups_to_be_merged[0].join_comparator
        == JoinComparatorEnum.LESS_THAN
    )
    assert (
        dataset_builder._feature_groups_to_be_merged[0].target_feature_name_in_base
        == "target-feature"
    )


def test_point_in_time_accurate_join(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.point_in_time_accurate_join()
    assert dataset_builder._point_in_time_accurate_join


def test_include_duplicated_records(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.include_duplicated_records()
    assert dataset_builder._include_duplicated_records


def test_include_deleted_records(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.include_deleted_records()
    assert dataset_builder._include_deleted_records


def test_with_number_of_recent_records_by_record_identifier(
    sagemaker_session_mock, feature_group_mock
):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.with_number_of_recent_records_by_record_identifier(5)
    assert dataset_builder._number_of_recent_records == 5


def test_with_number_of_records_from_query_results(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder.with_number_of_records_from_query_results(100)
    assert dataset_builder._number_of_records == 100


def test_with_event_time_range(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    start = datetime.datetime.now()
    end = start + datetime.timedelta(minutes=1)
    dataset_builder.with_event_time_range(start, end)
    assert dataset_builder._event_time_starting_timestamp == start
    assert dataset_builder._event_time_ending_timestamp == end


def test_to_csv_file_not_support_base_type(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    with pytest.raises(ValueError) as error:
        dataset_builder.to_csv_file()
    assert "Base must be either a FeatureGroup or a DataFrame." in str(error)


def test_to_csv_file_with_feature_group(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
    )
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {"DataCatalogConfig": {"TableName": "table", "Database": "database"}},
        "RecordIdentifierFeatureName": "feature-1",
        "EventTimeFeatureName": "feature-2",
        "FeatureDefinitions": [
            {"FeatureName": "feature-1", "FeatureType": "String"},
            {"FeatureName": "feature-2", "FeatureType": "String"},
        ],
    }
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query-id"}
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {"State": "SUCCEEDED"},
            "ResultConfiguration": {"OutputLocation": "s3-file-path"},
            "Query": "query-string",
        }
    }
    file_path, query_string = dataset_builder.to_csv_file()
    assert file_path == "s3-file-path"
    assert query_string == "query-string"


@patch("pandas.DataFrame.to_csv")
@patch("pandas.read_csv")
@patch("os.remove")
def test_to_dataframe_with_dataframe(
    remove_mock, read_csv_mock, to_csv_file_mock, sagemaker_session_mock
):
    dataframe = pd.DataFrame({"feature-1": [420, 380.0, 390], "feature-2": [50, 40.0, 45]})
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=dataframe,
        output_path="s3://file/to/path",
        event_time_identifier_feature_name="feature-2",
    )
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query-id"}
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {
            "Status": {"State": "SUCCEEDED"},
            "ResultConfiguration": {"OutputLocation": "s3://s3-file-path"},
            "Query": "query-string",
        }
    }
    to_csv_file_mock.return_value = None
    read_csv_mock.return_value = dataframe
    os.remove.return_value = None
    df, query_string = dataset_builder.to_dataframe()
    assert df.equals(dataframe)
    assert query_string == "query-string"


def test_construct_where_query_string(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
    )
    time = datetime.datetime.now().replace(microsecond=0)
    start = time + datetime.timedelta(minutes=1)
    end = start + datetime.timedelta(minutes=1)
    dataset_builder._write_time_ending_timestamp = time
    dataset_builder._event_time_starting_timestamp = start
    dataset_builder._event_time_ending_timestamp = end
    query_string = dataset_builder._construct_where_query_string(
        "suffix",
        FeatureDefinition("event-time", FeatureTypeEnum.STRING),
        ["NOT is_deleted"],
    )
    assert (
        query_string
        == "WHERE NOT is_deleted\n"
        + f"AND table_suffix.\"write_time\" <= to_timestamp('{time}', "
        + "'yyyy-mm-dd hh24:mi:ss')\n"
        + 'AND from_iso8601_timestamp(table_suffix."event-time") >= '
        + f"from_unixtime({start.timestamp()})\n"
        + 'AND from_iso8601_timestamp(table_suffix."event-time") <= '
        + f"from_unixtime({end.timestamp()})"
    )


def test_construct_query_string_with_duplicated_records(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    dataset_builder._include_duplicated_records = True

    dataset_builder._feature_groups_to_be_merged = [FEATURE_GROUP]
    query_string = dataset_builder._construct_query_string(BASE)
    assert (
        query_string
        == "WITH fg_base AS (WITH deleted_base AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_base."target-feature"\n'
        + 'ORDER BY origin_base."other-feature" DESC, origin_base."api_invocation_time" DESC, '
        + 'origin_base."write_time" DESC\n'
        + ") AS deleted_row_base\n"
        + 'FROM "database"."base-table" origin_base\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_base = 1\n"
        + ")\n"
        + 'SELECT table_base."target-feature", table_base."other-feature"\n'
        + "FROM (\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + 'FROM "database"."base-table" table_base\n'
        + "LEFT JOIN deleted_base\n"
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + 'WHERE deleted_base."target-feature" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + "FROM deleted_base\n"
        + 'JOIN "database"."base-table" table_base\n'
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + "AND (\n"
        + 'table_base."other-feature" > deleted_base."other-feature"\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" > deleted_base."api_invocation_time")\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" = deleted_base."api_invocation_time" AND '
        + 'table_base."write_time" > deleted_base."write_time")\n'
        + ")\n"
        + ") AS table_base\n"
        + "),\n"
        + "fg_0 AS (WITH deleted_0 AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_0."feature-1"\n'
        + 'ORDER BY origin_0."feature-2" DESC, origin_0."api_invocation_time" DESC, '
        + 'origin_0."write_time" DESC\n'
        + ") AS deleted_row_0\n"
        + 'FROM "database"."table-name" origin_0\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_0 = 1\n"
        + ")\n"
        + 'SELECT table_0."feature-1", table_0."feature-2"\n'
        + "FROM (\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + 'FROM "database"."table-name" table_0\n'
        + "LEFT JOIN deleted_0\n"
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + 'WHERE deleted_0."feature-1" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + "FROM deleted_0\n"
        + 'JOIN "database"."table-name" table_0\n'
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + "AND (\n"
        + 'table_0."feature-2" > deleted_0."feature-2"\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND table_0."api_invocation_time" > '
        + 'deleted_0."api_invocation_time")\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND table_0."api_invocation_time" = '
        + 'deleted_0."api_invocation_time" AND table_0."write_time" > deleted_0."write_time")\n'
        + ")\n"
        + ") AS table_0\n"
        + ")\n"
        + 'SELECT target-feature, other-feature, "feature-1.1", "feature-2.1"\n'
        + "FROM (\n"
        + 'SELECT fg_base.target-feature, fg_base.other-feature, fg_0."feature-1" as '
        + '"feature-1.1", fg_0."feature-2" as "feature-2.1", row_number() OVER (\n'
        + 'PARTITION BY fg_base."target-feature"\n'
        + 'ORDER BY fg_base."other-feature" DESC, fg_0."feature-2" DESC\n'
        + ") AS row_recent\n"
        + "FROM fg_base\n"
        + "JOIN fg_0\n"
        + 'ON fg_base."target-feature" = fg_0."feature-1"\n'
        + ")\n"
    )


def test_construct_query_string(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group,
        output_path="file/to/path",
    )
    dataset_builder._point_in_time_accurate_join = True
    dataset_builder._event_time_identifier_feature_name = "target-feature"
    dataset_builder._feature_groups_to_be_merged = [FEATURE_GROUP]
    query_string = dataset_builder._construct_query_string(BASE)
    assert (
        query_string
        == "WITH fg_base AS (WITH table_base AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_base."target-feature", origin_base."other-feature"\n'
        + 'ORDER BY origin_base."api_invocation_time" DESC, origin_base."write_time" DESC\n'
        + ") AS dedup_row_base\n"
        + 'FROM "database"."base-table" origin_base\n'
        + ")\n"
        + "WHERE dedup_row_base = 1\n"
        + "),\n"
        + "deleted_base AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_base."target-feature"\n'
        + 'ORDER BY origin_base."other-feature" DESC, origin_base."api_invocation_time" '
        + 'DESC, origin_base."write_time" DESC\n'
        + ") AS deleted_row_base\n"
        + 'FROM "database"."base-table" origin_base\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_base = 1\n"
        + ")\n"
        + 'SELECT table_base."target-feature", table_base."other-feature"\n'
        + "FROM (\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + "FROM table_base\n"
        + "LEFT JOIN deleted_base\n"
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + 'WHERE deleted_base."target-feature" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + "FROM deleted_base\n"
        + "JOIN table_base\n"
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + "AND (\n"
        + 'table_base."other-feature" > deleted_base."other-feature"\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" > deleted_base."api_invocation_time")\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" = deleted_base."api_invocation_time" AND '
        + 'table_base."write_time" > deleted_base."write_time")\n'
        + ")\n"
        + ") AS table_base\n"
        + "),\n"
        + "fg_0 AS (WITH table_0 AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_0."feature-1", origin_0."feature-2"\n'
        + 'ORDER BY origin_0."api_invocation_time" DESC, origin_0."write_time" DESC\n'
        + ") AS dedup_row_0\n"
        + 'FROM "database"."table-name" origin_0\n'
        + ")\n"
        + "WHERE dedup_row_0 = 1\n"
        + "),\n"
        + "deleted_0 AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_0."feature-1"\n'
        + 'ORDER BY origin_0."feature-2" DESC, origin_0."api_invocation_time" DESC, '
        + 'origin_0."write_time" DESC\n'
        + ") AS deleted_row_0\n"
        + 'FROM "database"."table-name" origin_0\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_0 = 1\n"
        + ")\n"
        + 'SELECT table_0."feature-1", table_0."feature-2"\n'
        + "FROM (\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + "FROM table_0\n"
        + "LEFT JOIN deleted_0\n"
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + 'WHERE deleted_0."feature-1" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + "FROM deleted_0\n"
        + "JOIN table_0\n"
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + "AND (\n"
        + 'table_0."feature-2" > deleted_0."feature-2"\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND '
        + 'table_0."api_invocation_time" > deleted_0."api_invocation_time")\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND '
        + 'table_0."api_invocation_time" = deleted_0."api_invocation_time" AND '
        + 'table_0."write_time" > deleted_0."write_time")\n'
        + ")\n"
        + ") AS table_0\n"
        + ")\n"
        + 'SELECT target-feature, other-feature, "feature-1.1", "feature-2.1"\n'
        + "FROM (\n"
        + 'SELECT fg_base.target-feature, fg_base.other-feature, fg_0."feature-1" as '
        + '"feature-1.1", fg_0."feature-2" as "feature-2.1", row_number() OVER (\n'
        + 'PARTITION BY fg_base."target-feature"\n'
        + 'ORDER BY fg_base."other-feature" DESC, fg_0."feature-2" DESC\n'
        + ") AS row_recent\n"
        + "FROM fg_base\n"
        + "JOIN fg_0\n"
        + 'ON fg_base."target-feature" = fg_0."feature-1"\n'
        + 'AND from_unixtime(fg_base."target-feature") >= from_unixtime(fg_0."feature-2")\n'
        + ")\n"
    )


# Tests the optional feature_name_in_target, join_comparator and join_type parameters
def test_with_feature_group_with_optional_params_query_string(sagemaker_session_mock):
    base_feature_group = FeatureGroup(
        name="base_feature_group", sagemaker_session=sagemaker_session_mock
    )
    target_feature_group = FeatureGroup(
        name="target_feature_group", sagemaker_session=sagemaker_session_mock
    )

    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=base_feature_group,
        output_path="file/to/path",
        record_identifier_feature_name="target-feature",
    )

    dataset_builder._event_time_identifier_feature_name = "target-feature"

    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {
            "DataCatalogConfig": {"TableName": "table-name", "Database": "database"}
        },
        "RecordIdentifierFeatureName": "feature-1",
        "EventTimeFeatureName": "feature-2",
        "FeatureDefinitions": [
            {"FeatureName": "feature-1", "FeatureType": "String"},
            {"FeatureName": "feature-2", "FeatureType": "String"},
        ],
    }

    dataset_builder.with_feature_group(
        target_feature_group,
        "target-feature",
        ["feature-1", "feature-2"],
        "feature-2",
        JoinComparatorEnum.GREATER_THAN,
        JoinTypeEnum.FULL_JOIN,
    )

    query_string = dataset_builder._construct_query_string(BASE)

    assert (
        query_string
        == "WITH fg_base AS (WITH table_base AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_base."target-feature", origin_base."other-feature"\n'
        + 'ORDER BY origin_base."api_invocation_time" DESC, origin_base."write_time" DESC\n'
        + ") AS dedup_row_base\n"
        + 'FROM "database"."base-table" origin_base\n'
        + ")\n"
        + "WHERE dedup_row_base = 1\n"
        + "),\n"
        + "deleted_base AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_base."target-feature"\n'
        + 'ORDER BY origin_base."other-feature" DESC, origin_base."api_invocation_time" '
        + 'DESC, origin_base."write_time" DESC\n'
        + ") AS deleted_row_base\n"
        + 'FROM "database"."base-table" origin_base\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_base = 1\n"
        + ")\n"
        + 'SELECT table_base."target-feature", table_base."other-feature"\n'
        + "FROM (\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + "FROM table_base\n"
        + "LEFT JOIN deleted_base\n"
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + 'WHERE deleted_base."target-feature" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_base."target-feature", table_base."other-feature", '
        + 'table_base."write_time"\n'
        + "FROM deleted_base\n"
        + "JOIN table_base\n"
        + 'ON table_base."target-feature" = deleted_base."target-feature"\n'
        + "AND (\n"
        + 'table_base."other-feature" > deleted_base."other-feature"\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" > deleted_base."api_invocation_time")\n'
        + 'OR (table_base."other-feature" = deleted_base."other-feature" AND '
        + 'table_base."api_invocation_time" = deleted_base."api_invocation_time" AND '
        + 'table_base."write_time" > deleted_base."write_time")\n'
        + ")\n"
        + ") AS table_base\n"
        + "),\n"
        + "fg_0 AS (WITH table_0 AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_0."feature-1", origin_0."feature-2"\n'
        + 'ORDER BY origin_0."api_invocation_time" DESC, origin_0."write_time" DESC\n'
        + ") AS dedup_row_0\n"
        + 'FROM "database"."table-name" origin_0\n'
        + ")\n"
        + "WHERE dedup_row_0 = 1\n"
        + "),\n"
        + "deleted_0 AS (\n"
        + "SELECT *\n"
        + "FROM (\n"
        + "SELECT *, row_number() OVER (\n"
        + 'PARTITION BY origin_0."feature-1"\n'
        + 'ORDER BY origin_0."feature-2" DESC, origin_0."api_invocation_time" DESC, '
        + 'origin_0."write_time" DESC\n'
        + ") AS deleted_row_0\n"
        + 'FROM "database"."table-name" origin_0\n'
        + "WHERE is_deleted\n"
        + ")\n"
        + "WHERE deleted_row_0 = 1\n"
        + ")\n"
        + 'SELECT table_0."feature-1", table_0."feature-2"\n'
        + "FROM (\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + "FROM table_0\n"
        + "LEFT JOIN deleted_0\n"
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + 'WHERE deleted_0."feature-1" IS NULL\n'
        + "UNION ALL\n"
        + 'SELECT table_0."feature-1", table_0."feature-2", table_0."write_time"\n'
        + "FROM deleted_0\n"
        + "JOIN table_0\n"
        + 'ON table_0."feature-1" = deleted_0."feature-1"\n'
        + "AND (\n"
        + 'table_0."feature-2" > deleted_0."feature-2"\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND '
        + 'table_0."api_invocation_time" > deleted_0."api_invocation_time")\n'
        + 'OR (table_0."feature-2" = deleted_0."feature-2" AND '
        + 'table_0."api_invocation_time" = deleted_0."api_invocation_time" AND '
        + 'table_0."write_time" > deleted_0."write_time")\n'
        + ")\n"
        + ") AS table_0\n"
        + ")\n"
        + 'SELECT target-feature, other-feature, "feature-1.1", "feature-2.1"\n'
        + "FROM (\n"
        + 'SELECT fg_base.target-feature, fg_base.other-feature, fg_0."feature-1" as '
        + '"feature-1.1", fg_0."feature-2" as "feature-2.1", row_number() OVER (\n'
        + 'PARTITION BY fg_base."target-feature"\n'
        + 'ORDER BY fg_base."other-feature" DESC, fg_0."feature-2" DESC\n'
        + ") AS row_recent\n"
        + "FROM fg_base\n"
        + "FULL JOIN fg_0\n"
        + 'ON fg_base."target-feature" > fg_0."feature-2"\n'
        + ")\n"
    )


def test_create_temp_table(sagemaker_session_mock):
    dataframe = pd.DataFrame({"feature-1": [420, 380, 390], "feature-2": [50, 40, 45]})
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=dataframe,
        output_path="file/to/path",
    )
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query-id"}
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }
    dataset_builder._create_temp_table("table-name", "s3-folder")
    assert sagemaker_session_mock.start_query_execution.call_count == 1
    sagemaker_session_mock.start_query_execution.assert_called_once_with(
        catalog="AwsDataCatalog",
        database="sagemaker_featurestore",
        query_string="CREATE EXTERNAL TABLE table-name (feature-1 INT, feature-2 INT) "
        + "ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde' "
        + 'WITH SERDEPROPERTIES ("separatorChar" = ",", "quoteChar" = "`", "escapeChar" = "\\\\") '
        + "LOCATION 's3-folder';",
        output_location="file/to/path",
        kms_key=None,
    )


@pytest.mark.parametrize(
    "column, expected",
    [
        ("feature-1", "feature-1 STRING"),
        ("feature-2", "feature-2 INT"),
        ("feature-3", "feature-3 DOUBLE"),
        ("feature-4", "feature-4 BOOLEAN"),
        ("feature-5", "feature-5 TIMESTAMP"),
    ],
)
def test_construct_athena_table_column_string(column, expected, sagemaker_session_mock):
    dataframe = pd.DataFrame(
        {
            "feature-1": ["420"],
            "feature-2": [50],
            "feature-3": [5.0],
            "feature-4": [True],
            "feature-5": [pd.Timestamp(1513393355)],
        }
    )
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=dataframe,
        output_path="file/to/path",
    )
    query_string = dataset_builder._construct_athena_table_column_string(column)
    assert query_string == expected


def test_construct_athena_table_column_string_not_support_column_type(
    sagemaker_session_mock,
):
    dataframe = pd.DataFrame({"feature": pd.Series([1] * 3, dtype="int8")})
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=dataframe,
        output_path="file/to/path",
    )
    with pytest.raises(RuntimeError) as error:
        dataset_builder._construct_athena_table_column_string("feature")
    assert "The dataframe type int8 is not supported yet." in str(error)


def test_run_query_throw_runtime_error(sagemaker_session_mock, feature_group_mock):
    dataset_builder = DatasetBuilder(
        sagemaker_session=sagemaker_session_mock,
        base=feature_group_mock,
        output_path="file/to/path",
    )
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query-id"}
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "FAILED"}}
    }
    with pytest.raises(RuntimeError) as error:
        dataset_builder._run_query("query-string", "catalog", "database")
    assert "Failed to execute query query-id." in str(error)
