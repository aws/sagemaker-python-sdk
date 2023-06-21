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
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import pytest
import test_data_helpers as tdh
from mock import Mock, patch, call
from pyspark.sql import SparkSession, DataFrame
from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    ParquetDataSource,
    IcebergTableDataSource,
)
from sagemaker.feature_store.feature_processor._input_loader import (
    SparkDataFrameInputLoader,
)
from sagemaker.feature_store.feature_processor._spark_factory import SparkSessionFactory
from sagemaker.feature_store.feature_processor._env import EnvironmentHelper
from sagemaker.session import Session


@pytest.fixture
def describe_fg_response():
    return tdh.DESCRIBE_FEATURE_GROUP_RESPONSE.copy()


@pytest.fixture
def sagemaker_session(describe_fg_response):
    return Mock(Session, describe_feature_group=Mock(return_value=describe_fg_response))


@pytest.fixture
def spark_session(mock_data_frame):
    return Mock(
        SparkSession,
        read=Mock(
            csv=Mock(return_value=Mock()),
            parquet=Mock(return_value=mock_data_frame),
            conf=Mock(set=Mock()),
        ),
        table=Mock(return_value=mock_data_frame),
    )


@pytest.fixture
def environment_helper():
    return Mock(
        EnvironmentHelper,
        get_job_scheduled_time=Mock(return_value="2023-05-05T15:22:57Z"),
    )


@pytest.fixture
def mock_data_frame():
    return Mock(DataFrame, filter=Mock())


@pytest.fixture
def spark_session_factory(spark_session):
    factory = Mock(SparkSessionFactory)
    factory.spark_session = spark_session
    factory.get_spark_session_with_iceberg_config = Mock(return_value=spark_session)
    return factory


@pytest.fixture
def fp_config():
    return tdh.create_fp_config()


@pytest.fixture
def input_loader(spark_session_factory, sagemaker_session, environment_helper):
    return SparkDataFrameInputLoader(
        spark_session_factory,
        environment_helper,
        sagemaker_session,
    )


def test_load_from_s3_with_csv_object(input_loader: SparkDataFrameInputLoader, spark_session):
    s3_data_source = CSVDataSource(
        s3_uri="s3://bucket/prefix/file.csv",
        csv_header=True,
        csv_infer_schema=True,
    )

    input_loader.load_from_s3(s3_data_source)

    spark_session.read.csv.assert_called_with(
        "s3a://bucket/prefix/file.csv", header=True, inferSchema=True
    )


def test_load_from_s3_with_parquet_object(input_loader, spark_session):
    s3_data_source = ParquetDataSource(s3_uri="s3://bucket/prefix/file.parquet")

    input_loader.load_from_s3(s3_data_source)

    spark_session.read.parquet.assert_called_with("s3a://bucket/prefix/file.parquet")


@pytest.mark.parametrize(
    "condition",
    [(None), ("condition")],
)
@patch(
    "sagemaker.feature_store.feature_processor._input_loader."
    "SparkDataFrameInputLoader._get_iceberg_offset_filter_condition"
)
def test_load_from_iceberg_table(
    mock_get_filter_condition,
    condition,
    input_loader,
    spark_session,
    spark_session_factory,
    mock_data_frame,
):
    iceberg_table_data_source = IcebergTableDataSource(
        warehouse_s3_uri="s3://bucket/prefix/",
        catalog="Catalog",
        database="Database",
        table="Table",
    )
    mock_get_filter_condition.return_value = condition

    input_loader.load_from_iceberg_table(iceberg_table_data_source, "event_time", "start", "end")
    spark_session_factory.get_spark_session_with_iceberg_config.assert_called_with(
        "s3://bucket/prefix/", "catalog"
    )
    spark_session.table.assert_called_with("catalog.database.table")
    mock_get_filter_condition.assert_called_with("event_time", "start", "end")

    if condition:
        mock_data_frame.filter.assert_called_with(condition)
    else:
        mock_data_frame.filter.assert_not_called()


@patch(
    "sagemaker.feature_store.feature_processor._input_loader.SparkDataFrameInputLoader.load_from_date_partitioned_s3"
)
def test_load_from_feature_group_with_arn(
    mock_load_from_date_partitioned_s3, sagemaker_session, input_loader
):
    fg_arn = tdh.INPUT_FEATURE_GROUP_ARN
    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(
        name=fg_arn, input_start_offset="start", input_end_offset="end"
    )

    input_loader.load_from_feature_group(fg_data_source)

    sagemaker_session.describe_feature_group.assert_called_with(fg_name)
    mock_load_from_date_partitioned_s3.assert_called_with(
        ParquetDataSource(tdh.INPUT_FEATURE_GROUP_RESOLVED_OUTPUT_S3_URI),
        "start",
        "end",
    )


def test_load_from_feature_group_offline_store_not_enabled(input_loader, describe_fg_response):
    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(name=fg_name)
    with pytest.raises(
        ValueError,
        match=(
            f"Input Feature Groups must have an enabled Offline Store."
            f" Feature Group: {fg_name} does not have an Offline Store enabled."
        ),
    ):
        del describe_fg_response["OfflineStoreConfig"]
        input_loader.load_from_feature_group(fg_data_source)


def test_load_from_feature_group_with_default_table_format(
    sagemaker_session, input_loader, spark_session
):
    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(name=fg_name)
    input_loader.load_from_feature_group(fg_data_source)

    sagemaker_session.describe_feature_group.assert_called_with(fg_name)
    spark_session.read.parquet.assert_called_with(
        tdh.INPUT_FEATURE_GROUP_RESOLVED_OUTPUT_S3_URI.replace("s3:", "s3a:")
    )


def test_load_from_feature_group_with_iceberg_table_format(
    describe_fg_response, spark_session_factory, spark_session, environment_helper
):
    describe_iceberg_fg_response = describe_fg_response.copy()
    describe_iceberg_fg_response["OfflineStoreConfig"]["TableFormat"] = "Iceberg"
    mocked_session = Mock(
        Session, describe_feature_group=Mock(return_value=describe_iceberg_fg_response)
    )
    mock_input_loader = SparkDataFrameInputLoader(
        spark_session_factory, environment_helper, mocked_session
    )

    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(name=fg_name)
    mock_input_loader.load_from_feature_group(fg_data_source)

    mocked_session.describe_feature_group.assert_called_with(fg_name)
    spark_session.table.assert_called_with(
        "awsdatacatalog.sagemaker_featurestore.input_fg_1680142547"
    )


@pytest.mark.parametrize(
    "param",
    [
        (None, None, None),
        ("start", None, "event_time >= 'start_time'"),
        (None, "end", "event_time < 'end_time'"),
        ("start", "end", "event_time >= 'start_time' AND event_time < 'end_time'"),
    ],
)
@patch(
    "sagemaker.feature_store.feature_processor._input_offset_parser.InputOffsetParser.get_iso_format_offset_date",
    side_effect=[
        "start_time",
        "end_time",
    ],
)
def test_get_iceberg_offset_filter_condition(mock_get_iso_date, param, input_loader):
    start_offset, end_offset, expected_condition_str = param

    condition = input_loader._get_iceberg_offset_filter_condition(
        "event_time", start_offset, end_offset
    )

    if start_offset or end_offset:
        mock_get_iso_date.assert_has_calls([call(start_offset), call(end_offset)])
    else:
        mock_get_iso_date.assert_not_called()

    assert condition == expected_condition_str


@pytest.mark.parametrize(
    "param",
    [
        (None, None, None),
        (
            "start",
            None,
            "(year >= 'year_start') AND NOT ((year = 'year_start' AND month < 'month_start') OR "
            "(year = 'year_start' AND month = 'month_start' AND day < 'day_start') OR "
            "(year = 'year_start' AND month = 'month_start' AND day = 'day_start' AND hour < 'hour_start'))",
        ),
        (
            None,
            "end",
            "(year <= 'year_end') AND NOT ((year = 'year_end' AND month > 'month_end') OR "
            "(year = 'year_end' AND month = 'month_end' AND day > 'day_end') OR (year = 'year_end' "
            "AND month = 'month_end' AND day = 'day_end' AND hour >= 'hour_end'))",
        ),
        (
            "start",
            "end",
            "(year >= 'year_start' AND year <= 'year_end') AND NOT ((year = 'year_start' AND "
            "month < 'month_start') OR (year = 'year_start' AND month = 'month_start' AND day < 'day_start') OR "
            "(year = 'year_start' AND month = 'month_start' AND day = 'day_start' AND hour < 'hour_start') OR "
            "(year = 'year_end' AND month > 'month_end') OR "
            "(year = 'year_end' AND month = 'month_end' AND day > 'day_end') OR "
            "(year = 'year_end' AND month = 'month_end' AND day = 'day_end' AND hour >= 'hour_end'))",
        ),
    ],
)
@patch(
    "sagemaker.feature_store.feature_processor._input_offset_parser.InputOffsetParser."
    "get_offset_date_year_month_day_hour",
    side_effect=[
        ("year_start", "month_start", "day_start", "hour_start"),
        ("year_end", "month_end", "day_end", "hour_end"),
    ],
)
def test_get_s3_partitions_offset_filter_condition(mock_get_ymdh, param, input_loader):
    start_offset, end_offset, expected_condition_str = param

    condition = input_loader._get_s3_partitions_offset_filter_condition(start_offset, end_offset)

    if start_offset or end_offset:
        mock_get_ymdh.assert_has_calls([call(start_offset), call(end_offset)])
    else:
        mock_get_ymdh.assert_not_called()

    assert condition == expected_condition_str


@pytest.mark.parametrize(
    "condition",
    [(None), ("condition")],
)
def test_load_from_date_partitioned_s3(input_loader, spark_session, mock_data_frame, condition):
    input_loader._get_s3_partitions_offset_filter_condition = Mock(return_value=condition)

    input_loader.load_from_date_partitioned_s3(
        ParquetDataSource("s3://path/to/file"), "start", "end"
    )
    df = spark_session.read.parquet
    df.assert_called_with("s3a://path/to/file")

    if condition:
        mock_data_frame.filter.assert_called_with(condition)
    else:
        mock_data_frame.filter.assert_not_called()
