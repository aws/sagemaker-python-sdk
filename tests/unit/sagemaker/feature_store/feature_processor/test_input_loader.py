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
from mock import Mock
from pyspark.sql import SparkSession

from sagemaker.feature_store.feature_processor._data_source import (
    CSVDataSource,
    FeatureGroupDataSource,
    IcebergTableDataSource,
    ParquetDataSource,
)
from sagemaker.feature_store.feature_processor._input_loader import SparkDataFrameInputLoader
from sagemaker.feature_store.feature_processor._spark_factory import SparkSessionFactory
from sagemaker.session import Session


@pytest.fixture
def describe_fg_response():
    return tdh.DESCRIBE_FEATURE_GROUP_RESPONSE.copy()


@pytest.fixture
def sagemaker_session(describe_fg_response):
    return Mock(Session, describe_feature_group=Mock(return_value=describe_fg_response))


@pytest.fixture
def spark_session():
    return Mock(
        SparkSession,
        read=Mock(
            csv=Mock(return_value=Mock()),
            parquet=Mock(return_value=Mock()),
            conf=Mock(set=Mock()),
            table=Mock(return_value=Mock()),
        ),
    )


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
def input_loader(spark_session_factory, sagemaker_session):
    return SparkDataFrameInputLoader(spark_session_factory, sagemaker_session)


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


def test_load_from_iceberg_table(input_loader, spark_session, spark_session_factory):
    iceberg_table_data_source = IcebergTableDataSource(
        warehouse_s3_uri="s3://bucket/prefix/",
        catalog="Catalog",
        database="Database",
        table="Table",
    )

    input_loader.load_from_iceberg_table(iceberg_table_data_source)
    spark_session_factory.get_spark_session_with_iceberg_config.assert_called_with(
        "s3://bucket/prefix/", "catalog"
    )
    spark_session.table.assert_called_with("catalog.database.table")


def test_load_from_feature_group_with_arn(sagemaker_session, input_loader):
    fg_arn = tdh.INPUT_FEATURE_GROUP_ARN
    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(name=fg_arn)

    input_loader.load_from_feature_group(fg_data_source)

    sagemaker_session.describe_feature_group.assert_called_with(fg_name)


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
    describe_fg_response, spark_session_factory, spark_session
):
    describe_iceberg_fg_response = describe_fg_response.copy()
    describe_iceberg_fg_response["OfflineStoreConfig"]["TableFormat"] = "Iceberg"
    mocked_session = Mock(
        Session, describe_feature_group=Mock(return_value=describe_iceberg_fg_response)
    )
    mock_input_loader = SparkDataFrameInputLoader(spark_session_factory, mocked_session)

    fg_name = tdh.INPUT_FEATURE_GROUP_NAME
    fg_data_source = FeatureGroupDataSource(name=fg_name)
    mock_input_loader.load_from_feature_group(fg_data_source)

    mocked_session.describe_feature_group.assert_called_with(fg_name)
    spark_session.table.assert_called_with(
        "awsdatacatalog.sagemaker_featurestore.input_fg_1680142547"
    )
