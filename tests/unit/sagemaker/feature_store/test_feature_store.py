# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


import pandas as pd
import pytest
from mock import Mock, patch, MagicMock

from sagemaker.feature_store.feature_definition import (
    FractionalFeatureDefinition,
    IntegralFeatureDefinition,
    StringFeatureDefinition,
    FeatureTypeEnum,
)
from sagemaker.feature_store.feature_group import FeatureGroup, IngestionManagerPandas, AthenaQuery


@pytest.fixture
def role_arn():
    return "arn:role"


@pytest.fixture
def s3_uri():
    return "s3://some/uri"


@pytest.fixture
def sagemaker_session_mock():
    return Mock()


@pytest.fixture
def feature_group_dummy_definitions():
    return [
        FractionalFeatureDefinition(feature_name="feature1"),
        IntegralFeatureDefinition(feature_name="feature2"),
        StringFeatureDefinition(feature_name="feature3"),
    ]


@pytest.fixture
def create_table_ddl():
    return (
        "CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (\n"
        "  feature1 FLOAT\n"
        "  feature2 INT\n"
        "  feature3 STRING\n)\n"
        "ROW FORMAT SERDE 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'\n"
        "  STORED AS\n"
        "  INPUTFORMAT 'parquet.hive.DeprecatedParquetInputFormat'\n"
        "  OUTPUTFORMAT 'parquet.hive.DeprecatedParquetOutputFormat'\n"
        "LOCATION 's3://some-bucket"
        "/{account}/sagemaker/{region}/offline-store/{feature_group_name}'"
    )


def test_feature_store_create(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions, s3_uri
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=s3_uri,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
        offline_store_config={
            "DisableGlueTableCreation": False,
            "S3StorageConfig": {"S3Uri": s3_uri},
        },
    )


def test_feature_store_create_online_only(
    sagemaker_session_mock, role_arn, feature_group_dummy_definitions
):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    feature_group.create(
        s3_uri=False,
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        role_arn=role_arn,
        enable_online_store=True,
    )
    sagemaker_session_mock.create_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup",
        record_identifier_name="feature1",
        event_time_feature_name="feature2",
        feature_definitions=[fd.to_dict() for fd in feature_group_dummy_definitions],
        role_arn=role_arn,
        description=None,
        tags=None,
        online_store_config={"EnableOnlineStore": True},
    )


def test_feature_store_delete(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.delete()
    sagemaker_session_mock.delete_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup"
    )


def test_feature_store_describe(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.describe()
    sagemaker_session_mock.describe_feature_group.assert_called_with(
        feature_group_name="MyFeatureGroup", next_token=None
    )


def test_put_record(sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyFeatureGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.put_record(record=[])
    sagemaker_session_mock.put_record.assert_called_with(
        feature_group_name="MyFeatureGroup", record=[]
    )


def test_load_feature_definition(sagemaker_session_mock):
    feature_group = FeatureGroup(name="SomeGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(
        {
            "float": pd.Series([2.0], dtype="float64"),
            "int": pd.Series([2], dtype="int64"),
            "string": pd.Series(["f1"], dtype="string"),
        }
    )
    feature_definitions = feature_group.load_feature_definitions(data_frame=df)
    names = [fd.feature_name for fd in feature_definitions]
    types = [fd.feature_type for fd in feature_definitions]
    assert names == ["float", "int", "string"]
    assert types == [FeatureTypeEnum.FRACTIONAL, FeatureTypeEnum.INTEGRAL, FeatureTypeEnum.STRING]


def test_load_feature_definition_unsupported_types(sagemaker_session_mock):
    feature_group = FeatureGroup(name="FailedGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(
        {
            "float": pd.Series([2.0], dtype="float64"),
            "int": pd.Series([2], dtype="int64"),
            "object": pd.Series(["f1"], dtype="object"),
        }
    )
    with pytest.raises(ValueError) as error:
        feature_group.load_feature_definitions(data_frame=df)
    assert "Failed to infer Feature type based on dtype object for column object." in str(error)


@patch("sagemaker.feature_store.feature_group.IngestionManagerPandas")
def test_ingest(ingestion_manager_init, sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame(dict((f"float{i}", pd.Series([2.0], dtype="float64")) for i in range(300)))

    mock_ingestion_manager_instance = Mock()
    ingestion_manager_init.return_value = mock_ingestion_manager_instance
    feature_group.ingest(data_frame=df, max_workers=10)

    ingestion_manager_init.assert_called_once_with(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        data_frame=df,
        max_workers=10,
    )
    mock_ingestion_manager_instance.run.assert_called_once_with(wait=True, timeout=None)


@patch("sagemaker.feature_store.feature_group.IngestionManagerPandas")
def test_ingest_default_max_workers(ingestion_manager_init, sagemaker_session_mock):
    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})

    mock_ingestion_manager_instance = Mock()
    ingestion_manager_init.return_value = mock_ingestion_manager_instance
    feature_group.ingest(data_frame=df)

    ingestion_manager_init.assert_called_once_with(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        data_frame=df,
        max_workers=1,
    )
    mock_ingestion_manager_instance.run.assert_called_once_with(wait=True, timeout=None)


def test_as_hive_ddl_with_default_values(
    create_table_ddl, feature_group_dummy_definitions, sagemaker_session_mock
):
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {"S3StorageConfig": {"S3Uri": "s3://some-bucket"}}
    }
    sagemaker_session_mock.account_id.return_value = "1234"
    sagemaker_session_mock.boto_session.region_name = "us-west-2"

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    assert (
        create_table_ddl.format(
            database="sagemaker_featurestore",
            table_name="MyGroup",
            account="1234",
            region="us-west-2",
            feature_group_name="MyGroup",
        )
        == feature_group.as_hive_ddl()
    )


def test_as_hive_ddl(create_table_ddl, feature_group_dummy_definitions, sagemaker_session_mock):
    sagemaker_session_mock.describe_feature_group.return_value = {
        "OfflineStoreConfig": {"S3StorageConfig": {"S3Uri": "s3://some-bucket"}}
    }
    sagemaker_session_mock.account_id.return_value = "1234"
    sagemaker_session_mock.boto_session.region_name = "us-west-2"

    feature_group = FeatureGroup(name="MyGroup", sagemaker_session=sagemaker_session_mock)
    feature_group.feature_definitions = feature_group_dummy_definitions
    assert (
        create_table_ddl.format(
            database="MyDatabase",
            table_name="MyTable",
            account="1234",
            region="us-west-2",
            feature_group_name="MyGroup",
        )
        == feature_group.as_hive_ddl(database="MyDatabase", table_name="MyTable")
    )


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    MagicMock(),
)
def test_ingestion_manager_run_success():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        data_frame=df,
        max_workers=10,
    )
    manager.run()


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    MagicMock(side_effect=Exception("Failed!")),
)
def test_ingestion_manager_run_failure():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_session=sagemaker_session_mock,
        data_frame=df,
        max_workers=10,
    )
    with pytest.raises(RuntimeError) as error:
        manager.run()
    assert "Failed to ingest some data into FeatureGroup MyGroup" in str(error)


@pytest.fixture
def query(sagemaker_session_mock):
    return AthenaQuery(
        catalog="catalog",
        database="database",
        table_name="table_name",
        sagemaker_session=sagemaker_session_mock,
    )


def test_athena_query_run(sagemaker_session_mock, query):
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query_id"}
    query.run(query_string="query", output_location="s3://some-bucket/some-path")
    sagemaker_session_mock.start_query_execution.assert_called_with(
        catalog="catalog",
        database="database",
        query_string="query",
        output_location="s3://some-bucket/some-path",
        kms_key=None,
    )
    assert "some-bucket" == query._result_bucket
    assert "some-path" == query._result_file_prefix
    assert "query_id" == query._current_query_execution_id


def test_athena_query_wait(sagemaker_session_mock, query):
    query._current_query_execution_id = "query_id"
    query.wait()
    sagemaker_session_mock.wait_for_athena_query.assert_called_with(query_execution_id="query_id")


def test_athena_query_get_query_execution(sagemaker_session_mock, query):
    query._current_query_execution_id = "query_id"
    query.get_query_execution()
    sagemaker_session_mock.get_query_execution.assert_called_with(query_execution_id="query_id")


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
@patch("pandas.read_csv")
def test_athena_query_as_dataframe(read_csv, sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "SUCCEEDED"}}
    }
    query._current_query_execution_id = "query_id"
    query._result_bucket = "bucket"
    query._result_file_prefix = "prefix"
    query.as_dataframe()
    sagemaker_session_mock.download_athena_query_result.assert_called_with(
        bucket="bucket",
        prefix="prefix",
        query_execution_id="query_id",
        filename="tmp/query_id.csv",
    )
    read_csv.assert_called_with("tmp/query_id.csv", delimiter=",")


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_failed(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "FAILED"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Failed to execute query query_id" in str(error)


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_queued(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "QUEUED"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Current query query_id is still being executed" in str(error)


@patch("tempfile.gettempdir", Mock(return_value="tmp"))
def test_athena_query_as_dataframe_query_running(sagemaker_session_mock, query):
    sagemaker_session_mock.get_query_execution.return_value = {
        "QueryExecution": {"Status": {"State": "RUNNING"}}
    }
    query._current_query_execution_id = "query_id"
    with pytest.raises(RuntimeError) as error:
        query.as_dataframe()
    assert "Current query query_id is still being executed" in str(error)
