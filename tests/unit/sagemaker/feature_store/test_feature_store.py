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
"""Test for Feature Store"""
from __future__ import absolute_import

import datetime

import pandas as pd
import pytest
from mock import Mock

from sagemaker.feature_store.feature_store import FeatureStore

DATAFRAME = pd.DataFrame({"feature_1": [420, 380, 390], "feature_2": [50, 40, 45]})

class PicklableMock(Mock):
    """Mock class use for tests"""

    def __reduce__(self):
        return (Mock, ())


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
def feature_group_mock():
    return Mock()


def test_minimal_create_dataset(sagemaker_session_mock, feature_group_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=feature_group_mock,
        output_path="file/to/path",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base == feature_group_mock
    assert dataset_builder._output_path == "file/to/path"


def test_complete_create_dataset(sagemaker_session_mock, feature_group_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=feature_group_mock,
        included_feature_names=["feature_1", "feature_2"],
        output_path="file/to/path",
        kms_key_id="kms-key-id",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base == feature_group_mock
    assert dataset_builder._included_feature_names == ["feature_1", "feature_2"]
    assert dataset_builder._output_path == "file/to/path"
    assert dataset_builder._kms_key_id == "kms-key-id"


def test_create_dataset_with_dataframe(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    dataset_builder = feature_store.create_dataset(
        base=DATAFRAME,
        record_identifier_feature_name="feature_1",
        event_time_identifier_feature_name="feature_2",
        included_feature_names=["feature_1", "feature_2"],
        output_path="file/to/path",
        kms_key_id="kms-key-id",
    )
    assert dataset_builder._sagemaker_session == sagemaker_session_mock
    assert dataset_builder._base.equals(DATAFRAME)
    assert dataset_builder._record_identifier_feature_name == "feature_1"
    assert dataset_builder._event_time_identifier_feature_name == "feature_2"
    assert dataset_builder._included_feature_names == ["feature_1", "feature_2"]
    assert dataset_builder._output_path == "file/to/path"
    assert dataset_builder._kms_key_id == "kms-key-id"


def test_create_dataset_with_dataframe_value_error(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    with pytest.raises(ValueError) as error:
        feature_store.create_dataset(
            base=DATAFRAME,
            included_feature_names=["feature_1", "feature_2"],
            output_path="file/to/path",
            kms_key_id="kms-key-id",
        )
    assert (
        "You must provide a record identifier feature name and an event time identifier feature "
        + "name if specify DataFrame as base."
        in str(error)
    )


def test_list_feature_groups_with_no_filter(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    feature_store.list_feature_groups()
    sagemaker_session_mock.list_feature_groups.assert_called_with(
        name_contains=None,
        feature_group_status_equals=None,
        offline_store_status_equals=None,
        creation_time_after=None,
        creation_time_before=None,
        sort_order=None,
        sort_by=None,
        max_results=None,
        next_token=None,
    )


def test_list_feature_groups_with_all_filters(sagemaker_session_mock):
    feature_store = FeatureStore(sagemaker_session=sagemaker_session_mock)
    feature_store.list_feature_groups(
        name_contains="MyFeatureGroup",
        feature_group_status_equals="Created",
        offline_store_status_equals="Active",
        creation_time_after=datetime.datetime(2020, 12, 1),
        creation_time_before=datetime.datetime(2022, 7, 1),
        sort_order="Ascending",
        sort_by="Name",
        max_results=50,
        next_token="token",
    )
    sagemaker_session_mock.list_feature_groups.assert_called_with(
        name_contains="MyFeatureGroup",
        feature_group_status_equals="Created",
        offline_store_status_equals="Active",
        creation_time_after=datetime.datetime(2020, 12, 1),
        creation_time_before=datetime.datetime(2022, 7, 1),
        sort_order="Ascending",
        sort_by="Name",
        max_results=50,
        next_token="token",
    )

    try:
        manager.run(df)
    except Exception as e:
        assert "The config profile (non_exist) could not be found" in str(e)


@patch(
    "sagemaker.feature_store.feature_group.IngestionManagerPandas._ingest_single_batch",
    PicklableMock(return_value=[1]),
)
def test_ingestion_manager_run_multi_process_failure():
    df = pd.DataFrame({"float": pd.Series([2.0], dtype="float64")})
    manager = IngestionManagerPandas(
        feature_group_name="MyGroup",
        sagemaker_fs_runtime_client_config=None,
        max_workers=2,
        max_processes=2,
    )

    with pytest.raises(IngestionError) as error:
        manager.run(df)

    assert "Failed to ingest some data into FeatureGroup MyGroup" in str(error)
    assert error.value.failed_rows == [1, 1, 1, 1]
    assert manager.failed_rows == [1, 1, 1, 1]


@pytest.fixture
def query(sagemaker_session_mock):
    return AthenaQuery(
        catalog="catalog",
        database="database",
        table_name="table_name",
        sagemaker_session=sagemaker_session_mock,
    )


def test_athena_query_run(sagemaker_session_mock, query):
    WORKGROUP = "workgroup"
    sagemaker_session_mock.start_query_execution.return_value = {"QueryExecutionId": "query_id"}
    query.run(
        query_string="query",
        output_location="s3://some-bucket/some-path",
        workgroup=WORKGROUP,
    )
    sagemaker_session_mock.start_query_execution.assert_called_with(
        catalog="catalog",
        database="database",
        query_string="query",
        output_location="s3://some-bucket/some-path",
        kms_key=None,
        workgroup=WORKGROUP,
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
