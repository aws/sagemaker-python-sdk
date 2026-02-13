# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for athena_query.py"""
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from sagemaker.mlops.feature_store.athena_query import AthenaQuery


class TestAthenaQuery:
    @pytest.fixture
    def mock_session(self):
        session = Mock()
        session.boto_session.client.return_value = Mock()
        session.boto_region_name = "us-west-2"
        return session

    @pytest.fixture
    def athena_query(self, mock_session):
        return AthenaQuery(
            catalog="AwsDataCatalog",
            database="sagemaker_featurestore",
            table_name="my_feature_group",
            sagemaker_session=mock_session,
        )

    def test_initialization(self, athena_query):
        assert athena_query.catalog == "AwsDataCatalog"
        assert athena_query.database == "sagemaker_featurestore"
        assert athena_query.table_name == "my_feature_group"
        assert athena_query._current_query_execution_id is None

    @patch("sagemaker.mlops.feature_store.athena_query.start_query_execution")
    def test_run_starts_query(self, mock_start, athena_query):
        mock_start.return_value = {"QueryExecutionId": "query-123"}

        result = athena_query.run(
            query_string="SELECT * FROM table",
            output_location="s3://bucket/output",
        )

        assert result == "query-123"
        assert athena_query._current_query_execution_id == "query-123"
        assert athena_query._result_bucket == "bucket"
        assert athena_query._result_file_prefix == "output"

    @patch("sagemaker.mlops.feature_store.athena_query.start_query_execution")
    def test_run_with_kms_key(self, mock_start, athena_query):
        mock_start.return_value = {"QueryExecutionId": "query-123"}

        athena_query.run(
            query_string="SELECT * FROM table",
            output_location="s3://bucket/output",
            kms_key="arn:aws:kms:us-west-2:123:key/abc",
        )

        mock_start.assert_called_once()
        call_kwargs = mock_start.call_args[1]
        assert call_kwargs["kms_key"] == "arn:aws:kms:us-west-2:123:key/abc"

    @patch("sagemaker.mlops.feature_store.athena_query.wait_for_athena_query")
    def test_wait_calls_helper(self, mock_wait, athena_query):
        athena_query._current_query_execution_id = "query-123"

        athena_query.wait()

        mock_wait.assert_called_once_with(athena_query.sagemaker_session, "query-123")

    @patch("sagemaker.mlops.feature_store.athena_query.get_query_execution")
    def test_get_query_execution(self, mock_get, athena_query):
        athena_query._current_query_execution_id = "query-123"
        mock_get.return_value = {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}

        result = athena_query.get_query_execution()

        assert result["QueryExecution"]["Status"]["State"] == "SUCCEEDED"

    @patch("sagemaker.mlops.feature_store.athena_query.get_query_execution")
    @patch("sagemaker.mlops.feature_store.athena_query.download_athena_query_result")
    @patch("pandas.read_csv")
    @patch("os.path.join")
    def test_as_dataframe_success(self, mock_join, mock_read_csv, mock_download, mock_get, athena_query):
        athena_query._current_query_execution_id = "query-123"
        athena_query._result_bucket = "bucket"
        athena_query._result_file_prefix = "prefix"

        mock_get.return_value = {"QueryExecution": {"Status": {"State": "SUCCEEDED"}}}
        mock_join.return_value = "/tmp/query-123.csv"
        mock_read_csv.return_value = pd.DataFrame({"col": [1, 2, 3]})

        with patch("tempfile.gettempdir", return_value="/tmp"):
            with patch("os.remove"):
                df = athena_query.as_dataframe()

        assert len(df) == 3

    @patch("sagemaker.mlops.feature_store.athena_query.get_query_execution")
    def test_as_dataframe_raises_when_running(self, mock_get, athena_query):
        athena_query._current_query_execution_id = "query-123"
        mock_get.return_value = {"QueryExecution": {"Status": {"State": "RUNNING"}}}

        with pytest.raises(RuntimeError, match="still executing"):
            athena_query.as_dataframe()

    @patch("sagemaker.mlops.feature_store.athena_query.get_query_execution")
    def test_as_dataframe_raises_when_failed(self, mock_get, athena_query):
        athena_query._current_query_execution_id = "query-123"
        mock_get.return_value = {"QueryExecution": {"Status": {"State": "FAILED"}}}

        with pytest.raises(RuntimeError, match="failed"):
            athena_query.as_dataframe()
