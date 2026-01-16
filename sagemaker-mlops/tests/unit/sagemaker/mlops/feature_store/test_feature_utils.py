# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for feature_utils.py"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import numpy as np

from sagemaker.mlops.feature_store.feature_utils import (
    load_feature_definitions_from_dataframe,
    as_hive_ddl,
    create_athena_query,
    ingest_dataframe,
    get_session_from_role,
    _is_collection_column,
    _generate_feature_definition,
)
from sagemaker.mlops.feature_store.feature_definition import (
    FeatureDefinition,
    ListCollectionType,
)


class TestLoadFeatureDefinitionsFromDataframe:
    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            "id": pd.Series([1, 2, 3], dtype="int64"),
            "value": pd.Series([1.1, 2.2, 3.3], dtype="float64"),
            "name": pd.Series(["a", "b", "c"], dtype="string"),
        })

    def test_infers_integral_type(self, sample_dataframe):
        defs = load_feature_definitions_from_dataframe(sample_dataframe)
        id_def = next(d for d in defs if d.feature_name == "id")
        assert id_def.feature_type == "Integral"

    def test_infers_fractional_type(self, sample_dataframe):
        defs = load_feature_definitions_from_dataframe(sample_dataframe)
        value_def = next(d for d in defs if d.feature_name == "value")
        assert value_def.feature_type == "Fractional"

    def test_infers_string_type(self, sample_dataframe):
        defs = load_feature_definitions_from_dataframe(sample_dataframe)
        name_def = next(d for d in defs if d.feature_name == "name")
        assert name_def.feature_type == "String"

    def test_returns_correct_count(self, sample_dataframe):
        defs = load_feature_definitions_from_dataframe(sample_dataframe)
        assert len(defs) == 3

    def test_collection_type_with_in_memory_storage(self):
        df = pd.DataFrame({
            "id": pd.Series([1, 2], dtype="int64"),
            "tags": pd.Series([["a", "b"], ["c"]], dtype="object"),
        })
        defs = load_feature_definitions_from_dataframe(df, online_storage_type="InMemory")
        tags_def = next(d for d in defs if d.feature_name == "tags")
        assert tags_def.collection_type == "List"


class TestIsCollectionColumn:
    def test_list_column_returns_true(self):
        series = pd.Series([[1, 2], [3, 4], [5]])
        assert _is_collection_column(series) == True

    def test_scalar_column_returns_false(self):
        series = pd.Series([1, 2, 3])
        assert _is_collection_column(series) == False

    def test_empty_series(self):
        series = pd.Series([], dtype="object")
        assert _is_collection_column(series) == False


class TestAsHiveDdl:
    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_generates_ddl_string(self, mock_fg_class):
        # Setup mock
        mock_fg = MagicMock()
        mock_fg.feature_definitions = [
            MagicMock(feature_name="id", feature_type="Integral"),
            MagicMock(feature_name="value", feature_type="Fractional"),
            MagicMock(feature_name="name", feature_type="String"),
        ]
        mock_fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = "s3://bucket/prefix"
        mock_fg_class.get.return_value = mock_fg

        ddl = as_hive_ddl("my-feature-group")

        assert "CREATE EXTERNAL TABLE" in ddl
        assert "my-feature-group" in ddl
        assert "id INT" in ddl
        assert "value FLOAT" in ddl
        assert "name STRING" in ddl
        assert "write_time TIMESTAMP" in ddl
        assert "event_time TIMESTAMP" in ddl
        assert "is_deleted BOOLEAN" in ddl
        assert "s3://bucket/prefix" in ddl

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_custom_database_and_table(self, mock_fg_class):
        mock_fg = MagicMock()
        mock_fg.feature_definitions = []
        mock_fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = "s3://bucket/prefix"
        mock_fg_class.get.return_value = mock_fg

        ddl = as_hive_ddl("my-fg", database="custom_db", table_name="custom_table")

        assert "custom_db.custom_table" in ddl


class TestCreateAthenaQuery:
    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_creates_athena_query(self, mock_fg_class):
        mock_fg = MagicMock()
        mock_fg.offline_store_config.data_catalog_config.catalog = "MyCatalog"
        mock_fg.offline_store_config.data_catalog_config.database = "MyDatabase"
        mock_fg.offline_store_config.data_catalog_config.table_name = "MyTable"
        mock_fg.offline_store_config.data_catalog_config.disable_glue_table_creation = False
        mock_fg_class.get.return_value = mock_fg

        session = Mock()
        query = create_athena_query("my-fg", session)

        assert query.catalog == "AwsDataCatalog"  # disable_glue=False uses default
        assert query.database == "MyDatabase"
        assert query.table_name == "MyTable"

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_raises_when_no_metastore(self, mock_fg_class):
        mock_fg = MagicMock()
        mock_fg.offline_store_config = None
        mock_fg_class.get.return_value = mock_fg

        session = Mock()
        with pytest.raises(RuntimeError, match="No metastore"):
            create_athena_query("my-fg", session)


class TestIngestDataframe:
    @patch("sagemaker.mlops.feature_store.feature_utils.IngestionManagerPandas")
    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_creates_manager_and_runs(self, mock_fg_class, mock_manager_class):
        mock_fg = MagicMock()
        mock_fg.feature_definitions = [
            MagicMock(feature_name="id", feature_type="Integral"),
        ]
        mock_fg_class.get.return_value = mock_fg

        mock_manager = MagicMock()
        mock_manager_class.return_value = mock_manager

        df = pd.DataFrame({"id": [1, 2, 3]})
        result = ingest_dataframe("my-fg", df, max_workers=2, max_processes=1)

        mock_manager_class.assert_called_once()
        mock_manager.run.assert_called_once()
        assert result == mock_manager

    def test_raises_on_invalid_max_workers(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="max_workers"):
            ingest_dataframe("my-fg", df, max_workers=0)

    def test_raises_on_invalid_max_processes(self):
        df = pd.DataFrame({"id": [1, 2, 3]})
        with pytest.raises(ValueError, match="max_processes"):
            ingest_dataframe("my-fg", df, max_processes=-1)


class TestGetSessionFromRole:
    @patch("sagemaker.mlops.feature_store.feature_utils.boto3")
    @patch("sagemaker.mlops.feature_store.feature_utils.Session")
    def test_creates_session_without_role(self, mock_session_class, mock_boto3):
        mock_boto_session = MagicMock()
        mock_boto3.Session.return_value = mock_boto_session

        get_session_from_role(region="us-west-2")

        mock_boto3.Session.assert_called_with(region_name="us-west-2")
        mock_session_class.assert_called_once()

    @patch("sagemaker.mlops.feature_store.feature_utils.boto3")
    @patch("sagemaker.mlops.feature_store.feature_utils.Session")
    def test_assumes_role_when_provided(self, mock_session_class, mock_boto3):
        mock_boto_session = MagicMock()
        mock_sts = MagicMock()
        mock_sts.assume_role.return_value = {
            "Credentials": {
                "AccessKeyId": "key",
                "SecretAccessKey": "secret",
                "SessionToken": "token",
            }
        }
        mock_boto_session.client.return_value = mock_sts
        mock_boto3.Session.return_value = mock_boto_session

        get_session_from_role(region="us-west-2", assume_role="arn:aws:iam::123:role/MyRole")

        mock_sts.assume_role.assert_called_once()
        assert mock_boto3.Session.call_count == 2  # Initial + after assume
