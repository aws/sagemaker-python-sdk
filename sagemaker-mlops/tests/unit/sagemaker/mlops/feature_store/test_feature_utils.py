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


class TestGetFeatureGroupAsDataframe:
    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.get_session_from_role")
    def test_with_session_provided(self, mock_get_session, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1, 2]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        result = get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            session=mock_session,
            latest_ingestion=False,
            verbose=False,
        )

        mock_fg_class.assert_called_once_with(name="test-fg", sagemaker_session=mock_session)
        mock_get_session.assert_not_called()
        mock_athena_query.run.assert_called_once()
        mock_athena_query.wait.assert_called_once()
        assert len(result) == 2

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.get_session_from_role")
    def test_with_region_provided(self, mock_get_session, mock_fg_class):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        result = get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            region="us-east-1",
            latest_ingestion=False,
        )

        mock_get_session.assert_called_once_with(region="us-east-1", assume_role=None)
        assert len(result) == 1

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.get_session_from_role")
    def test_with_region_and_role(self, mock_get_session, mock_fg_class):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        result = get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            region="us-east-1",
            role="arn:aws:iam::123:role/MyRole",
            latest_ingestion=False,
        )

        mock_get_session.assert_called_once_with(region="us-east-1", assume_role="arn:aws:iam::123:role/MyRole")

    def test_raises_when_no_session_or_region(self):
        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        with pytest.raises(Exception, match="Session or role and region must be specified"):
            get_feature_group_as_dataframe(
                feature_group_name="test-fg",
                athena_bucket="s3://bucket/path",
                latest_ingestion=False,
            )

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_raises_when_latest_ingestion_without_event_time(self, mock_fg_class):
        mock_session = MagicMock()

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        with pytest.raises(Exception, match="event_time_feature_name must be specified"):
            get_feature_group_as_dataframe(
                feature_group_name="test-fg",
                athena_bucket="s3://bucket/path",
                session=mock_session,
                latest_ingestion=True,
                event_time_feature_name=None,
            )

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_with_latest_ingestion_and_event_time(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1, 2], "event_time": [123, 123]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        result = get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            session=mock_session,
            latest_ingestion=True,
            event_time_feature_name="event_time",
            verbose=False,
        )

        call_args = mock_athena_query.run.call_args
        query_string = call_args[1]["query_string"]
        assert "event_time=(SELECT MAX(event_time)" in query_string
        assert len(result) == 2

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_custom_query_with_table_placeholder(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "actual_table_name"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        result = get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            session=mock_session,
            query='SELECT * FROM "sagemaker_featurestore"."#{table}" WHERE id > 0 ',
            latest_ingestion=False,
        )

        call_args = mock_athena_query.run.call_args
        query_string = call_args[1]["query_string"]
        assert "actual_table_name" in query_string
        assert "#{table}" not in query_string
        assert query_string.endswith(";")

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_verbose_logging(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe
        import logging

        with patch("sagemaker.mlops.feature_store.feature_utils.logger") as mock_logger:
            get_feature_group_as_dataframe(
                feature_group_name="test-fg",
                athena_bucket="s3://bucket/path",
                session=mock_session,
                latest_ingestion=False,
                verbose=True,
            )

            mock_logger.setLevel.assert_called_with(logging.INFO)
            assert mock_logger.info.call_count >= 1

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_silent_mode(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe
        import logging

        with patch("sagemaker.mlops.feature_store.feature_utils.logger") as mock_logger:
            get_feature_group_as_dataframe(
                feature_group_name="test-fg",
                athena_bucket="s3://bucket/path",
                session=mock_session,
                latest_ingestion=False,
                verbose=False,
            )

            mock_logger.setLevel.assert_called_with(logging.WARNING)

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_passes_kwargs_to_as_dataframe(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_athena_query = MagicMock()
        mock_athena_query.table_name = "my_table"
        mock_athena_query.as_dataframe.return_value = pd.DataFrame({"id": [1]})
        mock_fg.athena_query.return_value = mock_athena_query
        mock_fg_class.return_value = mock_fg

        from sagemaker.mlops.feature_store.feature_utils import get_feature_group_as_dataframe

        get_feature_group_as_dataframe(
            feature_group_name="test-fg",
            athena_bucket="s3://bucket/path",
            session=mock_session,
            latest_ingestion=False,
            dtype={"id": "int32"},
            na_values=["NA"],
        )

        mock_athena_query.as_dataframe.assert_called_once_with(dtype={"id": "int32"}, na_values=["NA"])


class TestPrepareFgFromDataframeOrFile:
    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_with_dataframe_and_session(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "value": [1.1, 2.2, 3.3],
        })
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        result = prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
            verbose=False,
        )
        
        mock_fg_class.assert_called_once()
        assert result == mock_fg
        assert "record_id" in df.columns
        assert "data_as_of_date" in df.columns

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.read_csv")
    def test_with_file_path(self, mock_read_csv, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2], "value": [1.1, 2.2]})
        mock_read_csv.return_value = df
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        result = prepare_fg_from_dataframe_or_file(
            dataframe_or_path="/path/to/file.csv",
            feature_group_name="test-fg",
            session=mock_session,
        )
        
        mock_read_csv.assert_called_once()
        assert result == mock_fg

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.get_session_from_role")
    def test_with_region_and_role(self, mock_get_session, mock_fg_class):
        mock_session = MagicMock()
        mock_get_session.return_value = mock_session
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            region="us-east-1",
            role="arn:aws:iam::123:role/MyRole",
        )
        
        mock_get_session.assert_called_once_with(region="us-east-1")

    def test_raises_on_invalid_type(self):
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        with pytest.raises(Exception, match="Invalid type"):
            prepare_fg_from_dataframe_or_file(
                dataframe_or_path=123,
                feature_group_name="test-fg",
                session=MagicMock(),
            )

    def test_raises_when_no_session_or_region(self):
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        df = pd.DataFrame({"id": [1, 2]})
        
        with pytest.raises(Exception, match="Session or role and region must be specified"):
            prepare_fg_from_dataframe_or_file(
                dataframe_or_path=df,
                feature_group_name="test-fg",
            )

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_creates_record_id_from_index(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"value": [1.1, 2.2, 3.3]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
        )
        
        assert "record_id" in df.columns
        assert list(df["record_id"]) == [0, 1, 2]

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_uses_existing_record_id(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"my_id": [10, 20, 30], "value": [1.1, 2.2, 3.3]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
            record_id="my_id",
        )
        
        assert "my_id" in df.columns
        assert "record_id" not in df.columns

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_raises_on_duplicate_record_ids(self, mock_fg_class):
        mock_session = MagicMock()
        
        df = pd.DataFrame({"my_id": [1, 1, 2], "value": [1.1, 2.2, 3.3]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        with pytest.raises(Exception, match="duplicated rows"):
            prepare_fg_from_dataframe_or_file(
                dataframe_or_path=df,
                feature_group_name="test-fg",
                session=mock_session,
                record_id="my_id",
            )

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.time")
    def test_creates_event_id_with_timestamp(self, mock_time, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        mock_time.time.return_value = 1234567890.5
        
        df = pd.DataFrame({"id": [1, 2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
        )
        
        assert "data_as_of_date" in df.columns
        assert all(df["data_as_of_date"] == 1234567891)

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_uses_existing_event_id(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2], "timestamp": [100, 200]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
            event_id="timestamp",
        )
        
        assert "timestamp" in df.columns
        assert "data_as_of_date" not in df.columns

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_formats_column_names(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"My Column": [1, 2], "Value.Test": [1.1, 2.2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
        )
        
        assert "my_column" in df.columns
        assert "valuetest" in df.columns

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_verbose_logging(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        import logging
        
        with patch("sagemaker.mlops.feature_store.feature_utils.logger") as mock_logger:
            prepare_fg_from_dataframe_or_file(
                dataframe_or_path=df,
                feature_group_name="test-fg",
                session=mock_session,
                verbose=True,
            )
            
            mock_logger.setLevel.assert_called_with(logging.INFO)

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_silent_mode(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        import logging
        
        with patch("sagemaker.mlops.feature_store.feature_utils.logger") as mock_logger:
            prepare_fg_from_dataframe_or_file(
                dataframe_or_path=df,
                feature_group_name="test-fg",
                session=mock_session,
                verbose=False,
            )
            
            mock_logger.setLevel.assert_called_with(logging.WARNING)

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    @patch("sagemaker.mlops.feature_store.feature_utils.read_csv")
    def test_passes_kwargs_to_read_csv(self, mock_read_csv, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2]})
        mock_read_csv.return_value = df
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path="/path/to/file.csv",
            feature_group_name="test-fg",
            session=mock_session,
            sep=";",
            encoding="utf-8",
        )
        
        mock_read_csv.assert_called_once()
        call_kwargs = mock_read_csv.call_args[1]
        assert call_kwargs["sep"] == ";"
        assert call_kwargs["encoding"] == "utf-8"

    @patch("sagemaker.mlops.feature_store.feature_utils.FeatureGroup")
    def test_calls_load_feature_definitions(self, mock_fg_class):
        mock_session = MagicMock()
        mock_fg = MagicMock()
        mock_fg_class.return_value = mock_fg
        
        df = pd.DataFrame({"id": [1, 2]})
        
        from sagemaker.mlops.feature_store.feature_utils import prepare_fg_from_dataframe_or_file
        
        prepare_fg_from_dataframe_or_file(
            dataframe_or_path=df,
            feature_group_name="test-fg",
            session=mock_session,
        )
        
        mock_fg.load_feature_definitions.assert_called_once()
