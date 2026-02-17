# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Tests for telemetry integration in feature store interfaces."""

import pytest
from unittest.mock import Mock, patch
import pandas as pd

from sagemaker.core.telemetry import Feature
from sagemaker.mlops.feature_store.dataset_builder import DatasetBuilder
from sagemaker.mlops.feature_store.ingestion_manager_pandas import IngestionManagerPandas
from sagemaker.mlops.feature_store.athena_query import AthenaQuery


class TestFeatureStoreTelemetry:
    """Test telemetry integration for feature store interfaces."""

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    def test_dataset_builder_telemetry(self, mock_send_telemetry):
        """Test that DatasetBuilder methods emit telemetry."""
        mock_session = Mock()
        mock_session.sagemaker_config = None
        mock_session.local_mode = False
        
        # Create a simple DataFrame
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})
        
        builder = DatasetBuilder.create(
            base=df,
            output_path="s3://test-bucket/output",
            session=mock_session,
            record_identifier_feature_name="feature1",
            event_time_identifier_feature_name="feature2"
        )
        
        # Mock the internal methods to avoid actual S3/Athena calls
        with patch.object(builder, '_to_csv_from_dataframe', return_value=("s3://test/file.csv", "SELECT * FROM test")):
            result = builder.to_csv_file()
            
        # Verify telemetry was called
        assert mock_send_telemetry.called
        # Verify the feature code for FEATURE_STORE was used
        call_args = mock_send_telemetry.call_args[0]
        assert 17 in call_args[1]  # FEATURE_STORE code

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    def test_ingestion_manager_telemetry(self, mock_send_telemetry):
        """Test that IngestionManagerPandas.run emits telemetry."""
        mock_session = Mock()
        mock_session.sagemaker_config = None
        mock_session.local_mode = False
        
        feature_definitions = {
            "feature1": {"FeatureType": "Integral", "CollectionType": None},
            "feature2": {"FeatureType": "String", "CollectionType": None}
        }
        
        manager = IngestionManagerPandas(
            feature_group_name="test-fg",
            feature_definitions=feature_definitions,
            max_workers=1,
            max_processes=1
        )
        
        # Add session to manager for telemetry
        manager.sagemaker_session = mock_session
        
        df = pd.DataFrame({"feature1": [1, 2, 3], "feature2": ["a", "b", "c"]})
        
        # Mock the internal methods to avoid actual ingestion
        with patch.object(manager, '_run_single_process_single_thread'):
            manager.run(df, wait=True)
            
        # Verify telemetry was called
        assert mock_send_telemetry.called
        # Verify the feature code for FEATURE_STORE was used
        call_args = mock_send_telemetry.call_args[0]
        assert 17 in call_args[1]  # FEATURE_STORE code

    @patch("sagemaker.core.telemetry.telemetry_logging._send_telemetry_request")
    def test_athena_query_telemetry(self, mock_send_telemetry):
        """Test that AthenaQuery methods emit telemetry."""
        mock_session = Mock()
        mock_session.sagemaker_config = None
        mock_session.local_mode = False
        
        query = AthenaQuery(
            catalog="test_catalog",
            database="test_db", 
            table_name="test_table",
            sagemaker_session=mock_session
        )
        
        # Mock the internal methods to avoid actual Athena calls
        with patch("sagemaker.mlops.feature_store.feature_utils.start_query_execution", 
                   return_value={"QueryExecutionId": "test-id"}):
            query.run("SELECT * FROM test", "s3://test-bucket/output")
            
        # Verify telemetry was called
        assert mock_send_telemetry.called
        # Verify the feature code for FEATURE_STORE was used
        call_args = mock_send_telemetry.call_args[0]
        assert 17 in call_args[1]  # FEATURE_STORE code