# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Conftest for feature_store tests."""
import pytest
from unittest.mock import Mock, MagicMock
import pandas as pd
import numpy as np


@pytest.fixture
def mock_session():
    """Create a mock Session."""
    session = Mock()
    session.boto_session = Mock()
    session.boto_region_name = "us-west-2"
    session.sagemaker_client = Mock()
    session.sagemaker_runtime_client = Mock()
    session.sagemaker_featurestore_runtime_client = Mock()
    return session


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "id": pd.Series([1, 2, 3, 4, 5], dtype="int64"),
        "value": pd.Series([1.1, 2.2, 3.3, 4.4, 5.5], dtype="float64"),
        "name": pd.Series(["a", "b", "c", "d", "e"], dtype="string"),
        "event_time": pd.Series(
            ["2024-01-01T00:00:00Z"] * 5,
            dtype="string"
        ),
    })


@pytest.fixture
def dataframe_with_collections():
    """Create a DataFrame with collection type columns."""
    return pd.DataFrame({
        "id": pd.Series([1, 2, 3], dtype="int64"),
        "tags": pd.Series([["a", "b"], ["c"], ["d", "e", "f"]], dtype="object"),
        "scores": pd.Series([[1.0, 2.0], [3.0], [4.0, 5.0]], dtype="object"),
        "event_time": pd.Series(["2024-01-01"] * 3, dtype="string"),
    })


@pytest.fixture
def feature_definitions_dict():
    """Create a feature definitions dictionary."""
    return {
        "id": {"FeatureName": "id", "FeatureType": "Integral"},
        "value": {"FeatureName": "value", "FeatureType": "Fractional"},
        "name": {"FeatureName": "name", "FeatureType": "String"},
        "event_time": {"FeatureName": "event_time", "FeatureType": "String"},
    }


@pytest.fixture
def mock_feature_group():
    """Create a mock FeatureGroup from core."""
    fg = MagicMock()
    fg.feature_group_name = "test-feature-group"
    fg.record_identifier_feature_name = "id"
    fg.event_time_feature_name = "event_time"
    fg.feature_definitions = [
        MagicMock(feature_name="id", feature_type="Integral"),
        MagicMock(feature_name="value", feature_type="Fractional"),
        MagicMock(feature_name="name", feature_type="String"),
        MagicMock(feature_name="event_time", feature_type="String"),
    ]
    fg.offline_store_config = MagicMock()
    fg.offline_store_config.s3_storage_config.s3_uri = "s3://bucket/prefix"
    fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = "s3://bucket/prefix/resolved"
    fg.offline_store_config.data_catalog_config.catalog = "AwsDataCatalog"
    fg.offline_store_config.data_catalog_config.database = "sagemaker_featurestore"
    fg.offline_store_config.data_catalog_config.table_name = "test_feature_group"
    fg.offline_store_config.data_catalog_config.disable_glue_table_creation = False
    fg.online_store_config = MagicMock()
    fg.online_store_config.enable_online_store = True
    return fg
