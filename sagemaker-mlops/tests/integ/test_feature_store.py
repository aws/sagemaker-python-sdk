"""Integration tests for sagemaker.mlops.feature_store."""
import time
import pytest
import pandas as pd
import boto3

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.feature_store import (
    FeatureGroup,
    OfflineStoreConfig,
    OnlineStoreConfig,
    S3StorageConfig,
)
from sagemaker.mlops.feature_store.feature_utils import (
    load_feature_definitions_from_dataframe,
    ingest_dataframe,
    create_athena_query,
)
from sagemaker.mlops.feature_store.dataset_builder import DatasetBuilder
from sagemaker.core.utils import unique_name_from_base
from sagemaker.core.resources import FeatureGroup as CoreFeatureGroup


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def role():
    return get_execution_role()


@pytest.fixture(scope="module")
def region():
    return boto3.Session().region_name


@pytest.fixture(scope="module")
def bucket(sagemaker_session):
    return sagemaker_session.default_bucket()


@pytest.fixture
def feature_group_name():
    return unique_name_from_base("integ-test-fg")


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    current_time = int(time.time())
    return pd.DataFrame({
        "record_id": [f"id-{i}" for i in range(10)],
        "feature_1": [i * 1.5 for i in range(10)],
        "feature_2": [i * 2 for i in range(10)],
        "event_time": [float(current_time + i) for i in range(10)],
    })


def cleanup_feature_group(feature_group_name):
    """Helper to cleanup feature group."""
    try:
        fg = FeatureGroup.get(feature_group_name=feature_group_name)
        fg.delete()
        time.sleep(2)
    except Exception:
        pass


# Test 1: Create FeatureGroup with both online and offline stores
def test_create_feature_group_with_both_stores(
    feature_group_name, sample_dataframe, bucket, role, region
):
    """Test creating a FeatureGroup with both online and offline stores."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            online_store_config=OnlineStoreConfig(enable_online_store=True),
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        assert fg.feature_group_name == feature_group_name
        assert fg.online_store_config is not None
        assert fg.offline_store_config is not None
        
        time.sleep(5)
        
        retrieved_fg = FeatureGroup.get(feature_group_name=feature_group_name)
        assert retrieved_fg.feature_group_name == feature_group_name
        
    finally:
        cleanup_feature_group(feature_group_name)


# Test 2: Ingest DataFrame and retrieve from online store
def test_ingest_and_retrieve_from_online_store(
    feature_group_name, sample_dataframe, bucket, role
):
    """Test ingesting data and retrieving from online store."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            online_store_config=OnlineStoreConfig(enable_online_store=True),
        )
        
        # Wait for FeatureGroup to become active
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(15)
        
        record = fg.get_record(record_identifier_value_as_string="id-0")
        assert record is not None
        assert len(record.record) > 0
        
    finally:
        cleanup_feature_group(feature_group_name)


# Test 3: Delete FeatureGroup
def test_delete_feature_group(feature_group_name, sample_dataframe, bucket, role):
    """Test deleting a FeatureGroup."""
    feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
    
    fg = FeatureGroup.create(
        feature_group_name=feature_group_name,
        record_identifier_feature_name="record_id",
        event_time_feature_name="event_time",
        feature_definitions=feature_definitions,
        role_arn=role,
        online_store_config=OnlineStoreConfig(enable_online_store=True),
    )
    
    fg.wait_for_status("Created")
    
    fg.delete()
    time.sleep(2)
    
    with pytest.raises(Exception):
        FeatureGroup.get(feature_group_name=feature_group_name)


# Test 7: Ingest to both OnlineStore and OfflineStore
def test_ingest_to_both_stores(feature_group_name, sample_dataframe, bucket, role):
    """Test ingesting data to both online and offline stores."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            online_store_config=OnlineStoreConfig(enable_online_store=True),
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        # Wait for FeatureGroup to become active
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(15)
        
        record = fg.get_record(record_identifier_value_as_string="id-0")
        assert record is not None
        
    finally:
        cleanup_feature_group(feature_group_name)


# Test 8: Query offline store with Athena and return DataFrame
def test_query_offline_store_with_athena(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test querying offline store with Athena."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        # Note: Offline store sync can take 15+ minutes, test may return empty results
        athena_query = create_athena_query(feature_group_name, sagemaker_session)
        query_string = f'SELECT * FROM "{athena_query.database}"."{athena_query.table_name}" LIMIT 10'
        output_location = f"s3://{bucket}/athena-results/"
        
        query_id = athena_query.run(query_string, output_location)
        assert query_id is not None
        
        athena_query.wait()
        df = athena_query.as_dataframe()
        
        assert df is not None
        
    finally:
        cleanup_feature_group(feature_group_name)


# Test 9: Query with WHERE conditions and aggregations
def test_query_with_conditions_and_aggregations(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test Athena queries with WHERE and aggregations."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        athena_query = create_athena_query(feature_group_name, sagemaker_session)
        query_string = f"""
            SELECT COUNT(*) as count, AVG(feature_1) as avg_feature
            FROM "{athena_query.database}"."{athena_query.table_name}"
            WHERE feature_2 > 5
        """
        output_location = f"s3://{bucket}/athena-results/"
        
        athena_query.run(query_string, output_location)
        athena_query.wait()
        df = athena_query.as_dataframe()
        
        assert df is not None
        
    finally:
        cleanup_feature_group(feature_group_name)



# Test 11: Create dataset from single FeatureGroup
def test_create_dataset_from_single_feature_group(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test creating a dataset from a single FeatureGroup."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        output_path = f"s3://{bucket}/dataset-output/"
        builder = DatasetBuilder.create(
            base=fg,
            output_path=output_path,
            session=sagemaker_session,
        )
        
        df, query = builder.to_dataframe()
        
        assert df is not None
        assert query is not None
        assert "SELECT" in query
        
    finally:
        cleanup_feature_group(feature_group_name)


# Test 15: Export dataset with deleted/duplicated records handling
def test_export_dataset_with_record_handling(
    feature_group_name, sample_dataframe, bucket, role, sagemaker_session
):
    """Test exporting dataset with options for deleted and duplicated records."""
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        
        fg = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
            ),
        )
        
        fg.wait_for_status("Created")
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=sample_dataframe,
            max_workers=1,
            max_processes=1,
        )
        
        updated_df = sample_dataframe.copy()
        updated_df["feature_1"] = updated_df["feature_1"] * 2
        updated_df["event_time"] = updated_df["event_time"] + 100
        
        ingest_dataframe(
            feature_group_name=feature_group_name,
            data_frame=updated_df,
            max_workers=1,
            max_processes=1,
        )
        
        time.sleep(300)
        
        output_path = f"s3://{bucket}/dataset-output/"
        
        builder = DatasetBuilder.create(
            base=fg,
            output_path=output_path,
            session=sagemaker_session,
        )
        builder.include_duplicated_records()
        
        df_with_dups, _ = builder.to_dataframe()
        assert df_with_dups is not None
        
        builder2 = DatasetBuilder.create(
            base=fg,
            output_path=output_path,
            session=sagemaker_session,
        )
        builder2.with_number_of_recent_records_by_record_identifier(1)
        
        df_recent, _ = builder2.to_dataframe()
        assert df_recent is not None
        
    finally:
        cleanup_feature_group(feature_group_name)
