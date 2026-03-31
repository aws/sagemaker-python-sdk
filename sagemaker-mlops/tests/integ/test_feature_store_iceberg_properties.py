"""Integration tests for FeatureGroupManager iceberg property handling."""
import time

import boto3
import pandas as pd
import pytest

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core.utils import unique_name_from_base
from sagemaker.mlops.feature_store import (
    OfflineStoreConfig,
    S3StorageConfig,
)
from sagemaker.mlops.feature_store import FeatureGroupManager
from sagemaker.mlops.feature_store.feature_group_manager import IcebergProperties
from sagemaker.mlops.feature_store.feature_utils import (
    load_feature_definitions_from_dataframe,
)


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
    return unique_name_from_base("integ-test-iceberg-fg")


@pytest.fixture
def sample_dataframe():
    current_time = int(time.time())
    return pd.DataFrame({
        "record_id": [f"id-{i}" for i in range(10)],
        "feature_1": [i * 1.5 for i in range(10)],
        "feature_2": [i * 2 for i in range(10)],
        "event_time": [float(current_time + i) for i in range(10)],
    })


def cleanup_feature_group(feature_group_name):
    try:
        fg = FeatureGroupManager.get(feature_group_name=feature_group_name)
        fg.delete()
        time.sleep(2)
    except Exception:
        pass


def test_create_with_iceberg_properties(
    feature_group_name, sample_dataframe, bucket, role
):
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)
        iceberg_props = IcebergProperties(properties={
            "write.target-file-size-bytes": "536870912",
            "history.expire.max-snapshot-age-ms": "432000000",
        })

        fg = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
                table_format="Iceberg",
            ),
            iceberg_properties=iceberg_props,
        )

        fg.wait_for_status("Created")

        retrieved = FeatureGroupManager.get(
            feature_group_name=feature_group_name,
            include_iceberg_properties=True,
        )
        assert retrieved.iceberg_properties is not None
        assert retrieved.iceberg_properties.properties["write.target-file-size-bytes"] == "536870912"
        assert retrieved.iceberg_properties.properties["history.expire.max-snapshot-age-ms"] == "432000000"
    finally:
        cleanup_feature_group(feature_group_name)


def test_update_iceberg_properties(
    feature_group_name, sample_dataframe, bucket, role
):
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)

        fg = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
                table_format="Iceberg",
            ),
            iceberg_properties=IcebergProperties(properties={
                "write.target-file-size-bytes": "536870912",
            }),
        )

        fg.wait_for_status("Created")

        fg.update(iceberg_properties=IcebergProperties(properties={
            "write.target-file-size-bytes": "268435456",
            "history.expire.min-snapshots-to-keep": "5",
        }))

        retrieved = FeatureGroupManager.get(
            feature_group_name=feature_group_name,
            include_iceberg_properties=True,
        )
        assert retrieved.iceberg_properties.properties["write.target-file-size-bytes"] == "268435456"
        assert retrieved.iceberg_properties.properties["history.expire.min-snapshots-to-keep"] == "5"
    finally:
        cleanup_feature_group(feature_group_name)


def test_get_with_include_iceberg_properties(
    feature_group_name, sample_dataframe, bucket, role
):
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)

        fg = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
                table_format="Iceberg",
            ),
            iceberg_properties=IcebergProperties(properties={
                "write.metadata.delete-after-commit.enabled": "true",
            }),
        )

        fg.wait_for_status("Created")

        retrieved = FeatureGroupManager.get(
            feature_group_name=feature_group_name,
            include_iceberg_properties=True,
        )
        assert retrieved.iceberg_properties is not None
        assert isinstance(retrieved.iceberg_properties.properties, dict)
        assert retrieved.iceberg_properties.properties["write.metadata.delete-after-commit.enabled"] == "true"
    finally:
        cleanup_feature_group(feature_group_name)


def test_create_with_iceberg_properties_none(
    feature_group_name, sample_dataframe, bucket, role
):
    try:
        feature_definitions = load_feature_definitions_from_dataframe(sample_dataframe)

        fg = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri=f"s3://{bucket}/feature-store"),
                table_format="Iceberg",
            ),
            iceberg_properties=None,
        )

        fg.wait_for_status("Created")

        assert fg.iceberg_properties is None
    finally:
        cleanup_feature_group(feature_group_name)


def test_create_iceberg_properties_without_offline_store_raises():
    with pytest.raises(ValueError, match="iceberg_properties requires offline_store_config"):
        FeatureGroupManager.create(
            feature_group_name="dummy-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[],
            role_arn="arn:aws:iam::000000000000:role/dummy",
            iceberg_properties=IcebergProperties(properties={
                "write.target-file-size-bytes": "536870912",
            }),
        )


def test_create_iceberg_properties_with_non_iceberg_table_format_raises():
    with pytest.raises(ValueError, match="table_format to be 'Iceberg'"):
        FeatureGroupManager.create(
            feature_group_name="dummy-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[],
            role_arn="arn:aws:iam::000000000000:role/dummy",
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/prefix"),
                table_format="Glue",
            ),
            iceberg_properties=IcebergProperties(properties={
                "write.target-file-size-bytes": "536870912",
            }),
        )
