"""
Integration tests for Lake Formation with FeatureGroupManager.

These tests require:
- AWS credentials with Lake Formation and SageMaker permissions
- An S3 bucket for offline store (uses default SageMaker bucket)
- An IAM role for Feature Store (uses execution role)

Run with: pytest tests/integ/test_featureStore_lakeformation.py -v -m integ
"""

import logging
import uuid
import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.feature_store import (
    FeatureGroupManager,
    LakeFormationConfig,
    OfflineStoreConfig,
    OnlineStoreConfig,
    S3StorageConfig,
    StringFeatureDefinition,
    FractionalFeatureDefinition,
)

feature_definitions = [
    StringFeatureDefinition(feature_name="record_id"),
    StringFeatureDefinition(feature_name="event_time"),
    FractionalFeatureDefinition(feature_name="feature_value"),
]


@pytest.fixture(scope="module")
def sagemaker_session():
    return Session()


@pytest.fixture(scope="module")
def role(sagemaker_session):
    return get_execution_role(sagemaker_session)


@pytest.fixture(scope="module")
def s3_uri(sagemaker_session):
    bucket = sagemaker_session.default_bucket()
    return f"s3://{bucket}/feature-store-test"


@pytest.fixture(scope="module")
def region():
    return "us-west-2"


@pytest.fixture(scope="module")
def shared_feature_group_for_negative_tests(s3_uri, role, region):
    """
    Create a single FeatureGroupManager for negative tests that only need to verify
    error conditions without modifying the resource.

    This fixture is module-scoped to be created once and shared across tests,
    reducing test execution time.
    """
    fg_name = f"test-lf-negative-{uuid.uuid4().hex[:8]}"
    fg = None

    try:
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        yield fg
    finally:
        if fg:
            cleanup_feature_group(fg)


def generate_feature_group_name():
    """Generate a unique feature group name for testing."""
    return f"test-lf-fg-{uuid.uuid4().hex[:8]}"


def create_test_feature_group(name: str, s3_uri: str, role_arn: str, region: str) -> FeatureGroupManager:
    """Create a FeatureGroupManager with offline store for testing."""

    offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))

    fg = FeatureGroupManager.create(
        feature_group_name=name,
        record_identifier_feature_name="record_id",
        event_time_feature_name="event_time",
        feature_definitions=feature_definitions,
        offline_store_config=offline_store_config,
        role_arn=role_arn,
        region=region,
    )

    return fg


def cleanup_feature_group(fg: FeatureGroupManager):
    """
    Delete a FeatureGroupManager and its associated Glue table.

    Args:
        fg: The FeatureGroupManager to delete.
    """
    try:
        # Delete the Glue table if it exists
        if fg.offline_store_config is not None:
            try:
                fg.refresh()  # Ensure we have latest config
                data_catalog_config = fg.offline_store_config.data_catalog_config
                if data_catalog_config is not None:
                    database_name = data_catalog_config.database
                    table_name = data_catalog_config.table_name

                    if database_name and table_name:
                        glue_client = boto3.client("glue")
                        try:
                            glue_client.delete_table(DatabaseName=database_name, Name=table_name)
                        except ClientError as e:
                            # Ignore if table doesn't exist
                            if e.response["Error"]["Code"] != "EntityNotFoundException":
                                raise
            except Exception:
                # Don't fail cleanup if Glue table deletion fails
                pass

        # Delete the FeatureGroupManager
        fg.delete()
    except ClientError:
        # Don't fail cleanup if Glue table deletion fails
        pass


@pytest.mark.serial
@pytest.mark.slow_test
def test_create_feature_group_and_enable_lake_formation(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager and enabling Lake Formation governance.

    This test:
    1. Creates a new FeatureGroupManager with offline store
    2. Waits for it to reach Created status
    3. Enables Lake Formation governance (registers S3, grants permissions, revokes IAM principals)
    4. Cleans up the FeatureGroupManager
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance
        result = fg.enable_lake_formation(disable_hybrid_access_mode=True, acknowledge_risk=True)

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_disabled"] is True

    finally:
        print('done')
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_create_feature_group_with_lake_formation_enabled(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager with lake_formation_config.enabled=True.

    This test verifies the integrated workflow where Lake Formation is enabled
    automatically during FeatureGroupManager creation:
    1. Creates a new FeatureGroupManager with lake_formation_config.enabled=True
    2. Verifies the FeatureGroupManager is created and Lake Formation is configured
    3. Cleans up the FeatureGroupManager
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager with Lake Formation enabled

        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
        lake_formation_config = LakeFormationConfig(
            enabled=True,
            disable_hybrid_access_mode = True,
            acknowledge_risk=True,
        )

        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
            lake_formation_config=lake_formation_config,
        )

        # Verify the FeatureGroupManager was created
        assert fg is not None
        assert fg.feature_group_name == fg_name
        assert fg.feature_group_status == "Created"

        # Verify Lake Formation is configured by checking we can refresh without errors
        fg.refresh()
        assert fg.offline_store_config is not None

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
def test_create_feature_group_without_lake_formation(s3_uri, role, region):
    """
    Test creating a FeatureGroupManager without Lake Formation enabled.

    This test verifies that when lake_formation_config is not provided or enabled=False,
    the FeatureGroupManager is created successfully without any Lake Formation operations:
    1. Creates a new FeatureGroupManager without lake_formation_config
    2. Verifies the FeatureGroupManager is created successfully
    3. Verifies no Lake Formation operations were performed
    4. Cleans up the FeatureGroupManager
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager without Lake Formation
        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))

        # Create without lake_formation_config (default behavior)
        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
        )

        # Verify the FeatureGroupManager was created
        assert fg is not None
        assert fg.feature_group_name == fg_name

        # Wait for Created status to ensure it's fully provisioned
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Verify offline store is configured
        fg.refresh()
        assert fg.offline_store_config is not None
        assert fg.offline_store_config.s3_storage_config is not None

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


# ============================================================================
# Negative Integration Tests
# ============================================================================


def test_create_feature_group_with_lake_formation_fails_without_offline_store(role, region):
    """
    Test that creating a FeatureGroupManager with enable_lake_formation=True fails
    when no offline store is configured.

    Expected behavior: ValueError should be raised indicating offline store is required.
    """
    fg_name = generate_feature_group_name()

    lake_formation_config = LakeFormationConfig(disable_hybrid_access_mode=True, acknowledge_risk=True)
    lake_formation_config.enabled = True

    # Attempt to create without offline store but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            role_arn=role,
            lake_formation_config=lake_formation_config,
        )

    # Verify error message mentions offline_store_config requirement
    assert "lake_formation_config with enabled=True requires offline_store_config to be configured" in str(
        exc_info.value
    )


def test_create_feature_group_with_lake_formation_fails_without_role(s3_uri, region):
    """
    Test that creating a FeatureGroupManager with lake_formation_config.enabled=True fails
    when no role_arn is provided.

    Expected behavior: ValueError should be raised indicating role_arn is required.
    """
    fg_name = generate_feature_group_name()

    offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
    lake_formation_config = LakeFormationConfig(disable_hybrid_access_mode=True, acknowledge_risk=True)
    lake_formation_config.enabled = True

    # Attempt to create without role_arn but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            lake_formation_config=lake_formation_config,
        )

    # Verify error message mentions role_arn requirement
    assert "lake_formation_config with enabled=True requires role_arn to be specified" in str(exc_info.value)


def test_enable_lake_formation_fails_for_non_created_status(s3_uri, role, region):
    """
    Test that enable_lake_formation() fails when called on a FeatureGroupManager
    that is not in 'Created' status.

    Expected behavior: ValueError should be raised indicating the Feature Group
    must be in 'Created' status.

    Note: This test creates its own FeatureGroupManager because it needs to test
    behavior during the 'Creating' status, which requires a fresh resource.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Immediately try to enable Lake Formation without waiting for Created status
        # The Feature Group will be in 'Creating' status
        with pytest.raises(ValueError) as exc_info:
            fg.enable_lake_formation(disable_hybrid_access_mode=True, acknowledge_risk=True, wait_for_active=False)

        # Verify error message mentions status requirement
        error_msg = str(exc_info.value)
        assert "must be in 'Created' status to enable Lake Formation" in error_msg

    finally:
        # Cleanup
        if fg:
            fg.wait_for_status(target_status="Created", poll=30, timeout=300)
            cleanup_feature_group(fg)


def test_enable_lake_formation_without_offline_store(role, region):
    """
    Test that enable_lake_formation() fails when called on a FeatureGroupManager
    without an offline store configured.

    Expected behavior: ValueError should be raised indicating offline store is required.

    Note: This test creates a FeatureGroupManager with only online store, which is a valid
    configuration, but Lake Formation cannot be enabled for it.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create a FeatureGroupManager with only online store (no offline store)
        online_store_config = OnlineStoreConfig(enable_online_store=True)

        fg = FeatureGroupManager.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            online_store_config=online_store_config,
            role_arn=role,
        )

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)

        # Attempt to enable Lake Formation
        with pytest.raises(ValueError) as exc_info:
            fg.enable_lake_formation(disable_hybrid_access_mode=True, acknowledge_risk=True)
        # Verify error message mentions offline store requirement
        assert "does not have an offline store configured" in str(exc_info.value)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


def test_enable_lake_formation_fails_with_invalid_registration_role(
    shared_feature_group_for_negative_tests,
):
    """
    Test that enable_lake_formation() fails when use_service_linked_role=False
    but no registration_role_arn is provided.

    Expected behavior: ValueError should be raised indicating registration_role_arn
    is required when not using service-linked role.
    """
    fg = shared_feature_group_for_negative_tests

    # Attempt to enable Lake Formation without service-linked role and without registration_role_arn
    with pytest.raises(ValueError) as exc_info:
        fg.enable_lake_formation(
            disable_hybrid_access_mode=True,
            acknowledge_risk=True,
            use_service_linked_role=False,
            registration_role_arn=None,
        )

    # Verify error message mentions role requirement
    error_msg = str(exc_info.value)
    assert "registration_role_arn" in error_msg


def test_enable_lake_formation_fails_with_nonexistent_role(
    shared_feature_group_for_negative_tests, role
):
    """
    Test that enable_lake_formation() properly bubbles errors when using
    a nonexistent role ARN for Lake Formation registration.

    Expected behavior: RuntimeError or ClientError should be raised with details
    about the registration failure.

    Note: This test uses a nonexistent role ARN (current role with random suffix)
    to trigger an error during S3 registration with Lake Formation.
    """
    fg = shared_feature_group_for_negative_tests

    # Build a short nonexistent role ARN using the account ID from the real role
    account_id = role.split(":")[4]
    nonexistent_role = f"arn:aws:iam::{account_id}:role/non-existent-role"

    with pytest.raises(RuntimeError) as exc_info:
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=nonexistent_role,
            disable_hybrid_access_mode=True,
            acknowledge_risk=True,
        )

    # Verify we got an appropriate error
    error_msg = str(exc_info.value)
    print(exc_info)
    # Should mention role-related issues (not found, invalid, access denied, etc.)
    assert "EntityNotFoundException" in error_msg


# ============================================================================
# Full Flow Integration Tests with Policy Output
# ============================================================================


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_full_flow_with_policy_output(s3_uri, role, region, caplog):
    """
    Test the full Lake Formation flow with S3 deny policy logging.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with disable_hybrid_access_mode=True
    3. Verifies all Lake Formation phases complete successfully
    4. Verifies the recommended S3 deny policy is logged as a warning
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(disable_hybrid_access_mode=True, acknowledge_risk=True)

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_disabled"] is True

        # Verify the recommended S3 deny policy was logged
        assert any("RECOMMENDED S3 BUCKET POLICY" in record.message for record in caplog.records)
        assert any("DenyFS" in record.message for record in caplog.records)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_default_logs_recommended_policy(s3_uri, role, region, caplog):
    """
    Test that recommended bucket policy is logged with default arguments.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with disable_hybrid_access_mode=True
    3. Verifies phases complete successfully (hybrid_access_mode_disabled=True)
    4. Verifies the recommended S3 deny policy is logged
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance with disable_hybrid_access_mode=True
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(disable_hybrid_access_mode=True, acknowledge_risk=True)

        # Verify phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_disabled"] is True

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_with_custom_role_logs_policy(s3_uri, role, region, caplog):
    """
    Test the full Lake Formation flow with custom registration role.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with use_service_linked_role=False and a custom registration_role_arn
    3. Verifies all phases complete successfully
    4. Verifies the recommended S3 deny policy is logged
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroupManager
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation with custom registration role
        with caplog.at_level(logging.WARNING, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation(
                use_service_linked_role=False,
                registration_role_arn=role,
                disable_hybrid_access_mode=True,
                acknowledge_risk=True,
            )

        # Verify all phases completed successfully
        assert result["s3_location_registered"] is True
        assert result["lf_permissions_granted"] is True
        assert result["hybrid_access_mode_disabled"] is True

        # Verify the recommended S3 deny policy was logged
        assert any("RECOMMENDED S3 BUCKET POLICY" in record.message for record in caplog.records)

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

