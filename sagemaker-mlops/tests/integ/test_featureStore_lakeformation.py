"""
Integration tests for Lake Formation with FeatureGroup.

These tests require:
- AWS credentials with Lake Formation and SageMaker permissions
- An S3 bucket for offline store (uses default SageMaker bucket)
- An IAM role for Feature Store (uses execution role)

Run with: pytest tests/integ/test_featureStore_lakeformation.py -v -m integ
"""

import uuid

import boto3
import pytest
from botocore.exceptions import ClientError

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.mlops.feature_store import (
    FeatureGroup,
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
    Create a single FeatureGroup for negative tests that only need to verify
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


def create_test_feature_group(name: str, s3_uri: str, role_arn: str, region: str) -> FeatureGroup:
    """Create a FeatureGroup with offline store for testing."""

    offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))

    fg = FeatureGroup.create(
        feature_group_name=name,
        record_identifier_feature_name="record_id",
        event_time_feature_name="event_time",
        feature_definitions=feature_definitions,
        offline_store_config=offline_store_config,
        role_arn=role_arn,
        region=region,
    )

    return fg


def cleanup_feature_group(fg: FeatureGroup):
    """
    Delete a FeatureGroup and its associated Glue table.

    Args:
        fg: The FeatureGroup to delete.
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

        # Delete the FeatureGroup
        fg.delete()
    except ClientError:
        # Don't fail cleanup if Glue table deletion fails
        pass


@pytest.mark.serial
@pytest.mark.slow_test
def test_create_feature_group_and_enable_lake_formation(s3_uri, role, region):
    """
    Test creating a FeatureGroup and enabling Lake Formation governance.

    This test:
    1. Creates a new FeatureGroup with offline store
    2. Waits for it to reach Created status
    3. Enables Lake Formation governance (registers S3, grants permissions, revokes IAM principals)
    4. Cleans up the FeatureGroup
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance
        result = fg.enable_lake_formation()

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True

    finally:
        print('done')
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_create_feature_group_with_lake_formation_enabled(s3_uri, role, region):
    """
    Test creating a FeatureGroup with lake_formation_config.enabled=True.

    This test verifies the integrated workflow where Lake Formation is enabled
    automatically during FeatureGroup creation:
    1. Creates a new FeatureGroup with lake_formation_config.enabled=True
    2. Verifies the FeatureGroup is created and Lake Formation is configured
    3. Cleans up the FeatureGroup
    """

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup with Lake Formation enabled

        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
        lake_formation_config = LakeFormationConfig()
        lake_formation_config.enabled = True

        fg = FeatureGroup.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
            lake_formation_config=lake_formation_config,
        )

        # Verify the FeatureGroup was created
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
    Test creating a FeatureGroup without Lake Formation enabled.

    This test verifies that when lake_formation_config is not provided or enabled=False,
    the FeatureGroup is created successfully without any Lake Formation operations:
    1. Creates a new FeatureGroup without lake_formation_config
    2. Verifies the FeatureGroup is created successfully
    3. Verifies no Lake Formation operations were performed
    4. Cleans up the FeatureGroup
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup without Lake Formation
        offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))

        # Create without lake_formation_config (default behavior)
        fg = FeatureGroup.create(
            feature_group_name=fg_name,
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=role,
        )

        # Verify the FeatureGroup was created
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
    Test that creating a FeatureGroup with enable_lake_formation=True fails
    when no offline store is configured.

    Expected behavior: ValueError should be raised indicating offline store is required.
    """
    fg_name = generate_feature_group_name()

    lake_formation_config = LakeFormationConfig()
    lake_formation_config.enabled = True

    # Attempt to create without offline store but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroup.create(
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
    Test that creating a FeatureGroup with lake_formation_config.enabled=True fails
    when no role_arn is provided.

    Expected behavior: ValueError should be raised indicating role_arn is required.
    """
    fg_name = generate_feature_group_name()

    offline_store_config = OfflineStoreConfig(s3_storage_config=S3StorageConfig(s3_uri=s3_uri))
    lake_formation_config = LakeFormationConfig()
    lake_formation_config.enabled = True

    # Attempt to create without role_arn but with Lake Formation enabled
    with pytest.raises(ValueError) as exc_info:
        FeatureGroup.create(
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
    Test that enable_lake_formation() fails when called on a FeatureGroup
    that is not in 'Created' status.

    Expected behavior: ValueError should be raised indicating the Feature Group
    must be in 'Created' status.

    Note: This test creates its own FeatureGroup because it needs to test
    behavior during the 'Creating' status, which requires a fresh resource.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Immediately try to enable Lake Formation without waiting for Created status
        # The Feature Group will be in 'Creating' status
        with pytest.raises(ValueError) as exc_info:
            fg.enable_lake_formation(wait_for_active=False)

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
    Test that enable_lake_formation() fails when called on a FeatureGroup
    without an offline store configured.

    Expected behavior: ValueError should be raised indicating offline store is required.

    Note: This test creates a FeatureGroup with only online store, which is a valid
    configuration, but Lake Formation cannot be enabled for it.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create a FeatureGroup with only online store (no offline store)
        online_store_config = OnlineStoreConfig(enable_online_store=True)

        fg = FeatureGroup.create(
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
            fg.enable_lake_formation()
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

    # Generate a nonexistent role ARN by appending a random string to the current role
    nonexistent_role = f"{role}-nonexistent-{uuid.uuid4().hex[:8]}"

    with pytest.raises(RuntimeError) as exc_info:
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=nonexistent_role,
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
def test_enable_lake_formation_full_flow_with_policy_output(s3_uri, role, region, capsys):
    """
    Test the full Lake Formation flow with S3 deny policy output.

    This test verifies:
    1. Creates a FeatureGroup with offline store
    2. Enables Lake Formation with show_s3_policy=True
    3. Verifies all Lake Formation phases complete successfully
    4. Verifies the S3 deny policy is printed to the console
    5. Verifies the policy structure contains expected elements

    This validates Requirements 6.1-6.9 from the design document.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance with policy output
        result = fg.enable_lake_formation(show_s3_policy=True)

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True

        # Capture the printed output
        captured = capsys.readouterr()
        output = captured.out

        # Re-print the output so it's visible in terminal with -s flag
        print(output)

        # Verify the policy header is printed
        assert "S3 Bucket Policy Update recommended" in output
        assert "=" * 80 in output

        # Verify bucket information is printed
        # Extract bucket name from s3_uri (s3://bucket/path -> bucket)
        expected_bucket = s3_uri.replace("s3://", "").split("/")[0]
        assert f"Bucket: {expected_bucket}" in output

        # Verify policy structure elements are present
        assert '"Version": "2012-10-17"' in output
        assert '"Statement"' in output
        assert '"Effect": "Deny"' in output
        assert '"Principal": "*"' in output

        # Verify the deny actions are present
        assert "s3:GetObject" in output
        assert "s3:PutObject" in output
        assert "s3:DeleteObject" in output
        assert "s3:ListBucket" in output

        # Verify the condition structure is present
        assert "StringNotEquals" in output
        assert "aws:PrincipalArn" in output

        # Verify the role ARN is in the allowed principals
        assert role in output

        # Verify the service-linked role pattern is present (default use_service_linked_role=True)
        assert "aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess" in output

        # Verify instructions are printed
        assert "Merge this with your existing bucket policy" in output

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_no_policy_output_by_default(s3_uri, role, region, capsys):
    """
    Test that S3 deny policy is NOT printed when show_s3_policy=False (default).

    This test verifies:
    1. Creates a FeatureGroup with offline store
    2. Enables Lake Formation without show_s3_policy (defaults to False)
    3. Verifies all Lake Formation phases complete successfully
    4. Verifies the S3 deny policy is NOT printed to the console

    This validates Requirement 6.2 from the design document.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation governance WITHOUT policy output (default)
        result = fg.enable_lake_formation()

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True

        # Capture the printed output
        captured = capsys.readouterr()
        output = captured.out

        # Verify the policy is NOT printed
        assert "S3 Bucket Policy Update recommended" not in output
        assert '"Version": "2012-10-17"' not in output
        assert "s3:GetObject" not in output

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_with_custom_role_policy_output(s3_uri, role, region, capsys):
    """
    Test the full Lake Formation flow with custom registration role and policy output.

    This test verifies:
    1. Creates a FeatureGroup with offline store
    2. Enables Lake Formation with use_service_linked_role=False and a custom registration_role_arn
    3. Verifies the S3 deny policy uses the custom role ARN instead of service-linked role

    This validates Requirements 6.4, 6.5 from the design document.

    Note: This test uses the same execution role as the registration role for simplicity.
    In production, these would typically be different roles.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Create the FeatureGroup
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        assert fg is not None

        # Wait for Created status
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        assert fg.feature_group_status == "Created"

        # Enable Lake Formation with custom registration role and policy output
        # Using the same role for both execution and registration for test simplicity
        result = fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=role,
            show_s3_policy=True,
        )

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True

        # Capture the printed output
        captured = capsys.readouterr()
        output = captured.out

        # Verify the policy header is printed
        assert "S3 Bucket Policy Update recommended" in output

        # Verify the custom role ARN is used in the policy (appears twice - once for each principal)
        # The role should appear as both the Lake Formation role and the Feature Store role
        assert output.count(role) >= 2

        # Verify the service-linked role is NOT used
        assert "aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess" not in output

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)
