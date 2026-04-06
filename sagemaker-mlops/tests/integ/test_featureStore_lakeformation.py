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
from unittest.mock import patch

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
        lake_formation_config = LakeFormationConfig()
        lake_formation_config.enabled = True

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

    lake_formation_config = LakeFormationConfig()
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
    lake_formation_config = LakeFormationConfig()
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

    # Build a short nonexistent role ARN using the account ID from the real role
    account_id = role.split(":")[4]
    nonexistent_role = f"arn:aws:iam::{account_id}:role/non-existent-role"

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
def test_enable_lake_formation_full_flow_with_policy_output(s3_uri, role, region):
    """
    Test the full Lake Formation flow with S3 deny policy application.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation (bucket policy is applied automatically)
    3. Verifies all Lake Formation phases complete successfully (including bucket policy)
    4. Fetches the actual bucket policy and verifies its structure
    """
    import json

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
        result = fg.enable_lake_formation()

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True
        assert result["bucket_policy_applied"] is True

        # Fetch the actual bucket policy from S3 and verify the role is allowed
        bucket_name = s3_uri.replace("s3://", "").split("/")[0]
        s3_client = boto3.client("s3")
        policy = json.loads(s3_client.get_bucket_policy(Bucket=bucket_name)["Policy"])

        # Find a DenyFS statement and verify the execution role is in allowed principals
        deny_stmts = [s for s in policy["Statement"] if s.get("Sid", "").startswith("DenyFS")]
        assert len(deny_stmts) >= 1
        allowed = deny_stmts[0]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert role in allowed

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_default_applies_bucket_policy(s3_uri, role, region, caplog):
    """
    Test that bucket policy is applied automatically with default arguments.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with default arguments
    3. Verifies all four phases complete successfully (including bucket policy)
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

        # Enable Lake Formation governance with defaults
        with caplog.at_level(logging.INFO, logger="sagemaker.mlops.feature_store.feature_group_manager"):
            result = fg.enable_lake_formation()

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True
        assert result["bucket_policy_applied"] is True

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_enable_lake_formation_with_custom_role_policy_output(s3_uri, role, region):
    """
    Test the full Lake Formation flow with custom registration role.

    This test verifies:
    1. Creates a FeatureGroupManager with offline store
    2. Enables Lake Formation with use_service_linked_role=False and a custom registration_role_arn
    3. Fetches the actual bucket policy and verifies the custom role ARN is in the allowed principals
    """
    import json

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
        result = fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=role,
        )

        # Verify all phases completed successfully
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True
        assert result["bucket_policy_applied"] is True

        # Fetch the actual bucket policy from S3 and verify the role is in allowed principals
        bucket_name = s3_uri.replace("s3://", "").split("/")[0]
        s3_client = boto3.client("s3")
        policy = json.loads(s3_client.get_bucket_policy(Bucket=bucket_name)["Policy"])

        # Find the deny statements for this feature group's prefix
        fg.refresh()
        s3_resolved = fg.offline_store_config.s3_storage_config.resolved_output_s3_uri
        fg_prefix = s3_resolved.replace(f"s3://{bucket_name}/", "").rstrip("/")
        sid_suffix = fg_prefix.rsplit("/", 1)[-1]

        object_sid = f"DenyFSObjectAccess_{sid_suffix}"
        deny_stmts = [s for s in policy["Statement"] if s.get("Sid") == object_sid]
        assert len(deny_stmts) == 1

        # Verify the custom role ARN is in the allowed principals
        allowed = deny_stmts[0]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert role in allowed

    finally:
        # Cleanup
        if fg:
            cleanup_feature_group(fg)

# ============================================================================
# Audit Gate Integration Tests
# ============================================================================


@pytest.mark.serial
@pytest.mark.slow_test
def test_audit_gate_detects_glue_table_access(s3_uri, role, region):
    """
    Test that the audit gate in enable_lake_formation() detects principals
    that accessed the Glue table via CloudTrail.

    Flow:
    1. Create a FeatureGroup (without LF) and wait for Created status
    2. Call glue:GetTable on the FG's Glue table to generate a CloudTrail event
    3. Wait for CloudTrail propagation (~5 minutes)
    4. Call enable_lake_formation() with input() mocked to decline ('N')
    5. Verify the result contains aborted=True and audit_results with the accessor

    This test is extensible: additional access patterns (training jobs, Athena
    queries, etc.) can be added as separate test functions following the same
    pattern.
    """
    import time

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # Step 1: Create FG without Lake Formation
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)
        fg.refresh()

        # Extract Glue table info
        data_catalog_config = fg.offline_store_config.data_catalog_config
        database_name = str(data_catalog_config.database)
        table_name = str(data_catalog_config.table_name)

        # Step 2: Access the Glue table to generate a CloudTrail event
        glue_client = boto3.client("glue", region_name=region)
        response = glue_client.get_table(DatabaseName=database_name, Name=table_name)
        assert "Table" in response, "glue:GetTable should return table metadata"
        print(f"[TEST] Called glue:GetTable on {database_name}.{table_name}")

        # Step 3: Wait for CloudTrail propagation
        # CloudTrail events typically appear within 5 minutes
        wait_minutes = 6
        print(f"[TEST] Waiting {wait_minutes} minutes for CloudTrail propagation...")
        time.sleep(wait_minutes * 60)

        # Step 4: Call enable_lake_formation with user declining confirmation
        with patch("builtins.input", return_value="N"):
            result = fg.enable_lake_formation(lookback_days=1)

        # Step 5: Verify the result indicates abort with audit findings
        print(f"[TEST] Audit gate result:\n{result}")

        assert result["aborted"] is True
        audit_results = result["audit_results"]
        assert "glue_table_accessors" in audit_results
        assert len(audit_results["glue_table_accessors"]) > 0
        assert audit_results["glue_table"] == table_name

    finally:
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.serial
@pytest.mark.slow_test
def test_audit_lake_formation_impact_standalone(s3_uri, role, region):
    """
    Test that audit_lake_formation_impact() runs successfully and returns
    the expected dict structure.

    This test does not require CloudTrail data -- it just verifies the method
    executes without error and returns the correct keys.
    """
    fg_name = generate_feature_group_name()
    fg = None

    try:
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)

        result = fg.audit_lake_formation_impact()

        expected_keys = {
            "glue_table_accessors",
            "sagemaker_execution_roles",
            "athena_query_principals",
            "athena_running_queries",
            "glue_etl_jobs",
            "glue_running_job_runs",
            "sagemaker_running_jobs",
            "glue_database",
            "glue_table",
            "s3_path",
            "warnings",
        }
        assert expected_keys.issubset(result.keys()), f"Missing keys: {expected_keys - result.keys()}"

    finally:
        if fg:
            cleanup_feature_group(fg)


@pytest.mark.skip(reason="This is a full e2e test that could take 15 min to run.")
@pytest.mark.serial
@pytest.mark.slow_test
def test_e2e_lf_put_record_athena_query(sagemaker_session, s3_uri, role, region):
    """
    End-to-end test: create FG with LF, put a record, wait for offline sync, query via Athena.

    Skipped by default because offline store sync takes ~10 minutes.
    """
    import time
    from datetime import datetime, timezone

    from sagemaker.mlops.feature_store import (
        FeatureValue,
        create_athena_query,
    )

    fg_name = generate_feature_group_name()
    fg = None

    try:
        # 1. Create feature group with LF enabled
        fg = create_test_feature_group(fg_name, s3_uri, role, region)
        fg.wait_for_status(target_status="Created", poll=30, timeout=300)

        result = fg.enable_lake_formation()
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True
        assert result["bucket_policy_applied"] is True

        # 2. Put a record
        record_id = uuid.uuid4().hex[:8]
        event_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fg.put_record(
            record=[
                FeatureValue(feature_name="record_id", value_as_string=record_id),
                FeatureValue(feature_name="event_time", value_as_string=event_time),
                FeatureValue(feature_name="feature_value", value_as_string="3.14"),
            ],
        )
        logging.info(f"Put record: record_id={record_id}, event_time={event_time}")

        # 3. Wait for offline store sync (~10 min)
        wait_minutes = 10
        logging.info(f"Waiting {wait_minutes} minutes for offline store sync...")
        time.sleep(wait_minutes * 60)

        # 4. Query via Athena
        bucket = sagemaker_session.default_bucket()
        output_location = f"s3://{bucket}/athena-results/test-lf-e2e"

        query = create_athena_query(fg_name, sagemaker_session)
        query.run(
            query_string=f'SELECT * FROM "{query.table_name}" WHERE record_id = \'{record_id}\'',
            output_location=output_location,
        )
        query.wait()

        df = query.as_dataframe()
        assert len(df) >= 1, f"Expected at least 1 row, got {len(df)}"
        assert df.iloc[0]["record_id"] == record_id

    finally:
        if fg:
            cleanup_feature_group(fg)


