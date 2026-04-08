"""Unit tests for FeatureGroupManager."""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest
from boto3 import Session

from sagemaker.mlops.feature_store import FeatureGroupManager, LakeFormationConfig


class TestS3UriToArn:
    """Tests for _s3_uri_to_arn static method."""

    def test_converts_s3_uri_to_arn(self):
        """Test S3 URI is converted to ARN format."""
        uri = "s3://my-bucket/my-prefix/data"
        result = FeatureGroupManager._s3_uri_to_arn(uri)
        assert result == "arn:aws:s3:::my-bucket/my-prefix/data"

    def test_handles_bucket_only_uri(self):
        """Test S3 URI with bucket only."""
        uri = "s3://my-bucket"
        result = FeatureGroupManager._s3_uri_to_arn(uri)
        assert result == "arn:aws:s3:::my-bucket"

    def test_returns_arn_unchanged(self):
        """Test ARN input is returned unchanged (idempotent)."""
        arn = "arn:aws:s3:::my-bucket/path"
        result = FeatureGroupManager._s3_uri_to_arn(arn)
        assert result == arn

    def test_uses_region_for_partition(self):
        """Test that region is used to determine partition."""
        uri = "s3://my-bucket/path"
        result = FeatureGroupManager._s3_uri_to_arn(uri, region="cn-north-1")
        assert result.startswith("arn:aws-cn:s3:::")



class TestGetLakeFormationClient:
    """Tests for _get_lake_formation_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_lake_formation_client = FeatureGroupManager._get_lake_formation_client.__get__(fg)

        client = fg._get_lake_formation_client(region="us-west-2")

        mock_session.client.assert_called_with("lakeformation", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_lake_formation_client = FeatureGroupManager._get_lake_formation_client.__get__(fg)

        client = fg._get_lake_formation_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("lakeformation", region_name="us-west-2")
        assert client == mock_client


class TestRegisterS3WithLakeFormation:
    """Tests for _register_s3_with_lake_formation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._s3_uri_to_arn = FeatureGroupManager._s3_uri_to_arn
        self.fg._register_s3_with_lake_formation = (
            FeatureGroupManager._register_s3_with_lake_formation.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_lake_formation_client = MagicMock(return_value=self.mock_client)

    def test_successful_registration_returns_true(self):
        """Test successful S3 registration returns True."""
        self.mock_client.register_resource.return_value = {}

        result = self.fg._register_s3_with_lake_formation("s3://test-bucket/prefix")

        assert result is True
        self.mock_client.register_resource.assert_called_with(
            ResourceArn="arn:aws:s3:::test-bucket/prefix",
            WithFederation=True,
            UseServiceLinkedRole=True,
        )

    def test_already_exists_exception_returns_true(self):
        """Test AlreadyExistsException is handled gracefully."""
        self.mock_client.register_resource.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AlreadyExistsException", "Message": "Already exists"}},
            "RegisterResource",
        )

        result = self.fg._register_s3_with_lake_formation("s3://test-bucket/prefix")

        assert result is True

    def test_other_exceptions_are_propagated(self):
        """Test non-AlreadyExistsException errors are propagated."""
        self.mock_client.register_resource.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "RegisterResource",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._register_s3_with_lake_formation("s3://test-bucket/prefix")

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_uses_service_linked_role(self):
        """Test UseServiceLinkedRole is set to True."""
        self.mock_client.register_resource.return_value = {}

        self.fg._register_s3_with_lake_formation("s3://bucket/path")

        call_args = self.mock_client.register_resource.call_args
        assert call_args[1]["UseServiceLinkedRole"] is True
        assert call_args[1]["WithFederation"] is True

    def test_uses_custom_role_arn_when_service_linked_role_disabled(self):
        """Test custom role ARN is used when use_service_linked_role is False."""
        self.mock_client.register_resource.return_value = {}
        custom_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        self.fg._register_s3_with_lake_formation(
            "s3://bucket/path",
            use_service_linked_role=False,
            role_arn=custom_role,
        )

        call_args = self.mock_client.register_resource.call_args
        assert call_args[1]["RoleArn"] == custom_role
        assert call_args[1]["WithFederation"] is True
        assert "UseServiceLinkedRole" not in call_args[1]

    def test_raises_error_when_role_arn_missing_and_service_linked_role_disabled(self):
        """Test ValueError when use_service_linked_role is False but role_arn not provided."""
        with pytest.raises(ValueError) as exc_info:
            self.fg._register_s3_with_lake_formation(
                "s3://bucket/path", use_service_linked_role=False
            )

        assert "role_arn must be provided when use_service_linked_role is False" in str(
            exc_info.value
        )



class TestRevokeIamAllowedPrincipal:
    """Tests for _revoke_iam_allowed_principal method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._revoke_iam_allowed_principal = FeatureGroupManager._revoke_iam_allowed_principal.__get__(
            self.fg
        )
        self.mock_client = MagicMock()
        self.fg._get_lake_formation_client = MagicMock(return_value=self.mock_client)

    def test_successful_revocation_returns_true(self):
        """Test successful revocation returns True when permissions exist."""
        self.mock_client.list_permissions.return_value = {
            "PrincipalResourcePermissions": [{"Principal": {}, "Resource": {}}]
        }
        self.mock_client.revoke_permissions.return_value = {}

        result = self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert result is True
        self.mock_client.revoke_permissions.assert_called_once()

    def test_revoke_permissions_call_structure(self):
        """Test that revoke_permissions is called with correct parameters."""
        self.mock_client.list_permissions.return_value = {
            "PrincipalResourcePermissions": [{"Principal": {}, "Resource": {}}]
        }
        self.mock_client.revoke_permissions.return_value = {}
        database_name = "my_database"
        table_name = "my_table"

        self.fg._revoke_iam_allowed_principal(database_name, table_name)

        call_args = self.mock_client.revoke_permissions.call_args
        assert call_args[1]["Principal"] == {
            "DataLakePrincipalIdentifier": "IAM_ALLOWED_PRINCIPALS"
        }
        assert call_args[1]["Permissions"] == ["ALL"]
        assert call_args[1]["Resource"] == {
            "Table": {
                "DatabaseName": database_name,
                "Name": table_name,
            }
        }

    def test_no_permissions_skips_revoke(self):
        """Test that empty list_permissions result skips revoke and returns True."""
        self.mock_client.list_permissions.return_value = {
            "PrincipalResourcePermissions": []
        }

        result = self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert result is True
        self.mock_client.revoke_permissions.assert_not_called()

    def test_list_permissions_error_propagates(self):
        """Test that errors from list_permissions propagate."""
        self.mock_client.list_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "ListPermissions",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        self.mock_client.revoke_permissions.assert_not_called()

    def test_revoke_permissions_error_propagates(self):
        """Test that errors from revoke_permissions propagate."""
        self.mock_client.list_permissions.return_value = {
            "PrincipalResourcePermissions": [{"Principal": {}, "Resource": {}}]
        }
        self.mock_client.revoke_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "RevokePermissions",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_passes_session_and_region_to_client(self):
        """Test session and region are passed to get_lake_formation_client."""
        self.mock_client.list_permissions.return_value = {
            "PrincipalResourcePermissions": []
        }
        mock_session = MagicMock()

        self.fg._revoke_iam_allowed_principal(
            "test_database", "test_table", session=mock_session, region="us-west-2"
        )

        self.fg._get_lake_formation_client.assert_called_with(mock_session, "us-west-2")



class TestGrantLakeFormationPermissions:
    """Tests for _grant_lake_formation_permissions method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._grant_lake_formation_permissions = (
            FeatureGroupManager._grant_lake_formation_permissions.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_lake_formation_client = MagicMock(return_value=self.mock_client)

    def test_successful_grant_returns_true(self):
        """Test successful permission grant returns True."""
        self.mock_client.grant_permissions.return_value = {}

        result = self.fg._grant_lake_formation_permissions(
            "arn:aws:iam::123456789012:role/TestRole", "test_database", "test_table"
        )

        assert result is True
        self.mock_client.grant_permissions.assert_called_once()

    def test_grant_permissions_call_structure(self):
        """Test that grant_permissions is called with correct parameters."""
        self.mock_client.grant_permissions.return_value = {}
        role_arn = "arn:aws:iam::123456789012:role/MyExecutionRole"

        self.fg._grant_lake_formation_permissions(role_arn, "my_database", "my_table")

        call_args = self.mock_client.grant_permissions.call_args
        assert call_args[1]["Principal"] == {"DataLakePrincipalIdentifier": role_arn}
        assert call_args[1]["Resource"] == {
            "Table": {
                "DatabaseName": "my_database",
                "Name": "my_table",
            }
        }
        assert call_args[1]["Permissions"] == ["SELECT", "INSERT", "DELETE", "DESCRIBE", "ALTER"]
        assert call_args[1]["PermissionsWithGrantOption"] == []

    def test_invalid_input_exception_returns_true(self):
        """Test InvalidInputException is handled gracefully (permissions may exist)."""
        self.mock_client.grant_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidInputException", "Message": "Permissions already exist"}},
            "GrantPermissions",
        )

        result = self.fg._grant_lake_formation_permissions(
            "arn:aws:iam::123456789012:role/TestRole", "test_database", "test_table"
        )

        assert result is True

    def test_other_exceptions_are_propagated(self):
        """Test non-InvalidInputException errors are propagated."""
        self.mock_client.grant_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "GrantPermissions",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._grant_lake_formation_permissions(
                "arn:aws:iam::123456789012:role/TestRole", "test_database", "test_table"
            )

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_passes_session_and_region_to_client(self):
        """Test session and region are passed to get_lake_formation_client."""
        self.mock_client.grant_permissions.return_value = {}
        mock_session = MagicMock()

        self.fg._grant_lake_formation_permissions(
            "arn:aws:iam::123456789012:role/TestRole",
            "test_database",
            "test_table",
            session=mock_session,
            region="us-west-2",
        )

        self.fg._get_lake_formation_client.assert_called_with(mock_session, "us-west-2")



class TestEnableLakeFormationValidation:
    """Tests for enable_lake_formation validation logic."""

    @patch.object(FeatureGroupManager, "refresh")
    def test_raises_error_when_no_offline_store(self, mock_refresh):
        """Test that enable_lake_formation raises ValueError when no offline store is configured."""
        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = None
        fg.feature_group_status = "Created"

        with pytest.raises(ValueError, match="does not have an offline store configured"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "refresh")
    def test_raises_error_when_no_role_arn(self, mock_refresh):
        """Test that enable_lake_formation raises ValueError when no role_arn is configured."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = None
        fg.feature_group_status = "Created"

        with pytest.raises(ValueError, match="does not have a role_arn configured"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "refresh")
    def test_raises_error_when_invalid_status(self, mock_refresh):
        """Test enable_lake_formation raises ValueError when Feature Group not in Created status."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        fg.feature_group_status = "Creating"

        with pytest.raises(ValueError, match="must be in 'Created' status"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_wait_for_active_calls_wait_for_status(
        self, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
    ):
        """Test that wait_for_active=True calls wait_for_status with 'Created' target."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call with wait_for_active=True
        fg.enable_lake_formation(wait_for_active=True, disable_hybrid_access_mode=True)

        # Verify wait_for_status was called with "Created"
        mock_wait.assert_called_once_with(target_status="Created")
        # Verify refresh was called after wait
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_wait_for_active_false_does_not_call_wait(
        self, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
    ):
        """Test that wait_for_active=False does not call wait_for_status."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call with wait_for_active=False (default)
        fg.enable_lake_formation(wait_for_active=False, disable_hybrid_access_mode=True)

        # Verify wait_for_status was NOT called
        mock_wait.assert_not_called()
        # Verify refresh was still called
        mock_refresh.assert_called_once()


    @pytest.mark.parametrize(
        "feature_group_name,role_arn,s3_uri,database_name,table_name",
        [
            ("test-fg", "TestRole", "path1", "db1", "table1"),
            ("my_feature_group", "ExecutionRole", "data/features", "feature_db", "feature_table"),
            ("fg123", "MyRole123", "ml/features/v1", "analytics", "features_v1"),
            ("simple", "SimpleRole", "simple-path", "simple_db", "simple_table"),
            (
                "complex-name",
                "ComplexExecutionRole",
                "complex/path/structure",
                "complex_database",
                "complex_table_name",
            ),
            (
                "underscore_name",
                "Underscore_Role",
                "underscore_path",
                "underscore_db",
                "underscore_table",
            ),
            ("mixed-123", "Mixed123Role", "mixed/path/123", "mixed_db_123", "mixed_table_123"),
            ("x", "XRole", "x", "x", "x"),
            (
                "very-long-name",
                "VeryLongRoleName",
                "very/long/path/structure",
                "very_long_database_name",
                "very_long_table_name",
            ),
        ],
    )
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_fail_fast_phase_execution(
        self,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
        feature_group_name,
        role_arn,
        s3_uri,
        database_name,
        table_name,
    ):
        """
        Test fail-fast behavior for Lake Formation phases.

        If Phase 1 (S3 registration) fails, Phase 2 and 3 should not execute.
        If Phase 2 fails, Phase 3 should not execute.
        RuntimeError should indicate which phase failed.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name=feature_group_name)
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri=f"s3://test-bucket/{s3_uri}",
                resolved_output_s3_uri=f"s3://test-bucket/resolved-{s3_uri}",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database=database_name, table_name=table_name
            ),
        )
        fg.role_arn = f"arn:aws:iam::123456789012:role/{role_arn}"
        fg.feature_group_status = "Created"

        # Test Phase 1 failure - subsequent phases should not be called
        mock_register.side_effect = Exception("Phase 1 failed")
        mock_grant.return_value = True
        mock_revoke.return_value = True

        with pytest.raises(
            RuntimeError, match="Failed to register S3 location with Lake Formation"
        ):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify Phase 1 was called but Phase 2 and 3 were not
        mock_register.assert_called_once()
        mock_grant.assert_not_called()
        mock_revoke.assert_not_called()

        # Reset mocks for Phase 2 failure test
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        # Test Phase 2 failure - Phase 3 should not be called
        mock_register.side_effect = None
        mock_register.return_value = True
        mock_grant.side_effect = Exception("Phase 2 failed")
        mock_revoke.return_value = True

        with pytest.raises(RuntimeError, match="Failed to grant Lake Formation permissions"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify Phase 1 and 2 were called but Phase 3 was not
        mock_register.assert_called_once()
        mock_grant.assert_called_once()
        mock_revoke.assert_not_called()

        # Reset mocks for Phase 3 failure test
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        # Test Phase 3 failure - all phases should be called
        mock_register.side_effect = None
        mock_register.return_value = True
        mock_grant.side_effect = None
        mock_grant.return_value = True
        mock_revoke.side_effect = Exception("Phase 3 failed")

        with pytest.raises(RuntimeError, match="Failed to revoke IAMAllowedPrincipal permissions"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # Verify all phases were called
        mock_register.assert_called_once()
        mock_grant.assert_called_once()
        mock_revoke.assert_called_once()



class TestUnhandledExceptionPropagation:
    """Tests for proper propagation of unhandled boto3 exceptions."""

    def test_register_s3_propagates_unhandled_exceptions(self):
        """
        Non-AlreadyExists Errors Propagate from S3 Registration

        For any error from Lake Formation's register_resource API that is not
        AlreadyExistsException, the error should be propagated to the caller unchanged.

        """
        fg = MagicMock(spec=FeatureGroupManager)
        fg._s3_uri_to_arn = FeatureGroupManager._s3_uri_to_arn
        fg._register_s3_with_lake_formation = FeatureGroupManager._register_s3_with_lake_formation.__get__(
            fg
        )
        mock_client = MagicMock()
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        # Configure mock to raise an unhandled error
        mock_client.register_resource.side_effect = botocore.exceptions.ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "User does not have permission",
                }
            },
            "RegisterResource",
        )

        # Verify the exception is propagated unchanged
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            fg._register_s3_with_lake_formation("s3://test-bucket/path")

        # Verify error details are preserved
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        assert exc_info.value.response["Error"]["Message"] == "User does not have permission"
        assert exc_info.value.operation_name == "RegisterResource"

    def test_revoke_iam_principal_propagates_unhandled_exceptions(self):
        """
        Non-InvalidInput Errors Propagate from IAM Principal Revocation

        For any error from Lake Formation's list_permissions API,
        the error should be propagated to the caller unchanged.

        """
        fg = MagicMock(spec=FeatureGroupManager)
        fg._revoke_iam_allowed_principal = FeatureGroupManager._revoke_iam_allowed_principal.__get__(fg)
        mock_client = MagicMock()
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        # Configure mock to raise an unhandled error from list_permissions
        mock_client.list_permissions.side_effect = botocore.exceptions.ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "User does not have permission",
                }
            },
            "ListPermissions",
        )

        # Verify the exception is propagated unchanged
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            fg._revoke_iam_allowed_principal("test_database", "test_table")

        # Verify error details are preserved
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        assert exc_info.value.response["Error"]["Message"] == "User does not have permission"
        assert exc_info.value.operation_name == "ListPermissions"

    def test_grant_permissions_propagates_unhandled_exceptions(self):
        """
        Non-InvalidInput Errors Propagate from Permission Grant

        For any error from Lake Formation's grant_permissions API that is not
        InvalidInputException, the error should be propagated to the caller unchanged.

        """
        fg = MagicMock(spec=FeatureGroupManager)
        fg._grant_lake_formation_permissions = (
            FeatureGroupManager._grant_lake_formation_permissions.__get__(fg)
        )
        mock_client = MagicMock()
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        # Configure mock to raise an unhandled error
        mock_client.grant_permissions.side_effect = botocore.exceptions.ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "User does not have permission",
                }
            },
            "GrantPermissions",
        )

        # Verify the exception is propagated unchanged
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            fg._grant_lake_formation_permissions(
                "arn:aws:iam::123456789012:role/TestRole", "test_database", "test_table"
            )

        # Verify error details are preserved
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        assert exc_info.value.response["Error"]["Message"] == "User does not have permission"
        assert exc_info.value.operation_name == "GrantPermissions"

    def test_handled_exceptions_do_not_propagate(self):
        """
        Verify that specifically handled exceptions (AlreadyExistsException, InvalidInputException)
        do NOT propagate but return True instead, while all other exceptions are propagated.
        """
        fg = MagicMock(spec=FeatureGroupManager)
        fg._s3_uri_to_arn = FeatureGroupManager._s3_uri_to_arn
        fg._register_s3_with_lake_formation = FeatureGroupManager._register_s3_with_lake_formation.__get__(
            fg
        )
        fg._revoke_iam_allowed_principal = FeatureGroupManager._revoke_iam_allowed_principal.__get__(fg)
        fg._grant_lake_formation_permissions = (
            FeatureGroupManager._grant_lake_formation_permissions.__get__(fg)
        )
        mock_client = MagicMock()
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        # Test AlreadyExistsException is handled (not propagated)
        mock_client.register_resource.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AlreadyExistsException", "Message": "Already exists"}},
            "RegisterResource",
        )
        result = fg._register_s3_with_lake_formation("s3://test-bucket/path")
        assert result is True  # Should return True, not raise

        # Test empty list_permissions returns True (no exception, no revoke needed)
        mock_client.list_permissions.return_value = {"PrincipalResourcePermissions": []}
        result = fg._revoke_iam_allowed_principal("db", "table")
        assert result is True  # Should return True, not raise

        # Test InvalidInputException is handled for grant (not propagated)
        mock_client.grant_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidInputException", "Message": "Invalid input"}},
            "GrantPermissions",
        )
        result = fg._grant_lake_formation_permissions(
            "arn:aws:iam::123456789012:role/TestRole", "db", "table"
        )
        assert result is True  # Should return True, not raise



class TestCreateWithLakeFormation:
    """Tests for create() method with Lake Formation integration."""

    @pytest.mark.parametrize(
        "feature_group_name,record_id_feature,event_time_feature",
        [
            ("test-fg", "record_id", "event_time"),
            ("my_feature_group", "id", "timestamp"),
            ("fg123", "identifier", "time"),
            ("simple", "rec_id", "evt_time"),
            ("complex-name", "record_identifier", "event_timestamp"),
            ("underscore_name", "record_id_field", "event_time_field"),
            ("mixed-123", "id_123", "time_123"),
            ("x", "x_id", "x_time"),
            ("very-long-name", "very_long_record_id", "very_long_event_time"),
        ],
    )
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_no_lake_formation_operations_when_disabled(
        self,
        mock_enable_lf,
        mock_wait,
        mock_get,
        mock_get_client,
        feature_group_name,
        record_id_feature,
        event_time_feature,
    ):
        """
        No Lake Formation Operations When Disabled

        For any call to FeatureGroupManager.create() where lake_formation_config is None or has enabled=False,
        no Lake Formation client methods should be invoked.

        """
        from sagemaker.core.shapes import FeatureDefinition

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        # Mock the get method to return a feature group
        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.feature_group_name = feature_group_name
        mock_get.return_value = mock_fg

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Test 1: lake_formation_config with enabled=False (explicit)
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = False
        result = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            lake_formation_config=lf_config,
        )

        # Verify enable_lake_formation was NOT called
        mock_enable_lf.assert_not_called()
        # Verify wait_for_status was NOT called
        mock_wait.assert_not_called()
        # Verify the feature group was returned
        assert result == mock_fg

        # Reset mocks for next test
        mock_enable_lf.reset_mock()
        mock_wait.reset_mock()
        mock_get.reset_mock()
        mock_get.return_value = mock_fg

        # Test 2: lake_formation_config not specified (defaults to None)
        result = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            # lake_formation_config not specified, should default to None
        )

        # Verify enable_lake_formation was NOT called
        mock_enable_lf.assert_not_called()
        # Verify wait_for_status was NOT called
        mock_wait.assert_not_called()
        # Verify the feature group was returned
        assert result == mock_fg

    @pytest.mark.parametrize(
        "feature_group_name,record_id_feature,event_time_feature,role_arn,s3_uri,database,table",
        [
            ("test-fg", "record_id", "event_time", "TestRole", "path1", "db1", "table1"),
            (
                "my_feature_group",
                "id",
                "timestamp",
                "ExecutionRole",
                "data/features",
                "feature_db",
                "feature_table",
            ),
            (
                "fg123",
                "identifier",
                "time",
                "MyRole123",
                "ml/features/v1",
                "analytics",
                "features_v1",
            ),
        ],
    )
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_enable_lake_formation_called_when_enabled(
        self,
        mock_enable_lf,
        mock_wait,
        mock_get,
        mock_get_client,
        feature_group_name,
        record_id_feature,
        event_time_feature,
        role_arn,
        s3_uri,
        database,
        table,
    ):
        """
        Test that enable_lake_formation is called when lake_formation_config has enabled=True.

        This verifies the integration between create() and enable_lake_formation().
        """
        from sagemaker.core.shapes import (
            FeatureDefinition,
            OfflineStoreConfig,
            S3StorageConfig,
            DataCatalogConfig,
        )

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        # Mock the get method to return a feature group
        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.feature_group_name = feature_group_name
        mock_fg.wait_for_status = mock_wait
        mock_fg.enable_lake_formation = mock_enable_lf
        mock_get.return_value = mock_fg

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Create offline store config
        offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri=f"s3://test-bucket/{s3_uri}"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database=database, table_name=table
            ),
        )

        # Create LakeFormationConfig with enabled=True
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True

        # Create with lake_formation_config enabled=True
        result = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=f"arn:aws:iam::123456789012:role/{role_arn}",
            lake_formation_config=lf_config,
        )

        # Verify wait_for_status was called with "Created"
        mock_wait.assert_called_once_with(target_status="Created")
        # Verify enable_lake_formation was called with default use_service_linked_role=True
        mock_enable_lf.assert_called_once_with(
            session=None,
            region=None,
            use_service_linked_role=True,
            registration_role_arn=None,
            disable_hybrid_access_mode=False,
        )
        # Verify the feature group was returned
        assert result == mock_fg

    @pytest.mark.parametrize(
        "feature_group_name,record_id_feature,event_time_feature",
        [
            ("test-fg", "record_id", "event_time"),
            ("my_feature_group", "id", "timestamp"),
            ("fg123", "identifier", "time"),
        ],
    )
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_lake_formation_enabled_without_offline_store(
        self, mock_get_client, feature_group_name, record_id_feature, event_time_feature
    ):
        """Test create() raises ValueError when lake_formation_config enabled=True without offline_store."""
        from sagemaker.core.shapes import FeatureDefinition

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Create LakeFormationConfig with enabled=True
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True

        # Test with lake_formation_config enabled=True but no offline_store_config
        with pytest.raises(
            ValueError,
            match="lake_formation_config with enabled=True requires offline_store_config to be configured",
        ):
            FeatureGroupManager.create(
                feature_group_name=feature_group_name,
                record_identifier_feature_name=record_id_feature,
                event_time_feature_name=event_time_feature,
                feature_definitions=feature_definitions,
                lake_formation_config=lf_config,
                # offline_store_config not provided
            )

    @pytest.mark.parametrize(
        "feature_group_name,record_id_feature,event_time_feature,s3_uri,database,table",
        [
            ("test-fg", "record_id", "event_time", "path1", "db1", "table1"),
            ("my_feature_group", "id", "timestamp", "data/features", "feature_db", "feature_table"),
            ("fg123", "identifier", "time", "ml/features/v1", "analytics", "features_v1"),
        ],
    )
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_lake_formation_enabled_without_role_arn(
        self,
        mock_get_client,
        feature_group_name,
        record_id_feature,
        event_time_feature,
        s3_uri,
        database,
        table,
    ):
        """Test create() raises ValueError when lake_formation_config enabled=True without role_arn."""
        from sagemaker.core.shapes import (
            FeatureDefinition,
            OfflineStoreConfig,
            S3StorageConfig,
            DataCatalogConfig,
        )

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Create offline store config
        offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri=f"s3://test-bucket/{s3_uri}"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database=database, table_name=table
            ),
        )

        # Create LakeFormationConfig with enabled=True
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True

        # Test with lake_formation_config enabled=True but no role_arn
        with pytest.raises(
            ValueError, match="lake_formation_config with enabled=True requires role_arn to be specified"
        ):
            FeatureGroupManager.create(
                feature_group_name=feature_group_name,
                record_identifier_feature_name=record_id_feature,
                event_time_feature_name=event_time_feature,
                feature_definitions=feature_definitions,
                offline_store_config=offline_store_config,
                lake_formation_config=lf_config,
                # role_arn not provided
            )


    @pytest.mark.parametrize(
        "feature_group_name,record_id_feature,event_time_feature,role_arn,s3_uri,database,table,use_slr",
        [
            ("test-fg", "record_id", "event_time", "TestRole", "path1", "db1", "table1", True),
            ("my_feature_group", "id", "timestamp", "ExecutionRole", "data/features", "feature_db", "feature_table", False),
            ("fg123", "identifier", "time", "MyRole123", "ml/features/v1", "analytics", "features_v1", True),
        ],
    )
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_use_service_linked_role_extraction_from_config(
        self,
        mock_enable_lf,
        mock_wait,
        mock_get,
        mock_get_client,
        feature_group_name,
        record_id_feature,
        event_time_feature,
        role_arn,
        s3_uri,
        database,
        table,
        use_slr,
    ):
        """
        Test that use_service_linked_role is correctly extracted from lake_formation_config.

        Verifies:
        - use_service_linked_role defaults to True when not specified
        - use_service_linked_role is passed correctly to enable_lake_formation()
        """
        from sagemaker.core.shapes import (
            FeatureDefinition,
            OfflineStoreConfig,
            S3StorageConfig,
            DataCatalogConfig,
        )

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        # Mock the get method to return a feature group
        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.feature_group_name = feature_group_name
        mock_fg.wait_for_status = mock_wait
        mock_fg.enable_lake_formation = mock_enable_lf
        mock_get.return_value = mock_fg

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Create offline store config
        offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri=f"s3://test-bucket/{s3_uri}"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database=database, table_name=table
            ),
        )

        # Build LakeFormationConfig with use_service_linked_role
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True
        lf_config.use_service_linked_role = use_slr
        # When use_service_linked_role is False, registration_role_arn is required
        expected_registration_role = None
        if not use_slr:
            lf_config.registration_role_arn = "arn:aws:iam::123456789012:role/LFRegistrationRole"
            expected_registration_role = "arn:aws:iam::123456789012:role/LFRegistrationRole"

        # Create with lake_formation_config
        result = FeatureGroupManager.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=f"arn:aws:iam::123456789012:role/{role_arn}",
            lake_formation_config=lf_config,
        )

        # Verify enable_lake_formation was called with correct use_service_linked_role value
        mock_enable_lf.assert_called_once_with(
            session=None,
            region=None,
            use_service_linked_role=use_slr,
            registration_role_arn=expected_registration_role,
            disable_hybrid_access_mode=False,
        )
        # Verify the feature group was returned
        assert result == mock_fg


class TestDisableHybridAccessMode:
    """Tests for IAM principal revocation being always-on in enable_lake_formation."""

    def setup_method(self):
        """Set up test fixtures."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        self.fg = FeatureGroupManager(feature_group_name="test-fg")
        self.fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        self.fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        self.fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        self.fg.feature_group_status = "Created"

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_revoke_called_when_disable_hybrid_access_mode_true(
        self, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that IAMAllowedPrincipal is revoked when disable_hybrid_access_mode=True."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        result = self.fg.enable_lake_formation(disable_hybrid_access_mode=True)

        mock_revoke.assert_called_once()
        assert result["hybrid_access_mode_disabled"] is True

    @patch("builtins.input", return_value="y")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_revoke_not_called_when_disable_hybrid_access_mode_false(
        self, mock_revoke, mock_grant, mock_register, mock_refresh, mock_input
    ):
        """Test that IAMAllowedPrincipal is NOT revoked when disable_hybrid_access_mode=False."""
        mock_register.return_value = True
        mock_grant.return_value = True

        result = self.fg.enable_lake_formation(disable_hybrid_access_mode=False)

        mock_revoke.assert_not_called()
        assert result["hybrid_access_mode_disabled"] is False

    @patch("builtins.input", return_value="n")
    @patch.object(FeatureGroupManager, "refresh")
    def test_raises_error_when_user_declines_hybrid_access_prompt(
        self, mock_refresh, mock_input
    ):
        """Test that RuntimeError is raised when user declines the hybrid access prompt."""
        with pytest.raises(RuntimeError, match="User chose not to proceed"):
            self.fg.enable_lake_formation(disable_hybrid_access_mode=False)

    @patch("builtins.input", return_value="")
    @patch.object(FeatureGroupManager, "refresh")
    def test_raises_error_when_user_enters_empty_at_hybrid_access_prompt(
        self, mock_refresh, mock_input
    ):
        """Test that RuntimeError is raised when user enters empty string at prompt."""
        with pytest.raises(RuntimeError, match="User chose not to proceed"):
            self.fg.enable_lake_formation(disable_hybrid_access_mode=False)


class TestCreateWithLakeFormationDisableHybridAccessMode:
    """Tests for create() passing disable_hybrid_access_mode from config."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_enable_lake_formation_called_with_disable_hybrid_access_mode(
        self, mock_enable_lf, mock_wait, mock_get, mock_get_client
    ):
        """Test that create() passes disable_hybrid_access_mode from config to enable_lake_formation."""
        from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig, S3StorageConfig

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.wait_for_status = mock_wait
        mock_fg.enable_lake_formation = mock_enable_lf
        mock_get.return_value = mock_fg

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True
        lf_config.disable_hybrid_access_mode = True

        FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path")
            ),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            lake_formation_config=lf_config,
        )

        mock_enable_lf.assert_called_once_with(
            session=None,
            region=None,
            use_service_linked_role=True,
            registration_role_arn=None,
            disable_hybrid_access_mode=True,
        )


class TestLakeFormationConfigDefaults:
    """Tests for LakeFormationConfig default values."""

    def test_has_expected_fields_only(self):
        """Test that LakeFormationConfig has only the expected fields."""
        config = LakeFormationConfig(disable_hybrid_access_mode=True)
        assert config.enabled is False
        assert config.use_service_linked_role is True
        assert config.registration_role_arn is None
        assert config.disable_hybrid_access_mode is True

    def test_disable_hybrid_access_mode_is_required(self):
        """Test that disable_hybrid_access_mode is a required field."""
        with pytest.raises(Exception):
            LakeFormationConfig()


class TestExtractAccountIdFromArn:
    """Tests for _extract_account_id_from_arn static method."""

    def test_extracts_account_id_from_sagemaker_arn(self):
        """Test extracting account ID from a SageMaker Feature Group ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/my-feature-group"
        result = FeatureGroupManager._extract_account_id_from_arn(arn)
        assert result == "123456789012"

    def test_raises_value_error_for_invalid_arn_too_few_parts(self):
        """Test that ValueError is raised for ARN with fewer than 5 colon-separated parts."""
        invalid_arn = "arn:aws:sagemaker:us-west-2"  # Only 4 parts
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroupManager._extract_account_id_from_arn(invalid_arn)

    def test_raises_value_error_for_empty_string(self):
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroupManager._extract_account_id_from_arn("")

    def test_raises_value_error_for_non_arn_string(self):
        """Test that ValueError is raised for non-ARN string."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroupManager._extract_account_id_from_arn("not-an-arn")

    def test_raises_value_error_for_s3_uri(self):
        """Test that ValueError is raised for S3 URI (not ARN)."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroupManager._extract_account_id_from_arn("s3://my-bucket/my-prefix")

    def test_handles_arn_with_resource_path(self):
        """Test extracting account ID from ARN with complex resource path."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/my-fg/version/1"
        result = FeatureGroupManager._extract_account_id_from_arn(arn)
        assert result == "123456789012"


class TestGetLakeFormationServiceLinkedRoleArn:
    """Tests for _get_lake_formation_service_linked_role_arn static method."""

    def test_generates_correct_service_linked_role_arn(self):
        """Test that the method generates the correct service-linked role ARN format."""
        account_id = "123456789012"
        result = FeatureGroupManager._get_lake_formation_service_linked_role_arn(account_id)
        expected = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        assert result == expected

    def test_uses_region_for_partition(self):
        """Test that region is used to determine partition."""
        account_id = "123456789012"
        result = FeatureGroupManager._get_lake_formation_service_linked_role_arn(account_id, region="cn-north-1")
        assert result.startswith("arn:aws-cn:iam::")



class TestGenerateS3DenyStatements:
    """Tests for _generate_s3_deny_statements method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._generate_s3_deny_statements = FeatureGroupManager._generate_s3_deny_statements.__get__(self.fg)

    def test_returns_list_not_dict(self):
        """Test that the method returns a list, not a dict."""
        result = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )
        assert isinstance(result, list)
        assert not isinstance(result, dict)

    def test_policy_includes_correct_bucket_arn_in_object_statement(self):
        """Test that the statements include correct bucket ARN and prefix in object actions statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        object_statement = statements[0]
        expected_resource = f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"
        assert object_statement["Resource"] == expected_resource

    def test_policy_includes_correct_bucket_arn_in_list_statement(self):
        """Test that the statements include correct bucket ARN in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        list_statement = statements[1]
        expected_resource = f"arn:aws:s3:::{bucket_name}"
        assert list_statement["Resource"] == expected_resource

    def test_policy_includes_correct_prefix_condition_in_list_statement(self):
        """Test that the statements include correct prefix condition in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        list_statement = statements[1]
        expected_prefix = f"{s3_prefix}/*"
        assert list_statement["Condition"]["StringLike"]["s3:prefix"] == expected_prefix

    def test_policy_preserves_bucket_name_exactly(self):
        """Test that bucket name is preserved exactly without modification."""
        test_cases = [
            "simple-bucket",
            "bucket.with.dots",
            "bucket-with-dashes-123",
            "mybucket",
            "a" * 63,
        ]

        for bucket_name in test_cases:
            statements = self.fg._generate_s3_deny_statements(
                bucket_name=bucket_name,
                s3_prefix="prefix",
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            assert bucket_name in statements[0]["Resource"]
            assert bucket_name in statements[1]["Resource"]

    def test_policy_preserves_prefix_exactly(self):
        """Test that S3 prefix is preserved exactly without modification."""
        test_cases = [
            "simple-prefix",
            "path/to/data",
            "feature-store/account-id/region/feature-group-name",
            "deep/nested/path/structure/data",
            "prefix_with_underscores",
            "prefix-with-dashes",
        ]

        for s3_prefix in test_cases:
            statements = self.fg._generate_s3_deny_statements(
                bucket_name="test-bucket",
                s3_prefix=s3_prefix,
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            assert f"{s3_prefix}/*" in statements[0]["Resource"]
            assert statements[1]["Condition"]["StringLike"]["s3:prefix"] == f"{s3_prefix}/*"

    def test_policy_has_correct_s3_arn_format(self):
        """Test that the statements use correct S3 ARN format (arn:aws:s3:::bucket/path)."""
        bucket_name = "test-bucket"
        s3_prefix = "test/prefix"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        object_resource = statements[0]["Resource"]
        assert object_resource.startswith("arn:aws:s3:::")
        assert object_resource == f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"

        list_resource = statements[1]["Resource"]
        assert list_resource.startswith("arn:aws:s3:::")
        assert list_resource == f"arn:aws:s3:::{bucket_name}"

    def test_policy_structure_validation(self):
        """Test that the statements have correct structure."""
        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        assert isinstance(statements, list)
        assert len(statements) == 2

        object_statement = statements[0]
        assert object_statement["Sid"] == "DenyFSObjectAccess_prefix"
        assert object_statement["Effect"] == "Deny"
        assert object_statement["Principal"] == "*"
        assert "Condition" in object_statement
        assert "StringNotEquals" in object_statement["Condition"]

        list_statement = statements[1]
        assert list_statement["Sid"] == "DenyFSListAccess_prefix"
        assert list_statement["Effect"] == "Deny"
        assert list_statement["Principal"] == "*"
        assert "Condition" in list_statement
        assert "StringLike" in list_statement["Condition"]
        assert "StringNotEquals" in list_statement["Condition"]

    def test_policy_includes_both_principals_in_allowed_list(self):
        """Test that both Lake Formation role and Feature Store role are in allowed principals."""
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        object_principals = statements[0]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in object_principals
        assert fs_role_arn in object_principals
        assert len(object_principals) == 2

        list_principals = statements[1]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in list_principals
        assert fs_role_arn in list_principals
        assert len(list_principals) == 2

    def test_policy_has_correct_actions_in_each_statement(self):
        """Test that each statement has the correct S3 actions."""
        statements = self.fg._generate_s3_deny_statements(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        object_actions = statements[0]["Action"]
        assert "s3:GetObject" in object_actions
        assert "s3:PutObject" in object_actions
        assert "s3:DeleteObject" in object_actions
        assert len(object_actions) == 3

        list_action = statements[1]["Action"]
        assert list_action == "s3:ListBucket"



class TestEnableLakeFormationServiceLinkedRoleInPolicy:
    """Tests for service-linked role ARN usage in Phase 4 deny policy generation."""

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_uses_service_linked_role_arn_when_use_service_linked_role_true(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that Phase 4 uses the auto-generated service-linked role ARN."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        fg.enable_lake_formation(use_service_linked_role=True, disable_hybrid_access_mode=True)

        expected_slr_arn = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert call_kwargs["feature_store_role_arn"] == fg.role_arn

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_uses_service_linked_role_arn_by_default(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that Phase 4 uses the service-linked role ARN by default."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::987654321098:role/MyFeatureStoreRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-east-1:987654321098:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        fg.enable_lake_formation(disable_hybrid_access_mode=True)

        expected_slr_arn = "arn:aws:iam::987654321098:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_service_linked_role_arn_uses_correct_account_id(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that the service-linked role ARN uses the correct account ID."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        account_id = "111222333444"
        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = f"arn:aws:iam::{account_id}:role/FeatureStoreRole"
        fg.feature_group_arn = f"arn:aws:sagemaker:us-west-2:{account_id}:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        fg.enable_lake_formation(use_service_linked_role=True, disable_hybrid_access_mode=True)

        expected_slr_arn = f"arn:aws:iam::{account_id}:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert account_id in call_kwargs["lake_formation_role_arn"]



class TestRegistrationRoleArnUsedWhenServiceLinkedRoleFalse:
    """Tests for verifying registration_role_arn is used when use_service_linked_role=False."""

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_uses_registration_role_arn_when_use_service_linked_role_false(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that registration_role_arn is used in Phase 4 when use_service_linked_role=False."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            disable_hybrid_access_mode=True,
        )

        mock_generate.assert_called_once()
        call_kwargs = mock_generate.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == custom_registration_role

        service_linked_role_pattern = "aws-service-role/lakeformation.amazonaws.com"
        assert service_linked_role_pattern not in call_kwargs["lake_formation_role_arn"]

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_registration_role_arn_passed_to_s3_registration(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that registration_role_arn is passed to _register_s3_with_lake_formation."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            disable_hybrid_access_mode=True,
        )

        mock_register.assert_called_once()
        call_args = mock_register.call_args
        assert call_args[1]["use_service_linked_role"] == False
        assert call_args[1]["role_arn"] == custom_registration_role

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_statements")
    def test_different_registration_role_arns_produce_different_policies(
        self,
        mock_generate,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """Test that different registration_role_arn values result in different role ARNs in Phase 4."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate.return_value = []

        first_role = "arn:aws:iam::123456789012:role/FirstLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=first_role,
            disable_hybrid_access_mode=True,
        )
        first_call_kwargs = mock_generate.call_args[1]
        first_lf_role = first_call_kwargs["lake_formation_role_arn"]

        mock_generate.reset_mock()
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        second_role = "arn:aws:iam::123456789012:role/SecondLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=second_role,
            disable_hybrid_access_mode=True,
        )
        second_call_kwargs = mock_generate.call_args[1]
        second_lf_role = second_call_kwargs["lake_formation_role_arn"]

        assert first_lf_role == first_role
        assert second_lf_role == second_role
        assert first_lf_role != second_lf_role



class TestFeatureGroupManagerReturnType:
    """Tests to verify create() and get() return FeatureGroupManager instances."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_create_returns_feature_group_manager_instance(self, mock_get_client):
        """Test that create() returns a FeatureGroupManager, not a FeatureGroup."""
        from sagemaker.core.shapes import FeatureDefinition
        from sagemaker.core.resources import FeatureGroup

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [{"FeatureName": "record_id", "FeatureType": "String"}],
            "FeatureGroupStatus": "Created",
        }
        mock_get_client.return_value = mock_client

        result = FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[FeatureDefinition(feature_name="record_id", feature_type="String")],
        )

        assert isinstance(result, FeatureGroupManager)
        assert not type(result) is FeatureGroup

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_get_returns_feature_group_manager_instance(self, mock_get_client):
        """Test that get() returns a FeatureGroupManager, not a FeatureGroup."""
        from sagemaker.core.resources import FeatureGroup

        mock_client = MagicMock()
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [{"FeatureName": "record_id", "FeatureType": "String"}],
            "FeatureGroupStatus": "Created",
        }
        mock_get_client.return_value = mock_client

        result = FeatureGroupManager.get(feature_group_name="test-fg")

        assert isinstance(result, FeatureGroupManager)
        assert not type(result) is FeatureGroup


class TestEnableLakeFormationIcebergTableFormat:
    """Tests for Iceberg table format S3 path handling in enable_lake_formation."""

    def setup_method(self):
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        self.fg = FeatureGroupManager(feature_group_name="test-fg")
        self.fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
            table_format="Iceberg",
        )
        self.fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        self.fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        self.fg.feature_group_status = "Created"

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_iceberg_strips_data_suffix_for_s3_registration(
        self, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that Iceberg tables register the parent S3 path (without /data suffix)."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        self.fg.enable_lake_formation(disable_hybrid_access_mode=True)

        # The registered S3 location should NOT end with /data
        call_args = mock_register.call_args
        registered_location = call_args[0][0]
        assert registered_location == "s3://test-bucket/resolved-path"
        assert not registered_location.endswith("/data")

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_non_iceberg_keeps_full_s3_path(
        self, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that non-Iceberg tables use the full resolved S3 URI."""
        self.fg.offline_store_config.table_format = None
        self.fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = (
            "s3://test-bucket/resolved-path/data"
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        self.fg.enable_lake_formation(disable_hybrid_access_mode=True)

        call_args = mock_register.call_args
        registered_location = call_args[0][0]
        assert registered_location == "s3://test-bucket/resolved-path/data"


class TestEnableLakeFormationMissingArn:
    """Tests for feature_group_arn validation in Phase 4."""

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_raises_error_when_feature_group_arn_is_none(
        self, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test ValueError when feature_group_arn is None (needed for Phase 4)."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        fg.feature_group_arn = None
        fg.feature_group_status = "Created"

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        with pytest.raises(ValueError, match="Feature Group ARN is required"):
            fg.enable_lake_formation(disable_hybrid_access_mode=True)


class TestEnableLakeFormationHappyPath:
    """Tests for the full happy-path return value of enable_lake_formation."""

    def setup_method(self):
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        self.fg = FeatureGroupManager(feature_group_name="test-fg")
        self.fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        self.fg.role_arn = "arn:aws:iam::123456789012:role/TestRole"
        self.fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        self.fg.feature_group_status = "Created"

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    def test_returns_all_true_on_success(
        self, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test full happy-path returns all phases as True."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        result = self.fg.enable_lake_formation(disable_hybrid_access_mode=True)

        assert result == {
            "s3_location_registered": True,
            "lf_permissions_granted": True,
            "hybrid_access_mode_disabled": True,
        }


class TestCreatePassesThroughSessionAndRegion:
    """Tests for session and region pass-through in create()."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_session_and_region_passed_to_enable_lake_formation(
        self, mock_enable_lf, mock_wait, mock_get, mock_get_client
    ):
        """Test that session and region are forwarded to enable_lake_formation."""
        from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig, S3StorageConfig

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.wait_for_status = mock_wait
        mock_fg.enable_lake_formation = mock_enable_lf
        mock_get.return_value = mock_fg

        mock_session = MagicMock(spec=Session)
        lf_config = LakeFormationConfig(disable_hybrid_access_mode=False)
        lf_config.enabled = True

        FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[FeatureDefinition(feature_name="record_id", feature_type="String")],
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path")
            ),
            role_arn="arn:aws:iam::123456789012:role/TestRole",
            lake_formation_config=lf_config,
            session=mock_session,
            region="eu-west-1",
        )

        mock_enable_lf.assert_called_once_with(
            session=mock_session,
            region="eu-west-1",
            use_service_linked_role=True,
            registration_role_arn=None,
            disable_hybrid_access_mode=False,
        )


class TestRegionInferenceFromSession:
    """Tests for region inference from session in LF methods."""

    def test_register_s3_infers_region_from_session(self):
        """Test that _register_s3_with_lake_formation infers region from session."""
        fg = MagicMock(spec=FeatureGroupManager)
        fg._s3_uri_to_arn = FeatureGroupManager._s3_uri_to_arn
        fg._register_s3_with_lake_formation = (
            FeatureGroupManager._register_s3_with_lake_formation.__get__(fg)
        )
        mock_client = MagicMock()
        mock_client.register_resource.return_value = {}
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        mock_session = MagicMock()
        mock_session.region_name = "ap-southeast-1"

        fg._register_s3_with_lake_formation("s3://bucket/path", session=mock_session)

        # Region should be inferred and passed to client
        fg._get_lake_formation_client.assert_called_with(mock_session, "ap-southeast-1")

    def test_revoke_iam_infers_region_from_session(self):
        """Test that _revoke_iam_allowed_principal infers region from session."""
        fg = MagicMock(spec=FeatureGroupManager)
        fg._revoke_iam_allowed_principal = (
            FeatureGroupManager._revoke_iam_allowed_principal.__get__(fg)
        )
        mock_client = MagicMock()
        mock_client.list_permissions.return_value = {"PrincipalResourcePermissions": []}
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        mock_session = MagicMock()
        mock_session.region_name = "ap-southeast-1"

        fg._revoke_iam_allowed_principal("db", "table", session=mock_session)

        fg._get_lake_formation_client.assert_called_with(mock_session, "ap-southeast-1")

    def test_grant_permissions_infers_region_from_session(self):
        """Test that _grant_lake_formation_permissions infers region from session."""
        fg = MagicMock(spec=FeatureGroupManager)
        fg._grant_lake_formation_permissions = (
            FeatureGroupManager._grant_lake_formation_permissions.__get__(fg)
        )
        mock_client = MagicMock()
        mock_client.grant_permissions.return_value = {}
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        mock_session = MagicMock()
        mock_session.region_name = "ap-southeast-1"

        fg._grant_lake_formation_permissions(
            "arn:aws:iam::123456789012:role/Role", "db", "table", session=mock_session
        )

        fg._get_lake_formation_client.assert_called_with(mock_session, "ap-southeast-1")
