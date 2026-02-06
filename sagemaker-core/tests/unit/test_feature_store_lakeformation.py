"""Unit tests for Lake Formation integration with FeatureGroup."""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from sagemaker.core.resources import FeatureGroup


class TestS3UriToArn:
    """Tests for _s3_uri_to_arn static method."""

    def test_converts_s3_uri_to_arn(self):
        """Test S3 URI is converted to ARN format."""
        uri = "s3://my-bucket/my-prefix/data"
        result = FeatureGroup._s3_uri_to_arn(uri)
        assert result == "arn:aws:s3:::my-bucket/my-prefix/data"

    def test_handles_bucket_only_uri(self):
        """Test S3 URI with bucket only."""
        uri = "s3://my-bucket"
        result = FeatureGroup._s3_uri_to_arn(uri)
        assert result == "arn:aws:s3:::my-bucket"

    def test_returns_arn_unchanged(self):
        """Test ARN input is returned unchanged (idempotent)."""
        arn = "arn:aws:s3:::my-bucket/path"
        result = FeatureGroup._s3_uri_to_arn(arn)
        assert result == arn


class TestGetLakeFormationClient:
    """Tests for _get_lake_formation_client method."""

    @patch("sagemaker.core.resources.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroup)
        fg._get_lake_formation_client = FeatureGroup._get_lake_formation_client.__get__(fg)

        client = fg._get_lake_formation_client(region="us-west-2")

        mock_session.client.assert_called_with("lakeformation", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroup)
        fg._get_lake_formation_client = FeatureGroup._get_lake_formation_client.__get__(fg)

        client = fg._get_lake_formation_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("lakeformation", region_name="us-west-2")
        assert client == mock_client


class TestRegisterS3WithLakeFormation:
    """Tests for _register_s3_with_lake_formation method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroup)
        self.fg._s3_uri_to_arn = FeatureGroup._s3_uri_to_arn
        self.fg._register_s3_with_lake_formation = (
            FeatureGroup._register_s3_with_lake_formation.__get__(self.fg)
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
        self.fg = MagicMock(spec=FeatureGroup)
        self.fg._revoke_iam_allowed_principal = FeatureGroup._revoke_iam_allowed_principal.__get__(
            self.fg
        )
        self.mock_client = MagicMock()
        self.fg._get_lake_formation_client = MagicMock(return_value=self.mock_client)

    def test_successful_revocation_returns_true(self):
        """Test successful revocation returns True."""
        self.mock_client.revoke_permissions.return_value = {}

        result = self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert result is True
        self.mock_client.revoke_permissions.assert_called_once()

    def test_revoke_permissions_call_structure(self):
        """Test that revoke_permissions is called with correct parameters."""
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

    def test_invalid_input_exception_returns_true(self):
        """Test InvalidInputException is handled gracefully (permissions may not exist)."""
        self.mock_client.revoke_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidInputException", "Message": "Permissions not found"}},
            "RevokePermissions",
        )

        result = self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert result is True

    def test_other_exceptions_are_propagated(self):
        """Test non-InvalidInputException errors are propagated."""
        self.mock_client.revoke_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "RevokePermissions",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._revoke_iam_allowed_principal("test_database", "test_table")

        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"

    def test_passes_session_and_region_to_client(self):
        """Test session and region are passed to get_lake_formation_client."""
        self.mock_client.revoke_permissions.return_value = {}
        mock_session = MagicMock()

        self.fg._revoke_iam_allowed_principal(
            "test_database", "test_table", session=mock_session, region="us-west-2"
        )

        self.fg._get_lake_formation_client.assert_called_with(mock_session, "us-west-2")


class TestGrantLakeFormationPermissions:
    """Tests for _grant_lake_formation_permissions method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroup)
        self.fg._grant_lake_formation_permissions = (
            FeatureGroup._grant_lake_formation_permissions.__get__(self.fg)
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

    @patch.object(FeatureGroup, "refresh")
    def test_raises_error_when_no_offline_store(self, mock_refresh):
        """Test that enable_lake_formation raises ValueError when no offline store is configured."""
        fg = FeatureGroup(feature_group_name="test-fg")
        fg.offline_store_config = None
        fg.feature_group_status = "Created"

        with pytest.raises(ValueError, match="does not have an offline store configured"):
            fg.enable_lake_formation()

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroup, "refresh")
    def test_raises_error_when_no_role_arn(self, mock_refresh):
        """Test that enable_lake_formation raises ValueError when no role_arn is configured."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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
            fg.enable_lake_formation()

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroup, "refresh")
    def test_raises_error_when_invalid_status(self, mock_refresh):
        """Test enable_lake_formation raises ValueError when Feature Group not in Created status."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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
            fg.enable_lake_formation()

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroup, "wait_for_status")
    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    def test_wait_for_active_calls_wait_for_status(
        self, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
    ):
        """Test that wait_for_active=True calls wait_for_status with 'Created' target."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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
        fg.feature_group_status = "Created"

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call with wait_for_active=True
        fg.enable_lake_formation(wait_for_active=True)

        # Verify wait_for_status was called with "Created"
        mock_wait.assert_called_once_with(target_status="Created")
        # Verify refresh was called after wait
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroup, "wait_for_status")
    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    def test_wait_for_active_false_does_not_call_wait(
        self, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
    ):
        """Test that wait_for_active=False does not call wait_for_status."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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
        fg.feature_group_status = "Created"

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call with wait_for_active=False (default)
        fg.enable_lake_formation(wait_for_active=False)

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
    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
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

        fg = FeatureGroup(feature_group_name=feature_group_name)
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
            fg.enable_lake_formation()

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
            fg.enable_lake_formation()

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
            fg.enable_lake_formation()

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
        fg = MagicMock(spec=FeatureGroup)
        fg._s3_uri_to_arn = FeatureGroup._s3_uri_to_arn
        fg._register_s3_with_lake_formation = FeatureGroup._register_s3_with_lake_formation.__get__(
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

        For any error from Lake Formation's revoke_permissions API that is not
        InvalidInputException, the error should be propagated to the caller unchanged.

        """
        fg = MagicMock(spec=FeatureGroup)
        fg._revoke_iam_allowed_principal = FeatureGroup._revoke_iam_allowed_principal.__get__(fg)
        mock_client = MagicMock()
        fg._get_lake_formation_client = MagicMock(return_value=mock_client)

        # Configure mock to raise an unhandled error
        mock_client.revoke_permissions.side_effect = botocore.exceptions.ClientError(
            {
                "Error": {
                    "Code": "AccessDeniedException",
                    "Message": "User does not have permission",
                }
            },
            "RevokePermissions",
        )

        # Verify the exception is propagated unchanged
        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            fg._revoke_iam_allowed_principal("test_database", "test_table")

        # Verify error details are preserved
        assert exc_info.value.response["Error"]["Code"] == "AccessDeniedException"
        assert exc_info.value.response["Error"]["Message"] == "User does not have permission"
        assert exc_info.value.operation_name == "RevokePermissions"

    def test_grant_permissions_propagates_unhandled_exceptions(self):
        """
        Non-InvalidInput Errors Propagate from Permission Grant

        For any error from Lake Formation's grant_permissions API that is not
        InvalidInputException, the error should be propagated to the caller unchanged.

        """
        fg = MagicMock(spec=FeatureGroup)
        fg._grant_lake_formation_permissions = (
            FeatureGroup._grant_lake_formation_permissions.__get__(fg)
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
        fg = MagicMock(spec=FeatureGroup)
        fg._s3_uri_to_arn = FeatureGroup._s3_uri_to_arn
        fg._register_s3_with_lake_formation = FeatureGroup._register_s3_with_lake_formation.__get__(
            fg
        )
        fg._revoke_iam_allowed_principal = FeatureGroup._revoke_iam_allowed_principal.__get__(fg)
        fg._grant_lake_formation_permissions = (
            FeatureGroup._grant_lake_formation_permissions.__get__(fg)
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

        # Test InvalidInputException is handled for revoke (not propagated)
        mock_client.revoke_permissions.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "InvalidInputException", "Message": "Invalid input"}},
            "RevokePermissions",
        )
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
    @patch.object(FeatureGroup, "get")
    @patch.object(FeatureGroup, "wait_for_status")
    @patch.object(FeatureGroup, "enable_lake_formation")
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

        For any call to FeatureGroup.create() where enable_lake_formation is False or not specified,
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
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_fg.feature_group_name = feature_group_name
        mock_get.return_value = mock_fg

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Test 1: enable_lake_formation=False (explicit)
        result = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            enable_lake_formation=False,
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

        # Test 2: enable_lake_formation not specified (defaults to False)
        result = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            # enable_lake_formation not specified, should default to False
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
    @patch.object(FeatureGroup, "get")
    @patch.object(FeatureGroup, "wait_for_status")
    @patch.object(FeatureGroup, "enable_lake_formation")
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
        Test that enable_lake_formation is called when enable_lake_formation=True.

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
        mock_fg = MagicMock(spec=FeatureGroup)
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

        # Create with enable_lake_formation=True
        result = FeatureGroup.create(
            feature_group_name=feature_group_name,
            record_identifier_feature_name=record_id_feature,
            event_time_feature_name=event_time_feature,
            feature_definitions=feature_definitions,
            offline_store_config=offline_store_config,
            role_arn=f"arn:aws:iam::123456789012:role/{role_arn}",
            enable_lake_formation=True,
        )

        # Verify wait_for_status was called with "Created"
        mock_wait.assert_called_once_with(target_status="Created")
        # Verify enable_lake_formation was called
        mock_enable_lf.assert_called_once()
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
    def test_validation_error_when_enable_lake_formation_without_offline_store(
        self, mock_get_client, feature_group_name, record_id_feature, event_time_feature
    ):
        """Test create() raises ValueError when enable_lake_formation=True without offline_store."""
        from sagemaker.core.shapes import FeatureDefinition

        # Mock the SageMaker client
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Create feature definitions
        feature_definitions = [
            FeatureDefinition(feature_name=record_id_feature, feature_type="String"),
            FeatureDefinition(feature_name=event_time_feature, feature_type="String"),
        ]

        # Test with enable_lake_formation=True but no offline_store_config
        with pytest.raises(
            ValueError,
            match="enable_lake_formation=True requires offline_store_config to be configured",
        ):
            FeatureGroup.create(
                feature_group_name=feature_group_name,
                record_identifier_feature_name=record_id_feature,
                event_time_feature_name=event_time_feature,
                feature_definitions=feature_definitions,
                enable_lake_formation=True,
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
    def test_validation_error_when_enable_lake_formation_without_role_arn(
        self,
        mock_get_client,
        feature_group_name,
        record_id_feature,
        event_time_feature,
        s3_uri,
        database,
        table,
    ):
        """Test create() raises ValueError when enable_lake_formation=True without role_arn."""
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

        # Test with enable_lake_formation=True but no role_arn
        with pytest.raises(
            ValueError, match="enable_lake_formation=True requires role_arn to be specified"
        ):
            FeatureGroup.create(
                feature_group_name=feature_group_name,
                record_identifier_feature_name=record_id_feature,
                event_time_feature_name=event_time_feature,
                feature_definitions=feature_definitions,
                offline_store_config=offline_store_config,
                enable_lake_formation=True,
                # role_arn not provided
            )
