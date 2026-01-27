"""Unit tests for Lake Formation integration with FeatureGroup."""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from sagemaker.mlops.feature_store import FeatureGroup, LakeFormationConfig


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

    def test_uses_region_for_partition(self):
        """Test that region is used to determine partition."""
        uri = "s3://my-bucket/path"
        result = FeatureGroup._s3_uri_to_arn(uri, region="cn-north-1")
        assert result.startswith("arn:aws-cn:s3:::")



class TestGetLakeFormationClient:
    """Tests for _get_lake_formation_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group.Session")
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
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
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
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
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

        For any call to FeatureGroup.create() where lake_formation_config is None or has enabled=False,
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

        # Test 1: lake_formation_config with enabled=False (explicit)
        lf_config = LakeFormationConfig()
        lf_config.enabled = False
        result = FeatureGroup.create(
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
        result = FeatureGroup.create(
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

        # Create LakeFormationConfig with enabled=True
        lf_config = LakeFormationConfig()
        lf_config.enabled = True

        # Create with lake_formation_config enabled=True
        result = FeatureGroup.create(
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
            show_s3_policy=False,
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
        lf_config = LakeFormationConfig()
        lf_config.enabled = True

        # Test with lake_formation_config enabled=True but no offline_store_config
        with pytest.raises(
            ValueError,
            match="lake_formation_config with enabled=True requires offline_store_config to be configured",
        ):
            FeatureGroup.create(
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
        lf_config = LakeFormationConfig()
        lf_config.enabled = True

        # Test with lake_formation_config enabled=True but no role_arn
        with pytest.raises(
            ValueError, match="lake_formation_config with enabled=True requires role_arn to be specified"
        ):
            FeatureGroup.create(
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
    @patch.object(FeatureGroup, "get")
    @patch.object(FeatureGroup, "wait_for_status")
    @patch.object(FeatureGroup, "enable_lake_formation")
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

        # Build LakeFormationConfig with use_service_linked_role
        lf_config = LakeFormationConfig()
        lf_config.enabled = True
        lf_config.use_service_linked_role = use_slr
        # When use_service_linked_role is False, registration_role_arn is required
        expected_registration_role = None
        if not use_slr:
            lf_config.registration_role_arn = "arn:aws:iam::123456789012:role/LFRegistrationRole"
            expected_registration_role = "arn:aws:iam::123456789012:role/LFRegistrationRole"

        # Create with lake_formation_config
        result = FeatureGroup.create(
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
            show_s3_policy=False,
        )
        # Verify the feature group was returned
        assert result == mock_fg


class TestExtractAccountIdFromArn:
    """Tests for _extract_account_id_from_arn static method."""

    def test_extracts_account_id_from_sagemaker_arn(self):
        """Test extracting account ID from a SageMaker Feature Group ARN."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/my-feature-group"
        result = FeatureGroup._extract_account_id_from_arn(arn)
        assert result == "123456789012"

    def test_raises_value_error_for_invalid_arn_too_few_parts(self):
        """Test that ValueError is raised for ARN with fewer than 5 colon-separated parts."""
        invalid_arn = "arn:aws:sagemaker:us-west-2"  # Only 4 parts
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroup._extract_account_id_from_arn(invalid_arn)

    def test_raises_value_error_for_empty_string(self):
        """Test that ValueError is raised for empty string."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroup._extract_account_id_from_arn("")

    def test_raises_value_error_for_non_arn_string(self):
        """Test that ValueError is raised for non-ARN string."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroup._extract_account_id_from_arn("not-an-arn")

    def test_raises_value_error_for_s3_uri(self):
        """Test that ValueError is raised for S3 URI (not ARN)."""
        with pytest.raises(ValueError, match="Invalid ARN format"):
            FeatureGroup._extract_account_id_from_arn("s3://my-bucket/my-prefix")

    def test_handles_arn_with_resource_path(self):
        """Test extracting account ID from ARN with complex resource path."""
        arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/my-fg/version/1"
        result = FeatureGroup._extract_account_id_from_arn(arn)
        assert result == "123456789012"


class TestGetLakeFormationServiceLinkedRoleArn:
    """Tests for _get_lake_formation_service_linked_role_arn static method."""

    def test_generates_correct_service_linked_role_arn(self):
        """Test that the method generates the correct service-linked role ARN format."""
        account_id = "123456789012"
        result = FeatureGroup._get_lake_formation_service_linked_role_arn(account_id)
        expected = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        assert result == expected

    def test_uses_region_for_partition(self):
        """Test that region is used to determine partition."""
        account_id = "123456789012"
        result = FeatureGroup._get_lake_formation_service_linked_role_arn(account_id, region="cn-north-1")
        assert result.startswith("arn:aws-cn:iam::")



class TestGenerateS3DenyPolicy:
    """Tests for _generate_s3_deny_policy method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroup)
        self.fg._generate_s3_deny_policy = FeatureGroup._generate_s3_deny_policy.__get__(self.fg)

    def test_policy_includes_correct_bucket_arn_in_object_statement(self):
        """Test that the policy includes correct bucket ARN and prefix in object actions statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        policy = self.fg._generate_s3_deny_policy(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        # Verify the object actions statement has correct Resource ARN
        object_statement = policy["Statement"][0]
        expected_resource = f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"
        assert object_statement["Resource"] == expected_resource

    def test_policy_includes_correct_bucket_arn_in_list_statement(self):
        """Test that the policy includes correct bucket ARN in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        policy = self.fg._generate_s3_deny_policy(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        # Verify the ListBucket statement has correct Resource ARN (bucket only)
        list_statement = policy["Statement"][1]
        expected_resource = f"arn:aws:s3:::{bucket_name}"
        assert list_statement["Resource"] == expected_resource

    def test_policy_includes_correct_prefix_condition_in_list_statement(self):
        """Test that the policy includes correct prefix condition in ListBucket statement."""
        bucket_name = "my-feature-store-bucket"
        s3_prefix = "feature-store/data/my-feature-group"
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        policy = self.fg._generate_s3_deny_policy(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        # Verify the ListBucket statement has correct prefix condition
        list_statement = policy["Statement"][1]
        expected_prefix = f"{s3_prefix}/*"
        assert list_statement["Condition"]["StringLike"]["s3:prefix"] == expected_prefix

    def test_policy_preserves_bucket_name_exactly(self):
        """Test that bucket name is preserved exactly without modification."""
        # Test with various bucket name formats
        test_cases = [
            "simple-bucket",
            "bucket.with.dots",
            "bucket-with-dashes-123",
            "mybucket",
            "a" * 63,  # Max bucket name length
        ]

        for bucket_name in test_cases:
            policy = self.fg._generate_s3_deny_policy(
                bucket_name=bucket_name,
                s3_prefix="prefix",
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            # Verify bucket name is preserved in both statements
            assert bucket_name in policy["Statement"][0]["Resource"]
            assert bucket_name in policy["Statement"][1]["Resource"]

    def test_policy_preserves_prefix_exactly(self):
        """Test that S3 prefix is preserved exactly without modification."""
        # Test with various prefix formats
        test_cases = [
            "simple-prefix",
            "path/to/data",
            "feature-store/account-id/region/feature-group-name",
            "deep/nested/path/structure/data",
            "prefix_with_underscores",
            "prefix-with-dashes",
        ]

        for s3_prefix in test_cases:
            policy = self.fg._generate_s3_deny_policy(
                bucket_name="test-bucket",
                s3_prefix=s3_prefix,
                lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
                feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
            )

            # Verify prefix is preserved in object statement Resource
            assert f"{s3_prefix}/*" in policy["Statement"][0]["Resource"]
            # Verify prefix is preserved in list statement Condition
            assert policy["Statement"][1]["Condition"]["StringLike"]["s3:prefix"] == f"{s3_prefix}/*"

    def test_policy_has_correct_s3_arn_format(self):
        """Test that the policy uses correct S3 ARN format (arn:aws:s3:::bucket/path)."""
        bucket_name = "test-bucket"
        s3_prefix = "test/prefix"

        policy = self.fg._generate_s3_deny_policy(
            bucket_name=bucket_name,
            s3_prefix=s3_prefix,
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        # Verify object statement Resource starts with correct ARN prefix
        object_resource = policy["Statement"][0]["Resource"]
        assert object_resource.startswith("arn:aws:s3:::")
        assert object_resource == f"arn:aws:s3:::{bucket_name}/{s3_prefix}/*"

        # Verify list statement Resource is bucket-only ARN
        list_resource = policy["Statement"][1]["Resource"]
        assert list_resource.startswith("arn:aws:s3:::")
        assert list_resource == f"arn:aws:s3:::{bucket_name}"

    def test_policy_structure_validation(self):
        """Test that the policy has correct overall structure."""
        policy = self.fg._generate_s3_deny_policy(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        # Verify policy version
        assert policy["Version"] == "2012-10-17"

        # Verify exactly two statements
        assert len(policy["Statement"]) == 2

        # Verify first statement structure (object actions)
        object_statement = policy["Statement"][0]
        assert object_statement["Sid"] == "DenyAllAccessToFeatureStorePrefixExceptAllowedPrincipals"
        assert object_statement["Effect"] == "Deny"
        assert object_statement["Principal"] == "*"
        assert "Condition" in object_statement
        assert "StringNotEquals" in object_statement["Condition"]

        # Verify second statement structure (list bucket)
        list_statement = policy["Statement"][1]
        assert list_statement["Sid"] == "DenyListOnPrefixExceptAllowedPrincipals"
        assert list_statement["Effect"] == "Deny"
        assert list_statement["Principal"] == "*"
        assert "Condition" in list_statement
        assert "StringLike" in list_statement["Condition"]
        assert "StringNotEquals" in list_statement["Condition"]

    def test_policy_includes_both_principals_in_allowed_list(self):
        """Test that both Lake Formation role and Feature Store role are in allowed principals."""
        lf_role_arn = "arn:aws:iam::123456789012:role/LakeFormationRole"
        fs_role_arn = "arn:aws:iam::123456789012:role/FeatureStoreRole"

        policy = self.fg._generate_s3_deny_policy(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn=lf_role_arn,
            feature_store_role_arn=fs_role_arn,
        )

        # Verify both principals in object statement
        object_principals = policy["Statement"][0]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in object_principals
        assert fs_role_arn in object_principals
        assert len(object_principals) == 2

        # Verify both principals in list statement
        list_principals = policy["Statement"][1]["Condition"]["StringNotEquals"]["aws:PrincipalArn"]
        assert lf_role_arn in list_principals
        assert fs_role_arn in list_principals
        assert len(list_principals) == 2

    def test_policy_has_correct_actions_in_each_statement(self):
        """Test that each statement has the correct S3 actions."""
        policy = self.fg._generate_s3_deny_policy(
            bucket_name="test-bucket",
            s3_prefix="test/prefix",
            lake_formation_role_arn="arn:aws:iam::123456789012:role/LFRole",
            feature_store_role_arn="arn:aws:iam::123456789012:role/FSRole",
        )

        # Verify object statement has correct actions
        object_actions = policy["Statement"][0]["Action"]
        assert "s3:GetObject" in object_actions
        assert "s3:PutObject" in object_actions
        assert "s3:DeleteObject" in object_actions
        assert len(object_actions) == 3

        # Verify list statement has correct action
        list_action = policy["Statement"][1]["Action"]
        assert list_action == "s3:ListBucket"



class TestEnableLakeFormationServiceLinkedRoleInPolicy:
    """Tests for service-linked role ARN usage in S3 deny policy generation."""

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_uses_service_linked_role_arn_when_use_service_linked_role_true(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that enable_lake_formation uses the auto-generated service-linked role ARN
        when use_service_linked_role=True.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # Call with use_service_linked_role=True (default)
        fg.enable_lake_formation(use_service_linked_role=True, show_s3_policy=True)

        # Verify _generate_s3_deny_policy was called with the service-linked role ARN
        expected_slr_arn = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert call_kwargs["feature_store_role_arn"] == fg.role_arn

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_uses_service_linked_role_arn_by_default(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that enable_lake_formation uses the service-linked role ARN by default
        (when use_service_linked_role is not explicitly specified).
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # Call without specifying use_service_linked_role (should default to True)
        fg.enable_lake_formation(show_s3_policy=True)

        # Verify _generate_s3_deny_policy was called with the service-linked role ARN
        expected_slr_arn = "arn:aws:iam::987654321098:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_service_linked_role_arn_uses_correct_account_id(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the service-linked role ARN is generated with the correct account ID
        extracted from the Feature Group ARN.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Use a specific account ID to verify it's extracted correctly
        account_id = "111222333444"
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # Call with use_service_linked_role=True
        fg.enable_lake_formation(use_service_linked_role=True, show_s3_policy=True)

        # Verify the service-linked role ARN contains the correct account ID
        expected_slr_arn = f"arn:aws:iam::{account_id}:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert account_id in call_kwargs["lake_formation_role_arn"]



class TestRegistrationRoleArnUsedWhenServiceLinkedRoleFalse:
    """Tests for verifying registration_role_arn is used when use_service_linked_role=False."""

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_uses_registration_role_arn_when_use_service_linked_role_false(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that when use_service_linked_role=False, the registration_role_arn is used
        in the S3 deny policy instead of the auto-generated service-linked role ARN.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # Custom registration role ARN
        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        # Call with use_service_linked_role=False and registration_role_arn
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            show_s3_policy=True,
        )

        # Verify _generate_s3_deny_policy was called with the custom registration role ARN
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == custom_registration_role

        # Verify it's NOT the service-linked role ARN
        service_linked_role_pattern = "aws-service-role/lakeformation.amazonaws.com"
        assert service_linked_role_pattern not in call_kwargs["lake_formation_role_arn"]

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_registration_role_arn_passed_to_s3_registration(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that when use_service_linked_role=False, the registration_role_arn is also
        passed to _register_s3_with_lake_formation.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # Custom registration role ARN
        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        # Call with use_service_linked_role=False and registration_role_arn
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            show_s3_policy=True,
        )

        # Verify _register_s3_with_lake_formation was called with the correct parameters
        mock_register.assert_called_once()
        call_args = mock_register.call_args
        assert call_args[1]["use_service_linked_role"] == False
        assert call_args[1]["role_arn"] == custom_registration_role

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroup, "_generate_s3_deny_policy")
    @patch("builtins.print")
    def test_different_registration_role_arns_produce_different_policies(
        self,
        mock_print,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that different registration_role_arn values result in different
        lake_formation_role_arn values in the generated policy.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        # First call with one registration role
        first_role = "arn:aws:iam::123456789012:role/FirstLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=first_role,
            show_s3_policy=True,
        )

        first_call_kwargs = mock_generate_policy.call_args[1]
        first_lf_role = first_call_kwargs["lake_formation_role_arn"]

        # Reset mocks
        mock_generate_policy.reset_mock()
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        # Second call with different registration role
        second_role = "arn:aws:iam::123456789012:role/SecondLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=second_role,
            show_s3_policy=True,
        )

        second_call_kwargs = mock_generate_policy.call_args[1]
        second_lf_role = second_call_kwargs["lake_formation_role_arn"]

        # Verify different roles were used
        assert first_lf_role == first_role
        assert second_lf_role == second_role
        assert first_lf_role != second_lf_role



class TestPolicyPrintedWithClearInstructions:
    """Tests for verifying the S3 deny policy is printed with clear instructions."""

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_printed_with_header_and_instructions(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that enable_lake_formation prints the S3 deny policy with clear
        header and instructions for the user.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation with show_s3_policy=True
        fg.enable_lake_formation(show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify header is printed
        assert "S3 Bucket Policy" in all_printed_text, "Header should mention 'S3 Bucket Policy'"

        # Verify instructions are printed
        assert (
            "Lake Formation" in all_printed_text
            or "deny policy" in all_printed_text
        ), "Instructions should mention Lake Formation or deny policy"

        # Verify bucket name is printed
        assert "test-bucket" in all_printed_text, "Bucket name should be printed"

        # Verify note about merging with existing policy is printed
        assert (
            "Merge" in all_printed_text or "existing" in all_printed_text
        ), "Note about merging with existing policy should be printed"

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_json_is_printed(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the S3 deny policy JSON is printed to the console when show_s3_policy=True.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation with show_s3_policy=True
        fg.enable_lake_formation(show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy JSON structure elements are printed
        assert "Version" in all_printed_text, "Policy JSON should contain 'Version'"
        assert "Statement" in all_printed_text, "Policy JSON should contain 'Statement'"
        assert "Effect" in all_printed_text, "Policy JSON should contain 'Effect'"
        assert "Deny" in all_printed_text, "Policy JSON should contain 'Deny' effect"

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_printed_only_after_successful_setup(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the S3 deny policy is only printed after all Lake Formation
        phases complete successfully.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock Phase 1 failure
        mock_register.side_effect = Exception("Phase 1 failed")
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            fg.enable_lake_formation(show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy was NOT printed when setup failed
        assert "S3 Bucket Policy" not in all_printed_text, "Policy should not be printed when setup fails"

        # Reset mocks
        mock_print.reset_mock()
        mock_register.reset_mock()
        mock_register.side_effect = None
        mock_register.return_value = True

        # Mock Phase 2 failure
        mock_grant.side_effect = Exception("Phase 2 failed")

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            fg.enable_lake_formation(show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy was NOT printed when setup fails at Phase 2
        assert "S3 Bucket Policy" not in all_printed_text, "Policy should not be printed when Phase 2 fails"

        # Reset mocks
        mock_print.reset_mock()
        mock_grant.reset_mock()
        mock_grant.side_effect = None
        mock_grant.return_value = True

        # Mock Phase 3 failure
        mock_revoke.side_effect = Exception("Phase 3 failed")

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            fg.enable_lake_formation(show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy was NOT printed when setup fails at Phase 3
        assert "S3 Bucket Policy" not in all_printed_text, "Policy should not be printed when Phase 3 fails"

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_includes_both_allowed_principals(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the printed policy includes both the Lake Formation role
        and the Feature Store execution role as allowed principals.
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri="s3://test-bucket/path",
                resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
        )
        feature_store_role = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        fg.role_arn = feature_store_role
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.feature_group_status = "Created"

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation with service-linked role and show_s3_policy=True
        fg.enable_lake_formation(use_service_linked_role=True, show_s3_policy=True)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify Feature Store role is in the printed output
        assert feature_store_role in all_printed_text, "Feature Store role should be in printed policy"

        # Verify Lake Formation service-linked role pattern is in the printed output
        assert "AWSServiceRoleForLakeFormationDataAccess" in all_printed_text, \
            "Lake Formation service-linked role should be in printed policy"

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_not_printed_when_show_s3_policy_false(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the S3 deny policy is NOT printed when show_s3_policy=False (default).
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation with show_s3_policy=False (default)
        fg.enable_lake_formation(show_s3_policy=False)

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy was NOT printed
        assert "S3 Bucket Policy" not in all_printed_text, "Policy should not be printed when show_s3_policy=False"
        assert "Version" not in all_printed_text, "Policy JSON should not be printed when show_s3_policy=False"

    @patch.object(FeatureGroup, "refresh")
    @patch.object(FeatureGroup, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroup, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroup, "_revoke_iam_allowed_principal")
    @patch("builtins.print")
    def test_policy_not_printed_by_default(
        self,
        mock_print,
        mock_revoke,
        mock_grant,
        mock_register,
        mock_refresh,
    ):
        """
        Test that the S3 deny policy is NOT printed by default (when show_s3_policy is not specified).
        """
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        # Set up Feature Group with required configuration
        fg = FeatureGroup(feature_group_name="test-fg")
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

        # Mock successful Lake Formation operations
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        # Call enable_lake_formation without specifying show_s3_policy (should default to False)
        fg.enable_lake_formation()

        # Collect all print calls
        print_calls = [str(call) for call in mock_print.call_args_list]
        all_printed_text = " ".join(print_calls)

        # Verify policy was NOT printed
        assert "S3 Bucket Policy" not in all_printed_text, "Policy should not be printed by default"
        assert "Version" not in all_printed_text, "Policy JSON should not be printed by default"
