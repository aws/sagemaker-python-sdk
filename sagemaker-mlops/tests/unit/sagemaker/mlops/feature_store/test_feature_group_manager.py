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


class TestGetS3Client:
    """Tests for _get_s3_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_s3_client = FeatureGroupManager._get_s3_client.__get__(fg)

        client = fg._get_s3_client(region="us-west-2")

        mock_session.client.assert_called_with("s3", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_s3_client = FeatureGroupManager._get_s3_client.__get__(fg)

        client = fg._get_s3_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("s3", region_name="us-west-2")
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
            fg.enable_lake_formation()

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
            fg.enable_lake_formation()

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
            fg.enable_lake_formation()

        # Verify refresh was called
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_wait_for_active_calls_wait_for_status(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
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
        mock_apply_policy.return_value = True

        # Call with wait_for_active=True
        fg.enable_lake_formation(wait_for_active=True)

        # Verify wait_for_status was called with "Created"
        mock_wait.assert_called_once_with(target_status="Created")
        # Verify refresh was called after wait
        mock_refresh.assert_called_once()

    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_wait_for_active_false_does_not_call_wait(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh, mock_wait
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
        mock_apply_policy.return_value = True

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
        lf_config = LakeFormationConfig()
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
        lf_config = LakeFormationConfig()
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
        lf_config = LakeFormationConfig()
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
        lf_config = LakeFormationConfig()
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
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_revoke_always_called(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that IAMAllowedPrincipal is always revoked."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_apply_policy.return_value = True

        result = self.fg.enable_lake_formation()

        mock_revoke.assert_called_once()
        assert result["iam_principal_revoked"] is True


class TestCreateWithLakeFormationDisableHybridAccessMode:
    """Tests for create() no longer passing disable_hybrid_access_mode."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "enable_lake_formation")
    def test_enable_lake_formation_called_without_disable_hybrid_access_mode(
        self, mock_enable_lf, mock_wait, mock_get, mock_get_client
    ):
        """Test that create() calls enable_lake_formation without disable_hybrid_access_mode."""
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

        lf_config = LakeFormationConfig()
        lf_config.enabled = True

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
        )


class TestLakeFormationConfigDefaults:
    """Tests for LakeFormationConfig default values."""

    def test_has_expected_fields_only(self):
        """Test that LakeFormationConfig has only the expected fields."""
        config = LakeFormationConfig()
        assert config.enabled is False
        assert config.use_service_linked_role is True
        assert config.registration_role_arn is None


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



class TestApplyBucketPolicy:
    """Tests for _apply_bucket_policy method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._apply_bucket_policy = FeatureGroupManager._apply_bucket_policy.__get__(self.fg)
        self.fg._generate_s3_deny_statements = FeatureGroupManager._generate_s3_deny_statements.__get__(self.fg)
        self.mock_s3_client = MagicMock()
        self.fg._get_s3_client = MagicMock(return_value=self.mock_s3_client)

    def test_no_existing_policy_creates_fresh(self):
        """Test that NoSuchBucketPolicy creates a fresh policy with our statements."""
        self.mock_s3_client.get_bucket_policy.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchBucketPolicy", "Message": "No policy"}},
            "GetBucketPolicy",
        )

        result = self.fg._apply_bucket_policy(
            "test-bucket", "prefix",
            "arn:aws:iam::123456789012:role/LFRole",
            "arn:aws:iam::123456789012:role/FSRole",
        )

        assert result is True
        self.mock_s3_client.put_bucket_policy.assert_called_once()
        put_args = self.mock_s3_client.put_bucket_policy.call_args
        import json as _json
        policy = _json.loads(put_args[1]["Policy"])
        assert len(policy["Statement"]) == 2

    def test_existing_policy_appends_statements(self):
        """Test that existing policy gets our statements appended."""
        import json as _json
        existing_policy = {
            "Version": "2012-10-17",
            "Statement": [{"Sid": "ExistingStatement", "Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::test-bucket/*"}]
        }
        self.mock_s3_client.get_bucket_policy.return_value = {"Policy": _json.dumps(existing_policy)}

        result = self.fg._apply_bucket_policy(
            "test-bucket", "prefix",
            "arn:aws:iam::123456789012:role/LFRole",
            "arn:aws:iam::123456789012:role/FSRole",
        )

        assert result is True
        put_args = self.mock_s3_client.put_bucket_policy.call_args
        policy = _json.loads(put_args[1]["Policy"])
        assert len(policy["Statement"]) == 3  # 1 existing + 2 new

    def test_sids_already_exist_skips_put(self):
        """Test that if our Sids already exist, put_bucket_policy is not called."""
        import json as _json
        existing_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Sid": "DenyFSObjectAccess_prefix", "Effect": "Deny"},
                {"Sid": "DenyFSListAccess_prefix", "Effect": "Deny"},
            ]
        }
        self.mock_s3_client.get_bucket_policy.return_value = {"Policy": _json.dumps(existing_policy)}

        result = self.fg._apply_bucket_policy(
            "test-bucket", "prefix",
            "arn:aws:iam::123456789012:role/LFRole",
            "arn:aws:iam::123456789012:role/FSRole",
        )

        assert result is True
        self.mock_s3_client.put_bucket_policy.assert_not_called()

    def test_put_bucket_policy_error_propagates(self):
        """Test that put_bucket_policy errors propagate."""
        self.mock_s3_client.get_bucket_policy.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchBucketPolicy", "Message": "No policy"}},
            "GetBucketPolicy",
        )
        self.mock_s3_client.put_bucket_policy.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "MalformedPolicy", "Message": "Bad policy"}},
            "PutBucketPolicy",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._apply_bucket_policy(
                "test-bucket", "prefix",
                "arn:aws:iam::123456789012:role/LFRole",
                "arn:aws:iam::123456789012:role/FSRole",
            )

        assert exc_info.value.response["Error"]["Code"] == "MalformedPolicy"

    def test_get_bucket_policy_non_nosuchbucketpolicy_error_propagates(self):
        """Test that non-NoSuchBucketPolicy errors from get_bucket_policy propagate."""
        self.mock_s3_client.get_bucket_policy.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access denied"}},
            "GetBucketPolicy",
        )

        with pytest.raises(botocore.exceptions.ClientError) as exc_info:
            self.fg._apply_bucket_policy(
                "test-bucket", "prefix",
                "arn:aws:iam::123456789012:role/LFRole",
                "arn:aws:iam::123456789012:role/FSRole",
            )

        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"

    def test_partial_sid_overlap_appends_only_missing(self):
        """Test that only missing statements are appended when one Sid already exists."""
        import json as _json
        existing_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {"Sid": "DenyFSObjectAccess_prefix", "Effect": "Deny"},
            ]
        }
        self.mock_s3_client.get_bucket_policy.return_value = {"Policy": _json.dumps(existing_policy)}

        result = self.fg._apply_bucket_policy(
            "test-bucket", "prefix",
            "arn:aws:iam::123456789012:role/LFRole",
            "arn:aws:iam::123456789012:role/FSRole",
        )

        assert result is True
        put_args = self.mock_s3_client.put_bucket_policy.call_args
        policy = _json.loads(put_args[1]["Policy"])
        assert len(policy["Statement"]) == 2  # 1 existing + 1 new
        sids = [s["Sid"] for s in policy["Statement"]]
        assert "DenyFSListAccess_prefix" in sids


class TestEnableLakeFormationServiceLinkedRoleInPolicy:
    """Tests for service-linked role ARN usage in Phase 4 bucket policy application."""

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_uses_service_linked_role_arn_when_use_service_linked_role_true(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        fg.enable_lake_formation(use_service_linked_role=True)

        expected_slr_arn = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_apply_policy.assert_called_once()
        call_kwargs = mock_apply_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert call_kwargs["feature_store_role_arn"] == fg.role_arn

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_uses_service_linked_role_arn_by_default(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        fg.enable_lake_formation()

        expected_slr_arn = "arn:aws:iam::987654321098:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_apply_policy.assert_called_once()
        call_kwargs = mock_apply_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_service_linked_role_arn_uses_correct_account_id(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        fg.enable_lake_formation(use_service_linked_role=True)

        expected_slr_arn = f"arn:aws:iam::{account_id}:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_apply_policy.assert_called_once()
        call_kwargs = mock_apply_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert account_id in call_kwargs["lake_formation_role_arn"]



class TestRegistrationRoleArnUsedWhenServiceLinkedRoleFalse:
    """Tests for verifying registration_role_arn is used when use_service_linked_role=False."""

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_uses_registration_role_arn_when_use_service_linked_role_false(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
        )

        mock_apply_policy.assert_called_once()
        call_kwargs = mock_apply_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == custom_registration_role

        service_linked_role_pattern = "aws-service-role/lakeformation.amazonaws.com"
        assert service_linked_role_pattern not in call_kwargs["lake_formation_role_arn"]

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_registration_role_arn_passed_to_s3_registration(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
        )

        mock_register.assert_called_once()
        call_args = mock_register.call_args
        assert call_args[1]["use_service_linked_role"] == False
        assert call_args[1]["role_arn"] == custom_registration_role

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_different_registration_role_arns_produce_different_policies(
        self,
        mock_apply_policy,
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
        mock_apply_policy.return_value = True

        first_role = "arn:aws:iam::123456789012:role/FirstLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=first_role,
        )
        first_call_kwargs = mock_apply_policy.call_args[1]
        first_lf_role = first_call_kwargs["lake_formation_role_arn"]

        mock_apply_policy.reset_mock()
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        second_role = "arn:aws:iam::123456789012:role/SecondLakeFormationRole"
        fg.enable_lake_formation(
            use_service_linked_role=False,
            registration_role_arn=second_role,
        )
        second_call_kwargs = mock_apply_policy.call_args[1]
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
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_iceberg_strips_data_suffix_for_s3_registration(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that Iceberg tables register the parent S3 path (without /data suffix)."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_apply_policy.return_value = True

        self.fg.enable_lake_formation()

        # The registered S3 location should NOT end with /data
        call_args = mock_register.call_args
        registered_location = call_args[0][0]
        assert registered_location == "s3://test-bucket/resolved-path"
        assert not registered_location.endswith("/data")

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_non_iceberg_keeps_full_s3_path(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test that non-Iceberg tables use the full resolved S3 URI."""
        self.fg.offline_store_config.table_format = None
        self.fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = (
            "s3://test-bucket/resolved-path/data"
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_apply_policy.return_value = True

        self.fg.enable_lake_formation()

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
            fg.enable_lake_formation()


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
    @patch.object(FeatureGroupManager, "_apply_bucket_policy")
    def test_returns_all_true_on_success(
        self, mock_apply_policy, mock_revoke, mock_grant, mock_register, mock_refresh
    ):
        """Test full happy-path returns all phases as True."""
        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_apply_policy.return_value = True

        result = self.fg.enable_lake_formation()

        assert result == {
            "s3_registration": True,
            "permissions_granted": True,
            "iam_principal_revoked": True,
            "bucket_policy_applied": True,
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
        lf_config = LakeFormationConfig()
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


class TestGetCloudTrailClient:
    """Tests for _get_cloudtrail_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_cloudtrail_client = FeatureGroupManager._get_cloudtrail_client.__get__(fg)

        client = fg._get_cloudtrail_client(region="us-west-2")

        mock_session.client.assert_called_with("cloudtrail", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_cloudtrail_client = FeatureGroupManager._get_cloudtrail_client.__get__(fg)

        client = fg._get_cloudtrail_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("cloudtrail", region_name="us-west-2")
        assert client == mock_client


class TestGetSageMakerClient:
    """Tests for _get_sagemaker_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_sagemaker_client = FeatureGroupManager._get_sagemaker_client.__get__(fg)

        client = fg._get_sagemaker_client(region="us-west-2")

        mock_session.client.assert_called_with("sagemaker", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_sagemaker_client = FeatureGroupManager._get_sagemaker_client.__get__(fg)

        client = fg._get_sagemaker_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("sagemaker", region_name="us-west-2")
        assert client == mock_client


class TestQueryGlueTableAccessors:
    """Tests for _query_glue_table_accessors method."""

    def setup_method(self):
        """Set up test fixtures."""
        import json as _json
        self._json = _json
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._query_glue_table_accessors = (
            FeatureGroupManager._query_glue_table_accessors.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_cloudtrail_client = MagicMock(return_value=self.mock_client)

    def _make_event(self, event_name, username=None, resources=None, cloud_trail_event=None, event_time=None):
        """Helper to build a CloudTrail event dict."""
        from datetime import datetime, timezone
        event = {
            "EventName": event_name,
            "EventTime": event_time or datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc),
        }
        if username:
            event["Username"] = username
        if resources:
            event["Resources"] = resources
        if cloud_trail_event:
            event["CloudTrailEvent"] = self._json.dumps(cloud_trail_event)
        return event

    def test_returns_matching_accessors(self):
        """Test that accessors are returned when Resources match the table name."""
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "GetTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "my_table"}],
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 1
        assert result["accessors"][0]["principal_arn"] == "arn:aws:iam::123:role/MyRole"

    def test_matches_via_request_parameters(self):
        """Test matching via CloudTrailEvent requestParameters when Resources don't match."""
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "GetTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "other_table"}],
                    cloud_trail_event={
                        "requestParameters": {
                            "databaseName": "my_db",
                            "tableName": "my_table",
                        }
                    },
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 1

    def test_filters_non_table_events(self):
        """Test that events with non-read EventNames are filtered out."""
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "CreateTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "my_table"}],
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 0

    def test_deduplicates_principals_keeps_latest(self):
        """Test that duplicate principals keep the latest event time."""
        from datetime import datetime, timezone
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "GetTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "my_table"}],
                    event_time=datetime(2024, 1, 10, tzinfo=timezone.utc),
                ),
                self._make_event(
                    "GetTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "my_table"}],
                    event_time=datetime(2024, 1, 20, tzinfo=timezone.utc),
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 1
        assert "2024-01-20" in result["accessors"][0]["last_access_time"]

    def test_pagination_with_next_token(self):
        """Test that pagination via NextToken processes all pages."""
        self.mock_client.lookup_events.side_effect = [
            {
                "Events": [
                    self._make_event("GetTable", username="arn:aws:iam::123:role/RoleA", resources=[{"ResourceName": "my_table"}]),
                ],
                "NextToken": "token1",
            },
            {
                "Events": [
                    self._make_event("GetTable", username="arn:aws:iam::123:role/RoleB", resources=[{"ResourceName": "my_table"}]),
                ],
            },
        ]

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 2
        arns = {a["principal_arn"] for a in result["accessors"]}
        assert "arn:aws:iam::123:role/RoleA" in arns
        assert "arn:aws:iam::123:role/RoleB" in arns

    def test_max_events_cap(self):
        """Test that processing stops after 1000 events."""
        # Create 1001 events, all matching
        events = [
            self._make_event(
                "GetTable",
                username=f"arn:aws:iam::123:role/Role{i}",
                resources=[{"ResourceName": "my_table"}],
            )
            for i in range(1001)
        ]
        self.mock_client.lookup_events.return_value = {"Events": events}

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        # Should have at most 1000 accessors (events_scanned caps at 1000)
        assert len(result["accessors"]) <= 1000

    def test_access_denied_returns_warning(self):
        """Test that AccessDeniedException returns empty accessors with warning."""
        self.mock_client.lookup_events.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "LookupEvents",
        )

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert result["accessors"] == []
        assert len(result["warnings"]) == 1
        assert "access denied" in result["warnings"][0].lower()

    def test_client_creation_failure(self):
        """Test that client creation failure returns empty accessors with warning."""
        self.fg._get_cloudtrail_client = MagicMock(side_effect=Exception("Connection error"))

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert result["accessors"] == []
        assert len(result["warnings"]) == 1
        assert "Failed to create CloudTrail client" in result["warnings"][0]

    def test_no_matching_events(self):
        """Test that events not matching the table return empty accessors."""
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "GetTable",
                    username="arn:aws:iam::123:role/MyRole",
                    resources=[{"ResourceName": "other_table"}],
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert result["accessors"] == []

    def test_extracts_principal_from_cloud_trail_event(self):
        """Test principal extraction from CloudTrailEvent JSON when Username is absent."""
        self.mock_client.lookup_events.return_value = {
            "Events": [
                self._make_event(
                    "GetTable",
                    resources=[{"ResourceName": "my_table"}],
                    cloud_trail_event={
                        "userIdentity": {"arn": "arn:aws:iam::123:role/ExtractedRole"},
                    },
                ),
            ],
        }

        result = self.fg._query_glue_table_accessors("my_db", "my_table")

        assert len(result["accessors"]) == 1
        assert result["accessors"][0]["principal_arn"] == "arn:aws:iam::123:role/ExtractedRole"


class TestQuerySageMakerExecutionRoles:
    """Tests for _query_sagemaker_execution_roles method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._query_sagemaker_execution_roles = (
            FeatureGroupManager._query_sagemaker_execution_roles.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_sagemaker_client = MagicMock(return_value=self.mock_client)

    def _setup_paginator(self, api_method, pages):
        """Helper to set up paginator mock for a specific API method."""
        paginator = MagicMock()
        paginator.paginate.return_value = pages
        # Make get_paginator return the right paginator for the right method
        original_side_effect = self.mock_client.get_paginator.side_effect

        def get_paginator_side_effect(method_name):
            if method_name == api_method:
                return paginator
            # Return empty paginator for other methods
            empty = MagicMock()
            empty.paginate.return_value = []
            return empty

        if original_side_effect is None:
            # First call - set up a mapping
            self._paginator_map = {api_method: paginator}
            self.mock_client.get_paginator.side_effect = lambda m: self._paginator_map.get(
                m, MagicMock(paginate=MagicMock(return_value=[]))
            )
        else:
            self._paginator_map[api_method] = paginator

    def test_returns_training_job_roles(self):
        """Test that training job roles are extracted from summaries."""
        self._setup_paginator("list_training_jobs", [
            {"TrainingJobSummaries": [
                {"TrainingJobName": "job1", "RoleArn": "arn:aws:iam::123:role/TrainRole"},
            ]},
        ])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_sagemaker_execution_roles()

        assert len(result["roles"]) == 1
        assert result["roles"][0]["role_arn"] == "arn:aws:iam::123:role/TrainRole"
        assert result["roles"][0]["job_type"] == "TrainingJob"

    def test_returns_processing_job_roles(self):
        """Test that processing job roles are extracted from summaries."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [
            {"ProcessingJobSummaries": [
                {"ProcessingJobName": "proc1", "RoleArn": "arn:aws:iam::123:role/ProcRole"},
            ]},
        ])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_sagemaker_execution_roles()

        assert len(result["roles"]) == 1
        assert result["roles"][0]["role_arn"] == "arn:aws:iam::123:role/ProcRole"
        assert result["roles"][0]["job_type"] == "ProcessingJob"

    def test_deduplicates_roles(self):
        """Test that the same role from different job types appears only once."""
        self._setup_paginator("list_training_jobs", [
            {"TrainingJobSummaries": [
                {"TrainingJobName": "job1", "RoleArn": "arn:aws:iam::123:role/SharedRole"},
            ]},
        ])
        self._setup_paginator("list_processing_jobs", [
            {"ProcessingJobSummaries": [
                {"ProcessingJobName": "proc1", "RoleArn": "arn:aws:iam::123:role/SharedRole"},
            ]},
        ])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_sagemaker_execution_roles()

        assert len(result["roles"]) == 1

    def test_transform_jobs_skip_role(self):
        """Test that transform jobs don't extract roles (too indirect via ModelName)."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [
            {"TransformJobSummaries": [
                {"TransformJobName": "transform1"},
            ]},
        ])
        # describe_transform_job returns no RoleArn (role is on the model, not the job)
        self.mock_client.describe_transform_job.return_value = {"ModelName": "my-model"}

        result = self.fg._query_sagemaker_execution_roles()

        assert len(result["roles"]) == 0

    def test_access_denied_per_job_type(self):
        """Test that AccessDeniedException on one job type still processes others."""
        # Set up processing and transform first so the map exists
        self._setup_paginator("list_processing_jobs", [
            {"ProcessingJobSummaries": [
                {"ProcessingJobName": "proc1", "RoleArn": "arn:aws:iam::123:role/ProcRole"},
            ]},
        ])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])
        # Training jobs paginator raises AccessDeniedException during iteration
        training_paginator = MagicMock()
        training_paginator.paginate.return_value = iter([])  # placeholder
        training_paginator.paginate.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "ListTrainingJobs",
        )
        self._paginator_map["list_training_jobs"] = training_paginator

        result = self.fg._query_sagemaker_execution_roles()

        assert len(result["roles"]) == 1
        assert result["roles"][0]["role_arn"] == "arn:aws:iam::123:role/ProcRole"
        assert len(result["warnings"]) == 1
        assert "access denied" in result["warnings"][0].lower()

    def test_client_creation_failure(self):
        """Test that client creation failure returns empty roles with warning."""
        self.fg._get_sagemaker_client = MagicMock(side_effect=Exception("Connection error"))

        result = self.fg._query_sagemaker_execution_roles()

        assert result["roles"] == []
        assert len(result["warnings"]) == 1
        assert "Failed to create SageMaker client" in result["warnings"][0]

    def test_no_jobs_found(self):
        """Test that empty summaries return empty roles."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_sagemaker_execution_roles()

        assert result["roles"] == []
        assert result["warnings"] == []


class TestRunAllAuditQueries:
    """Tests for _run_all_audit_queries method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._run_all_audit_queries = (
            FeatureGroupManager._run_all_audit_queries.__get__(self.fg)
        )
        self.fg._query_glue_table_accessors = MagicMock(return_value={"accessors": [], "warnings": []})
        self.fg._query_sagemaker_execution_roles = MagicMock(return_value={"roles": [], "warnings": []})
        self.fg._query_athena_query_principals = MagicMock(return_value={"principals": [], "running_queries": [], "warnings": []})
        self.fg._query_glue_etl_jobs = MagicMock(return_value={"jobs": [], "running_job_runs": [], "warnings": []})
        self.fg._query_running_jobs = MagicMock(return_value={"running_jobs": [], "warnings": []})

    def test_combines_all_query_results(self):
        """Test that results from all five sub-methods are aggregated."""
        self.fg._query_glue_table_accessors.return_value = {
            "accessors": [{"principal_arn": "arn:aws:iam::123:role/A", "last_access_time": "t1"}],
            "warnings": [],
        }
        self.fg._query_sagemaker_execution_roles.return_value = {
            "roles": [{"role_arn": "arn:aws:iam::123:role/B", "job_type": "TrainingJob", "job_name": "j1"}],
            "warnings": [],
        }
        self.fg._query_athena_query_principals.return_value = {
            "principals": [{"query_execution_id": "id-1"}],
            "running_queries": [{"query_execution_id": "id-2", "state": "RUNNING"}],
            "warnings": [],
        }
        self.fg._query_glue_etl_jobs.return_value = {
            "jobs": [{"job_name": "etl-1"}],
            "running_job_runs": [{"job_name": "etl-1", "run_id": "r1", "state": "RUNNING"}],
            "warnings": [],
        }
        self.fg._query_running_jobs.return_value = {
            "running_jobs": [{"job_name": "train-1", "status": "InProgress"}],
            "warnings": [],
        }

        result = self.fg._run_all_audit_queries("db", "tbl", "s3://bucket/path")

        assert len(result["glue_table_accessors"]) == 1
        assert len(result["sagemaker_execution_roles"]) == 1
        assert len(result["athena_query_principals"]) == 1
        assert len(result["athena_running_queries"]) == 1
        assert len(result["glue_etl_jobs"]) == 1
        assert len(result["glue_running_job_runs"]) == 1
        assert len(result["sagemaker_running_jobs"]) == 1
        assert result["glue_database"] == "db"
        assert result["glue_table"] == "tbl"
        assert result["s3_path"] == "s3://bucket/path"

    def test_merges_warnings(self):
        """Test that warnings from all sub-methods are merged."""
        self.fg._query_glue_table_accessors.return_value = {"accessors": [], "warnings": ["warn1"]}
        self.fg._query_sagemaker_execution_roles.return_value = {"roles": [], "warnings": ["warn2"]}
        self.fg._query_athena_query_principals.return_value = {"principals": [], "running_queries": [], "warnings": ["warn3"]}
        self.fg._query_glue_etl_jobs.return_value = {"jobs": [], "running_job_runs": [], "warnings": ["warn4"]}
        self.fg._query_running_jobs.return_value = {"running_jobs": [], "warnings": ["warn5"]}

        result = self.fg._run_all_audit_queries("db", "tbl", "s3://bucket/path")

        assert result["warnings"] == ["warn1", "warn2", "warn3", "warn4", "warn5"]

    def test_empty_results(self):
        """Test structure with empty results from all sub-methods."""
        result = self.fg._run_all_audit_queries("db", "tbl", "s3://bucket/path")

        assert result["glue_table_accessors"] == []
        assert result["sagemaker_execution_roles"] == []
        assert result["athena_query_principals"] == []
        assert result["athena_running_queries"] == []
        assert result["glue_etl_jobs"] == []
        assert result["glue_running_job_runs"] == []
        assert result["sagemaker_running_jobs"] == []
        assert result["warnings"] == []

    def test_passes_parameters_through(self):
        """Test that session, region, lookback_days are passed to all sub-methods."""
        mock_session = MagicMock()

        self.fg._run_all_audit_queries(
            "db", "tbl", "s3://bucket/path",
            session=mock_session, region="eu-west-1", lookback_days=7,
        )

        self.fg._query_glue_table_accessors.assert_called_once_with(
            database_name="db", table_name="tbl",
            session=mock_session, region="eu-west-1", lookback_days=7,
        )
        self.fg._query_sagemaker_execution_roles.assert_called_once_with(
            session=mock_session, region="eu-west-1", lookback_days=7,
        )
        self.fg._query_athena_query_principals.assert_called_once_with(
            database_name="db", table_name="tbl",
            session=mock_session, region="eu-west-1", lookback_days=7,
        )
        self.fg._query_glue_etl_jobs.assert_called_once_with(
            database_name="db", table_name="tbl",
            session=mock_session, region="eu-west-1",
        )
        self.fg._query_running_jobs.assert_called_once_with(
            session=mock_session, region="eu-west-1",
        )


class TestAuditGateInEnableLakeFormation:
    """Tests for the audit gate logic between Phase 2 and Phase 3 in enable_lake_formation."""

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

    def _empty_audit(self):
        return {
            "glue_table_accessors": [],
            "sagemaker_execution_roles": [],
            "athena_query_principals": [],
            "athena_running_queries": [],
            "glue_etl_jobs": [],
            "glue_running_job_runs": [],
            "sagemaker_running_jobs": [],
            "glue_database": "test_db",
            "glue_table": "test_table",
            "s3_path": "s3://test-bucket/resolved-path",
            "warnings": [],
        }

    def _audit_with_findings(self):
        audit = self._empty_audit()
        audit["glue_table_accessors"] = [{"principal_arn": "arn:aws:iam::123:role/Accessor"}]
        audit["sagemaker_execution_roles"] = [{"role_arn": "arn:aws:iam::123:role/SMRole"}]
        return audit

    @patch("builtins.input", return_value="N")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation", return_value=True)
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions", return_value=True)
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    def test_prompts_user_when_findings_exist(
        self, mock_audit, mock_grant, mock_register, mock_refresh, mock_input
    ):
        """Test that user is prompted and abort returned when declining."""
        mock_audit.return_value = self._audit_with_findings()

        result = self.fg.enable_lake_formation()

        mock_input.assert_called_once()
        assert result["aborted"] is True
        assert "audit_results" in result

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation", return_value=True)
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions", return_value=True)
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal", return_value=True)
    @patch.object(FeatureGroupManager, "_apply_bucket_policy", return_value=True)
    def test_proceeds_silently_when_no_findings(
        self, mock_apply, mock_revoke, mock_audit, mock_grant, mock_register, mock_refresh
    ):
        """Test that Phase 3 proceeds without input() when no findings."""
        mock_audit.return_value = self._empty_audit()

        result = self.fg.enable_lake_formation()

        mock_revoke.assert_called_once()
        assert result["iam_principal_revoked"] is True

    @patch("builtins.input", return_value="y")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation", return_value=True)
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions", return_value=True)
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal", return_value=True)
    @patch.object(FeatureGroupManager, "_apply_bucket_policy", return_value=True)
    def test_user_confirms_proceeds_to_phase3(
        self, mock_apply, mock_revoke, mock_audit, mock_grant, mock_register, mock_refresh, mock_input
    ):
        """Test that confirming 'y' proceeds to Phase 3."""
        mock_audit.return_value = self._audit_with_findings()

        result = self.fg.enable_lake_formation()

        mock_revoke.assert_called_once()
        assert result["iam_principal_revoked"] is True

    @patch("builtins.input", return_value="n")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation", return_value=True)
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions", return_value=True)
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    def test_user_declines_returns_audit_results(
        self, mock_audit, mock_grant, mock_register, mock_refresh, mock_input
    ):
        """Test that declining returns aborted result with audit data."""
        mock_audit.return_value = self._audit_with_findings()

        result = self.fg.enable_lake_formation()

        assert result["aborted"] is True
        assert result["audit_results"]["glue_table_accessors"][0]["principal_arn"] == "arn:aws:iam::123:role/Accessor"

    @patch("builtins.input", return_value="")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation", return_value=True)
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions", return_value=True)
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    def test_empty_input_defaults_to_abort(
        self, mock_audit, mock_grant, mock_register, mock_refresh, mock_input
    ):
        """Test that empty input defaults to abort."""
        mock_audit.return_value = self._audit_with_findings()

        result = self.fg.enable_lake_formation()

        assert result["aborted"] is True


class TestQueryRunningJobs:
    """Tests for _query_running_jobs method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._query_running_jobs = (
            FeatureGroupManager._query_running_jobs.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_sagemaker_client = MagicMock(return_value=self.mock_client)

    def _setup_paginator(self, api_method, pages):
        """Helper to set up paginator mock for a specific API method."""
        paginator = MagicMock()
        paginator.paginate.return_value = pages
        if not hasattr(self, "_paginator_map"):
            self._paginator_map = {}
            self.mock_client.get_paginator.side_effect = lambda m: self._paginator_map.get(
                m, MagicMock(paginate=MagicMock(return_value=[]))
            )
        self._paginator_map[api_method] = paginator

    def test_returns_in_progress_training_jobs(self):
        """Test that in-progress training jobs are returned."""
        self._setup_paginator("list_training_jobs", [
            {"TrainingJobSummaries": [
                {
                    "TrainingJobName": "train-1",
                    "TrainingJobArn": "arn:aws:sagemaker:us-east-1:123:training-job/train-1",
                    "RoleArn": "arn:aws:iam::123:role/TrainRole",
                },
            ]},
        ])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_running_jobs()

        assert len(result["running_jobs"]) == 1
        job = result["running_jobs"][0]
        assert job["service"] == "SageMaker"
        assert job["job_type"] == "TrainingJob"
        assert job["job_name"] == "train-1"
        assert job["status"] == "InProgress"
        assert job["role_arn"] == "arn:aws:iam::123:role/TrainRole"
        assert result["warnings"] == []

    def test_returns_in_progress_processing_jobs(self):
        """Test that in-progress processing jobs are returned."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [
            {"ProcessingJobSummaries": [
                {
                    "ProcessingJobName": "proc-1",
                    "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123:processing-job/proc-1",
                    "RoleArn": "arn:aws:iam::123:role/ProcRole",
                },
            ]},
        ])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_running_jobs()

        assert len(result["running_jobs"]) == 1
        job = result["running_jobs"][0]
        assert job["service"] == "SageMaker"
        assert job["job_type"] == "ProcessingJob"
        assert job["job_name"] == "proc-1"
        assert job["role_arn"] == "arn:aws:iam::123:role/ProcRole"

    def test_returns_in_progress_transform_jobs(self):
        """Test that in-progress transform jobs are returned with role_arn as None."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [
            {"TransformJobSummaries": [
                {
                    "TransformJobName": "transform-1",
                    "TransformJobArn": "arn:aws:sagemaker:us-east-1:123:transform-job/transform-1",
                },
            ]},
        ])

        result = self.fg._query_running_jobs()

        assert len(result["running_jobs"]) == 1
        job = result["running_jobs"][0]
        assert job["job_type"] == "TransformJob"
        assert job["job_name"] == "transform-1"
        assert job["role_arn"] is None

    def test_access_denied_per_service(self):
        """Test that AccessDeniedException on one service still processes others."""
        self._setup_paginator("list_processing_jobs", [
            {"ProcessingJobSummaries": [
                {"ProcessingJobName": "proc-1", "RoleArn": "arn:aws:iam::123:role/ProcRole"},
            ]},
        ])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])
        # Training jobs raises AccessDeniedException
        training_paginator = MagicMock()
        training_paginator.paginate.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "ListTrainingJobs",
        )
        self._paginator_map["list_training_jobs"] = training_paginator

        result = self.fg._query_running_jobs()

        assert len(result["running_jobs"]) == 1
        assert result["running_jobs"][0]["job_name"] == "proc-1"
        assert len(result["warnings"]) == 1
        assert "access denied" in result["warnings"][0].lower()

    def test_client_creation_failure(self):
        """Test that client creation failure returns empty jobs with warning."""
        self.fg._get_sagemaker_client = MagicMock(side_effect=Exception("Connection error"))

        result = self.fg._query_running_jobs()

        assert result["running_jobs"] == []
        assert len(result["warnings"]) == 1
        assert "Failed to create SageMaker client" in result["warnings"][0]

    def test_no_running_jobs(self):
        """Test that empty summaries return empty running_jobs."""
        self._setup_paginator("list_training_jobs", [{"TrainingJobSummaries": []}])
        self._setup_paginator("list_processing_jobs", [{"ProcessingJobSummaries": []}])
        self._setup_paginator("list_transform_jobs", [{"TransformJobSummaries": []}])

        result = self.fg._query_running_jobs()

        assert result["running_jobs"] == []
        assert result["warnings"] == []


class TestGetAthenaClient:
    """Tests for _get_athena_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_athena_client = FeatureGroupManager._get_athena_client.__get__(fg)

        client = fg._get_athena_client(region="us-west-2")

        mock_session.client.assert_called_with("athena", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_athena_client = FeatureGroupManager._get_athena_client.__get__(fg)

        client = fg._get_athena_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("athena", region_name="us-west-2")
        assert client == mock_client


class TestQueryAthenaQueryPrincipals:
    """Tests for _query_athena_query_principals method."""

    def setup_method(self):
        """Set up test fixtures."""
        from datetime import datetime, timedelta, timezone
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._query_athena_query_principals = (
            FeatureGroupManager._query_athena_query_principals.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_athena_client = MagicMock(return_value=self.mock_client)
        self.recent_time = datetime.now(timezone.utc) - timedelta(days=1)

    def _make_query_execution(self, query_id, query_sql, database, state="SUCCEEDED", submission_time=None):
        """Helper to build an Athena QueryExecution dict."""
        from datetime import datetime, timezone
        return {
            "QueryExecutionId": query_id,
            "Query": query_sql,
            "QueryExecutionContext": {"Database": database},
            "Status": {
                "State": state,
                "SubmissionDateTime": submission_time or self.recent_time,
            },
        }

    def test_returns_matching_queries(self):
        """Test that queries matching database and table are returned."""
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": ["id-1"]},
        ]
        self.mock_client.batch_get_query_execution.return_value = {
            "QueryExecutions": [
                self._make_query_execution("id-1", "SELECT * FROM my_table", "my_db"),
            ],
        }

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert len(result["principals"]) == 1
        assert result["principals"][0]["query_execution_id"] == "id-1"
        assert result["warnings"] == []

    def test_matches_explicit_db_table_reference(self):
        """Test matching when SQL contains database.table reference."""
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": ["id-1"]},
        ]
        self.mock_client.batch_get_query_execution.return_value = {
            "QueryExecutions": [
                self._make_query_execution(
                    "id-1", "SELECT * FROM my_db.my_table WHERE x=1", "other_db"
                ),
            ],
        }

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert len(result["principals"]) == 1

    def test_filters_queries_outside_lookback(self):
        """Test that queries older than lookback window are excluded."""
        from datetime import datetime, timezone
        old_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": ["id-1"]},
        ]
        self.mock_client.batch_get_query_execution.return_value = {
            "QueryExecutions": [
                self._make_query_execution(
                    "id-1", "SELECT * FROM my_table", "my_db",
                    submission_time=old_time,
                ),
            ],
        }

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert len(result["principals"]) == 0

    def test_detects_running_queries(self):
        """Test that RUNNING queries appear in running_queries."""
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": ["id-1"]},
        ]
        self.mock_client.batch_get_query_execution.return_value = {
            "QueryExecutions": [
                self._make_query_execution(
                    "id-1", "SELECT * FROM my_table", "my_db", state="RUNNING"
                ),
            ],
        }

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert len(result["running_queries"]) == 1
        assert result["running_queries"][0]["state"] == "RUNNING"

    def test_no_queries_found(self):
        """Test empty result when no query executions exist."""
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": []},
        ]

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert result["principals"] == []
        assert result["running_queries"] == []
        assert result["warnings"] == []

    def test_access_denied_returns_warning(self):
        """Test that AccessDeniedException returns empty results with warning."""
        self.mock_client.get_paginator.return_value.paginate.return_value.__iter__ = MagicMock(
            side_effect=botocore.exceptions.ClientError(
                {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
                "ListQueryExecutions",
            )
        )

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert result["principals"] == []
        assert result["running_queries"] == []
        assert len(result["warnings"]) == 1
        assert "access denied" in result["warnings"][0].lower()

    def test_client_creation_failure(self):
        """Test that client creation failure returns empty results with warning."""
        self.fg._get_athena_client = MagicMock(side_effect=Exception("Connection error"))

        result = self.fg._query_athena_query_principals("my_db", "my_table")

        assert result["principals"] == []
        assert result["running_queries"] == []
        assert len(result["warnings"]) == 1
        assert "Failed to create Athena client" in result["warnings"][0]

    def test_batch_chunking(self):
        """Test that >50 query IDs are batched correctly in chunks of 50."""
        ids = [f"id-{i}" for i in range(75)]
        self.mock_client.get_paginator.return_value.paginate.return_value = [
            {"QueryExecutionIds": ids},
        ]
        # Return empty for both batches
        self.mock_client.batch_get_query_execution.return_value = {"QueryExecutions": []}

        self.fg._query_athena_query_principals("my_db", "my_table")

        calls = self.mock_client.batch_get_query_execution.call_args_list
        assert len(calls) == 2
        assert len(calls[0][1]["QueryExecutionIds"]) == 50
        assert len(calls[1][1]["QueryExecutionIds"]) == 25


class TestGetGlueClient:
    """Tests for _get_glue_client method."""

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_creates_client_with_default_session(self, mock_session_class):
        """Test client creation with default session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client
        mock_session_class.return_value = mock_session

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_glue_client = FeatureGroupManager._get_glue_client.__get__(fg)

        client = fg._get_glue_client(region="us-west-2")

        mock_session.client.assert_called_with("glue", region_name="us-west-2")
        assert client == mock_client

    def test_creates_client_with_provided_session(self):
        """Test client creation with provided session."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_glue_client = FeatureGroupManager._get_glue_client.__get__(fg)

        client = fg._get_glue_client(session=mock_session, region="us-west-2")

        mock_session.client.assert_called_with("glue", region_name="us-west-2")
        assert client == mock_client


class TestQueryGlueEtlJobs:
    """Tests for _query_glue_etl_jobs method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._query_glue_etl_jobs = (
            FeatureGroupManager._query_glue_etl_jobs.__get__(self.fg)
        )
        self.mock_client = MagicMock()
        self.fg._get_glue_client = MagicMock(return_value=self.mock_client)

    def _make_visual_job(self, name, role, database, table, node_type="CatalogSource", use_tables=False):
        """Helper to create a visual-mode Glue job definition."""
        if use_tables:
            node_config = {node_type: {"Database": database, "Tables": [table]}}
        else:
            node_config = {node_type: {"Database": database, "Table": table}}
        return {
            "Name": name,
            "Role": role,
            "Command": {"Name": "glueetl"},
            "CodeGenConfigurationNodes": {"node1": node_config},
        }

    def test_returns_matching_visual_job(self):
        """Test that a visual job with CatalogSource matching database+table is returned."""
        job = self._make_visual_job("etl-job", "arn:aws:iam::123:role/GlueRole", "my_db", "my_table")
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Jobs": [job]}]
        self.mock_client.get_paginator.return_value = paginator
        self.mock_client.get_job_runs.return_value = {"JobRuns": []}

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert len(result["jobs"]) == 1
        assert result["jobs"][0]["job_name"] == "etl-job"
        assert result["jobs"][0]["role"] == "arn:aws:iam::123:role/GlueRole"
        assert result["warnings"] == []

    def test_filters_non_matching_table(self):
        """Test that a job referencing a different table is not included."""
        job = self._make_visual_job("etl-job", "role", "my_db", "other_table")
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Jobs": [job]}]
        self.mock_client.get_paginator.return_value = paginator

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert result["jobs"] == []

    def test_script_mode_job_skipped(self):
        """Test that a job without CodeGenConfigurationNodes is skipped."""
        job = {
            "Name": "script-job",
            "Role": "role",
            "Command": {"Name": "glueetl"},
        }
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Jobs": [job]}]
        self.mock_client.get_paginator.return_value = paginator

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert result["jobs"] == []

    def test_detects_running_job_runs(self):
        """Test that running job runs are detected for matching jobs."""
        job = self._make_visual_job("etl-job", "role", "my_db", "my_table")
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Jobs": [job]}]
        self.mock_client.get_paginator.return_value = paginator
        self.mock_client.get_job_runs.return_value = {
            "JobRuns": [
                {"Id": "run-1", "JobRunState": "RUNNING"},
                {"Id": "run-2", "JobRunState": "SUCCEEDED"},
            ]
        }

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert len(result["running_job_runs"]) == 1
        assert result["running_job_runs"][0]["run_id"] == "run-1"
        assert result["running_job_runs"][0]["state"] == "RUNNING"

    def test_no_jobs_found(self):
        """Test that empty get_jobs response returns empty results."""
        paginator = MagicMock()
        paginator.paginate.return_value = [{"Jobs": []}]
        self.mock_client.get_paginator.return_value = paginator

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert result["jobs"] == []
        assert result["running_job_runs"] == []
        assert result["warnings"] == []

    def test_access_denied_returns_warning(self):
        """Test that AccessDeniedException on get_jobs returns a warning."""
        paginator = MagicMock()
        paginator.paginate.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "GetJobs",
        )
        self.mock_client.get_paginator.return_value = paginator

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert result["jobs"] == []
        assert len(result["warnings"]) == 1
        assert "access denied" in result["warnings"][0].lower()

    def test_client_creation_failure(self):
        """Test that client creation failure returns empty results with warning."""
        self.fg._get_glue_client = MagicMock(side_effect=Exception("Connection error"))

        result = self.fg._query_glue_etl_jobs("my_db", "my_table")

        assert result["jobs"] == []
        assert result["running_job_runs"] == []
        assert len(result["warnings"]) == 1
        assert "Failed to create Glue client" in result["warnings"][0]


class TestFormatRiskReport:
    """Tests for _format_risk_report method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg.feature_group_name = "test-fg"
        self.fg._format_risk_report = (
            FeatureGroupManager._format_risk_report.__get__(self.fg)
        )

    def _base_audit(self):
        return {
            "glue_table_accessors": [],
            "sagemaker_execution_roles": [],
            "athena_query_principals": [],
            "athena_running_queries": [],
            "glue_etl_jobs": [],
            "glue_running_job_runs": [],
            "sagemaker_running_jobs": [],
            "glue_database": "my_db",
            "glue_table": "my_table",
            "s3_path": "s3://bucket/path",
            "warnings": [],
        }

    def test_all_sections_populated(self):
        """Test report with all findings present."""
        audit = self._base_audit()
        audit["glue_table_accessors"] = [{"principal_arn": "arn:aws:iam::123:role/A"}]
        audit["sagemaker_execution_roles"] = [{"role_arn": "arn:aws:iam::123:role/B"}]
        audit["athena_query_principals"] = [{"query_execution_id": "id-1", "query": "SELECT *"}]
        audit["glue_etl_jobs"] = [{"job_name": "etl-1"}]
        audit["sagemaker_running_jobs"] = [{"job_name": "train-1", "status": "InProgress"}]
        audit["warnings"] = ["some warning"]

        report = self.fg._format_risk_report(audit)

        assert "Glue table accessors" in report
        assert "SageMaker execution roles" in report
        assert "Athena query principals" in report
        assert "Glue ETL jobs" in report
        assert "[!] Running jobs/queries" in report
        assert "Warnings:" in report
        assert "[!] WARNING - Limitations:" in report

    def test_empty_sections_omitted(self):
        """Test that empty data sections are omitted."""
        report = self.fg._format_risk_report(self._base_audit())

        assert "Glue table accessors" not in report
        assert "SageMaker execution roles" not in report
        assert "Athena query principals" not in report
        assert "Glue ETL jobs" not in report
        assert "[!] Running jobs/queries" not in report
        assert "=== Lake Formation Impact Report ===" in report

    def test_limitations_always_shown(self):
        """Test that limitations section is always present."""
        report = self.fg._format_risk_report(self._base_audit())

        assert "[!] WARNING - Limitations:" in report
        assert "CloudTrail has 15-minute delivery delay" in report

    def test_running_jobs_section_has_warning_prefix(self):
        """Test that running jobs section uses [!] prefix."""
        audit = self._base_audit()
        audit["athena_running_queries"] = [{"query_execution_id": "id-1", "state": "RUNNING"}]

        report = self.fg._format_risk_report(audit)

        assert "[!] Running jobs/queries:" in report

    def test_warnings_displayed(self):
        """Test that warnings from audit results are shown."""
        audit = self._base_audit()
        audit["warnings"] = ["access denied warning"]

        report = self.fg._format_risk_report(audit)

        assert "Warnings:" in report
        assert "access denied warning" in report


class TestAuditLakeFormationImpact:
    """Tests for audit_lake_formation_impact public method."""

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
        self.fg.feature_group_status = "Created"

    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_run_all_audit_queries")
    @patch.object(FeatureGroupManager, "_format_risk_report", return_value="report")
    def test_returns_audit_results(self, mock_format, mock_audit, mock_refresh):
        """Test that audit results dict is returned."""
        expected = {"glue_table_accessors": [], "warnings": []}
        mock_audit.return_value = expected

        result = self.fg.audit_lake_formation_impact()

        assert result == expected

    @patch("builtins.print")
    @patch.object(FeatureGroupManager, "refresh")
    @patch.object(FeatureGroupManager, "_run_all_audit_queries", return_value={})
    @patch.object(FeatureGroupManager, "_format_risk_report", return_value="the report")
    def test_prints_formatted_report(self, mock_format, mock_audit, mock_refresh, mock_print):
        """Test that the formatted report is printed."""
        self.fg.audit_lake_formation_impact()

        mock_print.assert_called_once_with("the report")

    @patch.object(FeatureGroupManager, "refresh")
    def test_validates_fg_status(self, mock_refresh):
        """Test that non-Created status raises ValueError."""
        self.fg.feature_group_status = "Creating"

        with pytest.raises(ValueError, match="must be in 'Created' status"):
            self.fg.audit_lake_formation_impact()

    @patch.object(FeatureGroupManager, "refresh")
    def test_validates_offline_store(self, mock_refresh):
        """Test that missing offline store raises ValueError."""
        self.fg.offline_store_config = None

        with pytest.raises(ValueError, match="does not have an offline store"):
            self.fg.audit_lake_formation_impact()
