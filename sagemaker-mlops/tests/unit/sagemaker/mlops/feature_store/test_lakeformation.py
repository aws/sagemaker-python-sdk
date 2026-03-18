"""Unit tests for Lake Formation integration with FeatureGroupManager."""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from sagemaker.mlops.feature_store import FeatureGroupManager, LakeFormationConfig


def _make_mock_feature_group(
    feature_group_name="test-fg",
    role_arn="arn:aws:iam::123456789012:role/TestRole",
    feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
    feature_group_status="Created",
    s3_uri="s3://test-bucket/path",
    resolved_output_s3_uri="s3://test-bucket/resolved-path",
    database="test_db",
    table_name="test_table",
    table_format=None,
    offline_store_config=None,
):
    """Helper to create a mock FeatureGroup with standard attributes for enable_lake_formation tests."""
    from sagemaker.core.resources import FeatureGroup
    from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

    mock_fg = MagicMock(spec=FeatureGroup)
    mock_fg.feature_group_name = feature_group_name
    mock_fg.role_arn = role_arn
    mock_fg.feature_group_arn = feature_group_arn
    mock_fg.feature_group_status = feature_group_status

    if offline_store_config is not None:
        mock_fg.offline_store_config = offline_store_config
    else:
        mock_fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(
                s3_uri=s3_uri,
                resolved_output_s3_uri=resolved_output_s3_uri,
            ),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database=database, table_name=table_name
            ),
            table_format=table_format,
        )

    return mock_fg


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

    def test_caches_client_for_same_session_and_region(self):
        """Test that repeated calls with the same session and region reuse the cached client."""
        mock_session = MagicMock()
        mock_client = MagicMock()
        mock_session.client.return_value = mock_client

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_lake_formation_client = FeatureGroupManager._get_lake_formation_client.__get__(fg)

        client1 = fg._get_lake_formation_client(session=mock_session, region="us-west-2")
        client2 = fg._get_lake_formation_client(session=mock_session, region="us-west-2")

        assert client1 is client2
        mock_session.client.assert_called_once()

    def test_creates_new_client_for_different_region(self):
        """Test that a different region produces a new client."""
        mock_session = MagicMock()
        mock_client_west = MagicMock()
        mock_client_east = MagicMock()
        mock_session.client.side_effect = [mock_client_west, mock_client_east]

        fg = MagicMock(spec=FeatureGroupManager)
        fg._get_lake_formation_client = FeatureGroupManager._get_lake_formation_client.__get__(fg)

        client1 = fg._get_lake_formation_client(session=mock_session, region="us-west-2")
        client2 = fg._get_lake_formation_client(session=mock_session, region="us-east-1")

        assert client1 is not client2
        assert mock_session.client.call_count == 2


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


class TestCreateFeatureGroupWithLakeFormationDisableHybridAccessMode:
    """Tests for disable_hybrid_access_mode passed through create_feature_group() via LakeFormationConfig."""

    @patch.object(FeatureGroupManager, "enable_lake_formation")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.FeatureGroup")
    def test_disable_hybrid_access_mode_false_passed_through_create(
        self, mock_fg_class, mock_enable_lf
    ):
        """Test that disable_hybrid_access_mode=False is passed through create_feature_group() to enable_lake_formation."""
        from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig, S3StorageConfig

        mock_fg = MagicMock()
        mock_fg_class.create.return_value = mock_fg

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        lf_config = LakeFormationConfig()
        lf_config.enabled = True
        lf_config.disable_hybrid_access_mode = False

        FeatureGroupManager.create_feature_group(
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

        # Verify enable_lake_formation was called with the FG and the config
        mock_enable_lf.assert_called_once_with(mock_fg, lf_config, session=None, region=None)
        # Verify the config has the right value
        passed_config = mock_enable_lf.call_args[0][1]
        assert passed_config.disable_hybrid_access_mode is False


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



class TestGenerateS3DenyPolicy:
    """Tests for _generate_s3_deny_policy method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._generate_s3_deny_policy = FeatureGroupManager._generate_s3_deny_policy.__get__(self.fg)

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

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_uses_service_linked_role_arn_when_use_service_linked_role_true(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that enable_lake_formation uses the auto-generated service-linked role ARN
        when use_service_linked_role=True.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(
            use_service_linked_role=True,
            show_s3_policy=True,
        )

        manager.enable_lake_formation(mock_fg, lf_config)

        # Verify _generate_s3_deny_policy was called with the service-linked role ARN
        expected_slr_arn = "arn:aws:iam::123456789012:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert call_kwargs["feature_store_role_arn"] == "arn:aws:iam::123456789012:role/FeatureStoreRole"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_uses_service_linked_role_arn_by_default(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that enable_lake_formation uses the service-linked role ARN by default
        (when use_service_linked_role is not explicitly specified).
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::987654321098:role/MyFeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-east-1:987654321098:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(show_s3_policy=True)  # use_service_linked_role defaults to True

        manager.enable_lake_formation(mock_fg, lf_config)

        # Verify _generate_s3_deny_policy was called with the service-linked role ARN
        expected_slr_arn = "arn:aws:iam::987654321098:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_service_linked_role_arn_uses_correct_account_id(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the service-linked role ARN is generated with the correct account ID
        extracted from the Feature Group ARN.
        """
        account_id = "111222333444"
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn=f"arn:aws:iam::{account_id}:role/FeatureStoreRole",
            feature_group_arn=f"arn:aws:sagemaker:us-west-2:{account_id}:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(
            use_service_linked_role=True,
            show_s3_policy=True,
        )

        manager.enable_lake_formation(mock_fg, lf_config)

        # Verify the service-linked role ARN contains the correct account ID
        expected_slr_arn = f"arn:aws:iam::{account_id}:role/aws-service-role/lakeformation.amazonaws.com/AWSServiceRoleForLakeFormationDataAccess"
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == expected_slr_arn
        assert account_id in call_kwargs["lake_formation_role_arn"]



class TestRegistrationRoleArnUsedWhenServiceLinkedRoleFalse:
    """Tests for verifying registration_role_arn is used when use_service_linked_role=False."""

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_uses_registration_role_arn_when_use_service_linked_role_false(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that when use_service_linked_role=False, the registration_role_arn is used
        in the S3 deny policy instead of the auto-generated service-linked role ARN.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            show_s3_policy=True,
        )

        manager.enable_lake_formation(mock_fg, lf_config)

        # Verify _generate_s3_deny_policy was called with the custom registration role ARN
        mock_generate_policy.assert_called_once()
        call_kwargs = mock_generate_policy.call_args[1]
        assert call_kwargs["lake_formation_role_arn"] == custom_registration_role

        # Verify it's NOT the service-linked role ARN
        service_linked_role_pattern = "aws-service-role/lakeformation.amazonaws.com"
        assert service_linked_role_pattern not in call_kwargs["lake_formation_role_arn"]

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_registration_role_arn_passed_to_s3_registration(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that when use_service_linked_role=False, the registration_role_arn is also
        passed to _register_s3_with_lake_formation.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        custom_registration_role = "arn:aws:iam::123456789012:role/CustomLakeFormationRole"

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(
            use_service_linked_role=False,
            registration_role_arn=custom_registration_role,
            show_s3_policy=True,
        )

        manager.enable_lake_formation(mock_fg, lf_config)

        # Verify _register_s3_with_lake_formation was called with the correct parameters
        mock_register.assert_called_once()
        call_args = mock_register.call_args
        assert call_args[1]["use_service_linked_role"] == False
        assert call_args[1]["role_arn"] == custom_registration_role

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch.object(FeatureGroupManager, "_generate_s3_deny_policy")
    def test_different_registration_role_arns_produce_different_policies(
        self,
        mock_generate_policy,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that different registration_role_arn values result in different
        lake_formation_role_arn values in the generated policy.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True
        mock_generate_policy.return_value = {"Version": "2012-10-17", "Statement": []}

        manager = FeatureGroupManager()

        # First call with one registration role
        first_role = "arn:aws:iam::123456789012:role/FirstLakeFormationRole"
        lf_config_1 = LakeFormationConfig(
            use_service_linked_role=False,
            registration_role_arn=first_role,
            show_s3_policy=True,
        )
        manager.enable_lake_formation(mock_fg, lf_config_1)

        first_call_kwargs = mock_generate_policy.call_args[1]
        first_lf_role = first_call_kwargs["lake_formation_role_arn"]

        # Reset mocks
        mock_generate_policy.reset_mock()
        mock_register.reset_mock()
        mock_grant.reset_mock()
        mock_revoke.reset_mock()

        # Second call with different registration role
        second_role = "arn:aws:iam::123456789012:role/SecondLakeFormationRole"
        lf_config_2 = LakeFormationConfig(
            use_service_linked_role=False,
            registration_role_arn=second_role,
            show_s3_policy=True,
        )
        manager.enable_lake_formation(mock_fg, lf_config_2)

        second_call_kwargs = mock_generate_policy.call_args[1]
        second_lf_role = second_call_kwargs["lake_formation_role_arn"]

        # Verify different roles were used
        assert first_lf_role == first_role
        assert second_lf_role == second_role
        assert first_lf_role != second_lf_role



class TestPolicyPrintedWithClearInstructions:
    """Tests for verifying the S3 deny policy is printed with clear instructions."""

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_printed_with_header_and_instructions(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that enable_lake_formation logs the S3 deny policy with clear
        header and instructions for the user.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(show_s3_policy=True)

        manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify header is logged
        assert "S3 Bucket Policy" in all_logged_text, "Header should mention 'S3 Bucket Policy'"

        # Verify instructions are logged
        assert (
            "Lake Formation" in all_logged_text
            or "deny policy" in all_logged_text
        ), "Instructions should mention Lake Formation or deny policy"

        # Verify bucket name is logged
        assert "test-bucket" in all_logged_text, "Bucket name should be logged"

        # Verify note about merging with existing policy is logged
        assert (
            "Merge" in all_logged_text or "existing" in all_logged_text
        ), "Note about merging with existing policy should be logged"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_json_is_printed(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the S3 deny policy JSON is logged to the console when show_s3_policy=True.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(show_s3_policy=True)

        manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy JSON structure elements are logged
        assert "Version" in all_logged_text, "Policy JSON should contain 'Version'"
        assert "Statement" in all_logged_text, "Policy JSON should contain 'Statement'"
        assert "Effect" in all_logged_text, "Policy JSON should contain 'Effect'"
        assert "Deny" in all_logged_text, "Policy JSON should contain 'Deny' effect"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_printed_only_after_successful_setup(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the S3 deny policy is only logged after all Lake Formation
        phases complete successfully.
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(show_s3_policy=True)

        # Mock Phase 1 failure
        mock_register.side_effect = Exception("Phase 1 failed")

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy was NOT logged when setup failed
        assert "S3 Bucket Policy" not in all_logged_text, "Policy should not be logged when setup fails"

        # Reset mocks
        mock_logger.reset_mock()
        mock_register.reset_mock()
        mock_register.side_effect = None
        mock_register.return_value = True

        # Mock Phase 2 failure
        mock_grant.side_effect = Exception("Phase 2 failed")

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy was NOT logged when setup fails at Phase 2
        assert "S3 Bucket Policy" not in all_logged_text, "Policy should not be logged when Phase 2 fails"

        # Reset mocks
        mock_logger.reset_mock()
        mock_grant.reset_mock()
        mock_grant.side_effect = None
        mock_grant.return_value = True

        # Mock Phase 3 failure
        mock_revoke.side_effect = Exception("Phase 3 failed")

        # Call enable_lake_formation with show_s3_policy=True - should fail
        with pytest.raises(RuntimeError):
            manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy was NOT logged when setup fails at Phase 3
        assert "S3 Bucket Policy" not in all_logged_text, "Policy should not be logged when Phase 3 fails"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_includes_both_allowed_principals(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the logged policy includes both the Lake Formation role
        and the Feature Store execution role as allowed principals.
        """
        feature_store_role = "arn:aws:iam::123456789012:role/FeatureStoreRole"
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn=feature_store_role,
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(
            use_service_linked_role=True,
            show_s3_policy=True,
        )

        manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify Feature Store role is in the logged output
        assert feature_store_role in all_logged_text, "Feature Store role should be in logged policy"

        # Verify Lake Formation service-linked role pattern is in the logged output
        assert "AWSServiceRoleForLakeFormationDataAccess" in all_logged_text, \
            "Lake Formation service-linked role should be in logged policy"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_not_printed_when_show_s3_policy_false(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the S3 deny policy is NOT logged when show_s3_policy=False (default).
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(show_s3_policy=False)

        manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy was NOT logged
        assert "S3 Bucket Policy" not in all_logged_text, "Policy should not be logged when show_s3_policy=False"
        assert "Version" not in all_logged_text, "Policy JSON should not be logged when show_s3_policy=False"

    @patch.object(FeatureGroupManager, "_register_s3_with_lake_formation")
    @patch.object(FeatureGroupManager, "_grant_lake_formation_permissions")
    @patch.object(FeatureGroupManager, "_revoke_iam_allowed_principal")
    @patch("sagemaker.mlops.feature_store.feature_group_manager.logger")
    def test_policy_not_printed_by_default(
        self,
        mock_logger,
        mock_revoke,
        mock_grant,
        mock_register,
    ):
        """
        Test that the S3 deny policy is NOT logged by default (when show_s3_policy is not specified).
        """
        mock_fg = _make_mock_feature_group(
            resolved_output_s3_uri="s3://test-bucket/resolved-path/data",
            role_arn="arn:aws:iam::123456789012:role/FeatureStoreRole",
            feature_group_arn="arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
        )

        mock_register.return_value = True
        mock_grant.return_value = True
        mock_revoke.return_value = True

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig()  # show_s3_policy defaults to False

        manager.enable_lake_formation(mock_fg, lf_config)

        # Collect all logger.info calls
        log_calls = [str(call) for call in mock_logger.info.call_args_list]
        all_logged_text = " ".join(log_calls)

        # Verify policy was NOT logged
        assert "S3 Bucket Policy" not in all_logged_text, "Policy should not be logged by default"
        assert "Version" not in all_logged_text, "Policy JSON should not be logged by default"
