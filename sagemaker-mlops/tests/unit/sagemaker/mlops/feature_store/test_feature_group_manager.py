# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for FeatureGroupManager composition-based wrapper."""

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.mlops.feature_store.feature_group_manager import (
    FeatureGroupManager,
    LakeFormationConfig,
)
from sagemaker.core.resources import FeatureGroup
from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig


class TestFeatureGroupManagerStructure:
    """Verify FeatureGroupManager is a standalone class, not a FeatureGroup subclass."""

    def test_not_subclass_of_feature_group(self):
        """FeatureGroupManager must not inherit from FeatureGroup."""
        assert not issubclass(FeatureGroupManager, FeatureGroup)

    @pytest.mark.parametrize(
        "method_name",
        [
            "put_record",
            "get_record",
            "delete_record",
            "batch_get_record",
            "refresh",
            "wait_for_status",
            "delete",
            "update",
        ],
    )
    def test_feature_group_methods_not_on_manager(self, method_name):
        """FeatureGroup data-plane and lifecycle methods must not be callable on FeatureGroupManager."""
        assert not hasattr(FeatureGroupManager, method_name)

    def test_lake_formation_config_in_same_module(self):
        """LakeFormationConfig must be defined in the same module as FeatureGroupManager."""
        assert LakeFormationConfig.__module__ == FeatureGroupManager.__module__


class TestCreateFeatureGroup:
    """Verify create_feature_group delegation, LF workflow, and validation."""

    PATCH_FG_CREATE = "sagemaker.mlops.feature_store.feature_group_manager.FeatureGroup.create"
    PATCH_ENABLE_LF = (
        "sagemaker.mlops.feature_store.feature_group_manager.FeatureGroupManager.enable_lake_formation"
    )

    def _min_create_kwargs(self):
        """Return the minimum required kwargs for create_feature_group."""
        return dict(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[MagicMock(spec=FeatureDefinition)],
        )

    @patch(PATCH_FG_CREATE)
    def test_delegates_to_feature_group_create_and_returns_result(self, mock_create):
        """create_feature_group delegates to FeatureGroup.create() and returns the FeatureGroup."""
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_create.return_value = mock_fg

        result = FeatureGroupManager.create_feature_group(**self._min_create_kwargs())

        mock_create.assert_called_once()
        assert result is mock_fg

    @patch(PATCH_ENABLE_LF)
    @patch(PATCH_FG_CREATE)
    def test_lf_enabled_triggers_wait_and_enable(self, mock_create, mock_enable_lf):
        """When LF enabled with all prerequisites, calls create → wait_for_status → enable_lake_formation."""
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_create.return_value = mock_fg

        lf_config = LakeFormationConfig(enabled=True)
        kwargs = self._min_create_kwargs()
        kwargs["offline_store_config"] = MagicMock(spec=OfflineStoreConfig)
        kwargs["role_arn"] = "arn:aws:iam::123456789012:role/test-role"
        kwargs["lake_formation_config"] = lf_config

        result = FeatureGroupManager.create_feature_group(**kwargs)

        mock_create.assert_called_once()
        mock_fg.wait_for_status.assert_called_once_with(target_status="Created")
        mock_enable_lf.assert_called_once()
        # Verify enable_lake_formation was called with the feature_group and config
        call_args = mock_enable_lf.call_args
        assert call_args[0][0] is mock_fg  # feature_group
        assert call_args[0][1] is lf_config  # lake_formation_config
        assert call_args[1]["session"] is None  # session forwarded
        assert call_args[1]["region"] is None  # region forwarded
        assert result is mock_fg

    def test_lf_enabled_missing_offline_store_config_raises(self):
        """LF enabled without offline_store_config raises ValueError."""
        lf_config = LakeFormationConfig(enabled=True)
        kwargs = self._min_create_kwargs()
        kwargs["role_arn"] = "arn:aws:iam::123456789012:role/test-role"
        kwargs["lake_formation_config"] = lf_config
        # offline_store_config is None by default

        with pytest.raises(ValueError, match="requires offline_store_config"):
            FeatureGroupManager.create_feature_group(**kwargs)

    def test_lf_enabled_missing_role_arn_raises(self):
        """LF enabled without role_arn raises ValueError."""
        lf_config = LakeFormationConfig(enabled=True)
        kwargs = self._min_create_kwargs()
        kwargs["offline_store_config"] = MagicMock(spec=OfflineStoreConfig)
        kwargs["lake_formation_config"] = lf_config
        # role_arn is None by default

        with pytest.raises(ValueError, match="requires role_arn"):
            FeatureGroupManager.create_feature_group(**kwargs)

    def test_lf_enabled_slr_disabled_missing_registration_role_raises(self):
        """LF enabled with use_service_linked_role=False and no registration_role_arn raises ValueError."""
        lf_config = LakeFormationConfig(
            enabled=True, use_service_linked_role=False, registration_role_arn=None
        )
        kwargs = self._min_create_kwargs()
        kwargs["offline_store_config"] = MagicMock(spec=OfflineStoreConfig)
        kwargs["role_arn"] = "arn:aws:iam::123456789012:role/test-role"
        kwargs["lake_formation_config"] = lf_config

        with pytest.raises(ValueError, match="registration_role_arn must be provided"):
            FeatureGroupManager.create_feature_group(**kwargs)

    @patch(PATCH_FG_CREATE)
    def test_lf_disabled_skips_lf_operations(self, mock_create):
        """When LF config is None or enabled=False, no LF operations are performed."""
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_create.return_value = mock_fg

        # Case 1: lake_formation_config is None (default)
        result = FeatureGroupManager.create_feature_group(**self._min_create_kwargs())
        assert result is mock_fg
        mock_fg.wait_for_status.assert_not_called()

        mock_create.reset_mock()
        mock_fg.reset_mock()

        # Case 2: lake_formation_config with enabled=False
        kwargs = self._min_create_kwargs()
        kwargs["lake_formation_config"] = LakeFormationConfig(enabled=False)
        result = FeatureGroupManager.create_feature_group(**kwargs)
        assert result is mock_fg
        mock_fg.wait_for_status.assert_not_called()


class TestDescribeFeatureGroup:
    """Verify describe_feature_group delegation to FeatureGroup.get()."""

    PATCH_FG_GET = "sagemaker.mlops.feature_store.feature_group_manager.FeatureGroup.get"

    @patch(PATCH_FG_GET)
    def test_delegates_to_feature_group_get(self, mock_get):
        """describe_feature_group delegates to FeatureGroup.get() with the provided parameters."""
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_get.return_value = mock_fg

        FeatureGroupManager.describe_feature_group(
            feature_group_name="my-fg",
            next_token="token123",
            session=None,
            region="us-west-2",
        )

        mock_get.assert_called_once_with(
            feature_group_name="my-fg",
            next_token="token123",
            session=None,
            region="us-west-2",
        )

    @patch(PATCH_FG_GET)
    def test_returns_feature_group_from_get(self, mock_get):
        """describe_feature_group returns the exact FeatureGroup instance from FeatureGroup.get()."""
        mock_fg = MagicMock(spec=FeatureGroup)
        mock_get.return_value = mock_fg

        result = FeatureGroupManager.describe_feature_group(feature_group_name="my-fg")

        assert result is mock_fg


class TestEnableLakeFormation:
    """Verify enable_lake_formation 3-phase workflow, validation, fail-fast, and hybrid mode."""

    def _make_feature_group_mock(self):
        """Create a MagicMock FeatureGroup with all attributes needed by enable_lake_formation."""
        fg = MagicMock()
        fg.feature_group_status = "Created"
        fg.feature_group_name = "test-fg"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.role_arn = "arn:aws:iam::123456789012:role/test-role"

        # Offline store config
        fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = "s3://bucket/prefix"
        fg.offline_store_config.data_catalog_config.database = "my_database"
        fg.offline_store_config.data_catalog_config.table_name = "my_table"
        fg.offline_store_config.table_format = None
        return fg

    def _make_lf_config(self, **overrides):
        """Create a LakeFormationConfig with sensible defaults."""
        defaults = dict(
            enabled=True,
            use_service_linked_role=True,
            registration_role_arn=None,
            show_s3_policy=False,
            disable_hybrid_access_mode=True,
        )
        defaults.update(overrides)
        return LakeFormationConfig(**defaults)

    # --- 3-phase workflow tests ---

    def test_three_phase_workflow_executes_all_phases(self):
        """All 3 phases execute on a valid FeatureGroup with disable_hybrid_access_mode=True."""
        fg = self._make_feature_group_mock()
        lf_config = self._make_lf_config(disable_hybrid_access_mode=True)
        manager = FeatureGroupManager()

        manager._register_s3_with_lake_formation = MagicMock(return_value=True)
        manager._grant_lake_formation_permissions = MagicMock(return_value=True)
        manager._revoke_iam_allowed_principal = MagicMock(return_value=True)

        result = manager.enable_lake_formation(fg, lf_config)

        manager._register_s3_with_lake_formation.assert_called_once()
        manager._grant_lake_formation_permissions.assert_called_once()
        manager._revoke_iam_allowed_principal.assert_called_once()
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is True

    # --- Validation error tests ---

    def test_missing_offline_store_config_raises_value_error(self):
        """ValueError when FeatureGroup has no offline store config."""
        fg = self._make_feature_group_mock()
        fg.offline_store_config = None
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        with pytest.raises(ValueError, match="does not have an offline store"):
            manager.enable_lake_formation(fg, lf_config)

    def test_unassigned_offline_store_config_raises_value_error(self):
        """ValueError when FeatureGroup offline_store_config equals Unassigned()."""
        from sagemaker.core.shapes import Unassigned

        fg = self._make_feature_group_mock()
        fg.offline_store_config = Unassigned()
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        with pytest.raises(ValueError, match="does not have an offline store"):
            manager.enable_lake_formation(fg, lf_config)

    def test_missing_role_arn_raises_value_error(self):
        """ValueError when FeatureGroup has no role_arn."""
        fg = self._make_feature_group_mock()
        fg.role_arn = None
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        with pytest.raises(ValueError, match="does not have a role_arn"):
            manager.enable_lake_formation(fg, lf_config)

    def test_non_created_status_raises_value_error(self):
        """ValueError when FeatureGroup is not in 'Created' status."""
        fg = self._make_feature_group_mock()
        fg.feature_group_status = "Creating"
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        with pytest.raises(ValueError, match="must be in 'Created' status"):
            manager.enable_lake_formation(fg, lf_config)

    # --- Fail-fast behavior tests ---

    def test_phase1_failure_raises_runtime_error_and_skips_phases_2_3(self):
        """Phase 1 exception → RuntimeError, phases 2-3 not called."""
        fg = self._make_feature_group_mock()
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        manager._register_s3_with_lake_formation = MagicMock(
            side_effect=Exception("S3 registration failed")
        )
        manager._grant_lake_formation_permissions = MagicMock(return_value=True)
        manager._revoke_iam_allowed_principal = MagicMock(return_value=True)

        with pytest.raises(RuntimeError, match="Failed to register S3 location"):
            manager.enable_lake_formation(fg, lf_config)

        manager._grant_lake_formation_permissions.assert_not_called()
        manager._revoke_iam_allowed_principal.assert_not_called()

    def test_phase2_failure_raises_runtime_error_and_skips_phase_3(self):
        """Phase 2 exception → RuntimeError, phase 3 not called."""
        fg = self._make_feature_group_mock()
        lf_config = self._make_lf_config()
        manager = FeatureGroupManager()

        manager._register_s3_with_lake_formation = MagicMock(return_value=True)
        manager._grant_lake_formation_permissions = MagicMock(
            side_effect=Exception("Permission grant failed")
        )
        manager._revoke_iam_allowed_principal = MagicMock(return_value=True)

        with pytest.raises(RuntimeError, match="Failed to grant Lake Formation permissions"):
            manager.enable_lake_formation(fg, lf_config)

        manager._register_s3_with_lake_formation.assert_called_once()
        manager._revoke_iam_allowed_principal.assert_not_called()

    # --- Hybrid mode skip test ---

    def test_hybrid_mode_skips_phase3_and_sets_iam_principal_revoked_none(self):
        """When disable_hybrid_access_mode=False, Phase 3 is skipped and iam_principal_revoked=None."""
        fg = self._make_feature_group_mock()
        lf_config = self._make_lf_config(disable_hybrid_access_mode=False)
        manager = FeatureGroupManager()

        manager._register_s3_with_lake_formation = MagicMock(return_value=True)
        manager._grant_lake_formation_permissions = MagicMock(return_value=True)
        manager._revoke_iam_allowed_principal = MagicMock(return_value=True)

        result = manager.enable_lake_formation(fg, lf_config)

        manager._register_s3_with_lake_formation.assert_called_once()
        manager._grant_lake_formation_permissions.assert_called_once()
        manager._revoke_iam_allowed_principal.assert_not_called()
        assert result["s3_registration"] is True
        assert result["permissions_granted"] is True
        assert result["iam_principal_revoked"] is None


# --- Property-Based Tests (hypothesis) ---

from hypothesis import given, settings, strategies as st
import botocore.exceptions


class TestPropertyNoFeatureGroupMethodLeakage:
    """Property: FeatureGroup methods that are not part of the 4 public methods must not exist on FeatureGroupManager."""

    LEAKED_METHODS = [
        "put_record",
        "get_record",
        "delete_record",
        "batch_get_record",
        "refresh",
        "wait_for_status",
        "wait_for_delete",
        "delete",
        "update",
    ]

    @given(method_name=st.sampled_from(LEAKED_METHODS))
    @settings(max_examples=100)
    def test_feature_group_methods_do_not_leak_to_manager(self, method_name):
        """**Validates: Requirements 1.1, 1.2**"""
        assert not hasattr(FeatureGroupManager, method_name), (
            f"FeatureGroup method '{method_name}' should not exist on FeatureGroupManager"
        )


class TestPropertyCreateFeatureGroupDelegation:
    """Property: create_feature_group delegates to FeatureGroup.create and returns its result."""

    @given(feature_group_name=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_create_delegates_to_feature_group_create(self, feature_group_name):
        """**Validates: Requirements 2.2, 2.3**"""
        with patch(
            "sagemaker.mlops.feature_store.feature_group_manager.FeatureGroup.create"
        ) as mock_create:
            sentinel = MagicMock(spec=FeatureGroup)
            mock_create.return_value = sentinel

            result = FeatureGroupManager.create_feature_group(
                feature_group_name=feature_group_name,
                record_identifier_feature_name="rec_id",
                event_time_feature_name="evt_time",
                feature_definitions=[MagicMock(spec=FeatureDefinition)],
            )

            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args
            assert call_kwargs[1]["feature_group_name"] == feature_group_name
            assert result is sentinel


class TestPropertyCreateFeatureGroupValidation:
    """Property: create_feature_group rejects missing prerequisites when LF is enabled."""

    @given(use_service_linked_role=st.booleans())
    @settings(max_examples=100)
    def test_lf_enabled_missing_offline_store_raises(self, use_service_linked_role):
        """**Validates: Requirements 2.5, 2.6, 2.7**"""
        lf_config = LakeFormationConfig(
            enabled=True,
            use_service_linked_role=use_service_linked_role,
            registration_role_arn="arn:aws:iam::123456789012:role/reg-role" if not use_service_linked_role else None,
        )
        with pytest.raises(ValueError, match="requires offline_store_config"):
            FeatureGroupManager.create_feature_group(
                feature_group_name="test-fg",
                record_identifier_feature_name="rec_id",
                event_time_feature_name="evt_time",
                feature_definitions=[MagicMock(spec=FeatureDefinition)],
                role_arn="arn:aws:iam::123456789012:role/test-role",
                lake_formation_config=lf_config,
                # offline_store_config is None
            )

    @given(use_service_linked_role=st.booleans())
    @settings(max_examples=100)
    def test_lf_enabled_missing_role_arn_raises(self, use_service_linked_role):
        """**Validates: Requirements 2.5, 2.6, 2.7**"""
        lf_config = LakeFormationConfig(
            enabled=True,
            use_service_linked_role=use_service_linked_role,
            registration_role_arn="arn:aws:iam::123456789012:role/reg-role" if not use_service_linked_role else None,
        )
        with pytest.raises(ValueError, match="requires role_arn"):
            FeatureGroupManager.create_feature_group(
                feature_group_name="test-fg",
                record_identifier_feature_name="rec_id",
                event_time_feature_name="evt_time",
                feature_definitions=[MagicMock(spec=FeatureDefinition)],
                offline_store_config=MagicMock(spec=OfflineStoreConfig),
                lake_formation_config=lf_config,
                # role_arn is None
            )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_lf_enabled_slr_disabled_missing_registration_role_raises(self, data):
        """**Validates: Requirements 2.5, 2.6, 2.7**"""
        lf_config = LakeFormationConfig(
            enabled=True,
            use_service_linked_role=False,
            registration_role_arn=None,
        )
        with pytest.raises(ValueError, match="registration_role_arn must be provided"):
            FeatureGroupManager.create_feature_group(
                feature_group_name="test-fg",
                record_identifier_feature_name="rec_id",
                event_time_feature_name="evt_time",
                feature_definitions=[MagicMock(spec=FeatureDefinition)],
                offline_store_config=MagicMock(spec=OfflineStoreConfig),
                role_arn="arn:aws:iam::123456789012:role/test-role",
                lake_formation_config=lf_config,
            )


class TestPropertyDescribeFeatureGroupDelegation:
    """Property: describe_feature_group delegates to FeatureGroup.get and returns its result."""

    @given(feature_group_name=st.text(min_size=1, max_size=50))
    @settings(max_examples=100)
    def test_describe_delegates_to_feature_group_get(self, feature_group_name):
        """**Validates: Requirements 3.2, 3.3**"""
        with patch(
            "sagemaker.mlops.feature_store.feature_group_manager.FeatureGroup.get"
        ) as mock_get:
            sentinel = MagicMock(spec=FeatureGroup)
            mock_get.return_value = sentinel

            result = FeatureGroupManager.describe_feature_group(
                feature_group_name=feature_group_name,
            )

            mock_get.assert_called_once()
            assert mock_get.call_args[1]["feature_group_name"] == feature_group_name
            assert result is sentinel


class TestPropertyEnableLakeFormationValidation:
    """Property: enable_lake_formation rejects invalid FeatureGroup state."""

    INVALID_STATUSES = ["Creating", "DeleteFailed", "Deleting"]

    @given(status=st.sampled_from(INVALID_STATUSES))
    @settings(max_examples=100)
    def test_non_created_status_raises_value_error(self, status):
        """**Validates: Requirements 5.6, 5.7, 5.8**"""
        fg = MagicMock()
        fg.feature_group_status = status
        fg.feature_group_name = "test-fg"
        fg.offline_store_config = MagicMock()
        fg.role_arn = "arn:aws:iam::123456789012:role/test-role"

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(enabled=True)

        with pytest.raises(ValueError, match="must be in 'Created' status"):
            manager.enable_lake_formation(fg, lf_config)

    @given(data=st.data())
    @settings(max_examples=100)
    def test_missing_offline_store_raises_value_error(self, data):
        """**Validates: Requirements 5.6, 5.7, 5.8**"""
        fg = MagicMock()
        fg.feature_group_status = "Created"
        fg.feature_group_name = "test-fg"
        fg.offline_store_config = None
        fg.role_arn = "arn:aws:iam::123456789012:role/test-role"

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(enabled=True)

        with pytest.raises(ValueError, match="does not have an offline store"):
            manager.enable_lake_formation(fg, lf_config)

    @given(data=st.data())
    @settings(max_examples=100)
    def test_missing_role_arn_raises_value_error(self, data):
        """**Validates: Requirements 5.6, 5.7, 5.8**"""
        fg = MagicMock()
        fg.feature_group_status = "Created"
        fg.feature_group_name = "test-fg"
        fg.offline_store_config = MagicMock()
        fg.role_arn = None

        manager = FeatureGroupManager()
        lf_config = LakeFormationConfig(enabled=True)

        with pytest.raises(ValueError, match="does not have a role_arn"):
            manager.enable_lake_formation(fg, lf_config)


class TestPropertyEnableLakeFormationFailFast:
    """Property: If Phase N raises an exception, all phases after N do not execute and RuntimeError is raised."""

    def _make_fg_mock(self):
        fg = MagicMock()
        fg.feature_group_status = "Created"
        fg.feature_group_name = "test-fg"
        fg.feature_group_arn = "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg"
        fg.role_arn = "arn:aws:iam::123456789012:role/test-role"
        fg.offline_store_config.s3_storage_config.resolved_output_s3_uri = "s3://bucket/prefix"
        fg.offline_store_config.data_catalog_config.database = "my_database"
        fg.offline_store_config.data_catalog_config.table_name = "my_table"
        fg.offline_store_config.table_format = None
        return fg

    @given(failing_phase=st.integers(min_value=1, max_value=3))
    @settings(max_examples=100)
    def test_fail_fast_skips_subsequent_phases(self, failing_phase):
        """**Validates: Requirements 5.3, 5.4, 5.5**"""
        fg = self._make_fg_mock()
        lf_config = LakeFormationConfig(enabled=True, disable_hybrid_access_mode=True)
        manager = FeatureGroupManager()

        phase1_effect = MagicMock(return_value=True)
        phase2_effect = MagicMock(return_value=True)
        phase3_effect = MagicMock(return_value=True)

        if failing_phase == 1:
            phase1_effect = MagicMock(side_effect=Exception("Phase 1 failed"))
        elif failing_phase == 2:
            phase2_effect = MagicMock(side_effect=Exception("Phase 2 failed"))
        elif failing_phase == 3:
            phase3_effect = MagicMock(side_effect=Exception("Phase 3 failed"))

        with patch.object(manager, "_register_s3_with_lake_formation", phase1_effect), \
             patch.object(manager, "_grant_lake_formation_permissions", phase2_effect), \
             patch.object(manager, "_revoke_iam_allowed_principal", phase3_effect):

            with pytest.raises(RuntimeError):
                manager.enable_lake_formation(fg, lf_config)

            # Verify phases after the failing one were not called
            if failing_phase == 1:
                phase2_effect.assert_not_called()
                phase3_effect.assert_not_called()
            elif failing_phase == 2:
                phase1_effect.assert_called_once()
                phase3_effect.assert_not_called()
            elif failing_phase == 3:
                phase1_effect.assert_called_once()
                phase2_effect.assert_called_once()


class TestPropertyS3UriToArnConversion:
    """Property: ARN inputs are returned unchanged; valid S3 URIs produce correct ARN format."""

    REGIONS = ["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1", "cn-north-1", "us-gov-west-1"]

    @given(
        bucket=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-"),
            min_size=3,
            max_size=30,
        ),
        key=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="-/"),
            min_size=1,
            max_size=30,
        ).filter(lambda k: k.strip("/") != ""),
        region=st.sampled_from(REGIONS),
    )
    @settings(max_examples=100)
    def test_s3_uri_converts_to_correct_arn(self, bucket, key, region):
        """**Validates: Requirements 6.1**"""
        s3_uri = f"s3://{bucket}/{key}"
        result = FeatureGroupManager._s3_uri_to_arn(s3_uri, region=region)

        if region and region.startswith("cn-"):
            partition = "aws-cn"
        elif region and region.startswith("us-gov-"):
            partition = "aws-us-gov"
        else:
            partition = "aws"

        # parse_s3_url strips leading slashes from the key
        normalized_key = key.lstrip("/")
        expected = f"arn:{partition}:s3:::{bucket}/{normalized_key}"
        assert result == expected

    @given(
        arn_suffix=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=100)
    def test_arn_input_returned_unchanged(self, arn_suffix):
        """**Validates: Requirements 6.1**"""
        arn_input = f"arn:{arn_suffix}"
        result = FeatureGroupManager._s3_uri_to_arn(arn_input)
        assert result == arn_input


class TestPropertyLFHelperExceptionHandling:
    """Property: LF helpers handle expected exceptions gracefully and propagate others."""

    GRACEFUL_REGISTER_CODE = "AlreadyExistsException"
    GRACEFUL_REVOKE_GRANT_CODE = "InvalidInputException"
    OTHER_ERROR_CODES = ["AccessDeniedException", "InternalServiceException", "OperationTimeoutException"]

    def _make_client_error(self, code, message="test error"):
        return botocore.exceptions.ClientError(
            {"Error": {"Code": code, "Message": message}},
            "TestOperation",
        )

    @given(data=st.data())
    @settings(max_examples=100)
    def test_register_s3_handles_already_exists_gracefully(self, data):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.register_resource.side_effect = self._make_client_error(
            self.GRACEFUL_REGISTER_CODE
        )

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            result = manager._register_s3_with_lake_formation("s3://bucket/prefix")
            assert result is True

    @given(data=st.data())
    @settings(max_examples=100)
    def test_revoke_iam_handles_invalid_input_gracefully(self, data):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.revoke_permissions.side_effect = self._make_client_error(
            self.GRACEFUL_REVOKE_GRANT_CODE
        )

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            result = manager._revoke_iam_allowed_principal("db", "table")
            assert result is True

    @given(data=st.data())
    @settings(max_examples=100)
    def test_grant_permissions_handles_invalid_input_gracefully(self, data):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.grant_permissions.side_effect = self._make_client_error(
            self.GRACEFUL_REVOKE_GRANT_CODE
        )

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            result = manager._grant_lake_formation_permissions(
                "arn:aws:iam::123456789012:role/test", "db", "table"
            )
            assert result is True

    @given(error_code=st.sampled_from(OTHER_ERROR_CODES))
    @settings(max_examples=100)
    def test_register_s3_propagates_other_client_errors(self, error_code):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.register_resource.side_effect = self._make_client_error(error_code)

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            with pytest.raises(botocore.exceptions.ClientError) as exc_info:
                manager._register_s3_with_lake_formation("s3://bucket/prefix")
            assert exc_info.value.response["Error"]["Code"] == error_code

    @given(error_code=st.sampled_from(OTHER_ERROR_CODES))
    @settings(max_examples=100)
    def test_revoke_iam_propagates_other_client_errors(self, error_code):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.revoke_permissions.side_effect = self._make_client_error(error_code)

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            with pytest.raises(botocore.exceptions.ClientError) as exc_info:
                manager._revoke_iam_allowed_principal("db", "table")
            assert exc_info.value.response["Error"]["Code"] == error_code

    @given(error_code=st.sampled_from(OTHER_ERROR_CODES))
    @settings(max_examples=100)
    def test_grant_permissions_propagates_other_client_errors(self, error_code):
        """**Validates: Requirements 6.6, 6.7, 6.8**"""
        manager = FeatureGroupManager()
        mock_client = MagicMock()
        mock_client.grant_permissions.side_effect = self._make_client_error(error_code)

        with patch.object(manager, "_get_lake_formation_client", return_value=mock_client):
            with pytest.raises(botocore.exceptions.ClientError) as exc_info:
                manager._grant_lake_formation_permissions(
                    "arn:aws:iam::123456789012:role/test", "db", "table"
                )
            assert exc_info.value.response["Error"]["Code"] == error_code
