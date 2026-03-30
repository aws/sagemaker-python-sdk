"""Unit tests for Iceberg properties in FeatureGroupManager."""
from unittest.mock import MagicMock, patch

import botocore.exceptions
import pytest

from boto3 import Session
from sagemaker.mlops.feature_store import FeatureGroupManager
from sagemaker.mlops.feature_store.feature_group_manager import IcebergProperties


class TestIcebergPropertiesConfig:
    """Tests for IcebergProperties default values."""

    def test_properties_defaults_to_none(self):
        """Test that properties defaults to None."""
        config = IcebergProperties()
        assert config.properties is None

    def test_properties_can_be_set(self):
        """Test that properties can be set with a dict."""
        props = {"write.target-file-size-bytes": "536870912"}
        config = IcebergProperties(properties=props)
        assert config.properties == props

    def test_valid_approved_keys_accepted(self):
        """Test that all approved keys are accepted."""
        props = {
            "write.target-file-size-bytes": "536870912",
            "write.metadata.delete-after-commit.enabled": "true",
            "history.expire.max-snapshot-age-ms": "432000000",
        }
        config = IcebergProperties(properties=props)
        assert config.properties == props

    def test_single_invalid_key_raises_error(self):
        """Test that a single invalid key raises ValueError."""
        with pytest.raises(ValueError, match="Invalid iceberg properties"):
            IcebergProperties(properties={"not.a.valid.key": "value"})

    def test_multiple_invalid_keys_raises_error(self):
        """Test that multiple invalid keys raise ValueError."""
        with pytest.raises(ValueError, match="Invalid iceberg properties"):
            IcebergProperties(properties={"bad.key.one": "1", "bad.key.two": "2"})

    def test_mix_valid_and_invalid_keys_raises_error(self):
        """Test that a mix of valid and invalid keys raises ValueError."""
        with pytest.raises(ValueError, match="Invalid iceberg properties"):
            IcebergProperties(properties={
                "write.target-file-size-bytes": "536870912",
                "invalid.key": "value",
            })

    def test_error_message_contains_invalid_key_names(self):
        """Test that the error message includes the invalid key names."""
        with pytest.raises(ValueError, match="fake.property"):
            IcebergProperties(properties={"fake.property": "value"})

    def test_duplicate_keys_raises_error(self):
        """Test that duplicate property keys raise ValueError."""
        config = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})
        mock_props = MagicMock()
        mock_props.keys.return_value = [
            "write.target-file-size-bytes",
            "write.target-file-size-bytes",
        ]
        object.__setattr__(config, "properties", mock_props)
        with pytest.raises(ValueError, match="Invalid duplicate properties"):
            config.validate_property_keys()

    def test_no_duplicate_keys_passes(self):
        """Test that unique approved keys pass duplicate validation."""
        config = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})
        result = config.validate_property_keys()
        assert result is config


class TestGetIcebergProperties:
    """Tests for get_iceberg_properties method."""

    def setup_method(self):
        """Set up test fixtures."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._get_iceberg_properties = FeatureGroupManager._get_iceberg_properties.__get__(self.fg)
        self.fg.feature_group_name = "test-fg"
        self.fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri="s3://test-bucket/path"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="test_db", table_name="test_table"
            ),
            table_format="Iceberg",
        )

    def test_raises_error_when_no_offline_store_config(self):
        """Test ValueError when offline_store_config is None."""
        self.fg.offline_store_config = None

        with pytest.raises(ValueError, match="offline_store_config is not configured"):
            self.fg._get_iceberg_properties()

    def test_raises_error_when_offline_store_config_is_unassigned(self):
        """Test ValueError when offline_store_config is Unassigned()."""
        from sagemaker.core.shapes import Unassigned

        self.fg.offline_store_config = Unassigned()

        with pytest.raises(ValueError, match="offline_store_config is not configured"):
            self.fg._get_iceberg_properties()

    def test_raises_error_when_table_format_not_iceberg(self):
        """Test ValueError when table_format is not Iceberg."""
        self.fg.offline_store_config.table_format = None

        with pytest.raises(ValueError, match="table_format must be 'Iceberg'"):
            self.fg._get_iceberg_properties()

    def test_raises_error_when_no_data_catalog_config(self):
        """Test ValueError when data_catalog_config is None."""
        self.fg.offline_store_config.data_catalog_config = None

        with pytest.raises(ValueError, match="data_catalog_config is not available"):
            self.fg._get_iceberg_properties()

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_successful_get_table_and_field_stripping(self, mock_session_class):
        """Test successful Glue get_table call strips non-TableInput fields."""
        mock_session = MagicMock()
        mock_glue_client = MagicMock()
        mock_session.client.return_value = mock_glue_client
        mock_session_class.return_value = mock_session

        mock_glue_client.get_table.return_value = {
            "Table": {
                "Name": "test_table",
                "Parameters": {"table_type": "ICEBERG"},
                "DatabaseName": "test_db",
                "CreateTime": "2024-01-01",
                "UpdateTime": "2024-01-02",
                "CreatedBy": "user",
                "IsRegisteredWithLakeFormation": False,
                "CatalogId": "123456789012",
                "VersionId": "1",
                "FederatedTable": {},
            }
        }

        result = self.fg._get_iceberg_properties()

        assert result["database_name"] == "test_db"
        assert result["table_name"] == "test_table"
        assert result["glue_client"] == mock_glue_client
        # Verify stripped fields are not in table_input
        for field in ["DatabaseName", "CreateTime", "UpdateTime", "CreatedBy",
                      "IsRegisteredWithLakeFormation", "CatalogId", "VersionId", "FederatedTable"]:
            assert field not in result["table_input"]
        # Verify kept fields remain
        assert result["table_input"]["Name"] == "test_table"
        assert result["table_input"]["Parameters"] == {"table_type": "ICEBERG"}

    def test_uses_provided_session_and_region(self):
        """Test that provided session and region are used instead of defaults."""
        mock_session = MagicMock()
        mock_glue_client = MagicMock()
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.return_value = {
            "Table": {"Name": "test_table"}
        }

        result = self.fg._get_iceberg_properties(session=mock_session, region="eu-west-1")

        mock_session.client.assert_called_once_with("glue", region_name="eu-west-1")

    def test_uses_session_region_when_region_not_provided(self):
        """Test that session.region_name is used when region is None."""
        mock_session = MagicMock()
        mock_session.region_name = "ap-southeast-1"
        mock_glue_client = MagicMock()
        mock_session.client.return_value = mock_glue_client

        mock_glue_client.get_table.return_value = {
            "Table": {"Name": "test_table"}
        }

        result = self.fg._get_iceberg_properties(session=mock_session)

        mock_session.client.assert_called_once_with("glue", region_name="ap-southeast-1")

    @patch("sagemaker.mlops.feature_store.feature_group_manager.Session")
    def test_raises_runtime_error_on_client_error(self, mock_session_class):
        """Test RuntimeError wrapping ClientError from Glue."""
        mock_session = MagicMock()
        mock_glue_client = MagicMock()
        mock_session.client.return_value = mock_glue_client
        mock_session_class.return_value = mock_session

        mock_glue_client.get_table.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "EntityNotFoundException", "Message": "Table not found"}},
            "GetTable",
        )

        with pytest.raises(RuntimeError, match="Failed to update Iceberg properties"):
            self.fg._get_iceberg_properties()


class TestUpdateIcebergProperties:
    """Tests for update_iceberg_properties method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.fg = MagicMock(spec=FeatureGroupManager)
        self.fg._update_iceberg_properties = FeatureGroupManager._update_iceberg_properties.__get__(self.fg)
        self.fg.feature_group_name = "test-fg"

    def test_raises_error_when_iceberg_properties_is_none(self):
        """Test ValueError when iceberg_properties is None."""
        with pytest.raises(ValueError, match="must contain at least one property"):
            self.fg._update_iceberg_properties(iceberg_properties=None)

    def test_raises_error_when_properties_dict_is_empty(self):
        """Test ValueError when properties dict is empty."""
        props = IcebergProperties(properties={})

        with pytest.raises(ValueError, match="must contain at least one property"):
            self.fg._update_iceberg_properties(iceberg_properties=props)

    def test_raises_error_when_properties_is_none_on_object(self):
        """Test ValueError when IcebergProperties.properties is None."""
        props = IcebergProperties()

        with pytest.raises(ValueError, match="must contain at least one property"):
            self.fg._update_iceberg_properties(iceberg_properties=props)

    def test_successful_update_merges_properties(self):
        """Test successful update merges properties and calls update_table."""
        mock_glue_client = MagicMock()
        self.fg._get_iceberg_properties.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {
                "Name": "test_table",
                "Parameters": {"table_type": "ICEBERG", "existing_key": "existing_value"},
            },
            "glue_client": mock_glue_client,
        }

        props = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})
        result = self.fg._update_iceberg_properties(iceberg_properties=props)

        # Verify update_table was called with merged parameters
        call_args = mock_glue_client.update_table.call_args
        updated_params = call_args[1]["TableInput"]["Parameters"]
        assert updated_params["table_type"] == "ICEBERG"
        assert updated_params["existing_key"] == "existing_value"
        assert updated_params["write.target-file-size-bytes"] == "536870912"

        assert result["database"] == "test_db"
        assert result["table"] == "test_table"
        assert result["properties_updated"] == props.properties

    def test_creates_parameters_dict_when_missing(self):
        """Test that Parameters dict is created when not present in table_input."""
        mock_glue_client = MagicMock()
        self.fg._get_iceberg_properties.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {"Name": "test_table"},
            "glue_client": mock_glue_client,
        }

        props = IcebergProperties(properties={"write.target-file-size-bytes": "value"})
        result = self.fg._update_iceberg_properties(iceberg_properties=props)

        call_args = mock_glue_client.update_table.call_args
        updated_params = call_args[1]["TableInput"]["Parameters"]
        assert updated_params == {"write.target-file-size-bytes": "value"}

    def test_raises_runtime_error_on_update_table_client_error(self):
        """Test RuntimeError wrapping ClientError from Glue update_table."""
        mock_glue_client = MagicMock()
        self.fg._get_iceberg_properties.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {"Name": "test_table", "Parameters": {}},
            "glue_client": mock_glue_client,
        }
        mock_glue_client.update_table.side_effect = botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Access denied"}},
            "UpdateTable",
        )

        props = IcebergProperties(properties={"write.target-file-size-bytes": "value"})

        with pytest.raises(RuntimeError, match="Failed to update Iceberg properties"):
            self.fg._update_iceberg_properties(iceberg_properties=props)

    def test_raises_error_on_duplicate_keys(self):
        """Test ValueError when iceberg_properties has duplicate keys."""
        props = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})
        mock_props = MagicMock()
        mock_props.keys.return_value = [
            "write.target-file-size-bytes",
            "write.target-file-size-bytes",
        ]
        mock_props.__bool__ = lambda self: True
        object.__setattr__(props, "properties", mock_props)

        with pytest.raises(ValueError, match="Invalid duplicate properties"):
            self.fg._update_iceberg_properties(iceberg_properties=props)


class TestCreateWithIcebergProperties:
    """Tests for create() method with iceberg_properties parameter."""

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    def test_no_iceberg_operations_when_none(self, mock_get, mock_get_client):
        """Test no iceberg operations when iceberg_properties is None."""
        from sagemaker.core.shapes import FeatureDefinition

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_get.return_value = mock_fg

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
        )

        mock_fg.wait_for_status.assert_not_called()
        mock_fg._update_iceberg_properties.assert_not_called()

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    def test_no_iceberg_operations_when_properties_empty(self, mock_get, mock_get_client):
        """Test no iceberg operations when iceberg_properties.properties is empty."""
        from sagemaker.core.shapes import FeatureDefinition

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_get.return_value = mock_fg

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            iceberg_properties=IcebergProperties(),
        )

        mock_fg._update_iceberg_properties.assert_not_called()

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_without_offline_store_config(self, mock_get_client):
        """Test ValueError when iceberg_properties provided without offline_store_config."""
        from sagemaker.core.shapes import FeatureDefinition

        mock_get_client.return_value = MagicMock()

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        with pytest.raises(ValueError, match="iceberg_properties requires offline_store_config"):
            FeatureGroupManager.create(
                feature_group_name="test-fg",
                record_identifier_feature_name="record_id",
                event_time_feature_name="event_time",
                feature_definitions=feature_definitions,
                iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "value"}),
            )

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_table_format_not_iceberg(self, mock_get_client):
        """Test ValueError when table_format is not Iceberg."""
        from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig, S3StorageConfig

        mock_get_client.return_value = MagicMock()

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        with pytest.raises(ValueError, match="table_format to be 'Iceberg'"):
            FeatureGroupManager.create(
                feature_group_name="test-fg",
                record_identifier_feature_name="record_id",
                event_time_feature_name="event_time",
                feature_definitions=feature_definitions,
                offline_store_config=OfflineStoreConfig(
                    s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
                ),
                iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "value"}),
            )

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_table_format_is_glue(self, mock_get_client):
        """Test ValueError when table_format is explicitly Glue."""
        from sagemaker.core.shapes import FeatureDefinition, OfflineStoreConfig, S3StorageConfig

        mock_get_client.return_value = MagicMock()

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        with pytest.raises(ValueError, match="table_format to be 'Iceberg'"):
            FeatureGroupManager.create(
                feature_group_name="test-fg",
                record_identifier_feature_name="record_id",
                event_time_feature_name="event_time",
                feature_definitions=feature_definitions,
                offline_store_config=OfflineStoreConfig(
                    s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
                    table_format="Glue",
                ),
                iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "value"}),
            )

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    def test_update_called_after_create_with_properties(
        self, mock_update, mock_wait, mock_get, mock_get_client
    ):
        """Test update_iceberg_properties called after create when properties provided."""
        from sagemaker.core.shapes import (
            FeatureDefinition,
            OfflineStoreConfig,
            S3StorageConfig,
            DataCatalogConfig,
        )

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.wait_for_status = mock_wait
        mock_fg._update_iceberg_properties = mock_update
        mock_get.return_value = mock_fg

        feature_definitions = [
            FeatureDefinition(feature_name="record_id", feature_type="String"),
            FeatureDefinition(feature_name="event_time", feature_type="String"),
        ]

        iceberg_props = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})

        result = FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=feature_definitions,
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
                data_catalog_config=DataCatalogConfig(
                    catalog="AwsDataCatalog", database="test_db", table_name="test_table"
                ),
                table_format="Iceberg",
            ),
            iceberg_properties=iceberg_props,
        )

        # Verify wait_for_status called before update
        mock_wait.assert_called_once_with(target_status="Created")
        mock_update.assert_called_once_with(
            iceberg_properties=iceberg_props,
            session=None,
            region=None,
        )
        assert result == mock_fg

    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    @patch.object(FeatureGroupManager, "get")
    @patch.object(FeatureGroupManager, "wait_for_status")
    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    def test_create_passes_session_and_region_to_update(
        self, mock_update, mock_wait, mock_get, mock_get_client
    ):
        """Test that session and region are forwarded to _update_iceberg_properties."""
        from sagemaker.core.shapes import (
            FeatureDefinition,
            OfflineStoreConfig,
            S3StorageConfig,
            DataCatalogConfig,
        )

        mock_client = MagicMock()
        mock_client.create_feature_group.return_value = {
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test"
        }
        mock_get_client.return_value = mock_client

        mock_fg = MagicMock(spec=FeatureGroupManager)
        mock_fg.wait_for_status = mock_wait
        mock_fg._update_iceberg_properties = mock_update
        mock_get.return_value = mock_fg

        mock_session = MagicMock(spec=Session)
        iceberg_props = IcebergProperties(properties={"write.target-file-size-bytes": "val"})

        FeatureGroupManager.create(
            feature_group_name="test-fg",
            record_identifier_feature_name="record_id",
            event_time_feature_name="event_time",
            feature_definitions=[
                FeatureDefinition(feature_name="record_id", feature_type="String"),
                FeatureDefinition(feature_name="event_time", feature_type="String"),
            ],
            offline_store_config=OfflineStoreConfig(
                s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
                data_catalog_config=DataCatalogConfig(
                    catalog="AwsDataCatalog", database="db", table_name="tbl"
                ),
                table_format="Iceberg",
            ),
            iceberg_properties=iceberg_props,
            session=mock_session,
            region="eu-west-1",
        )

        mock_update.assert_called_once_with(
            iceberg_properties=iceberg_props,
            session=mock_session,
            region="eu-west-1",
        )


class TestUpdateWithIcebergProperties:
    """Tests for update() method with iceberg_properties parameter."""

    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_no_iceberg_operations_when_none(self, mock_get_client, mock_refresh, mock_update_iceberg):
        """Test no iceberg operations when iceberg_properties is None."""
        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.update(description="new description")

        mock_update_iceberg.assert_not_called()

    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_no_iceberg_operations_when_properties_empty(self, mock_get_client, mock_refresh, mock_update_iceberg):
        """Test no iceberg operations when iceberg_properties.properties is None."""
        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.update(iceberg_properties=IcebergProperties())

        mock_update_iceberg.assert_not_called()

    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_iceberg_update_called_with_properties(self, mock_get_client, mock_refresh, mock_update_iceberg):
        """Test _update_iceberg_properties called when properties provided."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="db", table_name="tbl"
            ),
            table_format="Iceberg",
        )
        iceberg_props = IcebergProperties(properties={"write.target-file-size-bytes": "536870912"})
        fg.update(description="new description", iceberg_properties=iceberg_props, session=None, region=None)

        mock_update_iceberg.assert_called_once_with(
            iceberg_properties=iceberg_props,
            session=None,
            region=None,
        )

    @patch.object(FeatureGroupManager, "_update_iceberg_properties")
    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_parent_update_receives_only_standard_params(self, mock_get_client, mock_refresh, mock_update_iceberg):
        """Test that iceberg_properties is not passed to the parent update()."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig, DataCatalogConfig

        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
            data_catalog_config=DataCatalogConfig(
                catalog="AwsDataCatalog", database="db", table_name="tbl"
            ),
            table_format="Iceberg",
        )
        fg.update(
            description="new desc",
            iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "val"}),
        )

        # Verify the SageMaker API call does NOT contain iceberg_properties
        call_args = mock_client.update_feature_group.call_args
        assert "IcebergProperties" not in call_args[1]
        assert "Description" in call_args[1]

    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_no_offline_store(self, mock_get_client, mock_refresh):
        """Test ValueError when iceberg_properties provided without offline_store_config."""
        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = None

        with pytest.raises(ValueError, match="iceberg_properties requires offline_store_config"):
            fg.update(iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "val"}))

    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_offline_store_is_unassigned(self, mock_get_client, mock_refresh):
        """Test ValueError when offline_store_config is Unassigned()."""
        from sagemaker.core.shapes import Unassigned

        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        object.__setattr__(fg, "offline_store_config", Unassigned())

        with pytest.raises(ValueError, match="iceberg_properties requires offline_store_config"):
            fg.update(iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "val"}))

    @patch.object(FeatureGroupManager, "refresh")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_validation_error_when_table_format_not_iceberg(self, mock_get_client, mock_refresh):
        """Test ValueError when table_format is not Iceberg."""
        from sagemaker.core.shapes import OfflineStoreConfig, S3StorageConfig

        mock_client = MagicMock()
        mock_client.update_feature_group.return_value = {}
        mock_get_client.return_value = mock_client

        fg = FeatureGroupManager(feature_group_name="test-fg")
        fg.offline_store_config = OfflineStoreConfig(
            s3_storage_config=S3StorageConfig(s3_uri="s3://bucket/path"),
            table_format="Glue",
        )

        with pytest.raises(ValueError, match="table_format to be 'Iceberg'"):
            fg.update(iceberg_properties=IcebergProperties(properties={"write.target-file-size-bytes": "val"}))


class TestGetWithIcebergProperties:
    """Tests for get() method with include_iceberg_properties flag."""

    @patch.object(FeatureGroupManager, "_get_iceberg_properties")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_no_iceberg_fetch_by_default(self, mock_get_client, mock_get_iceberg):
        """Test that Iceberg properties are not fetched when flag is False (default)."""
        from sagemaker.core.shapes import FeatureDefinition

        mock_client = MagicMock()
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [
                {"FeatureName": "record_id", "FeatureType": "String"},
            ],
            "CreationTime": "2024-01-01T00:00:00Z",
        }
        mock_get_client.return_value = mock_client

        result = FeatureGroupManager.get(feature_group_name="test-fg")

        mock_get_iceberg.assert_not_called()

    @patch.object(FeatureGroupManager, "_get_iceberg_properties")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_iceberg_properties_fetched_when_flag_true(self, mock_get_client, mock_get_iceberg):
        """Test that Iceberg properties are fetched and stored when flag is True."""
        mock_client = MagicMock()
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [
                {"FeatureName": "record_id", "FeatureType": "String"},
            ],
            "CreationTime": "2024-01-01T00:00:00Z",
        }
        mock_get_client.return_value = mock_client

        mock_get_iceberg.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {
                "Parameters": {
                    "write.target-file-size-bytes": "536870912",
                },
            },
            "glue_client": MagicMock(),
        }

        result = FeatureGroupManager.get(
            feature_group_name="test-fg",
            include_iceberg_properties=True,
        )

        mock_get_iceberg.assert_called_once_with(session=None, region=None)
        assert result.iceberg_properties.properties == {
            "write.target-file-size-bytes": "536870912",
        }

    @patch.object(FeatureGroupManager, "_get_iceberg_properties")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_iceberg_properties_empty_parameters(self, mock_get_client, mock_get_iceberg):
        """Test that empty Parameters dict results in empty properties."""
        mock_client = MagicMock()
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [
                {"FeatureName": "record_id", "FeatureType": "String"},
            ],
            "CreationTime": "2024-01-01T00:00:00Z",
        }
        mock_get_client.return_value = mock_client

        mock_get_iceberg.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {},
            "glue_client": MagicMock(),
        }

        result = FeatureGroupManager.get(
            feature_group_name="test-fg",
            include_iceberg_properties=True,
        )

        assert result.iceberg_properties.properties == {}

    @patch.object(FeatureGroupManager, "_get_iceberg_properties")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_passes_session_and_region_to_get_iceberg_properties(self, mock_get_client, mock_get_iceberg):
        """Test that session and region kwargs are forwarded to _get_iceberg_properties."""
        mock_client = MagicMock()
        mock_client.describe_feature_group.return_value = {
            "FeatureGroupName": "test-fg",
            "FeatureGroupArn": "arn:aws:sagemaker:us-west-2:123456789012:feature-group/test-fg",
            "RecordIdentifierFeatureName": "record_id",
            "EventTimeFeatureName": "event_time",
            "FeatureDefinitions": [
                {"FeatureName": "record_id", "FeatureType": "String"},
            ],
            "CreationTime": "2024-01-01T00:00:00Z",
        }
        mock_get_client.return_value = mock_client

        mock_session = MagicMock(spec=Session)
        mock_get_iceberg.return_value = {
            "database_name": "test_db",
            "table_name": "test_table",
            "table_input": {"Parameters": {"write.target-file-size-bytes": "val"}},
            "glue_client": MagicMock(),
        }

        FeatureGroupManager.get(
            feature_group_name="test-fg",
            include_iceberg_properties=True,
            session=mock_session,
            region="us-east-1",
        )

        mock_get_iceberg.assert_called_once_with(session=mock_session, region="us-east-1")
