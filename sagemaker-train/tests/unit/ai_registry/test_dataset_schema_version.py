# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

"""Tests for document_schema_version parameter behavior in DataSet.create()."""

import json
import pytest
from unittest.mock import Mock, patch

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.air_constants import (
    DATASET_DOCUMENT_SCHEMA_VERSION,
    DATASET_SUPPORTED_SCHEMA_VERSIONS,
    RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_HUB_CONTENT_VERSION,
)


@pytest.fixture(autouse=True)
def _isolate_role_and_hub_name():
    """Keep DataSet.create offline in these tests.

    TrainDefaults.get_role now validates the role via resolve_and_validate_role (STS
    call), and DataSet.__init__ resolves the hub name via AIRHub.get_hub_name (STS
    call). Patch both so the tests don't depend on credentials or test ordering.
    """
    with (
        patch(
            "sagemaker.train.defaults.resolve_and_validate_role",
            return_value="arn:aws:iam::123456789012:role/SageMakerRole",
        ),
        patch(
            "sagemaker.ai_registry.air_hub_entity.AIRHub.get_hub_name",
            return_value="test-hub",
        ),
    ):
        yield


class TestDataSetSchemaVersionConstants:
    """Tests for schema version constant values."""

    def test_default_schema_version_constant_is_5_0_0(self):
        """Verify the default document schema version is 5.0.0."""
        assert DATASET_DOCUMENT_SCHEMA_VERSION == "5.0.0"

    def test_supported_schema_versions_list(self):
        """Verify the supported schema versions include all expected versions."""
        assert DATASET_SUPPORTED_SCHEMA_VERSIONS == ["2.0.0", "3.0.0", "4.0.0", "5.0.0"]


class TestDataSetSchemaVersionDefault:
    """Tests for default schema version behavior when not explicitly provided."""

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_create_without_schema_version_uses_5_0_0(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """When document_schema_version is not provided, it defaults to 5.0.0."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        assert call_kwargs["document_schema_version"] == "5.0.0"


class TestDataSetSchemaVersionOverride:
    """Tests for explicitly providing a valid document_schema_version."""

    @pytest.mark.parametrize("version", ["2.0.0", "3.0.0", "4.0.0", "5.0.0"])
    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_create_with_valid_schema_version_passes_it_through(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
        version,
    ):
        """When a valid document_schema_version is provided, it is passed to import_hub_content."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                document_schema_version=version,
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        assert call_kwargs["document_schema_version"] == version


class TestDataSetSchemaVersionValidation:
    """Tests for invalid document_schema_version values raising errors."""

    @pytest.mark.parametrize("invalid_version", ["1.0.0", "6.0.0", "", "abc", "5.0"])
    def test_create_with_invalid_schema_version_raises_value_error(self, invalid_version):
        """When an unsupported document_schema_version is provided, a ValueError is raised."""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )

        with pytest.raises(ValueError):
            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                document_schema_version=invalid_version,
                sagemaker_session=mock_session,
                wait=False,
            )

    def test_invalid_schema_version_error_message_lists_supported_versions(self):
        """The ValueError message includes the list of supported versions."""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )

        with pytest.raises(
            ValueError, match=r"Supported versions:.*\['2.0.0', '3.0.0', '4.0.0', '5.0.0'\]"
        ):
            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                document_schema_version="1.0.0",
                sagemaker_session=mock_session,
                wait=False,
            )


class TestDataSetFileFormat:
    """Tests for the dataset_file_format parameter behavior in DataSet.create()."""

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_create_without_file_format_defaults_to_custom(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """When dataset_file_format is not provided, it defaults to Custom in the document."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert doc["DatasetFileFormat"] == "Custom"

    @pytest.mark.parametrize(
        "file_format", ["Bedrock_Converse", "HF_Prompt_Completion", "HF_Preference", "Verl"]
    )
    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_create_with_valid_file_format_passes_it_through(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
        file_format,
    ):
        """When a valid dataset_file_format is provided, it appears in the document."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                dataset_file_format=file_format,
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert doc["DatasetFileFormat"] == file_format

    @pytest.mark.parametrize("invalid_format", ["invalid", "jsonl", "CSV", ""])
    def test_create_with_invalid_file_format_raises_value_error(self, invalid_format):
        """When an unsupported dataset_file_format is provided, a ValueError is raised."""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )

        with pytest.raises(ValueError):
            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                dataset_file_format=invalid_format,
                sagemaker_session=mock_session,
                wait=False,
            )

    def test_invalid_file_format_error_message_lists_valid_formats(self):
        """The ValueError message includes the list of valid format names."""
        mock_session = Mock()
        mock_session.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )

        with pytest.raises(ValueError, match=r"Valid formats:.*Bedrock_Converse"):
            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                dataset_file_format="invalid",
                sagemaker_session=mock_session,
                wait=False,
            )


class TestDataSetV5DocumentStructure:
    """Tests for the v5 hub content document body structure."""

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v5_document_contains_required_fields(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """With default schema version (5.0.0), the document contains all required fields."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert "DatasetType" in doc
        assert "DatasetFileFormat" in doc
        assert "DatasetS3Bucket" in doc
        assert "DatasetS3Prefix" in doc

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v5_document_does_not_contain_legacy_fields(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """With default schema version (5.0.0), the document does not contain legacy fields."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert "ConversationId" not in doc
        assert "ConversationCheckpointId" not in doc
        assert "DatasetContextS3Uri" not in doc
        assert "DatasetRoleArn" not in doc
        assert "Dependencies" not in doc

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v2_document_contains_legacy_fields(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """With schema version 2.0.0, the document contains legacy fields."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                document_schema_version="2.0.0",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert "ConversationId" in doc
        assert "Dependencies" in doc
        assert "DatasetRoleArn" in doc

    @patch("boto3.client")
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v5_document_dataset_type_is_byod(
        self,
        mock_air_hub,
        mock_validate_format,
        mock_validate_file,
        mock_get_domain_id,
        mock_session,
        mock_boto_client,
    ):
        """With default schema version, DatasetType is BYOD in the document."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_session.return_value = mock_session_instance

        mock_sts_client = Mock()
        mock_sts_client.get_caller_identity.return_value = {"Account": "123456789012"}
        mock_boto_client.return_value = mock_sts_client

        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        mock_air_hub.download_from_s3 = Mock()
        mock_validate_format.return_value = None
        mock_validate_file.return_value = None

        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"

            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/data.jsonl",
                sagemaker_session=mock_session_instance,
                wait=False,
            )

        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        doc = json.loads(call_kwargs["hub_content_document"])
        assert doc["DatasetType"] == "BYOD"


class TestDataSetV5ContentMetadata:
    """Tests that Feature Store ContentMetadata is carried in the v5.0.0 document.

    The v5.0.0 dataset schema defines ``ContentMetadata`` (SourceFeatureGroups required;
    ExtractionMethod and AthenaQueryExecutionId optional), so DataSet.create must emit it
    when ``content_metadata`` is passed and omit it otherwise.
    """

    CONTENT_METADATA = {
        "SourceFeatureGroups": [
            "arn:aws:sagemaker:us-west-2:123456789012:feature-group/fg1",
        ],
        "ExtractionMethod": "FeatureStoreDatasetBuilder",
        "AthenaQueryExecutionId": "query-xyz",
    }

    def _create_and_get_document(
        self, mock_air_hub, content_metadata, document_schema_version=None
    ):
        mock_session_instance = Mock()
        mock_session_instance.get_caller_identity_arn.return_value = (
            "arn:aws:iam::123456789012:role/SageMakerRole"
        )
        mock_air_hub.get_hub_name.return_value = "test-hub"
        mock_air_hub.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_air_hub.describe_hub_content.return_value = {
            RESPONSE_KEY_HUB_CONTENT_ARN: "test-arn",
            RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0",
            "CreationTime": "2024-01-01",
            "LastModifiedTime": "2024-01-01",
        }
        with (
            patch("tempfile.NamedTemporaryFile") as mock_temp,
            patch("os.path.exists", return_value=False),
            patch("os.remove"),
            patch("sagemaker.ai_registry.dataset.DataSet.wait"),
        ):
            mock_temp.return_value.__enter__.return_value.name = "/tmp/test.jsonl"
            kwargs = {}
            if document_schema_version is not None:
                kwargs["document_schema_version"] = document_schema_version
            DataSet.create(
                name="test-dataset",
                source="s3://test-bucket/output/result.csv",
                content_metadata=content_metadata,
                sagemaker_session=mock_session_instance,
                wait=False,
                **kwargs,
            )
        mock_air_hub.import_hub_content.assert_called_once()
        call_kwargs = mock_air_hub.import_hub_content.call_args[1]
        return call_kwargs["document_schema_version"], json.loads(
            call_kwargs["hub_content_document"]
        )

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v5_document_includes_content_metadata_when_provided(
        self, mock_air_hub, mock_validate_format, mock_validate_file, mock_domain
    ):
        version, doc = self._create_and_get_document(mock_air_hub, dict(self.CONTENT_METADATA))
        assert version == "5.0.0"
        assert doc["ContentMetadata"] == self.CONTENT_METADATA
        # Still a v5 document: required field present, legacy fields absent.
        assert doc["DatasetFileFormat"] == "Custom"
        assert doc["DatasetType"] == "BYOD"
        for legacy in ("Dependencies", "ConversationId", "DatasetRoleArn"):
            assert legacy not in doc

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v5_document_omits_content_metadata_when_not_provided(
        self, mock_air_hub, mock_validate_format, mock_validate_file, mock_domain
    ):
        mock_air_hub.download_from_s3 = Mock()
        version, doc = self._create_and_get_document(mock_air_hub, None)
        assert version == "5.0.0"
        assert "ContentMetadata" not in doc

    @patch("sagemaker.train.common_utils.finetune_utils._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_v2_document_keeps_content_metadata(
        self, mock_air_hub, mock_validate_format, mock_validate_file, mock_domain
    ):
        version, doc = self._create_and_get_document(
            mock_air_hub, dict(self.CONTENT_METADATA), document_schema_version="2.0.0"
        )
        assert version == "2.0.0"
        assert doc["ContentMetadata"] == self.CONTENT_METADATA
        assert doc["Dependencies"] == []
