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

import logging
import os
import subprocess
import sys
import textwrap
from unittest.mock import patch, MagicMock

from sagemaker.ai_registry.air_constants import AIR_DEFAULT_PAGE_SIZE
from sagemaker.ai_registry.air_hub import AIRHub


class TestAIRHubLazyClients:
    """Importing the module must have no AWS side effects (issue #6542)."""

    def test_import_without_region_has_no_aws_side_effects(self):
        """Importing AIRHub must not require a configured AWS region.

        Reproduces issue #6542: AIRHub used to create boto3 clients as class
        attributes, which ran at import time and raised NoRegionError when no
        region was configured. Clients are now created lazily, so the import
        must succeed in a subprocess with every region source scrubbed.

        Imports sagemaker.mlops / sagemaker.serve too when they are installed
        (the transitive path from the original report), but tolerates their
        absence: the sagemaker-train unit-test environment installs only
        sagemaker-train and sagemaker-core, and the defect lives in air_hub
        itself, which the direct import below exercises.
        """
        script = textwrap.dedent("""
            for _mod in ("sagemaker.mlops", "sagemaker.serve"):
                try:
                    __import__(_mod)
                except ImportError:
                    pass
            from sagemaker.ai_registry.air_hub import AIRHub

            assert AIRHub._sagemaker_client is None, "sagemaker client created at import"
            assert AIRHub._s3_client is None, "s3 client created at import"
            print("OK")
            """)
        env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")}
        # Point AWS config/credentials at a non-existent path so no region leaks in.
        env["AWS_CONFIG_FILE"] = os.devnull
        env["AWS_SHARED_CREDENTIALS_FILE"] = os.devnull

        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )

        assert (
            result.returncode == 0
        ), f"import failed without a region (issue #6542):\n{result.stderr}"
        assert "OK" in result.stdout

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_get_sagemaker_client_created_lazily_and_cached(self, mock_boto3):
        AIRHub._sagemaker_client = None
        sentinel = MagicMock()
        mock_boto3.client.return_value = sentinel

        first = AIRHub._get_sagemaker_client()
        second = AIRHub._get_sagemaker_client()

        assert first is sentinel
        assert second is sentinel
        mock_boto3.client.assert_called_once_with("sagemaker")

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_get_s3_client_created_lazily_and_cached(self, mock_boto3):
        AIRHub._s3_client = None
        sentinel = MagicMock()
        mock_boto3.client.return_value = sentinel

        first = AIRHub._get_s3_client()
        second = AIRHub._get_s3_client()

        assert first is sentinel
        assert second is sentinel
        mock_boto3.client.assert_called_once_with("s3")


class TestAIRHub:
    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_import_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_client.describe_hub.return_value = {"HubName": "test-hub"}

        # Reset the class variable to use our mock
        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.import_hub_content(
            hub_content_type="DataSet",
            hub_content_name="test-dataset",
            document_schema_version="1.0.0",
            hub_content_document='{"test": "document"}',
        )

        assert result["HubContentArn"] == "test-arn"
        mock_client.import_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion="1.0.0",
            DocumentSchemaVersion="1.0.0",
            HubContentDocument='{"test": "document"}',
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_import_hub_content_custom_hub_name(self, mock_boto3):
        # A caller-supplied hub_name is used verbatim in the request and is
        # verified to exist (describe_hub) but never auto-created (no create_hub).
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.import_hub_content.return_value = {"HubContentArn": "custom-arn"}
        mock_client.describe_hub.return_value = {"HubName": "CustomHub"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.import_hub_content(
            hub_content_type="DataSet",
            hub_content_name="test-dataset",
            document_schema_version="1.0.0",
            hub_content_document='{"test": "document"}',
            hub_name="CustomHub",
        )

        assert result["HubContentArn"] == "custom-arn"
        mock_client.describe_hub.assert_called_once_with(HubName="CustomHub")
        mock_client.create_hub.assert_not_called()
        mock_client.import_hub_content.assert_called_once_with(
            HubName="CustomHub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion='1.0.0',
            DocumentSchemaVersion="1.0.0",
            HubContentDocument='{"test": "document"}'
        )

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_import_hub_content_custom_hub_missing_creates(self, mock_boto3, caplog):
        # A caller-supplied hub that does not exist is created on demand (using
        # the supplied name for both HubName and HubDisplayName), then imported into.
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        class _ResourceNotFound(Exception):
            pass
        mock_client.exceptions.ResourceNotFound = _ResourceNotFound
        mock_client.describe_hub.side_effect = _ResourceNotFound("missing")
        mock_client.import_hub_content.return_value = {"HubContentArn": "custom-arn"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        with caplog.at_level(logging.INFO, logger="sagemaker.ai_registry.air_hub"):
            result = AIRHub.import_hub_content(
                hub_content_type="DataSet",
                hub_content_name="test-dataset",
                document_schema_version="1.0.0",
                hub_content_document='{"test": "document"}',
                hub_name="NewHub",
            )

        assert result["HubContentArn"] == "custom-arn"
        mock_client.describe_hub.assert_called_once_with(HubName="NewHub")
        mock_client.create_hub.assert_called_once_with(
            HubName="NewHub",
            HubDisplayName="NewHub",
            HubDescription="AI Registry Hub",
        )
        mock_client.import_hub_content.assert_called_once_with(
            HubName="NewHub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion='1.0.0',
            DocumentSchemaVersion="1.0.0",
            HubContentDocument='{"test": "document"}'
        )
        # Auto-creation of a caller-supplied hub is logged so it is observable.
        assert "NewHub" in caplog.text
        assert "creating it on demand" in caplog.text

    @patch('sagemaker.ai_registry.air_hub.boto3')
    def test_import_hub_content_with_tags(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.import_hub_content.return_value = {"HubContentArn": "test-arn"}
        mock_client.describe_hub.return_value = {"HubName": "test-hub"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.import_hub_content(
            hub_content_type="DataSet",
            hub_content_name="test-dataset",
            document_schema_version="1.0.0",
            hub_content_document='{"test": "document"}',
            tags=[("method", "generated"), ("team", "ml")],
        )

        assert result["HubContentArn"] == "test-arn"
        # Tags flow to both HubContentSearchKeywords and the API Tags field
        mock_client.import_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion='1.0.0',
            DocumentSchemaVersion="1.0.0",
            HubContentDocument='{"test": "document"}',
            HubContentSearchKeywords=["method:generated", "team:ml"],
            Tags=[
                {"Key": "method", "Value": "generated"},
                {"Key": "team", "Value": "ml"},
            ],
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_list_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.list_hub_contents.return_value = {"HubContentSummaries": []}
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.list_hub_content("DataSet")

        assert "items" in result
        assert "next_token" in result
        mock_client.list_hub_contents.assert_called_once_with(
            HubName="test-hub", HubContentType="DataSet", MaxResults=AIR_DEFAULT_PAGE_SIZE
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_describe_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.describe_hub_content("DataSet", "test-dataset")

        assert result["HubContentName"] == "test"
        mock_client.describe_hub_content.assert_called_once_with(
            HubName="test-hub", HubContentType="DataSet", HubContentName="test-dataset"
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_describe_hub_content_with_version(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.describe_hub_content.return_value = {"HubContentName": "test"}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        AIRHub.describe_hub_content("DataSet", "test-dataset", "1.0.0")

        mock_client.describe_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion="1.0.0",
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_list_hub_content_versions(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.list_hub_content_versions.return_value = {"HubContentSummaries": []}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        result = AIRHub.list_hub_content_versions("DataSet", "test-dataset")

        assert isinstance(result, list)
        mock_client.list_hub_content_versions.assert_called_once_with(
            HubName="test-hub", HubContentType="DataSet", HubContentName="test-dataset"
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_delete_hub_content(self, mock_boto3):
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client
        mock_client.delete_hub_content.return_value = {}

        AIRHub._sagemaker_client = mock_client
        AIRHub.hubName = "test-hub"

        AIRHub.delete_hub_content("DataSet", "test-dataset", "1.0.0")

        mock_client.delete_hub_content.assert_called_once_with(
            HubName="test-hub",
            HubContentType="DataSet",
            HubContentName="test-dataset",
            HubContentVersion="1.0.0",
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_upload_to_s3(self, mock_boto3):
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        AIRHub._s3_client = mock_s3_client

        result = AIRHub.upload_to_s3("test-bucket", "test/key", "/local/path")

        assert result == "s3://test-bucket/test/key"
        # Non-default bucket: no ExpectedBucketOwner enforced (explicit buckets untouched).
        mock_s3_client.upload_file.assert_called_once_with(
            "/local/path", "test-bucket", "test/key", ExtraArgs=None
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_download_from_s3(self, mock_boto3):
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client

        AIRHub._s3_client = mock_s3_client

        AIRHub.download_from_s3("s3://test-bucket/test/key", "/local/path")

        # Non-default bucket: no ExpectedBucketOwner enforced.
        mock_s3_client.download_file.assert_called_once_with(
            "test-bucket", "test/key", "/local/path", ExtraArgs=None
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_upload_to_s3_default_bucket_enforces_owner(self, mock_boto3):
        """Uploading to the SDK-derived default bucket passes ExpectedBucketOwner."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "111122223333"}
        mock_boto3.client.return_value = mock_sts
        mock_boto3.session.Session.return_value.region_name = "us-west-2"
        mock_s3_client = MagicMock()
        AIRHub._s3_client = mock_s3_client

        default_bucket = "sagemaker-us-west-2-111122223333"
        AIRHub.upload_to_s3(default_bucket, "test/key", "/local/path")

        mock_s3_client.upload_file.assert_called_once_with(
            "/local/path",
            default_bucket,
            "test/key",
            ExtraArgs={"ExpectedBucketOwner": "111122223333"},
        )

    @patch("sagemaker.ai_registry.air_hub.boto3")
    def test_download_from_s3_default_bucket_enforces_owner(self, mock_boto3):
        """Downloading from the SDK-derived default bucket passes ExpectedBucketOwner."""
        mock_sts = MagicMock()
        mock_sts.get_caller_identity.return_value = {"Account": "111122223333"}
        mock_boto3.client.return_value = mock_sts
        mock_boto3.session.Session.return_value.region_name = "us-west-2"
        mock_s3_client = MagicMock()
        AIRHub._s3_client = mock_s3_client

        default_bucket = "sagemaker-us-west-2-111122223333"
        AIRHub.download_from_s3(f"s3://{default_bucket}/test/key", "/local/path")

        mock_s3_client.download_file.assert_called_once_with(
            default_bucket,
            "test/key",
            "/local/path",
            ExtraArgs={"ExpectedBucketOwner": "111122223333"},
        )

    def test_generate_hub_names_no_padding(self):
        """Test that generated hub names don't contain = padding characters."""
        # Clear any existing hubName to ensure clean test
        if hasattr(AIRHub, "hubName"):
            delattr(AIRHub, "hubName")
        if hasattr(AIRHub, "hubDisplayName"):
            delattr(AIRHub, "hubDisplayName")

        # Generate hub names
        AIRHub._generate_hub_names("us-west-2", "123456789012")

        # Verify hubName doesn't contain = padding
        assert (
            "=" not in AIRHub.hubName
        ), f"Hub name should not contain '=' padding: {AIRHub.hubName}"

        # Verify hubDisplayName is correctly formatted
        assert AIRHub.hubDisplayName == "AiRegistry-us-west-2-123456789012"

        # Verify hubName is not empty and is a valid base32 string
        assert len(AIRHub.hubName) > 0
        assert (
            AIRHub.hubName.isalnum()
        ), f"Hub name should only contain alphanumeric characters: {AIRHub.hubName}"
