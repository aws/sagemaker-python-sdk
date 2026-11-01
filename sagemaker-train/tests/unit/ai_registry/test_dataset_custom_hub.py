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

"""Tests that a caller-supplied ``hub_name`` is honored end to end."""

import json
import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest

from sagemaker.ai_registry.air_constants import (
    DATASET_HUB_CONTENT_TYPE,
    RESPONSE_KEY_CREATION_TIME,
    RESPONSE_KEY_HUB_CONTENT_ARN,
    RESPONSE_KEY_HUB_CONTENT_DOCUMENT,
    RESPONSE_KEY_HUB_CONTENT_NAME,
    RESPONSE_KEY_HUB_CONTENT_STATUS,
    RESPONSE_KEY_HUB_CONTENT_VERSION,
    RESPONSE_KEY_LAST_MODIFIED_TIME,
)
from sagemaker.ai_registry.air_hub import AIRHub
from sagemaker.ai_registry.air_utils import _determine_new_version
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.dataset_utils import DataSetHubContentDocument

CUSTOM_HUB = "my-custom-hub"


def _describe_response(status="Available", version="2.0.0"):
    return {
        RESPONSE_KEY_HUB_CONTENT_NAME: "ds",
        RESPONSE_KEY_HUB_CONTENT_ARN: "arn:aws:sagemaker:us-west-2:123456789012:hub-content/ds",
        RESPONSE_KEY_HUB_CONTENT_VERSION: version,
        RESPONSE_KEY_HUB_CONTENT_STATUS: status,
        RESPONSE_KEY_HUB_CONTENT_DOCUMENT: json.dumps(
            {"DatasetS3Bucket": "b", "DatasetS3Prefix": "p"}
        ),
        RESPONSE_KEY_CREATION_TIME: "2024-01-01",
        RESPONSE_KEY_LAST_MODIFIED_TIME: "2024-01-01",
    }


class TestDetermineNewVersionHub:
    @patch("sagemaker.ai_registry.air_utils.AIRHub")
    def test_queries_custom_hub(self, mock_hub):
        mock_hub.describe_hub_content.return_value = {RESPONSE_KEY_HUB_CONTENT_VERSION: "3.0.0"}

        assert _determine_new_version("T", "n", None, hub_name=CUSTOM_HUB) == "4.0.0"
        assert mock_hub.describe_hub_content.call_args.kwargs["hub_name"] == CUSTOM_HUB

    @patch("sagemaker.ai_registry.air_utils.AIRHub")
    def test_default_hub_call_unchanged(self, mock_hub):
        mock_hub.describe_hub_content.return_value = {RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0"}

        assert _determine_new_version("T", "n") == "2.0.0"
        assert "hub_name" not in mock_hub.describe_hub_content.call_args.kwargs


class TestListVersionsHub:
    def test_list_hub_content_versions_uses_custom_hub(self):
        client = MagicMock()
        client.list_hub_content_versions.return_value = {"HubContentSummaries": []}
        session = Mock(sagemaker_client=client)
        AIRHub.hubName = "derived-hub"

        AIRHub.list_hub_content_versions("Dataset", "ds", session=session, hub_name=CUSTOM_HUB)
        assert client.list_hub_content_versions.call_args.kwargs["HubName"] == CUSTOM_HUB

        AIRHub.list_hub_content_versions("Dataset", "ds", session=session)
        assert client.list_hub_content_versions.call_args.kwargs["HubName"] == "derived-hub"


class TestDataSetCustomHub:
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_get_stores_hub_and_refresh_uses_it(self, mock_hub):
        mock_hub.describe_hub_content.return_value = _describe_response()
        mock_hub.get_hub_name.return_value = "derived-hub"
        session = Mock()

        with patch(
            "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session", return_value=session
        ):
            dataset = DataSet.get("ds", hub_name=CUSTOM_HUB)

        assert dataset.hub_name == CUSTOM_HUB

        dataset.refresh()
        assert mock_hub.describe_hub_content.call_args.kwargs["hub_name"] == CUSTOM_HUB

    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_get_versions_uses_custom_hub(self, mock_hub):
        mock_hub.get_hub_name.return_value = "derived-hub"
        mock_hub.list_hub_content_versions.return_value = [
            {RESPONSE_KEY_HUB_CONTENT_VERSION: "1.0.0"}
        ]
        mock_hub.describe_hub_content.return_value = _describe_response()
        dataset = DataSet(
            name="ds",
            arn="arn",
            version="1.0.0",
            source="s3://b/p",
            status="Available",
            hub_name=CUSTOM_HUB,
        )

        versions = dataset.get_versions()

        assert mock_hub.list_hub_content_versions.call_args.kwargs["hub_name"] == CUSTOM_HUB
        assert mock_hub.describe_hub_content.call_args.kwargs["hub_name"] == CUSTOM_HUB
        assert versions[0].hub_name == CUSTOM_HUB

    @patch("sagemaker.ai_registry.dataset._determine_new_version", return_value="2.0.0")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_create_threads_hub_into_version_and_entity(
        self, mock_hub, _validate_format, _domain, mock_new_version
    ):
        mock_hub.describe_hub_content.return_value = _describe_response(status="Importing")
        mock_hub.get_hub_name.return_value = "derived-hub"
        mock_hub.download_from_s3.return_value = None
        session = Mock()

        with (
            patch(
                "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session", return_value=session
            ),
            patch(
                "sagemaker.train.defaults.TrainDefaults.get_role",
                return_value="arn:aws:iam::123456789012:role/R",
            ),
        ):
            dataset = DataSet.create(
                name="ds",
                source="s3://b/p/file.jsonl",
                hub_name=CUSTOM_HUB,
                sagemaker_session=session,
                wait=False,
            )

        # Version is computed against the hub the content is imported into.
        assert mock_new_version.call_args.kwargs["hub_name"] == CUSTOM_HUB
        # The entity remembers its hub so wait()/refresh() poll the right place.
        assert dataset.hub_name == CUSTOM_HUB
        mock_hub.describe_hub_content.reset_mock()
        mock_hub.describe_hub_content.return_value = _describe_response(status="Available")
        dataset.wait(poll=0)
        assert dataset.status == "Available"
        assert mock_hub.describe_hub_content.call_args.kwargs["hub_name"] == CUSTOM_HUB
        assert DATASET_HUB_CONTENT_TYPE == mock_hub.describe_hub_content.call_args.args[0]


class TestToJsonSchemaVersionOrdering:
    def test_two_digit_major_version_is_treated_as_v3_plus(self):
        doc = DataSetHubContentDocument(
            dataset_s3_bucket="b",
            dataset_s3_prefix="p",
            dataset_file_format="Custom",
            dataset_type="BYOD",
        )

        out = json.loads(doc.to_json(schema_version="10.0.0"))

        assert out["DatasetFileFormat"] == "Custom"
        assert "DatasetRoleArn" not in out

    def test_v2_still_uses_legacy_branch(self):
        out = json.loads(DataSetHubContentDocument(dataset_s3_bucket="b").to_json("2.0.0"))
        assert out["DatasetType"] == "CUSTOMER_PROVIDED"


class TestToJsonValidationRoleArn:
    def test_v5_emits_validation_role_arn_when_set(self):
        doc = DataSetHubContentDocument(
            dataset_file_format="Custom",
            dataset_type="BYOD",
            validation_role_arn="arn:aws:iam::123456789012:role/V",
        )
        assert json.loads(doc.to_json("5.0.0"))["ValidationRoleArn"].endswith("role/V")

    def test_v5_omits_validation_role_arn_when_unset(self):
        doc = DataSetHubContentDocument(dataset_file_format="Custom", dataset_type="BYOD")
        assert "ValidationRoleArn" not in json.loads(doc.to_json("5.0.0"))


class TestFormatValidationDefault:
    """An unspecified dataset_file_format must still be validated.

    Regression guard: create() defaults the *stored* format to "Custom", but that
    default must not silently skip format detection. Only an explicit "Custom"
    opts out. The integ test test_create_dataset_with_invalid_format_local covers
    the same contract end to end.
    """

    @patch("sagemaker.ai_registry.dataset._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix", return_value="datasets/ds")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket", return_value="bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_unspecified_format_still_validates(self, mock_hub, _bucket, _prefix, _domain):
        """create() with no format passes None to the detector, so a bad file raises."""
        mock_hub.get_hub_name.return_value = "derived-hub"
        session = Mock()

        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            f.write("not a valid dataset")
            f.flush()
            path = f.name
        try:
            with (
                patch(
                    "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session",
                    return_value=session,
                ),
                patch(
                    "sagemaker.train.defaults.TrainDefaults.get_role",
                    return_value="arn:aws:iam::123456789012:role/R",
                ),
            ):
                with pytest.raises(ValueError, match="Unable to detect format"):
                    DataSet.create(name="ds", source=path, wait=False)
        finally:
            os.unlink(path)

    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id", return_value=None)
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix", return_value="datasets/ds")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket", return_value="bucket")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_file")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    def test_explicit_custom_opts_out_of_validation(
        self, mock_hub, _file, _bucket, _prefix, _domain, mock_validate_format
    ):
        """An explicit Custom format is forwarded as "Custom", which skips detection."""
        mock_hub.describe_hub_content.return_value = _describe_response(status="Importing")
        mock_hub.get_hub_name.return_value = "derived-hub"
        session = Mock()

        with (
            patch(
                "sagemaker.train.defaults.TrainDefaults.get_sagemaker_session", return_value=session
            ),
            patch(
                "sagemaker.train.defaults.TrainDefaults.get_role",
                return_value="arn:aws:iam::123456789012:role/R",
            ),
        ):
            DataSet.create(
                name="ds",
                source="/local/path/file.jsonl",
                dataset_file_format="Custom",
                sagemaker_session=session,
                wait=False,
            )

        mock_validate_format.assert_called_once_with("/local/path/file.jsonl", "Custom")
