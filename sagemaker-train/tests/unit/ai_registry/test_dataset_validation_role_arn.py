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
"""Unit tests for the validation_role_arn kwarg on DataSet.create.

Locks in the Phase 4 contract for the ``validation-role-arn-optional`` spec:

* When the caller supplies ``validation_role_arn`` on a v4 or v5 schema, the SDK
  writes it onto the outgoing HubContentDocument as ``"ValidationRoleArn"``.
* When the caller omits it, the field is absent from the document -- even if
  ``role`` is supplied. The SDK does NOT silently reuse the ``role`` kwarg.
* When the caller supplies it with an unsupported schema (v2/v3), the SDK
  raises ``ValueError`` up-front rather than silently dropping the field --
  otherwise a caller's data-validation opt-in would vanish without notice.
"""

import json
import tempfile
import os
import pytest
from unittest.mock import Mock, patch
from sagemaker.ai_registry.dataset import DataSet

# Sample RLVR format dataset (GSM8K style) -- mirrors test_dataset_domain_id.py
SAMPLE_DATASET = {
    "data_source": "openai/gsm8k",
    "prompt": [
        {
            "content": 'Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Let\'s think step by step and output the final answer after "####".',  # noqa: E501
            "role": "user",
        }
    ],
    "ability": "math",
    "reward_model": {"ground_truth": "72", "style": "rule"},
    "extra_info": {
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",  # noqa: E501
        "index": 0,
        "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",  # noqa: E501
        "split": "train",
    },
}


@pytest.fixture
def sample_dataset_file():
    """Create a temporary JSONL file with sample dataset."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        json.dump(SAMPLE_DATASET, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


def _describe_hub_content_response():
    """Standard describe_hub_content mock response used by every test."""
    return {
        "HubContentName": "test-dataset",
        "HubContentArn": "arn:aws:sagemaker:us-west-2:123:hub-content/test",
        "HubContentVersion": "1.0.0",
        "HubContentStatus": "Available",
        "CreationTime": "2024-01-01",
        "LastModifiedTime": "2024-01-01",
        "HubContentDocument": '{"DatasetS3Bucket": "bucket", "DatasetS3Prefix": "prefix"}',
    }


class TestDataSetValidationRoleArn:
    """Test the validation_role_arn kwarg is emitted correctly on the outgoing document."""

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_written_to_document_when_kwarg_supplied(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """When the caller supplies validation_role_arn on v5, the field is written to the document."""
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        document_str = call_args[1]["hub_content_document"]
        parsed = json.loads(document_str)

        assert parsed["ValidationRoleArn"] == "arn:aws:iam::123456789012:role/dv-role"

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_absent_from_document_when_kwarg_omitted_even_if_role_supplied(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """The SDK does not silently reuse the ``role`` kwarg for ``validation_role_arn``.

        When the caller supplies ``role=<some ARN>`` but omits ``validation_role_arn``,
        the outgoing document MUST NOT carry ``ValidationRoleArn``. Absence must remain
        absence -- Hub Service reads absence as an explicit opt-out of data validation.
        A regression like ``content["ValidationRoleArn"] = validation_role_arn or role``
        would fail this test.
        """
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                role="arn:aws:iam::123456789012:role/execution-role",
                # No validation_role_arn kwarg
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        document_str = call_args[1]["hub_content_document"]
        parsed = json.loads(document_str)

        assert "ValidationRoleArn" not in parsed, (
            "SDK must not silently fall back from role to validation_role_arn; got: " + document_str
        )

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_empty_string_is_forwarded_to_document(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """An empty-string ``validation_role_arn`` is forwarded to the document verbatim.

        The SDK uses ``is not None`` (not a truthiness check) so an empty string
        reaches Hub Service, which rejects it via its own ARN syntax validator with
        an ``INVALID_VALIDATION_ROLE_ARN`` error. This test locks that semantic in
        place against a regression to ``if validation_role_arn:`` (truthiness),
        which would silently omit the field for empty strings and let a caller's
        malformed opt-in go undetected until they wondered why DV never ran.
        """
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                validation_role_arn="",
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        document_str = call_args[1]["hub_content_document"]
        parsed = json.loads(document_str)

        assert "ValidationRoleArn" in parsed, (
            "empty string must be forwarded, not omitted -- Hub rejects it via its own"
            " syntax validator; got: " + document_str
        )
        assert (
            parsed["ValidationRoleArn"] == ""
        ), "empty string must be forwarded verbatim; got value: " + repr(
            parsed.get("ValidationRoleArn")
        )

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_on_v3_schema_raises_ValueError(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_domain_id,
        mock_session,
        sample_dataset_file,
    ):
        """v3 schema has no ValidationRoleArn slot; the SDK must fail loudly rather than silently drop.

        A caller who typo'd the schema version and supplied ``validation_role_arn`` would
        otherwise silently lose their data-validation opt-in.
        """
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        with pytest.raises(ValueError, match="only supported on document_schema_version"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="3.0.0",
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        # No I/O should have happened: the raise short-circuits before any file
        # validation, S3 upload, or Hub import.
        mock_air_hub.upload_to_s3.assert_not_called()
        mock_air_hub.import_hub_content.assert_not_called()

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_on_v2_schema_raises_ValueError(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_domain_id,
        mock_session,
        sample_dataset_file,
    ):
        """v2 legacy schema has no ValidationRoleArn slot; same fail-loud contract as v3."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        with pytest.raises(ValueError, match="only supported on document_schema_version"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="2.0.0",
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        # No I/O should have happened: the raise short-circuits before any file
        # validation, S3 upload, or Hub import.
        mock_air_hub.upload_to_s3.assert_not_called()
        mock_air_hub.import_hub_content.assert_not_called()

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_written_on_v4_schema(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """On the v4 schema, validation_role_arn is emitted just like on v5."""
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="4.0.0",
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        document_str = call_args[1]["hub_content_document"]
        parsed = json.loads(document_str)

        assert parsed["ValidationRoleArn"] == "arn:aws:iam::123456789012:role/dv-role"

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_validation_role_arn_written_on_default_schema_version(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """The default ``document_schema_version`` accepts ``validation_role_arn``.

        Callers who don't pass ``document_schema_version`` get whatever
        ``DATASET_DOCUMENT_SCHEMA_VERSION`` resolves to (currently "5.0.0"). If a
        future bump changes the default to a schema that doesn't carry
        ``ValidationRoleArn``, this test will fail loudly rather than silently
        breaking DV opt-ins for the majority of callers who rely on the default.
        """
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                # No document_schema_version -- exercises the DATASET_DOCUMENT_SCHEMA_VERSION default.
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        document_str = call_args[1]["hub_content_document"]
        parsed = json.loads(document_str)

        assert parsed["ValidationRoleArn"] == "arn:aws:iam::123456789012:role/dv-role", (
            "the SDK's default document_schema_version must support validation_role_arn"
            " so callers who rely on the default keep their DV opt-in working"
        )


class TestDataSetValidationRoleArnRequiredForKnownFormat:
    """validation_role_arn is required for a known (non-"Custom") format on v4/v5.

    A known dataset_file_format routes the import through Hub Service's
    data-validation pipeline, which needs an IAM role to read and validate the
    data. "Custom" opts out of format validation, so no role is required there.
    The SDK rejects the missing role up-front rather than letting the import fail
    server-side with a less obvious error.
    """

    @pytest.mark.parametrize("schema_version", ["4.0.0", "5.0.0"])
    @pytest.mark.parametrize(
        "file_format",
        ["Bedrock_Converse", "HF_Prompt_Completion", "HF_Preference", "Verl", "Nova_PPO"],
    )
    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_known_format_without_validation_role_arn_raises_ValueError(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_domain_id,
        mock_session,
        sample_dataset_file,
        file_format,
        schema_version,
    ):
        """A known format with no validation_role_arn on v4/v5 raises up-front."""
        mock_get_domain_id.return_value = None
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        with pytest.raises(ValueError, match="validation_role_arn is required when"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version=schema_version,
                dataset_file_format=file_format,
                # No validation_role_arn kwarg
            )

        # The raise short-circuits before any file validation, S3 upload, or Hub import.
        mock_air_hub.upload_to_s3.assert_not_called()
        mock_air_hub.import_hub_content.assert_not_called()

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_custom_format_without_validation_role_arn_is_allowed(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """ "Custom" opts out of format validation, so no validation_role_arn is required."""
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                dataset_file_format="Custom",
                # No validation_role_arn kwarg -- allowed for Custom
            )

        assert mock_air_hub.import_hub_content.called

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_default_format_without_validation_role_arn_is_allowed(
        self,
        mock_get_role,
        mock_get_session,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """dataset_file_format defaults to "Custom", so omitting it needs no role."""
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                # No dataset_file_format (defaults to Custom), no validation_role_arn
            )

        assert mock_air_hub.import_hub_content.called

    @patch("sagemaker.core.helper.session_helper.Session")
    @patch("sagemaker.ai_registry.air_hub_entity.AIRHub")
    @patch("sagemaker.ai_registry.dataset._get_current_domain_id")
    @patch("sagemaker.ai_registry.dataset._get_default_s3_prefix")
    @patch("sagemaker.ai_registry.dataset._get_default_bucket")
    @patch("sagemaker.ai_registry.dataset.AIRHub")
    @patch("sagemaker.ai_registry.dataset.DataSet._validate_dataset_format")
    @patch("sagemaker.train.defaults.TrainDefaults.get_sagemaker_session")
    @patch("sagemaker.train.defaults.TrainDefaults.get_role")
    def test_known_format_with_validation_role_arn_is_allowed(
        self,
        mock_get_role,
        mock_get_session,
        mock_validate_format,
        mock_air_hub,
        mock_get_bucket,
        mock_get_prefix,
        mock_get_domain_id,
        mock_entity_air_hub,
        mock_session,
        sample_dataset_file,
    ):
        """A known format with validation_role_arn supplied proceeds and emits the field."""
        mock_get_domain_id.return_value = None
        mock_get_bucket.return_value = "test-bucket"
        mock_get_prefix.return_value = "datasets/test-dataset"
        mock_entity_air_hub.get_hub_name.return_value = "test-hub"
        mock_session_instance = Mock()
        mock_session.return_value = mock_session_instance
        mock_get_session.return_value = mock_session_instance
        mock_get_role.return_value = "arn:aws:iam::123456789012:role/test-role"
        mock_validate_format.return_value = None

        mock_air_hub.upload_to_s3 = Mock()
        mock_air_hub.import_hub_content = Mock()
        mock_air_hub.describe_hub_content = Mock(return_value=_describe_hub_content_response())

        with patch("sagemaker.ai_registry.dataset.DataSet.wait"):
            DataSet.create(
                name="test-dataset",
                source=sample_dataset_file,
                document_schema_version="5.0.0",
                dataset_file_format="Verl",
                validation_role_arn="arn:aws:iam::123456789012:role/dv-role",
            )

        assert mock_air_hub.import_hub_content.called
        call_args = mock_air_hub.import_hub_content.call_args
        parsed = json.loads(call_args[1]["hub_content_document"])
        assert parsed["ValidationRoleArn"] == "arn:aws:iam::123456789012:role/dv-role"
        assert parsed["DatasetFileFormat"] == "Verl"
