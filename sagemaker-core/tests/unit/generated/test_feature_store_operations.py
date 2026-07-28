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
"""Unit tests for FeatureGroup batch_write_record and list_records methods."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.core.resources import FeatureGroup
from sagemaker.core.shapes.shapes import (
    BatchWriteRecordEntry,
    BatchWriteRecordResponse,
    FeatureValue,
    ListRecordsResponse,
    TtlDuration,
)


@pytest.fixture
def mock_feature_group():
    """Create a FeatureGroup instance with mocked internals."""
    fg = FeatureGroup.model_construct(
        feature_group_name="test-feature-group",
        next_token=None,
    )
    return fg


class TestBatchWriteRecord:
    """Tests for FeatureGroup.batch_write_record method."""

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_batch_write_record_success(self, mock_get_client, mock_transform, mock_feature_group):
        """Test that batch_write_record calls the client with correct arguments."""
        mock_client = MagicMock()
        mock_client.batch_write_record.return_value = {
            "Errors": [],
            "UnprocessedEntries": [],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "errors": [],
            "unprocessed_entries": [],
        }

        entries = [
            BatchWriteRecordEntry(
                feature_group_name="test-feature-group",
                record=[FeatureValue(feature_name="feature1", value_as_string="value1")],
            )
        ]

        mock_feature_group.batch_write_record(entries=entries)

        mock_get_client.assert_called_once_with(
            session=None, region_name=None, service_name="sagemaker-featurestore-runtime"
        )
        mock_client.batch_write_record.assert_called_once()
        call_kwargs = mock_client.batch_write_record.call_args[1]
        assert "Entries" in call_kwargs
        mock_transform.assert_called_once()

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_batch_write_record_with_ttl_duration(
        self, mock_get_client, mock_transform, mock_feature_group
    ):
        """Test batch_write_record with optional ttl_duration parameter."""
        mock_client = MagicMock()
        mock_client.batch_write_record.return_value = {
            "Errors": [],
            "UnprocessedEntries": [],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "errors": [],
            "unprocessed_entries": [],
        }

        entries = [
            BatchWriteRecordEntry(
                feature_group_name="test-feature-group",
                record=[FeatureValue(feature_name="feature1", value_as_string="value1")],
            )
        ]
        ttl = TtlDuration(unit="Hours", value=24)

        mock_feature_group.batch_write_record(entries=entries, ttl_duration=ttl)

        mock_get_client.assert_called_once_with(
            session=None, region_name=None, service_name="sagemaker-featurestore-runtime"
        )
        mock_client.batch_write_record.assert_called_once()
        call_kwargs = mock_client.batch_write_record.call_args[1]
        assert "Entries" in call_kwargs
        assert "TtlDuration" in call_kwargs

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_batch_write_record_returns_response(
        self, mock_get_client, mock_transform, mock_feature_group
    ):
        """Test that batch_write_record returns a BatchWriteRecordResponse."""
        mock_client = MagicMock()
        mock_client.batch_write_record.return_value = {
            "Errors": [],
            "UnprocessedEntries": [],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "errors": [],
            "unprocessed_entries": [],
        }

        entries = [
            BatchWriteRecordEntry(
                feature_group_name="test-feature-group",
                record=[FeatureValue(feature_name="feature1", value_as_string="value1")],
            )
        ]

        result = mock_feature_group.batch_write_record(entries=entries)

        assert isinstance(result, BatchWriteRecordResponse)
        assert result.errors == []
        assert result.unprocessed_entries == []

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_batch_write_record_returns_response_with_nested_data(
        self, mock_get_client, mock_transform, mock_feature_group
    ):
        """Test that batch_write_record correctly deserializes nested response data."""
        mock_client = MagicMock()
        mock_client.batch_write_record.return_value = {
            "Errors": [
                {
                    "Entry": {
                        "FeatureGroupName": "test-feature-group",
                        "Record": [{"FeatureName": "f1", "ValueAsString": "v1"}],
                    },
                    "ErrorCode": "ValidationError",
                    "ErrorMessage": "Invalid feature value",
                }
            ],
            "UnprocessedEntries": [
                {
                    "FeatureGroupName": "test-feature-group",
                    "Record": [{"FeatureName": "f2", "ValueAsString": "v2"}],
                }
            ],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "errors": [
                {
                    "entry": {
                        "feature_group_name": "test-feature-group",
                        "record": [{"feature_name": "f1", "value_as_string": "v1"}],
                    },
                    "error_code": "ValidationError",
                    "error_message": "Invalid feature value",
                }
            ],
            "unprocessed_entries": [
                {
                    "feature_group_name": "test-feature-group",
                    "record": [{"feature_name": "f2", "value_as_string": "v2"}],
                }
            ],
        }

        entries = [
            BatchWriteRecordEntry(
                feature_group_name="test-feature-group",
                record=[FeatureValue(feature_name="feature1", value_as_string="value1")],
            )
        ]

        result = mock_feature_group.batch_write_record(entries=entries)

        assert isinstance(result, BatchWriteRecordResponse)
        assert len(result.errors) == 1
        assert len(result.unprocessed_entries) == 1


class TestListRecords:
    """Tests for FeatureGroup.list_records method."""

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_list_records_success(self, mock_get_client, mock_transform, mock_feature_group):
        """Test that list_records calls the client with FeatureGroupName."""
        mock_client = MagicMock()
        mock_client.list_records.return_value = {
            "RecordIdentifiers": ["id1", "id2"],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "record_identifiers": ["id1", "id2"],
        }

        mock_feature_group.list_records()

        mock_get_client.assert_called_once_with(
            session=None, region_name=None, service_name="sagemaker-featurestore-runtime"
        )
        mock_client.list_records.assert_called_once()
        call_kwargs = mock_client.list_records.call_args[1]
        assert call_kwargs["FeatureGroupName"] == "test-feature-group"
        mock_transform.assert_called_once()

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_list_records_with_parameters(
        self, mock_get_client, mock_transform, mock_feature_group
    ):
        """Test list_records with max_results and include_soft_deleted_records."""
        mock_client = MagicMock()
        mock_client.list_records.return_value = {
            "RecordIdentifiers": ["id1"],
            "NextToken": "token123",
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "record_identifiers": ["id1"],
            "next_token": "token123",
        }

        mock_feature_group.list_records(
            max_results=10, include_soft_deleted_records=True
        )

        mock_get_client.assert_called_once_with(
            session=None, region_name=None, service_name="sagemaker-featurestore-runtime"
        )
        mock_client.list_records.assert_called_once()
        call_kwargs = mock_client.list_records.call_args[1]
        assert call_kwargs["FeatureGroupName"] == "test-feature-group"
        assert call_kwargs["MaxResults"] == 10
        assert call_kwargs["IncludeSoftDeletedRecords"] is True

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_list_records_returns_response(
        self, mock_get_client, mock_transform, mock_feature_group
    ):
        """Test that list_records returns a ListRecordsResponse."""
        mock_client = MagicMock()
        mock_client.list_records.return_value = {
            "RecordIdentifiers": ["id1", "id2", "id3"],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "record_identifiers": ["id1", "id2", "id3"],
        }

        result = mock_feature_group.list_records()

        assert isinstance(result, ListRecordsResponse)
        assert result.record_identifiers == ["id1", "id2", "id3"]

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_list_records_does_not_use_self_next_token(
        self, mock_get_client, mock_transform
    ):
        """Test that list_records does NOT pass self.next_token (from DescribeFeatureGroup) to ListRecords."""
        fg = FeatureGroup.model_construct(
            feature_group_name="test-feature-group",
            next_token="describe-pagination-token",
        )
        mock_client = MagicMock()
        mock_client.list_records.return_value = {
            "RecordIdentifiers": ["id1"],
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "record_identifiers": ["id1"],
        }

        fg.list_records()

        call_kwargs = mock_client.list_records.call_args[1]
        # self.next_token should NOT be passed to ListRecords
        assert "NextToken" not in call_kwargs

    @patch("sagemaker.core.resources.transform")
    @patch("sagemaker.core.resources.Base.get_sagemaker_client")
    def test_list_records_accepts_next_token_parameter(
        self, mock_get_client, mock_transform
    ):
        """Test that list_records accepts next_token as a pagination parameter."""
        fg = FeatureGroup.model_construct(
            feature_group_name="test-feature-group",
            next_token="describe-pagination-token",
        )
        mock_client = MagicMock()
        mock_client.list_records.return_value = {
            "RecordIdentifiers": ["id1"],
            "NextToken": "next-page-token",
        }
        mock_get_client.return_value = mock_client
        mock_transform.return_value = {
            "record_identifiers": ["id1"],
            "next_token": "next-page-token",
        }

        fg.list_records(next_token="list-records-page-2-token")

        call_kwargs = mock_client.list_records.call_args[1]
        # The explicitly passed next_token should be used, not self.next_token
        assert call_kwargs["NextToken"] == "list-records-page-2-token"
