# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
"""Unit tests for list_records function."""
import pytest
from unittest.mock import Mock, patch


class TestListRecords:
    """Tests for list_records function."""

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_list_records_single_page(self, mock_fg_class):
        from sagemaker.mlops.feature_store.feature_utils import list_records

        mock_fg = Mock()
        mock_fg_class.get.return_value = mock_fg
        mock_response = Mock()
        mock_response.record_identifiers = ["id-1", "id-2", "id-3"]
        mock_response.next_token = None
        mock_fg.list_records.return_value = mock_response

        result = list_records("test-fg", region="us-west-2")

        assert result.record_identifiers == ["id-1", "id-2", "id-3"]
        assert result.next_token is None

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_list_records_with_all_params(self, mock_fg_class):
        from sagemaker.mlops.feature_store.feature_utils import list_records

        mock_fg = Mock()
        mock_fg_class.get.return_value = mock_fg
        mock_response = Mock()
        mock_response.record_identifiers = ["id-1"]
        mock_response.next_token = "token-abc"
        mock_fg.list_records.return_value = mock_response

        list_records(
            "test-fg", max_results=1, next_token="prev-token",
            include_soft_deleted_records=True, region="us-west-2"
        )

        mock_fg.list_records.assert_called_once_with(
            max_results=1, next_token="prev-token",
            include_soft_deleted_records=True, region="us-west-2"
        )

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_list_records_no_optional_params(self, mock_fg_class):
        from sagemaker.mlops.feature_store.feature_utils import list_records

        mock_fg = Mock()
        mock_fg_class.get.return_value = mock_fg
        mock_response = Mock()
        mock_response.record_identifiers = []
        mock_response.next_token = None
        mock_fg.list_records.return_value = mock_response

        list_records("test-fg", region="us-east-1")

        mock_fg.list_records.assert_called_once_with(region="us-east-1")

    @patch("sagemaker.mlops.feature_store.feature_utils.CoreFeatureGroup")
    def test_list_records_does_not_pass_none_params(self, mock_fg_class):
        """None values for optional params should not be passed to the API."""
        from sagemaker.mlops.feature_store.feature_utils import list_records

        mock_fg = Mock()
        mock_fg_class.get.return_value = mock_fg
        mock_response = Mock()
        mock_response.record_identifiers = []
        mock_response.next_token = None
        mock_fg.list_records.return_value = mock_response

        list_records("test-fg", max_results=None, next_token=None, region="us-west-2")

        # max_results=None and next_token=None should NOT be in the kwargs
        mock_fg.list_records.assert_called_once_with(region="us-west-2")
