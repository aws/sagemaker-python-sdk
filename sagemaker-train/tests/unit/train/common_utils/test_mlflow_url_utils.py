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
"""Unit tests for mlflow_url_utils module."""

import pytest
from unittest.mock import patch, MagicMock

from sagemaker.train.common_utils.mlflow_url_utils import (
    _build_mlflow_deep_link,
    _build_mlflow_deep_link_by_name,
    _resolve_experiment_id,
    get_presigned_mlflow_experiment_url,
)


class TestBuildMlflowDeepLink:
    """Tests for _build_mlflow_deep_link."""

    def test_with_experiment_id_only(self):
        url = "https://app-ABC.mlflow.sagemaker.us-west-2.app.aws/auth?authToken=token123"
        result = _build_mlflow_deep_link(url, "25")
        assert "authToken=token123" in result
        assert "deepLink=/#/experiments/25" in result
        assert result.startswith("https://app-ABC.mlflow.sagemaker.us-west-2.app.aws/auth?")

    def test_with_experiment_id_and_run_id(self):
        url = "https://app-ABC.mlflow.sagemaker.us-west-2.app.aws/auth?authToken=token123"
        result = _build_mlflow_deep_link(url, "25", "run-abc")
        assert "deepLink=/#/experiments/25/runs/run-abc" in result

    def test_empty_url(self):
        assert _build_mlflow_deep_link("", "25") == ""

    def test_extracts_token_from_long_jwt(self):
        url = "https://app.mlflow.aws/auth?authToken=eyJhbGciOiJIUzI1NiJ9.payload.sig"
        result = _build_mlflow_deep_link(url, "5")
        assert "authToken=eyJhbGciOiJIUzI1NiJ9.payload.sig" in result
        assert "deepLink=/#/experiments/5" in result


class TestResolveExperimentId:
    """Tests for _resolve_experiment_id."""

    @patch("requests.Session")
    def test_successful_resolution(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = [
            MagicMock(status_code=200),  # auth call
            MagicMock(
                status_code=200,
                json=lambda: {"experiment": {"experiment_id": "23"}},
            ),
        ]

        result = _resolve_experiment_id(
            "https://app.mlflow.aws/auth?authToken=tok",
            "mtrl-eval-openai-reasoning-gpt-oss-20b",
        )
        assert result == "23"

    @patch("requests.Session")
    def test_api_returns_404(self, mock_session_cls):
        mock_session = MagicMock()
        mock_session_cls.return_value = mock_session
        mock_session.get.side_effect = [
            MagicMock(status_code=200),
            MagicMock(status_code=404),
        ]

        result = _resolve_experiment_id(
            "https://app.mlflow.aws/auth?authToken=tok", "nonexistent"
        )
        assert result is None

    @patch("requests.Session")
    def test_connection_error(self, mock_session_cls):
        mock_session_cls.side_effect = Exception("network error")
        result = _resolve_experiment_id(
            "https://app.mlflow.aws/auth?authToken=tok", "exp"
        )
        assert result is None


class TestBuildMlflowDeepLinkByName:
    """Tests for _build_mlflow_deep_link_by_name."""

    @patch(
        "sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id"
    )
    def test_with_resolved_id(self, mock_resolve):
        mock_resolve.return_value = "23"
        url = "https://app.mlflow.aws/auth?authToken=tok123"
        result = _build_mlflow_deep_link_by_name(url, "mtrl-eval-model")
        assert "deepLink=/#/experiments/23" in result
        assert "authToken=tok123" in result

    @patch(
        "sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id"
    )
    def test_fallback_to_search_filter(self, mock_resolve):
        mock_resolve.return_value = None
        url = "https://app.mlflow.aws/auth?authToken=tok123"
        result = _build_mlflow_deep_link_by_name(url, "mtrl-eval-model")
        assert "deepLink=/#/experiments?searchFilter=mtrl-eval-model" in result

    def test_empty_url(self):
        assert _build_mlflow_deep_link_by_name("", "exp") == ""


class TestGetPresignedMlflowExperimentUrl:
    """Tests for get_presigned_mlflow_experiment_url."""

    @patch(
        "sagemaker.train.common_utils.mlflow_url_utils._resolve_experiment_id"
    )
    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_with_experiment_name(self, mock_sm_class, mock_resolve):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": "https://app.mlflow.aws/auth?authToken=abc"
        }
        mock_resolve.return_value = "42"

        result = get_presigned_mlflow_experiment_url(
            "arn:aws:sagemaker:us-west-2:123:mlflow-app/app-XYZ",
            "my-experiment",
        )
        assert "deepLink=/#/experiments/42" in result

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_without_experiment_name(self, mock_sm_class):
        mock_client = MagicMock()
        mock_sm_class.return_value.sagemaker_client = mock_client
        mock_client.create_presigned_mlflow_app_url.return_value = {
            "AuthorizedUrl": "https://app.mlflow.aws/auth?authToken=abc"
        }

        result = get_presigned_mlflow_experiment_url(
            "arn:aws:sagemaker:us-west-2:123:mlflow-app/app-XYZ"
        )
        assert result == "https://app.mlflow.aws/auth?authToken=abc"

    @patch("sagemaker.core.utils.utils.SageMakerClient")
    def test_presigned_url_failure(self, mock_sm_class):
        mock_sm_class.side_effect = Exception("service error")
        result = get_presigned_mlflow_experiment_url(
            "arn:aws:sagemaker:us-west-2:123:mlflow-app/app-XYZ"
        )
        assert result is None
