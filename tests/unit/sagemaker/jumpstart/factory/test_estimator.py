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
from __future__ import absolute_import
import pytest
from unittest.mock import patch
from sagemaker.jumpstart.constants import JUMPSTART_MODEL_HUB_NAME
from sagemaker.jumpstart.factory.estimator import (
    _add_model_uri_to_kwargs,
    get_model_info_default_kwargs,
)
from sagemaker.jumpstart.types import JumpStartEstimatorInitKwargs
from sagemaker.jumpstart.enums import JumpStartScriptScope


class TestAddModelUriToKwargs:
    @pytest.fixture
    def mock_kwargs(self):
        return JumpStartEstimatorInitKwargs(
            model_id="test-model",
            model_version="1.0.0",
            instance_type="ml.m5.large",
            model_uri=None,
        )

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=True,
    )
    @patch("sagemaker.jumpstart.factory.estimator.model_uris.retrieve")
    def test_add_model_uri_to_kwargs_default_uri(
        self, mock_retrieve, mock_supports_training, mock_kwargs
    ):
        """Test adding default model URI when none is provided."""
        default_uri = "s3://jumpstart-models/training/test-model/1.0.0"
        mock_retrieve.return_value = default_uri

        result = _add_model_uri_to_kwargs(mock_kwargs)

        mock_supports_training.assert_called_once()
        mock_retrieve.assert_called_once_with(
            model_scope=JumpStartScriptScope.TRAINING,
            instance_type=mock_kwargs.instance_type,
            **get_model_info_default_kwargs(mock_kwargs),
        )
        assert result.model_uri == default_uri

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=True,
    )
    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_incremental_training",
        return_value=True,
    )
    @patch("sagemaker.jumpstart.factory.estimator.model_uris.retrieve")
    def test_add_model_uri_to_kwargs_custom_uri_with_incremental(
        self, mock_retrieve, mock_supports_incremental, mock_supports_training, mock_kwargs
    ):
        """Test using custom model URI with incremental training support."""
        default_uri = "s3://jumpstart-models/training/test-model/1.0.0"
        custom_uri = "s3://custom-bucket/my-model"
        mock_retrieve.return_value = default_uri
        mock_kwargs.model_uri = custom_uri

        result = _add_model_uri_to_kwargs(mock_kwargs)

        mock_supports_training.assert_called_once()
        mock_supports_incremental.assert_called_once()
        assert result.model_uri == custom_uri

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=True,
    )
    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_incremental_training",
        return_value=False,
    )
    @patch("sagemaker.jumpstart.factory.estimator.model_uris.retrieve")
    @patch("sagemaker.jumpstart.factory.estimator.JUMPSTART_LOGGER.warning")
    def test_add_model_uri_to_kwargs_custom_uri_without_incremental(
        self,
        mock_warning,
        mock_retrieve,
        mock_supports_incremental,
        mock_supports_training,
        mock_kwargs,
    ):
        """Test using custom model URI without incremental training support logs warning."""
        default_uri = "s3://jumpstart-models/training/test-model/1.0.0"
        custom_uri = "s3://custom-bucket/my-model"
        mock_retrieve.return_value = default_uri
        mock_kwargs.model_uri = custom_uri

        result = _add_model_uri_to_kwargs(mock_kwargs)

        mock_supports_training.assert_called_once()
        mock_supports_incremental.assert_called_once()
        mock_warning.assert_called_once()
        assert "does not support incremental training" in mock_warning.call_args[0][0]
        assert result.model_uri == custom_uri

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=False,
    )
    def test_add_model_uri_to_kwargs_no_training_support(self, mock_supports_training, mock_kwargs):
        """Test when model doesn't support training model URI."""
        result = _add_model_uri_to_kwargs(mock_kwargs)

        mock_supports_training.assert_called_once()
        assert result.model_uri is None

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=False,
    )
    @patch("sagemaker.jumpstart.factory.estimator.model_uris.retrieve")
    def test_add_model_uri_to_kwargs_private_hub(
        self, mock_retrieve, mock_supports_training, mock_kwargs
    ):
        """Test when model is from a private hub."""
        default_uri = "s3://jumpstart-models/training/test-model/1.0.0"
        mock_retrieve.return_value = default_uri
        mock_kwargs.hub_arn = "arn:aws:sagemaker:us-west-2:123456789012:hub/private-hub"

        result = _add_model_uri_to_kwargs(mock_kwargs)

        # Should not check if model supports training model URI for private hub
        mock_supports_training.assert_not_called()
        mock_retrieve.assert_called_once()
        assert result.model_uri == default_uri

    @patch(
        "sagemaker.jumpstart.factory.estimator._model_supports_training_model_uri",
        return_value=False,
    )
    @patch("sagemaker.jumpstart.factory.estimator.model_uris.retrieve")
    def test_add_model_uri_to_kwargs_public_hub(
        self, mock_retrieve, mock_supports_training, mock_kwargs
    ):
        """Test when model is from the public hub."""
        mock_kwargs.hub_arn = (
            f"arn:aws:sagemaker:us-west-2:123456789012:hub/{JUMPSTART_MODEL_HUB_NAME}"
        )

        result = _add_model_uri_to_kwargs(mock_kwargs)

        # Should check if model supports training model URI for public hub
        mock_supports_training.assert_called_once()
        mock_retrieve.assert_not_called()
        assert result.model_uri is None
