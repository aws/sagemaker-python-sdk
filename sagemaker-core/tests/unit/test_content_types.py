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
from unittest.mock import Mock, patch

from sagemaker.core import content_types
from sagemaker.core.jumpstart.enums import JumpStartModelType


def test_content_type_constants():
    """Test that content type constants are defined correctly."""
    assert content_types.CONTENT_TYPE_JSON == "application/json"
    assert content_types.CONTENT_TYPE_CSV == "text/csv"
    assert content_types.CONTENT_TYPE_OCTET_STREAM == "application/octet-stream"
    assert content_types.CONTENT_TYPE_NPY == "application/x-npy"


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_supported_content_types")
def test_retrieve_options_success(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_options with valid JumpStart model inputs."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = ["application/json", "text/csv"]

    result = content_types.retrieve_options(
        region="us-west-2", model_id="test-model", model_version="1.0.0"
    )

    assert result == ["application/json", "text/csv"]
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")
    mock_retrieve.assert_called_once()


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
def test_retrieve_options_missing_model_id(mock_is_jumpstart):
    """Test retrieve_options raises ValueError when model_id is missing."""
    mock_is_jumpstart.return_value = False

    with pytest.raises(ValueError, match="Must specify JumpStart"):
        content_types.retrieve_options(region="us-west-2", model_version="1.0.0")


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_supported_content_types")
def test_retrieve_options_with_hub_arn(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_options with hub_arn parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = ["application/json"]

    result = content_types.retrieve_options(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0",
        hub_arn="arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub",
    )

    assert result == ["application/json"]
    assert (
        mock_retrieve.call_args[1]["hub_arn"]
        == "arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub"
    )


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_supported_content_types")
def test_retrieve_options_with_tolerance_flags(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_options with vulnerability and deprecation tolerance flags."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = ["application/json"]

    content_types.retrieve_options(
        model_id="test-model",
        model_version="1.0.0",
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
    )

    assert mock_retrieve.call_args[1]["tolerate_vulnerable_model"] is True
    assert mock_retrieve.call_args[1]["tolerate_deprecated_model"] is True


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_default_content_type")
def test_retrieve_default_success(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with valid JumpStart model inputs."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "application/json"

    result = content_types.retrieve_default(
        region="us-west-2", model_id="test-model", model_version="1.0.0"
    )

    assert result == "application/json"
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")
    mock_retrieve.assert_called_once()


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
def test_retrieve_default_missing_model_version(mock_is_jumpstart):
    """Test retrieve_default raises ValueError when model_version is missing."""
    mock_is_jumpstart.return_value = False

    with pytest.raises(ValueError, match="Must specify JumpStart"):
        content_types.retrieve_default(region="us-west-2", model_id="test-model")


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_default_content_type")
def test_retrieve_default_with_model_type(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with custom model_type."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "application/json"

    content_types.retrieve_default(
        model_id="test-model", model_version="1.0.0", model_type=JumpStartModelType.PROPRIETARY
    )

    assert mock_retrieve.call_args[1]["model_type"] == JumpStartModelType.PROPRIETARY


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_default_content_type")
def test_retrieve_default_with_config_name(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with config_name parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "application/json"

    content_types.retrieve_default(
        model_id="test-model", model_version="1.0.0", config_name="test-config"
    )

    assert mock_retrieve.call_args[1]["config_name"] == "test-config"


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_default_content_type")
def test_retrieve_default_with_session(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with custom sagemaker_session."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "application/json"
    mock_session = Mock()

    content_types.retrieve_default(
        model_id="test-model", model_version="1.0.0", sagemaker_session=mock_session
    )

    assert mock_retrieve.call_args[1]["sagemaker_session"] == mock_session


@patch("sagemaker.core.content_types.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.content_types.artifacts._retrieve_supported_content_types")
def test_retrieve_options_all_parameters(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_options with all parameters specified."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = ["application/json", "text/csv", "application/x-npy"]
    mock_session = Mock()

    result = content_types.retrieve_options(
        region="eu-west-1",
        model_id="test-model",
        model_version="2.0.0",
        hub_arn="arn:aws:sagemaker:eu-west-1:123456789012:hub/test-hub",
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=mock_session,
    )

    assert len(result) == 3
    assert "application/json" in result
    mock_retrieve.assert_called_once()
