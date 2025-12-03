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

from sagemaker.core import script_uris
from sagemaker.core.jumpstart.enums import JumpStartModelType


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_success(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with valid JumpStart model inputs."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    result = script_uris.retrieve(region="us-west-2", model_id="test-model", model_version="1.0.0")

    assert result == "s3://bucket/scripts/inference.py"
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")
    mock_retrieve.assert_called_once()


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
def test_retrieve_missing_model_id(mock_is_jumpstart):
    """Test retrieve raises ValueError when model_id is missing."""
    mock_is_jumpstart.return_value = False

    with pytest.raises(ValueError, match="Must specify JumpStart"):
        script_uris.retrieve(region="us-west-2", model_version="1.0.0")


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_script_scope(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with script_scope parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/training.py"

    script_uris.retrieve(model_id="test-model", model_version="1.0.0", script_scope="training")

    assert mock_retrieve.call_args[1]["script_scope"] == "training"


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_hub_arn(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with hub_arn parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    script_uris.retrieve(
        model_id="test-model",
        model_version="1.0.0",
        hub_arn="arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub",
    )

    assert (
        mock_retrieve.call_args[1]["hub_arn"]
        == "arn:aws:sagemaker:us-west-2:123456789012:hub/test-hub"
    )


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_tolerance_flags(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with vulnerability and deprecation tolerance flags."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    script_uris.retrieve(
        model_id="test-model",
        model_version="1.0.0",
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
    )

    assert mock_retrieve.call_args[1]["tolerate_vulnerable_model"] is True
    assert mock_retrieve.call_args[1]["tolerate_deprecated_model"] is True


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_model_type(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with custom model_type."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    script_uris.retrieve(
        model_id="test-model", model_version="1.0.0", model_type=JumpStartModelType.PROPRIETARY
    )

    assert mock_retrieve.call_args[1]["model_type"] == JumpStartModelType.PROPRIETARY


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_config_name(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with config_name parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    script_uris.retrieve(model_id="test-model", model_version="1.0.0", config_name="test-config")

    assert mock_retrieve.call_args[1]["config_name"] == "test-config"


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_with_session(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with custom sagemaker_session."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"
    mock_session = Mock()

    script_uris.retrieve(
        model_id="test-model", model_version="1.0.0", sagemaker_session=mock_session
    )

    assert mock_retrieve.call_args[1]["sagemaker_session"] == mock_session


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_inference_scope(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with inference script_scope."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/inference.py"

    result = script_uris.retrieve(
        model_id="test-model", model_version="1.0.0", script_scope="inference"
    )

    assert result == "s3://bucket/scripts/inference.py"
    assert mock_retrieve.call_args[1]["script_scope"] == "inference"


@patch("sagemaker.core.script_uris.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.script_uris.artifacts._retrieve_script_uri")
def test_retrieve_all_parameters(mock_retrieve, mock_is_jumpstart):
    """Test retrieve with all parameters specified."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = "s3://bucket/scripts/training.py"
    mock_session = Mock()

    result = script_uris.retrieve(
        region="eu-west-1",
        model_id="test-model",
        model_version="2.0.0",
        hub_arn="arn:aws:sagemaker:eu-west-1:123456789012:hub/test-hub",
        script_scope="training",
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True,
        sagemaker_session=mock_session,
        config_name="test-config",
        model_type=JumpStartModelType.PROPRIETARY,
    )

    assert result == "s3://bucket/scripts/training.py"
    mock_retrieve.assert_called_once()
