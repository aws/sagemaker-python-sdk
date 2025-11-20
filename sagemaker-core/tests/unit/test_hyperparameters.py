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

from sagemaker.core import hyperparameters
from sagemaker.core.jumpstart.enums import HyperparameterValidationMode, JumpStartModelType


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.artifacts._retrieve_default_hyperparameters")
def test_retrieve_default_success(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with valid JumpStart model inputs."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = {"learning_rate": "0.001", "epochs": "10"}
    
    result = hyperparameters.retrieve_default(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0"
    )
    
    assert result == {"learning_rate": "0.001", "epochs": "10"}
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")
    mock_retrieve.assert_called_once()


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
def test_retrieve_default_missing_model_id(mock_is_jumpstart):
    """Test retrieve_default raises ValueError when model_id is missing."""
    mock_is_jumpstart.return_value = False
    
    with pytest.raises(ValueError, match="Must specify JumpStart"):
        hyperparameters.retrieve_default(
            region="us-west-2",
            model_version="1.0.0"
        )


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.artifacts._retrieve_default_hyperparameters")
def test_retrieve_default_with_instance_type(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with instance_type parameter."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = {"learning_rate": "0.001"}
    
    hyperparameters.retrieve_default(
        model_id="test-model",
        model_version="1.0.0",
        instance_type="ml.p3.2xlarge"
    )
    
    assert mock_retrieve.call_args[1]["instance_type"] == "ml.p3.2xlarge"


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.artifacts._retrieve_default_hyperparameters")
def test_retrieve_default_with_container_hyperparameters(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with include_container_hyperparameters flag."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = {"learning_rate": "0.001", "sagemaker_program": "train.py"}
    
    hyperparameters.retrieve_default(
        model_id="test-model",
        model_version="1.0.0",
        include_container_hyperparameters=True
    )
    
    assert mock_retrieve.call_args[1]["include_container_hyperparameters"] is True


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.artifacts._retrieve_default_hyperparameters")
def test_retrieve_default_with_model_type(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_default with custom model_type."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = {"learning_rate": "0.001"}
    
    hyperparameters.retrieve_default(
        model_id="test-model",
        model_version="1.0.0",
        model_type=JumpStartModelType.PROPRIETARY
    )
    
    assert mock_retrieve.call_args[1]["model_type"] == JumpStartModelType.PROPRIETARY


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.validate_hyperparameters")
def test_validate_success(mock_validate, mock_is_jumpstart):
    """Test validate with valid hyperparameters."""
    mock_is_jumpstart.return_value = True
    mock_validate.return_value = None
    
    hyperparameters.validate(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0",
        hyperparameters={"learning_rate": "0.001"}
    )
    
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")
    mock_validate.assert_called_once()


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
def test_validate_missing_model_id(mock_is_jumpstart):
    """Test validate raises ValueError when model_id is missing."""
    mock_is_jumpstart.return_value = False
    
    with pytest.raises(ValueError, match="Must specify JumpStart"):
        hyperparameters.validate(
            region="us-west-2",
            model_version="1.0.0",
            hyperparameters={"learning_rate": "0.001"}
        )


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
def test_validate_missing_hyperparameters(mock_is_jumpstart):
    """Test validate raises ValueError when hyperparameters is None."""
    mock_is_jumpstart.return_value = True
    
    with pytest.raises(ValueError, match="Must specify hyperparameters"):
        hyperparameters.validate(
            region="us-west-2",
            model_id="test-model",
            model_version="1.0.0",
            hyperparameters=None
        )


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.validate_hyperparameters")
def test_validate_with_validation_mode(mock_validate, mock_is_jumpstart):
    """Test validate with custom validation_mode."""
    mock_is_jumpstart.return_value = True
    mock_validate.return_value = None
    
    hyperparameters.validate(
        model_id="test-model",
        model_version="1.0.0",
        hyperparameters={"learning_rate": "0.001"},
        validation_mode=HyperparameterValidationMode.VALIDATE_ALL
    )
    
    assert mock_validate.call_args[1]["validation_mode"] == HyperparameterValidationMode.VALIDATE_ALL


@patch("sagemaker.core.hyperparameters.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.hyperparameters.validate_hyperparameters")
def test_validate_with_tolerance_flags(mock_validate, mock_is_jumpstart):
    """Test validate with vulnerability and deprecation tolerance flags."""
    mock_is_jumpstart.return_value = True
    mock_validate.return_value = None
    
    hyperparameters.validate(
        model_id="test-model",
        model_version="1.0.0",
        hyperparameters={"learning_rate": "0.001"},
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True
    )
    
    assert mock_validate.call_args[1]["tolerate_vulnerable_model"] is True
    assert mock_validate.call_args[1]["tolerate_deprecated_model"] is True
