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
from unittest.mock import Mock, patch, MagicMock

from sagemaker.core import payloads
from sagemaker.core.jumpstart.enums import JumpStartModelType
from sagemaker.core.jumpstart.types import JumpStartSerializablePayload


@patch("sagemaker.core.payloads.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.payloads.artifacts._retrieve_example_payloads")
def test_retrieve_all_examples_success(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_all_examples with valid JumpStart model inputs."""
    mock_is_jumpstart.return_value = True
    mock_payload = JumpStartSerializablePayload({
        "content_type": "application/json",
        "body": '{"input": "test"}',
        "accept": "application/json"
    })
    mock_retrieve.return_value = {"example1": mock_payload}
    
    result = payloads.retrieve_all_examples(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0",
        serialize=False
    )
    
    assert len(result) == 1
    assert result[0].content_type == "application/json"
    mock_is_jumpstart.assert_called_once_with("test-model", "1.0.0")


@patch("sagemaker.core.payloads.jumpstart_utils.is_jumpstart_model_input")
def test_retrieve_all_examples_missing_model_id(mock_is_jumpstart):
    """Test retrieve_all_examples raises ValueError when model_id is missing."""
    mock_is_jumpstart.return_value = False
    
    with pytest.raises(ValueError, match="Must specify JumpStart"):
        payloads.retrieve_all_examples(
            region="us-west-2",
            model_version="1.0.0"
        )


@patch("sagemaker.core.payloads.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.payloads.artifacts._retrieve_example_payloads")
def test_retrieve_all_examples_returns_none(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_all_examples returns None when no payloads available."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = None
    
    result = payloads.retrieve_all_examples(
        model_id="test-model",
        model_version="1.0.0"
    )
    
    assert result is None


@patch("sagemaker.core.payloads.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.payloads.artifacts._retrieve_example_payloads")
@patch("sagemaker.core.payloads.PayloadSerializer")
def test_retrieve_all_examples_with_serialization(mock_serializer_class, mock_retrieve, mock_is_jumpstart):
    """Test retrieve_all_examples with serialization enabled."""
    mock_is_jumpstart.return_value = True
    mock_payload = JumpStartSerializablePayload({
        "content_type": "application/json",
        "body": '{"input": "test"}',
        "accept": "application/json"
    })
    mock_retrieve.return_value = {"example1": mock_payload}
    
    mock_serializer = MagicMock()
    mock_serializer.serialize.return_value = b'{"input": "test"}'
    mock_serializer_class.return_value = mock_serializer
    
    mock_session = Mock()
    mock_session.s3_client = Mock()
    
    result = payloads.retrieve_all_examples(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0",
        serialize=True,
        sagemaker_session=mock_session
    )
    
    assert len(result) == 1
    mock_serializer.serialize.assert_called_once()


@patch("sagemaker.core.payloads.jumpstart_utils.is_jumpstart_model_input")
@patch("sagemaker.core.payloads.artifacts._retrieve_example_payloads")
def test_retrieve_all_examples_with_model_type(mock_retrieve, mock_is_jumpstart):
    """Test retrieve_all_examples with custom model_type."""
    mock_is_jumpstart.return_value = True
    mock_retrieve.return_value = {}
    
    payloads.retrieve_all_examples(
        model_id="test-model",
        model_version="1.0.0",
        model_type=JumpStartModelType.PROPRIETARY
    )
    
    assert mock_retrieve.call_args[1]["model_type"] == JumpStartModelType.PROPRIETARY


@patch("sagemaker.core.payloads.retrieve_all_examples")
def test_retrieve_example_success(mock_retrieve_all):
    """Test retrieve_example returns first payload."""
    mock_payload = JumpStartSerializablePayload({
        "content_type": "application/json",
        "body": '{"input": "test"}',
        "accept": "application/json"
    })
    mock_retrieve_all.return_value = [mock_payload]
    
    result = payloads.retrieve_example(
        region="us-west-2",
        model_id="test-model",
        model_version="1.0.0"
    )
    
    assert result == mock_payload
    assert result.content_type == "application/json"


@patch("sagemaker.core.payloads.retrieve_all_examples")
def test_retrieve_example_returns_none_when_empty(mock_retrieve_all):
    """Test retrieve_example returns None when no payloads available."""
    mock_retrieve_all.return_value = []
    
    result = payloads.retrieve_example(
        model_id="test-model",
        model_version="1.0.0"
    )
    
    assert result is None


@patch("sagemaker.core.payloads.retrieve_all_examples")
def test_retrieve_example_returns_none_when_none(mock_retrieve_all):
    """Test retrieve_example returns None when retrieve_all_examples returns None."""
    mock_retrieve_all.return_value = None
    
    result = payloads.retrieve_example(
        model_id="test-model",
        model_version="1.0.0"
    )
    
    assert result is None


@patch("sagemaker.core.payloads.retrieve_all_examples")
def test_retrieve_example_with_serialization(mock_retrieve_all):
    """Test retrieve_example passes serialize parameter correctly."""
    mock_payload = JumpStartSerializablePayload({
        "content_type": "application/json",
        "body": b'{"input": "test"}',
        "accept": "application/json"
    })
    mock_retrieve_all.return_value = [mock_payload]
    
    result = payloads.retrieve_example(
        model_id="test-model",
        model_version="1.0.0",
        serialize=True
    )
    
    assert result == mock_payload
    assert mock_retrieve_all.call_args[1]["serialize"] is True


@patch("sagemaker.core.payloads.retrieve_all_examples")
def test_retrieve_example_with_tolerance_flags(mock_retrieve_all):
    """Test retrieve_example passes tolerance flags correctly."""
    mock_payload = JumpStartSerializablePayload({
        "content_type": "application/json",
        "body": '{"input": "test"}',
        "accept": "application/json"
    })
    mock_retrieve_all.return_value = [mock_payload]
    
    payloads.retrieve_example(
        model_id="test-model",
        model_version="1.0.0",
        tolerate_vulnerable_model=True,
        tolerate_deprecated_model=True
    )
    
    assert mock_retrieve_all.call_args[1]["tolerate_vulnerable_model"] is True
    assert mock_retrieve_all.call_args[1]["tolerate_deprecated_model"] is True
