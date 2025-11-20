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
"""Unit tests for sagemaker.core.deserializers.implementations module."""
from __future__ import absolute_import

import pytest
from unittest.mock import Mock, patch, MagicMock
from sagemaker.core.deserializers import implementations
from sagemaker.core.deserializers.base import JSONDeserializer


class TestRetrieveOptions:
    """Test retrieve_options function."""

    def test_retrieve_options_missing_model_id(self):
        """Test that ValueError is raised when model_id is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_options(
                region="us-west-2",
                model_version="1.0"
            )

    def test_retrieve_options_missing_model_version(self):
        """Test that ValueError is raised when model_version is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_options(
                region="us-west-2",
                model_id="test-model"
            )

    @patch('sagemaker.core.deserializers.implementations.jumpstart_utils.is_jumpstart_model_input')
    @patch('sagemaker.core.deserializers.implementations.artifacts._retrieve_deserializer_options')
    def test_retrieve_options_success(self, mock_retrieve, mock_is_jumpstart):
        """Test successful retrieval of deserializer options."""
        mock_is_jumpstart.return_value = True
        mock_deserializers = [JSONDeserializer()]
        mock_retrieve.return_value = mock_deserializers
        
        result = implementations.retrieve_options(
            region="us-west-2",
            model_id="test-model",
            model_version="1.0"
        )
        
        assert result == mock_deserializers
        mock_retrieve.assert_called_once()

    @patch('sagemaker.core.deserializers.implementations.jumpstart_utils.is_jumpstart_model_input')
    @patch('sagemaker.core.deserializers.implementations.artifacts._retrieve_deserializer_options')
    def test_retrieve_options_with_all_params(self, mock_retrieve, mock_is_jumpstart):
        """Test retrieve_options with all parameters."""
        mock_is_jumpstart.return_value = True
        mock_deserializers = [JSONDeserializer()]
        mock_retrieve.return_value = mock_deserializers
        mock_session = Mock()
        
        result = implementations.retrieve_options(
            region="us-east-1",
            model_id="test-model",
            model_version="2.0",
            hub_arn="arn:aws:sagemaker:us-east-1:123456789012:hub/test-hub",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
            sagemaker_session=mock_session
        )
        
        assert result == mock_deserializers
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["model_version"] == "2.0"
        assert call_kwargs["region"] == "us-east-1"
        assert call_kwargs["tolerate_vulnerable_model"] is True
        assert call_kwargs["tolerate_deprecated_model"] is True


class TestRetrieveDefault:
    """Test retrieve_default function."""

    def test_retrieve_default_missing_model_id(self):
        """Test that ValueError is raised when model_id is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_default(
                region="us-west-2",
                model_version="1.0"
            )

    def test_retrieve_default_missing_model_version(self):
        """Test that ValueError is raised when model_version is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_default(
                region="us-west-2",
                model_id="test-model"
            )

    @patch('sagemaker.core.deserializers.implementations.jumpstart_utils.is_jumpstart_model_input')
    @patch('sagemaker.core.deserializers.implementations.artifacts._retrieve_default_deserializer')
    def test_retrieve_default_success(self, mock_retrieve, mock_is_jumpstart):
        """Test successful retrieval of default deserializer."""
        mock_is_jumpstart.return_value = True
        mock_deserializer = JSONDeserializer()
        mock_retrieve.return_value = mock_deserializer
        
        result = implementations.retrieve_default(
            region="us-west-2",
            model_id="test-model",
            model_version="1.0"
        )
        
        assert result == mock_deserializer
        mock_retrieve.assert_called_once()

    @patch('sagemaker.core.deserializers.implementations.jumpstart_utils.is_jumpstart_model_input')
    @patch('sagemaker.core.deserializers.implementations.artifacts._retrieve_default_deserializer')
    def test_retrieve_default_with_all_params(self, mock_retrieve, mock_is_jumpstart):
        """Test retrieve_default with all parameters."""
        mock_is_jumpstart.return_value = True
        mock_deserializer = JSONDeserializer()
        mock_retrieve.return_value = mock_deserializer
        mock_session = Mock()
        
        result = implementations.retrieve_default(
            region="us-east-1",
            model_id="test-model",
            model_version="2.0",
            hub_arn="arn:aws:sagemaker:us-east-1:123456789012:hub/test-hub",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
            sagemaker_session=mock_session,
            config_name="test-config"
        )
        
        assert result == mock_deserializer
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["model_version"] == "2.0"
        assert call_kwargs["config_name"] == "test-config"


class TestBackwardCompatibility:
    """Test backward compatibility imports."""

    def test_base_deserializer_import(self):
        """Test that BaseDeserializer can be imported."""
        from sagemaker.core.deserializers.implementations import BaseDeserializer
        assert BaseDeserializer is not None

    def test_bytes_deserializer_import(self):
        """Test that BytesDeserializer can be imported."""
        from sagemaker.core.deserializers.implementations import BytesDeserializer
        assert BytesDeserializer is not None

    def test_json_deserializer_import(self):
        """Test that JSONDeserializer can be imported."""
        from sagemaker.core.deserializers.implementations import JSONDeserializer
        assert JSONDeserializer is not None

    def test_numpy_deserializer_import(self):
        """Test that NumpyDeserializer can be imported."""
        from sagemaker.core.deserializers.implementations import NumpyDeserializer
        assert NumpyDeserializer is not None

    def test_record_deserializer_deprecated(self):
        """Test that record_deserializer is available as deprecated."""
        assert hasattr(implementations, 'record_deserializer')
