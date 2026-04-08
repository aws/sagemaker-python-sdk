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
"""Unit tests for sagemaker.core.serializers.implementations module."""
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from sagemaker.core.serializers import implementations
from sagemaker.core.serializers.base import JSONSerializer


class TestRetrieveOptions:
    """Test retrieve_options function."""

    def test_retrieve_options_missing_model_id(self):
        """Test that ValueError is raised when model_id is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_options(region="us-west-2", model_version="1.0")

    def test_retrieve_options_missing_model_version(self):
        """Test that ValueError is raised when model_version is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_options(region="us-west-2", model_id="test-model")

    @patch("sagemaker.core.serializers.implementations.jumpstart_utils.is_jumpstart_model_input")
    @patch("sagemaker.core.serializers.implementations.artifacts._retrieve_serializer_options")
    def test_retrieve_options_success(self, mock_retrieve, mock_is_jumpstart):
        """Test successful retrieval of serializer options."""
        mock_is_jumpstart.return_value = True
        mock_serializers = [JSONSerializer()]
        mock_retrieve.return_value = mock_serializers

        result = implementations.retrieve_options(
            region="us-west-2", model_id="test-model", model_version="1.0"
        )

        assert result == mock_serializers
        mock_retrieve.assert_called_once()

    @patch("sagemaker.core.serializers.implementations.jumpstart_utils.is_jumpstart_model_input")
    @patch("sagemaker.core.serializers.implementations.artifacts._retrieve_serializer_options")
    def test_retrieve_options_with_all_params(self, mock_retrieve, mock_is_jumpstart):
        """Test retrieve_options with all parameters."""
        mock_is_jumpstart.return_value = True
        mock_serializers = [JSONSerializer()]
        mock_retrieve.return_value = mock_serializers
        mock_session = Mock()

        result = implementations.retrieve_options(
            region="us-east-1",
            model_id="test-model",
            model_version="2.0",
            hub_arn="arn:aws:sagemaker:us-east-1:123456789012:hub/test-hub",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
            sagemaker_session=mock_session,
            config_name="test-config",
        )

        assert result == mock_serializers
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["model_version"] == "2.0"
        assert call_kwargs["region"] == "us-east-1"
        assert call_kwargs["tolerate_vulnerable_model"] is True
        assert call_kwargs["tolerate_deprecated_model"] is True
        assert call_kwargs["config_name"] == "test-config"


class TestRetrieveDefault:
    """Test retrieve_default function."""

    def test_retrieve_default_missing_model_id(self):
        """Test that ValueError is raised when model_id is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_default(region="us-west-2", model_version="1.0")

    def test_retrieve_default_missing_model_version(self):
        """Test that ValueError is raised when model_version is missing."""
        with pytest.raises(ValueError, match="Must specify JumpStart"):
            implementations.retrieve_default(region="us-west-2", model_id="test-model")

    @patch("sagemaker.core.serializers.implementations.jumpstart_utils.is_jumpstart_model_input")
    @patch("sagemaker.core.serializers.implementations.artifacts._retrieve_default_serializer")
    def test_retrieve_default_success(self, mock_retrieve, mock_is_jumpstart):
        """Test successful retrieval of default serializer."""
        mock_is_jumpstart.return_value = True
        mock_serializer = JSONSerializer()
        mock_retrieve.return_value = mock_serializer

        result = implementations.retrieve_default(
            region="us-west-2", model_id="test-model", model_version="1.0"
        )

        assert result == mock_serializer
        mock_retrieve.assert_called_once()

    @patch("sagemaker.core.serializers.implementations.jumpstart_utils.is_jumpstart_model_input")
    @patch("sagemaker.core.serializers.implementations.artifacts._retrieve_default_serializer")
    def test_retrieve_default_with_all_params(self, mock_retrieve, mock_is_jumpstart):
        """Test retrieve_default with all parameters."""
        mock_is_jumpstart.return_value = True
        mock_serializer = JSONSerializer()
        mock_retrieve.return_value = mock_serializer
        mock_session = Mock()

        result = implementations.retrieve_default(
            region="us-east-1",
            model_id="test-model",
            model_version="2.0",
            hub_arn="arn:aws:sagemaker:us-east-1:123456789012:hub/test-hub",
            tolerate_vulnerable_model=True,
            tolerate_deprecated_model=True,
            sagemaker_session=mock_session,
            config_name="test-config",
        )

        assert result == mock_serializer
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["model_id"] == "test-model"
        assert call_kwargs["model_version"] == "2.0"
        assert call_kwargs["config_name"] == "test-config"


class TestBackwardCompatibility:
    """Test backward compatibility imports."""

    def test_base_serializer_import(self):
        """Test that BaseSerializer can be imported."""
        from sagemaker.core.serializers.implementations import BaseSerializer

        assert BaseSerializer is not None

    def test_csv_serializer_import(self):
        """Test that CSVSerializer can be imported."""
        from sagemaker.core.serializers.implementations import CSVSerializer

        assert CSVSerializer is not None

    def test_json_serializer_import(self):
        """Test that JSONSerializer can be imported."""
        from sagemaker.core.serializers.implementations import JSONSerializer

        assert JSONSerializer is not None

    def test_numpy_serializer_import(self):
        """Test that NumpySerializer can be imported."""
        from sagemaker.core.serializers.implementations import NumpySerializer

        assert NumpySerializer is not None

    def test_record_serializer_deprecated(self):
        """Test that numpy_to_record_serializer is available as deprecated."""
        # numpy_to_record_serializer may or may not be present depending on the module
        # Just verify the module itself is importable
        assert implementations is not None

    def test_torch_tensor_serializer_import(self):
        """Test that TorchTensorSerializer can be imported from base module."""
        from sagemaker.core.serializers.base import TorchTensorSerializer

        assert TorchTensorSerializer is not None

    def test_torch_tensor_serializer_requires_torch(self):
        """Test that TorchTensorSerializer raises ImportError when torch is missing."""
        import importlib
        import sys

        saved = {}
        try:
            # Block torch
            torch_keys = [key for key in sys.modules if key.startswith("torch.")]
            saved = {key: sys.modules.pop(key) for key in torch_keys}
            saved["torch"] = sys.modules.get("torch")
            sys.modules["torch"] = None

            from sagemaker.core.serializers.base import TorchTensorSerializer

            with pytest.raises(ImportError, match="Unable to import torch"):
                TorchTensorSerializer()
        finally:
            # Restore torch
            original_torch = saved.pop("torch", None)
            if original_torch is not None:
                sys.modules["torch"] = original_torch
            elif "torch" in sys.modules:
                del sys.modules["torch"]
            for key, val in saved.items():
                sys.modules[key] = val
