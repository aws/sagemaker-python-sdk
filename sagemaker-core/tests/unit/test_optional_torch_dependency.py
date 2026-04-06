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
"""Tests to verify torch dependency is optional in sagemaker-core."""
from __future__ import absolute_import

import io
import sys
from unittest import mock

import numpy as np
import pytest


def test_serializer_module_imports_without_torch():
    """Verify that importing serializers module succeeds without torch installed."""
    # The serializers module should be importable even without torch
    # because TorchTensorSerializer uses lazy import in __init__
    from sagemaker.core.serializers.base import (
        CSVSerializer,
        NumpySerializer,
        JSONSerializer,
        IdentitySerializer,
        SparseMatrixSerializer,
        JSONLinesSerializer,
        LibSVMSerializer,
        DataSerializer,
        StringSerializer,
    )
    # Verify non-torch serializers can be instantiated
    assert CSVSerializer() is not None
    assert NumpySerializer() is not None
    assert JSONSerializer() is not None
    assert IdentitySerializer() is not None


def test_deserializer_module_imports_without_torch():
    """Verify that importing deserializers module succeeds without torch installed."""
    from sagemaker.core.deserializers.base import (
        StringDeserializer,
        BytesDeserializer,
        CSVDeserializer,
        StreamDeserializer,
        NumpyDeserializer,
        JSONDeserializer,
        PandasDeserializer,
        JSONLinesDeserializer,
    )
    # Verify non-torch deserializers can be instantiated
    assert StringDeserializer() is not None
    assert BytesDeserializer() is not None
    assert CSVDeserializer() is not None
    assert NumpyDeserializer() is not None
    assert JSONDeserializer() is not None


def test_torch_tensor_serializer_raises_import_error_without_torch():
    """Verify TorchTensorSerializer raises ImportError when torch is not installed."""
    import importlib
    import sagemaker.core.serializers.base as ser_module

    # Save original torch module if present
    original_torch = sys.modules.get('torch')
    
    try:
        # Simulate torch not being installed
        sys.modules['torch'] = None
        # Need to also handle the case where torch submodules are cached
        torch_keys = [key for key in sys.modules if key.startswith('torch.')]
        saved = {key: sys.modules.pop(key) for key in torch_keys}
        
        with pytest.raises(ImportError, match="Unable to import torch"):
            ser_module.TorchTensorSerializer()
    finally:
        # Restore original state
        if original_torch is not None:
            sys.modules['torch'] = original_torch
        elif 'torch' in sys.modules:
            del sys.modules['torch']
        for key, val in saved.items():
            sys.modules[key] = val


def test_torch_tensor_deserializer_raises_import_error_without_torch():
    """Verify TorchTensorDeserializer raises ImportError when torch is not installed."""
    import sagemaker.core.deserializers.base as deser_module

    # Save original torch module if present
    original_torch = sys.modules.get('torch')
    
    try:
        # Simulate torch not being installed
        sys.modules['torch'] = None
        torch_keys = [key for key in sys.modules if key.startswith('torch.')]
        saved = {key: sys.modules.pop(key) for key in torch_keys}
        
        with pytest.raises(ImportError, match="Unable to import torch"):
            deser_module.TorchTensorDeserializer()
    finally:
        # Restore original state
        if original_torch is not None:
            sys.modules['torch'] = original_torch
        elif 'torch' in sys.modules:
            del sys.modules['torch']
        for key, val in saved.items():
            sys.modules[key] = val


def test_torch_tensor_serializer_works_with_torch():
    """Verify TorchTensorSerializer works when torch is available."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")

    from sagemaker.core.serializers.base import TorchTensorSerializer

    serializer = TorchTensorSerializer()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = serializer.serialize(tensor)
    assert result is not None
    # Verify the result can be loaded back as numpy
    array = np.load(io.BytesIO(result))
    assert np.array_equal(array, np.array([1.0, 2.0, 3.0]))


def test_torch_tensor_deserializer_works_with_torch():
    """Verify TorchTensorDeserializer works when torch is available."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch is not installed")

    from sagemaker.core.deserializers.base import TorchTensorDeserializer

    deserializer = TorchTensorDeserializer()
    # Create a numpy array, save it, and deserialize to tensor
    array = np.array([1.0, 2.0, 3.0])
    buffer = io.BytesIO()
    np.save(buffer, array)
    buffer.seek(0)

    result = deserializer.deserialize(buffer, "tensor/pt")
    assert isinstance(result, torch.Tensor)
    assert torch.equal(result, torch.tensor([1.0, 2.0, 3.0]))
