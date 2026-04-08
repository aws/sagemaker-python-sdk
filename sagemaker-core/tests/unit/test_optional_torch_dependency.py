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
from __future__ import annotations

import importlib
import io
import sys

import numpy as np
import pytest


def _block_torch():
    """Block torch imports by setting sys.modules['torch'] to None.

    Returns a dict of saved torch submodule entries so they can be restored.
    """
    saved = {}
    torch_keys = [key for key in sys.modules if key.startswith("torch.")]
    saved = {key: sys.modules.pop(key) for key in torch_keys}
    saved["torch"] = sys.modules.get("torch")
    sys.modules["torch"] = None
    return saved


def _restore_torch(saved):
    """Restore torch modules from saved dict."""
    original_torch = saved.pop("torch", None)
    if original_torch is not None:
        sys.modules["torch"] = original_torch
    elif "torch" in sys.modules:
        del sys.modules["torch"]
    for key, val in saved.items():
        sys.modules[key] = val


def test_serializer_module_imports_without_torch():
    """Verify that importing non-torch serializers succeeds without torch installed."""
    saved = {}
    try:
        saved = _block_torch()

        # Reload the module so it re-evaluates imports with torch blocked
        import sagemaker.core.serializers.base as ser_module

        importlib.reload(ser_module)

        # Verify non-torch serializers can be instantiated
        assert ser_module.CSVSerializer() is not None
        assert ser_module.NumpySerializer() is not None
        assert ser_module.JSONSerializer() is not None
        assert ser_module.IdentitySerializer() is not None
    finally:
        _restore_torch(saved)


def test_deserializer_module_imports_without_torch():
    """Verify that importing non-torch deserializers succeeds without torch installed."""
    saved = {}
    try:
        saved = _block_torch()

        import sagemaker.core.deserializers.base as deser_module

        importlib.reload(deser_module)

        # Verify non-torch deserializers can be instantiated
        assert deser_module.StringDeserializer() is not None
        assert deser_module.BytesDeserializer() is not None
        assert deser_module.CSVDeserializer() is not None
        assert deser_module.NumpyDeserializer() is not None
        assert deser_module.JSONDeserializer() is not None
    finally:
        _restore_torch(saved)


def test_torch_tensor_serializer_raises_import_error_without_torch():
    """Verify TorchTensorSerializer raises ImportError when torch is not installed."""
    import sagemaker.core.serializers.base as ser_module

    saved = {}
    try:
        saved = _block_torch()

        with pytest.raises(ImportError, match="Unable to import torch"):
            ser_module.TorchTensorSerializer()
    finally:
        _restore_torch(saved)


def test_torch_tensor_deserializer_raises_import_error_without_torch():
    """Verify TorchTensorDeserializer raises ImportError when torch is not installed."""
    import sagemaker.core.deserializers.base as deser_module

    saved = {}
    try:
        saved = _block_torch()

        with pytest.raises(ImportError, match="Unable to import torch"):
            deser_module.TorchTensorDeserializer()
    finally:
        _restore_torch(saved)


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
