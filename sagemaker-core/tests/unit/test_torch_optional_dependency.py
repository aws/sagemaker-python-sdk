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
"""Tests for torch optional dependency behavior."""
from __future__ import absolute_import

import sys
from unittest.mock import patch

import numpy as np
import pytest


def test_torch_tensor_serializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorSerializer raises ImportError with helpful message when torch is missing."""
    import importlib
    import sagemaker.core.serializers.base as base_module

    with patch.dict(sys.modules, {"torch": None}):
        # Reload to clear any cached imports
        importlib.reload(base_module)
        with pytest.raises(ImportError, match="pip install 'sagemaker-core\\[torch\\]'"):
            base_module.TorchTensorSerializer()

    # Reload again to restore normal state
    importlib.reload(base_module)


def test_torch_tensor_deserializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorDeserializer raises ImportError when torch is missing."""
    import importlib
    import sagemaker.core.deserializers.base as base_module

    with patch.dict(sys.modules, {"torch": None}):
        importlib.reload(base_module)
        with pytest.raises(ImportError, match="pip install 'sagemaker-core\\[torch\\]'"):
            base_module.TorchTensorDeserializer()

    # Reload again to restore normal state
    importlib.reload(base_module)


def test_torch_tensor_serializer_works_when_torch_installed():
    """Verify TorchTensorSerializer can be instantiated when torch is available."""
    pytest.importorskip("torch")
    from sagemaker.core.serializers.base import TorchTensorSerializer

    serializer = TorchTensorSerializer()
    assert serializer is not None
    assert serializer.CONTENT_TYPE == "tensor/pt"


def test_torch_tensor_deserializer_works_when_torch_installed():
    """Verify TorchTensorDeserializer can be instantiated when torch is available."""
    pytest.importorskip("torch")
    from sagemaker.core.deserializers.base import TorchTensorDeserializer

    deserializer = TorchTensorDeserializer()
    assert deserializer is not None
    assert deserializer.ACCEPT == ("tensor/pt",)


def test_sagemaker_core_imports_without_torch():
    """Verify that importing serializers/deserializers modules does not fail without torch."""
    import importlib
    import sagemaker.core.serializers.base as ser_base
    import sagemaker.core.deserializers.base as deser_base

    with patch.dict(sys.modules, {"torch": None}):
        # Reloading the modules should not raise since torch imports are lazy (in __init__)
        importlib.reload(ser_base)
        importlib.reload(deser_base)

    # Restore
    importlib.reload(ser_base)
    importlib.reload(deser_base)


def test_other_serializers_work_without_torch():
    """Verify non-torch serializers work normally even if torch is unavailable."""
    import importlib
    import sagemaker.core.serializers.base as base_module

    with patch.dict(sys.modules, {"torch": None}):
        importlib.reload(base_module)

        csv_ser = base_module.CSVSerializer()
        assert csv_ser.serialize([1, 2, 3]) == "1,2,3"

        json_ser = base_module.JSONSerializer()
        assert json_ser.serialize([1, 2, 3]) == "[1, 2, 3]"

        numpy_ser = base_module.NumpySerializer()
        result = numpy_ser.serialize(np.array([1, 2, 3]))
        assert result is not None

    # Restore
    importlib.reload(base_module)
