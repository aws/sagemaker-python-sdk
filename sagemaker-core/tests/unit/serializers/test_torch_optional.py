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

import sys
from unittest.mock import patch, MagicMock

import pytest
import numpy as np


def test_torch_tensor_serializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorSerializer() raises ImportError with helpful install message
    when torch is not installed."""
    import sagemaker.core.serializers.base as base_module

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="pip install.*torch"):
            base_module.TorchTensorSerializer()


def test_torch_tensor_deserializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorDeserializer() raises ImportError with helpful install message
    when torch is not installed."""
    import sagemaker.core.deserializers.base as base_module

    with patch.dict(sys.modules, {"torch": None}):
        with pytest.raises(ImportError, match="pip install.*torch"):
            base_module.TorchTensorDeserializer()


def test_non_torch_serializers_work_without_torch():
    """Verify CSVSerializer, JSONSerializer, NumpySerializer etc. all work fine
    even if torch is not available."""
    from sagemaker.core.serializers.base import (
        CSVSerializer,
        JSONSerializer,
        NumpySerializer,
        IdentitySerializer,
    )

    csv_ser = CSVSerializer()
    assert csv_ser.serialize([1, 2, 3]) == "1,2,3"

    json_ser = JSONSerializer()
    assert json_ser.serialize({"a": 1}) == '{"a": 1}'

    numpy_ser = NumpySerializer()
    result = numpy_ser.serialize(np.array([1, 2, 3]))
    assert result is not None

    identity_ser = IdentitySerializer()
    assert identity_ser.serialize(b"hello") == b"hello"


def test_torch_tensor_serializer_works_when_torch_available():
    """Verify TorchTensorSerializer works normally when torch is installed."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from sagemaker.core.serializers.base import TorchTensorSerializer

    serializer = TorchTensorSerializer()
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = serializer.serialize(tensor)
    assert result is not None


def test_torch_tensor_deserializer_works_when_torch_available():
    """Verify TorchTensorDeserializer works normally when torch is installed."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")

    from sagemaker.core.deserializers.base import TorchTensorDeserializer

    deserializer = TorchTensorDeserializer()
    assert deserializer is not None
