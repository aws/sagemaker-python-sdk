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
from unittest import mock

import pytest


def test_torch_tensor_serializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorSerializer raises ImportError with helpful message when torch is missing."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        # Need to reload the module to pick up the mocked import
        from sagemaker.core.serializers.base import TorchTensorSerializer

        with pytest.raises(ImportError, match="pip install"):
            TorchTensorSerializer()


def test_torch_tensor_deserializer_raises_import_error_when_torch_missing():
    """Verify TorchTensorDeserializer raises ImportError with helpful message when torch is missing."""
    with mock.patch.dict(sys.modules, {"torch": None}):
        from sagemaker.core.deserializers.base import TorchTensorDeserializer

        with pytest.raises(ImportError, match="pip install"):
            TorchTensorDeserializer()


def test_torch_tensor_serializer_works_when_torch_available():
    """Verify TorchTensorSerializer can be instantiated when torch is available."""
    torch = pytest.importorskip("torch")
    from sagemaker.core.serializers.base import TorchTensorSerializer

    serializer = TorchTensorSerializer()
    assert serializer.CONTENT_TYPE == "tensor/pt"

    # Test serialization of a simple tensor
    tensor = torch.tensor([1.0, 2.0, 3.0])
    result = serializer.serialize(tensor)
    assert result is not None


def test_torch_tensor_deserializer_works_when_torch_available():
    """Verify TorchTensorDeserializer can be instantiated when torch is available."""
    pytest.importorskip("torch")
    from sagemaker.core.deserializers.base import TorchTensorDeserializer

    deserializer = TorchTensorDeserializer()
    assert deserializer.ACCEPT == ("tensor/pt",)


def test_base_serializers_importable_without_torch():
    """Verify non-torch serializers can be imported and used without torch."""
    from sagemaker.core.serializers.base import (
        CSVSerializer,
        NumpySerializer,
        JSONSerializer,
        IdentitySerializer,
        JSONLinesSerializer,
        LibSVMSerializer,
        DataSerializer,
        StringSerializer,
    )

    # Verify they can be instantiated
    assert CSVSerializer() is not None
    assert NumpySerializer() is not None
    assert JSONSerializer() is not None
    assert IdentitySerializer() is not None
    assert JSONLinesSerializer() is not None
    assert LibSVMSerializer() is not None
    assert DataSerializer() is not None
    assert StringSerializer() is not None


def test_base_deserializers_importable_without_torch():
    """Verify non-torch deserializers can be imported and used without torch."""
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

    # Verify they can be instantiated
    assert StringDeserializer() is not None
    assert BytesDeserializer() is not None
    assert CSVDeserializer() is not None
    assert StreamDeserializer() is not None
    assert NumpyDeserializer() is not None
    assert JSONDeserializer() is not None
    assert PandasDeserializer() is not None
    assert JSONLinesDeserializer() is not None
