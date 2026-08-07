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
"""Tests to verify torch dependency is optional in sagemaker-serve.

Runs in a subprocess so that blocking torch cannot affect other tests in
the session.
"""
from __future__ import absolute_import

import subprocess
import sys
import textwrap


def _run_without_torch(body):
    """Execute body in a subprocess where importing torch raises ImportError."""
    script = textwrap.dedent(
        """
        import sys

        class _TorchBlocker:
            \"\"\"Meta path finder that makes `import torch` fail.\"\"\"

            def find_spec(self, name, path=None, target=None):
                if name == "torch" or name.startswith("torch."):
                    raise ImportError("torch is blocked for this test")
                return None

        sys.meta_path.insert(0, _TorchBlocker())
        for _name in [n for n in sys.modules if n == "torch" or n.startswith("torch.")]:
            del sys.modules[_name]
        """
    ) + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )


def test_tensor_serializer_works_without_torch():
    """Serializing a tensor needs only its own detach()/numpy(), not the torch import."""
    result = _run_without_torch(
        """
        import numpy as np

        from sagemaker.core.serializers import TorchTensorSerializer

        class FakeTensor:
            __module__ = "torch"

            def detach(self):
                return self

            def numpy(self):
                return np.array([1, 2, 3])

        assert TorchTensorSerializer().serialize(FakeTensor())
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_tensor_serializer_still_rejects_non_tensors():
    """The structural check must not widen to arbitrary objects."""
    result = _run_without_torch(
        """
        from sagemaker.core.serializers import TorchTensorSerializer

        try:
            TorchTensorSerializer().serialize([1, 2, 3])
            raise AssertionError("expected ValueError")
        except ValueError:
            print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_in_process_server_imports_without_torch():
    """The in-process server module must not import torch at module scope."""
    result = _run_without_torch(
        """
        from sagemaker.serve.model_server.in_process_model_server.app import InProcessServer

        assert InProcessServer is not None
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_constants_imports_without_torch():
    """serve.constants must not instantiate TorchTensorSerializer at import time."""
    result = _run_without_torch(
        """
        from sagemaker.serve.constants import (
            Framework,
            DEFAULT_SERIALIZERS_BY_FRAMEWORK,
        )

        assert Framework.PYTORCH in DEFAULT_SERIALIZERS_BY_FRAMEWORK
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_model_builder_imports_without_torch():
    """ModelBuilder must be importable for API-only use without torch installed."""
    result = _run_without_torch(
        """
        from sagemaker.serve import ModelBuilder

        assert ModelBuilder is not None
        print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_tensor_deserializer_names_the_extra():
    """Deserializing must construct a real tensor, so it still needs torch - and the
    error must name an extra the caller can actually install."""
    result = _run_without_torch(
        """
        from sagemaker.core.deserializers import TorchTensorDeserializer

        try:
            TorchTensorDeserializer()
            raise AssertionError("expected ImportError")
        except ImportError as e:
            assert "[torch]" in str(e)
            print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr


def test_triton_translator_names_the_extra():
    """Translating to torch.Tensor needs torch; the error must name the extra."""
    result = _run_without_torch(
        """
        from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator

        try:
            TorchTensorTranslator()
            raise AssertionError("expected ImportError")
        except ImportError as e:
            assert "sagemaker-serve[torch]" in str(e)
            print("OK")
        """
    )
    assert "OK" in result.stdout, result.stderr
