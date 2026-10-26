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
"""sagemaker-serve must not force test tooling or serving backends on every user (issue #6502).

The base dependencies used to include pytest plus torch, onnx, onnxruntime,
tritonclient[http] and mlflow. The top-level ``sagemaker`` package depends on
sagemaker-serve, so that download cost reached users who never import
``sagemaker.serve``. The backends are now extras, which only works while nothing
on the import path needs them eagerly.
"""

from __future__ import absolute_import

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None

from sagemaker.serve.constants import (
    DEFAULT_SERIALIZERS_BY_FRAMEWORK,
    Framework,
    _LazySerializerRegistry,
)

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
ROOT_PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"
OPTIONAL_BACKENDS = {"torch", "onnx", "onnxruntime", "tritonclient", "mlflow"}
# Modules that must not be imported just by importing the package.
MUST_NOT_IMPORT = sorted(OPTIONAL_BACKENDS | {"pytest"})


def _requirement_name(requirement):
    """'tritonclient[http]>=1.0 ; python_version>"3.9"' -> 'tritonclient'."""
    name = requirement.split(";")[0].split("[")[0].strip()
    for separator in ("==", ">=", "<=", "!=", "~=", ">", "<", "="):
        name = name.split(separator)[0]
    return name.strip().lower()


@pytest.fixture(scope="module")
def project():
    if tomllib is None:
        pytest.skip("tomllib requires Python 3.11+")
    with open(PYPROJECT, "rb") as handle:
        return tomllib.load(handle)["project"]


class TestDeclaredDependencies:
    def test_pytest_is_not_a_runtime_dependency(self, project):
        names = {_requirement_name(r) for r in project["dependencies"]}

        assert "pytest" not in names

    def test_serving_backends_are_not_runtime_dependencies(self, project):
        names = {_requirement_name(r) for r in project["dependencies"]}

        assert not names & OPTIONAL_BACKENDS

    @pytest.mark.parametrize("extra", ["torch", "onnx", "triton", "mlflow", "all"])
    def test_backend_extras_are_declared(self, project, extra):
        assert extra in project["optional-dependencies"]

    def test_test_extra_still_installs_the_backends(self, project):
        """CI exercises the backend code paths, so the test extra must keep them."""
        names = {_requirement_name(r) for r in project["optional-dependencies"]["test"]}

        assert OPTIONAL_BACKENDS <= names

    @pytest.mark.parametrize("extra", ["torch", "onnx", "triton", "mlflow"])
    def test_top_level_package_passes_the_backend_extras_through(self, extra):
        """``pip install sagemaker[torch]`` must work, not just ``sagemaker-serve[torch]``."""
        if tomllib is None:
            pytest.skip("tomllib requires Python 3.11+")
        if not ROOT_PYPROJECT.exists():
            pytest.skip("top-level pyproject.toml is not available outside the repo")

        with open(ROOT_PYPROJECT, "rb") as handle:
            root = tomllib.load(handle)["project"]

        assert root["name"] == "sagemaker"
        assert root["optional-dependencies"][extra] == [f"sagemaker-serve[{extra}]"]


class TestNoEagerBackendImport:
    def test_importing_sagemaker_serve_needs_no_optional_backend(self):
        """Run in a subprocess so an already-imported backend cannot mask the defect."""
        script = textwrap.dedent(f"""
            import sys

            blocked = {MUST_NOT_IMPORT!r}

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] in blocked:
                        raise ImportError(name + " must not be imported eagerly")
                    return None

            sys.meta_path.insert(0, Blocker())

            import sagemaker.serve
            from sagemaker.serve import ModelBuilder
            import sagemaker.serve.model_builder_utils
            import sagemaker.serve.builder.schema_builder
            import sagemaker.serve.marshalling.triton_translator
            import sagemaker.serve.mode.in_process_mode

            leaked = sorted(m for m in sys.modules if m.split(".")[0] in blocked)
            assert not leaked, leaked
            """)

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr


class TestPyTorchDefaultsWithoutTorch:
    def test_pytorch_default_pair_is_selectable_without_torch(self):
        """Deploying a prebuilt PyTorch container needs no local torch: the framework is
        detected from the image URI, so picking its default pair must not require the
        backend. Subprocess so blocking torch cannot leak into the session."""
        script = textwrap.dedent("""
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "torch":
                        raise ImportError("torch is blocked for this test")
                    return None

            sys.meta_path.insert(0, Blocker())

            from sagemaker.serve.constants import DEFAULT_SERIALIZERS_BY_FRAMEWORK, Framework

            serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.PYTORCH]

            assert type(serializer).__name__ == "TorchTensorSerializer"
            assert serializer.CONTENT_TYPE == "tensor/pt"
            assert "torch" not in sys.modules
            """)

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr

    def test_serializer_rejects_non_tensors_without_torch(self):
        """The no-torch path must stay a type error, not an ImportError."""
        script = textwrap.dedent("""
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "torch":
                        raise ImportError("torch is blocked for this test")
                    return None

            sys.meta_path.insert(0, Blocker())

            from sagemaker.core.serializers import TorchTensorSerializer

            try:
                TorchTensorSerializer().serialize([1.0, 2.0, 3.0])
            except ValueError as e:
                assert "not a torch.Tensor" in str(e), str(e)
            else:
                raise AssertionError("expected ValueError")
            """)

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr


class TestMissingBackendErrors:
    def test_triton_translator_names_the_extra(self):
        """A bare ``import torch`` here surfaced a raw ModuleNotFoundError that left the
        caller no idea which extra to install. Subprocess so blocking torch is contained."""
        script = textwrap.dedent("""
            import sys

            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name.split(".")[0] == "torch":
                        raise ImportError("torch is blocked for this test")
                    return None

            sys.meta_path.insert(0, Blocker())

            from sagemaker.serve.marshalling.triton_translator import TorchTensorTranslator

            try:
                TorchTensorTranslator()
            except ImportError as e:
                assert "sagemaker-serve[torch]" in str(e), str(e)
            else:
                raise AssertionError("expected ImportError")
            """)

        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )

        assert result.returncode == 0, result.stderr


class TestLazySerializerRegistry:
    def test_pairs_are_not_built_until_requested(self):
        built = []

        class Probe:
            def __init__(self):
                built.append(type(self).__name__)

        registry = _LazySerializerRegistry({Framework.PYTORCH: (Probe, Probe)})

        assert built == []
        registry[Framework.PYTORCH]
        assert built == ["Probe", "Probe"]

    def test_instances_are_cached(self):
        class Probe:
            pass

        registry = _LazySerializerRegistry({Framework.PYTORCH: (Probe, Probe)})

        assert registry[Framework.PYTORCH] is registry[Framework.PYTORCH]

    def test_registry_still_behaves_like_the_mapping_it_replaced(self):
        assert Framework.PYTORCH in DEFAULT_SERIALIZERS_BY_FRAMEWORK
        assert Framework.XGBOOST in DEFAULT_SERIALIZERS_BY_FRAMEWORK
        assert len(DEFAULT_SERIALIZERS_BY_FRAMEWORK) == 12

        serializer, deserializer = DEFAULT_SERIALIZERS_BY_FRAMEWORK[Framework.SKLEARN]

        assert serializer.CONTENT_TYPE == "application/x-npy"
        assert deserializer is not None

    def test_membership_does_not_build_the_pair(self):
        """``Mapping.__contains__`` answers via ``__getitem__``, which would both
        defeat the laziness and raise a backend's ImportError out of an ``in`` test."""
        built = []

        class Boom:
            def __init__(self):
                built.append("built")
                raise ImportError("backend missing")

        registry = _LazySerializerRegistry({Framework.PYTORCH: (Boom, Boom)})

        assert Framework.PYTORCH in registry
        assert Framework.SKLEARN not in registry
        assert built == []
