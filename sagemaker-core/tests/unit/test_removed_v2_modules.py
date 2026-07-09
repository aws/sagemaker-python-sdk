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
"""Tests that removed v2 modules raise actionable guidance in v3."""
from __future__ import absolute_import

import importlib
import warnings

import pytest

from sagemaker.core.deprecations import (
    raise_removed_in_v3,
    register_removed_module_finder,
    _RemovedV2ModuleFinder,
    V3_MIGRATION_URL,
)

# Removed v2 module -> a substring we expect in the guidance message.
REMOVED_MODULES = {
    "sagemaker.estimator": "ModelTrainer",
    "sagemaker.model": "ModelBuilder",
    "sagemaker.predictor": "Endpoint",
    "sagemaker.base_predictor": "Endpoint",
    "sagemaker.predictor_async": "ModelBuilder",
    "sagemaker.transformer": "TransformJob",
    "sagemaker.tuner": "hyperparameter tuning",
    "sagemaker.processing": "DataProcessor",
    "sagemaker.pipeline": "Pipeline",
    "sagemaker.multidatamodel": "ModelBuilder",
    "sagemaker.automl": "AutoML",
    "sagemaker.algorithm": "ModelTrainer",
}


def _fresh_import(module):
    """Import a module fresh, bypassing any cached (failed) import state."""
    import sys

    sys.modules.pop(module, None)
    return importlib.import_module(module)


@pytest.mark.parametrize("module,expected", sorted(REMOVED_MODULES.items()))
def test_removed_module_raises_actionable_error(module, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as exc:
            _fresh_import(module)
    message = str(exc.value)
    assert module in message
    assert "removed in the SageMaker Python SDK v3" in message
    assert expected in message
    assert V3_MIGRATION_URL in message


@pytest.mark.parametrize("module", sorted(REMOVED_MODULES))
def test_removed_module_emits_deprecation_warning(module):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(ModuleNotFoundError):
            _fresh_import(module)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_helper_includes_import_when_provided():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as exc:
            raise_removed_in_v3(
                module="sagemaker.estimator",
                replacement="`ModelTrainer` in the sagemaker-train package",
                v3_import="from sagemaker.train import ModelTrainer",
            )
    message = str(exc.value)
    assert "from sagemaker.train import ModelTrainer" in message
    # ModuleNotFoundError carries the module name for tooling.
    assert exc.value.name == "sagemaker.estimator"


def test_helper_without_replacement_still_actionable():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as exc:
            raise_removed_in_v3(module="sagemaker.something")
    message = str(exc.value)
    assert "sagemaker.something" in message
    assert V3_MIGRATION_URL in message


# --- Fallback finder (catch-all for non-tombstoned removed modules) ---


def test_finder_is_registered_on_meta_path():
    import sys
    import sagemaker.core  # noqa: F401  -- import registers the finder

    assert any(isinstance(f, _RemovedV2ModuleFinder) for f in sys.meta_path)


def test_register_is_idempotent():
    import sys

    register_removed_module_finder()
    register_removed_module_finder()
    finders = [f for f in sys.meta_path if isinstance(f, _RemovedV2ModuleFinder)]
    assert len(finders) == 1


def test_finder_is_appended_not_prepended():
    # As a last resort it must sit at the end so real/tombstone modules resolve
    # first. Being in the second half of meta_path is a sufficient proxy.
    import sys

    register_removed_module_finder()
    idx = next(i for i, f in enumerate(sys.meta_path) if isinstance(f, _RemovedV2ModuleFinder))
    assert idx >= len(sys.meta_path) // 2


@pytest.mark.parametrize("module", ["sagemaker.clarify", "sagemaker.session", "sagemaker.network"])
def test_uncovered_module_falls_back_to_guidance(module):
    import sagemaker.core  # noqa: F401  -- ensure finder registered

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(ModuleNotFoundError) as exc:
            _fresh_import(module)
    message = str(exc.value)
    assert module in message
    assert "not available in the SageMaker Python SDK v3" in message
    assert V3_MIGRATION_URL in message
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.parametrize(
    "module",
    [
        "sagemaker.core",
        "sagemaker.train",
        "sagemaker.serve",
        "sagemaker.mlops",
        "sagemaker.lineage",
    ],
)
def test_finder_does_not_shadow_real_v3_packages(module):
    import sagemaker.core  # noqa: F401  -- ensure finder registered

    # The fallback finder must not intercept real v3 packages: find_spec must
    # return None (pass-through), regardless of whether the package happens to
    # be installed in the current environment. (The sagemaker-core unit-test job
    # installs core only, so sagemaker.train/serve/mlops are not importable
    # there -- asserting find_spec pass-through avoids that false dependency.)
    finder = _RemovedV2ModuleFinder()
    assert finder.find_spec(module) is None


def test_finder_ignores_non_sagemaker_and_nested():
    import sagemaker.core  # noqa: F401

    finder = _RemovedV2ModuleFinder()
    # Not a sagemaker module -> pass through (None).
    assert finder.find_spec("numpy") is None
    # Nested path under a (removed) top-level -> only the top-level is guarded.
    assert finder.find_spec("sagemaker.workflow.steps") is None
    # Real v3 top-level -> never guarded.
    assert finder.find_spec("sagemaker.train") is None
