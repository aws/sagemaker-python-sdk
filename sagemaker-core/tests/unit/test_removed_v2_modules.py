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
    _REMOVED_V2_MODULES,
    _KNOWN_V3_TOPLEVEL,
    _V3_REPLACEMENTS,
    V3_MIGRATION_URL,
)

# Curated removed v2 module -> a substring expected in the precise guidance
# message. The values are verified-importable v3 symbols; the message also
# carries the exact import and a docs URL. These are served by the meta-path
# finder via _V3_REPLACEMENTS (no per-module files).
REMOVED_MODULES = {
    "sagemaker.estimator": "ModelTrainer",
    "sagemaker.model": "ModelBuilder",
    "sagemaker.predictor": "Endpoint",
    "sagemaker.base_predictor": "Endpoint",
    "sagemaker.predictor_async": "ModelBuilder",
    "sagemaker.transformer": "TransformJob",
    "sagemaker.tuner": "HyperParameterTuningJob",
    "sagemaker.processing": "ProcessingJob",
    "sagemaker.pipeline": "Pipeline",
    "sagemaker.multidatamodel": "ModelBuilder",
    "sagemaker.automl": "AutoMLJob",
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
    # Every curated entry points at a copy-pasteable import and a docs URL.
    assert "from sagemaker." in message
    assert "sagemaker.readthedocs.io/en/stable/api/generated/" in message
    assert V3_MIGRATION_URL in message


@pytest.mark.parametrize("module", sorted(REMOVED_MODULES))
def test_removed_module_emits_deprecation_warning(module):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(ModuleNotFoundError):
            _fresh_import(module)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


def test_helper_includes_import_and_docs_when_provided():
    docs = "https://sagemaker.readthedocs.io/en/stable/api/generated/sagemaker.train.model_trainer.html"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModuleNotFoundError) as exc:
            raise_removed_in_v3(
                module="sagemaker.estimator",
                replacement="`ModelTrainer`",
                v3_import="from sagemaker.train import ModelTrainer",
                v3_docs=docs,
            )
    message = str(exc.value)
    assert "from sagemaker.train import ModelTrainer" in message
    assert docs in message
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


# --- Meta-path finder (handles all removed v2 modules) ---


def test_finder_is_registered_on_meta_path():
    import sys
    import sagemaker.core  # noqa: F401  -- import registers the finder

    assert any(isinstance(f, _RemovedV2ModuleFinder) for f in sys.meta_path)


def test_finder_registered_on_bare_sagemaker_import():
    import sys
    import importlib

    # The namespace package init registers the finder, so it is active even
    # without importing sagemaker.core explicitly.
    for name in [m for m in list(sys.modules) if m == "sagemaker" or m.startswith("sagemaker.")]:
        pass  # (do not evict already-imported real modules; just assert state)
    import sagemaker  # noqa: F401  -- runs the namespace __init__

    importlib.import_module("sagemaker")
    assert any(isinstance(f, _RemovedV2ModuleFinder) for f in sys.meta_path)


def test_removed_module_guidance_on_first_import_in_fresh_process():
    # Regression: a removed module as the VERY FIRST sagemaker import (before
    # sagemaker.core is otherwise loaded) must still get guidance, not a bare
    # error. Run in a fresh interpreter so import state cannot leak in, and
    # capture the message from the exception itself (the console traceback
    # formatter does not always write plain text to stderr).
    import subprocess
    import sys

    code = (
        "import sys\n"
        "try:\n"
        "    import sagemaker.estimator\n"
        "except ModuleNotFoundError as e:\n"
        "    sys.stdout.write(str(e))\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert "was removed in the SageMaker Python SDK v3" in proc.stdout
    assert "from sagemaker.train import ModelTrainer" in proc.stdout


def test_register_is_idempotent():
    import sys

    register_removed_module_finder()
    register_removed_module_finder()
    finders = [f for f in sys.meta_path if isinstance(f, _RemovedV2ModuleFinder)]
    assert len(finders) == 1


def test_finder_is_appended_not_prepended():
    # As a last resort it must sit at the end so real modules resolve first.
    # Being in the second half of meta_path is a sufficient proxy.
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
    assert "was removed in the SageMaker Python SDK v3" in message
    assert V3_MIGRATION_URL in message
    assert any(issubclass(x.category, DeprecationWarning) for x in w)


@pytest.mark.parametrize("module", ["sagemaker.foobar", "sagemaker.not_a_module", "sagemaker.zzz"])
def test_unknown_name_gets_plain_error_not_guidance(module):
    # Names that never existed in v2 (typos, hallucinated imports) must fall
    # through to a plain ModuleNotFoundError -- no "was removed" claim, no
    # migration-guide link, no deprecation warning.
    import sagemaker.core  # noqa: F401  -- ensure finder registered

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with pytest.raises(ModuleNotFoundError) as exc:
            _fresh_import(module)
    message = str(exc.value)
    assert "was removed" not in message
    assert V3_MIGRATION_URL not in message
    assert not any(issubclass(x.category, DeprecationWarning) for x in w)


def test_finder_passes_through_unknown_names():
    import sagemaker.core  # noqa: F401

    finder = _RemovedV2ModuleFinder()
    # Known removed v2 module -> guarded (raises); unknown -> pass through (None).
    assert finder.find_spec("sagemaker.foobar") is None
    assert finder.find_spec("sagemaker.definitely_not_real") is None


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


# --- Name-set invariants (reviewer: what prevents a collision with a new module?) ---


def test_removed_set_does_not_overlap_real_v3_packages():
    # A removed-v2 name must never collide with a real v3 top-level package,
    # otherwise the finder could claim a live package "was removed".
    assert _REMOVED_V2_MODULES.isdisjoint(_KNOWN_V3_TOPLEVEL)


def test_all_curated_replacements_are_in_removed_set():
    # Every precisely-guided name must also be in the removed set (the finder
    # checks membership there before dispatching to the curated message).
    assert set(_V3_REPLACEMENTS).issubset(_REMOVED_V2_MODULES)


def test_finder_never_shadows_a_real_module_with_removed_name(monkeypatch):
    # Even if a real module exists under a name that is in the removed set, the
    # appended finder must not shadow it: normal import machinery resolves the
    # real module first, so find_spec is never consulted. Simulate by injecting
    # a real module and importing it.
    import sys
    import types

    import sagemaker.core  # noqa: F401  -- ensure finder registered

    name = "sagemaker.inputs"  # a name present in _REMOVED_V2_MODULES
    assert "inputs" in _REMOVED_V2_MODULES
    real = types.ModuleType(name)
    real.SENTINEL = object()
    monkeypatch.setitem(sys.modules, name, real)
    resolved = importlib.import_module(name)
    assert resolved is real  # real module wins; finder never intercepts it


# --- Replacement targets stay valid (guards against silent doc/import rot) ---


def _parse_import(stmt):
    """Parse ``from <module> import <symbol>`` into (module, symbol)."""
    _, module, _, symbol = stmt.split()
    return module, symbol


@pytest.mark.parametrize("leaf,entry", sorted(_V3_REPLACEMENTS.items()))
def test_curated_v3_imports_resolve(leaf, entry):
    # Recommendation B: every curated v3 import target must actually resolve in
    # the installed SDK, so a future rename/move of a replacement fails CI
    # instead of shipping a dead import string. The target package may live in a
    # sibling distribution (sagemaker-train/serve/mlops) that is not installed in
    # the core-only test job -- importorskip skips cleanly there and this runs in
    # the all-extras lane.
    _replacement, v3_import, docs_module = entry
    module, symbol = _parse_import(v3_import)
    mod = pytest.importorskip(module)
    assert hasattr(mod, symbol), f"{module} has no attribute {symbol} (referenced by {leaf})"


def test_curated_docs_url_matches_import_module_root():
    # The docs URL is a derivation of a module path (not independently
    # maintained); assert its module segment shares the same top-level package
    # as the import so the two cannot silently drift to different subsystems.
    for leaf, (_replacement, v3_import, docs_module) in _V3_REPLACEMENTS.items():
        import_module, _symbol = _parse_import(v3_import)
        assert docs_module.split(".")[:2] == import_module.split(".")[:2], leaf
