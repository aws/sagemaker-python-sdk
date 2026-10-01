"""Every module-level third-party import must have a declared distribution.

Regression guard for #6152. Switching ``mlflow`` to ``mlflow-skinny`` silently dropped
``docker``: full mlflow requires ``docker>=4.0.0,<8`` while mlflow-skinny requires
nothing of the sort. ``docker`` is imported at module scope by
``sagemaker.serve.mode.local_container_mode``, which ``sagemaker.serve.__init__``
imports eagerly, so losing it breaks ``import sagemaker.serve`` outright. Nothing went
red until a CI runner happened not to have ``docker`` already installed, which is
exactly the kind of gap a declared-dependency check closes.

The check is deliberately limited to *module-level* imports. Imports nested inside a
function are opt-in code paths (matplotlib for plotting, sklearn for framework
detection) and are allowed to rely on an undeclared package.
"""

from __future__ import absolute_import

import ast
import os
import re
import sys
from pathlib import Path

import pytest

SERVE_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SERVE_ROOT.parent
SERVE_SRC = SERVE_ROOT / "src"

# sagemaker-serve depends on sagemaker-core and sagemaker-train, so anything those two
# declare is also importable from serve.
PYPROJECTS = (
    SERVE_ROOT / "pyproject.toml",
    REPO_ROOT / "sagemaker-core" / "pyproject.toml",
    REPO_ROOT / "sagemaker-train" / "pyproject.toml",
)

# Import name -> distribution name, for the cases where they differ.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
}

# Modules provided by the runtime environment rather than by our wheels.
EXEMPT_IMPORTS = {
    # Injected by the Triton Python backend inside the serving container; it is not
    # importable from the SDK and must never be a declared dependency.
    "triton_python_backend_utils",
}

FIRST_PARTY = {"sagemaker"}

# Matches the top-level `dependencies = [...]` array. The optional-dependency arrays are
# keyed by extra name (`full`, `test`, `dev`), so they cannot match this pattern.
_DEPENDENCIES_ARRAY = re.compile(r"^dependencies = \[(.*?)^\]", re.MULTILINE | re.DOTALL)


def _normalize(name):
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions():
    """Return the normalized distribution names declared across the three packages."""
    declared = set()
    for pyproject in PYPROJECTS:
        match = _DEPENDENCIES_ARRAY.search(pyproject.read_text(encoding="utf-8"))
        assert match, "no top-level dependencies array found in {}".format(pyproject)
        for line in match.group(1).splitlines():
            line = line.split("#", 1)[0].strip().rstrip(",").strip()
            if not line:
                continue
            requirement = ast.literal_eval(line)
            # Strip extras, version specifiers and environment markers.
            name = re.split(r"[\[<>=!~;\s]", requirement, maxsplit=1)[0]
            declared.add(_normalize(name))
    return declared


def _module_level_imports():
    """Map each module-level third-party import to the files that import it."""
    imports = {}
    for dirpath, _, filenames in os.walk(SERVE_SRC):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = Path(dirpath) / filename
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in tree.body:  # module level only, by construction
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    if top in sys.stdlib_module_names or top in FIRST_PARTY:
                        continue
                    if top in EXEMPT_IMPORTS:
                        continue
                    imports.setdefault(top, set()).add(str(path.relative_to(REPO_ROOT)))
    return imports


def test_module_level_imports_are_declared_dependencies():
    """Fail if sagemaker-serve imports a package none of the three packages declare."""
    declared = _declared_distributions()
    undeclared = {}
    for module, files in _module_level_imports().items():
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        if _normalize(distribution) not in declared:
            undeclared[module] = sorted(files)

    assert not undeclared, "module-level imports with no declared distribution: " + "; ".join(
        "{} (imported by {})".format(module, ", ".join(files))
        for module, files in sorted(undeclared.items())
    )


def test_docker_is_declared():
    """Pin the specific regression from #6152 so it cannot silently come back.

    ``docker`` has to be a hard dependency, not an extra: the import sits at module
    scope on the eager path out of ``sagemaker.serve.__init__``.
    """
    assert "docker" in _declared_distributions()


@pytest.mark.parametrize(
    "module",
    ["docker", "numpy", "pandas", "yaml"],
)
def test_sanity_known_module_level_imports_are_detected(module):
    """Guard the detector itself: these are known module-level imports in serve."""
    assert module in _module_level_imports()
