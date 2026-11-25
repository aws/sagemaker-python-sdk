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
"""Unit tests for the data_preparer subpackage layout and import surface."""

from __future__ import absolute_import

import importlib
import inspect
import pkgutil
import re

import pytest

PUBLIC_NAMES = [
    "DataPreparer",
    "DatasetJobConfig",
    "JobCategory",
    "JobStatus",
    "LabelingOptions",
    "ClassificationLabelingOptions",
    "NerLabelingOptions",
    "QaLabelingOptions",
    "OrdinalLabelingOptions",
    "LlmJudgeLabelingOptions",
    "SummarizationLabelingOptions",
    "Demos",
    "EvaluationData",
    "Grader",
    "Routing",
    "Scoring",
    "LabelingSummary",
    "DataPreparerError",
    "DataPreparerConfigurationError",
    "DataPreparerSchemaError",
    "DataPreparerSchemaNotFoundError",
    "DataPreparerSchemaValidationError",
    "DataPreparerJobError",
    "DataPreparerJobFailedError",
    "DataPreparerJobStoppedError",
    "DataPreparerJobTimeoutError",
    "DataPreparerOperatorError",
]

# The subset that also lives under data_preparer.auto_labeling, which is a third
# supported import path alongside sagemaker.ai_registry and data_preparer.
LABELING_NAMES = [
    "LabelingOptions",
    "ClassificationLabelingOptions",
    "NerLabelingOptions",
    "QaLabelingOptions",
    "OrdinalLabelingOptions",
    "LlmJudgeLabelingOptions",
    "SummarizationLabelingOptions",
    "Demos",
    "EvaluationData",
    "Grader",
    "Routing",
    "Scoring",
    "Confidence",
    "JudgeModel",
    "Optimizer",
    "Review",
    "LabelingSummary",
]

_DATA_PREPARER = "sagemaker.ai_registry.data_preparer"
_OPERATORS = f"{_DATA_PREPARER}.auto_labeling.operators"

# One folder per primitive, layer second: a primitive owns its shapes and its operators.
PRIMITIVES = ["auto_labeling"]

# The operator subsystems of the labeling primitive, by the question each answers.
OPERATOR_PACKAGES = ["inspect", "seed", "spec", "job", "score", "review", "report"]

# Every module under auto_labeling/operators, by its path under the package.
OPERATOR_MODULES = [
    "_bedrock",
    "_digest",
    "_fields",
    "_rowrule",
    "_rows",
    "_run_log",
    "_s3",
    "inspect.inputs",
    "inspect.task",
    "inspect.taxonomy",
    "seed._bounds",
    "seed.bootstrap",
    "seed.cold_start",
    "seed.demos",
    "seed.folds",
    "seed.labels",
    "seed.review_export",
    "seed.split",
    "seed.split_state",
    "spec._floor",
    "spec.adopt",
    "spec.induce",
    "spec.rubric",
    "spec.select",
    "job._round_parts",
    "job._run_context",
    "job._scratch",
    "job.client",
    "job.dt_handoff",
    "job.hub_project",
    "job.register",
    "job.request",
    "job.round",
    "job.run_log",
    "score._metrics",
    "score.scorers",
    "review.decisions",
    "review.golden",
    "review.request",
    "report.render",
]


def _imports_of(module):
    """The dotted module paths a module imports, from its source (any indentation)."""
    src = inspect.getsource(module)
    found = set()
    for m in re.finditer(
        r"^\s*from\s+(sagemaker\.[\w.]+)\s+import\s+(\([^)]*\)|[^\n]+)", src, re.M
    ):
        base, names = m.group(1), m.group(2).strip("()")
        for part in names.split(","):
            name = part.split("#")[0].split(" as ")[0].strip()
            if name:
                found.add(f"{base}.{name}")
        found.add(base)
    for m in re.finditer(r"^\s*import\s+(sagemaker\.[\w.]+)", src, re.M):
        found.add(m.group(1))
    return found


class TestSubpackageExists:
    def test_data_preparer_package_importable(self):
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer")
        assert mod is not None

    def test_constants_module_importable(self):
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer.constants")
        assert mod is not None

    def test_operator_result_is_a_root_module(self):
        """OperatorResult is shared by every primitive, so it lives at the package root."""
        mod = importlib.import_module(f"{_DATA_PREPARER}._operator_result")
        assert hasattr(mod, "OperatorResult")

    @pytest.mark.parametrize("name", OPERATOR_MODULES)
    def test_operator_module_importable(self, name):
        mod = importlib.import_module(f"{_OPERATORS}.{name}")
        assert mod is not None

    def test_operator_modules_listed(self):
        """OPERATOR_MODULES names every module on disk, so a new one is not missed."""
        root = importlib.import_module(_OPERATORS)
        prefix = f"{_OPERATORS}."
        found = [
            info.name[len(prefix) :]
            for info in pkgutil.walk_packages(root.__path__, prefix)
            if not info.ispkg
        ]
        assert sorted(found) == sorted(OPERATOR_MODULES)

    def test_operator_packages_listed(self):
        root = importlib.import_module(_OPERATORS)
        prefix = f"{_OPERATORS}."
        found = [
            info.name[len(prefix) :]
            for info in pkgutil.walk_packages(root.__path__, prefix)
            if info.ispkg
        ]
        assert sorted(found) == sorted(OPERATOR_PACKAGES)


class TestLayoutRules:
    """One folder per primitive; sharing goes through the package root or not at all."""

    @pytest.mark.parametrize("name", OPERATOR_MODULES)
    def test_no_primitive_imports_another_primitive(self, name):
        mod = importlib.import_module(f"{_OPERATORS}.{name}")
        for imported in _imports_of(mod):
            if not imported.startswith(_DATA_PREPARER + "."):
                continue
            rest = imported[len(_DATA_PREPARER) + 1 :]
            head = rest.split(".", 1)[0]
            if head in PRIMITIVES:
                assert head == "auto_labeling", f"{name} imports primitive {head}: {imported}"
            else:
                # the package root: its public modules, or a private _*.py shared module
                assert head.startswith("_") or head in (
                    "exceptions",
                    "preparer",
                    "configs",
                    "constants",
                    "defaults",
                ), f"{name} imports {imported}, which is neither a root module nor its primitive"

    def test_no_common_folder(self):
        """Shared code is a private root module, promoted when a second primitive imports it."""
        root = importlib.import_module(_DATA_PREPARER)
        names = [info.name for info in pkgutil.walk_packages(root.__path__, _DATA_PREPARER + ".")]
        assert not any(n.endswith(".common") or ".common." in n for n in names)


class TestOperatorExports:
    """What a customer imports: listed in __all__, one owner per name, each function with an example."""

    @staticmethod
    def _check_export(package_mod, name, prefix):
        obj = getattr(package_mod, name)
        if inspect.isfunction(obj):
            assert obj.__module__.startswith(prefix), f"{name} is not {prefix}'s"
            assert "Example::" in (obj.__doc__ or ""), f"{name} has no Example::"
        elif inspect.isclass(obj):
            assert issubclass(
                obj, Exception
            ), f"{name}: a class export must be an exception to catch"
            assert obj.__module__.startswith(prefix), f"{name} is not {prefix}'s"
        else:
            assert isinstance(
                obj, (str, int, float, tuple, frozenset)
            ), f"{name}: a constant export is immutable"

    def test_each_subsystem_lists_its_exports(self):
        seen = {}
        for package in OPERATOR_PACKAGES:
            mod = importlib.import_module(f"{_OPERATORS}.{package}")
            assert mod.__all__, f"{package} exports nothing"
            for name in mod.__all__:
                assert name not in seen, f"{name} exported by both {seen[name]} and {package}"
                seen[name] = package
                self._check_export(mod, name, f"{_OPERATORS}.{package}.")

    def test_client_reads_are_exported_at_the_top(self):
        ops = importlib.import_module(_OPERATORS)
        assert ops.CLIENT_READ_EXPORTS
        for name in ops.CLIENT_READ_EXPORTS:
            self._check_export(ops, name, f"{_OPERATORS}._")

    def test_the_package_joins_the_subsystems(self):
        ops = importlib.import_module(_OPERATORS)
        joined = [
            n
            for p in OPERATOR_PACKAGES
            for n in importlib.import_module(f"{_OPERATORS}.{p}").__all__
        ]
        joined += list(ops.CLIENT_READ_EXPORTS)
        assert sorted(ops.__all__) == sorted(joined)
        assert len(set(ops.__all__)) == len(ops.__all__)
        operators = [n for n in ops.__all__ if inspect.isfunction(getattr(ops, n))]
        assert (
            40 <= len(operators) <= 110
        ), "an export list a customer can read, not a catalogue of 337"

    def test_the_primitive_re_exports_them_lazily(self):
        """``from ...data_preparer.auto_labeling import prepare_round`` works, without an import cycle."""
        al = importlib.import_module(f"{_DATA_PREPARER}.auto_labeling")
        ops = importlib.import_module(_OPERATORS)
        for name in ops.__all__:
            assert getattr(al, name) is getattr(ops, name)
        assert set(ops.__all__) <= set(dir(al))
        with pytest.raises(AttributeError):
            getattr(al, "not_an_operator")

    def test_api_reference_documents_every_function_export_and_only_exports(self):
        import os

        root = os.path.dirname(os.path.abspath(__file__))
        while not os.path.isdir(os.path.join(root, "docs")) and os.path.dirname(root) != root:
            root = os.path.dirname(root)
        path = os.path.join(root, "docs", "data_preparation", "api_reference.rst")
        text = open(path, encoding="utf-8").read()
        ops = importlib.import_module(_OPERATORS)
        documented = re.findall(
            r"^\.\. autofunction:: " + re.escape(_OPERATORS) + r"\.(\S+)$", text, re.M
        )
        functions = sorted(n for n in ops.__all__ if inspect.isfunction(getattr(ops, n)))
        assert sorted(d.rsplit(".", 1)[1] for d in documented) == functions
        operators_section = text[text.index("Operators\n---------") :]
        assert ":undoc-members:" not in operators_section


class TestPublicImportSurface:
    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_importable_from_data_preparer(self, name):
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer")
        assert hasattr(mod, name), f"{name} not found in sagemaker.ai_registry.data_preparer"

    @pytest.mark.parametrize("name", LABELING_NAMES)
    def test_labeling_names_importable_from_auto_labeling(self, name):
        """The auto_labeling subpackage is a supported import path in its own right."""
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer.auto_labeling")
        assert hasattr(mod, name), f"{name} not found in ...data_preparer.auto_labeling"

    @pytest.mark.parametrize("name", PUBLIC_NAMES)
    def test_importable_from_ai_registry(self, name):
        mod = importlib.import_module("sagemaker.ai_registry")
        assert hasattr(mod, name), f"{name} not found in sagemaker.ai_registry"


class TestPrivateNotExported:
    def test_schema_resolver_not_importable_from_ai_registry(self):
        """_schema_resolver should not be importable via from sagemaker.ai_registry import _schema_resolver."""
        try:
            from sagemaker.ai_registry import _schema_resolver  # noqa: F401

            pytest.fail("_schema_resolver should not be importable from sagemaker.ai_registry")
        except ImportError:
            pass  # Expected

    def test_schema_resolver_class_not_in_ai_registry(self):
        mod = importlib.import_module("sagemaker.ai_registry")
        assert not hasattr(mod, "_SchemaResolver")
