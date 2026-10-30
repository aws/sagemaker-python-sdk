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
    "LabelingSummary",
]


class TestSubpackageExists:
    def test_data_preparer_package_importable(self):
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer")
        assert mod is not None

    def test_constants_module_importable(self):
        mod = importlib.import_module("sagemaker.ai_registry.data_preparer.constants")
        assert mod is not None


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
