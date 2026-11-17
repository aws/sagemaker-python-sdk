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
"""Auto-labeling support for ``DataPreparer.generate``.

A SyntheticDataGeneration job runs in auto-labeling mode when its request
carries ``GenerationConfig.DatasetOptions`` with a ``labeling.*`` ``Type``.

    * :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.shapes` — the
      typed per-task request options, passed as
      ``DatasetJobConfig(dataset_options=...)``.
    * :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.summary` — the
      reader for a finished job's ``labeling_summary.json``.
    * :mod:`~sagemaker.ai_registry.data_preparer.auto_labeling.operators` — the
      client-side operators of a labeling project, re-exported here by name.

Every name here is also re-exported from ``sagemaker.ai_registry``, which is the
stable import path for callers.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.shapes import (  # noqa: F401
    ClassificationLabelingOptions,
    Confidence,
    Demos,
    EvaluationData,
    Grader,
    JudgeModel,
    LabelingOptions,
    LlmJudgeLabelingOptions,
    NerLabelingOptions,
    Optimizer,
    OrdinalLabelingOptions,
    QaLabelingOptions,
    Review,
    Routing,
    Scoring,
    SummarizationLabelingOptions,
)
from sagemaker.ai_registry.data_preparer.auto_labeling.summary import (  # noqa: F401
    LabelingSummary,
)

__all__ = [
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


# The operators are re-exported by name, resolved on first access: importing them here
# eagerly would cycle (configs -> shapes -> this package -> operators -> request -> configs).
def _operators():
    import importlib

    return importlib.import_module("sagemaker.ai_registry.data_preparer.auto_labeling.operators")


def __getattr__(name):
    if not name.startswith("_") and name != "operators":
        operators = _operators()
        if name in operators.__all__:
            return getattr(operators, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_operators().__all__))
