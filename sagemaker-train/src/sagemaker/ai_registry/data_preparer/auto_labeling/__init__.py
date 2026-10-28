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

Every name here is also re-exported from ``sagemaker.ai_registry``, which is the
stable import path for callers.
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.auto_labeling.shapes import (  # noqa: F401
    ClassificationLabelingOptions,
    Demos,
    EvaluationData,
    Grader,
    LabelingOptions,
    LlmJudgeLabelingOptions,
    NerLabelingOptions,
    OrdinalLabelingOptions,
    QaLabelingOptions,
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
    "LabelingSummary",
]
