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
"""DataPreparer subpackage: data preparation jobs.

Covers SyntheticDataGeneration, DataQualityEvaluation, and DataTransformation.

Public surface:
    DataPreparer, DatasetJobConfig, JobCategory, JobStatus, and the full exception taxonomy.
    The auto_labeling subpackage's typed options and LabelingSummary.
    _schema_resolver is NOT re-exported (private to the subpackage).
"""

from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.preparer import DataPreparer  # noqa: F401
from sagemaker.ai_registry.data_preparer.configs import DatasetJobConfig  # noqa: F401
from sagemaker.ai_registry.data_preparer.constants import JobCategory, JobStatus  # noqa: F401
from sagemaker.ai_registry.data_preparer.auto_labeling import (  # noqa: F401
    ClassificationLabelingOptions,
    Confidence,
    Demos,
    EvaluationData,
    Grader,
    JudgeModel,
    LabelingOptions,
    LabelingSummary,
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
from sagemaker.ai_registry.data_preparer.exceptions import (  # noqa: F401
    DataPreparerConfigurationError,
    DataPreparerError,
    DataPreparerJobError,
    DataPreparerJobFailedError,
    DataPreparerJobStoppedError,
    DataPreparerJobTimeoutError,
    DataPreparerOperatorError,
    DataPreparerSchemaError,
    DataPreparerSchemaNotFoundError,
    DataPreparerSchemaValidationError,
)

__all__ = [
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
    "Confidence",
    "JudgeModel",
    "Optimizer",
    "Review",
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
