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
"""Constants and enums for DataPreparer.

JobCategory / JobStatus are defined here rather than re-exported from
sagemaker.core.shapes. The public service model's JobCategory enum does not list
the DataPreparer categories, so enums generated from it would drop them. Keep these
definitions until the service model includes every DataPreparer category, and keep
the import path (sagemaker.ai_registry.JobCategory / JobStatus) stable for callers.
"""

from __future__ import absolute_import

from enum import Enum

from sagemaker.core.config.config_schema import (
    _simple_path,
    SAGEMAKER,
    PYTHON_SDK,
    MODULES,
)

# Module-name constant under SageMaker.PythonSDK.Modules
DATA_PREPARER = "DataPreparer"
GLOBAL_DEFAULTS = "GlobalDefaults"
JOB = "Job"
SYNTHETIC_DATA_GENERATION = "SyntheticDataGeneration"
DATA_QUALITY_EVALUATION = "DataQualityEvaluation"

# Job-name constraints (Smithy JobName length constraint)
MAX_JOB_NAME_LENGTH = 63

# Config paths
DATA_PREPARER_BASE_PATH = _simple_path(SAGEMAKER, PYTHON_SDK, MODULES, DATA_PREPARER)
DATA_PREPARER_GLOBAL_PATH = _simple_path(DATA_PREPARER_BASE_PATH, GLOBAL_DEFAULTS)
DATA_PREPARER_JOB_PATH = _simple_path(DATA_PREPARER_BASE_PATH, JOB)


def data_preparer_job_path(category: "JobCategory") -> str:
    """Return the per-category config path prefix."""
    return _simple_path(DATA_PREPARER_JOB_PATH, category.value)


# ---------------------------------------------------------------------------
# Local enums (see the module docstring for why they are not generated).
# Keep the public import path (sagemaker.ai_registry.JobCategory) stable.
# ---------------------------------------------------------------------------
class JobCategory(str, Enum):
    """Job category for DataPreparer operations."""

    SYNTHETIC_DATA_GENERATION = "SyntheticDataGeneration"
    DATA_QUALITY_EVALUATION = "DataQualityEvaluation"
    DATA_TRANSFORMATION = "DataTransformation"


# Major version ceiling per category. SDK resolves the latest version
# within this major. Bumping requires an explicit SDK release.
#
# DataQualityEvaluation is at 3: an unpinned evaluate() resolves "3.0.0", the
# Operations shape, wherever the service shows it to the account, and "2.0.0"
# where it does not. Explicit schema_version="2.0.0" pins still resolve to v2.
#
# SyntheticDataGeneration is at 2 for the same reason: the service publishes a
# "2.0.0" schema that adds the plural ModelConfig.SampleGenerationModelConfigs
# list, SampleGenerationStrategy, and the auto-labeling DatasetOptions envelope.
# With a ceiling of 1 an unpinned generate() filtered "2.0.0" out and sent the
# v1 shape, so auto-labeling needed a manual schema_version="2.0.0" pin. The one
# field v2 drops is the singular ModelConfig.SampleGenerationModelConfig;
# _promote_singular_sample_generation_model (configs.py) rewrites it into the
# plural list so a v1-style single-model caller survives the bump. Explicit
# schema_version="1.0.0" pins still resolve to v1.
#
# DataTransformation only publishes a v1 schema, so it stays at 1.
SCHEMA_VERSION_CEILING = {
    JobCategory.SYNTHETIC_DATA_GENERATION: 2,
    JobCategory.DATA_QUALITY_EVALUATION: 3,
    JobCategory.DATA_TRANSFORMATION: 1,
}


class JobStatus(str, Enum):
    """Job status reported by the service."""

    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
