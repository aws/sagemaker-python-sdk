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
"""Unit tests for sagemaker.ai_registry.data_preparer.constants."""
from __future__ import absolute_import

from sagemaker.ai_registry.data_preparer.constants import (
    DATA_PREPARER,
    DATA_PREPARER_BASE_PATH,
    DATA_PREPARER_GLOBAL_PATH,
    DATA_PREPARER_JOB_PATH,
    DATA_QUALITY_EVALUATION,
    GLOBAL_DEFAULTS,
    JOB,
    MAX_JOB_NAME_LENGTH,
    SYNTHETIC_DATA_GENERATION,
    JobCategory,
    JobStatus,
    data_preparer_job_path,
)


class TestJobCategoryEnum:
    """Tests for the JobCategory enum members and wire values."""

    def test_member_names(self):
        assert hasattr(JobCategory, "SYNTHETIC_DATA_GENERATION")
        assert hasattr(JobCategory, "DATA_QUALITY_EVALUATION")

    def test_wire_values(self):
        assert JobCategory.SYNTHETIC_DATA_GENERATION.value == "SyntheticDataGeneration"
        assert JobCategory.DATA_QUALITY_EVALUATION.value == "DataQualityEvaluation"

    def test_is_str_enum(self):
        assert isinstance(JobCategory.SYNTHETIC_DATA_GENERATION, str)


class TestJobStatusEnum:
    """Tests for the JobStatus enum members and wire values."""

    def test_member_names(self):
        assert hasattr(JobStatus, "IN_PROGRESS")
        assert hasattr(JobStatus, "COMPLETED")
        assert hasattr(JobStatus, "FAILED")
        assert hasattr(JobStatus, "STOPPING")
        assert hasattr(JobStatus, "STOPPED")

    def test_wire_values(self):
        assert JobStatus.IN_PROGRESS.value == "InProgress"
        assert JobStatus.COMPLETED.value == "Completed"
        assert JobStatus.FAILED.value == "Failed"
        assert JobStatus.STOPPING.value == "Stopping"
        assert JobStatus.STOPPED.value == "Stopped"


class TestConfigPathConstants:
    """Tests for the config path constants and path-builder function."""

    def test_data_preparer_base_path(self):
        assert DATA_PREPARER_BASE_PATH == "SageMaker.PythonSDK.Modules.DataPreparer"

    def test_data_preparer_global_path(self):
        assert DATA_PREPARER_GLOBAL_PATH == "SageMaker.PythonSDK.Modules.DataPreparer.GlobalDefaults"

    def test_data_preparer_job_path(self):
        assert DATA_PREPARER_JOB_PATH == "SageMaker.PythonSDK.Modules.DataPreparer.Job"

    def test_data_preparer_job_path_function_sdg(self):
        assert (
            data_preparer_job_path(JobCategory.SYNTHETIC_DATA_GENERATION)
            == "SageMaker.PythonSDK.Modules.DataPreparer.Job.SyntheticDataGeneration"
        )

    def test_data_preparer_job_path_function_dq(self):
        assert (
            data_preparer_job_path(JobCategory.DATA_QUALITY_EVALUATION)
            == "SageMaker.PythonSDK.Modules.DataPreparer.Job.DataQualityEvaluation"
        )


class TestStringConstants:
    """Tests for module-level string constants."""

    def test_data_preparer(self):
        assert DATA_PREPARER == "DataPreparer"

    def test_global_defaults(self):
        assert GLOBAL_DEFAULTS == "GlobalDefaults"

    def test_job(self):
        assert JOB == "Job"

    def test_synthetic_data_generation(self):
        assert SYNTHETIC_DATA_GENERATION == "SyntheticDataGeneration"

    def test_data_quality_evaluation(self):
        assert DATA_QUALITY_EVALUATION == "DataQualityEvaluation"

    def test_max_job_name_length(self):
        assert MAX_JOB_NAME_LENGTH == 63
