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
"""Unit tests for sagemaker.core.debugger.profiler_constants module."""
from __future__ import absolute_import

from sagemaker.core.debugger import profiler_constants


class TestProfilerConstants:
    """Test profiler constants module."""

    def test_base_folder_default(self):
        """Test BASE_FOLDER_DEFAULT constant."""
        assert profiler_constants.BASE_FOLDER_DEFAULT == "/opt/ml/output/profiler"

    def test_max_file_size_default(self):
        """Test MAX_FILE_SIZE_DEFAULT constant."""
        assert profiler_constants.MAX_FILE_SIZE_DEFAULT == 10485760  # 10MB

    def test_close_file_interval_default(self):
        """Test CLOSE_FILE_INTERVAL_DEFAULT constant."""
        assert profiler_constants.CLOSE_FILE_INTERVAL_DEFAULT == 60  # 60 seconds

    def test_file_open_fail_threshold_default(self):
        """Test FILE_OPEN_FAIL_THRESHOLD_DEFAULT constant."""
        assert profiler_constants.FILE_OPEN_FAIL_THRESHOLD_DEFAULT == 50

    def test_profiling_config_names(self):
        """Test profiling config name constants."""
        assert profiler_constants.DETAILED_PROFILING_CONFIG_NAME == "DetailedProfilingConfig"
        assert profiler_constants.DATALOADER_PROFILING_CONFIG_NAME == "DataloaderProfilingConfig"
        assert profiler_constants.PYTHON_PROFILING_CONFIG_NAME == "PythonProfilingConfig"
        assert profiler_constants.HOROVOD_PROFILING_CONFIG_NAME == "HorovodProfilingConfig"
        assert (
            profiler_constants.SMDATAPARALLEL_PROFILING_CONFIG_NAME
            == "SMDataParallelProfilingConfig"
        )

    def test_profiling_start_step_defaults(self):
        """Test profiling start step default constants."""
        assert profiler_constants.DETAILED_PROFILING_START_STEP_DEFAULT == 5
        assert profiler_constants.DATALOADER_PROFILING_START_STEP_DEFAULT == 7
        assert profiler_constants.PYTHON_PROFILING_START_STEP_DEFAULT == 9
        assert profiler_constants.HOROVOD_PROFILING_START_STEP_DEFAULT == 13
        assert profiler_constants.SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT == 15

    def test_profiling_num_steps_defaults(self):
        """Test profiling num steps default constants."""
        assert profiler_constants.PROFILING_NUM_STEPS_DEFAULT == 1
        assert profiler_constants.START_STEP_DEFAULT == 0
        assert profiler_constants.PYTHON_PROFILING_NUM_STEPS_DEFAULT == 3

    def test_cpu_profiling_duration(self):
        """Test CPU_PROFILING_DURATION constant."""
        assert profiler_constants.CPU_PROFILING_DURATION == 3600  # 1 hour

    def test_file_rotation_interval_default(self):
        """Test FILE_ROTATION_INTERVAL_DEFAULT constant."""
        assert profiler_constants.FILE_ROTATION_INTERVAL_DEFAULT == 600  # 600 seconds

    def test_detail_prof_processing_defaults(self):
        """Test detail profiler processing default constants."""
        assert profiler_constants.DETAIL_PROF_PROCESSING_DEFAULT_INSTANCE_TYPE == "ml.m5.4xlarge"
        assert profiler_constants.DETAIL_PROF_PROCESSING_DEFAULT_VOLUME_SIZE == 128

    def test_start_steps_are_unique(self):
        """Test that different profiling types have unique start steps."""
        start_steps = [
            profiler_constants.DETAILED_PROFILING_START_STEP_DEFAULT,
            profiler_constants.DATALOADER_PROFILING_START_STEP_DEFAULT,
            profiler_constants.PYTHON_PROFILING_START_STEP_DEFAULT,
            profiler_constants.HOROVOD_PROFILING_START_STEP_DEFAULT,
            profiler_constants.SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
        ]
        # All start steps should be unique to avoid conflicts
        assert len(start_steps) == len(set(start_steps))

    def test_start_steps_are_positive(self):
        """Test that all start step defaults are positive integers."""
        start_steps = [
            profiler_constants.DETAILED_PROFILING_START_STEP_DEFAULT,
            profiler_constants.DATALOADER_PROFILING_START_STEP_DEFAULT,
            profiler_constants.PYTHON_PROFILING_START_STEP_DEFAULT,
            profiler_constants.HOROVOD_PROFILING_START_STEP_DEFAULT,
            profiler_constants.SMDATAPARALLEL_PROFILING_START_STEP_DEFAULT,
            profiler_constants.START_STEP_DEFAULT,
        ]
        for step in start_steps:
            assert isinstance(step, int)
            assert step >= 0
