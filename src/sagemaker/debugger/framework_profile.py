# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
"""Configuration for collecting framework metrics in SageMaker training jobs."""
from __future__ import absolute_import

from sagemaker.debugger.metrics_config import (
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    HerringProfilingConfig,
    HorovodProfilingConfig,
    PythonProfilingConfig,
)
from sagemaker.debugger.profiler_constants import (
    BASE_FOLDER_DEFAULT,
    CLOSE_FILE_INTERVAL_DEFAULT,
    FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
    MAX_FILE_SIZE_DEFAULT,
)
from sagemaker.debugger.utils import ErrorMessages

ALL_METRIC_CONFIGS = [
    DetailedProfilingConfig,
    DataloaderProfilingConfig,
    PythonProfilingConfig,
    HorovodProfilingConfig,
    HerringProfilingConfig,
]


class FrameworkProfile:
    """Configuration for the collection of framework metrics in the profiler.

    Validates user input and fills in default values wherever necessary.
    """

    def __init__(
        self,
        local_path=BASE_FOLDER_DEFAULT,
        file_max_size=MAX_FILE_SIZE_DEFAULT,
        file_close_interval=CLOSE_FILE_INTERVAL_DEFAULT,
        file_open_fail_threshold=FILE_OPEN_FAIL_THRESHOLD_DEFAULT,
        detailed_profiling_config=None,
        dataloader_profiling_config=None,
        python_profiling_config=None,
        horovod_profiling_config=None,
        herring_profiling_config=None,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
    ):
        """Set up the profiling configuration for framework metrics based on user input.

        There are three main options for the user to choose from.
        1. No custom metrics configs or step range or time range specified. Default profiling is
        done for each set of framework metrics.
        2. Custom metrics configs are specified. Do profiling for the metrics whose configs are
        specified and no profiling for the rest of the metrics.
        3. Custom step range or time range is specified. Profiling for all of the metrics will
        occur with the provided step/time range. Configs with additional parameters beyond
        step/time range will use defaults for those additional parameters.

        If custom metrics configs are specified in addition to step or time range being specified,
        then we ignore the step/time range and default to using custom metrics configs.

        Args:
            local_path (str): The path where profiler events have to be saved.
            file_max_size (int): Max size a trace file can be, before being rotated.
            file_close_interval (float): Interval in seconds from the last close, before being
                rotated.
            file_open_fail_threshold (int): Number of times to attempt to open a trace fail before
                marking the writer as unhealthy.
            detailed_profiling_config (DetailedProfilingConfig): The configuration for detailed
                profiling done by the framework.
            dataloader_profiling_config (DataloaderProfilingConfig): The configuration for metrics
                collected in the data loader.
            python_profiling_config (PythonProfilingConfig): The configuration for stats
                collected by the Python profiler (cProfile or Pyinstrument).
            horovod_profiling_config (HorovodProfilingConfig): The configuration for metrics
                collected by horovod when using horovod for distributed training.
            herring_profiling_config (HerringProfilingConfig): The configuration for metrics
                collected by herring when using herring for distributed training.
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
        """
        self.profiling_parameters = {}
        self._use_default_metrics_configs = False
        self._use_one_config_for_all_metrics = False
        self._use_custom_metrics_configs = False

        self._process_trace_file_parameters(
            local_path, file_max_size, file_close_interval, file_open_fail_threshold
        )
        use_custom_metrics_configs = self._process_metrics_configs(
            detailed_profiling_config,
            dataloader_profiling_config,
            python_profiling_config,
            horovod_profiling_config,
            herring_profiling_config,
        )

        use_one_config_for_all_metrics = (
            self._process_range_fields(start_step, num_steps, start_unix_time, duration)
            if not use_custom_metrics_configs
            else False
        )

        if not use_custom_metrics_configs and not use_one_config_for_all_metrics:
            self._create_default_metrics_configs()

    def _process_trace_file_parameters(
        self, local_path, file_max_size, file_close_interval, file_open_fail_threshold
    ):
        """Helper function to validate and set the provided trace file parameters.

        Args:
            local_path (str): The path where profiler events have to be saved.
            file_max_size (int): Max size a trace file can be, before being rotated.
            file_close_interval (float): Interval in seconds from the last close, before being
                rotated.
            file_open_fail_threshold (int): Number of times to attempt to open a trace fail before
                marking the writer as unhealthy.
        """
        assert isinstance(local_path, str), ErrorMessages.INVALID_LOCAL_PATH.value
        assert (
            isinstance(file_max_size, int) and file_max_size > 0
        ), ErrorMessages.INVALID_FILE_MAX_SIZE.value
        assert (
            isinstance(file_close_interval, (float, int)) and file_close_interval > 0
        ), ErrorMessages.INVALID_FILE_CLOSE_INTERVAL.value
        assert (
            isinstance(file_open_fail_threshold, int) and file_open_fail_threshold > 0
        ), ErrorMessages.INVALID_FILE_OPEN_FAIL_THRESHOLD.value

        self.profiling_parameters["LocalPath"] = local_path
        self.profiling_parameters["RotateMaxFileSizeInBytes"] = str(file_max_size)
        self.profiling_parameters["RotateFileCloseIntervalInSeconds"] = str(file_close_interval)
        self.profiling_parameters["FileOpenFailThreshold"] = str(file_open_fail_threshold)

    def _process_metrics_configs(self, *metrics_configs):
        """Helper function to validate and set the provided metrics_configs.

        In this case, the user specifies configs for the metrics they want profiled.
        Profiling does not occur for metrics if configs are not specified for them.

        Args:
            metrics_configs: The list of metrics configs specified by the user.
        Returns:
            bool: Whether custom metrics configs will be used for profiling.
        """
        metrics_configs = [config for config in metrics_configs if config is not None]
        if len(metrics_configs) == 0:
            return False

        for config in metrics_configs:
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _process_range_fields(self, start_step, num_steps, start_unix_time, duration):
        """Helper function to validate and set the provided range fields.

        Profiling will occur for all of the metrics using these fields as the specified
        range and default parameters for the rest of the config fields (if necessary).

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
        Returns:
            bool: Whether custom step or time range will be used for profiling.
        """
        if start_step is num_steps is start_unix_time is duration is None:
            return False

        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(
                start_step=start_step,
                num_steps=num_steps,
                start_unix_time=start_unix_time,
                duration=duration,
            )
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
        return True

    def _create_default_metrics_configs(self):
        """Helper function for creating the default configs for each set of metrics."""
        for config_class in ALL_METRIC_CONFIGS:
            config = config_class(profile_default_steps=True)
            config_name = config.name
            config_json = config.to_json_string()
            self.profiling_parameters[config_name] = config_json
