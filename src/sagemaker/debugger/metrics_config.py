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
"""The various types of metrics configs that can be specified in FrameworkProfile"""
from __future__ import absolute_import

from sagemaker.debugger.profiler_constants import (
    DATALOADER_PROFILING_CONFIG_NAME,
    DATALOADER_PROFILING_START_STEP_DEFAULT,
    DETAILED_PROFILING_CONFIG_NAME,
    DETAILED_PROFILING_START_STEP_DEFAULT,
    HERRING_PROFILING_CONFIG_NAME,
    HERRING_PROFILING_START_STEP_DEFAULT,
    HOROVOD_PROFILING_CONFIG_NAME,
    HOROVOD_PROFILING_START_STEP_DEFAULT,
    PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_CONFIG_NAME,
    PYTHON_PROFILING_NUM_STEPS_DEFAULT,
    PYTHON_PROFILING_START_STEP_DEFAULT,
    START_STEP_DEFAULT,
)
from sagemaker.debugger.utils import (
    convert_json_config_to_string,
    cProfileTimer,
    is_valid_regex,
    is_valid_unix_time,
    ErrorMessages,
    PythonProfiler,
)


class StepRange:
    """Configuration for what range of steps to profile."""

    def __init__(self, start_step, num_steps):
        """Set the start step and num steps.

        If the start step is not specified, profile starting
        at step 0. If num steps is not specified, profile for 1 step.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
        """
        if start_step is None:
            start_step = START_STEP_DEFAULT
        elif num_steps is None:
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        self.start_step = start_step
        self.num_steps = num_steps

    def to_json(self):
        """Convert the step range into a dictionary.

        Returns:
            dict: The step range as a dictionary.
        """
        return {"StartStep": self.start_step, "NumSteps": self.num_steps}


class TimeRange:
    """Configuration for what range of time to profile."""

    def __init__(self, start_unix_time, duration):
        """Set the start unix time and duration.

        If the start unix time is not specified, profile starting at step 0.
        If the duration is not specified, profile for 1 step.

        Args:
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
        """
        self.start_unix_time = start_unix_time
        self.duration = duration

    def to_json(self):
        """Convert the time range into a dictionary.

        Returns:
            dict: The time range as a dictionary.
        """
        time_range_json = {}
        if self.start_unix_time is not None:
            time_range_json["StartTimeInSecSinceEpoch"] = self.start_unix_time
        if self.duration is not None:
            time_range_json["Duration"] = self.duration
        return time_range_json


class MetricsConfigBase:
    """The base class for a metrics config.

    It determines what step or time range will be profiled.
    Validates the provided fields and that both step and time fields are not specified.
    """

    def __init__(self, name, start_step, num_steps, start_unix_time, duration):
        """Validate the provided range fields and set the range to be profiled accordingly.

        Args:
            name (str): The name of the metrics config.
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
        """
        self.name = name

        assert (
            start_step is None or isinstance(start_step, int) and start_step >= 0
        ), ErrorMessages.INVALID_START_STEP.value
        assert (
            num_steps is None or isinstance(num_steps, int) and num_steps > 0
        ), ErrorMessages.INVALID_NUM_STEPS.value
        assert (
            start_unix_time is None
            or isinstance(start_unix_time, int)
            and is_valid_unix_time(start_unix_time)
        ), ErrorMessages.INVALID_START_UNIX_TIME.value
        assert (
            duration is None or isinstance(duration, (float, int)) and duration > 0
        ), ErrorMessages.INVALID_DURATION.value

        has_step_range = start_step is not None or num_steps is not None
        has_time_range = start_unix_time is not None or duration is not None
        assert not (
            has_step_range and has_time_range
        ), ErrorMessages.FOUND_BOTH_STEP_AND_TIME_FIELDS.value

        self.range = (
            StepRange(start_step, num_steps)
            if has_step_range
            else TimeRange(start_unix_time, duration)
        )

    def _to_json(self):
        """Convert this metrics config to a dictionary.

        It is done by converting the range object into a dictionary.

        Returns:
            dict: This metrics config as a dictionary.
        """
        return self.range.to_json()

    def to_json_string(self):
        """Converts this metrics config to dictionary formatted as a string.

        Calling eval on the return value is the same as calling _to_json directly.

        Returns:
            str: This metrics config as a dictionary formatted as a string.
        """
        return convert_json_config_to_string(self._to_json())


class DetailedProfilingConfig(MetricsConfigBase):
    """The configuration for detailed profiling done by the framework.

    By default, profile step 5 of training.
    """

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """If profile_default_steps is set to True or none of the range fields are specified,
        use the default config for detailed profiling. Otherwise, profile according to the
        specified range fields.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
            profile_default_steps (bool): Whether the default config should be used.
        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = DETAILED_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            DETAILED_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )


class DataloaderProfilingConfig(MetricsConfigBase):
    """The configuration for metrics collected in the data loader.

    By default, profile step 7 of training.
    """

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
        metrics_regex=".*",
    ):
        """If profile_default_steps is set to True or none of the range fields are specified,
        use the default config for dataloader profiling. Otherwise, profile according to the
        specified range fields.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
            profile_default_steps (bool): Whether the default config should be used.
            metrics_regex (str): The regex used for collecting metrics. Metrics will be collected
                for data loader events that this this regex matches. By default, collect metrics
                for all events.
        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = DATALOADER_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            DATALOADER_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )

        assert is_valid_regex(metrics_regex), ErrorMessages.INVALID_METRICS_REGEX.value
        self.metrics_regex = metrics_regex

    def _to_json(self):
        """Convert the dataloader profiling config to a dictionary.

        Build off the base metrics config dictionary to add the metrics regex.

        Returns:
            dict: The dataloader profiling config as a dictionary.
        """
        dataloader_profiling_config = super()._to_json()
        dataloader_profiling_config["MetricsRegex"] = self.metrics_regex
        return dataloader_profiling_config


class PythonProfilingConfig(MetricsConfigBase):
    """The configuration for stats collected by the Python profiler (cProfile or Pyinstrument).

    By default, profile steps 9, 10 and 11 of training using cProfile and collecting metrics
    based on total time, cpy time and off cpu time for these three steps respectively.
    """

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
        python_profiler=PythonProfiler.CPROFILE,
        cprofile_timer=cProfileTimer.TOTAL_TIME,
    ):
        """If profile_default_steps is set to True or none of the range fields are specified,
        use the default config for Python profiling. Otherwise, profile according to the
        specified range fields.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
            profile_default_steps (bool): Whether the default config should be used.
            python_profiler (PythonProfiler): The Python profiler to be used for collecting
                python profiling stats. By default, use cProfile.
            cprofile_timer (cProfileTimer): The timer to be used by cProfile when collecting
                python profiling stats. By default, use total time. If Pyinstrument is used as
                the Python profiler, this field is ignored.
        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = PYTHON_PROFILING_START_STEP_DEFAULT
            num_steps = PYTHON_PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            PYTHON_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )

        assert isinstance(
            python_profiler, PythonProfiler
        ), ErrorMessages.INVALID_PYTHON_PROFILER.value
        assert isinstance(cprofile_timer, cProfileTimer), ErrorMessages.INVALID_CPROFILE_TIMER.value

        self.python_profiler = python_profiler

        # The cprofile timer can only be used when the python profiler is cProfile.
        if python_profiler == PythonProfiler.PYINSTRUMENT:
            self.cprofile_timer = None
        else:
            self.cprofile_timer = cprofile_timer

    def _to_json(self):
        """Convert the Python profiling config to a dictionary.

        Build off the base metrics config dictionary to add the Python profiler and cProfile timer.

        Returns:
            dict: The python profiling config as a dictionary.
        """
        python_profiling_config = super()._to_json()
        python_profiling_config["ProfilerName"] = self.python_profiler.value
        if self.cprofile_timer is not None:
            python_profiling_config["cProfileTimer"] = self.cprofile_timer.value
        return python_profiling_config


class HorovodProfilingConfig(MetricsConfigBase):
    """Configuration for metrics collected by horovod when using horovod for distributed training.

    By default, profile step 13 of training.
    """

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """If profile_default_steps is set to True or none of the range fields are specified,
        use the default config for horovod profiling. Otherwise, profile according to the
        specified range fields.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
            profile_default_steps (bool): Whether the default config should be used.
        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = HOROVOD_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            HOROVOD_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )


class HerringProfilingConfig(MetricsConfigBase):
    """Configuration for metrics collected by herring when using herring for distributed training.

    By default, profile step 15 of training.
    """

    def __init__(
        self,
        start_step=None,
        num_steps=None,
        start_unix_time=None,
        duration=None,
        profile_default_steps=False,
    ):
        """If profile_default_steps is set to True or none of the range fields are specified,
        use the default config for herring profiling. Otherwise, profile according to the
        specified range fields.

        Args:
            start_step (int): The step at which to start profiling.
            num_steps (int): The number of steps to profile.
            start_unix_time (int): The UNIX time at which to start profiling.
            duration (float): The duration in seconds to profile for.
            profile_default_steps (bool): Whether the default config should be used.
        """
        assert isinstance(
            profile_default_steps, bool
        ), ErrorMessages.INVALID_PROFILE_DEFAULT_STEPS.value
        if profile_default_steps or start_step is num_steps is start_unix_time is duration is None:
            start_step = HERRING_PROFILING_START_STEP_DEFAULT
            num_steps = PROFILING_NUM_STEPS_DEFAULT

        super().__init__(
            HERRING_PROFILING_CONFIG_NAME, start_step, num_steps, start_unix_time, duration
        )
