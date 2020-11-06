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
"""Configuration for collecting system and framework metrics in SageMaker training jobs."""
from __future__ import absolute_import

from sagemaker.debugger.framework_profile import FrameworkProfile


class ProfilerConfig(object):
    """Profiler allows customers to gather system and framework profiling information."""

    def __init__(
        self,
        s3_output_path=None,
        system_monitor_interval_millis=None,
        framework_profile_params=None,
    ):
        """Initialize a ``Profiler`` instance.

        Args:
            s3_output_path (str): The location in S3 to store the output. (default: ``None``).
            system_monitor_interval_millis (int): How often profiling system metrics are
                collected; Unit: Milliseconds. (default: ``None``).
            framework_profile_params (:class:`~sagemaker.debugger.FrameworkProfile`): A dictionary
                of parameters for the profiler framework metrics. (default: ``None``).
        """
        assert framework_profile_params is None or isinstance(
            framework_profile_params, FrameworkProfile
        ), "framework_profile_params must be of type FrameworkProfile if specified."

        self.s3_output_path = s3_output_path
        self.system_monitor_interval_millis = system_monitor_interval_millis
        self.framework_profile_params = framework_profile_params

    def _to_request_dict(self):
        """Generates a request dictionary using the parameters provided when initializing object.

        Returns:
            dict: An portion of an API request as a dictionary.
        """
        profiler_config_request = {}

        if self.s3_output_path is not None:
            profiler_config_request["S3OutputPath"] = self.s3_output_path

        if self.system_monitor_interval_millis is not None:
            profiler_config_request[
                "ProfilingIntervalInMilliseconds"
            ] = self.system_monitor_interval_millis

        if self.framework_profile_params is not None:
            profiler_config_request[
                "ProfilingParameters"
            ] = self.framework_profile_params.profiling_parameters

        return profiler_config_request

    @classmethod
    def _to_profiler_disabled_request_dict(cls):
        """Generates a request dictionary for updating training job to disable profiler.

        Returns:
            dict: An portion of an API request as a dictionary.
        """

        profiler_config_request = {"DisableProfiler": True}
        return profiler_config_request
