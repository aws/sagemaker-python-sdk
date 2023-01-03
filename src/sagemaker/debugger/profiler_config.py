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
"""Configuration for collecting system and framework metrics in SageMaker training jobs."""
from __future__ import absolute_import

from typing import Optional, Union

from sagemaker.debugger.framework_profile import FrameworkProfile
from sagemaker.workflow.entities import PipelineVariable


class ProfilerConfig(object):
    """Configuration for collecting system and framework metrics of SageMaker training jobs.

    SageMaker Debugger collects system and framework profiling
    information of training jobs and identify performance bottlenecks.

    """

    def __init__(
        self,
        s3_output_path: Optional[Union[str, PipelineVariable]] = None,
        system_monitor_interval_millis: Optional[Union[int, PipelineVariable]] = None,
        framework_profile_params: Optional[FrameworkProfile] = None,
        disable_profiler: Optional[Union[str, PipelineVariable]] = False,
    ):
        """Initialize a ``ProfilerConfig`` instance.

        Pass the output of this class
        to the ``profiler_config`` parameter of the generic :class:`~sagemaker.estimator.Estimator`
        class and SageMaker Framework estimators.

        Args:
            s3_output_path (str or PipelineVariable): The location in Amazon S3 to store
                the output.
                The default Debugger output path for profiling data is created under the
                default output path of the :class:`~sagemaker.estimator.Estimator` class.
                For example,
                s3://sagemaker-<region>-<12digit_account_id>/<training-job-name>/profiler-output/.
            system_monitor_interval_millis (int or PipelineVariable): The time interval in
                milliseconds to collect system metrics. Available values are 100, 200, 500,
                1000 (1 second), 5000 (5 seconds), and 60000 (1 minute) milliseconds.
                The default is 500 milliseconds.
            framework_profile_params (:class:`~sagemaker.debugger.FrameworkProfile`):
                A parameter object for framework metrics profiling. Configure it using
                the :class:`~sagemaker.debugger.FrameworkProfile` class.
                To use the default framework profile parameters, pass ``FrameworkProfile()``.
                For more information about the default values,
                see :class:`~sagemaker.debugger.FrameworkProfile`.

        **Example**: The following example shows the basic ``profiler_config``
        parameter configuration, enabling system monitoring every 5000 milliseconds
        and framework profiling with default parameter values.

        .. code-block:: python

            from sagemaker.debugger import ProfilerConfig, FrameworkProfile

            profiler_config = ProfilerConfig(
                system_monitor_interval_millis = 5000
                framework_profile_params = FrameworkProfile()
            )

        """
        assert framework_profile_params is None or isinstance(
            framework_profile_params, FrameworkProfile
        ), "framework_profile_params must be of type FrameworkProfile if specified."

        self.s3_output_path = s3_output_path
        self.system_monitor_interval_millis = system_monitor_interval_millis
        self.framework_profile_params = framework_profile_params
        self.disable_profiler = disable_profiler

    def _to_request_dict(self):
        """Generate a request dictionary using the parameters provided when initializing the object.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_config_request = {}

        if (
            self.s3_output_path is not None
            and self.disable_profiler is not None
            and self.disable_profiler is False
        ):
            profiler_config_request["S3OutputPath"] = self.s3_output_path

        profiler_config_request["DisableProfiler"] = self.disable_profiler

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
        """Generate a request dictionary for updating the training job to disable profiler.

        Returns:
            dict: An portion of an API request as a dictionary.

        """
        profiler_config_request = {"DisableProfiler": True}
        return profiler_config_request
