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
"""The step definitions for EMR Serverless workflow."""
from __future__ import absolute_import

from typing import Any, Dict, List, Union, Optional

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import Properties
from sagemaker.mlops.workflow.retry import StepRetryPolicy
from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import ConfigurableRetryStep, Step, StepTypeEnum, CacheConfig


class EMRServerlessJobConfig:
    """Config for EMR Serverless job."""

    def __init__(
        self,
        job_driver: Dict,
        execution_role_arn: str,
        configuration_overrides: Optional[Dict] = None,
        execution_timeout_minutes: Optional[int] = None,
        name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):  # pylint: disable=too-many-positional-arguments
        """Create a definition for EMR Serverless job configuration.

        Args:
            job_driver (Dict): The job driver for the job run.
            execution_role_arn (str): The execution role ARN for the job run.
            configuration_overrides (Dict, optional): Configuration overrides for the job run.
            execution_timeout_minutes (int, optional): The maximum duration for the job run.
            name (str, optional): The optional job run name.
            tags (Dict[str, str], optional): The tags assigned to the job run.
        """
        self.job_driver = job_driver
        self.execution_role_arn = execution_role_arn
        self.configuration_overrides = configuration_overrides
        self.execution_timeout_minutes = execution_timeout_minutes
        self.name = name
        self.tags = tags

    def to_request(self, application_id: Optional[str] = None) -> RequestType:
        """Convert EMRServerlessJobConfig object to request dict."""
        config = {"executionRoleArn": self.execution_role_arn, "jobDriver": self.job_driver}
        if application_id is not None:
            config["applicationId"] = application_id
        if self.configuration_overrides is not None:
            config["configurationOverrides"] = self.configuration_overrides
        if self.execution_timeout_minutes is not None:
            config["executionTimeoutMinutes"] = self.execution_timeout_minutes
        if self.name is not None:
            config["name"] = self.name
        if self.tags is not None:
            config["tags"] = self.tags
        return config


ERR_STR_WITH_BOTH_APP_ID_AND_APP_CONFIG = (
    "EMRServerlessStep {step_name} cannot have both application_id and application_config. "
    "To use EMRServerlessStep with application_config, "
    "application_id must be explicitly set to None."
)

ERR_STR_WITHOUT_APP_ID_AND_APP_CONFIG = (
    "EMRServerlessStep {step_name} must have either application_id or application_config"
)


class EMRServerlessStep(ConfigurableRetryStep):
    """EMR Serverless step for workflow with configurable retry policies."""

    def __init__(
        self,
        name: str,
        display_name: str,
        description: str,
        job_config: EMRServerlessJobConfig,
        application_id: Optional[str] = None,
        application_config: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
        cache_config: Optional[CacheConfig] = None,
        retry_policies: Optional[List[StepRetryPolicy]] = None,
    ):  # pylint: disable=too-many-positional-arguments
        """Constructs an `EMRServerlessStep`.

        Args:
            name (str): The name of the EMR Serverless step.
            display_name (str): The display name of the EMR Serverless step.
            description (str): The description of the EMR Serverless step.
            job_config (EMRServerlessJobConfig): Job configuration for the EMR Serverless job.
            application_id (str, optional): The ID of the existing EMR Serverless application.
            application_config (Dict[str, Any], optional): Configuration for creating a new
                EMR Serverless application.
            depends_on (List[Union[str, Step, StepCollection]], optional): A list of
                `Step`/`StepCollection` names or `Step` instances or `StepCollection` instances
                that this `EMRServerlessStep` depends on.
            cache_config (CacheConfig, optional): A `sagemaker.workflow.steps.CacheConfig` instance.
            retry_policies (List[StepRetryPolicy], optional): A list of retry policies.
        """
        super().__init__(
            name=name,
            step_type=StepTypeEnum.EMR_SERVERLESS,
            display_name=display_name,
            description=description,
            depends_on=depends_on,
            retry_policies=retry_policies,
        )

        if application_id is None and application_config is None:
            raise ValueError(ERR_STR_WITHOUT_APP_ID_AND_APP_CONFIG.format(step_name=name))

        if application_id is not None and application_config is not None:
            raise ValueError(ERR_STR_WITH_BOTH_APP_ID_AND_APP_CONFIG.format(step_name=name))

        emr_serverless_args = {
            "ExecutionRoleArn": job_config.execution_role_arn,  # Top-level role (used by backend)
            "JobConfig": job_config.to_request(
                application_id
            ),  # Role also in JobConfig (structure requirement)
        }

        if application_id is not None:
            emr_serverless_args["ApplicationId"] = application_id
        elif application_config is not None:
            emr_serverless_args["ApplicationConfig"] = application_config

        self.args = emr_serverless_args
        self.cache_config = cache_config

        root_property = Properties(
            step_name=name, step=self, shape_name="GetJobRunResponse", service_name="emr-serverless"
        )
        self._properties = root_property

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call EMR Serverless APIs."""
        return self.args

    @property
    def properties(self) -> RequestType:
        """A Properties object representing the EMR Serverless GetJobRunResponse model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration and retry policies."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        return request_dict
