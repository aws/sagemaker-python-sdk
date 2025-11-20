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
"""Local pipeline execution entities."""
from __future__ import absolute_import

import enum
import datetime
import logging
from uuid import uuid4
from copy import deepcopy
from botocore.exceptions import ClientError

from sagemaker.mlops.local.exceptions import StepExecutionException

logger = logging.getLogger(__name__)


class _LocalPipeline(object):
    """Class representing a local SageMaker Pipeline"""

    _executions = {}

    def __init__(
        self,
        pipeline,
        pipeline_description=None,
        local_session=None,
    ):
        from sagemaker.core.local import LocalSession

        self.local_session = local_session or LocalSession()
        self.pipeline = pipeline
        self.pipeline_description = pipeline_description
        self.creation_time = datetime.datetime.now().timestamp()
        self.last_modified_time = self.creation_time

    def describe(self):
        """Describe Pipeline"""
        response = {
            "PipelineArn": self.pipeline.name,
            "PipelineDefinition": self.pipeline.definition(),
            "PipelineDescription": self.pipeline_description,
            "PipelineName": self.pipeline.name,
            "PipelineStatus": "Active",
            "RoleArn": "<no_role>",
            "CreationTime": self.creation_time,
            "LastModifiedTime": self.last_modified_time,
        }
        return response

    def start(self, **kwargs):
        """Start a pipeline execution. Returns a _LocalPipelineExecution object."""
        from sagemaker.mlops.local.pipeline import LocalPipelineExecutor

        execution_id = str(uuid4())
        execution = _LocalPipelineExecution(
            execution_id=execution_id,
            pipeline=self.pipeline,
            local_session=self.local_session,
            **kwargs,
        )

        self._executions[execution_id] = execution
        logger.info(
            "Starting execution for pipeline %s. Execution ID is %s",
            self.pipeline.name,
            execution_id,
        )
        self.last_modified_time = datetime.datetime.now().timestamp()

        return LocalPipelineExecutor(execution, self.local_session).execute()


class _LocalPipelineExecution(object):
    """Class representing a local SageMaker pipeline execution."""

    def __init__(
        self,
        execution_id,
        pipeline,
        PipelineParameters=None,
        PipelineExecutionDescription=None,
        PipelineExecutionDisplayName=None,
        local_session=None,
    ):
        from sagemaker.mlops.workflow.pipeline import PipelineGraph
        from sagemaker.core.local import LocalSession

        self.pipeline = pipeline
        self.pipeline_execution_name = execution_id
        self.pipeline_execution_description = PipelineExecutionDescription
        self.pipeline_execution_display_name = PipelineExecutionDisplayName
        self.local_session = local_session or LocalSession()
        self.status = _LocalExecutionStatus.EXECUTING.value
        self.failure_reason = None
        self.creation_time = datetime.datetime.now().timestamp()
        self.last_modified_time = self.creation_time
        self.step_execution = {}
        self.pipeline_dag = PipelineGraph.from_pipeline(self.pipeline)
        self._initialize_step_execution(self.pipeline_dag.step_map.values())
        self.pipeline_parameters = self._initialize_and_validate_parameters(PipelineParameters)
        self._blocked_steps = {}

    def describe(self):
        """Describe Pipeline Execution."""
        response = {
            "CreationTime": self.creation_time,
            "LastModifiedTime": self.last_modified_time,
            "FailureReason": self.failure_reason,
            "PipelineArn": self.pipeline.name,
            "PipelineExecutionArn": self.pipeline_execution_name,
            "PipelineExecutionDescription": self.pipeline_execution_description,
            "PipelineExecutionDisplayName": self.pipeline_execution_display_name,
            "PipelineExecutionStatus": self.status,
        }
        filtered_response = {k: v for k, v in response.items() if v is not None}
        return filtered_response

    def list_steps(self):
        """List pipeline execution steps."""
        return {
            "PipelineExecutionSteps": [
                step.to_list_steps_response()
                for step in self.step_execution.values()
                if step.status is not None
            ]
        }

    def result(self, step_name: str):
        """Retrieves the output of the provided step if it is a ``@step`` decorated function.

        Args:
            step_name (str): The name of the pipeline step.
        Returns:
            The step output.

        Raises:
              ValueError if the provided step is not a ``@step`` decorated function.
              RuntimeError if the provided step is not in "Completed" status.
        """
        from sagemaker.mlops.workflow.pipeline import get_function_step_result

        return get_function_step_result(
            step_name=step_name,
            step_list=self.list_steps()["PipelineExecutionSteps"],
            execution_id=self.pipeline_execution_name,
            sagemaker_session=self.local_session,
        )

    def update_execution_success(self):
        """Mark execution as succeeded."""
        self.status = _LocalExecutionStatus.SUCCEEDED.value
        self.last_modified_time = datetime.datetime.now().timestamp()
        logger.info("Pipeline execution %s SUCCEEDED", self.pipeline_execution_name)

    def update_execution_failure(self, step_name, failure_message):
        """Mark execution as failed."""
        self.status = _LocalExecutionStatus.FAILED.value
        self.failure_reason = f"Step '{step_name}' failed with message: {failure_message}"
        self.last_modified_time = datetime.datetime.now().timestamp()
        logger.info(
            "Pipeline execution %s FAILED because step '%s' failed.",
            self.pipeline_execution_name,
            step_name,
        )

    def update_step_properties(self, step_name, step_properties):
        """Update pipeline step execution output properties."""
        self.step_execution.get(step_name).update_step_properties(step_properties)
        logger.info("Pipeline step '%s' SUCCEEDED.", step_name)

    def update_step_failure(self, step_name, failure_message):
        """Mark step_name as failed."""
        logger.info("Pipeline step '%s' FAILED. Failure message is: %s", step_name, failure_message)
        self.step_execution.get(step_name).update_step_failure(failure_message)

    def mark_step_executing(self, step_name):
        """Update pipelines step's status to EXECUTING and start_time to now."""
        logger.info("Starting pipeline step: '%s'", step_name)
        self.step_execution.get(step_name).mark_step_executing()

    def _initialize_step_execution(self, steps):
        """Initialize step_execution dict."""
        from sagemaker.mlops.workflow.steps import StepTypeEnum, Step

        supported_steps_types = (
            StepTypeEnum.TRAINING,
            StepTypeEnum.PROCESSING,
            StepTypeEnum.TRANSFORM,
            StepTypeEnum.CONDITION,
            StepTypeEnum.FAIL,
            StepTypeEnum.CREATE_MODEL,
        )

        for step in steps:
            if isinstance(step, Step):
                if step.step_type not in supported_steps_types:
                    error_msg = self._construct_validation_exception_message(
                        "Step type {} is not supported in local mode.".format(step.step_type.value)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                self.step_execution[step.name] = _LocalPipelineExecutionStep(
                    step.name, step.step_type, step.description, step.display_name
                )
                if step.step_type == StepTypeEnum.CONDITION:
                    self._initialize_step_execution(step.if_steps + step.else_steps)

    def _initialize_and_validate_parameters(self, overridden_parameters):
        """Initialize and validate pipeline parameters."""
        merged_parameters = {}
        default_parameters = {parameter.name: parameter for parameter in self.pipeline.parameters}
        if overridden_parameters is not None:
            for param_name, param_value in overridden_parameters.items():
                if param_name not in default_parameters:
                    error_msg = self._construct_validation_exception_message(
                        "Unknown parameter '{}'".format(param_name)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                parameter_type = default_parameters[param_name].parameter_type
                if type(param_value) != parameter_type.python_type:  # pylint: disable=C0123
                    error_msg = self._construct_validation_exception_message(
                        "Unexpected type for parameter '{}'. Expected {} but found "
                        "{}.".format(param_name, parameter_type.python_type, type(param_value))
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                if param_value == "":
                    error_msg = self._construct_validation_exception_message(
                        'Parameter {} value "" is too short (length: 0, '
                        "required minimum: 1).".format(param_name)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                merged_parameters[param_name] = param_value
        for param_name, default_parameter in default_parameters.items():
            if param_name not in merged_parameters:
                if default_parameter.default_value is None:
                    error_msg = self._construct_validation_exception_message(
                        "Parameter '{}' is undefined.".format(param_name)
                    )
                    raise ClientError(error_msg, "start_pipeline_execution")
                merged_parameters[param_name] = default_parameter.default_value
        return merged_parameters

    @staticmethod
    def _construct_validation_exception_message(exception_msg):
        """Construct error response for botocore.exceptions.ClientError"""
        return {"Error": {"Code": "ValidationException", "Message": exception_msg}}


class _LocalPipelineExecutionStep(object):
    """Class representing a local pipeline execution step."""

    def __init__(
        self,
        name,
        step_type,
        description,
        display_name=None,
        start_time=None,
        end_time=None,
        status=None,
        properties=None,
        failure_reason=None,
    ):
        from sagemaker.mlops.workflow.steps import StepTypeEnum

        self.name = name
        self.type = step_type
        self.description = description
        self.display_name = display_name
        self.status = status
        self.failure_reason = failure_reason
        self.properties = properties or {}
        self.start_time = start_time
        self.end_time = end_time
        self._step_type_to_output_format_map = {
            StepTypeEnum.TRAINING: self._construct_training_metadata,
            StepTypeEnum.PROCESSING: self._construct_processing_metadata,
            StepTypeEnum.TRANSFORM: self._construct_transform_metadata,
            StepTypeEnum.CREATE_MODEL: self._construct_model_metadata,
            StepTypeEnum.CONDITION: self._construct_condition_metadata,
            StepTypeEnum.FAIL: self._construct_fail_metadata,
        }

    def update_step_properties(self, properties):
        """Update pipeline step execution output properties."""
        self.properties = deepcopy(properties)
        self.status = _LocalExecutionStatus.SUCCEEDED.value
        self.end_time = datetime.datetime.now().timestamp()

    def update_step_failure(self, failure_message):
        """Update pipeline step execution failure status and message."""
        self.failure_reason = failure_message
        self.status = _LocalExecutionStatus.FAILED.value
        self.end_time = datetime.datetime.now().timestamp()
        raise StepExecutionException(self.name, failure_message)

    def mark_step_executing(self):
        """Update pipelines step's status to EXECUTING and start_time to now"""
        self.status = _LocalExecutionStatus.EXECUTING.value
        self.start_time = datetime.datetime.now().timestamp()

    def to_list_steps_response(self):
        """Convert to response dict for list_steps calls."""
        response = {
            "EndTime": self.end_time,
            "FailureReason": self.failure_reason,
            "Metadata": self._construct_metadata(),
            "StartTime": self.start_time,
            "StepDescription": self.description,
            "StepDisplayName": self.display_name,
            "StepName": self.name,
            "StepStatus": self.status,
        }
        filtered_response = {k: v for k, v in response.items() if v is not None}
        return filtered_response

    def _construct_metadata(self):
        """Constructs the metadata shape for the list_steps_response."""
        if self.properties:
            return self._step_type_to_output_format_map[self.type]()
        return None

    def _construct_training_metadata(self):
        """Construct training job metadata response."""
        return {"TrainingJob": {"Arn": self.properties["TrainingJobName"]}}

    def _construct_processing_metadata(self):
        """Construct processing job metadata response."""
        return {"ProcessingJob": {"Arn": self.properties["ProcessingJobName"]}}

    def _construct_transform_metadata(self):
        """Construct transform job metadata response."""
        return {"TransformJob": {"Arn": self.properties["TransformJobName"]}}

    def _construct_model_metadata(self):
        """Construct create model step metadata response."""
        return {"Model": {"Arn": self.properties["ModelName"]}}

    def _construct_condition_metadata(self):
        """Construct condition step metadata response."""
        return {"Condition": {"Outcome": self.properties["Outcome"]}}

    def _construct_fail_metadata(self):
        """Construct fail step metadata response."""
        return {"Fail": {"ErrorMessage": self.properties["ErrorMessage"]}}


class _LocalExecutionStatus(enum.Enum):
    """Pipeline execution status."""

    EXECUTING = "Executing"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"
