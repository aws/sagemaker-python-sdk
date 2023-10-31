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
"""Pipeline parameters and conditions for workflow."""
from __future__ import absolute_import

from typing import List, TYPE_CHECKING
from sagemaker.workflow.entities import (
    RequestType,
    PipelineVariable,
)

if TYPE_CHECKING:
    from sagemaker.workflow.steps import Step


class ExecutionVariable(PipelineVariable):
    """Pipeline execution variables for workflow."""

    def __init__(self, name: str):
        """Create a pipeline execution variable.

        Args:
            name (str): The name of the execution variable.
        """
        self.name = name

    def __eq__(self, other):
        """Override default equals method"""
        if not isinstance(other, ExecutionVariable):
            return NotImplemented
        return self.name == other.name

    def to_string(self) -> PipelineVariable:
        """Prompt the pipeline to convert the pipeline variable to String in runtime

        As ExecutionVariable is treated as String in runtime, no extra actions are needed.
        """
        return self

    @property
    def expr(self) -> RequestType:
        """The 'Get' expression dict for an `ExecutionVariable`."""
        return {"Get": f"Execution.{self.name}"}

    @property
    def _pickleable(self):
        """The pickleable object that can be passed to a remote function invocation."""

        from sagemaker.remote_function.core.pipeline_variables import _ExecutionVariable

        return _ExecutionVariable(name=self.name)

    @property
    def _referenced_steps(self) -> List["Step"]:
        """List of step that this function depends on."""
        return []


class ExecutionVariables:
    """Provide access to all available execution variables:

    - ExecutionVariables.START_DATETIME
    - ExecutionVariables.CURRENT_DATETIME
    - ExecutionVariables.PIPELINE_NAME
    - ExecutionVariables.PIPELINE_ARN
    - ExecutionVariables.PIPELINE_EXECUTION_ID
    - ExecutionVariables.PIPELINE_EXECUTION_ARN
    - ExecutionVariables.TRAINING_JOB_NAME
    - ExecutionVariables.PROCESSING_JOB_NAME
    """

    START_DATETIME = ExecutionVariable("StartDateTime")
    CURRENT_DATETIME = ExecutionVariable("CurrentDateTime")
    PIPELINE_NAME = ExecutionVariable("PipelineName")
    PIPELINE_ARN = ExecutionVariable("PipelineArn")
    PIPELINE_EXECUTION_ID = ExecutionVariable("PipelineExecutionId")
    PIPELINE_EXECUTION_ARN = ExecutionVariable("PipelineExecutionArn")
    TRAINING_JOB_NAME = ExecutionVariable("TrainingJobName")
    PROCESSING_JOB_NAME = ExecutionVariable("ProcessingJobName")
