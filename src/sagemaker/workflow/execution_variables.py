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

from sagemaker.workflow.entities import (
    Expression,
    RequestType,
)


class ExecutionVariable(Expression):
    """Pipeline execution variables for workflow."""

    def __init__(self, name: str):
        """Create a pipeline execution variable.

        Args:
            name (str): The name of the execution variable.
        """
        self.name = name

    @property
    def expr(self) -> RequestType:
        """The 'Get' expression dict for an `ExecutionVariable`."""
        return {"Get": f"Execution.{self.name}"}


class ExecutionVariables:
    """All available ExecutionVariable."""

    START_DATETIME = ExecutionVariable("StartDateTime")
    CURRENT_DATETIME = ExecutionVariable("CurrentDateTime")
    PIPELINE_NAME = ExecutionVariable("PipelineName")
    PIPELINE_ARN = ExecutionVariable("PipelineArn")
    PIPELINE_EXECUTION_ID = ExecutionVariable("PipelineExecutionId")
    PIPELINE_EXECUTION_ARN = ExecutionVariable("PipelineExecutionArn")
