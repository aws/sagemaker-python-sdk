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
"""Pipeline parameters and conditions for workflow."""
from __future__ import absolute_import

from typing import Dict

from sagemaker.workflow.entities import (
    Entity,
    RequestType,
)


class ExecutionVariable(Entity, str):
    """Pipeline execution variables for workflow."""

    def __new__(cls, *args, **kwargs):  # pylint: disable=unused-argument
        """Subclass str"""
        value = ""
        if len(args) == 1:
            value = args[0] or value
        elif kwargs:
            value = kwargs.get("name", value)
        return str.__new__(cls, ExecutionVariable._expr(value))

    def __init__(self, name: str):
        """Create a pipeline execution variable.

        Args:
            name (str): The name of the execution variable.
        """
        super(ExecutionVariable, self).__init__()
        self.name = name

    def __hash__(self):
        """Hash function for execution variable types"""
        return hash(tuple(self.to_request()))

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        return self.expr

    @property
    def expr(self) -> Dict[str, str]:
        """The 'Get' expression dict for an `ExecutionVariable`."""
        return ExecutionVariable._expr(self.name)

    @classmethod
    def _expr(cls, name):
        """An internal classmethod for the 'Get' expression dict for an `ExecutionVariable`.

        Args:
            name (str): The name of the execution variable.
        """
        return {"Get": f"Execution.{name}"}


class ExecutionVariables:
    """Enum-like class for all ExecutionVariable instances.

    Considerations to move these as module-level constants should be made.
    """

    START_DATETIME = ExecutionVariable("StartDateTime")
    CURRENT_DATETIME = ExecutionVariable("CurrentDateTime")
    PIPELINE_NAME = ExecutionVariable("PipelineName")
    PIPELINE_ARN = ExecutionVariable("PipelineArn")
    PIPELINE_EXECUTION_ID = ExecutionVariable("PipelineExecutionId")
    PIPELINE_EXECUTION_ARN = ExecutionVariable("PipelineExecutionArn")
