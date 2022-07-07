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
"""Local Pipeline Executor"""
from __future__ import absolute_import

import logging
from copy import deepcopy
from datetime import datetime

from sagemaker.workflow.entities import PipelineVariable
from sagemaker.workflow.parameters import Parameter
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.execution_variables import ExecutionVariable, ExecutionVariables
from sagemaker.workflow.pipeline import PipelineGraph
from sagemaker.local.exceptions import StepExecutionException
from sagemaker.local.utils import get_using_dot_notation

logger = logging.getLogger(__name__)

PRIMITIVES = (str, int, bool, float)


class LocalPipelineExecutor(object):
    """An executor that executes SageMaker Pipelines locally."""

    def __init__(self, execution, sagemaker_session):
        """Initialize StepExecutor.

        Args:
            sagemaker_session (sagemaker.session.Session): a session to use to read configurations
                from, and use its boto client.
        """
        self.sagemaker_session = sagemaker_session
        self.execution = execution
        self.pipeline_dag = PipelineGraph.from_pipeline(self.execution.pipeline)

    def execute(self):
        """Execute a local pipeline."""
        try:
            for step in self.pipeline_dag:
                self.execute_step(step)
        except StepExecutionException as e:
            self.execution.update_execution_failure(e.step_name, e.message)
        return self.execution

    def execute_step(self, step):
        """Execute a local pipeline step."""
        self.execution.mark_step_starting(step.name)
        step_arguments = self.evaluate_step_arguments(step)  # noqa: F841; pylint: disable=W0612
        # TODO execute step

    def evaluate_step_arguments(self, step):
        """Parses and evaluate step arguments."""

        def _parse_arguments(obj):
            if isinstance(obj, dict):
                obj_copy = deepcopy(obj)
                for k, v in obj.items():
                    if isinstance(v, dict):
                        obj_copy[k] = _parse_arguments(v)
                    elif isinstance(v, list):
                        list_copy = []
                        for item in v:
                            list_copy.append(_parse_arguments(item))
                        obj_copy[k] = list_copy
                    elif isinstance(v, PipelineVariable):
                        obj_copy[k] = self.evaluate_pipeline_variable(v, step.name)
                return obj_copy
            return obj

        return _parse_arguments(step.arguments)

    def evaluate_pipeline_variable(self, pipeline_variable, step_name):
        """Evaluate pipeline variable runtime value."""
        value = None
        if isinstance(pipeline_variable, PRIMITIVES):
            value = pipeline_variable
        elif isinstance(pipeline_variable, Parameter):
            value = self.execution.pipeline_parameters.get(pipeline_variable.name)
        elif isinstance(pipeline_variable, Join):
            evaluated = [
                self.evaluate_pipeline_variable(v, step_name) for v in pipeline_variable.values
            ]
            value = pipeline_variable.on.join(evaluated)
        elif isinstance(pipeline_variable, Properties):
            value = self._evaluate_property_reference(pipeline_variable, step_name)
        elif isinstance(pipeline_variable, ExecutionVariable):
            value = self._evaluate_execution_variable(pipeline_variable)
        elif isinstance(pipeline_variable, JsonGet):
            # TODO
            raise NotImplementedError
        else:
            self.execution.update_step_failure(
                step_name, f"Unrecognized pipeline variable {pipeline_variable.expr}."
            )

        if value is None:
            self.execution.update_step_failure(step_name, f"{pipeline_variable.expr} is undefined.")
        return value

    def _evaluate_property_reference(self, pipeline_variable, step_name):
        """Evaluate property reference runtime value."""
        try:
            referenced_step_name = pipeline_variable.step_name
            step_properties = self.execution.step_execution.get(referenced_step_name).properties
            return get_using_dot_notation(step_properties, pipeline_variable.path)
        except (KeyError, IndexError):
            self.execution.update_step_failure(step_name, f"{pipeline_variable.expr} is undefined.")

    def _evaluate_execution_variable(self, pipeline_variable):
        """Evaluate pipeline execution variable runtime value."""
        if pipeline_variable in (ExecutionVariables.PIPELINE_NAME, ExecutionVariables.PIPELINE_ARN):
            return self.execution.pipeline.name
        if pipeline_variable in (
            ExecutionVariables.PIPELINE_EXECUTION_ID,
            ExecutionVariables.PIPELINE_EXECUTION_ARN,
        ):
            return self.execution.pipeline_execution_name
        if pipeline_variable == ExecutionVariables.START_DATETIME:
            return self.execution.creation_time
        if pipeline_variable == ExecutionVariables.CURRENT_DATETIME:
            return datetime.now()
        return None
