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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import List, Union, Optional
from sagemaker.core.workflow.conditions import Condition
from sagemaker.mlops.workflow.steps import (
    Step,
    StepTypeEnum,
)

from sagemaker.core.workflow.utilities import list_to_request
from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import (
    Properties,
)


class ConditionStep(Step):
    """Conditional step for pipelines to support conditional branching in the execution of steps."""

    def __init__(
        self,
        name: str,
        depends_on: Optional[List[Union[str, Step]]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        conditions: Optional[List[Condition]] = None,
        if_steps: Optional[List[Step]] = None,
        else_steps: Optional[List[Step]] = None,
    ):
        """Construct a ConditionStep for pipelines to support conditional branching.

        If all the conditions in the condition list evaluate to True, the `if_steps` are
        marked as ready for execution. Otherwise, the `else_steps` are marked as ready for
        execution.

        Args:
            name (str): The name of the condition step.
            depends_on (List[Union[str, Step]]): The list of `Step`
                names or `Step` instances that the current `Step`
                depends on.
            display_name (str): The display name of the condition step.
            description (str): The description of the condition step.
            conditions (List[Condition]): A list of `sagemaker.workflow.conditions.Condition`
                instances.
            if_steps (List[Step]): A list of `sagemaker.workflow.steps.Step`
                instances that are marked as ready for execution if the list of conditions evaluates to True.
            else_steps (List[Step]): A list of `sagemaker.workflow.steps.Step`
                instances that are marked as ready for execution if the list of conditions evaluates to False.
        """
        super(ConditionStep, self).__init__(
            name, display_name, description, StepTypeEnum.CONDITION, depends_on
        )
        self.conditions = conditions or []
        self.if_steps = if_steps or []
        self.else_steps = else_steps or []

        root_prop = Properties(step_name=name)
        root_prop.__dict__["Outcome"] = Properties(step_name=name, path="Outcome")
        self._properties = root_prop

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to define the conditional branching in the pipeline."""
        return dict(
            Conditions=[condition.to_request() for condition in self.conditions],
            IfSteps=list_to_request(self.if_steps),
            ElseSteps=list_to_request(self.else_steps),
        )

    @property
    def step_only_arguments(self):
        """Argument dict pertaining to the step only, and not the `if_steps` or `else_steps`."""
        return dict(Conditions=[condition.to_request() for condition in self.conditions])

    @property
    def properties(self):
        """A simple Properties object with `Outcome` as the only property"""
        return self._properties