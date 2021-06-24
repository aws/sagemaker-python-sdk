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
"""The step definitions for workflow."""
from __future__ import absolute_import

from typing import List, Union

from sagemaker.workflow.conditions import Condition
from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.properties import Properties
from sagemaker.workflow.steps import (
    Step,
    StepTypeEnum,
)
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.utilities import list_to_request


class ConditionStep(Step):
    """Conditional step for pipelines to support conditional branching in the execution of steps."""

    def __init__(
        self,
        name: str,
        depends_on: Union[List[str], List[Step]] = None,
        conditions: List[Condition] = None,
        if_steps: List[Union[Step, StepCollection]] = None,
        else_steps: List[Union[Step, StepCollection]] = None,
    ):
        """Construct a ConditionStep for pipelines to support conditional branching.

        If all of the conditions in the condition list evaluate to True, the `if_steps` are
        marked as ready for execution. Otherwise, the `else_steps` are marked as ready for
        execution.

        Args:
            conditions (List[Condition]): A list of `sagemaker.workflow.conditions.Condition`
                instances.
            if_steps (List[Union[Step, StepCollection]]): A list of `sagemaker.workflow.steps.Step`
                and `sagemaker.workflow.step_collections.StepCollection` instances that are
                marked as ready for execution if the list of conditions evaluates to True.
            else_steps (List[Union[Step, StepCollection]]): A list of `sagemaker.workflow.steps.Step`
                and `sagemaker.workflow.step_collections.StepCollection` instances that are
                marked as ready for execution if the list of conditions evaluates to False.
        """
        super(ConditionStep, self).__init__(name, StepTypeEnum.CONDITION, depends_on)
        self.conditions = conditions or []
        self.if_steps = if_steps or []
        self.else_steps = else_steps or []

        root_path = f"Steps.{name}"
        root_prop = Properties(path=root_path)
        root_prop.__dict__["Outcome"] = Properties(f"{root_path}.Outcome")
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
    def properties(self):
        """A simple Properties object with `Outcome` as the only property"""
        return self._properties
