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
"""Base class representing step decorator outputs"""
from __future__ import absolute_import

import abc
from typing import List, TYPE_CHECKING

from sagemaker.workflow.entities import RequestType, PipelineVariable

if TYPE_CHECKING:
    from sagemaker.workflow.steps import Step


class StepOutput(PipelineVariable):
    """Base class representing ``@step`` decorator outputs."""

    def __init__(self, step: "Step" = None):
        """Initializes a `StepOutput` object.

        Args:
            step: A `sagemaker.workflow.steps.Step` instance.
        """

        self._step = step

    def __repr__(self):
        """Formatted representation of the output class"""
        return str(self.__dict__)

    @property
    @abc.abstractmethod
    def expr(self) -> RequestType:
        """Get the expression structure for workflow service calls."""

    @property
    def _referenced_steps(self) -> List[str]:
        """List of step names that this function depends on."""
        # TODO: Implement this
        return []


def get_step(step_output: StepOutput):
    """Get the step associated with this output.

    Args:
        step_output: A `sagemaker.workflow.steps.StepOutput` instance.

    Returns:
        A `sagemaker.workflow.steps.Step` instance.
    """
    return step_output._step
