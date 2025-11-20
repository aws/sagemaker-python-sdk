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

from typing import List, Union

import attr
from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.step_outputs import StepOutput
from sagemaker.mlops.workflow.steps import Step


@attr.s
class StepCollection:
    """A wrapper of pipeline steps for workflow.

    Attributes:
        name (str): The name of the `StepCollection`.
        steps (List[Step]): A list of steps.
        depends_on (List[Union[str, Step, StepCollection, StepOutput]]):
            The list of `Step`/`StepCollection` names or `Step`/`StepCollection`/`StepOutput`
            instances that the current `Step` depends on.
    """

    name: str = attr.ib()
    steps: List[Step] = attr.ib(factory=list)
    depends_on: List[Union[str, Step, "StepCollection", StepOutput]] = attr.ib(default=None)

    def request_dicts(self) -> List[RequestType]:
        """Get the request structure for workflow service calls."""
        return [step.to_request() for step in self.steps]

    @property
    def properties(self):
        """The properties of the particular `StepCollection`."""
        if not self.steps:
            return None
        return self.steps[-1].properties