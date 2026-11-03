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
"""Run conditions for step dependencies in SageMaker Pipelines."""

from __future__ import absolute_import

from enum import Enum
from typing import Union, TYPE_CHECKING

from sagemaker.core.workflow.step_outputs import StepOutput, get_step

if TYPE_CHECKING:
    from sagemaker.mlops.workflow.steps import Step


class TriggerCondition(Enum):
    """Upstream step outcome that allows a dependent step to run."""

    COMPLETED = "Completed"
    FAILED = "Failed"
    STOPPED = "Stopped"
    ANY = "Any"


class StepDependency:
    """A `depends_on` entry that runs the dependent step on a chosen upstream outcome.

    Plain entries in `depends_on` (a step or step name) keep their existing meaning: the
    dependent step runs only after the upstream step completes. Wrap an entry in
    `StepDependency` to run the dependent step when the upstream step fails, stops, or
    reaches any terminal state, for example to send a notification when training fails:

    .. code-block:: python

        notify = LambdaStep(
            name="NotifyOnFailure",
            lambda_func=notify_lambda,
            depends_on=[StepDependency(train, TriggerCondition.FAILED)],
        )

    A pipeline whose steps run on a failed upstream step ends in the
    `CompletedWithFailures` status instead of `Failed`. Run conditions are not supported
    in local mode.
    """

    def __init__(
        self,
        step: Union[str, "Step", StepOutput],
        trigger: TriggerCondition = TriggerCondition.COMPLETED,
    ):
        """Create a step dependency with a run condition.

        Args:
            step (Union[str, Step, StepOutput]): The upstream `Step`, its name, or a
                `StepOutput` produced by it.
            trigger (TriggerCondition): The upstream outcome that lets the dependent step
                run (default: `TriggerCondition.COMPLETED`).
        """
        if not isinstance(trigger, TriggerCondition):
            raise TypeError(f"trigger must be a TriggerCondition, got {type(trigger).__name__}.")
        self.step = get_step(step) if isinstance(step, StepOutput) else step
        self.trigger = trigger

    @property
    def step_name(self) -> str:
        """The name of the upstream step."""
        return self.step if isinstance(self.step, str) else self.step.name

    def to_request(self, step_name: str = None) -> Union[str, dict]:
        """Get the `DependsOn` entry for this dependency.

        A `COMPLETED` dependency is emitted as a plain step name, so definitions that do not
        use run conditions are unchanged.
        """
        name = step_name or self.step_name
        if self.trigger == TriggerCondition.COMPLETED:
            return name
        return {"Step": name, "Trigger": self.trigger.value}
