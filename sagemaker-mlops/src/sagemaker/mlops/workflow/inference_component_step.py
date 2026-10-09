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
"""Step definition for SageMaker InferenceComponent in Pipelines.

Follows the ``step_args`` convention: call
:meth:`~sagemaker.core.helper.session_helper.Session.create_inference_component`
under a :class:`~sagemaker.core.workflow.pipeline_context.PipelineSession`
and pass the returned step arguments to the step.

Example::

    pipeline_session = PipelineSession()

    step_args = pipeline_session.create_inference_component(
        inference_component_name="my-component",
        endpoint_name="my-endpoint",
        variant_name="AllTraffic",
        specification={...},
    )
    step = InferenceComponentStep(name="CreateComponent", step_args=step_args)
"""

from __future__ import absolute_import

from typing import List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.pipeline_context import _JobStepArguments
from sagemaker.core.workflow.properties import Properties
from sagemaker.core.workflow.utilities import validate_step_args_input

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class InferenceComponentStep(Step):
    """Creates or updates a SageMaker Inference Component within a pipeline.

    Wraps the SageMaker ``CreateInferenceComponent``/``UpdateInferenceComponent``
    API -- the pipeline chooses create-vs-update based on component
    existence. Inference components enable multi-model endpoint
    deployments with independent scaling per model.

    The ``step_args`` must be obtained by calling
    :meth:`~sagemaker.core.helper.session_helper.Session.create_inference_component`
    on a ``PipelineSession``.

    ``InferenceComponent`` is neither cacheable nor retryable at the
    pipeline level.
    """

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct an ``InferenceComponentStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments): The arguments for this step,
                obtained from
                ``pipeline_session.create_inference_component()``.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.INFERENCE_COMPONENT,
            depends_on=depends_on,
        )
        validate_step_args_input(
            step_args=step_args,
            expected_caller={"create_inference_component"},
            error_message=(
                "The step_args of InferenceComponentStep must be obtained from "
                "pipeline_session.create_inference_component()."
            ),
        )
        self.step_args = step_args
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeInferenceComponentOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call ``create_inference_component``."""
        return self.step_args.args

    @property
    def properties(self):
        """A ``Properties`` object shaped like ``DescribeInferenceComponentOutput``."""
        return self._properties
