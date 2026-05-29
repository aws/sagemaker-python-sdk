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

Design note: mirrors Tioga's ``InferenceComponentStep`` in
``IronmanTiogaPipelineDefinitionRepository``. The ``Arguments`` block is
an opaque ``StructureArgument`` validated against SageMaker's
``CreateInferenceComponentInput`` Coral model with no field exclusions —
any field the AWS API accepts, Tioga accepts.
"""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import Properties

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class InferenceComponentStep(Step):
    """Creates or updates a SageMaker Inference Component within a pipeline.

    Wraps the SageMaker ``CreateInferenceComponent``/``UpdateInferenceComponent``
    API — the pipeline chooses create-vs-update based on component existence.
    Inference components enable multi-model endpoint deployments with
    independent scaling per model.

    The ``arguments`` dict is passed through to the service; it accepts
    any field of ``CreateInferenceComponentInput`` (no exclusions).

    Per the Zimmer step contract, ``InferenceComponent`` is neither
    cacheable nor retryable at the pipeline level.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct an ``InferenceComponentStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block for the
                ``CreateInferenceComponent``/``UpdateInferenceComponent``
                call. Typical fields: ``InferenceComponentName``,
                ``EndpointName``, ``VariantName``, ``Specification``,
                ``Specifications`` (plural, for multi-spec deployments),
                ``RuntimeConfig``. Values may be pipeline variables.
                Note: ``ComputeResourceRequirements.NumberOfCpuCoresRequired``
                is a float — pass ``2.0`` not ``2``.
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
        if arguments is None:
            raise ValueError("arguments is required for InferenceComponentStep.")
        self._arguments = arguments
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeInferenceComponentOutput"
        )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block for the Create/Update InferenceComponent call."""
        return self._arguments

    @property
    def properties(self):
        """A ``Properties`` object shaped like ``DescribeInferenceComponentOutput``."""
        return self._properties
