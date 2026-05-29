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
"""Step definition for SageMaker Lineage tracking in Pipelines.

Design note: mirrors Tioga's ``LineageStep`` in
``IronmanTiogaPipelineDefinitionRepository``. The ``Arguments`` block
conforms to Tioga's ``LineageStepArgument`` structure — four optional
lists:

* ``Actions`` — list of ``CreateActionRequest`` shapes
* ``Artifacts`` — list of ``CreateArtifactRequest`` shapes
* ``Contexts`` — list of ``CreateContextRequest`` shapes
* ``Associations`` — list of ``LineageAssociation`` shapes
  (``Source``/``Destination``/``AssociationType``)

The SDK passes the ``arguments`` dict through verbatim.
"""

from __future__ import absolute_import

from typing import Any, Dict, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.properties import Properties

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


class LineageStep(Step):
    """Creates and associates lineage entities in SageMaker's lineage system.

    Wraps SageMaker's ``CreateAction``/``CreateArtifact``/``CreateContext``
    and lineage ``AddAssociation`` APIs. A single step may create
    multiple entities of any of the four types (Actions, Artifacts,
    Contexts, Associations). Property references use
    ``Steps.<StepName>.ActionArns['<name>']``,
    ``Steps.<StepName>.ArtifactArns['<name>']``,
    ``Steps.<StepName>.ContextArns['<name>']``, and
    ``Steps.<StepName>.Associations``.
    """

    def __init__(
        self,
        name: str,
        arguments: Dict[str, Any],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``LineageStep``.

        Args:
            name (str): The name of the step.
            arguments (Dict[str, Any]): The ``Arguments`` block. Recognized
                top-level keys: ``Actions``, ``Artifacts``, ``Contexts``,
                ``Associations`` — each is a list of dicts conforming to
                the corresponding SageMaker API shape (or Tioga's
                ``LineageAssociation`` for ``Associations``). At least
                one of the four keys must be present.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.

        Raises:
            ValueError: If ``arguments`` is None or contains none of the
                recognized keys.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.LINEAGE,
            depends_on=depends_on,
        )
        if arguments is None:
            raise ValueError("arguments is required for LineageStep.")
        recognized = {"Actions", "Artifacts", "Contexts", "Associations"}
        if not recognized & set(arguments.keys()):
            raise ValueError(
                "LineageStep.arguments must contain at least one of: "
                + ", ".join(sorted(recognized))
            )
        self._arguments = arguments

        root = Properties(step_name=name, step=self)
        for field in ("ActionArns", "ArtifactArns", "ContextArns", "Associations"):
            root.__dict__[field] = Properties(step_name=name, path=field)
        self._properties = root

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block describing lineage entities and associations."""
        return self._arguments

    @property
    def properties(self):
        """Exposes ``ActionArns``, ``ArtifactArns``, ``ContextArns``, ``Associations``."""
        return self._properties
