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

Follows the ``step_args`` convention: create a lineage entity with the
corresponding class from :mod:`sagemaker.core.lineage` under a
:class:`~sagemaker.core.workflow.pipeline_context.PipelineSession` and
pass the returned step arguments to the step. Each step creates one
lineage entity; associations between entities created in different
steps reference them by ARN via step property references.

Example::

    pipeline_session = PipelineSession()

    action_args = Action.create(
        action_name="my-action",
        source_uri="s3://bucket/model.tar.gz",
        source_type="S3ETag",
        action_type="ModelTraining",
        status="Completed",
        sagemaker_session=pipeline_session,
    )
    action_step = LineageStep(name="RecordAction", step_args=action_args)

    association_args = Association.create(
        source_arn=action_step.properties.ActionArns["my-action"],
        destination_arn="arn:aws:sagemaker:...:artifact/abc",
        association_type="Produced",
        sagemaker_session=pipeline_session,
    )
    association_step = LineageStep(
        name="RecordAssociation",
        step_args=association_args,
        depends_on=[action_step],
    )
"""

from __future__ import absolute_import

from typing import List, Optional, Union

from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.pipeline_context import _JobStepArguments
from sagemaker.core.workflow.properties import Properties
from sagemaker.core.workflow.utilities import validate_step_args_input

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum

# Maps the captured lineage create call to the Arguments key the pipeline
# service expects.
_CALLER_TO_ARGUMENTS_KEY = {
    "create_action": "Actions",
    "create_artifact": "Artifacts",
    "create_context": "Contexts",
    "add_association": "Associations",
}


class _EntityArnMap(Properties):
    """Map-style property access for service-native ARN maps.

    The lineage step's ``ActionArns``/``ArtifactArns``/``ContextArns``
    outputs are maps keyed by entity name. They are pipeline-service
    outputs with no botocore shape, so this supports ``['name']`` access
    without a shape lookup.
    """

    def __getitem__(self, item: str) -> Properties:
        """Reference the ARN of the entity created under the given name."""
        return Properties(step_name=self.step_name, path=f"{self.path}['{item}']")


class LineageStep(Step):
    """Creates a lineage entity or association in SageMaker's lineage system.

    Wraps SageMaker's ``CreateAction``/``CreateArtifact``/``CreateContext``
    and lineage ``AddAssociation`` APIs. Each step creates one entity;
    the ``step_args`` must be obtained by calling ``Action.create()``,
    ``Artifact.create()``, ``Context.create()``, or
    ``Association.create()`` from :mod:`sagemaker.core.lineage` with a
    ``PipelineSession``.

    Property references expose the created entity:
    ``Steps.<StepName>.ActionArns['<name>']``,
    ``Steps.<StepName>.ArtifactArns['<name>']``,
    ``Steps.<StepName>.ContextArns['<name>']``, and
    ``Steps.<StepName>.Associations``.
    """

    def __init__(
        self,
        name: str,
        step_args: _JobStepArguments,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``LineageStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments): The arguments for this step,
                obtained from ``Action.create()``, ``Artifact.create()``,
                ``Context.create()``, or ``Association.create()`` called
                with a ``PipelineSession``.
            display_name (str): Optional display name.
            description (str): Optional description.
            depends_on (List[Union[str, Step, StepCollection]]): Optional
                explicit step dependencies.
        """
        super().__init__(
            name=name,
            display_name=display_name,
            description=description,
            step_type=StepTypeEnum.LINEAGE,
            depends_on=depends_on,
        )
        validate_step_args_input(
            step_args=step_args,
            expected_caller=set(_CALLER_TO_ARGUMENTS_KEY),
            error_message=(
                "The step_args of LineageStep must be obtained from "
                "Action.create(), Artifact.create(), Context.create(), or "
                "Association.create() called with a PipelineSession."
            ),
        )
        self.step_args = step_args

        root = Properties(step_name=name, step=self)
        for field in ("ActionArns", "ArtifactArns", "ContextArns"):
            root.__dict__[field] = _EntityArnMap(step_name=name, path=field)
        root.__dict__["Associations"] = Properties(step_name=name, path="Associations")
        self._properties = root

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block describing the lineage entity to create."""
        key = _CALLER_TO_ARGUMENTS_KEY[self.step_args.caller_name]
        entity = self.step_args.args
        if key == "Associations":
            # The AddAssociation API uses SourceArn/DestinationArn; the
            # pipeline service models associations as entity references.
            entity = {
                "Source": {"Arn": entity["SourceArn"]},
                "Destination": {"Arn": entity["DestinationArn"]},
            }
            if "AssociationType" in self.step_args.args:
                entity["AssociationType"] = self.step_args.args["AssociationType"]
        return {key: [entity]}

    @property
    def properties(self):
        """Exposes ``ActionArns``, ``ArtifactArns``, ``ContextArns``, ``Associations``."""
        return self._properties
