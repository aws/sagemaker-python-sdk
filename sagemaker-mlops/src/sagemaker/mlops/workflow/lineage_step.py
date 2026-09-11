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

A single ``LineageStep`` records a batch of lineage entities and the
associations between them. Entities follow the ``step_args`` convention:
create each one with the corresponding class from
:mod:`sagemaker.core.lineage` under a
:class:`~sagemaker.core.workflow.pipeline_context.PipelineSession` and pass
the captured arguments to the step.

Associations are declared rather than captured. ``Association.create()``
takes source and destination ARNs, but an entity created by the same step
has no ARN until the step runs, so a sibling is referenced by name and
type instead. An association may also reference a pre-existing entity by
literal ARN.

Example::

    pipeline_session = PipelineSession()

    step = LineageStep(
        name="RecordLineage",
        step_args=[
            Action.create(
                action_name="training-run",
                source_uri="s3://bucket/run",
                source_type="S3ETag",
                action_type="ModelTraining",
                sagemaker_session=pipeline_session,
            ),
            Artifact.create(
                artifact_name="trained-model",
                source_uri="s3://bucket/model.tar.gz",
                artifact_type="Model",
                sagemaker_session=pipeline_session,
            ),
        ],
        associations=[
            LineageAssociation(
                source=LineageEntityReference(name="training-run", type="Action"),
                destination=LineageEntityReference(name="trained-model", type="Artifact"),
                association_type="Produced",
            ),
        ],
    )
"""

from __future__ import absolute_import

from typing import Dict, List, Optional, Union

from sagemaker.core.helper.pipeline_variable import PipelineVariable, RequestType
from sagemaker.core.workflow.pipeline_context import _JobStepArguments
from sagemaker.core.workflow.properties import Properties
from sagemaker.core.workflow.utilities import validate_step_args_input

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum

ENTITY_TYPE_ACTION = "Action"
ENTITY_TYPE_ARTIFACT = "Artifact"
ENTITY_TYPE_CONTEXT = "Context"

# Maps the captured lineage create call to the ``Arguments`` key the pipeline
# service expects, the entity type used when referencing the entity from an
# association, and the request field holding the entity name.
_ENTITY_SPECS = {
    "create_action": ("Actions", ENTITY_TYPE_ACTION, "ActionName"),
    "create_artifact": ("Artifacts", ENTITY_TYPE_ARTIFACT, "ArtifactName"),
    "create_context": ("Contexts", ENTITY_TYPE_CONTEXT, "ContextName"),
}


class LineageEntityReference:
    """Reference to a lineage entity, used as an association endpoint.

    Reference an entity created by the same ``LineageStep`` with ``name``
    and ``type``, or an entity that already exists with ``arn``.

    A name and type pair is resolved by the pipeline service against the
    entities created by that same step. It cannot refer to an entity
    created by a different step; use ``arn`` for anything created
    elsewhere.
    """

    def __init__(
        self,
        name: Optional[Union[str, PipelineVariable]] = None,
        type: Optional[str] = None,  # pylint: disable=redefined-builtin
        arn: Optional[Union[str, PipelineVariable]] = None,
    ):
        """Construct a ``LineageEntityReference``.

        Args:
            name (str or PipelineVariable): Name of an entity created by the
                same step. Must be given together with ``type``.
            type (str): One of ``"Action"``, ``"Artifact"`` or ``"Context"``.
            arn (str or PipelineVariable): ARN of an existing entity. Mutually
                exclusive with ``name``/``type``.
        """
        if arn is not None:
            if name is not None or type is not None:
                raise ValueError(
                    "A LineageEntityReference takes either arn, or name and type -- not both."
                )
        else:
            if name is None or type is None:
                raise ValueError(
                    "A LineageEntityReference requires either arn, or both name and type."
                )
            if type not in (ENTITY_TYPE_ACTION, ENTITY_TYPE_ARTIFACT, ENTITY_TYPE_CONTEXT):
                raise ValueError(
                    f"Unsupported lineage entity type '{type}'. Expected one of "
                    f"{ENTITY_TYPE_ACTION}, {ENTITY_TYPE_ARTIFACT}, {ENTITY_TYPE_CONTEXT}."
                )
        self.name = name
        self.type = type
        self.arn = arn

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        if self.arn is not None:
            return {"Arn": self.arn}
        return {"Name": self.name, "Type": self.type}


class LineageAssociation:
    """An association between two lineage entities."""

    def __init__(
        self,
        source: LineageEntityReference,
        destination: LineageEntityReference,
        association_type: Optional[str] = None,
    ):
        """Construct a ``LineageAssociation``.

        Args:
            source (LineageEntityReference): The source entity.
            destination (LineageEntityReference): The destination entity.
            association_type (str): The association type, for example
                ``ContributedTo``, ``AssociatedWith``, ``DerivedFrom`` or
                ``Produced``.
        """
        for role, ref in (("source", source), ("destination", destination)):
            if not isinstance(ref, LineageEntityReference):
                raise TypeError(
                    f"The {role} of a LineageAssociation must be a "
                    f"LineageEntityReference, got {type(ref).__name__}."
                )
        self.source = source
        self.destination = destination
        self.association_type = association_type

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request = {
            "Source": self.source.to_request(),
            "Destination": self.destination.to_request(),
        }
        if self.association_type is not None:
            request["AssociationType"] = self.association_type
        return request


class _EntityArnMap(Properties):
    """Map-style property access for the step's ARN outputs.

    ``ActionArns``, ``ArtifactArns`` and ``ContextArns`` are maps keyed by
    entity name. They are pipeline-service outputs with no botocore shape, so
    this supports ``['name']`` access without a shape lookup.
    """

    def __getitem__(self, item: str) -> Properties:
        """Reference the ARN of the entity created under the given name."""
        return Properties(step_name=self.step_name, path=f"{self.path}['{item}']")


class LineageStep(Step):
    """Records lineage entities and their associations in one pipeline step.

    Wraps SageMaker's ``CreateAction``, ``CreateArtifact``, ``CreateContext``
    and ``AddAssociation`` APIs. The pipeline service creates every entity in
    the step, then adds the associations between them, so associations can
    reference siblings by name and type.

    The step exposes the ARNs of what it created as
    ``Steps.<StepName>.ActionArns['<name>']``,
    ``Steps.<StepName>.ArtifactArns['<name>']`` and
    ``Steps.<StepName>.ContextArns['<name>']``, for downstream steps to
    consume. Note these cannot be used as an association endpoint in another
    ``LineageStep``: the service resolves name and type only against the
    entities created by the same step.
    """

    def __init__(
        self,
        name: str,
        step_args: Optional[Union[_JobStepArguments, List[_JobStepArguments]]] = None,
        associations: Optional[List[LineageAssociation]] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, StepCollection]]] = None,
    ):
        """Construct a ``LineageStep``.

        Args:
            name (str): The name of the step.
            step_args (_JobStepArguments or list): The captured arguments for
                each entity this step creates, obtained from
                ``Action.create()``, ``Artifact.create()`` or
                ``Context.create()`` called with a ``PipelineSession``. A
                single value is accepted for a step that creates one entity.
            associations (List[LineageAssociation]): Associations to add after
                the entities are created.
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

        if step_args is None:
            entities = []
        elif isinstance(step_args, (list, tuple)):
            entities = list(step_args)
        else:
            entities = [step_args]

        self.associations = list(associations) if associations else []

        if not entities and not self.associations:
            raise ValueError(
                "A LineageStep requires at least one entity in step_args, or one association."
            )

        for entity in entities:
            validate_step_args_input(
                step_args=entity,
                expected_caller=set(_ENTITY_SPECS),
                error_message=(
                    "The step_args of LineageStep must be obtained from Action.create(), "
                    "Artifact.create() or Context.create() called with a PipelineSession. "
                    "Associations are passed to the associations argument instead, because "
                    "Association.create() cannot reference an entity created by the same step."
                ),
            )

        for association in self.associations:
            if not isinstance(association, LineageAssociation):
                raise TypeError(
                    "Each entry in associations must be a LineageAssociation, got "
                    f"{type(association).__name__}."
                )

        self.step_args = entities
        self._validate_sibling_references()

        root = Properties(step_name=name, step=self)
        for field in ("ActionArns", "ArtifactArns", "ContextArns"):
            root.__dict__[field] = _EntityArnMap(step_name=name, path=field)
        root.__dict__["Associations"] = Properties(step_name=name, path="Associations")
        self._properties = root

    def _created_entities(self) -> Dict[str, str]:
        """Map the name of each entity created by this step to its type."""
        created = {}
        for entity in self.step_args:
            _, entity_type, name_field = _ENTITY_SPECS[entity.caller_name]
            name = entity.args.get(name_field)
            if isinstance(name, str):
                created[name] = entity_type
        return created

    def _validate_sibling_references(self) -> None:
        """Reject a name and type reference that no entity in this step creates.

        The pipeline service resolves a name and type pair only against the
        entities created by the same step, and fails the execution otherwise.
        Catching it here turns a runtime failure into a construction error.
        """
        created = self._created_entities()
        for association in self.associations:
            endpoints = (
                ("source", association.source),
                ("destination", association.destination),
            )
            for role, ref in endpoints:
                if ref.arn is not None or not isinstance(ref.name, str):
                    # An ARN needs no lookup, and a pipeline variable cannot be
                    # compared against the names known at construction time.
                    continue
                if created.get(ref.name) != ref.type:
                    raise ValueError(
                        f"The {role} of an association references {ref.type} '{ref.name}', "
                        f"which this step does not create. A name and type reference must "
                        f"name an entity created by the same LineageStep; use arn to "
                        f"reference an entity created elsewhere."
                    )

    @property
    def arguments(self) -> RequestType:
        """The ``Arguments`` block: the entities and associations for this step."""
        request: RequestType = {}
        for entity in self.step_args:
            key, _, _ = _ENTITY_SPECS[entity.caller_name]
            request.setdefault(key, []).append(entity.args)
        if self.associations:
            request["Associations"] = [a.to_request() for a in self.associations]
        return request

    @property
    def properties(self):
        """Exposes ``ActionArns``, ``ArtifactArns``, ``ContextArns``, ``Associations``."""
        return self._properties
