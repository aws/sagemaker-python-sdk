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
"""The step definitions for the Cleanup step in SageMaker Pipelines."""

from __future__ import absolute_import

from typing import List, Optional, Union

from sagemaker.core.helper.pipeline_variable import PipelineVariable, RequestType
from sagemaker.core.workflow.step_outputs import StepOutput, get_step

from sagemaker.mlops.workflow.step_collections import StepCollection
from sagemaker.mlops.workflow.step_dependency import StepDependency
from sagemaker.mlops.workflow.steps import Step, StepTypeEnum


def _dependency_name(dep: Union[str, Step, StepCollection, StepOutput, StepDependency]) -> str:
    """The name of the step a `depends_on` entry refers to."""
    if isinstance(dep, StepDependency):
        return _dependency_name(dep.step)
    if isinstance(dep, str):
        return dep
    if isinstance(dep, StepCollection):
        # Matches how the pipeline resolves a StepCollection dependency.
        return dep.steps[-1].name
    if isinstance(dep, StepOutput):
        return get_step(dep).name
    return dep.name


class S3CleanupSource:
    """An upstream step whose Amazon S3 outputs a `CleanupStep` deletes.

    Omit `s3_uris` to let the service find the outputs from the source step
    (Training and Athena steps). Set `s3_uris` to delete exact objects
    written by a Processing or Transform step; each URI must be under that
    step's output location.
    """

    def __init__(
        self,
        step: Union[str, Step],
        s3_uris: Optional[List[Union[str, PipelineVariable]]] = None,
    ):
        """Constructs an `S3CleanupSource`.

        Args:
            step (str or Step): The source step, or its name.
            s3_uris (List[str or PipelineVariable]): Exact `s3://bucket/key`
                object URIs to delete. Omit for Training and Athena steps
                (default: None).
        """
        if s3_uris is not None and len(s3_uris) == 0:
            raise ValueError("s3_uris must be omitted or contain at least one URI.")
        self.step = step
        self.s3_uris = s3_uris

    @property
    def step_name(self) -> str:
        """The name of the source step."""
        return self.step.name if isinstance(self.step, Step) else self.step

    def to_request(self) -> RequestType:
        """Gets the request structure for this cleanup source."""
        request = {"StepName": self.step_name}
        if self.s3_uris is not None:
            request["S3Uris"] = list(self.s3_uris)
        return request


class CleanupStep(Step):
    """`CleanupStep` for SageMaker Pipelines Workflows.

    Deletes the Amazon S3 outputs of upstream steps. The pipeline execution
    role needs `s3:DeleteObject` on the target objects. A source step that
    sets a `VpcConfig` can't be cleaned up.
    """

    def __init__(
        self,
        name: str,
        s3_sources: List[S3CleanupSource],
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[
            List[Union[str, Step, StepCollection, StepOutput, StepDependency]]
        ] = None,
    ):
        """Constructs a `CleanupStep`.

        Args:
            name (str): The name of the `CleanupStep`.
            s3_sources (List[S3CleanupSource]): The steps whose S3 outputs
                to delete.
            display_name (str): The display name of the `CleanupStep`
                (default: None).
            description (str): The description of the `CleanupStep`
                (default: None).
            depends_on (List[str or Step or StepCollection or StepOutput or
                StepDependency]): Additional steps this `CleanupStep` depends
                on. Source steps not already listed are added with the
                default `Completed` trigger; list a source in a
                `StepDependency` to clean up on another outcome, for example
                `TriggerCondition.ANY` (default: None).
        """
        if not s3_sources:
            raise ValueError("s3_sources must contain at least one S3CleanupSource.")
        # The service can only delete outputs of steps that already ran, so the
        # sources are always dependencies, in addition to any the caller passes.
        depends_on = list(depends_on or [])
        dependency_names = {_dependency_name(dep) for dep in depends_on}
        for source in s3_sources:
            if source.step_name not in dependency_names:
                depends_on.append(source.step)
                dependency_names.add(source.step_name)
        super(CleanupStep, self).__init__(
            name, display_name, description, StepTypeEnum.CLEANUP, depends_on
        )
        self.s3_sources = s3_sources

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to define the `CleanupStep`."""
        return {"Resources": {"S3": [source.to_request() for source in self.s3_sources]}}

    @property
    def properties(self):
        """A `Properties` object is not available for the `CleanupStep`.

        Cleanup results are reported in the execution step metadata.
        """
        raise RuntimeError("The Properties object is not available for the CleanupStep.")
