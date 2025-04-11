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
"""Classes for compiling pipeline steps."""
from __future__ import absolute_import

import logging
import secrets
from typing import Sequence, Union, List

from sagemaker.workflow.entities import RequestType
from sagemaker.workflow.function_step import _FunctionStep
from sagemaker.workflow.steps import Step, StepTypeEnum, PropertyFile
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.step_collections import StepCollection
from sagemaker.workflow.step_outputs import get_step
from sagemaker.workflow.pipeline_definition_config import PipelineDefinitionConfig
from sagemaker.workflow.step_outputs import StepOutput
from sagemaker.workflow.utilities import (
    step_compilation_context_manager,
    get_config_hash,
    get_code_hash,
)
from sagemaker.utils import sagemaker_timestamp

logger = logging.getLogger(__name__)


class CompiledStep(Step):
    """A compiled step.

    A compiled step is a lightweight and somewhat immutable representation of a step.
    For example, invoking to_request() on a compiled step will not go through the heavy
    process of preparing the artifacts and generating the request dictionary.

    It can be compiled to pipeline definition and also directly used by
    the LocalPipelineExecutor.
    """

    def __init__(self, request_dict: RequestType):
        """Initialize a CompiledStep."""

        self._request_dict = request_dict
        super().__init__(
            name=request_dict["Name"],
            step_type=StepTypeEnum(request_dict["Type"]),
            depends_on=request_dict.get("DependsOn", []),
            description=request_dict.get("Description"),
            display_name=request_dict.get("DisplayName"),
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments to the particular `Step` service call."""
        return self._request_dict["Arguments"]

    @property
    def property_files(self) -> List[PropertyFile]:
        """The property files of a processing step."""
        return [
            PropertyFile(
                name=property_file["PropertyFileName"],
                output_name=property_file["OutputName"],
                path=property_file["FilePath"],
            )
            for property_file in self._request_dict.get("PropertyFiles", [])
        ]

    @property
    def properties(self):
        """The properties of the particular `Step`.

        This method is disabled for compiled steps.
        """
        raise NotImplementedError

    def add_depends_on(self, step_names: List[Union[str, "Step", "StepCollection", StepOutput]]):
        """Add `Step` names or `Step` instances to the current `Step` depends on list.

        This method is disabled for compiled steps.
        """
        raise NotImplementedError

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        return self._request_dict


class _StepsSet:
    """A simple ordered set to dedup steps."""

    def __init__(self):
        """Initialize a _StepSet."""
        self._steps = dict()
        self._steps_list = list()

    def __contains__(self, step: Step):
        """Check if a step is in the set."""
        return id(step) in self._steps

    def __getitem__(self, index):
        """Get a step by index."""
        return self._steps_list[index]

    def __len__(self):
        """Get the number of steps in the set."""
        return len(self._steps_list)

    def add(self, step: Step):
        """Add a step to the set."""
        if step not in self:
            self._steps_list.append(step)
        self._steps[id(step)] = step

    def add_list(self, steps: List[Step]):
        """Add a list of steps to the set."""
        for step in steps:
            self.add(step)


class _BuildQueue:
    """A FIFO queue for the steps to build."""

    def __init__(self):
        """Initialize a _BuildQueue."""
        self._steps_to_build = list()

    def push(self, steps: List[Step]):
        """Push a list of steps to the queue.

        Steps already queued will be ignored.
        """
        for step in steps:
            self._steps_to_build.append(step)

    def pop(self) -> Step:
        """Pop a step from the queue."""
        if self._steps_to_build:
            step = self._steps_to_build.pop(0)
            return step

        return None

    def __len__(self):
        """Return the length of the queue."""
        return len(self._steps_to_build)


class StepsCompiler(object):
    """Class used to compile steps. Should be instantiated once per compilation."""

    def __init__(
        self,
        pipeline_name: str,
        sagemaker_session,
        steps: Sequence[Union[Step, StepCollection, StepOutput]],
        pipeline_definition_config: PipelineDefinitionConfig = None,
    ):
        """Initialize a StepsCompiler.

        Args:
            pipeline_name (str): The name of the pipeline, passed down from pipeline.to_request()
            sagemaker_session (sagemaker.session.Session): The session object used for compile the
              steps.
            pipeline_definition_config (PipelineDefinitionConfig):
              A pipeline definition configuration for a pipeline containing feature flag toggles
        """
        self.pipeline_name = pipeline_name
        self.sagemaker_session = sagemaker_session
        self.pipeline_definition_config = pipeline_definition_config
        self.upload_runtime_scripts = True
        self.upload_workspace = True
        self.pipeline_build_time = sagemaker_timestamp()

        self._input_steps = list(steps)
        self._input_steps_map = dict()
        StepsCompiler._generate_step_map(self._input_steps, self._input_steps_map)

        self._all_known_steps = _StepsSet()
        self._build_queue = _BuildQueue()

        self._function_step_secret_token = secrets.token_hex(32)

        self._build_count = 0
        self._steps_need_json_serialization = set()

    @staticmethod
    def _generate_step_map(
        steps: Sequence[Union[Step, StepCollection, StepOutput]], step_map: dict
    ):
        """Helper method to create a mapping from Step/Step Collection name to itself."""
        for item in steps:
            if isinstance(item, StepOutput):
                step = get_step(item)
            else:
                step = item
            if step.name in step_map:
                raise ValueError(
                    "Pipeline steps cannot have duplicate names. In addition, steps added in "
                    "the ConditionStep cannot be added in the Pipeline steps list."
                )
            step_map[step.name] = step
            if isinstance(step, ConditionStep):
                StepsCompiler._generate_step_map(step.if_steps + step.else_steps, step_map)
            if isinstance(step, StepCollection):
                StepsCompiler._generate_step_map(step.steps, step_map)

    def _simplify_step_list(
        self,
        step_list: List[Union[str, Step, StepCollection, StepOutput]],
    ) -> List[Step]:
        """Simplify an input step list containing strings, StepCollections, StepOutput."""

        if not step_list:
            return []

        resolved_step_set = _StepsSet()
        for item in step_list:
            if isinstance(item, str):
                step_name = item
                step = self._input_steps_map.get(item)
                if step is None:
                    raise ValueError(
                        "The input steps do not contain the step of name: %s" % step_name
                    )
            elif isinstance(item, StepOutput):
                step = get_step(item)
            else:
                step = item

            if isinstance(step, Step):
                if step not in resolved_step_set:
                    resolved_step_set.add(step)
            elif isinstance(step, StepCollection):
                for sub_step in step.steps:
                    if sub_step not in resolved_step_set:
                        resolved_step_set.add(sub_step)
            else:
                raise ValueError(f"Invalid input step type: {type(step)}")

        return list(resolved_step_set)

    def _flatten_condition_step(self, step: ConditionStep) -> List[Step]:
        """Flatten a ConditionStep into a list of steps."""
        flattened_steps = _StepsSet()
        flattened_steps.add(step)
        sub_steps = self._simplify_step_list(step.if_steps + step.else_steps)
        for sub_step in sub_steps:
            if isinstance(sub_step, ConditionStep):
                flattened_steps.add_list(self._flatten_condition_step(sub_step))
            else:
                flattened_steps.add(sub_step)
        return list(flattened_steps)

    def _push_to_build_queue(self, steps: List[Step]):
        """Push steps to the build queue."""
        for step in steps:
            if step not in self._all_known_steps:
                self._build_queue.push([step])

            if isinstance(step, ConditionStep):
                self._all_known_steps.add_list(self._flatten_condition_step(step))
            else:
                self._all_known_steps.add(step)

    @staticmethod
    def get_upstream_steps_from_step_arguments(args: RequestType) -> List[Step]:
        """Get upstream steps from step arguments."""
        pipeline_variables = Step._find_pipeline_variables_in_step_arguments(args)
        upstream_steps = []
        for pipeline_variable in pipeline_variables:
            upstream_steps.extend(pipeline_variable._referenced_steps)

        return upstream_steps

    def _get_upstream_steps(self, step: Step, arguments: dict):
        """Get all upstream steps of the given step"""
        depends_on = _StepsSet()
        depends_on.add_list(self._simplify_step_list(step.depends_on))

        upstream_step_in_args = self._simplify_step_list(
            StepsCompiler.get_upstream_steps_from_step_arguments(arguments)
        )

        for upstream_step in upstream_step_in_args:
            if isinstance(upstream_step, _FunctionStep):
                self._steps_need_json_serialization.add(upstream_step.name)
                depends_on.add(upstream_step)

        return list(depends_on), upstream_step_in_args

    def _set_serialize_output_to_json_flag(self, compiled_steps: List[CompiledStep]):
        """Set the serialize_output_to_json to true in container arguments

        if the function step is data referenced by other steps
        """
        for step in compiled_steps:
            if isinstance(step, ConditionStep):
                self._set_serialize_output_to_json_flag(step.if_steps)
                self._set_serialize_output_to_json_flag(step.else_steps)
            elif step.name in self._steps_need_json_serialization:
                step_container_args = step._request_dict["Arguments"]["AlgorithmSpecification"][
                    "ContainerArguments"
                ]
                step_container_args.extend(["--serialize_output_to_json", "true"])

    def _build_step(self, step: Step) -> CompiledStep:
        """Build a step."""

        with step_compilation_context_manager(
            pipeline_name=self.pipeline_name,
            step_name=step.name,
            sagemaker_session=self.sagemaker_session,
            code_hash=get_code_hash(step),
            config_hash=get_config_hash(step),
            pipeline_definition_config=self.pipeline_definition_config,
            upload_runtime_scripts=self.upload_runtime_scripts,
            upload_workspace=self.upload_workspace,
            pipeline_build_time=self.pipeline_build_time,
            function_step_secret_token=self._function_step_secret_token,
        ) as context:
            request_dict = step.to_request()

            self.upload_runtime_scripts = context.upload_runtime_scripts
            self.upload_workspace = context.upload_workspace

            depends_on, upstream_steps_in_args = self._get_upstream_steps(
                step, request_dict["Arguments"]
            )
            if depends_on:
                request_dict["DependsOn"] = [s.name for s in depends_on]

            self._push_to_build_queue(depends_on)
            self._push_to_build_queue(upstream_steps_in_args)
        return CompiledStep(request_dict)

    def _build_condition_step(self, condition_step: ConditionStep) -> ConditionStep:
        """Build a condition step."""
        depends_on, upstream_steps_in_args = self._get_upstream_steps(
            condition_step, condition_step.step_only_arguments
        )
        self._push_to_build_queue(depends_on)
        self._push_to_build_queue(upstream_steps_in_args)

        compiled_condition_step = ConditionStep(
            name=condition_step.name,
            display_name=condition_step.display_name,
            description=condition_step.description,
            conditions=list(condition_step.conditions),
            if_steps=self._build_steps(condition_step.if_steps),
            else_steps=self._build_steps(condition_step.else_steps),
            depends_on=[step.name for step in depends_on],
        )

        return compiled_condition_step

    def _build_steps(self, steps: List[Union[Step, StepCollection, StepOutput]]):
        """Build a list of steps."""
        simple_steps = self._simplify_step_list(steps)

        compiled_steps = []
        for step in simple_steps:
            if isinstance(step, ConditionStep):
                compiled_steps.append(self._build_condition_step(step))
            else:
                compiled_steps.append(self._build_step(step))
        return compiled_steps

    def _initialize_queue_and_build(self, steps: List[Union[Step, StepCollection, StepOutput]]):
        """Build a list of steps."""

        simple_steps = self._simplify_step_list(steps)
        self._push_to_build_queue(simple_steps)

        compiled_steps = []
        while len(self._build_queue) > 0:
            step = self._build_queue.pop()
            if isinstance(step, ConditionStep):
                compiled_steps.append(self._build_condition_step(step))
            else:
                compiled_steps.append(self._build_step(step))

        self._set_serialize_output_to_json_flag(compiled_steps)
        return compiled_steps

    def build(self):
        """Build a list of steps.

        Returns:
            list: A request structure object for a service call for the list of pipeline steps
        """
        self._build_count += 1

        if self._build_count > 1:
            raise RuntimeError("Cannot build a pipeline more than once with the same compiler.")

        return self._initialize_queue_and_build(self._input_steps)
