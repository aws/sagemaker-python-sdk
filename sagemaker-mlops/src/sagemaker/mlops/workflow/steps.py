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
"""The `Step` definitions for SageMaker Pipelines Workflows."""
from __future__ import absolute_import

import abc

from enum import Enum
from typing import Dict, List, Set, Union, Optional, Any, TYPE_CHECKING
import attr

from sagemaker.core.local.local_session import LocalSagemakerClient
# Primitive imports (stay in core)
from sagemaker.core.workflow.entities import Entity
from sagemaker.core.helper.pipeline_variable import RequestType
from sagemaker.core.workflow.pipeline_context import _JobStepArguments
from sagemaker.core.workflow.properties import (
    PropertyFile,
    Properties,
)
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.functions import Join, JsonGet
# Orchestration imports (now in mlops)
from sagemaker.mlops.workflow.retry import RetryPolicy
from sagemaker.core.workflow.step_outputs import StepOutput
from sagemaker.core.workflow.utilities import trim_request_dict
from sagemaker.core.processing import Processor

# Lazy import to avoid circular dependency
# ModelTrainer imports from core, and core.workflow imports ModelTrainer
if TYPE_CHECKING:
    from sagemaker.mlops.workflow.step_collections import StepCollection


class StepTypeEnum(Enum):
    """Enum of `Step` types."""

    CONDITION = "Condition"
    CREATE_MODEL = "Model"
    PROCESSING = "Processing"
    REGISTER_MODEL = "RegisterModel"
    TRAINING = "Training"
    TRANSFORM = "Transform"
    CALLBACK = "Callback"
    TUNING = "Tuning"
    LAMBDA = "Lambda"
    QUALITY_CHECK = "QualityCheck"
    CLARIFY_CHECK = "ClarifyCheck"
    EMR = "EMR"
    FAIL = "Fail"
    AUTOML = "AutoML"


class Step(Entity):
    """Pipeline `Step` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        step_type: StepTypeEnum = None,
        depends_on: Optional[List[Union[str, "Step", "StepCollection", StepOutput]]] = None,
    ):
        """Initialize a Step

        Args:
            name (str): The name of the `Step`.
            display_name (str): The display name of the `Step`.
            description (str): The description of the `Step`.
            step_type (StepTypeEnum): The type of the `Step`.
            depends_on (List[Union[str, Step, StepCollection]]): The list of `Step`/`StepCollection`
                names or `Step` or `StepCollection`, `StepOutput` instances that the current `Step`
                depends on.
        """
        self.name = name
        self.display_name = display_name
        self.description = description
        self.step_type = step_type
        if depends_on is not None:
            self._depends_on = depends_on
        else:
            self._depends_on = None

    @property
    def depends_on(self) -> Optional[List[Union[str, "Step", "StepCollection", StepOutput]]]:
        """The list of steps the current `Step` depends on."""

        return self._depends_on

    @depends_on.setter
    def depends_on(self, depends_on: List[Union[str, "Step", "StepCollection", StepOutput]]):
        """Set the list of  steps the current step explicitly depends on."""

        if depends_on is not None:
            self._depends_on = depends_on
        else:
            self._depends_on = None

    @property
    @abc.abstractmethod
    def arguments(self) -> RequestType:
        """The arguments to the particular `Step` service call."""

    @property
    def step_only_arguments(self) -> RequestType:
        """The arguments to this Step only.

        Compound Steps such as the ConditionStep will have to
        override this method to return arguments pertaining to only that step.
        """
        return self.arguments

    @property
    @abc.abstractmethod
    def properties(self):
        """The properties of the particular `Step`."""

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        request_dict = {
            "Name": self.name,
            "Type": self.step_type.value,
            "Arguments": self.arguments,
        }
        if self.depends_on:
            request_dict["DependsOn"] = list(self.depends_on)
        if self.display_name:
            request_dict["DisplayName"] = self.display_name
        if self.description:
            request_dict["Description"] = self.description

        return request_dict

    def add_depends_on(self, step_names: List[Union[str, "Step", "StepCollection", StepOutput]]):
        """Add `Step` names or `Step` instances to the current `Step` depends on list."""

        if not step_names:
            return

        if not self._depends_on:
            self._depends_on = []

        self._depends_on.extend(step_names)

    @property
    def ref(self) -> Dict[str, str]:
        """Gets a reference dictionary for `Step` instances."""
        return {"Name": self.name}

    # TODO: move this method to CompiledStep
    def _find_step_dependencies(
        self, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> List[str]:
        """Find all step names this step is dependent on."""
        step_dependencies = set()
        if self.depends_on:
            step_dependencies.update(self._find_dependencies_in_depends_on_list(step_map))
        step_dependencies.update(
            self._find_dependencies_in_step_arguments(self.step_only_arguments, step_map)
        )
        return list(step_dependencies)

    def _find_dependencies_in_depends_on_list(
        self, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> Set[str]:
        """Find dependency steps referenced in the depends-on field of this step."""
        # import here to prevent circular import
        from sagemaker.mlops.workflow.step_collections import StepCollection

        dependencies = set()
        for step in self.depends_on:
            if isinstance(step, Step):
                dependencies.add(step.name)
            elif isinstance(step, StepCollection):
                dependencies.add(step.steps[-1].name)
            elif isinstance(step, str):
                # step could be the name of a `Step` or a `StepCollection`
                dependencies.add(self._get_step_name_from_str(step, step_map))
        return dependencies

    def _find_dependencies_in_step_arguments(
        self, obj: Any, step_map: Dict[str, Union["Step", "StepCollection"]]
    ):
        """Find the step dependencies referenced in the arguments of this step."""
        dependencies = set()
        pipeline_variables = Step._find_pipeline_variables_in_step_arguments(obj)
        for pipeline_variable in pipeline_variables:
            for referenced_step in pipeline_variable._referenced_steps:
                if isinstance(referenced_step, Step):
                    dependencies.add(referenced_step.name)
                else:
                    dependencies.add(self._get_step_name_from_str(referenced_step, step_map))

            from sagemaker.core.workflow.function_step import DelayedReturn

            # TODO: we can remove the if-elif once move the validators to JsonGet constructor
            if isinstance(pipeline_variable, JsonGet):
                self._validate_json_get_function(pipeline_variable, step_map)
            elif isinstance(pipeline_variable, DelayedReturn):
                # DelayedReturn showing up in arguments, meaning that it's data referenced
                # We should convert it to JsonGet and validate the JsonGet object
                self._validate_json_get_function(pipeline_variable._to_json_get(), step_map)

        return dependencies

    def _validate_json_get_function(
        self, json_get: JsonGet, step_map: Dict[str, Union["Step", "StepCollection"]]
    ):
        """Validate the JsonGet function inputs."""
        if json_get.property_file:
            self._validate_json_get_property_file_reference(json_get=json_get, step_map=step_map)

    # TODO: move it to JsonGet constructor
    def _validate_json_get_property_file_reference(self, json_get: JsonGet, step_map: dict):
        """Validate the property file reference in JsonGet"""
        property_file_reference = json_get.property_file
        processing_step = step_map[json_get.step_name]
        property_file = None
        if isinstance(property_file_reference, str):
            if not processing_step.step_type == StepTypeEnum.PROCESSING:
                raise ValueError(
                    f"Invalid JsonGet function {json_get.expr} in step '{self.name}'. "
                    f"JsonGet function (with property_file) can only be evaluated "
                    f"on processing step outputs."
                )
            for file in processing_step.property_files:
                if file.name == property_file_reference:
                    property_file = file
                    break
        elif isinstance(property_file_reference, PropertyFile):
            property_file = property_file_reference
        if property_file is None:
            raise ValueError(
                f"Invalid JsonGet function {json_get.expr} in step '{self.name}'. Property file "
                f"reference '{property_file_reference}' is undefined in step "
                f"'{processing_step.name}'."
            )
        property_file_output = None
        if "ProcessingOutputConfig" in processing_step.arguments:
            for output in processing_step.arguments["ProcessingOutputConfig"]["Outputs"]:
                if output["OutputName"] == property_file.output_name:
                    property_file_output = output
        if property_file_output is None:
            raise ValueError(
                f"Processing output name '{property_file.output_name}' defined in property file "
                f"'{property_file.name}' not found in processing step '{processing_step.name}'."
            )

    @staticmethod
    def _find_pipeline_variables_in_step_arguments(obj: RequestType) -> List[PipelineVariable]:
        """Recursively find all the pipeline variables in the step arguments."""
        pipeline_variables = list()
        if isinstance(obj, dict):
            for value in obj.values():
                if isinstance(value, PipelineVariable):
                    pipeline_variables.append(value)
                else:
                    pipeline_variables.extend(
                        Step._find_pipeline_variables_in_step_arguments(value)
                    )
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, PipelineVariable):
                    pipeline_variables.append(item)
                else:
                    pipeline_variables.extend(Step._find_pipeline_variables_in_step_arguments(item))
        return pipeline_variables

    @staticmethod
    def _get_step_name_from_str(
        str_input: str, step_map: Dict[str, Union["Step", "StepCollection"]]
    ) -> str:
        """Convert a Step or StepCollection name input to step name."""
        from sagemaker.mlops.workflow.step_collections import StepCollection

        if str_input not in step_map:
            raise ValueError(f"Step {str_input} is undefined.")
        if isinstance(step_map[str_input], StepCollection):
            return step_map[str_input].steps[-1].name
        return str_input

    @staticmethod
    def _trim_experiment_config(request_dict: Dict):
        """For job steps, trim the experiment config to keep the trial component display name."""
        if request_dict.get("ExperimentConfig", {}).get("TrialComponentDisplayName"):
            request_dict["ExperimentConfig"] = {
                "TrialComponentDisplayName": request_dict["ExperimentConfig"][
                    "TrialComponentDisplayName"
                ]
            }
        else:
            request_dict.pop("ExperimentConfig", None)


@attr.s
class CacheConfig:
    """Configuration class to enable caching in SageMaker Pipelines Workflows.

    If caching is enabled, the pipeline attempts to find a previous execution of a `Step`
    that was called with the same arguments. `Step` caching only considers successful execution.
    If a successful previous execution is found, the pipeline propagates the values
    from the previous execution rather than recomputing the `Step`.
    When multiple successful executions exist within the timeout period,
    it uses the result for the most recent successful execution.


    Attributes:
        enable_caching (bool): To enable `Step` caching. Defaults to `False`.
        expire_after (str): If `Step` caching is enabled, a timeout also needs to defined.
            It defines how old a previous execution can be to be considered for reuse.
            Value should be an ISO 8601 duration string. Defaults to `None`.

            Examples::

                'p30d' # 30 days
                'P4DT12H' # 4 days and 12 hours
                'T12H' # 12 hours
    """

    enable_caching: bool = attr.ib(default=False)
    expire_after = attr.ib(
        default=None, validator=attr.validators.optional(attr.validators.instance_of(str))
    )

    @property
    def config(self):
        """Configures `Step` caching for SageMaker Pipelines Workflows."""
        config = {"Enabled": self.enable_caching}
        if self.expire_after is not None:
            config["ExpireAfter"] = self.expire_after
        return {"CacheConfig": config}


class ConfigurableRetryStep(Step):
    """`ConfigurableRetryStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_type: StepTypeEnum,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        depends_on: Optional[List[Union[str, Step, "StepCollection"]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        super().__init__(
            name=name,
            display_name=display_name,
            step_type=step_type,
            description=description,
            depends_on=depends_on,
        )
        self.retry_policies = [] if not retry_policies else retry_policies

    def add_retry_policy(self, retry_policy: RetryPolicy):
        """Add a policy to the current `ConfigurableRetryStep` retry policies list."""
        if not retry_policy:
            return

        if not self.retry_policies:
            self.retry_policies = []
        self.retry_policies.append(retry_policy)

    def to_request(self) -> RequestType:
        """Gets the request structure for `ConfigurableRetryStep`."""
        step_dict = super().to_request()
        if self.retry_policies:
            step_dict["RetryPolicies"] = self._resolve_retry_policy(self.retry_policies)
        return step_dict

    @staticmethod
    def _resolve_retry_policy(retry_policy_list: List[RetryPolicy]) -> List[RequestType]:
        """Resolve the `ConfigurableRetryStep` retry policy list."""
        return [retry_policy.to_request() for retry_policy in retry_policy_list]


class TrainingStep(ConfigurableRetryStep):
    """`TrainingStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: Optional[_JobStepArguments] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        cache_config: Optional[CacheConfig] = None,
        depends_on: Optional[List[Union[str, Step]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Construct a `TrainingStep` using step_args from model_trainer.train().

        Args:
            name (str): The name of the `TrainingStep`.
            step_args (_JobStepArguments): The arguments for the `TrainingStep` definition.
            display_name (str): The display name of the `TrainingStep`.
            description (str): The description of the `TrainingStep`.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step]]): A list of `Step`
                names or `Step` instances that this `TrainingStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
        """
        super(TrainingStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, depends_on, retry_policies
        )

        if step_args:
            from sagemaker.core.workflow.utilities import validate_step_args_input
            # Lazy import to avoid circular dependency
            from sagemaker.train.model_trainer import ModelTrainer

            validate_step_args_input(
                step_args=step_args,
                expected_caller={ModelTrainer.train.__name__},
                error_message="The step_args of TrainingStep must be obtained from model_trainer.train().",
            )

        self.step_args = step_args
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeTrainingJobResponse"
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_training_job`.

        NOTE: The `CreateTrainingJob` request is not quite the args list that workflow needs. 
        """
        from sagemaker.core.workflow.utilities import execute_job_functions
        from sagemaker.core.workflow.utilities import _pipeline_config

        if self.step_args:
            # execute fit function with saved parameters,
            # and store args in PipelineSession's _context
            execute_job_functions(self.step_args)

            # populate request dict with args
            model_trainer = self.step_args.func_args[0]
            request_dict = model_trainer.sagemaker_session.context.args
        else:
            raise ValueError("step_args input is required.")
        
        if "HyperParameters" in request_dict:
            request_dict["HyperParameters"].pop("sagemaker_job_name", None)

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "TrainingJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeTrainingJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the request dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict


class TransformStep(ConfigurableRetryStep):
    """`TransformStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: Optional[_JobStepArguments] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        cache_config: Optional[CacheConfig] = None,
        depends_on: Optional[List[Union[str, Step]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Constructs a `TransformStep`, given a `Transformer` instance.

        In addition to the `Transformer` instance, the other arguments are those
        that are supplied to the `transform` method of the `sagemaker.transformer.Transformer`.

        Args:
            name (str): The name of the `TransformStep`.
            step_args (_JobStepArguments): The arguments for the `TransformStep` definition.
            cache_config (CacheConfig): A `sagemaker.workflow.steps.CacheConfig` instance.
            display_name (str): The display name of the `TransformStep`.
            description (str): The description of the `TransformStep`.
            depends_on (List[Union[str, Step]]): A list of `Step`
                names or `Step` instances that this `TransformStep`
                depends on.
            retry_policies (List[RetryPolicy]): A list of retry policies.
        """
        super(TransformStep, self).__init__(
            name, StepTypeEnum.TRANSFORM, display_name, description, depends_on, retry_policies
        )

        if not step_args:
            raise ValueError("step_args is required for TransformStep.")

        from sagemaker.core.workflow.utilities import validate_step_args_input

        validate_step_args_input(
            step_args=step_args,
            expected_caller={"transform", LocalSagemakerClient().create_transform_job.__name__},
            error_message="The step_args of TransformStep must be obtained "
            "from transformer.transform().",
        )

        self.step_args = step_args
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeTransformJobResponse"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_transform_job`.

        NOTE: The `CreateTransformJob` request is not quite the args list that workflow needs.
        `ExperimentConfig` cannot be included in the arguments.
        """
        from sagemaker.core.workflow.utilities import execute_job_functions
        from sagemaker.core.workflow.utilities import _pipeline_config

        # execute transform function with saved parameters,
        # and store args in PipelineSession's _context
        execute_job_functions(self.step_args)

        # populate request dict with args
        transformer = self.step_args.func_args[0]
        request_dict = transformer.sagemaker_session.context.args

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "TransformJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeTransformJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict


class ProcessingStep(ConfigurableRetryStep):
    """`ProcessingStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: Optional[_JobStepArguments] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        property_files: Optional[List[PropertyFile]] = None,
        cache_config: Optional[CacheConfig] = None,
        depends_on: Optional[List[Union[str, Step]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Construct a `ProcessingStep`, given a `Processor` instance.

        In addition to the `Processor` instance, the other arguments are those that are supplied to
        the `process` method of the `sagemaker.processing.Processor`.

        Args:
            name (str): The name of the `ProcessingStep`.
            step_args (_JobStepArguments): The arguments for the `ProcessingStep` definition.
            display_name (str): The display name of the `ProcessingStep`.
            description (str): The description of the `ProcessingStep`
            property_files (List[PropertyFile]): A list of property files that workflow looks
                for and resolves from the configured processing output list.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step]]): A list of `Step`
                names or `Step` instances that this `ProcessingStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
        """
        super(ProcessingStep, self).__init__(
            name, StepTypeEnum.PROCESSING, display_name, description, depends_on, retry_policies
        )

        if not step_args:
            raise ValueError("step_args is required for ProcessingStep.")

        from sagemaker.core.workflow.utilities import validate_step_args_input
        

        validate_step_args_input(
            step_args=step_args,
            expected_caller={Processor.run.__name__, LocalSagemakerClient().create_processing_job.__name__},
            error_message=f"The step_args of ProcessingStep must be obtained from processor.run() or in local mode, not {step_args.caller_name}",
        )

        self.step_args = step_args
        self.property_files = property_files or []
        self.cache_config = cache_config
        self._properties = Properties(
            step_name=name, step=self, shape_name="DescribeProcessingJobResponse"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_processing_job`.

        NOTE: The `CreateProcessingJob` request is not quite the args list that workflow needs.
        `ExperimentConfig` cannot be included in the arguments.
        """
        from sagemaker.core.workflow.utilities import execute_job_functions
        from sagemaker.core.workflow.utilities import _pipeline_config

        # execute run function with saved parameters,
        # and store args in PipelineSession's _context
        execute_job_functions(self.step_args)

        # populate request dict with args
        processor = self.step_args.func_args[0]
        request_dict = processor.sagemaker_session.context.args
            
        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(request_dict, "ProcessingJobName", _pipeline_config)

        Step._trim_experiment_config(request_dict)

        return request_dict

    @property
    def properties(self):
        """A `Properties` object representing the `DescribeProcessingJobResponse` data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Get the request structure for workflow service calls."""
        request_dict = super(ProcessingStep, self).to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)
        if self.property_files:
            request_dict["PropertyFiles"] = [
                property_file.expr for property_file in self.property_files
            ]
        return request_dict




class TuningStep(ConfigurableRetryStep):
    """`TuningStep` for SageMaker Pipelines Workflows."""

    def __init__(
        self,
        name: str,
        step_args: Optional[_JobStepArguments] = None,
        display_name: Optional[str] = None,
        description: Optional[str] = None,
        cache_config: Optional[CacheConfig] = None,
        depends_on: Optional[List[Union[str, Step]]] = None,
        retry_policies: Optional[List[RetryPolicy]] = None,
    ):
        """Construct a `TuningStep`, given a `HyperparameterTuner` instance.

        In addition to the `HyperparameterTuner` instance, the other arguments are those
        that are supplied to the `fit` method of the `sagemaker.tuner.HyperparameterTuner`.

        Args:
            name (str): The name of the `TuningStep`.
            step_args (_JobStepArguments): The arguments for the `TuningStep` definition.
            display_name (str): The display name of the `TuningStep`.
            description (str): The description of the `TuningStep`.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[Union[str, Step]]): A list of `Step`
                names or `Step` instances that this `TuningStep`
                depends on.
            retry_policies (List[RetryPolicy]):  A list of retry policies.
        """
        super(TuningStep, self).__init__(
            name, StepTypeEnum.TUNING, display_name, description, depends_on, retry_policies
        )

        if not step_args :
            raise ValueError("step_args is required for TuningStep.")

        from sagemaker.core.workflow.utilities import validate_step_args_input

        validate_step_args_input(
            step_args=step_args,
            expected_caller={"tune"},
            error_message="The step_args of TuningStep must be obtained from tuner.tune().",
        )

        self.step_args = step_args
        self._properties = Properties(
            step_name=name,
            step=self,
            shape_names=[
                "DescribeHyperParameterTuningJobResponse",
                "ListTrainingJobsForHyperParameterTuningJobResponse",
            ],
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dictionary that is used to call `create_hyper_parameter_tuning_job`.

        NOTE: The `CreateHyperParameterTuningJob` request is not quite the
            args list that workflow needs.
        """
        from sagemaker.core.workflow.utilities import execute_job_functions
        from sagemaker.core.workflow.utilities import _pipeline_config

        # execute fit function with saved parameters,
        # and store args in PipelineSession's _context
        execute_job_functions(self.step_args)

        # populate request dict with args
        tuner = self.step_args.func_args[0]
        request_dict = tuner.sagemaker_session.context.args
        

        # Continue to pop job name if not explicitly opted-in via config
        request_dict = trim_request_dict(
            request_dict, "HyperParameterTuningJobName", _pipeline_config
        )

        return request_dict

    @property
    def properties(self):
        """A `Properties` object

        A `Properties` object representing `DescribeHyperParameterTuningJobResponse` and
        `ListTrainingJobsForHyperParameterTuningJobResponse` data model.
        """
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict

    def get_top_model_s3_uri(self, top_k: int, s3_bucket: str, prefix: str = "") -> Join:
        """Get the model artifact S3 URI from the top performing training jobs.

        Args:
            top_k (int): The index of the top performing training job
                tuning step stores up to 50 top performing training jobs.
                A valid top_k value is from 0 to 49. The best training job
                model is at index 0.
            s3_bucket (str): The S3 bucket to store the training job output artifact.
            prefix (str): The S3 key prefix to store the training job output artifact.
        """
        values = ["s3:/", s3_bucket]
        if prefix != "" and prefix is not None:
            values.append(prefix)

        return Join(
            on="/",
            values=values
            + [
                self.properties.TrainingJobSummaries[top_k].TrainingJobName,
                "output/model.tar.gz",
            ],
        )