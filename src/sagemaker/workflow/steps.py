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

import abc
import warnings
from enum import Enum
from typing import Dict, List, Union
from urllib.parse import urlparse

import attr

from sagemaker.estimator import EstimatorBase, _TrainingJob
from sagemaker.inputs import (
    CompilationInput,
    CreateModelInput,
    FileSystemInput,
    TrainingInput,
    TransformInput,
)
from sagemaker.model import Model
from sagemaker.processing import (
    ProcessingInput,
    ProcessingJob,
    ProcessingOutput,
    Processor,
)
from sagemaker.transformer import Transformer, _TransformJob
from sagemaker.tuner import HyperparameterTuner, _TuningJob
from sagemaker.workflow.entities import DefaultEnumMeta, Entity, RequestType
from sagemaker.workflow.functions import Join
from sagemaker.workflow.properties import Properties, PropertyFile
from sagemaker.workflow.retry import RetryPolicy


class StepTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Enum of step types."""

    CONDITION = "Condition"
    CREATE_MODEL = "Model"
    PROCESSING = "Processing"
    REGISTER_MODEL = "RegisterModel"
    TRAINING = "Training"
    TRANSFORM = "Transform"
    CALLBACK = "Callback"
    TUNING = "Tuning"
    COMPILATION = "Compilation"
    LAMBDA = "Lambda"
    QUALITY_CHECK = "QualityCheck"
    CLARIFY_CHECK = "ClarifyCheck"


@attr.s
class Step(Entity):
    """Pipeline step for workflow.

    Attributes:
        name (str): The name of the step.
        display_name (str): The display name of the step.
        description (str): The description of the step.
        step_type (StepTypeEnum): The type of the step.
        depends_on (List[str] or List[Step]): The list of step names or step
            instances the current step depends on
        retry_policies (List[RetryPolicy]): The custom retry policy configuration
    """

    name: str = attr.ib(factory=str)
    display_name: str = attr.ib(default=None)
    description: str = attr.ib(default=None)
    step_type: StepTypeEnum = attr.ib(factory=StepTypeEnum.factory)
    depends_on: Union[List[str], List["Step"]] = attr.ib(default=None)

    @property
    @abc.abstractmethod
    def arguments(self) -> RequestType:
        """The arguments to the particular step service call."""

    @property
    @abc.abstractmethod
    def properties(self):
        """The properties of the particular step."""

    def to_request(self) -> RequestType:
        """Gets the request structure for workflow service calls."""
        request_dict = {
            "Name": self.name,
            "Type": self.step_type.value,
            "Arguments": self.arguments,
        }
        if self.depends_on:
            request_dict["DependsOn"] = self._resolve_depends_on(self.depends_on)
        if self.display_name:
            request_dict["DisplayName"] = self.display_name
        if self.description:
            request_dict["Description"] = self.description

        return request_dict

    def add_depends_on(self, step_names: Union[List[str], List["Step"]]):
        """Add step names or step instances to the current step depends on list"""

        if not step_names:
            return

        if not self.depends_on:
            self.depends_on = []
        self.depends_on.extend(step_names)

    @property
    def ref(self) -> Dict[str, str]:
        """Gets a reference dict for steps"""
        return {"Name": self.name}

    @staticmethod
    def _resolve_depends_on(depends_on_list: Union[List[str], List["Step"]]) -> List[str]:
        """Resolve the step depends on list"""
        depends_on = []
        for step in depends_on_list:
            if isinstance(step, Step):
                depends_on.append(step.name)
            elif isinstance(step, str):
                depends_on.append(step)
            else:
                raise ValueError(f"Invalid input step name: {step}")
        return depends_on


@attr.s
class CacheConfig:
    """Configuration class to enable caching in pipeline workflow.

    If caching is enabled, the pipeline attempts to find a previous execution of a step
    that was called with the same arguments. Step caching only considers successful execution.
    If a successful previous execution is found, the pipeline propagates the values
    from previous execution rather than recomputing the step. When multiple successful executions
    exist within the timeout period, it uses the result for the most recent successful execution.


    Attributes:
        enable_caching (bool): To enable step caching. Defaults to `False`.
        expire_after (str): If step caching is enabled, a timeout also needs to defined.
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
        """Configures caching in pipeline steps."""
        config = {"Enabled": self.enable_caching}
        if self.expire_after is not None:
            config["ExpireAfter"] = self.expire_after
        return {"CacheConfig": config}


class ConfigurableRetryStep(Step):
    """ConfigurableRetryStep step for workflow."""

    def __init__(
        self,
        name: str,
        step_type: StepTypeEnum,
        display_name: str = None,
        description: str = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
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
        """Add a retry policy to the current step retry policies list."""
        if not retry_policy:
            return

        if not self.retry_policies:
            self.retry_policies = []
        self.retry_policies.append(retry_policy)

    def to_request(self) -> RequestType:
        """Gets the request structure for ConfigurableRetryStep"""
        step_dict = super().to_request()
        if self.retry_policies:
            step_dict["RetryPolicies"] = self._resolve_retry_policy(self.retry_policies)
        return step_dict

    @staticmethod
    def _resolve_retry_policy(retry_policy_list: List[RetryPolicy]) -> List[RequestType]:
        """Resolve the step retry policy list"""
        return [retry_policy.to_request() for retry_policy in retry_policy_list]


class TrainingStep(ConfigurableRetryStep):
    """Training step for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        display_name: str = None,
        description: str = None,
        inputs: Union[TrainingInput, dict, str, FileSystemInput] = None,
        cache_config: CacheConfig = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a TrainingStep, given an `EstimatorBase` instance.

        In addition to the estimator instance, the other arguments are those that are supplied to
        the `fit` method of the `sagemaker.estimator.Estimator`.

        Args:
            name (str): The name of the training step.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            display_name (str): The display name of the training step.
            description (str): The description of the training step.
            inputs (Union[str, dict, TrainingInput, FileSystemInput]): Information
                about the training data. This can be one of three types:

                * (str) the S3 location where training data is saved, or a file:// path in
                  local mode.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) If using multiple
                  channels for training data, you can specify a dict mapping channel names to
                  strings or :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - channel configuration for S3 data sources
                  that can provide additional information as well as the path to the training
                  dataset.
                  See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.inputs.FileSystemInput) - channel configuration for
                  a file system data source that can provide additional information as well as
                  the path to the training dataset.

            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[str] or List[Step]): A list of step names or step instances
                this `sagemaker.workflow.steps.TrainingStep` depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
        """
        super(TrainingStep, self).__init__(
            name, StepTypeEnum.TRAINING, display_name, description, depends_on, retry_policies
        )
        self.estimator = estimator
        self.inputs = inputs
        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeTrainingJobResponse"
        )
        self.cache_config = cache_config

        if self.cache_config is not None and not self.estimator.disable_profiler:
            msg = (
                "Profiling is enabled on the provided estimator. "
                "The default profiler rule includes a timestamp "
                "which will change each time the pipeline is "
                "upserted, causing cache misses. If profiling "
                "is not needed, set disable_profiler to True on the estimator."
            )
            warnings.warn(msg)

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_training_job`.

        NOTE: The CreateTrainingJob request is not quite the args list that workflow needs.
        The TrainingJobName and ExperimentConfig attributes cannot be included.
        """

        self.estimator._prepare_for_training()
        train_args = _TrainingJob._get_train_args(
            self.estimator, self.inputs, experiment_config=dict()
        )
        request_dict = self.estimator.sagemaker_session._get_train_request(**train_args)
        request_dict.pop("TrainingJobName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict


class CreateModelStep(ConfigurableRetryStep):
    """CreateModel step for workflow."""

    def __init__(
        self,
        name: str,
        model: Model,
        inputs: CreateModelInput,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
        display_name: str = None,
        description: str = None,
    ):
        """Construct a CreateModelStep, given an `sagemaker.model.Model` instance.

        In addition to the Model instance, the other arguments are those that are supplied to
        the `_create_sagemaker_model` method of the `sagemaker.model.Model._create_sagemaker_model`.

        Args:
            name (str): The name of the CreateModel step.
            model (Model): A `sagemaker.model.Model` instance.
            inputs (CreateModelInput): A `sagemaker.inputs.CreateModelInput` instance.
                Defaults to `None`.
            depends_on (List[str] or List[Step]): A list of step names or step instances
                this `sagemaker.workflow.steps.CreateModelStep` depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
            display_name (str): The display name of the CreateModel step.
            description (str): The description of the CreateModel step.
        """
        super(CreateModelStep, self).__init__(
            name, StepTypeEnum.CREATE_MODEL, display_name, description, depends_on, retry_policies
        )
        self.model = model
        self.inputs = inputs or CreateModelInput()

        self._properties = Properties(path=f"Steps.{name}", shape_name="DescribeModelOutput")

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_model`.

        NOTE: The CreateModelRequest is not quite the args list that workflow needs.
        ModelName cannot be included in the arguments.
        """

        request_dict = self.model.sagemaker_session._create_model_request(
            name="",
            role=self.model.role,
            container_defs=self.model.prepare_container_def(
                instance_type=self.inputs.instance_type,
                accelerator_type=self.inputs.accelerator_type,
            ),
            vpc_config=self.model.vpc_config,
            enable_network_isolation=self.model.enable_network_isolation(),
        )
        request_dict.pop("ModelName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeModelResponse data model."""
        return self._properties


class TransformStep(ConfigurableRetryStep):
    """Transform step for workflow."""

    def __init__(
        self,
        name: str,
        transformer: Transformer,
        inputs: TransformInput,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Constructs a TransformStep, given an `Transformer` instance.

        In addition to the transformer instance, the other arguments are those that are supplied to
        the `transform` method of the `sagemaker.transformer.Transformer`.

        Args:
            name (str): The name of the transform step.
            transformer (Transformer): A `sagemaker.transformer.Transformer` instance.
            inputs (TransformInput): A `sagemaker.inputs.TransformInput` instance.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            display_name (str): The display name of the transform step.
            description (str): The description of the transform step.
            depends_on (List[str]): A list of step names this `sagemaker.workflow.steps.TransformStep`
                depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
        """
        super(TransformStep, self).__init__(
            name, StepTypeEnum.TRANSFORM, display_name, description, depends_on, retry_policies
        )
        self.transformer = transformer
        self.inputs = inputs
        self.cache_config = cache_config
        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeTransformJobResponse"
        )

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_transform_job`.

        NOTE: The CreateTransformJob request is not quite the args list that workflow needs.
        TransformJobName and ExperimentConfig cannot be included in the arguments.
        """
        transform_args = _TransformJob._get_transform_args(
            transformer=self.transformer,
            data=self.inputs.data,
            data_type=self.inputs.data_type,
            content_type=self.inputs.content_type,
            compression_type=self.inputs.compression_type,
            split_type=self.inputs.split_type,
            input_filter=self.inputs.input_filter,
            output_filter=self.inputs.output_filter,
            join_source=self.inputs.join_source,
            model_client_config=self.inputs.model_client_config,
            experiment_config=dict(),
        )

        request_dict = self.transformer.sagemaker_session._get_transform_request(**transform_args)
        request_dict.pop("TransformJobName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeTransformJobResponse data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict


class ProcessingStep(ConfigurableRetryStep):
    """Processing step for workflow."""

    def __init__(
        self,
        name: str,
        processor: Processor,
        display_name: str = None,
        description: str = None,
        inputs: List[ProcessingInput] = None,
        outputs: List[ProcessingOutput] = None,
        job_arguments: List[str] = None,
        code: str = None,
        property_files: List[PropertyFile] = None,
        cache_config: CacheConfig = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a ProcessingStep, given a `Processor` instance.

        In addition to the processor instance, the other arguments are those that are supplied to
        the `process` method of the `sagemaker.processing.Processor`.

        Args:
            name (str): The name of the processing step.
            processor (Processor): A `sagemaker.processing.Processor` instance.
            display_name (str): The display name of the processing step.
            description (str): The description of the processing step.
            inputs (List[ProcessingInput]): A list of `sagemaker.processing.ProcessorInput`
                instances. Defaults to `None`.
            outputs (List[ProcessingOutput]): A list of `sagemaker.processing.ProcessorOutput`
                instances. Defaults to `None`.
            job_arguments (List[str]): A list of strings to be passed into the processing job.
                Defaults to `None`.
            code (str): This can be an S3 URI or a local path to a file with the framework
                script to run. Defaults to `None`.
            property_files (List[PropertyFile]): A list of property files that workflow looks
                for and resolves from the configured processing output list.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[str] or List[Step]): A list of step names or step instance
                this `sagemaker.workflow.steps.ProcessingStep` depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
        """
        super(ProcessingStep, self).__init__(
            name, StepTypeEnum.PROCESSING, display_name, description, depends_on, retry_policies
        )
        self.processor = processor
        self.inputs = inputs
        self.outputs = outputs
        self.job_arguments = job_arguments
        self.code = code
        self.property_files = property_files
        self.job_name = None

        # Examine why run method in sagemaker.processing.Processor mutates the processor instance
        # by setting the instance's arguments attribute. Refactor Processor.run, if possible.
        self.processor.arguments = job_arguments

        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeProcessingJobResponse"
        )
        self.cache_config = cache_config

        if code:
            code_url = urlparse(code)
            if code_url.scheme == "" or code_url.scheme == "file":
                # By default, Processor will upload the local code to an S3 path
                # containing a timestamp. This causes cache misses whenever a
                # pipeline is updated, even if the underlying script hasn't changed.
                # To avoid this, hash the contents of the script and include it
                # in the job_name passed to the Processor, which will be used
                # instead of the timestamped path.
                self.job_name = self._generate_code_upload_path()

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_processing_job`.

        NOTE: The CreateProcessingJob request is not quite the args list that workflow needs.
        ProcessingJobName and ExperimentConfig cannot be included in the arguments.
        """
        normalized_inputs, normalized_outputs = self.processor._normalize_args(
            job_name=self.job_name,
            arguments=self.job_arguments,
            inputs=self.inputs,
            outputs=self.outputs,
            code=self.code,
        )

        process_args = ProcessingJob._get_process_args(
            self.processor, normalized_inputs, normalized_outputs, experiment_config=dict()
        )
        request_dict = self.processor.sagemaker_session._get_process_request(**process_args)
        request_dict.pop("ProcessingJobName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeProcessingJobResponse data model."""
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

    def _generate_code_upload_path(self) -> str:
        """Generate an upload path for local processing scripts based on its contents"""
        from sagemaker.workflow.utilities import hash_file

        code_hash = hash_file(self.code)
        return f"{self.name}-{code_hash}"[:1024]


class TuningStep(ConfigurableRetryStep):
    """Tuning step for workflow."""

    def __init__(
        self,
        name: str,
        tuner: HyperparameterTuner,
        display_name: str = None,
        description: str = None,
        inputs=None,
        job_arguments: List[str] = None,
        cache_config: CacheConfig = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
    ):
        """Construct a TuningStep, given a `HyperparameterTuner` instance.

        In addition to the tuner instance, the other arguments are those that are supplied to
        the `fit` method of the `sagemaker.tuner.HyperparameterTuner`.

        Args:
            name (str): The name of the tuning step.
            tuner (HyperparameterTuner): A `sagemaker.tuner.HyperparameterTuner` instance.
            display_name (str): The display name of the tuning step.
            description (str): The description of the tuning step.
            inputs: Information about the training data. Please refer to the
                ``fit()`` method of the associated estimator, as this can take
                any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str] or dict[str, sagemaker.inputs.TrainingInput]) -
                    If using multiple channels for training data, you can specify
                    a dict mapping channel names to strings or
                    :func:`~sagemaker.inputs.TrainingInput` objects.
                * (sagemaker.inputs.TrainingInput) - Channel configuration for S3 data sources
                    that can provide additional information about the training dataset.
                    See :func:`sagemaker.inputs.TrainingInput` for full details.
                * (sagemaker.session.FileSystemInput) - channel configuration for
                    a file system data source that can provide additional information as well as
                    the path to the training dataset.
                * (sagemaker.amazon.amazon_estimator.RecordSet) - A collection of
                    Amazon :class:~`Record` objects serialized and stored in S3.
                    For use with an estimator for an Amazon algorithm.
                * (sagemaker.amazon.amazon_estimator.FileSystemRecordSet) -
                    Amazon SageMaker channel configuration for a file system data source for
                    Amazon algorithms.
                * (list[sagemaker.amazon.amazon_estimator.RecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.RecordSet` objects,
                    where each instance is a different channel of training data.
                * (list[sagemaker.amazon.amazon_estimator.FileSystemRecordSet]) - A list of
                    :class:~`sagemaker.amazon.amazon_estimator.FileSystemRecordSet` objects,
                    where each instance is a different channel of training data.
            job_arguments (List[str]): A list of strings to be passed into the processing job.
                Defaults to `None`.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
            depends_on (List[str] or List[Step]): A list of step names or step instance
                this `sagemaker.workflow.steps.ProcessingStep` depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
        """
        super(TuningStep, self).__init__(
            name, StepTypeEnum.TUNING, display_name, description, depends_on, retry_policies
        )
        self.tuner = tuner
        self.inputs = inputs
        self.job_arguments = job_arguments
        self._properties = Properties(
            path=f"Steps.{name}",
            shape_names=[
                "DescribeHyperParameterTuningJobResponse",
                "ListTrainingJobsForHyperParameterTuningJobResponse",
            ],
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_hyper_parameter_tuning_job`.

        NOTE: The CreateHyperParameterTuningJob request is not quite the
            args list that workflow needs.
        The HyperParameterTuningJobName attribute cannot be included.
        """
        if self.tuner.estimator is not None:
            self.tuner.estimator._prepare_for_training()
        else:
            for _, estimator in self.tuner.estimator_dict.items():
                estimator._prepare_for_training()

        self.tuner._prepare_for_tuning()
        tuner_args = _TuningJob._get_tuner_args(self.tuner, self.inputs)
        request_dict = self.tuner.sagemaker_session._get_tuning_request(**tuner_args)
        request_dict.pop("HyperParameterTuningJobName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing

        `DescribeHyperParameterTuningJobResponse` and
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
        """Get the model artifact s3 uri from the top performing training jobs.

        Args:
            top_k (int): the index of the top performing training job
                tuning step stores up to 50 top performing training jobs, hence
                a valid top_k value is from 0 to 49. The best training job
                model is at index 0
            s3_bucket (str): the s3 bucket to store the training job output artifact
            prefix (str): the s3 key prefix to store the training job output artifact
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


class CompilationStep(ConfigurableRetryStep):
    """Compilation step for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        model: Model,
        inputs: CompilationInput = None,
        job_arguments: List[str] = None,
        depends_on: Union[List[str], List[Step]] = None,
        retry_policies: List[RetryPolicy] = None,
        display_name: str = None,
        description: str = None,
        cache_config: CacheConfig = None,
    ):
        """Construct a CompilationStep.

        Given an `EstimatorBase` and a `sagemaker.model.Model` instance construct a CompilationStep.

        In addition to the estimator and Model instances, the other arguments are those that are
        supplied to the `compile_model` method of the `sagemaker.model.Model.compile_model`.

        Args:
            name (str): The name of the compilation step.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            model (Model): A `sagemaker.model.Model` instance.
            inputs (CompilationInput): A `sagemaker.inputs.CompilationInput` instance.
                Defaults to `None`.
            job_arguments (List[str]): A list of strings to be passed into the processing job.
                Defaults to `None`.
            depends_on (List[str] or List[Step]): A list of step names or step instances
                this `sagemaker.workflow.steps.CompilationStep` depends on
            retry_policies (List[RetryPolicy]):  A list of retry policy
            display_name (str): The display name of the compilation step.
            description (str): The description of the compilation step.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
        """
        super(CompilationStep, self).__init__(
            name, StepTypeEnum.COMPILATION, display_name, description, depends_on, retry_policies
        )
        self.estimator = estimator
        self.model = model
        self.inputs = inputs
        self.job_arguments = job_arguments
        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeCompilationJobResponse"
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_compilation_job`.

        NOTE: The CreateTrainingJob request is not quite the args list that workflow needs.
        The TrainingJobName and ExperimentConfig attributes cannot be included.
        """

        compilation_args = self.model._get_compilation_args(self.estimator, self.inputs)
        request_dict = self.model.sagemaker_session._get_compilation_request(**compilation_args)
        request_dict.pop("CompilationJobName")

        return request_dict

    @property
    def properties(self):
        """A Properties object representing the DescribeTrainingJobResponse data model."""
        return self._properties

    def to_request(self) -> RequestType:
        """Updates the dictionary with cache configuration."""
        request_dict = super().to_request()
        if self.cache_config:
            request_dict.update(self.cache_config.config)

        return request_dict
