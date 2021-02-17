# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

from enum import Enum
from typing import Dict, List

import attr

from sagemaker.estimator import EstimatorBase, _TrainingJob
from sagemaker.inputs import (
    CreateModelInput,
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
from sagemaker.workflow.entities import (
    DefaultEnumMeta,
    Entity,
    RequestType,
)
from sagemaker.workflow.properties import (
    PropertyFile,
    Properties,
)


class StepTypeEnum(Enum, metaclass=DefaultEnumMeta):
    """Enum of step types."""

    CONDITION = "Condition"
    CREATE_MODEL = "Model"
    PROCESSING = "Processing"
    REGISTER_MODEL = "RegisterModel"
    TRAINING = "Training"
    TRANSFORM = "Transform"


@attr.s
class Step(Entity):
    """Pipeline step for workflow.

    Attributes:
        name (str): The name of the step.
        step_type (StepTypeEnum): The type of the step.
    """

    name: str = attr.ib(factory=str)
    step_type: StepTypeEnum = attr.ib(factory=StepTypeEnum.factory)

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
        return {
            "Name": self.name,
            "Type": self.step_type.value,
            "Arguments": self.arguments,
        }

    @property
    def ref(self) -> Dict[str, str]:
        """Gets a reference dict for steps"""
        return {"Name": self.name}


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


class TrainingStep(Step):
    """Training step for workflow."""

    def __init__(
        self,
        name: str,
        estimator: EstimatorBase,
        inputs: TrainingInput = None,
        cache_config: CacheConfig = None,
    ):
        """Construct a TrainingStep, given an `EstimatorBase` instance.

        In addition to the estimator instance, the other arguments are those that are supplied to
        the `fit` method of the `sagemaker.estimator.Estimator`.

        Args:
            name (str): The name of the training step.
            estimator (EstimatorBase): A `sagemaker.estimator.EstimatorBase` instance.
            inputs (TrainingInput): A `sagemaker.inputs.TrainingInput` instance. Defaults to `None`.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
        """
        super(TrainingStep, self).__init__(name, StepTypeEnum.TRAINING)
        self.estimator = estimator
        self.inputs = inputs
        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeTrainingJobResponse"
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_training_job`.

        NOTE: The CreateTrainingJob request is not quite the args list that workflow needs.
        The TrainingJobName and ExperimentConfig attributes cannot be included.
        """
        self.estimator.disable_profiler = True
        self.estimator.profiler_config = None
        self.estimator.profiler_rules = None

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


class CreateModelStep(Step):
    """CreateModel step for workflow."""

    def __init__(
        self,
        name: str,
        model: Model,
        inputs: CreateModelInput,
    ):
        """Construct a CreateModelStep, given an `sagemaker.model.Model` instance.

        In addition to the Model instance, the other arguments are those that are supplied to
        the `_create_sagemaker_model` method of the `sagemaker.model.Model._create_sagemaker_model`.

        Args:
            name (str): The name of the CreateModel step.
            model (Model): A `sagemaker.model.Model` instance.
            inputs (CreateModelInput): A `sagemaker.inputs.CreateModelInput` instance.
                Defaults to `None`.
        """
        super(CreateModelStep, self).__init__(name, StepTypeEnum.CREATE_MODEL)
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


class TransformStep(Step):
    """Transform step for workflow."""

    def __init__(
        self,
        name: str,
        transformer: Transformer,
        inputs: TransformInput,
        cache_config: CacheConfig = None,
    ):
        """Constructs a TransformStep, given an `Transformer` instance.

        In addition to the transformer instance, the other arguments are those that are supplied to
        the `transform` method of the `sagemaker.transformer.Transformer`.

        Args:
            name (str): The name of the transform step.
            transformer (Transformer): A `sagemaker.transformer.Transformer` instance.
            inputs (TransformInput): A `sagemaker.inputs.TransformInput` instance.
            cache_config (CacheConfig):  A `sagemaker.workflow.steps.CacheConfig` instance.
        """
        super(TransformStep, self).__init__(name, StepTypeEnum.TRANSFORM)
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


class ProcessingStep(Step):
    """Processing step for workflow."""

    def __init__(
        self,
        name: str,
        processor: Processor,
        inputs: List[ProcessingInput] = None,
        outputs: List[ProcessingOutput] = None,
        job_arguments: List[str] = None,
        code: str = None,
        property_files: List[PropertyFile] = None,
        cache_config: CacheConfig = None,
    ):
        """Construct a ProcessingStep, given a `Processor` instance.

        In addition to the processor instance, the other arguments are those that are supplied to
        the `process` method of the `sagemaker.processing.Processor`.

        Args:
            name (str): The name of the training step.
            processor (Processor): A `sagemaker.processing.Processor` instance.
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
        """
        super(ProcessingStep, self).__init__(name, StepTypeEnum.PROCESSING)
        self.processor = processor
        self.inputs = inputs
        self.outputs = outputs
        self.job_arguments = job_arguments
        self.code = code
        self.property_files = property_files

        # Examine why run method in sagemaker.processing.Processor mutates the processor instance
        # by setting the instance's arguments attribute. Refactor Processor.run, if possible.
        self.processor.arguments = job_arguments

        self._properties = Properties(
            path=f"Steps.{name}", shape_name="DescribeProcessingJobResponse"
        )
        self.cache_config = cache_config

    @property
    def arguments(self) -> RequestType:
        """The arguments dict that is used to call `create_processing_job`.

        NOTE: The CreateProcessingJob request is not quite the args list that workflow needs.
        ProcessingJobName and ExperimentConfig cannot be included in the arguments.
        """
        normalized_inputs, normalized_outputs = self.processor._normalize_args(
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
