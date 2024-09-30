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
"""ModelTrainer class module."""
from __future__ import absolute_import

from typing import Optional, List, Union, Dict
from pydantic import BaseModel, ConfigDict, model_validator
from pydantic_core import PydanticCustomError

from sagemaker_core.resources import TrainingJob
from sagemaker_core.shapes import AlgorithmSpecification

from sagemaker import get_execution_role, Session
from sagemaker.modules.configs import (
    ResourceConfig,
    StoppingCondition,
    OutputDataConfig,
    SourceCodeConfig,
    TrainingImageConfig,
    Channel,
    VpcConfig,
)
from sagemaker.modules.utils import (
    _get_repo_name_from_image,
    _get_unique_name,
)
from sagemaker.modules.types import DataSourceType
from sagemaker.modules.constants import DEFAULT_INSTANCE_TYPE
from sagemaker.modules.image_spec import ImageSpec
from sagemaker.modules import logger


class ModelTrainer(BaseModel):
    """Class that trains a model using AWS SageMaker.

    Attributes:
        session (Optiona(Session)):
            The SageMaker session.
            If not specified, a new session will be created.
        role (Optional(str)):
            The IAM role ARN for the training job.
            If not specified, the default SageMaker execution role will be used.
        base_name (Optional[str]):
            The base name for the training job.
            If not specified, a default name will be generated using the algorithm name
            or training image.
        resource_config (Optional[ResourceConfig]):
            The resource configuration. This is used to specify the compute resources for
            the training job.
            If not specified, will default to 1 instance of ml.m5.xlarge.
        stopping_condition (Optional[StoppingCondition]):
            The stopping condition. This is used to specify the different stopping
            conditions for the training job.
            If not specified, will default to 1 hour max run time.
        output_data_config (Optional[OutputDataConfig]):
            The output data configuration. This is used to specify the output data location
            for the training job.
            If not specified, will default to s3://<default_bucket>/<base_name>/output/.
        input_data_channels (Optional[Union[List[Channel], Dict[str, DataSourceType]]]):
            The input data channels for the training job.
            Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
            DataSourceType can be an S3 URI string, local file path string,
            S3DataSource object, or FileSystemDataSource object.
        source_code_config (Optional[SourceCodeConfig]):
            The source code configuration. This is used to configure the source code for
            running the training job.
        algorithm_name (Optional[str]):
            The SageMaker marketplace algorithm name to use for the training job.
            algorithm_name cannot be specified if training_image is specified.
        training_image (Optional[Union[str, ImageSpec]]):
            The training image URI to use for the training job container.
            training_image cannot be specified if algorithm_name is specified.
        training_input_mode (Optional[str]):
            The input mode for the training job. Valid values are "Pipe", "File", "FastFile".
            Defaults to "File".
        training_image_config (Optional[TrainingImageConfig]:
            Training image Config. This is the configuration to use an image from a private
            Docker registry for a training job.
        environment (Optional[Dict[str, str]]):
            The environment variables for the training job.
        hyper_parameters (Optional[Dict[str, str]]):
            The hyperparameters for the training job.
        vpc_config: (Optional[VpcConfig]):
            The VPC configuration.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: Optional[Session] = None
    role: Optional[str] = None
    base_name: Optional[str] = None
    resource_config: Optional[ResourceConfig] = None
    stopping_condition: Optional[StoppingCondition] = None
    output_data_config: Optional[OutputDataConfig] = None
    input_data_channels: Optional[Union[List[Channel], Dict[str, DataSourceType]]] = None
    source_code_config: Optional[SourceCodeConfig] = None
    algorithm_name: Optional[str] = None
    training_image: Optional[Union[str, ImageSpec]] = None
    training_input_mode: Optional[str] = None
    training_image_config: Optional[TrainingImageConfig] = None
    environment: Optional[Dict[str, str]] = None
    hyper_parameters: Optional[Dict[str, str]] = None
    vpc_config: Optional[VpcConfig] = None

    @model_validator(mode="after")
    def _set_defaults(self):
        """Set default values for session, role, base_name, and other interdependent fields."""

        if not self.training_image and not self.algorithm_name:
            raise PydanticCustomError(
                "validation_error",
                "Atleast one of 'training_image' or 'algorithm_name' must be provided.",
            )

        if self.training_image and self.algorithm_name:
            raise PydanticCustomError(
                "validation_error",
                "Only one of 'training_image' or 'algorithm_name' must be provided.",
            )

        if self.session is None:
            self.session = Session()
            logger.warn("Session not provided. Using default Session.")

        if self.role is None:
            self.role = get_execution_role()
            logger.warn(f"Role not provided. Using default role:\n{self.role}")

        if self.base_name is None:
            if self.algorithm_name:
                self.base_name = self.algorithm_name
            elif self.training_image:
                self.base_name = _get_repo_name_from_image(self.training_image)

            logger.warn(f"Base name not provided. Using default name:\n{self.base_name}")

        if self.resource_config is None:
            self.resource_config = ResourceConfig(
                volume_size_in_gb=30,
                instance_count=1,
                instance_type=DEFAULT_INSTANCE_TYPE,
                volume_kms_key_id=None,
                keep_alive_period_in_seconds=None,
                instance_groups=None,
            )
            logger.warn(f"ResourceConfig not provided. Using default:\n{self.resource_config}")

        if self.stopping_condition is None:
            self.stopping_condition = StoppingCondition(
                max_runtime_in_seconds=3600,
                max_pending_time_in_seconds=None,
                max_wait_time_in_seconds=None,
            )
            logger.warn(
                f"StoppingCondition not provided. Using default:\n{self.stopping_condition}"
            )

        if self.output_data_config is None:
            session = self.session
            base_name = self.base_name
            self.output_data_config = OutputDataConfig(
                s3_output_path=f"s3://{session.default_bucket()}/{base_name}/output",
                compression_type="NONE",
                kms_key_id=None,
            )
            logger.warn(f"OutputDataConfig not provided. Using default:\n{self.output_data_config}")
        return self

    def train(
        self,
        input_data_channels: Optional[Union[List[Channel], Dict[str, DataSourceType]]] = None,
        source_code_config: Optional[SourceCodeConfig] = None,
        hyper_parameters: Optional[Dict[str, str]] = None,
        environment: Optional[Dict[str, str]] = None,
        wait: bool = True,
        logs: bool = True,
    ):
        """Train a model using AWS SageMaker.

        Args:
            input_data_channels (Optional[Union[List[Channel], Dict[str, DataSourceType]]]):
                The input data channels for the training job.
                Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
            source_code_config (Optional[SourceCodeConfig]):
                The source code configuration. This is used to configure the source code for
                running the training job.
            hyper_parameters (Optional[Dict[str, str]]):
                The hyperparameters for the training job.
            environment (Optional[Dict[str,str]]):
                The environment variables for the training job.
            wait (Optional[bool]):
                Whether to wait for the training job to complete before returning.
                Defaults to True.
            logs (Optional[bool]):
                Whether to display the training container logs while training.
                Defaults to True.
        """
        if input_data_channels:
            self.input_data_channels = input_data_channels
        if source_code_config:
            self.source_code_config = source_code_config
        if hyper_parameters:
            self.hyper_parameters = hyper_parameters
        if environment:
            self.environment = environment

        algorithm_specification = AlgorithmSpecification(
            algorithm_name=self.algorithm_name,
            training_image=self.training_image,  # TODO: Implement ImageSpec
            training_input_mode=self.training_input_mode,
            training_image_config=self.training_image_config,
        )

        if self.source_code_config:
            # TODO: Implement creating the run.sh script
            pass
        if self.input_data_channels:
            # TODO: Implement InputDataConfig
            pass

        training_job = TrainingJob.create(
            session=self.session.boto_session,
            role_arn=self.role,
            training_job_name=_get_unique_name(self.base_name),
            algorithm_specification=algorithm_specification,
            resource_config=self.resource_config,
            stopping_condition=self.stopping_condition,
            output_data_config=self.output_data_config,
            environment=self.environment,
            hyper_parameters=self.hyper_parameters,
            vpc_config=self.vpc_config,
        )

        if wait:
            training_job.wait(logs=logs)
        if logs and not wait:
            logger.warning("Not displaing the training container logs as 'wait' is set to False.")
