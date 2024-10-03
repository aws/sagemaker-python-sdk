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

import os

from typing import Optional, List, Union, Dict
from pydantic import BaseModel, ConfigDict

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
    DataSource,
    S3DataSource,
    FileSystemDataSource,
    VpcConfig,
)
from sagemaker.modules.utils import (
    _get_repo_name_from_image,
    _get_unique_name,
    _is_valid_path,
    _is_valid_s3_uri,
)
from sagemaker.modules.types import DataSourceType
from sagemaker.modules.constants import (
    DEFAULT_INSTANCE_TYPE,
    SOURCE_CODE_CONTAINER_PATH,
    SM_CODE_LOCAL_PATH,
    TRAIN_SCRIPT,
    DEFAULT_CONTAINER_ENTRYPOINT,
    DEFAULT_CONTAINER_ARGUMENTS,
)
from sagemaker.modules.templates import TRAIN_SCRIPT_TEMPLATE
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
            The SageMaker marketplace algorithm name/arn to use for the training job.
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
    training_input_mode: Optional[str] = "File"
    training_image_config: Optional[TrainingImageConfig] = None
    environment: Optional[Dict[str, str]] = None
    hyper_parameters: Optional[Dict[str, str]] = None
    vpc_config: Optional[VpcConfig] = None

    def _validate_training_image_and_algorithm_name(
        self, training_image: Optional[str], algorithm_name: Optional[str]
    ):
        """Validate that only one of 'training_image' or 'algorithm_name' is provided."""
        if not training_image and not algorithm_name:
            raise ValueError(
                "Atleast one of 'training_image' or 'algorithm_name' must be provided.",
            )
        if training_image and algorithm_name:
            raise ValueError(
                "Only one of 'training_image' or 'algorithm_name' must be provided.",
            )

    def _validate_source_code_config(self, source_code_config: Optional[SourceCodeConfig]):
        """Validate the source code configuration."""
        if source_code_config:
            if source_code_config.requirements or source_code_config.entry_script:
                source_dir = source_code_config.source_dir
                requirements = source_code_config.requirements
                entry_script = source_code_config.entry_script
                if not source_dir:
                    raise ValueError(
                        "If 'requirements' or 'entry_script' is provided in 'source_code_config', "
                        + "'source_dir' must also be provided.",
                    )
                if not _is_valid_path(source_dir, path_type="Directory"):
                    raise ValueError(
                        f"Invalid 'source_dir' path: {source_dir}. " + "Must be a valid directory.",
                    )
                if requirements:
                    if not _is_valid_path(
                        f"{source_dir}/{entry_script}",
                        path_type="File",
                    ):
                        raise ValueError(
                            f"Invalid 'requirements': {requirements}. "
                            + "Must be a valid file within the 'source_dir'.",
                        )
                if entry_script:
                    if not _is_valid_path(
                        f"{source_dir}/{entry_script}",
                        path_type="File",
                    ):
                        raise ValueError(
                            f"Invalid 'entry_script': {entry_script}. "
                            + "Must be a valid file within the 'source_dir'.",
                        )

    def __init__(self, **data):
        """Set default values for session, role, base_name, and other interdependent fields."""
        # Validate types with pydantic
        super().__init__(**data)

        self._validate_training_image_and_algorithm_name(self.training_image, self.algorithm_name)
        self._validate_source_code_config(self.source_code_config)

        if self.session is None:
            self.session = Session()
            logger.warning("Session not provided. Using default Session.")

        if self.role is None:
            self.role = get_execution_role()
            logger.warning(f"Role not provided. Using default role:\n{self.role}")

        if self.base_name is None:
            if self.algorithm_name:
                self.base_name = f"{self.algorithm_name}-job"
            elif self.training_image:
                if isinstance(self.training_image, ImageSpec):
                    self.base_name = f"{self.training_image.framework_name}-job"
                else:
                    self.base_name = f"{_get_repo_name_from_image(self.training_image)}-job"
            logger.warning(f"Base name not provided. Using default name:\n{self.base_name}")

        if self.resource_config is None:
            self.resource_config = ResourceConfig(
                volume_size_in_gb=30,
                instance_count=1,
                instance_type=DEFAULT_INSTANCE_TYPE,
                volume_kms_key_id=None,
                keep_alive_period_in_seconds=None,
                instance_groups=None,
            )
            logger.warning(f"ResourceConfig not provided. Using default:\n{self.resource_config}")

        if self.stopping_condition is None:
            self.stopping_condition = StoppingCondition(
                max_runtime_in_seconds=3600,
                max_pending_time_in_seconds=None,
                max_wait_time_in_seconds=None,
            )
            logger.warning(
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
            logger.warning(
                f"OutputDataConfig not provided. Using default:\n{self.output_data_config}"
            )

        if self.training_image:
            if isinstance(self.training_image, ImageSpec):
                self.training_image = self.training_image.get_image_uri(
                    image_scope="training", instance_type=self.resource_config.instance_type
                )
            logger.info(f"Training image URI: {self.training_image}")

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

        input_data_config = self._get_input_data_config(self.input_data_channels)

        container_entrypoint = None
        container_arguments = None
        if self.source_code_config:
            if not input_data_config:
                input_data_config = []

            # If source code is provided, create a channel for the source code
            # The source code will be mounted at /opt/ml/input/data/code in the container
            # and set as the working directory
            working_dir = ""
            if self.source_code_config.source_dir:
                source_code_channel = self.create_input_data_channel(
                    "code", self.source_code_config.source_dir
                )
                input_data_config.append(source_code_channel)
                working_dir = SOURCE_CODE_CONTAINER_PATH

            # Get the commands to execute in the training job container
            # and prepare the train.sh script
            commands = self._get_script_mode_command(self.source_code_config)
            self._prepare_train_script(
                command=commands,
                requirements=self.source_code_config.requirements,
                working_dir=working_dir,
            )

            # Create an input channel for scripts packaged by the sdk
            sm_scripts_channel = self.create_input_data_channel("sm_code", SM_CODE_LOCAL_PATH)
            input_data_config.append(sm_scripts_channel)

            # If source_code_config is provided, we will always use
            # the default container entrypoint and arguments
            # to execute the train.sh script.
            # Any commands generated from the source_code_config will be
            # executed from the train.sh script.
            container_entrypoint = DEFAULT_CONTAINER_ENTRYPOINT
            container_arguments = DEFAULT_CONTAINER_ARGUMENTS

        algorithm_specification = AlgorithmSpecification(
            algorithm_name=self.algorithm_name,
            training_image=self.training_image,
            training_input_mode=self.training_input_mode,
            training_image_config=self.training_image_config,
            container_entrypoint=container_entrypoint,
            container_arguments=container_arguments,
        )

        # Unfortunately, API requires hyperparameters to be strings
        string_hyper_parameters = {}
        if hyper_parameters:
            for hyper_parameter, value in hyper_parameters.items():
                string_hyper_parameters[hyper_parameter] = str(value)

        training_job = TrainingJob.create(
            session=self.session.boto_session,
            role_arn=self.role,
            training_job_name=_get_unique_name(self.base_name),
            algorithm_specification=algorithm_specification,
            resource_config=self.resource_config,
            stopping_condition=self.stopping_condition,
            input_data_config=input_data_config,
            output_data_config=self.output_data_config,
            environment=self.environment,
            hyper_parameters=string_hyper_parameters,
            vpc_config=self.vpc_config,
        )

        if wait:
            training_job.wait(logs=logs)
        if logs and not wait:
            logger.warning("Not displaing the training container logs as 'wait' is set to False.")

    def create_input_data_channel(self, channel_name: str, data_source: DataSourceType) -> Channel:
        """Create an input data channel for the training job.

        Args:
            channel_name (str): The name of the input data channel.
            data_source (DataSourceType): The data source for the input data channel.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
        """
        channel = None
        if isinstance(data_source, str):
            if _is_valid_s3_uri(data_source):
                channel = Channel(
                    channel_name=channel_name,
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=data_source,
                            s3_data_distribution_type="FullyReplicated",
                        ),
                    ),
                    input_mode="File",
                )
            elif _is_valid_path(data_source):
                s3_uri = self.session.upload_data(
                    path=data_source,
                    bucket=self.session.default_bucket(),
                    key_prefix=f"{self.base_name}/input/{channel_name}",
                )
                channel = Channel(
                    channel_name=channel_name,
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=s3_uri,
                            s3_data_distribution_type="FullyReplicated",
                        ),
                    ),
                    input_mode="File",
                )
            else:
                raise ValueError(f"Not a valid S3 URI or local file path: {data_source}.")
        elif isinstance(data_source, S3DataSource):
            channel = Channel(channel_name=channel_name, data_source=data_source)
        elif isinstance(data_source, FileSystemDataSource):
            channel = Channel(channel_name=channel_name, data_source=data_source)
        return channel

    def _get_input_data_config(
        self, input_data_channels: Optional[Union[List[Channel], Dict[str, DataSourceType]]]
    ) -> List[Channel]:
        """Get the input data configuration for the training job.

        Args:
            input_data_channels (Optional[Union[List[Channel], Dict[str, DataSourceType]]]):
                The input data channels for the training job.
                Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
        """
        if input_data_channels is None:
            return []

        if isinstance(input_data_channels, list):
            return input_data_channels
        return [
            self.create_input_data_channel(channel_name, data_source)
            for channel_name, data_source in input_data_channels.items()
        ]

    def _get_script_mode_command(self, source_code_config: SourceCodeConfig) -> str:
        """Get the command to execute in the training job container for script mode.

        Args:
            source_code_config (SourceCodeConfig): The source code configuration.
        """
        if source_code_config.command:
            if source_code_config.entry_script:
                logger.warning(
                    "Both 'command' and 'entry_script' are provided in the SourceCodeConfig. "
                    + "Defaulting to 'command'."
                )
            commands = source_code_config.command.split()
            return " ".join(commands)

        assert (
            source_code_config.entry_script
        ), "Either 'command' or 'entry_script' must be provided."

        if source_code_config.entry_script.endswith(".py"):
            return f"python {source_code_config.entry_script}"
        if source_code_config.entry_script.endswith(".sh"):
            return (
                f"chmod +x {source_code_config.entry_script} && "
                f"bash {source_code_config.entry_script}"
            )

        raise ValueError(
            f"Unsupported entry script: {source_code_config.entry_script}."
            + "Only .py and .sh scripts are supported."
        )

    def _prepare_train_script(
        self, command: str, requirements: Optional[str], working_dir: Optional[str]
    ):
        """Prepare the training script to be executed in the training job container.

        Args:
            command (str): The command to execute in the training job container.
            requirements (str): The path to the requirements file within the source code directory.
            working_dir (str): The working directory for the training job container
        """
        install_requirements = ""
        if requirements:
            install_requirements = "echo 'Installing requirements'\n"
            install_requirements = f"pip install -r {requirements}"

        if working_dir:
            working_dir = f"cd {working_dir}"

        train_script = TRAIN_SCRIPT_TEMPLATE.format(
            working_dir=working_dir,
            install_requirements=install_requirements,
            command=command,
        )

        os.makedirs(SM_CODE_LOCAL_PATH, exist_ok=True)
        with open(os.path.join(SM_CODE_LOCAL_PATH, TRAIN_SCRIPT), "w") as f:
            f.write(train_script)

    def get_training_image_uri(self) -> str:
        """Get the training image URI for the training job."""
        return self.training_image
