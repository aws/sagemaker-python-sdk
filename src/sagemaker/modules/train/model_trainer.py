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

from enum import Enum
import os
import json
import shutil
from tempfile import TemporaryDirectory

from typing import Optional, List, Union, Dict, Any, ClassVar

from graphene.utils.str_converters import to_camel_case, to_snake_case

from sagemaker_core.main import resources
from sagemaker_core.resources import TrainingJob
from sagemaker_core.shapes import AlgorithmSpecification

from pydantic import BaseModel, ConfigDict, PrivateAttr, validate_call

from sagemaker.config.config_schema import (
    _simple_path,
    SAGEMAKER,
    MODEL_TRAINER,
    MODULES,
    PYTHON_SDK,
    TRAINING_JOB_ENVIRONMENT_PATH,
    TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
    TRAINING_JOB_VPC_CONFIG_PATH,
    TRAINING_JOB_SUBNETS_PATH,
    TRAINING_JOB_SECURITY_GROUP_IDS_PATH,
    TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH,
    TRAINING_JOB_RESOURCE_CONFIG_PATH,
    TRAINING_JOB_ROLE_ARN_PATH,
    TRAINING_JOB_TAGS_PATH,
)

from sagemaker.utils import resolve_value_from_config
from sagemaker.modules import Session, get_execution_role
from sagemaker.modules.configs import (
    Compute,
    StoppingCondition,
    RetryStrategy,
    OutputDataConfig,
    SourceCode,
    TrainingImageConfig,
    Channel,
    DataSource,
    S3DataSource,
    FileSystemDataSource,
    Networking,
    Tag,
    InfraCheckConfig,
    RemoteDebugConfig,
    SessionChainingConfig,
    TensorBoardOutputConfig,
    CheckpointConfig,
    InputData,
)

from sagemaker.modules.local_core.local_container import _LocalContainer
from sagemaker.modules.distributed import Torchrun, MPI, DistributedConfig
from sagemaker.modules.utils import (
    _get_repo_name_from_image,
    _get_unique_name,
    _is_valid_path,
    _is_valid_s3_uri,
    safe_serialize,
)
from sagemaker.modules.types import DataSourceType
from sagemaker.modules.constants import (
    DEFAULT_INSTANCE_TYPE,
    SM_CODE,
    SM_CODE_CONTAINER_PATH,
    SM_DRIVERS,
    SM_DRIVERS_LOCAL_PATH,
    TRAIN_SCRIPT,
    DEFAULT_CONTAINER_ENTRYPOINT,
    DEFAULT_CONTAINER_ARGUMENTS,
    SOURCE_CODE_JSON,
    DISTRIBUTED_JSON,
)
from sagemaker.modules.templates import (
    TRAIN_SCRIPT_TEMPLATE,
    EXECUTE_BASE_COMMANDS,
    EXECUTE_MPI_DRIVER,
    EXEUCTE_TORCHRUN_DRIVER,
    EXECUTE_BASIC_SCRIPT_DRIVER,
)
from sagemaker.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.telemetry.constants import Feature
from sagemaker.modules import logger
from sagemaker.modules.train.sm_recipes.utils import _get_args_from_recipe, _determine_device_type


class Mode(Enum):
    """Enum class for training mode."""

    LOCAL_CONTAINER = "LOCAL_CONTAINER"
    SAGEMAKER_TRAINING_JOB = "SAGEMAKER_TRAINING_JOB"


class ModelTrainer(BaseModel):
    """Class that trains a model using AWS SageMaker.

    Example:
    ```python
    from sagemaker.modules.train import ModelTrainer
    from sagemaker.modules.configs import SourceCode, Compute, InputData

    source_code = SourceCode(source_dir="source", entry_script="train.py")
    training_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-training-image"
    model_trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code,
    )

    train_data = InputData(channel_name="train", data_source="s3://bucket/train")
    model_trainer.train(input_data_config=[train_data])
    ```

    Attributes:
        sagemaker_session (Optiona(Session)):
            The SageMakerCore session. For convinience, can be imported like:
            `from sagemaker.modules import Session`.
            If not specified, a new session will be created.
            If the default bucket for the artifacts needs to be updated, it can be done by
            passing it in the Session object.
        role (Optional(str)):
            The IAM role ARN for the training job.
            If not specified, the default SageMaker execution role will be used.
        base_job_name (Optional[str]):
            The base name for the training job.
            If not specified, a default name will be generated using the algorithm name
            or training image.
        source_code (Optional[SourceCode]):
            The source code configuration. This is used to configure the source code for
            running the training job.
        distributed (Optional[Union[MPI, Torchrun]]):
            The distributed runner for the training job. This is used to configure
            a distributed training job. If specifed, `source_code` must also
            be provided.
        compute (Optional[Compute]):
            The compute configuration. This is used to specify the compute resources for
            the training job. If not specified, will default to 1 instance of ml.m5.xlarge.
        networking (Optional[Networking]):
            The networking configuration. This is used to specify the networking settings
            for the training job.
        stopping_condition (Optional[StoppingCondition]):
            The stopping condition. This is used to specify the different stopping
            conditions for the training job.
            If not specified, will default to 1 hour max run time.
        algorithm_name (Optional[str]):
            The SageMaker marketplace algorithm name/arn to use for the training job.
            algorithm_name cannot be specified if training_image is specified.
        training_image (Optional[str]):
            The training image URI to use for the training job container.
            training_image cannot be specified if algorithm_name is specified.
            To find available sagemaker distributed images,
            see: https://docs.aws.amazon.com/sagemaker/latest/dg-ecr-paths/sagemaker-algo-docker-registry-paths
        training_image_config (Optional[TrainingImageConfig]):
            Training image Config. This is the configuration to use an image from a private
            Docker registry for a training job.
        output_data_config (Optional[OutputDataConfig]):
            The output data configuration. This is used to specify the output data location
            for the training job.
            If not specified in the session, will default to
            `s3://<default_bucket>/<default_prefix>/<base_job_name>/`.
        input_data_config (Optional[List[Union[Channel, InputData]]]):
            The input data config for the training job.
            Takes a list of Channel or InputData objects. An InputDataSource can be an S3 URI
            string, local file path string, S3DataSource object, or FileSystemDataSource object.
        checkpoint_config (Optional[CheckpointConfig]):
            Contains information about the output location for managed spot training checkpoint
            data.
        training_input_mode (Optional[str]):
            The input mode for the training job. Valid values are "Pipe", "File", "FastFile".
            Defaults to "File".
        environment (Optional[Dict[str, str]]):
            The environment variables for the training job.
        hyperparameters (Optional[Dict[str, Any]]):
            The hyperparameters for the training job.
        tags (Optional[List[Tag]]):
            An array of key-value pairs. You can use tags to categorize your AWS resources
            in different ways, for example, by purpose, owner, or environment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    training_mode: Mode = Mode.SAGEMAKER_TRAINING_JOB
    sagemaker_session: Optional[Session] = None
    role: Optional[str] = None
    base_job_name: Optional[str] = None
    source_code: Optional[SourceCode] = None
    distributed: Optional[Union[MPI, Torchrun]] = None
    compute: Optional[Compute] = None
    networking: Optional[Networking] = None
    stopping_condition: Optional[StoppingCondition] = None
    training_image: Optional[str] = None
    training_image_config: Optional[TrainingImageConfig] = None
    algorithm_name: Optional[str] = None
    output_data_config: Optional[OutputDataConfig] = None
    input_data_config: Optional[List[Union[Channel, InputData]]] = None
    checkpoint_config: Optional[CheckpointConfig] = None
    training_input_mode: Optional[str] = "File"
    environment: Optional[Dict[str, str]] = {}
    hyperparameters: Optional[Dict[str, Any]] = {}
    tags: Optional[List[Tag]] = None
    local_container_root: Optional[str] = os.getcwd()

    # Created Artifacts
    _latest_training_job: Optional[resources.TrainingJob] = PrivateAttr(default=None)

    # Private TrainingJob Parameters
    _tensorboard_output_config: Optional[TensorBoardOutputConfig] = PrivateAttr(default=None)
    _retry_strategy: Optional[RetryStrategy] = PrivateAttr(default=None)
    _infra_check_config: Optional[InfraCheckConfig] = PrivateAttr(default=None)
    _session_chaining_config: Optional[SessionChainingConfig] = PrivateAttr(default=None)
    _remote_debug_config: Optional[RemoteDebugConfig] = PrivateAttr(default=None)

    _temp_recipe_train_dir: Optional[TemporaryDirectory] = PrivateAttr(default=None)

    CONFIGURABLE_ATTRIBUTES: ClassVar[List[str]] = [
        "role",
        "base_job_name",
        "source_code",
        "compute",
        "networking",
        "stopping_condition",
        "training_image",
        "training_image_config",
        "algorithm_name",
        "output_data_config",
        "checkpoint_config",
        "training_input_mode",
        "environment",
        "hyperparameters",
    ]

    SERIALIZABLE_CONFIG_ATTRIBUTES: ClassVar[Any] = {
        "source_code": SourceCode,
        "compute": Compute,
        "networking": Networking,
        "stopping_condition": StoppingCondition,
        "training_image_config": TrainingImageConfig,
        "output_data_config": OutputDataConfig,
        "checkpoint_config": CheckpointConfig,
    }

    def _populate_intelligent_defaults(self):
        """Function to populate all the possible default configs

        Model Trainer specific configs take precedence over the generic training job ones.
        """
        self._populate_intelligent_defaults_from_model_trainer_space()
        self._populate_intelligent_defaults_from_training_job_space()

    def _populate_intelligent_defaults_from_training_job_space(self):
        """Function to populate all the possible default configs from Training Job Space"""
        if not self.environment:
            self.environment = resolve_value_from_config(
                config_path=TRAINING_JOB_ENVIRONMENT_PATH, sagemaker_session=self.sagemaker_session
            )

        default_enable_network_isolation = resolve_value_from_config(
            config_path=TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        default_vpc_config = resolve_value_from_config(
            config_path=TRAINING_JOB_VPC_CONFIG_PATH, sagemaker_session=self.sagemaker_session
        )

        if not self.networking:
            if default_enable_network_isolation is not None or default_vpc_config is not None:
                self.networking = Networking(
                    default_enable_network_isolation=default_enable_network_isolation,
                    subnets=resolve_value_from_config(config_path=TRAINING_JOB_SUBNETS_PATH),
                    security_group_ids=resolve_value_from_config(
                        config_path=TRAINING_JOB_SECURITY_GROUP_IDS_PATH
                    ),
                )
        else:
            if self.networking.enable_network_isolation is None:
                self.networking.enable_network_isolation = default_enable_network_isolation
            if self.networking.subnets is None:
                self.networking.subnets = resolve_value_from_config(
                    config_path=TRAINING_JOB_SUBNETS_PATH
                )
            if self.networking.security_group_ids is None:
                self.networking.subnets = resolve_value_from_config(
                    config_path=TRAINING_JOB_SUBNETS_PATH
                )

        if not self.output_data_config:
            default_output_data_config = resolve_value_from_config(
                config_path=TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH
            )
            if default_output_data_config:
                self.output_data_config = OutputDataConfig(
                    **self._convert_keys_to_snake(default_output_data_config)
                )

        if not self.compute:
            default_resource_config = resolve_value_from_config(
                config_path=TRAINING_JOB_RESOURCE_CONFIG_PATH
            )
            if default_resource_config:
                self.compute = Compute(**self._convert_keys_to_snake(default_resource_config))

        if not self.role:
            self.role = resolve_value_from_config(config_path=TRAINING_JOB_ROLE_ARN_PATH)

        if not self.tags:
            self.tags = resolve_value_from_config(config_path=TRAINING_JOB_TAGS_PATH)

    def _convert_keys_to_snake(self, config: dict) -> dict:
        """Utility helper function that converts the keys of a dictionary into snake case"""
        return {to_snake_case(key): value for key, value in config.items()}

    def _populate_intelligent_defaults_from_model_trainer_space(self):
        """Function to populate all the possible default configs from Model Trainer Space"""

        for configurable_attribute in self.CONFIGURABLE_ATTRIBUTES:
            if getattr(self, configurable_attribute) is None:
                default_config = resolve_value_from_config(
                    config_path=_simple_path(
                        SAGEMAKER,
                        PYTHON_SDK,
                        MODULES,
                        MODEL_TRAINER,
                        to_camel_case(configurable_attribute),
                    ),
                    sagemaker_session=self.sagemaker_session,
                )
                if default_config is not None:
                    if configurable_attribute in self.SERIALIZABLE_CONFIG_ATTRIBUTES:
                        default_config = self.SERIALIZABLE_CONFIG_ATTRIBUTES.get(
                            configurable_attribute
                        )(
                            **default_config  # pylint: disable=E1134
                        )
                    setattr(self, configurable_attribute, default_config)

    def __del__(self):
        """Destructor method to clean up the temporary directory."""
        # Clean up the temporary directory if it exists
        if self._temp_recipe_train_dir is not None:
            self._temp_recipe_train_dir.cleanup()

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

    def _validate_distributed_config(
        self,
        source_code: Optional[SourceCode],
        distributed: Optional[DistributedConfig],
    ):
        """Validate the distribution configuration."""
        if distributed and not source_code.entry_script:
            raise ValueError(
                "Must provide 'entry_script' if 'distribution' " + "is provided in 'source_code'.",
            )

    # TODO: Move to use pydantic model validators
    def _validate_source_code(self, source_code: Optional[SourceCode]):
        """Validate the source code configuration."""
        if source_code:
            if source_code.requirements or source_code.entry_script:
                source_dir = source_code.source_dir
                requirements = source_code.requirements
                entry_script = source_code.entry_script
                if not source_dir:
                    raise ValueError(
                        "If 'requirements' or 'entry_script' is provided in 'source_code', "
                        + "'source_dir' must also be provided.",
                    )
                if not _is_valid_path(source_dir, path_type="Directory"):
                    raise ValueError(
                        f"Invalid 'source_dir' path: {source_dir}. " + "Must be a valid directory.",
                    )
                if requirements:
                    if not _is_valid_path(
                        f"{source_dir}/{requirements}",
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

    def model_post_init(self, __context: Any):
        """Post init method to perform custom validation and set default values."""
        self._validate_training_image_and_algorithm_name(self.training_image, self.algorithm_name)
        self._validate_source_code(self.source_code)
        self._validate_distributed_config(self.source_code, self.distributed)

        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            if self.sagemaker_session is None:
                self.sagemaker_session = Session()
                logger.warning("SageMaker session not provided. Using default Session.")

            if self.role is None:
                self.role = get_execution_role(sagemaker_session=self.sagemaker_session)
                logger.warning(f"Role not provided. Using default role:\n{self.role}")

        if self.base_job_name is None:
            if self.algorithm_name:
                self.base_job_name = f"{self.algorithm_name}-job"
            elif self.training_image:
                self.base_job_name = f"{_get_repo_name_from_image(self.training_image)}-job"
            logger.warning(f"Base name not provided. Using default name:\n{self.base_job_name}")

        if self.compute is None:
            self.compute = Compute(
                instance_type=DEFAULT_INSTANCE_TYPE,
                instance_count=1,
                volume_size_in_gb=30,
            )
            logger.warning(f"Compute not provided. Using default:\n{self.compute}")

        if self.stopping_condition is None:
            self.stopping_condition = StoppingCondition(
                max_runtime_in_seconds=3600,
                max_pending_time_in_seconds=None,
                max_wait_time_in_seconds=None,
            )
            logger.warning(
                f"StoppingCondition not provided. Using default:\n{self.stopping_condition}"
            )

        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB and self.output_data_config is None:
            session = self.sagemaker_session
            base_job_name = self.base_job_name
            self.output_data_config = OutputDataConfig(
                s3_output_path=f"s3://{self._fetch_bucket_name_and_prefix(session)}"
                f"/{base_job_name}",
                compression_type="GZIP",
                kms_key_id=None,
            )
            logger.warning(
                f"OutputDataConfig not provided. Using default:\n{self.output_data_config}"
            )

        # TODO: Autodetect which image to use if source_code is provided
        if self.training_image:
            logger.info(f"Training image URI: {self.training_image}")

    def _fetch_bucket_name_and_prefix(self, session: Session) -> str:
        """Helper function to get the bucket name with the corresponding prefix if applicable"""
        if session.default_bucket_prefix is not None:
            return f"{session.default_bucket()}/{session.default_bucket_prefix}"
        return session.default_bucket()

    @_telemetry_emitter(feature=Feature.MODEL_TRAINER, func_name="model_trainer.train")
    @validate_call
    def train(
        self,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        wait: Optional[bool] = True,
        logs: Optional[bool] = True,
    ):
        """Train a model using AWS SageMaker.

        Args:
            input_data_config (Optional[Union[List[Channel], Dict[str, DataSourceType]]]):
                The input data config for the training job.
                Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
            wait (Optional[bool]):
                Whether to wait for the training job to complete before returning.
                Defaults to True.
            logs (Optional[bool]):
                Whether to display the training container logs while training.
                Defaults to True.
        """
        self._populate_intelligent_defaults()
        current_training_job_name = _get_unique_name(self.base_job_name)
        input_data_key_prefix = f"{self.base_job_name}/{current_training_job_name}/input"
        if input_data_config:
            self.input_data_config = input_data_config

        input_data_config = []
        if self.input_data_config:
            input_data_config = self._get_input_data_config(
                self.input_data_config, input_data_key_prefix
            )

        string_hyper_parameters = {}
        if self.hyperparameters:
            for hyper_parameter, value in self.hyperparameters.items():
                string_hyper_parameters[hyper_parameter] = safe_serialize(value)

        container_entrypoint = None
        container_arguments = None
        if self.source_code:
            if self.training_mode == Mode.LOCAL_CONTAINER:
                drivers_dir = TemporaryDirectory(
                    prefix=os.path.join(self.local_container_root + "/")
                )
            else:
                drivers_dir = TemporaryDirectory()
            shutil.copytree(SM_DRIVERS_LOCAL_PATH, drivers_dir.name, dirs_exist_ok=True)

            # If source code is provided, create a channel for the source code
            # The source code will be mounted at /opt/ml/input/data/code in the container
            if self.source_code.source_dir:
                source_code_channel = self.create_input_data_channel(
                    channel_name=SM_CODE,
                    data_source=self.source_code.source_dir,
                    key_prefix=input_data_key_prefix,
                )
                input_data_config.append(source_code_channel)

            self._prepare_train_script(
                tmp_dir=drivers_dir,
                source_code=self.source_code,
                distributed=self.distributed,
            )

            if isinstance(self.distributed, Torchrun) and self.distributed.smp:
                mp_parameters = self.distributed.smp._to_mp_hyperparameters()
                string_hyper_parameters.update(mp_parameters)

            self._write_source_code_json(tmp_dir=drivers_dir, source_code=self.source_code)
            self._write_distributed_json(tmp_dir=drivers_dir, distributed=self.distributed)

            # Create an input channel for drivers packaged by the sdk
            sm_drivers_channel = self.create_input_data_channel(
                channel_name=SM_DRIVERS,
                data_source=drivers_dir.name,
                key_prefix=input_data_key_prefix,
            )
            input_data_config.append(sm_drivers_channel)

            # If source_code is provided, we will always use
            # the default container entrypoint and arguments
            # to execute the sm_train.sh script.
            # Any commands generated from the source_code will be
            # executed from the sm_train.sh script.
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

        resource_config = self.compute._to_resource_config()
        vpc_config = self.networking._to_vpc_config() if self.networking else None

        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            training_job = TrainingJob.create(
                training_job_name=current_training_job_name,
                algorithm_specification=algorithm_specification,
                hyper_parameters=string_hyper_parameters,
                input_data_config=input_data_config,
                resource_config=resource_config,
                vpc_config=vpc_config,
                # Public Instance Attributes
                session=self.sagemaker_session.boto_session,
                role_arn=self.role,
                tags=self.tags,
                stopping_condition=self.stopping_condition,
                output_data_config=self.output_data_config,
                checkpoint_config=self.checkpoint_config,
                environment=self.environment,
                enable_managed_spot_training=self.compute.enable_managed_spot_training,
                enable_inter_container_traffic_encryption=(
                    self.networking.enable_inter_container_traffic_encryption
                    if self.networking
                    else None
                ),
                enable_network_isolation=(
                    self.networking.enable_network_isolation if self.networking else None
                ),
                # Private Instance Attributes
                remote_debug_config=self._remote_debug_config,
                tensor_board_output_config=self._tensorboard_output_config,
                retry_strategy=self._retry_strategy,
                infra_check_config=self._infra_check_config,
                session_chaining_config=self._session_chaining_config,
            )
            self._latest_training_job = training_job

            if wait:
                training_job.wait(logs=logs)
            if logs and not wait:
                logger.warning(
                    "Not displaing the training container logs as 'wait' is set to False."
                )
        else:
            local_container = _LocalContainer(
                training_job_name=_get_unique_name(self.base_job_name),
                instance_type=resource_config.instance_type,
                instance_count=resource_config.instance_count,
                image=algorithm_specification.training_image,
                container_root=self.local_container_root,
                sagemaker_session=self.sagemaker_session,
                container_entrypoint=algorithm_specification.container_entrypoint,
                container_arguments=algorithm_specification.container_arguments,
                input_data_config=input_data_config,
                hyper_parameters=string_hyper_parameters,
                environment=self.environment,
            )
            local_container.train(wait)

    def create_input_data_channel(
        self, channel_name: str, data_source: DataSourceType, key_prefix: Optional[str] = None
    ) -> Channel:
        """Create an input data channel for the training job.

        Args:
            channel_name (str): The name of the input data channel.
            data_source (DataSourceType): The data source for the input data channel.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
            key_prefix (Optional[str]): The key prefix to use when uploading data to S3.
                Only applicable when data_source is a local file path string.
                If not specified, local data will be uploaded to:
                    s3://<default_bucket_path>/<base_job_name>/input/<channel_name>/
                If specified, local data will be uploaded to:
                    s3://<default_bucket_path>/<key_prefix>/<channel_name>/
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
                if key_prefix:
                    logger.warning(
                        "key_prefix is only applicable when data_source is a local file path."
                    )
            elif _is_valid_path(data_source):
                if self.training_mode == Mode.LOCAL_CONTAINER:
                    channel = Channel(
                        channel_name=channel_name,
                        data_source=DataSource(
                            file_system_data_source=FileSystemDataSource.model_construct(
                                directory_path=data_source,
                                file_system_type="EFS",
                            ),
                        ),
                        input_mode="File",
                    )
                else:
                    key_prefix = (
                        f"{key_prefix}/{channel_name}"
                        if key_prefix
                        else f"{self.base_job_name}/input/{channel_name}"
                    )
                    if self.sagemaker_session.default_bucket_prefix:
                        key_prefix = f"{self.sagemaker_session.default_bucket_prefix}/{key_prefix}"
                    s3_uri = self.sagemaker_session.upload_data(
                        path=data_source,
                        bucket=self.sagemaker_session.default_bucket(),
                        key_prefix=key_prefix,
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
            channel = Channel(
                channel_name=channel_name, data_source=DataSource(s3_data_source=data_source)
            )
        elif isinstance(data_source, FileSystemDataSource):
            channel = Channel(
                channel_name=channel_name,
                data_source=DataSource(file_system_data_source=data_source),
            )
        return channel

    def _get_input_data_config(
        self,
        input_data_channels: Optional[List[Union[Channel, InputData]]],
        key_prefix: Optional[str] = None,
    ) -> List[Channel]:
        """Get the input data configuration for the training job.

        Args:
            input_data_channels (Optional[List[Union[Channel, InputData]]]):
                The input data config for the training job.
                Takes a list of Channel or InputData objects. An InputDataSource can be an S3 URI
                string, local file path string, S3DataSource object, or FileSystemDataSource object.
        """
        if input_data_channels is None:
            return []

        channels = []
        for input_data in input_data_channels:
            if isinstance(input_data, Channel):
                channels.append(input_data)
            elif isinstance(input_data, InputData):
                channel = self.create_input_data_channel(
                    input_data.channel_name, input_data.data_source, key_prefix=key_prefix
                )
                channels.append(channel)
            else:
                raise ValueError(
                    f"Invalid input data channel: {input_data}. "
                    + "Must be a Channel or InputDataSource."
                )
        return channels

    def _write_source_code_json(self, tmp_dir: TemporaryDirectory, source_code: SourceCode):
        """Write the source code configuration to a JSON file."""
        file_path = os.path.join(tmp_dir.name, SOURCE_CODE_JSON)
        with open(file_path, "w") as f:
            dump = source_code.model_dump(exclude_none=True) if source_code else {}
            f.write(json.dumps(dump))

    def _write_distributed_json(
        self,
        tmp_dir: TemporaryDirectory,
        distributed: Optional[DistributedConfig] = None,
    ):
        """Write the distributed runner configuration to a JSON file."""
        file_path = os.path.join(tmp_dir.name, DISTRIBUTED_JSON)
        with open(file_path, "w") as f:
            dump = distributed.model_dump(exclude_none=True) if distributed else {}
            f.write(json.dumps(dump))

    def _prepare_train_script(
        self,
        tmp_dir: TemporaryDirectory,
        source_code: SourceCode,
        distributed: Optional[DistributedConfig] = None,
    ):
        """Prepare the training script to be executed in the training job container.

        Args:
            source_code (SourceCodeConfig): The source code configuration.
        """

        base_command = ""
        if source_code.command:
            if source_code.entry_script:
                logger.warning(
                    "Both 'command' and 'entry_script' are provided in the SourceCodeConfig. "
                    + "Defaulting to 'command'."
                )
            base_command = source_code.command.split()
            base_command = " ".join(base_command)

        install_requirements = ""
        if source_code.requirements:
            install_requirements = "echo 'Installing requirements'\n"
            install_requirements = f"$SM_PIP_CMD install -r {source_code.requirements}"

        working_dir = ""
        if source_code.source_dir:
            working_dir = f"cd {SM_CODE_CONTAINER_PATH}"

        if base_command:
            execute_driver = EXECUTE_BASE_COMMANDS.format(base_command=base_command)
        elif distributed:
            distribution_type = distributed._type
            if distribution_type == "mpi":
                execute_driver = EXECUTE_MPI_DRIVER
            elif distribution_type == "torchrun":
                execute_driver = EXEUCTE_TORCHRUN_DRIVER
            else:
                raise ValueError(f"Unsupported distribution type: {distribution_type}.")
        elif source_code.entry_script and not source_code.command and not distributed:
            if not source_code.entry_script.endswith((".py", ".sh")):
                raise ValueError(
                    f"Unsupported entry script: {source_code.entry_script}."
                    + "Only .py and .sh scripts are supported."
                )
            execute_driver = EXECUTE_BASIC_SCRIPT_DRIVER

        train_script = TRAIN_SCRIPT_TEMPLATE.format(
            working_dir=working_dir,
            install_requirements=install_requirements,
            execute_driver=execute_driver,
        )

        with open(os.path.join(tmp_dir.name, TRAIN_SCRIPT), "w") as f:
            f.write(train_script)

    @classmethod
    def from_recipe(
        cls,
        training_recipe: str,
        compute: Compute,
        recipe_overrides: Optional[Dict[str, Any]] = None,
        networking: Optional[Networking] = None,
        stopping_condition: Optional[StoppingCondition] = None,
        requirements: Optional[str] = None,
        training_image: Optional[str] = None,
        training_image_config: Optional[TrainingImageConfig] = None,
        output_data_config: Optional[OutputDataConfig] = None,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
        training_input_mode: Optional[str] = "File",
        environment: Optional[Dict[str, str]] = None,
        tags: Optional[List[Tag]] = None,
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None,
        base_job_name: Optional[str] = None,
    ) -> "ModelTrainer":
        """Create a ModelTrainer from a training recipe.

        Args:
            training_recipe (str):
                The training recipe to use for training the model. This must be the name of
                a sagemaker training recipe or a path to a local training recipe .yaml file.
            compute (Compute):
                The compute configuration. This is used to specify the compute resources for
                the training job. If not specified, will default to 1 instance of ml.m5.xlarge.
            recipe_overrides (Optional[Dict[str, Any]]):
                The recipe overrides. This is used to override the default recipe parameters.
            networking (Optional[Networking]):
                The networking configuration. This is used to specify the networking settings
                for the training job.
            stopping_condition (Optional[StoppingCondition]):
                The stopping condition. This is used to specify the different stopping
                conditions for the training job.
                If not specified, will default to 1 hour max run time.
            requirements (Optional[str]):
                The path to a requirements file to install in the training job container.
            training_image (Optional[str]):
                The training image URI to use for the training job container. If not specified,
                the training image will be determined from the recipe.
            training_image_config (Optional[TrainingImageConfig]):
                Training image Config. This is the configuration to use an image from a private
                Docker registry for a training job.
            output_data_config (Optional[OutputDataConfig]):
                The output data configuration. This is used to specify the output data location
                for the training job.
                If not specified, will default to `s3://<default_bucket>/<base_job_name>/output/`.
            input_data_config (Optional[List[Union[Channel, InputData]]]):
                The input data config for the training job.
                Takes a list of Channel or InputData objects. An InputDataSource can be an S3 URI
                string, local file path string, S3DataSource object, or FileSystemDataSource object.
            checkpoint_config (Optional[CheckpointConfig]):
                Contains information about the output location for managed spot training checkpoint
                data.
            training_input_mode (Optional[str]):
                The input mode for the training job. Valid values are "Pipe", "File", "FastFile".
                Defaults to "File".
            environment (Optional[Dict[str, str]]):
                The environment variables for the training job.
            tags (Optional[List[Tag]]):
                An array of key-value pairs. You can use tags to categorize your AWS resources
                in different ways, for example, by purpose, owner, or environment.
            sagemaker_session (Optional[Session]):
                The SageMakerCore session.
                If not specified, a new session will be created.
            role (Optional[str]):
                The IAM role ARN for the training job.
                If not specified, the default SageMaker execution role will be used.
            base_job_name (Optional[str]):
                The base name for the training job.
                If not specified, a default name will be generated using the algorithm name
                or training image.
        """
        if compute.instance_type is None:
            raise ValueError(
                "Must set `instance_type` in compute_config when using training recipes."
            )
        device_type = _determine_device_type(compute.instance_type)
        if device_type == "cpu":
            raise ValueError(
                "Training recipes are not supported for CPU instances. "
                + "Please provide a GPU or Tranium instance type."
            )

        if training_image_config and training_image is None:
            raise ValueError("training_image must be provided when using training_image_config.")

        if sagemaker_session is None:
            sagemaker_session = Session()
            logger.warning("SageMaker session not provided. Using default Session.")
        if role is None:
            role = get_execution_role(sagemaker_session=sagemaker_session)
            logger.warning(f"Role not provided. Using default role:\n{role}")

        # The training recipe is used to prepare the following args:
        # - source_code
        # - training_image
        # - distributed
        # - compute
        # - hyperparameters
        model_trainer_args, recipe_train_dir = _get_args_from_recipe(
            training_recipe=training_recipe,
            recipe_overrides=recipe_overrides,
            requirements=requirements,
            compute=compute,
            region_name=sagemaker_session.boto_region_name,
        )
        if training_image is not None:
            model_trainer_args["training_image"] = training_image

        model_trainer = cls(
            sagemaker_session=sagemaker_session,
            role=role,
            base_job_name=base_job_name,
            networking=networking,
            stopping_condition=stopping_condition,
            training_image_config=training_image_config,
            output_data_config=output_data_config,
            input_data_config=input_data_config,
            checkpoint_config=checkpoint_config,
            training_input_mode=training_input_mode,
            environment=environment,
            tags=tags,
            **model_trainer_args,
        )

        model_trainer._temp_recipe_train_dir = recipe_train_dir
        return model_trainer

    def with_tensorboard_output_config(
        self, tensorboard_output_config: TensorBoardOutputConfig
    ) -> "ModelTrainer":
        """Set the TensorBoard output configuration.

        Args:
            tensorboard_output_config (TensorBoardOutputConfig):
                The TensorBoard output configuration.
        """
        self._tensorboard_output_config = tensorboard_output_config
        return self

    def with_retry_strategy(self, retry_strategy: RetryStrategy) -> "ModelTrainer":
        """Set the retry strategy for the training job.

        Args:
            retry_strategy (RetryStrategy):
                The retry strategy for the training job.
        """
        self._retry_strategy = retry_strategy
        return self

    def with_infra_check_config(self, infra_check_config: InfraCheckConfig) -> "ModelTrainer":
        """Set the infra check configuration for the training job.

        Args:
            infra_check_config (InfraCheckConfig):
                The infra check configuration for the training job.
        """
        self._infra_check_config = infra_check_config
        return self

    def with_session_chaining_config(
        self, session_chaining_config: SessionChainingConfig
    ) -> "ModelTrainer":
        """Set the session chaining configuration for the training job.

        Args:
            session_chaining_config (SessionChainingConfig):
                The session chaining configuration for the training job.
        """
        self._session_chaining_config = session_chaining_config
        return self

    def with_remote_debug_config(self, remote_debug_config: RemoteDebugConfig) -> "ModelTrainer":
        """Set the remote debug configuration for the training job.

        Args:
            remote_debug_config (RemoteDebugConfig):
                The remote debug configuration for the training job.
        """
        self._remote_debug_config = remote_debug_config
        return self
