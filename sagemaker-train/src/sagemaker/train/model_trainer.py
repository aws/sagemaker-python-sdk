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
import yaml

from graphene.utils.str_converters import to_camel_case, to_snake_case
from sagemaker.core.config.config_manager import SageMakerConfig
from sagemaker.core import resources
from sagemaker.core.resources import TrainingJob
from sagemaker.core import shapes
from sagemaker.core.shapes import AlgorithmSpecification
from sagemaker.core.utils.utils import serialize
from sagemaker.core.apiutils._boto_functions import to_pascal_case

from pydantic import BaseModel, ConfigDict, PrivateAttr, validate_call
from sagemaker.core.config.config_schema import (
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

from sagemaker.core.helper.session_helper import Session
from sagemaker.train import configs
from sagemaker.train.configs import (
    Compute,
    StoppingCondition,
    RetryStrategy,
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
    InputData,
    MetricDefinition,
)

from sagemaker.train.distributed import Torchrun, DistributedConfig
from sagemaker.train.utils import (
    _default_s3_uri,
    _get_unique_name,
    _is_valid_path,
    _is_valid_s3_uri,
    safe_serialize,
)
from sagemaker.train.types import DataSourceType
from sagemaker.train.constants import (
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
from sagemaker.train.templates import (
    TRAIN_SCRIPT_TEMPLATE,
    EXECUTE_BASE_COMMANDS,
    EXEUCTE_DISTRIBUTED_DRIVER,
    EXECUTE_BASIC_SCRIPT_DRIVER,
    INSTALL_AUTO_REQUIREMENTS,
    INSTALL_REQUIREMENTS,
)
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train import logger
from sagemaker.train.sm_recipes.utils import _get_args_from_recipe, _determine_device_type

from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.jumpstart.document import get_hub_content_and_document
from sagemaker.core.jumpstart.utils import get_eula_url
from sagemaker.train.defaults import TrainDefaults, JumpStartTrainDefaults
from sagemaker.core.workflow.pipeline_context import PipelineSession, runnable_by_pipeline

from sagemaker.train.local.local_container import _LocalContainer


class Mode(Enum):
    """Enum class for training mode."""

    LOCAL_CONTAINER = "LOCAL_CONTAINER"
    SAGEMAKER_TRAINING_JOB = "SAGEMAKER_TRAINING_JOB"


class ModelTrainer(BaseModel):
    """Class that trains a model using AWS SageMaker.

    Example:

    .. code:: python

        from sagemaker.train import ModelTrainer
        from sagemaker.train.configs import SourceCode, Compute, InputData

        ignore_patterns = ['.env', '.git', '__pycache__', '.DS_Store', 'data']
        source_code = SourceCode(source_dir="source", entry_script="train.py", ignore_patterns=ignore_patterns)
        training_image = "123456789012.dkr.ecr.us-west-2.amazonaws.com/my-training-image"
        model_trainer = ModelTrainer(
            training_image=training_image,
            source_code=source_code,
        )

        train_data = InputData(channel_name="train", data_source="s3://bucket/train")
        model_trainer.train(input_data_config=[train_data])

        training_job = model_trainer._latest_training_job

    Parameters:
        training_mode (Mode):
            The training mode. Valid values are "Mode.LOCAL_CONTAINER" or
            "Mode.SAGEMAKER_TRAINING_JOB".
        sagemaker_session (Optiona(Session)):
            The SageMaker Session. This object can be used to manage underlying
            boto3 clients and to specify aritfact upload paths via the ``default_bucket``
            and ``default_bucket_prefix`` attributes.
            If not specified, a new session will be created.
        role (Optional(str)):
            The IAM role ARN for the training job.
            If not specified, the default SageMaker execution role will be used.
        base_job_name (Optional[str]):
            The base name for the training job.
            If not specified, a default name will be generated using the algorithm name
            or training image name.
        source_code (Optional[SourceCode]):
            The source code configuration. This is used to configure the source code for
            running the training job.
        distributed (Optional[DistributedConfig]):
            The distributed runner for the training job. This is used to configure
            a distributed training job. If specifed, ``source_code`` must also
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
            ``s3://<default_bucket>/<default_prefix>/<base_job_name>/``.
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
        hyperparameters (Optional[Union[Dict[str, Any], str]):
            The hyperparameters for the training job. Can be a dictionary of hyperparameters
            or a path to hyperparameters json/yaml file.
        tags (Optional[List[Tag]]):
            An array of key-value pairs. You can use tags to categorize your AWS resources
            in different ways, for example, by purpose, owner, or environment.
        local_container_root (Optional[str]):
            The local root directory to store artifacts from a training job launched in
            "LOCAL_CONTAINER" mode.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="forbid"
    )

    training_mode: Mode = Mode.SAGEMAKER_TRAINING_JOB
    sagemaker_session: Optional[Session] = None
    role: Optional[str] = None
    base_job_name: Optional[str] = None
    source_code: Optional[SourceCode] = None
    distributed: Optional[DistributedConfig] = None
    compute: Optional[Compute] = None
    networking: Optional[Networking] = None
    stopping_condition: Optional[StoppingCondition] = None
    training_image: Optional[str] = None
    training_image_config: Optional[TrainingImageConfig] = None
    algorithm_name: Optional[str] = None
    output_data_config: Optional[shapes.OutputDataConfig] = None
    input_data_config: Optional[List[Union[Channel, InputData]]] = None
    checkpoint_config: Optional[shapes.CheckpointConfig] = None
    training_input_mode: Optional[str] = "File"
    environment: Optional[Dict[str, str]] = {}
    hyperparameters: Optional[Union[Dict[str, Any], str]] = {}
    tags: Optional[List[Tag]] = None
    local_container_root: Optional[str] = os.getcwd()

    # Created Artifacts
    _latest_training_job: Optional[resources.TrainingJob] = PrivateAttr(default=None)

    # Private TrainingJob Parameters
    _tensorboard_output_config: Optional[shapes.TensorBoardOutputConfig] = PrivateAttr(default=None)
    _retry_strategy: Optional[RetryStrategy] = PrivateAttr(default=None)
    _infra_check_config: Optional[InfraCheckConfig] = PrivateAttr(default=None)
    _session_chaining_config: Optional[SessionChainingConfig] = PrivateAttr(default=None)
    _remote_debug_config: Optional[RemoteDebugConfig] = PrivateAttr(default=None)
    _metric_definitions: Optional[List[MetricDefinition]] = PrivateAttr(default=None)

    # Private Attributes for Recipes
    _temp_recipe_train_dir: Optional[TemporaryDirectory] = PrivateAttr(default=None)

    # Private Attributes for JumpStart
    _jumpstart_config: Optional[JumpStartConfig] = PrivateAttr(default=None)

    # Private Attributes for AWS_Batch
    _temp_code_dir: Optional[TemporaryDirectory] = PrivateAttr(default=None)

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
        "output_data_config": configs.OutputDataConfig,
        "checkpoint_config": configs.CheckpointConfig,
    }

    config_mgr: SageMakerConfig = SageMakerConfig()

    def _populate_intelligent_defaults(self):
        """Function to populate all the possible default configs

        Model Trainer specific configs take precedence over the generic training job ones.
        """
        self._populate_intelligent_defaults_from_model_trainer_space()
        self._populate_intelligent_defaults_from_training_job_space()

    def _populate_intelligent_defaults_from_training_job_space(self):
        """Function to populate all the possible default configs from Training Job Space"""

        if not self.environment:
            self.environment = self.config_mgr.resolve_value_from_config(
                config_path=TRAINING_JOB_ENVIRONMENT_PATH, sagemaker_session=self.sagemaker_session
            )

        default_enable_network_isolation = self.config_mgr.resolve_value_from_config(
            config_path=TRAINING_JOB_ENABLE_NETWORK_ISOLATION_PATH,
            sagemaker_session=self.sagemaker_session,
        )
        default_vpc_config = self.config_mgr.resolve_value_from_config(
            config_path=TRAINING_JOB_VPC_CONFIG_PATH, sagemaker_session=self.sagemaker_session
        )

        if not self.networking:
            if default_enable_network_isolation is not None or default_vpc_config is not None:
                self.networking = Networking(
                    default_enable_network_isolation=default_enable_network_isolation,
                    subnets=self.config_mgr.resolve_value_from_config(
                        config_path=TRAINING_JOB_SUBNETS_PATH
                    ),
                    security_group_ids=self.config_mgr.resolve_value_from_config(
                        config_path=TRAINING_JOB_SECURITY_GROUP_IDS_PATH
                    ),
                )
        else:
            if self.networking.enable_network_isolation is None:
                self.networking.enable_network_isolation = default_enable_network_isolation
            if self.networking.subnets is None:
                self.networking.subnets = self.config_mgr.resolve_value_from_config(
                    config_path=TRAINING_JOB_SUBNETS_PATH
                )
            if self.networking.security_group_ids is None:
                self.networking.subnets = self.config_mgr.resolve_value_from_config(
                    config_path=TRAINING_JOB_SUBNETS_PATH
                )

        if not self.output_data_config:
            default_output_data_config = self.config_mgr.resolve_value_from_config(
                config_path=TRAINING_JOB_OUTPUT_DATA_CONFIG_PATH
            )
            if default_output_data_config:
                self.output_data_config = configs.OutputDataConfig(
                    **self._convert_keys_to_snake(default_output_data_config)
                )

        if not self.compute:
            default_resource_config = self.config_mgr.resolve_value_from_config(
                config_path=TRAINING_JOB_RESOURCE_CONFIG_PATH
            )
            if default_resource_config:
                self.compute = Compute(**self._convert_keys_to_snake(default_resource_config))

        if not self.role:
            self.role = self.config_mgr.resolve_value_from_config(
                config_path=TRAINING_JOB_ROLE_ARN_PATH
            )

        if not self.tags:
            self.tags = self.config_mgr.resolve_value_from_config(
                config_path=TRAINING_JOB_TAGS_PATH
            )

    def _convert_keys_to_snake(self, config: dict) -> dict:
        """Utility helper function that converts the keys of a dictionary into snake case"""
        return {to_snake_case(key): value for key, value in config.items()}

    def _populate_intelligent_defaults_from_model_trainer_space(self):
        """Function to populate all the possible default configs from Model Trainer Space"""

        for configurable_attribute in self.CONFIGURABLE_ATTRIBUTES:
            if getattr(self, configurable_attribute) is None:
                default_config = self.config_mgr.resolve_value_from_config(
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
        # Clean up the temporary directory if it exists and class was initialized
        if hasattr(self, "__pydantic_fields_set__"):
            if self._temp_recipe_train_dir is not None:
                self._temp_recipe_train_dir.cleanup()
            if self._temp_code_dir is not None:
                self._temp_code_dir.cleanup()

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
                        "'source_dir' must also be provided."
                    )
                if not (
                    _is_valid_path(source_dir, path_type="Directory")
                    or _is_valid_s3_uri(source_dir, path_type="Directory")
                    or (
                        _is_valid_path(source_dir, path_type="File")
                        and source_dir.endswith(".tar.gz")
                    )
                    or (
                        _is_valid_s3_uri(source_dir, path_type="File")
                        and source_dir.endswith(".tar.gz")
                    )
                ):
                    raise ValueError(
                        f"Invalid 'source_dir' path: {source_dir}. "
                        "Must be a valid local directory, "
                        "s3 uri or path to tar.gz file stored locally or in s3."
                    )
                if requirements:
                    if not source_dir.endswith(".tar.gz"):
                        if not _is_valid_path(
                            f"{source_dir}/{requirements}", path_type="File"
                        ) and not _is_valid_s3_uri(
                            f"{source_dir}/{requirements}", path_type="File"
                        ):
                            raise ValueError(
                                f"Invalid 'requirements': {requirements}. "
                                "Must be a valid file within the 'source_dir'.",
                            )
                if entry_script:
                    if not source_dir.endswith(".tar.gz"):
                        if not _is_valid_path(
                            f"{source_dir}/{entry_script}", path_type="File"
                        ) and not _is_valid_s3_uri(
                            f"{source_dir}/{entry_script}", path_type="File"
                        ):
                            raise ValueError(
                                f"Invalid 'entry_script': {entry_script}. "
                                "Must be a valid file within the 'source_dir'.",
                            )

    @staticmethod
    def _validate_and_fetch_hyperparameters_file(hyperparameters_file: str):
        """Validate and fetch hyperparameters from a file."""
        if not os.path.exists(hyperparameters_file):
            raise ValueError(f"Hyperparameters file not found: {hyperparameters_file}")
        logger.info(f"Loading hyperparameters from file: {hyperparameters_file}")
        with open(hyperparameters_file, "r") as f:
            contents = f.read()
            try:
                hyperparameters = json.loads(contents)
                logger.debug("Hyperparameters loaded as JSON")
            except json.JSONDecodeError:
                try:
                    logger.info(f"contents: {contents}")
                    hyperparameters = yaml.safe_load(contents)
                    if not isinstance(hyperparameters, dict):
                        raise ValueError("YAML contents must be a valid mapping")
                    logger.info(f"hyperparameters: {hyperparameters}")
                    logger.debug("Hyperparameters loaded as YAML")
                except (yaml.YAMLError, ValueError):
                    raise ValueError(
                        f"Invalid hyperparameters file: {hyperparameters_file}. "
                        "Must be a valid JSON or YAML file."
                    )
        return hyperparameters

    def model_post_init(self, __context: Any):
        """Post init method to perform custom validation and set default values."""
        self._validate_training_image_and_algorithm_name(self.training_image, self.algorithm_name)
        self._validate_source_code(self.source_code)
        self._validate_distributed_config(self.source_code, self.distributed)

        if self.hyperparameters and isinstance(self.hyperparameters, str):
            self.hyperparameters = self._validate_and_fetch_hyperparameters_file(
                hyperparameters_file=self.hyperparameters
            )

        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            self.sagemaker_session = TrainDefaults.get_sagemaker_session(self.sagemaker_session)
            self.role = TrainDefaults.get_role(
                role=self.role, sagemaker_session=self.sagemaker_session
            )

        self.base_job_name = TrainDefaults.get_base_job_name(
            base_job_name=self.base_job_name,
            algorithm_name=self.algorithm_name,
            training_image=self.training_image,
        )
        self.compute = TrainDefaults.get_compute(compute=self.compute)
        self.stopping_condition = TrainDefaults.get_stopping_condition(
            stopping_condition=self.stopping_condition
        )

        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            self.output_data_config = TrainDefaults.get_output_data_config(
                base_job_name=self.base_job_name,
                output_data_config=self.output_data_config,
                sagemaker_session=self.sagemaker_session,
            )

        if self.training_image:
            logger.info(f"Training image URI: {self.training_image}")
    

    def _create_training_job_args(
        self,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        boto3: bool = False,
    ) -> Dict[str, Any]:
        """Create the training job arguments.
        Args:
            input_data_config (Optional[List[Union[Channel, InputData]]]):
            input_data_config (Optional[List[Union[Channel, InputData]]]):
                The input data config for the training job.
                Takes a list of Channel objects or a dictionary of channel names to DataSourceType.
                DataSourceType can be an S3 URI string, local file path string,
                S3DataSource object, or FileSystemDataSource object.
            boto3 (bool): Whether to return the arguments in boto3 format. Defaults to False.
                By default, the arguments are returned in the format used by the SageMaker Core.
        Returns:
            Dict[str, Any]: The training job arguments.
        """
        self._populate_intelligent_defaults()
        current_training_job_name = _get_unique_name(self.base_job_name)
        input_data_key_prefix = f"{self.base_job_name}/{current_training_job_name}/input"

        final_input_data_config = self.input_data_config.copy() if self.input_data_config else []

        if input_data_config:
            # merge the inputs with method parameter taking precedence
            existing_channels = {input.channel_name: input for input in final_input_data_config}
            new_channels = []
            for new_input in input_data_config:
                if new_input.channel_name in existing_channels:
                    existing_channels[new_input.channel_name] = new_input
                else:
                    new_channels.append(new_input)

            final_input_data_config = list(existing_channels.values()) + new_channels

        if final_input_data_config:
            final_input_data_config = self._get_input_data_config(
                final_input_data_config, input_data_key_prefix
            )

        if self.checkpoint_config and not self.checkpoint_config.s3_uri:
            self.checkpoint_config.s3_uri = _default_s3_uri(
                self.sagemaker_session,
                f"{self.base_job_name}/{current_training_job_name}/checkpoints",
            )

        if self._tensorboard_output_config and not self._tensorboard_output_config.s3_output_path:
            self._tensorboard_output_config.s3_output_path = _default_s3_uri(
                self.sagemaker_session, self.base_job_name
            )

        string_hyper_parameters = {}
        if self.hyperparameters:
            for hyper_parameter, value in self.hyperparameters.items():
                string_hyper_parameters[hyper_parameter] = safe_serialize(value)

        container_entrypoint = None
        container_arguments = None
        if self.source_code:
            if self.training_mode == Mode.LOCAL_CONTAINER:
                self._temp_code_dir = TemporaryDirectory(prefix=os.path.join(self.local_container_root + "/"))
            else:
                self._temp_code_dir = TemporaryDirectory()
            # Copy everything under container_drivers/ to a temporary directory
            shutil.copytree(SM_DRIVERS_LOCAL_PATH, self._temp_code_dir.name, dirs_exist_ok=True)

            # If distributed is provided, overwrite code under <root>/drivers
            if self.distributed:
                distributed_driver_dir = self.distributed.driver_dir
                driver_dir = os.path.join(self._temp_code_dir.name, "distributed_drivers")
                shutil.copytree(distributed_driver_dir, driver_dir, dirs_exist_ok=True)

            # If source code is provided, create a channel for the source code
            # The source code will be mounted at /opt/ml/input/data/code in the container
            if self.source_code.source_dir:
                source_code_channel = self.create_input_data_channel(
                    channel_name=SM_CODE,
                    data_source=self.source_code.source_dir,
                    key_prefix=input_data_key_prefix,
                    ignore_patterns=self.source_code.ignore_patterns,
                )
                final_input_data_config.append(source_code_channel)

            self._prepare_train_script(
                tmp_dir=self._temp_code_dir,
                source_code=self.source_code,
                distributed=self.distributed,
            )

            if isinstance(self.distributed, Torchrun) and self.distributed.smp:
                mp_parameters = self.distributed.smp._to_mp_hyperparameters()
                string_hyper_parameters.update(mp_parameters)

            self._write_source_code_json(tmp_dir=self._temp_code_dir, source_code=self.source_code)
            self._write_distributed_json(tmp_dir=self._temp_code_dir, distributed=self.distributed)

            # Create an input channel for drivers packaged by the sdk
            sm_drivers_channel = self.create_input_data_channel(
                channel_name=SM_DRIVERS,
                data_source=self._temp_code_dir.name,
                key_prefix=input_data_key_prefix,
                ignore_patterns=self.source_code.ignore_patterns,
            )
            final_input_data_config.append(sm_drivers_channel)

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
            metric_definitions=self._metric_definitions,
        )

        resource_config = self.compute._to_resource_config()
        vpc_config = self.networking._to_vpc_config() if self.networking else None

        # Convert tags to dictionaries if they are Tag objects
        tags_as_dicts = None
        if self.tags:
            tags_as_dicts = []
            for tag in self.tags:
                if hasattr(tag, 'model_dump'):
                    tags_as_dicts.append(tag.model_dump())
                elif isinstance(tag, dict):
                    tags_as_dicts.append(tag)
                else:
                    # Fallback for any other tag-like object
                    tags_as_dicts.append({"key": getattr(tag, 'key', ''), "value": getattr(tag, 'value', '')})
        
        # Build training request with snake_case keys (Python SDK convention)
        training_request = {
            "training_job_name": current_training_job_name,
            "algorithm_specification": algorithm_specification,
            "hyper_parameters": string_hyper_parameters,
            "input_data_config": final_input_data_config,
            "resource_config": resource_config,
            "vpc_config": vpc_config,
            "role_arn": self.role,
            "tags": tags_as_dicts,
            "stopping_condition": self.stopping_condition,
            "output_data_config": self.output_data_config,
            "checkpoint_config": self.checkpoint_config,
            "environment": self.environment,
            "enable_managed_spot_training": self.compute.enable_managed_spot_training,
            "enable_inter_container_traffic_encryption": (
                self.networking.enable_inter_container_traffic_encryption
                if self.networking
                else None
            ),
            "enable_network_isolation": (
                self.networking.enable_network_isolation if self.networking else None
            ),
            "remote_debug_config": self._remote_debug_config,
            "tensor_board_output_config": self._tensorboard_output_config,
            "retry_strategy": self._retry_strategy,
            "infra_check_config": self._infra_check_config,
            "session_chaining_config": self._session_chaining_config,
        }

        if boto3 or isinstance(self.sagemaker_session, PipelineSession):
            if isinstance(self.sagemaker_session, PipelineSession):
                training_request.pop("training_job_name", None)
            # Convert snake_case to PascalCase for AWS API
            pipeline_request = {to_pascal_case(k): v for k, v in training_request.items()}
            serialized_request = serialize(pipeline_request)
            return serialized_request
        
        return training_request


    @_telemetry_emitter(feature=Feature.MODEL_TRAINER, func_name="model_trainer.train")
    @runnable_by_pipeline
    @validate_call
    def train(
        self,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        wait: Optional[bool] = True,
        logs: Optional[bool] = True,
    ):
        """Train a model using AWS SageMaker.

        Args:
            input_data_config (Optional[List[Union[Channel, InputData]]]):
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
        training_request = self._create_training_job_args(input_data_config=input_data_config)
            
        # Handle PipelineSession
        if self.training_mode == Mode.SAGEMAKER_TRAINING_JOB:
            if isinstance(self.sagemaker_session, PipelineSession):
                self.sagemaker_session._intercept_create_request(training_request, None, "train")
                return
        
            training_job = TrainingJob.create(
                session=self.sagemaker_session.boto_session,
                **training_request
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
                training_job_name=training_request["training_job_name"],
                instance_type=training_request["resource_config"].instance_type,
                instance_count=training_request["resource_config"].instance_count,
                image=training_request["algorithm_specification"].training_image,
                container_root=self.local_container_root,
                sagemaker_session=self.sagemaker_session,
                container_entrypoint=training_request["algorithm_specification"].container_entrypoint,
                container_arguments=training_request["algorithm_specification"].container_arguments,
                input_data_config=training_request["input_data_config"],
                hyper_parameters=training_request["hyper_parameters"],
                environment=training_request["environment"],
            )
            local_container.train(wait)
        if self._temp_code_dir is not None:
            self._temp_code_dir.cleanup()


    def create_input_data_channel(
        self,
        channel_name: str,
        data_source: DataSourceType,
        key_prefix: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
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
                ``s3://<default_bucket_path>/<base_job_name>/input/<channel_name>/``

                If specified, local data will be uploaded to:
                ``s3://<default_bucket_path>/<key_prefix>/<channel_name>/``
            ignore_patterns: (Optional[List[str]]) :
                The ignore patterns to ignore specific files/folders when uploading to S3.
                If not specified, default to: ['.env', '.git', '__pycache__', '.DS_Store', '.cache', '.ipynb_checkpoints'].
        """
        from sagemaker.core.helper.pipeline_variable import PipelineVariable
        
        channel = None
        if isinstance(data_source, PipelineVariable):
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
        elif isinstance(data_source, str):
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
                    if ignore_patterns and _is_valid_path(data_source, path_type="Directory"):
                        tmp_dir = TemporaryDirectory()
                        copied_path = os.path.join(
                            tmp_dir.name, os.path.basename(os.path.normpath(data_source))
                        )
                        shutil.copytree(
                            data_source,
                            copied_path,
                            dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*ignore_patterns),
                        )
                        s3_uri = self.sagemaker_session.upload_data(
                            path=copied_path,
                            bucket=self.sagemaker_session.default_bucket(),
                            key_prefix=key_prefix,
                        )
                    else:
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
        else:
            raise ValueError(f"Unsupported data_source type: {type(data_source)}")
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
                if input_data.content_type:
                    channel.content_type = input_data.content_type
                channels.append(channel)
            else:
                raise ValueError(
                    f"Invalid input data channel: {input_data}. "
                    "Must be a Channel or InputDataSource."
                )
        return channels

    def _write_source_code_json(self, tmp_dir: TemporaryDirectory, source_code: SourceCode):
        """Write the source code configuration to a JSON file."""
        file_path = os.path.join(tmp_dir.name, SOURCE_CODE_JSON)
        with open(file_path, "w") as f:
            dump = source_code.model_dump() if source_code else {}
            f.write(json.dumps(dump))

    def _write_distributed_json(
        self,
        tmp_dir: TemporaryDirectory,
        distributed: Optional[DistributedConfig] = None,
    ):
        """Write the distributed runner configuration to a JSON file."""
        file_path = os.path.join(tmp_dir.name, DISTRIBUTED_JSON)
        with open(file_path, "w") as f:
            dump = distributed.model_dump() if distributed else {}
            f.write(json.dumps(dump))

    def _prepare_train_script(
        self,
        tmp_dir: TemporaryDirectory,
        source_code: SourceCode,
        distributed: Optional[DistributedConfig] = None,
    ):
        """Prepare the training script to be executed in the training job container.

        Args:
            source_code (SourceCode): The source code configuration.
        """

        base_command = ""
        if source_code.command:
            if source_code.entry_script:
                logger.warning(
                    "Both 'command' and 'entry_script' are provided in the SourceCode. "
                    "Defaulting to 'command'."
                )
            base_command = source_code.command.split()
            base_command = " ".join(base_command)

        install_requirements = ""
        if source_code.requirements:
            if self._jumpstart_config and source_code.requirements == "auto":
                install_requirements = INSTALL_AUTO_REQUIREMENTS
            else:
                install_requirements = INSTALL_REQUIREMENTS.format(
                    requirements_file=source_code.requirements
                )
        working_dir = ""
        if source_code.source_dir:
            working_dir = f"cd {SM_CODE_CONTAINER_PATH} \n"
            if source_code.source_dir.endswith(".tar.gz"):
                tarfile_name = os.path.basename(source_code.source_dir)
                working_dir += f"tar -xzf {tarfile_name} \n"

        if base_command:
            execute_driver = EXECUTE_BASE_COMMANDS.format(base_command=base_command)
        elif distributed:
            execute_driver = EXEUCTE_DISTRIBUTED_DRIVER.format(
                driver_name=distributed.__class__.__name__,
                driver_script=distributed.driver_script,
            )
        elif source_code.entry_script and not source_code.command and not distributed:
            if not source_code.entry_script.endswith((".py", ".sh")):
                raise ValueError(
                    f"Unsupported entry script: {source_code.entry_script}."
                    + "Only .py and .sh scripts are supported."
                )
            execute_driver = EXECUTE_BASIC_SCRIPT_DRIVER
        else:
            # This should never be reached, as the source_code should have been validated.
            raise ValueError(
                f"Unsupported SourceCode or DistributedConfig: {source_code}, {distributed}."
                "Please provide a valid configuration with atleast one of 'command'"
                " or 'entry_script'."
            )

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
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        checkpoint_config: Optional[shapes.CheckpointConfig] = None,
        training_input_mode: Optional[str] = "File",
        environment: Optional[Dict[str, str]] = None,
        tags: Optional[List[Tag]] = None,
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None,
        base_job_name: Optional[str] = None,
    ) -> "ModelTrainer":  # noqa: D412
        """Create a ModelTrainer from a training recipe.

        Example:

        .. code:: python

            from sagemaker.train import ModelTrainer
            from sagemaker.train.configs import Compute

            recipe_overrides = {
                "run": {
                    "results_dir": "/opt/ml/model",
                },
                "model": {
                    "data": {
                        "use_synthetic_data": True
                    }
                }
            }

            compute = Compute(
                instance_type="ml.p5.48xlarge",
                keep_alive_period_in_seconds=3600
            )

            model_trainer = ModelTrainer.from_recipe(
                training_recipe="fine-tuning/deepseek/hf_deepseek_r1_distilled_llama_8b_seq8k_gpu_fine_tuning",
                recipe_overrides=recipe_overrides,
                compute=compute,
            )

            model_trainer.train(wait=False)

        Args:
            training_recipe (str):
                The training recipe to use for training the model. This must be the name of
                a sagemaker training recipe or a path to a local training recipe .yaml file.
                For available training recipes, see: https://github.com/aws/sagemaker-hyperpod-recipes/
            compute (Compute):
                The compute configuration. This is used to specify the compute resources for
                the training job. Specifying instance_type is required for training recipes.
                Must be a GPU or Tranium instance type.
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
                If not specified, will default to ``s3://<default_bucket>/<base_job_name>/output/``.
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
            raise ValueError("Must set ``instance_type`` in Compute when using training recipes.")
        device_type = _determine_device_type(compute.instance_type)
        if device_type == "cpu":
            raise ValueError(
                "Training recipes are not supported for CPU instances. "
                "Please provide a GPU or Tranium instance type."
            )

        if training_image_config and training_image is None:
            raise ValueError("training_image must be provided when using training_image_config.")

        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session)
        role = TrainDefaults.get_role(role=role, sagemaker_session=sagemaker_session)

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

    @classmethod
    def from_jumpstart_config(
        cls,
        jumpstart_config: JumpStartConfig,
        source_code: Optional[SourceCode] = None,
        compute: Optional[Compute] = None,
        networking: Optional[Networking] = None,
        stopping_condition: Optional[StoppingCondition] = None,
        training_image: Optional[str] = None,
        training_image_config: Optional[TrainingImageConfig] = None,
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        checkpoint_config: Optional[shapes.CheckpointConfig] = None,
        training_input_mode: Optional[str] = "File",
        environment: Optional[Dict[str, str]] = {},
        hyperparameters: Optional[Union[Dict[str, Any], str]] = {},
        tags: Optional[List[Tag]] = None,
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None,
        base_job_name: Optional[str] = None,
    ) -> "ModelTrainer":  # noqa: D412
        """Create a ModelTrainer from a JumpStart Model ID.

        .. code:: python

            from sagemaker.train import ModelTrainer
            from sagemaker.train.configs import InputData
            from sagemaker.core.jumpstart import JumpStartConfig

            jumpstart_config = JumpStartConfig(model_id="xxxxxxx")

            model_trainer = ModelTrainer.from_jumpstart_config(
                jumpstart_config=jumpstart_config
            )

            training_data = InputData(channel_name="training", data_source="s3://bucket/path")
            model_trainer.train(input_data_config=[training_data])

        Args:
            jumpstart_config (JumpStart):
                The JumpStart model configuration. This is used to specify the model ID,
                version, and other parameters for the training job.
            source_code (Optional[SourceCode]):
                The source code configuration. This is used to configure the source code for
                running the training job.
            compute (Optional[Compute]):
                The compute configuration. This is used to specify the compute resources for
                the training job.
            networking (Optional[Networking]):
                The networking configuration. This is used to specify the networking settings
                for the training job.
            stopping_condition (Optional[StoppingCondition]):
                The stopping condition. This is used to specify the different stopping
                conditions for the training job.
                If not specified, will default to 1 hour max run time.
            training_image (Optional[str]):
                The training image URI to use for the training job container. If not specified,
                the training image will be determined from the recipe.
            training_image_config (Optional[TrainingImageConfig]):
                Training image Config. This is the configuration to use an image from a private
                Docker registry for a training job.
            output_data_config (Optional[OutputDataConfig]):
                The output data configuration. This is used to specify the output data location
                for the training job. If not specified, will default
                to ``s3://<default_bucket>/<default_prefix>/<base_job_name>/``.
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
            hyperparameters (Optional[Union[Dict[str, Any], str]]):
                The hyperparameters for the training job. Can be a dictionary of hyperparameters
                or a path to hyperparameters json/yaml file.
            tags (Optional[List[Tag]]):
                An array of key-value pairs. You can use tags to categorize your AWS resources
                in different ways, for example, by purpose, owner, or environment.
            sagemaker_session (Optional[Session]):
                The SageMaker Session. This object can be used to manage underlying
                boto3 clients and to specify aritfact upload paths via the ``default_bucket``
                and ``default_bucket_prefix`` attributes.
                If not specified, a new session will be created.
            role (Optional[str]):
                The IAM role ARN for the training job.
                If not specified, the default SageMaker execution role will be used.
            base_job_name (Optional[str]):
                The base name for the training job.
                If not specified, a default name will be generated using the algorithm name
                or training image name.
        """
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        role = TrainDefaults.get_role(role=role, sagemaker_session=sagemaker_session)

        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config, sagemaker_session=sagemaker_session
        )
        # Basic Validation
        if not document.TrainingSupported:
            raise ValueError(
                f"Training is not supported for the model ID: {jumpstart_config.model_id}.\n"
                "Please check that the model ID is available for training."
            )
        if compute and document.SupportedTrainingInstanceTypes:
            if compute.instance_type not in document.SupportedTrainingInstanceTypes:
                raise ValueError(
                    "Training is not supported for model ID with instance type: "
                    f" {compute.instance_type}.\n"
                    "This model ID is only supported for the following instance types:\n"
                    f"{document.SupportedTrainingInstanceTypes}.\n"
                )
        if document.GatedBucket:
            eula_url = get_eula_url(sagemaker_session=sagemaker_session, document=document)
            if not jumpstart_config.accept_eula:
                raise ValueError(
                    f"Model {jumpstart_config.model_id} is a gated model "
                    "and requires accepting the EULA via the `accept_eula` parameter.\n"
                    f"See {eula_url} for terms of use."
                )
            logger.warning(f"Model {jumpstart_config.model_id} is a gated model ")
            print(f"See {eula_url} for terms of use.")

        compute = JumpStartTrainDefaults.get_compute(
            jumpstart_config=jumpstart_config,
            compute=compute,
            sagemaker_session=sagemaker_session,
        )
        networking = JumpStartTrainDefaults.get_networking(
            jumpstart_config=jumpstart_config,
            networking=networking,
            sagemaker_session=sagemaker_session,
        )
        training_image = JumpStartTrainDefaults.get_training_image(
            jumpstart_config=jumpstart_config,
            compute=compute,
            training_image=training_image,
            sagemaker_session=sagemaker_session,
        )
        base_job_name = JumpStartTrainDefaults.get_base_job_name(
            jumpstart_config=jumpstart_config,
            base_job_name=base_job_name,
        )
        environment = JumpStartTrainDefaults.get_enviornment(
            jumpstart_config=jumpstart_config,
            compute=compute,
            environment=environment,
            sagemaker_session=sagemaker_session,
        )

        if hyperparameters and isinstance(hyperparameters, str):
            hyperparameters = cls._validate_and_fetch_hyperparameters_file(hyperparameters)

        hyperparameters = JumpStartTrainDefaults.get_hyperparameters(
            jumpstart_config=jumpstart_config,
            compute=compute,
            hyperparameters=hyperparameters,
            environment=environment,
            sagemaker_session=sagemaker_session,
        )
        source_code = JumpStartTrainDefaults.get_source_code(
            jumpstart_config=jumpstart_config,
            source_code=source_code,
            sagemaker_session=sagemaker_session,
        )
        input_data_config = JumpStartTrainDefaults.get_training_dataset_input(
            jumpstart_config=jumpstart_config,
            input_data_config=input_data_config,
            sagemaker_session=sagemaker_session,
        )
        input_data_config = JumpStartTrainDefaults.get_model_artifact_input(
            jumpstart_config=jumpstart_config,
            compute=compute,
            input_data_config=input_data_config,
            environment=environment,
            sagemaker_session=sagemaker_session,
        )
        output_data_config = JumpStartTrainDefaults.get_output_data_config(
            jumpstart_config=jumpstart_config,
            base_job_name=base_job_name,
            output_data_config=output_data_config,
            sagemaker_session=sagemaker_session,
        )
        tags = JumpStartTrainDefaults.get_tags(
            jumpstart_config=jumpstart_config,
            tags=tags,
            sagemaker_session=sagemaker_session,
        )

        model_trainer = cls(
            source_code=source_code,
            compute=compute,
            networking=networking,
            stopping_condition=stopping_condition,
            training_image=training_image,
            training_image_config=training_image_config,
            output_data_config=output_data_config,
            input_data_config=input_data_config,
            checkpoint_config=checkpoint_config,
            training_input_mode=training_input_mode,
            environment=environment,
            hyperparameters=hyperparameters,
            tags=tags,
            sagemaker_session=sagemaker_session,
            role=role,
            base_job_name=base_job_name,
        )
        model_trainer._jumpstart_config = jumpstart_config
        return model_trainer

    def with_tensorboard_output_config(
        self, tensorboard_output_config: Optional[shapes.TensorBoardOutputConfig] = None
    ) -> "ModelTrainer":  # noqa: D412
        """Set the TensorBoard output configuration.

        Example:

        .. code:: python

            from sagemaker.train import ModelTrainer

            model_trainer = ModelTrainer(
                ...
            ).with_tensorboard_output_config()

        Args:
            tensorboard_output_config (sagemaker.train.configs.TensorBoardOutputConfig):
                The TensorBoard output configuration.
        """
        self._tensorboard_output_config = (
            tensorboard_output_config or configs.TensorBoardOutputConfig()
        )
        return self

    def with_retry_strategy(self, retry_strategy: RetryStrategy) -> "ModelTrainer":  # noqa: D412
        """Set the retry strategy for the training job.

        Example:

        ..code:: python

            from sagemaker.train import ModelTrainer
            from sagemaker.train.configs import RetryStrategy

            retry_strategy = RetryStrategy(maximum_retry_attempts=3)

            model_trainer = ModelTrainer(
                ...
            ).with_retry_strategy(retry_strategy)

        Args:
            retry_strategy (sagemaker.train.configs.RetryStrategy):
                The retry strategy for the training job.
        """
        self._retry_strategy = retry_strategy
        return self

    def with_infra_check_config(
        self, infra_check_config: Optional[InfraCheckConfig] = None
    ) -> "ModelTrainer":  # noqa: D412
        """Set the infra check configuration for the training job.

        Example:

        ..code:: python
            from sagemaker.train import ModelTrainer

            model_trainer = ModelTrainer(
                ...
            ).with_infra_check_config(infra_check_config)

        Args:
            infra_check_config (sagemaker.train.configs.InfraCheckConfig):
                The infra check configuration for the training job.
        """
        self._infra_check_config = infra_check_config or InfraCheckConfig(enable_infra_check=True)
        return self

    def with_session_chaining_config(
        self, session_chaining_config: Optional[SessionChainingConfig] = None
    ) -> "ModelTrainer":  # noqa: D412
        """Set the session chaining configuration for the training job.

        Example:

        ..code:: python

            from sagemaker.train import ModelTrainer

            model_trainer = ModelTrainer(
                ...
            ).with_session_chaining_config()

        Args:
            session_chaining_config (sagemaker.train.configs.SessionChainingConfig):
                The session chaining configuration for the training job.
        """
        self._session_chaining_config = session_chaining_config or SessionChainingConfig(
            enable_session_tag_chaining=True
        )
        return self

    def with_remote_debug_config(
        self, remote_debug_config: RemoteDebugConfig
    ) -> "ModelTrainer":  # noqa: D412
        """Set the remote debug configuration for the training job.

        Example:

        ..code:: python

            from sagemaker.train import ModelTrainer

            model_trainer = ModelTrainer(
                ...
            ).with_remote_debug_config()

        Args:
            remote_debug_config (sagemaker.train.configs.RemoteDebugConfig):
                The remote debug configuration for the training job.
        """
        self._remote_debug_config = remote_debug_config or RemoteDebugConfig(
            enable_remote_debug=True
        )
        return self

    def with_checkpoint_config(
        self, checkpoint_config: Optional[shapes.CheckpointConfig] = None
    ) -> "ModelTrainer":  # noqa: D412
        """Set the checkpoint configuration for the training job.

        Example:

        .. code:: python

            from sagemaker.train import ModelTrainer

            model_trainer = ModelTrainer(
                ...
            ).with_checkpoint_config()

        Args:
            checkpoint_config (sagemaker.modules.configs.CheckpointConfig):
                The checkpoint configuration for the training job.
        """
        self.checkpoint_config = checkpoint_config or configs.CheckpointConfig()
        return self

    def with_metric_definitions(
        self,
        metric_definitions: List[MetricDefinition]
    ) -> "ModelTrainer":  # noqa: D412
        """Set the metric definitions for the training job.
        Example:
        .. code:: python
            from sagemaker.modules.train import ModelTrainer
            from sagemaker.modules.configs import MetricDefinition
            metric_definitions = [
                MetricDefinition(
                    name="loss",
                    regex="Loss: (.*?)",
                )
            ]
            model_trainer = ModelTrainer(
                ...
            ).with_metric_definitions(metric_definitions)
        Args:
            metric_definitions (List[MetricDefinition]):
                The metric definitions for the training job.
        """
        self._metric_definitions = metric_definitions

        return self