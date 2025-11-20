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
"""This module contains logic for setting defaults in ModelTrainer."""
from __future__ import absolute_import

from typing import Optional, Dict, Any, Union, List

from sagemaker.core.helper.session_helper import Session, get_execution_role
from sagemaker.core import shapes

from sagemaker.core.jumpstart.document import get_hub_content_and_document
from sagemaker.core.jumpstart.configs import JumpStartConfig
from sagemaker.core.jumpstart.constants import DEFAULT_TRAINING_ENTRY_POINT
from sagemaker.core.jumpstart.models import (
    TrainingComponentsModel,
    HubContentDocument,
    TrainingVariantModel,
)

from sagemaker.train import logger
from sagemaker.train.utils import _get_repo_name_from_image, _default_s3_uri
from sagemaker.train import configs
from sagemaker.train.configs import (
    Compute,
    StoppingCondition,
    Networking,
    SourceCode,
    Channel,
    InputData,
    S3DataSource,
    HubAccessConfig,
    ModelAccessConfig,
    DataSource,
    Tag,
)

DEFAULT_INSTANCE_TYPE = "ml.m5.xlarge"
DEFAULT_INSTANCE_COUNT = 1
DEFAULT_VOLUME_SIZE = 30
DEFAULT_MAX_RUNTIME_IN_SECONDS = 3600


class TrainDefaults:
    """Class to set the base default values for ModelTrainer."""

    @staticmethod
    def get_sagemaker_session(sagemaker_session: Optional[Session] = None) -> Session:
        """Get the default SageMaker session."""
        if sagemaker_session is None:
            sagemaker_session = Session()
            logger.info("SageMaker session not provided. Using default Session.")
        return sagemaker_session

    @staticmethod
    def get_role(
        role: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> str:
        """Get the default execution role."""
        if role is None:
            sagemaker_session = TrainDefaults.get_sagemaker_session(
                sagemaker_session=sagemaker_session
            )
            role = get_execution_role(sagemaker_session)
            logger.info(f"Role not provided. Using default role:\n{role}")
        return role

    @staticmethod
    def get_base_job_name(
        base_job_name: Optional[str] = None,
        algorithm_name: Optional[str] = None,
        training_image: Optional[str] = None,
    ) -> str:
        """Get the default base job name."""
        if base_job_name is None:
            if algorithm_name:
                base_job_name = f"{algorithm_name}-job"
            elif training_image:
                base_job_name = f"{_get_repo_name_from_image(training_image)}-job"
            logger.info(f"Base name not provided. Using default name:\n{base_job_name}")
        return base_job_name

    @staticmethod
    def get_compute(compute: Optional[Compute] = None) -> Compute:
        """Get the default compute."""
        if compute is None:
            compute = Compute(
                instance_type=DEFAULT_INSTANCE_TYPE,
                instance_count=DEFAULT_INSTANCE_COUNT,
                volume_size_in_gb=DEFAULT_VOLUME_SIZE,
            )
            logger.info(f"Compute not provided. Using default:\n{compute}")
        if compute.instance_type is None:
            compute.instance_type = DEFAULT_INSTANCE_TYPE
            logger.info(f"Instance type not provided. Using default:\n{DEFAULT_INSTANCE_TYPE}")
        if compute.instance_count is None:
            compute.instance_count = DEFAULT_INSTANCE_COUNT
            logger.info(f"Instance count not provided. Using default:\n{compute.instance_count}")
        if compute.volume_size_in_gb is None:
            compute.volume_size_in_gb = DEFAULT_VOLUME_SIZE
            logger.info(f"Volume size not provided. Using default:\n{compute.volume_size_in_gb}")
        return compute

    @staticmethod
    def get_stopping_condition(
        stopping_condition: Optional[StoppingCondition] = None,
    ) -> StoppingCondition:
        """Get the default stopping condition."""
        if stopping_condition is None:
            stopping_condition = StoppingCondition(
                max_runtime_in_seconds=DEFAULT_MAX_RUNTIME_IN_SECONDS,
                max_pending_time_in_seconds=None,
                max_wait_time_in_seconds=None,
            )
            logger.info(f"StoppingCondition not provided. Using default:\n{stopping_condition}")
        if stopping_condition.max_runtime_in_seconds is None:
            stopping_condition.max_runtime_in_seconds = DEFAULT_MAX_RUNTIME_IN_SECONDS
            logger.info(
                "Max runtime not provided. Using default:\n"
                f"{stopping_condition.max_runtime_in_seconds}"
            )
        return stopping_condition

    @staticmethod
    def get_output_data_config(
        base_job_name: str,
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> shapes.OutputDataConfig:
        """Get the default output data config."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        if output_data_config is None:
            output_data_config = configs.OutputDataConfig(
                s3_output_path=_default_s3_uri(
                    session=sagemaker_session, additional_path=base_job_name
                ),
                compression_type="GZIP",
                kms_key_id=None,
            )
            logger.info(f"OutputDataConfig not provided. Using default:\n{output_data_config}")
        if output_data_config.s3_output_path is None:
            base_job_name = base_job_name
            output_data_config.s3_output_path = _default_s3_uri(
                session=sagemaker_session, additional_path=base_job_name
            )
            logger.info(
                f"OutputDataConfig s3_output_path not provided. Using default:\n"
                f"{output_data_config.s3_output_path}"
            )
        if output_data_config.compression_type is None:
            output_data_config.compression_type = "GZIP"
            logger.info(
                f"OutputDataConfig compression type not provided. Using default:\n"
                f"{output_data_config.compression_type}"
            )
        return output_data_config


class JumpStartTrainDefaults:
    """Class for the JumpStart Defaults."""

    @staticmethod
    def _get_training_components_model(
        document: HubContentDocument,
        jumpstart_config: JumpStartConfig,
    ) -> TrainingComponentsModel:
        """Get the training components model."""
        training_components_model = document
        if jumpstart_config.training_config_name:
            if jumpstart_config.training_config_name not in document.TrainingConfigs:
                raise ValueError(
                    f"Training config {jumpstart_config.training_config_name} not found for model "
                    f"{jumpstart_config.model_id}.\n"
                    f"Available configs - {document.TrainingConfigs}."
                )
            training_components_model = document.TrainingConfigComponents[jumpstart_config]
        return training_components_model

    @staticmethod
    def _get_training_variant(
        training_components_model: TrainingComponentsModel,
        compute: Compute,
    ) -> TrainingVariantModel:
        """Get the training variant model."""
        variants = {} or training_components_model.TrainingInstanceTypeVariants.Variants
        instance_family = compute.instance_type.split(".")[1]
        variant = variants.get(instance_family)
        if not variant:
            variant = variants.get(compute.instance_type)
        return variant

    @staticmethod
    def get_compute(
        jumpstart_config: JumpStartConfig,
        compute: Optional[Compute] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Compute:
        """Get the default compute for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )

        if compute is None:
            compute = Compute(
                instance_type=training_components_model.DefaultTrainingInstanceType,
                instance_count=DEFAULT_INSTANCE_COUNT,
                volume_size_in_gb=(
                    training_components_model.TrainingVolumeSize or DEFAULT_VOLUME_SIZE
                ),
            )
            logger.info(f"Compute not provided. Using default compute:\n{compute}")
        if compute.instance_type is None and training_components_model.DefaultTrainingInstanceType:
            compute.instance_type = training_components_model.DefaultTrainingInstanceType
            logger.info(
                f"Instance type not provided. Using default instance type:\n{compute.instance_type}"
            )
        if compute.volume_size_in_gb is None:
            compute.volume_size_in_gb = (
                training_components_model.TrainingVolumeSize or DEFAULT_VOLUME_SIZE
            )
            logger.info(
                f"Volume size not provided. Using default volume size:\n{compute.volume_size_in_gb}"
            )
        if compute.instance_count is None:
            compute.instance_count = DEFAULT_INSTANCE_COUNT
            logger.info(f"Instance count not provided. Using default instance count:\n{compute}")
        return compute

    def get_networking(
        jumpstart_config: JumpStartConfig,
        networking: Optional[Networking] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Networking:
        """Get the default networking for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        if training_components_model.TrainingEnableNetworkIsolation:
            if networking is None:
                networking = Networking(
                    enable_network_isolation=True,
                )
                logger.info(f"Networking not provided. Using default networking:\n{networking}")
            else:
                networking.enable_network_isolation = True
                logger.info(
                    f"Networking provided. Setting enable_network_isolation to True:\n{networking}"
                )
        return networking

    def get_training_image(
        jumpstart_config: JumpStartConfig,
        compute: Compute,
        training_image: Optional[str] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> str:
        """Get the default training image for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        if training_image is None:
            variant = JumpStartTrainDefaults._get_training_variant(
                training_components_model=training_components_model,
                compute=compute,
            )
            training_image = (
                variant.Properties.ImageUri if variant else training_components_model.TrainingEcrUri
            )
            logger.info(f"Training image not provided. Using default:\n{training_image}")
        return training_image

    def get_base_job_name(
        jumpstart_config: JumpStartConfig,
        base_job_name: Optional[str] = None,
    ) -> str:
        """Get the default base job name for JumpStart."""
        if base_job_name is None:
            base_job_name = f"{jumpstart_config.model_id}-job"
            logger.info(f"Base name not provided. Using default name:\n{base_job_name}")
        return base_job_name

    def get_hyperparameters(
        jumpstart_config: JumpStartConfig,
        compute: Compute,
        hyperparameters: Optional[Dict[str, Any]] = None,
        environment: Optional[Dict[str, str]] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Dict[str, Any]:
        """Get the default hyperparameters for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        if hyperparameters is None:
            hyperparameters = {}
            logger.info(f"Hyperparameters not provided. Using defaults")
        variant = JumpStartTrainDefaults._get_training_variant(
            training_components_model=training_components_model,
            compute=compute,
        )
        default_hyperparameters = {
            hp.Name: hp.Default for hp in training_components_model.Hyperparameters
        }
        if variant and variant.Properties.Hyperparameters:
            default_hyperparameters.update(
                {hp.name: hp.default for hp in variant.Properties.Hyperparameters}
            )

        # Merge, giving precedence to user-provided values
        final_hyperparameters = default_hyperparameters.copy()
        final_hyperparameters.update(hyperparameters)

        # Handle Legacy Hyperparameters
        if final_hyperparameters:
            if "sagemaker_container_log_level" in final_hyperparameters:
                environment["SM_LOG_LEVEL"] = str(
                    final_hyperparameters["sagemaker_container_log_level"]
                )
                del final_hyperparameters["sagemaker_container_log_level"]
            for key in list(final_hyperparameters.keys()):
                if key.startswith("sagemaker_"):
                    del final_hyperparameters[key]

        return final_hyperparameters

    def get_enviornment(
        jumpstart_config: JumpStartConfig,
        compute: Compute,
        environment: Optional[Dict[str, str]] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> Dict[str, str]:
        """Get the default environment for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        if environment is None:
            environment = {}

        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )

        variant = JumpStartTrainDefaults._get_training_variant(
            training_components_model=training_components_model,
            compute=compute,
        )

        environment = environment or {}
        if variant:
            if variant.Properties.EnvironmentVariables:
                environment.update(variant.Properties.EnvironmentVariables)
        return environment

    def get_source_code(
        jumpstart_config: JumpStartConfig,
        source_code: Optional[SourceCode] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> SourceCode:
        """Get the default source code for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        if not source_code:
            source_code = SourceCode(
                source_dir=training_components_model.TrainingScriptUri,
                entry_script=DEFAULT_TRAINING_ENTRY_POINT,
                requirements="auto",
            )
        elif source_code.source_dir is None or source_code.entry_script is None:
            source_code.source_dir = training_components_model.TrainingScriptUri
            source_code.entry_script = DEFAULT_TRAINING_ENTRY_POINT
            if source_code.requirements is None:
                source_code.requirements = "auto"
        return source_code

    def get_training_dataset_input(
        jumpstart_config: JumpStartConfig,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> List[Union[Channel, InputData]]:
        """Get the default training dataset input for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        hub_content, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        train_channel_exists = False
        if input_data_config:
            # Only one of "training" or "train" channel is expected
            for input_data in input_data_config:
                if input_data.channel_name in ["training", "train"]:
                    if train_channel_exists:
                        raise ValueError(
                            "Only one of 'training' or 'train' channel is expected for JumpStart."
                        )
                    train_channel_exists = True
        if not input_data_config or not train_channel_exists:
            if not training_components_model.DefaultTrainingDatasetUri:
                logger.warning(
                    "No default training dataset is availble "
                    f"for the model ID {jumpstart_config.model_id}.\n"
                    "Provide a custom training dataset to the 'training' or 'train' input channel."
                )
            else:
                input_data_config = [] if input_data_config is None else input_data_config
                logger.warning(
                    f"Using default training dataset. "
                    "To override, provide custom input data to the 'training' "
                    "or 'train' input channel.\n"
                )
                input_data = InputData(
                    channel_name="training",
                    data_source=S3DataSource(
                        s3_data_type="S3Prefix",
                        s3_uri=training_components_model.DefaultTrainingDatasetUri,
                        attribute_names=None,
                        s3_data_distribution_type="FullyReplicated",
                        model_access_config=ModelAccessConfig(
                            accept_eula=jumpstart_config.accept_eula,
                        ),
                    ),
                )
                logger.info(f"Using default training dataset: {input_data}")
                if hub_content.hub_content_type == "ModelReference":
                    input_data.data_source.hub_access_config = HubAccessConfig(
                        hub_content_arn=hub_content.hub_content_arn
                    )
                input_data_config.append(input_data)
        return input_data_config

    def get_model_artifact_input(
        jumpstart_config: JumpStartConfig,
        compute: Compute,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        environment: Optional[Dict[str, str]] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> List[Union[Channel, InputData]]:
        """Get the default model artifact input for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        hub_content, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )
        variant = JumpStartTrainDefaults._get_training_variant(
            training_components_model=training_components_model,
            compute=compute,
        )

        model_channel_exists = False
        if input_data_config:
            model_channel_exists = any(
                input.channel_name == "model" for input in input_data_config if input_data_config
            )
        gated_model_env_var = environment.get("SageMakerGatedModelS3Uri")
        if model_channel_exists:
            if not document.IncrementalTrainingSupported:
                raise ValueError(
                    f"Model ID {jumpstart_config.model_id} does not support incremental training,\n"
                    "but a custom 'model' channel was provided.\n"
                )
            if gated_model_env_var:
                raise ValueError(
                    "A model channel and SageMakerGatedModelS3Uri environment variable cannot be used together.\n"
                    "Please provide either a model channel or the SageMakerGatedModelS3Uri environment variable.\n"
                )

        if gated_model_env_var:
            logger.warning(
                "SageMakerGatedModelS3Uri environment variable is provided.\n"
                "This will be used to fetch the model artifacts."
            )
            return input_data_config

        if not input_data_config or not model_channel_exists:
            model_artifact_uri = training_components_model.TrainingArtifactUri
            if variant:
                model_artifact_uri = (
                    variant.Properties.GatedModelEnvVarUri
                    or variant.Properties.TrainingArtifactUri
                    or model_artifact_uri
                )
            if not model_artifact_uri:
                logger.warning(
                    "No default model artifact is availble "
                    f"for the model ID {hub_content.hub_content_name}."
                )
            else:
                input_data_config = [] if input_data_config is None else input_data_config
                input_data = Channel(
                    channel_name="model",
                    compression_type=(
                        training_components_model.TrainingArtifactCompressionType or "None"
                    ),
                    input_mode="File",
                    content_type="application/x-sagemaker-model",
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=model_artifact_uri,
                            attribute_names=None,
                            s3_data_distribution_type="FullyReplicated",
                            model_access_config=ModelAccessConfig(
                                accept_eula=jumpstart_config.accept_eula,
                            ),
                        ),
                    ),
                )
                logger.info(f"Using default model artifact: {input_data}")
                if hub_content.hub_content_type == "ModelReference":
                    input_data.data_source.s3_data_source.hub_access_config = HubAccessConfig(
                        hub_content_arn=hub_content.hub_content_arn
                    )
                input_data_config.append(input_data)
        return input_data_config

    def get_output_data_config(
        jumpstart_config: JumpStartConfig,
        base_job_name: str,
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> shapes.OutputDataConfig:
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        _, document = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        training_components_model = JumpStartTrainDefaults._get_training_components_model(
            document=document,
            jumpstart_config=jumpstart_config,
        )

        compression_type = (
            "GZIP" if not training_components_model.DisableOutputCompression else "NONE"
        )
        if not output_data_config:
            output_data_config = configs.OutputDataConfig(
                s3_output_path=_default_s3_uri(
                    session=sagemaker_session, additional_path=base_job_name
                ),
                kms_key_id=None,
            )
            logger.info(
                f"Output data config not provided. Using default output data config:\n"
                f"{output_data_config}"
            )
        if output_data_config.s3_output_path is None:
            output_data_config.s3_output_path = _default_s3_uri(
                session=sagemaker_session, additional_path=base_job_name
            )
            logger.info(
                f"Output data path not provided. Using default output data path:\n"
                f"{output_data_config.s3_output_path}"
            )
        output_data_config.compression_type = compression_type
        return output_data_config

    def get_tags(
        jumpstart_config: JumpStartConfig,
        tags: Optional[List[Tag]] = None,
        sagemaker_session: Optional[Session] = None,
    ) -> List[Tag]:
        """Get the default tags for JumpStart."""
        sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=sagemaker_session)
        hub_content, _ = get_hub_content_and_document(
            jumpstart_config=jumpstart_config,
            sagemaker_session=sagemaker_session,
        )
        tags = tags or []
        if len(tags) >= 49:
            logger.warning("Skipping adding JumpStart tags as the limit is reached.")
        else:
            model_id_tag = Tag(
                key="sagemaker-sdk:jumpstart-model-id", value=hub_content.hub_content_name
            )
            model_version_tag = Tag(
                key="sagemaker-sdk:jumpstart-model-version", value=hub_content.hub_content_version
            )
            tags.extend([model_id_tag, model_version_tag])
            logger.info(f"Adding JumpStart Tags:\n{model_id_tag},\n{model_version_tag}")
        return tags
