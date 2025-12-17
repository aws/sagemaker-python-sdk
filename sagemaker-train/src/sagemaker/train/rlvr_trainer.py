from typing import Optional, Union
import logging
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import TrainingType, CustomizationTechnique, JOB_TYPE
from sagemaker.core.resources import TrainingJob, ModelPackageGroup, MlflowTrackingServer, ModelPackage
from sagemaker.core.shapes import VpcConfig
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name, _get_studio_tags
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.evaluator import Evaluator
from sagemaker.train.common_utils.finetune_utils import (
    _get_fine_tuning_options_and_model_arn,
    _validate_and_resolve_model_package_group,
    _extract_evaluator_arn,
    _resolve_model_and_name,
    _create_input_data_config,
    _convert_input_data_to_channels,
    _create_output_config,
    _create_serverless_config,
    _create_mlflow_config,
    _create_model_package_config,
    _validate_eula_for_gated_model,
    _validate_hyperparameter_values
)
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.constants import HUB_NAME

logger = logging.getLogger(__name__)


class RLVRTrainer(BaseTrainer):
    """Class that performs Reinforcement Learning from Verifiable Rewards (RLVR) fine-tuning on foundation models using AWS SageMaker.

    Example:

    .. code:: python

        from sagemaker.train import RLVRTrainer
        from sagemaker.train.common import TrainingType

        trainer = RLVRTrainer(
            model="meta-llama/Llama-2-7b-hf",
            training_type=TrainingType.LORA,
            model_package_group="my-model-group",
            custom_reward_function="arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-evaluator/1.0",
            training_dataset="s3://bucket/rlvr_data.jsonl"
        )

        trainer.train()

        # Complete workflow: create -> wait -> get model package ARN
        trainer = RLVRTrainer(
            model="meta-llama/Llama-2-7b-hf",
            model_package_group="my-rlvr-models",
            custom_reward_function="arn:aws:sagemaker:us-east-1:123456789012:hub-content/SageMakerPublicHub/JsonDoc/my-evaluator/1.0"
        )
        
        # Create training job (non-blocking)
        training_job = trainer.train(
            training_dataset="s3://bucket/rlvr_data.jsonl",
            wait=False
        )
        
        # Wait for completion
        training_job.wait()
        
        # Refresh job status
        training_job.refresh()
        
        # Get the fine-tuned model package ARN
        model_package_arn = training_job.output_model_package_arn

    Parameters:
        model (Union[str, ModelPackage]):
            The foundation model to fine-tune. Can be a model name string, model package ARN,
            or ModelPackage object.
        training_type (Union[TrainingType, str]):
            The fine-tuning approach. Valid values are TrainingType.LORA (default),
            TrainingType.FULL.
        model_package_group (Optional[Union[str, ModelPackageGroup]]):
            The model package group for storing the fine-tuned model. Can be a group name,
            ARN, or ModelPackageGroup object. Required when model is not a ModelPackage.
        custom_reward_function (Optional[Union[str, Evaluator]]):
            The custom reward function evaluator. Can be an evaluator ARN string or Evaluator object.
            Required for RLVR training to provide reward signals.
        mlflow_resource_arn (Optional[Union[str, MlflowTrackingServer]]):
            The MLflow tracking server ARN for experiment tracking.
            If not specified, uses default MLflow experience.
        mlflow_experiment_name (Optional[str]):
            The MLflow experiment name for organizing runs.
        mlflow_run_name (Optional[str]):
            The MLflow run name for this training job.
        training_dataset (Optional[Union[str, DataSet]]):
            The training dataset. Can be a dataset ARN, or DataSet object.
        validation_dataset (Optional[Union[str, DataSet]]):
            The validation dataset. Can be a dataset ARN, or DataSet object.
        s3_output_path (Optional[str]):
            The S3 path for training job outputs.
            If not specified, defaults to s3://sagemaker-<region>-<account>/output.
        kms_key_id (Optional[str]):
            The KMS key ID for encrypting training job outputs.
        networking (Optional[VpcConfig]):
            The VPC configuration for the training job.
    """

    def __init__(
        self,
        model: Union[str, ModelPackage],
        training_type: Union[TrainingType, str] = TrainingType.LORA,
        model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
        custom_reward_function: Optional[Union[str, Evaluator]] = None,
        mlflow_resource_arn: Optional[Union[str, MlflowTrackingServer]] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        training_dataset: Optional[Union[str, DataSet]] = None,
        validation_dataset: Optional[Union[str, DataSet]] = None,
        s3_output_path: Optional[str] = None,
        # Additional OutputDataConfig parameters
        kms_key_id: Optional[str] = None,
        # vpc config
        networking: Optional[VpcConfig] = None,
        accept_eula: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Resolve model and model name
        self.model, self._model_name = _resolve_model_and_name(model, self.sagemaker_session)

        self.training_type = training_type
        self.model_package_group = _validate_and_resolve_model_package_group(model,
                                                                                 model_package_group)
        self.custom_reward_function = custom_reward_function
        self.mlflow_resource_arn = mlflow_resource_arn
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.s3_output_path = s3_output_path
        self.kms_key_id = kms_key_id
        self.networking = networking

        # Initialize fine-tuning options with beta session fallback
        self.hyperparameters, self._model_arn, is_gated_model = _get_fine_tuning_options_and_model_arn(self._model_name,
                                                                     CustomizationTechnique.RLVR.value,
                                                                     self.training_type,
                                                                     self.sagemaker_session or TrainDefaults.get_sagemaker_session(
                                                                     sagemaker_session=self.sagemaker_session
                                                                    ))
        
        # Remove constructor-handled hyperparameters
        self._process_hyperparameters()
        
        # Validate and set EULA acceptance
        self.accept_eula = _validate_eula_for_gated_model(model, accept_eula, is_gated_model)

    def _process_hyperparameters(self):
        """Remove hyperparameter keys that are handled by constructor inputs."""
        if self.hyperparameters:
            # Remove keys that are handled by constructor inputs
            if hasattr(self.hyperparameters, 'data_s3_path'):
                delattr(self.hyperparameters, 'data_s3_path')
                self.hyperparameters._specs.pop('data_s3_path', None)
            if hasattr(self.hyperparameters, 'reward_lambda_arn'):
                delattr(self.hyperparameters, 'reward_lambda_arn')
                self.hyperparameters._specs.pop('reward_lambda_arn', None)
            if hasattr(self.hyperparameters, 'data_path'):
                delattr(self.hyperparameters, 'data_path')
                self.hyperparameters._specs.pop('data_path', None)
            if hasattr(self.hyperparameters, 'validation_data_path'):
                delattr(self.hyperparameters, 'validation_data_path')
                self.hyperparameters._specs.pop('validation_data_path', None)
            if hasattr(self.hyperparameters, 'output_path'):
                delattr(self.hyperparameters, 'output_path')
                self.hyperparameters._specs.pop('output_path', None)

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="RLVRTrainer.train")
    def train(self, training_dataset: Optional[Union[str, DataSet]] = None,
              validation_dataset: Optional[Union[str, DataSet]] = None, wait: bool = True):
        """Execute the RLVR training job.

        Parameters:
            training_dataset (Optional[Union[str, DataSet]]):
                The training dataset for this job. Overrides the dataset specified in __init__.
                Can be an S3 URI, dataset ARN, or DataSet object.
            validation_dataset (Optional[Union[str, DataSet]]):
                The validation dataset for this job. Overrides the dataset specified in __init__.
                Can be an S3 URI, dataset ARN, or DataSet object.
            wait (bool):
                Whether to wait for the training job to complete. Defaults to True.

        Returns:
            TrainingJob: The SageMaker training job object.
        """

        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )

        role = TrainDefaults.get_role(role=self.role, sagemaker_session=sagemaker_session)
        current_training_job_name = _get_unique_name(
            self.base_job_name or f"{self._model_name}-rlvr"
        )

        logger.info(f"Training Job Name: {current_training_job_name}")

        #data
        input_data_config = _create_input_data_config(training_dataset or self.training_dataset,
                                                     validation_dataset or self.validation_dataset
                                                     )
        channels = _convert_input_data_to_channels(input_data_config)

        output_config = _create_output_config(
            s3_output_path=self.s3_output_path,
            sagemaker_session=sagemaker_session,
            kms_key_id=self.kms_key_id
        )

        # Extract and validate evaluator ARN
        evaluator_arn = _extract_evaluator_arn(self.custom_reward_function) if self.custom_reward_function else None
        serverless_config = _create_serverless_config(model_arn=self._model_arn,
                                                     customization_technique=CustomizationTechnique.RLVR.value,
                                                     training_type=self.training_type,
                                                     accept_eula=self.accept_eula,
                                                     evaluator_arn=evaluator_arn,
                                                     job_type=JOB_TYPE
                                                     )
        mlflow_config = _create_mlflow_config(
            sagemaker_session,
            mlflow_resource_arn=self.mlflow_resource_arn,
            mlflow_experiment_name=self.mlflow_experiment_name,
            mlflow_run_name=self.mlflow_run_name,
        )

        final_hyperparameters = self.hyperparameters.to_dict()
        
        # Validate hyperparameter values
        _validate_hyperparameter_values(final_hyperparameters)

        model_package_config = _create_model_package_config(
            model_package_group_name=self.model_package_group,
            model=self.model,
            sagemaker_session=sagemaker_session
        )

        vpc_config = self.networking if self.networking else None
        tags = _get_studio_tags(self._model_name, HUB_NAME)

        try:
            training_job = TrainingJob.create(
                training_job_name=current_training_job_name,
                role_arn=role,
                input_data_config=channels,
                output_data_config=output_config,
                serverless_job_config=serverless_config,
                mlflow_config=mlflow_config,
                hyper_parameters=final_hyperparameters,
                model_package_config=model_package_config,
                vpc_config=vpc_config,
                session=sagemaker_session.boto_session,
                region=sagemaker_session.boto_session.region_name,
                tags=tags,
            )
        except Exception as e:
            logger.error("Error: %s", e)
            raise e

        if wait:
            from sagemaker.train.common_utils.trainer_wait import wait as _wait
            from sagemaker.core.utils.exceptions import TimeoutExceededError
            try:
                _wait(training_job)
            except TimeoutExceededError as e:
                logger.error("Error: %s", e)

        self._latest_training_job = training_job
        return training_job
