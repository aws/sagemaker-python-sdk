from typing import Optional, Union
import logging
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import TrainingType, CustomizationTechnique, JOB_TYPE
from sagemaker.core.resources import TrainingJob, ModelPackageGroup, ModelPackage
from sagemaker.core.shapes import VpcConfig
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name, _get_jumpstart_tags
from sagemaker.train.configs import StoppingCondition
from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute
from sagemaker.train.common_utils.finetune_utils import (
    _get_fine_tuning_options_and_model_arn,
    _validate_and_resolve_model_package_group,
    _is_nova_model,
    _resolve_model_and_name,
    _resolve_model_with_checkpoint,
    _create_input_data_config,
    _convert_input_data_to_channels,
    _create_output_config,
    _create_serverless_config,
    _create_mlflow_config,
    _create_model_package_config,
    _validate_eula_for_gated_model,
    _validate_hyperparameter_values
)
from sagemaker.train.common_utils.data_utils import is_multimodal_data
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter, TelemetryParamType
from sagemaker.train.common_utils.telemetry_params import BASE_TRAINER_TELEMETRY_PARAMS
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.constants import get_sagemaker_hub_name

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DPOTrainer(BaseTrainer):
    """Class that performs Direct Preference Optimization (DPO) fine-tuning on foundation models using AWS SageMaker.

    Example:

    .. code:: python

        from sagemaker.train import DPOTrainer
        from sagemaker.train.common import TrainingType

        trainer = DPOTrainer(
            model="meta-llama/Llama-2-7b-hf",
            training_type=TrainingType.LORA,
            model_package_group="my-model-group",
            training_dataset="s3://bucket/preference_data.jsonl"
        )

        trainer.train()

        # Complete workflow: create -> wait -> get model package ARN
        trainer = DPOTrainer(
            model="meta-llama/Llama-2-7b-hf",
            model_package_group="my-dpo-models"
        )
        
        # Create training job (non-blocking)
        training_job = trainer.train(
            training_dataset="s3://bucket/preference_data.jsonl",
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
        mlflow_resource_arn (Optional[str]):
            The MLflow tracking server ARN for experiment tracking.
            If not specified, uses default MLflow experience.
        mlflow_experiment_name (Optional[str]):
            The MLflow experiment name for organizing runs.
        mlflow_run_name (Optional[str]):
            The MLflow run name for this training job.
        training_dataset (Optional[Union[str, DataSet]]):
            The training dataset with preference pairs. Can be a dataset ARN, or DataSet object.
        validation_dataset (Optional[Union[str, DataSet]]):
            The validation dataset. Can be a dataset ARN, or DataSet object.
        s3_output_path (Optional[str]):
            The S3 path for training job outputs.
            If not specified, defaults to s3://sagemaker-<region>-<account>/output.
        kms_key_id (Optional[str]):
            The KMS key ID for encrypting training job outputs.
        networking (Optional[VpcConfig]):
            The VPC configuration for the training job.
        stopping_condition (Optional[StoppingCondition]):
            The stopping condition to override training runtime limit.
            If not specified, uses SageMaker service default (24 hours for serverless training).
        is_multimodal (Optional[bool]):
            Whether the training dataset contains multimodal data. If None (default),
            auto-detected from the training dataset at train time.
    """

    _customization_technique = CustomizationTechnique.DPO.value

    def __init__(
            self,
            model: Union[str, ModelPackage],
            training_type: Union[TrainingType, str] = TrainingType.LORA,
            model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
            compute: Optional[Union[TrainingJobCompute, HyperPodCompute]] = None,
            mlflow_resource_arn: Optional[str] = None,
            mlflow_experiment_name: Optional[str] = None,
            mlflow_run_name: Optional[str] = None,
            training_dataset: Optional[Union[str, DataSet]] = None,
            validation_dataset: Optional[Union[str, DataSet]] = None,
            s3_output_path: Optional[str] = None,
            kms_key_id: Optional[str] = None,
            networking: Optional[VpcConfig] = None,
            accept_eula: bool = False,
            stopping_condition: Optional[StoppingCondition] = None,
            recipe: Optional[str] = None,
            overrides: Optional[dict] = None,
            is_multimodal: Optional[bool] = None,
            base_model_name: Optional[str] = None,
        disable_output_compression: Optional[bool] = False,
        **kwargs,
    ):
        super().__init__(base_model_name=base_model_name, disable_output_compression=disable_output_compression, **kwargs)

        self.model, self._model_name, self.model_source = _resolve_model_with_checkpoint(
            model, self.base_model_name, compute, self.sagemaker_session,
            resolve_fn=_resolve_model_and_name,
        )
        self.training_type = training_type

        self.compute = compute
        if compute is not None and not isinstance(compute, (TrainingJobCompute, HyperPodCompute)):
            raise TypeError(
                f"compute must be a TrainingJobCompute or HyperPodCompute instance, got {type(compute).__name__}"
            )

        if compute is None:
            self.model_package_group = _validate_and_resolve_model_package_group(
                model, model_package_group
            )
        else:
            self.model_package_group = model_package_group
        self.mlflow_resource_arn = mlflow_resource_arn
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.s3_output_path = s3_output_path
        self.kms_key_id = kms_key_id
        self.networking = networking
        self.stopping_condition = stopping_condition
        self._recipe_path = recipe
        self._overrides = overrides
        self._recipe_resolver = None
        self._resolved_recipe_cache = None
        self.is_multimodal = is_multimodal

        # Initialize fine-tuning options with beta session fallback
        self.hyperparameters, self._model_arn, is_gated_model = _get_fine_tuning_options_and_model_arn(self._model_name,
                                                                                      CustomizationTechnique.DPO.value,
                                                                                      self.training_type,
                                                                                      self.sagemaker_session or TrainDefaults.get_sagemaker_session(
                                                                                      sagemaker_session=self.sagemaker_session

                                                                                    ),
                                                                                      compute=self.compute)

        # Process hyperparameters
        self._process_hyperparameters()

        # Validate and set EULA acceptance
        self.accept_eula = _validate_eula_for_gated_model(model, accept_eula, is_gated_model)

    def _process_hyperparameters(self):
        """Remove hyperparameter keys that are handled by constructor inputs."""
        if self.hyperparameters:
            # Remove keys that are handled by constructor inputs
            if hasattr(self.hyperparameters, 'data_path'):
                delattr(self.hyperparameters, 'data_path')
                self.hyperparameters._specs.pop('data_path', None)
            if hasattr(self.hyperparameters, 'output_path'):
                delattr(self.hyperparameters, 'output_path')
                self.hyperparameters._specs.pop('output_path', None)
            if hasattr(self.hyperparameters, 'data_s3_path'):
                delattr(self.hyperparameters, 'data_s3_path')
                self.hyperparameters._specs.pop('data_s3_path', None)
            if hasattr(self.hyperparameters, 'output_s3_path'):
                delattr(self.hyperparameters, 'output_s3_path')
                self.hyperparameters._specs.pop('output_s3_path', None)
            if hasattr(self.hyperparameters, 'training_data_name'):
                delattr(self.hyperparameters, 'training_data_name')
                self.hyperparameters._specs.pop('training_data_name', None)
            if hasattr(self.hyperparameters, 'validation_data_name'):
                delattr(self.hyperparameters, 'validation_data_name')
                self.hyperparameters._specs.pop('validation_data_name', None)
            if hasattr(self.hyperparameters, 'validation_data_path'):
                delattr(self.hyperparameters, 'validation_data_path')
                self.hyperparameters._specs.pop('validation_data_path', None)

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="DPOTrainer.train",
        telemetry_params=BASE_TRAINER_TELEMETRY_PARAMS + [
            ("compute", TelemetryParamType.ATTR_TYPE),
        ],
    )
    def train(self,
              training_dataset: Optional[Union[str, DataSet]] = None,
              validation_dataset: Optional[Union[str, DataSet]] = None,
              wait: bool = True,
              wait_timeout: Optional[int] = None,
              poll: int = 5):
        """Execute the DPO training job.

        Parameters:
            training_dataset (Optional[Union[str, DataSet]]):
                The training dataset with preference pairs for this job. Overrides the dataset specified in __init__.
                Can be an S3 URI, dataset ARN, or DataSet object.
            validation_dataset (Optional[Union[str, DataSet]]):
                The validation dataset for this job. Overrides the dataset specified in __init__.
                Can be an S3 URI, dataset ARN, or DataSet object.
            wait (bool):
                Whether to wait for the training job to complete. Defaults to True.
            wait_timeout (Optional[int]):
                Maximum time in seconds to wait for the training job to complete. Only used when wait=True.
                If None, uses the default timeout from the wait utility.
            poll (int):
                Polling interval in seconds for checking training job status. Defaults to 5.

        Returns:
            TrainingJob: The SageMaker training job object.
        """
        # Dispatch based on compute type
        if isinstance(self.compute, HyperPodCompute):
            return self._train_hyperpod(
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                wait=wait,
                wait_timeout=wait_timeout,
                poll=poll,
            )
        elif isinstance(self.compute, TrainingJobCompute):
            return self._train_serverful_smtj(
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                wait=wait,
                wait_timeout=wait_timeout,
                poll=poll,
            )

        # Default: serverless compute (None)
        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )

        role = TrainDefaults.get_role(role=self.role, sagemaker_session=sagemaker_session)
        current_training_job_name = _get_unique_name(
            self.base_job_name or f"{self._model_name}-dpo"
        )

        logger.info(f"Training Job Name: {current_training_job_name}")
        print(f"Training Job Name: {current_training_job_name}")

        #data
        input_data_config = _create_input_data_config(training_dataset or self.training_dataset,
                                                     validation_dataset or self.validation_dataset
                                                     )
        channels = _convert_input_data_to_channels(
            input_data_config,
            s3_data_type="Converse" if _is_nova_model(self._model_name) else "S3Prefix",
        )

        output_config = _create_output_config(
            s3_output_path=self.s3_output_path,
            sagemaker_session=sagemaker_session,
            kms_key_id=self.kms_key_id,
            disable_output_compression=getattr(self, 'disable_output_compression', False),
        )

        serverless_config = _create_serverless_config(model_arn=self._model_arn,
                                                     customization_technique=CustomizationTechnique.DPO.value,
                                                     training_type=self.training_type,
                                                     accept_eula=self.accept_eula,
                                                     job_type=JOB_TYPE
                                                     )

        mlflow_config = _create_mlflow_config(
            sagemaker_session,
            mlflow_resource_arn=self.mlflow_resource_arn,
            mlflow_experiment_name=self.mlflow_experiment_name,
            mlflow_run_name=self.mlflow_run_name,
        )

        final_hyperparameters = self.hyperparameters.to_dict()

        # Apply recipe/overrides if provided (overrides > recipe > Hub defaults)
        final_hyperparameters = self._apply_recipe_to_hyperparameters(final_hyperparameters)
        # Resolve is_multimodal: auto-detect from training dataset if not explicitly set
        if self.is_multimodal is None:
            effective_training_dataset = training_dataset or self.training_dataset
            if effective_training_dataset is not None:
                self.is_multimodal = is_multimodal_data(effective_training_dataset)

        _validate_hyperparameter_values(final_hyperparameters)

        model_package_config = _create_model_package_config(
            model_package_group_name=self.model_package_group,
            model=self.model,
            sagemaker_session=sagemaker_session
        )

        vpc_config = self.networking if self.networking else None
        tags = _get_jumpstart_tags(self._model_name, get_sagemaker_hub_name())

        # Build TrainingJob.create() arguments
        create_args = {
            "training_job_name": current_training_job_name,
            "role_arn": role,
            "input_data_config": channels,
            "output_data_config": output_config,
            "serverless_job_config": serverless_config,
            "mlflow_config": mlflow_config,
            "hyper_parameters": final_hyperparameters,
            "model_package_config": model_package_config,
            "vpc_config": vpc_config,
            "session": sagemaker_session.boto_session,
            "region": sagemaker_session.boto_session.region_name,
            "tags": tags,
        }
        
        # Only pass stopping_condition if explicitly provided by user
        if self.stopping_condition is not None:
            create_args["stopping_condition"] = self.stopping_condition

        try:
            training_job = TrainingJob.create(**create_args)
        except Exception as e:
            logger.error("Error: %s", e)
            raise e

        if wait:
            from sagemaker.train.common_utils.trainer_wait import wait as _wait
            from sagemaker.core.utils.exceptions import TimeoutExceededError
            try :
                wait_kwargs = {}
                if wait_timeout is not None:
                    wait_kwargs['timeout'] = wait_timeout
                wait_kwargs['poll'] = poll
                _wait(training_job, **wait_kwargs)
            except TimeoutExceededError as e:
                logger.error("Error: %s", e)

        self.latest_training_job = training_job
        return training_job
