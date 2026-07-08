import json
import logging
from typing import Any, Dict, List, Optional, Union

from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import TrainingType, CustomizationTechnique, JOB_TYPE
from sagemaker.core.resources import TrainingJob, ModelPackageGroup, MlflowTrackingServer, ModelPackage
from sagemaker.core.shapes import VpcConfig
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name, _get_jumpstart_tags
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.ai_registry.evaluator import Evaluator
from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute
from sagemaker.train.configs import StoppingCondition
from sagemaker.core.training.configs import TrainingJobCompute, HyperPodCompute
from sagemaker.train.common_utils.finetune_utils import (
    _get_fine_tuning_options_and_model_arn,
    _validate_and_resolve_model_package_group,
    _extract_evaluator_arn,
    _is_lambda_arn,
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
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter, TelemetryParamType
from sagemaker.train.common_utils.telemetry_params import BASE_TRAINER_TELEMETRY_PARAMS
from sagemaker.train.common_utils.data_utils import is_multimodal_data, load_file_content
from sagemaker.train.common_utils.rlvr_reward_verifier import verify_reward_function
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.constants import get_sagemaker_hub_name

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

        # Using a Lambda ARN directly (Evaluator is auto-created):
        trainer = RLVRTrainer(
            model="meta-llama/Llama-2-7b-hf",
            training_type=TrainingType.LORA,
            model_package_group="my-model-group",
            custom_reward_function="arn:aws:lambda:us-east-1:123456789012:function:my-reward-fn",
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
            The custom reward function evaluator. Can be an evaluator ARN string, a Lambda
            function ARN string, or an Evaluator object. If a Lambda ARN is provided
            (e.g., "arn:aws:lambda:us-east-1:123456789012:function:my-reward"), an Evaluator
            will be automatically created in the AI Registry and used for training.
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
        stopping_condition (Optional[StoppingCondition]):
            The stopping condition to override training runtime limit.
            If not specified, uses SageMaker service default (24 hours for serverless training).
        is_multimodal (Optional[bool]):
            Whether the training dataset contains multimodal data. If None (default),
            auto-detected from the training dataset at train time.
    """

    _customization_technique = CustomizationTechnique.RLVR.value

    def __init__(
        self,
        model: Union[str, ModelPackage],
        training_type: Union[TrainingType, str] = TrainingType.LORA,
        model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
        custom_reward_function: Optional[Union[str, Evaluator]] = None,
        compute: Optional[Union[TrainingJobCompute, HyperPodCompute]] = None,
        mlflow_resource_arn: Optional[Union[str, MlflowTrackingServer]] = None,
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
        self.custom_reward_function = custom_reward_function
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
                                                                     CustomizationTechnique.RLVR.value,
                                                                     self.training_type,
                                                                     self.sagemaker_session or TrainDefaults.get_sagemaker_session(
                                                                     sagemaker_session=self.sagemaker_session
                                                                    ),
                                                                     compute=self.compute)

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
            if hasattr(self.hyperparameters, 'preset_reward_function'):
                delattr(self.hyperparameters, 'preset_reward_function')
                self.hyperparameters._specs.pop('preset_reward_function', None)
            if hasattr(self.hyperparameters, 'data_path'):
                delattr(self.hyperparameters, 'data_path')
                self.hyperparameters._specs.pop('data_path', None)
            if hasattr(self.hyperparameters, 'validation_data_path'):
                delattr(self.hyperparameters, 'validation_data_path')
                self.hyperparameters._specs.pop('validation_data_path', None)
            if hasattr(self.hyperparameters, 'output_path'):
                delattr(self.hyperparameters, 'output_path')
                self.hyperparameters._specs.pop('output_path', None)

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="RLVRTrainer.train",
        telemetry_params=BASE_TRAINER_TELEMETRY_PARAMS + [
            ("custom_reward_function", TelemetryParamType.ATTR_EXISTS),
        ],
    )
    def _verify_reward_function(
        self,
        sample_count: int = 3,
        training_dataset: Optional[Union[str, DataSet]] = None,
    ) -> Dict[str, Any]:
        """Verifies the reward function by invoking it with sample data from the training dataset.

        Reads a small number of samples from the training dataset and invokes the configured
        reward function (Lambda ARN or local Python file) to validate it returns the expected
        output format before submitting a full training job.

        Args:
            sample_count: Number of samples to read from the training dataset for verification.
                Defaults to 3.
            training_dataset: Training dataset to read samples from. Can be an S3 URI, dataset
                ARN, or DataSet object. If not provided, uses the dataset configured on the
                trainer instance.

        Returns:
            None. Logs the verification result dict on success.

        Raises:
            ValueError: If the reward function is not configured, no training dataset is
                available, or verification fails with detailed error messages.
        """
        # Resolve the reward function
        reward_function = self.custom_reward_function
        if reward_function is None:
            raise ValueError(
                "Cannot verify reward function: 'custom_reward_function' is not set. "
                "Please provide custom_reward_function when initializing RLVRTrainer."
            )

        is_nova = _is_nova_model(self._model_name)

        # If it's an Evaluator object, extract the Lambda ARN (reference)
        if isinstance(reward_function, Evaluator):
            reward_function = reward_function.reference
            if not reward_function:
                raise ValueError(
                    "Cannot verify reward function: Evaluator object does not have a "
                    "Lambda ARN reference. Verification requires a Lambda ARN or local file path."
                )
        elif isinstance(reward_function, str) and not _is_lambda_arn(reward_function):
            # It's a string but not a Lambda ARN — treat as an evaluator ARN,
            # fetch the Evaluator object and extract the Lambda ARN from it.
            # Evaluator Lambdas always use OSS format (statusCode/body envelope)
            try:
                # Parse evaluator name from the ARN
                # ARN format: arn:aws:sagemaker:region:account:hub-content/hub/type/name/version
                evaluator_name = reward_function.split("/")[-2]
                evaluator = Evaluator.get(evaluator_name, sagemaker_session=TrainDefaults.get_sagemaker_session(sagemaker_session=self.sagemaker_session))
                reward_function = evaluator.reference
                if not reward_function:
                    raise ValueError(
                        f"Evaluator '{evaluator_name}' does not have a Lambda ARN reference. "
                        "Verification requires a Lambda ARN."
                    )
                logger.info(f"Resolved evaluator ARN to Lambda ARN: {reward_function}")
            except ValueError:
                raise
            except Exception as e:
                raise ValueError(
                    f"Failed to resolve evaluator ARN '{self.custom_reward_function}' "
                    f"to a Lambda ARN for verification: {str(e)}"
                )

        # Resolve DataSet object to S3 URI
        if isinstance(training_dataset, DataSet):
            data_s3_path = training_dataset.source
        else:
            data_s3_path = training_dataset

        # Read sample data from the training dataset
        samples: List[Dict[str, Any]] = []
        try:
            for line in load_file_content(data_s3_path, extension=".jsonl", encoding="utf-8-sig"):
                if len(samples) >= sample_count:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    samples.append(sample)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"Failed to parse JSON from line {len(samples) + 1} in "
                        f"{data_s3_path}: {str(e)}"
                    )
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to read samples from {data_s3_path}: {str(e)}\n"
                "Please verify the S3 path is correct and you have read permissions."
            )

        if not samples:
            raise ValueError(
                f"No samples found in {data_s3_path}. "
                "Please ensure the data file contains valid JSONL data."
            )

        logger.info(f"Verifying reward function with {len(samples)} sample(s)...")

        result = verify_reward_function(
            reward_function=reward_function,
            sample_data=samples,
            validate_format=True,
            compute=self.compute,
            is_nova=is_nova,
        )

        logger.info(
            f"Reward function verification result: {json.dumps(result, indent=2, default=str)}"
        )

        if not result.get("success"):
            raise ValueError(
                f"Reward function verification failed: "
                f"Details: {json.dumps(result, default=str)}"
            )

        logger.info(
            f"Reward function verification successful: "
            f"{result['successful_samples']}/{result['total_samples']} sample(s) passed"
        )

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="RLVRTrainer.train")
    def train(self, training_dataset: Optional[Union[str, DataSet]] = None,
              validation_dataset: Optional[Union[str, DataSet]] = None, wait: bool = True, wait_timeout: Optional[int] = None, poll: int = 5):
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
            kms_key_id=self.kms_key_id,
            disable_output_compression=getattr(self, 'disable_output_compression', False),
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

        # Apply recipe/overrides if provided (overrides > recipe > Hub defaults)
        final_hyperparameters = self._apply_recipe_to_hyperparameters(final_hyperparameters)
        # Resolve is_multimodal: auto-detect from training dataset if not explicitly set
        if self.is_multimodal is None:
            effective_training_dataset = training_dataset or self.training_dataset
            if effective_training_dataset is not None:
                self.is_multimodal = is_multimodal_data(effective_training_dataset)

        # Validate hyperparameter values
        _validate_hyperparameter_values(final_hyperparameters)

        model_package_config = _create_model_package_config(
            model_package_group_name=self.model_package_group,
            model=self.model,
            sagemaker_session=sagemaker_session
        )

        # Verify reward function before submitting training job
        if self.custom_reward_function:
            effective_dataset = training_dataset or self.training_dataset
            if effective_dataset is not None:
                logger.info("Verifying reward function before submitting training job...")
                self._verify_reward_function(training_dataset=effective_dataset)

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
            try:
                wait_kwargs = {}
                if wait_timeout is not None:
                    wait_kwargs['timeout'] = wait_timeout
                wait_kwargs['poll'] = poll
                _wait(training_job, **wait_kwargs)
            except TimeoutExceededError as e:
                logger.error("Error: %s", e)

        self._latest_training_job = training_job
        return training_job

    def _get_extra_smtj_hyperparameters(self) -> Dict[str, Any]:
        """Return RLVR-specific hyperparameters for SMTJ training.

        Injects reward_lambda_arn from the custom_reward_function if set.
        """
        extra_hp = {}
        if self.custom_reward_function:
            reward_fn = self.custom_reward_function
            if isinstance(reward_fn, str) and (
                reward_fn.startswith("arn:aws:lambda:") or "hub-content" in reward_fn
            ):
                extra_hp["reward_lambda_arn"] = reward_fn
            else:
                evaluator_arn = _extract_evaluator_arn(reward_fn)
                if evaluator_arn:
                    extra_hp["reward_lambda_arn"] = evaluator_arn
        return extra_hp
