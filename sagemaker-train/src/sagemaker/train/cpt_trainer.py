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
"""CPTTrainer — Continued Pre-Training on foundation models using SageMaker HyperPod."""

from typing import Optional, Union
import logging

from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import TrainingType, CustomizationTechnique
from sagemaker.core.resources import ModelPackageGroup, ModelPackage
from sagemaker.core.shapes import VpcConfig
from sagemaker.ai_registry.dataset import DataSet
from sagemaker.train.configs import StoppingCondition
from sagemaker.train.data_mixing_config import DataMixingConfig
from sagemaker.core.training.configs import HyperPodCompute
from sagemaker.train.common_utils.finetune_utils import (
    _validate_and_resolve_model_package_group,
    _resolve_model_and_name,
    _validate_eula_for_gated_model,
)
from sagemaker.train.common_utils.data_mixing_utils import (
    validate_data_mixing_model,
    validate_data_mixing_categories,
    resolve_hyperpod_datamix_context,
    build_hyperpod_datamix_recipe_from_context,
)
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CPTTrainer(BaseTrainer):
    """Performs Continued Pre-Training (CPT) on foundation models using SageMaker HyperPod.

    CPT extends a foundation model's knowledge by further pre-training on domain-specific
    unlabeled text data. This is useful for adapting models to specialized domains
    (legal, medical, finance, etc.) before applying task-specific fine-tuning.

    CPT is only supported on HyperPod compute.

    Example:

    .. code:: python

        from sagemaker.train import CPTTrainer
        from sagemaker.core.training.configs import HyperPodCompute

        trainer = CPTTrainer(
            model="amazon.nova-lite-v2",
            model_package_group="my-cpt-models",
            training_dataset="s3://bucket/domain_corpus.jsonl",
            s3_output_path="s3://bucket/output/",
            compute=HyperPodCompute(
                cluster_name="my-cluster",
                instance_type="ml.p5.48xlarge",
                node_count=4,
            ),
            recipe="training/nova/nova_2_0/nova_lite/CPT/nova_lite_2_0_p5x8_gpu_pretrain",
            overrides={"recipes.training_config.trainer.max_steps": 100},
        )

        training_job = trainer.train(wait=False)

    Parameters:
        model (Union[str, ModelPackage]):
            The foundation model to continue pre-training.
        model_package_group (Optional[Union[str, ModelPackageGroup]]):
            The model package group for storing the trained model.
        compute (Optional[HyperPodCompute]):
            HyperPod compute configuration. Required — CPT only runs on HyperPod.
        mlflow_resource_arn (Optional[str]):
            The MLflow tracking server ARN for experiment tracking.
        mlflow_experiment_name (Optional[str]):
            The MLflow experiment name.
        mlflow_run_name (Optional[str]):
            The MLflow run name.
        training_dataset (Optional[Union[str, DataSet]]):
            S3 URI or DataSet object pointing to unlabeled text data.
        validation_dataset (Optional[Union[str, DataSet]]):
            Validation dataset for computing validation loss during training.
        s3_output_path (Optional[str]):
            S3 path for training job outputs.
        kms_key_id (Optional[str]):
            KMS key ID for encrypting outputs.
        networking (Optional[VpcConfig]):
            VPC configuration for the training job.
        stopping_condition (Optional[StoppingCondition]):
            Stopping condition to override training runtime limit.
        recipe (Optional[str]):
            Path to a user recipe YAML file or HyperPod recipe name. If not provided,
            the recipe is auto-resolved from SageMaker Hub based on model and training type.
        overrides (Optional[dict]):
            Programmatic overrides dict.
        training_image (Optional[str]):
            Custom training container image URI. If not provided, auto-resolved from Hub.
    """

    _customization_technique = CustomizationTechnique.CPT.value

    def __init__(
        self,
        model: Union[str, ModelPackage],
        model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
        compute: Optional[HyperPodCompute] = None,
        mlflow_resource_arn: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        training_dataset: Optional[Union[str, DataSet]] = None,
        validation_dataset: Optional[Union[str, DataSet]] = None,
        s3_output_path: Optional[str] = None,
        kms_key_id: Optional[str] = None,
        networking: Optional[VpcConfig] = None,
        accept_eula: Optional[bool] = False,
        stopping_condition: Optional[StoppingCondition] = None,
        recipe: Optional[str] = None,
        overrides: Optional[dict] = None,
        training_image: Optional[str] = None,
        data_mixing_config: Optional[DataMixingConfig] = None,
        **kwargs,
    ):
        super().__init__(training_image=training_image, **kwargs)

        self.model, self._model_name = _resolve_model_and_name(model, self.sagemaker_session)
        self.training_type = TrainingType.FULL

        self.model_package_group = _validate_and_resolve_model_package_group(
            model, model_package_group, compute=compute
        )

        if compute is not None and not isinstance(compute, HyperPodCompute):
            raise TypeError(
                f"CPT only supports HyperPod compute. Got {type(compute).__name__}. "
                f"Pass a HyperPodCompute instance with cluster_name and recipe."
            )
        self.compute = compute
        self.data_mixing_config = data_mixing_config

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

        # CPT is HyperPod-only and the recipe is auto-resolved from Hub if not
        # provided by the user. No Hub lookup for hyperparameters is needed —
        # they are managed entirely by the HyperPod recipe and overrides.
        self.hyperparameters = None
        self._model_arn = None
        self.accept_eula = _validate_eula_for_gated_model(model, accept_eula, False)

    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="CPTTrainer.train")
    def train(
        self,
        training_dataset: Optional[Union[str, DataSet]] = None,
        validation_dataset: Optional[Union[str, DataSet]] = None,
        wait: bool = True,
        wait_timeout: Optional[int] = None,
        poll: int = 5,
    ):
        """Execute the CPT training job on HyperPod.

        Parameters:
            training_dataset (Optional[Union[str, DataSet]]):
                Training dataset. Overrides the dataset specified in __init__.
            validation_dataset (Optional[Union[str, DataSet]]):
                Validation dataset. Overrides the dataset specified in __init__.
            wait (bool):
                Whether to wait for the job to complete. Defaults to True.
            wait_timeout (Optional[int]):
                Maximum time in seconds to wait.
            poll (int):
                Polling interval in seconds. Defaults to 5.

        Returns:
            str: The HyperPod job name.

        Raises:
            ValueError: If compute is not configured or recipe is missing.
        """
        if not isinstance(self.compute, HyperPodCompute):
            raise ValueError(
                "CPT requires HyperPod compute. Pass compute=HyperPodCompute(...) "
                "when creating the CPTTrainer."
            )

        if self.data_mixing_config is not None:
            if not isinstance(self.compute, HyperPodCompute):
                raise ValueError(
                    "Data mixing is only supported on HyperPod. "
                    "Provide a HyperPodCompute instance as compute."
                )

            validate_data_mixing_model(self._model_name)
            is_multimodal = getattr(self, "is_multimodal", False) or False

            from sagemaker.train.defaults import TrainDefaults

            sagemaker_session = TrainDefaults.get_sagemaker_session(
                sagemaker_session=self.sagemaker_session
            )

            context = resolve_hyperpod_datamix_context(
                model_name=self._model_name,
                is_multimodal=is_multimodal,
                sagemaker_session=sagemaker_session,
                training_type="FULL",
                customization_technique="CPT",
            )
            validated_config = validate_data_mixing_categories(
                self.data_mixing_config, context.categories
            )
            recipe_path, hp_image_uri = build_hyperpod_datamix_recipe_from_context(
                context, validated_config
            )
            self._recipe_path = recipe_path
            if hp_image_uri and not self.training_image:
                self.training_image = hp_image_uri

        return self._train_hyperpod(
            training_dataset=training_dataset,
            validation_dataset=validation_dataset,
            wait=wait,
            wait_timeout=wait_timeout,
            poll=poll,
        )
