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
"""Placeholder docstring"""

from __future__ import absolute_import

import logging
from enum import Enum
from typing import Dict, List, Optional, Union
from sagemaker.core.analytics import HyperparameterTuningJobAnalytics

from sagemaker.core.jumpstart.utils import (
    add_jumpstart_uri_tags,
    get_jumpstart_base_name_if_jumpstart_model,
)
from sagemaker.core.parameter import (
    CategoricalParameter,
    ContinuousParameter,
    IntegerParameter,
    ParameterRange,
)
from sagemaker.core.shapes import (
    HyperParameterTuningJobWarmStartConfig,
    HyperParameterTuningJobStrategyConfig,
    HyperParameterTuningInstanceConfig,
    TuningJobCompletionCriteria,
    Channel,

)
from sagemaker.core.resources import HyperParameterTuningJob
from sagemaker.core.common_utils import (
    Tags,
    base_from_name,
    base_name_from_image,
    format_tags,
    name_from_base,
    to_string,
)
from sagemaker.core.helper.pipeline_variable import PipelineVariable
from sagemaker.core.workflow.pipeline_context import PipelineSession, runnable_by_pipeline
# Lazy import to avoid circular dependency - ModelTrainer imports from core
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from sagemaker.train.model_trainer import ModelTrainer
from sagemaker.core.training.configs import InputData
from sagemaker.core.training.utils import _is_valid_s3_uri

HYPERPARAMETER_TUNING_JOB_NAME = "HyperParameterTuningJobName"
PARENT_HYPERPARAMETER_TUNING_JOBS = "ParentHyperParameterTuningJobs"
WARM_START_TYPE = "WarmStartType"
HYPERBAND_STRATEGY_CONFIG = "HyperbandStrategyConfig"
HYPERBAND_MIN_RESOURCE = "MinResource"
HYPERBAND_MAX_RESOURCE = "MaxResource"
GRID_SEARCH = "Grid"
MAX_NUMBER_OF_TRAINING_JOBS_NOT_IMPROVING = "MaxNumberOfTrainingJobsNotImproving"
BEST_OBJECTIVE_NOT_IMPROVING = "BestObjectiveNotImproving"
CONVERGENCE_DETECTED = "ConvergenceDetected"
COMPLETE_ON_CONVERGENCE_DETECTED = "CompleteOnConvergence"
TARGET_OBJECTIVE_METRIC_VALUE = "TargetObjectiveMetricValue"
MAX_RUNTIME_IN_SECONDS = "MaxRuntimeInSeconds"

logger = logging.getLogger(__name__)


class WarmStartTypes(Enum):
    IDENTICAL_DATA_AND_ALGORITHM = "IdenticalDataAndAlgorithm"
    TRANSFER_LEARNING = "TransferLearning"


class HyperparameterTuner(object):
    """Defines interaction with Amazon SageMaker hyperparameter tuning jobs.

    It also supports deploying the resulting models.
    """

    TUNING_JOB_NAME_MAX_LENGTH = 32

    def __init__(
        self,
        model_trainer: "ModelTrainer",
        objective_metric_name: Union[str, PipelineVariable],
        hyperparameter_ranges: Dict[str, ParameterRange],
        metric_definitions: Optional[List[Dict[str, Union[str, PipelineVariable]]]] = None,
        strategy: Union[str, PipelineVariable] = "Bayesian",
        objective_type: Union[str, PipelineVariable] = "Maximize",
        max_jobs: Union[int, PipelineVariable] = None,
        max_parallel_jobs: Union[int, PipelineVariable] = 1,
        max_runtime_in_seconds: Optional[Union[int, PipelineVariable]] = None,
        tags: Optional[Tags] = None,
        base_tuning_job_name: Optional[str] = None,
        warm_start_config: Optional[HyperParameterTuningJobWarmStartConfig] = None,
        strategy_config: Optional[HyperParameterTuningJobStrategyConfig] = None,
        completion_criteria_config: Optional[TuningJobCompletionCriteria] = None,
        early_stopping_type: Union[str, PipelineVariable] = "Off",
        model_trainer_name: Optional[str] = None,
        random_seed: Optional[int] = None,
        autotune: bool = False,
        hyperparameters_to_keep_static: Optional[List[str]] = None,
    ):
        """Creates a ``HyperparameterTuner`` instance.

        It takes a model_trainer to obtain configuration information for training
        jobs that are created as the result of a hyperparameter tuning job.

        Args:
            model_trainer (sagemaker.train.model_trainer.ModelTrainer): A model_trainer object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.
            objective_metric_name (str or PipelineVariable): Name of the metric for evaluating
                training jobs.
            hyperparameter_ranges (dict[str, sagemaker.parameter.ParameterRange]): Dictionary of
                parameter ranges. These parameter ranges can be one
                of three types: Continuous, Integer, or Categorical. The keys of
                the dictionary are the names of the hyperparameter, and the
                values are the appropriate parameter range class to represent
                the range.
            metric_definitions (list[dict[str, str] or list[dict[str, PipelineVariable]]): A list of
                dictionaries that defines the metric(s) used to evaluate the training jobs (default:
                None). Each dictionary contains two keys: 'Name' for the name of
                the metric, and 'Regex' for the regular expression used to
                extract the metric from the logs. This should be defined only
                for hyperparameter tuning jobs that don't use an Amazon
                algorithm.
            strategy (str or PipelineVariable): Strategy to be used for hyperparameter model_trainer.
                More information about different strategies:
                https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html.
                Available options are: 'Bayesian', 'Random', 'Hyperband',
                'Grid' (default: 'Bayesian')
            objective_type (str or PipelineVariable): The type of the objective metric for
                evaluating training jobs. This value can be either 'Minimize' or
                'Maximize' (default: 'Maximize').
            max_jobs (int or PipelineVariable): Maximum total number of training jobs to start for
                the hyperparameter tuning job. The default value is unspecified fot the 'Grid'
                strategy and the default value is 1 for all others strategies (default: None).
            max_parallel_jobs (int or PipelineVariable): Maximum number of parallel training jobs to
                start (default: 1).
            max_runtime_in_seconds (int or PipelineVariable): The maximum time in seconds
                 that a hyperparameter tuning job can run.
            tags (Optional[Tags]): Tags for labeling the tuning job (default: None).
                For more, see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job
                name when the :meth:`~sagemaker.core.shapes.HyperparameterTuner.train`
                method launches. If not specified, a default job name is
                generated, based on the training image name and current
                timestamp.
            warm_start_config (sagemaker.core.shapes.HyperParameterTuningJobWarmStartConfig): A
                ``HyperParameterTuningJobWarmStartConfig`` object that has been initialized with the
                configuration defining the nature of warm start tuning job.
            strategy_config (sagemaker.core.shapes.StrategyConfig): A configuration for "Hyperparameter"
                tuning job optimisation strategy.
            completion_criteria_config (sagemaker.core.shapes.TuningJobCompletionCriteria): A
                 configuration for the completion criteria.
            early_stopping_type (str or PipelineVariable): Specifies whether early stopping is
                enabled for the job. Can be either 'Auto' or 'Off' (default:
                'Off'). If set to 'Off', early stopping will not be attempted.
                If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.
            model_trainer_name (str): A unique name to identify a model_trainer within the
                hyperparameter tuning job, when more than one model_trainer is used with
                the same tuning job (default: None).
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically (default: False).
            hyperparameters_to_keep_static: list[str]: Names of hyperparameters that will be kept
                static and will not be assigned a tunable range with Autotune functionality.
                (default: None).
        """
        if hyperparameter_ranges is None or len(hyperparameter_ranges) == 0:
            if not autotune:
                raise ValueError("Need to specify hyperparameter ranges or set autotune=True.")

        if not autotune and hyperparameters_to_keep_static is not None:
            raise ValueError(
                "hyperparameters_to_keep_static parameter is set, however Autotune mode is not "
                "enabled. Either do not set value for hyperparameters_to_keep_static parameter, "
                "or enable Autotune mode by setting autotune=True."
            )

        if hyperparameters_to_keep_static is not None:
            if len(hyperparameters_to_keep_static) != len(set(hyperparameters_to_keep_static)):
                raise ValueError("Please remove duplicate names in hyperparameters_to_keep_static.")

        if model_trainer_name is not None:
            self.model_trainer = None
            self.objective_metric_name = None
            self._hyperparameter_ranges = None
            self.metric_definitions = None
            self.model_trainer_dict = {model_trainer_name: model_trainer}
            self.objective_metric_name_dict = {model_trainer_name: objective_metric_name}
            self._hyperparameter_ranges_dict = {model_trainer_name: hyperparameter_ranges}
            self.metric_definitions_dict = (
                {model_trainer_name: metric_definitions} if metric_definitions is not None else {}
            )
            self.static_hyperparameters = None
            self.auto_parameters = None
            self.auto_parameters_dict = None
            self.hyperparameters_to_keep_static = None
            self.hyperparameters_to_keep_static_dict = {
                model_trainer_name: hyperparameters_to_keep_static
            }
        else:
            self.model_trainer = model_trainer
            self.objective_metric_name = objective_metric_name
            self._hyperparameter_ranges = hyperparameter_ranges
            self.metric_definitions = metric_definitions
            self.model_trainer_dict = None
            self.objective_metric_name_dict = None
            self._hyperparameter_ranges_dict = None
            self.metric_definitions_dict = None
            self.static_hyperparameters_dict = None
            self.auto_parameters = None
            self.auto_parameters_dict = None
            self.hyperparameters_to_keep_static = hyperparameters_to_keep_static
            self.hyperparameters_to_keep_static_dict = None

        self._validate_parameter_ranges(model_trainer, hyperparameter_ranges)

        self.strategy = strategy
        self.strategy_config = strategy_config
        self.completion_criteria_config = completion_criteria_config
        self.objective_type = objective_type
        # For the GridSearch strategy we expect the max_jobs equals None and recalculate it later.
        # For all other strategies for the backward compatibility we keep
        # the default value as 1 (previous default value).
        self.max_jobs = max_jobs
        if max_jobs is None and strategy != GRID_SEARCH:
            self.max_jobs = 1
        self.max_parallel_jobs = max_parallel_jobs
        self.max_runtime_in_seconds = max_runtime_in_seconds

        self.tags = format_tags(tags)
        self.base_tuning_job_name = base_tuning_job_name
        self._current_job_name = None
        self.latest_tuning_job = None
        self.warm_start_config = warm_start_config
        self.early_stopping_type = early_stopping_type
        self.random_seed = random_seed
        self.instance_configs_dict = None
        self.instance_configs = None
        self.autotune = autotune

    def override_resource_config(
        self,
        instance_configs: Union[List[HyperParameterTuningInstanceConfig], Dict[str, List[HyperParameterTuningInstanceConfig]]],
    ):
        """Override the instance configuration of the model_trainers used by the tuner.

        Args:
            instance_configs (List[HyperParameterTuningInstanceConfig] or Dict[str, List[HyperParameterTuningInstanceConfig]):
                The InstanceConfigs to use as an override for the instance configuration
                of the model_trainer. ``None`` will remove the override.
        """
        if isinstance(instance_configs, dict):
            self._validate_dict_argument(
                name="instance_configs",
                value=instance_configs,
                allowed_keys=list(self.model_trainer_dict.keys()),
            )
            self.instance_configs_dict = instance_configs
        else:
            self.instance_configs = instance_configs
            if self.model_trainer_dict is not None and self.model_trainer_dict.keys():
                model_trainer_names = list(self.model_trainer_dict.keys())
                self.instance_configs_dict = {model_trainer_names[0]: instance_configs}

    def _prepare_for_tuning(self, job_name=None):
        """Prepare the tuner instance for tuning (train)."""
        self._prepare_job_name_for_tuning(job_name=job_name)
        self._prepare_static_hyperparameters_for_tuning()
        self._prepare_auto_parameters_for_tuning()
        self._prepare_tags_for_tuning()

    def _get_model_uri(
        self,
        model_trainer,
    ):
        """Return the model artifact URI used by the ModelTrainer instance.

        This attribute can live in multiple places, and accessing the attribute can
        raise a TypeError, which needs to be handled.
        """
        try:
            return getattr(model_trainer, "model_data", None)
        except TypeError:
            return getattr(model_trainer, "model_uri", None)

    def _prepare_tags_for_tuning(self):
        """Add tags to tuning job (from ModelTrainer and JumpStart tags)."""

        # Add tags from ModelTrainer class
        model_trainer = self.model_trainer or self.model_trainer_dict[sorted(self.model_trainer_dict.keys())[0]]

        model_trainer_tags = getattr(model_trainer, "tags", []) or []

        if self.tags is None and len(model_trainer_tags) > 0:
            self.tags = []

        for tag in model_trainer_tags:
            if tag not in self.tags:
                self.tags.append(tag)

        if self.sagemaker_session.settings.include_jumpstart_tags:
            self.tags = add_jumpstart_uri_tags(
                tags=self.tags,
                training_script_uri=getattr(model_trainer, "source_code", None),
                training_model_uri=self._get_model_uri(model_trainer),
            )

    def _prepare_job_name_for_tuning(self, job_name=None):
        """Set current job name before starting tuning."""
        if job_name is not None:
            self._current_job_name = job_name
        else:
            base_name = self.base_tuning_job_name
            if base_name is None:
                model_trainer = (
                    self.model_trainer or self.model_trainer_dict[sorted(self.model_trainer_dict.keys())[0]]
                )
                base_name = base_name_from_image(
                    model_trainer.training_image,
                    default_base_name="ModelTrainer",
                )

                jumpstart_base_name = get_jumpstart_base_name_if_jumpstart_model(
                    getattr(model_trainer, "source_code", None),
                    self._get_model_uri(model_trainer),
                )
                base_name = jumpstart_base_name or base_name
            self._current_job_name = name_from_base(
                base_name, max_length=self.TUNING_JOB_NAME_MAX_LENGTH, short=True
            )

    def _prepare_static_hyperparameters_for_tuning(self):
        """Prepare static hyperparameters for all model_trainers before tuning."""
        self.static_hyperparameters = None
        if self.model_trainer is not None:
            self.static_hyperparameters = self._prepare_static_hyperparameters(
                self.model_trainer, self._hyperparameter_ranges
            )

        self.static_hyperparameters_dict = None
        if self.model_trainer_dict is not None:
            self.static_hyperparameters_dict = {
                model_trainer_name: self._prepare_static_hyperparameters(
                    model_trainer,
                    self._hyperparameter_ranges_dict[model_trainer_name],
                )
                for (model_trainer_name, model_trainer) in self.model_trainer_dict.items()
            }

    def _prepare_auto_parameters_for_tuning(self):
        """Prepare auto parameters for all model_trainers before tuning."""
        self.auto_parameters = None
        if self.model_trainer is not None:
            self.static_hyperparameters, self.auto_parameters = self._prepare_auto_parameters(
                self.static_hyperparameters, self.hyperparameters_to_keep_static
            )

        self.auto_parameters_dict = None
        if self.model_trainer_dict is not None:
            static_auto_parameters_dict = {
                model_trainer_name: self._prepare_auto_parameters(
                    self.static_hyperparameters_dict[model_trainer_name],
                    (
                        self.hyperparameters_to_keep_static_dict.get(model_trainer_name, None)
                        if self.hyperparameters_to_keep_static_dict
                        else None
                    ),
                )
                for model_trainer_name in sorted(self.model_trainer_dict.keys())
            }

            self.static_hyperparameters_dict = {}
            self.auto_parameters_dict = {}
            for model_trainer_name, (
                static_hyperparameters,
                auto_parameters,
            ) in static_auto_parameters_dict.items():
                self.static_hyperparameters_dict[model_trainer_name] = static_hyperparameters
                self.auto_parameters_dict[model_trainer_name] = auto_parameters

    @classmethod
    def _prepare_static_hyperparameters(
        cls, model_trainer, hyperparameter_ranges
    ):
        """Prepare static hyperparameters for one model_trainer before tuning."""
        # Initialize hyperparameters if None
        if model_trainer.hyperparameters is None:
            model_trainer.hyperparameters = {}
        
        # Remove any hyperparameter that will be tuned
        static_hyperparameters = {
            str(k): to_string(v) for (k, v) in model_trainer.hyperparameters.items()
        }
        if hyperparameter_ranges is not None:
            for hyperparameter_name in hyperparameter_ranges.keys():
                static_hyperparameters.pop(hyperparameter_name, None)

        return static_hyperparameters

    def _prepare_auto_parameters(self, static_hyperparameters, hyperparameters_to_keep_static):
        """Prepare auto parameters for one model_trainer before tuning."""
        if not self.autotune:
            return static_hyperparameters, None

        if hyperparameters_to_keep_static is None:
            hyperparameters_to_keep_static = {}

        if not set(hyperparameters_to_keep_static).issubset(set(static_hyperparameters.keys())):
            raise ValueError(
                "Names in hyperparameters_to_keep_static must be members of model_trainer's "
                "hyperparameters."
            )

        new_static_hyperparameters = {
            k: v for k, v in static_hyperparameters.items() if k in hyperparameters_to_keep_static
        }
        auto_parameters = {
            k: v
            for k, v in static_hyperparameters.items()
            if k not in hyperparameters_to_keep_static
        }

        return new_static_hyperparameters, auto_parameters

    @classmethod
    def _prepare_model_trainer_for_tuning(cls, model_trainer, inputs=None, job_name=None, **kwargs):
        """Prepare ModelTrainer before tuning by uploading source code and configuring hyperparameters.
        
        This method mimics V2's _prepare_estimator_for_tuning() pattern, adapted for V3's
        ModelTrainer architecture. It ensures that script mode hyperparameters are set before
        the tuning job is created, which framework containers (PyTorch, TensorFlow) require.
        
        Args:
            model_trainer: ModelTrainer instance to prepare
            inputs: Training inputs (unused, for V2 compatibility)
            job_name: Job name (unused, for V2 compatibility)
            **kwargs: Additional arguments (unused, for V2 compatibility)
        """
        # Only proceed if source_code is configured
        if hasattr(model_trainer, 'source_code') and model_trainer.source_code is not None:
            cls._upload_source_code_and_configure_hyperparameters(model_trainer)

    @classmethod
    def _upload_source_code_and_configure_hyperparameters(cls, model_trainer):
        """Upload source code to S3 and add script mode hyperparameters.
        
        Framework containers (PyTorch, TensorFlow) expect sagemaker_program and
        sagemaker_submit_directory hyperparameters for script mode execution. This method:
        1. Checks if source_dir is a local path or S3 URI
        2. Creates a tar.gz archive and uploads to S3
        3. Adds required script mode hyperparameters to model_trainer.hyperparameters
        
        This follows V2's pattern of creating sourcedir.tar.gz files.
        
        Args:
            model_trainer: ModelTrainer instance with source_code configured
        """
        import os
        import tarfile
        import tempfile
        import time
        
        source_code = model_trainer.source_code
        
        # Get source directory and entry script
        source_dir = source_code.source_dir
        entry_script = source_code.entry_script
        
        # Check if already an S3 URI
        if _is_valid_s3_uri(source_dir):
            # Already uploaded, use as-is
            source_s3_uri = source_dir
        else:
            # Local directory - need to create tar.gz and upload
            session = model_trainer.sagemaker_session
            bucket = session.default_bucket()
            
            # Generate S3 key
            timestamp = int(time.time())
            s3_key = f"{model_trainer.base_job_name or 'source'}/source-{timestamp}/sourcedir.tar.gz"
            
            # Create tar.gz file
            with tempfile.NamedTemporaryFile(suffix='.tar.gz', delete=False) as tmp_file:
                tar_path = tmp_file.name
            
            try:
                # Create tar.gz archive
                with tarfile.open(tar_path, 'w:gz') as tar:
                    # Add all files from source_dir
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # Calculate arcname to preserve directory structure
                            arcname = os.path.relpath(file_path, source_dir)
                            tar.add(file_path, arcname=arcname)
                
                # Upload to S3
                s3_client = session.boto_session.client('s3', region_name=session.boto_region_name)
                s3_client.upload_file(tar_path, bucket, s3_key)
                
                # Construct S3 URI
                source_s3_uri = f"s3://{bucket}/{s3_key}"
            finally:
                # Clean up temp file
                if os.path.exists(tar_path):
                    os.remove(tar_path)
        
        # Initialize hyperparameters dict if None
        if model_trainer.hyperparameters is None:
            model_trainer.hyperparameters = {}
        
        # Add script mode hyperparameters required by framework containers
        model_trainer.hyperparameters['sagemaker_program'] = entry_script
        model_trainer.hyperparameters['sagemaker_submit_directory'] = source_s3_uri

    @runnable_by_pipeline
    def tune(
        self,
        inputs: Optional[
            Union[
                str,
                Dict[str, str],
                List[Union[Channel, InputData]],
            ]
        ] = None,
        job_name: Optional[str] = None,
        model_trainer_kwargs: Optional[Dict[str, dict]] = None,
        wait: bool = True,
        **kwargs,
    ):
        """Start a hyperparameter tuning job.

        Args:
            inputs: Information about the training data. Please refer to the
                ``train()`` method of the associated model_trainer, as this can take
                any of the following forms:

                * (str) - The S3 location where training data is saved.
                * (dict[str, str]) - If using multiple channels for training data, you can specify
                    a dict mapping channel names to S3 URI strings.
                * (list[sagemaker.train.configs.Channel]) - A list of Channel objects for
                    detailed input data configuration.
                * (list[sagemaker.train.configs.InputData]) - A list of InputData objects for
                    simplified input data specification.

            job_name (str): Tuning job name. If not specified, the tuner
                generates a default job name, based on the training image name
                and current timestamp.
            model_trainer_kwargs (dict[str, dict]): Dictionary for other arguments needed for
                training. Should be used only for tuners created via the factory method create().
                The keys are the model_trainer names for the model_trainer_dict argument of create()
                method. Each value is a dictionary for the other arguments needed for training
                of the corresponding model_trainer.
            wait (bool): Whether the call should wait until the job completes (default: ``True``).
            **kwargs: Other arguments needed for training. Please refer to the
                ``train()`` method of the associated model_trainer to see what other
                arguments are needed.
        """
        if self.model_trainer is not None:
            self._train_with_model_trainer(inputs, job_name, **kwargs)
        else:
            self._train_with_model_trainer_dict(inputs, job_name, model_trainer_kwargs)

        if wait:
            self.latest_tuning_job.wait()

    def _train_with_model_trainer(self, inputs, job_name):
        """Start tuning for tuner instances that have the ``model_trainer`` field set."""
        # Prepare model_trainer before tuning (upload source code, set hyperparameters)
        self._prepare_model_trainer_for_tuning(self.model_trainer, inputs, job_name)
        self._prepare_for_tuning(job_name=job_name)
        self.latest_tuning_job = self._start_tuning_job(inputs)

    def _train_with_model_trainer_dict(self, inputs, job_name, model_trainer_kwargs):
        """Start tuning for tuner instances that have the ``model_trainer_dict`` field set."""
        model_trainer_names = sorted(self.model_trainer_dict.keys())
        self._validate_dict_argument(name="inputs", value=inputs, allowed_keys=model_trainer_names)
        self._validate_dict_argument(
            name="model_trainer_kwargs",
            value=model_trainer_kwargs,
            allowed_keys=model_trainer_names,
        )

        # Prepare each model_trainer before tuning (upload source code, set hyperparameters)
        for model_trainer_name, model_trainer in self.model_trainer_dict.items():
            ins = inputs.get(model_trainer_name, None) if inputs is not None else None
            self._prepare_model_trainer_for_tuning(model_trainer, ins, job_name)

        self._prepare_for_tuning(job_name=job_name)

        self.latest_tuning_job = self._start_tuning_job(inputs)

    def stop_tuning_job(self):
        """Stop latest running hyperparameter tuning job."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.stop()

    def describe(self):
        """Returns a response from the DescribeHyperParameterTuningJob API call."""
        self._ensure_last_tuning_job()
        return self.latest_tuning_job.refresh()

    def wait(self):
        """Wait for latest hyperparameter tuning job to finish."""
        self._ensure_last_tuning_job()
        self.latest_tuning_job.wait()

    def best_training_job(self):
        """Return name of the best training job for the latest hyperparameter tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        return self._get_best_training_job()["TrainingJobName"]

    def _get_best_training_job(self):
        """Return the best training job for the latest hyperparameter tuning job.

        Raises:
            Exception: If there is no best training job available for the
                hyperparameter tuning job.
        """
        self._ensure_last_tuning_job()
        
        # Refresh the tuning job to get latest status
        tuning_job = self.latest_tuning_job.refresh()
        
        if tuning_job.best_training_job:
            # Convert the best training job to the expected format
            best_job = tuning_job.best_training_job
            return {
                "TrainingJobName": best_job.training_job_name,
                "TrainingJobDefinitionName": best_job.training_job_definition_name or "training-job-definition"
            }
        else:
            raise Exception(
                f"Best training job not available for tuning job: {tuning_job.hyper_parameter_tuning_job_name}"
            )

    def _ensure_last_tuning_job(self):
        """Placeholder docstring"""
        if self.latest_tuning_job is None:
            raise ValueError("No tuning job available")

    @classmethod
    def _prepare_init_params_from_job_description(cls, job_details):
        """Placeholder docstring"""
        tuning_config = job_details["HyperParameterTuningJobConfig"]

        params = {
            "strategy": tuning_config["Strategy"],
            "max_jobs": tuning_config["ResourceLimits"]["MaxNumberOfTrainingJobs"],
            "max_parallel_jobs": tuning_config["ResourceLimits"]["MaxParallelTrainingJobs"],
            "warm_start_config": HyperParameterTuningJobWarmStartConfig.from_job_desc(
                job_details.get("HyperParameterTuningJobWarmStartConfig", None)
            ),
            "early_stopping_type": tuning_config["TrainingJobEarlyStoppingType"],
            "base_tuning_job_name": base_from_name(job_details["HyperParameterTuningJobName"]),
        }

        if "TuningJobCompletionCriteria" in tuning_config:
            params["completion_criteria_config"] = TuningJobCompletionCriteria.from_job_desc(
                tuning_config["TuningJobCompletionCriteria"]
            )

        if MAX_RUNTIME_IN_SECONDS in tuning_config["ResourceLimits"]:
            params["max_runtime_in_seconds"] = tuning_config["ResourceLimits"][
                MAX_RUNTIME_IN_SECONDS
            ]

        if "RandomSeed" in tuning_config:
            params["random_seed"] = tuning_config["RandomSeed"]

        if "HyperParameterTuningJobObjective" in tuning_config:
            params["objective_metric_name"] = tuning_config["HyperParameterTuningJobObjective"][
                "MetricName"
            ]
            params["objective_type"] = tuning_config["HyperParameterTuningJobObjective"]["Type"]

        if "ParameterRanges" in tuning_config:
            params["hyperparameter_ranges"] = cls._prepare_parameter_ranges_from_job_description(
                tuning_config["ParameterRanges"]
            )

        if "TrainingJobDefinition" in job_details:
            params["metric_definitions"] = job_details["TrainingJobDefinition"][
                "AlgorithmSpecification"
            ]["MetricDefinitions"]

        if "TrainingJobDefinitions" in job_details:
            params["objective_type"] = job_details["TrainingJobDefinitions"][0]["TuningObjective"][
                "Type"
            ]

        return params

    @classmethod
    def _prepare_parameter_ranges_from_job_description(cls, parameter_ranges):
        """Placeholder docstring"""
        ranges = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            ranges[parameter["Name"]] = CategoricalParameter(parameter["Values"])

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            ranges[parameter["Name"]] = ContinuousParameter(
                float(parameter["MinValue"]), float(parameter["MaxValue"])
            )

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            ranges[parameter["Name"]] = IntegerParameter(
                int(parameter["MinValue"]), int(parameter["MaxValue"])
            )

        return ranges

    @classmethod
    def _extract_hyperparameters_from_parameter_ranges(cls, parameter_ranges):
        """Placeholder docstring"""
        hyperparameters = {}

        for parameter in parameter_ranges["CategoricalParameterRanges"]:
            hyperparameters[parameter["Name"]] = parameter["Values"][0]

        for parameter in parameter_ranges["ContinuousParameterRanges"]:
            hyperparameters[parameter["Name"]] = float(parameter["MinValue"])

        for parameter in parameter_ranges["IntegerParameterRanges"]:
            hyperparameters[parameter["Name"]] = int(parameter["MinValue"])

        return hyperparameters

    def hyperparameter_ranges(self):
        """Return the hyperparameter ranges in a dictionary.

        Dictionary to be used as part of a request for creating a hyperparameter tuning job.
        """
        if self._hyperparameter_ranges is None:
            return None

        return self._prepare_parameter_ranges_for_tuning(
            self._hyperparameter_ranges
        )

    def hyperparameter_ranges_dict(self):
        """Return a dictionary of hyperparameter ranges for all model_trainers in ``model_trainer_dict``"""
        if self._hyperparameter_ranges_dict is None:
            return None

        return {
            model_trainer_name: self._prepare_parameter_ranges_for_tuning(
                self._hyperparameter_ranges_dict[model_trainer_name]
            )
            for model_trainer_name in sorted(self.model_trainer_dict.keys())
        }

    @classmethod
    def _prepare_parameter_ranges_for_tuning(cls, parameter_ranges):
        """Prepare hyperparameter ranges for tuning"""
        processed_parameter_ranges = dict()
        for range_type in ParameterRange.__all_types__:
            hp_ranges = []
            for parameter_name, parameter in parameter_ranges.items():
                if parameter is not None and parameter.__name__ == range_type:
                    # Get tuning range and convert keys to snake_case for v3 Pydantic models
                    tuning_range = parameter.as_tuning_range(parameter_name)
                    # Convert PascalCase keys to snake_case
                    tuning_range_snake = {}
                    for key, value in tuning_range.items():
                        # Convert PascalCase to snake_case
                        snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                        tuning_range_snake[snake_key] = value
                    hp_ranges.append(tuning_range_snake)
            processed_parameter_ranges[range_type + "ParameterRanges"] = hp_ranges
        return processed_parameter_ranges

    @property
    def sagemaker_session(self):
        """Convenience method for accessing the SageMaker session.

        It access :class:`~sagemaker.session.Session` object associated with the model_trainer
        for the ``HyperparameterTuner``.
        """
        model_trainer = self.model_trainer
        if model_trainer is None:
            first_model_trainer_name = sorted(self.model_trainer_dict.keys())[0]
            model_trainer = self.model_trainer_dict[first_model_trainer_name]
        return model_trainer.sagemaker_session

    def analytics(self):
        """An instance of HyperparameterTuningJobAnalytics for this latest tuning job of this tuner.

        Analytics olbject gives you access to tuning results summarized into a pandas dataframe.
        """
        self._ensure_last_tuning_job()
        return HyperparameterTuningJobAnalytics(
            self.latest_tuning_job.hyper_parameter_tuning_job_name, 
            self.sagemaker_session
        )

    def _validate_parameter_ranges(self, model_trainer, hyperparameter_ranges):
        """Validate hyperparameter ranges for a model_trainer"""
        # ModelTrainer uses a different hyperparameter structure
        # Skip validation for now as ModelTrainer handles this internally

    def _validate_parameter_range(self, value_hp, parameter_range):
        """Placeholder docstring"""
        for (
            parameter_range_key,
            parameter_range_value,
        ) in parameter_range.__dict__.items():
            if parameter_range_key == "scaling_type":
                continue

            # Categorical ranges
            if isinstance(parameter_range_value, list):
                for categorical_value in parameter_range_value:
                    value_hp.validate(categorical_value)
            # Continuous, Integer ranges
            else:
                value_hp.validate(parameter_range_value)

    def transfer_learning_tuner(self, additional_parents=None, model_trainer=None):
        """Creates a new ``HyperparameterTuner``.

        Creation is done by copying the request fields from the provided parent
        to the new instance of ``HyperparameterTuner``.
        Followed by addition of warm start configuration with the type as
        "TransferLearning" and parents as the union of provided list of
        ``additional_parents`` and the ``self``. Also, training image in the new
        tuner's model_trainer is updated with the provided ``training_image``.

        Examples:
            >>> parent_tuner = HyperparameterTuner.attach(tuning_job_name="parent-job-1")
            >>> transfer_learning_tuner = parent_tuner.transfer_learning_tuner(
            >>>                                             additional_parents={"parent-job-2"})
            Later On:
            >>> transfer_learning_tuner.train(inputs={})

        Args:
            additional_parents (set{str}): Set of additional parents along with
                the self to be used in warm starting
            model_trainer (sagemaker.train.model_trainer.ModelTrainer): A ModelTrainer object
                that has been initialized with the desired configuration. There
                does not need to be a training job associated with this
                instance.

        Returns:
            sagemaker.core.shapes.HyperparameterTuner: ``HyperparameterTuner``
            instance which can be used to launch transfer learning tuning job.
        """

        return self._create_warm_start_tuner(
            additional_parents=additional_parents,
            warm_start_type=WarmStartTypes.TRANSFER_LEARNING,
            model_trainer=model_trainer,
        )

    def _create_warm_start_tuner(self, additional_parents, warm_start_type, model_trainer=None):
        """Creates a new ``HyperparameterTuner`` with ``HyperParameterTuningJobWarmStartConfig``.

        Where type will be equal to ``warm_start_type`` and``parents`` would be equal
        to union of ``additional_parents`` and self.

        Args:
            additional_parents (set{str}): Additional parents along with self,
                to be used for warm starting.
            warm_start_type (sagemaker.core.shapes.WarmStartTypes): Type of warm start
                job.
            model_trainer:

        Returns:
            sagemaker.core.shapes.HyperparameterTuner: Instance with the request
            fields copied from self along with the warm start configuration
        """
        self._ensure_last_tuning_job()
        all_parents = {self.latest_tuning_job.hyper_parameter_tuning_job_name}
        if additional_parents:
            all_parents = all_parents.union(additional_parents)

        if self.model_trainer is not None:
            return HyperparameterTuner(
                model_trainer=model_trainer if model_trainer else self.model_trainer,
                objective_metric_name=self.objective_metric_name,
                hyperparameter_ranges=self._hyperparameter_ranges,
                strategy=self.strategy,
                strategy_config=self.strategy_config,
                completion_criteria_config=self.completion_criteria_config,
                objective_type=self.objective_type,
                max_jobs=self.max_jobs,
                max_parallel_jobs=self.max_parallel_jobs,
                max_runtime_in_seconds=self.max_runtime_in_seconds,
                warm_start_config=HyperParameterTuningJobWarmStartConfig(
                    warm_start_type=warm_start_type, parents=all_parents
                ),
                early_stopping_type=self.early_stopping_type,
                random_seed=self.random_seed,
            )

        if len(self.model_trainer_dict) > 1:
            raise ValueError(
                "Warm start is not supported currently for tuners with multiple model_trainers"
            )

        if model_trainer is not None:
            model_trainer_name = list(self.model_trainer_dict.keys())[0]
            model_trainer_dict = {model_trainer_name: model_trainer}
        else:
            model_trainer_dict = self.model_trainer_dict

        return HyperparameterTuner.create(
            model_trainer_dict=model_trainer_dict,
            objective_metric_name_dict=self.objective_metric_name_dict,
            hyperparameter_ranges_dict=self._hyperparameter_ranges_dict,
            metric_definitions_dict=self.metric_definitions_dict,
            strategy=self.strategy,
            strategy_config=self.strategy_config,
            completion_criteria_config=self.completion_criteria_config,
            objective_type=self.objective_type,
            max_jobs=self.max_jobs,
            max_parallel_jobs=self.max_parallel_jobs,
            max_runtime_in_seconds=self.max_runtime_in_seconds,
            warm_start_config=HyperParameterTuningJobWarmStartConfig(warm_start_type=warm_start_type, parents=all_parents),
            early_stopping_type=self.early_stopping_type,
            random_seed=self.random_seed,
        )

    @classmethod
    def create(
        cls,
        model_trainer_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
        base_tuning_job_name=None,
        strategy="Bayesian",
        strategy_config=None,
        completion_criteria_config=None,
        objective_type="Maximize",
        max_jobs=None,
        max_parallel_jobs=1,
        max_runtime_in_seconds=None,
        tags=None,
        warm_start_config=None,
        early_stopping_type="Off",
        random_seed=None,
        autotune=False,
        hyperparameters_to_keep_static_dict=None,
    ):
        """Factory method to create a ``HyperparameterTuner`` instance.

        It takes one or more model_trainers to obtain configuration information for training jobs
        that are created as the result of a hyperparameter tuning job. The model_trainers are provided
        through a  dictionary (i.e. ``model_trainer_dict``) with unique model_trainer names as the keys.
        For  individual model_trainers separate objective metric names and hyperparameter ranges
        should be provided in two dictionaries, i.e. ``objective_metric_name_dict`` and
        ``hyperparameter_ranges_dict``, with the same model_trainer names as the keys. Optional
        metrics definitions could also be provided for individual model_trainers via another dictionary
        ``metric_definitions_dict``.

        Args:
            model_trainer_dict (dict[str, sagemaker.train.model_trainer.ModelTrainer]): Dictionary of model_trainer
                instances that have been initialized with the desired configuration. There does not
                need to be a training job associated with the model_trainer instances. The keys of the
                dictionary would be referred to as "model_trainer names".
            objective_metric_name_dict (dict[str, str]): Dictionary of names of the objective
                metric for evaluating training jobs. The keys are the same set of model_trainer names
                as in ``model_trainer_dict``, and there must be one entry for each model_trainer in
                ``model_trainer_dict``.
            hyperparameter_ranges_dict (dict[str, dict[str, sagemaker.parameter.ParameterRange]]):
                Dictionary of tunable hyperparameter ranges. The keys are the same set of model_trainer
                names as in model_trainer_dict, and there must be one entry for each model_trainer in
                model_trainer_dict. Each value is a dictionary of sagemaker.parameter.ParameterRange
                instance, which can be one of three types: Continuous, Integer, or Categorical.
                The keys of each ParameterRange dictionaries are the names of the hyperparameter,
                and the values are the appropriate parameter range class to represent the range.
            metric_definitions_dict (dict(str, list[dict]]): Dictionary of metric definitions.
                The keys are the same set or a subset of model_trainer names as in model_trainer_dict,
                and there must be one entry for each model_trainer in model_trainer_dict. Each value is
                a list of dictionaries that defines the metric(s) used to evaluate the training
                jobs (default: None). Each of these dictionaries contains two keys: 'Name' for the
                name of the metric, and 'Regex' for the regular expression used to extract the
                metric from the logs. This should be defined only for hyperparameter tuning jobs
                that don't use an Amazon algorithm.
            base_tuning_job_name (str): Prefix for the hyperparameter tuning job name when the
                :meth:`~sagemaker.core.shapes.HyperparameterTuner.train` method launches.
                If not specified, a default job name is generated,
                based on the training image name and current timestamp.
            strategy (str or PipelineVariable): Strategy to be used for hyperparameter estimations.
                More information about different strategies:
                https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html.
                Available options are: 'Bayesian', 'Random', 'Hyperband',
                'Grid' (default: 'Bayesian')
            strategy_config (dict): The configuration for a training job launched by a
                hyperparameter tuning job.
            completion_criteria_config (dict): The configuration for tuning job completion criteria.
            objective_type (str): The type of the objective metric for evaluating training jobs.
                This value can be either 'Minimize' or 'Maximize' (default: 'Maximize').
            max_jobs (int): Maximum total number of training jobs to start for the hyperparameter
                tuning job. The default value is unspecified fot the 'Grid' strategy
                and the value is 1 for all others strategies (default: None).
            max_parallel_jobs (int): Maximum number of parallel training jobs to start
                (default: 1).
            max_runtime_in_seconds (int): The maximum time in seconds
                 that a hyperparameter tuning job can run.
            tags (Optional[Tags]): List of tags for labeling the tuning job (default: None).
                For more,
                see https://docs.aws.amazon.com/sagemaker/latest/dg/API_Tag.html.
            warm_start_config (sagemaker.core.shapes.HyperParameterTuningJobWarmStartConfig): A ``HyperParameterTuningJobWarmStartConfig`` object that
                has been initialized with the configuration defining the nature of warm start
                tuning job.
            early_stopping_type (str): Specifies whether early stopping is enabled for the job.
                Can be either 'Auto' or 'Off' (default: 'Off'). If set to 'Off', early stopping
                will not be attempted. If set to 'Auto', early stopping of some training jobs may
                happen, but is not guaranteed to.
            random_seed (int): An initial value used to initialize a pseudo-random number generator.
                Setting a random seed will make the hyperparameter tuning search strategies to
                produce more consistent configurations for the same tuning job.
            autotune (bool): Whether the parameter ranges or other unset settings of a tuning job
                should be chosen automatically (default: False).
            hyperparameters_to_keep_static_dict (dict(str, list[str]]): Dictionary of
                hyperparameter names that will be kept static. The keys are the same set or a subset
                of model_trainer names as in model_trainer_dict, and there must be one entry for each
                model_trainer in model_trainer_dict. Each value is a list of hyperparameter names that will
                be kept static and will not be assigned a tunable range with Autotune functionality
                (default: None).

        Returns:
            sagemaker.core.shapes.HyperparameterTuner: a new ``HyperparameterTuner`` object that can
            start a hyperparameter tuning job with one or more model_trainers.

        """

        cls._validate_create_tuner_inputs(
            model_trainer_dict,
            objective_metric_name_dict,
            hyperparameter_ranges_dict,
            metric_definitions_dict,
            hyperparameters_to_keep_static_dict,
        )

        model_trainer_names = sorted(model_trainer_dict.keys())
        first_model_trainer_name = model_trainer_names[0]

        metric_definitions = (
            metric_definitions_dict.get(first_model_trainer_name, None)
            if metric_definitions_dict is not None
            else None
        )

        hyperparameters_to_keep_static = (
            hyperparameters_to_keep_static_dict.get(first_model_trainer_name, None)
            if hyperparameters_to_keep_static_dict is not None
            else None
        )

        tuner = HyperparameterTuner(
            base_tuning_job_name=base_tuning_job_name,
            model_trainer_name=first_model_trainer_name,
            model_trainer=model_trainer_dict[first_model_trainer_name],
            objective_metric_name=objective_metric_name_dict[first_model_trainer_name],
            hyperparameter_ranges=hyperparameter_ranges_dict[first_model_trainer_name],
            metric_definitions=metric_definitions,
            strategy=strategy,
            strategy_config=strategy_config,
            completion_criteria_config=completion_criteria_config,
            objective_type=objective_type,
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            max_runtime_in_seconds=max_runtime_in_seconds,
            tags=format_tags(tags),
            warm_start_config=warm_start_config,
            early_stopping_type=early_stopping_type,
            random_seed=random_seed,
            autotune=autotune,
            hyperparameters_to_keep_static=hyperparameters_to_keep_static,
        )

        for model_trainer_name in model_trainer_names[1:]:
            metric_definitions = (
                metric_definitions_dict.get(model_trainer_name, None)
                if metric_definitions_dict is not None
                else None
            )
            hyperparameters_to_keep_static = (
                hyperparameters_to_keep_static_dict.get(model_trainer_name, None)
                if hyperparameters_to_keep_static_dict is not None
                else None
            )
            tuner._add_model_trainer(
                model_trainer_name=model_trainer_name,
                model_trainer=model_trainer_dict[model_trainer_name],
                objective_metric_name=objective_metric_name_dict[model_trainer_name],
                hyperparameter_ranges=hyperparameter_ranges_dict[model_trainer_name],
                metric_definitions=metric_definitions,
                hyperparameters_to_keep_static=hyperparameters_to_keep_static,
            )
        return tuner

    @classmethod
    def _validate_create_tuner_inputs(
        cls,
        model_trainer_dict,
        objective_metric_name_dict,
        hyperparameter_ranges_dict,
        metric_definitions_dict=None,
        hyperparameters_to_keep_static_dict=None,
    ):
        """Validate inputs for ``HyperparameterTuner.create()``"""
        cls._validate_model_trainer_dict(model_trainer_dict)

        model_trainer_names = sorted(model_trainer_dict.keys())

        cls._validate_dict_argument(
            name="objective_metric_name_dict",
            value=objective_metric_name_dict,
            allowed_keys=model_trainer_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="hyperparameter_ranges_dict",
            value=hyperparameter_ranges_dict,
            allowed_keys=model_trainer_names,
            require_same_keys=True,
        )
        cls._validate_dict_argument(
            name="metric_definitions_dict",
            value=metric_definitions_dict,
            allowed_keys=model_trainer_names,
        )
        cls._validate_dict_argument(
            name="hyperparameters_to_keep_static_dict",
            value=hyperparameters_to_keep_static_dict,
            allowed_keys=model_trainer_names,
        )

    @classmethod
    def _validate_model_trainer_dict(cls, model_trainer_dict):
        """Validate ``model_trainer_dict`` in inputs for ``HyperparameterTuner.create()``"""
        if model_trainer_dict is None or len(model_trainer_dict) == 0:
            raise ValueError("At least one model_trainer should be provided")
        if None in model_trainer_dict.keys():
            raise ValueError("ModelTrainer names cannot be None")

    @classmethod
    def _validate_dict_argument(cls, name, value, allowed_keys, require_same_keys=False):
        """Check if an argument is an dictionary with correct key set."""
        if value is None:
            return

        if not isinstance(value, dict):
            raise ValueError(f"Argument '{name}' must be a dictionary using {allowed_keys} as keys")

        value_keys = sorted(value.keys())

        if require_same_keys:
            if value_keys != allowed_keys:
                raise ValueError(
                    f"The keys of argument '{name}' must be the same as {allowed_keys}"
                )
        else:
            if not set(value_keys).issubset(set(allowed_keys)):
                raise ValueError(
                    f"The keys of argument '{name}' must be a subset of {allowed_keys}"
                )

    def _add_model_trainer(
        self,
        model_trainer_name,
        model_trainer,
        objective_metric_name,
        hyperparameter_ranges,
        metric_definitions=None,
        hyperparameters_to_keep_static=None,
    ):
        """Add a model_trainer with corresponding attributes, if applicable.

        The objective metric name, parameter ranges and metric definitions are added to
        the model_trainer, if populated.
        """
        self.model_trainer_dict[model_trainer_name] = model_trainer
        self.objective_metric_name_dict[model_trainer_name] = objective_metric_name
        self._hyperparameter_ranges_dict[model_trainer_name] = hyperparameter_ranges
        if hyperparameters_to_keep_static is not None:
            self.hyperparameters_to_keep_static_dict[model_trainer_name] = (
                hyperparameters_to_keep_static
            )
        if metric_definitions is not None:
            self.metric_definitions_dict[model_trainer_name] = metric_definitions


    def _start_tuning_job(self, inputs):
        """Start a new hyperparameter tuning job using HyperParameterTuningJob."""
        tuning_job_config = self._build_tuning_job_config()
        training_job_definition = self._build_training_job_definition(inputs)
        
        # Prepare autotune parameter
        autotune_param = None
        if self.autotune:
            from sagemaker.core.shapes import Autotune
            autotune_param = Autotune(mode="Enabled")
        
        # Convert tags to proper Tag objects
        tag_objects = None
        if self.tags:
            from sagemaker.core.shapes import Tag
            tag_objects = [Tag(key=tag["Key"], value=tag["Value"]) for tag in self.tags]
        
        # Build tuning request
        tuning_request = {
            "hyper_parameter_tuning_job_name": self._current_job_name,
            "hyper_parameter_tuning_job_config": tuning_job_config,
            "training_job_definition": training_job_definition,
            "warm_start_config": self.warm_start_config,
            "tags": tag_objects,
            "autotune": autotune_param,
        }
        
        # Handle PipelineSession
        if isinstance(self.sagemaker_session, PipelineSession):
            from sagemaker.core.utils.utils import serialize
            from sagemaker.core.apiutils._boto_functions import to_pascal_case
            
            # Remove job name for pipeline as it's auto-generated at execution time
            tuning_request.pop("hyper_parameter_tuning_job_name", None)
            # Convert snake_case to PascalCase for AWS API
            pipeline_request = {to_pascal_case(k): v for k, v in tuning_request.items()}
            serialized_request = serialize(pipeline_request)
            self.sagemaker_session._intercept_create_request(serialized_request, None, "tune")
            return None
        
        # Create the tuning job using HyperParameterTuningJob for regular session
        tuning_job = HyperParameterTuningJob.create(
            session=self.sagemaker_session.boto_session if hasattr(self.sagemaker_session, 'boto_session') else None,
            region=self.sagemaker_session.boto_region_name if hasattr(self.sagemaker_session, 'boto_region_name') else None,
            **tuning_request
        )
        
        return tuning_job
    
    def _build_tuning_job_config(self):
        """Build the hyperparameter tuning job configuration."""
        from sagemaker.core.shapes import (
            HyperParameterTuningJobConfig,
            HyperParameterTuningJobObjective,
            ResourceLimits,
            ParameterRanges
        )
        
        # Build objective
        objective = None
        if self.objective_metric_name:
            objective = HyperParameterTuningJobObjective(
                type=self.objective_type,
                metric_name=self.objective_metric_name
            )
        
        # Build resource limits
        resource_limits = ResourceLimits(
            max_number_of_training_jobs=self.max_jobs,
            max_parallel_training_jobs=self.max_parallel_jobs
        )
        
        if self.max_runtime_in_seconds:
            resource_limits.max_runtime_in_seconds = self.max_runtime_in_seconds
        
        # Build parameter ranges
        parameter_ranges = None
        if self._hyperparameter_ranges:
            ranges_dict = self.hyperparameter_ranges()
            parameter_ranges = ParameterRanges(
                integer_parameter_ranges=ranges_dict.get("IntegerParameterRanges", []),
                continuous_parameter_ranges=ranges_dict.get("ContinuousParameterRanges", []),
                categorical_parameter_ranges=ranges_dict.get("CategoricalParameterRanges", [])
            )
        
        config = HyperParameterTuningJobConfig(
            strategy=self.strategy,
            hyper_parameter_tuning_job_objective=objective,
            resource_limits=resource_limits,
            parameter_ranges=parameter_ranges,
            training_job_early_stopping_type=self.early_stopping_type
        )
        
        if self.random_seed:
            config.random_seed = self.random_seed
        
        if self.strategy_config:
            config.strategy_config = self.strategy_config
        
        if self.completion_criteria_config:
            config.tuning_job_completion_criteria = self.completion_criteria_config
        
        return config
    
    def _build_training_job_definition(self, inputs):
        """Build the training job definition for the tuning job."""
        from sagemaker.core.shapes import (
            HyperParameterTrainingJobDefinition,
            HyperParameterAlgorithmSpecification,
            OutputDataConfig,
            ResourceConfig,
            StoppingCondition,
            Channel,
            DataSource,
            S3DataSource
        )
        
        model_trainer = self.model_trainer
        
        # Build algorithm specification - use HyperParameterAlgorithmSpecification for tuning
        algorithm_spec = HyperParameterAlgorithmSpecification(
            training_image=model_trainer.training_image,
            training_input_mode=model_trainer.training_input_mode or "File"
        )
        
        if self.metric_definitions:
            # Convert metric definitions to snake_case for v3 Pydantic models
            metric_defs_snake = []
            for metric_def in self.metric_definitions:
                metric_def_snake = {}
                for key, value in metric_def.items():
                    # Convert PascalCase to snake_case
                    snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
                    metric_def_snake[snake_key] = value
                metric_defs_snake.append(metric_def_snake)
            algorithm_spec.metric_definitions = metric_defs_snake
        
        # Build input data config from inputs
        input_data_config = []
        if inputs:
            if isinstance(inputs, str):
                # Single S3 URI string
                input_data_config = [Channel(
                    channel_name="training",
                    data_source=DataSource(
                        s3_data_source=S3DataSource(
                            s3_data_type="S3Prefix",
                            s3_uri=inputs,
                            s3_data_distribution_type="FullyReplicated"
                        )
                    )
                )]
            elif isinstance(inputs, list):
                # List of InputData or Channel objects
                for inp in inputs:
                    if isinstance(inp, InputData):
                        # Convert InputData to Channel
                        input_data_config.append(Channel(
                            channel_name=inp.channel_name,
                            data_source=DataSource(
                                s3_data_source=S3DataSource(
                                    s3_data_type="S3Prefix",
                                    s3_uri=inp.data_source,
                                    s3_data_distribution_type="FullyReplicated"
                                )
                            )
                        ))
                    elif isinstance(inp, Channel):
                        # Already a Channel object
                        input_data_config.append(inp)
            elif isinstance(inputs, dict):
                # Dict mapping channel names to S3 URIs
                for channel_name, s3_uri in inputs.items():
                    input_data_config.append(Channel(
                        channel_name=channel_name,
                        data_source=DataSource(
                            s3_data_source=S3DataSource(
                                s3_data_type="S3Prefix",
                                s3_uri=s3_uri,
                                s3_data_distribution_type="FullyReplicated"
                            )
                        )
                    ))
        
        # Build output data config
        output_config = OutputDataConfig(
            s3_output_path=model_trainer.output_data_config.s3_output_path if model_trainer.output_data_config else None
        )
        
        # Build resource config
        resource_config = ResourceConfig(
            instance_type=model_trainer.compute.instance_type if model_trainer.compute else "ml.m5.xlarge",
            instance_count=model_trainer.compute.instance_count if model_trainer.compute else 1,
            volume_size_in_gb=model_trainer.compute.volume_size_in_gb if model_trainer.compute else 30
        )
        
        # Build stopping condition
        stopping_condition = StoppingCondition()
        if model_trainer.stopping_condition and model_trainer.stopping_condition.max_runtime_in_seconds:
            stopping_condition.max_runtime_in_seconds = model_trainer.stopping_condition.max_runtime_in_seconds
        
        definition = HyperParameterTrainingJobDefinition(
            algorithm_specification=algorithm_spec,
            role_arn=model_trainer.role,
            input_data_config=input_data_config if input_data_config else None,
            output_data_config=output_config,
            resource_config=resource_config,
            stopping_condition=stopping_condition,
            static_hyper_parameters=self.static_hyperparameters or {}
        )
        
        return definition
