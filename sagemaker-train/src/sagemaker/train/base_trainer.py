import copy
import os
import time
import yaml
from abc import ABC, abstractmethod
from datetime import datetime as _datetime
from typing import Optional, Dict, Any, List, Union
import json
import logging
import re
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse

import yaml
import boto3

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Tag, Networking, InputData, Channel, OutputDataConfig, HyperPodCompute
from sagemaker.core.utils.logs import MultiLogStreamHandler
from sagemaker.core.shapes import shapes
from sagemaker.core.resources import TrainingJob
from sagemaker.train.common_utils.recipe_utils import _is_nova_model, resolve_recipe, get_resolved_recipe_from_context, NoRecipeError
from sagemaker.core.s3.utils import resolve_s3_uri_placeholders
from sagemaker.train.recipe_resolver import flatten_resolved_recipe
from sagemaker.train.common_utils.finetune_utils import (
    get_training_image,
    get_hyperpod_training_image,
    get_hyperpod_recipe_path,
    get_recipe_s3_uri,
    _validate_hyperparameter_values,
    _get_smhp_replicas_enum,
)
from sagemaker.train.common_utils.metrics_visualizer import plot_training_metrics
from sagemaker.train.common_utils.mlflow_config_utils import resolve_mlflow_tracking_fields
from sagemaker.train.common_utils.validator import validate_hyperpod_compute
from sagemaker.train.common_utils.cloudwatch_metrics import fetch_and_plot_metrics, _get_smhp_log_group
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base class for all SageMaker training workflows.

    This class provides the common interface and shared functionality for all trainer implementations
    including SFT, DPO, RLVR, and RLAIF trainers. It defines the standard parameters and abstract
    methods that concrete trainer classes must implement.

    Parameters:
        sagemaker_session (Optional[Session]):
            The SageMaker session for managing API calls and resources.
            If not specified, a default session will be created.
        role (Optional[str]):
            The IAM role ARN for the training job execution.
            If not specified, the default SageMaker execution role will be used.
        base_job_name (Optional[str]):
            The base name for training jobs. A unique suffix will be appended.
            If not specified, a default name will be generated based on the trainer type.
        tags (Optional[List[Tag]]):
            List of tags to apply to the training job for resource management and billing.
        hyperparameters (Optional[Dict[str, Any]]):
            Dictionary of hyperparameters for the training job.
            Trainer-specific defaults will be applied if not specified.
        output_data_config (Optional[shapes.OutputDataConfig]):
            Configuration for training job outputs including S3 paths and encryption.
            If not specified, default output configuration will be used.
        input_data_config (Optional[List[Union[Channel, InputData]]]):
            List of input data channels for the training job.
            Can include training and validation datasets.
        environment (Optional[Dict[str, str]]):
            Environment variables to set in the training container.
        training_image (Optional[str]):
            Custom training container image URI. If not provided, the image is
            auto-resolved from the model's recipe metadata in SageMaker Hub.
    """
    
    # Class-level attributes with default values
    sagemaker_session: Optional[Session] = None
    role: Optional[str] = None
    base_job_name: Optional[str] = None
    tags: Optional[List[Tag]] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    output_data_config: Optional[shapes.OutputDataConfig] = None
    input_data_config: Optional[List[Union[Channel, InputData]]] = None
    environment: Optional[Dict[str, str]] = None
    training_image: Optional[str] = None
    latest_training_job: Optional[TrainingJob] = None

    def __init__(
        self,
        sagemaker_session: Optional[Session] = None,
        role: Optional[str] = None,
        base_job_name: Optional[str] = None,
        tags: Optional[List[Tag]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_data_config: Optional[shapes.OutputDataConfig] = None,
        input_data_config: Optional[List[Union[Channel, InputData]]] = None,
        environment: Optional[Dict[str, str]] = None,
        training_image: Optional[str] = None,
        base_model_name: Optional[str] = None,
        disable_output_compression: Optional[bool] = False,
    ):
        self.sagemaker_session = sagemaker_session
        self.role = role
        self.base_job_name = base_job_name
        self.tags = tags
        self.hyperparameters = hyperparameters or {}
        self.output_data_config = output_data_config
        self.input_data_config = input_data_config
        self.environment = environment or {}
        self.training_image = training_image
        self.base_model_name = base_model_name
        self.disable_output_compression = disable_output_compression
        self._checkpoint_s3_uri = None

    def _is_nova_model_for_telemetry(self) -> bool:
        """Check if the model is a Nova model for telemetry tracking."""
        model_name = getattr(self, "_model_name", None)
        return _is_nova_model(model_name) if model_name else False

    def get_resolved_recipe(self) -> Dict[str, Any]:
        """Return the fully resolved recipe configuration.

        Shows the final merged result of base defaults + user recipe + overrides
        after interpolation resolution and validation. Callable before or after train().

        When neither ``recipe`` nor ``overrides`` were provided at construction time
        but hyperparameters have been set directly (e.g. ``trainer.hyperparameters.x = val``),
        those user-set values are treated as implicit overrides so the resolved recipe
        still reflects the user's intent.

        Returns:
            dict: Deep copy of the resolved recipe configuration.

        Raises:
            ValueError: If no recipe, overrides, or direct hyperparameter assignments
                were provided.
        """
        # Fetch full recipe template from Hub to preserve YAML structure
        full_recipe_template = self._fetch_full_recipe_template()

        resolved = get_resolved_recipe_from_context(
            recipe_path=getattr(self, '_recipe_path', None),
            overrides=getattr(self, '_overrides', None),
            hyperparameters=self.hyperparameters if hasattr(self, 'hyperparameters') else None,
            resolved_cache=getattr(self, '_resolved_recipe_cache', None),
            template_section="training_config",
            protected_keys={"model_type", "model_name_or_path", "dataset_catalog"},
            full_recipe_template=full_recipe_template,
            compute=getattr(self, 'compute', None),
        )

        # Post-resolution patches for display accuracy
        self._patch_resolved_recipe(resolved)

        self._resolved_recipe_cache = resolved
        return copy.deepcopy(resolved)

    def _fetch_full_recipe_template(self) -> Optional[Dict[str, Any]]:
        """Fetch the full recipe template from Hub to preserve YAML structure.

        Returns None if the template can't be fetched (fallback to synthetic template).
        """
        logger = logging.getLogger(__name__)
        frt = getattr(self.hyperparameters, '_full_recipe_template', None) if hasattr(self, 'hyperparameters') else None
        if isinstance(frt, dict):
            return frt

        if not hasattr(self, '_model_name') or not hasattr(self, '_customization_technique'):
            return None

        try:
            from sagemaker.core.training.configs import HyperPodCompute
            from sagemaker.train.common_utils.finetune_utils import (
                _get_recipe_entry_and_override_spec,
                _extract_recipe_from_helm_template,
            )

            is_hyperpod = isinstance(getattr(self, 'compute', None), HyperPodCompute)
            sagemaker_session = TrainDefaults.get_sagemaker_session(sagemaker_session=self.sagemaker_session)
            platform = "hyperpod" if is_hyperpod else "smtj"

            recipe_entry, _ = _get_recipe_entry_and_override_spec(
                model_name=self._model_name,
                customization_technique=self._customization_technique,
                training_type=self.training_type,
                sagemaker_session=sagemaker_session,
                platform=platform,
            )

            s3_client = sagemaker_session.boto_session.client("s3")

            if is_hyperpod:
                hp_uri = recipe_entry["HpEksPayloadTemplateS3Uri"]
                bucket, key = hp_uri.replace("s3://", "").split("/", 1)
                raw = s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
                return yaml.safe_load(_extract_recipe_from_helm_template(raw))
            else:
                smtj_uri = resolve_s3_uri_placeholders(recipe_entry["SmtjRecipeTemplateS3Uri"], sagemaker_session)
                uri_path = smtj_uri.replace("s3://", "")
                if uri_path.startswith("arn:"):
                    match = re.match(r'(arn:aws:s3:[^:]*:[^:]*:accesspoint/[^/]+)/(.*)', uri_path)
                    bucket, key = (match.group(1), match.group(2)) if match else uri_path.split("/", 1)
                else:
                    bucket, key = uri_path.split("/", 1)
                tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False)
                s3_client.download_file(bucket, key, tmp.name)
                with open(tmp.name, "r") as f:
                    return yaml.safe_load(f)
        except Exception as e:
            logger.debug(f"Could not fetch full recipe template: {e}")
            return None

    def _patch_resolved_recipe(self, resolved: Dict[str, Any]) -> None:
        """Apply post-resolution patches to make the preview match actual job config."""
        from sagemaker.train.recipe_resolver import _set_nested_value, _build_key_path_map

        # Build a map of where keys live in the resolved structure
        patch_values = {}

        # base_job_name → name
        if self.base_job_name:
            patch_values["name"] = _get_unique_name(self.base_job_name)

        # output_s3_path and data_s3_path from trainer config
        if getattr(self, 's3_output_path', None):
            patch_values["output_s3_path"] = self.s3_output_path
        if getattr(self, 'training_dataset', None):
            patch_values["data_s3_path"] = self.training_dataset

        # Subclass-specific hyperparameters (e.g. reward_lambda_arn for RLVR)
        patch_values.update(self._get_extra_smtj_hyperparameters())

        if not patch_values:
            return

        # Find where each key lives in the resolved dict and set it
        key_path_map = _build_key_path_map(resolved, set(patch_values.keys()))
        for key, value in patch_values.items():
            dotpath = key_path_map.get(key)
            if dotpath:
                _set_nested_value(resolved, dotpath, value)

    def _apply_recipe_to_hyperparameters(self, final_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resolved recipe values to final_hyperparameters dict.

        If recipe/overrides were provided, or if the user set hyperparameters
        directly via ``.hyperparameters.*``, merges resolved recipe values into
        the hyperparameters dict. All leaf values from the resolved recipe are
        applied — including keys not in the Hub spec subset — enabling
        power users to override any parameter in the full recipe.
        Values are converted to strings (matching the SageMaker API
        expectation for hyperparameter values).

        Args:
            final_hyperparameters: The hyperparameters dict from to_dict().

        Returns:
            The updated hyperparameters dict with recipe values applied.
        """
        if not hasattr(self, 'hyperparameters') or not isinstance(getattr(self.hyperparameters, '_specs', None), dict):
            return final_hyperparameters

        try:
            resolved = self.get_resolved_recipe()
        except NoRecipeError:
            return final_hyperparameters

        flat = flatten_resolved_recipe(resolved)
        for k, v in flat.items():
            if v is not None:
                final_hyperparameters[k] = str(v) if not isinstance(v, str) else v

        return final_hyperparameters

    def show_metrics(
        self,
        metrics: Optional[List[str]] = None,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
    ) -> Any:
        """Plot training metrics from CloudWatch logs (Nova) or MLflow (OSS).

        For Nova models, parses CloudWatch logs for training_loss, lr, and reward_score.
        For non-Nova (OSS) models, pulls metrics from MLflow (requires mlflow_resource_arn
        to be configured on the trainer or auto-resolved).

        Args:
            metrics: Optional list of metric names to plot. If None, plots all
                available metrics for the training technique. 
            starting_step: Only plot metrics from this global step onwards.
            ending_step: Only plot metrics up to this global step.
            start_time: Optional start time for log retrieval. Accepts a
                datetime object or epoch milliseconds (int). When not provided,
                auto-resolved from the training job's start time.
            end_time: Optional end time for log retrieval. Accepts a
                datetime object or epoch milliseconds (int). When not provided,
                defaults to now.

        Returns:
            pandas.DataFrame containing the extracted metrics.

        Raises:
            NotImplementedError: If the training technique does not support metric
                extraction (e.g., DPO).
            ValueError: If no training job has been run yet, no logs/metrics
                are found, or MLflow is not configured for OSS models.
        """
        # Validate that we have a training job to get metrics from
        if not hasattr(self, '_latest_training_job') or self._latest_training_job is None:
            raise ValueError(
                "No training job found. Call .train() first, then call .show_metrics() "
                "to view training metrics."
            )

        # Route based on model type
        model_name = getattr(self, '_model_name', None)
        is_nova = _is_nova_model(model_name) if model_name else False

        if is_nova:
            return self._show_metrics_cloudwatch(metrics, starting_step, ending_step, start_time, end_time)
        else:
            return self._show_metrics_mlflow(metrics, starting_step, ending_step)

    def _show_metrics_mlflow(
        self,
        metrics: Optional[List[str]] = None,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
    ) -> None:
        """Pull and plot training metrics from MLflow for non-Nova models."""
        training_job = self._latest_training_job

        # Resolve the TrainingJob object if it's a string
        if isinstance(training_job, str):
            logger.info(f"Resolving training job: {training_job}")
            training_job = TrainingJob.get(training_job_name=training_job)

        # Validate MLflow is configured
        mlflow_config = getattr(training_job, 'mlflow_config', None)
        if not mlflow_config or not getattr(mlflow_config, 'mlflow_resource_arn', None):
            raise ValueError(
                "show_metrics() for non-Nova models requires MLflow to be configured. "
                "Either pass mlflow_resource_arn when creating the trainer, or ensure "
                "your account has an MLflow app set up."
            )

        mlflow_details = getattr(training_job, 'mlflow_details', None)
        if not mlflow_details or not getattr(mlflow_details, 'mlflow_run_id', None):
            raise ValueError(
                "No MLflow run ID found on the training job. "
                "MLflow metrics are only available after the job completes. "
                "If the job is still running, wait for it to finish and try again. "
                f"MLflow app ARN: {mlflow_config.mlflow_resource_arn}"
            )

        logger.info(
            f"Fetching metrics from MLflow app: {mlflow_config.mlflow_resource_arn}, "
            f"run: {mlflow_details.mlflow_run_id}"
        )

        plot_training_metrics(training_job, metrics=metrics)

    def _show_metrics_cloudwatch(
        self,
        metrics: Optional[List[str]] = None,
        starting_step: Optional[int] = None,
        ending_step: Optional[int] = None,
        start_time: Optional[Any] = None,
        end_time: Optional[Any] = None,
    ) -> Any:
        """Parse and plot training metrics from CloudWatch logs (Nova models)."""
        
        training_job = self._latest_training_job
        if hasattr(training_job, 'training_job_name'):
            job_id = training_job.training_job_name
        elif isinstance(training_job, str):
            job_id = training_job
        else:
            job_id = str(training_job)

        # Determine platform from compute config
        compute = getattr(self, 'compute', None)

        # Get customization technique
        customization_technique = getattr(self, '_customization_technique', None)
        if not customization_technique:
            raise ValueError(
                "Could not determine training technique. "
                "show_metrics() requires a trainer with a known customization technique."
            )

        # Resolve session
        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )

        # Resolve start_time: user-provided > training job metadata > None
        start_time_ms = None
        if start_time is not None:
            if isinstance(start_time, _datetime):
                start_time_ms = int(start_time.timestamp() * 1000)
            else:
                start_time_ms = int(start_time)
        elif hasattr(training_job, 'training_start_time') and training_job.training_start_time:
            try:
                start_time_ms = int(training_job.training_start_time.timestamp() * 1000)
            except Exception:
                pass

        # Resolve end_time: user-provided > None (defaults to now in fetch layer)
        end_time_ms = None
        if end_time is not None:
            if isinstance(end_time, _datetime):
                end_time_ms = int(end_time.timestamp() * 1000)
            else:
                end_time_ms = int(end_time)

        return fetch_and_plot_metrics(
            job_id=job_id,
            compute=compute,
            customization_technique=customization_technique,
            sagemaker_session=sagemaker_session,
            metrics=metrics,
            starting_step=starting_step,
            ending_step=ending_step,
            start_time=start_time_ms,
            end_time=end_time_ms,
        )

    def stream_logs(self, poll: int = 5, start_time: Optional[Any] = None) -> None:
        """Stream CloudWatch logs in real-time (like ``kubectl logs -f``).

        Continuously polls for new log events and prints them as they arrive.
        Blocks until the training job reaches a terminal state (SMTJ) or
        the user interrupts with Ctrl+C (HyperPod).

        Args:
            poll: Polling interval in seconds between log fetches. Defaults to 5.
            start_time: Optional start time to stream logs from. Accepts a
                datetime object or epoch milliseconds (int). Useful when
                attaching to a job that's already running. If not provided,
                auto-resolved from the training job's start time (SMTJ) or
                defaults to now (HyperPod).

        Raises:
            ValueError: If no training job has been run yet.
        """
        if not hasattr(self, '_latest_training_job') or self._latest_training_job is None:
            raise ValueError(
                "No training job found. Call .train(wait=False) first, "
                "then call .stream_logs() to stream logs in real-time."
            )

        # Resolve start_time for SMHP jobs
        start_time_ms = None
        if start_time is not None:
            if isinstance(start_time, _datetime):
                start_time_ms = int(start_time.timestamp() * 1000)
            else:
                start_time_ms = int(start_time)

        training_job = self._latest_training_job
        compute = getattr(self, 'compute', None)

        if isinstance(compute, HyperPodCompute):
            self._stream_logs_smhp(training_job, compute, poll, start_time_ms)
        else:
            self._stream_logs_smtj(training_job, poll)

    def _stream_logs_smtj(self, training_job, poll: int) -> None:
        """Stream logs for an SMTJ training job using MultiLogStreamHandler."""

        # Resolve job name
        if hasattr(training_job, 'training_job_name'):
            job_name = training_job.training_job_name
        else:
            job_name = str(training_job)

        log_group = "/aws/sagemaker/TrainingJobs"
        instance_count = 1
        if hasattr(self, 'compute') and self.compute and hasattr(self.compute, 'instance_count'):
            instance_count = self.compute.instance_count or 1

        handler = MultiLogStreamHandler(
            log_group_name=log_group,
            log_stream_name_prefix=job_name,
            expected_stream_count=instance_count,
        )

        logger.info(f"Streaming logs for job: {job_name}")
        logger.info(f"Log group: {log_group}")

        terminal_statuses = {"Completed", "Failed", "Stopped"}

        while True:
            for stream_name, event in handler.get_latest_log_events():
                message = event.get("message", "").rstrip()
                if message:
                    logger.info(message)

            # Check job status
            try:
                job = TrainingJob.get(training_job_name=job_name)
                status = job.training_job_status
                if status in terminal_statuses:
                    # Final flush
                    for stream_name, event in handler.get_latest_log_events():
                        message = event.get("message", "").rstrip()
                        if message:
                            logger.info(message)
                    logger.info(f"Job {job_name} finished with status: {status}")
                    return
            except Exception:
                pass

            time.sleep(poll)

    def _stream_logs_smhp(self, training_job, compute, poll: int, start_time_ms=None) -> None:
        """Stream logs for a HyperPod job using filter_log_events polling."""

        job_id = training_job if isinstance(training_job, str) else str(training_job)

        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )
        region_name = sagemaker_session.boto_session.region_name
        logs_client = sagemaker_session.boto_session.client("logs", region_name=region_name)
        log_group = _get_smhp_log_group(compute.cluster_name, sagemaker_session.sagemaker_client)

        logger.info(f"Streaming logs for HyperPod job: {job_id}")
        logger.info(f"Cluster: {compute.cluster_name}")
        logger.info(f"Log group: {log_group}")
        logger.info("Press Ctrl+C to stop streaming.")

        # Pick start time (user-provided > training job start time > now)
        if start_time_ms is not None:
            last_timestamp = start_time_ms
        elif hasattr(training_job, 'training_start_time') and training_job.training_start_time:
            try:
                last_timestamp = int(training_job.training_start_time.timestamp() * 1000)
            except Exception:
                last_timestamp = int(time.time() * 1000)
        else:
            last_timestamp = int(time.time() * 1000)
        seen_event_ids = set()

        while True:
            try:
                params = {
                    "logGroupName": log_group,
                    "logStreamNamePrefix": "SagemakerHyperPodTrainingJob",
                    "filterPattern": f'"{job_id}"',
                    "startTime": last_timestamp,
                }
                response = logs_client.filter_log_events(**params)
                events = response.get("events", [])

                for event in events:
                    event_id = event.get("eventId", "")
                    if event_id not in seen_event_ids:
                        seen_event_ids.add(event_id)
                        message = event.get("message", "").rstrip()
                        if message:
                            logger.info(message)
                        ts = event.get("timestamp", 0)
                        if ts > last_timestamp:
                            last_timestamp = ts
            except Exception as e:
                logger.debug(f"Error fetching HP logs: {e}")

            # Note: HyperPod jobs don't have a simple status API to poll for completion.
            # This polls till the user interrupts with Ctrl+C. 
            try:
                time.sleep(poll)
            except KeyboardInterrupt:
                logger.info("Log streaming stopped by user.")
                return

    def _validate_instance_count(self, instance_count, sagemaker_session):
        """Validate instance/node count against allowed values from SMHP recipe."""
        smhp_replicas_enum = _get_smhp_replicas_enum(
            model_name=self._model_name,
            customization_technique=self._customization_technique,
            training_type=self.training_type,
            sagemaker_session=sagemaker_session,
        )
        if smhp_replicas_enum and instance_count not in smhp_replicas_enum:
            raise ValueError(
                f"Node/Instance count '{instance_count}' is not supported. "
                f"Allowed values: {sorted(smhp_replicas_enum)}."
            )
        return smhp_replicas_enum

    @abstractmethod
    def train(self, input_data_config: List[InputData], wait: bool = True, logs: bool = True, wait_timeout: Optional[int] = None):
        """Common training method that calls the specific implementation."""
        pass

    def _get_extra_smtj_hyperparameters(self) -> Dict[str, Any]:
        """Return extra hyperparameters to inject for SMTJ training.

        Subclasses can override this to add trainer-specific hyperparameters
        (e.g. RLVRTrainer adds ``reward_lambda_arn``).

        Returns:
            Dict of additional hyperparameters to merge.
        """
        return {}

    def _train_serverful_smtj(self, training_dataset=None, validation_dataset=None,
                    wait=True, wait_timeout=None, poll=5):
        """Execute training on serverful SageMaker Training Job (SMTJ) compute.

        Uses ModelTrainer.from_recipe() with the model's recipe template from
        SageMaker Hub, running on user-specified instances.

        This method is shared across SFT, DPO, and RLVR trainers. The only
        trainer-specific variation is the ``customization_technique`` (derived
        from ``self._customization_technique``) and any extra hyperparameters
        from ``_get_extra_smtj_hyperparameters()``.
        """
        import logging
        import tempfile
        from sagemaker.train.model_trainer import ModelTrainer
        from sagemaker.core.training.configs import TrainingJobCompute, InputData, Networking
        from sagemaker.core.shapes import S3DataSource
        from sagemaker.train.common_utils.finetune_utils import (
            get_recipe_s3_uri,
            get_training_image,
            _validate_hyperparameter_values,
        )
        from sagemaker.train.defaults import TrainDefaults

        logger = logging.getLogger(__name__)

        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )
        role = TrainDefaults.get_role(role=self.role, sagemaker_session=sagemaker_session)

        compute = self.compute
        customization_technique = self._customization_technique

        # Resolve the recipe S3 URI from hub metadata
        recipe_s3_uri = get_recipe_s3_uri(
            model_name=self._model_name,
            customization_technique=customization_technique,
            training_type=self.training_type,
            sagemaker_session=sagemaker_session,
        )

        logger.info(f"SMTJ recipe S3 URI: {recipe_s3_uri}")

        # Download recipe from S3 to a local temp file
        recipe_s3_uri = resolve_s3_uri_placeholders(recipe_s3_uri, sagemaker_session)

        s3_client = sagemaker_session.boto_session.client("s3")
        uri_path = recipe_s3_uri.replace("s3://", "")

        # Handle S3 access point ARN URIs
        if uri_path.startswith("arn:"):
            match = re.match(r'(arn:aws:s3:[^:]*:[^:]*:accesspoint/[^/]+)/(.*)', uri_path)
            if match:
                bucket = match.group(1)
                key = match.group(2)
            else:
                raise ValueError(f"Cannot parse S3 access point ARN: {uri_path}")
        else:
            bucket, key = uri_path.split("/", 1)

        recipe_tmp = tempfile.NamedTemporaryFile(
            prefix="smtj_recipe_", suffix=".yaml", delete=False
        )
        s3_client.download_file(bucket, key, recipe_tmp.name)
        recipe_local_path = recipe_tmp.name

        # Resolve datasets up front so their paths can be injected into the recipe
        # before rendering. The recipe maps data.train_files -> {{data_path}} and
        # data.val_files -> {{validation_data_path}}; these are mounted into the
        # container as SageMaker input channels (train/validation), so the recipe
        # must point at the local channel mount paths, not be left empty.
        resolved_training_dataset = training_dataset or self.training_dataset
        resolved_validation_dataset = validation_dataset or self.validation_dataset

        def _channel_mount_path(dataset_uri, channel_name):
            """Map an S3 dataset URI to the container path where its channel mounts."""
            mount_dir = "/opt/ml/input/data/" + channel_name
            basename = dataset_uri.rstrip("/").rsplit("/", 1)[-1]
            # A trailing object key with an extension is mounted as that file; an
            # S3 prefix (directory) is mounted as the channel directory itself.
            if "." in basename:
                return mount_dir + "/" + basename
            return mount_dir

        # Render {{placeholder}} values in the recipe template with defaults
        from sagemaker.train.common_utils.finetune_utils import (
            _render_recipe_placeholders,
            _get_smtj_override_spec,
            _get_smhp_replicas_enum,
            _resolve_base_model_weights_s3_uri,
        )
        override_spec = _get_smtj_override_spec(
            model_name=self._model_name,
            customization_technique=customization_technique,
            training_type=self.training_type,
            sagemaker_session=sagemaker_session,
        )

        # Validate instance count against allowed values from SMHP recipe.
        smhp_replicas_enum = self._validate_instance_count(compute.instance_count, sagemaker_session)
        if smhp_replicas_enum:
            override_spec.setdefault("replicas", {})["enum"] = smhp_replicas_enum
            if hasattr(self, 'hyperparameters') and hasattr(self.hyperparameters, '_specs'):
                self.hyperparameters._specs.setdefault("replicas", {})["enum"] = smhp_replicas_enum
                if not hasattr(self.hyperparameters, 'replicas'):
                    object.__setattr__(self.hyperparameters, 'replicas', compute.instance_count)

        # Inject the resolved dataset channel paths so the rendered recipe's
        # train_files / val_files are non-empty (the container aborts otherwise).
        def _set_spec_default(spec, key, value):
            entry = spec.get(key)
            if isinstance(entry, dict):
                entry["default"] = value
            else:
                spec[key] = {"default": value, "type": "string"}

        # For OSS/LLMFT models the recipe's model_name_or_path feeds straight into
        # AutoModelForCausalLM.from_pretrained(), so it must point at HF-format weights
        # on the local filesystem. When the Hub override spec leaves it empty, deliver
        # the SageMaker-prepared base weights via a dedicated "model" input channel and
        # point model_name_or_path at that channel's local mount.
        # Scoped to non-Nova: Nova recipes resolve model_name_or_path through
        # _get_args_from_nova_recipe (into the base_model hyperparameter), so this
        # OSS-specific workaround must never touch the Nova flow.
        base_model_weights_uri = getattr(self, 'model_source', None) if not _is_nova_model(self._model_name) else None
        if not _is_nova_model(self._model_name):
            model_name_or_path_spec = override_spec.get("model_name_or_path")
            if model_name_or_path_spec is not None:
                current_default = model_name_or_path_spec.get("default", "") if isinstance(model_name_or_path_spec, dict) else model_name_or_path_spec
                if not current_default and not base_model_weights_uri:
                    base_model_weights_uri = _resolve_base_model_weights_s3_uri(
                        model_name=self._model_name,
                        sagemaker_session=sagemaker_session,
                    )
                if base_model_weights_uri:
                    _set_spec_default(
                        override_spec, "model_name_or_path",
                        "/opt/ml/input/data/model",
                    )

        if resolved_training_dataset:
            _set_spec_default(
                override_spec, "data_path",
                _channel_mount_path(resolved_training_dataset, "train"),
            )
        if resolved_validation_dataset:
            _set_spec_default(
                override_spec, "validation_data_path",
                _channel_mount_path(resolved_validation_dataset, "validation"),
            )

        # Point the recipe's output/training dir at the local SageMaker model dir so the
        # trained model is written there and SageMaker uploads it to s3_output_path as
        # model.tar.gz. Without this, the recipe's {{output_path}} renders empty and the
        # container writes the model to a local cwd that never gets uploaded (job succeeds
        # but no artifact lands in S3). The llmft container uses local paths for output
        # (e.g. the metering callback writes to /opt/ml/metering), so /opt/ml/model is the
        # correct target. Scoped to non-Nova: Nova uses a managed escrow output mechanism.
        if not _is_nova_model(self._model_name) and "output_path" in override_spec:
            _set_spec_default(override_spec, "output_path", "/opt/ml/model")

        # MLflow configuration: inject tracking URI, experiment name, and run name
        # into the recipe override spec so they render into {{mlflow_*}} placeholders.
        # Uses the shared resolve helper to default empty names to base_job_name when
        # a tracking URI is set (prevents OSS container recipe validation failures).
        job_base_name = self.base_job_name or f"{self._model_name}-{customization_technique}"
        mlflow_tracking_uri, mlflow_experiment_name, mlflow_run_name = (
            resolve_mlflow_tracking_fields(
                mlflow_tracking_uri=getattr(self, 'mlflow_resource_arn', None),
                mlflow_experiment_name=getattr(self, 'mlflow_experiment_name', None),
                mlflow_run_name=getattr(self, 'mlflow_run_name', None),
                base_job_name=job_base_name,
            )
        )
        if mlflow_tracking_uri:
            _set_spec_default(override_spec, "mlflow_tracking_uri", mlflow_tracking_uri)
            _set_spec_default(override_spec, "mlflow_experiment_name", mlflow_experiment_name)
            _set_spec_default(override_spec, "mlflow_run_name", mlflow_run_name)

        # Inject user-set hyperparameters into the recipe before rendering.
        # For LLMFT/SMTJ the recipe YAML is the source of truth: ModelTrainer.from_recipe
        # ignores the hyperparameters dict for non-Nova recipes, so values the user set on
        # self.hyperparameters (e.g. global_batch_size, learning_rate, max_epochs) must be
        # rendered into the recipe's {{placeholders}} or they are silently dropped in favor
        # of the Hub spec defaults.
        def _yaml_safe_default(value):
            # Render floats in decimal form: scientific notation like "5e-06" is parsed
            # as a string by YAML, which breaks numeric recipe fields.
            if isinstance(value, float):
                s = format(value, ".12f").rstrip("0")
                return s + "0" if s.endswith(".") else s
            return value

        for hp_key in (getattr(self.hyperparameters, "_user_set", None) or []):
            if hp_key in override_spec:
                hp_value = getattr(self.hyperparameters, hp_key, None)
                if hp_value is not None:
                    _set_spec_default(override_spec, hp_key, _yaml_safe_default(hp_value))

        # Build hyperparameters early to inject into recipe template before runtime.
        final_hyperparameters = self.hyperparameters.to_dict()
        _validate_hyperparameter_values(final_hyperparameters)

        # Allow subclasses to inject extra hyperparameters
        extra_hp = self._get_extra_smtj_hyperparameters()
        if extra_hp:
            final_hyperparameters.update(extra_hp)

        # Merge user-provided recipe/overrides into hyperparameters
        final_hyperparameters = self._apply_recipe_to_hyperparameters(final_hyperparameters)

        # Inject all final hyperparameters into the override spec
        for hp_key, hp_value in final_hyperparameters.items():
            if hp_value is not None and hp_value != "":
                _set_spec_default(override_spec, hp_key, hp_value)

        with open(recipe_local_path, "r") as f:
            recipe_content = f.read()
        recipe_content = _render_recipe_placeholders(recipe_content, override_spec)

        # Inject model_source into the recipe as model_name_or_path for iterative
        # training (resuming from a previously trained checkpoint).
        # Only applies to Nova models — OSS models handle this via the input channel.
        if getattr(self, 'model_source', None) and _is_nova_model(self._model_name):
            import yaml as _yaml
            recipe_dict = _yaml.safe_load(recipe_content)

            applied = False
            if "run" in recipe_dict and isinstance(recipe_dict["run"], dict):
                recipe_dict["run"]["model_name_or_path"] = self.model_source
                applied = True

            if not applied:
                logger.warning(
                    "model checkpoint path was provided but the expected recipe path for "
                    "'model_name_or_path' was not found. The checkpoint path will not be applied."
                )
            else:
                recipe_content = _yaml.dump(recipe_dict, default_flow_style=False, sort_keys=False)
                logger.info(f"Overriding model_name_or_path with checkpoint: {self.model_source}")

        with open(recipe_local_path, "w") as f:
            f.write(recipe_content)

        logger.info(f"Recipe downloaded and rendered to: {recipe_local_path}")

        # Resolve training image
        training_image = self.training_image
        if not training_image:
            training_image = get_training_image(
                model_name=self._model_name,
                customization_technique=customization_technique,
                training_type=self.training_type,
                sagemaker_session=sagemaker_session,
            )
        if not training_image:
            raise ValueError(
                "training_image is required for SMTJ compute but could not be resolved "
                "from model metadata. Pass it explicitly via the trainer's "
                "training_image parameter."
            )

        # Build compute config for ModelTrainer
        trainer_compute = TrainingJobCompute(
            instance_type=compute.instance_type,
            instance_count=compute.instance_count,
            volume_size_in_gb=compute.volume_size_in_gb,
            keep_alive_period_in_seconds=compute.keep_alive_period_in_seconds,
        )

        # Build input data config (datasets resolved earlier for recipe injection)
        # Build input data config
        resolved_training_dataset = training_dataset or self.training_dataset
        resolved_validation_dataset = validation_dataset or self.validation_dataset

        # Use "Converse" S3DataType for Nova SFT and DPO datasets
        is_nova = _is_nova_model(self._model_name)
        use_converse = is_nova and customization_technique not in ("RLVR", "RLAIF")
        s3_data_type = "Converse" if use_converse else "S3Prefix"

        input_data_list = []
        if resolved_training_dataset:
            input_data_list.append(
                InputData(
                    channel_name="train",
                    data_source=S3DataSource(
                        s3_uri=resolved_training_dataset,
                        s3_data_type=s3_data_type,
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
            )
        if resolved_validation_dataset:
            input_data_list.append(
                InputData(
                    channel_name="validation",
                    data_source=S3DataSource(
                        s3_uri=resolved_validation_dataset,
                        s3_data_type=s3_data_type,
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
            )

        # For OSS/LLMFT models, deliver the SageMaker-prepared base model weights as a
        # "model" channel (mounted at /opt/ml/input/data/model). The recipe's
        # model_name_or_path was pointed at that mount above.
        if base_model_weights_uri:
            input_data_list.append(
                InputData(
                    channel_name="model",
                    data_source=S3DataSource(
                        s3_uri=base_model_weights_uri,
                        s3_data_type="S3Prefix",
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
            )

        # Build networking config
        networking = None
        if self.networking:
            networking = Networking(
                security_group_ids=getattr(self.networking, 'security_group_ids', None),
                subnets=getattr(self.networking, 'subnets', None),
            )

        # Create ModelTrainer from recipe
        base_job_name = self.base_job_name or f"{self._model_name}-{customization_technique}"

        # Build output data config from s3_output_path if provided
        output_data_config = None
        if self.s3_output_path:
            output_config_kwargs = {"s3_output_path": self.s3_output_path}
            if getattr(self, "disable_output_compression", False):
                output_config_kwargs["compression_type"] = "NONE"
            output_data_config = OutputDataConfig(**output_config_kwargs)

        model_trainer = ModelTrainer.from_recipe(
            training_recipe=recipe_local_path,
            compute=trainer_compute,
            networking=networking,
            stopping_condition=self.stopping_condition,
            training_image=training_image,
            input_data_config=input_data_list if input_data_list else None,
            output_data_config=output_data_config,
            hyperparameters=final_hyperparameters,
            environment=self.environment or None,
            sagemaker_session=sagemaker_session,
            role=role,
            base_job_name=base_job_name,
        )

        # Execute training
        model_trainer.train(
            wait=wait,
            logs=wait,
        )

        # Store latest training job reference
        self._latest_training_job = model_trainer._latest_training_job

        if wait:
            job_name = None
            if hasattr(self._latest_training_job, 'training_job_name'):
                job_name = self._latest_training_job.training_job_name
            elif hasattr(self._latest_training_job, 'name'):
                job_name = self._latest_training_job.name
            if job_name:
                try:
                    checkpoint_path = self._resolve_checkpoint_from_manifest(
                        job_name=job_name,
                        output_s3_path=self.s3_output_path,
                        sagemaker_session=sagemaker_session,
                    )
                    if checkpoint_path:
                        self._latest_training_job.model_artifacts = shapes.ModelArtifacts(
                            s3_model_artifacts=checkpoint_path
                        )
                        logger.info(
                            "Resolved checkpoint for %s: %s", job_name, checkpoint_path
                        )
                except Exception as e:
                    logger.warning(
                        "Could not resolve checkpoint from manifest for %s: %s",
                        job_name,
                        e,
                    )

        return self._latest_training_job

    @staticmethod
    def _resolve_checkpoint_from_manifest(
        job_name: str,
        output_s3_path: Optional[str],
        sagemaker_session=None,
    ) -> Optional[str]:
        """Resolve the model checkpoint S3 path from a training job's manifest.

        Supports both platforms:
        - **SMHP (HyperPod)**: reads ``{output_s3_path}/{job_name}/manifest.json``
          directly from S3.
        - **SMTJ (Serverful)**: downloads
          ``{output_s3_path}/{job_name}/output/output.tar.gz``, extracts
          ``manifest.json`` from the archive.

        The manifest contains a ``checkpoint_s3_bucket`` field pointing to the
        final checkpoint location on S3 (e.g. in the customer-escrow bucket).

        Args:
            job_name: The training job name.
            output_s3_path: The S3 output path configured for the training job.
            sagemaker_session: SageMaker session (used for region/boto client).

        Returns:
            The S3 URI of the checkpoint, or None if unavailable.
        """
        if not output_s3_path:
            return None

        parsed = urlparse(output_s3_path)
        bucket = parsed.netloc
        base_key = parsed.path.lstrip("/").rstrip("/")

        region = None
        if sagemaker_session and hasattr(sagemaker_session, 'boto_session'):
            region = sagemaker_session.boto_session.region_name

        s3_client = boto3.client("s3", region_name=region) if region else boto3.client("s3")

        manifest = None

        # Try SMHP format first: manifest.json directly in S3
        manifest_key = f"{base_key}/{job_name}/manifest.json"
        try:
            response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
            manifest = json.loads(response["Body"].read())
        except Exception:
            pass

        # Try SMTJ format: manifest.json inside output.tar.gz
        if manifest is None:
            tar_key = f"{base_key}/{job_name}/output/output.tar.gz"
            try:
                with tempfile.NamedTemporaryFile() as tmp_file:
                    s3_client.download_file(bucket, tar_key, tmp_file.name)
                    with tarfile.open(tmp_file.name, "r:gz") as tar:
                        manifest_file = tar.extractfile("manifest.json")
                        if manifest_file is not None:
                            manifest = json.loads(manifest_file.read())
            except Exception:
                pass

        if manifest is None:
            return None

        checkpoint_path = manifest.get("checkpoint_s3_bucket")
        if not checkpoint_path or not checkpoint_path.strip():
            return None

        # The manifest may store a relative path (SMHP convention). If it
        # doesn't start with s3://, it's relative and we cannot resolve it
        # without knowing the escrow bucket. Return as-is if it's absolute.
        if not checkpoint_path.startswith("s3://"):
            return None

        checkpoint_path = checkpoint_path.strip()

        return checkpoint_path

    def _train_hyperpod(self, training_dataset=None, validation_dataset=None,
                        wait=True, wait_timeout=None, poll=5):
        """Execute training on a SageMaker HyperPod cluster.

        Uses the HyperPod CLI to connect to the cluster and submit a training job
        using a recipe-based approach. Shared across trainers that support HyperPod
        (SFT, DPO, RLVR).
        """

        logger = logging.getLogger(__name__)

        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )

        compute = self.compute

        if not compute.cluster_name:
            raise ValueError(
                "cluster_name is required in HyperPodCompute for HyperPod training."
            )

        # HyperPod submits via the HyperPod CLI running as the *caller's* identity,
        # so there is no execution role to resolve here; this verifies the caller's
        # cluster-connect permissions (warn, non-blocking).
        TrainDefaults.verify_hyperpod_caller_permissions(
            sagemaker_session=sagemaker_session,
            cluster_name=compute.cluster_name,
        )

        # Validate HyperPod cluster capacity before proceeding
        is_nova = _is_nova_model(self._model_name)
        validate_hyperpod_compute(
            compute=compute,
            sagemaker_session=sagemaker_session,
            is_nova=is_nova,
        )

        namespace = compute.namespace or "kubeflow"

        # Connect to the HyperPod cluster
        try:
            subprocess.run(
                [
                    "hyperpod", "connect-cluster",
                    "--cluster-name", compute.cluster_name,
                    "--namespace", namespace,
                ],
                capture_output=True, text=True, check=True,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "The 'hyperpod' CLI is not installed or not on PATH. "
                "Install it with: pip install hyperpod"
            )

        # Resolve training image
        training_image = self.training_image
        if not training_image:
            smtj_image = get_training_image(
                model_name=self._model_name,
                customization_technique=self._customization_technique,
                training_type=self.training_type,
                sagemaker_session=sagemaker_session,
            )
            if smtj_image:
                training_image = smtj_image.replace("SM-TJ-", "SM-HP-")
            else:
                training_image = get_hyperpod_training_image(
                    model_name=self._model_name,
                    customization_technique=self._customization_technique,
                    training_type=self.training_type,
                    sagemaker_session=sagemaker_session,
                )

        if not training_image:
            raise ValueError(
                "training_image is required for HyperPod compute but could not be resolved "
                f"from model metadata for model '{self._model_name}' with customization "
                f"technique '{self._customization_technique}'. Pass it explicitly via the "
                "trainer's training_image parameter."
            )

        job_base_name = self.base_job_name or f"{self._model_name}-{self._customization_technique}"

        # Validate node_count against allowed values from SMHP recipe
        self._validate_instance_count(compute.node_count, sagemaker_session)

        # Resolve and validate the recipe (3-level merge: base → user recipe → overrides)
        try:
            resolved = self.get_resolved_recipe()
            additional_overrides = flatten_resolved_recipe(resolved)
        except NoRecipeError:
            additional_overrides = {}

        # Add HyperPod-specific fields not in the recipe
        additional_overrides["name"] = _get_unique_name(job_base_name)
        if compute.node_count:
            additional_overrides["replicas"] = compute.node_count

        # Data paths
        resolved_training_dataset = training_dataset or self.training_dataset
        resolved_validation_dataset = validation_dataset or self.validation_dataset
        if resolved_training_dataset:
            additional_overrides["data_s3_path"] = resolved_training_dataset
        if resolved_validation_dataset:
            additional_overrides["validation_data_s3_path"] = resolved_validation_dataset

        # Output path
        if self.s3_output_path:
            additional_overrides["output_s3_path"] = self.s3_output_path

        # MLflow configuration
        mlflow_uri, mlflow_exp, mlflow_run = resolve_mlflow_tracking_fields(
            mlflow_tracking_uri=getattr(self, 'mlflow_resource_arn', None),
            mlflow_experiment_name=getattr(self, 'mlflow_experiment_name', None),
            mlflow_run_name=getattr(self, 'mlflow_run_name', None),
            base_job_name=job_base_name,
        )
        if mlflow_uri:
            additional_overrides["mlflow_tracking_uri"] = mlflow_uri
            additional_overrides["mlflow_experiment_name"] = mlflow_exp
            additional_overrides["mlflow_run_name"] = mlflow_run

        # Render recipe with all overrides baked in and write to CLI directory
        recipe_cli_path = get_hyperpod_recipe_path(
            model_name=self._model_name,
            customization_technique=self._customization_technique,
            training_type=self.training_type,
            sagemaker_session=sagemaker_session,
            job_name=job_base_name,
            additional_overrides=additional_overrides,
        )
        logger.info(f"HyperPod recipe resolved: {recipe_cli_path}")

        # Only instance_type, container, and model_name_or_path remain as override parameters
        override_parameters = {}
        if compute.instance_type:
            override_parameters["instance_type"] = compute.instance_type
        if training_image:
            override_parameters["container"] = training_image
        if getattr(self, 'model_source', None):
            override_parameters["recipes.run.model_name_or_path"] = self.model_source

        # Submit job
        start_job_cmd = [
            "hyperpod", "start-job",
            "--namespace", namespace,
            "--recipe", recipe_cli_path,
        ]
        if override_parameters:
            start_job_cmd.extend(["--override-parameters", json.dumps(override_parameters)])

        logger.info(f"Submitting HyperPod job: {' '.join(start_job_cmd)}")

        try:
            start_result = subprocess.run(
                start_job_cmd, capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start HyperPod job: {e.stderr}")
            raise

        # Extract job name from output
        matched = re.search(r"NAME: (\S+)", start_result.stdout)
        if not matched:
            raise ValueError(
                f"Could not find job name in HyperPod CLI output: {start_result.stdout}"
            )

        job_name = matched.group(1)
        logger.info(f"HyperPod job submitted: {job_name}")

        training_job = TrainingJob(training_job_name=job_name)
        if wait:
            try:
                checkpoint_path = self._resolve_checkpoint_from_manifest(
                    job_name=job_name,
                    output_s3_path=self.s3_output_path,
                    sagemaker_session=sagemaker_session,
                )
                if checkpoint_path:
                    training_job.model_artifacts = shapes.ModelArtifacts(
                        s3_model_artifacts=checkpoint_path
                    )
            except Exception as e:
                logger.warning(
                    "Could not resolve checkpoint from manifest for %s: %s", job_name, e
                )
        self._latest_training_job = training_job

        return job_name
