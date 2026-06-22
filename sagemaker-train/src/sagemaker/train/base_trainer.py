import copy
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
import json
import logging
import re
import subprocess
import tarfile
import tempfile
from urllib.parse import urlparse

import boto3

from sagemaker.core.helper.session_helper import Session
from sagemaker.core.training.configs import Tag, Networking, InputData, Channel, OutputDataConfig
from sagemaker.core.shapes import shapes
from sagemaker.core.resources import TrainingJob
from sagemaker.train.common_utils.recipe_utils import _is_nova_model, resolve_recipe
from sagemaker.train.recipe_resolver import flatten_resolved_recipe
from sagemaker.train.common_utils.finetune_utils import (
    get_training_image,
    get_hyperpod_training_image,
    get_hyperpod_recipe_path,
    _validate_hyperparameter_values,
)
from sagemaker.train.common_utils.validator import validate_hyperpod_compute
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name

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
        self._checkpoint_s3_uri = None

    def _is_nova_model_for_telemetry(self) -> bool:
        """Check if the model is a Nova model for telemetry tracking."""
        model_name = getattr(self, "_model_name", None)
        return _is_nova_model(model_name) if model_name else False

    def get_resolved_recipe(self) -> Dict[str, Any]:
        """Return the fully resolved recipe configuration.

        Shows the final merged result of base defaults + user recipe + overrides
        after interpolation resolution and validation. Callable before or after train().

        Returns:
            dict: Deep copy of the resolved recipe configuration.

        Raises:
            ValueError: If no recipe or overrides were provided at construction time.
        """
        if getattr(self, '_resolved_recipe_cache', None) is not None:
            return copy.deepcopy(self._resolved_recipe_cache)

        recipe_path = getattr(self, '_recipe_path', None)
        overrides = getattr(self, '_overrides', None)

        if not recipe_path and not overrides:
            raise ValueError(
                "get_resolved_recipe() requires a 'recipe' or 'overrides' to be provided "
                "at construction time."
            )

        override_spec = {}
        full_recipe_template = None
        if hasattr(self, 'hyperparameters') and hasattr(self.hyperparameters, '_specs'):
            override_spec = self.hyperparameters._specs
        frt = getattr(self.hyperparameters, '_full_recipe_template', None) if hasattr(self, 'hyperparameters') else None
        if isinstance(frt, dict):
            full_recipe_template = frt

        resolved = resolve_recipe(
            recipe_path=recipe_path,
            overrides=overrides,
            override_spec=override_spec,
            template_section="training_config",
            # ``dataset_catalog`` is part of the curated recipe fetched from Hub at
            # runtime (not a hardcoded local key); protecting it prevents user
            # recipe/overrides from tampering with the curated dataset mix
            # (AppSec finding #7 — data-mixing). The resolver locates the key by
            # name in the runtime full recipe template via _build_key_path_map.
            protected_keys={"model_type", "model_name_or_path", "dataset_catalog"},
            full_recipe_template=full_recipe_template,
            compute=getattr(self, 'compute', None),
        )

        self._resolved_recipe_cache = resolved
        return copy.deepcopy(resolved)

    def _apply_recipe_to_hyperparameters(self, final_hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Apply resolved recipe values to final_hyperparameters dict.

        If recipe/overrides were provided, merges resolved values into the
        hyperparameters dict. All leaf values from the resolved recipe are
        applied — including keys not in the Hub spec subset — enabling
        power users to override any parameter in the full recipe.
        Values are converted to strings (matching the SageMaker API
        expectation for hyperparameter values).

        Args:
            final_hyperparameters: The hyperparameters dict from to_dict().

        Returns:
            The updated hyperparameters dict with recipe values applied.
        """
        if not getattr(self, '_recipe_path', None) and not getattr(self, '_overrides', None):
            return final_hyperparameters

        resolved = self.get_resolved_recipe()
        flat = flatten_resolved_recipe(resolved)
        for k, v in flat.items():
            if v is not None:
                final_hyperparameters[k] = str(v) if not isinstance(v, str) else v

        return final_hyperparameters

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
        s3_client = sagemaker_session.boto_session.client("s3")
        uri_path = recipe_s3_uri.replace("s3://", "")
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
        )
        override_spec = _get_smtj_override_spec(
            model_name=self._model_name,
            customization_technique=customization_technique,
            training_type=self.training_type,
            sagemaker_session=sagemaker_session,
        )

        # Inject the resolved dataset channel paths so the rendered recipe's
        # train_files / val_files are non-empty (the container aborts otherwise).
        def _set_spec_default(spec, key, value):
            entry = spec.get(key)
            if isinstance(entry, dict):
                entry["default"] = value
            else:
                spec[key] = {"default": value, "type": "string"}

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

        with open(recipe_local_path, "r") as f:
            recipe_content = f.read()
        recipe_content = _render_recipe_placeholders(recipe_content, override_spec)
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

        # Build hyperparameters dict for the recipe
        final_hyperparameters = self.hyperparameters.to_dict()
        _validate_hyperparameter_values(final_hyperparameters)

        # Allow subclasses to inject extra hyperparameters
        extra_hp = self._get_extra_smtj_hyperparameters()
        if extra_hp:
            final_hyperparameters.update(extra_hp)

        # Merge user-provided recipe/overrides into hyperparameters
        final_hyperparameters = self._apply_recipe_to_hyperparameters(final_hyperparameters)

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
            output_data_config = OutputDataConfig(s3_output_path=self.s3_output_path)

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
                self._try_resolve_checkpoint(job_name, sagemaker_session)

        return self._latest_training_job

    def _try_resolve_checkpoint(self, job_name: str, sagemaker_session=None):
        """Attempt to resolve the checkpoint path from the job manifest.

        Sets ``_checkpoint_s3_uri`` on success; logs a warning on failure.
        """
        logger = logging.getLogger(__name__)
        try:
            checkpoint_path = self._resolve_checkpoint_from_manifest(
                job_name=job_name,
                output_s3_path=self.s3_output_path,
                sagemaker_session=sagemaker_session,
            )
            if checkpoint_path:
                self._checkpoint_s3_uri = checkpoint_path
                logger.info("Resolved checkpoint for %s: %s", job_name, checkpoint_path)
        except Exception as e:
            logger.warning(
                "Could not resolve checkpoint from manifest for %s: %s", job_name, e
            )

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

        # Resolve recipe: use user-provided recipe or auto-resolve from Hub
        recipe_name = getattr(self, '_recipe_path', None)
        if not recipe_name:
            job_base_name = self.base_job_name or f"{self._model_name}-{self._customization_technique}"
            recipe_name = get_hyperpod_recipe_path(
                model_name=self._model_name,
                customization_technique=self._customization_technique,
                training_type=self.training_type,
                sagemaker_session=sagemaker_session,
                job_name=job_base_name,
            )
            logger.info(f"Auto-resolved HyperPod recipe from Hub: {recipe_name}")

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

        # Build override parameters
        override_parameters = {}
        if compute.instance_type:
            override_parameters["instance_type"] = compute.instance_type
        if training_image:
            override_parameters["container"] = training_image
        if compute.node_count:
            override_parameters["recipes.run.replicas"] = compute.node_count

        job_base_name = self.base_job_name or f"{self._model_name}-{self._customization_technique}"
        override_parameters["recipes.run.name"] = _get_unique_name(job_base_name)

        # Data paths
        resolved_training_dataset = training_dataset or self.training_dataset
        resolved_validation_dataset = validation_dataset or self.validation_dataset
        if resolved_training_dataset:
            override_parameters["recipes.run.data_s3_path"] = resolved_training_dataset
        if resolved_validation_dataset:
            override_parameters["recipes.run.validation_data_s3_path"] = resolved_validation_dataset

        # Output path
        if self.s3_output_path:
            override_parameters["recipes.run.output_s3_path"] = self.s3_output_path

        # MLflow configuration
        if getattr(self, 'mlflow_resource_arn', None):
            override_parameters["recipes.run.mlflow_tracking_uri"] = self.mlflow_resource_arn
        if getattr(self, 'mlflow_experiment_name', None):
            override_parameters["recipes.run.mlflow_experiment_name"] = self.mlflow_experiment_name
        if getattr(self, 'mlflow_run_name', None):
            override_parameters["recipes.run.mlflow_run_name"] = self.mlflow_run_name

        # Hyperparameters — only pass user-explicitly-set values for HyperPod
        if self.hyperparameters:
            for hp_key in getattr(self.hyperparameters, '_user_set', []):
                hp_value = getattr(self.hyperparameters, hp_key, None)
                if hp_value is not None:
                    override_parameters[f"recipes.training_config.{hp_key}"] = hp_value

        # Apply user-provided overrides (supports nested recipe paths)
        if getattr(self, '_overrides', None):
            for key, value in self._overrides.items():
                override_parameters[key] = value

        # Submit job
        start_job_cmd = [
            "hyperpod", "start-job",
            "--namespace", namespace,
            "--recipe", recipe_name,
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

        self._latest_training_job = job_name

        if wait:
            self._try_resolve_checkpoint(job_name, sagemaker_session)

        return job_name
