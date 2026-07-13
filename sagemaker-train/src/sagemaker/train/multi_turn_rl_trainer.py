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

"""MultiTurnRLTrainer — trainer for Agentic Reinforcement Fine-Tuning (Multi-Turn RL) jobs."""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Union

import boto3

from sagemaker.ai_registry.dataset import DataSet
from sagemaker.core.resources import Job, ModelPackageGroup, ModelPackage, MlflowApp
from sagemaker.core.shapes import VpcConfig
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter, TelemetryParamType
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.custom_agent_lambda import CustomAgentLambda
from sagemaker.train.agent_rft_job import AgentRFTJob
from sagemaker.train.base_trainer import BaseTrainer
from sagemaker.train.common import CustomizationTechnique
from sagemaker.train.common_utils.finetune_utils import (
    _get_default_s3_output_path,
    _get_fine_tuning_options_and_model_arn,
    _resolve_mlflow_resource_arn,
    _resolve_model_and_name,
    _resolve_model_package_arn,
    _validate_eula_for_gated_model,
    _validate_hyperparameter_values,
    _validate_s3_path_exists,
)
from sagemaker.train.common_utils.constants import MIN_MLFLOW_VERSION
from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe, _is_nova_model
from sagemaker.train.constants import get_sagemaker_hub_name
from sagemaker.train.defaults import TrainDefaults
from sagemaker.train.utils import _get_unique_name, _get_jumpstart_tags

logger = logging.getLogger(__name__)



# ARN patterns
BEDROCK_AGENT_CORE_ARN_PATTERN = re.compile(
    r"^arn:aws[a-z-]*:bedrock-agentcore:[a-z0-9-]+:[0-9]{12}:runtime/[a-zA-Z0-9_-]+$"
)
LAMBDA_ARN_PATTERN = re.compile(
    r"^arn:aws[a-z-]*:lambda:[a-z0-9-]+:[0-9]{12}:function:[a-zA-Z0-9-_.]+"
    r"(:\$LATEST|:[a-zA-Z0-9-_]+)?$"
)
S3_URI_PATTERN = re.compile(r"^s3://[^/]+(/.*)?$")
MLFLOW_APP_ARN_PATTERN = re.compile(
    r"^arn:[a-z0-9-.]+:sagemaker:[^:]+:[^:]+:mlflow-app/.+$"
)

# Pattern for bare Bedrock AgentCore runtime IDs (not full ARNs).
AGENT_RUNTIME_ID_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_]{0,99}-[a-zA-Z0-9]{10}$")

MAX_HYPERPARAMETERS = 50
# Intentionlly hardcode this version for each PySDK version. 
# If we need upgrade the schema version, it should upgrade PySDK version as well.
JOB_CONFIG_SCHEMA_VERSION = "1.0.0" 
JOB_CATEGORY = "AgentRFT"
MTRL_TECHNIQUE = "MTRL"


def _resolve_agent_runtime_arn(agent_runtime_id: str, session=None) -> str:
    """Resolve a bare agent runtime ID to its full ARN via GetAgentRuntime.

    Args:
        agent_runtime_id: The agent runtime ID (e.g. ``"myRuntime-aBcDeFgHiJ"``).
        session: Optional boto3 session.

    Returns:
        The full Bedrock AgentCore runtime ARN.

    Raises:
        ValueError: If the runtime cannot be found or the API call fails.
    """
    try:
        client = (session or boto3.Session()).client("bedrock-agentcore-control")
        response = client.get_agent_runtime(agentRuntimeId=agent_runtime_id)
        arn = response.get("agentRuntimeArn")
        if not arn:
            raise ValueError(
                f"GetAgentRuntime returned no ARN for runtime ID '{agent_runtime_id}'."
            )
        logger.info("Resolved agent runtime ID '%s' to ARN '%s'", agent_runtime_id, arn)
        return arn
    except Exception as e:
        if "agentRuntimeArn" not in str(e):
            raise ValueError(
                f"Failed to resolve agent runtime ID '{agent_runtime_id}': {e}"
            ) from e
        raise


def _list_all_mtrl_models(session=None) -> list[str]:
    """List all models in SageMakerPublicHub that support the MTRL technique.

    Delegates to :func:`_list_hub_models_by_recipe` with
    ``recipe_type="FineTuning"`` and ``technique="MTRL"``.

    Args:
        session: Optional boto3 session.

    Returns:
        Sorted list of hub content model names supporting MTRL.
    """
    return _list_hub_models_by_recipe(
        recipe_type="FineTuning", technique=MTRL_TECHNIQUE, session=session
    )


class MultiTurnRLTrainer(BaseTrainer):
    """Trainer for Agentic Reinforcement Fine-Tuning (Multi-Turn RL) jobs.

    Uses CreateJob API (not CreateTrainingJob) with a JobConfigDocument JSON string.

    Example:

    .. code:: python

        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

        trainer = MultiTurnRLTrainer(
            model="huggingface-reasoning-qwen3-32b",
            agent_env="arn:aws:bedrock-agentcore::us-west-2:123456789012:runtime/AGENTID",
            training_dataset="s3://my-bucket/",
            output_model_package_group="arn:aws:sagemaker:us-west-2:123456789012:model-package-group/grp",
            mlflow_app_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/srv",
            s3_output_path="s3://my-bucket/output/",
            accept_eula=True,
        )
        job = trainer.train()

    Args:
        model: JumpStart model ID string or JumpStart hub content Model ARN.
        agent_env: Bedrock AgentCore ARN, agent runtime ID, Lambda ARN, or CustomAgentLambda.
            When a bare agent runtime ID is provided (e.g. ``"myRuntime-aBcDeFgHiJ"``),
            it is resolved to the full ARN via ``GetAgentRuntime``.
        training_dataset: S3 URI, DataSet object, or DataSet ARN string (optional).
            Must be provided at ``__init__`` or ``train()`` time.
        mlflow_app_arn: MLflow app ARN or MlflowApp object (optional).
            If not specified, uses the default MLflow experience.
        s3_output_path: S3 path for output artifacts (optional).
            If not specified, defaults to ``s3://sagemaker-<region>-<account>/output``.
        output_model_package_group: ModelPackageGroup object or ARN string (optional).
        intermediate_checkpoint_model_package_group: ModelPackageGroup object or ARN string
            for intermediate checkpoints (optional). If not provided, auto-creates
            ``{model_name}-mtrl-checkpoint-mpg``. Must differ from ``output_model_package_group``.
        validation_dataset: S3 URI, DataSet object, or DataSet ARN string (optional).
        bedrock_agentcore_qualifier: Bedrock AgentCore qualifier (default: ``"DEFAULT"``).
        mlflow_experiment_name: MLflow experiment name (optional).
        mlflow_run_name: MLflow run name (optional).
        networking: VpcConfig for the job (optional).
        kms_key_arn: KMS key ID for output encryption (optional).
        accept_eula: Boolean for EULA acceptance (optional).
        **kwargs: Passed to BaseTrainer (sagemaker_session, role, base_job_name, tags).
    """

    def __init__(
        self,
        model: Union[str, ModelPackage],
        agent_env: Union[str, CustomAgentLambda],
        training_dataset: Optional[Union[str, DataSet]] = None,
        mlflow_app_arn: Optional[Union[str, MlflowApp]] = None,
        s3_output_path: Optional[str] = None,
        output_model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
        intermediate_checkpoint_model_package_group: Optional[Union[str, ModelPackageGroup]] = None,
        validation_dataset: Optional[Union[str, DataSet]] = None,
        bedrock_agentcore_qualifier: str = "DEFAULT",
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        networking: Optional[VpcConfig] = None,
        kms_key_arn: Optional[str] = None,
        accept_eula: bool = False,
        recipe: Optional[str] = None,
        overrides: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._recipe_path = recipe
        self._overrides = overrides
        self._resolved_recipe_cache = None

        self._validate_agent_config(agent_env)
        self._validate_networking(networking)

        # Resolve bare agent runtime ID to full ARN
        if (
            isinstance(agent_env, str)
            and not agent_env.startswith("arn:")
            and AGENT_RUNTIME_ID_PATTERN.match(agent_env)
        ):
            agent_env = _resolve_agent_runtime_arn(agent_env)

        self.model, self._model_name = _resolve_model_and_name(model, self.sagemaker_session)
        self.agent_env = agent_env
        self.bedrock_agentcore_qualifier = bedrock_agentcore_qualifier
        self.training_dataset = training_dataset
        self.validation_dataset = validation_dataset
        self.output_model_package_group = output_model_package_group
        self.mlflow_app_arn = mlflow_app_arn
        if isinstance(mlflow_app_arn, str) and not MLFLOW_APP_ARN_PATTERN.match(mlflow_app_arn):
            raise ValueError(
                f"Invalid mlflow_app_arn: '{mlflow_app_arn}'. "
                "Must match pattern: arn:<partition>:sagemaker:<region>:<account>:mlflow-app/<name>"
            )
        self.s3_output_path = s3_output_path
        self.mlflow_experiment_name = mlflow_experiment_name
        self.mlflow_run_name = mlflow_run_name
        self.networking = networking
        self.kms_key_arn = kms_key_arn

        session = self.sagemaker_session or TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )

        # Resolve defaults for optional parameters
        if s3_output_path is None:
            self.s3_output_path = _get_default_s3_output_path(session)
            logger.info("Using default S3 output path: %s", self.s3_output_path)
        _validate_s3_path_exists(self.s3_output_path, session)

        self.output_model_package_group = self._resolve_model_package_group(
            model, output_model_package_group, session
        )
        self.intermediate_checkpoint_model_package_group = (
            self._resolve_intermediate_checkpoint_mpg(
                intermediate_checkpoint_model_package_group, session
            )
        )
        self.hyperparameters, self._model_arn, is_gated_model = (
            _get_fine_tuning_options_and_model_arn(
                self._model_name, MTRL_TECHNIQUE, "LORA", session
            )
        )
        self.accept_eula = _validate_eula_for_gated_model(model, accept_eula, is_gated_model)
        self._process_hyperparameters()
        self._latest_job: AgentRFTJob | None = None

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLTrainer.train",
        telemetry_params=[
            ("_model_name", TelemetryParamType.ATTR_VALUE),
            ("bedrock_agentcore_qualifier", TelemetryParamType.ATTR_VALUE),
            ("networking", TelemetryParamType.ATTR_EXISTS),
            ("kms_key_arn", TelemetryParamType.ATTR_EXISTS),
            ("mlflow_app_arn", TelemetryParamType.ATTR_EXISTS),
            ("agent_env", TelemetryParamType.ATTR_EXISTS),
            ("wait", TelemetryParamType.KWARG_EXISTS),
        ],
    )
    def train(
        self,
        training_dataset: Optional[Union[str, DataSet]] = None,
        wait: bool = True,
    ) -> AgentRFTJob:
        """Launch an Agentic RFT job.

        Args:
            training_dataset: Training dataset override.
            wait: If True (default), block until job reaches terminal status.

        Returns:
            AgentRFTJob instance for tracking the job.
        """
        sagemaker_session = TrainDefaults.get_sagemaker_session(
            sagemaker_session=self.sagemaker_session
        )
        role = TrainDefaults.get_role(role=self.role, sagemaker_session=sagemaker_session)

        current_job_name = _get_unique_name(
            self.base_job_name or f"{self._model_name}-mtrl"
        )
        logger.info(f"Job Name: {current_job_name}")

        self._final_hyperparameters = self.hyperparameters.to_dict()

        # Apply recipe/overrides if provided (overrides > recipe > Hub defaults).
        # Restrict to Hub-allowlisted overridable keys so the serverless
        # HyperParameters override map stays within the API's 100-entry limit
        # and the _build_training_config delta filter compares like-for-like
        # against the spec defaults snapshot (P467902218).
        self._final_hyperparameters = self._apply_recipe_to_hyperparameters(
            self._final_hyperparameters, serverless=True
        )

        _validate_hyperparameter_values(self._final_hyperparameters)

        if training_dataset is not None:
            self.training_dataset = training_dataset
        job_config_doc = self._build_job_config_document()

        tags = _get_jumpstart_tags(self._model_name, get_sagemaker_hub_name())

        try:
            job = Job.create(
                job_name=current_job_name,
                job_category=JOB_CATEGORY,
                role_arn=role,
                job_config_schema_version=JOB_CONFIG_SCHEMA_VERSION,
                job_config_document=job_config_doc,
                session=sagemaker_session.boto_session,
                region=sagemaker_session.boto_session.region_name,
            )
        except Exception as e:
            logger.error("Error: %s", e)
            raise

        agent_rft_job = AgentRFTJob.from_job(job)
        logger.info(f"Created Job: {agent_rft_job.job_arn}")

        hp = self._final_hyperparameters
        agent_rft_job.description = f"Multi-turn RFT training using {self._model_name}"

        if wait:
            from sagemaker.core.utils.exceptions import TimeoutExceededError

            try:
                agent_rft_job.wait()
            except TimeoutExceededError as e:
                logger.error("Error: %s", e)

        self._latest_job = agent_rft_job
        return agent_rft_job

    @property
    def output_model_package_arn(self) -> str | None:
        """The output model package ARN from the latest completed training job."""
        if self._latest_job is not None:
            return self._latest_job.output_model_package_arn
        return None

    @classmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION, func_name="MultiTurnRLTrainer.attach"
    )
    def attach(cls, job_name: str, session=None) -> AgentRFTJob:
        """Attach to an existing Agentic RFT job by name.

        Args:
            job_name: The name of the job.
            session: Optional boto3 session.

        Returns:
            AgentRFTJob wrapping the existing job.
        """
        return AgentRFTJob.get(job_name=job_name, session=session)

    # ---- Private: JobConfigDocument construction ----

    def _build_job_config_document(self) -> str:
        """Build the JobConfigDocument JSON string conforming to v1_0_0 schema."""
        config = {
            "AgentConfig": self._build_agent_config(),
            "InputDataConfig": self._build_input_data_config(),
            "OutputDataConfig": self._build_output_data_config(),
            "ModelPackageConfig": self._build_model_package_config(),
            "TrainingConfig": self._build_training_config(),
        }
        if self.networking:
            config["VpcConfig"] = {
                "SecurityGroupIds": self.networking.security_group_ids,
                "Subnets": self.networking.subnets,
            }
        doc = json.dumps(config, indent=2)
        logger.info(f"JobConfigDocument:\n{doc}")
        return doc

    def _build_agent_config(self) -> dict:
        agent_env = self.agent_env
        if isinstance(agent_env, CustomAgentLambda):
            return {
                "CustomAgentLambdaConfig": {"LambdaArn": agent_env.lambda_arn},
            }
        if BEDROCK_AGENT_CORE_ARN_PATTERN.match(agent_env):
            config = {"AgentRuntimeArn": agent_env}
            if self.bedrock_agentcore_qualifier:
                config["Qualifier"] = self.bedrock_agentcore_qualifier
            return {"BedrockAgentCoreConfig": config}
        if LAMBDA_ARN_PATTERN.match(agent_env):
            return {
                "CustomAgentLambdaConfig": {"LambdaArn": agent_env},
            }
        raise ValueError(f"Unrecognized agent config: {agent_env}")

    def _build_input_data_config(self) -> list:
        channels = [self._resolve_channel("train", self.training_dataset)]
        if self.validation_dataset is not None:
            channels.append(self._resolve_channel("validation", self.validation_dataset))
        return channels

    @staticmethod
    def _resolve_channel(channel_name: str, data) -> dict:
        if isinstance(data, DataSet):
            return {
                "ChannelName": channel_name,
                "DataSource": {"DatasetSource": {"DatasetArn": data.arn}},
            }
        if isinstance(data, str) and S3_URI_PATTERN.match(data):
            return {
                "ChannelName": channel_name,
                "DataSource": {
                    "S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": data}
                },
            }
        # Assume DataSet ARN string
        return {
            "ChannelName": channel_name,
            "DataSource": {"DatasetSource": {"DatasetArn": data}},
        }

    def _build_output_data_config(self) -> dict:
        config = {"S3OutputPath": self.s3_output_path}
        if self.kms_key_arn:
            config["KmsKeyArn"] = self.kms_key_arn
        return config

    def _build_model_package_config(self) -> dict:
        arn = (
            self.output_model_package_group.model_package_group_arn
            if isinstance(self.output_model_package_group, ModelPackageGroup)
            else self.output_model_package_group
        )
        config = {"OutputModelPackageGroupArn": arn}
        if isinstance(self.model, ModelPackage):
            source_arn = _resolve_model_package_arn(self.model)
            if source_arn:
                config["InputModelPackageArn"] = source_arn
        config["IntermediateCheckpointModelPackageGroupArn"] = (
            self.intermediate_checkpoint_model_package_group
        )
        return config

    def _build_training_config(self) -> dict:
        hyperparameters = getattr(self, "_final_hyperparameters", {})
        config = {
            "BaseModelArn": self._model_arn,
        }
        mlflow_config = self._build_mlflow_config()
        if mlflow_config:
            config["MlflowConfig"] = mlflow_config
        if self.accept_eula is not None:
            config["AcceptEula"] = self.accept_eula
        if hyperparameters:
            # Only send hyperparameters the user explicitly changed
            defaults = getattr(self, "_hp_defaults", {})
            user_set = {k: v for k, v in hyperparameters.items() if v != defaults.get(k)}
            if user_set:
                config["HyperParameters"] = user_set
        return config

    def _build_mlflow_config(self) -> Optional[dict]:
        arn = (
            self.mlflow_app_arn.arn
            if isinstance(self.mlflow_app_arn, MlflowApp)
            else self.mlflow_app_arn
        )
        if not arn:
            session = self.sagemaker_session or TrainDefaults.get_sagemaker_session(
                sagemaker_session=self.sagemaker_session
            )
            arn = _resolve_mlflow_resource_arn(session, None, min_mlflow_version=MIN_MLFLOW_VERSION)
            if not arn:
                return None
            logger.info("MLflow resource ARN: %s", arn)
        config = {"MlflowResourceArn": arn}
        if self.mlflow_experiment_name:
            config["MlflowExperimentName"] = self.mlflow_experiment_name
        if self.mlflow_run_name:
            config["MlflowRunName"] = self.mlflow_run_name
        return config

    def _process_hyperparameters(self):
        """Snapshot defaults for MTRL so we only send user-changed values."""
        if not self.hyperparameters or not hasattr(self.hyperparameters, "_specs"):
            return
        self._hp_defaults = self.hyperparameters.to_dict().copy()

    # ---- Validation ----

    @staticmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLTrainer.list_supported_models",
    )
    def list_supported_models(session=None) -> list[str]:
        """Return the list of supported model names.

        Queries SageMakerPublicHub to discover all models with MTRL
        recipes in their ``RecipeCollection``.

        Args:
            session: Optional boto3 session.

        Returns:
            List of hub content model names supporting MTRL.
        """
        return _list_all_mtrl_models(session=session)

    @staticmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLTrainer.list_bedrock_agentcore_runtimes",
    )
    def list_bedrock_agentcore_runtimes(session=None) -> list[dict]:
        """List Bedrock AgentCore runtimes.

        Args:
            session: Optional boto3 session.

        Returns:
            List of dicts, each with keys ``name``, ``runtime_id``, ``arn``,
            and ``status``.
        """
        client = (session or boto3.Session()).client("bedrock-agentcore-control")
        runtimes: list[dict] = []
        next_token = None

        while True:
            kwargs: dict = {}
            if next_token:
                kwargs["nextToken"] = next_token
            response = client.list_agent_runtimes(**kwargs)

            for rt in response.get("agentRuntimes", []):
                entry = {
                    "name": rt.get("agentRuntimeName", ""),
                    "runtime_id": rt.get("agentRuntimeId", ""),
                    "arn": rt["agentRuntimeArn"],
                    "status": rt.get("status", ""),
                }
                runtimes.append(entry)

            next_token = response.get("nextToken")
            if not next_token:
                break

        return runtimes

    @staticmethod
    def _validate_agent_config(agent_env):
        if isinstance(agent_env, CustomAgentLambda):
            return
        if not isinstance(agent_env, str):
            raise ValueError(
                f"agent_env must be a string ARN, agent runtime ID, or CustomAgentLambda, "
                f"got {type(agent_env).__name__}."
            )
        if not (
            BEDROCK_AGENT_CORE_ARN_PATTERN.match(agent_env)
            or LAMBDA_ARN_PATTERN.match(agent_env)
            or AGENT_RUNTIME_ID_PATTERN.match(agent_env)
        ):
            raise ValueError(
                f"Invalid agent_env: '{agent_env}'. "
                "Must be a Bedrock AgentCore ARN, Lambda ARN, agent runtime ID, "
                "or CustomAgentLambda."
            )

    @staticmethod
    def _validate_networking(vpc):
        if vpc is None:
            return
        sg = getattr(vpc, "security_group_ids", None)
        subnets = getattr(vpc, "subnets", None)
        if not sg or not subnets:
            raise ValueError(
                "VPC config requires both non-empty 'security_group_ids' and 'subnets'."
            )

    def _get_or_create_mpg(self, value, default_name: str, session, managed_configuration=None) -> str:
        """Resolve an existing ModelPackageGroup or auto-create one.

        If ``value`` is provided (object or string), validates it exists and returns its ARN.
        If ``value`` is None, creates a ModelPackageGroup with ``default_name`` (get-or-create).

        Returns:
            The ModelPackageGroup ARN.
        """
        if value:
            if isinstance(value, ModelPackageGroup):
                return value.model_package_group_arn
            mpg = ModelPackageGroup.get(
                model_package_group_name=value,
                session=session.boto_session,
                region=session.boto_session.region_name,
            )
            return mpg.model_package_group_arn

        # Auto-create (get-or-create with deterministic name)
        logger.info("Auto-resolving ModelPackageGroup: %s", default_name)
        try:
            mpg = ModelPackageGroup.get(
                model_package_group_name=default_name,
                session=session.boto_session,
                region=session.boto_session.region_name,
            )
        except Exception:
            try:
                create_kwargs = {
                    "model_package_group_name": default_name,
                    "session": session.boto_session,
                    "region": session.boto_session.region_name,
                }
                if managed_configuration:
                    create_kwargs["managed_configuration"] = managed_configuration
                mpg = ModelPackageGroup.create(**create_kwargs)
                logger.info("Created ModelPackageGroup: %s", mpg.model_package_group_arn)
            except Exception as e:
                raise ValueError(
                    f"Failed to create ModelPackageGroup '{default_name}': {e}"
                ) from e
        return mpg.model_package_group_arn

    def _resolve_model_package_group(self, model, output_model_package_group, session):
        """Resolve, validate, or auto-create the output ModelPackageGroup.

        Resolution order:
        1. If ``output_model_package_group`` is provided, validates it exists.
        2. If ``model`` is a ModelPackage, derives the group from it.
        3. Otherwise, auto-creates ``{model_name}-mtrl-mpg`` (get-or-create).

        Returns:
            The ModelPackageGroup ARN.
        """
        if output_model_package_group:
            return self._get_or_create_mpg(output_model_package_group, None, session)

        # Derive from ModelPackage
        if isinstance(model, ModelPackage):
            group_name = model.model_package_group_name
            if group_name:
                return self._get_or_create_mpg(group_name, None, session)

        managed_config = None
        if _is_nova_model(self._model_name):
            from sagemaker.core.shapes import ManagedConfiguration
            managed_config = ManagedConfiguration(managed_storage_type="Restricted")

        return self._get_or_create_mpg(
            None, f"{self._model_name}-mtrl-mpg", session, managed_configuration=managed_config
        )

    def _resolve_intermediate_checkpoint_mpg(self, intermediate_checkpoint_mpg, session) -> str:
        """Resolve or auto-create the intermediate checkpoint ModelPackageGroup.

        If provided, validates it exists. Otherwise auto-creates
        ``{model_name}-mtrl-checkpoint-mpg`` (get-or-create).
        Raises ValueError if the resolved ARN is the same as ``output_model_package_group``.

        Returns:
            The ModelPackageGroup ARN.
        """
        managed_config = None
        if not intermediate_checkpoint_mpg and _is_nova_model(self._model_name):
            from sagemaker.core.shapes import ManagedConfiguration
            managed_config = ManagedConfiguration(managed_storage_type="Restricted")

        arn = self._get_or_create_mpg(
            intermediate_checkpoint_mpg,
            f"{self._model_name}-mtrl-checkpoint-mpg",
            session,
            managed_configuration=managed_config,
        )
        if arn == self.output_model_package_group:
            raise ValueError(
                "intermediate_checkpoint_model_package_group must differ from "
                "output_model_package_group."
            )
        return arn
