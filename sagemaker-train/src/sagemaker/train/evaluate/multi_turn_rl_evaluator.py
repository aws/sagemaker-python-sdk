"""MultiTurnRLEvaluator — evaluate MTRL agents on held-out prompts.

This module implements :class:`MultiTurnRLEvaluator`, the SDK surface for
evaluating Multi-Turn Reinforcement Learning (MTRL) agent models via the
AgentRFT ``CreateJob`` pipeline step. Mirrors the architecture of
:class:`sagemaker.train.evaluate.BenchMarkEvaluator`, with MTRL-specific
fields, validators, and the three-template rendering surface defined in
:mod:`sagemaker.train.evaluate.mtrl_pipeline_templates`.
"""

from __future__ import absolute_import

import logging
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import Field, root_validator, validator

from .base_evaluator import BaseEvaluator
from .constants import EvalType
from .mtrl_pipeline_templates import (
    MTRL_TEMPLATE,
    MTRL_TEMPLATE_BASE_MODEL_ONLY,
    MTRL_TEMPLATE_FINE_TUNED_ONLY,
)
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter, TelemetryParamType
from sagemaker.train.common_utils.telemetry_params import BASE_EVALUATOR_TELEMETRY_PARAMS
from sagemaker.core.telemetry.constants import Feature

if TYPE_CHECKING:
    from .execution import MTRLEvaluationExecution

_logger = logging.getLogger(__name__)

# Validation patterns.
_BEDROCK_AGENTCORE_ARN_RE = re.compile(
    r"^arn:aws[a-z\-]*:bedrock-agentcore:[a-z0-9\-]+:[0-9]{12}:(?:agent-runtime|runtime)/.+$"
)
_LAMBDA_ARN_RE = re.compile(
    r"^arn:aws[a-z\-]*:lambda:[a-z0-9\-]+:[0-9]{12}:function:.+$"
)

# Stopping-condition bounds (seconds): 0 < v <= 72 hours.
_MAX_STOPPING_CONDITION_SECONDS = 72 * 60 * 60


class MultiTurnRLEvaluator(BaseEvaluator):
    """Evaluate a multi-turn RL agent model against a held-out prompt dataset.

    The evaluator runs rollouts of the agent against an environment
    (Bedrock AgentCore runtime or a Lambda-wrapped agent) and computes
    aggregate metrics (pass@k, mean reward, etc.). Execution routes through
    SageMaker Pipelines using the new AgentRFT ``Job`` step type
    (``JobCategory="AgentRFTEvaluation"``).

    The evaluator supports three evaluation shapes, selected automatically
    based on the provided inputs:

    * **Base-model only** — pass a base model (JumpStart ID or ModelPackage)
      with an explicit ``agent_config``.
    * **Fine-tuned only** — pass a ``MultiTurnRLTrainer`` or a
      fine-tuned ``ModelPackage``; the evaluator extracts the source model
      package ARN and evaluates it only.
    * **Base + fine-tuned comparison** — pass ``evaluate_base_model=True``
      along with a fine-tuned trainer / ModelPackage; both runs land in
      the same MLflow experiment for side-by-side comparison.

    Attributes:
        dataset (Union[str, Any]): Prompt dataset — S3 URI, hub-content
            DataSet ARN, or object exposing an ``.arn`` attribute.
            Required.
        agent_config (Optional[Union[str, Any]]): Agent environment —
            Bedrock AgentCore ARN or Lambda ARN. Auto-resolved from a
            ``MultiTurnRLTrainer`` when provided as ``model``.
        agent_qualifier (Optional[str]): Bedrock AgentCore qualifier
            (e.g. ``"PROD"``). Ignored when ``agent_config`` is a Lambda.
        accept_eula (bool): Forwarded to
            ``JobConfigDocument.EvaluationConfig.AcceptEula``. Defaults
            to ``True`` (templates emit ``true`` unconditionally; flag
            kept for future backend schemas).
        evaluate_base_model (bool): When ``True`` and a fine-tuned model is
            present, render the comparison template (both base and
            fine-tuned are evaluated). Defaults to ``False`` — fine-tuned
            only.
        stopping_condition (int): Maximum job duration in seconds. Default
            ``86400`` (24 hours); must be in ``(0, 259200]``.
        tags (Optional[List[Dict[str, str]]]): Customer tags propagated to
            the pipeline + step ``Tags`` list.

        See :class:`BaseEvaluator` for inherited fields (``model``,
        ``s3_output_path``, ``mlflow_resource_arn``,
        ``mlflow_experiment_name``, ``networking``, ``kms_key_id``,
        ``model_package_group``, ``base_eval_name``, ``region``, ``role``,
        ``sagemaker_session``).

    Example:

        .. code:: python

            from sagemaker.train.evaluate import MultiTurnRLEvaluator

            # Evaluate a fine-tuned MTRL trainer output
            evaluator = MultiTurnRLEvaluator(
                model=completed_mtrl_trainer,
                dataset='s3://my-bucket/eval-prompts.jsonl',
                s3_output_path='s3://my-bucket/mtrl-eval-output/',
            )

            execution = evaluator.evaluate()
            execution.wait()
            execution.show_results()
    """

    # --- Declared fields -------------------------------------------------
    dataset: Any = Field(..., description="Prompt dataset (S3 URI, ARN, or object with .arn).")
    agent_config: Optional[Any] = Field(default=None, description="Agent environment.")
    agent_qualifier: Optional[str] = Field(default=None, description="Bedrock AgentCore qualifier.")
    accept_eula: bool = Field(default=True, description="Accept EULA for the base model.")
    evaluate_base_model: bool = Field(
        default=False,
        description="When True, render the base + fine-tuned comparison template.",
    )
    stopping_condition: int = Field(
        default=86400,
        description="Maximum job duration in seconds; must be in (0, 259200].",
    )
    tags: Optional[List[Dict[str, str]]] = Field(default=None, description="Customer tags.")

    # Private instance state (populated during resolution).
    _base_model_arn_cache: Optional[str] = None
    _base_model_name_cache: Optional[str] = None
    _source_model_package_arn_cache: Optional[str] = None
    _agent_arn_resolved: Optional[str] = None
    _agent_kind: Optional[str] = None  # "bedrock" | "lambda"
    _hyperparameters: Optional[Any] = None


    # --- Validators ------------------------------------------------------

    @validator("dataset", pre=True, always=True)
    def _resolve_dataset(cls, v):
        if v is None:
            raise ValueError(
                "[PySDK Error] 'dataset' is required. Accepted: S3 URI "
                "(s3://...), hub-content DataSet ARN, or an object with an "
                "`.arn` attribute."
            )
        return BaseEvaluator._validate_and_resolve_dataset(v)

    @validator("agent_config", pre=True, always=True)
    def _resolve_agent_config(cls, v):
        if v is None:
            return None
        # AgentLambdaAdapter-like object with .materialize() → ARN is deferred
        # to evaluate() time; here we only accept pre-resolved strings.
        if not isinstance(v, str):
            # Pass through non-string objects; evaluate() will materialize.
            return v
        if _BEDROCK_AGENTCORE_ARN_RE.match(v) or _LAMBDA_ARN_RE.match(v):
            return v
        raise ValueError(
            f"[PySDK Error] 'agent_config' value '{v}' is not a recognized "
            f"Bedrock AgentCore ARN or Lambda ARN."
        )

    @validator("stopping_condition", always=True)
    def _validate_stopping_condition(cls, v):
        if v is None:
            return 86400
        if v <= 0:
            raise ValueError(
                f"[PySDK Error] 'stopping_condition' must be > 0; got {v}."
            )
        if v > _MAX_STOPPING_CONDITION_SECONDS:
            raise ValueError(
                f"[PySDK Error] 'stopping_condition' must be <= "
                f"{_MAX_STOPPING_CONDITION_SECONDS} seconds (72 hours); "
                f"got {v}."
            )
        return v

    @root_validator(skip_on_failure=True)
    def _check_agent_config_for_non_trainer_models(cls, values):
        """When the model is not a ``MultiTurnRLTrainer``, require ``agent_config``.

        When the customer passes a trainer instance, the evaluator
        auto-resolves the agent config from the trainer's stored
        configuration. For any other model type (string JumpStart ID,
        ``ModelPackage`` object, ModelPackage ARN string) the customer
        must supply ``agent_config`` explicitly.
        """
        model = values.get("model")
        agent_config = values.get("agent_config")
        if agent_config is not None:
            return values
        # Avoid a hard import cycle on MultiTurnRLTrainer; check by class name.
        model_cls_name = type(model).__name__ if model is not None else ""
        if model_cls_name not in ("MultiTurnRLTrainer", "AgentRFTJob"):
            raise ValueError(
                "[PySDK Error] 'agent_config' is required when 'model' is "
                "not a MultiTurnRLTrainer. Provide a Bedrock AgentCore ARN "
                "or a Lambda ARN."
            )
        return values


    # --- Trainer / model resolution -------------------------------------

    def _resolve_trainer_defaults(self) -> None:
        """Pull base/source ARNs and agent defaults from a MultiTurnRLTrainer.

        Idempotent; re-reading a trainer yields the same values. Customer-
        provided ``agent_config`` / ``agent_qualifier`` always win over
        trainer-sourced values.
        """
        if type(self.model).__name__ not in ("MultiTurnRLTrainer", "AgentRFTJob"):
            return

        trainer = self.model
        # Resolve the output model package ARN from the completed job.
        # MultiTurnRLTrainer stores the job in _latest_job (AgentRFTJob),
        # which exposes output_model_package_arn as a property.
        source_mp = (
            getattr(trainer, "output_model_package_arn", None)
            or getattr(trainer, "model_package_arn", None)
        )
        if not source_mp and hasattr(trainer, "_latest_job") and trainer._latest_job is not None:
            source_mp = getattr(trainer._latest_job, "output_model_package_arn", None)

        if not source_mp:
            raise ValueError(
                "[PySDK Error] The provided MultiTurnRLTrainer has no "
                "completed training job (output model package ARN is "
                "unavailable). Run trainer.wait() and retry."
            )

        self._source_model_package_arn_cache = source_mp
        self._base_model_arn_cache = (
            getattr(trainer, "base_model_arn", None)
            or getattr(trainer, "_base_model_arn", None)
            or getattr(trainer, "_model_arn", None)
        )
        self._base_model_name_cache = (
            getattr(trainer, "base_model_name", None)
            or getattr(trainer, "_base_model_name", None)
            or getattr(trainer, "_model_name", None)
        )

        # Customer values win.
        if self.agent_config is None:
            resolved_agent = (
                getattr(trainer, "agent_config", None)
                or getattr(trainer, "_agent_config", None)
                or getattr(trainer, "agent_env", None)
            )
            # AgentRFTJob.agent_config returns a dict like {"AgentRuntimeArn": "..."}
            if isinstance(resolved_agent, dict):
                self.agent_config = (
                    resolved_agent.get("AgentRuntimeArn")
                    or resolved_agent.get("AgentLambdaArn")
                    or resolved_agent.get("LambdaArn")
                )
            else:
                self.agent_config = resolved_agent
        if self.agent_qualifier is None:
            self.agent_qualifier = (
                getattr(trainer, "agent_qualifier", None)
                or getattr(trainer, "_agent_qualifier", None)
                or getattr(trainer, "bedrock_agentcore_qualifier", None)
            )

    # --- Hyperparameters property ---------------------------------------

    @property
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLEvaluator.hyperparameters",
    )
    def hyperparameters(self):
        """Lazy-load evaluation hyperparameters from the JumpStart hub.

        Returns a ``FineTuningOptions`` object exposing ``to_dict()``,
        ``get_info()``, and attribute-style read/write access with
        hub-sourced validation (type + range).

        Supported parameters (sourced from the AgentRFT evaluation recipe):
        ``eval_group_size``, ``sampling_temperature``, ``top_p``,
        ``max_tokens``, ``pass_k_values``, ``success_threshold``.

        Raises:
            ValueError: If the base model name is not available or the hub
                does not expose an AgentRFTEvaluation override spec for
                the model.
        """
        if self._hyperparameters is not None:
            return self._hyperparameters

        from ..common import FineTuningOptions
        from ..common_utils.recipe_utils import (
            _extract_eval_override_options,
            _get_evaluation_override_params,
        )

        hub_content_name = self._base_model_name_cache or self._base_model_name
        if not hub_content_name:
            raise ValueError(
                "[PySDK Error] Cannot load MTRL hyperparameters: base "
                "model name not available. Ensure `model` resolves to a "
                "JumpStart / hub-backed base model."
            )

        boto_session = (
            self.sagemaker_session.boto_session
            if hasattr(self.sagemaker_session, "boto_session")
            else self.sagemaker_session
        )

        override_params = _get_evaluation_override_params(
            hub_content_name=hub_content_name,
            hub_name="SageMakerPublicHub",
            evaluation_type="MTRLEvaluation",
            region=self.region,
            session=boto_session,
        )
        if not override_params:
            raise ValueError(
                f"[PySDK Error] Base model '{hub_content_name}' does not "
                f"expose AgentRFTEvaluation hyperparameter overrides in the "
                f"JumpStart hub."
            )

        spec = _extract_eval_override_options(
            override_params, param_names=list(override_params.keys()), return_full_spec=True
        )
        self._hyperparameters = FineTuningOptions(spec)
        return self._hyperparameters


    # --- Helpers ---------------------------------------------------------

    def _resolve_agent_arn(self) -> None:
        """Resolve ``agent_config`` to a concrete ARN string + kind.

        * String ARN: classify as ``bedrock`` or ``lambda`` by regex.
        * ``AgentLambdaAdapter``-like object: call ``.materialize()`` which
          returns a Lambda ARN string. Gated — returns a clear error if no
          ``.materialize()`` is available.
        """
        if self.agent_config is None:
            self._agent_arn_resolved = None
            self._agent_kind = None
            return

        if isinstance(self.agent_config, str):
            arn = self.agent_config
        else:
            materialize = getattr(self.agent_config, "materialize", None)
            if callable(materialize):
                arn = materialize()
            else:
                arn = (
                    getattr(self.agent_config, "lambda_arn", None)
                    or getattr(self.agent_config, "arn", None)
                )
            if not isinstance(arn, str):
                raise ValueError(
                    "[PySDK Error] Could not resolve agent_config to an ARN. "
                    "Pass a Bedrock AgentCore ARN string, a Lambda ARN "
                    "string, or an object exposing `.lambda_arn` or `.materialize()`."
                )

        if _BEDROCK_AGENTCORE_ARN_RE.match(arn):
            self._agent_kind = "bedrock"
        elif _LAMBDA_ARN_RE.match(arn):
            self._agent_kind = "lambda"
        else:
            raise ValueError(
                f"[PySDK Error] Resolved agent ARN '{arn}' is neither a "
                f"Bedrock AgentCore ARN nor a Lambda ARN."
            )
        self._agent_arn_resolved = arn

    def _select_mtrl_template(self) -> str:
        """Pick the right template based on fine-tuned vs base vs comparison."""
        has_ft = bool(self._source_model_package_arn_cache)
        if not has_ft:
            return MTRL_TEMPLATE_BASE_MODEL_ONLY
        if has_ft and self.evaluate_base_model:
            return MTRL_TEMPLATE
        return MTRL_TEMPLATE_FINE_TUNED_ONLY

    def _build_template_context(
        self,
        aws_context: Dict[str, str],
        artifacts: Dict[str, str],
        model_package_group_arn: Optional[str],
    ) -> Dict[str, Any]:
        """Assemble the rendering context expected by the MTRL templates."""
        import json as _json

        hparams: Dict[str, Any] = {}
        try:
            hparams = self._get_effective_hyperparameters()
        except Exception as e:  # hub fetch can fail in offline envs
            _logger.info(f"Skipping hub-sourced hyperparameters: {e}")

        def _str_or_none(v):
            return str(v) if v is not None else None

        action_arn_prefix = (
            f"arn:aws:sagemaker:{aws_context['region']}:{aws_context['account_id']}:action"
        )

        networking = getattr(self, "networking", None)
        vpc_security_group_ids: List[str] = []
        vpc_subnets: List[str] = []
        if networking is not None:
            vpc_security_group_ids = list(getattr(networking, "security_group_ids", []) or [])
            vpc_subnets = list(getattr(networking, "subnets", []) or [])

        base_model_arn = (
            self._base_model_arn_cache
            or self._base_model_arn
            or artifacts.get("base_model_arn")
        )

        # --- Build JobConfigDocument as a dict, then json.dumps() it ----
        # The Pipelines service expects JobConfigDocument as a JSON string
        # (double-encoded), not a nested object.
        def _build_job_config_doc(include_mpc: bool, mlflow_run_name: str) -> str:
            agent_arn = self._agent_arn_resolved

            # AgentConfig
            if self._agent_kind == "bedrock":
                agent_cfg = {"BedrockAgentCoreConfig": {"AgentRuntimeArn": agent_arn}}
                if self.agent_qualifier:
                    agent_cfg["BedrockAgentCoreConfig"]["Qualifier"] = self.agent_qualifier
            else:
                agent_cfg = {"CustomAgentLambdaConfig": {"LambdaArn": agent_arn}}

            # InputDataConfig
            ds = self.dataset
            if ds.startswith("arn:") and "hub-content" in ds and "/DataSet/" in ds:
                data_source = {"DatasetSource": {"DatasetArn": ds}}
            else:
                data_source = {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": ds}}
            input_data = [{"ChannelName": "evaluation", "DataSource": data_source}]

            # OutputDataConfig
            output_data: Dict[str, Any] = {"S3OutputPath": self.s3_output_path}
            if getattr(self, "kms_key_id", None):
                output_data["KmsKeyArn"] = self.kms_key_id
            if self.mlflow_resource_arn:
                mlflow_cfg: Dict[str, Any] = {"MlflowResourceArn": self.mlflow_resource_arn}
                exp_name = (
                    getattr(self, "mlflow_experiment_name", None)
                    or f"mtrl-eval-{self._base_model_name_cache or 'default'}"
                )
                mlflow_cfg["MlflowExperimentName"] = exp_name
                mlflow_cfg["MlflowRunName"] = mlflow_run_name
                output_data["MlflowConfig"] = mlflow_cfg

            eval_cfg: Dict[str, Any] = {
                "BaseModelArn": base_model_arn,
                "AcceptEula": True,
            }
            hp: Dict[str, str] = {}
            for k in ("eval_group_size", "sampling_temperature", "top_p",
                       "max_tokens", "pass_k_values", "success_threshold"):
                v = hparams.get(k)
                if v is not None:
                    hp[k] = str(v)
            if hp:
                eval_cfg["HyperParameters"] = hp

            doc: Dict[str, Any] = {
                "AgentConfig": agent_cfg,
                "InputDataConfig": input_data,
                "OutputDataConfig": output_data,
                "EvaluationConfig": eval_cfg,
            }

            # ModelPackageConfig (fine-tuned only)
            if include_mpc and self._source_model_package_arn_cache:
                mpc: Dict[str, str] = {
                    "InputModelPackageArn": self._source_model_package_arn_cache,
                }
                doc["ModelPackageConfig"] = mpc

            # StoppingCondition
            doc["StoppingCondition"] = {
                "MaxRuntimeInSeconds": self.stopping_condition,
            }

            return _json.dumps(doc)

        # Build both variants (base-only and fine-tuned).
        job_config_doc_str = _build_job_config_doc(include_mpc=False, mlflow_run_name="base-model-eval")
        job_config_doc_ft_str = _build_job_config_doc(include_mpc=True, mlflow_run_name="fine-tuned-model-eval")

        return {
            "pipeline_name": aws_context.get("pipeline_name")
                or artifacts.get("pipeline_name")
                or f"SagemakerEvaluation-MTRLEvaluation",
            "role_arn": aws_context["role_arn"],
            "base_model_arn": base_model_arn,
            "agent_arn": self._agent_arn_resolved,
            "agent_qualifier": self.agent_qualifier,
            "dataset_uri": self.dataset,
            "s3_output_path": self.s3_output_path,
            "mlflow_resource_arn": self.mlflow_resource_arn,
            "mlflow_experiment_name": getattr(self, "mlflow_experiment_name", None)
                or aws_context.get("pipeline_name"),
            "eval_group_size": _str_or_none(hparams.get("eval_group_size")),
            "sampling_temperature": _str_or_none(hparams.get("sampling_temperature")),
            "top_p": _str_or_none(hparams.get("top_p")),
            "max_tokens": _str_or_none(hparams.get("max_tokens")),
            "pass_k_values": _str_or_none(hparams.get("pass_k_values")),
            "success_threshold": _str_or_none(hparams.get("success_threshold")),
            "stopping_condition": self.stopping_condition,
            "model_package_group_arn": model_package_group_arn,
            "source_model_package_arn": self._source_model_package_arn_cache,
            "action_arn_prefix": action_arn_prefix,
            "dataset_artifact_arn": None,
            "kms_key_arn": getattr(self, "kms_key_id", None),
            "vpc_config": bool(networking),
            "vpc_security_group_ids": vpc_security_group_ids,
            "vpc_subnets": vpc_subnets,
            "tags": self.tags,
            # Pre-stringified JobConfigDocument for the templates.
            "job_config_document_str": job_config_doc_str,
            "job_config_document_ft_str": job_config_doc_ft_str,
        }

    # --- Public entry points --------------------------------------------

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLEvaluator.evaluate",
        telemetry_params=[
            ("agent_qualifier", TelemetryParamType.ATTR_VALUE),
            ("agent_config", TelemetryParamType.ATTR_EXISTS),
            ("stopping_condition", TelemetryParamType.ATTR_EXISTS),
        ] + BASE_EVALUATOR_TELEMETRY_PARAMS,
    )
    def evaluate(self, dry_run: bool = False) -> Optional['MTRLEvaluationExecution']:
        """Render the MTRL pipeline and start a non-blocking execution.

        Args:
            dry_run (bool):
                If True, runs all validation (IAM, agent resolution, model
                resolution, template rendering) without submitting the
                evaluation. Returns None on success, raises on validation
                failure. Defaults to False.

        Returns:
            MTRLEvaluationExecution: The started pipeline execution, or None
            if dry_run=True.
            Call ``.wait()`` to block until completion and ``.show_results()``
            to render the aggregate report.

        Example:

            .. code:: python

                execution = evaluator.evaluate()
                execution.wait()
                execution.show_results()

                # Validate without submitting:
                evaluator.evaluate(dry_run=True)
        """
        # 1. Trainer-sourced resolution (no-op if model is not a trainer).
        self._resolve_trainer_defaults()

        # 2. Resolve agent ARN.
        self._resolve_agent_arn()

        if not self._agent_arn_resolved:
            raise ValueError(
                "[PySDK Error] 'agent_config' resolved to None. A valid agent "
                "ARN is required for evaluation. Provide either:\n"
                "  - A Bedrock AgentCore ARN: arn:aws:bedrock-agentcore:<region>:<account>:runtime/<id>\n"
                "  - A Lambda ARN: arn:aws:lambda:<region>:<account>:function:<name>"
            )

        # 3. AWS context + model artifacts (reuses BaseEvaluator plumbing).
        aws_context = self._get_aws_execution_context()
        artifacts = self._resolve_model_artifacts(aws_context["region"])
        if not self._base_model_arn_cache:
            self._base_model_arn_cache = self._base_model_arn
        if not self._base_model_name_cache:
            self._base_model_name_cache = self._base_model_name
        if not self._source_model_package_arn_cache:
            self._source_model_package_arn_cache = self._source_model_package_arn

        model_package_group_arn = self._get_model_package_group_arn()

        # 4. Template context.
        template_context = self._build_template_context(
            aws_context=aws_context,
            artifacts=artifacts,
            model_package_group_arn=model_package_group_arn,
        )

        # 5. Template selection + render.
        template_str = self._select_mtrl_template()
        pipeline_definition = self._render_pipeline_definition(template_str, template_context)

        if dry_run:
            _logger.info("Dry-run validation passed. No evaluation submitted.")
            return None

        # Dump the pipeline definition to a local JSON file for debugging.
        import json as _json_mod
        _debug_path = "mtrl_eval_pipeline_input.json"
        with open(_debug_path, "w") as _f:
            _json_mod.dump(_json_mod.loads(pipeline_definition), _f, indent=2)
        _logger.info(f"Pipeline definition written to {_debug_path}")

        # 6. Start execution via custom boto3 path. The MTRL pipeline uses the
        #    "Job" step type which requires the beta endpoint for CreatePipeline.
        #    We still tag the pipeline for discoverability via get_all().
        name = self.base_eval_name or f"mtrl-eval-{(self._base_model_name_cache or 'model')}"
        return self._start_mtrl_execution(
            pipeline_definition=pipeline_definition,
            name=name,
            role_arn=aws_context["role_arn"],
            region=aws_context["region"],
        )

    def _get_mlflow_presigned_url(self, region: str, sm_client=None) -> Optional[str]:
        """Generate a presigned MLflow tracking server URL for the user.

        Uses the provided sm_client to call
        create_presigned_mlflow_app_url. Falls back to a console deep-link if
        the presigned URL call fails.
        """
        if not self.mlflow_resource_arn:
            return None

        eval_experiment_name = (
            getattr(self, "mlflow_experiment_name", None)
            or f"mtrl-eval-{self._base_model_name_cache or 'default'}"
        )

        base_url = None

        # Try presigned URL via the provided client first (respects beta endpoint).
        if sm_client is not None:
            try:
                response = sm_client.create_presigned_mlflow_app_url(
                    Arn=self.mlflow_resource_arn
                )
                base_url = response.get("AuthorizedUrl")
            except Exception as e:
                _logger.debug(f"Presigned MLflow URL via sm_client failed: {e}")

        # Fallback: get presigned URL via default SageMakerClient
        if not base_url:
            try:
                from sagemaker.core.utils.utils import SageMakerClient
                client = SageMakerClient().sagemaker_client
                response = client.create_presigned_mlflow_app_url(
                    Arn=self.mlflow_resource_arn
                )
                base_url = response.get("AuthorizedUrl")
            except Exception as e:
                _logger.debug(f"Presigned MLflow URL via SageMakerClient failed: {e}")

        if not base_url:
            return None

        # Build deep link with experiment name in the URL
        # We can't resolve experiment name → ID without an authenticated MLflow session,
        # so we use the experiment name directly in the search filter deep link
        from sagemaker.train.common_utils.mlflow_url_utils import _build_mlflow_deep_link_by_name
        return _build_mlflow_deep_link_by_name(base_url, eval_experiment_name)

    def _start_mtrl_execution(self, pipeline_definition, name, role_arn, region):
        """Start MTRL pipeline execution via boto3.

        This method handles pipeline get-or-create with proper evaluation
        tagging so executions are discoverable via ``get_all()``.
        """
        import uuid
        import boto3
        from .execution import MTRLEvaluationExecution, PipelineExecutionStatus
        from .constants import _get_pipeline_name_prefix, _TAG_SAGEMAKER_MODEL_EVALUATION

        sm_client = boto3.client("sagemaker", region_name=region)

        pipeline_prefix = _get_pipeline_name_prefix(EvalType.MTRL)
        pipeline_name = pipeline_prefix

        # Search for existing MTRL pipeline
        existing_pipeline_name = None
        try:
            resp = sm_client.list_pipelines(PipelineNamePrefix=pipeline_prefix)
            for p in resp.get("PipelineSummaries", []):
                existing_pipeline_name = p["PipelineName"]
                break
        except Exception:
            pass

        if existing_pipeline_name:
            pipeline_name = existing_pipeline_name
            sm_client.update_pipeline(
                PipelineName=pipeline_name,
                PipelineDefinition=pipeline_definition,
                RoleArn=role_arn,
            )
            _logger.info(f"Updated existing pipeline: {pipeline_name}")
        else:
            sm_client.create_pipeline(
                PipelineName=pipeline_name,
                PipelineDefinition=pipeline_definition,
                RoleArn=role_arn,
                PipelineDisplayName=pipeline_name,
                PipelineDescription="MTRL evaluation pipeline",
                ClientRequestToken=str(uuid.uuid4()),
                Tags=[{"Key": _TAG_SAGEMAKER_MODEL_EVALUATION, "Value": "true"}],
            )
            _logger.info(f"Created pipeline: {pipeline_name}")

        # Start execution
        resp = sm_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName=f"{name}-{int(__import__('time').time())}",
            ClientRequestToken=str(uuid.uuid4()),
        )
        exec_arn = resp["PipelineExecutionArn"]
        _logger.info(f"Started MTRL pipeline execution: {exec_arn}")

        # Build execution object using the shared subclass
        from sagemaker.core.resources import PipelineExecution

        execution = MTRLEvaluationExecution(
            name=name,
            arn=exec_arn,
            eval_type=EvalType.MTRL,
            s3_output_path=self.s3_output_path,
            status=PipelineExecutionStatus(overall_status="Executing"),
        )

        # Store the pipeline execution reference for wait/refresh
        try:
            pe = PipelineExecution.get(pipeline_execution_arn=exec_arn, region=region)
            execution._pipeline_execution = pe
        except Exception as e:
            _logger.debug(f"Could not fetch PipelineExecution for wait/refresh: {e}")

        # Print job summary and MLflow URL for the user.
        mlflow_url = self._get_mlflow_presigned_url(region, sm_client=sm_client)
        template_type = self._select_mtrl_template()
        if "BASE_MODEL_ONLY" in template_type:
            eval_mode = "Base model only"
        elif "FINE_TUNED_ONLY" in template_type:
            eval_mode = "Fine-tuned model only"
        else:
            eval_mode = "Base + Fine-tuned comparison"

        print(f"\n{'─' * 60}")
        print(f"  MTRL Evaluation Job")
        print(f"{'─' * 60}")
        print(f"  Model          : {self._base_model_name_cache or self.model}")
        print(f"  Eval mode      : {eval_mode}")
        print(f"  Dataset        : {self.dataset}")
        print(f"  Agent          : {self._agent_arn_resolved or self.agent_config}")
        print(f"  Output         : {self.s3_output_path}")
        print(f"  Region         : {region}")
        if mlflow_url:
            print(f"  MLflow URL     : {mlflow_url}")
        else:
            print(f"  MLflow ARN     : {self.mlflow_resource_arn}")
        print(f"{'─' * 60}")
        print(f"  Pipeline execution started: {exec_arn}\n")

        # Store MLflow URL and config on execution for later access.
        execution.mlflow_url = mlflow_url
        execution.mlflow_resource_arn = self.mlflow_resource_arn
        execution.mlflow_experiment_name = (
            getattr(self, "mlflow_experiment_name", None)
            or f"mtrl-eval-{self._base_model_name_cache or 'default'}"
        )

        return execution

    @classmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLEvaluator.get_all",
    )
    def get_all(cls, session=None, region=None):
        """List all MTRL evaluation executions in the account / region.

        Args:
            session: Optional boto3 session.
            region: Optional AWS region.

        Yields:
            EvaluationPipelineExecution: MTRL evaluation execution instances.
        """
        from .execution import EvaluationPipelineExecution
        yield from EvaluationPipelineExecution.get_all(
            eval_type=EvalType.MTRL, session=session, region=region
        )

    @staticmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLEvaluator.list_supported_models",
    )
    def list_supported_models(session=None) -> list:
        """Return the list of models that support MTRL evaluation.

        Queries SageMakerPublicHub to discover all models with MTRL
        recipes in their ``RecipeCollection``.

        Args:
            session: Optional boto3 session.

        Returns:
            List of hub content model names supporting MTRL evaluation.
        """
        from sagemaker.train.common_utils.recipe_utils import _list_hub_models_by_recipe
        return _list_hub_models_by_recipe(
            recipe_type="FineTuning", technique="MTRL", session=session
        )

    @staticmethod
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="MultiTurnRLEvaluator.list_bedrock_agentcore_runtimes",
    )
    def list_bedrock_agentcore_runtimes(session=None) -> list:
        """List Bedrock AgentCore runtimes.

        Args:
            session: Optional boto3 session.

        Returns:
            List of dicts, each with keys ``name``, ``runtime_id``, ``arn``,
            and ``status``.
        """
        import boto3

        client = (session or boto3.Session()).client("bedrock-agentcore-control")
        runtimes: list = []
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
