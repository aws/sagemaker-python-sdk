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
"""InspectAI Evaluator for SageMaker Model Evaluation Module.

This module provides evaluation capabilities using InspectAI as a backend,
enabling a broad set of benchmarks and methodologies via the InspectAI framework.
The evaluator runs InspectAI tasks inside a dedicated container on SageMaker
Training infrastructure.
"""

import logging
import os
import re
import uuid
from typing import Any, Dict, Iterator, List, Optional

import yaml
from pydantic import root_validator, validator
from sagemaker.core.s3.client import S3Uploader
from sagemaker.core.telemetry.constants import Feature
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter

from .base_evaluator import BaseEvaluator
from .constants import EvalType, _get_inspect_ai_default_image_uri, _NOVA_ESCROW_ACCOUNTS
from .execution import EvaluationPipelineExecution
from .pipeline_templates import INSPECT_AI_TEMPLATE

_logger = logging.getLogger(__name__)

_ECR_URI_PATTERN = re.compile(
    r"^\d{12}\.dkr\.ecr\.[a-z0-9-]+\.amazonaws\.com/[a-z0-9._/-]+(:[a-zA-Z0-9._-]+)?$"
)
_IAM_ROLE_ARN_PATTERN = re.compile(r"^arn:aws(-cn|-us-gov|-iso-f)?:iam::\d{12}:role/.+$")


class InspectAIEvaluator(BaseEvaluator):
    """InspectAI evaluation job.

    Runs InspectAI tasks inside a SageMaker Training container, supporting three
    inference provider modes: Bedrock, existing SageMaker endpoint, or creating a
    new endpoint.

    The evaluator serializes configuration to a YAML file (``inspect_config.yaml``),
    uploads it to S3, and launches a single-step SageMaker Pipeline that runs the
    InspectAI container with that config as input.

    Supports resource chaining: a completed trainer (e.g., ``SFTTrainer``,
    ``DPOTrainer``, ``MultiTurnRLTrainer``) can be passed directly as the
    ``model`` parameter. The evaluator will automatically resolve the trainer's
    output model package artifacts and configure endpoint creation for evaluation.

    Attributes:
        benchmarks_path (str): S3 URI pointing to benchmark ``.py`` files with
            ``@task`` decorators. Required.
        tasks (Optional[List[Dict[str, Any]]]): List of task configurations. Each dict
            must have a ``"name"`` key. Optional keys: ``"path"`` (must end with .py),
            ``"limit"`` (int >= 1), ``"epochs"`` (int >= 1), ``"task_args"`` (dict).
            If None or empty, all tasks at ``benchmarks_path`` are run.
        output_format (Optional[str]): Output format for results. One of
            ``"eval"``, ``"csv"``, ``"jsonl"``, ``"json"``.
        bedrock_model_id (Optional[str]): Explicit Bedrock model ID for bedrock
            inference mode. Falls back to the model's bedrock_model_id if not set.
        endpoint_name (Optional[str]): Existing SageMaker endpoint name. Mutually
            exclusive with ``model_s3_uri``/``inference_image_uri``.
        model_s3_uri (Optional[str]): S3 URI of model artifacts for creating a new
            endpoint. Must be paired with ``inference_image_uri``.
        inference_image_uri (Optional[str]): ECR image URI for creating a new endpoint.
            Must be paired with ``model_s3_uri``.
        endpoint_instance_type (Optional[str]): Instance type for new endpoint
            (must start with ``ml.``).
        endpoint_instance_count (int): Instance count for new endpoint. Defaults to 1.
        endpoint_execution_role_arn (Optional[str]): IAM role ARN for new endpoint.
        context_length (Optional[str]): Context length as string integer.
        max_concurrency (Optional[str]): Max concurrency as string integer.
        cleanup_endpoint (bool): Delete endpoint after evaluation. Defaults to True.
        endpoint_prefix (str): Prefix for auto-created endpoint names.
        endpoint_environment (Optional[Dict[str, str]]): Env vars for the inference
            endpoint container.
        extra_args (Optional[List[str]]): Additional CLI args forwarded to
            ``inspect eval``.
        environment (Optional[Dict[str, str]]): Env vars for the SageMaker Training
            Job container.
        image_uri (Optional[str]): Override for the InspectAI container image URI.
        instance_type (str): Instance type for the orchestrator Training Job (CPU-only).
            Defaults to ``"ml.m5.large"``.
        max_runtime_seconds (int): Max runtime for the Training Job in seconds.
            Defaults to 86400 (24 hours).
        max_connections (int): Max concurrent inference connections used by the
            InspectAI eval runner. Defaults to 16.
        max_retries (int): Max retries per inference request. Defaults to 100.
        timeout (int): Per-request timeout in seconds. Defaults to 600.
        temperature (float): Sampling temperature in [0.0, 2.0]. Defaults to 0.0.
        top_p (float): Nucleus sampling cutoff in [0.0, 1.0]. Defaults to 1.0.
        top_k (int): Top-k sampling cutoff. Use ``-1`` to disable. Defaults to -1.
        max_tokens (int): Max tokens to generate per response. Defaults to 8192.

    Example:
        .. code:: python

            from sagemaker.train.evaluate import InspectAIEvaluator

            evaluator = InspectAIEvaluator(
                model="amazon-nova-lite-v1",
                benchmarks_path="s3://my-bucket/benchmarks/",
                tasks=[{"name": "boolq_pt", "limit": 10}],
                s3_output_path="s3://my-bucket/eval-output/",
            )
            execution = evaluator.evaluate()
            execution.wait()
            execution.show_results()

        Resource chaining with a trainer:

        .. code:: python

            from sagemaker.train import SFTTrainer
            from sagemaker.train.evaluate import InspectAIEvaluator

            # Train a model
            trainer = SFTTrainer(model="llama3-2-1b-instruct", ...)
            trainer.train(training_dataset="s3://bucket/data.jsonl")

            # Evaluate the fine-tuned model directly
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path="s3://my-bucket/benchmarks/",
                tasks=[{"name": "boolq_pt", "limit": 10}],
                s3_output_path="s3://my-bucket/eval-output/",
            )
            execution = evaluator.evaluate()
    """

    # InspectAI-specific fields
    benchmarks_path: str
    tasks: Optional[List[Dict[str, Any]]] = None
    output_format: Optional[str] = None
    bedrock_model_id: Optional[str] = None
    endpoint_name: Optional[str] = None
    model_s3_uri: Optional[str] = None
    inference_image_uri: Optional[str] = None
    endpoint_instance_type: Optional[str] = None
    endpoint_instance_count: int = 1
    endpoint_execution_role_arn: Optional[str] = None
    context_length: Optional[str] = None
    max_concurrency: Optional[str] = None
    cleanup_endpoint: bool = True
    endpoint_prefix: str = "inspectai"
    endpoint_environment: Optional[Dict[str, str]] = None
    extra_args: Optional[List[str]] = None
    environment: Optional[Dict[str, str]] = None
    image_uri: Optional[str] = None
    instance_type: str = "ml.m5.large"
    max_runtime_seconds: int = 86400

    # Eval orchestration tunables (forwarded into eval section of inspect_config.yaml)
    max_connections: int = 16
    max_retries: int = 100
    timeout: int = 600

    # Decoding tunables (forwarded into eval.decoding section of inspect_config.yaml)
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 8192

    @validator("environment")
    def _validate_environment(cls, v):
        if v is None:
            return v
        for key, val in v.items():
            if not isinstance(key, str) or not isinstance(val, str):
                raise ValueError("environment must be a flat Dict[str, str]")
        return v

    @validator("benchmarks_path")
    def _validate_benchmarks_path(cls, v):
        if not v or not v.strip():
            raise ValueError("benchmarks_path is required and cannot be empty")
        if not v.startswith("s3://"):
            raise ValueError(f"benchmarks_path must start with 's3://'. Got: '{v}'")
        return v

    @validator("tasks")
    def _validate_tasks(cls, v):
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError("tasks must be a list of dicts")
        for i, task in enumerate(v):
            if not isinstance(task, dict):
                raise ValueError(f"tasks[{i}] must be a dict, got {type(task).__name__}")
            if "name" not in task:
                raise ValueError(f"tasks[{i}] must have a 'name' key")
            if "path" in task and not task["path"].endswith(".py"):
                raise ValueError(f"tasks[{i}]['path'] must end with '.py'. Got: '{task['path']}'")
            if "limit" in task:
                if not isinstance(task["limit"], int) or task["limit"] < 1:
                    raise ValueError(f"tasks[{i}]['limit'] must be an integer >= 1")
            if "epochs" in task:
                if not isinstance(task["epochs"], int) or task["epochs"] < 1:
                    raise ValueError(f"tasks[{i}]['epochs'] must be an integer >= 1")
            if "task_args" in task and not isinstance(task["task_args"], dict):
                raise ValueError(f"tasks[{i}]['task_args'] must be a dict")
        return v

    @validator("output_format")
    def _validate_output_format(cls, v):
        if v is None:
            return v
        allowed = ("eval", "csv", "jsonl", "json")
        if v not in allowed:
            raise ValueError(f"output_format must be one of {allowed}. Got: '{v}'")
        return v

    @validator("model_s3_uri")
    def _validate_model_s3_uri(cls, v):
        if v is not None and not v.startswith("s3://"):
            raise ValueError(f"model_s3_uri must start with 's3://'. Got: '{v}'")
        return v

    @validator("inference_image_uri")
    def _validate_inference_image_uri(cls, v):
        if v is not None and not _ECR_URI_PATTERN.match(v):
            raise ValueError(f"inference_image_uri must be a valid ECR URI. Got: '{v}'")
        return v

    @validator("endpoint_instance_type")
    def _validate_endpoint_instance_type(cls, v):
        if v is not None and not v.startswith("ml."):
            raise ValueError(f"endpoint_instance_type must start with 'ml.'. Got: '{v}'")
        return v

    @validator("endpoint_execution_role_arn")
    def _validate_endpoint_execution_role_arn(cls, v):
        if v is not None and not _IAM_ROLE_ARN_PATTERN.match(v):
            raise ValueError(
                f"endpoint_execution_role_arn must be a valid IAM role ARN. Got: '{v}'"
            )
        return v

    @root_validator(skip_on_failure=True)
    def _validate_inference_mode_consistency(cls, values):
        endpoint_name = values.get("endpoint_name")
        model_s3_uri = values.get("model_s3_uri")
        inference_image_uri = values.get("inference_image_uri")

        if endpoint_name and model_s3_uri:
            raise ValueError(
                "endpoint_name and model_s3_uri are mutually exclusive. "
                "Use endpoint_name for an existing endpoint, or model_s3_uri + "
                "inference_image_uri to create a new endpoint."
            )
        if model_s3_uri and not inference_image_uri:
            raise ValueError(
                "inference_image_uri is required when model_s3_uri is provided "
                "(create_endpoint mode)."
            )
        if inference_image_uri and not model_s3_uri:
            raise ValueError(
                "model_s3_uri is required when inference_image_uri is provided "
                "(create_endpoint mode)."
            )
        return values

    @root_validator(skip_on_failure=True)
    def _resolve_trainer_model(cls, values):
        """Auto-resolve model artifacts from a BaseTrainer for endpoint creation.

        When a trainer is passed as ``model`` and no explicit inference mode
        (``endpoint_name``, ``model_s3_uri``, ``bedrock_model_id``) is configured,
        this resolver extracts the model S3 URI and inference image URI from the
        trainer's output model package, enabling automatic ``create_endpoint`` mode.

        This supports resource chaining where a completed trainer can be fed
        directly into the evaluator without manual artifact lookup.
        """
        from sagemaker.train.base_trainer import BaseTrainer

        model = values.get("model")
        if not isinstance(model, BaseTrainer):
            return values

        # Only auto-resolve if no explicit inference mode is configured
        endpoint_name = values.get("endpoint_name")
        model_s3_uri = values.get("model_s3_uri")
        bedrock_model_id = values.get("bedrock_model_id")
        if endpoint_name or model_s3_uri or bedrock_model_id:
            return values

        # Resolve model package ARN from the trainer
        source_mp_arn = None
        # MultiTurnRLTrainer uses _latest_job
        if hasattr(model, "_latest_job") and model._latest_job is not None:
            source_mp_arn = getattr(model._latest_job, "output_model_package_arn", None)
        # Standard trainers (SFT, DPO, RLVR, RLAIF) use _latest_training_job
        if not source_mp_arn and hasattr(model, "_latest_training_job") and model._latest_training_job is not None:
            source_mp_arn = getattr(model._latest_training_job, "output_model_package_arn", None)

        if not source_mp_arn:
            _logger.info(
                "Trainer has no completed training job output; falling back to bedrock mode."
            )
            return values

        # Resolve model artifacts from the model package
        try:
            session = values.get("sagemaker_session")
            from sagemaker.core.resources import ModelPackage as _MP

            boto_session = (
                session.boto_session if hasattr(session, "boto_session") else session
            )
            region = boto_session.region_name if boto_session else None

            mp = _MP.get(
                model_package_name=source_mp_arn,
                session=boto_session,
                region=region,
            )

            # Extract model data URL and image URI from inference specification
            if (
                mp.inference_specification
                and mp.inference_specification.containers
            ):
                container = mp.inference_specification.containers[0]

                # Resolve model S3 URI: try model_data_url first, then model_data_source
                resolved_model_s3 = getattr(container, "model_data_url", None)
                if not resolved_model_s3:
                    model_data_source = getattr(container, "model_data_source", None)
                    if model_data_source:
                        s3_data_source = getattr(model_data_source, "s3_data_source", None)
                        if s3_data_source:
                            resolved_model_s3 = getattr(s3_data_source, "s3_uri", None)

                # Resolve inference image: try explicit image first, then derive
                # from base_model for Nova models using escrow account pattern
                resolved_image = getattr(container, "image", None)
                if not resolved_image:
                    base_model = getattr(container, "base_model", None)
                    if base_model:
                        hub_content_name = getattr(base_model, "hub_content_name", None)
                        if hub_content_name and "nova" in (hub_content_name or "").lower():
                            escrow_account = _NOVA_ESCROW_ACCOUNTS.get(region)
                            if escrow_account:
                                resolved_image = (
                                    f"{escrow_account}.dkr.ecr.{region}.amazonaws.com"
                                    f"/nova-inference-repo:SM-Inference-latest"
                                )

                if resolved_model_s3 and resolved_image:
                    _logger.info(
                        "Auto-resolved trainer model artifacts for create_endpoint mode: "
                        "model_s3_uri=%s, inference_image_uri=%s",
                        resolved_model_s3,
                        resolved_image,
                    )
                    values["model_s3_uri"] = resolved_model_s3
                    values["inference_image_uri"] = resolved_image
                else:
                    _logger.warning(
                        "Trainer output model package does not contain model S3 URI "
                        "or inference image in inference_specification; "
                        "falling back to bedrock mode. "
                        "(resolved_model_s3=%s, resolved_image=%s)",
                        resolved_model_s3,
                        resolved_image,
                    )
            else:
                _logger.warning(
                    "Trainer output model package has no inference_specification; "
                    "falling back to bedrock mode."
                )
        except Exception as e:
            _logger.warning(
                "Failed to resolve trainer model artifacts: %s. "
                "Falling back to bedrock mode.",
                e,
            )

        return values

    @validator("image_uri")
    def _validate_image_uri(cls, v):
        if v is not None and not _ECR_URI_PATTERN.match(v):
            raise ValueError(f"image_uri must be a valid ECR URI. Got: '{v}'")
        return v

    @validator("instance_type")
    def _validate_instance_type(cls, v):
        if not v.startswith("ml."):
            raise ValueError(f"instance_type must start with 'ml.'. Got: '{v}'")
        return v

    @validator("max_connections")
    def _validate_max_connections(cls, v):
        if v < 1:
            raise ValueError(f"max_connections must be >= 1. Got: {v}")
        return v

    @validator("max_retries")
    def _validate_max_retries(cls, v):
        if v < 1:
            raise ValueError(f"max_retries must be >= 1. Got: {v}")
        return v

    @validator("max_tokens")
    def _validate_max_tokens(cls, v):
        if v < 1:
            raise ValueError(f"max_tokens must be >= 1. Got: {v}")
        return v

    @validator("timeout")
    def _validate_timeout(cls, v):
        if v < 1:
            raise ValueError(f"timeout must be >= 1 (seconds). Got: {v}")
        return v

    @validator("temperature")
    def _validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError(f"temperature must be in [0.0, 2.0]. Got: {v}")
        return v

    @validator("top_p")
    def _validate_top_p(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError(f"top_p must be in [0.0, 1.0]. Got: {v}")
        return v

    @validator("top_k")
    def _validate_top_k(cls, v):
        # -1 disables top-k sampling; otherwise must be a positive int
        if v != -1 and v < 1:
            raise ValueError(f"top_k must be -1 (disabled) or >= 1. Got: {v}")
        return v

    def _infer_scenario(self) -> str:
        """Determine the inference provider mode.

        Returns:
            One of 'bedrock', 'existing_endpoint', 'create_endpoint'.
        """
        if self.endpoint_name:
            return "existing_endpoint"
        if self.model_s3_uri:
            return "create_endpoint"
        return "bedrock"

    def _get_bedrock_model_id(self, region: str) -> str:
        """Resolve the Bedrock model ID for bedrock inference mode.

        Priority: explicit bedrock_model_id > model's bedrock_model_id > model string.
        """
        if self.bedrock_model_id:
            return self.bedrock_model_id
        # Try to derive from model info (cross-region inference profile format)
        try:
            model_info = self._get_resolved_model_info()
            if hasattr(model_info, "bedrock_model_id") and model_info.bedrock_model_id:
                return model_info.bedrock_model_id
        except Exception:
            pass
        # Fall back to model string if it looks like a model ID
        # Use cross-region inference profile format: <continent_prefix>.<model_id>
        if isinstance(self.model, str) and not self.model.startswith("arn:"):
            region_prefix = region.split("-")[0]
            return f"{region_prefix}.{self.model}"
        raise ValueError(
            "Cannot determine Bedrock model ID. Provide bedrock_model_id explicitly "
            "or use a model that has a bedrock_model_id mapping."
        )

    def _build_inference_provider_config(self, region: str) -> dict:
        """Build the inference_provider section of the YAML config."""
        scenario = self._infer_scenario()

        if scenario == "bedrock":
            model_id = self._get_bedrock_model_id(region)
            return {
                "bedrock": {
                    "model_id": model_id,
                    "region": region,
                }
            }
        elif scenario == "existing_endpoint":
            config = {
                "sagemaker_endpoint": {
                    "endpoint_name": self.endpoint_name,
                    "region": region,
                }
            }
            if self.context_length:
                config["sagemaker_endpoint"]["context_length"] = self.context_length
            if self.max_concurrency:
                config["sagemaker_endpoint"]["max_concurrency"] = self.max_concurrency
            return config
        else:  # create_endpoint
            config = {
                "sagemaker_endpoint": {
                    "endpoint_name": None,
                    "region": region,
                    "model_s3_uri": self.model_s3_uri,
                    "inference_image_uri": self.inference_image_uri,
                    "cleanup_endpoint": self.cleanup_endpoint,
                    "instance_count": self.endpoint_instance_count,
                    "endpoint_prefix": self.endpoint_prefix,
                }
            }
            ep = config["sagemaker_endpoint"]
            # execution_role_arn: use explicit endpoint role, fall back to evaluator role
            execution_role = self.endpoint_execution_role_arn or self.role
            if execution_role:
                ep["execution_role_arn"] = execution_role
            if self.endpoint_instance_type:
                ep["instance_type"] = self.endpoint_instance_type
            if self.context_length:
                ep["context_length"] = self.context_length
            if self.max_concurrency:
                ep["max_concurrency"] = self.max_concurrency
            if self.endpoint_environment:
                ep["environment"] = self.endpoint_environment
            return config

    def _build_yaml_config(self, region: str) -> dict:
        """Build the complete YAML config dict for the InspectAI container.

        Matches the structure expected by the sagemaker-inspect-ai container:
        inference_provider, benchmarks, eval, output.
        """
        config = {}

        # Inference provider
        config["inference_provider"] = self._build_inference_provider_config(region)

        # Benchmarks
        benchmarks = {}
        if self.benchmarks_path:
            benchmarks["s3_path"] = self.benchmarks_path
        if self.tasks:
            benchmarks["tasks"] = []
            for task in self.tasks:
                task_entry = {"name": task["name"]}
                if "path" in task:
                    task_entry["path"] = task["path"]
                if "limit" in task:
                    task_entry["limit"] = task["limit"]
                if "epochs" in task:
                    task_entry["epochs"] = task["epochs"]
                if "task_args" in task:
                    task_entry["task_args"] = task["task_args"]
                benchmarks["tasks"].append(task_entry)
        config["benchmarks"] = benchmarks

        # Eval settings
        eval_config = {
            "max_connections": self.max_connections,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "decoding": {
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "max_tokens": self.max_tokens,
            },
        }
        if self.extra_args:
            eval_config["extra_args"] = self.extra_args
        config["eval"] = eval_config

        # Output
        output = {}
        output_path = self.s3_output_path.rstrip("/")
        output["s3_path"] = f"{output_path}/inspectai-results/"
        if self.output_format:
            output["output_format"] = self.output_format
        config["output"] = output

        return config

    def _upload_yaml_config(self, config: dict, region: str) -> str:
        """Serialize config to YAML and upload to S3.

        Returns the S3 prefix URI where inspect_config.yaml was uploaded.
        """
        yaml_content = yaml.dump(config, default_flow_style=False, sort_keys=False)

        s3_base = self.s3_output_path.rstrip("/")
        config_prefix = f"{s3_base}/inspectai-config/{uuid.uuid4()}"
        config_s3_uri = f"{config_prefix}/inspect_config.yaml"

        _logger.info(f"Uploading InspectAI config to: {config_s3_uri}")
        S3Uploader.upload_string_as_file_body(
            body=yaml_content,
            desired_s3_uri=config_s3_uri,
            kms_key=self.kms_key_id,
            sagemaker_session=self.sagemaker_session,
        )

        return config_prefix

    def upload_benchmarks(self, local_path: str) -> str:
        """Upload local benchmark files to S3.

        Uploads all files from a local directory to an S3 prefix under the
        configured output path. The uploaded path can be used as ``benchmarks_path``.

        Args:
            local_path: Local directory path containing ``.py`` files with
                ``@task`` decorators.

        Returns:
            S3 URI prefix where benchmarks were uploaded.

        Raises:
            ValueError: If local_path does not exist or is not a directory.
        """
        if not os.path.isdir(local_path):
            raise ValueError(f"local_path must be an existing directory. Got: '{local_path}'")

        s3_base = self.s3_output_path.rstrip("/")
        s3_prefix = f"{s3_base}/benchmarks/{uuid.uuid4()}"

        _logger.info(f"Uploading benchmarks from '{local_path}' to '{s3_prefix}'")
        S3Uploader.upload(
            local_path=local_path,
            desired_s3_uri=s3_prefix,
            kms_key=self.kms_key_id,
            sagemaker_session=self.sagemaker_session,
        )

        _logger.info(f"Benchmarks uploaded to: {s3_prefix}")
        return s3_prefix

    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION, func_name="InspectAIEvaluator.evaluate"
    )
    def evaluate(self) -> EvaluationPipelineExecution:
        """Create and start an InspectAI evaluation job.

        Serializes the InspectAI configuration to YAML, uploads it to S3, and
        launches a single-step SageMaker Pipeline with the InspectAI container.

        Returns:
            EvaluationPipelineExecution: The started evaluation execution with
                ``.wait()``, ``.refresh()``, and ``.show_results()`` methods.

        Example:
            .. code:: python

                evaluator = InspectAIEvaluator(
                    model="amazon-nova-lite-v1",
                    benchmarks_path="s3://my-bucket/benchmarks/",
                    tasks=[{"name": "boolq_pt", "limit": 10}],
                    s3_output_path="s3://my-bucket/eval-output/",
                )
                execution = evaluator.evaluate()
                execution.wait()
                execution.show_results()
        """
        # Get AWS execution context
        aws_context = self._get_aws_execution_context()
        region = aws_context["region"]
        role_arn = aws_context["role_arn"]

        # Build and upload YAML config
        yaml_config = self._build_yaml_config(region)
        config_s3_prefix = self._upload_yaml_config(yaml_config, region)

        # Resolve container image URI
        resolved_image_uri = self.image_uri or _get_inspect_ai_default_image_uri(region)

        # Build job name prefix (keep total under 63 chars after pipeline exec ID appended)
        base_name = self.base_eval_name or "inspectai-eval"
        job_name_prefix = base_name[:26]

        # Build template context
        template_context = {
            "job_name_prefix": job_name_prefix,
            "image_uri": resolved_image_uri,
            "role_arn": role_arn,
            "instance_type": self.instance_type,
            "max_runtime_seconds": self.max_runtime_seconds,
            "config_s3_uri": config_s3_prefix,
            "s3_output_path": self.s3_output_path.rstrip("/"),
            "kms_key_id": self.kms_key_id,
            "environment": self.environment,
            "vpc_config": self.networking is not None,
        }

        if self.networking:
            template_context["vpc_security_group_ids"] = self.networking.security_group_ids
            template_context["vpc_subnets"] = self.networking.subnets

        # Render pipeline definition
        pipeline_definition = self._render_pipeline_definition(
            INSPECT_AI_TEMPLATE, template_context
        )

        # Start execution
        name = self.base_eval_name or "inspectai-eval"
        return self._start_execution(
            eval_type=EvalType.INSPECT_AI,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=role_arn,
            region=region,
        )

    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="InspectAIEvaluator.get_all")
    def get_all(
        cls, session: Optional[Any] = None, region: Optional[str] = None
    ) -> Iterator[EvaluationPipelineExecution]:
        """Get all InspectAI evaluation executions.

        Args:
            session (Optional[Any]): Optional boto3 session.
            region (Optional[str]): Optional AWS region.

        Yields:
            EvaluationPipelineExecution: InspectAI evaluation execution instances.
        """
        yield from EvaluationPipelineExecution.get_all(
            eval_type=EvalType.INSPECT_AI,
            session=session,
            region=region,
        )
