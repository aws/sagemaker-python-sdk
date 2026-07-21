"""LLM-as-Judge Evaluator for SageMaker Model Evaluation Module.

This module provides evaluation capabilities using foundation models as judges
to evaluate LLM responses based on quality and responsible AI metrics.
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Union

from pydantic import root_validator, validator

from .base_evaluator import BaseEvaluator
from .constants import (
    EvalType,
    _get_inspect_ai_default_image_uri,
    _get_nova_inference_image_uri,
    _REGION_TO_BEDROCK_PREFIX,
)
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.common_utils.data_utils import validate_data_path_exists
from sagemaker.train.common_utils.model_aliases import NOVA_BEDROCK_MODEL_IDS
from sagemaker.train.common_utils.recipe_utils import _is_nova_model
from sagemaker.train.constants import _ALLOWED_EVALUATOR_MODELS
from sagemaker.train.defaults import TrainDefaults

_logger = logging.getLogger(__name__)


def _resolve_bedrock_model_id(base_model_name: str, region: str) -> Optional[str]:
    """Derive Bedrock inference profile ID from JumpStart model name + region.

    Returns None if the model isn't in the Nova→Bedrock mapping (i.e., it's not
    a Nova model or isn't supported for Bedrock routing).

    Args:
        base_model_name: Resolved JumpStart model name (e.g., "nova-textgeneration-lite")
        region: AWS region from session (e.g., "us-east-1")

    Returns:
        Full Bedrock model ID (e.g., "us.amazon.nova-lite-v1:0") or None
    """
    if not base_model_name:
        return None

    base_id = NOVA_BEDROCK_MODEL_IDS.get(base_model_name)
    if not base_id:
        return None

    prefix = _REGION_TO_BEDROCK_PREFIX.get(region)
    if not prefix:
        return None

    return f"{prefix}.{base_id}"


class LLMAsJudgeEvaluator(BaseEvaluator):
    """LLM-as-judge evaluation job.
    
    This evaluator uses foundation models to evaluate LLM responses
    based on various quality and responsible AI metrics.

    This feature is powered by Amazon Bedrock Evaluations. Your use of this feature is subject to pricing of
    Amazon Bedrock Evaluations, the Service Terms applicable to Amazon Bedrock, and the terms that apply to your
    usage of third-party models. Amazon Bedrock Evaluations may securely transmit data across AWS Regions within your
    geography for processing. For more information, access Amazon Bedrock Evaluations documentation.

    Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html

    Attributes:
        evaluator_model (str): AWS Bedrock foundation model identifier to use as the judge.
            Required. For supported models, see:
            https://docs.aws.amazon.com/bedrock/latest/userguide/evaluation-judge.html#evaluation-judge-supported
        dataset (Union[str, Any]): Evaluation dataset. Required. Accepts:
            - S3 URI (str): e.g., 's3://bucket/path/dataset.jsonl'
            - Dataset ARN (str): e.g., 'arn:aws:sagemaker:...:hub-content/AIRegistry/DataSet/...'
            - DataSet object: sagemaker.ai_registry.dataset.DataSet instance (ARN inferred automatically)
        builtin_metrics (Optional[List[str]]): List of built-in evaluation metric names to compute.
            The 'Builtin.' prefix from Bedrock documentation is optional and will be automatically
            removed if present. Examples: ['Correctness', 'Faithfulness'] or
            ['Builtin.Correctness', 'Builtin.Faithfulness']. Optional.
        custom_metrics (Optional[str]): JSON string containing array of custom metric definitions.
            Optional. For format details, see:
            https://docs.aws.amazon.com/bedrock/latest/userguide/model-evaluation-custom-metrics-prompt-formats.html
        mlflow_resource_arn (Optional[str]): ARN of the MLflow tracking server for experiment tracking.
            Optional. If not provided, the system will attempt to resolve it using the default
            MLflow app experience (checks domain match, account default, or creates a new app).
            Inherited from BaseEvaluator.
        evaluate_base_model (bool): Whether to evaluate the base model in addition to the custom
            model. Set to False to skip base model evaluation and only evaluate the custom model.
            Defaults to True (evaluates both models).
        region (Optional[str]): AWS region. Inherited from BaseEvaluator.
        sagemaker_session (Optional[Any]): SageMaker session object. Inherited from BaseEvaluator.
        model (Union[str, Any]): Model for evaluation. Inherited from BaseEvaluator.
        base_eval_name (Optional[str]): Base name for evaluation jobs. Inherited from BaseEvaluator.
        s3_output_path (str): S3 location for evaluation outputs. Inherited from BaseEvaluator.
        mlflow_experiment_name (Optional[str]): MLflow experiment name. Inherited from BaseEvaluator.
        mlflow_run_name (Optional[str]): MLflow run name. Inherited from BaseEvaluator.
        networking (Optional[VpcConfig]): VPC configuration. Inherited from BaseEvaluator.
        kms_key_id (Optional[str]): KMS key ID for encryption. Inherited from BaseEvaluator.
        model_package_group (Optional[Union[str, ModelPackageGroup]]): Model package group.
            Inherited from BaseEvaluator.
    
    Example:
        .. code:: python
        
            from sagemaker.train.evaluate import LLMAsJudgeEvaluator
            
            # Example with built-in metrics (prefix optional)
            # Both formats work - with or without 'Builtin.' prefix
            evaluator = LLMAsJudgeEvaluator(
                base_model="llama-3-3-70b-instruct",
                evaluator_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                dataset="s3://my-bucket/my-dataset.jsonl",
                builtin_metrics=["Correctness", "Helpfulness"],  # Prefix optional
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server",
                s3_output_path="s3://my-bucket/output"
            )
            execution = evaluator.evaluate()
            
            # Example with custom metrics
            custom_metrics = [
                {
                    "customMetricDefinition": {
                        "name": "PositiveSentiment",
                        "instructions": "Assess if the response has positive sentiment. Prompt: {{prompt}}\\nResponse: {{prediction}}",
                        "ratingScale": [
                            {"definition": "Good", "value": {"floatValue": 1.0}},
                            {"definition": "Poor", "value": {"floatValue": 0.0}}
                        ]
                    }
                }
            ]
            
            evaluator = LLMAsJudgeEvaluator(
                base_model="llama-3-3-70b-instruct",
                evaluator_model="anthropic.claude-3-haiku-20240307-v1:0",
                dataset="s3://my-bucket/dataset.jsonl",
                custom_metrics=custom_metrics,
                s3_output_path="s3://my-bucket/output"
            )
            execution = evaluator.evaluate()
            
            # Example evaluating only custom model (skip base model)
            evaluator = LLMAsJudgeEvaluator(
                base_model="llama-3-3-70b-instruct",
                evaluator_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                dataset="s3://my-bucket/my-dataset.jsonl",
                builtin_metrics=["Correctness"],  # Prefix optional
                evaluate_base_model=False,
                s3_output_path="s3://my-bucket/output"
            )
            execution = evaluator.evaluate()
    """
    
    evaluator_model: str
    dataset: Union[str, Any]
    builtin_metrics: Optional[List[str]] = None
    custom_metrics: Optional[str] = None

    # Template-required fields
    evaluate_base_model: bool = False
    
    @validator('dataset', pre=True)
    def _resolve_dataset(cls, v):
        """Resolve dataset to string (S3 URI or ARN) and validate format.

        Uses BaseEvaluator's common validation logic to avoid code duplication.
        """
        return BaseEvaluator._validate_and_resolve_dataset(v)
    
    @root_validator(skip_on_failure=True)
    def _validate_model_compatibility(cls, values):
        """Validate Nova model region compatibility for LLM-as-Judge.

        Nova JumpStart models are automatically routed to the InspectAI+Bedrock
        inference path. This validator ensures the session region supports
        Bedrock cross-region inference for Nova models.
        """
        
        # Get resolved model info if available
        resolved_info = values.get('_resolved_model_info')
        if resolved_info and resolved_info.base_model_name:
            base_model_name = resolved_info.base_model_name
            is_nova = _is_nova_model(base_model_name)
            
            if is_nova:
                session = values.get('sagemaker_session')
                region = session.boto_region_name if session and hasattr(session, 'boto_region_name') else None
                if region and region not in _REGION_TO_BEDROCK_PREFIX:
                    raise ValueError(
                        f"Nova model '{base_model_name}' is not supported for "
                        f"LLM-as-Judge evaluation in region '{region}'. "
                        f"Supported regions: {list(_REGION_TO_BEDROCK_PREFIX.keys())}"
                    )
        
        return values

    @validator('evaluator_model')
    def _validate_evaluator_model(cls, v, values):
        """Validate evaluator_model is allowed and check region compatibility."""
        
        if v not in _ALLOWED_EVALUATOR_MODELS:
            raise ValueError(
                f"Invalid evaluator_model '{v}'. "
                f"Allowed models are: {list(_ALLOWED_EVALUATOR_MODELS.keys())}"
            )
        
        # Get current region from session
        session = values.get('sagemaker_session')
        if session and hasattr(session, 'boto_region_name'):
            current_region = session.boto_region_name
            allowed_regions = _ALLOWED_EVALUATOR_MODELS[v]
            
            if current_region not in allowed_regions:
                raise ValueError(
                    f"Evaluator model '{v}' is not available in region '{current_region}'. "
                    f"Available regions for this model: {allowed_regions}"
                )
            
        return v
    
    def _should_use_inspectai_path(self) -> bool:
        """Determine if the InspectAI path should be used for Phase 1 inference.

        The InspectAI path is required when the model is a Nova model, which
        cannot use the existing ServerlessJobConfig inference-only path:

        1. Nova JumpStart model → InspectAI + Bedrock cross-region inference.
        2. Fine-tuned Nova model (identified by having a
           ``source_model_package_arn``) → InspectAI + SageMaker Endpoint.

        Non-Nova models (both JumpStart and fine-tuned model packages) continue
        to use the existing ServerlessJobConfig path.

        Returns:
            bool: True if the InspectAI path should be used, False otherwise.
        """
        if self._base_model_name and _is_nova_model(self._base_model_name):
            return True

        return False

    def _process_builtin_metrics(self, metrics: Optional[List[str]]) -> List[str]:
        """Process builtin metrics by removing 'Builtin.' prefix if present.
        
        Args:
            metrics: List of metric names, potentially with 'Builtin.' prefix
            
        Returns:
            List[str]: Processed metric names without 'Builtin.' prefix
        """
        if not metrics:
            return []
        
        processed_metrics = []
        for metric in metrics:
            # Remove 'Builtin.' prefix if present (case-insensitive)
            if metric.lower().startswith('builtin.'):
                processed_metric = metric[8:]  # Remove first 8 characters ('Builtin.')
            else:
                processed_metric = metric
            processed_metrics.append(processed_metric)
        
        return processed_metrics
    
    def _validate_custom_metrics_json(self, custom_metrics_json: Optional[str]) -> Optional[str]:
        """Validate custom metrics JSON string if provided.
        
        Args:
            custom_metrics_json: JSON string to validate
            
        Returns:
            Optional[str]: Validated JSON string or None
            
        Raises:
            ValueError: If JSON is invalid
        """
        if not custom_metrics_json:
            return None
        
        try:
            json.loads(custom_metrics_json)  # Validate JSON
            return custom_metrics_json
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in custom_metrics: {e}")
    
    def _resolve_llmaj_proxy_model_arn(self, region: str) -> str:
        """Resolve a non-Nova model ARN for the LLMAJEvaluation judging step.

        Nova models do not support EvaluationType=LLMAJEvaluation. Since the
        InspectAI path handles inference in Phase 1, Phase 2 (judging) only
        needs a BaseModelArn that the backend accepts. This method resolves
        a known non-Nova model from the SageMaker Public Hub to use as a proxy.

        Args:
            region: AWS region for hub content lookup.

        Returns:
            str: Hub content ARN for a non-Nova model that supports LLMAJEvaluation.
        """
        from sagemaker.train.common_utils.model_resolution import _resolve_base_model

        _LLMAJ_PROXY_MODEL_ID = "meta-textgeneration-llama-3-2-1b-instruct"

        try:
            proxy_info = _resolve_base_model(
                _LLMAJ_PROXY_MODEL_ID,
                sagemaker_session=self.sagemaker_session,
            )
            _logger.info(
                f"Resolved LLMAJ proxy model ARN for Nova: {proxy_info.base_model_arn}"
            )
            return proxy_info.base_model_arn
        except Exception as e:
            raise ValueError(
                f"Nova models do not support LLMAJEvaluation directly. "
                f"Failed to resolve proxy model '{_LLMAJ_PROXY_MODEL_ID}' "
                f"for the judging step: {e}"
            )

    def _emit_cost_warning(self, instance_type: str, inference_mode: str) -> None:
        """Emit a warning about additional InspectAI infrastructure costs.

        Informs the user that the InspectAI inference path incurs costs for
        both the orchestrator Training instance and the inference backend
        (Bedrock or SageMaker Endpoint), in addition to the LLMAJEvaluation
        judging step.

        Args:
            instance_type: The ML instance type used for the InspectAI
                orchestrator Training job (e.g., ``"ml.m5.large"``).
            inference_mode: Description of the inference backend
                (e.g., ``"Bedrock"`` or ``"SageMaker Endpoint"``).
        """
        _logger.warning(
            "This evaluation uses the InspectAI inference path. "
            "You will incur additional costs: "
            f"(1) A SageMaker Training instance ({instance_type}) for the InspectAI orchestrator. "
            f"(2) {inference_mode} inference costs for generating model responses. "
            "These costs are in addition to the LLMAJEvaluation judging step."
        )

    def _get_inference_model_id(self, region: str) -> Optional[str]:
        """Resolve the Bedrock model ID for InspectAI inference.

        For Nova JumpStart models, derives the Bedrock cross-region inference
        profile ID from the model name and session region. For custom models
        (model packages), returns None — they use endpoint-based inference.

        Args:
            region: AWS region from execution context.

        Returns:
            Bedrock model ID string (e.g., ``"us.amazon.nova-lite-v1:0"``),
            or None if endpoint-based inference should be used.
        """
        if self._source_model_package_arn is not None:
            # Custom model → endpoint path, no Bedrock model ID needed
            return None

        return _resolve_bedrock_model_id(self._base_model_name, region)

    def _build_inspectai_config(
        self, region: str, benchmark_s3_path: str, output_s3_uri: str
    ) -> dict:
        """Build the InspectAI YAML configuration for inference-only mode.

        Constructs the configuration dictionary that will be serialized to YAML
        and uploaded to S3 for the InspectAI Training job. The config controls
        which inference provider is used, which benchmark tasks to run, and
        eval/output settings.

        :param region: AWS region for the inference provider.
        :param benchmark_s3_path: S3 path prefix where benchmark files
            (``inference_only.py``, ``pyproject.toml``, ``dataset.jsonl``) are stored.
        :param output_s3_uri: S3 URI where the benchmark should write the
            ``{prompt, response}`` JSONL output for Phase 2.
        :returns: Configuration dictionary ready for YAML serialization.
        """
        config: dict = {}

        bedrock_model_id = self._get_inference_model_id(region)
        if bedrock_model_id is not None:
            config["inference_provider"] = {
                "bedrock": {
                    "model_id": bedrock_model_id,
                    "region": region,
                }
            }
        else:
            model_s3_uri, inference_image_uri = (
                self._resolve_model_artifacts_for_endpoint(region)
            )
            config["inference_provider"] = {
                "sagemaker_endpoint": {
                    "model_s3_uri": model_s3_uri,
                    "inference_image_uri": inference_image_uri,
                    "cleanup_endpoint": True,
                    "instance_count": 1,
                    "endpoint_prefix": "llmaj-infer",
                }
            }

        config["benchmarks"] = {
            "s3_path": benchmark_s3_path,
            "tasks": [
                {
                    "name": "inference_only",
                    "task_args": {
                        "dataset": "dataset.jsonl",
                        "output_s3_uri": output_s3_uri,
                        "region": region,
                    },
                }
            ],
        }

        # Eval settings — max_connections set low for rate-limit reliability
        config["eval"] = {
            "max_connections": 1,
            "max_retries": 100,
            "timeout": 600,
            "decoding": {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_tokens": 8192,
            },
        }

        s3_base = self.s3_output_path.rstrip("/")
        config["output"] = {
            "s3_path": f"{s3_base}/inspectai-results/",
        }

        return config

    def _resolve_model_artifacts_for_endpoint(self, region: str) -> tuple[str, str]:
        """Extract model S3 URI and inference image URI from the model package.

        Retrieves the model package specified by ``self._source_model_package_arn``
        and resolves the deployable artifacts needed to create a SageMaker endpoint
        for InspectAI inference.

        The method attempts to resolve:
        - **model_s3_uri**: First from ``model_data_url``, then from
          ``model_data_source.s3_data_source.s3_uri`` on the first container.
        - **inference_image_uri**: From the container's ``image`` field. If absent,
          derives the image URI from the ``base_model.hub_content_name`` for Nova
          models using region-specific escrow accounts.

        :param region: AWS region for model package retrieval and image URI derivation.
        :type region: str
        :returns: A tuple of ``(model_s3_uri, inference_image_uri)``.
        :rtype: tuple[str, str]
        :raises ValueError: If the model package cannot be retrieved, lacks an
            inference specification, or the model S3 URI or inference image URI
            cannot be resolved.
        """
        from sagemaker.core.resources import ModelPackage
        from sagemaker.core.utils.utils import Unassigned

        model_package_arn = self._source_model_package_arn

        try:
            session = self.sagemaker_session
            boto_session = (
                session.boto_session if hasattr(session, "boto_session") else session
            )
            mp = ModelPackage.get(
                model_package_name=model_package_arn,
                session=boto_session,
                region=region,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve model package '{model_package_arn}': {e}. "
                "Ensure the model package ARN is correct and accessible in the "
                f"current region ({region})."
            )

        # Validate inference spec has containers
        if (
            not mp.inference_specification
            or isinstance(mp.inference_specification, Unassigned)
            or not mp.inference_specification.containers
        ):
            raise ValueError(
                f"Model package '{model_package_arn}' does not have an "
                "inference_specification with containers. Cannot resolve model "
                "artifacts for endpoint creation. Ensure the model package was "
                "created via SageMaker fine-tuning and has deployable artifacts."
            )

        container = mp.inference_specification.containers[0]

        # Resolve model S3 URI: try model_data_url first, then model_data_source
        model_s3_uri = getattr(container, "model_data_url", None)
        if isinstance(model_s3_uri, Unassigned) or not model_s3_uri:
            model_s3_uri = None
            model_data_source = getattr(container, "model_data_source", None)
            if model_data_source and not isinstance(model_data_source, Unassigned):
                s3_data_source = getattr(model_data_source, "s3_data_source", None)
                if s3_data_source and not isinstance(s3_data_source, Unassigned):
                    s3_uri = getattr(s3_data_source, "s3_uri", None)
                    if s3_uri and not isinstance(s3_uri, Unassigned):
                        model_s3_uri = s3_uri

        if not model_s3_uri:
            raise ValueError(
                f"Cannot resolve model S3 URI from model package "
                f"'{model_package_arn}'. The inference_specification.containers[0] "
                "has neither 'model_data_url' nor "
                "'model_data_source.s3_data_source.s3_uri'. Ensure the model "
                "package contains deployable model artifacts."
            )

        # Resolve inference image URI: try explicit image first
        inference_image_uri = getattr(container, "image", None)
        if isinstance(inference_image_uri, Unassigned) or not inference_image_uri:
            inference_image_uri = None

            # Derive from base_model for Nova models using escrow account pattern
            base_model = getattr(container, "base_model", None)
            if base_model and not isinstance(base_model, Unassigned):
                hub_content_name = getattr(base_model, "hub_content_name", None)
                if (
                    hub_content_name
                    and not isinstance(hub_content_name, Unassigned)
                    and "nova" in hub_content_name.lower()
                ):
                    inference_image_uri = _get_nova_inference_image_uri(region)

        if not inference_image_uri:
            raise ValueError(
                f"Cannot resolve inference image URI from model package "
                f"'{model_package_arn}'. The inference_specification.containers[0] "
                "does not have an 'image' field, and the base model is not a "
                "recognized Nova model for automatic image derivation. Ensure "
                "the model package has a valid inference container image specified."
            )

        _logger.info(
            "Resolved model artifacts for endpoint: model_s3_uri=%s, "
            "inference_image_uri=%s",
            model_s3_uri,
            inference_image_uri,
        )

        return (model_s3_uri, inference_image_uri)

    def _upload_benchmark_and_dataset(self, region: str, output_s3_uri: str) -> str:
        """Generate benchmark files, download and convert the dataset, and upload to S3.

        This method handles the full preparation of InspectAI benchmark artifacts:

        1. Generates the InspectAI benchmark Python file and ``pyproject.toml``.
        2. Downloads the customer's evaluation dataset from S3 (resolving Dataset
           ARN to S3 URI if necessary).
        3. Converts the dataset to InspectAI format (``{"input": ..., "target": ""}``).
        4. Uploads all files to a unique S3 prefix under the configured output path.

        :param region: AWS region for S3 operations.
        :type region: str
        :param output_s3_uri: The S3 URI where inference output will be written.
            Embedded in the benchmark config for the InspectAI task.
        :type output_s3_uri: str
        :returns: The S3 prefix where benchmark files were uploaded.
        :rtype: str
        :raises ValueError: If the dataset cannot be downloaded or converted.
        """
        from sagemaker.core.s3.client import S3Downloader, S3Uploader

        from .llmaj_inference_benchmark import (
            convert_dataset_to_inspectai_format,
            generate_benchmark_files,
        )

        # 1. Generate benchmark files
        benchmark_files = generate_benchmark_files()

        # 2. Resolve dataset URI (handle ARN → S3 URI if needed)
        dataset_uri = self.dataset
        if dataset_uri.startswith("arn:") and "hub-content" in dataset_uri and "/DataSet/" in dataset_uri:
            dataset_uri = self._resolve_dataset_arn_to_s3_uri(dataset_uri)

        # 3. Download customer dataset from S3
        try:
            raw_content = S3Downloader.read_file(
                s3_uri=dataset_uri,
                sagemaker_session=self.sagemaker_session,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to download dataset from {dataset_uri}: {e}"
            ) from e

        # 4. Convert to InspectAI format
        converted_dataset = convert_dataset_to_inspectai_format(raw_content)

        # 5. Upload everything to a unique S3 prefix
        s3_base = self.s3_output_path.rstrip("/")
        benchmark_prefix = f"{s3_base}/llmaj-benchmarks/{uuid.uuid4()}"

        for filename, content in benchmark_files.items():
            S3Uploader.upload_string_as_file_body(
                body=content,
                desired_s3_uri=f"{benchmark_prefix}/{filename}",
                kms_key=self.kms_key_id,
                sagemaker_session=self.sagemaker_session,
            )

        S3Uploader.upload_string_as_file_body(
            body=converted_dataset,
            desired_s3_uri=f"{benchmark_prefix}/dataset.jsonl",
            kms_key=self.kms_key_id,
            sagemaker_session=self.sagemaker_session,
        )

        _logger.info(f"Uploaded benchmark and dataset to: {benchmark_prefix}")
        return benchmark_prefix

    def _resolve_dataset_arn_to_s3_uri(self, dataset_arn: str) -> str:
        """Resolve a hub-content Dataset ARN to its S3 URI.

        Calls ``AIRHub.describe_hub_content`` to retrieve the dataset's S3
        bucket and prefix, then constructs the full S3 URI.

        :param dataset_arn: The hub-content Dataset ARN to resolve.
        :type dataset_arn: str
        :returns: The S3 URI pointing to the dataset file.
        :rtype: str
        :raises ValueError: If the ARN cannot be resolved to an S3 location.
        """
        import json as _json

        from sagemaker.ai_registry.air_hub import AIRHub
        from sagemaker.ai_registry.air_constants import (
            DOC_KEY_DATASET_S3_BUCKET,
            DOC_KEY_DATASET_S3_PREFIX,
            DATASET_HUB_CONTENT_TYPE,
            RESPONSE_KEY_HUB_CONTENT_DOCUMENT,
        )

        # ARN format: arn:aws:sagemaker:region:account:hub-content/HubName/DataSet/Name/Version
        try:
            arn_parts = dataset_arn.split(":")
            resource_parts = arn_parts[-1].split("/")
            hub_content_name = resource_parts[3]
        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Failed to parse dataset ARN '{dataset_arn}': {e}"
            ) from e

        try:
            response = AIRHub.describe_hub_content(
                hub_content_type=DATASET_HUB_CONTENT_TYPE,
                hub_content_name=hub_content_name,
                session=self.sagemaker_session,
            )
            doc = _json.loads(response[RESPONSE_KEY_HUB_CONTENT_DOCUMENT])
            bucket = doc.get(DOC_KEY_DATASET_S3_BUCKET, "")
            prefix = doc.get(DOC_KEY_DATASET_S3_PREFIX, "")

            if not bucket or not prefix:
                raise ValueError(
                    f"Dataset ARN '{dataset_arn}' resolved to empty S3 location "
                    f"(bucket={bucket!r}, prefix={prefix!r})."
                )

            return f"s3://{bucket}/{prefix}"
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(
                f"Failed to resolve dataset ARN '{dataset_arn}' to S3 URI: {e}"
            ) from e

    def _upload_custom_metrics_to_s3(self, custom_metrics_json: str, eval_name: str) -> str:
        """Upload custom metrics JSON to S3 and return the S3 path.
        
        Args:
            custom_metrics_json: JSON string of custom metrics
            eval_name: Evaluation name for path generation
            
        Returns:
            str: S3 path where custom metrics were uploaded
        """
        from datetime import datetime
        from sagemaker.core.s3.client import S3Uploader
        
        # Generate timestamp
        timestamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        
        # Strip trailing slash from S3 output path
        s3_base = self.s3_output_path.rstrip('/')
        
        # Construct S3 path: s3_output_path/evaluationinputs/{evaluation_name}{timestamp}/custom-metrics.json
        s3_path = f"{s3_base}/evaluationinputs/{eval_name}{timestamp}/custom-metrics.json"
        
        # Upload to S3 using S3Uploader
        _logger.info(f"Uploading custom metrics to S3: {s3_path}")
        S3Uploader.upload_string_as_file_body(
            body=custom_metrics_json,
            desired_s3_uri=s3_path,
            kms_key=self.kms_key_id,
            sagemaker_session=self.sagemaker_session
        )
        
        _logger.info(f"Successfully uploaded custom metrics to: {s3_path}")
        return s3_path
    
    def _get_llmaj_template_additions(self, eval_name: str) -> dict:
        """Get LLM-as-judge specific template context additions.
        
        Args:
            eval_name: Evaluation name for S3 path generation
        
        Returns:
            dict: LLM-as-judge specific template context fields
        """
        # Process builtin_metrics - remove 'Builtin.' prefix and convert to JSON string
        processed_metrics = self._process_builtin_metrics(self.builtin_metrics)
        llmaj_metrics_json = json.dumps(processed_metrics)
        
        # Validate custom_metrics JSON string if provided
        custom_metrics_json = self._validate_custom_metrics_json(self.custom_metrics)
        
        # Upload custom_metrics to S3 and get path if provided
        custom_metrics_s3_path = None
        if custom_metrics_json:
            custom_metrics_s3_path = self._upload_custom_metrics_to_s3(
                custom_metrics_json, 
                eval_name
            )
        
        # Strip trailing slash from S3 output path to avoid double slashes
        s3_output_path = self.s3_output_path.rstrip('/') if self.s3_output_path else self.s3_output_path
        
        return {
            'judge_model_id': self.evaluator_model,
            's3_output_path': s3_output_path,
            'llmaj_metrics': llmaj_metrics_json,
            'custom_metrics': custom_metrics_s3_path,
            'max_new_tokens': str(8192),
            'temperature': str(0),
            'top_k': str(-1),
            'top_p': str(1.0),
            'evaluate_base_model': self.evaluate_base_model,
        }
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="LLMAsJudgeEvaluator.evaluate")
    def evaluate(self, dry_run: bool = False):
        """Create and start an LLM-as-judge evaluation job.
        
        This method initiates a 2-phase evaluation job:
        
        1. Phase 1: Generate inference responses from base and custom models
        2. Phase 2: Use judge model to evaluate responses with built-in and custom metrics
        
        When the InspectAI path is active (custom model or Nova JumpStart model),
        Phase 1 runs inside an InspectAI container that generates
        inference responses and writes them to S3. Phase 2 remains unchanged —
        the LLMAJEvaluation judging step evaluates those responses with the
        judge model.

        Args:
            dry_run (bool):
                If True, runs all validation (IAM, model resolution, data paths)
                without submitting the evaluation. Returns None on success, raises
                on validation failure. Defaults to False.

        Returns:
            EvaluationPipelineExecution: The created LLM-as-judge evaluation execution,
            or None if dry_run=True.
        
        Raises:
            ValueError: If invalid model, dataset, or metric configurations are provided
        
        Example:
            .. code:: python
            
                evaluator = LLMAsJudgeEvaluator(
                    base_model="llama-3-3-70b-instruct",
                    evaluator_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    dataset="s3://my-bucket/my-dataset.jsonl",
                    builtin_metrics=["Correctness", "Helpfulness"],
                    s3_output_path="s3://my-bucket/output"
                )
                execution = evaluator.evaluate()
                execution.wait()
        """
        from .constants import EvalType, _get_inspect_ai_default_image_uri
        from .pipeline_templates import (
            LLMAJ_INSPECTAI_TEMPLATE,
            LLMAJ_TEMPLATE,
            LLMAJ_TEMPLATE_BASE_MODEL_ONLY,
        )
        
        # S3 checkpoint paths are not supported on serverless evaluation
        from sagemaker.train.common_utils.model_resolution import _ModelType
        info = self._get_resolved_model_info()
        if info and info.model_type == _ModelType.S3_CHECKPOINT:
            raise ValueError(
                "S3 checkpoint paths cannot be used with serverless evaluation. "
                "LLM-as-judge evaluation currently only supports serverless compute. "
                "Please use a Model Package ARN or JumpStart model ID instead."
            )

        # Get AWS execution context (role ARN, region, account ID)
        aws_context = self._get_aws_execution_context()
        region = aws_context['region']
        role_arn = aws_context['role_arn']
        
        # Resolve model artifacts
        artifacts = self._resolve_model_artifacts(region)
        
        # Get or infer model_package_group ARN (handles all cases internally)
        model_package_group_arn = self._get_model_package_group_arn()
        
        # Log resolved model information for debugging
        _logger.info(
            f"Resolved model info - base_model_name: {self._base_model_name}, "
            f"base_model_arn: {self._base_model_arn}, "
            f"source_model_package_arn: {self._source_model_package_arn}"
        )

        # Generate execution name early so we can use it for S3 paths
        name = self.base_eval_name or "llm-judge-eval"

        # InspectAI path (Nova models)
        if self._should_use_inspectai_path():
            import yaml
            from sagemaker.core.s3.client import S3Uploader

            inspectai_instance_type = "ml.m5.large"
            bedrock_model_id = self._get_inference_model_id(region)
            inference_mode = "Bedrock" if bedrock_model_id else "SageMaker Endpoint"
            self._emit_cost_warning(inspectai_instance_type, inference_mode)

            inference_run_id = str(uuid.uuid4())
            s3_base = self.s3_output_path.rstrip("/")
            inference_output_s3_uri = (
                f"{s3_base}/inference/{inference_run_id}/inference_output.jsonl"
            )

            benchmark_s3_path = self._upload_benchmark_and_dataset(
                region, inference_output_s3_uri
            )

            inspectai_config = self._build_inspectai_config(
                region, benchmark_s3_path, inference_output_s3_uri
            )
            yaml_content = yaml.dump(
                inspectai_config, default_flow_style=False, sort_keys=False
            )
            config_s3_prefix = (
                f"{s3_base}/inspectai-config/{inference_run_id}"
            )
            config_s3_uri = f"{config_s3_prefix}/config.yaml"
            _logger.info(f"Uploading InspectAI config to: {config_s3_uri}")
            S3Uploader.upload_string_as_file_body(
                body=yaml_content,
                desired_s3_uri=config_s3_uri,
                kms_key=self.kms_key_id,
                sagemaker_session=self.sagemaker_session,
            )

            inspectai_image_uri = _get_inspect_ai_default_image_uri(region)

            # Resolve mlflow_experiment_name: required when ModelPackageGroupArn is absent
            mlflow_experiment_name = self.mlflow_experiment_name
            if not mlflow_experiment_name and self.mlflow_resource_arn:
                mlflow_experiment_name = '{{ pipeline_name }}'
                _logger.info(
                    "No mlflow_experiment_name provided for InspectAI path, "
                    "using pipeline_name as default"
                )

            # Nova models don't support EvaluationType=LLMAJEvaluation. BaseModelArn
            # is required by the API, so we resolve a non-Nova proxy model ARN.
            if self._base_model_name and _is_nova_model(self._base_model_name):
                judge_step_base_model_arn = self._resolve_llmaj_proxy_model_arn(region)
            else:
                judge_step_base_model_arn = self._base_model_arn or self.model

            template_context = {
                'role_arn': role_arn,
                'mlflow_resource_arn': self.mlflow_resource_arn,
                'mlflow_experiment_name': mlflow_experiment_name,
                'inspectai_image_uri': inspectai_image_uri,
                'inspectai_instance_type': inspectai_instance_type,
                'inspectai_config_s3_uri': config_s3_prefix,
                's3_output_path': s3_base,
                'base_model_arn': judge_step_base_model_arn,
                'inference_output_s3_uri': inference_output_s3_uri,
            }

            llmaj_additions = self._get_llmaj_template_additions(name)
            template_context.update(llmaj_additions)

            if self._source_model_package_arn and model_package_group_arn:
                template_context['model_package_config'] = True
                template_context['model_package_group_arn'] = model_package_group_arn
                template_context['source_model_package_arn'] = (
                    self._source_model_package_arn
                )

            template_context = self._add_vpc_and_kms_to_context(template_context)

            pipeline_definition = self._render_pipeline_definition(
                LLMAJ_INSPECTAI_TEMPLATE, template_context
            )

            if dry_run:
                _logger.info("Dry-run validation passed. No evaluation submitted.")
                return None

            return self._start_execution(
                eval_type=EvalType.LLM_AS_JUDGE,
                name=name,
                pipeline_definition=pipeline_definition,
                role_arn=role_arn,
                region=region,
            )

        # Standard LLMAJ path (non-Nova JumpStart models)
        # Build base template context
        template_context = self._get_base_template_context(
            role_arn=role_arn,
            region=region,
            account_id=aws_context['account_id'],
            model_package_group_arn=model_package_group_arn,
            resolved_model_artifact_arn=artifacts['resolved_model_artifact_arn']
        )
        
        # Add dataset URI
        template_context['dataset_uri'] = self.dataset
        
        # Add LLM-as-judge specific template additions (needs eval name for S3 upload)
        llmaj_additions = self._get_llmaj_template_additions(name)
        template_context.update(llmaj_additions)
        
        # Add VPC and KMS configuration
        template_context = self._add_vpc_and_kms_to_context(template_context)
        
        # Select appropriate template
        template_str = self._select_template(
            LLMAJ_TEMPLATE_BASE_MODEL_ONLY,
            LLMAJ_TEMPLATE
        )
        
        # Render pipeline definition
        pipeline_definition = self._render_pipeline_definition(template_str, template_context)

        # Validate dataset path exists
        if hasattr(self, 'dataset') and self.dataset:
            session = TrainDefaults.get_sagemaker_session(
                sagemaker_session=self.sagemaker_session
            )
            validate_data_path_exists(
                self.dataset, session, label="evaluation dataset"
            )

        if dry_run:
            _logger.info("Dry-run validation passed. No evaluation submitted.")
            return None

        # Start execution (name already generated earlier)
        return self._start_execution(
            eval_type=EvalType.LLM_AS_JUDGE,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=role_arn,
            region=region,
        )
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="LLMAsJudgeEvaluator.get_all")
    def get_all(cls, session: Optional[Any] = None, region: Optional[str] = None):
        """Get all LLM-as-judge evaluation executions.
        
        Uses ``EvaluationPipelineExecution.get_all()`` to retrieve all LLM-as-judge
        evaluation executions as an iterator.
        
        Args:
            session (Optional[Any]): Optional boto3 session. If not provided, will be inferred.
            region (Optional[str]): Optional AWS region. If not provided, will be inferred.
        
        Yields:
            EvaluationPipelineExecution: LLM-as-judge evaluation execution instances
        
        Example:
            .. code:: python
            
                # Get all LLM-as-judge evaluations as iterator
                evaluations = LLMAsJudgeEvaluator.get_all()
                all_executions = list(evaluations)
                
                # Or iterate directly
                for execution in LLMAsJudgeEvaluator.get_all():
                    print(f"{execution.name}: {execution.status.overall_status}")
                
                # With specific session/region
                evaluations = LLMAsJudgeEvaluator.get_all(session=my_session, region='us-west-2')
                all_executions = list(evaluations)
        """
        from .execution import EvaluationPipelineExecution
        from .constants import EvalType
        
        # Use EvaluationPipelineExecution.get_all() with LLM_AS_JUDGE eval_type
        # This returns a generator, so we yield from it
        yield from EvaluationPipelineExecution.get_all(
            eval_type=EvalType.LLM_AS_JUDGE,
            session=session,
            region=region
        )
