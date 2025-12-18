"""LLM-as-Judge Evaluator for SageMaker Model Evaluation Module.

This module provides evaluation capabilities using foundation models as judges
to evaluate LLM responses based on quality and responsible AI metrics.
"""

import json
import logging
from typing import Any, List, Optional, Union

from pydantic import validator

from .base_evaluator import BaseEvaluator
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.constants import _ALLOWED_EVALUATOR_MODELS

_logger = logging.getLogger(__name__)


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
    
    @validator('model')
    def _validate_model_compatibility(cls, v, values):
        """Validate that Nova models are not used with LLM-as-judge evaluator"""
        from ..common_utils.recipe_utils import _is_nova_model
        
        # Get resolved model info if available
        resolved_info = values.get('_resolved_model_info')
        if resolved_info and resolved_info.base_model_name:
            base_model_name = resolved_info.base_model_name
            is_nova = _is_nova_model(base_model_name)
            
            # LLM-as-judge is not allowed for Nova models
            if is_nova:
                raise ValueError(
                    f"LLM-as-judge evaluation is not supported for Nova models. "
                    f"The current model '{base_model_name}' is a Nova model."
                )
        
        return v

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
    def evaluate(self):
        """Create and start an LLM-as-judge evaluation job.
        
        This method initiates a 2-phase evaluation job:
        
        1. Phase 1: Generate inference responses from base and custom models
        2. Phase 2: Use judge model to evaluate responses with built-in and custom metrics
        
        Returns:
            EvaluationPipelineExecution: The created LLM-as-judge evaluation execution
        
        Raises:
            ValueError: If invalid model, dataset, or metric configurations are provided
        
        Example:
            .. code:: python
            
                evaluator = LLMAsJudgeEvaluator(
                    base_model="llama-3-3-70b-instruct",
                evaluator_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                dataset="s3://my-bucket/my-dataset.jsonl",
                builtin_metrics=["Correctness", "Helpfulness"],  # Prefix optional
                s3_output_path="s3://my-bucket/output"
            )
                    evaluator_model="anthropic.claude-3-5-sonnet-20240620-v1:0",
                    dataset="s3://my-bucket/my-dataset.jsonl",
                    builtin_metrics=["Correctness", "Helpfulness"],
                    s3_output_path="s3://my-bucket/output"
                )
                execution = evaluator.evaluate()
                execution.wait()
        """
        from .pipeline_templates import LLMAJ_TEMPLATE, LLMAJ_TEMPLATE_BASE_MODEL_ONLY
        from .constants import EvalType
        
        # Get AWS execution context (role ARN, region, account ID)
        aws_context = self._get_aws_execution_context()
        
        # Resolve model artifacts
        artifacts = self._resolve_model_artifacts(aws_context['region'])
        
        # Get or infer model_package_group ARN (handles all cases internally)
        model_package_group_arn = self._get_model_package_group_arn()
        
        # Log resolved model information for debugging
        _logger.info(f"Resolved model info - base_model_name: {self._base_model_name}, base_model_arn: {self._base_model_arn}, source_model_package_arn: {self._source_model_package_arn}")
        
        # Build base template context
        template_context = self._get_base_template_context(
            role_arn=aws_context['role_arn'],
            region=aws_context['region'],
            account_id=aws_context['account_id'],
            model_package_group_arn=model_package_group_arn,
            resolved_model_artifact_arn=artifacts['resolved_model_artifact_arn']
        )
        
        # Add dataset URI
        template_context['dataset_uri'] = self.dataset
        
        # Generate execution name early so we can use it for S3 paths
        name = self.base_eval_name or f"llm-judge-eval"
        
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
        
        # Start execution (name already generated earlier)
        return self._start_execution(
            eval_type=EvalType.LLM_AS_JUDGE,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=aws_context['role_arn'],
            region=aws_context['region']
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
