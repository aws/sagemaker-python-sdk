"""Custom Scorer Evaluator for SageMaker Model Evaluation Module.

This module provides evaluation capabilities using custom scorer metrics,
supporting both built-in preset metrics and custom evaluator implementations
for flexible model evaluation workflows.
"""

import logging
from enum import Enum
from typing import Any, Optional, Type, Union

from pydantic import validator

from .base_evaluator import BaseEvaluator
from .constants import EvalType
from .execution import EvaluationPipelineExecution
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter, TelemetryParamType
from sagemaker.train.common_utils.telemetry_params import BASE_EVALUATOR_TELEMETRY_PARAMS
from sagemaker.core.telemetry.constants import Feature
from sagemaker.train.constants import get_sagemaker_hub_name

_logger = logging.getLogger(__name__)


class _BuiltInMetric(str, Enum):
    """Internal: Preset metrics for custom scorer evaluation.
    
    These metrics provide built-in evaluation capabilities for common use cases.
    
    Note:
        This is an internal class. Users should use ``get_builtin_metrics()`` instead.
    """
    PRIME_MATH = "prime_math"
    PRIME_CODE = "prime_code"



def get_builtin_metrics() -> Type[_BuiltInMetric]:
    """Get the built-in metrics enum for custom scorer evaluation.
    
    This utility function provides access to preset metrics for custom scorer evaluation.
    
    Returns:
        Type[_BuiltInMetric]: The built-in metric enum class
    
    Example:
        .. code:: python
        
            from sagemaker.train.evaluate import get_builtin_metrics
            
            BuiltInMetric = get_builtin_metrics()
            evaluator = CustomScorerEvaluator(
                evaluator=BuiltInMetric.PRIME_MATH,
                dataset=my_dataset,
                base_model="my-model",
                s3_output_path="s3://bucket/output",
                mlflow_resource_arn="arn:..."
            )
    """
    return _BuiltInMetric


class CustomScorerEvaluator(BaseEvaluator):
    """Custom scorer evaluation job for preset or custom evaluator metrics.
    
    This evaluator supports both preset metrics (via built-in metrics enum) and
    custom evaluator implementations for specialized evaluation needs.
    
    Attributes:
        evaluator (Union[str, Any]): Built-in metric enum value, Evaluator object, or Evaluator
            ARN string. Required. Use ``get_builtin_metrics()`` for available preset metrics.
        dataset (Any): Dataset for evaluation. Required. Accepts S3 URI, Dataset ARN, or DataSet object.
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
        
            from sagemaker.train.evaluate.custom_scorer_evaluator import (
                CustomScorerEvaluator,
                get_builtin_metrics
            )
            from sagemaker.ai_registry.evaluator import Evaluator
            
            # Using preset metric
            BuiltInMetric = get_builtin_metrics()
            evaluator = CustomScorerEvaluator(
                evaluator=BuiltInMetric.PRIME_MATH,
                dataset=my_dataset,
                base_model="my-model",
                s3_output_path="s3://bucket/output",
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
            )
            
            # Using custom evaluator
            my_evaluator = Evaluator.create(
                name="my-custom-evaluator",
                function_source="/path/to/evaluator.py",
                sub_type="AWS/Evaluator"
            )
            evaluator = CustomScorerEvaluator(
                evaluator=my_evaluator,
                dataset=my_dataset,
                base_model="my-model",
                s3_output_path="s3://bucket/output",
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
            )
            
            # Using evaluator ARN string
            evaluator = CustomScorerEvaluator(
                evaluator="arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/Evaluator/my-evaluator/1",
                dataset=my_dataset,
                base_model="my-model",
                s3_output_path="s3://bucket/output",
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
            )
            
            job = evaluator.evaluate()
    """
    
    evaluator: Union[str, Any]
    dataset: Any
    _hyperparameters: Optional[Any] = None
    
    # Template-required fields
    evaluate_base_model: bool = False

    def _get_eval_recipe_display_name_filter(self) -> str:
        """Prefer 'custom' or 'scorer' recipes for CustomScorerEvaluator."""
        return "custom"
    
    @validator('dataset', pre=True)
    def _resolve_dataset(cls, v):
        """Resolve dataset to string (S3 URI or ARN) and validate format.
        
        Uses BaseEvaluator's common validation logic to avoid code duplication.
        """
        return BaseEvaluator._validate_and_resolve_dataset(v)
    
    @validator('evaluator')
    def _validate_evaluator(cls, v):
        """Validate evaluator parameter is a built-in metric, Evaluator object, or ARN string"""
        # Check if it's a built-in metric enum
        if isinstance(v, _BuiltInMetric):
            return v
        
        # Check if it's an Evaluator object (has 'arn' attribute)
        if hasattr(v, 'arn'):
            _logger.info(f"Resolving Evaluator object to ARN: {v.arn}")
            return v.arn
        
        # Check if it's a string (should be an ARN)
        if isinstance(v, str):
            # Validate it looks like an ARN or is a valid built-in metric name
            if v.startswith('arn:'):
                return v
            # Try to match as built-in metric name
            try:
                return _BuiltInMetric(v)
            except ValueError:
                raise ValueError(
                    f"Invalid evaluator: '{v}'. Must be a built-in metric enum value, "
                    f"Evaluator object, or valid Evaluator ARN. "
                    f"Available built-in metrics: {', '.join(m.value for m in _BuiltInMetric)}"
                )
        
        raise ValueError(
            f"Invalid evaluator type: {type(v).__name__}. "
            f"Must be a built-in metric enum value, Evaluator object, or ARN string."
        )
    
    @property
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="CustomScorerEvaluator.hyperparameters")
    def hyperparameters(self):
        """Get evaluation hyperparameters as a FineTuningOptions object.
        
        This property provides access to evaluation hyperparameters with validation,
        type checking, and user-friendly information display. Hyperparameters are
        lazily loaded from the JumpStart Hub when first accessed.
        
        Returns:
            FineTuningOptions: Dynamic object with evaluation hyperparameters
        
        Raises:
            ValueError: If base model name is not available or if hyperparameters cannot be loaded
        
        Example:
            .. code:: python
            
                evaluator = CustomScorerEvaluator(...)
                
                # Access current values
                print(evaluator.hyperparameters.temperature)
                
                # Modify values (with validation)
                evaluator.hyperparameters.temperature = 0.5
                
                # Get as dictionary
                params = evaluator.hyperparameters.to_dict()
                
                # Display parameter information
                evaluator.hyperparameters.get_info()
                evaluator.hyperparameters.get_info('temperature')
        """
        if self._hyperparameters is None:
            from ..common import FineTuningOptions
            from ..common_utils.recipe_utils import _get_evaluation_override_params, _extract_eval_override_options
            
            # Get the hub content name from the base model
            hub_content_name = self._base_model_name
            if not hub_content_name:
                raise ValueError(
                    "Base model name not available. Cannot load hyperparameters. "
                    "Ensure base_model is properly configured. "
                    "The base_model parameter must be set to a valid model identifier (e.g., JumpStart model ID, "
                    "model package ARN, or model ARN) to enable hyperparameter configuration."
                )
            
            # Get region
            region = self.region
            
            # Fetch override parameters from hub (let exceptions propagate)
            _logger.info(f"Fetching evaluation override parameters for hyperparameters property")
            
            # Extract boto_session from sagemaker_core Session
            # HubContent.get() in recipe_utils expects boto3 session, not sagemaker_core Session
            boto_session = (self.sagemaker_session.boto_session 
                           if hasattr(self.sagemaker_session, 'boto_session') 
                           else self.sagemaker_session)
            
            override_params = _get_evaluation_override_params(
                hub_content_name=hub_content_name,
                hub_name=get_sagemaker_hub_name(),
                evaluation_type="DeterministicEvaluation",
                region=region,
                session=boto_session
            )
            
            # Extract full parameter specifications
            configurable_params = _extract_eval_override_options(override_params, return_full_spec=True)
            
            # Create FineTuningOptions object from full specifications
            self._hyperparameters = FineTuningOptions(configurable_params)
        
        return self._hyperparameters
    
    def _resolve_evaluator_config(self) -> dict:
        """Resolve evaluator configuration (ARN vs preset).
        
        Returns:
            dict: Dictionary with:
                - evaluator_arn (Optional[str]): Custom evaluator ARN or None
                - preset_reward_function (Optional[str]): Preset function name or None
        """
        evaluator_arn = None
        preset_reward_function = None
        
        if isinstance(self.evaluator, _BuiltInMetric):
            # Built-in metric enum - use as preset_reward_function
            preset_reward_function = self.evaluator.value
        elif isinstance(self.evaluator, str) and self.evaluator.startswith('arn:'):
            # Custom evaluator ARN
            evaluator_arn = self.evaluator
        elif isinstance(self.evaluator, str):
            # Built-in metric as string
            preset_reward_function = self.evaluator
        
        return {
            'evaluator_arn': evaluator_arn,
            'preset_reward_function': preset_reward_function
        }
    
    def _get_custom_scorer_template_additions(self, evaluator_config: dict) -> dict:
        """Get custom scorer specific template context additions.
        
        Args:
            evaluator_config: Dictionary with evaluator_arn and preset_reward_function
            
        Returns:
            dict: Custom scorer specific template context fields
        """
        from ..common_utils.recipe_utils import _is_nova_model

        # Get effective hyperparameters (recipe/overrides take precedence if provided)
        configured_params = self._get_effective_hyperparameters()
        _logger.info(f"Using configured hyperparameters: {configured_params}")
        
        # Determine if this is a Nova model
        is_nova = _is_nova_model(self._base_model_name)
        metric_key = 'metric' if is_nova else 'evaluation_metric'
        
        # Build custom scorer specific context
        custom_scorer_context = {
            'task': 'gen_qa',  # Fixed task for custom scorer
            'strategy': 'gen_qa',  # Fixed strategy for gen_qa task
            metric_key: "all",  # Use 'metric' for Nova, 'evaluation_metric' for OpenWeights
            'evaluate_base_model': self.evaluate_base_model,
            'evaluator_arn': evaluator_config['evaluator_arn'],
        }
        
        # Add lambda_type for Nova models
        if is_nova:
            custom_scorer_context['lambda_type'] = 'rft'
        
        # Add preset_reward_function if present
        if evaluator_config['preset_reward_function']:
            custom_scorer_context['preset_reward_function'] = evaluator_config['preset_reward_function']
        
        # Add all configured hyperparameters
        for key in configured_params.keys():
            custom_scorer_context[key] = configured_params[key]
        
        # Determine postprocessing and aggregation values
        # When evaluator_arn is provided, postprocessing must be enabled for Lambda execution
        if evaluator_config['evaluator_arn']:
            custom_scorer_context['postprocessing'] = 'True'
            if not custom_scorer_context.get('aggregation'):
                custom_scorer_context['aggregation'] = 'mean'
        
        return custom_scorer_context
    
    def _get_inference_params_from_hub(self, region: str) -> dict:
        """Fetch inference parameters from JumpStart Hub for the base model
        
        This method retrieves the evaluation recipe override parameters from the hub
        and extracts the inference parameters (max_new_tokens, temperature, top_k, top_p).
        
        Args:
            region: AWS region
            
        Returns:
            Dict containing inference parameters as strings. Returns fallback values if fetch fails.
        """
        from ..common_utils.recipe_utils import _get_evaluation_override_params, _extract_eval_override_options
        
        # Default fallback values
        fallback_params = {
            'max_new_tokens': '8192',
            'temperature': '0',
            'top_k': '-1',
            'top_p': '1.0'
        }
        
        try:
            # Get the hub content name from the base model
            hub_content_name = self._base_model_name
            if not hub_content_name:
                logger.warning("Base model name not available, using fallback inference parameters")
                return fallback_params
            
            # Get boto session for API calls
            session = self.sagemaker_session.boto_session if hasattr(self.sagemaker_session, 'boto_session') else None
            
            # Fetch override parameters from hub
            _logger.info(f"Fetching evaluation recipe override parameters from hub for model: {hub_content_name}")
            override_params = _get_evaluation_override_params(
                hub_content_name=hub_content_name,
                hub_name=get_sagemaker_hub_name(),
                evaluation_type="DeterministicEvaluation",
                region=region,
                session=session
            )
            
            # Extract evaluation override options
            inference_params = _extract_eval_override_options(override_params)
            
            _logger.info(f"Successfully fetched inference parameters from hub: {inference_params}")
            return inference_params
            
        except Exception as e:
            _logger.warning(
                f"Failed to fetch inference parameters from hub for model '{self._base_model_name}': {e}. "
                f"Using fallback values: {fallback_params}"
            )
            return fallback_params
    
    @_telemetry_emitter(
        feature=Feature.MODEL_CUSTOMIZATION,
        func_name="CustomScorerEvaluator.evaluate",
        telemetry_params=[
            ("evaluator", TelemetryParamType.ATTR_EXISTS),
        ] + BASE_EVALUATOR_TELEMETRY_PARAMS,
    )
    def evaluate(self) -> EvaluationPipelineExecution:
        """Create and start a custom scorer evaluation job.
        
        Supports multiple compute backends via the ``compute`` parameter set at
        construction time:
        - **Serverless** (default): Runs via SageMaker Pipelines.
        - **SMTJ**: Runs on user-managed instances via ModelTrainer.
        - **HyperPod**: Submits to a HyperPod cluster via the HyperPod CLI.
        
        Returns:
            EvaluationPipelineExecution: The created custom scorer evaluation execution
        
        Example:
            .. code:: python
            
                evaluator = CustomScorerEvaluator(
                    evaluator=BuiltInMetric.CODE_EXECUTIONS,
                    dataset=my_dataset,
                    base_model="my-model",
                    s3_output_path="s3://bucket/output",
                    mlflow_resource_arn="arn:..."
                )
                execution = evaluator.evaluate()
                execution.wait()
        """
        from sagemaker.core.training.configs import Compute, HyperPodCompute

        # Validate platform compatibility (HP checkpoints must eval on HP, SMTJ on SMTJ)
        from sagemaker.train.common_utils.finetune_utils import validate_eval_platform_compatibility
        model_info = self._get_resolved_model_info()
        model_path = getattr(model_info, 's3_model_path', None) if model_info else None
        validate_eval_platform_compatibility(model_path, self.compute)

        # Dispatch based on compute type
        if isinstance(self.compute, Compute) and not isinstance(self.compute, HyperPodCompute):
            return self._evaluate_serverful_smtj()
        elif isinstance(self.compute, HyperPodCompute):
            return self._evaluate_hyperpod()

        # Default: serverless compute via SageMaker Pipelines
        # S3 checkpoint paths are not supported on serverless — require SMTJ or HyperPod compute
        from sagemaker.train.common_utils.model_resolution import _ModelType
        info = self._get_resolved_model_info()
        if info and info.model_type == _ModelType.S3_CHECKPOINT:
            raise ValueError(
                "S3 checkpoint paths cannot be used with serverless evaluation. "
                "Please provide a 'compute' parameter (e.g., TrainingJobCompute or HyperPodCompute) "
                "to run evaluation on dedicated instances."
            )

        from .pipeline_templates import CUSTOM_SCORER_TEMPLATE, CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY
        
        # Get AWS execution context (role ARN, region, account ID)
        aws_context = self._get_aws_execution_context()
        
        # Resolve model artifacts
        artifacts = self._resolve_model_artifacts(aws_context['region'])
        
        # Get or infer model_package_group ARN (handles all cases internally)
        model_package_group_arn = self._get_model_package_group_arn()
        
        # Log resolved model information for debugging
        _logger.info(f"Resolved model info - base_model_name: {self._base_model_name}, base_model_arn: {self._base_model_arn}, source_model_package_arn: {self._source_model_package_arn}")
        
        # Resolve evaluator configuration
        evaluator_config = self._resolve_evaluator_config()
        
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
        
        # Add custom scorer specific template additions
        custom_scorer_additions = self._get_custom_scorer_template_additions(evaluator_config)
        template_context.update(custom_scorer_additions)
        
        # Add VPC and KMS configuration
        template_context = self._add_vpc_and_kms_to_context(template_context)
        
        # Select appropriate template
        template_str = self._select_template(
            CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY,
            CUSTOM_SCORER_TEMPLATE
        )
        
        # Render pipeline definition
        pipeline_definition = self._render_pipeline_definition(template_str, template_context)
        
        # Generate execution name
        name = self.base_eval_name or f"custom-scorer-eval"
        
        # Start execution
        return self._start_execution(
            eval_type=EvalType.CUSTOM_SCORER,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=aws_context['role_arn'],
            region=aws_context['region']
        )
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="CustomScorerEvaluator.get_all")
    def get_all(cls, session: Optional[Any] = None, region: Optional[str] = None):
        """Get all custom scorer evaluation executions.
        
        Uses ``EvaluationPipelineExecution.get_all()`` to retrieve all custom scorer
        evaluation executions as an iterator.
        
        Args:
            session (Optional[Any]): Optional boto3 session. If not provided, will be inferred.
            region (Optional[str]): Optional AWS region. If not provided, will be inferred.
        
        Yields:
            EvaluationPipelineExecution: Custom scorer evaluation execution instances
        
        Example:
            .. code:: python
            
                # Get all custom scorer evaluations as iterator
                evaluations = CustomScorerEvaluator.get_all()
                all_executions = list(evaluations)
                
                # Or iterate directly
                for execution in CustomScorerEvaluator.get_all():
                    print(f"{execution.name}: {execution.status.overall_status}")
                
                # With specific session/region
                evaluations = CustomScorerEvaluator.get_all(session=my_session, region='us-west-2')
                all_executions = list(evaluations)
        """
        # Use EvaluationPipelineExecution.get_all() with CUSTOM_SCORER eval_type
        # This returns a generator, so we yield from it
        yield from EvaluationPipelineExecution.get_all(
            eval_type=EvalType.CUSTOM_SCORER,
            session=session,
            region=region
        )

    def _evaluate_serverful_smtj(self):
        """Execute custom scorer evaluation on SMTJ compute via ModelTrainer.

        Fetches the evaluation recipe template from SageMaker Hub (filtered by
        Type=Evaluation), injects custom scorer-specific parameters, and launches
        via ModelTrainer.from_recipe(). Follows the same Hub lookup pattern as
        BenchMarkEvaluator._evaluate_serverful_smtj().
        """
        from sagemaker.train.utils import _get_unique_name

        # --- Validate platform compatibility ---
        from sagemaker.train.common_utils.model_resolution import _ModelType, _detect_checkpoint_platform, _CheckpointPlatform
        info = self._get_resolved_model_info()
        if info and info.model_type == _ModelType.S3_CHECKPOINT and info.s3_model_path:
            checkpoint_platform = _detect_checkpoint_platform(info.s3_model_path)
            if checkpoint_platform == _CheckpointPlatform.HYPERPOD:
                raise ValueError(
                    f"HyperPod-trained checkpoints cannot be evaluated on SMTJ compute. "
                    f"The checkpoint at '{info.s3_model_path}' was trained on HyperPod. "
                    f"Please use HyperPodCompute for evaluation instead:\n\n"
                    f"    compute=HyperPodCompute(\n"
                    f"        cluster_name='your-cluster',\n"
                    f"        namespace='kubeflow',\n"
                    f"        instance_type='ml.p5.48xlarge',\n"
                    f"        node_count=1,\n"
                    f"    )"
                )

        # --- Common setup ---
        sagemaker_session, role, region = self._get_smtj_session_and_role()

        # --- Find SMTJ evaluation recipe ---
        smtj_eval_recipes = self._get_smtj_eval_recipes(sagemaker_session, region)

        # Prefer "custom scorer" recipes if available, otherwise fall back to first
        custom_scorer_recipes = [
            r for r in smtj_eval_recipes
            if "custom" in r.get("DisplayName", "").lower()
            or "scorer" in r.get("DisplayName", "").lower()
        ]
        recipe_metadata = custom_scorer_recipes[0] if custom_scorer_recipes else smtj_eval_recipes[0]

        # Resolve training image
        training_image = self.training_image
        if not training_image:
            training_image = recipe_metadata.get("SmtjImageUri")

        # --- Download and load recipe ---
        recipe_dict, recipe_tmp_path = self._download_and_load_recipe(
            recipe_metadata["SmtjRecipeTemplateS3Uri"], sagemaker_session
        )

        # --- Fetch the Hub-declared overridable field set + defaults ---
        # SmtjOverrideParamsS3Uri declares the fields this recipe accepts and
        # their defaults/types, so the overridable set is Hub-driven rather than
        # hardcoded (mirrors the Nova Forge SDK RecipeBuilder).
        override_spec = self._download_eval_override_spec(recipe_metadata, sagemaker_session)

        # --- Build SDK-derived semantic values ---
        base_job_name = self.base_eval_name or "custom-eval"
        # The eval container requires non-empty experiment and run names whenever
        # an MLflow tracking URI is set (OSS); Nova tolerates empty names but
        # still benefits from a meaningful default. Default both to base_job_name
        # for a consistent MLflow experience. See
        # BaseEvaluator._resolve_mlflow_tracking_fields.
        mlflow_tracking_uri, mlflow_experiment_name, mlflow_run_name = (
            self._resolve_mlflow_tracking_fields(base_job_name)
        )
        # Infra field names span recipe families: Nova uses 'output_s3_path',
        # the OSS eval recipe uses 'output_path' and also carries 'base_model_name'
        # and 'instance_count' (run.replicas).
        semantic_values = {
            "name": _get_unique_name(base_job_name),
            "base_model_name": self._base_model_name or "",
            "instance_count": self.compute.instance_count,
            "kms_key_id": self.kms_key_id or "",
            "mlflow_tracking_uri": mlflow_tracking_uri,
            "mlflow_experiment_name": mlflow_experiment_name,
            "mlflow_run_name": mlflow_run_name,
        }

        evaluator_config = None
        if self.evaluator:
            evaluator_config = self._resolve_evaluator_config()
            # Custom-scorer semantic fields: task/strategy/metric (or
            # evaluation_metric for OpenWeights), evaluator_arn, lambda_type,
            # preset_reward_function, postprocessing, aggregation, + hyperparams.
            semantic_values.update(
                self._get_custom_scorer_template_additions(evaluator_config)
            )

        # --- Resolve model path (fine-tuned checkpoint or OSS base weights) ---
        # For OSS base models the container loads weights via model_name_or_path,
        # so it must be resolved from Hub when there is no fine-tuned checkpoint.
        # OSS artifacts are delivered via a dedicated "model" input channel so the
        # container's checkpoints/hf_merged resolution runs against a local mount
        # (reproducing the serverless experience); Nova keeps the raw S3 path.
        model_path, model_channel = self._resolve_eval_model_input(
            sagemaker_session, region
        )
        if model_path:
            semantic_values["model_name_or_path"] = model_path

        # --- Dataset as input channel ---
        input_data_config = []
        if self.dataset:
            from sagemaker.core.training.configs import InputData
            from sagemaker.core.shapes import S3DataSource
            from ..common_utils.recipe_utils import _is_nova_model

            dataset_uri = str(self.dataset)
            input_data_config.append(
                InputData(
                    channel_name="train",
                    data_source=S3DataSource(
                        s3_uri=dataset_uri,
                        s3_data_type="S3Prefix",
                        s3_data_distribution_type="FullyReplicated",
                    ),
                )
            )
            # The dataset is always delivered via the mounted "train" channel.
            # Nova's container resolves the S3 dataset path itself, so it keeps
            # the S3 URI. The OSS container instead validates the dataset path on
            # the local filesystem (same as the model path), so point it at the
            # mounted channel directory; passing the S3 URI makes it report
            # "does not exist". This matches the proven serverless OSS recipe,
            # which sets data to '/opt/ml/input/data/train'.
            if _is_nova_model(self._base_model_name):
                semantic_values["data_s3_path"] = dataset_uri
            else:
                semantic_values["data_s3_path"] = "/opt/ml/input/data/train"

        # --- OSS base model weights as input channel (Nova adds none) ---
        if model_channel:
            input_data_config.append(model_channel)

        input_data_config = input_data_config or None

        if self.s3_output_path:
            semantic_values["output_s3_path"] = self.s3_output_path
            semantic_values["output_path"] = self.s3_output_path

        # --- Merge: spec defaults < semantic values < user overrides ---
        value_map = self._build_eval_value_map(
            override_spec, semantic_values=semantic_values, user_overrides=self.overrides
        )

        # --- Inject by leaf-key name / placeholder across any recipe structure ---
        self._apply_eval_recipe_values(recipe_dict, value_map)

        # --- Custom-scorer Lambda wiring (depends on section presence) ---
        if evaluator_config:
            if evaluator_config.get('evaluator_arn'):
                recipe_dict.setdefault("run", {})["eval_lambda_arn"] = evaluator_config['evaluator_arn']
            elif evaluator_config.get('preset_reward_function'):
                # Using a preset reward function, not a custom Lambda. The container
                # schema expects lambda_arn to be present but empty.
                if "processor" in recipe_dict:
                    recipe_dict["processor"]["lambda_arn"] = ""
                recipe_dict.get("run", {}).pop("eval_lambda_arn", None)

            # Remove fields from the run section that the container doesn't accept
            # there (they belong in the processor section only).
            for key in ("preset_reward_function", "lambda_type"):
                recipe_dict.get("run", {}).pop(key, None)

        # Nova recipes carry infra fields in a `run` section; ensure they are
        # present even if the template omitted a key. Scoped to Nova: OSS recipes
        # use different infra field names (output_path, output.mlflow_*) already
        # covered by the spec/injection, so adding Nova-style keys here would
        # pollute the OSS recipe's run section.
        from ..common_utils.recipe_utils import _is_nova_model
        if "run" in recipe_dict and _is_nova_model(self._base_model_name):
            run = recipe_dict["run"]
            run.setdefault("name", semantic_values["name"])
            if self.dataset:
                run.setdefault("data_s3_path", str(self.dataset))
            if self.s3_output_path:
                run.setdefault("output_s3_path", self.s3_output_path)
            if model_path:
                run.setdefault("model_name_or_path", model_path)

        # --- Common: resolve any remaining inference placeholders ---
        self._resolve_inference_placeholders(recipe_dict)

        # Blank any remaining optional infra placeholders so the recipe has no
        # unresolved {{...}} tokens before submission.
        blanked = self._blank_unresolved_placeholders(recipe_dict)
        if blanked:
            _logger.warning(
                f"Blanked unresolved eval recipe placeholders (not declared in the "
                f"override spec or set by the SDK): {blanked}"
            )

        # --- Common: write recipe and submit ---
        return self._write_and_submit_smtj_recipe(
            recipe_dict, recipe_tmp_path, training_image, sagemaker_session, role, base_job_name,
            input_data_config=input_data_config,
        )
    def _evaluate_hyperpod(self):
        """Execute custom scorer evaluation on HyperPod cluster.

        Builds evaluator-specific override parameters (processor config, dataset)
        and delegates to BaseEvaluator._submit_hyperpod_eval_job().
        """
        override_parameters = {}

        if hasattr(self, 'evaluator') and self.evaluator:
            evaluator_config = self._resolve_evaluator_config()
            if evaluator_config.get('evaluator_arn'):
                override_parameters["recipes.processor.lambda_arn"] = evaluator_config['evaluator_arn']
            elif evaluator_config.get('preset_reward_function'):
                override_parameters["recipes.processor.preset_reward_function"] = evaluator_config['preset_reward_function']
        if hasattr(self, 'dataset') and self.dataset:
            override_parameters["recipes.run.data_s3_path"] = str(self.dataset)

        # User-provided overrides (e.g. inference params)
        if self.overrides:
            override_parameters.update(self.overrides)

        return self._submit_hyperpod_eval_job(
            override_parameters=override_parameters,
            base_job_name="custom-eval",
        )
