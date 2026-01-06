"""Benchmark evaluator module for SageMaker Model Evaluation.

This module provides benchmark evaluation capabilities for SageMaker models, supporting
various standard benchmarks like MMLU, BBH, MATH, and others. It handles benchmark
configuration, validation, and execution of evaluation pipelines.
"""

from __future__ import absolute_import

import logging
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Type, Union

from pydantic import BaseModel, Field, validator

from sagemaker.core.resources import ModelPackageGroup

from .base_evaluator import BaseEvaluator
from .constants import EvalType
from .execution import EvaluationPipelineExecution
from sagemaker.core.telemetry.telemetry_logging import _telemetry_emitter
from sagemaker.core.telemetry.constants import Feature

_logger = logging.getLogger(__name__)


# Internal enums and classes - not meant for direct user access
class _Benchmark(str, Enum):
    """Internal benchmark types for model evaluation"""
    MMLU = "mmlu"
    MMLU_PRO = "mmlu_pro"
    BBH = "bbh"
    GPQA = "gpqa"
    MATH = "math"
    STRONG_REJECT = "strong_reject"
    IFEVAL = "ifeval"
    MMMU = "mmmu"
    LLM_JUDGE = "llm_judge"


# Internal benchmark configuration mapping - using plain dictionaries
_BENCHMARK_CONFIG: Dict[_Benchmark, Dict[str, Any]] = {
    _Benchmark.MMLU: {
        "modality": "Text",
        "description": "Multi-task Language Understanding – Tests knowledge across 57 subjects.",
        "metrics": ["accuracy"],
        "strategy": "zs_cot",
        "subtask_available": True,
        "subtasks": [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_medicine",
            "college_physics", "computer_security", "conceptual_physics",
            "econometrics", "electrical_engineering", "elementary_mathematics",
            "formal_logic", "global_facts", "high_school_biology",
            "high_school_chemistry", "high_school_computer_science",
            "high_school_european_history", "high_school_geography",
            "high_school_government_and_politics", "high_school_macroeconomics",
            "high_school_mathematics", "high_school_microeconomics",
            "high_school_physics", "high_school_psychology",
            "high_school_statistics", "high_school_us_history",
            "high_school_world_history", "human_aging", "human_sexuality",
            "international_law", "jurisprudence", "logical_fallacies",
            "machine_learning", "management", "marketing", "medical_genetics",
            "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
            "philosophy", "prehistory", "professional_accounting",
            "professional_law", "professional_medicine", "professional_psychology",
            "public_relations", "security_studies", "sociology",
            "us_foreign_policy", "virology", "world_religions"
        ]
    },
    _Benchmark.MMLU_PRO: {
        "modality": "Text",
        "description": "MMLU – Professional Subset – Focuses on professional domains such as law, medicine, accounting, and engineering.",
        "metrics": ["accuracy"],
        "strategy": "zs_cot",
        "subtask_available": False,
        "subtasks": None
    },
    _Benchmark.BBH: {
        "modality": "Text",
        "description": "Advanced Reasoning Tasks – A collection of challenging problems that test higher-level cognitive and problem-solving skills.",
        "metrics": ["accuracy"],
        "strategy": "fs_cot",
        "subtask_available": True,
        "subtasks": [
            "boolean_expressions", "causal_judgement", "date_understanding",
            "disambiguation_qa", "dyck_languages", "formal_fallacies",
            "geometric_shapes", "hyperbaton", "logical_deduction_five_objects",
            "logical_deduction_seven_objects", "logical_deduction_three_objects",
            "movie_recommendation", "multistep_arithmetic_two", "navigate",
            "object_counting", "penguins_in_a_table",
            "reasoning_about_colored_objects", "ruin_names",
            "salient_translation_error_detection", "snarks",
            "sports_understanding", "temporal_sequences",
            "tracking_shuffled_objects_five_objects",
            "tracking_shuffled_objects_seven_objects",
            "tracking_shuffled_objects_three_objects", "web_of_lies",
            "word_sorting"
        ]
    },
    _Benchmark.GPQA: {
        "modality": "Text",
        "description": "General Physics Question Answering – Assesses comprehension of physics concepts and related problem-solving abilities.",
        "metrics": ["accuracy"],
        "strategy": "zs_cot",
        "subtask_available": False,
        "subtasks": None
    },
    _Benchmark.MATH: {
        "modality": "Text",
        "description": "Mathematical Problem Solving – Measures mathematical reasoning across topics including algebra, calculus, and word problems.",
        "metrics": ["exact_match"],
        "strategy": "zs_cot",
        "subtask_available": True,
        "subtasks": [
            "algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra",
            "precalculus"
        ]
    },
    _Benchmark.STRONG_REJECT: {
        "modality": "Text",
        "description": "Quality-Control Task – Tests the model's ability to detect and reject inappropriate, harmful, or incorrect content.",
        "metrics": ["deflection"],
        "strategy": "zs",
        "subtask_available": True,
        "subtasks": None  # Documentation doesn't specify subtasks for strong_reject
    },
    _Benchmark.IFEVAL: {
        "modality": "Text",
        "description": "Instruction-Following Evaluation – Gauges how accurately a model follows given instructions and completes tasks to specification.",
        "metrics": ["accuracy"],
        "strategy": "zs",
        "subtask_available": False,
        "subtasks": None
    },
    _Benchmark.MMMU: {
        "modality": "Multi-Modal",
        "description": "Massive Multidiscipline Multimodal Understanding (MMMU) – College-level benchmark comprising multiple-choice and open-ended questions from 30 disciplines.",
        "metrics": ["accuracy"],
        "strategy": "zs_cot",
        "subtask_available": True,
        "subtasks": [
            "Accounting", "Agriculture", "Architecture_and_Engineering",
            "Art", "Art_Theory", "Basic_Medical_Science", "Biology",
            "Chemistry", "Clinical_Medicine", "Computer_Science", "Design",
            "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
            "Energy_and_Power", "Finance", "Geography", "History",
            "Literature", "Manage", "Marketing", "Materials", "Math",
            "Mechanical_Engineering", "Music", "Pharmacy", "Physics",
            "Psychology", "Public_Health", "Sociology"
        ]
    },
    _Benchmark.LLM_JUDGE: {
        "modality": "Text",
        "description": "LLM-as-a-Judge - Uses a user-selected judge model to judge a set of customer-provided inference responses.",
        "metrics": ["all"],
        "strategy": "judge",
        "subtask_available": False,
        "subtasks": None
    },
}


# Public utility methods
def get_benchmarks() -> Type[_Benchmark]:
    """Get the Benchmark enum for selecting available benchmarks.
    
    This utility method provides access to the internal Benchmark enum,
    allowing users to reference available benchmarks without directly
    accessing internal implementation details.
    
    Returns:
        Type[_Benchmark]: The Benchmark enum class containing all available benchmarks.
        
    Example:
    
        .. code:: python
        
            Benchmark = get_benchmarks()
            evaluator = BenchMarkEvaluator(
                benchmark=Benchmark.MMLU,
                sagemaker_session=session,
                s3_output_path="s3://bucket/output"
            )
    
    Note:
        In the future, this will be extended to dynamically generate the
        enum from a backend API call to fetch the latest available benchmarks.
    """
    return _Benchmark


def get_benchmark_properties(benchmark: _Benchmark) -> Dict[str, Any]:
    """Get properties for a specific benchmark.
    
    This utility method returns the properties associated with a given benchmark
    as a dictionary, including information about modality, metrics, strategy,
    and available subtasks.
    
    Args:
        benchmark (_Benchmark): The benchmark to get properties for (from ``get_benchmarks()``).
    
    Returns:
        Dict[str, Any]: Dictionary containing benchmark properties with keys:
            
            - ``modality`` (str): The modality type (e.g., "Text", "Multi-Modal")
            - ``description`` (str): Description of the benchmark
            - ``metrics`` (list[str]): List of supported metrics
            - ``strategy`` (str): The evaluation strategy used
            - ``subtask_available`` (bool): Whether subtasks are supported
            - ``subtasks`` (Optional[list[str]]): List of available subtasks, if applicable
    
    Raises:
        ValueError: If the provided benchmark is not found in the configuration.
        
    Example:
    
        .. code:: python
        
            Benchmark = get_benchmarks()
            props = get_benchmark_properties(Benchmark.MMLU)
            print(props['description'])
            # 'Multi-task Language Understanding – Tests knowledge across 57 subjects.'
            print(props['subtasks'][:3])
            # ['abstract_algebra', 'anatomy', 'astronomy']
    
    Note:
        In the future, this will be extended to dynamically fetch benchmark
        properties from a backend API call instead of using the internal static configuration.
    """
    config = _BENCHMARK_CONFIG.get(benchmark)
    if config is None:
        raise ValueError(
            f"Benchmark '{benchmark.value}' not found in configuration. "
            f"Available benchmarks: {', '.join(b.value for b in _BENCHMARK_CONFIG.keys())}"
        )
    
    # Return a copy of the configuration dictionary
    return config.copy()


class BenchMarkEvaluator(BaseEvaluator):
    """Benchmark evaluator for standard model evaluation tasks.
    
    This evaluator accepts a benchmark enum and automatically deduces the appropriate
    metrics, strategy, and subtask availability based on the benchmark configuration.
    Supports various standard benchmarks like MMLU, BBH, MATH, MMMU, and others.
    
    Attributes:
        benchmark (_Benchmark): Benchmark type from the Benchmark enum obtained via ``get_benchmarks()``.
            Required. Use get_benchmarks() to access available benchmark types.
        subtasks (Optional[Union[str, list[str]]]): Benchmark subtask(s) to evaluate. Defaults to
            'ALL' for benchmarks that support subtasks. Can be a single subtask string, a list of
            subtasks, or 'ALL' to run all subtasks. For benchmarks without subtask support, must
            be None.
        mlflow_resource_arn (Optional[str]): ARN of the MLflow tracking server for experiment tracking.
            Optional. If not provided, the system will attempt to resolve it using the default
            MLflow app experience (checks domain match, account default, or creates a new app).
            Format: arn:aws:sagemaker:region:account:mlflow-tracking-server/name
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
        
            # Get available benchmarks
            Benchmark = get_benchmarks()
            
            # Create evaluator with benchmark and subtasks
            evaluator = BenchMarkEvaluator(
                benchmark=Benchmark.MMLU,
                subtasks=["abstract_algebra", "anatomy", "astronomy"],
                model="llama3-2-1b-instruct",
                s3_output_path="s3://bucket/outputs/",
                mlflow_resource_arn="arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/my-server"
            )
            
            # Run evaluation with configured subtasks
            execution = evaluator.evaluate()
            execution.wait()
            
            # Or override subtasks at evaluation time
            execution = evaluator.evaluate(subtask="abstract_algebra")
    """
    
    benchmark: _Benchmark
    subtasks: Optional[Union[str, List[str]]] = None
    evaluate_base_model: bool = False
    _hyperparameters: Optional[Any] = None

    
    @validator('benchmark')
    def _validate_benchmark_model_compatibility(cls, v, values):
        """Validate that benchmark is compatible with model type (Nova vs non-Nova)"""
        from ..common_utils.recipe_utils import _is_nova_model
        
        # Get resolved model info if available
        resolved_info = values.get('_resolved_model_info')
        if resolved_info and resolved_info.base_model_name:
            base_model_name = resolved_info.base_model_name
            is_nova = _is_nova_model(base_model_name)
            benchmark_value = v.value
            
            # mmmu is only allowed for Nova models
            if benchmark_value == "mmmu" and not is_nova:
                raise ValueError(
                    f"Benchmark 'mmmu' is only supported for Nova models. "
                    f"The current model '{base_model_name}' is not a Nova model."
                )
            
            # llm_judge is not allowed for Nova models
            if benchmark_value == "llm_judge" and is_nova:
                raise ValueError(
                    f"Benchmark 'llm_judge' is not supported for Nova models. "
                    f"The current model '{base_model_name}' is a Nova model."
                )
        
        return v
    
    @validator('subtasks', always=True)
    def _validate_subtasks(cls, v, values):
        """Validate that subtasks is provided when required and in correct format"""
        if 'benchmark' in values:
            benchmark = values['benchmark']
            config = _BENCHMARK_CONFIG.get(benchmark)
            
            if config and config.get("subtask_available"):
                # Default to "ALL" if not provided for benchmarks that support subtasks
                if v is None:
                    return "ALL"
                
                # Validate format
                if isinstance(v, list):
                    if len(v) == 0:
                        raise ValueError(
                            f"Subtask list cannot be empty for benchmark '{benchmark.value}'. "
                            f"Provide at least one subtask or use 'ALL'."
                        )

                    # Validate each subtask in the list
                    for subtask in v:
                        if not isinstance(subtask, str):
                            raise ValueError(
                                f"All subtasks in the list must be strings. "
                                f"Found {type(subtask).__name__}: {subtask}"
                            )

                        # Validate against available subtasks if defined
                        if config.get("subtasks") and subtask not in config["subtasks"]:
                            raise ValueError(
                                f"Invalid subtask '{subtask}' for benchmark '{benchmark.value}'. "
                                f"Available subtasks: {', '.join(config['subtasks'])}"
                            )
                
                elif isinstance(v, str):
                    # Skip validation for "ALL" keyword
                    if v.upper() != "ALL":
                        # Validate single subtask against available subtasks if defined
                        if config.get("subtasks") and v not in config["subtasks"]:
                            raise ValueError(
                                f"Invalid subtask '{v}' for benchmark '{benchmark.value}'. "
                                f"Available subtasks: {', '.join(config['subtasks'])}"
                            )
                else:
                    raise ValueError(
                        f"Subtask must be a string, a list of strings, or 'ALL'. "
                        f"Got {type(v).__name__}"
                    )
            
            if config and not config.get("subtask_available") and v is not None:
                raise ValueError(
                    f"Subtask is not supported for benchmark '{benchmark.value}'. "
                    f"Please set subtasks to None."
                )
        
        return v
    
    @property
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BenchMarkEvaluator.hyperparameters")
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
            
                evaluator = BenchMarkEvaluator(...)
                
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
            from ..common_utils.recipe_utils import _get_evaluation_override_params, _extract_eval_override_options, _is_nova_model
            
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
            # region = (self.sagemaker_session.boto_region_name 
            #          if hasattr(self.sagemaker_session, 'boto_region_name') 
            #          else 'us-west-2')
            region = self.region
            
            # Determine evaluation type based on model and task
            evaluation_type = "DeterministicEvaluation"  # Default for non-Nova models
            if _is_nova_model(hub_content_name):
                # For Nova models, evaluation type depends on the task
                task = self.benchmark.value
                if task == "mmmu":
                    evaluation_type = "DeterministicMultiModalBenchmark"
                else:
                    evaluation_type = "DeterministicTextBenchmark"
            
            # Fetch override parameters from hub (let exceptions propagate)
            _logger.info(f"Fetching evaluation override parameters for hyperparameters property")
            
            # Extract boto_session from sagemaker_core Session
            # HubContent.get() in recipe_utils expects boto3 session, not sagemaker_core Session
            boto_session = (self.sagemaker_session.boto_session 
                           if hasattr(self.sagemaker_session, 'boto_session') 
                           else self.sagemaker_session)
            
            override_params = _get_evaluation_override_params(
                hub_content_name=hub_content_name,
                hub_name="SageMakerPublicHub",
                evaluation_type=evaluation_type,
                region=region,
                session=boto_session
            )
            
            # Extract full parameter specifications
            configurable_params = _extract_eval_override_options(override_params, return_full_spec=True)
            
            # Create FineTuningOptions object from full specifications
            self._hyperparameters = FineTuningOptions(configurable_params)
        
        return self._hyperparameters
    
    def _resolve_subtask_for_evaluation(self, subtask: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
        """Resolve and validate subtask for evaluation.
        
        Args:
            subtask: Subtask parameter from evaluate() call
            
        Returns:
            Optional[Union[str, List[str]]]: Resolved subtask (uses constructor value if not provided)
            
        Raises:
            ValueError: If subtask is invalid for the benchmark
        """
        # Use provided subtask or fall back to constructor subtasks
        eval_subtask = subtask if subtask is not None else self.subtasks

        if eval_subtask is None or (isinstance(eval_subtask, str) and eval_subtask.upper() == "ALL"):
            #TODO : Check All Vs None subtask for evaluation
            return None

        # Validate the subtask
        config = _BENCHMARK_CONFIG.get(self.benchmark)
        if config and config.get("subtask_available"):
            if isinstance(eval_subtask, str):
                if eval_subtask.upper() != "ALL" and config.get("subtasks") and eval_subtask not in config["subtasks"]:
                    raise ValueError(
                        f"Invalid subtask '{eval_subtask}' for benchmark '{self.benchmark.value}'. "
                        f"Available subtasks: {', '.join(config['subtasks'])}"
                    )
            elif isinstance(eval_subtask, list):
                if len(eval_subtask) == 0:
                    raise ValueError(
                        f"Subtask list cannot be empty for benchmark '{self.benchmark.value}'. "
                        f"Provide at least one subtask or use 'ALL'."
                    )
                # Validate each subtask in the list
                for st in eval_subtask:
                    if config.get("subtasks") and st not in config["subtasks"]:
                        raise ValueError(
                            f"Invalid subtask '{st}' for benchmark '{self.benchmark.value}'. "
                            f"Available subtasks: {', '.join(config['subtasks'])}"
                        )

        
        return eval_subtask
    
    def _get_benchmark_template_additions(self, eval_subtask: Optional[Union[str, List[str]]], 
                                         config: Dict[str, Any]) -> dict:
        """Get benchmark-specific template context additions.
        
        Args:
            eval_subtask: Resolved subtask value
            config: Benchmark configuration dictionary
            
        Returns:
            dict: Benchmark-specific template context fields
        """
        from ..common_utils.recipe_utils import _is_nova_model
        
        # Get configured hyperparameters (triggers lazy load from hub with Nova model logic)
        configured_params = self.hyperparameters.to_dict()
        _logger.info(f"Using configured hyperparameters: {configured_params}")
        
        # Determine if this is a Nova model
        is_nova = _is_nova_model(self._base_model_name)
        metric_key = 'metric' if is_nova else 'evaluation_metric'
        
        # Build benchmark-specific context
        benchmark_context = {
            'task': self.benchmark.value,
            'strategy': config["strategy"],
            metric_key: config["metrics"][0] if config.get("metrics") else 'accuracy',
            'evaluate_base_model': self.evaluate_base_model,
        }
        
        if isinstance(eval_subtask, str):
            benchmark_context['subtask'] = eval_subtask
        elif isinstance(eval_subtask, list):
            # Convert list to comma-separated string
            benchmark_context['subtask'] = ','.join(eval_subtask)

        # Add all configured hyperparameters
        for key in configured_params.keys():
            benchmark_context[key] = configured_params[key]
        
        return benchmark_context
    
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BenchMarkEvaluator.evaluate")
    def evaluate(self, subtask: Optional[Union[str, List[str]]] = None) -> EvaluationPipelineExecution:
        """Create and start a benchmark evaluation job.
        
        Args:
            subtask (Optional[Union[str, list[str]]]): Optional subtask(s) to evaluate. If not provided, 
                uses the subtasks from constructor. Can be a single subtask string, a list of 
                subtasks, or 'ALL' to run all subtasks.
        
        Returns:
            EvaluationPipelineExecution: The created benchmark evaluation execution.
            
        Example:
        
            .. code:: python
            
                Benchmark = get_benchmarks()
                evaluator = BenchMarkEvaluator(
                    benchmark=Benchmark.MMLU,
                    subtasks="ALL",
                    model="llama3-2-1b-instruct",
                    s3_output_path="s3://bucket/outputs/"
                )
                
                # Evaluate single subtask
                execution = evaluator.evaluate(subtask="abstract_algebra")
                
                # Evaluate multiple subtasks
                execution = evaluator.evaluate(subtask=["abstract_algebra", "anatomy"])
                
                # Evaluate all subtasks (uses constructor default)
                execution = evaluator.evaluate()
        """
        from .pipeline_templates import DETERMINISTIC_TEMPLATE, DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY
        
        # Resolve and validate subtask
        eval_subtask = self._resolve_subtask_for_evaluation(subtask)
        
        # Get benchmark configuration
        config = _BENCHMARK_CONFIG.get(self.benchmark)
        
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

        
        # Add benchmark-specific template additions
        benchmark_additions = self._get_benchmark_template_additions(eval_subtask, config)
        template_context.update(benchmark_additions)
        
        # Add VPC and KMS configuration
        template_context = self._add_vpc_and_kms_to_context(template_context)
        
        # Select appropriate template
        template_str = self._select_template(
            DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY,
            DETERMINISTIC_TEMPLATE
        )
        
        # Render pipeline definition
        pipeline_definition = self._render_pipeline_definition(template_str, template_context)
        
        # Generate execution name
        name = self.base_eval_name or f"benchmark-eval-{self.benchmark.value}"
        
        # Start execution
        return self._start_execution(
            eval_type=EvalType.BENCHMARK,
            name=name,
            pipeline_definition=pipeline_definition,
            role_arn=aws_context['role_arn'],
            region=aws_context['region']
        )
    
    @classmethod
    @_telemetry_emitter(feature=Feature.MODEL_CUSTOMIZATION, func_name="BenchMarkEvaluator.get_all")
    def get_all(
        cls,
        session: Optional[Any] = None,
        region: Optional[str] = None
    ) -> Iterator[EvaluationPipelineExecution]:
        """Get all benchmark evaluation executions.
        
        Uses ``EvaluationPipelineExecution.get_all()`` to retrieve all benchmark
        evaluation executions as an iterator.
        
        Args:
            session (Optional[Any]): Optional boto3 session. If not provided, will be inferred.
            region (Optional[str]): Optional AWS region. If not provided, will be inferred.
        
        Yields:
            EvaluationPipelineExecution: Benchmark evaluation execution instances.
            
        Example:
        
            .. code:: python
            
                # Get all benchmark evaluations as iterator
                eval_iter = BenchMarkEvaluator.get_all()
                all_executions = list(eval_iter)
                
                # Or iterate directly
                for execution in BenchMarkEvaluator.get_all():
                    print(f"{execution.name}: {execution.status.overall_status}")
                
                # With specific session/region
                eval_iter = BenchMarkEvaluator.get_all(session=my_session, region='us-west-2')
                all_executions = list(eval_iter)
        """
        # Use EvaluationPipelineExecution.get_all() with BENCHMARK eval_type
        # This returns a generator, so we yield from it
        yield from EvaluationPipelineExecution.get_all(
            eval_type=EvalType.BENCHMARK,
            session=session,
            region=region
        )
