"""SageMaker Model Evaluation Module.

This module provides comprehensive evaluation capabilities for SageMaker models:

Classes:
    - BaseEvaluator: Abstract base class for all evaluators
    - BenchMarkEvaluator: Standard benchmark evaluations
    - CustomScorerEvaluator: Custom scorer and preset metrics evaluations
    - LLMAsJudgeEvaluator: LLM-as-judge evaluations
    - EvaluationPipelineExecution: Pipeline-based evaluation execution implementation
    - PipelineExecutionStatus: Combined status with step details and failure reason
    - StepDetail: Individual pipeline step information
"""

from .base_evaluator import BaseEvaluator
from .benchmark_evaluator import (
    BenchMarkEvaluator,
    get_benchmark_properties,
    get_benchmarks,
)
from .custom_scorer_evaluator import (
    CustomScorerEvaluator,
    get_builtin_metrics,
)
from .execution import (
    EvaluationPipelineExecution,
    PipelineExecutionStatus,
    StepDetail,
)
from .llm_as_judge_evaluator import LLMAsJudgeEvaluator

__all__ = [
    # Evaluator classes
    "BaseEvaluator",
    "BenchMarkEvaluator", 
    "CustomScorerEvaluator",
    "LLMAsJudgeEvaluator",
    
    # Benchmark utility functions
    "get_benchmarks",
    "get_benchmark_properties",
    
    # Custom scorer utility functions
    "get_builtin_metrics",
    
    # Execution classes
    "EvaluationPipelineExecution",
    "PipelineExecutionStatus",
    "StepDetail",
]

__version__ = "1.0.0"
