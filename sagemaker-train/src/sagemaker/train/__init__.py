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
"""SageMaker Python SDK Train Module."""
from __future__ import absolute_import

# Lazy imports to avoid circular dependencies
# Session and get_execution_role are available from sagemaker.core.helper.session_helper
# Import them directly from there if needed, or use lazy import pattern

def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "Session":
        from sagemaker.core.helper.session_helper import Session
        return Session
    elif name == "get_execution_role":
        from sagemaker.core.helper.session_helper import get_execution_role
        return get_execution_role
    elif name == "ModelTrainer":
        from sagemaker.train.model_trainer import ModelTrainer
        return ModelTrainer
    elif name == "SFTTrainer":
        from sagemaker.train.sft_trainer import SFTTrainer
        return SFTTrainer
    elif name == "DPOTrainer":
        from sagemaker.train.dpo_trainer import DPOTrainer
        return DPOTrainer
    elif name == "RLVRTrainer":
        from sagemaker.train.rlvr_trainer import RLVRTrainer
        return RLVRTrainer
    elif name == "RLAIFTrainer":
        from sagemaker.train.rlaif_trainer import RLAIFTrainer
        return RLAIFTrainer
    elif name == "CPTTrainer":
        from sagemaker.train.cpt_trainer import CPTTrainer
        return CPTTrainer
    elif name == "DataMixingConfig":
        from sagemaker.train.data_mixing_config import DataMixingConfig
        return DataMixingConfig
    elif name == "TrainingType":
        from sagemaker.train.common import TrainingType
        return TrainingType
    elif name == "CustomizationTechnique":
        from sagemaker.train.common import CustomizationTechnique
        return CustomizationTechnique
    elif name == "logger":
        from sagemaker.core.utils.utils import logger
        return logger
    # Evaluate module exports
    elif name == "BaseEvaluator":
        from sagemaker.train.evaluate import BaseEvaluator
        return BaseEvaluator
    elif name == "BenchMarkEvaluator":
        from sagemaker.train.evaluate import BenchMarkEvaluator
        return BenchMarkEvaluator
    elif name == "CustomScorerEvaluator":
        from sagemaker.train.evaluate import CustomScorerEvaluator
        return CustomScorerEvaluator
    elif name == "LLMAsJudgeEvaluator":
        from sagemaker.train.evaluate import LLMAsJudgeEvaluator
        return LLMAsJudgeEvaluator
    elif name == "EvaluationPipelineExecution":
        from sagemaker.train.evaluate import EvaluationPipelineExecution
        return EvaluationPipelineExecution
    elif name == "get_benchmarks":
        from sagemaker.train.evaluate import get_benchmarks
        return get_benchmarks
    elif name == "get_benchmark_properties":
        from sagemaker.train.evaluate import get_benchmark_properties
        return get_benchmark_properties
    elif name == "get_builtin_metrics":
        from sagemaker.train.evaluate import get_builtin_metrics
        return get_builtin_metrics
    elif name == "MultiTurnRLTrainer":
        from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer
        return MultiTurnRLTrainer
    elif name == "AgentRFTJob":
        from sagemaker.train.agent_rft_job import AgentRFTJob
        return AgentRFTJob
    elif name == "CustomAgentLambda":
        from sagemaker.train.custom_agent_lambda import CustomAgentLambda
        return CustomAgentLambda
    elif name == "plot_training_metrics":
        from sagemaker.train.common_utils.metrics_visualizer import plot_training_metrics
        return plot_training_metrics
    elif name == "get_available_metrics":
        from sagemaker.train.common_utils.metrics_visualizer import get_available_metrics
        return get_available_metrics
    elif name == "get_studio_url":
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        return get_studio_url
    elif name == "get_mlflow_url":
        from sagemaker.train.common_utils.trainer_wait import get_mlflow_url
        return get_mlflow_url
    elif name == "plot_training_metrics":
        from sagemaker.train.common_utils.metrics_visualizer import plot_training_metrics
        return plot_training_metrics
    elif name == "get_available_metrics":
        from sagemaker.train.common_utils.metrics_visualizer import get_available_metrics
        return get_available_metrics
    elif name == "get_studio_url":
        from sagemaker.train.common_utils.metrics_visualizer import get_studio_url
        return get_studio_url
    elif name == "get_mlflow_url":
        from sagemaker.train.common_utils.trainer_wait import get_mlflow_url
        return get_mlflow_url
    elif name == "Compute":
        from sagemaker.core.training.configs import Compute
        return Compute
    elif name == "HyperPodCompute":
        from sagemaker.core.training.configs import HyperPodCompute
        return HyperPodCompute
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
