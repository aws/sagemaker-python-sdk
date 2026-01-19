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
"""BenchmarkEvaluator Tests."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock, Mock
from pydantic import ValidationError

from sagemaker.train.evaluate.benchmark_evaluator import (
    BenchMarkEvaluator,
    get_benchmarks,
    get_benchmark_properties,
    _Benchmark,
    _BENCHMARK_CONFIG,
)
from sagemaker.train.evaluate.constants import EvalType
from sagemaker.core.shapes import VpcConfig

# Test constants
DEFAULT_REGION = "us-west-2"
DEFAULT_ROLE = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_MODEL = "llama3-2-1b-instruct"
DEFAULT_DATASET = "s3://test-bucket/dataset.jsonl"
DEFAULT_S3_OUTPUT = "s3://test-bucket/outputs/"
DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server"
DEFAULT_MODEL_PACKAGE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group"
DEFAULT_BASE_MODEL_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/llama3-2-1b-instruct/1.0.0"
DEFAULT_ARTIFACT_ARN = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact"


def test_get_benchmarks():
    """Test get_benchmarks returns Benchmark enum."""
    Benchmark = get_benchmarks()
    assert Benchmark == _Benchmark
    assert hasattr(Benchmark, 'MMLU')
    assert hasattr(Benchmark, 'BBH')
    assert hasattr(Benchmark, 'MATH')


@pytest.mark.parametrize(
    "benchmark,expected_keys",
    [
        (_Benchmark.MMLU, ['modality', 'description', 'metrics', 'strategy', 'subtask_available', 'subtasks']),
        (_Benchmark.BBH, ['modality', 'description', 'metrics', 'strategy', 'subtask_available', 'subtasks']),
        (_Benchmark.MATH, ['modality', 'description', 'metrics', 'strategy', 'subtask_available', 'subtasks']),
    ],
    ids=['mmlu', 'bbh', 'math']
)
def test_get_benchmark_properties(benchmark, expected_keys):
    """Test get_benchmark_properties returns correct properties."""
    props = get_benchmark_properties(benchmark)
    
    for key in expected_keys:
        assert key in props
    
    assert props is not _BENCHMARK_CONFIG[benchmark]
    assert isinstance(props['modality'], str)
    assert isinstance(props['description'], str)
    assert isinstance(props['metrics'], list)


def test_get_benchmark_properties_invalid_benchmark():
    """Test get_benchmark_properties raises error for invalid benchmark."""
    class FakeBenchmark:
        value = "invalid_benchmark"
    
    with pytest.raises(ValueError, match="Benchmark 'invalid_benchmark' not found"):
        get_benchmark_properties(FakeBenchmark())


@pytest.mark.parametrize(
    "benchmark,expected_strategy,expected_metric",
    [
        (_Benchmark.MMLU, "zs_cot", "accuracy"),
        (_Benchmark.BBH, "fs_cot", "accuracy"),
        (_Benchmark.MATH, "zs_cot", "exact_match"),
        (_Benchmark.STRONG_REJECT, "zs", "deflection"),
        (_Benchmark.IFEVAL, "zs", "accuracy"),
    ],
    ids=['mmlu', 'bbh', 'math', 'strong_reject', 'ifeval']
)
def test_benchmark_config_strategy_and_metrics(benchmark, expected_strategy, expected_metric):
    """Test benchmark configuration has correct strategy and metrics."""
    config = _BENCHMARK_CONFIG[benchmark]
    assert config['strategy'] == expected_strategy
    assert expected_metric in config['metrics']


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_initialization_minimal(mock_artifact, mock_resolve):
    """Test BenchmarkEvaluator initialization with minimal parameters."""
    # Setup mocks
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.benchmark == _Benchmark.MMLU
    assert evaluator.model == DEFAULT_MODEL
    assert evaluator.evaluate_base_model is False
    assert evaluator.subtasks == "ALL"


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_subtask_defaults_to_all(mock_artifact, mock_resolve):
    """Test subtasks default to ALL for benchmarks that support them."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.subtasks == "ALL"


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_subtask_validation_invalid(mock_artifact, mock_resolve):
    """Test invalid subtask raises error."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    with pytest.raises(ValueError, match="Invalid subtask"):
        BenchMarkEvaluator(
            benchmark=_Benchmark.MMLU,
            subtasks=["invalid_subtask"],
            model=DEFAULT_MODEL,
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_no_subtask_for_unsupported_benchmark(mock_artifact, mock_resolve):
    """Test error when providing subtask for benchmark that doesn't support it."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    with pytest.raises(ValueError, match="Subtask is not supported"):
        BenchMarkEvaluator(
            benchmark=_Benchmark.GPQA,
            subtasks="some_subtask",
            model=DEFAULT_MODEL,
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_dataset_resolution_from_object(mock_artifact, mock_resolve):
    """Test dataset resolution from DataSet object."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    mock_dataset = Mock()
    mock_dataset.arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/AIRegistry/DataSet/test/1.0.0"
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Dataset field is commented out, so no assertion needed


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_evaluate_method_exists(mock_artifact, mock_resolve):
    """Test evaluate method exists and has correct signature."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks=["abstract_algebra"],
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Verify evaluate method exists
    assert hasattr(evaluator, 'evaluate')
    assert callable(evaluator.evaluate)
    
    # Verify method accepts optional subtask parameter
    import inspect
    sig = inspect.signature(evaluator.evaluate)
    assert 'subtask' in sig.parameters


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_evaluate_invalid_subtask_override(mock_artifact, mock_resolve, mock_resolve_mlflow):
    """Test evaluate with invalid subtask override raises error."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None  # Prevent config validation issues
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    with pytest.raises(ValueError, match="Invalid subtask"):
        evaluator.evaluate(subtask="invalid_subtask")


@patch('sagemaker.train.evaluate.benchmark_evaluator.EvaluationPipelineExecution')
def test_benchmark_evaluator_get_all(mock_execution_class):
    """Test get_all class method."""
    mock_execution1 = Mock()
    mock_execution2 = Mock()
    mock_execution_class.get_all.return_value = iter([mock_execution1, mock_execution2])
    
    mock_session = Mock()
    executions = list(BenchMarkEvaluator.get_all(session=mock_session, region=DEFAULT_REGION))
    
    mock_execution_class.get_all.assert_called_once_with(
        eval_type=EvalType.BENCHMARK,
        session=mock_session,
        region=DEFAULT_REGION
    )
    
    assert len(executions) == 2
    assert executions[0] == mock_execution1
    assert executions[1] == mock_execution2


def test_benchmark_evaluator_missing_required_fields():
    """Test error when required fields are missing."""
    mock_session = Mock()
    
    # Missing dataset
    with pytest.raises(ValidationError):
        BenchMarkEvaluator(
            benchmark=_Benchmark.MMLU,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
    
    # Missing mlflow_resource_arn
    with pytest.raises(ValidationError):
        BenchMarkEvaluator(
            benchmark=_Benchmark.MMLU,
            model=DEFAULT_MODEL,
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_resolve_subtask_for_evaluation(mock_artifact, mock_resolve):
    """Test _resolve_subtask_for_evaluation method."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks="abstract_algebra",  # Use a specific subtask instead of "ALL"
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # When None is passed, should return the evaluator's subtasks value
    result = evaluator._resolve_subtask_for_evaluation(None)
    assert result == "abstract_algebra"
    
    # When a specific subtask is passed, should return that subtask
    result = evaluator._resolve_subtask_for_evaluation("anatomy")
    assert result == "anatomy"


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_hyperparameters_property(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_is_nova, mock_resolve_mlflow):
    """Test hyperparameters property lazy loading."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None  # Prevent config validation issues
    
    # Mock recipe utils
    mock_is_nova.return_value = False
    mock_get_params.return_value = {'temperature': 0.7, 'max_tokens': 2048}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}, 'max_tokens': {'value': 2048}}
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Access hyperparameters (triggers lazy load)
    hyperparams = evaluator.hyperparameters
    
    # Verify mocks were called
    mock_get_params.assert_called_once()
    mock_extract_options.assert_called_once()
    
    # Verify hyperparameters object is cached
    assert evaluator._hyperparameters is not None
    assert evaluator.hyperparameters is hyperparams  # Same instance


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_get_benchmark_template_additions(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_is_nova, mock_resolve_mlflow):
    """Test _get_benchmark_template_additions method."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None  # Prevent config validation issues
    
    # Mock recipe utils
    mock_is_nova.return_value = False
    mock_get_params.return_value = {'temperature': 0.7, 'max_tokens': 2048}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}, 'max_tokens': {'value': 2048}}
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks=["abstract_algebra"],
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = {'strategy': 'zs_cot', 'metrics': ['accuracy']}
    additions = evaluator._get_benchmark_template_additions("abstract_algebra", config)
    
    # Verify required fields
    assert additions['task'] == 'mmlu'
    assert additions['strategy'] == 'zs_cot'
    assert additions['evaluation_metric'] == 'accuracy'
    assert additions['subtask'] == 'abstract_algebra'
    assert additions['evaluate_base_model'] is False


@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_mmmu_nova_validation(mock_artifact, mock_resolve, mock_is_nova):
    """Test that mmmu benchmark requires Nova models."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    # Mock that model is NOT a Nova model
    mock_is_nova.return_value = False
    
    # Try to create evaluator with MMMU benchmark (should fail for non-Nova)
    with pytest.raises(ValueError, match="Benchmark 'mmmu' is only supported for Nova models"):
        BenchMarkEvaluator(
            benchmark=_Benchmark.MMMU,
            model=DEFAULT_MODEL,
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_llm_judge_nova_validation(mock_artifact, mock_resolve, mock_is_nova):
    """Test that llm_judge benchmark is not allowed for Nova models."""
    mock_info = Mock()
    mock_info.base_model_name = "nova-pro"
    mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/nova-pro/1.0.0"
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    # Mock that model IS a Nova model
    mock_is_nova.return_value = True
    
    # Try to create evaluator with LLM_JUDGE benchmark (should fail for Nova)
    with pytest.raises(ValueError, match="Benchmark 'llm_judge' is not supported for Nova models"):
        BenchMarkEvaluator(
            benchmark=_Benchmark.LLM_JUDGE,
            model="nova-pro",
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_subtask_list_validation(mock_artifact, mock_resolve):
    """Test subtask validation with list of subtasks."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    # Valid list of subtasks
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks=["abstract_algebra", "anatomy"],
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    assert evaluator.subtasks == ["abstract_algebra", "anatomy"]
    
    # Empty list should fail
    with pytest.raises(ValueError, match="Subtask list cannot be empty"):
        BenchMarkEvaluator(
            benchmark=_Benchmark.MMLU,
            subtasks=[],
            model=DEFAULT_MODEL,
            
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_resolve_subtask_list(mock_artifact, mock_resolve):
    """Test _resolve_subtask_for_evaluation with list of subtasks."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks=["abstract_algebra", "anatomy"],
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    result = evaluator._resolve_subtask_for_evaluation(None)
    assert result == ["abstract_algebra", "anatomy"]
    
    # Test with list override
    result = evaluator._resolve_subtask_for_evaluation(["abstract_algebra"])
    assert result == ["abstract_algebra"]
    
    # Test with invalid subtask in list
    with pytest.raises(ValueError, match="Invalid subtask"):
        evaluator._resolve_subtask_for_evaluation(["invalid_subtask"])


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_template_additions_with_list_subtasks(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_is_nova, mock_resolve_mlflow):
    """Test _get_benchmark_template_additions with list of subtasks."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None  # Prevent config validation issues
    
    # Mock recipe utils
    mock_is_nova.return_value = False
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        subtasks=["abstract_algebra", "anatomy"],
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = {'strategy': 'zs_cot', 'metrics': ['accuracy']}
    additions = evaluator._get_benchmark_template_additions(["abstract_algebra", "anatomy"], config)
    
    # Verify subtask is comma-separated
    assert additions['subtask'] == 'abstract_algebra,anatomy'



# Additional tests for improved coverage

@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_with_subtask_list(mock_resolve):
    """Test BenchmarkEvaluator with subtask as list."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        model=DEFAULT_MODEL,
        
        benchmark=_Benchmark.MMLU,
        subtasks=['abstract_algebra', 'anatomy'],
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.subtasks == ['abstract_algebra', 'anatomy']


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_with_subtask_string(mock_resolve):
    """Test BenchmarkEvaluator with subtask as string."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    evaluator = BenchMarkEvaluator(
        model=DEFAULT_MODEL,
        
        benchmark=_Benchmark.MMLU,
        subtasks='abstract_algebra',
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Subtasks remain as string if passed as string
    assert evaluator.subtasks == 'abstract_algebra'


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_invalid_subtask(mock_resolve):
    """Test BenchmarkEvaluator with invalid subtask."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    with pytest.raises(ValidationError, match="Invalid subtask"):
        BenchMarkEvaluator(
            model=DEFAULT_MODEL,
            
            benchmark=_Benchmark.MMLU,
            subtasks=['invalid_subtask'],
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_no_subtask_available(mock_resolve):
    """Test BenchmarkEvaluator with benchmark that doesn't support subtasks."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    # IFEVAL doesn't support subtasks
    evaluator = BenchMarkEvaluator(
        model=DEFAULT_MODEL,
        
        benchmark=_Benchmark.IFEVAL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.subtasks is None


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_with_networking(mock_resolve):
    """Test BenchmarkEvaluator with networking configuration."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    vpc_config = VpcConfig(
        security_group_ids=['sg-123'],
        subnets=['subnet-123']
    )
    
    evaluator = BenchMarkEvaluator(
        model=DEFAULT_MODEL,
        
        benchmark=_Benchmark.MMLU,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        networking=vpc_config,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.networking == vpc_config


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_benchmark_evaluator_with_kms_key(mock_resolve):
    """Test BenchmarkEvaluator with KMS key."""
    mock_info = MagicMock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = MagicMock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    kms_key = "arn:aws:kms:us-west-2:123456789012:key/test-key"
    
    evaluator = BenchMarkEvaluator(
        model=DEFAULT_MODEL,
        
        benchmark=_Benchmark.MMLU,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        kms_key_id=kms_key,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.kms_key_id == kms_key


# Tests for conditional metric key (Nova vs OpenWeights)

@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_uses_metric_key_for_nova(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_is_nova, mock_resolve_mlflow):
    """Test that Nova models use 'metric' key instead of 'evaluation_metric'."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = "nova-pro"
    mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/nova-pro/1.0.0"
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None
    
    # Mock that model IS a Nova model
    mock_is_nova.return_value = True
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model="nova-pro",
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = {'strategy': 'zs_cot', 'metrics': ['accuracy']}
    additions = evaluator._get_benchmark_template_additions("ALL", config)
    
    # Verify Nova model uses 'metric' key
    assert 'metric' in additions
    assert additions['metric'] == 'accuracy'
    assert 'evaluation_metric' not in additions


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_benchmark_evaluator_uses_evaluation_metric_key_for_non_nova(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_is_nova, mock_resolve_mlflow):
    """Test that non-Nova models use 'evaluation_metric' key."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None
    
    # Mock that model is NOT a Nova model
    mock_is_nova.return_value = False
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    evaluator = BenchMarkEvaluator(
        benchmark=_Benchmark.MMLU,
        model=DEFAULT_MODEL,
        
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = {'strategy': 'zs_cot', 'metrics': ['accuracy']}
    additions = evaluator._get_benchmark_template_additions("ALL", config)
    
    # Verify non-Nova model uses 'evaluation_metric' key
    assert 'evaluation_metric' in additions
    assert additions['evaluation_metric'] == 'accuracy'
    assert 'metric' not in additions
