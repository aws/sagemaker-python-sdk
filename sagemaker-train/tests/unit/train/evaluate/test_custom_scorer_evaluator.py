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
"""CustomScorerEvaluator Tests."""
from __future__ import absolute_import

import pytest
from unittest.mock import patch, MagicMock, Mock
from pydantic import ValidationError

from sagemaker.train.evaluate.custom_scorer_evaluator import (
    CustomScorerEvaluator,
    get_builtin_metrics,
    _BuiltInMetric,
)
from sagemaker.train.evaluate.constants import EvalType

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
DEFAULT_EVALUATOR_ARN = "arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/Evaluator/my-evaluator/1"


def test_get_builtin_metrics():
    """Test get_builtin_metrics returns BuiltInMetric enum."""
    BuiltInMetric = get_builtin_metrics()
    assert BuiltInMetric == _BuiltInMetric
    assert hasattr(BuiltInMetric, 'PRIME_MATH')
    assert hasattr(BuiltInMetric, 'PRIME_CODE')


def test_builtin_metric_values():
    """Test built-in metric enum values."""
    assert _BuiltInMetric.PRIME_MATH.value == "prime_math"
    assert _BuiltInMetric.PRIME_CODE.value == "prime_code"


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_initialization_minimal(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator initialization with minimal parameters."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluator == _BuiltInMetric.PRIME_MATH
    assert evaluator.dataset == DEFAULT_DATASET
    assert evaluator.model == DEFAULT_MODEL
    assert evaluator.evaluate_base_model is True


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_custom_evaluator_arn(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with custom evaluator ARN."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=DEFAULT_EVALUATOR_ARN,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluator == DEFAULT_EVALUATOR_ARN


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_evaluator_object(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with Evaluator object."""
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
    
    mock_evaluator_obj = Mock()
    mock_evaluator_obj.arn = DEFAULT_EVALUATOR_ARN
    
    evaluator = CustomScorerEvaluator(
        evaluator=mock_evaluator_obj,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluator == DEFAULT_EVALUATOR_ARN


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_builtin_metric_string(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with built-in metric as string."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator="prime_math",
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluator == _BuiltInMetric.PRIME_MATH


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_custom_scorer_evaluator_invalid_evaluator_string(mock_resolve):
    """Test CustomScorerEvaluator with invalid evaluator string."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    with pytest.raises(ValueError, match="Invalid evaluator"):
        CustomScorerEvaluator(
            evaluator="invalid_metric",
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
def test_custom_scorer_evaluator_invalid_evaluator_type(mock_resolve):
    """Test CustomScorerEvaluator with invalid evaluator type."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info
    
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    
    with pytest.raises(ValueError, match="Invalid evaluator type"):
        CustomScorerEvaluator(
            evaluator=12345,
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_dataset_resolution_from_object(mock_artifact, mock_resolve):
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=mock_dataset,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.dataset == mock_dataset.arn


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_evaluate_base_model_false(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with evaluate_base_model=False."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        evaluate_base_model=False,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluate_base_model is False


def test_custom_scorer_evaluator_missing_required_fields():
    """Test error when required fields are missing."""
    mock_session = Mock()
    
    # Missing evaluator
    with pytest.raises(ValidationError):
        CustomScorerEvaluator(
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
    
    # Missing dataset
    with pytest.raises(ValidationError):
        CustomScorerEvaluator(
            evaluator=_BuiltInMetric.PRIME_MATH,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
    
    # Missing mlflow_resource_arn
    with pytest.raises(ValidationError):
        CustomScorerEvaluator(
            evaluator=_BuiltInMetric.PRIME_MATH,
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_resolve_evaluator_config_builtin(mock_artifact, mock_resolve):
    """Test _resolve_evaluator_config with built-in metric."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = evaluator._resolve_evaluator_config()
    assert config['evaluator_arn'] is None
    assert config['preset_reward_function'] == "prime_math"


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_resolve_evaluator_config_arn(mock_artifact, mock_resolve):
    """Test _resolve_evaluator_config with custom evaluator ARN."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=DEFAULT_EVALUATOR_ARN,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    config = evaluator._resolve_evaluator_config()
    assert config['evaluator_arn'] == DEFAULT_EVALUATOR_ARN
    assert config['preset_reward_function'] is None


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_get_custom_scorer_template_additions_with_aggregation(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow
):
    """Test _get_custom_scorer_template_additions with evaluator ARN sets default aggregation."""
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.5}
    mock_extract_options.return_value = {'temperature': {'value': 0.5}}
    
    evaluator = CustomScorerEvaluator(
        evaluator=DEFAULT_EVALUATOR_ARN,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    evaluator_config = {'evaluator_arn': DEFAULT_EVALUATOR_ARN, 'preset_reward_function': None}
    additions = evaluator._get_custom_scorer_template_additions(evaluator_config)
    
    # Verify postprocessing is True and aggregation defaults to mean
    assert additions['postprocessing'] == 'True'
    assert additions['aggregation'] == 'mean'


@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_get_inference_params_from_hub(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options
):
    """Test _get_inference_params_from_hub method."""
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'max_new_tokens': '4096', 'temperature': '0.5'}
    mock_extract_options.return_value = {'max_new_tokens': '4096', 'temperature': '0.5'}
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    params = evaluator._get_inference_params_from_hub(DEFAULT_REGION)
    
    # Verify mocks were called
    mock_get_params.assert_called_once()
    mock_extract_options.assert_called_once()
    
    # Verify inference params returned
    assert 'max_new_tokens' in params
    assert 'temperature' in params


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_get_inference_params_from_hub_no_base_model(mock_artifact, mock_resolve):
    """Test _get_inference_params_from_hub with no base model name returns fallback."""
    mock_info = Mock()
    mock_info.base_model_name = None
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
    
    # Provide explicit base_eval_name to avoid None.split() error
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        base_eval_name="test-eval",
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    params = evaluator._get_inference_params_from_hub(DEFAULT_REGION)
    
    # Verify fallback values
    assert params['max_new_tokens'] == '8192'
    assert params['temperature'] == '0'
    assert params['top_k'] == '-1'
    assert params['top_p'] == '1.0'


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.evaluate.custom_scorer_evaluator.EvaluationPipelineExecution')
def test_custom_scorer_evaluator_get_all(mock_execution_class, mock_resolve_mlflow):
    """Test get_all class method."""
    mock_execution1 = Mock()
    mock_execution2 = Mock()
    mock_execution_class.get_all.return_value = iter([mock_execution1, mock_execution2])
    
    mock_session = Mock()
    executions = list(CustomScorerEvaluator.get_all(session=mock_session, region=DEFAULT_REGION))
    
    mock_execution_class.get_all.assert_called_once_with(
        eval_type=EvalType.CUSTOM_SCORER,
        session=mock_session,
        region=DEFAULT_REGION
    )
    
    assert len(executions) == 2
    assert executions[0] == mock_execution1
    assert executions[1] == mock_execution2


@pytest.mark.skip(reason="Integration test - requires full pipeline execution setup")
@patch('sagemaker.train.evaluate.execution.Pipeline')
@patch('sagemaker.train.evaluate.custom_scorer_evaluator.EvaluationPipelineExecution')
@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_evaluate_method(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow, mock_execution_class, mock_pipeline
):
    """Test evaluate method creates and starts execution."""
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    # Mock Pipeline and execution
    mock_pipeline_instance = Mock()
    mock_pipeline_instance.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline"
    mock_pipeline.create.return_value = mock_pipeline_instance
    
    mock_execution = Mock()
    mock_execution_class.start.return_value = mock_execution
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    result = evaluator.evaluate()
    
    # Verify execution was started
    mock_execution_class.start.assert_called_once()
    
    # Verify result is the mock execution
    assert result == mock_execution


@pytest.mark.skip(reason="Integration test - requires full pipeline execution setup")
@patch('sagemaker.train.evaluate.execution.Pipeline')
@patch('sagemaker.train.evaluate.custom_scorer_evaluator.EvaluationPipelineExecution')
@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_evaluate_with_model_package(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow, mock_execution_class, mock_pipeline
):
    """Test evaluate method with ModelPackage (fine-tuned model)."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    model_package_arn = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package/1"
    
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = model_package_arn
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    # Mock Pipeline and execution
    mock_pipeline_instance = Mock()
    mock_pipeline_instance.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline"
    mock_pipeline.create.return_value = mock_pipeline_instance
    
    mock_execution = Mock()
    mock_execution_class.start.return_value = mock_execution
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=model_package_arn,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        sagemaker_session=mock_session,
    )
    
    result = evaluator.evaluate()
    
    # Verify execution was started
    mock_execution_class.start.assert_called_once()
    
    # Verify result is the mock execution
    assert result == mock_execution


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_vpc_config(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with VPC configuration."""
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
    
    from sagemaker.core.shapes import VpcConfig
    vpc_config = VpcConfig(
        security_group_ids=["sg-123456"],
        subnets=["subnet-123456"]
    )
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        networking=vpc_config,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.networking == vpc_config


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_kms_key(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with KMS key."""
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
    
    kms_key_id = "arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012"
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        kms_key_id=kms_key_id,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.kms_key_id == kms_key_id


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_with_mlflow_names(mock_artifact, mock_resolve):
    """Test CustomScorerEvaluator with MLflow experiment and run names."""
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
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        mlflow_experiment_name="my-experiment",
        mlflow_run_name="my-run",
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.mlflow_experiment_name == "my-experiment"
    assert evaluator.mlflow_run_name == "my-run"
@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_hyperparameters_property(mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow):
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
    mock_session.sagemaker_config = None
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.7, 'max_new_tokens': 2048}
    mock_extract_options.return_value = {
        'temperature': {'value': 0.7, 'type': 'float', 'min': 0.0, 'max': 1.0},
        'max_new_tokens': {'value': 2048, 'type': 'int', 'min': 1, 'max': 8192}
    }
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
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
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_hyperparameters_no_base_model(mock_artifact, mock_resolve, mock_resolve_mlflow):
    """Test hyperparameters property when base model name is not available."""
    mock_resolve_mlflow.return_value = DEFAULT_MLFLOW_ARN
    mock_info = Mock()
    mock_info.base_model_name = None
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
    
    # Provide explicit base_eval_name to avoid None.split() error
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        base_eval_name="test-eval",
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    with pytest.raises(ValueError, match="Base model name not available"):
        _ = evaluator.hyperparameters


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_get_custom_scorer_template_additions_builtin(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow
):
    """Test _get_custom_scorer_template_additions with built-in metric."""
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.7}
    mock_extract_options.return_value = {'temperature': {'value': 0.7}}
    
    evaluator = CustomScorerEvaluator(
        evaluator=_BuiltInMetric.PRIME_MATH,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    evaluator_config = {'evaluator_arn': None, 'preset_reward_function': 'prime_math'}
    additions = evaluator._get_custom_scorer_template_additions(evaluator_config)
    
    # Verify required fields
    assert additions['task'] == 'gen_qa'
    assert additions['strategy'] == 'gen_qa'
    assert additions['evaluation_metric'] == 'all'
    assert additions['evaluate_base_model'] is True
    assert additions['evaluator_arn'] is None
    assert additions['preset_reward_function'] == 'prime_math'
    assert 'temperature' in additions


@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.recipe_utils._extract_eval_override_options')
@patch('sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_custom_scorer_evaluator_get_custom_scorer_template_additions_custom_arn(
    mock_artifact, mock_resolve, mock_get_params, mock_extract_options, mock_resolve_mlflow
):
    """Test _get_custom_scorer_template_additions with custom evaluator ARN."""
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
    
    # Mock recipe utils
    mock_get_params.return_value = {'temperature': 0.5, 'aggregation': 'median'}
    mock_extract_options.return_value = {
        'temperature': {'value': 0.5},
        'aggregation': {'value': 'median'}
    }
    
    evaluator = CustomScorerEvaluator(
        evaluator=DEFAULT_EVALUATOR_ARN,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Mock the hyperparameters property to return a mock with to_dict method
    mock_hyperparams = Mock()
    mock_hyperparams.to_dict.return_value = {'temperature': 0.5, 'aggregation': 'median'}
    evaluator._hyperparameters = mock_hyperparams
    
    evaluator_config = {'evaluator_arn': DEFAULT_EVALUATOR_ARN, 'preset_reward_function': None}
    additions = evaluator._get_custom_scorer_template_additions(evaluator_config)
    
    # Verify required fields
    assert additions['evaluator_arn'] == DEFAULT_EVALUATOR_ARN
    assert 'preset_reward_function' not in additions
    assert additions['postprocessing'] == 'True'
    # Verify aggregation is set from configured params
    assert additions['aggregation'] == 'median'
