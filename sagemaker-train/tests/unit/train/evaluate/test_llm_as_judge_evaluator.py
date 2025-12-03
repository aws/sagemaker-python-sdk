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
"""LLMAsJudgeEvaluator Tests."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, Mock
from pydantic import ValidationError

from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator
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
DEFAULT_EVALUATOR_MODEL = "anthropic.claude-3-5-sonnet-20240620-v1:0"


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_initialization_minimal(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator initialization with minimal parameters."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluator_model == DEFAULT_EVALUATOR_MODEL
    assert evaluator.dataset == DEFAULT_DATASET
    assert evaluator.model == DEFAULT_MODEL
    assert evaluator.evaluate_base_model is True
    assert evaluator.builtin_metrics is None
    assert evaluator.custom_metrics is None


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_with_builtin_metrics(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with builtin metrics."""
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
    
    builtin_metrics = ["Correctness", "Helpfulness"]
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        builtin_metrics=builtin_metrics,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.builtin_metrics == builtin_metrics


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_with_custom_metrics(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with custom metrics."""
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
    
    custom_metrics = json.dumps([{
        "customMetricDefinition": {
            "name": "PositiveSentiment",
            "instructions": "Assess if the response has positive sentiment",
            "ratingScale": [
                {"definition": "Good", "value": {"floatValue": 1.0}},
                {"definition": "Poor", "value": {"floatValue": 0.0}}
            ]
        }
    }])
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        custom_metrics=custom_metrics,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.custom_metrics == custom_metrics


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_dataset_resolution_from_object(mock_artifact, mock_resolve):
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=mock_dataset,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.dataset == mock_dataset.arn


@patch('sagemaker.train.common_utils.recipe_utils._is_nova_model')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_nova_model_validation(mock_artifact, mock_resolve, mock_is_nova):
    """Test that Nova models are rejected for LLM-as-judge evaluation."""
    mock_info = Mock()
    mock_info.base_model_name = "amazon-nova-lite-v1"
    mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/amazon-nova-lite-v1/1.0.0"
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
    
    mock_is_nova.return_value = True
    
    with pytest.raises(ValueError, match="LLM-as-judge evaluation is not supported for Nova models"):
        LLMAsJudgeEvaluator(
            evaluator_model=DEFAULT_EVALUATOR_MODEL,
            dataset=DEFAULT_DATASET,
            model="amazon-nova-lite-v1",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_evaluate_base_model_false(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with evaluate_base_model=False."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        evaluate_base_model=False,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    assert evaluator.evaluate_base_model is False


def test_llm_as_judge_evaluator_missing_required_fields():
    """Test error when required fields are missing."""
    mock_session = Mock()
    
    # Missing evaluator_model
    with pytest.raises(ValidationError):
        LLMAsJudgeEvaluator(
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
    
    # Missing dataset
    with pytest.raises(ValidationError):
        LLMAsJudgeEvaluator(
            evaluator_model=DEFAULT_EVALUATOR_MODEL,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            sagemaker_session=mock_session,
        )
    
    # Missing mlflow_resource_arn
    with pytest.raises(ValidationError):
        LLMAsJudgeEvaluator(
            evaluator_model=DEFAULT_EVALUATOR_MODEL,
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_process_builtin_metrics(mock_artifact, mock_resolve):
    """Test _process_builtin_metrics removes 'Builtin.' prefix."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    # Test with 'Builtin.' prefix
    metrics_with_prefix = ["Builtin.Correctness", "Builtin.Helpfulness", "Faithfulness"]
    processed = evaluator._process_builtin_metrics(metrics_with_prefix)
    assert processed == ["Correctness", "Helpfulness", "Faithfulness"]
    
    # Test without prefix
    metrics_without_prefix = ["Correctness", "Helpfulness"]
    processed = evaluator._process_builtin_metrics(metrics_without_prefix)
    assert processed == ["Correctness", "Helpfulness"]
    
    # Test with mixed case
    metrics_mixed_case = ["builtin.Correctness", "BUILTIN.Helpfulness"]
    processed = evaluator._process_builtin_metrics(metrics_mixed_case)
    assert processed == ["Correctness", "Helpfulness"]
    
    # Test with None
    processed = evaluator._process_builtin_metrics(None)
    assert processed == []
    
    # Test with empty list
    processed = evaluator._process_builtin_metrics([])
    assert processed == []


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_validate_custom_metrics_json_valid(mock_artifact, mock_resolve):
    """Test _validate_custom_metrics_json with valid JSON."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    valid_json = json.dumps([{"name": "test"}])
    result = evaluator._validate_custom_metrics_json(valid_json)
    assert result == valid_json
    
    # Test with None
    result = evaluator._validate_custom_metrics_json(None)
    assert result is None


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_validate_custom_metrics_json_invalid(mock_artifact, mock_resolve):
    """Test _validate_custom_metrics_json with invalid JSON."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    invalid_json = "not valid json {"
    with pytest.raises(ValueError, match="Invalid JSON in custom_metrics"):
        evaluator._validate_custom_metrics_json(invalid_json)


@patch('sagemaker.core.s3.client.S3Uploader.upload_string_as_file_body')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_get_llmaj_template_additions(mock_artifact, mock_resolve, mock_s3_upload):
    """Test _get_llmaj_template_additions method."""
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
    
    builtin_metrics = ["Builtin.Correctness", "Helpfulness"]
    custom_metrics = json.dumps([{"name": "test"}])
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        builtin_metrics=builtin_metrics,
        custom_metrics=custom_metrics,
        s3_output_path="s3://test-bucket/outputs/",
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    eval_name = "test-eval"
    additions = evaluator._get_llmaj_template_additions(eval_name)
    
    assert additions['judge_model_id'] == DEFAULT_EVALUATOR_MODEL
    assert additions['s3_output_path'] == "s3://test-bucket/outputs"  # Trailing slash removed
    assert additions['llmaj_metrics'] == json.dumps(["Correctness", "Helpfulness"])
    # custom_metrics now uploaded to S3
    assert 'custom_metrics' in additions
    assert additions['custom_metrics'].startswith("s3://test-bucket/outputs/evaluationinputs/")
    assert additions['max_new_tokens'] == '8192'
    assert additions['temperature'] == '0'
    assert additions['top_k'] == '-1'
    assert additions['top_p'] == '1.0'
    # pipeline_name is no longer in template additions - it's resolved dynamically in execution.py
    assert 'pipeline_name' not in additions
    assert additions['evaluate_base_model'] is True
    
    # Verify S3 upload was called
    mock_s3_upload.assert_called_once()


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_get_llmaj_template_additions_no_metrics(mock_artifact, mock_resolve):
    """Test _get_llmaj_template_additions with no metrics specified."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    eval_name = "test-eval"
    additions = evaluator._get_llmaj_template_additions(eval_name)
    
    assert additions['llmaj_metrics'] == json.dumps([])
    assert additions['custom_metrics'] is None


@pytest.mark.skip(reason="Integration test - requires full pipeline execution setup")
@patch('sagemaker.train.evaluate.execution.Pipeline')
@patch('sagemaker.train.evaluate.llm_as_judge_evaluator.EvaluationPipelineExecution')
@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_evaluate_method(mock_artifact, mock_resolve, mock_resolve_mlflow, mock_execution_class, mock_pipeline):
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
    
    # Mock Pipeline and execution
    mock_pipeline_instance = Mock()
    mock_pipeline_instance.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline"
    mock_pipeline.create.return_value = mock_pipeline_instance
    
    mock_execution = Mock()
    mock_execution_class.start.return_value = mock_execution
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        builtin_metrics=["Correctness"],
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    
    result = evaluator.evaluate()
    
    # Verify execution was started
    mock_execution_class.start.assert_called_once()
    assert result == mock_execution


@pytest.mark.skip(reason="Integration test - requires full pipeline execution setup")
@patch('sagemaker.train.evaluate.execution.Pipeline')
@patch('sagemaker.train.evaluate.llm_as_judge_evaluator.EvaluationPipelineExecution')
@patch('sagemaker.train.common_utils.finetune_utils._resolve_mlflow_resource_arn')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_evaluate_with_model_package(mock_artifact, mock_resolve, mock_resolve_mlflow, mock_execution_class, mock_pipeline):
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
    
    # Mock Pipeline and execution
    mock_pipeline_instance = Mock()
    mock_pipeline_instance.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/test-pipeline"
    mock_pipeline.create.return_value = mock_pipeline_instance
    
    mock_execution = Mock()
    mock_execution_class.start.return_value = mock_execution
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=model_package_arn,
        builtin_metrics=["Correctness"],
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        sagemaker_session=mock_session,
    )
    
    result = evaluator.evaluate()
    
    # Verify execution was started
    mock_execution_class.start.assert_called_once()
    assert result == mock_execution


@patch('sagemaker.train.evaluate.execution.EvaluationPipelineExecution')
def test_llm_as_judge_evaluator_get_all(mock_execution_class):
    """Test get_all class method."""
    mock_execution1 = Mock()
    mock_execution2 = Mock()
    mock_execution_class.get_all.return_value = iter([mock_execution1, mock_execution2])
    
    mock_session = Mock()
    executions = list(LLMAsJudgeEvaluator.get_all(session=mock_session, region=DEFAULT_REGION))
    
    mock_execution_class.get_all.assert_called_once_with(
        eval_type=EvalType.LLM_AS_JUDGE,
        session=mock_session,
        region=DEFAULT_REGION
    )
    
    assert len(executions) == 2
    assert executions[0] == mock_execution1
    assert executions[1] == mock_execution2


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_with_vpc_config(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with VPC configuration."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
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
def test_llm_as_judge_evaluator_with_kms_key(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with KMS key."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
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
def test_llm_as_judge_evaluator_with_mlflow_names(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with MLflow experiment and run names."""
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
    
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
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
