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
from datetime import datetime, timedelta, timezone

import pytest
from unittest.mock import patch, Mock
from botocore.exceptions import ClientError
from pydantic import ValidationError

from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator
from sagemaker.train.evaluate.constants import EvalType

# Where the evaluator reads the supported-judge-models list from.
_S3_READ_FILE_PATH = "sagemaker.core.s3.client.S3Downloader.read_file"


def _configure_bedrock_get_model(mock_session, lifecycle=None, side_effect=None):
    """Wire mock_session.boto_session.client('bedrock').get_foundation_model.

    Args:
        mock_session: Mock session whose boto_session is a Mock.
        lifecycle: dict placed at modelDetails.modelLifecycle in the response.
        side_effect: if set, raised by get_foundation_model instead of returning.
    """
    bedrock_client = Mock()
    if side_effect is not None:
        bedrock_client.get_foundation_model.side_effect = side_effect
    else:
        bedrock_client.get_foundation_model.return_value = {
            "modelDetails": {"modelLifecycle": lifecycle or {"status": "ACTIVE"}}
        }
    mock_session.boto_session.client.return_value = bedrock_client
    return bedrock_client


def _supported_models_doc(model_ids):
    """Build a supported-llmaj-judge-models.json body listing ``model_ids``."""
    return json.dumps(
        {
            "schema_version": "1.0",
            "supported_judge_models": [{"model_id": mid} for mid in model_ids],
        }
    )


def _patch_supported_models(model_ids=None, side_effect=None):
    """Patch S3Downloader.read_file to serve a supported-judge-models list.

    Args:
        model_ids: iterable of model_ids the list should contain.
        side_effect: if provided, set as read_file's side_effect (e.g. an error)
            instead of returning a document body.
    """
    if side_effect is not None:
        return patch(_S3_READ_FILE_PATH, side_effect=side_effect)
    return patch(
        _S3_READ_FILE_PATH, return_value=_supported_models_doc(model_ids or [])
    )


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
DEFAULT_EVALUATOR_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"


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
    assert evaluator.evaluate_base_model is False
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
def test_llm_as_judge_evaluator_nova_model_auto_routed(mock_artifact, mock_resolve, mock_is_nova):
    """Test that Nova models are accepted and auto-routed to InspectAI+Bedrock."""
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
    
    # Nova models are now allowed — they auto-route to InspectAI+Bedrock
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model="amazon-nova-lite-v1",
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    assert evaluator._should_use_inspectai_path() is True


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
    assert additions['evaluate_base_model'] is False
    
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


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_builtin_metrics_only_no_custom(mock_artifact, mock_resolve):
    """Test that evaluator handles builtin_metrics with custom_metrics=None correctly."""
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
        builtin_metrics=["Completeness", "Faithfulness"],
        custom_metrics=None,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )

    assert evaluator.builtin_metrics == ["Completeness", "Faithfulness"]
    assert evaluator.custom_metrics is None

    eval_name = "test-eval"
    additions = evaluator._get_llmaj_template_additions(eval_name)

    assert additions['llmaj_metrics'] == json.dumps(["Completeness", "Faithfulness"])
    assert additions['custom_metrics'] is None


@patch('sagemaker.core.s3.client.S3Uploader.upload_string_as_file_body')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_custom_metrics_only_no_builtin(mock_artifact, mock_resolve, mock_s3_upload):
    """Test that evaluator handles custom_metrics with builtin_metrics=None correctly."""
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

    custom_metrics_json = json.dumps([{"customMetricDefinition": {"name": "TestMetric"}}])

    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        builtin_metrics=None,
        custom_metrics=custom_metrics_json,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )

    assert evaluator.builtin_metrics is None
    assert evaluator.custom_metrics == custom_metrics_json

    eval_name = "test-eval"
    additions = evaluator._get_llmaj_template_additions(eval_name)

    assert additions['llmaj_metrics'] == json.dumps([])
    assert additions['custom_metrics'] is not None
    assert additions['custom_metrics'].startswith("s3://")
    mock_s3_upload.assert_called_once()


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


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_valid_evaluator_models(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator with valid evaluator models."""
    # us-west-2 models only; Claude 3.5 Sonnet v1/v2 are ap-northeast-1-only
    # (covered by test_llm_as_judge_evaluator_region_restriction).
    valid_models = [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-sonnet-4-5-20250929-v1:0",
        "anthropic.claude-opus-4-5-20251101-v1:0",
        "meta.llama3-1-70b-instruct-v1:0",
        "mistral.mistral-large-2402-v1:0",
        "amazon.nova-pro-v1:0",
        "amazon.nova-2-lite-v1:0",
        "amazon.nova-micro-v1:0",
        "amazon.nova-premier-v1:0",
    ]
    
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
    mock_session.boto_region_name = "us-west-2"  # Region where all models including nova-pro are available
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE

    # The supported-judge-models list reports every model under test as supported.
    with _patch_supported_models(model_ids=valid_models):
        for model in valid_models:
            evaluator = LLMAsJudgeEvaluator(
                model=DEFAULT_MODEL,
                evaluator_model=model,
                dataset=DEFAULT_DATASET,
                builtin_metrics=["Correctness"],
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
            assert evaluator.evaluator_model == model


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_invalid_evaluator_model(mock_artifact, mock_resolve):
    """Test LLMAsJudgeEvaluator fails fast when the model is not in the supported list.

    Covers both never-supported models and EOL models: neither appears in the
    service-maintained supported-judge-models list, so construction raises.
    """
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

    # The supported list contains real models, but not "invalid-model".
    supported = [DEFAULT_EVALUATOR_MODEL, "anthropic.claude-3-haiku-20240307-v1:0"]
    with _patch_supported_models(model_ids=supported):
        with pytest.raises(ValidationError) as exc_info:
            LLMAsJudgeEvaluator(
                model=DEFAULT_MODEL,
                evaluator_model="invalid-model",
                dataset=DEFAULT_DATASET,
                builtin_metrics=["Correctness"],
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
    assert "is not a supported LLM-as-Judge model" in str(exc_info.value)
    assert "invalid-model" in str(exc_info.value)


@patch('sagemaker.train.defaults.TrainDefaults.get_sagemaker_session')
@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_llm_as_judge_evaluator_region_restriction(mock_artifact, mock_resolve, mock_get_session):
    """Test LLMAsJudgeEvaluator raises when the model is absent from a region's list."""
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
    mock_session.boto_region_name = "eu-central-1"  # Region not supported for nova-pro
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_get_session.return_value = mock_session

    # The eu-central-1 supported-judge-models list does not include nova-pro.
    with _patch_supported_models(model_ids=["anthropic.claude-3-haiku-20240307-v1:0"]):
        with pytest.raises(ValidationError) as exc_info:
            LLMAsJudgeEvaluator(
                model=DEFAULT_MODEL,
                evaluator_model="amazon.nova-pro-v1:0",
                dataset=DEFAULT_DATASET,
                builtin_metrics=["Correctness"],
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=mock_session,
            )
    assert "is not a supported LLM-as-Judge model" in str(exc_info.value)
    assert "eu-central-1" in str(exc_info.value)


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_nova_model_allowed_auto_routed(mock_artifact, mock_resolve):
    """Test that Nova JumpStart model is allowed — auto-routes to InspectAI+Bedrock."""
    mock_info = Mock()
    mock_info.base_model_name = "nova-textgeneration-lite"
    mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/nova-textgeneration-lite/1.0.0"
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

    # Should NOT raise ValueError — Nova is auto-routed to InspectAI+Bedrock
    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model="nova-textgeneration-lite",
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )

    assert evaluator.model == "nova-textgeneration-lite"
    assert evaluator._should_use_inspectai_path() is True


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_non_nova_jumpstart_model_uses_existing_path(mock_artifact, mock_resolve):
    """Test that non-Nova JumpStart model uses the existing ServerlessJobConfig path."""
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

    assert evaluator._should_use_inspectai_path() is False


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_nova_model_rejected_in_unsupported_region(mock_artifact, mock_resolve):
    """Test that a Nova base model in an unsupported region fails validation.

    evaluator_model validation degrades gracefully here (no Bedrock list is
    available from the bare mock), so the Nova cross-region-inference
    compatibility root-validator is what blocks construction.
    """
    mock_info = Mock()
    mock_info.base_model_name = "nova-textgeneration-lite"
    mock_info.base_model_arn = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/nova-textgeneration-lite/1.0.0"
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info

    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance

    mock_session = Mock()
    mock_session.boto_region_name = "af-south-1"  # Unsupported region
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE

    with pytest.raises(ValueError, match="not supported for"):
        LLMAsJudgeEvaluator(
            evaluator_model=DEFAULT_EVALUATOR_MODEL,
            dataset=DEFAULT_DATASET,
            model="nova-textgeneration-lite",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_evaluator_model_validation_degrades_when_list_unreadable(mock_artifact, mock_resolve):
    """If the supported-judge-models list can't be read, construction must NOT block.

    This is the degradation route (e.g. denied access or a missing object): we
    warn that the model can't be verified but continue rather than block the user.
    """
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

    # Reading the list fails (denied access / missing object / network error).
    with _patch_supported_models(side_effect=Exception("access denied")):
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


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_evaluator_model_validation_degrades_on_malformed_list(mock_artifact, mock_resolve):
    """A malformed/unexpected list document must NOT block construction."""
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

    # File is present but not the expected shape (no supported_judge_models array).
    with patch(_S3_READ_FILE_PATH, return_value=json.dumps({"unexpected": True})):
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


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_evaluator_model_validation_degrades_without_region(mock_artifact, mock_resolve):
    """No resolvable region means validation is skipped (non-blocking) with a warning."""
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
    mock_session.boto_region_name = None  # No region resolvable from session
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE

    with patch(_S3_READ_FILE_PATH) as mock_read_file:
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
    # The list should not be fetched when no region is available.
    mock_read_file.assert_not_called()


# ---------------------------------------------------------------------------
# _check_evaluator_model_lifecycle (evaluate()-time end-of-life check)
# ---------------------------------------------------------------------------
def _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session):
    """Construct an evaluator (supported-model check stubbed out) for lifecycle tests."""
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_resolve.return_value = mock_info

    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance

    with _patch_supported_models(model_ids=[DEFAULT_EVALUATOR_MODEL]):
        return LLMAsJudgeEvaluator(
            evaluator_model=DEFAULT_EVALUATOR_MODEL,
            dataset=DEFAULT_DATASET,
            model=DEFAULT_MODEL,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=mock_session,
        )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_active_model_passes(mock_artifact, mock_resolve):
    """An in-service (ACTIVE) judge model passes the lifecycle check."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    bedrock_client = _configure_bedrock_get_model(
        mock_session, lifecycle={"status": "ACTIVE"}
    )
    evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)  # no raise

    bedrock_client.get_foundation_model.assert_called_once_with(
        modelIdentifier=DEFAULT_EVALUATOR_MODEL
    )


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_future_eol_passes(mock_artifact, mock_resolve):
    """A LEGACY model whose end-of-life is still in the future is still usable."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    future = datetime.now(timezone.utc) + timedelta(days=30)
    _configure_bedrock_get_model(
        mock_session, lifecycle={"status": "LEGACY", "endOfLifeTime": future}
    )
    evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)  # no raise


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_past_eol_raises(mock_artifact, mock_resolve):
    """A model past its end-of-life fails fast before the job is submitted."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    past = datetime.now(timezone.utc) - timedelta(days=1)
    _configure_bedrock_get_model(
        mock_session, lifecycle={"status": "LEGACY", "endOfLifeTime": past}
    )
    with pytest.raises(ValueError, match="reached end of life"):
        evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_model_not_found_raises(mock_artifact, mock_resolve):
    """A model absent from the region (ResourceNotFound) fails fast."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    not_found = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "no such model"}},
        "GetFoundationModel",
    )
    _configure_bedrock_get_model(mock_session, side_effect=not_found)
    with pytest.raises(ValueError, match="not available in region"):
        evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_access_denied_warns_and_continues(mock_artifact, mock_resolve):
    """AccessDenied from GetFoundationModel → warn about the permission, don't block.

    We call Bedrock directly (no SimulatePrincipalPolicy pre-gate), so a missing —
    including a scoped — permission surfaces here as AccessDenied and degrades.
    """
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    denied = ClientError(
        {"Error": {"Code": "AccessDeniedException", "Message": "not authorized"}},
        "GetFoundationModel",
    )
    _configure_bedrock_get_model(mock_session, side_effect=denied)
    # Should NOT raise — degrades with a permission-specific warning.
    evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_lifecycle_transient_bedrock_error_does_not_block(mock_artifact, mock_resolve):
    """A transient Bedrock error (e.g. throttling) must NOT block construction/submit."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    throttling = ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        "GetFoundationModel",
    )
    _configure_bedrock_get_model(mock_session, side_effect=throttling)
    evaluator._check_evaluator_model_lifecycle(DEFAULT_REGION)  # no raise


@patch('sagemaker.train.common_utils.model_resolution._resolve_base_model')
@patch('sagemaker.core.resources.Artifact')
def test_evaluate_invokes_lifecycle_check(mock_artifact, mock_resolve):
    """evaluate() must call _check_evaluator_model_lifecycle with the resolved region."""
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = None  # let the telemetry decorator resolve cleanly
    evaluator = _build_lifecycle_evaluator(mock_artifact, mock_resolve, mock_session)

    sentinel = RuntimeError("lifecycle-check-invoked")
    aws_context = {
        "role_arn": DEFAULT_ROLE,
        "region": DEFAULT_REGION,
        "account_id": "123456789012",
    }
    with patch.object(evaluator, "_get_resolved_model_info", return_value=None), \
         patch.object(evaluator, "_get_aws_execution_context", return_value=aws_context), \
         patch.object(
             evaluator, "_check_evaluator_model_lifecycle", side_effect=sentinel
         ) as mock_lifecycle:
        with pytest.raises(RuntimeError, match="lifecycle-check-invoked"):
            evaluator.evaluate()

    mock_lifecycle.assert_called_once_with(DEFAULT_REGION)
