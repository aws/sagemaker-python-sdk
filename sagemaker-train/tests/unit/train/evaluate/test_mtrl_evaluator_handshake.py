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
"""Unit tests for MTRLTrainer handshake with BenchMarkEvaluator and CustomScorerEvaluator.

Tests that model resolution correctly handles MultiTurnRLTrainer instances when
passed to existing evaluators (BenchMarkEvaluator, CustomScorerEvaluator).
"""
from __future__ import absolute_import

import os
import json
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
os.environ.setdefault("SAGEMAKER_REGION", "us-west-2")
os.environ.setdefault("AWS_REGION", "us-west-2")

from sagemaker.train.common_utils.model_resolution import (
    _ModelResolver,
    _ModelInfo,
    _ModelType,
    _resolve_base_model,
)
from sagemaker.train.base_trainer import BaseTrainer


# ============================================================
# Fixtures
# ============================================================

MODEL_PACKAGE_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/my-finetuned-model/1"
BASE_MODEL_ARN = "arn:aws:sagemaker:us-west-2:aws:hub-content/SageMakerPublicHub/Model/openai-reasoning-gpt-oss-20b/1.0.0"
BASE_MODEL_NAME = "openai-reasoning-gpt-oss-20b"
MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-app/app-ABCDEF"
S3_OUTPUT = "s3://sagemaker-us-west-2-123456789012/eval-output/"
MODEL_PACKAGE_GROUP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/my-finetuned-model"


def _make_mock_agent_rft_job(output_model_package_arn=MODEL_PACKAGE_ARN):
    """Create a mock AgentRFTJob with output_model_package_arn."""
    job = MagicMock()
    job.job_name = "test-mtrl-job-123"
    job.job_arn = "arn:aws:sagemaker:us-west-2:123456789012:job/test-mtrl-job-123"
    job.job_status = "Completed"
    job.output_model_package_arn = output_model_package_arn
    return job


def _make_mock_mtrl_trainer(with_job=True):
    """Create a mock MultiTurnRLTrainer with _latest_job and _model_arn attributes."""
    from sagemaker.train.multi_turn_rl_trainer import MultiTurnRLTrainer

    trainer = MagicMock(spec=MultiTurnRLTrainer)
    trainer.__class__ = MultiTurnRLTrainer
    trainer._model_arn = BASE_MODEL_ARN
    trainer._model_name = BASE_MODEL_NAME

    if with_job:
        trainer._latest_job = _make_mock_agent_rft_job()
    else:
        trainer._latest_job = None

    # BaseTrainer attributes
    trainer._latest_training_job = None
    trainer.sagemaker_session = None
    trainer.role = None
    return trainer


# ============================================================
# Tests: Model Resolution with MTRLTrainer
# ============================================================

class TestModelResolutionWithMTRLTrainer:
    """Test that _ModelResolver correctly handles MultiTurnRLTrainer instances."""

    def test_resolve_mtrl_trainer_with_model_arn_and_job(self):
        """MTRLTrainer with _model_arn and _latest_job should resolve directly without API calls."""
        trainer = _make_mock_mtrl_trainer(with_job=True)

        resolver = _ModelResolver(sagemaker_session=None)
        result = resolver.resolve_model_info(trainer)

        assert isinstance(result, _ModelInfo)
        assert result.base_model_name == BASE_MODEL_NAME
        assert result.base_model_arn == BASE_MODEL_ARN
        assert result.source_model_package_arn == MODEL_PACKAGE_ARN
        assert result.model_type == _ModelType.FINE_TUNED

    def test_resolve_mtrl_trainer_with_model_arn_no_job(self):
        """MTRLTrainer with _model_arn but no _latest_job should resolve as JumpStart-like (no source_model_package_arn)."""
        trainer = _make_mock_mtrl_trainer(with_job=False)

        resolver = _ModelResolver(sagemaker_session=None)
        result = resolver.resolve_model_info(trainer)

        assert isinstance(result, _ModelInfo)
        assert result.base_model_name == BASE_MODEL_NAME
        assert result.base_model_arn == BASE_MODEL_ARN
        assert result.source_model_package_arn is None
        assert result.model_type == _ModelType.JUMPSTART

    def test_resolve_mtrl_trainer_without_model_arn_falls_back_to_job(self):
        """MTRLTrainer without _model_arn should fall back to _latest_job resolution."""
        trainer = _make_mock_mtrl_trainer(with_job=True)
        trainer._model_arn = None
        trainer._model_name = None

        # Mock the _resolve_model_package_arn to avoid actual API calls
        mock_info = _ModelInfo(
            base_model_name=BASE_MODEL_NAME,
            base_model_arn=BASE_MODEL_ARN,
            source_model_package_arn=MODEL_PACKAGE_ARN,
            model_type=_ModelType.FINE_TUNED,
            hub_content_name=BASE_MODEL_NAME,
            additional_metadata={},
        )

        resolver = _ModelResolver(sagemaker_session=None)
        with patch.object(resolver, '_resolve_model_package_arn', return_value=mock_info) as mock_resolve:
            result = resolver.resolve_model_info(trainer)

        mock_resolve.assert_called_once_with(MODEL_PACKAGE_ARN)
        assert result == mock_info

    def test_resolve_mtrl_trainer_no_job_no_model_arn_raises(self):
        """MTRLTrainer without _model_arn and without a completed job should raise ValueError."""
        trainer = _make_mock_mtrl_trainer(with_job=False)
        trainer._model_arn = None
        trainer._model_name = None

        resolver = _ModelResolver(sagemaker_session=None)
        with pytest.raises(ValueError, match="completed training job"):
            resolver.resolve_model_info(trainer)

    def test_resolve_agent_rft_job_directly(self):
        """AgentRFTJob passed directly (duck-typed) should resolve via output_model_package_arn."""
        job = _make_mock_agent_rft_job()

        mock_info = _ModelInfo(
            base_model_name=BASE_MODEL_NAME,
            base_model_arn=BASE_MODEL_ARN,
            source_model_package_arn=MODEL_PACKAGE_ARN,
            model_type=_ModelType.FINE_TUNED,
            hub_content_name=BASE_MODEL_NAME,
            additional_metadata={},
        )

        resolver = _ModelResolver(sagemaker_session=None)
        with patch.object(resolver, '_resolve_model_package_arn', return_value=mock_info) as mock_resolve:
            result = resolver.resolve_model_info(job)

        mock_resolve.assert_called_once_with(MODEL_PACKAGE_ARN)
        assert result == mock_info


# ============================================================
# Tests: BenchMarkEvaluator with MTRLTrainer
# ============================================================

class TestBenchmarkEvaluatorWithMTRLTrainer:
    """Test that BenchMarkEvaluator accepts MTRLTrainer as model input."""

    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_arn')
    def test_benchmark_evaluator_accepts_mtrl_trainer(self, mock_resolve_mp, mock_mlflow):
        """BenchMarkEvaluator should accept a MultiTurnRLTrainer with completed job."""
        from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

        mock_mlflow.return_value = MLFLOW_ARN

        Benchmark = get_benchmarks()
        trainer = _make_mock_mtrl_trainer(with_job=True)

        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            model=trainer,
            s3_output_path=S3_OUTPUT,
            model_package_group=MODEL_PACKAGE_GROUP_ARN,
            region="us-west-2",
        )

        assert evaluator is not None
        assert evaluator.model is trainer
        assert evaluator._base_model_name == BASE_MODEL_NAME
        assert evaluator._base_model_arn == BASE_MODEL_ARN
        assert evaluator._source_model_package_arn == MODEL_PACKAGE_ARN

    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    def test_benchmark_evaluator_rejects_mtrl_trainer_without_job(self, mock_mlflow):
        """BenchMarkEvaluator should reject a MultiTurnRLTrainer without a completed job or _model_arn."""
        from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

        mock_mlflow.return_value = MLFLOW_ARN

        Benchmark = get_benchmarks()
        trainer = _make_mock_mtrl_trainer(with_job=False)
        trainer._model_arn = None
        trainer._model_name = None

        with pytest.raises(ValueError, match="Failed to resolve model"):
            BenchMarkEvaluator(
                benchmark=Benchmark.MMLU,
                model=trainer,
                s3_output_path=S3_OUTPUT,
                model_package_group=MODEL_PACKAGE_GROUP_ARN,
                region="us-west-2",
            )


# ============================================================
# Tests: CustomScorerEvaluator with MTRLTrainer
# ============================================================

class TestCustomScorerEvaluatorWithMTRLTrainer:
    """Test that CustomScorerEvaluator accepts MTRLTrainer as model input."""

    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_arn')
    def test_custom_scorer_evaluator_accepts_mtrl_trainer(self, mock_resolve_mp, mock_mlflow):
        """CustomScorerEvaluator should accept a MultiTurnRLTrainer with completed job."""
        from sagemaker.train.evaluate import CustomScorerEvaluator, get_builtin_metrics

        mock_mlflow.return_value = MLFLOW_ARN

        trainer = _make_mock_mtrl_trainer(with_job=True)

        evaluator = CustomScorerEvaluator(
            evaluator="arn:aws:sagemaker:us-west-2:123456789012:hub-content/myhub/JsonDoc/eval-test/0.0.1",
            dataset="s3://my-bucket/dataset.jsonl",
            model=trainer,
            s3_output_path=S3_OUTPUT,
            model_package_group=MODEL_PACKAGE_GROUP_ARN,
            region="us-west-2",
        )

        assert evaluator is not None
        assert evaluator.model is trainer
        assert evaluator._base_model_name == BASE_MODEL_NAME
        assert evaluator._base_model_arn == BASE_MODEL_ARN
        assert evaluator._source_model_package_arn == MODEL_PACKAGE_ARN

    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    def test_custom_scorer_evaluator_rejects_mtrl_trainer_without_job(self, mock_mlflow):
        """CustomScorerEvaluator should reject a MultiTurnRLTrainer without completed job or _model_arn."""
        from sagemaker.train.evaluate import CustomScorerEvaluator

        mock_mlflow.return_value = MLFLOW_ARN

        trainer = _make_mock_mtrl_trainer(with_job=False)
        trainer._model_arn = None
        trainer._model_name = None

        with pytest.raises(ValueError, match="Failed to resolve model"):
            CustomScorerEvaluator(
                evaluator="arn:aws:sagemaker:us-west-2:123456789012:hub-content/myhub/JsonDoc/eval-test/0.0.1",
                dataset="s3://my-bucket/dataset.jsonl",
                model=trainer,
                s3_output_path=S3_OUTPUT,
                model_package_group=MODEL_PACKAGE_GROUP_ARN,
                region="us-west-2",
            )


# ============================================================
# Tests: Evaluate submission (mock pipeline start)
# ============================================================

class TestEvaluateSubmissionWithMTRLTrainer:
    """Test that evaluate() can be called successfully when model is an MTRLTrainer."""

    @patch('sagemaker.train.evaluate.base_evaluator.resolve_and_validate_role', side_effect=lambda provided_role, **kwargs: provided_role)
    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_arn')
    @patch('sagemaker.train.evaluate.base_evaluator.BaseEvaluator._get_or_create_artifact_arn')
    @patch('sagemaker.train.evaluate.execution.EvaluationPipelineExecution.start')
    @patch('sagemaker.train.evaluate.benchmark_evaluator.BenchMarkEvaluator.hyperparameters', new_callable=PropertyMock)
    def test_benchmark_evaluate_submission_with_mtrl_trainer(
        self, mock_hyperparams, mock_start, mock_artifact, mock_resolve_mp, mock_mlflow, mock_role
    ):
        """BenchMarkEvaluator.evaluate() should successfully submit when model is MTRLTrainer."""
        from sagemaker.train.evaluate import BenchMarkEvaluator, get_benchmarks

        mock_mlflow.return_value = MLFLOW_ARN

        # Mock hyperparameters
        hp_mock = MagicMock()
        hp_mock.to_dict.return_value = {"max_new_tokens": 256, "temperature": 0.7}
        mock_hyperparams.return_value = hp_mock

        # Mock artifact creation
        mock_artifact.return_value = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact"

        # Mock pipeline execution start
        mock_execution = MagicMock()
        mock_execution.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/eval/execution/abc123"
        mock_start.return_value = mock_execution

        Benchmark = get_benchmarks()
        trainer = _make_mock_mtrl_trainer(with_job=True)

        evaluator = BenchMarkEvaluator(
            benchmark=Benchmark.MMLU,
            model=trainer,
            s3_output_path=S3_OUTPUT,
            model_package_group=MODEL_PACKAGE_GROUP_ARN,
            region="us-west-2",
            role="arn:aws:iam::123456789012:role/TestRole",
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        mock_start.assert_called_once()

    @patch('sagemaker.train.evaluate.base_evaluator.resolve_and_validate_role', side_effect=lambda provided_role, **kwargs: provided_role)
    @patch('sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn')
    @patch('sagemaker.train.common_utils.model_resolution._ModelResolver._resolve_model_package_arn')
    @patch('sagemaker.train.evaluate.base_evaluator.BaseEvaluator._get_or_create_artifact_arn')
    @patch('sagemaker.train.evaluate.execution.EvaluationPipelineExecution.start')
    @patch('sagemaker.train.evaluate.custom_scorer_evaluator.CustomScorerEvaluator.hyperparameters', new_callable=PropertyMock)
    def test_custom_scorer_evaluate_submission_with_mtrl_trainer(
        self, mock_hyperparams, mock_start, mock_artifact, mock_resolve_mp, mock_mlflow, mock_role
    ):
        """CustomScorerEvaluator.evaluate() should successfully submit when model is MTRLTrainer."""
        from sagemaker.train.evaluate import CustomScorerEvaluator

        mock_mlflow.return_value = MLFLOW_ARN

        # Mock hyperparameters
        hp_mock = MagicMock()
        hp_mock.to_dict.return_value = {"max_new_tokens": 256, "temperature": 0.7}
        mock_hyperparams.return_value = hp_mock

        # Mock artifact creation
        mock_artifact.return_value = "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact"

        # Mock pipeline execution start
        mock_execution = MagicMock()
        mock_execution.arn = "arn:aws:sagemaker:us-west-2:123456789012:pipeline/eval/execution/def456"
        mock_start.return_value = mock_execution

        trainer = _make_mock_mtrl_trainer(with_job=True)

        evaluator = CustomScorerEvaluator(
            evaluator="arn:aws:sagemaker:us-west-2:123456789012:hub-content/myhub/JsonDoc/eval-test/0.0.1",
            dataset="s3://my-bucket/dataset.jsonl",
            model=trainer,
            s3_output_path=S3_OUTPUT,
            model_package_group=MODEL_PACKAGE_GROUP_ARN,
            region="us-west-2",
            role="arn:aws:iam::123456789012:role/TestRole",
        )

        execution = evaluator.evaluate()

        assert execution is not None
        assert execution.arn is not None
        mock_start.assert_called_once()
