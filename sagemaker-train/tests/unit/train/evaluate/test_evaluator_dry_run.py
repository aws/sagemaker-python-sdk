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
"""Unit tests for dry_run=True on evaluator types.

Tests that:
1. dry_run=True returns None (no execution submitted)
2. dry_run=True does NOT call _start_execution / upload to S3
3. _get_aws_execution_context is called during evaluate()

Strategy: mock only at the evaluate() method's two key boundaries:
  - _get_aws_execution_context (the IAM/session resolver — avoids network)
  - _start_execution / _start_mtrl_execution (the pipeline submit — we assert not called)
For evaluators that call S3/Hub before those boundaries, we mock the specific
network-calling helpers rather than the entire evaluate flow.
"""
from __future__ import absolute_import

from unittest.mock import Mock, patch, PropertyMock

import pytest

from sagemaker.train.common_utils.model_resolution import _ModelInfo, _ModelType
from sagemaker.train.evaluate.benchmark_evaluator import BenchMarkEvaluator
from sagemaker.train.evaluate.constants import EvalType
from sagemaker.train.evaluate.custom_scorer_evaluator import CustomScorerEvaluator
from sagemaker.train.evaluate.inspect_ai_evaluator import InspectAIEvaluator
from sagemaker.train.evaluate.llm_as_judge_evaluator import LLMAsJudgeEvaluator
from sagemaker.train.evaluate.multi_turn_rl_evaluator import MultiTurnRLEvaluator


DEFAULT_REGION = "us-east-1"
DEFAULT_ROLE = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_MODEL = "amazon-nova-lite-v1"
DEFAULT_S3_OUTPUT = "s3://test-bucket/eval-output/"
DEFAULT_BENCHMARKS_PATH = "s3://test-bucket/benchmarks/"
DEFAULT_MLFLOW_ARN = "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/test-server"
DEFAULT_MODEL_PACKAGE_GROUP_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-group"
)
DEFAULT_BASE_MODEL_ARN = (
    "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/amazon-nova-lite-v1/1.0.0"
)
DEFAULT_ARTIFACT_ARN = "arn:aws:sagemaker:us-east-1:123456789012:artifact/test-artifact"
DEFAULT_DATASET = "s3://test-bucket/dataset.jsonl"


def _mock_session():
    session = Mock()
    session.boto_region_name = DEFAULT_REGION
    session.boto_session = Mock()
    session.boto_session.region_name = DEFAULT_REGION
    session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    session.sagemaker_config = {}
    session.default_bucket.return_value = "test-bucket"
    session.default_bucket_prefix = None
    return session


def _mock_model_info():
    return _ModelInfo(
        base_model_name=DEFAULT_MODEL,
        base_model_arn=DEFAULT_BASE_MODEL_ARN,
        source_model_package_arn=None,
        model_type=_ModelType.JUMPSTART,
        hub_content_name="amazon-nova-lite-v1",
        additional_metadata={
            "bedrock_model_id": "us.amazon.nova-lite-v1:0",
            "resolved_model_artifact_arn": DEFAULT_ARTIFACT_ARN,
        },
        s3_model_path="s3://test-bucket/models/amazon-nova-lite-v1/output",
    )


def _aws_context():
    return {
        "role_arn": DEFAULT_ROLE,
        "region": DEFAULT_REGION,
        "account_id": "123456789012",
    }


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIDryRun:

    def _create(self, mock_artifact, mock_resolve):

        mock_resolve.return_value = _mock_model_info()
        mock_artifact.get_all.return_value = iter([])
        inst = Mock()
        inst.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = inst

        return InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_dry_run_returns_none_and_skips_s3(self, mock_uploader, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)

        with patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()):
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_uploader.upload_string_as_file_body.assert_not_called()

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_dry_run_does_not_start_execution(self, mock_uploader, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            evaluator.evaluate(dry_run=True)

        mock_start.assert_not_called()

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_dry_run_passes_flag(self, mock_uploader, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)

        with patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()) as ctx:
            evaluator.evaluate(dry_run=True)

        ctx.assert_called_once_with()

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_default_starts_execution(self, mock_uploader, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)
        mock_exec = Mock()

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_start_execution", return_value=mock_exec) as mock_start,
        ):
            result = evaluator.evaluate()

        mock_start.assert_called_once()
        assert result is mock_exec


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestMultiTurnRLDryRun:

    def _create(self, mock_artifact, mock_resolve):

        mock_resolve.return_value = _mock_model_info()
        mock_artifact.get_all.return_value = iter([])
        inst = Mock()
        inst.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = inst

        return MultiTurnRLEvaluator(
            model=DEFAULT_MODEL,
            agent_config="arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test-agent-aBcDeFgHiJ",
            dataset="s3://test-bucket/prompts.parquet",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

    def test_dry_run_returns_none(self, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_resolve_trainer_defaults"),
            patch.object(evaluator, "_resolve_agent_arn"),
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_resolve_model_artifacts", return_value={}),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            patch.object(evaluator, "_build_template_context", return_value={}),
            patch.object(evaluator, "_select_mtrl_template", return_value="{}"),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
            patch.object(evaluator, "_start_mtrl_execution") as mock_start,
        ):
            evaluator._agent_arn_resolved = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test"
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_start.assert_not_called()

    def test_dry_run_passes_flag(self, mock_artifact, mock_resolve):
        evaluator = self._create(mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_resolve_trainer_defaults"),
            patch.object(evaluator, "_resolve_agent_arn"),
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()) as ctx,
            patch.object(evaluator, "_resolve_model_artifacts", return_value={}),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            patch.object(evaluator, "_build_template_context", return_value={}),
            patch.object(evaluator, "_select_mtrl_template", return_value="{}"),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
        ):
            evaluator._agent_arn_resolved = "arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/test"
            evaluator.evaluate(dry_run=True)

        ctx.assert_called_once_with()


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
@patch("sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn")
class TestBenchmarkDryRun:

    def _create(self, mock_mlflow, mock_artifact, mock_resolve):

        mock_resolve.return_value = _mock_model_info()
        mock_mlflow.return_value = DEFAULT_MLFLOW_ARN
        mock_artifact.get_all.return_value = iter([])
        inst = Mock()
        inst.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = inst

        return BenchMarkEvaluator(
            model=DEFAULT_MODEL,
            benchmark="mmlu",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

    def test_dry_run_returns_none(self, mock_mlflow, mock_artifact, mock_resolve):
        evaluator = self._create(mock_mlflow, mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_resolve_model_artifacts", return_value={
                "resolved_model_artifact_arn": DEFAULT_ARTIFACT_ARN,
            }),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            patch.object(evaluator, "_get_base_template_context", return_value={"evaluate_base_model": False}),
            patch.object(evaluator, "_get_benchmark_template_additions", return_value={"task": "mmlu"}),
            patch.object(evaluator, "_add_vpc_and_kms_to_context", side_effect=lambda c: c),
            patch.object(evaluator, "_select_template", return_value="{}"),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_start.assert_not_called()


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
@patch("sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn")
class TestCustomScorerDryRun:

    def _create(self, mock_mlflow, mock_artifact, mock_resolve):

        mock_resolve.return_value = _mock_model_info()
        mock_mlflow.return_value = DEFAULT_MLFLOW_ARN
        mock_artifact.get_all.return_value = iter([])
        inst = Mock()
        inst.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = inst

        return CustomScorerEvaluator(
            model=DEFAULT_MODEL,
            dataset=DEFAULT_DATASET,
            evaluator="arn:aws:lambda:us-east-1:123456789012:function:my-scorer",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

    def test_dry_run_returns_none(self, mock_mlflow, mock_artifact, mock_resolve):
        evaluator = self._create(mock_mlflow, mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_resolve_model_artifacts", return_value={
                "resolved_model_artifact_arn": DEFAULT_ARTIFACT_ARN,
            }),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            patch.object(evaluator, "_resolve_evaluator_config", return_value={
                "evaluator_arn": "arn:aws:lambda:us-east-1:123456789012:function:my-scorer",
                "preset_reward_function": None,
            }),
            patch.object(evaluator, "_get_base_template_context", return_value={"evaluate_base_model": False}),
            patch.object(evaluator, "_add_vpc_and_kms_to_context", side_effect=lambda c: c),
            patch.object(evaluator, "_select_template", return_value="{}"),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_start.assert_not_called()


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
@patch("sagemaker.train.evaluate.base_evaluator._resolve_mlflow_resource_arn")
class TestLLMAsJudgeDryRun:

    def _create(self, mock_mlflow, mock_artifact, mock_resolve):

        mock_resolve.return_value = _mock_model_info()
        mock_mlflow.return_value = DEFAULT_MLFLOW_ARN
        mock_artifact.get_all.return_value = iter([])
        inst = Mock()
        inst.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = inst

        return LLMAsJudgeEvaluator(
            model=DEFAULT_MODEL,
            evaluator_model="amazon.nova-pro-v1:0",
            dataset=DEFAULT_DATASET,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

    def test_dry_run_returns_none(self, mock_mlflow, mock_artifact, mock_resolve):
        evaluator = self._create(mock_mlflow, mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_resolve_model_artifacts", return_value={
                "resolved_model_artifact_arn": DEFAULT_ARTIFACT_ARN,
            }),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            patch.object(evaluator, "_get_base_template_context", return_value={"evaluate_base_model": False}),
            patch.object(evaluator, "_add_vpc_and_kms_to_context", side_effect=lambda c: c),
            patch.object(evaluator, "_upload_benchmark_and_dataset", return_value="s3://test-bucket/benchmarks/converted"),
            patch.object(evaluator, "_build_inspectai_config", return_value={}),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_start.assert_not_called()

    def test_dry_run_standard_path_returns_none(self, mock_mlflow, mock_artifact, mock_resolve):
        """Test dry_run on the standard (non-InspectAI) LLMAJ path."""
        evaluator = self._create(mock_mlflow, mock_artifact, mock_resolve)

        with (
            patch.object(evaluator, "_get_aws_execution_context", return_value=_aws_context()),
            patch.object(evaluator, "_resolve_model_artifacts", return_value={
                "resolved_model_artifact_arn": DEFAULT_ARTIFACT_ARN,
            }),
            patch.object(evaluator, "_get_model_package_group_arn", return_value=DEFAULT_MODEL_PACKAGE_GROUP_ARN),
            # Force the standard (non-InspectAI) path
            patch.object(evaluator, "_should_use_inspectai_path", return_value=False),
            patch.object(evaluator, "_get_base_template_context", return_value={"evaluate_base_model": False}),
            patch.object(evaluator, "_add_vpc_and_kms_to_context", side_effect=lambda c: c),
            patch.object(evaluator, "_render_pipeline_definition", return_value='{"Steps": []}'),
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            result = evaluator.evaluate(dry_run=True)

        assert result is None
        mock_start.assert_not_called()
