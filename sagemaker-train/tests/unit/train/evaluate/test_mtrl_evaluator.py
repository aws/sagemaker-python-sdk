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
"""Unit tests for MultiTurnRLEvaluator — pipeline search and model resolution."""
from __future__ import absolute_import

import json
import pytest
from unittest.mock import patch, MagicMock, Mock

from sagemaker.train.evaluate.multi_turn_rl_evaluator import MultiTurnRLEvaluator
from sagemaker.train.evaluate.constants import EvalType, _get_pipeline_name_prefix


# --- Constants ---
REGION = "us-west-2"
ROLE = "arn:aws:iam::123456789012:role/test-role"
AGENT_ARN = "arn:aws:bedrock-agentcore:us-west-2:123456789012:runtime/test-agent-aBcDeFgHiJ"
DATASET = "s3://test-bucket/prompts.parquet"
OUTPUT = "s3://test-bucket/output/"
MLFLOW_ARN = "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server"
BASE_MODEL = "huggingface-reasoning-qwen3-32b"
MPG_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-mpg"
SOURCE_MP_ARN = "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-mpg/1"
PIPELINE_PREFIX = "SagemakerEvaluation-MTRLEvaluation"


class TestStartMtrlExecution:
    """Tests for _start_mtrl_execution pipeline search and create/update logic."""

    def _make_evaluator(self):
        """Create a mock evaluator with the real _start_mtrl_execution method bound."""
        evaluator = MagicMock()
        evaluator.s3_output_path = OUTPUT
        evaluator._start_mtrl_execution = MultiTurnRLEvaluator._start_mtrl_execution.__get__(evaluator)
        return evaluator

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_creates_pipeline_when_not_found(self, mock_boto3_client, mock_pe_cls):
        """Test that a new pipeline is created when list_pipelines finds nothing."""
        evaluator = self._make_evaluator()
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {"PipelineSummaries": []}
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-1"
        }

        result = evaluator._start_mtrl_execution(
            pipeline_definition='{"Steps": []}',
            name="test-eval",
            role_arn=ROLE,
            region=REGION,
        )

        mock_client.create_pipeline.assert_called_once()
        assert result.arn.endswith("exec-1")

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_updates_pipeline_when_found(self, mock_boto3_client, mock_pe_cls):
        """Test that existing pipeline is updated when list_pipelines finds one."""
        evaluator = self._make_evaluator()
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {
            "PipelineSummaries": [{"PipelineName": PIPELINE_PREFIX}]
        }
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-2"
        }

        result = evaluator._start_mtrl_execution(
            pipeline_definition='{"Steps": []}',
            name="test-eval",
            role_arn=ROLE,
            region=REGION,
        )

        mock_client.update_pipeline.assert_called_once_with(
            PipelineName=PIPELINE_PREFIX,
            PipelineDefinition='{"Steps": []}',
            RoleArn=ROLE,
        )
        mock_client.create_pipeline.assert_not_called()

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_uses_correct_pipeline_prefix(self, mock_boto3_client, mock_pe_cls):
        """Test that the pipeline prefix matches the constants module."""
        evaluator = self._make_evaluator()
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {"PipelineSummaries": []}
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test/execution/exec-1"
        }

        evaluator._start_mtrl_execution(
            pipeline_definition='{}',
            name="test",
            role_arn=ROLE,
            region=REGION,
        )

        expected_prefix = _get_pipeline_name_prefix(EvalType.MTRL)
        assert expected_prefix == "SagemakerEvaluation-MTRLEvaluation"
        mock_client.list_pipelines.assert_called_once_with(PipelineNamePrefix=expected_prefix)

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_find_existing_pipeline_search_fails(self, mock_boto3_client, mock_pe_cls):
        """Test that pipeline is created when search raises an exception."""
        evaluator = self._make_evaluator()
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.side_effect = Exception("AccessDenied")
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": "arn:aws:sagemaker:us-west-2:123:pipeline/test/execution/exec-1"
        }

        result = evaluator._start_mtrl_execution(
            pipeline_definition='{}',
            name="test",
            role_arn=ROLE,
            region=REGION,
        )

        mock_client.create_pipeline.assert_called_once()
        assert result.arn.endswith("exec-1")


class TestModelResolutionWithLatestJob:
    """Tests for model resolution handling _latest_job (AgentRFT flow)."""

    def test_resolve_base_model_with_latest_job(self):
        """Test that _resolve_base_model handles BaseTrainer with _latest_job."""
        from sagemaker.train.common_utils.model_resolution import _ModelResolver
        from sagemaker.train.base_trainer import BaseTrainer

        # Create a mock trainer with _latest_job
        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_trainer._latest_job = MagicMock()
        mock_trainer._latest_job.output_model_package_arn = SOURCE_MP_ARN
        # Ensure _latest_training_job is not set
        mock_trainer._latest_training_job = None

        resolver = _ModelResolver()

        with patch.object(resolver, '_resolve_model_package_arn') as mock_resolve:
            mock_resolve.return_value = MagicMock(
                base_model_name=BASE_MODEL,
                base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/test",
                source_model_package_arn=SOURCE_MP_ARN,
            )
            result = resolver.resolve_model_info(mock_trainer)

            mock_resolve.assert_called_once_with(SOURCE_MP_ARN)

    def test_resolve_base_model_latest_job_no_output_arn(self):
        """Test that resolution falls through when _latest_job has no output ARN."""
        from sagemaker.train.common_utils.model_resolution import _ModelResolver
        from sagemaker.train.base_trainer import BaseTrainer

        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_trainer._latest_job = MagicMock()
        mock_trainer._latest_job.output_model_package_arn = None
        # No _latest_training_job either
        del mock_trainer._latest_training_job

        resolver = _ModelResolver()

        with pytest.raises(ValueError, match="must have completed training job"):
            resolver.resolve_model_info(mock_trainer)

    def test_resolve_base_model_no_latest_job_uses_training_job(self):
        """Test fallback to _latest_training_job when _latest_job is not set."""
        from sagemaker.train.common_utils.model_resolution import _ModelResolver
        from sagemaker.train.base_trainer import BaseTrainer

        mock_trainer = MagicMock(spec=BaseTrainer)
        mock_trainer._latest_job = None
        mock_trainer._latest_training_job = MagicMock()
        mock_trainer._latest_training_job.output_model_package_arn = SOURCE_MP_ARN

        resolver = _ModelResolver()

        with patch.object(resolver, '_resolve_model_package_arn') as mock_resolve:
            mock_resolve.return_value = MagicMock(
                base_model_name=BASE_MODEL,
                base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/test",
                source_model_package_arn=SOURCE_MP_ARN,
            )
            result = resolver.resolve_model_info(mock_trainer)

            mock_resolve.assert_called_once_with(SOURCE_MP_ARN)
