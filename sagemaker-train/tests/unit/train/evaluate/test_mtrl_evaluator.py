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

import pytest
from unittest.mock import patch, MagicMock

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
        evaluator._start_mtrl_execution = MultiTurnRLEvaluator._start_mtrl_execution.__get__(
            evaluator
        )
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
            "PipelineExecutionArn": (
                f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-1"
            )
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

        evaluator._start_mtrl_execution(
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
            pipeline_definition="{}",
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
            pipeline_definition="{}",
            name="test",
            role_arn=ROLE,
            region=REGION,
        )

        mock_client.create_pipeline.assert_called_once()
        assert result.arn.endswith("exec-1")

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_lowercase_user_tags_sent_capitalized(self, mock_boto3_client, mock_pe_cls):
        """Lowercase tags must be converted before reaching boto3 CreatePipeline.

        This path uses raw boto3, which rejects the lowercase key/value form that the other
        evaluators pass to the pydantic-validated Pipeline.create.
        """
        evaluator = self._make_evaluator()
        evaluator.tags = [{"key": "sagemaker:project-id", "value": "p-12345"}]
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {"PipelineSummaries": []}
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": (
                f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-1"
            )
        }

        evaluator._start_mtrl_execution(
            pipeline_definition='{"Steps": []}',
            name="test-eval",
            role_arn=ROLE,
            region=REGION,
        )

        sent_tags = mock_client.create_pipeline.call_args.kwargs["Tags"]
        assert {"Key": "sagemaker:project-id", "Value": "p-12345"} in sent_tags
        # Every entry must use the capitalized form boto3 requires.
        for tag in sent_tags:
            assert set(tag.keys()) == {"Key", "Value"}

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_capitalized_user_tags_passed_through(self, mock_boto3_client, mock_pe_cls):
        """Tags already in the capitalized form must be preserved unchanged."""
        evaluator = self._make_evaluator()
        evaluator.tags = [{"Key": "sagemaker:project-id", "Value": "p-12345"}]
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {"PipelineSummaries": []}
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": (
                f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-1"
            )
        }

        evaluator._start_mtrl_execution(
            pipeline_definition='{"Steps": []}',
            name="test-eval",
            role_arn=ROLE,
            region=REGION,
        )

        sent_tags = mock_client.create_pipeline.call_args.kwargs["Tags"]
        assert {"Key": "sagemaker:project-id", "Value": "p-12345"} in sent_tags

    @patch("sagemaker.core.resources.PipelineExecution")
    @patch("boto3.client")
    def test_no_user_tags_keeps_only_discovery_tag(self, mock_boto3_client, mock_pe_cls):
        """Omitting tags must leave the evaluation discovery tag as the only tag."""
        evaluator = self._make_evaluator()
        evaluator.tags = None
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client
        mock_client.list_pipelines.return_value = {"PipelineSummaries": []}
        mock_client.start_pipeline_execution.return_value = {
            "PipelineExecutionArn": (
                f"arn:aws:sagemaker:us-west-2:123:pipeline/{PIPELINE_PREFIX}/execution/exec-1"
            )
        }

        evaluator._start_mtrl_execution(
            pipeline_definition='{"Steps": []}',
            name="test-eval",
            role_arn=ROLE,
            region=REGION,
        )

        sent_tags = mock_client.create_pipeline.call_args.kwargs["Tags"]
        assert sent_tags == [{"Key": "SagemakerModelEvaluation", "Value": "true"}]


class TestTagsWithCapitalizedKeys:
    """Tests for the tag-casing conversion used by the MTRL evaluator paths."""

    def test_converts_lowercase(self):
        from sagemaker.train.evaluate.multi_turn_rl_evaluator import _tags_with_capitalized_keys

        assert _tags_with_capitalized_keys([{"key": "a", "value": "b"}]) == [
            {"Key": "a", "Value": "b"}
        ]

    def test_preserves_capitalized(self):
        from sagemaker.train.evaluate.multi_turn_rl_evaluator import _tags_with_capitalized_keys

        assert _tags_with_capitalized_keys([{"Key": "a", "Value": "b"}]) == [
            {"Key": "a", "Value": "b"}
        ]

    def test_returns_empty_for_none(self):
        from sagemaker.train.evaluate.multi_turn_rl_evaluator import _tags_with_capitalized_keys

        assert _tags_with_capitalized_keys(None) == []

    def test_skips_entries_missing_key_or_value(self):
        from sagemaker.train.evaluate.multi_turn_rl_evaluator import _tags_with_capitalized_keys

        assert _tags_with_capitalized_keys([{"nope": "x"}, {"key": "ok", "value": "1"}]) == [
            {"Key": "ok", "Value": "1"}
        ]


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

        with patch.object(resolver, "_resolve_model_package_arn") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                base_model_name=BASE_MODEL,
                base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/test",
                source_model_package_arn=SOURCE_MP_ARN,
            )
            resolver.resolve_model_info(mock_trainer)

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

        with patch.object(resolver, "_resolve_model_package_arn") as mock_resolve:
            mock_resolve.return_value = MagicMock(
                base_model_name=BASE_MODEL,
                base_model_arn="arn:aws:sagemaker:us-west-2:aws:hub-content/test",
                source_model_package_arn=SOURCE_MP_ARN,
            )
            resolver.resolve_model_info(mock_trainer)

            mock_resolve.assert_called_once_with(SOURCE_MP_ARN)


# Mirrors the MTRLEvaluation override spec the hub publishes for MTRL models.
_HUB_MTRL_EVAL_SPEC = {
    "base_model_name": {"type": "string", "default": BASE_MODEL, "required": True},
    "model_name": {"type": "string", "default": BASE_MODEL, "required": True},
    "name": {"type": "string", "default": "mtrl-eval-abcde", "required": True},
    "eval_group_size": {"type": "integer", "min": 1, "max": 128, "default": 1},
    "pass_k_values": {"type": "array", "default": [1, 2, 4, 8, 16, 32]},
    "success_threshold": {"type": "float", "default": 1.0},
    "temperature": {"type": "float", "min": 0, "max": 2, "default": 0},
    "sampling_top_p": {"type": "float", "min": 0, "max": 1, "default": 1},
    "sampling_max_tokens": {"type": "integer", "min": 512, "max": 8192, "default": 4096},
    "rollout_max_concurrency": {"type": "integer", "min": 32, "max": 96, "default": 96},
    "rollout_timeout": {"type": "float", "min": 300, "max": 86400, "default": 600.0},
    "rollout_max_retries": {"type": "integer", "min": 1, "max": 10, "default": 3},
}

_AWS_CONTEXT = {"region": REGION, "account_id": "123456789012", "role_arn": ROLE}


@patch(
    "sagemaker.train.common_utils.recipe_utils._get_evaluation_override_params",
    return_value=_HUB_MTRL_EVAL_SPEC,
)
@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestHyperparameterForwarding:
    """The JobConfigDocument must carry user-changed values under the recipe's names."""

    def _create(self, mock_artifact, mock_resolve, **kwargs):
        from sagemaker.train.common_utils.model_resolution import _ModelInfo, _ModelType

        mock_resolve.return_value = _ModelInfo(
            base_model_name=BASE_MODEL,
            base_model_arn=(
                f"arn:aws:sagemaker:{REGION}:aws:hub-content/SageMakerPublicHub/Model/"
                f"{BASE_MODEL}/1.0.0"
            ),
            source_model_package_arn=None,
            model_type=_ModelType.JUMPSTART,
            hub_content_name=BASE_MODEL,
            additional_metadata={},
        )
        mock_artifact.get_all.return_value = iter([])
        session = MagicMock()
        session.boto_region_name = REGION
        session.boto_session.region_name = REGION
        session.sagemaker_config = {}
        evaluator = MultiTurnRLEvaluator(
            model=BASE_MODEL,
            agent_config=AGENT_ARN,
            dataset=DATASET,
            s3_output_path=OUTPUT,
            sagemaker_session=session,
            region=REGION,
            **kwargs,
        )
        evaluator._resolve_agent_arn()
        return evaluator

    @staticmethod
    def _sent_hyperparameters(evaluator):
        import json

        ctx = evaluator._build_template_context(_AWS_CONTEXT, {}, None)
        for key in ("job_config_document_str", "job_config_document_ft_str"):
            doc = json.loads(ctx[key])
            yield doc["EvaluationConfig"].get("HyperParameters")

    def test_defaults_send_no_hyperparameters(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(mock_artifact, mock_resolve)
        assert list(self._sent_hyperparameters(evaluator)) == [None, None]

    def test_direct_assignment_uses_recipe_names(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(mock_artifact, mock_resolve)
        evaluator.hyperparameters.sampling_max_tokens = 8192
        evaluator.hyperparameters.sampling_top_p = 0.9
        evaluator.hyperparameters.temperature = 0.7
        evaluator.hyperparameters.rollout_timeout = 1200.0
        evaluator.hyperparameters.rollout_max_concurrency = 48
        evaluator.hyperparameters.rollout_max_retries = 5

        expected = {
            "sampling_max_tokens": "8192",
            "sampling_top_p": "0.9",
            "temperature": "0.7",
            "rollout_timeout": "1200.0",
            "rollout_max_concurrency": "48",
            "rollout_max_retries": "5",
        }
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == expected

    def test_list_values_are_forwarded(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(mock_artifact, mock_resolve)
        evaluator.hyperparameters.pass_k_values = [1, 4]
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == {"pass_k_values": "[1, 4]"}

    def test_overrides_dict_is_forwarded(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(
            mock_artifact,
            mock_resolve,
            overrides={"inference": {"sampling_max_tokens": 6000, "eval_group_size": 4}},
        )
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == {"sampling_max_tokens": "6000", "eval_group_size": "4"}

    def test_list_value_in_overrides_is_forwarded(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(
            mock_artifact,
            mock_resolve,
            overrides={"inference": {"pass_k_values": [1, 4], "sampling_max_tokens": 6000}},
        )
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == {"pass_k_values": "[1, 4]", "sampling_max_tokens": "6000"}

    def test_list_value_in_recipe_is_forwarded(self, mock_artifact, mock_resolve, _, tmp_path):
        recipe = tmp_path / "recipe.yaml"
        recipe.write_text("inference:\n  pass_k_values: [1, 2]\n  eval_group_size: 2\n")
        evaluator = self._create(mock_artifact, mock_resolve, recipe=str(recipe))
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == {"pass_k_values": "[1, 2]", "eval_group_size": "2"}

    def test_value_equal_to_default_is_not_sent(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(mock_artifact, mock_resolve)
        evaluator.hyperparameters.sampling_max_tokens = 4096
        evaluator.hyperparameters.eval_group_size = 8
        for sent in self._sent_hyperparameters(evaluator):
            assert sent == {"eval_group_size": "8"}

    def test_range_is_validated_against_recipe(self, mock_artifact, mock_resolve, _):
        evaluator = self._create(mock_artifact, mock_resolve)
        with pytest.raises(ValueError, match="sampling_max_tokens must be <= 8192"):
            evaluator.hyperparameters.sampling_max_tokens = 16384

    def test_hub_failure_sends_no_hyperparameters(self, mock_artifact, mock_resolve, mock_hub):
        mock_hub.side_effect = RuntimeError("hub unavailable")
        evaluator = self._create(mock_artifact, mock_resolve)
        assert list(self._sent_hyperparameters(evaluator)) == [None, None]
