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
from __future__ import absolute_import

import json
from unittest.mock import Mock, patch

import pytest

from sagemaker.train.evaluate.llmaj_inference_benchmark import (
    convert_dataset_to_inspectai_format,
    generate_benchmark_files,
)
from sagemaker.train.evaluate.llm_as_judge_evaluator import (
    _resolve_bedrock_model_id,
    _REGION_TO_BEDROCK_PREFIX,
)
from sagemaker.train.common_utils.model_aliases import NOVA_BEDROCK_MODEL_IDS
from sagemaker.train.evaluate.pipeline_templates import LLMAJ_INSPECTAI_TEMPLATE


# Test constants
DEFAULT_REGION = "us-east-1"
DEFAULT_ROLE = "arn:aws:iam::123456789012:role/test-role"
DEFAULT_MODEL = "nova-textgeneration-lite"
DEFAULT_DATASET = "s3://test-bucket/dataset.jsonl"
DEFAULT_S3_OUTPUT = "s3://test-bucket/outputs/"
DEFAULT_MLFLOW_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:mlflow-tracking-server/test-server"
)
DEFAULT_MODEL_PACKAGE_GROUP_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:model-package-group/test-group"
)
DEFAULT_BASE_MODEL_ARN = (
    "arn:aws:sagemaker:us-east-1:aws:hub-content/SageMakerPublicHub/Model/"
    "nova-textgeneration-lite/1.0.0"
)
DEFAULT_ARTIFACT_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:artifact/test-artifact"
)
DEFAULT_EVALUATOR_MODEL = "amazon.nova-pro-v1:0"
DEFAULT_MODEL_PACKAGE_ARN = (
    "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-pkg/1"
)


def _create_evaluator(
    mock_resolve,
    mock_artifact,
    base_model_name=DEFAULT_MODEL,
    source_model_package_arn=None,
):
    """Create a minimally-mocked LLMAsJudgeEvaluator for unit testing.

    Args:
        mock_resolve: Active mock for model_resolution._resolve_base_model
        mock_artifact: Active mock for sagemaker.core.resources.Artifact
        base_model_name: Model name to use for resolution (default: non-Nova model)
        source_model_package_arn: Optional model package ARN to set on resolved info
    """
    mock_info = Mock()
    mock_info.base_model_name = base_model_name
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = source_model_package_arn
    mock_resolve.return_value = mock_info

    mock_artifact.get_all.return_value = iter([])
    mock_artifact_instance = Mock()
    mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
    mock_artifact.create.return_value = mock_artifact_instance

    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE

    from sagemaker.train.evaluate.llm_as_judge_evaluator import (
        LLMAsJudgeEvaluator,
    )

    evaluator = LLMAsJudgeEvaluator(
        evaluator_model=DEFAULT_EVALUATOR_MODEL,
        dataset=DEFAULT_DATASET,
        model=DEFAULT_MODEL,
        s3_output_path=DEFAULT_S3_OUTPUT,
        mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
        model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
        sagemaker_session=mock_session,
    )
    return evaluator


class TestShouldUseInspectaiPath:
    """Tests for _should_use_inspectai_path routing decision."""

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_should_use_inspectai_path_jumpstart_model(
        self, mock_resolve, mock_artifact
    ):
        """Non-Nova JumpStart model uses existing ServerlessJobConfig path."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="llama3-2-1b-instruct",
            source_model_package_arn=None,
        )
        assert evaluator._should_use_inspectai_path() is False

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_should_use_inspectai_path_nova_model_package(
        self, mock_resolve, mock_artifact
    ):
        """Nova fine-tuned model (model package ARN) routes to InspectAI path."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=DEFAULT_MODEL_PACKAGE_ARN,
        )
        assert evaluator._should_use_inspectai_path() is True

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_should_use_inspectai_path_non_nova_model_package(
        self, mock_resolve, mock_artifact
    ):
        """Non-Nova fine-tuned model (model package ARN) uses existing path."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="llama3-2-1b-instruct",
            source_model_package_arn=DEFAULT_MODEL_PACKAGE_ARN,
        )
        assert evaluator._should_use_inspectai_path() is False

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_should_use_inspectai_path_nova_model(
        self, mock_resolve, mock_artifact
    ):
        """Nova JumpStart model auto-routes to InspectAI+Bedrock path."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=None,
        )
        assert evaluator._should_use_inspectai_path() is True


class TestGenerateBenchmarkFiles:
    """Tests for generate_benchmark_files() output validity."""

    def test_generate_benchmark_files_valid_python(self):
        """Generated benchmark .py file must be compilable Python."""
        files = generate_benchmark_files()
        py_content = files["inference_only.py"]
        # compile() raises SyntaxError on invalid Python
        compile(py_content, "inference_only.py", "exec")

    def test_generate_benchmark_files_has_pyproject(self):
        """Generated files must include pyproject.toml with [project] section."""
        files = generate_benchmark_files()
        assert "pyproject.toml" in files
        assert "[project]" in files["pyproject.toml"]


class TestConvertDataset:
    """Tests for convert_dataset_to_inspectai_format."""

    @pytest.mark.parametrize(
        "prompt_text",
        [
            "What is machine learning?",
            "Explain quantum computing",
            "",
        ],
    )
    def test_convert_dataset_prompt_field(self, prompt_text):
        """Lines with 'prompt' field convert to {input, target} format."""
        line = json.dumps({"prompt": prompt_text})
        result = convert_dataset_to_inspectai_format(line)
        records = [json.loads(r) for r in result.strip().split("\n") if r.strip()]
        assert len(records) == 1
        assert records[0] == {"input": prompt_text, "target": ""}

    @pytest.mark.parametrize(
        "query_text",
        [
            "How does a transformer work?",
            "Summarize this document",
            "",
        ],
    )
    def test_convert_dataset_query_field(self, query_text):
        """Lines with 'query' field convert to {input, target} format."""
        line = json.dumps({"query": query_text})
        result = convert_dataset_to_inspectai_format(line)
        records = [json.loads(r) for r in result.strip().split("\n") if r.strip()]
        assert len(records) == 1
        assert records[0] == {"input": query_text, "target": ""}

    def test_convert_dataset_missing_field_raises(self):
        """Lines with neither 'prompt' nor 'query' raise ValueError."""
        line = json.dumps({"text": "some content"})
        with pytest.raises(ValueError, match="neither 'prompt' nor 'query'"):
            convert_dataset_to_inspectai_format(line)


class TestBuildInspectaiConfig:
    """Tests for _build_inspectai_config output."""

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_build_inspectai_config_bedrock_mode(self, mock_resolve, mock_artifact):
        """Config uses bedrock inference provider for Nova JumpStart models."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=None,
        )
        config = evaluator._build_inspectai_config(
            region="us-east-1",
            benchmark_s3_path="s3://bucket/benchmarks/uuid",
            output_s3_uri="s3://bucket/inference/uuid/inference_output.jsonl",
        )
        assert "bedrock" in config["inference_provider"]
        assert config["inference_provider"]["bedrock"]["model_id"] == (
            "us.amazon.nova-lite-v1:0"
        )
        assert config["inference_provider"]["bedrock"]["region"] == "us-east-1"

    @patch(
        "sagemaker.train.evaluate.llm_as_judge_evaluator."
        "LLMAsJudgeEvaluator._resolve_model_artifacts_for_endpoint"
    )
    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_build_inspectai_config_endpoint_mode(
        self, mock_resolve, mock_artifact, mock_resolve_artifacts
    ):
        """Config uses sagemaker_endpoint provider for model package."""
        mock_resolve_artifacts.return_value = (
            "s3://bucket/model.tar.gz",
            "123456789012.dkr.ecr.us-west-2.amazonaws.com/image:latest",
        )
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            source_model_package_arn=DEFAULT_MODEL_PACKAGE_ARN,
        )
        config = evaluator._build_inspectai_config(
            region="us-west-2",
            benchmark_s3_path="s3://bucket/benchmarks/uuid",
            output_s3_uri="s3://bucket/inference/uuid/inference_output.jsonl",
        )
        assert "sagemaker_endpoint" in config["inference_provider"]
        endpoint_cfg = config["inference_provider"]["sagemaker_endpoint"]
        assert endpoint_cfg["model_s3_uri"] == "s3://bucket/model.tar.gz"
        assert endpoint_cfg["cleanup_endpoint"] is True

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_build_inspectai_config_eval_defaults(
        self, mock_resolve, mock_artifact
    ):
        """Config contains expected eval defaults for rate limiting."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=None,
        )
        config = evaluator._build_inspectai_config(
            region="us-east-1",
            benchmark_s3_path="s3://bucket/benchmarks/uuid",
            output_s3_uri="s3://bucket/inference/uuid/inference_output.jsonl",
        )
        eval_settings = config["eval"]
        assert eval_settings["max_connections"] == 1
        assert eval_settings["decoding"]["temperature"] == 0.0
        assert eval_settings["decoding"]["max_tokens"] == 8192


class TestCostWarning:
    """Tests for _emit_cost_warning behavior."""

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_cost_warning_emitted_for_inspectai_path(
        self, mock_resolve, mock_artifact
    ):
        """Warning logged with instance type when InspectAI path is used."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=None,
        )
        with patch(
            "sagemaker.train.evaluate.llm_as_judge_evaluator._logger"
        ) as mock_logger:
            evaluator._emit_cost_warning("ml.m5.large", "Bedrock")
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "ml.m5.large" in warning_msg

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_no_cost_warning_for_jumpstart_path(
        self, mock_resolve, mock_artifact
    ):
        """No warning emitted when JumpStart path (non-InspectAI) is taken."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="llama3-2-1b-instruct",
            source_model_package_arn=None,
        )
        with patch(
            "sagemaker.train.evaluate.llm_as_judge_evaluator._logger"
        ) as mock_logger:
            # JumpStart path does not call _emit_cost_warning, so the
            # logger should not receive any warning calls for this evaluator.
            # We verify by checking that the path is not InspectAI and
            # therefore _emit_cost_warning would never be called.
            assert evaluator._should_use_inspectai_path() is False
            mock_logger.warning.assert_not_called()


class TestLlmajInspectaiTemplate:
    """Tests for LLMAJ_INSPECTAI_TEMPLATE Jinja2 rendering."""

    def _render_template(self, context):
        """Render LLMAJ_INSPECTAI_TEMPLATE with given context via Jinja2."""
        from jinja2 import Template

        template = Template(LLMAJ_INSPECTAI_TEMPLATE)
        return template.render(**context)

    def _sample_context(self):
        """Return a minimal valid template context."""
        return {
            "mlflow_resource_arn": DEFAULT_MLFLOW_ARN,
            "mlflow_experiment_name": "test-experiment",
            "inspectai_image_uri": (
                "123456789012.dkr.ecr.us-west-2.amazonaws.com/inspectai:latest"
            ),
            "role_arn": DEFAULT_ROLE,
            "inspectai_instance_type": "ml.m5.large",
            "inspectai_config_s3_uri": "s3://bucket/config/",
            "s3_output_path": "s3://bucket/output",
            "kms_key_id": None,
            "environment": None,
            "vpc_config": None,
            "base_model_arn": DEFAULT_BASE_MODEL_ARN,
            "judge_model_id": DEFAULT_EVALUATOR_MODEL,
            "inference_output_s3_uri": (
                "s3://bucket/output/inference/abc-123/inference_output.jsonl"
            ),
            "llmaj_metrics": ["Correctness", "Helpfulness"],
            "custom_metrics": None,
            "max_new_tokens": "8192",
            "temperature": "0",
            "top_k": "-1",
            "top_p": "1.0",
            "model_package_config": None,
        }

    def test_llmaj_inspectai_template_renders_valid_json(self):
        """Rendered template with sample context produces valid JSON."""
        context = self._sample_context()
        rendered = self._render_template(context)
        parsed = json.loads(rendered)
        assert parsed["Version"] == "2020-12-01"
        assert len(parsed["Steps"]) == 2
        assert parsed["Steps"][0]["Name"] == "InspectAIInference"
        assert parsed["Steps"][1]["Name"] == "EvaluateWithJudge"

    def test_inference_output_path_matches_between_steps(self):
        """Config output_s3_uri matches Phase 2 inference_data_s3_path."""
        inference_output_s3_uri = (
            "s3://bucket/output/inference/run-id-123/inference_output.jsonl"
        )
        context = self._sample_context()
        context["inference_output_s3_uri"] = inference_output_s3_uri

        rendered = self._render_template(context)
        parsed = json.loads(rendered)

        # Phase 2 HyperParameters.inference_data_s3_path should equal the
        # output_s3_uri that Phase 1 writes to (coordinated via UUID).
        phase2_hypers = parsed["Steps"][1]["Arguments"]["HyperParameters"]
        assert phase2_hypers["inference_data_s3_path"] == inference_output_s3_uri


class TestResolveBedrocklModelId:
    """Tests for _resolve_bedrock_model_id utility function."""

    def test_nova_lite_us_east_1(self):
        """Nova Lite in us-east-1 resolves to us.amazon.nova-lite-v1:0."""
        result = _resolve_bedrock_model_id("nova-textgeneration-lite", "us-east-1")
        assert result == "us.amazon.nova-lite-v1:0"

    def test_nova_pro_us_west_2(self):
        """Nova Pro in us-west-2 resolves to us.amazon.nova-pro-v1:0."""
        result = _resolve_bedrock_model_id("nova-textgeneration-pro", "us-west-2")
        assert result == "us.amazon.nova-pro-v1:0"

    def test_nova_micro_eu_west_2(self):
        """Nova Micro in eu-west-2 resolves to eu.amazon.nova-micro-v1:0."""
        result = _resolve_bedrock_model_id("nova-textgeneration-micro", "eu-west-2")
        assert result == "eu.amazon.nova-micro-v1:0"

    def test_nova_lite_unsupported_ap_region(self):
        """Nova Lite in ap-northeast-1 returns None (InspectAI not available)."""
        result = _resolve_bedrock_model_id("nova-textgeneration-lite", "ap-northeast-1")
        assert result is None

    def test_non_nova_model_returns_none(self):
        """Non-Nova model returns None (not in mapping)."""
        result = _resolve_bedrock_model_id("llama3-2-1b-instruct", "us-east-1")
        assert result is None

    def test_unsupported_region_returns_none(self):
        """Nova model in unsupported region returns None."""
        result = _resolve_bedrock_model_id("nova-textgeneration-lite", "me-south-1")
        assert result is None

    def test_all_mapped_models_resolve(self):
        """All models in NOVA_BEDROCK_MODEL_IDS resolve in supported regions."""
        for model_name in NOVA_BEDROCK_MODEL_IDS:
            for region in _REGION_TO_BEDROCK_PREFIX:
                result = _resolve_bedrock_model_id(model_name, region)
                assert result is not None, f"Failed for {model_name} in {region}"
                prefix = _REGION_TO_BEDROCK_PREFIX[region]
                assert result.startswith(f"{prefix}.")


class TestGetInferenceModelId:
    """Tests for _get_inference_model_id method on evaluator."""

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_nova_model_returns_bedrock_id(self, mock_resolve, mock_artifact):
        """Nova model returns derived Bedrock model ID."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="nova-textgeneration-lite",
            source_model_package_arn=None,
        )
        result = evaluator._get_inference_model_id("us-east-1")
        assert result == "us.amazon.nova-lite-v1:0"

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_model_package_returns_none(self, mock_resolve, mock_artifact):
        """Custom model (model package) returns None — uses endpoint instead."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            source_model_package_arn=DEFAULT_MODEL_PACKAGE_ARN,
        )
        result = evaluator._get_inference_model_id("us-east-1")
        assert result is None

    @patch("sagemaker.core.resources.Artifact")
    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    def test_non_nova_jumpstart_returns_none(self, mock_resolve, mock_artifact):
        """Non-Nova JumpStart model returns None (uses ServerlessJobConfig)."""
        evaluator = _create_evaluator(
            mock_resolve,
            mock_artifact,
            base_model_name="llama3-2-1b-instruct",
            source_model_package_arn=None,
        )
        result = evaluator._get_inference_model_id("us-east-1")
        assert result is None