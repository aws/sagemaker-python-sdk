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
"""InspectAIEvaluator unit tests."""
from __future__ import absolute_import

import os
import tempfile
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError
from sagemaker.train.evaluate.constants import EvalType, _get_inspect_ai_default_image_uri
from sagemaker.train.evaluate.inspect_ai_evaluator import InspectAIEvaluator

# Test constants
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


def _mock_session():
    mock_session = Mock()
    mock_session.boto_region_name = DEFAULT_REGION
    mock_session.boto_session = Mock()
    mock_session.boto_session.region_name = DEFAULT_REGION
    mock_session.get_caller_identity_arn.return_value = DEFAULT_ROLE
    mock_session.sagemaker_config = {}
    return mock_session


def _mock_model_resolution():
    mock_info = Mock()
    mock_info.base_model_name = DEFAULT_MODEL
    mock_info.base_model_arn = DEFAULT_BASE_MODEL_ARN
    mock_info.source_model_package_arn = None
    mock_info.bedrock_model_id = "us.amazon.nova-lite-v1:0"
    return mock_info


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIEvaluatorConstruction:
    """Test InspectAIEvaluator construction and validation."""

    def test_construction_bedrock_minimal(self, mock_artifact, mock_resolve):
        """Test construction with minimal bedrock config."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator.benchmarks_path == DEFAULT_BENCHMARKS_PATH
        assert evaluator.instance_type == "ml.m5.large"
        assert evaluator.max_runtime_seconds == 86400
        assert evaluator.cleanup_endpoint is True
        assert evaluator.endpoint_prefix == "inspectai"
        assert evaluator.tasks is None

    def test_construction_with_tasks(self, mock_artifact, mock_resolve):
        """Test construction with tasks list."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        tasks = [
            {"name": "boolq_pt", "limit": 10},
            {"name": "hellaswag", "epochs": 2, "task_args": {"n_shots": 5}},
        ]

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            tasks=tasks,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator.tasks == tasks

    def test_construction_existing_endpoint(self, mock_artifact, mock_resolve):
        """Test construction with existing endpoint config."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            endpoint_name="my-endpoint",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator.endpoint_name == "my-endpoint"
        assert evaluator._infer_scenario() == "existing_endpoint"

    def test_construction_create_endpoint(self, mock_artifact, mock_resolve):
        """Test construction with create endpoint config."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            model_s3_uri="s3://bucket/model/",
            inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:latest",
            endpoint_instance_type="ml.g5.xlarge",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator._infer_scenario() == "create_endpoint"


class TestInspectAIEvaluatorValidation:
    """Test InspectAIEvaluator field validation."""

    def test_benchmarks_path_required_non_empty(self):
        """Test benchmarks_path cannot be empty."""
        with pytest.raises(ValidationError, match="benchmarks_path is required"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path="",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_benchmarks_path_must_be_s3(self):
        """Test benchmarks_path must start with s3://."""
        with pytest.raises(ValidationError, match="must start with 's3://'"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path="/local/path/benchmarks/",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_tasks_must_have_name_key(self):
        """Test each task dict must have a 'name' key."""
        with pytest.raises(ValidationError, match="must have a 'name' key"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"limit": 10}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_tasks_limit_must_be_positive_int(self):
        """Test task limit must be int >= 1."""
        with pytest.raises(ValidationError, match="must be an integer >= 1"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"name": "boolq", "limit": 0}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_tasks_epochs_must_be_positive_int(self):
        """Test task epochs must be int >= 1."""
        with pytest.raises(ValidationError, match="must be an integer >= 1"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"name": "boolq", "epochs": -1}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_tasks_path_must_end_py(self):
        """Test task path must end with .py."""
        with pytest.raises(ValidationError, match="must end with '.py'"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"name": "boolq", "path": "boolq.yaml"}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_tasks_task_args_must_be_dict(self):
        """Test task_args must be a dict."""
        with pytest.raises(ValidationError, match="must be a dict"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"name": "boolq", "task_args": "invalid"}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_output_format_invalid(self):
        """Test invalid output format raises error."""
        with pytest.raises(ValidationError, match="output_format must be one of"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                output_format="xml",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact")
    def test_mutual_exclusion_endpoint_and_model_s3(self, mock_artifact, mock_resolve):
        """Test endpoint_name and model_s3_uri are mutually exclusive."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        with pytest.raises(ValidationError, match="mutually exclusive"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                endpoint_name="my-endpoint",
                model_s3_uri="s3://bucket/model/",
                inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img:v1",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact")
    def test_model_s3_uri_requires_inference_image(self, mock_artifact, mock_resolve):
        """Test model_s3_uri requires inference_image_uri."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        with pytest.raises(ValidationError, match="inference_image_uri is required"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                model_s3_uri="s3://bucket/model/",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

    @patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
    @patch("sagemaker.core.resources.Artifact")
    def test_inference_image_requires_model_s3(self, mock_artifact, mock_resolve):
        """Test inference_image_uri requires model_s3_uri."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        with pytest.raises(ValidationError, match="model_s3_uri is required"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img:v1",
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

    def test_endpoint_instance_type_must_start_with_ml(self):
        """Test endpoint_instance_type must start with ml."""
        with pytest.raises(ValidationError, match="must start with 'ml.'"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                endpoint_instance_type="p4d.24xlarge",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_instance_type_must_start_with_ml(self):
        """Test orchestrator instance_type must start with ml."""
        with pytest.raises(ValidationError, match="must start with 'ml.'"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                instance_type="m5.large",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    def test_model_s3_uri_must_be_s3(self):
        """Test model_s3_uri must start with s3://."""
        with pytest.raises(ValidationError, match="must start with 's3://'"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                model_s3_uri="/local/model/",
                inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img:v1",
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
            )

    @pytest.mark.parametrize(
        "field,value",
        [
            ("max_connections", 0),
            ("max_retries", 0),
            ("max_tokens", 0),
            ("timeout", 0),
        ],
    )
    def test_positive_int_fields_reject_zero(self, field, value):
        """max_connections, max_retries, max_tokens, timeout must be >= 1."""
        with pytest.raises(ValidationError):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
                **{field: value},
            )

    def test_temperature_out_of_range(self):
        with pytest.raises(ValidationError, match="temperature must be in"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
                temperature=2.5,
            )

    def test_top_p_out_of_range(self):
        with pytest.raises(ValidationError, match="top_p must be in"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
                top_p=1.5,
            )

    def test_top_k_zero_rejected(self):
        with pytest.raises(ValidationError, match="top_k must be"):
            InspectAIEvaluator(
                model=DEFAULT_MODEL,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                sagemaker_session=_mock_session(),
                top_k=0,
            )


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIYAMLSerialization:
    """Test YAML config serialization for each inference mode."""

    def _create_evaluator(self, mock_artifact, mock_resolve, **kwargs):
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        defaults = dict(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )
        defaults.update(kwargs)
        return InspectAIEvaluator(**defaults)

    def test_yaml_bedrock_mode(self, mock_artifact, mock_resolve):
        """Test YAML serialization for bedrock inference mode."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            tasks=[{"name": "boolq_pt", "limit": 10}],
            bedrock_model_id="us.amazon.nova-lite-v1:0",
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)

        assert "inference_provider" in config
        assert "bedrock" in config["inference_provider"]
        assert config["inference_provider"]["bedrock"]["model_id"] == "us.amazon.nova-lite-v1:0"
        assert config["inference_provider"]["bedrock"]["region"] == DEFAULT_REGION

        assert "benchmarks" in config
        assert config["benchmarks"]["s3_path"] == DEFAULT_BENCHMARKS_PATH
        assert len(config["benchmarks"]["tasks"]) == 1
        assert config["benchmarks"]["tasks"][0]["name"] == "boolq_pt"
        assert config["benchmarks"]["tasks"][0]["limit"] == 10

        assert "eval" in config
        assert config["eval"]["max_connections"] == 16
        assert config["eval"]["decoding"]["temperature"] == 0.0
        assert config["eval"]["decoding"]["max_tokens"] == 8192

        assert "output" in config
        assert config["output"]["s3_path"].endswith("/inspectai-results/")

    def test_yaml_existing_endpoint_mode(self, mock_artifact, mock_resolve):
        """Test YAML serialization for existing endpoint inference mode."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            endpoint_name="my-endpoint",
            context_length="4096",
            max_concurrency="8",
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)

        provider = config["inference_provider"]
        assert "sagemaker_endpoint" in provider
        ep = provider["sagemaker_endpoint"]
        assert ep["endpoint_name"] == "my-endpoint"
        assert ep["region"] == DEFAULT_REGION
        assert ep["context_length"] == "4096"
        assert ep["max_concurrency"] == "8"

    def test_yaml_create_endpoint_mode(self, mock_artifact, mock_resolve):
        """Test YAML serialization for create endpoint inference mode."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            model_s3_uri="s3://bucket/model/",
            inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:v1",
            endpoint_instance_type="ml.g5.xlarge",
            endpoint_instance_count=2,
            endpoint_execution_role_arn="arn:aws:iam::123456789012:role/endpoint-role",
            cleanup_endpoint=False,
            endpoint_prefix="myprefix",
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)

        provider = config["inference_provider"]
        assert "sagemaker_endpoint" in provider
        ep = provider["sagemaker_endpoint"]
        assert ep["endpoint_name"] is None
        assert ep["model_s3_uri"] == "s3://bucket/model/"
        assert (
            ep["inference_image_uri"] == "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:v1"
        )
        assert ep["instance_type"] == "ml.g5.xlarge"
        assert ep["instance_count"] == 2
        assert ep["execution_role_arn"] == "arn:aws:iam::123456789012:role/endpoint-role"
        assert ep["cleanup_endpoint"] is False
        assert ep["endpoint_prefix"] == "myprefix"

    def test_yaml_with_extra_args(self, mock_artifact, mock_resolve):
        """Test YAML serialization includes extra_args when set."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            extra_args=["--log-level", "debug"],
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)
        assert config["eval"]["extra_args"] == ["--log-level", "debug"]

    def test_yaml_with_output_format(self, mock_artifact, mock_resolve):
        """Test YAML serialization includes output_format when set."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            output_format="json",
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)
        assert config["output"]["output_format"] == "json"

    def test_yaml_without_tasks_omits_tasks_key(self, mock_artifact, mock_resolve):
        """Test YAML serialization when tasks is None."""
        evaluator = self._create_evaluator(mock_artifact, mock_resolve)

        config = evaluator._build_yaml_config(DEFAULT_REGION)
        assert "tasks" not in config["benchmarks"]

    def test_yaml_eval_and_decoding_overrides(self, mock_artifact, mock_resolve):
        """User-supplied eval/decoding values override the defaults in the YAML."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            max_connections=4,
            max_retries=5,
            timeout=120,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            max_tokens=4096,
        )

        config = evaluator._build_yaml_config(DEFAULT_REGION)

        assert config["eval"]["max_connections"] == 4
        assert config["eval"]["max_retries"] == 5
        assert config["eval"]["timeout"] == 120
        assert config["eval"]["decoding"]["temperature"] == 0.7
        assert config["eval"]["decoding"]["top_p"] == 0.9
        assert config["eval"]["decoding"]["top_k"] == 50
        assert config["eval"]["decoding"]["max_tokens"] == 4096


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIUploadBenchmarks:
    """Test upload_benchmarks method."""

    def test_upload_benchmarks_success(self, mock_artifact, mock_resolve):
        """Test upload_benchmarks uploads to S3 and returns URI."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy benchmark file
            with open(os.path.join(tmpdir, "boolq_pt.py"), "w") as f:
                f.write("from inspect_ai import task\n@task\ndef boolq_pt(): pass\n")

            with patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader") as mock_uploader:
                result = evaluator.upload_benchmarks(tmpdir)

            assert result.startswith("s3://test-bucket/eval-output/benchmarks/")
            mock_uploader.upload.assert_called_once()

    def test_upload_benchmarks_invalid_path(self, mock_artifact, mock_resolve):
        """Test upload_benchmarks raises for non-existent path."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        with pytest.raises(ValueError, match="must be an existing directory"):
            evaluator.upload_benchmarks("/nonexistent/path")


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIEvaluateFlow:
    """Test the evaluate() method orchestration."""

    def _create_evaluator(self, mock_artifact, mock_resolve, **kwargs):
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        defaults = dict(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )
        defaults.update(kwargs)
        return InspectAIEvaluator(**defaults)

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_evaluate_calls_start_execution(self, mock_uploader, mock_artifact, mock_resolve):
        """Test evaluate() orchestrates config upload and pipeline start."""
        evaluator = self._create_evaluator(mock_artifact, mock_resolve)

        mock_execution = Mock()

        with (
            patch.object(evaluator, "_get_aws_execution_context") as mock_ctx,
            patch.object(evaluator, "_start_execution", return_value=mock_execution) as mock_start,
        ):
            mock_ctx.return_value = {
                "role_arn": DEFAULT_ROLE,
                "region": DEFAULT_REGION,
                "account_id": "123456789012",
            }
            result = evaluator.evaluate()

        mock_uploader.upload_string_as_file_body.assert_called_once()
        mock_start.assert_called_once()
        call_kwargs = mock_start.call_args.kwargs
        assert call_kwargs["eval_type"] == EvalType.INSPECT_AI
        assert call_kwargs["role_arn"] == DEFAULT_ROLE
        assert call_kwargs["region"] == DEFAULT_REGION
        assert result is mock_execution

    @patch("sagemaker.train.evaluate.inspect_ai_evaluator.S3Uploader")
    def test_evaluate_uploads_valid_yaml(self, mock_uploader, mock_artifact, mock_resolve):
        """Test that evaluate() uploads valid YAML config to S3."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            tasks=[{"name": "boolq_pt", "limit": 10}],
            bedrock_model_id="us.amazon.nova-lite-v1:0",
        )

        with (
            patch.object(evaluator, "_get_aws_execution_context") as mock_ctx,
            patch.object(evaluator, "_start_execution") as mock_start,
        ):
            mock_ctx.return_value = {
                "role_arn": DEFAULT_ROLE,
                "region": DEFAULT_REGION,
                "account_id": "123456789012",
            }
            mock_start.return_value = Mock()
            evaluator.evaluate()

        # Verify YAML was uploaded
        upload_call = mock_uploader.upload_string_as_file_body.call_args
        import yaml

        yaml_body = upload_call.kwargs["body"]
        config = yaml.safe_load(yaml_body)
        assert config["inference_provider"]["bedrock"]["model_id"] == "us.amazon.nova-lite-v1:0"
        assert config["benchmarks"]["tasks"][0]["name"] == "boolq_pt"


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAIGetBedrockModelId:
    """Test _get_bedrock_model_id() resolution logic."""

    def _create_evaluator(self, mock_artifact, mock_resolve, **kwargs):
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        defaults = dict(
            model=DEFAULT_MODEL,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )
        defaults.update(kwargs)
        return InspectAIEvaluator(**defaults)

    def test_explicit_bedrock_model_id_takes_priority(self, mock_artifact, mock_resolve):
        """Test explicit bedrock_model_id is returned directly."""
        evaluator = self._create_evaluator(
            mock_artifact,
            mock_resolve,
            bedrock_model_id="us.amazon.nova-pro-v1:0",
        )
        assert evaluator._get_bedrock_model_id("us-east-1") == "us.amazon.nova-pro-v1:0"

    def test_fallback_to_model_string(self, mock_artifact, mock_resolve):
        """Test fallback to {region_prefix}.{model} when model is a string."""
        mock_info = _mock_model_resolution()
        mock_info.bedrock_model_id = None
        mock_resolve.return_value = mock_info
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model="amazon-nova-lite-v1",
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        with patch.object(evaluator, "_get_resolved_model_info") as mock_get_info:
            mock_get_info.side_effect = Exception("no model info")
            result = evaluator._get_bedrock_model_id("us-east-1")

        assert result == "us.amazon-nova-lite-v1"

    def test_fallback_to_model_string_non_us_region(self, mock_artifact, mock_resolve):
        """Test fallback uses region prefix for non-US regions."""
        mock_info = _mock_model_resolution()
        mock_info.bedrock_model_id = None
        mock_resolve.return_value = mock_info
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        evaluator = InspectAIEvaluator(
            model="amazon-nova-lite-v1",
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        with patch.object(evaluator, "_get_resolved_model_info") as mock_get_info:
            mock_get_info.side_effect = Exception("no model info")
            assert evaluator._get_bedrock_model_id("eu-west-2") == "eu.amazon-nova-lite-v1"
            assert evaluator._get_bedrock_model_id("ap-northeast-1") == "ap.amazon-nova-lite-v1"


class TestInspectAIConstants:
    """Test InspectAI-related constants."""

    def test_eval_type_inspect_ai_exists(self):
        """Test EvalType.INSPECT_AI enum member exists."""
        assert EvalType.INSPECT_AI.value == "inspectai"

    def test_default_image_uri(self):
        """Test default image URI construction."""
        uri = _get_inspect_ai_default_image_uri("us-east-1")
        assert uri == "763104351884.dkr.ecr.us-east-1.amazonaws.com/sagemaker-inspect-ai"

    def test_default_image_uri_different_region(self):
        """Test default image URI with different region."""
        uri = _get_inspect_ai_default_image_uri("eu-west-2")
        assert uri == "763104351884.dkr.ecr.eu-west-2.amazonaws.com/sagemaker-inspect-ai"

    @pytest.mark.parametrize(
        "region",
        [
            "us-east-1",
            "us-west-2",
            "eu-west-1",
            "ap-northeast-1",
        ],
    )
    def test_default_image_uri_supported_regions(self, region):
        """Supported regions resolve to the standard commercial DLC account."""
        uri = _get_inspect_ai_default_image_uri(region)
        assert uri == f"763104351884.dkr.ecr.{region}.amazonaws.com/sagemaker-inspect-ai"

    @pytest.mark.parametrize(
        "region",
        [
            "cn-north-1",
            "cn-northwest-1",
            "us-gov-east-1",
            "us-gov-west-1",
            "ap-east-1",
            "not-a-real-region",
        ],
    )
    def test_default_image_uri_unsupported_region_raises(self, region):
        """Unmapped regions raise upfront; users must override via image_uri.

        The InspectAI image is only published to standard commercial-partition
        regions today. Users in China, GovCloud, ISO, or opt-in regions must
        pass image_uri= explicitly to point at their own ECR mirror.
        """
        with pytest.raises(ValueError):
            _get_inspect_ai_default_image_uri(region)


@patch("sagemaker.train.common_utils.model_resolution._resolve_base_model")
@patch("sagemaker.core.resources.Artifact")
class TestInspectAITrainerResourceChaining:
    """Test resource chaining: passing a trainer directly as the model parameter."""

    def _mock_trainer(self, with_training_job=True, model_s3_uri=None, image_uri=None):
        """Create a mock trainer with a completed training job."""
        from sagemaker.train.base_trainer import BaseTrainer

        trainer = Mock(spec=BaseTrainer)
        trainer.__class__ = BaseTrainer
        # Make isinstance check work
        trainer._model_arn = DEFAULT_BASE_MODEL_ARN
        trainer._model_name = DEFAULT_MODEL

        if with_training_job:
            mock_job = Mock()
            mock_job.output_model_package_arn = (
                "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"
            )
            trainer._latest_training_job = mock_job
            trainer._latest_job = None
        else:
            trainer._latest_training_job = None
            trainer._latest_job = None

        return trainer

    def _mock_model_package(self, model_s3_uri="s3://bucket/model/output/model.tar.gz",
                            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/inference:latest"):
        """Create a mock ModelPackage with inference specification."""
        mp = Mock()
        container = Mock()
        container.model_data_url = model_s3_uri
        container.image = image_uri
        container.base_model = Mock()
        container.base_model.hub_content_name = DEFAULT_MODEL
        container.base_model.hub_content_arn = DEFAULT_BASE_MODEL_ARN
        mp.inference_specification = Mock()
        mp.inference_specification.containers = [container]
        mp.model_package_arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/my-group/1"
        return mp

    def test_trainer_auto_resolves_create_endpoint(self, mock_artifact, mock_resolve):
        """Test that passing a trainer auto-resolves model_s3_uri and inference_image_uri."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()
        mock_mp = self._mock_model_package()

        with patch("sagemaker.core.resources.ModelPackage.get", return_value=mock_mp):
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

        assert evaluator._infer_scenario() == "create_endpoint"
        assert evaluator.model_s3_uri == "s3://bucket/model/output/model.tar.gz"
        assert evaluator.inference_image_uri == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/inference:latest"
        )

    def test_trainer_with_explicit_endpoint_name_skips_resolution(self, mock_artifact, mock_resolve):
        """Test that explicit endpoint_name prevents trainer artifact resolution."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()

        evaluator = InspectAIEvaluator(
            model=trainer,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            endpoint_name="my-existing-endpoint",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator._infer_scenario() == "existing_endpoint"
        assert evaluator.endpoint_name == "my-existing-endpoint"

    def test_trainer_with_explicit_bedrock_model_id_skips_resolution(self, mock_artifact, mock_resolve):
        """Test that explicit bedrock_model_id prevents trainer artifact resolution."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()

        evaluator = InspectAIEvaluator(
            model=trainer,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            bedrock_model_id="us.amazon.nova-lite-v1:0",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator._infer_scenario() == "bedrock"
        assert evaluator.model_s3_uri is None

    def test_trainer_with_explicit_model_s3_uri_skips_resolution(self, mock_artifact, mock_resolve):
        """Test that explicit model_s3_uri prevents trainer artifact resolution."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()

        evaluator = InspectAIEvaluator(
            model=trainer,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            model_s3_uri="s3://my-custom-bucket/model/",
            inference_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/custom:v1",
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator._infer_scenario() == "create_endpoint"
        assert evaluator.model_s3_uri == "s3://my-custom-bucket/model/"

    def test_trainer_without_completed_job_falls_back_to_bedrock(self, mock_artifact, mock_resolve):
        """Test trainer without completed job gracefully falls back to bedrock mode."""
        mock_info = _mock_model_resolution()
        mock_info.bedrock_model_id = "us.amazon.nova-lite-v1:0"
        mock_resolve.return_value = mock_info
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer(with_training_job=False)

        evaluator = InspectAIEvaluator(
            model=trainer,
            benchmarks_path=DEFAULT_BENCHMARKS_PATH,
            s3_output_path=DEFAULT_S3_OUTPUT,
            mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
            model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
            sagemaker_session=_mock_session(),
        )

        assert evaluator._infer_scenario() == "bedrock"
        assert evaluator.model_s3_uri is None

    def test_trainer_model_package_without_inference_spec_falls_back(self, mock_artifact, mock_resolve):
        """Test fallback when model package has no inference specification."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()

        # Model package without inference spec
        mock_mp = Mock()
        mock_mp.inference_specification = None

        with patch("sagemaker.core.resources.ModelPackage.get", return_value=mock_mp):
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

        assert evaluator._infer_scenario() == "bedrock"
        assert evaluator.model_s3_uri is None

    def test_trainer_model_package_get_fails_falls_back(self, mock_artifact, mock_resolve):
        """Test fallback when ModelPackage.get() raises an exception."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()

        with patch("sagemaker.core.resources.ModelPackage.get", side_effect=Exception("API error")):
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

        assert evaluator._infer_scenario() == "bedrock"
        assert evaluator.model_s3_uri is None

    def test_trainer_yaml_config_uses_create_endpoint(self, mock_artifact, mock_resolve):
        """Test that YAML config is built correctly when trainer resolves to create_endpoint."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        trainer = self._mock_trainer()
        mock_mp = self._mock_model_package(
            model_s3_uri="s3://output-bucket/jobs/model-artifacts/model.tar.gz",
            image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-inference:2.0",
        )

        with patch("sagemaker.core.resources.ModelPackage.get", return_value=mock_mp):
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                tasks=[{"name": "mmlu", "limit": 5}],
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

        config = evaluator._build_yaml_config(DEFAULT_REGION)

        provider = config["inference_provider"]
        assert "sagemaker_endpoint" in provider
        ep = provider["sagemaker_endpoint"]
        assert ep["model_s3_uri"] == "s3://output-bucket/jobs/model-artifacts/model.tar.gz"
        assert ep["inference_image_uri"] == (
            "123456789012.dkr.ecr.us-east-1.amazonaws.com/sagemaker-inference:2.0"
        )
        assert ep["cleanup_endpoint"] is True

    def test_trainer_with_latest_job_mtrl_style(self, mock_artifact, mock_resolve):
        """Test resource chaining for MultiTurnRLTrainer-style trainer using _latest_job."""
        mock_resolve.return_value = _mock_model_resolution()
        mock_artifact.get_all.return_value = iter([])
        mock_artifact_instance = Mock()
        mock_artifact_instance.artifact_arn = DEFAULT_ARTIFACT_ARN
        mock_artifact.create.return_value = mock_artifact_instance

        from sagemaker.train.base_trainer import BaseTrainer

        trainer = Mock(spec=BaseTrainer)
        trainer.__class__ = BaseTrainer
        trainer._model_arn = DEFAULT_BASE_MODEL_ARN
        trainer._model_name = DEFAULT_MODEL
        trainer._latest_training_job = None
        # MultiTurnRLTrainer uses _latest_job instead
        mock_job = Mock()
        mock_job.output_model_package_arn = (
            "arn:aws:sagemaker:us-east-1:123456789012:model-package/mtrl-group/1"
        )
        trainer._latest_job = mock_job

        mock_mp = self._mock_model_package()

        with patch("sagemaker.core.resources.ModelPackage.get", return_value=mock_mp):
            evaluator = InspectAIEvaluator(
                model=trainer,
                benchmarks_path=DEFAULT_BENCHMARKS_PATH,
                s3_output_path=DEFAULT_S3_OUTPUT,
                mlflow_resource_arn=DEFAULT_MLFLOW_ARN,
                model_package_group=DEFAULT_MODEL_PACKAGE_GROUP_ARN,
                sagemaker_session=_mock_session(),
            )

        assert evaluator._infer_scenario() == "create_endpoint"
        assert evaluator.model_s3_uri == "s3://bucket/model/output/model.tar.gz"
