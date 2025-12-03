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
"""Tests for pipeline_templates module."""
from __future__ import absolute_import

import json
import pytest
from jinja2 import Template

from sagemaker.train.evaluate.pipeline_templates import (
    DETERMINISTIC_TEMPLATE,
    LLMAJ_TEMPLATE_BASE_MODEL_ONLY,
    DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY,
    CUSTOM_SCORER_TEMPLATE,
    CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY,
    LLMAJ_TEMPLATE,
)


# Base context for all templates
BASE_CONTEXT = {
    "mlflow_resource_arn": "arn:aws:sagemaker:us-west-2:123456789012:mlflow-tracking-server/test-server",
    "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
    "base_model_arn": "arn:aws:bedrock:us-west-2::foundation-model/test-model",
    "s3_output_path": "s3://test-bucket/output",
    "dataset_uri": "s3://test-bucket/dataset",
}


class TestDeterministicTemplate:
    """Tests for DETERMINISTIC_TEMPLATE rendering."""

    def test_deterministic_template_minimal_context(self):
        """Test DETERMINISTIC_TEMPLATE with minimal required context."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        # Verify it's valid JSON
        pipeline_def = json.loads(rendered)
        
        # Verify basic structure
        assert pipeline_def["Version"] == "2020-12-01"
        assert "MlflowConfig" in pipeline_def
        assert pipeline_def["MlflowConfig"]["MlflowResourceArn"] == context["mlflow_resource_arn"]
        assert "Steps" in pipeline_def
        
        # Should have 2 steps: CreateEvaluationAction and EvaluateCustomModel (no base model)
        assert len(pipeline_def["Steps"]) == 3
        assert pipeline_def["Steps"][0]["Name"] == "CreateEvaluationAction"
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateCustomModel"
        assert pipeline_def["Steps"][2]["Name"] == "AssociateLineage"

    def test_deterministic_template_with_base_model(self):
        """Test DETERMINISTIC_TEMPLATE with evaluate_base_model flag."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "evaluate_base_model": True,
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should have 3 steps: CreateEvaluationAction, EvaluateBaseModel, EvaluateCustomModel, AssociateLineage
        assert len(pipeline_def["Steps"]) == 4
        assert pipeline_def["Steps"][0]["Name"] == "CreateEvaluationAction"
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateBaseModel"
        assert pipeline_def["Steps"][2]["Name"] == "EvaluateCustomModel"
        assert pipeline_def["Steps"][3]["Name"] == "AssociateLineage"

    def test_deterministic_template_with_optional_mlflow_params(self):
        """Test DETERMINISTIC_TEMPLATE with optional MLflow parameters."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "mlflow_experiment_name": "test-experiment",
            "mlflow_run_name": "test-run",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        assert pipeline_def["MlflowConfig"]["MlflowExperimentName"] == "test-experiment"
        assert pipeline_def["MlflowConfig"]["MlflowRunName"] == "test-run"

    def test_deterministic_template_with_all_hyperparameters(self):
        """Test DETERMINISTIC_TEMPLATE with all optional hyperparameters."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "max_new_tokens": 100,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.9,
            "max_model_len": 2048,
            "aggregation": "mean",
            "postprocessing": "normalize",
            "preset_reward_function": "default",
            "subtask": "summarization",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Check EvaluateCustomModel step has all hyperparameters
        custom_model_step = pipeline_def["Steps"][1]
        hyperparams = custom_model_step["Arguments"]["HyperParameters"]
        
        assert hyperparams["task"] == "text-generation"
        assert hyperparams["max_new_tokens"] == "100"
        assert hyperparams["temperature"] == "0.7"
        assert hyperparams["top_k"] == "50"
        assert hyperparams["top_p"] == "0.9"
        assert hyperparams["max_model_len"] == "2048"
        assert hyperparams["aggregation"] == "mean"
        assert hyperparams["postprocessing"] == "normalize"
        assert hyperparams["preset_reward_function"] == "default"
        assert hyperparams["subtask"] == "summarization"

    def test_deterministic_template_with_kms_key(self):
        """Test DETERMINISTIC_TEMPLATE with KMS key encryption."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/test-key",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        custom_model_step = pipeline_def["Steps"][1]
        output_config = custom_model_step["Arguments"]["OutputDataConfig"]
        
        assert output_config["KmsKeyId"] == "arn:aws:kms:us-west-2:123456789012:key/test-key"

    def test_deterministic_template_with_vpc_config(self):
        """Test DETERMINISTIC_TEMPLATE with VPC configuration."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "vpc_config": True,
            "vpc_security_group_ids": ["sg-12345", "sg-67890"],
            "vpc_subnets": ["subnet-abc", "subnet-def"],
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        custom_model_step = pipeline_def["Steps"][1]
        vpc_config = custom_model_step["Arguments"]["VpcConfig"]
        
        assert vpc_config["SecurityGroupIds"] == ["sg-12345", "sg-67890"]
        assert vpc_config["Subnets"] == ["subnet-abc", "subnet-def"]

    def test_deterministic_template_with_dataset_arn(self):
        """Test DETERMINISTIC_TEMPLATE with AIRegistry dataset ARN."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "dataset_uri": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/DataSet/test-dataset",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        custom_model_step = pipeline_def["Steps"][1]
        data_source = custom_model_step["Arguments"]["InputDataConfig"][0]["DataSource"]
        
        # Should use DatasetSource instead of S3DataSource
        assert "DatasetSource" in data_source
        assert data_source["DatasetSource"]["DatasetArn"] == context["dataset_uri"]

    def test_deterministic_template_with_evaluator_arn(self):
        """Test DETERMINISTIC_TEMPLATE with evaluator ARN."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "evaluator_arn": "arn:aws:sagemaker:us-west-2:123456789012:evaluator/test-evaluator",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        custom_model_step = pipeline_def["Steps"][1]
        serverless_config = custom_model_step["Arguments"]["ServerlessJobConfig"]
        
        assert serverless_config["EvaluatorArn"] == "arn:aws:sagemaker:us-west-2:123456789012:evaluator/test-evaluator"


class TestDeterministicTemplateBaseModelOnly:
    """Tests for DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY rendering."""

    def test_deterministic_base_model_only_minimal(self):
        """Test DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY with minimal context."""
        context = {
            **BASE_CONTEXT,
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
        }

        template = Template(DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should have only 1 step: EvaluateBaseModel
        assert len(pipeline_def["Steps"]) == 1
        assert pipeline_def["Steps"][0]["Name"] == "EvaluateBaseModel"

    def test_deterministic_base_model_only_with_all_params(self):
        """Test DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY with all parameters."""
        context = {
            **BASE_CONTEXT,
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "mlflow_experiment_name": "test-experiment",
            "mlflow_run_name": "test-run",
            "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/test-key",
            "vpc_config": True,
            "vpc_security_group_ids": ["sg-12345"],
            "vpc_subnets": ["subnet-abc"],
            "max_new_tokens": 100,
            "temperature": 0.7,
            "evaluator_arn": "arn:aws:sagemaker:us-west-2:123456789012:evaluator/test-evaluator",
        }

        template = Template(DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        base_model_step = pipeline_def["Steps"][0]
        
        # Verify MLflow config is not present in BASE_MODEL_ONLY template
        assert "MlflowConfig" not in pipeline_def
        
        # Verify KMS key
        assert base_model_step["Arguments"]["OutputDataConfig"]["KmsKeyId"] == context["kms_key_id"]
        
        # Verify VPC config
        assert base_model_step["Arguments"]["VpcConfig"]["SecurityGroupIds"] == ["sg-12345"]
        
        # Verify hyperparameters
        assert base_model_step["Arguments"]["HyperParameters"]["max_new_tokens"] == "100"
        assert base_model_step["Arguments"]["HyperParameters"]["temperature"] == "0.7"
        
        # Verify evaluator ARN
        assert base_model_step["Arguments"]["ServerlessJobConfig"]["EvaluatorArn"] == context["evaluator_arn"]


class TestCustomScorerTemplate:
    """Tests for CUSTOM_SCORER_TEMPLATE rendering."""

    def test_custom_scorer_template_minimal(self):
        """Test CUSTOM_SCORER_TEMPLATE with minimal context."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "gen_qa",
            "strategy": "gen_qa",
            "evaluation_metric": "all",
        }

        template = Template(CUSTOM_SCORER_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Verify EvaluationType is CustomScorerEvaluation
        custom_model_step = pipeline_def["Steps"][1]
        assert custom_model_step["Arguments"]["ServerlessJobConfig"]["EvaluationType"] == "CustomScorerEvaluation"

    def test_custom_scorer_template_with_base_model(self):
        """Test CUSTOM_SCORER_TEMPLATE with base model evaluation."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "gen_qa",
            "strategy": "gen_qa",
            "evaluation_metric": "all",
            "evaluate_base_model": True,
        }

        template = Template(CUSTOM_SCORER_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should have both base and custom model evaluation steps
        assert len(pipeline_def["Steps"]) == 4
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateBaseModel"
        assert pipeline_def["Steps"][2]["Name"] == "EvaluateCustomModel"


class TestCustomScorerTemplateBaseModelOnly:
    """Tests for CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY rendering."""

    def test_custom_scorer_base_model_only_minimal(self):
        """Test CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY with minimal context."""
        context = {
            **BASE_CONTEXT,
            "task": "gen_qa",
            "strategy": "gen_qa",
            "evaluation_metric": "all",
        }

        template = Template(CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Verify MLflow config is not present in BASE_MODEL_ONLY template
        assert "MlflowConfig" not in pipeline_def
        
        # Should have only 1 step
        assert len(pipeline_def["Steps"]) == 1
        assert pipeline_def["Steps"][0]["Name"] == "EvaluateBaseModel"
        
        # Verify it's CustomScorerEvaluation
        base_model_step = pipeline_def["Steps"][0]
        assert base_model_step["Arguments"]["ServerlessJobConfig"]["EvaluationType"] == "CustomScorerEvaluation"


class TestLLMAJTemplate:
    """Tests for LLMAJ_TEMPLATE rendering."""

    def test_llmaj_template_minimal(self):
        """Test LLMAJ_TEMPLATE with minimal context."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence", "relevance"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should have CreateEvaluationAction, EvaluateCustomInferenceModel, EvaluateCustomModelMetrics, AssociateLineage
        assert len(pipeline_def["Steps"]) == 4
        assert pipeline_def["Steps"][0]["Name"] == "CreateEvaluationAction"
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateCustomInferenceModel"
        assert pipeline_def["Steps"][2]["Name"] == "EvaluateCustomModelMetrics"
        assert pipeline_def["Steps"][3]["Name"] == "AssociateLineage"
        
        # Verify inference step
        inference_step = pipeline_def["Steps"][1]
        assert inference_step["Arguments"]["HyperParameters"]["task"] == "inference_only"
        assert inference_step["Arguments"]["ServerlessJobConfig"]["EvaluationType"] == "BenchmarkEvaluation"
        
        # Verify metrics step
        metrics_step = pipeline_def["Steps"][2]
        assert metrics_step["Arguments"]["ServerlessJobConfig"]["EvaluationType"] == "LLMAJEvaluation"
        assert metrics_step["Arguments"]["HyperParameters"]["judge_model_id"] == "anthropic.claude-v2"
        assert metrics_step["Arguments"]["HyperParameters"]["llmaj_metrics"] == ["coherence", "relevance"]

    def test_llmaj_template_with_base_model(self):
        """Test LLMAJ_TEMPLATE with base model evaluation."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence", "relevance"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
            "evaluate_base_model": True,
        }

        template = Template(LLMAJ_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should have 6 steps with base model evaluation
        assert len(pipeline_def["Steps"]) == 6
        assert pipeline_def["Steps"][0]["Name"] == "CreateEvaluationAction"
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateBaseInferenceModel"
        assert pipeline_def["Steps"][2]["Name"] == "EvaluateCustomInferenceModel"
        assert pipeline_def["Steps"][3]["Name"] == "EvaluateBaseModelMetrics"
        assert pipeline_def["Steps"][4]["Name"] == "EvaluateCustomModelMetrics"
        assert pipeline_def["Steps"][5]["Name"] == "AssociateLineage"

    def test_llmaj_template_with_custom_metrics(self):
        """Test LLMAJ_TEMPLATE with custom metrics S3 path."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence", "relevance"],
            "custom_metrics": "s3://test-bucket/custom-metrics.json",
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        metrics_step = pipeline_def["Steps"][2]
        hyperparams = metrics_step["Arguments"]["HyperParameters"]
        
        assert hyperparams["custom_metrics"] == "s3://test-bucket/custom-metrics.json"

    def test_llmaj_template_with_vpc_and_kms(self):
        """Test LLMAJ_TEMPLATE with VPC config and KMS key."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
            "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/test-key",
            "vpc_config": True,
            "vpc_security_group_ids": ["sg-12345"],
            "vpc_subnets": ["subnet-abc"],
        }

        template = Template(LLMAJ_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        inference_step = pipeline_def["Steps"][1]
        
        # Verify KMS key
        assert inference_step["Arguments"]["OutputDataConfig"]["KmsKeyId"] == context["kms_key_id"]
        
        # Verify VPC config
        assert inference_step["Arguments"]["VpcConfig"]["SecurityGroupIds"] == ["sg-12345"]
        assert inference_step["Arguments"]["VpcConfig"]["Subnets"] == ["subnet-abc"]


class TestLLMAJTemplateBaseModelOnly:
    """Tests for LLMAJ_TEMPLATE_BASE_MODEL_ONLY rendering."""

    def test_llmaj_base_model_only_minimal(self):
        """Test LLMAJ_TEMPLATE_BASE_MODEL_ONLY with minimal context."""
        context = {
            **BASE_CONTEXT,
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Verify MLflow config is not present in BASE_MODEL_ONLY template
        assert "MlflowConfig" not in pipeline_def
        
        # Should have 2 steps: EvaluateBaseInferenceModel and EvaluateBaseModelMetrics
        assert len(pipeline_def["Steps"]) == 2
        assert pipeline_def["Steps"][0]["Name"] == "EvaluateBaseInferenceModel"
        assert pipeline_def["Steps"][1]["Name"] == "EvaluateBaseModelMetrics"
        
        # Verify inference step
        inference_step = pipeline_def["Steps"][0]
        assert inference_step["Arguments"]["HyperParameters"]["task"] == "inference_only"
        
        # Verify metrics step
        metrics_step = pipeline_def["Steps"][1]
        assert metrics_step["Arguments"]["HyperParameters"]["judge_model_id"] == "anthropic.claude-v2"
        assert metrics_step["Arguments"]["HyperParameters"]["llmaj_metrics"] == ["coherence"]

    def test_llmaj_base_model_only_with_custom_metrics(self):
        """Test LLMAJ_TEMPLATE_BASE_MODEL_ONLY with custom metrics S3 path."""
        context = {
            **BASE_CONTEXT,
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "custom_metrics": "s3://test-bucket/custom-metrics.json",
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        metrics_step = pipeline_def["Steps"][1]
        assert metrics_step["Arguments"]["HyperParameters"]["custom_metrics"] == "s3://test-bucket/custom-metrics.json"

    def test_llmaj_base_model_only_with_kms_and_vpc(self):
        """Test LLMAJ_TEMPLATE_BASE_MODEL_ONLY with KMS and VPC."""
        context = {
            **BASE_CONTEXT,
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
            "kms_key_id": "arn:aws:kms:us-west-2:123456789012:key/test-key",
            "vpc_config": True,
            "vpc_security_group_ids": ["sg-12345"],
            "vpc_subnets": ["subnet-abc"],
        }

        template = Template(LLMAJ_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        inference_step = pipeline_def["Steps"][0]
        
        # Verify KMS
        assert inference_step["Arguments"]["OutputDataConfig"]["KmsKeyId"] == context["kms_key_id"]
        
        # Verify VPC
        assert inference_step["Arguments"]["VpcConfig"]["SecurityGroupIds"] == ["sg-12345"]
        assert inference_step["Arguments"]["VpcConfig"]["Subnets"] == ["subnet-abc"]


class TestTemplateEdgeCases:
    """Tests for edge cases and comprehensive coverage."""

    def test_all_templates_are_valid_jinja2(self):
        """Verify all templates are valid Jinja2 templates."""
        templates = [
            DETERMINISTIC_TEMPLATE,
            LLMAJ_TEMPLATE_BASE_MODEL_ONLY,
            DETERMINISTIC_TEMPLATE_BASE_MODEL_ONLY,
            CUSTOM_SCORER_TEMPLATE,
            CUSTOM_SCORER_TEMPLATE_BASE_MODEL_ONLY,
            LLMAJ_TEMPLATE,
        ]
        
        for template_str in templates:
            # Should not raise exception
            template = Template(template_str)
            assert template is not None

    def test_deterministic_template_without_optional_params(self):
        """Test that optional parameters can be omitted."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            # Not including: mlflow_experiment_name, mlflow_run_name, kms_key_id, vpc_config, etc.
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Should not have optional fields in MlflowConfig
        assert "MlflowExperimentName" not in pipeline_def["MlflowConfig"]
        assert "MlflowRunName" not in pipeline_def["MlflowConfig"]
        
        # Should not have KMS key in output config
        custom_model_step = pipeline_def["Steps"][1]
        assert "KmsKeyId" not in custom_model_step["Arguments"]["OutputDataConfig"]
        
        # Should not have VPC config
        assert "VpcConfig" not in custom_model_step["Arguments"]

    def test_s3_data_source_vs_dataset_arn(self):
        """Test conditional rendering based on dataset_uri content."""
        # Test with S3 URI
        context_s3 = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "dataset_uri": "s3://test-bucket/dataset",
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered_s3 = template.render(**context_s3)
        pipeline_def_s3 = json.loads(rendered_s3)
        
        data_source_s3 = pipeline_def_s3["Steps"][1]["Arguments"]["InputDataConfig"][0]["DataSource"]
        assert "S3DataSource" in data_source_s3
        assert data_source_s3["S3DataSource"]["S3Uri"] == "s3://test-bucket/dataset"

        # Test with AIRegistry Dataset ARN
        context_dataset = {
            **context_s3,
            "dataset_uri": "arn:aws:sagemaker:us-west-2:123456789012:hub-content/AIRegistry/DataSet/test-dataset",
        }

        rendered_dataset = template.render(**context_dataset)
        pipeline_def_dataset = json.loads(rendered_dataset)
        
        data_source_dataset = pipeline_def_dataset["Steps"][1]["Arguments"]["InputDataConfig"][0]["DataSource"]
        assert "DatasetSource" in data_source_dataset
        assert data_source_dataset["DatasetSource"]["DatasetArn"] == context_dataset["dataset_uri"]

    def test_llmaj_template_dependency_chain(self):
        """Test that LLMAJ_TEMPLATE has correct DependsOn relationships."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
            "evaluate_base_model": True,
        }

        template = Template(LLMAJ_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Check dependencies
        base_inference_step = pipeline_def["Steps"][1]
        custom_inference_step = pipeline_def["Steps"][2]
        base_metrics_step = pipeline_def["Steps"][3]
        custom_metrics_step = pipeline_def["Steps"][4]
        
        assert base_inference_step["DependsOn"] == ["CreateEvaluationAction"]
        assert custom_inference_step["DependsOn"] == ["CreateEvaluationAction"]
        assert base_metrics_step["DependsOn"] == ["EvaluateBaseInferenceModel"]
        assert custom_metrics_step["DependsOn"] == ["EvaluateCustomInferenceModel"]

    def test_lineage_associations_count(self):
        """Test that lineage associations are correctly generated."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "evaluate_base_model": True,
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # AssociateLineage step
        lineage_step = pipeline_def["Steps"][3]
        assert lineage_step["Name"] == "AssociateLineage"
        
        # Should have 2 artifacts (base and custom eval reports)
        assert len(lineage_step["Arguments"]["Artifacts"]) == 2
        
        # Should have 2 associations
        assert len(lineage_step["Arguments"]["Associations"]) == 2

    def test_hyperparameter_type_conversion(self):
        """Test that numeric hyperparameters are converted to strings."""
        context = {
            **BASE_CONTEXT,
            "pipeline_name": "test-pipeline",
            "source_model_package_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package/test-package",
            "dataset_artifact_arn": "arn:aws:sagemaker:us-west-2:123456789012:artifact/test-artifact",
            "action_arn_prefix": "arn:aws:sagemaker:us-west-2:123456789012:action",
            "model_package_group_arn": "arn:aws:sagemaker:us-west-2:123456789012:model-package-group/test-group",
            "task": "text-generation",
            "strategy": "greedy",
            "evaluation_metric": "accuracy",
            "max_new_tokens": 100,  # int
            "temperature": 0.7,  # float
            "top_k": 50,  # int
            "top_p": 0.9,  # float
        }

        template = Template(DETERMINISTIC_TEMPLATE)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        hyperparams = pipeline_def["Steps"][1]["Arguments"]["HyperParameters"]
        
        # All should be strings in JSON
        assert hyperparams["max_new_tokens"] == "100"
        assert hyperparams["temperature"] == "0.7"
        assert hyperparams["top_k"] == "50"
        assert hyperparams["top_p"] == "0.9"

    def test_training_job_name_generation_in_llmaj(self):
        """Test that LLMAJ templates generate training job names correctly."""
        context = {
            **BASE_CONTEXT,
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        # Check that training job name is set
        inference_step = pipeline_def["Steps"][0]
        assert inference_step["Arguments"]["TrainingJobName"] == "BaseInference"
        
        # Check that metrics step has dynamic training job name
        metrics_step = pipeline_def["Steps"][1]
        training_job_name = metrics_step["Arguments"]["TrainingJobName"]
        assert "Std:Join" in training_job_name
        assert training_job_name["Std:Join"]["On"] == "-"
        assert "base-llmaj-eval" in training_job_name["Std:Join"]["Values"]

    def test_inference_data_s3_path_construction(self):
        """Test that inference data S3 path is correctly constructed in LLMAJ."""
        context = {
            **BASE_CONTEXT,
            "judge_model_id": "anthropic.claude-v2",
            "llmaj_metrics": ["coherence"],
            "max_new_tokens": "100",
            "temperature": "0.7",
            "top_k": "50",
            "top_p": "0.9",
        }

        template = Template(LLMAJ_TEMPLATE_BASE_MODEL_ONLY)
        rendered = template.render(**context)
        
        pipeline_def = json.loads(rendered)
        
        metrics_step = pipeline_def["Steps"][1]
        inference_path = metrics_step["Arguments"]["HyperParameters"]["inference_data_s3_path"]
        
        # Should be a Std:Join expression
        assert "Std:Join" in inference_path
        assert inference_path["Std:Join"]["On"] == ""
        
        # Should reference the inference step output
        values = inference_path["Std:Join"]["Values"]
        assert any("Steps.EvaluateBaseInferenceModel.OutputDataConfig.S3OutputPath" in str(v) for v in values)
        assert "BaseInference" in values
        assert "/eval_results/inference_output.jsonl" in values
