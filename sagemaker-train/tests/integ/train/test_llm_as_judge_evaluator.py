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
"""Integration tests for LLMAsJudgeEvaluator"""
from __future__ import absolute_import

import json
import pytest
import logging

from sagemaker.train.evaluate import (
    LLMAsJudgeEvaluator,
    EvaluationPipelineExecution,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Test timeout configuration (in seconds)
EVALUATION_TIMEOUT_SECONDS = 14400  # 4 hours

# Custom metrics definition from notebook
CUSTOM_METRIC_DICT = {
    "customMetricDefinition": {
        "name": "PositiveSentiment",
        "instructions": (
            "You are an expert evaluator. Your task is to assess if the sentiment of the response is positive. "
            "Rate the response based on whether it conveys positive sentiment, helpfulness, and constructive tone.\n\n"
            "Consider the following:\n"
            "- Does the response have a positive, encouraging tone?\n"
            "- Is the response helpful and constructive?\n"
            "- Does it avoid negative language or criticism?\n\n"
            "Rate on this scale:\n"
            "- Good: Response has positive sentiment\n"
            "- Poor: Response lacks positive sentiment\n\n"
            "Here is the actual task:\n"
            "Prompt: {{prompt}}\n"
            "Response: {{prediction}}"
        ),
        "ratingScale": [
            {"definition": "Good", "value": {"floatValue": 1}},
            {"definition": "Poor", "value": {"floatValue": 0}}
        ]
    }
}

# Test configuration values from llm_as_judge_demo.ipynb
# TEST_CONFIG = {
#     "model_package_arn": "arn:aws:sagemaker:us-west-2:052150106756:model-package/test-finetuned-models-gamma/28",
#     "evaluator_model": "anthropic.claude-3-5-haiku-20241022-v1:0",
#     "dataset_s3_uri": "s3://my-sagemaker-sherpa-dataset/dataset/gen-qa-formatted-dataset/gen_qa.jsonl",
#     "builtin_metrics": ["Completeness", "Faithfulness"],
#     "custom_metrics_json": json.dumps([CUSTOM_METRIC_DICT]),
#     "s3_output_path": "s3://mufi-test-serverless-smtj/eval/",
#     "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:052150106756:mlflow-tracking-server/mmlu-eval-experiment",
#     "evaluate_base_model": False,
#     "region": "us-west-2",
# }

TEST_CONFIG = {
    "model_package_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package/sdk-test-finetuned-models/1",
    "evaluator_model": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "dataset_s3_uri": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/gen_qa.jsonl",
    "builtin_metrics": ["Completeness", "Faithfulness"],
    "custom_metrics_json": json.dumps([CUSTOM_METRIC_DICT]),
    "s3_output_path": "s3://sagemaker-us-west-2-729646638167/model-customization/eval/",
    "mlflow_tracking_server_arn": "arn:aws:sagemaker:us-west-2:729646638167:mlflow-app/app-W7FOBBXZANVX",
    # "model_package_group_arn": "arn:aws:sagemaker:us-west-2:729646638167:model-package-group/sdk-test-finetuned-models",
    "evaluate_base_model": False,
    "region": "us-west-2",
}


@pytest.mark.skip(reason="Temporarily skipped - moved from tests/integ/sagemaker/modules/evaluate/")
class TestLLMAsJudgeEvaluatorIntegration:
    """Integration tests for LLMAsJudgeEvaluator"""

    def test_llm_as_judge_evaluation_full_flow(self):
        """
        Test complete LLM-as-Judge evaluation flow with custom and built-in metrics.
        
        This test mirrors the flow from llm_as_judge_demo.ipynb and covers:
        1. Creating LLMAsJudgeEvaluator with custom and built-in metrics
        2. Starting evaluation
        3. Monitoring execution
        4. Waiting for completion
        5. Viewing results with pagination
        6. Retrieving execution by ARN
        7. Listing all evaluations
        
        Test configuration values are taken directly from the notebook example.
        """
        # Step 1: Create LLMAsJudgeEvaluator
        logger.info("Creating LLMAsJudgeEvaluator with custom and built-in metrics")
        
        # Create evaluator (matching notebook configuration)
        evaluator = LLMAsJudgeEvaluator(
            model=TEST_CONFIG["model_package_arn"],
            evaluator_model=TEST_CONFIG["evaluator_model"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            builtin_metrics=TEST_CONFIG["builtin_metrics"],
            custom_metrics=TEST_CONFIG["custom_metrics_json"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            evaluate_base_model=TEST_CONFIG["evaluate_base_model"],
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.model == TEST_CONFIG["model_package_arn"]
        assert evaluator.evaluator_model == TEST_CONFIG["evaluator_model"]
        assert evaluator.dataset == TEST_CONFIG["dataset_s3_uri"]
        assert evaluator.builtin_metrics == TEST_CONFIG["builtin_metrics"]
        assert evaluator.custom_metrics == TEST_CONFIG["custom_metrics_json"]
        assert evaluator.evaluate_base_model == TEST_CONFIG["evaluate_base_model"]
        
        logger.info(f"Created evaluator with judge model: {evaluator.evaluator_model}")
        
        # Step 2: Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        assert execution.name is not None
        assert execution.eval_type is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        logger.info(f"Initial Status: {execution.status.overall_status}")
        
        # Step 3: Monitor execution
        logger.info("Refreshing execution status")
        execution.refresh()
        
        # Verify status was updated
        assert execution.status.overall_status is not None
        
        # Log step details if available
        if execution.status.step_details:
            logger.info("Step Details:")
            for step in execution.status.step_details:
                logger.info(f"  {step.name}: {step.status}")
        
        # Step 4: Wait for completion
        logger.info(f"Waiting for evaluation to complete (timeout: {EVALUATION_TIMEOUT_SECONDS}s / {EVALUATION_TIMEOUT_SECONDS//3600}h)")
        
        try:
            execution.wait(target_status="Succeeded", poll=30, timeout=EVALUATION_TIMEOUT_SECONDS)
            logger.info(f"Final Status: {execution.status.overall_status}")
            
            # Verify completion
            assert execution.status.overall_status == "Succeeded"
            
            # Step 5: View results with pagination
            logger.info("Displaying results (limit=5)")
            execution.show_results(limit=5, offset=0, show_explanations=False)
            
            # Verify S3 output path is set
            assert execution.s3_output_path is not None
            logger.info(f"Results stored at: {execution.s3_output_path}")
            
        except Exception as e:
            logger.error(f"Evaluation failed or timed out: {e}")
            logger.error(f"Final status: {execution.status.overall_status}")
            if execution.status.failure_reason:
                logger.error(f"Failure reason: {execution.status.failure_reason}")
            
            # Log step failures
            if execution.status.step_details:
                for step in execution.status.step_details:
                    if "failed" in step.status.lower():
                        logger.error(f"Failed step: {step.name}")
                        if step.failure_reason:
                            logger.error(f"  Reason: {step.failure_reason}")
            
            # Re-raise to fail the test
            raise
        
        # Step 6: Retrieve execution by ARN
        logger.info("Retrieving execution by ARN")
        retrieved_execution = EvaluationPipelineExecution.get(
            arn=execution.arn,
            region=TEST_CONFIG["region"]
        )
        
        # Verify retrieved execution matches
        assert retrieved_execution.arn == execution.arn
        
        logger.info(f"Retrieved execution status: {retrieved_execution.status.overall_status}")
        
        # Step 7: List all LLM-as-Judge evaluations
        logger.info("Listing all LLM-as-Judge evaluations")
        all_executions_iter = LLMAsJudgeEvaluator.get_all(region=TEST_CONFIG["region"])
        all_executions = list(all_executions_iter)
        
        if all_executions:
            # Verify our execution is in the list
            execution_arns = [exec.arn for exec in all_executions]
            assert execution.arn in execution_arns
        
        logger.info("Integration test completed successfully")

    def test_llm_as_judge_evaluator_validation(self):
        """Test LLMAsJudgeEvaluator validation of inputs"""
        # Test invalid MLflow ARN format (validated in constructor)
        # Use a JumpStart model ID to avoid AWS API calls
        with pytest.raises(ValueError, match="Invalid MLFlow resource ARN"):
            LLMAsJudgeEvaluator(
                model="meta-textgeneration-llama-3-2-1b-instruct",
                evaluator_model=TEST_CONFIG["evaluator_model"],
                dataset=TEST_CONFIG["dataset_s3_uri"],
                s3_output_path=TEST_CONFIG["s3_output_path"],
                mlflow_resource_arn="invalid-arn",
            )        
        logger.info("Validation tests passed")

    def test_llm_as_judge_builtin_metrics_prefix_handling(self):
        """Test that built-in metrics work with or without 'Builtin.' prefix"""
        # Test with prefix
        evaluator_with_prefix = LLMAsJudgeEvaluator(
            model=TEST_CONFIG["model_package_arn"],
            evaluator_model=TEST_CONFIG["evaluator_model"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            builtin_metrics=["Builtin.Correctness", "Builtin.Helpfulness"],
        )
        assert evaluator_with_prefix.builtin_metrics == ["Builtin.Correctness", "Builtin.Helpfulness"]
        
        # Test without prefix
        evaluator_without_prefix = LLMAsJudgeEvaluator(
            model=TEST_CONFIG["model_package_arn"],
            evaluator_model=TEST_CONFIG["evaluator_model"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            builtin_metrics=["Correctness", "Helpfulness"],
        )
        assert evaluator_without_prefix.builtin_metrics == ["Correctness", "Helpfulness"]
        
        logger.info("Built-in metrics prefix handling tests passed")

    @pytest.mark.skip(reason="Built-in metrics only test - to be enabled when needed")
    def test_llm_as_judge_builtin_metrics_only(self):
        """
        Test LLM-as-Judge evaluation with only built-in metrics (no custom metrics).
        
        This test uses only built-in metrics without custom metrics.
        
        Note: This test is currently skipped. Remove the @pytest.mark.skip decorator
        when you want to enable it.
        """
        logger.info("Creating LLMAsJudgeEvaluator with built-in metrics only")
        
        # Create evaluator with only built-in metrics
        evaluator = LLMAsJudgeEvaluator(
            model=TEST_CONFIG["model_package_arn"],
            evaluator_model=TEST_CONFIG["evaluator_model"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            builtin_metrics=["Completeness", "Faithfulness", "Helpfulness"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            evaluate_base_model=False,
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.builtin_metrics == ["Completeness", "Faithfulness", "Helpfulness"]
        assert evaluator.custom_metrics is None
        
        logger.info("Created evaluator with built-in metrics only")
        
        # Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        
        # Wait for completion
        logger.info(f"Waiting for evaluation to complete (timeout: {EVALUATION_TIMEOUT_SECONDS}s / {EVALUATION_TIMEOUT_SECONDS//3600}h)")
        execution.wait(target_status="Succeeded", poll=30, timeout=EVALUATION_TIMEOUT_SECONDS)
        
        # Verify completion
        assert execution.status.overall_status == "Succeeded"
        logger.info("Built-in metrics only evaluation completed successfully")

    @pytest.mark.skip(reason="Custom metrics only test - to be enabled when needed")
    def test_llm_as_judge_custom_metrics_only(self):
        """
        Test LLM-as-Judge evaluation with only custom metrics (no built-in metrics).
        
        This test uses only custom metrics without built-in metrics.
        
        Note: This test is currently skipped. Remove the @pytest.mark.skip decorator
        when you want to enable it.
        """
        logger.info("Creating LLMAsJudgeEvaluator with custom metrics only")
        
        # Create evaluator with only custom metrics
        evaluator = LLMAsJudgeEvaluator(
            model=TEST_CONFIG["model_package_arn"],
            evaluator_model=TEST_CONFIG["evaluator_model"],
            dataset=TEST_CONFIG["dataset_s3_uri"],
            custom_metrics=TEST_CONFIG["custom_metrics_json"],
            # mlflow_resource_arn=TEST_CONFIG["mlflow_tracking_server_arn"],
            s3_output_path=TEST_CONFIG["s3_output_path"],
            evaluate_base_model=False,
        )
        
        # Verify evaluator was created
        assert evaluator is not None
        assert evaluator.custom_metrics == TEST_CONFIG["custom_metrics_json"]
        assert evaluator.builtin_metrics is None
        
        logger.info("Created evaluator with custom metrics only")
        
        # Start evaluation
        logger.info("Starting evaluation execution")
        execution = evaluator.evaluate()
        
        # Verify execution was created
        assert execution is not None
        assert execution.arn is not None
        
        logger.info(f"Pipeline Execution ARN: {execution.arn}")
        
        # Wait for completion
        logger.info(f"Waiting for evaluation to complete (timeout: {EVALUATION_TIMEOUT_SECONDS}s / {EVALUATION_TIMEOUT_SECONDS//3600}h)")
        execution.wait(target_status="Succeeded", poll=30, timeout=EVALUATION_TIMEOUT_SECONDS)
        
        # Verify completion
        assert execution.status.overall_status == "Succeeded"
        logger.info("Custom metrics only evaluation completed successfully")
